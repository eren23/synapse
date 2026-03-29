use crate::config::position::RoPEStyle;
use crate::kv_cache::KVCacheLayer;
use crate::ops::activation::{gelu, is_gated_ffn, softmax_slice};
use crate::ops::matmul::matmul_t;
#[cfg(feature = "metal")]
use crate::ops::norm::layernorm;
use crate::ops::norm::{apply_headwise_rmsnorm, apply_norm};
use crate::ops::rope::apply_rope_inplace;
use crate::ops::vector::{add_vecs, add_vecs_inplace};
use crate::registry::{AttentionVariant, FFNVariant, NormVariant};
use crate::weight_loading::{AlignedBuffer, RawTensor, WeightError};

/// A single decoder transformer layer (pre-norm architecture).
///
/// Forward: norm → attention → residual → norm → FFN → residual.
pub struct DecoderLayer {
    pub attn_norm: Box<dyn NormVariant>,
    pub attention: Box<dyn AttentionVariant>,
    pub ffn_norm: Box<dyn NormVariant>,
    pub ffn: Box<dyn FFNVariant>,
    pub hidden_size: usize,
    /// Whether this layer uses per-head Q/K norms (e.g. Qwen3 does, LLaMA/Mistral do not).
    pub has_head_norms: bool,
    /// RoPE dimension pairing convention.
    pub rope_style: RoPEStyle,

    // ── Weights (64-byte aligned for SIMD) ───────────────────────────
    pub attn_norm_weight: AlignedBuffer,
    pub w_q: AlignedBuffer,
    pub w_k: AlignedBuffer,
    pub w_v: AlignedBuffer,
    pub w_o: AlignedBuffer,
    pub q_norm_weight: AlignedBuffer,
    pub k_norm_weight: AlignedBuffer,
    // ── Attention biases (empty if model doesn't use them) ──────────
    pub q_bias: AlignedBuffer,
    pub k_bias: AlignedBuffer,
    pub v_bias: AlignedBuffer,
    // ── FFN weights ─────────────────────────────────────────────────
    pub ffn_norm_weight: AlignedBuffer,
    pub ffn_gate: AlignedBuffer,
    pub ffn_up: AlignedBuffer,
    pub ffn_down: AlignedBuffer,
}

impl DecoderLayer {
    /// Pre-norm forward: norm→attention→residual→norm→FFN→residual.
    ///
    /// `x` is `[seq_len, hidden_size]` (flat). Returns same shape.
    /// `rope_cos` / `rope_sin` are `[max_pos, head_dim/2]` precomputed tables.
    pub fn forward(
        &self,
        x: &[f32],
        seq_len: usize,
        rope_cos: &[f32],
        rope_sin: &[f32],
    ) -> Vec<f32> {
        let h = self.hidden_size;

        // 1. Attention sub-layer
        let normed = apply_norm(x, &self.attn_norm_weight, &*self.attn_norm, h);
        let attn_out = self.apply_attention(&normed, seq_len, rope_cos, rope_sin);
        let mut residual = add_vecs(x, &attn_out);

        // 2. FFN sub-layer
        let normed = apply_norm(&residual, &self.ffn_norm_weight, &*self.ffn_norm, h);
        let ffn_out = self.apply_ffn(&normed);
        add_vecs_inplace(&mut residual, &ffn_out);

        residual
    }

    /// Batched prefill: same as [`forward`] but also populates the KV cache.
    ///
    /// Runs the fast batched attention (all positions at once), then saves
    /// the RoPE'd K and raw V for each position into `cache_layer` so that
    /// subsequent [`forward_one`] decode steps can reuse them.
    pub fn forward_prefill_batched(
        &self,
        x: &[f32],
        seq_len: usize,
        cache_layer: &mut KVCacheLayer,
        rope_cos: &[f32],
        rope_sin: &[f32],
    ) -> Vec<f32> {
        let h = self.hidden_size;

        // 1. Attention sub-layer (batched + cache populate)
        let normed = apply_norm(x, &self.attn_norm_weight, &*self.attn_norm, h);
        let attn_out =
            self.apply_attention_and_cache(&normed, seq_len, cache_layer, rope_cos, rope_sin);
        let mut residual = add_vecs(x, &attn_out);

        // 2. FFN sub-layer
        let normed = apply_norm(&residual, &self.ffn_norm_weight, &*self.ffn_norm, h);
        let ffn_out = self.apply_ffn(&normed);
        add_vecs_inplace(&mut residual, &ffn_out);

        residual
    }

    /// Single-token forward using KV cache for autoregressive decode.
    ///
    /// `hidden` is `[1, hidden_size]` (flat). `cache_layer` holds K/V from
    /// prior positions. `pos` is the 0-based position of this token.
    /// Returns `[1, hidden_size]`.
    ///
    /// The key difference from [`forward`]: attention reads K/V from the cache
    /// instead of recomputing them for all positions.
    pub fn forward_one(
        &self,
        hidden: &[f32],
        cache_layer: &mut KVCacheLayer,
        pos: usize,
        rope_cos: &[f32],
        rope_sin: &[f32],
    ) -> Vec<f32> {
        let h = self.hidden_size;
        debug_assert_eq!(
            hidden.len(),
            h,
            "forward_one: hidden must be [1, hidden_size]"
        );

        // 1. Attention sub-layer
        let normed = apply_norm(hidden, &self.attn_norm_weight, &*self.attn_norm, h);
        let attn_out = self.apply_attention_cached(&normed, cache_layer, pos, rope_cos, rope_sin);
        let mut residual = add_vecs(hidden, &attn_out);

        // 2. FFN sub-layer
        let normed = apply_norm(&residual, &self.ffn_norm_weight, &*self.ffn_norm, h);
        let ffn_out = self.apply_ffn(&normed);
        add_vecs_inplace(&mut residual, &ffn_out);

        residual
    }

    /// Number of trainable parameters in this layer.
    pub fn param_count(&self) -> usize {
        let h = self.hidden_size;
        let q_dim = self.attention.num_heads() * self.attention.head_dim();
        let kv_dim = self.attention.num_kv_heads() * self.attention.head_dim();
        let inter = self.ffn.intermediate_size();

        let head_norms = if self.has_head_norms {
            2 * self.attention.head_dim()
        } else {
            0
        };
        let norms = 2 * h + head_norms; // attn_norm + ffn_norm + optional q/k norm
        let attn = q_dim * h + kv_dim * h + kv_dim * h + h * q_dim;
        let ffn = if is_gated_ffn(self.ffn.name()) {
            3 * inter * h
        } else {
            2 * inter * h
        };

        norms + attn + ffn
    }

    /// Weight keys this layer expects (relative to `layers[i].`).
    pub fn weight_keys(&self, layer_idx: usize) -> Vec<String> {
        let i = layer_idx;
        let mut keys = vec![
            format!("layers[{i}].attn_norm.weight"),
            format!("layers[{i}].attention.w_q"),
            format!("layers[{i}].attention.w_k"),
            format!("layers[{i}].attention.w_v"),
            format!("layers[{i}].attention.w_o"),
        ];
        if self.has_head_norms {
            keys.push(format!("layers[{i}].attention.q_norm"));
            keys.push(format!("layers[{i}].attention.k_norm"));
        }
        keys.push(format!("layers[{i}].ffn_norm.weight"));
        if is_gated_ffn(self.ffn.name()) {
            keys.push(format!("layers[{i}].ffn.w_gate"));
        }
        keys.push(format!("layers[{i}].ffn.w_up"));
        keys.push(format!("layers[{i}].ffn.w_down"));
        keys
    }

    /// Assign a weight by its field name (e.g. "attention.w_q").
    pub fn set_weight(&mut self, field: &str, tensor: &RawTensor) -> Result<(), WeightError> {
        self.validate_weight_shape(field, &tensor.shape)?;
        match field {
            "attn_norm.weight" => self.attn_norm_weight = tensor.data.clone(),
            "attention.w_q" => self.w_q = tensor.data.clone(),
            "attention.w_k" => self.w_k = tensor.data.clone(),
            "attention.w_v" => self.w_v = tensor.data.clone(),
            "attention.w_o" => self.w_o = tensor.data.clone(),
            "attention.q_norm" => self.q_norm_weight = tensor.data.clone(),
            "attention.k_norm" => self.k_norm_weight = tensor.data.clone(),
            "attention.q_bias" => self.q_bias = tensor.data.clone(),
            "attention.k_bias" => self.k_bias = tensor.data.clone(),
            "attention.v_bias" => self.v_bias = tensor.data.clone(),
            "ffn_norm.weight" => self.ffn_norm_weight = tensor.data.clone(),
            "ffn.w_gate" => self.ffn_gate = tensor.data.clone(),
            "ffn.w_up" => self.ffn_up = tensor.data.clone(),
            "ffn.w_down" => self.ffn_down = tensor.data.clone(),
            _ => {}
        }
        Ok(())
    }

    fn validate_weight_shape(&self, field: &str, actual: &[usize]) -> Result<(), WeightError> {
        let h = self.hidden_size;
        let q_dim = self.attention.num_heads() * self.attention.head_dim();
        let kv_dim = self.attention.num_kv_heads() * self.attention.head_dim();
        let inter = self.ffn.intermediate_size();

        let expected = match field {
            "attn_norm.weight" | "ffn_norm.weight" => vec![h],
            "attention.w_q" => vec![q_dim, h],
            "attention.w_k" | "attention.w_v" => vec![kv_dim, h],
            "attention.w_o" => vec![h, q_dim],
            "attention.q_norm" | "attention.k_norm" => vec![self.attention.head_dim()],
            "attention.q_bias" => vec![q_dim],
            "attention.k_bias" | "attention.v_bias" => vec![kv_dim],
            "ffn.w_gate" | "ffn.w_up" => vec![inter, h],
            "ffn.w_down" => vec![h, inter],
            _ => return Ok(()),
        };

        if actual != expected {
            return Err(WeightError::ShapeMismatch(format!(
                "{field}: expected {:?}, got {:?}",
                expected, actual
            )));
        }

        Ok(())
    }

    // ── Backend-dispatched forward (Metal feature) ──────────────────

    /// Forward pass dispatched through ComputeBackend (GPU or CPU).
    #[cfg(feature = "metal")]
    pub fn forward_with_backend(
        &self,
        x: &[f32],
        seq_len: usize,
        backend: &crate::metal::ComputeBackend,
        rope_cos: &[f32],
        rope_sin: &[f32],
    ) -> Vec<f32> {
        let h = self.hidden_size;

        let normed = apply_norm_dispatch(x, &self.attn_norm_weight, &*self.attn_norm, h, backend);
        let attn_out = self.apply_attention_dispatch(&normed, seq_len, backend, rope_cos, rope_sin);
        let mut residual = add_vecs(x, &attn_out);

        let normed = apply_norm_dispatch(
            &residual,
            &self.ffn_norm_weight,
            &*self.ffn_norm,
            h,
            backend,
        );
        let ffn_out = self.apply_ffn_dispatch(&normed, backend);
        add_vecs_inplace(&mut residual, &ffn_out);

        residual
    }

    /// Single-token decode with backend dispatch.
    #[cfg(feature = "metal")]
    pub fn forward_one_with_backend(
        &self,
        hidden: &[f32],
        cache_layer: &mut KVCacheLayer,
        pos: usize,
        backend: &crate::metal::ComputeBackend,
        rope_cos: &[f32],
        rope_sin: &[f32],
    ) -> Vec<f32> {
        let h = self.hidden_size;
        debug_assert_eq!(hidden.len(), h);

        let normed =
            apply_norm_dispatch(hidden, &self.attn_norm_weight, &*self.attn_norm, h, backend);
        let attn_out = self.apply_attention_cached_dispatch(
            &normed,
            cache_layer,
            pos,
            backend,
            rope_cos,
            rope_sin,
        );
        let mut residual = add_vecs(hidden, &attn_out);

        let normed = apply_norm_dispatch(
            &residual,
            &self.ffn_norm_weight,
            &*self.ffn_norm,
            h,
            backend,
        );
        let ffn_out = self.apply_ffn_dispatch(&normed, backend);
        add_vecs_inplace(&mut residual, &ffn_out);

        residual
    }

    #[cfg(feature = "metal")]
    fn apply_attention_dispatch(
        &self,
        x: &[f32],
        seq_len: usize,
        backend: &crate::metal::ComputeBackend,
        rope_cos: &[f32],
        rope_sin: &[f32],
    ) -> Vec<f32> {
        let h = self.hidden_size;
        let num_heads = self.attention.num_heads();
        let num_kv_heads = self.attention.num_kv_heads();
        let head_dim = self.attention.head_dim();
        let q_dim = num_heads * head_dim;
        let kv_dim = num_kv_heads * head_dim;
        let groups = num_heads / num_kv_heads;
        let mut q = backend.matmul_t(x, &self.w_q, seq_len, h, q_dim);
        Self::add_bias(&mut q, &self.q_bias, seq_len, q_dim);
        let mut k = backend.matmul_t(x, &self.w_k, seq_len, h, kv_dim);
        Self::add_bias(&mut k, &self.k_bias, seq_len, kv_dim);
        let mut v = backend.matmul_t(x, &self.w_v, seq_len, h, kv_dim);
        Self::add_bias(&mut v, &self.v_bias, seq_len, kv_dim);
        let mut q = apply_headwise_rmsnorm(
            &q,
            &self.q_norm_weight,
            seq_len,
            num_heads,
            head_dim,
            self.attn_norm.eps() as f32,
        );
        let mut k = apply_headwise_rmsnorm(
            &k,
            &self.k_norm_weight,
            seq_len,
            num_kv_heads,
            head_dim,
            self.attn_norm.eps() as f32,
        );

        apply_rope_inplace(
            &mut q,
            rope_cos,
            rope_sin,
            seq_len,
            num_heads,
            head_dim,
            0,
            self.rope_style,
        );
        apply_rope_inplace(
            &mut k,
            rope_cos,
            rope_sin,
            seq_len,
            num_kv_heads,
            head_dim,
            0,
            self.rope_style,
        );

        let scale = 1.0 / (head_dim as f32).sqrt();
        let mut attn_output = vec![0.0f32; seq_len * q_dim];
        for head in 0..num_heads {
            let kv_head = head / groups;
            for t in 0..seq_len {
                let mut scores = vec![f32::NEG_INFINITY; seq_len];
                for s in 0..=t {
                    let mut dot = 0.0f32;
                    for d in 0..head_dim {
                        dot += q[t * q_dim + head * head_dim + d]
                            * k[s * kv_dim + kv_head * head_dim + d];
                    }
                    scores[s] = dot * scale;
                }
                softmax_slice(&mut scores[..=t]);
                for d in 0..head_dim {
                    let mut sum = 0.0f32;
                    for s in 0..=t {
                        sum += scores[s] * v[s * kv_dim + kv_head * head_dim + d];
                    }
                    attn_output[t * q_dim + head * head_dim + d] = sum;
                }
            }
        }

        backend.matmul_t(&attn_output, &self.w_o, seq_len, q_dim, h)
    }

    #[cfg(feature = "metal")]
    fn apply_attention_cached_dispatch(
        &self,
        x: &[f32],
        cache_layer: &mut KVCacheLayer,
        pos: usize,
        backend: &crate::metal::ComputeBackend,
        rope_cos: &[f32],
        rope_sin: &[f32],
    ) -> Vec<f32> {
        let h = self.hidden_size;
        let num_heads = self.attention.num_heads();
        let num_kv_heads = self.attention.num_kv_heads();
        let head_dim = self.attention.head_dim();
        let q_dim = num_heads * head_dim;
        let kv_dim = num_kv_heads * head_dim;

        // GPU-dispatched Q/K/V projections
        let mut q = backend.matmul_t(x, &self.w_q, 1, h, q_dim);
        Self::add_bias(&mut q, &self.q_bias, 1, q_dim);
        let mut k = backend.matmul_t(x, &self.w_k, 1, h, kv_dim);
        Self::add_bias(&mut k, &self.k_bias, 1, kv_dim);
        let mut v = backend.matmul_t(x, &self.w_v, 1, h, kv_dim);
        Self::add_bias(&mut v, &self.v_bias, 1, kv_dim);

        let attn_out = crate::ops::attention::cached_attention_decode(
            &q,
            &k,
            &v,
            num_heads,
            num_kv_heads,
            head_dim,
            cache_layer,
            pos,
            rope_cos,
            rope_sin,
            self.rope_style,
            &self.q_norm_weight,
            &self.k_norm_weight,
            self.attn_norm.eps() as f32,
            self.attention.window_size(),
        );

        backend.matmul_t(&attn_out, &self.w_o, 1, q_dim, h)
    }

    #[cfg(feature = "metal")]
    fn apply_ffn_dispatch(&self, x: &[f32], backend: &crate::metal::ComputeBackend) -> Vec<f32> {
        let h = self.hidden_size;
        let inter = self.ffn.intermediate_size();
        let tokens = x.len() / h;

        match self.ffn.name() {
            "SwiGLU" => {
                let gate = backend.matmul_t(x, &self.ffn_gate, tokens, h, inter);
                let up = backend.matmul_t(x, &self.ffn_up, tokens, h, inter);
                let hidden = backend.swiglu(&gate, &up);
                backend.matmul_t(&hidden, &self.ffn_down, tokens, inter, h)
            }
            "GeGLU" => {
                let gate = backend.matmul_t(x, &self.ffn_gate, tokens, h, inter);
                let up = backend.matmul_t(x, &self.ffn_up, tokens, h, inter);
                let mut hidden = vec![0.0f32; tokens * inter];
                for i in 0..hidden.len() {
                    hidden[i] = gelu(gate[i]) * up[i];
                }
                backend.matmul_t(&hidden, &self.ffn_down, tokens, inter, h)
            }
            _ => {
                let mut activated = backend.matmul_t(x, &self.ffn_up, tokens, h, inter);
                for v in activated.iter_mut() {
                    *v = gelu(*v);
                }
                backend.matmul_t(&activated, &self.ffn_down, tokens, inter, h)
            }
        }
    }

    // ── Attention ────────────────────────────────────────────────────

    /// Add a per-column bias to a row-major matrix `[m, n]` in place.
    /// No-op when `bias` is empty (model has no attention biases).
    fn add_bias(x: &mut [f32], bias: &[f32], m: usize, n: usize) {
        if bias.is_empty() {
            return;
        }
        for row in 0..m {
            for col in 0..n {
                x[row * n + col] += bias[col];
            }
        }
    }

    fn apply_attention(
        &self,
        x: &[f32],
        seq_len: usize,
        rope_cos: &[f32],
        rope_sin: &[f32],
    ) -> Vec<f32> {
        let h = self.hidden_size;
        let num_heads = self.attention.num_heads();
        let num_kv_heads = self.attention.num_kv_heads();
        let head_dim = self.attention.head_dim();
        let q_dim = num_heads * head_dim;
        let kv_dim = num_kv_heads * head_dim;
        let groups = num_heads / num_kv_heads;
        let scale = 1.0 / (head_dim as f32).sqrt();

        // Q, K, V projections: x is [seq_len, h]
        let mut q = matmul_t(x, &self.w_q, seq_len, h, q_dim);
        Self::add_bias(&mut q, &self.q_bias, seq_len, q_dim);
        let mut k = matmul_t(x, &self.w_k, seq_len, h, kv_dim);
        Self::add_bias(&mut k, &self.k_bias, seq_len, kv_dim);
        let mut v = matmul_t(x, &self.w_v, seq_len, h, kv_dim);
        Self::add_bias(&mut v, &self.v_bias, seq_len, kv_dim);

        let mut q = apply_headwise_rmsnorm(
            &q,
            &self.q_norm_weight,
            seq_len,
            num_heads,
            head_dim,
            self.attn_norm.eps() as f32,
        );
        let mut k = apply_headwise_rmsnorm(
            &k,
            &self.k_norm_weight,
            seq_len,
            num_kv_heads,
            head_dim,
            self.attn_norm.eps() as f32,
        );

        // Apply RoPE to Q and K
        apply_rope_inplace(
            &mut q,
            rope_cos,
            rope_sin,
            seq_len,
            num_heads,
            head_dim,
            0,
            self.rope_style,
        );
        apply_rope_inplace(
            &mut k,
            rope_cos,
            rope_sin,
            seq_len,
            num_kv_heads,
            head_dim,
            0,
            self.rope_style,
        );

        // Multi-head causal attention with GQA support
        let mut attn_output = vec![0.0f32; seq_len * q_dim];

        for head in 0..num_heads {
            let kv_head = head / groups;

            for t in 0..seq_len {
                // Compute scores for position t attending to 0..=t
                let mut scores = vec![f32::NEG_INFINITY; seq_len];
                for s in 0..=t {
                    let mut dot = 0.0f32;
                    for d in 0..head_dim {
                        dot += q[t * q_dim + head * head_dim + d]
                            * k[s * kv_dim + kv_head * head_dim + d];
                    }
                    scores[s] = dot * scale;
                }

                softmax_slice(&mut scores[..=t]);

                // Weighted sum of values
                for d in 0..head_dim {
                    let mut sum = 0.0f32;
                    for s in 0..=t {
                        sum += scores[s] * v[s * kv_dim + kv_head * head_dim + d];
                    }
                    attn_output[t * q_dim + head * head_dim + d] = sum;
                }
            }
        }

        // Output projection
        matmul_t(&attn_output, &self.w_o, seq_len, q_dim, h)
    }

    // ── Cached attention (single-token decode) ────────────────────
    //
    // Computes Q/K/V for one token, delegates to shared attention logic,
    // then applies the output projection.

    fn apply_attention_cached(
        &self,
        x: &[f32],
        cache_layer: &mut KVCacheLayer,
        pos: usize,
        rope_cos: &[f32],
        rope_sin: &[f32],
    ) -> Vec<f32> {
        let h = self.hidden_size;
        let num_heads = self.attention.num_heads();
        let num_kv_heads = self.attention.num_kv_heads();
        let head_dim = self.attention.head_dim();
        let q_dim = num_heads * head_dim;
        let kv_dim = num_kv_heads * head_dim;

        // Q, K, V projections for single token: x is [1, h]
        let mut q = matmul_t(x, &self.w_q, 1, h, q_dim);
        Self::add_bias(&mut q, &self.q_bias, 1, q_dim);
        let mut k = matmul_t(x, &self.w_k, 1, h, kv_dim);
        Self::add_bias(&mut k, &self.k_bias, 1, kv_dim);
        let mut v = matmul_t(x, &self.w_v, 1, h, kv_dim);
        Self::add_bias(&mut v, &self.v_bias, 1, kv_dim);

        let attn_out = crate::ops::attention::cached_attention_decode(
            &q,
            &k,
            &v,
            num_heads,
            num_kv_heads,
            head_dim,
            cache_layer,
            pos,
            rope_cos,
            rope_sin,
            self.rope_style,
            &self.q_norm_weight,
            &self.k_norm_weight,
            self.attn_norm.eps() as f32,
            self.attention.window_size(),
        );

        // Output projection
        matmul_t(&attn_out, &self.w_o, 1, q_dim, h)
    }

    // ── Batched attention + cache populate (for prefill) ────────────

    /// Like [`apply_attention`] but also saves RoPE'd K and raw V to the
    /// cache for each position, so subsequent decode steps can reuse them.
    fn apply_attention_and_cache(
        &self,
        x: &[f32],
        seq_len: usize,
        cache_layer: &mut KVCacheLayer,
        rope_cos: &[f32],
        rope_sin: &[f32],
    ) -> Vec<f32> {
        let h = self.hidden_size;
        let num_heads = self.attention.num_heads();
        let num_kv_heads = self.attention.num_kv_heads();
        let head_dim = self.attention.head_dim();
        let q_dim = num_heads * head_dim;
        let kv_dim = num_kv_heads * head_dim;

        // Q, K, V projections (batched over all positions)
        let mut q = matmul_t(x, &self.w_q, seq_len, h, q_dim);
        Self::add_bias(&mut q, &self.q_bias, seq_len, q_dim);
        let mut k = matmul_t(x, &self.w_k, seq_len, h, kv_dim);
        Self::add_bias(&mut k, &self.k_bias, seq_len, kv_dim);
        let mut v = matmul_t(x, &self.w_v, seq_len, h, kv_dim);
        Self::add_bias(&mut v, &self.v_bias, seq_len, kv_dim);

        let attn_out = crate::ops::attention::cached_attention_prefill(
            &q,
            &k,
            &v,
            seq_len,
            num_heads,
            num_kv_heads,
            head_dim,
            cache_layer,
            rope_cos,
            rope_sin,
            self.rope_style,
            &self.q_norm_weight,
            &self.k_norm_weight,
            self.attn_norm.eps() as f32,
        );

        matmul_t(&attn_out, &self.w_o, seq_len, q_dim, h)
    }

    // ── FFN ──────────────────────────────────────────────────────────

    fn apply_ffn(&self, x: &[f32]) -> Vec<f32> {
        let h = self.hidden_size;
        let inter = self.ffn.intermediate_size();
        let tokens = x.len() / h;

        match self.ffn.name() {
            "SwiGLU" => {
                let gate = matmul_t(x, &self.ffn_gate, tokens, h, inter);
                let up = matmul_t(x, &self.ffn_up, tokens, h, inter);
                #[cfg(feature = "zig-ffi")]
                let hidden = {
                    let mut hidden = vec![0.0f32; tokens * inter];
                    let len = hidden.len();
                    let status = unsafe {
                        synapse_sys::syn_swiglu(hidden.as_mut_ptr(), gate.as_ptr(), up.as_ptr(), len)
                    };
                    debug_assert_eq!(status, synapse_sys::SYN_OK, "syn_swiglu failed: {status}");
                    hidden
                };
                #[cfg(not(feature = "zig-ffi"))]
                let hidden = crate::ops::pure_rust_ops::swiglu(&gate, &up);
                matmul_t(&hidden, &self.ffn_down, tokens, inter, h)
            }
            "GeGLU" => {
                let gate = matmul_t(x, &self.ffn_gate, tokens, h, inter);
                let up = matmul_t(x, &self.ffn_up, tokens, h, inter);
                let mut hidden = vec![0.0f32; tokens * inter];
                for i in 0..hidden.len() {
                    hidden[i] = gelu(gate[i]) * up[i];
                }
                matmul_t(&hidden, &self.ffn_down, tokens, inter, h)
            }
            _ => {
                // GELU or others: y = activation(x @ up^T) @ down^T
                let mut activated = matmul_t(x, &self.ffn_up, tokens, h, inter);
                for v in activated.iter_mut() {
                    *v = gelu(*v);
                }
                matmul_t(&activated, &self.ffn_down, tokens, inter, h)
            }
        }
    }
}

// ── Dispatch helpers (Metal-specific, kept here) ─────────────────────

/// Apply normalization dispatched through ComputeBackend.
#[cfg(feature = "metal")]
pub(crate) fn apply_norm_dispatch(
    x: &[f32],
    weight: &[f32],
    norm: &dyn NormVariant,
    hidden_size: usize,
    backend: &crate::metal::ComputeBackend,
) -> Vec<f32> {
    let eps = norm.eps() as f32;
    match norm.name() {
        "RMSNorm" => backend.rmsnorm(x, weight, eps, hidden_size),
        "LayerNorm" => layernorm(x, weight, eps, hidden_size),
        _ => x.to_vec(),
    }
}

#[cfg(test)]
#[path = "decoder_layer_tests.rs"]
mod tests;
