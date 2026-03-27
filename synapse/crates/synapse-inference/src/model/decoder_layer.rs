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
                let mut hidden = vec![0.0f32; tokens * inter];
                let len = hidden.len();
                let status = unsafe {
                    synapse_sys::syn_swiglu(hidden.as_mut_ptr(), gate.as_ptr(), up.as_ptr(), len)
                };
                debug_assert_eq!(status, synapse_sys::SYN_OK, "syn_swiglu failed: {status}");
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

// ── Test-only naive reference implementations ────────────────────────

/// Naive triple-loop reference implementation of y = A * B^T.
///
/// Kept for test comparison against the SIMD path.
#[cfg(test)]
pub(crate) fn matmul_t_naive(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
    debug_assert_eq!(a.len(), m * k, "matmul_t_naive: a.len() != m*k");
    debug_assert_eq!(b.len(), n * k, "matmul_t_naive: b.len() != n*k");
    let mut out = vec![0.0f32; m * n];
    for i in 0..m {
        let a_row = &a[i * k..(i + 1) * k];
        for j in 0..n {
            let b_row = &b[j * k..(j + 1) * k];
            let mut sum = 0.0f32;
            for d in 0..k {
                sum += a_row[d] * b_row[d];
            }
            out[i * n + j] = sum;
        }
    }
    out
}

/// Naive scalar RMS normalization (reference for test comparison).
///
/// Uses `black_box` on the accumulator to prevent LLVM auto-vectorization,
/// giving a fair scalar-vs-SIMD benchmark comparison.
#[cfg(test)]
#[inline(never)]
pub(crate) fn rmsnorm_naive(x: &[f32], weight: &[f32], eps: f32, hidden_size: usize) -> Vec<f32> {
    let n = x.len() / hidden_size;
    let mut out = vec![0.0f32; x.len()];
    for i in 0..n {
        let off = i * hidden_size;
        let slice = &x[off..off + hidden_size];
        let mut ms = 0.0f32;
        for j in 0..hidden_size {
            ms += slice[j] * slice[j];
            ms = std::hint::black_box(ms);
        }
        ms /= hidden_size as f32;
        let scale = 1.0 / (ms + eps).sqrt();
        for j in 0..hidden_size {
            out[off + j] = std::hint::black_box(slice[j] * scale) * weight[j];
        }
    }
    out
}

/// Naive scalar headwise RMS normalization (reference for test comparison).
#[cfg(test)]
pub(crate) fn apply_headwise_rmsnorm_naive(
    x: &[f32],
    weight: &[f32],
    rows: usize,
    heads: usize,
    head_dim: usize,
    eps: f32,
) -> Vec<f32> {
    if weight.is_empty() {
        return x.to_vec();
    }

    let mut out = vec![0.0f32; x.len()];
    let stride = heads * head_dim;
    for row in 0..rows {
        let row_offset = row * stride;
        for head in 0..heads {
            let head_offset = row_offset + head * head_dim;
            let slice = &x[head_offset..head_offset + head_dim];
            let ms = slice.iter().map(|v| v * v).sum::<f32>() / head_dim as f32;
            let scale = 1.0 / (ms + eps).sqrt();
            for idx in 0..head_dim {
                out[head_offset + idx] = slice[idx] * scale * weight[idx];
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ops::activation::silu;
    use crate::ops::norm::rmsnorm;

    /// Generate deterministic pseudo-random f32 values in [-1, 1].
    fn pseudo_rand(len: usize, seed: u64) -> Vec<f32> {
        let mut state = seed;
        (0..len)
            .map(|_| {
                // xorshift64
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                ((state as f64) / (u64::MAX as f64) * 2.0 - 1.0) as f32
            })
            .collect()
    }

    /// Assert element-wise closeness: |a - e| <= atol + rtol * |e|.
    /// Same semantics as numpy.allclose.
    fn assert_close(actual: &[f32], expected: &[f32], rtol: f32, label: &str) {
        const ATOL: f32 = 2e-5; // Allows for GEMV vs tiled SGEMM accumulation order differences
        assert_eq!(actual.len(), expected.len(), "{label}: length mismatch");
        for (i, (&a, &e)) in actual.iter().zip(expected.iter()).enumerate() {
            let diff = (a - e).abs();
            let bound = ATOL + rtol * e.abs();
            assert!(
                diff <= bound,
                "{label}[{i}]: simd={a}, naive={e}, diff={diff} > bound={bound} \
                 (atol={ATOL}, rtol={rtol})"
            );
        }
    }

    // ── Correctness tests: SIMD matches naive ─────────────────────────
    // Tolerance: SIMD tiling reorders accumulation vs the scalar loop,
    // causing O(k · eps) relative error.  Use 1e-5 for small k, 1e-4
    // for k ≥ 512 (consistent with the full-model 1e-4 requirement).

    #[test]
    fn matmul_simd_vs_naive_1x512_x_512x1024() {
        let (m, k, n) = (1, 512, 1024);
        let a = pseudo_rand(m * k, 42);
        let b = pseudo_rand(n * k, 123);
        let simd = matmul_t(&a, &b, m, k, n);
        let naive = matmul_t_naive(&a, &b, m, k, n);
        assert_close(&simd, &naive, 1e-4, "[1,512]x[512,1024]");
    }

    #[test]
    fn matmul_simd_vs_naive_4x1024_x_1024x3072() {
        let (m, k, n) = (4, 1024, 3072);
        let a = pseudo_rand(m * k, 7);
        let b = pseudo_rand(n * k, 99);
        let simd = matmul_t(&a, &b, m, k, n);
        let naive = matmul_t_naive(&a, &b, m, k, n);
        assert_close(&simd, &naive, 1e-4, "[4,1024]x[1024,3072]");
    }

    #[test]
    fn matmul_simd_vs_naive_128x1024_x_1024x1024() {
        let (m, k, n) = (128, 1024, 1024);
        let a = pseudo_rand(m * k, 314);
        let b = pseudo_rand(n * k, 159);
        let simd = matmul_t(&a, &b, m, k, n);
        let naive = matmul_t_naive(&a, &b, m, k, n);
        assert_close(&simd, &naive, 1e-4, "[128,1024]x[1024,1024]");
    }

    // ── Edge cases ───────────────────────────────────────────────────

    #[test]
    fn matmul_edge_m1_single_token() {
        let (m, k, n) = (1, 64, 128);
        let a = pseudo_rand(m * k, 1);
        let b = pseudo_rand(n * k, 2);
        let simd = matmul_t(&a, &b, m, k, n);
        let naive = matmul_t_naive(&a, &b, m, k, n);
        assert_close(&simd, &naive, 1e-5, "m=1 single token");
    }

    #[test]
    fn matmul_edge_k1() {
        let (m, k, n) = (8, 1, 16);
        let a = pseudo_rand(m * k, 10);
        let b = pseudo_rand(n * k, 20);
        let simd = matmul_t(&a, &b, m, k, n);
        let naive = matmul_t_naive(&a, &b, m, k, n);
        assert_close(&simd, &naive, 1e-5, "k=1");
    }

    #[test]
    fn matmul_edge_non_power_of_2() {
        let (m, k, n) = (13, 67, 101);
        let a = pseudo_rand(m * k, 55);
        let b = pseudo_rand(n * k, 77);
        let simd = matmul_t(&a, &b, m, k, n);
        let naive = matmul_t_naive(&a, &b, m, k, n);
        assert_close(&simd, &naive, 1e-4, "non-pow2 [13,67]x[67,101]");
    }

    #[test]
    fn matmul_edge_small_1x1() {
        let a = vec![3.0f32];
        let b = vec![5.0f32];
        let simd = matmul_t(&a, &b, 1, 1, 1);
        let naive = matmul_t_naive(&a, &b, 1, 1, 1);
        assert_close(&simd, &naive, 1e-5, "1x1");
    }

    // ── RMSNorm: SIMD vs naive correctness ────────────────────────────

    #[test]
    fn rmsnorm_simd_vs_naive_1x1024() {
        let hidden = 1024;
        let x = pseudo_rand(1 * hidden, 42);
        let w = pseudo_rand(hidden, 100)
            .iter()
            .map(|v| v.abs() + 0.1)
            .collect::<Vec<_>>();
        let simd = rmsnorm(&x, &w, 1e-5, hidden);
        let naive = rmsnorm_naive(&x, &w, 1e-5, hidden);
        assert_close(&simd, &naive, 1e-5, "rmsnorm [1,1024]");
    }

    #[test]
    fn rmsnorm_simd_vs_naive_4x1024() {
        let hidden = 1024;
        let x = pseudo_rand(4 * hidden, 7);
        let w = pseudo_rand(hidden, 200)
            .iter()
            .map(|v| v.abs() + 0.1)
            .collect::<Vec<_>>();
        let simd = rmsnorm(&x, &w, 1e-5, hidden);
        let naive = rmsnorm_naive(&x, &w, 1e-5, hidden);
        assert_close(&simd, &naive, 1e-5, "rmsnorm [4,1024]");
    }

    #[test]
    fn rmsnorm_simd_vs_naive_128x1024() {
        let hidden = 1024;
        let x = pseudo_rand(128 * hidden, 314);
        let w = pseudo_rand(hidden, 300)
            .iter()
            .map(|v| v.abs() + 0.1)
            .collect::<Vec<_>>();
        let simd = rmsnorm(&x, &w, 1e-5, hidden);
        let naive = rmsnorm_naive(&x, &w, 1e-5, hidden);
        assert_close(&simd, &naive, 1e-5, "rmsnorm [128,1024]");
    }

    #[test]
    fn rmsnorm_weighted_vs_unweighted() {
        let hidden = 64;
        let x = pseudo_rand(2 * hidden, 55);
        let ones = vec![1.0f32; hidden];
        let gamma = pseudo_rand(hidden, 77)
            .iter()
            .map(|v| v.abs() + 0.5)
            .collect::<Vec<_>>();

        let out_unit = rmsnorm(&x, &ones, 1e-5, hidden);
        let out_weighted = rmsnorm(&x, &gamma, 1e-5, hidden);

        // Weighted output should equal unit output * gamma element-wise.
        let out_unit_naive = rmsnorm_naive(&x, &ones, 1e-5, hidden);
        let out_weighted_naive = rmsnorm_naive(&x, &gamma, 1e-5, hidden);

        // Verify the naive weighted = naive_unit * gamma
        for i in 0..out_weighted_naive.len() {
            let j = i % hidden;
            let expected = out_unit_naive[i] * gamma[j];
            assert!(
                (out_weighted_naive[i] - expected).abs() < 1e-5,
                "naive weighted[{i}] mismatch: {} vs {}",
                out_weighted_naive[i],
                expected,
            );
        }

        // Verify SIMD matches naive for both
        assert_close(&out_unit, &out_unit_naive, 1e-5, "rmsnorm unit weight");
        assert_close(&out_weighted, &out_weighted_naive, 1e-5, "rmsnorm weighted");
    }

    #[test]
    fn rmsnorm_edge_hidden1_batch1() {
        let x = vec![3.0f32];
        let w = vec![2.0f32];
        let simd = rmsnorm(&x, &w, 1e-5, 1);
        let naive = rmsnorm_naive(&x, &w, 1e-5, 1);
        assert_close(&simd, &naive, 1e-5, "rmsnorm h=1 b=1");
    }

    #[test]
    fn headwise_rmsnorm_simd_vs_naive() {
        let (rows, heads, head_dim) = (4, 8, 128);
        let total = rows * heads * head_dim;
        let x = pseudo_rand(total, 42);
        let w = pseudo_rand(head_dim, 99)
            .iter()
            .map(|v| v.abs() + 0.1)
            .collect::<Vec<_>>();
        let eps = 1e-5;

        let simd = apply_headwise_rmsnorm(&x, &w, rows, heads, head_dim, eps);
        let naive = apply_headwise_rmsnorm_naive(&x, &w, rows, heads, head_dim, eps);
        assert_close(&simd, &naive, 1e-5, "headwise rmsnorm");
    }

    // ── Benchmark: RMSNorm SIMD >= 4× throughput vs naive ───────────

    #[test]
    fn bench_rmsnorm_simd_vs_naive_throughput() {
        if cfg!(debug_assertions) {
            eprintln!("Skipping rmsnorm throughput benchmark in debug mode");
            return;
        }

        let hidden = 1024;
        let batch = 1;
        let total = batch * hidden;
        let x = pseudo_rand(total, 42);
        let w = pseudo_rand(hidden, 99)
            .iter()
            .map(|v| v.abs() + 0.1)
            .collect::<Vec<_>>();

        // Warm up
        let _ = rmsnorm(&x, &w, 1e-5, hidden);
        let _ = rmsnorm_naive(&x, &w, 1e-5, hidden);

        let iters = 500;

        let t0 = std::time::Instant::now();
        for _ in 0..iters {
            let _ = rmsnorm_naive(&x, &w, 1e-5, hidden);
        }
        let naive_ns = t0.elapsed().as_nanos() as f64 / iters as f64;

        let t0 = std::time::Instant::now();
        for _ in 0..iters {
            let _ = rmsnorm(&x, &w, 1e-5, hidden);
        }
        let simd_ns = t0.elapsed().as_nanos() as f64 / iters as f64;

        let speedup = naive_ns / simd_ns;
        eprintln!(
            "rmsnorm [{batch},{hidden}]: naive={naive_ns:.0}ns, \
             simd={simd_ns:.0}ns, speedup={speedup:.1}×"
        );

        assert!(
            speedup >= 4.0,
            "RMSNorm SIMD speedup {speedup:.1}× is below 4× threshold \
             (naive={naive_ns:.0}ns, simd={simd_ns:.0}ns)"
        );
    }

    // ── Benchmark: matmul SIMD >= 4× throughput vs naive ────────────

    #[test]
    fn bench_simd_vs_naive_throughput() {
        // Only meaningful in release mode — skip in debug to avoid timeout.
        if cfg!(debug_assertions) {
            eprintln!("Skipping throughput benchmark in debug mode");
            return;
        }

        let (m, k, n) = (1024, 1024, 3072);
        let a = pseudo_rand(m * k, 42);
        let b = pseudo_rand(n * k, 99);

        // Warm up
        let _ = matmul_t(&a, &b, m, k, n);
        let _ = matmul_t_naive(&a, &b, m, k, n);

        let iters = 5;

        let t0 = std::time::Instant::now();
        for _ in 0..iters {
            let _ = matmul_t_naive(&a, &b, m, k, n);
        }
        let naive_ns = t0.elapsed().as_nanos() as f64 / iters as f64;

        let t0 = std::time::Instant::now();
        for _ in 0..iters {
            let _ = matmul_t(&a, &b, m, k, n);
        }
        let simd_ns = t0.elapsed().as_nanos() as f64 / iters as f64;

        let speedup = naive_ns / simd_ns;
        let flops = 2.0 * m as f64 * n as f64 * k as f64;
        let naive_gflops = flops / naive_ns;
        let simd_gflops = flops / simd_ns;

        eprintln!(
            "matmul_t [{m},{k}]×[{k},{n}]: naive={naive_gflops:.2} GFLOP/s, \
             simd={simd_gflops:.2} GFLOP/s, speedup={speedup:.1}×"
        );

        assert!(
            speedup >= 4.0,
            "SIMD speedup {speedup:.1}× is below 4× threshold \
             (naive={naive_gflops:.2}, simd={simd_gflops:.2} GFLOP/s)"
        );
    }

    // ── SwiGLU fused vs manual correctness ──────────────────────────

    /// Manual silu(gate)*up reference (scalar, not using FFI).
    fn swiglu_manual(gate: &[f32], up: &[f32]) -> Vec<f32> {
        gate.iter()
            .zip(up.iter())
            .map(|(&g, &u)| silu(g) * u)
            .collect()
    }

    /// Fused SwiGLU via syn_swiglu FFI.
    fn swiglu_fused(gate: &[f32], up: &[f32]) -> Vec<f32> {
        let len = gate.len();
        let mut out = vec![0.0f32; len];
        let status =
            unsafe { synapse_sys::syn_swiglu(out.as_mut_ptr(), gate.as_ptr(), up.as_ptr(), len) };
        assert_eq!(status, synapse_sys::SYN_OK, "syn_swiglu failed: {status}");
        out
    }

    #[test]
    fn swiglu_fused_vs_manual_1024() {
        let gate = pseudo_rand(1024, 42);
        let up = pseudo_rand(1024, 99);
        let fused = swiglu_fused(&gate, &up);
        let manual = swiglu_manual(&gate, &up);
        assert_close(&fused, &manual, 1e-5, "swiglu [1024]");
    }

    #[test]
    fn swiglu_fused_vs_manual_3072() {
        let gate = pseudo_rand(3072, 7);
        let up = pseudo_rand(3072, 13);
        let fused = swiglu_fused(&gate, &up);
        let manual = swiglu_manual(&gate, &up);
        assert_close(&fused, &manual, 1e-5, "swiglu [3072]");
    }

    #[test]
    fn swiglu_fused_vs_manual_11008() {
        let gate = pseudo_rand(11008, 314);
        let up = pseudo_rand(11008, 159);
        let fused = swiglu_fused(&gate, &up);
        let manual = swiglu_manual(&gate, &up);
        assert_close(&fused, &manual, 1e-5, "swiglu [11008]");
    }

    // ── GeGLU / GELU paths still work ───────────────────────────────

    #[test]
    fn geglu_path_correctness() {
        let len = 1024;
        let gate = pseudo_rand(len, 55);
        let up = pseudo_rand(len, 77);
        let mut result = vec![0.0f32; len];
        for i in 0..len {
            result[i] = gelu(gate[i]) * up[i];
        }
        // Sanity: GeGLU output should differ from SwiGLU output
        let swiglu_result = swiglu_manual(&gate, &up);
        let differs = result
            .iter()
            .zip(swiglu_result.iter())
            .any(|(a, b)| (a - b).abs() > 1e-3);
        assert!(differs, "GeGLU and SwiGLU outputs should differ");

        // Verify GeGLU values are reasonable (not NaN/Inf)
        for (i, &v) in result.iter().enumerate() {
            assert!(v.is_finite(), "GeGLU[{i}] is not finite: {v}");
        }
    }

    #[test]
    fn gelu_path_correctness() {
        let len = 1024;
        let x = pseudo_rand(len, 42);
        let mut activated = x.clone();
        for v in activated.iter_mut() {
            *v = gelu(*v);
        }
        // GELU should be approximately x for large positive x, ~0 for large negative x
        for (i, (&orig, &act)) in x.iter().zip(activated.iter()).enumerate() {
            assert!(act.is_finite(), "GELU[{i}] is not finite: {act}");
            if orig > 3.0 {
                // For large positive inputs, gelu(x) ≈ x
                assert!(
                    (act - orig).abs() < 0.1,
                    "GELU[{i}]: expected ~{orig}, got {act}"
                );
            }
            if orig < -3.0 {
                // For large negative inputs, gelu(x) ≈ 0
                assert!(act.abs() < 0.1, "GELU[{i}]: expected ~0, got {act}");
            }
        }
    }

    // ── Benchmark: syn_swiglu >= 2× throughput vs separate silu+mul ─

    /// Scalar SwiGLU with serial dependency chain.
    ///
    /// Loop-carried `dep` prevents LLVM from vectorizing or pipelining
    /// multiple `exp()` calls — same strategy as `rmsnorm_naive`.
    /// The `dep * 0.0` is a no-op on finite values but the compiler
    /// cannot prove finiteness through `black_box`, so the chain holds.
    #[inline(never)]
    fn swiglu_separate_scalar(dst: &mut [f32], gate: &[f32], up: &[f32]) {
        // Pass 1: silu(gate) → dst
        let mut dep = 0.0f32;
        for i in 0..dst.len() {
            let g = gate[i] + dep * 0.0;
            dst[i] = g / (1.0 + (-g).exp());
            dep = std::hint::black_box(dst[i]);
        }
        // Pass 2: dst *= up
        dep = 0.0;
        for i in 0..dst.len() {
            let d = dst[i] + dep * 0.0;
            dst[i] = d * up[i];
            dep = std::hint::black_box(dst[i]);
        }
    }

    #[test]
    fn bench_swiglu_fused_vs_separate_throughput() {
        if cfg!(debug_assertions) {
            eprintln!("Skipping swiglu throughput benchmark in debug mode");
            return;
        }

        let len = 3072; // [1, 3072] single-token FFN intermediate
        let gate = pseudo_rand(len, 42);
        let up = pseudo_rand(len, 99);
        let mut dst = vec![0.0f32; len];

        // Warm up
        for _ in 0..100 {
            swiglu_separate_scalar(&mut dst, &gate, &up);
            unsafe {
                synapse_sys::syn_swiglu(dst.as_mut_ptr(), gate.as_ptr(), up.as_ptr(), len);
            }
        }

        // Use min-of-runs: noise only adds latency, minimum is most
        // representative (standard microbenchmark practice).
        let runs = 5;
        let iters_per_run = 2000;

        let mut best_separate = f64::MAX;
        for _ in 0..runs {
            let t0 = std::time::Instant::now();
            for _ in 0..iters_per_run {
                swiglu_separate_scalar(&mut dst, &gate, &up);
            }
            let ns = t0.elapsed().as_nanos() as f64 / iters_per_run as f64;
            if ns < best_separate {
                best_separate = ns;
            }
        }

        let mut best_fused = f64::MAX;
        for _ in 0..runs {
            let t0 = std::time::Instant::now();
            for _ in 0..iters_per_run {
                unsafe {
                    synapse_sys::syn_swiglu(dst.as_mut_ptr(), gate.as_ptr(), up.as_ptr(), len);
                }
            }
            let ns = t0.elapsed().as_nanos() as f64 / iters_per_run as f64;
            if ns < best_fused {
                best_fused = ns;
            }
        }

        let speedup = best_separate / best_fused;
        eprintln!(
            "swiglu [1,{len}]: separate={best_separate:.0}ns, \
             fused={best_fused:.0}ns, speedup={speedup:.1}×"
        );

        assert!(
            speedup >= 2.0,
            "syn_swiglu speedup {speedup:.1}× is below 2× threshold \
             (separate={best_separate:.0}ns, fused={best_fused:.0}ns)"
        );
    }

    // ── forward_one / KV-cache tests ────────────────────────────────

    use crate::kv_cache::KVCacheLayer;
    use crate::weight_loading::AlignedBuffer;

    /// Minimal AttentionVariant for tests.
    #[derive(Debug)]
    struct TestAttn {
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
    }
    impl crate::registry::AttentionVariant for TestAttn {
        fn num_heads(&self) -> usize {
            self.num_heads
        }
        fn head_dim(&self) -> usize {
            self.head_dim
        }
        fn num_kv_heads(&self) -> usize {
            self.num_kv_heads
        }
        fn name(&self) -> &str {
            "GQA"
        }
    }

    /// Minimal NormVariant for tests (delegates to RMSNorm path via name).
    #[derive(Debug)]
    struct TestNorm {
        eps: f64,
    }
    impl crate::registry::NormVariant for TestNorm {
        fn eps(&self) -> f64 {
            self.eps
        }
        fn name(&self) -> &str {
            "RMSNorm"
        }
    }

    /// Minimal FFNVariant for tests (dispatches to SwiGLU path).
    #[derive(Debug)]
    struct TestFFN {
        inter: usize,
    }
    impl crate::registry::FFNVariant for TestFFN {
        fn intermediate_size(&self) -> usize {
            self.inter
        }
        fn name(&self) -> &str {
            "SwiGLU"
        }
    }

    /// Build test RoPE cos/sin tables.
    fn make_test_rope(head_dim: usize, max_pos: usize) -> (Vec<f32>, Vec<f32>) {
        let half_d = head_dim / 2;
        let base: f32 = 10_000.0;
        let mut cos = vec![0.0f32; max_pos * half_d];
        let mut sin = vec![0.0f32; max_pos * half_d];
        for pos in 0..max_pos {
            for i in 0..half_d {
                let freq = 1.0 / base.powf(2.0 * i as f32 / head_dim as f32);
                let angle = pos as f32 * freq;
                cos[pos * half_d + i] = angle.cos();
                sin[pos * half_d + i] = angle.sin();
            }
        }
        (cos, sin)
    }

    /// Build a test DecoderLayer with pseudo-random weights.
    fn make_test_layer(
        hidden: usize,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        inter: usize,
    ) -> DecoderLayer {
        let q_dim = num_heads * head_dim;
        let kv_dim = num_kv_heads * head_dim;
        DecoderLayer {
            attn_norm: Box::new(TestNorm { eps: 1e-5 }),
            attention: Box::new(TestAttn {
                num_heads,
                num_kv_heads,
                head_dim,
            }),
            ffn_norm: Box::new(TestNorm { eps: 1e-5 }),
            ffn: Box::new(TestFFN { inter }),
            hidden_size: hidden,
            has_head_norms: true,
            rope_style: RoPEStyle::default(),
            attn_norm_weight: AlignedBuffer::from_vec(
                pseudo_rand(hidden, 1000)
                    .iter()
                    .map(|v| v.abs() + 0.1)
                    .collect(),
            ),
            w_q: AlignedBuffer::from_vec(pseudo_rand(q_dim * hidden, 2000)),
            w_k: AlignedBuffer::from_vec(pseudo_rand(kv_dim * hidden, 3000)),
            w_v: AlignedBuffer::from_vec(pseudo_rand(kv_dim * hidden, 4000)),
            w_o: AlignedBuffer::from_vec(pseudo_rand(hidden * q_dim, 5000)),
            q_norm_weight: AlignedBuffer::from_vec(
                pseudo_rand(head_dim, 6000)
                    .iter()
                    .map(|v| v.abs() + 0.1)
                    .collect(),
            ),
            k_norm_weight: AlignedBuffer::from_vec(
                pseudo_rand(head_dim, 7000)
                    .iter()
                    .map(|v| v.abs() + 0.1)
                    .collect(),
            ),
            q_bias: AlignedBuffer::new_zeroed(0),
            k_bias: AlignedBuffer::new_zeroed(0),
            v_bias: AlignedBuffer::new_zeroed(0),
            ffn_norm_weight: AlignedBuffer::from_vec(
                pseudo_rand(hidden, 8000)
                    .iter()
                    .map(|v| v.abs() + 0.1)
                    .collect(),
            ),
            ffn_gate: AlignedBuffer::from_vec(pseudo_rand(inter * hidden, 9000)),
            ffn_up: AlignedBuffer::from_vec(pseudo_rand(inter * hidden, 10000)),
            ffn_down: AlignedBuffer::from_vec(pseudo_rand(hidden * inter, 11000)),
        }
    }

    /// forward_one() output at position N must match the last position of
    /// forward() when given the same input sequence.
    #[test]
    fn forward_one_matches_forward_last_position() {
        let (hidden, num_heads, num_kv_heads, head_dim, inter) = (64, 4, 4, 16, 128);
        let layer = make_test_layer(hidden, num_heads, num_kv_heads, head_dim, inter);
        let (rope_cos, rope_sin) = make_test_rope(head_dim, 64);
        let seq_len = 5;

        // Build a multi-token input: [seq_len, hidden]
        let input = pseudo_rand(seq_len * hidden, 42);

        // Full forward pass
        let full_out = layer.forward(&input, seq_len, &rope_cos, &rope_sin);

        // Incremental forward_one for each token
        let mut cache = KVCacheLayer::new(64, num_kv_heads, head_dim).unwrap();
        let mut last_out = vec![0.0f32; hidden];
        for t in 0..seq_len {
            let token = &input[t * hidden..(t + 1) * hidden];
            last_out = layer.forward_one(token, &mut cache, t, &rope_cos, &rope_sin);
        }

        // Compare: forward_one output at last position should match
        // forward() output at the last position.
        let full_last = &full_out[(seq_len - 1) * hidden..seq_len * hidden];
        assert_close(
            &last_out,
            full_last,
            1e-4,
            "forward_one vs forward last pos",
        );
    }

    /// KV-cache values after forward_one() must match K/V computed by the
    /// full forward() path at the same positions.
    #[test]
    fn kv_cache_values_match_full_forward() {
        let (hidden, num_heads, num_kv_heads, head_dim, inter) = (64, 4, 4, 16, 128);
        let layer = make_test_layer(hidden, num_heads, num_kv_heads, head_dim, inter);
        let kv_dim = num_kv_heads * head_dim;
        let seq_len = 4;

        let input = pseudo_rand(seq_len * hidden, 77);
        let h = hidden;
        let eps = 1e-5f32;

        // Compute what full forward would produce for K/V:
        // 1. norm the input
        let normed = apply_norm(&input, &layer.attn_norm_weight, &*layer.attn_norm, h);
        // 2. project K, V
        let k_full = matmul_t(&normed, &layer.w_k, seq_len, h, kv_dim);
        let v_full = matmul_t(&normed, &layer.w_v, seq_len, h, kv_dim);
        // 3. apply headwise norm to K (V is raw)
        let k_full_normed = apply_headwise_rmsnorm(
            &k_full,
            &layer.k_norm_weight,
            seq_len,
            num_kv_heads,
            head_dim,
            eps,
        );

        // Apply RoPE to the normed K (matching what forward_one does)
        let (rope_cos, rope_sin) = make_test_rope(head_dim, 64);
        let mut k_full_roped = k_full_normed.clone();
        apply_rope_inplace(
            &mut k_full_roped,
            &rope_cos,
            &rope_sin,
            seq_len,
            num_kv_heads,
            head_dim,
            0,
            RoPEStyle::RotateHalf,
        );

        // Now run forward_one incrementally and collect cache contents
        let mut cache = KVCacheLayer::new(64, num_kv_heads, head_dim).unwrap();
        for t in 0..seq_len {
            let token = &input[t * h..(t + 1) * h];
            let _ = layer.forward_one(token, &mut cache, t, &rope_cos, &rope_sin);
        }

        let (cached_k, cached_v, cached_len) = cache.slice().unwrap();
        assert_eq!(cached_len, seq_len);

        // Cached K should match RoPE'd normed K from full forward
        assert_close(
            cached_k,
            &k_full_roped,
            1e-5,
            "cached K vs full RoPE'd normed K",
        );
        // Cached V should match raw V from full forward
        assert_close(cached_v, &v_full, 1e-5, "cached V vs full raw V");
    }

    /// GQA test: n_kv_heads < n_heads. forward_one must still match
    /// the last position of forward().
    #[test]
    fn forward_one_gqa_heads_expansion() {
        // 8 Q-heads, 2 KV-heads → groups = 4
        let (hidden, num_heads, num_kv_heads, head_dim, inter) = (64, 8, 2, 8, 128);
        let layer = make_test_layer(hidden, num_heads, num_kv_heads, head_dim, inter);
        let (rope_cos, rope_sin) = make_test_rope(head_dim, 64);
        let seq_len = 6;

        let input = pseudo_rand(seq_len * hidden, 314);

        // Full forward
        let full_out = layer.forward(&input, seq_len, &rope_cos, &rope_sin);

        // Incremental forward_one
        let mut cache = KVCacheLayer::new(64, num_kv_heads, head_dim).unwrap();
        let mut last_out = vec![0.0f32; hidden];
        for t in 0..seq_len {
            let token = &input[t * hidden..(t + 1) * hidden];
            last_out = layer.forward_one(token, &mut cache, t, &rope_cos, &rope_sin);
        }

        let full_last = &full_out[(seq_len - 1) * hidden..seq_len * hidden];
        assert_close(&last_out, full_last, 1e-4, "forward_one GQA (8h/2kv)");
    }

    #[test]
    fn rope_interleaved_vs_rotate_half_differ() {
        // Verify that the two RoPE styles produce different outputs for the same input.
        let head_dim = 8;
        let half_d = head_dim / 2;
        let num_heads = 1;
        let seq_len = 1;
        let max_pos = 4;

        // Precompute tables
        let base = 10_000.0f32;
        let mut cos = vec![0.0f32; max_pos * half_d];
        let mut sin = vec![0.0f32; max_pos * half_d];
        for pos in 0..max_pos {
            for i in 0..half_d {
                let freq = 1.0 / base.powf(2.0 * i as f32 / head_dim as f32);
                let angle = pos as f32 * freq;
                cos[pos * half_d + i] = angle.cos();
                sin[pos * half_d + i] = angle.sin();
            }
        }

        // Test at pos=2 where angles are non-trivial
        let input: Vec<f32> = (1..=head_dim).map(|x| x as f32).collect();

        let mut rh = input.clone();
        apply_rope_inplace(
            &mut rh,
            &cos,
            &sin,
            seq_len,
            num_heads,
            head_dim,
            2,
            RoPEStyle::RotateHalf,
        );

        let mut il = input.clone();
        apply_rope_inplace(
            &mut il,
            &cos,
            &sin,
            seq_len,
            num_heads,
            head_dim,
            2,
            RoPEStyle::Interleaved,
        );

        // Outputs should differ (different dimension pairing)
        assert_ne!(
            rh, il,
            "RotateHalf and Interleaved should produce different results"
        );
    }

    #[test]
    fn rope_interleaved_roundtrip() {
        // Applying interleaved RoPE at pos=0 with angle=0 should be identity.
        let head_dim = 8;
        let half_d = head_dim / 2;
        // At pos=0, all angles are 0 → cos=1, sin=0 → identity
        let cos = vec![1.0f32; half_d];
        let sin = vec![0.0f32; half_d];

        let input: Vec<f32> = (1..=head_dim).map(|x| x as f32).collect();
        let mut out = input.clone();
        apply_rope_inplace(
            &mut out,
            &cos,
            &sin,
            1,
            1,
            head_dim,
            0,
            RoPEStyle::Interleaved,
        );
        assert_eq!(out, input, "RoPE at pos=0 should be identity");
    }

    // ── Sliding window attention tests ──────────────────────────────

    /// AttentionVariant with sliding window support for tests.
    #[derive(Debug)]
    struct TestSlidingWindowAttn {
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        window_size: usize,
    }
    impl crate::registry::AttentionVariant for TestSlidingWindowAttn {
        fn num_heads(&self) -> usize {
            self.num_heads
        }
        fn head_dim(&self) -> usize {
            self.head_dim
        }
        fn num_kv_heads(&self) -> usize {
            self.num_kv_heads
        }
        fn window_size(&self) -> Option<usize> {
            Some(self.window_size)
        }
        fn name(&self) -> &str {
            "SlidingWindow"
        }
    }

    /// Build a test DecoderLayer with sliding window attention.
    fn make_test_layer_sliding_window(
        hidden: usize,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        inter: usize,
        window_size: usize,
    ) -> DecoderLayer {
        let q_dim = num_heads * head_dim;
        let kv_dim = num_kv_heads * head_dim;
        DecoderLayer {
            attn_norm: Box::new(TestNorm { eps: 1e-5 }),
            attention: Box::new(TestSlidingWindowAttn {
                num_heads,
                num_kv_heads,
                head_dim,
                window_size,
            }),
            ffn_norm: Box::new(TestNorm { eps: 1e-5 }),
            ffn: Box::new(TestFFN { inter }),
            hidden_size: hidden,
            has_head_norms: true,
            rope_style: RoPEStyle::default(),
            attn_norm_weight: AlignedBuffer::from_vec(
                pseudo_rand(hidden, 1000)
                    .iter()
                    .map(|v| v.abs() + 0.1)
                    .collect(),
            ),
            w_q: AlignedBuffer::from_vec(pseudo_rand(q_dim * hidden, 2000)),
            w_k: AlignedBuffer::from_vec(pseudo_rand(kv_dim * hidden, 3000)),
            w_v: AlignedBuffer::from_vec(pseudo_rand(kv_dim * hidden, 4000)),
            w_o: AlignedBuffer::from_vec(pseudo_rand(hidden * q_dim, 5000)),
            q_norm_weight: AlignedBuffer::from_vec(
                pseudo_rand(head_dim, 6000)
                    .iter()
                    .map(|v| v.abs() + 0.1)
                    .collect(),
            ),
            k_norm_weight: AlignedBuffer::from_vec(
                pseudo_rand(head_dim, 7000)
                    .iter()
                    .map(|v| v.abs() + 0.1)
                    .collect(),
            ),
            q_bias: AlignedBuffer::new_zeroed(0),
            k_bias: AlignedBuffer::new_zeroed(0),
            v_bias: AlignedBuffer::new_zeroed(0),
            ffn_norm_weight: AlignedBuffer::from_vec(
                pseudo_rand(hidden, 8000)
                    .iter()
                    .map(|v| v.abs() + 0.1)
                    .collect(),
            ),
            ffn_gate: AlignedBuffer::from_vec(pseudo_rand(inter * hidden, 9000)),
            ffn_up: AlignedBuffer::from_vec(pseudo_rand(inter * hidden, 10000)),
            ffn_down: AlignedBuffer::from_vec(pseudo_rand(hidden * inter, 11000)),
        }
    }

    /// Sliding window decode: with window_size=4 and 8 cached positions,
    /// forward_one at position 8 should only attend to the last 4 positions
    /// (positions 5-8), producing a different result than full attention.
    #[test]
    fn sliding_window_attention_limits_decode_context() {
        let (hidden, num_heads, num_kv_heads, head_dim, inter) = (64, 4, 4, 16, 128);
        let window_size = 4;
        let total_positions = 8;

        // Build two layers with identical weights: one with sliding window, one without
        let sw_layer = make_test_layer_sliding_window(
            hidden,
            num_heads,
            num_kv_heads,
            head_dim,
            inter,
            window_size,
        );
        let full_layer = make_test_layer(hidden, num_heads, num_kv_heads, head_dim, inter);

        let (rope_cos, rope_sin) = make_test_rope(head_dim, 64);

        // Generate input tokens
        let input = pseudo_rand((total_positions + 1) * hidden, 42);

        // Fill both caches with 8 positions
        let mut sw_cache = KVCacheLayer::new(64, num_kv_heads, head_dim).unwrap();
        let mut full_cache = KVCacheLayer::new(64, num_kv_heads, head_dim).unwrap();

        for t in 0..total_positions {
            let token = &input[t * hidden..(t + 1) * hidden];
            let _ = sw_layer.forward_one(token, &mut sw_cache, t, &rope_cos, &rope_sin);
            let _ = full_layer.forward_one(token, &mut full_cache, t, &rope_cos, &rope_sin);
        }

        // Both caches should have 8 entries
        let (_, _, sw_len) = sw_cache.slice().unwrap();
        let (_, _, full_len) = full_cache.slice().unwrap();
        assert_eq!(sw_len, total_positions);
        assert_eq!(full_len, total_positions);

        // Now decode at position 8 with both layers
        let decode_token = &input[total_positions * hidden..(total_positions + 1) * hidden];
        let sw_out = sw_layer.forward_one(
            decode_token,
            &mut sw_cache,
            total_positions,
            &rope_cos,
            &rope_sin,
        );
        let full_out = full_layer.forward_one(
            decode_token,
            &mut full_cache,
            total_positions,
            &rope_cos,
            &rope_sin,
        );

        // The outputs should differ because sliding window attends to only
        // the last 4 positions while full attention attends to all 9.
        let differs = sw_out
            .iter()
            .zip(full_out.iter())
            .any(|(a, b)| (a - b).abs() > 1e-5);
        assert!(
            differs,
            "Sliding window (ws=4) output should differ from full attention with 9 cached positions"
        );

        // Verify the cache still has all 9 entries (sliding window only limits
        // attention, not cache storage)
        let (_, _, sw_len_after) = sw_cache.slice().unwrap();
        assert_eq!(
            sw_len_after,
            total_positions + 1,
            "Cache should still store all positions, not be truncated by sliding window"
        );

        // Additional verification: sliding window with window_size >= seq_len
        // should behave identically to full attention
        let big_window_layer = make_test_layer_sliding_window(
            hidden,
            num_heads,
            num_kv_heads,
            head_dim,
            inter,
            100, // window >> total positions
        );
        let mut big_cache = KVCacheLayer::new(64, num_kv_heads, head_dim).unwrap();
        let mut full_cache2 = KVCacheLayer::new(64, num_kv_heads, head_dim).unwrap();

        for t in 0..total_positions {
            let token = &input[t * hidden..(t + 1) * hidden];
            let _ = big_window_layer.forward_one(token, &mut big_cache, t, &rope_cos, &rope_sin);
            let _ = full_layer.forward_one(token, &mut full_cache2, t, &rope_cos, &rope_sin);
        }

        let big_out = big_window_layer.forward_one(
            decode_token,
            &mut big_cache,
            total_positions,
            &rope_cos,
            &rope_sin,
        );
        let full_out2 = full_layer.forward_one(
            decode_token,
            &mut full_cache2,
            total_positions,
            &rope_cos,
            &rope_sin,
        );

        assert_close(
            &big_out,
            &full_out2,
            1e-5,
            "Sliding window with window >= seq_len should match full attention",
        );
    }
}
