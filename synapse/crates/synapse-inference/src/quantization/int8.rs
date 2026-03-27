use std::mem;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::OnceLock;
use std::time::Instant;

use crate::config::position::RoPEStyle;
use crate::config::ModelConfig;
use crate::kv_cache::{KVCache, KVCacheLayer};
use crate::model::causal_lm::ModelOutput;
use crate::model::CausalLM;
use crate::ops::activation::{gelu, silu, softmax_slice};
use crate::ops::matmul::matmul_t;
use crate::ops::norm::{apply_headwise_rmsnorm, apply_norm};
use crate::ops::rope::apply_rope_inplace;
use crate::ops::vector::{add_vecs, add_vecs_inplace};
use crate::quantization::QuantizedLinear;
use crate::registry::{
    create_attention, create_ffn, create_norm, AttentionVariant, FFNVariant, NormVariant,
};

/// A quantized decoder layer with INT8 linear weights.
///
/// Norm weights stay f32 (small). All linear projections (Q/K/V/O, FFN)
/// are quantized to INT8 with per-channel scales.
pub struct QuantizedDecoderLayer {
    pub attn_norm: Box<dyn NormVariant>,
    pub attention: Box<dyn AttentionVariant>,
    pub ffn_norm: Box<dyn NormVariant>,
    pub ffn: Box<dyn FFNVariant>,
    pub hidden_size: usize,
    pub rope_style: RoPEStyle,

    pub attn_norm_weight: Vec<f32>,
    pub q_norm_weight: Vec<f32>,
    pub k_norm_weight: Vec<f32>,
    pub q_bias: Vec<f32>,
    pub k_bias: Vec<f32>,
    pub v_bias: Vec<f32>,
    pub ffn_norm_weight: Vec<f32>,

    pub w_q: QuantizedLinear,
    pub w_k: QuantizedLinear,
    pub w_v: QuantizedLinear,
    pub w_o: QuantizedLinear,
    pub ffn_gate: QuantizedLinear,
    pub ffn_up: QuantizedLinear,
    pub ffn_down: QuantizedLinear,
}

/// A causal language model with INT8-quantized linear layers.
pub struct QuantizedCausalLM {
    pub config: ModelConfig,
    pub layers: Vec<QuantizedDecoderLayer>,
    pub final_norm: Box<dyn NormVariant>,
    pub embed_tokens: Vec<f32>,
    pub final_norm_weight: Vec<f32>,
    pub lm_head_weight: Option<Vec<f32>>,
    /// Quantized LM head for fast INT8 vocabulary projection.
    pub lm_head_quantized: Option<QuantizedLinear>,
    pub rope_cos: Vec<f32>,
    pub rope_sin: Vec<f32>,
}

fn decode_profile_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var_os("SYNAPSE_PROFILE_INT8_DECODE").is_some())
}

fn should_emit_decode_profile() -> bool {
    static EMITTED: AtomicBool = AtomicBool::new(false);
    decode_profile_enabled() && !EMITTED.swap(true, Ordering::Relaxed)
}

/// Quantize all Linear layers in a CausalLM to weight-only INT8.
///
/// Embedding, norm weights, and lm_head stay f32. All attention and FFN
/// projection matrices are quantized per-channel.
pub fn quantize_model(model: &CausalLM) -> QuantizedCausalLM {
    let cfg = &model.config;
    let h = cfg.architecture.hidden_size;
    let q_dim = cfg.attention.num_heads() * cfg.attention.head_dim();
    let kv_dim = cfg.attention.num_kv_heads() * cfg.attention.head_dim();
    let inter = cfg.ffn.intermediate_size();

    let layers = model
        .layers
        .iter()
        .map(|layer| {
            let w_q = QuantizedLinear::from_f32(&layer.w_q, q_dim, h);
            let w_k = QuantizedLinear::from_f32(&layer.w_k, kv_dim, h);
            let w_v = QuantizedLinear::from_f32(&layer.w_v, kv_dim, h);
            let w_o = QuantizedLinear::from_f32(&layer.w_o, h, q_dim);

            let ffn_gate = if !layer.ffn_gate.is_empty() {
                QuantizedLinear::from_f32(&layer.ffn_gate, inter, h)
            } else {
                QuantizedLinear::empty()
            };
            let ffn_up = QuantizedLinear::from_f32(&layer.ffn_up, inter, h);
            let ffn_down = QuantizedLinear::from_f32(&layer.ffn_down, h, inter);

            QuantizedDecoderLayer {
                attn_norm: create_norm(&cfg.norm),
                attention: create_attention(&cfg.attention),
                ffn_norm: create_norm(&cfg.norm),
                ffn: create_ffn(&cfg.ffn),
                hidden_size: h,
                rope_style: layer.rope_style,
                attn_norm_weight: layer.attn_norm_weight.to_vec(),
                q_norm_weight: layer.q_norm_weight.to_vec(),
                k_norm_weight: layer.k_norm_weight.to_vec(),
                q_bias: layer.q_bias.to_vec(),
                k_bias: layer.k_bias.to_vec(),
                v_bias: layer.v_bias.to_vec(),
                ffn_norm_weight: layer.ffn_norm_weight.to_vec(),
                w_q,
                w_k,
                w_v,
                w_o,
                ffn_gate,
                ffn_up,
                ffn_down,
            }
        })
        .collect();

    // Quantize LM head if it exists (not tied to embeddings)
    let lm_head_quantized = model
        .lm_head_weight
        .as_ref()
        .map(|w| QuantizedLinear::from_f32(w, cfg.architecture.vocab_size, h));
    // For tied embeddings, quantize from embed_tokens
    let lm_head_quantized = lm_head_quantized.or_else(|| {
        Some(QuantizedLinear::from_f32(
            &model.embed_tokens,
            cfg.architecture.vocab_size,
            h,
        ))
    });

    QuantizedCausalLM {
        config: model.config.clone(),
        layers,
        final_norm: create_norm(&cfg.norm),
        embed_tokens: model.embed_tokens.to_vec(),
        final_norm_weight: model.final_norm_weight.to_vec(),
        lm_head_weight: model.lm_head_weight.as_ref().map(|w| w.to_vec()),
        lm_head_quantized,
        rope_cos: model.rope_cos.clone(),
        rope_sin: model.rope_sin.clone(),
    }
}

// ── QuantizedDecoderLayer forward ────────────────────────────────────

impl QuantizedDecoderLayer {
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

    /// Pre-norm forward: norm→attention→residual→norm→FFN→residual.
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

    /// Prefill forward: batched attention + KV cache populate.
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

    /// Single-token decode using KV cache.
    pub fn forward_one(
        &self,
        hidden: &[f32],
        cache_layer: &mut KVCacheLayer,
        pos: usize,
        rope_cos: &[f32],
        rope_sin: &[f32],
    ) -> Vec<f32> {
        self.forward_one_impl(hidden, cache_layer, pos, rope_cos, rope_sin, None)
    }

    fn forward_one_profiled(
        &self,
        hidden: &[f32],
        cache_layer: &mut KVCacheLayer,
        pos: usize,
        rope_cos: &[f32],
        rope_sin: &[f32],
        layer_idx: usize,
    ) -> Vec<f32> {
        self.forward_one_impl(hidden, cache_layer, pos, rope_cos, rope_sin, Some(layer_idx))
    }

    fn forward_one_impl(
        &self,
        hidden: &[f32],
        cache_layer: &mut KVCacheLayer,
        pos: usize,
        rope_cos: &[f32],
        rope_sin: &[f32],
        profile_layer_idx: Option<usize>,
    ) -> Vec<f32> {
        let h = self.hidden_size;

        // 1. Attention sub-layer
        let attn_norm_start = profile_layer_idx.map(|_| Instant::now());
        let normed = apply_norm(hidden, &self.attn_norm_weight, &*self.attn_norm, h);
        let attn_norm_elapsed = attn_norm_start.map(|start| start.elapsed());
        let attention_start = profile_layer_idx.map(|_| Instant::now());
        let attn_out = self.apply_attention_cached(&normed, cache_layer, pos, rope_cos, rope_sin);
        let attention_elapsed = attention_start.map(|start| start.elapsed());
        let mut residual = add_vecs(hidden, &attn_out);

        // 2. FFN sub-layer
        let ffn_norm_start = profile_layer_idx.map(|_| Instant::now());
        let normed = apply_norm(&residual, &self.ffn_norm_weight, &*self.ffn_norm, h);
        let ffn_norm_elapsed = ffn_norm_start.map(|start| start.elapsed());
        let ffn_start = profile_layer_idx.map(|_| Instant::now());
        let ffn_out = self.apply_ffn(&normed);
        let ffn_elapsed = ffn_start.map(|start| start.elapsed());
        add_vecs_inplace(&mut residual, &ffn_out);

        if let Some(layer_idx) = profile_layer_idx {
            eprintln!(
                "INT8 decode layer[{layer_idx:02}]: attn_norm={:.1}ms attention={:.1}ms ffn_norm={:.1}ms ffn={:.1}ms",
                attn_norm_elapsed.unwrap().as_secs_f64() * 1000.0,
                attention_elapsed.unwrap().as_secs_f64() * 1000.0,
                ffn_norm_elapsed.unwrap().as_secs_f64() * 1000.0,
                ffn_elapsed.unwrap().as_secs_f64() * 1000.0,
            );
        }

        residual
    }

    fn apply_attention(
        &self,
        x: &[f32],
        seq_len: usize,
        rope_cos: &[f32],
        rope_sin: &[f32],
    ) -> Vec<f32> {
        let num_heads = self.attention.num_heads();
        let num_kv_heads = self.attention.num_kv_heads();
        let head_dim = self.attention.head_dim();
        let q_dim = num_heads * head_dim;
        let kv_dim = num_kv_heads * head_dim;
        let groups = num_heads / num_kv_heads;
        let scale = 1.0 / (head_dim as f32).sqrt();

        // Quantize x once and share across Q/K/V projections.
        let k_dim = self.hidden_size;
        let (x_int8, scales_x) = synapse_core::quantize_per_channel_int8(x, seq_len, k_dim)
            .expect("quantize_per_channel_int8 failed for attention input");
        let mut q = self.w_q.forward_pre_quantized(&x_int8, &scales_x, seq_len);
        Self::add_bias(&mut q, &self.q_bias, seq_len, q_dim);
        let mut k = self.w_k.forward_pre_quantized(&x_int8, &scales_x, seq_len);
        Self::add_bias(&mut k, &self.k_bias, seq_len, kv_dim);
        let mut v = self.w_v.forward_pre_quantized(&x_int8, &scales_x, seq_len);
        Self::add_bias(&mut v, &self.v_bias, seq_len, kv_dim);

        // Apply headwise Q/K norms, then RoPE
        let eps = self.attn_norm.eps() as f32;
        let mut q =
            apply_headwise_rmsnorm(&q, &self.q_norm_weight, seq_len, num_heads, head_dim, eps);
        let mut k = apply_headwise_rmsnorm(
            &k,
            &self.k_norm_weight,
            seq_len,
            num_kv_heads,
            head_dim,
            eps,
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

        self.w_o.forward(&attn_output, seq_len)
    }

    /// Batched attention with KV cache populate (for prefill).
    fn apply_attention_and_cache(
        &self,
        x: &[f32],
        seq_len: usize,
        cache_layer: &mut KVCacheLayer,
        rope_cos: &[f32],
        rope_sin: &[f32],
    ) -> Vec<f32> {
        let num_heads = self.attention.num_heads();
        let num_kv_heads = self.attention.num_kv_heads();
        let head_dim = self.attention.head_dim();
        let q_dim = num_heads * head_dim;
        let kv_dim = num_kv_heads * head_dim;

        // INT8 Q/K/V projections (quantize input once, share across projections)
        let k_in = self.hidden_size;
        let (x_int8, scales_x) = synapse_core::quantize_per_channel_int8(x, seq_len, k_in)
            .expect("quantize failed for prefill attention");
        let mut q = self.w_q.forward_pre_quantized(&x_int8, &scales_x, seq_len);
        Self::add_bias(&mut q, &self.q_bias, seq_len, q_dim);
        let mut k = self.w_k.forward_pre_quantized(&x_int8, &scales_x, seq_len);
        Self::add_bias(&mut k, &self.k_bias, seq_len, kv_dim);
        let mut v = self.w_v.forward_pre_quantized(&x_int8, &scales_x, seq_len);
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

        self.w_o.forward(&attn_out, seq_len)
    }

    /// Single-token cached attention (for decode).
    fn apply_attention_cached(
        &self,
        x: &[f32],
        cache_layer: &mut KVCacheLayer,
        pos: usize,
        rope_cos: &[f32],
        rope_sin: &[f32],
    ) -> Vec<f32> {
        let num_heads = self.attention.num_heads();
        let num_kv_heads = self.attention.num_kv_heads();
        let head_dim = self.attention.head_dim();
        let q_dim = num_heads * head_dim;
        let kv_dim = num_kv_heads * head_dim;
        let tokens = 1;

        // Quantize the single-token input once and share it across Q/K/V.
        let (x_int8, scales_x) = synapse_core::quantize_per_channel_int8(x, tokens, self.hidden_size)
            .expect("quantize failed for cached attention input");
        let mut q = self.w_q.forward_pre_quantized(&x_int8, &scales_x, tokens);
        Self::add_bias(&mut q, &self.q_bias, tokens, q_dim);
        let mut k = self.w_k.forward_pre_quantized(&x_int8, &scales_x, tokens);
        Self::add_bias(&mut k, &self.k_bias, tokens, kv_dim);
        let mut v = self.w_v.forward_pre_quantized(&x_int8, &scales_x, tokens);
        Self::add_bias(&mut v, &self.v_bias, tokens, kv_dim);

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

        // INT8 output projection
        self.w_o.forward(&attn_out, tokens)
    }

    fn apply_ffn(&self, x: &[f32]) -> Vec<f32> {
        let h = self.hidden_size;
        let inter = self.ffn.intermediate_size();
        let tokens = x.len() / h;

        match self.ffn.name() {
            "SwiGLU" => {
                // Quantize x once and share across gate/up projections.
                let (x_int8, scales_x) = synapse_core::quantize_per_channel_int8(x, tokens, h)
                    .expect("quantize_per_channel_int8 failed for SwiGLU input");
                let gate = self
                    .ffn_gate
                    .forward_pre_quantized(&x_int8, &scales_x, tokens);
                let up = self
                    .ffn_up
                    .forward_pre_quantized(&x_int8, &scales_x, tokens);
                let mut hidden = vec![0.0f32; tokens * inter];
                for i in 0..hidden.len() {
                    hidden[i] = silu(gate[i]) * up[i];
                }
                self.ffn_down.forward(&hidden, tokens)
            }
            "GeGLU" => {
                // Quantize x once and share across gate/up projections.
                let (x_int8, scales_x) = synapse_core::quantize_per_channel_int8(x, tokens, h)
                    .expect("quantize_per_channel_int8 failed for GeGLU input");
                let gate = self
                    .ffn_gate
                    .forward_pre_quantized(&x_int8, &scales_x, tokens);
                let up = self
                    .ffn_up
                    .forward_pre_quantized(&x_int8, &scales_x, tokens);
                let mut hidden = vec![0.0f32; tokens * inter];
                for i in 0..hidden.len() {
                    hidden[i] = gelu(gate[i]) * up[i];
                }
                self.ffn_down.forward(&hidden, tokens)
            }
            _ => {
                let mut activated = self.ffn_up.forward(x, tokens);
                for v in activated.iter_mut() {
                    *v = gelu(*v);
                }
                self.ffn_down.forward(&activated, tokens)
            }
        }
    }

    /// Memory in bytes for this layer's weights.
    pub fn memory_bytes(&self) -> usize {
        let norm_bytes =
            (self.attn_norm_weight.len() + self.ffn_norm_weight.len()) * mem::size_of::<f32>();
        let linear_bytes = self.w_q.memory_bytes()
            + self.w_k.memory_bytes()
            + self.w_v.memory_bytes()
            + self.w_o.memory_bytes()
            + self.ffn_gate.memory_bytes()
            + self.ffn_up.memory_bytes()
            + self.ffn_down.memory_bytes();
        norm_bytes + linear_bytes
    }
}

// ── QuantizedCausalLM forward ────────────────────────────────────────

impl QuantizedCausalLM {
    /// Forward pass: token_ids → logits.
    pub fn forward(&self, token_ids: &[u32]) -> ModelOutput {
        let seq_len = token_ids.len();
        let h = self.config.architecture.hidden_size;
        let vocab = self.config.architecture.vocab_size;

        // 1. Embedding lookup → [seq_len, h]
        let mut x = vec![0.0f32; seq_len * h];
        for (t, &id) in token_ids.iter().enumerate() {
            let id = id as usize;
            if id < vocab {
                let src = &self.embed_tokens[id * h..(id + 1) * h];
                x[t * h..(t + 1) * h].copy_from_slice(src);
            }
        }

        // 2. Quantized decoder layers
        for layer in &self.layers {
            x = layer.forward(&x, seq_len, &self.rope_cos, &self.rope_sin);
        }

        // 3. Final norm
        x = apply_norm(&x, &self.final_norm_weight, &*self.final_norm, h);

        // 4. LM head projection (INT8 quantized for speed)
        let logits = if let Some(ref lm_q) = self.lm_head_quantized {
            lm_q.forward(&x, seq_len)
        } else {
            let lm_weight = self.lm_head_weight.as_ref().unwrap_or(&self.embed_tokens);
            matmul_t(&x, lm_weight, seq_len, h, vocab)
        };

        ModelOutput {
            logits,
            shape: [1, seq_len, vocab],
        }
    }

    /// Prefill: processes all prompt tokens, populates KV cache, returns logits for last token.
    pub fn forward_prefill(&self, token_ids: &[u32], cache: &mut KVCache) -> ModelOutput {
        let seq_len = token_ids.len();
        let h = self.config.architecture.hidden_size;
        let vocab = self.config.architecture.vocab_size;

        // 1. Embedding lookup → [seq_len, h]
        let mut x = vec![0.0f32; seq_len * h];
        for (t, &id) in token_ids.iter().enumerate() {
            let id = id as usize;
            if id < vocab {
                let src = &self.embed_tokens[id * h..(id + 1) * h];
                x[t * h..(t + 1) * h].copy_from_slice(src);
            }
        }

        // 2. Quantized decoder layers with cache populate
        for (i, layer) in self.layers.iter().enumerate() {
            x = layer.forward_prefill_batched(
                &x,
                seq_len,
                cache.layer_mut(i),
                &self.rope_cos,
                &self.rope_sin,
            );
        }

        // 3. Final norm (last token only)
        let last_hidden = &x[(seq_len - 1) * h..seq_len * h];
        let normed = apply_norm(last_hidden, &self.final_norm_weight, &*self.final_norm, h);

        // 4. LM head projection → [1, vocab]
        let logits = if let Some(ref lm_q) = self.lm_head_quantized {
            lm_q.forward(&normed, 1)
        } else {
            let lm_weight = self.lm_head_weight.as_ref().unwrap_or(&self.embed_tokens);
            matmul_t(&normed, lm_weight, 1, h, vocab)
        };

        ModelOutput {
            logits,
            shape: [1, 1, vocab],
        }
    }

    /// Single-token decode using KV cache.
    pub fn forward_one(&self, token: u32, cache: &mut KVCache) -> ModelOutput {
        let h = self.config.architecture.hidden_size;
        let vocab = self.config.architecture.vocab_size;
        let pos = cache.current_len().expect("failed to query cache length");
        let emit_profile = should_emit_decode_profile();

        // 1. Embedding lookup → [1, h]
        let embed_start = if emit_profile {
            Some(Instant::now())
        } else {
            None
        };
        let mut x = vec![0.0f32; h];
        let id = token as usize;
        if id < vocab {
            x.copy_from_slice(&self.embed_tokens[id * h..(id + 1) * h]);
        }
        let embed_elapsed = embed_start.map(|start| start.elapsed());

        // 2. Quantized decoder layers with KV cache
        let layers_start = if emit_profile {
            Some(Instant::now())
        } else {
            None
        };
        for (i, layer) in self.layers.iter().enumerate() {
            x = if emit_profile {
                layer.forward_one_profiled(&x, cache.layer_mut(i), pos, &self.rope_cos, &self.rope_sin, i)
            } else {
                layer.forward_one(&x, cache.layer_mut(i), pos, &self.rope_cos, &self.rope_sin)
            };
        }
        let layers_elapsed = layers_start.map(|start| start.elapsed());

        // 3. Final norm
        let norm_start = if emit_profile {
            Some(Instant::now())
        } else {
            None
        };
        x = apply_norm(&x, &self.final_norm_weight, &*self.final_norm, h);
        let norm_elapsed = norm_start.map(|start| start.elapsed());

        // 4. LM head projection → [1, vocab]
        let lm_head_start = if emit_profile {
            Some(Instant::now())
        } else {
            None
        };
        let logits = if let Some(ref lm_q) = self.lm_head_quantized {
            lm_q.forward(&x, 1)
        } else {
            let lm_weight = self.lm_head_weight.as_ref().unwrap_or(&self.embed_tokens);
            matmul_t(&x, lm_weight, 1, h, vocab)
        };
        let lm_head_elapsed = lm_head_start.map(|start| start.elapsed());

        if emit_profile {
            eprintln!(
                "INT8 decode token summary: embed={:.1}ms layers={:.1}ms final_norm={:.1}ms lm_head={:.1}ms",
                embed_elapsed.unwrap().as_secs_f64() * 1000.0,
                layers_elapsed.unwrap().as_secs_f64() * 1000.0,
                norm_elapsed.unwrap().as_secs_f64() * 1000.0,
                lm_head_elapsed.unwrap().as_secs_f64() * 1000.0,
            );
        }

        ModelOutput {
            logits,
            shape: [1, 1, vocab],
        }
    }

    /// Total memory in bytes for all stored weights.
    pub fn memory_bytes(&self) -> usize {
        let embed = self.embed_tokens.len() * mem::size_of::<f32>();
        let layers: usize = self.layers.iter().map(|l| l.memory_bytes()).sum();
        let norm = self.final_norm_weight.len() * mem::size_of::<f32>();
        let lm_head = self
            .lm_head_weight
            .as_ref()
            .map_or(0, |w| w.len() * mem::size_of::<f32>());
        embed + layers + norm + lm_head
    }
}

impl crate::model::traits::Model for QuantizedCausalLM {
    fn forward(&self, token_ids: &[u32]) -> ModelOutput {
        QuantizedCausalLM::forward(self, token_ids)
    }

    fn forward_prefill(&self, token_ids: &[u32], cache: &mut KVCache) -> ModelOutput {
        QuantizedCausalLM::forward_prefill(self, token_ids, cache)
    }

    fn forward_one(&self, token: u32, cache: &mut KVCache) -> ModelOutput {
        QuantizedCausalLM::forward_one(self, token, cache)
    }

    // No forward_one_draft for INT8 — use default (falls back to forward_one)

    fn num_layers(&self) -> usize {
        self.layers.len()
    }

    fn config(&self) -> &ModelConfig {
        &self.config
    }
}

/// Compute the f32 memory footprint of a CausalLM for comparison.
pub fn f32_model_memory_bytes(model: &CausalLM) -> usize {
    let sz = mem::size_of::<f32>();
    let embed = model.embed_tokens.len() * sz;
    let layers: usize = model
        .layers
        .iter()
        .map(|layer| {
            (layer.attn_norm_weight.len()
                + layer.w_q.len()
                + layer.w_k.len()
                + layer.w_v.len()
                + layer.w_o.len()
                + layer.ffn_norm_weight.len()
                + layer.ffn_gate.len()
                + layer.ffn_up.len()
                + layer.ffn_down.len())
                * sz
        })
        .sum();
    let norm = model.final_norm_weight.len() * sz;
    let lm_head = model.lm_head_weight.as_ref().map_or(0, |w| w.len() * sz);
    embed + layers + norm + lm_head
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    use crate::config::*;
    use crate::model::ModelBuilder;
    use crate::weight_loading::{AlignedBuffer, RawTensor, WeightMapper};

    fn test_config() -> ModelConfig {
        ModelConfig {
            name: "QuantizedCausalLMTest".to_string(),
            architecture: ArchitectureConfig {
                hidden_size: 64,
                num_layers: 4,
                vocab_size: 256,
                max_sequence_length: 64,
                tie_word_embeddings: true,
            },
            attention: AttentionConfig::GQA {
                num_heads: 4,
                num_kv_heads: 2,
                head_dim: 16,
            },
            norm: NormConfig::RMSNorm { eps: 1e-6 },
            ffn: FFNConfig::SwiGLU {
                intermediate_size: 128,
            },
            position: PositionConfig::RoPE {
                base: 10000.0,
                max_position_embeddings: 64,
                style: Default::default(),
                scaling: Default::default(),
            },
            quantization: QuantConfig::F32,
        }
    }

    fn gen_weights(len: usize, seed: u32) -> Vec<f32> {
        (0..len)
            .map(|i| {
                let x = ((i as u32).wrapping_mul(2654435761).wrapping_add(seed)) as f32;
                (x / u32::MAX as f32) * 0.36 - 0.18
            })
            .collect()
    }

    fn generate_fake_hf_weights(cfg: &ModelConfig) -> HashMap<String, RawTensor> {
        let h = cfg.architecture.hidden_size;
        let vocab = cfg.architecture.vocab_size;
        let q_dim = cfg.attention.num_heads() * cfg.attention.head_dim();
        let kv_dim = cfg.attention.num_kv_heads() * cfg.attention.head_dim();
        let inter = cfg.ffn.intermediate_size();
        let nl = cfg.architecture.num_layers;

        let fake = |shape: Vec<usize>, seed: u32| -> RawTensor {
            let n: usize = shape.iter().product();
            RawTensor {
                data: AlignedBuffer::from_slice(&gen_weights(n, seed)),
                shape,
            }
        };

        let mut w = HashMap::new();
        w.insert("model.embed_tokens.weight".into(), fake(vec![vocab, h], 1));
        for i in 0..nl {
            let s = (i as u32 + 1) * 100;
            w.insert(
                format!("model.layers.{i}.input_layernorm.weight"),
                fake(vec![h], s),
            );
            w.insert(
                format!("model.layers.{i}.self_attn.q_proj.weight"),
                fake(vec![q_dim, h], s + 1),
            );
            w.insert(
                format!("model.layers.{i}.self_attn.k_proj.weight"),
                fake(vec![kv_dim, h], s + 2),
            );
            w.insert(
                format!("model.layers.{i}.self_attn.v_proj.weight"),
                fake(vec![kv_dim, h], s + 3),
            );
            w.insert(
                format!("model.layers.{i}.self_attn.o_proj.weight"),
                fake(vec![h, q_dim], s + 4),
            );
            w.insert(
                format!("model.layers.{i}.self_attn.q_norm.weight"),
                fake(vec![cfg.attention.head_dim()], s + 5),
            );
            w.insert(
                format!("model.layers.{i}.self_attn.k_norm.weight"),
                fake(vec![cfg.attention.head_dim()], s + 6),
            );
            w.insert(
                format!("model.layers.{i}.post_attention_layernorm.weight"),
                fake(vec![h], s + 7),
            );
            w.insert(
                format!("model.layers.{i}.mlp.gate_proj.weight"),
                fake(vec![inter, h], s + 8),
            );
            w.insert(
                format!("model.layers.{i}.mlp.up_proj.weight"),
                fake(vec![inter, h], s + 9),
            );
            w.insert(
                format!("model.layers.{i}.mlp.down_proj.weight"),
                fake(vec![h, inter], s + 10),
            );
        }
        w.insert("model.norm.weight".into(), fake(vec![h], 9999));
        w.insert("lm_head.weight".into(), fake(vec![vocab, h], 9998));
        w
    }

    fn build_quantized_test_model() -> QuantizedCausalLM {
        let cfg = test_config();
        let mut model = ModelBuilder::from_config(&cfg);
        let mapper = WeightMapper::qwen3();
        let weights = generate_fake_hf_weights(&cfg);
        let result = model.load_weights(weights, &mapper).unwrap();
        assert!(result.missing.is_empty());
        quantize_model(&model)
    }

    fn make_cache(cfg: &ModelConfig, max_seq: usize) -> KVCache {
        KVCache::new(
            cfg.architecture.num_layers,
            max_seq,
            cfg.attention.num_kv_heads(),
            cfg.attention.head_dim(),
        )
        .unwrap()
    }

    #[test]
    fn quantized_prefill_matches_quantized_forward_last_logits() {
        let model = build_quantized_test_model();
        let prompt = vec![1u32, 2, 3, 4];
        let vocab = model.config.architecture.vocab_size;

        let full_out = model.forward(&prompt);
        let full_last_logits = &full_out.logits[(prompt.len() - 1) * vocab..prompt.len() * vocab];

        let mut cache = make_cache(&model.config, prompt.len() + 1);
        let prefill_out = model.forward_prefill(&prompt, &mut cache);

        assert_eq!(prefill_out.logits.len(), vocab);
        for (i, (&a, &b)) in prefill_out
            .logits
            .iter()
            .zip(full_last_logits.iter())
            .enumerate()
        {
            assert!(
                (a - b).abs() < 1e-5,
                "Quantized prefill mismatch at logit {i}: cached={a}, full={b}"
            );
        }
    }

    #[test]
    fn quantized_prefill_then_one_matches_quantized_forward_last_logits() {
        let model = build_quantized_test_model();
        let prompt = vec![1u32, 2, 3];
        let next_token = 42u32;
        let vocab = model.config.architecture.vocab_size;

        let mut cache = make_cache(&model.config, prompt.len() + 2);
        let _ = model.forward_prefill(&prompt, &mut cache);
        let one_out = model.forward_one(next_token, &mut cache);

        let mut full_tokens = prompt.clone();
        full_tokens.push(next_token);
        let full_out = model.forward(&full_tokens);
        let full_last_logits =
            &full_out.logits[(full_tokens.len() - 1) * vocab..full_tokens.len() * vocab];

        assert_eq!(one_out.logits.len(), vocab);
        for (i, (&a, &b)) in one_out
            .logits
            .iter()
            .zip(full_last_logits.iter())
            .enumerate()
        {
            assert!(
                (a - b).abs() < 1e-5,
                "Quantized decode mismatch at logit {i}: cached={a}, full={b}"
            );
        }
    }
}
