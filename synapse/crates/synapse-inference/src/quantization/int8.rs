use std::mem;

use crate::config::position::RoPEStyle;
use crate::config::ModelConfig;
use crate::kv_cache::{KVCache, KVCacheLayer};
use crate::model::causal_lm::ModelOutput;
use crate::model::decoder_layer::{
    add_vecs, add_vecs_inplace, apply_headwise_rmsnorm, apply_norm, apply_rope_inplace, gelu,
    matmul_nn, matmul_t, silu, softmax_slice,
};
use crate::model::CausalLM;
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
    pub rope_cos: Vec<f32>,
    pub rope_sin: Vec<f32>,
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

    QuantizedCausalLM {
        config: model.config.clone(),
        layers,
        final_norm: create_norm(&cfg.norm),
        embed_tokens: model.embed_tokens.to_vec(),
        final_norm_weight: model.final_norm_weight.to_vec(),
        lm_head_weight: model.lm_head_weight.as_ref().map(|w| w.to_vec()),
        rope_cos: model.rope_cos.clone(),
        rope_sin: model.rope_sin.clone(),
    }
}

// ── QuantizedDecoderLayer forward ────────────────────────────────────

impl QuantizedDecoderLayer {
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
        let h = self.hidden_size;

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
        let (x_int8, scales_x) =
            synapse_core::quantize_per_channel_int8(x, seq_len, k_dim)
                .expect("quantize_per_channel_int8 failed for attention input");
        let q = self.w_q.forward_pre_quantized(&x_int8, &scales_x, seq_len);
        let k = self.w_k.forward_pre_quantized(&x_int8, &scales_x, seq_len);
        let v = self.w_v.forward_pre_quantized(&x_int8, &scales_x, seq_len);

        // Apply headwise Q/K norms, then RoPE
        let eps = self.attn_norm.eps() as f32;
        let mut q = apply_headwise_rmsnorm(&q, &self.q_norm_weight, seq_len, num_heads, head_dim, eps);
        let mut k = apply_headwise_rmsnorm(&k, &self.k_norm_weight, seq_len, num_kv_heads, head_dim, eps);
        apply_rope_inplace(&mut q, rope_cos, rope_sin, seq_len, num_heads, head_dim, 0, self.rope_style);
        apply_rope_inplace(&mut k, rope_cos, rope_sin, seq_len, num_kv_heads, head_dim, 0, self.rope_style);

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
        let groups = num_heads / num_kv_heads;
        let scale = 1.0 / (head_dim as f32).sqrt();

        // INT8 Q/K/V projections (quantize input once, share across projections)
        let k_in = self.hidden_size;
        let (x_int8, scales_x) =
            synapse_core::quantize_per_channel_int8(x, seq_len, k_in)
                .expect("quantize failed for prefill attention");
        let q = self.w_q.forward_pre_quantized(&x_int8, &scales_x, seq_len);
        let k = self.w_k.forward_pre_quantized(&x_int8, &scales_x, seq_len);
        let v = self.w_v.forward_pre_quantized(&x_int8, &scales_x, seq_len);

        let eps = self.attn_norm.eps() as f32;
        let mut q = apply_headwise_rmsnorm(&q, &self.q_norm_weight, seq_len, num_heads, head_dim, eps);
        let mut k = apply_headwise_rmsnorm(&k, &self.k_norm_weight, seq_len, num_kv_heads, head_dim, eps);
        apply_rope_inplace(&mut q, rope_cos, rope_sin, seq_len, num_heads, head_dim, 0, self.rope_style);
        apply_rope_inplace(&mut k, rope_cos, rope_sin, seq_len, num_kv_heads, head_dim, 0, self.rope_style);

        // Populate KV cache
        for t in 0..seq_len {
            let k_token = &k[t * kv_dim..(t + 1) * kv_dim];
            let v_token = &v[t * kv_dim..(t + 1) * kv_dim];
            cache_layer
                .append(k_token, v_token)
                .expect("KV cache append failed during prefill");
        }

        // Batched causal attention (identical to f32 path — attention is always f32)
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
        let groups = num_heads / num_kv_heads;
        let scale = 1.0 / (head_dim as f32).sqrt();

        // INT8 Q/K/V projections for single token
        let q = self.w_q.forward(x, 1);
        let k = self.w_k.forward(x, 1);
        let v = self.w_v.forward(x, 1);

        let eps = self.attn_norm.eps() as f32;
        let mut q = apply_headwise_rmsnorm(&q, &self.q_norm_weight, 1, num_heads, head_dim, eps);
        let mut k = apply_headwise_rmsnorm(&k, &self.k_norm_weight, 1, num_kv_heads, head_dim, eps);
        apply_rope_inplace(&mut q, rope_cos, rope_sin, 1, num_heads, head_dim, pos, self.rope_style);
        apply_rope_inplace(&mut k, rope_cos, rope_sin, 1, num_kv_heads, head_dim, pos, self.rope_style);

        // Append to cache
        cache_layer
            .append(&k, &v)
            .expect("KV cache append failed");

        // Get full cached K/V
        let (cached_k, cached_v, seq_len) = cache_layer
            .slice()
            .expect("KV cache slice failed");

        // Sliding window: limit attention to the last `window_size` positions
        let (effective_k, effective_v, effective_len) = if let Some(ws) = self.attention.window_size() {
            if seq_len > ws {
                let offset = (seq_len - ws) * kv_dim;
                (&cached_k[offset..], &cached_v[offset..], ws)
            } else {
                (cached_k, cached_v, seq_len)
            }
        } else {
            (cached_k, cached_v, seq_len)
        };

        // Attention: single Q against effective cached K/V (same logic as f32 path)
        let mut attn_output = vec![0.0f32; q_dim];

        if effective_len >= 16 {
            // SIMD path: gather + matmul
            let mut k_heads = Vec::with_capacity(num_kv_heads);
            let mut v_heads = Vec::with_capacity(num_kv_heads);
            for kv_head in 0..num_kv_heads {
                let mut k_buf = vec![0.0f32; effective_len * head_dim];
                let mut v_buf = vec![0.0f32; effective_len * head_dim];
                for s in 0..effective_len {
                    let off = s * kv_dim + kv_head * head_dim;
                    k_buf[s * head_dim..(s + 1) * head_dim]
                        .copy_from_slice(&effective_k[off..off + head_dim]);
                    v_buf[s * head_dim..(s + 1) * head_dim]
                        .copy_from_slice(&effective_v[off..off + head_dim]);
                }
                k_heads.push(k_buf);
                v_heads.push(v_buf);
            }

            for head in 0..num_heads {
                let kv_head = head / groups;
                let q_head = &q[head * head_dim..(head + 1) * head_dim];

                let mut scores = matmul_t(q_head, &k_heads[kv_head], 1, head_dim, effective_len);
                for s in &mut scores {
                    *s *= scale;
                }
                softmax_slice(&mut scores);

                let sv = matmul_nn(&scores, &v_heads[kv_head], 1, effective_len, head_dim);
                attn_output[head * head_dim..(head + 1) * head_dim]
                    .copy_from_slice(&sv);
            }
        } else {
            // Scalar path for short sequences
            for head in 0..num_heads {
                let kv_head = head / groups;
                let mut scores = vec![0.0f32; effective_len];
                for s in 0..effective_len {
                    let mut dot = 0.0f32;
                    for d in 0..head_dim {
                        dot += q[head * head_dim + d]
                            * effective_k[s * kv_dim + kv_head * head_dim + d];
                    }
                    scores[s] = dot * scale;
                }
                softmax_slice(&mut scores);
                for d in 0..head_dim {
                    let mut sum = 0.0f32;
                    for s in 0..effective_len {
                        sum += scores[s]
                            * effective_v[s * kv_dim + kv_head * head_dim + d];
                    }
                    attn_output[head * head_dim + d] = sum;
                }
            }
        }

        // INT8 output projection
        self.w_o.forward(&attn_output, 1)
    }

    fn apply_ffn(&self, x: &[f32]) -> Vec<f32> {
        let h = self.hidden_size;
        let inter = self.ffn.intermediate_size();
        let tokens = x.len() / h;

        match self.ffn.name() {
            "SwiGLU" => {
                // Quantize x once and share across gate/up projections.
                let (x_int8, scales_x) =
                    synapse_core::quantize_per_channel_int8(x, tokens, h)
                        .expect("quantize_per_channel_int8 failed for SwiGLU input");
                let gate = self.ffn_gate.forward_pre_quantized(&x_int8, &scales_x, tokens);
                let up = self.ffn_up.forward_pre_quantized(&x_int8, &scales_x, tokens);
                let mut hidden = vec![0.0f32; tokens * inter];
                for i in 0..hidden.len() {
                    hidden[i] = silu(gate[i]) * up[i];
                }
                self.ffn_down.forward(&hidden, tokens)
            }
            "GeGLU" => {
                // Quantize x once and share across gate/up projections.
                let (x_int8, scales_x) =
                    synapse_core::quantize_per_channel_int8(x, tokens, h)
                        .expect("quantize_per_channel_int8 failed for GeGLU input");
                let gate = self.ffn_gate.forward_pre_quantized(&x_int8, &scales_x, tokens);
                let up = self.ffn_up.forward_pre_quantized(&x_int8, &scales_x, tokens);
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

        // 4. LM head projection (stays f32)
        let lm_weight = self.lm_head_weight.as_ref().unwrap_or(&self.embed_tokens);
        let logits = matmul_t(&x, lm_weight, seq_len, h, vocab);

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
                &x, seq_len, cache.layer_mut(i), &self.rope_cos, &self.rope_sin,
            );
        }

        // 3. Final norm (last token only)
        let last_hidden = &x[(seq_len - 1) * h..seq_len * h];
        let normed = apply_norm(last_hidden, &self.final_norm_weight, &*self.final_norm, h);

        // 4. LM head projection → [1, vocab]
        let lm_weight = self.lm_head_weight.as_ref().unwrap_or(&self.embed_tokens);
        let logits = matmul_t(&normed, lm_weight, 1, h, vocab);

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

        // 1. Embedding lookup → [1, h]
        let mut x = vec![0.0f32; h];
        let id = token as usize;
        if id < vocab {
            x.copy_from_slice(&self.embed_tokens[id * h..(id + 1) * h]);
        }

        // 2. Quantized decoder layers with KV cache
        for (i, layer) in self.layers.iter().enumerate() {
            x = layer.forward_one(&x, cache.layer_mut(i), pos, &self.rope_cos, &self.rope_sin);
        }

        // 3. Final norm
        x = apply_norm(&x, &self.final_norm_weight, &*self.final_norm, h);

        // 4. LM head projection → [1, vocab]
        let lm_weight = self.lm_head_weight.as_ref().unwrap_or(&self.embed_tokens);
        let logits = matmul_t(&x, lm_weight, 1, h, vocab);

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
    let lm_head = model
        .lm_head_weight
        .as_ref()
        .map_or(0, |w| w.len() * sz);
    embed + layers + norm + lm_head
}
