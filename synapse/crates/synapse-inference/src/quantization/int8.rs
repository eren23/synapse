use std::mem;

use crate::config::ModelConfig;
use crate::model::causal_lm::ModelOutput;
use crate::model::decoder_layer::{
    add_vecs, add_vecs_inplace, apply_norm, gelu, matmul_t, silu, softmax_slice,
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

    pub attn_norm_weight: Vec<f32>,
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
                attn_norm_weight: layer.attn_norm_weight.clone(),
                ffn_norm_weight: layer.ffn_norm_weight.clone(),
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
        embed_tokens: model.embed_tokens.clone(),
        final_norm_weight: model.final_norm_weight.clone(),
        lm_head_weight: model.lm_head_weight.clone(),
    }
}

// ── QuantizedDecoderLayer forward ────────────────────────────────────

impl QuantizedDecoderLayer {
    /// Pre-norm forward: norm→attention→residual→norm→FFN→residual.
    pub fn forward(&self, x: &[f32], seq_len: usize) -> Vec<f32> {
        let h = self.hidden_size;

        // 1. Attention sub-layer
        let normed = apply_norm(x, &self.attn_norm_weight, &*self.attn_norm, h);
        let attn_out = self.apply_attention(&normed, seq_len);
        let mut residual = add_vecs(x, &attn_out);

        // 2. FFN sub-layer
        let normed = apply_norm(&residual, &self.ffn_norm_weight, &*self.ffn_norm, h);
        let ffn_out = self.apply_ffn(&normed);
        add_vecs_inplace(&mut residual, &ffn_out);

        residual
    }

    fn apply_attention(&self, x: &[f32], seq_len: usize) -> Vec<f32> {
        let num_heads = self.attention.num_heads();
        let num_kv_heads = self.attention.num_kv_heads();
        let head_dim = self.attention.head_dim();
        let q_dim = num_heads * head_dim;
        let kv_dim = num_kv_heads * head_dim;
        let groups = num_heads / num_kv_heads;
        let scale = 1.0 / (head_dim as f32).sqrt();

        let q = self.w_q.forward(x, seq_len);
        let k = self.w_k.forward(x, seq_len);
        let v = self.w_v.forward(x, seq_len);

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

    fn apply_ffn(&self, x: &[f32]) -> Vec<f32> {
        let h = self.hidden_size;
        let inter = self.ffn.intermediate_size();
        let tokens = x.len() / h;

        match self.ffn.name() {
            "SwiGLU" => {
                let gate = self.ffn_gate.forward(x, tokens);
                let up = self.ffn_up.forward(x, tokens);
                let mut hidden = vec![0.0f32; tokens * inter];
                for i in 0..hidden.len() {
                    hidden[i] = silu(gate[i]) * up[i];
                }
                self.ffn_down.forward(&hidden, tokens)
            }
            "GeGLU" => {
                let gate = self.ffn_gate.forward(x, tokens);
                let up = self.ffn_up.forward(x, tokens);
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
            x = layer.forward(&x, seq_len);
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
