//! Q4_0-quantized RoBERTa / UniXcoder encoder.
//!
//! Mirrors [`super::roberta::RoBERTaEncoder`] with every per-layer linear
//! (Q/K/V/O + intermediate.up/output.down) replaced by [`Q4Linear`].
//! Embeddings stay fp32 — they're already table-lookup-bound and the
//! padding-aware position id cumsum trick is easier to keep bit-exact.
//!
//! Memory footprint for `microsoft/unixcoder-base` (D=768, 12 layers):
//!   - fp32 safetensors on disk: 480 MB (125 M params × ~4 B)
//!   - Q4 linears in RAM:        ~70 MB (6.4× smaller)
//!   - Embeddings remain fp32:   ~160 MB (word 51 416 × 768 + pos + type)
//!   ——— Total RAM: ~230 MB. Half-precision embeddings would trim it to
//!   ~150 MB if the on-disk format is fp16.

use crate::models::text_encoder::roberta::{
    RoBERTaConfig, RoBERTaEmbeddings, RoBERTaEncoder, RoBERTaLayer,
};
use crate::ops::pure_rust_ops::layernorm_with_bias;
use crate::quantization::primitives::Q4Linear;
use crate::weight_loading::AlignedBuffer;

// ── Exact-erf GELU helpers — duplicated from roberta.rs so the Q4 module
// doesn't reach into the sibling's private fn table. Accuracy ~1.5e-7.

fn erf(x: f32) -> f32 {
    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let x = x.abs();
    let a1 = 0.254829592f32;
    let a2 = -0.284496736f32;
    let a3 = 1.421413741f32;
    let a4 = -1.453152027f32;
    let a5 = 1.061405429f32;
    let p = 0.3275911f32;
    let t = 1.0 / (1.0 + p * x);
    let y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * (-x * x).exp();
    sign * y
}

fn gelu_erf(x: f32) -> f32 {
    0.5 * x * (1.0 + erf(x / std::f32::consts::SQRT_2))
}

// Duplicate of the padding-aware attention in `roberta.rs`. Kept local to
// avoid exporting it from the public surface.
fn masked_bidirectional_attention(
    q: &[f32], k: &[f32], v: &[f32],
    seq_len: usize, num_heads: usize, head_dim: usize,
    attention_mask: &[i64],
) -> Vec<f32> {
    let qk_dim = num_heads * head_dim;
    let scale = 1.0 / (head_dim as f32).sqrt();
    let mut output = vec![0.0f32; seq_len * qk_dim];
    for head in 0..num_heads {
        for qi in 0..seq_len {
            let mut scores = vec![f32::NEG_INFINITY; seq_len];
            let mut max = f32::NEG_INFINITY;
            for ki in 0..seq_len {
                if attention_mask[ki] == 0 { continue; }
                let mut dot = 0.0f32;
                for d in 0..head_dim {
                    dot += q[qi * qk_dim + head * head_dim + d]
                         * k[ki * qk_dim + head * head_dim + d];
                }
                let s = dot * scale;
                scores[ki] = s;
                if s > max { max = s; }
            }
            let mut sum = 0.0f32;
            for s in scores.iter_mut() {
                if s.is_finite() {
                    *s = (*s - max).exp();
                    sum += *s;
                } else { *s = 0.0; }
            }
            if sum > 0.0 { for s in scores.iter_mut() { *s /= sum; } }
            for d in 0..head_dim {
                let mut val = 0.0f32;
                for ki in 0..seq_len {
                    val += scores[ki] * v[ki * qk_dim + head * head_dim + d];
                }
                output[qi * qk_dim + head * head_dim + d] = val;
            }
        }
    }
    output
}

pub struct Q4RoBERTaLayer {
    pub hidden_size: usize,
    pub num_heads: usize,
    pub intermediate_size: usize,
    pub layer_norm_eps: f32,

    pub w_q: Q4Linear, pub q_bias: AlignedBuffer,
    pub w_k: Q4Linear, pub k_bias: AlignedBuffer,
    pub w_v: Q4Linear, pub v_bias: AlignedBuffer,
    pub w_o: Q4Linear, pub o_bias: AlignedBuffer,
    pub attn_ln_w: AlignedBuffer, pub attn_ln_b: AlignedBuffer,

    pub w_up:   Q4Linear, pub up_bias:   AlignedBuffer,
    pub w_down: Q4Linear, pub down_bias: AlignedBuffer,
    pub ffn_ln_w: AlignedBuffer, pub ffn_ln_b: AlignedBuffer,
}

impl Q4RoBERTaLayer {
    pub fn from_fp32(l: &RoBERTaLayer) -> Self {
        let h = l.hidden_size;
        let inter = l.intermediate_size;
        Self {
            hidden_size: h,
            num_heads: l.num_heads,
            intermediate_size: inter,
            layer_norm_eps: l.layer_norm_eps,

            w_q: Q4Linear::from_f32(&l.w_q, h, h), q_bias: l.q_bias.clone(),
            w_k: Q4Linear::from_f32(&l.w_k, h, h), k_bias: l.k_bias.clone(),
            w_v: Q4Linear::from_f32(&l.w_v, h, h), v_bias: l.v_bias.clone(),
            w_o: Q4Linear::from_f32(&l.w_o, h, h), o_bias: l.o_bias.clone(),
            attn_ln_w: l.attn_ln_weight.clone(), attn_ln_b: l.attn_ln_bias.clone(),

            w_up:   Q4Linear::from_f32(&l.w_up,   inter, h), up_bias:   l.up_bias.clone(),
            w_down: Q4Linear::from_f32(&l.w_down, h, inter), down_bias: l.down_bias.clone(),
            ffn_ln_w: l.ffn_ln_weight.clone(), ffn_ln_b: l.ffn_ln_bias.clone(),
        }
    }

    fn add_col_bias(x: &mut [f32], bias: &[f32], rows: usize, cols: usize) {
        if bias.is_empty() { return; }
        for r in 0..rows {
            for c in 0..cols {
                x[r * cols + c] += bias[c];
            }
        }
    }

    pub fn forward(&self, x: &[f32], seq_len: usize, mask: Option<&[i64]>) -> Vec<f32> {
        let h = self.hidden_size;
        let head_dim = h / self.num_heads;

        let mut q = self.w_q.forward_batched(x, seq_len);
        Self::add_col_bias(&mut q, &self.q_bias, seq_len, h);
        let mut k = self.w_k.forward_batched(x, seq_len);
        Self::add_col_bias(&mut k, &self.k_bias, seq_len, h);
        let mut v = self.w_v.forward_batched(x, seq_len);
        Self::add_col_bias(&mut v, &self.v_bias, seq_len, h);

        let attn = match mask {
            Some(m) => masked_bidirectional_attention(&q, &k, &v, seq_len, self.num_heads, head_dim, m),
            None => crate::ops::attention::bidirectional_attention(&q, &k, &v, seq_len, self.num_heads, head_dim),
        };

        let mut out = self.w_o.forward_batched(&attn, seq_len);
        Self::add_col_bias(&mut out, &self.o_bias, seq_len, h);

        let mut resid = vec![0.0f32; seq_len * h];
        for i in 0..resid.len() { resid[i] = x[i] + out[i]; }
        let normed = layernorm_with_bias(&resid, &self.attn_ln_w, &self.attn_ln_b,
                                         self.layer_norm_eps, h);

        let mut up = self.w_up.forward_batched(&normed, seq_len);
        Self::add_col_bias(&mut up, &self.up_bias, seq_len, self.intermediate_size);
        for v in up.iter_mut() { *v = gelu_erf(*v); }
        let mut down = self.w_down.forward_batched(&up, seq_len);
        Self::add_col_bias(&mut down, &self.down_bias, seq_len, h);

        for i in 0..resid.len() { resid[i] = normed[i] + down[i]; }
        layernorm_with_bias(&resid, &self.ffn_ln_w, &self.ffn_ln_b,
                            self.layer_norm_eps, h)
    }

    pub fn q4_memory_bytes(&self) -> usize {
        self.w_q.memory_bytes()
            + self.w_k.memory_bytes()
            + self.w_v.memory_bytes()
            + self.w_o.memory_bytes()
            + self.w_up.memory_bytes()
            + self.w_down.memory_bytes()
    }
}

pub struct Q4RoBERTaEncoder {
    pub config: RoBERTaConfig,
    pub embeddings: RoBERTaEmbeddings,
    pub layers: Vec<Q4RoBERTaLayer>,
}

impl Q4RoBERTaEncoder {
    pub fn from_fp32(m: &RoBERTaEncoder) -> Self {
        Self {
            config: m.config.clone(),
            embeddings: RoBERTaEmbeddings {
                hidden_size: m.embeddings.hidden_size,
                max_position_embeddings: m.embeddings.max_position_embeddings,
                type_vocab_size: m.embeddings.type_vocab_size,
                pad_token_id: m.embeddings.pad_token_id,
                layer_norm_eps: m.embeddings.layer_norm_eps,
                word:       m.embeddings.word.clone(),
                position:   m.embeddings.position.clone(),
                token_type: m.embeddings.token_type.clone(),
                ln_weight:  m.embeddings.ln_weight.clone(),
                ln_bias:    m.embeddings.ln_bias.clone(),
            },
            layers: m.layers.iter().map(Q4RoBERTaLayer::from_fp32).collect(),
        }
    }

    pub fn forward(&self, input_ids: &[i64], attention_mask: &[i64]) -> Vec<f32> {
        let seq_len = input_ids.len();
        let mut h = self.embeddings.forward(input_ids, attention_mask);
        for layer in &self.layers {
            h = layer.forward(&h, seq_len, Some(attention_mask));
        }
        h
    }

    pub fn cls_feature(&self, input_ids: &[i64], attention_mask: &[i64]) -> Vec<f32> {
        let h = self.forward(input_ids, attention_mask);
        h[..self.config.hidden_size].to_vec()
    }

    pub fn q4_memory_bytes(&self) -> usize {
        self.layers.iter().map(|l| l.q4_memory_bytes()).sum()
    }
}
