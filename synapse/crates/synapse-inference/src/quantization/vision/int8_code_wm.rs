//! INT8 quantization for Code WM (CWM).
//!
//! Only the 4 Linear weight matrices per transformer block are quantized
//! (attn in/out projections + MLP up/down). These are ~92% of the model's
//! matmul weights. Embeddings, positional encoding, LayerNorm params,
//! biases, and the tiny action encoder stay f32 — quantizing them hurts
//! accuracy for minimal size savings.
//!
//! Expected size: 3.0 MB f32 → ~1.0 MB INT8 (3x compression).

use std::collections::HashMap;

use crate::models::vision::code_wm::{CodeWorldModel, CodeWorldModelConfig, GeluKind};
use crate::ops::attention::bidirectional_attention;
use crate::ops::pure_rust_ops::layernorm_with_bias;
use crate::quantization::QuantizedLinear;
use crate::weight_loading::{AlignedBuffer, RawTensor, WeightError};

// ── GELU (mirror of code_wm.rs — duplicated to keep quantization module self-contained) ──
#[inline]
fn erf_f32(x: f32) -> f32 {
    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let x = x.abs();
    let t = 1.0 / (1.0 + 0.3275911 * x);
    let y = 1.0
        - (((((1.061405429_f32 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592)
            * t
            * (-x * x).exp();
    sign * y
}

#[inline]
fn gelu_erf(x: f32) -> f32 {
    const INV_SQRT2: f32 = 0.70710678118654752440_f32;
    0.5 * x * (1.0 + erf_f32(x * INV_SQRT2))
}

#[inline]
fn gelu_tanh(x: f32) -> f32 {
    const K0: f32 = 0.7978845608028654;
    0.5 * x * (1.0 + (K0 * (x + 0.044715 * x * x * x)).tanh())
}

fn apply_gelu(buf: &mut [f32], kind: GeluKind) {
    match kind {
        GeluKind::Erf => {
            for v in buf.iter_mut() {
                *v = gelu_erf(*v);
            }
        }
        GeluKind::Tanh => {
            for v in buf.iter_mut() {
                *v = gelu_tanh(*v);
            }
        }
    }
}

/// Small helper: `y = x @ w.T + b` applied row-by-row using a raw f32 weight matrix.
/// Used for the kept-f32 paths (action encoder, biases).
fn linear_f32(x: &[f32], weight: &[f32], bias: &[f32], m: usize, in_dim: usize, out_dim: usize) -> Vec<f32> {
    let mut out = crate::ops::matmul::matmul_t(x, weight, m, in_dim, out_dim);
    if !bias.is_empty() {
        for r in 0..m {
            for j in 0..out_dim {
                out[r * out_dim + j] += bias[j];
            }
        }
    }
    out
}

/// INT8-quantized transformer block. Weight matrices are INT8; biases, norms stay f32.
pub struct QuantizedTransformerBlock {
    pub norm1_w: AlignedBuffer,
    pub norm1_b: AlignedBuffer,
    pub attn_in_proj: QuantizedLinear, // logical [3*D, D]
    pub attn_in_proj_bias: AlignedBuffer,
    pub attn_out_proj: QuantizedLinear, // logical [D, D]
    pub attn_out_proj_bias: AlignedBuffer,
    pub norm2_w: AlignedBuffer,
    pub norm2_b: AlignedBuffer,
    pub mlp_up: QuantizedLinear,   // logical [mlp_hidden, D]
    pub mlp_up_bias: AlignedBuffer,
    pub mlp_down: QuantizedLinear, // logical [D, mlp_hidden]
    pub mlp_down_bias: AlignedBuffer,
}

impl QuantizedTransformerBlock {
    fn forward(&self, x: &[f32], seq_len: usize, cfg: &CodeWorldModelConfig) -> Vec<f32> {
        let d = cfg.model_dim;

        // Attention branch
        let normed1 = layernorm_with_bias(x, &self.norm1_w, &self.norm1_b, cfg.layernorm_eps, d);
        let mut qkv = self.attn_in_proj.forward(&normed1, seq_len);
        // Add in_proj_bias (broadcast per row)
        for r in 0..seq_len {
            for j in 0..3 * d {
                qkv[r * 3 * d + j] += self.attn_in_proj_bias[j];
            }
        }

        let mut q = vec![0.0_f32; seq_len * d];
        let mut k = vec![0.0_f32; seq_len * d];
        let mut v = vec![0.0_f32; seq_len * d];
        for t in 0..seq_len {
            let off_src = t * 3 * d;
            let off_dst = t * d;
            q[off_dst..off_dst + d].copy_from_slice(&qkv[off_src..off_src + d]);
            k[off_dst..off_dst + d].copy_from_slice(&qkv[off_src + d..off_src + 2 * d]);
            v[off_dst..off_dst + d].copy_from_slice(&qkv[off_src + 2 * d..off_src + 3 * d]);
        }

        let attn_out = bidirectional_attention(&q, &k, &v, seq_len, cfg.num_heads, cfg.head_dim);

        let mut attn_proj = self.attn_out_proj.forward(&attn_out, seq_len);
        for r in 0..seq_len {
            for j in 0..d {
                attn_proj[r * d + j] += self.attn_out_proj_bias[j];
            }
        }
        let mut res1 = vec![0.0_f32; seq_len * d];
        for i in 0..res1.len() {
            res1[i] = x[i] + attn_proj[i];
        }

        // MLP branch
        let normed2 = layernorm_with_bias(&res1, &self.norm2_w, &self.norm2_b, cfg.layernorm_eps, d);
        let mut up = self.mlp_up.forward(&normed2, seq_len);
        for r in 0..seq_len {
            for j in 0..cfg.mlp_hidden {
                up[r * cfg.mlp_hidden + j] += self.mlp_up_bias[j];
            }
        }
        apply_gelu(&mut up, cfg.gelu_kind);
        let mut down = self.mlp_down.forward(&up, seq_len);
        for r in 0..seq_len {
            for j in 0..d {
                down[r * d + j] += self.mlp_down_bias[j];
            }
        }

        let mut out = vec![0.0_f32; seq_len * d];
        for i in 0..out.len() {
            out[i] = res1[i] + down[i];
        }
        out
    }

    fn bytes(&self) -> usize {
        self.attn_in_proj.memory_bytes()
            + self.attn_out_proj.memory_bytes()
            + self.mlp_up.memory_bytes()
            + self.mlp_down.memory_bytes()
            + 4 * self.norm1_w.len()
            + 4 * self.norm1_b.len()
            + 4 * self.norm2_w.len()
            + 4 * self.norm2_b.len()
            + 4 * self.attn_in_proj_bias.len()
            + 4 * self.attn_out_proj_bias.len()
            + 4 * self.mlp_up_bias.len()
            + 4 * self.mlp_down_bias.len()
    }
}

/// INT8-quantized Code WM.
pub struct QuantizedCodeWorldModel {
    pub config: CodeWorldModelConfig,
    // f32 (kept for accuracy): embedding, CLS, PE, layernorm scales, small MLP
    pub token_embedding: AlignedBuffer, // [vocab, D]
    pub cls_token: AlignedBuffer,       // [D]
    pub pos_enc: AlignedBuffer,         // [max_seq+1, D]
    pub encoder_block: QuantizedTransformerBlock,
    pub encoder_final_norm_w: AlignedBuffer,
    pub encoder_final_norm_b: AlignedBuffer,
    // Action encoder stays f32 — only 17.5K params (<0.1 MB)
    pub action_fc1_w: AlignedBuffer,
    pub action_fc1_b: AlignedBuffer,
    pub action_fc2_w: AlignedBuffer,
    pub action_fc2_b: AlignedBuffer,
    pub predictor_blocks: Vec<QuantizedTransformerBlock>,
    pub predictor_final_norm_w: AlignedBuffer,
    pub predictor_final_norm_b: AlignedBuffer,
}

impl QuantizedCodeWorldModel {
    /// Total in-memory size in bytes (weights + scales).
    pub fn memory_bytes(&self) -> usize {
        4 * self.token_embedding.len()
            + 4 * self.cls_token.len()
            + 4 * self.pos_enc.len()
            + self.encoder_block.bytes()
            + 4 * self.encoder_final_norm_w.len()
            + 4 * self.encoder_final_norm_b.len()
            + 4 * self.action_fc1_w.len()
            + 4 * self.action_fc1_b.len()
            + 4 * self.action_fc2_w.len()
            + 4 * self.action_fc2_b.len()
            + self.predictor_blocks.iter().map(|b| b.bytes()).sum::<usize>()
            + 4 * self.predictor_final_norm_w.len()
            + 4 * self.predictor_final_norm_b.len()
    }

    /// Encode tokens → 128-d latent.
    pub fn encode(&self, tokens: &[i64]) -> Vec<f32> {
        let d = self.config.model_dim;
        let s = tokens.len();
        let seq_with_cls = s + 1;

        let mut h = vec![0.0_f32; seq_with_cls * d];
        h[..d].copy_from_slice(&self.cls_token);
        for (i, &tok) in tokens.iter().enumerate() {
            let t = tok as usize;
            let src = &self.token_embedding[t * d..(t + 1) * d];
            h[(i + 1) * d..(i + 2) * d].copy_from_slice(src);
        }
        for i in 0..seq_with_cls {
            let off = i * d;
            for j in 0..d {
                h[off + j] += self.pos_enc[off + j];
            }
        }
        for _ in 0..self.config.encoder_loops {
            h = self.encoder_block.forward(&h, seq_with_cls, &self.config);
        }
        let cls_out: Vec<f32> = h[..d].to_vec();
        layernorm_with_bias(
            &cls_out,
            &self.encoder_final_norm_w,
            &self.encoder_final_norm_b,
            self.config.layernorm_eps,
            d,
        )
    }

    /// Encode action vector → 128-d latent.
    pub fn encode_action(&self, action: &[f32]) -> Vec<f32> {
        let d = self.config.model_dim;
        let mut h = linear_f32(action, &self.action_fc1_w, &self.action_fc1_b, 1, self.config.action_dim, d);
        apply_gelu(&mut h, self.config.gelu_kind);
        linear_f32(&h, &self.action_fc2_w, &self.action_fc2_b, 1, d, d)
    }

    /// Predict next latent from (z_state, z_action).
    pub fn predict(&self, z_state: &[f32], z_action: &[f32]) -> Vec<f32> {
        let d = self.config.model_dim;
        let mut x = vec![0.0_f32; 2 * d];
        x[..d].copy_from_slice(z_state);
        x[d..].copy_from_slice(z_action);
        for block in &self.predictor_blocks {
            for _ in 0..self.config.predictor_loops {
                x = block.forward(&x, 2, &self.config);
            }
        }
        let tok0: Vec<f32> = x[..d].to_vec();
        layernorm_with_bias(
            &tok0,
            &self.predictor_final_norm_w,
            &self.predictor_final_norm_b,
            self.config.layernorm_eps,
            d,
        )
    }
}

/// Convert an f32 CodeWorldModel into its INT8 quantized counterpart.
pub fn quantize_code_wm(model: &CodeWorldModel) -> QuantizedCodeWorldModel {
    let cfg = model.config.clone();
    let d = cfg.model_dim;
    let mlp_h = cfg.mlp_hidden;

    let quantize_block = |src: &crate::models::vision::code_wm::TransformerBlock| -> QuantizedTransformerBlock {
        QuantizedTransformerBlock {
            norm1_w: AlignedBuffer::from_slice(&src.norm1.weight),
            norm1_b: AlignedBuffer::from_slice(&src.norm1.bias),
            attn_in_proj: QuantizedLinear::from_f32(&src.attn_in_proj.weight, 3 * d, d),
            attn_in_proj_bias: AlignedBuffer::from_slice(&src.attn_in_proj.bias),
            attn_out_proj: QuantizedLinear::from_f32(&src.attn_out_proj.weight, d, d),
            attn_out_proj_bias: AlignedBuffer::from_slice(&src.attn_out_proj.bias),
            norm2_w: AlignedBuffer::from_slice(&src.norm2.weight),
            norm2_b: AlignedBuffer::from_slice(&src.norm2.bias),
            mlp_up: QuantizedLinear::from_f32(&src.mlp_up.weight, mlp_h, d),
            mlp_up_bias: AlignedBuffer::from_slice(&src.mlp_up.bias),
            mlp_down: QuantizedLinear::from_f32(&src.mlp_down.weight, d, mlp_h),
            mlp_down_bias: AlignedBuffer::from_slice(&src.mlp_down.bias),
        }
    };

    QuantizedCodeWorldModel {
        config: cfg,
        token_embedding: AlignedBuffer::from_slice(&model.token_embedding),
        cls_token: AlignedBuffer::from_slice(&model.cls_token),
        pos_enc: AlignedBuffer::from_slice(&model.pos_enc),
        encoder_block: quantize_block(&model.encoder_block),
        encoder_final_norm_w: AlignedBuffer::from_slice(&model.encoder_final_norm.weight),
        encoder_final_norm_b: AlignedBuffer::from_slice(&model.encoder_final_norm.bias),
        action_fc1_w: AlignedBuffer::from_slice(&model.action_fc1.weight),
        action_fc1_b: AlignedBuffer::from_slice(&model.action_fc1.bias),
        action_fc2_w: AlignedBuffer::from_slice(&model.action_fc2.weight),
        action_fc2_b: AlignedBuffer::from_slice(&model.action_fc2.bias),
        predictor_blocks: model.predictor_blocks.iter().map(&quantize_block).collect(),
        predictor_final_norm_w: AlignedBuffer::from_slice(&model.predictor_final_norm.weight),
        predictor_final_norm_b: AlignedBuffer::from_slice(&model.predictor_final_norm.bias),
    }
}

/// Load an f32 Code WM from safetensors + config, then quantize.
pub fn load_and_quantize(
    config: &CodeWorldModelConfig,
    weights: HashMap<String, RawTensor>,
) -> Result<QuantizedCodeWorldModel, WeightError> {
    let mut f32_model = CodeWorldModel::from_config(config);
    f32_model.load_weights(weights)?;
    Ok(quantize_code_wm(&f32_model))
}
