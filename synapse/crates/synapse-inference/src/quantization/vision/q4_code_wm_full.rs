//! Full-model quantization for Code WM: Q4 matmul layers + INT8 per-row
//! token_embedding + pos_enc. Biases, layernorms, and the tiny action encoder
//! stay f32.
//!
//! Q4 alone leaves ~660 KB of f32 weights (embedding 336 KB + PE 257 KB +
//! action 68 KB); quantizing the two big tables shrinks the model further.
//!
//! Expected: 1038 KB (Q4) → ~590 KB (Q4-full), ~5.1x vs f32 baseline.
//! Quality delta: token embedding dequantize-on-gather is a 128-op multiply
//! per token — negligible overhead.

use crate::models::vision::code_wm::{CodeWorldModel, CodeWorldModelConfig, GeluKind};
use crate::ops::attention::bidirectional_attention;
use crate::ops::pure_rust_ops::layernorm_with_bias;
use crate::quantization::Q4Linear;
use crate::weight_loading::AlignedBuffer;

// ── GELU (mirror of code_wm.rs) ────────────────────────────────────
#[inline]
fn erf_f32(x: f32) -> f32 {
    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let x = x.abs();
    let t = 1.0 / (1.0 + 0.3275911 * x);
    let y = 1.0
        - (((((1.061405429_f32 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592)
            * t * (-x * x).exp();
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
        GeluKind::Erf => for v in buf.iter_mut() { *v = gelu_erf(*v); },
        GeluKind::Tanh => for v in buf.iter_mut() { *v = gelu_tanh(*v); },
    }
}

fn linear_f32(x: &[f32], weight: &[f32], bias: &[f32], m: usize, in_dim: usize, out_dim: usize) -> Vec<f32> {
    let mut out = crate::ops::matmul::matmul_t(x, weight, m, in_dim, out_dim);
    if !bias.is_empty() {
        for r in 0..m { for j in 0..out_dim { out[r * out_dim + j] += bias[j]; } }
    }
    out
}

// ── Per-row symmetric INT8 quantization for tables ────────────────
//
// Given a [rows, cols] f32 matrix, compute one f32 scale per row and
// store int8 values. Dequantize on gather: `value_f32 = int8 * scale`.
// Memory: rows * cols i8 + rows f32  (vs rows * cols * 4 for f32).
// For 662 × 128: 84.7 KB + 2.6 KB = 87 KB vs 339 KB f32 (3.9x smaller).

pub struct Int8Table {
    pub data: Vec<i8>,      // [rows, cols]
    pub scales: Vec<f32>,   // [rows]
    pub rows: usize,
    pub cols: usize,
}

impl Int8Table {
    pub fn from_f32(values: &[f32], rows: usize, cols: usize) -> Self {
        assert_eq!(values.len(), rows * cols);
        let mut data = vec![0_i8; rows * cols];
        let mut scales = Vec::with_capacity(rows);
        for r in 0..rows {
            let row = &values[r * cols..(r + 1) * cols];
            let max_abs = row.iter().fold(0f32, |acc, &v| acc.max(v.abs()));
            let scale = if max_abs == 0.0 { 1.0 } else { max_abs / 127.0 };
            let inv = 1.0 / scale;
            for j in 0..cols {
                let q = (values[r * cols + j] * inv).round().clamp(-128.0, 127.0) as i8;
                data[r * cols + j] = q;
            }
            scales.push(scale);
        }
        Self { data, scales, rows, cols }
    }

    /// Dequantize one row (row index `r`) into the output slice.
    #[inline]
    pub fn dequant_row(&self, r: usize, out: &mut [f32]) {
        debug_assert!(r < self.rows);
        debug_assert_eq!(out.len(), self.cols);
        let scale = self.scales[r];
        let row = &self.data[r * self.cols..(r + 1) * self.cols];
        for j in 0..self.cols {
            out[j] = (row[j] as f32) * scale;
        }
    }

    /// Dequantize a row and add it into the output slice (`out[j] += row[j] * scale`).
    #[inline]
    pub fn dequant_row_add(&self, r: usize, out: &mut [f32]) {
        debug_assert!(r < self.rows);
        debug_assert_eq!(out.len(), self.cols);
        let scale = self.scales[r];
        let row = &self.data[r * self.cols..(r + 1) * self.cols];
        for j in 0..self.cols {
            out[j] += (row[j] as f32) * scale;
        }
    }

    pub fn memory_bytes(&self) -> usize {
        self.data.len() + 4 * self.scales.len()
    }
}

// ── Q4 transformer block (reuse from q4_code_wm) ──────────────────
pub struct Q4FullTransformerBlock {
    pub norm1_w: AlignedBuffer,
    pub norm1_b: AlignedBuffer,
    pub attn_in_proj: Q4Linear,
    pub attn_in_proj_bias: AlignedBuffer,
    pub attn_out_proj: Q4Linear,
    pub attn_out_proj_bias: AlignedBuffer,
    pub norm2_w: AlignedBuffer,
    pub norm2_b: AlignedBuffer,
    pub mlp_up: Q4Linear,
    pub mlp_up_bias: AlignedBuffer,
    pub mlp_down: Q4Linear,
    pub mlp_down_bias: AlignedBuffer,
}

impl Q4FullTransformerBlock {
    fn forward(&self, x: &[f32], seq_len: usize, cfg: &CodeWorldModelConfig) -> Vec<f32> {
        let d = cfg.model_dim;
        let normed1 = layernorm_with_bias(x, &self.norm1_w, &self.norm1_b, cfg.layernorm_eps, d);
        let mut qkv = self.attn_in_proj.forward(&normed1, seq_len);
        for r in 0..seq_len {
            for j in 0..3 * d { qkv[r * 3 * d + j] += self.attn_in_proj_bias[j]; }
        }
        let mut q = vec![0.0_f32; seq_len * d];
        let mut k = vec![0.0_f32; seq_len * d];
        let mut v = vec![0.0_f32; seq_len * d];
        for t in 0..seq_len {
            let o_src = t * 3 * d; let o_dst = t * d;
            q[o_dst..o_dst + d].copy_from_slice(&qkv[o_src..o_src + d]);
            k[o_dst..o_dst + d].copy_from_slice(&qkv[o_src + d..o_src + 2 * d]);
            v[o_dst..o_dst + d].copy_from_slice(&qkv[o_src + 2 * d..o_src + 3 * d]);
        }
        let attn_out = bidirectional_attention(&q, &k, &v, seq_len, cfg.num_heads, cfg.head_dim);
        let mut attn_proj = self.attn_out_proj.forward(&attn_out, seq_len);
        for r in 0..seq_len { for j in 0..d { attn_proj[r * d + j] += self.attn_out_proj_bias[j]; } }
        let mut res1 = vec![0.0_f32; seq_len * d];
        for i in 0..res1.len() { res1[i] = x[i] + attn_proj[i]; }

        let normed2 = layernorm_with_bias(&res1, &self.norm2_w, &self.norm2_b, cfg.layernorm_eps, d);
        let mut up = self.mlp_up.forward(&normed2, seq_len);
        for r in 0..seq_len { for j in 0..cfg.mlp_hidden { up[r * cfg.mlp_hidden + j] += self.mlp_up_bias[j]; } }
        apply_gelu(&mut up, cfg.gelu_kind);
        let mut down = self.mlp_down.forward(&up, seq_len);
        for r in 0..seq_len { for j in 0..d { down[r * d + j] += self.mlp_down_bias[j]; } }
        let mut out = vec![0.0_f32; seq_len * d];
        for i in 0..out.len() { out[i] = res1[i] + down[i]; }
        out
    }

    fn bytes(&self) -> usize {
        self.attn_in_proj.memory_bytes() + self.attn_out_proj.memory_bytes()
            + self.mlp_up.memory_bytes() + self.mlp_down.memory_bytes()
            + 4 * (self.norm1_w.len() + self.norm1_b.len() + self.norm2_w.len() + self.norm2_b.len()
                + self.attn_in_proj_bias.len() + self.attn_out_proj_bias.len()
                + self.mlp_up_bias.len() + self.mlp_down_bias.len())
    }
}

// ── Q4-full Code WM with INT8 embedding + PE ───────────────────────
pub struct Q4FullCodeWorldModel {
    pub config: CodeWorldModelConfig,
    // INT8-quantized tables (per-row scales)
    pub token_embedding: Int8Table,   // [vocab, D]
    pub pos_enc: Int8Table,           // [max_seq+1, D]
    // f32 (unchanged — small / precision-sensitive)
    pub cls_token: AlignedBuffer,
    pub encoder_block: Q4FullTransformerBlock,
    pub encoder_final_norm_w: AlignedBuffer,
    pub encoder_final_norm_b: AlignedBuffer,
    pub action_fc1_w: AlignedBuffer,
    pub action_fc1_b: AlignedBuffer,
    pub action_fc2_w: AlignedBuffer,
    pub action_fc2_b: AlignedBuffer,
    pub predictor_blocks: Vec<Q4FullTransformerBlock>,
    pub predictor_final_norm_w: AlignedBuffer,
    pub predictor_final_norm_b: AlignedBuffer,
}

impl Q4FullCodeWorldModel {
    pub fn memory_bytes(&self) -> usize {
        self.token_embedding.memory_bytes()
            + self.pos_enc.memory_bytes()
            + 4 * self.cls_token.len()
            + self.encoder_block.bytes()
            + 4 * (self.encoder_final_norm_w.len() + self.encoder_final_norm_b.len()
                + self.action_fc1_w.len() + self.action_fc1_b.len()
                + self.action_fc2_w.len() + self.action_fc2_b.len()
                + self.predictor_final_norm_w.len() + self.predictor_final_norm_b.len())
            + self.predictor_blocks.iter().map(|b| b.bytes()).sum::<usize>()
    }

    pub fn encode(&self, tokens: &[i64]) -> Vec<f32> {
        let d = self.config.model_dim;
        let s = tokens.len();
        let seq_with_cls = s + 1;

        let mut h = vec![0.0_f32; seq_with_cls * d];
        // CLS token at position 0 (f32)
        h[..d].copy_from_slice(&self.cls_token);
        // Dequantize embedding rows into positions 1..s+1
        for (i, &tok) in tokens.iter().enumerate() {
            let t = tok as usize;
            let dst = &mut h[(i + 1) * d..(i + 2) * d];
            self.token_embedding.dequant_row(t, dst);
        }
        // Add dequantized PE (for positions 0..seq_with_cls)
        for i in 0..seq_with_cls {
            let off = i * d;
            self.pos_enc.dequant_row_add(i, &mut h[off..off + d]);
        }

        for _ in 0..self.config.encoder_loops {
            h = self.encoder_block.forward(&h, seq_with_cls, &self.config);
        }
        let cls_out: Vec<f32> = h[..d].to_vec();
        layernorm_with_bias(
            &cls_out, &self.encoder_final_norm_w, &self.encoder_final_norm_b,
            self.config.layernorm_eps, d,
        )
    }

    pub fn encode_action(&self, action: &[f32]) -> Vec<f32> {
        let d = self.config.model_dim;
        let mut h = linear_f32(action, &self.action_fc1_w, &self.action_fc1_b, 1, self.config.action_dim, d);
        apply_gelu(&mut h, self.config.gelu_kind);
        linear_f32(&h, &self.action_fc2_w, &self.action_fc2_b, 1, d, d)
    }

    pub fn predict(&self, z_state: &[f32], z_action: &[f32]) -> Vec<f32> {
        let d = self.config.model_dim;
        let mut x = vec![0.0_f32; 2 * d];
        x[..d].copy_from_slice(z_state);
        x[d..].copy_from_slice(z_action);
        for block in &self.predictor_blocks {
            for _ in 0..self.config.predictor_loops { x = block.forward(&x, 2, &self.config); }
        }
        let tok0: Vec<f32> = x[..d].to_vec();
        layernorm_with_bias(
            &tok0, &self.predictor_final_norm_w, &self.predictor_final_norm_b,
            self.config.layernorm_eps, d,
        )
    }
}

/// Convert an f32 CodeWorldModel to Q4 matmul + INT8 embedding/PE.
pub fn quantize_code_wm_q4_full(model: &CodeWorldModel) -> Q4FullCodeWorldModel {
    let cfg = model.config.clone();
    let d = cfg.model_dim;
    let mlp_h = cfg.mlp_hidden;
    let pe_rows = cfg.max_seq_len + 1;

    let quantize_block = |src: &crate::models::vision::code_wm::TransformerBlock| -> Q4FullTransformerBlock {
        Q4FullTransformerBlock {
            norm1_w: AlignedBuffer::from_slice(&src.norm1.weight),
            norm1_b: AlignedBuffer::from_slice(&src.norm1.bias),
            attn_in_proj: Q4Linear::from_f32(&src.attn_in_proj.weight, 3 * d, d),
            attn_in_proj_bias: AlignedBuffer::from_slice(&src.attn_in_proj.bias),
            attn_out_proj: Q4Linear::from_f32(&src.attn_out_proj.weight, d, d),
            attn_out_proj_bias: AlignedBuffer::from_slice(&src.attn_out_proj.bias),
            norm2_w: AlignedBuffer::from_slice(&src.norm2.weight),
            norm2_b: AlignedBuffer::from_slice(&src.norm2.bias),
            mlp_up: Q4Linear::from_f32(&src.mlp_up.weight, mlp_h, d),
            mlp_up_bias: AlignedBuffer::from_slice(&src.mlp_up.bias),
            mlp_down: Q4Linear::from_f32(&src.mlp_down.weight, d, mlp_h),
            mlp_down_bias: AlignedBuffer::from_slice(&src.mlp_down.bias),
        }
    };

    Q4FullCodeWorldModel {
        config: cfg.clone(),
        token_embedding: Int8Table::from_f32(&model.token_embedding, cfg.vocab_size, d),
        pos_enc: Int8Table::from_f32(&model.pos_enc, pe_rows, d),
        cls_token: AlignedBuffer::from_slice(&model.cls_token),
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
