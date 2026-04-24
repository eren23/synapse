//! Q4_0-quantized CodeDeltaTok head.
//!
//! In-memory quantization of [`super::code_deltatok::CodeDeltaTokHead`]:
//! every `Linear`-style weight matrix (per-block attention Q/K/V/O plus
//! SwiGLU gate/up/down and the final `out_proj`) is compressed into
//! [`Q4Linear`] blocks (32-element rows, f32 scale + 16 nibble bytes per
//! block). Biases, LayerNorm weights/biases, LayerScale vectors, and the
//! small parameter embeddings (`z_embed`, `pos_prev`, `pos_next`, `pos_z`)
//! stay fp32 — they're tiny relative to the linears and sensitive to the
//! last bits of precision.
//!
//! Memory footprint (paper default: D=768, 4×2 blocks, K=1):
//!   - fp32 head:    ~305 MB on disk (76 M params × 4 bytes)
//!   - Q4 head:      ~48 MB in RAM  (roughly 6.4×; biases/norms dominate
//!                  the residual bytes)
//!
//! Forward semantics mirror `CodeDeltaTokHead` exactly except that the big
//! matmuls go through `Q4Linear::forward`. Expect cosine ~0.99+ vs fp32;
//! Q4_0 drift is additive per block, so deeper stacks drift more.

use crate::models::text_encoder::code_deltatok::{CodeDeltaTokConfig, CodeDeltaTokHead, DeltaTokBlock};
use crate::ops::pure_rust_ops::{layernorm_with_bias, silu};
use crate::quantization::primitives::Q4Linear;
use crate::weight_loading::AlignedBuffer;

pub struct Q4DeltaTokBlock {
    pub dim: usize,
    pub num_heads: usize,
    pub intermediate: usize,
    pub layer_norm_eps: f32,

    pub norm1_w: AlignedBuffer, pub norm1_b: AlignedBuffer,
    pub norm2_w: AlignedBuffer, pub norm2_b: AlignedBuffer,

    pub w_q: Q4Linear, pub q_bias: AlignedBuffer,
    pub w_k: Q4Linear, pub k_bias: AlignedBuffer,
    pub w_v: Q4Linear, pub v_bias: AlignedBuffer,
    pub w_o: Q4Linear, pub o_bias: AlignedBuffer,

    pub mlp_gate: Q4Linear, pub mlp_gate_b: AlignedBuffer,
    pub mlp_up:   Q4Linear, pub mlp_up_b:   AlignedBuffer,
    pub mlp_down: Q4Linear, pub mlp_down_b: AlignedBuffer,

    pub scale1: AlignedBuffer,
    pub scale2: AlignedBuffer,
}

impl Q4DeltaTokBlock {
    /// Quantize the linear weights of a fp32 [`DeltaTokBlock`] into Q4_0
    /// and copy the fp32-staying tensors (biases, norms, LayerScale) by
    /// value.
    pub fn from_fp32(blk: &DeltaTokBlock) -> Self {
        let d = blk.dim;
        let inter = blk.intermediate;

        Self {
            dim: d,
            num_heads: blk.num_heads,
            intermediate: inter,
            layer_norm_eps: blk.layer_norm_eps,

            norm1_w: blk.norm1_w.clone(), norm1_b: blk.norm1_b.clone(),
            norm2_w: blk.norm2_w.clone(), norm2_b: blk.norm2_b.clone(),

            w_q: Q4Linear::from_f32(&blk.w_q, d, d),
            q_bias: blk.q_bias.clone(),
            w_k: Q4Linear::from_f32(&blk.w_k, d, d),
            k_bias: blk.k_bias.clone(),
            w_v: Q4Linear::from_f32(&blk.w_v, d, d),
            v_bias: blk.v_bias.clone(),
            w_o: Q4Linear::from_f32(&blk.w_o, d, d),
            o_bias: blk.o_bias.clone(),

            mlp_gate: Q4Linear::from_f32(&blk.mlp_gate_w, inter, d),
            mlp_gate_b: blk.mlp_gate_b.clone(),
            mlp_up:   Q4Linear::from_f32(&blk.mlp_up_w, inter, d),
            mlp_up_b: blk.mlp_up_b.clone(),
            mlp_down: Q4Linear::from_f32(&blk.mlp_down_w, d, inter),
            mlp_down_b: blk.mlp_down_b.clone(),

            scale1: blk.scale1.clone(),
            scale2: blk.scale2.clone(),
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

    pub fn forward(&self, x: &[f32], seq_len: usize) -> Vec<f32> {
        let d = self.dim;
        let head_dim = d / self.num_heads;

        // Attention sub-layer.
        let h1 = layernorm_with_bias(x, &self.norm1_w, &self.norm1_b, self.layer_norm_eps, d);
        let mut q = self.w_q.forward_batched(&h1, seq_len);
        Self::add_col_bias(&mut q, &self.q_bias, seq_len, d);
        let mut k = self.w_k.forward_batched(&h1, seq_len);
        Self::add_col_bias(&mut k, &self.k_bias, seq_len, d);
        let mut v = self.w_v.forward_batched(&h1, seq_len);
        Self::add_col_bias(&mut v, &self.v_bias, seq_len, d);

        let attn = crate::ops::attention::bidirectional_attention(
            &q, &k, &v, seq_len, self.num_heads, head_dim);
        let mut attn_out = self.w_o.forward_batched(&attn, seq_len);
        Self::add_col_bias(&mut attn_out, &self.o_bias, seq_len, d);

        let mut y = vec![0.0f32; seq_len * d];
        for i in 0..seq_len {
            for j in 0..d {
                y[i * d + j] = x[i * d + j] + self.scale1[j] * attn_out[i * d + j];
            }
        }

        // Feed-forward sub-layer.
        let h2 = layernorm_with_bias(&y, &self.norm2_w, &self.norm2_b, self.layer_norm_eps, d);
        let mut gate = self.mlp_gate.forward_batched(&h2, seq_len);
        Self::add_col_bias(&mut gate, &self.mlp_gate_b, seq_len, self.intermediate);
        for g in gate.iter_mut() { *g = silu(*g); }

        let mut up = self.mlp_up.forward_batched(&h2, seq_len);
        Self::add_col_bias(&mut up, &self.mlp_up_b, seq_len, self.intermediate);

        for i in 0..gate.len() {
            gate[i] *= up[i];
        }

        let mut down = self.mlp_down.forward_batched(&gate, seq_len);
        Self::add_col_bias(&mut down, &self.mlp_down_b, seq_len, d);

        for i in 0..seq_len {
            for j in 0..d {
                y[i * d + j] += self.scale2[j] * down[i * d + j];
            }
        }

        y
    }

    /// Total Q4 block storage in bytes (linear weights only; biases etc.
    /// are tiny fp32 buffers and reported separately in the head summary).
    pub fn q4_memory_bytes(&self) -> usize {
        self.w_q.memory_bytes()
            + self.w_k.memory_bytes()
            + self.w_v.memory_bytes()
            + self.w_o.memory_bytes()
            + self.mlp_gate.memory_bytes()
            + self.mlp_up.memory_bytes()
            + self.mlp_down.memory_bytes()
    }
}

pub struct Q4CodeDeltaTokHead {
    pub config: CodeDeltaTokConfig,

    pub z_embed:  AlignedBuffer,
    pub pos_prev: AlignedBuffer,
    pub pos_next: AlignedBuffer,
    pub pos_z:    AlignedBuffer,

    pub encoder: Vec<Q4DeltaTokBlock>,
    pub enc_norm_w: AlignedBuffer,
    pub enc_norm_b: AlignedBuffer,

    pub decoder: Vec<Q4DeltaTokBlock>,
    pub dec_norm_w: AlignedBuffer,
    pub dec_norm_b: AlignedBuffer,

    pub out_proj: Q4Linear,
    pub out_proj_b: AlignedBuffer,
}

impl Q4CodeDeltaTokHead {
    /// Quantize an fp32 head to Q4_0 in memory. The source head is left
    /// untouched.
    pub fn from_fp32(h: &CodeDeltaTokHead) -> Self {
        let cfg = h.config.clone();
        let d = cfg.feature_dim;
        let encoder = h.encoder.iter().map(Q4DeltaTokBlock::from_fp32).collect();
        let decoder = h.decoder.iter().map(Q4DeltaTokBlock::from_fp32).collect();
        Self {
            config: cfg,
            z_embed:  h.z_embed.clone(),
            pos_prev: h.pos_prev.clone(),
            pos_next: h.pos_next.clone(),
            pos_z:    h.pos_z.clone(),
            encoder,
            enc_norm_w: h.enc_norm_w.clone(), enc_norm_b: h.enc_norm_b.clone(),
            decoder,
            dec_norm_w: h.dec_norm_w.clone(), dec_norm_b: h.dec_norm_b.clone(),
            out_proj:   Q4Linear::from_f32(&h.out_proj_w, d, d),
            out_proj_b: h.out_proj_b.clone(),
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

    pub fn encode(&self, h_b: &[f32], h_a: &[f32]) -> Vec<f32> {
        let cfg = &self.config;
        let d = cfg.feature_dim;
        let k = cfg.num_delta_tokens;

        let mut seq = vec![0.0f32; (k + 2) * d];
        for i in 0..k {
            for j in 0..d {
                seq[i * d + j] = self.z_embed[i * d + j] + self.pos_z[i * d + j];
            }
        }
        for j in 0..d {
            seq[k * d + j] = h_b[j] + self.pos_prev[j];
        }
        for j in 0..d {
            seq[(k + 1) * d + j] = h_a[j] + self.pos_next[j];
        }

        let mut h = seq;
        for blk in &self.encoder {
            h = blk.forward(&h, k + 2);
        }
        let h = layernorm_with_bias(&h, &self.enc_norm_w, &self.enc_norm_b,
                                    cfg.layer_norm_eps, d);

        let mut delta = vec![0.0f32; k * d];
        delta.copy_from_slice(&h[..k * d]);
        delta
    }

    pub fn decode(&self, delta: &[f32], h_b: &[f32]) -> Vec<f32> {
        let cfg = &self.config;
        let d = cfg.feature_dim;
        let k = cfg.num_delta_tokens;

        let mut seq = vec![0.0f32; (k + 1) * d];
        seq[..k * d].copy_from_slice(delta);
        for j in 0..d {
            seq[k * d + j] = h_b[j] + self.pos_prev[j];
        }

        let mut h = seq;
        for blk in &self.decoder {
            h = blk.forward(&h, k + 1);
        }
        let h = layernorm_with_bias(&h, &self.dec_norm_w, &self.dec_norm_b,
                                    cfg.layer_norm_eps, d);

        let slice = &h[k * d..(k + 1) * d];
        let mut recon = self.out_proj.forward_batched(slice, 1);
        Self::add_col_bias(&mut recon, &self.out_proj_b, 1, d);
        recon
    }

    /// Total Q4 storage across all linear weights. fp32 buffers (norms,
    /// biases, pos/z embeddings) are negligible but not counted here.
    pub fn q4_memory_bytes(&self) -> usize {
        let mut s = self.out_proj.memory_bytes();
        for b in &self.encoder { s += b.q4_memory_bytes(); }
        for b in &self.decoder { s += b.q4_memory_bytes(); }
        s
    }
}
