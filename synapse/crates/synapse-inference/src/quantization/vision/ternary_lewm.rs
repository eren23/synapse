//! Ternary (2-bit) quantized LEWM predictor with TerDiT RMSNorm stabilization.
//!
//! The key insight from TerDiT (2025): ternary DiT models require adding RMSNorm
//! after the adaLN modulation MLP to stabilize the scale/shift/gate values.
//! Without this, large modulation values destabilize the ternary forward pass.
//!
//! Architecture:
//! - adaLN conditioning: **QuantizedLinear (INT8)** — most quantization-sensitive
//! - QKV, attn_out, MLP: **TernaryLinear (2-bit)** — robust, 16x compression
//! - Norms, biases: f32
//! - ViT encoder: f32 (unchanged, only ~2.8M params)
//!
//! This gives ~8-10x total compression vs f32, with the RMSNorm fix
//! maintaining prediction quality.

use crate::models::vision::lewm::{LeWMConfig, LeWorldModel, ProjectionHead};
use crate::models::vision::vit::ViTModel;
use crate::ops::activation::gelu;
use crate::ops::attention::bidirectional_attention;
use crate::ops::matmul::matmul_t;
use crate::ops::norm::layernorm;
use crate::quantization::QuantizedLinear;
use crate::quantization::TernaryLinear;

/// RMSNorm: normalize by root-mean-square (no learned weight, just stabilization).
#[inline]
fn rmsnorm_unweighted(x: &mut [f32], eps: f32) {
    let n = x.len();
    if n == 0 { return; }
    let rms: f32 = (x.iter().map(|v| v * v).sum::<f32>() / n as f32 + eps).sqrt();
    let inv_rms = 1.0 / rms;
    for v in x.iter_mut() {
        *v *= inv_rms;
    }
}

/// A ternary adaLN layer with INT8 conditioning and RMSNorm stabilization.
pub struct TernaryAdaLNLayer {
    // adaLN modulation: INT8 (sensitive to quantization)
    pub adaln_linear: QuantizedLinear,  // [6*hidden, hidden]
    pub adaln_bias: Vec<f32>,
    // QKV: ternary (large, robust)
    pub to_qkv: TernaryLinear,          // [3*inner_dim, hidden]
    // Output projection: ternary
    pub attn_out: TernaryLinear,         // [hidden, inner_dim]
    pub attn_out_bias: Vec<f32>,
    // Norms stay f32
    pub attn_norm_weight: Vec<f32>,
    pub attn_norm_bias: Vec<f32>,
    pub mlp_norm_weight: Vec<f32>,
    pub mlp_norm_bias: Vec<f32>,
    // MLP: ternary (largest layers, most benefit from compression)
    pub mlp_up: TernaryLinear,           // [inter, hidden]
    pub mlp_up_bias: Vec<f32>,
    pub mlp_down: TernaryLinear,         // [hidden, inter]
    pub mlp_down_bias: Vec<f32>,
}

impl TernaryAdaLNLayer {
    /// Forward pass with TerDiT RMSNorm stabilization.
    ///
    /// The key difference from QuantizedAdaLNLayer: RMSNorm is applied to the
    /// adaLN modulation vector before splitting into scale/shift/gate components.
    pub fn forward(
        &self,
        x: &[f32],
        conditioning: &[f32],
        seq_len: usize,
        hidden: usize,
        num_heads: usize,
        inner_dim: usize,
        inter: usize,
    ) -> Vec<f32> {
        let head_dim = inner_dim / num_heads;
        let mod_dim = 6 * hidden;

        // 1. Compute adaLN modulation via INT8: conditioning [hidden] → mod_vec [6*hidden]
        let mut mod_vec = self.adaln_linear.forward(conditioning, 1);
        debug_assert_eq!(mod_vec.len(), mod_dim);
        for j in 0..mod_dim.min(self.adaln_bias.len()) {
            mod_vec[j] += self.adaln_bias[j];
        }

        // ★ TerDiT RMSNorm fix: normalize modulation vector to prevent
        // large scale/shift values from destabilizing ternary computation.
        rmsnorm_unweighted(&mut mod_vec, 1e-6);

        // Split into 6 vectors of [hidden]
        let scale1 = &mod_vec[0..hidden];
        let shift1 = &mod_vec[hidden..2 * hidden];
        let gate1 = &mod_vec[2 * hidden..3 * hidden];
        let scale2 = &mod_vec[3 * hidden..4 * hidden];
        let shift2 = &mod_vec[4 * hidden..5 * hidden];
        let gate2 = &mod_vec[5 * hidden..6 * hidden];

        let mut residual = x.to_vec();

        // 2. Pre-attention: layernorm + modulate
        let normed = layernorm(x, &self.attn_norm_weight, 1e-6, hidden);
        let mut modulated = vec![0.0f32; seq_len * hidden];
        for t in 0..seq_len {
            for j in 0..hidden {
                let idx = t * hidden + j;
                modulated[idx] = normed[idx] * (1.0 + scale1[j]) + shift1[j];
            }
        }

        // 3. QKV via ternary: [seq_len, hidden] → [seq_len, 3*inner_dim]
        let qkv = self.to_qkv.forward(&modulated, seq_len);

        let mut q = vec![0.0f32; seq_len * inner_dim];
        let mut k = vec![0.0f32; seq_len * inner_dim];
        let mut v = vec![0.0f32; seq_len * inner_dim];
        for t in 0..seq_len {
            let qkv_off = t * 3 * inner_dim;
            let off = t * inner_dim;
            q[off..off + inner_dim].copy_from_slice(&qkv[qkv_off..qkv_off + inner_dim]);
            k[off..off + inner_dim]
                .copy_from_slice(&qkv[qkv_off + inner_dim..qkv_off + 2 * inner_dim]);
            v[off..off + inner_dim]
                .copy_from_slice(&qkv[qkv_off + 2 * inner_dim..qkv_off + 3 * inner_dim]);
        }

        let attn_out = bidirectional_attention(&q, &k, &v, seq_len, num_heads, head_dim);

        // Output via ternary: [seq_len, inner_dim] → [seq_len, hidden]
        let mut proj = self.attn_out.forward(&attn_out, seq_len);
        for t in 0..seq_len {
            for j in 0..hidden.min(self.attn_out_bias.len()) {
                proj[t * hidden + j] += self.attn_out_bias[j];
            }
        }

        // 4. Gated residual
        for t in 0..seq_len {
            for j in 0..hidden {
                let idx = t * hidden + j;
                residual[idx] += gate1[j] * proj[idx];
            }
        }

        // 5. Pre-FFN: layernorm + modulate
        let normed2 = layernorm(&residual, &self.mlp_norm_weight, 1e-6, hidden);
        let mut modulated2 = vec![0.0f32; seq_len * hidden];
        for t in 0..seq_len {
            for j in 0..hidden {
                let idx = t * hidden + j;
                modulated2[idx] = normed2[idx] * (1.0 + scale2[j]) + shift2[j];
            }
        }

        // 6. MLP via ternary: up → GELU → down
        let mut up = self.mlp_up.forward(&modulated2, seq_len);
        for t in 0..seq_len {
            for j in 0..inter.min(self.mlp_up_bias.len()) {
                up[t * inter + j] += self.mlp_up_bias[j];
            }
        }
        for v in up.iter_mut() {
            *v = gelu(*v);
        }
        let mut down = self.mlp_down.forward(&up, seq_len);
        for t in 0..seq_len {
            for j in 0..hidden.min(self.mlp_down_bias.len()) {
                down[t * hidden + j] += self.mlp_down_bias[j];
            }
        }

        // 7. Gated residual
        for t in 0..seq_len {
            for j in 0..hidden {
                let idx = t * hidden + j;
                residual[idx] += gate2[j] * down[idx];
            }
        }

        residual
    }
}

/// Ternary-quantized LEWM: 2-bit predictor with INT8 adaLN + RMSNorm fix.
pub struct TernaryLeWM {
    pub config: LeWMConfig,
    pub encoder: ViTModel,            // stays f32
    pub predictor_layers: Vec<TernaryAdaLNLayer>,
    pub predictor_pos_embed: Vec<f32>,
    pub predictor_norm_weight: Vec<f32>,
    pub predictor_norm_bias: Vec<f32>,
    pub action_conv_weight: Vec<f32>,
    pub action_conv_bias: Vec<f32>,
    pub action_mlp1_weight: Vec<f32>,
    pub action_mlp1_bias: Vec<f32>,
    pub action_mlp2_weight: Vec<f32>,
    pub action_mlp2_bias: Vec<f32>,
    pub projector: ProjectionHead,
    pub pred_proj: ProjectionHead,
}

impl TernaryLeWM {
    pub fn encode(&self, image: &[f32], h: usize, w: usize) -> Vec<f32> {
        let vit_out = self.encoder.forward_image(image, h, w);
        self.projector.forward(&vit_out.embeddings)
    }

    pub fn predict_next(&self, z_t: &[f32], action: &[f32]) -> Vec<f32> {
        let hidden = self.config.predictor_hidden;
        let num_heads = self.config.predictor_heads;
        let inner_dim = self.config.predictor_inner_dim;
        let inter = self.config.predictor_inter;
        let latent_dim = self.config.latent_dim;

        // Action embedding (same as f32 version)
        let action_hidden = self.encode_action(action, hidden);

        // Build input sequence: [z_t_projected] with positional embedding
        let seq_len = latent_dim / hidden;
        let mut x = vec![0.0f32; seq_len * hidden];
        for t in 0..seq_len {
            for j in 0..hidden {
                let idx = t * hidden + j;
                x[idx] = z_t[t * hidden + j] + self.predictor_pos_embed[idx];
            }
        }

        // Run through ternary predictor layers
        for layer in &self.predictor_layers {
            x = layer.forward(&x, &action_hidden, seq_len, hidden, num_heads, inner_dim, inter);
        }

        // Final norm + projection
        let normed = layernorm(&x, &self.predictor_norm_weight, 1e-6, hidden);
        self.pred_proj.forward(&normed)
    }

    pub fn rollout(&self, z_start: &[f32], actions: &[Vec<f32>]) -> Vec<Vec<f32>> {
        let mut states = Vec::with_capacity(actions.len());
        let mut z = z_start.to_vec();
        for action in actions {
            z = self.predict_next(&z, action);
            states.push(z.clone());
        }
        states
    }

    fn encode_action(&self, action: &[f32], hidden: usize) -> Vec<f32> {
        let act_dim = self.config.action_dim;

        // 1. 1D conv with kernel_size=1: action [act_dim] → [act_dim]
        let mut conv_out = vec![0.0f32; act_dim];
        if !self.action_conv_weight.is_empty() {
            conv_out = matmul_t(action, &self.action_conv_weight, 1, act_dim, act_dim);
            for j in 0..act_dim.min(self.action_conv_bias.len()) {
                conv_out[j] += self.action_conv_bias[j];
            }
        } else {
            conv_out.copy_from_slice(action);
        }

        // 2. MLP: [act_dim] → [inter] (GELU) → [hidden]
        let inter = if !self.action_mlp1_weight.is_empty() {
            self.action_mlp1_weight.len() / act_dim
        } else {
            hidden * 4
        };

        let mut h1 = if !self.action_mlp1_weight.is_empty() {
            matmul_t(&conv_out, &self.action_mlp1_weight, 1, act_dim, inter)
        } else {
            vec![0.0f32; inter]
        };
        for j in 0..inter.min(self.action_mlp1_bias.len()) {
            h1[j] += self.action_mlp1_bias[j];
        }
        for val in h1.iter_mut() {
            *val = gelu(*val);
        }

        let mut out = if !self.action_mlp2_weight.is_empty() {
            matmul_t(&h1, &self.action_mlp2_weight, 1, inter, hidden)
        } else {
            vec![0.0f32; hidden]
        };
        for j in 0..hidden.min(self.action_mlp2_bias.len()) {
            out[j] += self.action_mlp2_bias[j];
        }

        out
    }

    /// Estimated model size in bytes.
    pub fn model_size_bytes(&self) -> usize {
        let mut size = 0;

        // Encoder (f32)
        // Approximate: encoder is ~2.8M params * 4 bytes
        size += 2_800_000 * 4;

        // Predictor layers
        for layer in &self.predictor_layers {
            // INT8 adaLN: weight + scale per row
            size += layer.adaln_linear.memory_bytes();
            size += layer.adaln_bias.len() * 4;
            // Ternary projections: ~0.5 bytes/weight + scale overhead
            size += layer.to_qkv.memory_bytes();
            size += layer.attn_out.memory_bytes();
            size += layer.attn_out_bias.len() * 4;
            size += layer.mlp_up.memory_bytes();
            size += layer.mlp_up_bias.len() * 4;
            size += layer.mlp_down.memory_bytes();
            size += layer.mlp_down_bias.len() * 4;
            // Norms (f32, small)
            size += (layer.attn_norm_weight.len() + layer.attn_norm_bias.len()
                + layer.mlp_norm_weight.len() + layer.mlp_norm_bias.len()) * 4;
        }

        // Pos embed, norms, action embed, projectors
        size += self.predictor_pos_embed.len() * 4;
        size += (self.predictor_norm_weight.len() + self.predictor_norm_bias.len()) * 4;
        size += (self.action_conv_weight.len() + self.action_conv_bias.len()
            + self.action_mlp1_weight.len() + self.action_mlp1_bias.len()
            + self.action_mlp2_weight.len() + self.action_mlp2_bias.len()) * 4;

        size
    }
}

/// Quantize a f32 LEWM to ternary predictor with INT8 adaLN.
pub fn quantize_lewm_ternary(model: &LeWorldModel) -> TernaryLeWM {
    let cfg = &model.config;
    let hidden = cfg.predictor_hidden;
    let inner_dim = cfg.predictor_inner_dim;
    let inter = cfg.predictor_inter;

    let layers: Vec<TernaryAdaLNLayer> = model.predictor_layers.iter().map(|layer| {
        TernaryAdaLNLayer {
            // adaLN stays INT8 (most sensitive)
            adaln_linear: QuantizedLinear::from_f32(&layer.adaln_weight, 6 * hidden, hidden),
            adaln_bias: layer.adaln_bias.to_vec(),
            // QKV and projections go ternary (2-bit)
            to_qkv: TernaryLinear::from_f32(&layer.to_qkv, 3 * inner_dim, hidden),
            attn_out: TernaryLinear::from_f32(&layer.attn_out_weight, hidden, inner_dim),
            attn_out_bias: layer.attn_out_bias.to_vec(),
            // Norms stay f32
            attn_norm_weight: layer.attn_norm_weight.to_vec(),
            attn_norm_bias: layer.attn_norm_bias.to_vec(),
            mlp_norm_weight: layer.mlp_norm_weight.to_vec(),
            mlp_norm_bias: layer.mlp_norm_bias.to_vec(),
            // MLP goes ternary
            mlp_up: TernaryLinear::from_f32(&layer.mlp_up_weight, inter, hidden),
            mlp_up_bias: layer.mlp_up_bias.to_vec(),
            mlp_down: TernaryLinear::from_f32(&layer.mlp_down_weight, hidden, inter),
            mlp_down_bias: layer.mlp_down_bias.to_vec(),
        }
    }).collect();

    use super::int8_lewm::{clone_vit_encoder, clone_projection_head};

    TernaryLeWM {
        config: cfg.clone(),
        encoder: clone_vit_encoder(&model.encoder),
        predictor_layers: layers,
        predictor_pos_embed: model.predictor_pos_embed.to_vec(),
        predictor_norm_weight: model.predictor_norm_weight.to_vec(),
        predictor_norm_bias: model.predictor_norm_bias.to_vec(),
        action_conv_weight: model.action_conv_weight.to_vec(),
        action_conv_bias: model.action_conv_bias.to_vec(),
        action_mlp1_weight: model.action_mlp1_weight.to_vec(),
        action_mlp1_bias: model.action_mlp1_bias.to_vec(),
        action_mlp2_weight: model.action_mlp2_weight.to_vec(),
        action_mlp2_bias: model.action_mlp2_bias.to_vec(),
        projector: clone_projection_head(&model.projector),
        pred_proj: clone_projection_head(&model.pred_proj),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rmsnorm_unweighted() {
        let mut v = vec![3.0, 4.0];
        rmsnorm_unweighted(&mut v, 1e-6);
        let rms = (9.0 + 16.0_f32) / 2.0;
        let expected_scale = 1.0 / rms.sqrt();
        assert!((v[0] - 3.0 * expected_scale).abs() < 1e-5);
        assert!((v[1] - 4.0 * expected_scale).abs() < 1e-5);
    }

    #[test]
    fn test_rmsnorm_unweighted_zero() {
        let mut v = vec![0.0, 0.0, 0.0];
        rmsnorm_unweighted(&mut v, 1e-6);
        assert!(v.iter().all(|x| x.abs() < 1e-3));
    }

    #[test]
    fn test_ternary_adaln_layer_forward_produces_finite() {
        let hidden = 8;
        let inner_dim = 16;
        let inter = 32;
        let num_heads = 2;
        let seq_len = 4;

        let layer = TernaryAdaLNLayer {
            adaln_linear: QuantizedLinear::from_f32(
                &vec![0.01f32; 6 * hidden * hidden], 6 * hidden, hidden),
            adaln_bias: vec![0.0; 6 * hidden],
            to_qkv: TernaryLinear::from_f32(
                &(0..3 * inner_dim * hidden).map(|i| (i as f32 * 0.01).sin() * 0.1).collect::<Vec<_>>(),
                3 * inner_dim, hidden),
            attn_out: TernaryLinear::from_f32(
                &vec![0.01; hidden * inner_dim], hidden, inner_dim),
            attn_out_bias: vec![0.0; hidden],
            attn_norm_weight: vec![1.0; hidden],
            attn_norm_bias: vec![0.0; hidden],
            mlp_norm_weight: vec![1.0; hidden],
            mlp_norm_bias: vec![0.0; hidden],
            mlp_up: TernaryLinear::from_f32(
                &vec![0.01; inter * hidden], inter, hidden),
            mlp_up_bias: vec![0.0; inter],
            mlp_down: TernaryLinear::from_f32(
                &vec![0.01; hidden * inter], hidden, inter),
            mlp_down_bias: vec![0.0; hidden],
        };

        let x = vec![0.1f32; seq_len * hidden];
        let cond = vec![0.5f32; hidden];
        let out = layer.forward(&x, &cond, seq_len, hidden, num_heads, inner_dim, inter);

        assert_eq!(out.len(), seq_len * hidden);
        assert!(out.iter().all(|v| v.is_finite()), "output has non-finite values");
    }
}
