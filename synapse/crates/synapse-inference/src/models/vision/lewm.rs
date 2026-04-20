//! LeWorldModel (LeWM) — JEPA-based world model for planning in latent space.
//!
//! Architecture: ViT encoder → projector → DiT predictor (adaLN) → pred_proj.
//! The predictor uses adaptive layer normalization (adaLN) conditioned on the
//! action embedding, following the DiT/LeJEPA design.
//!
//! Paper: <https://le-wm.github.io/>

use std::collections::HashMap;

use crate::ops::activation::gelu;
use crate::ops::attention::bidirectional_attention;
use crate::ops::matmul::matmul_t;
use crate::ops::norm::layernorm;
#[cfg(not(feature = "zig-ffi"))]
use crate::ops::fused_ops::{
    fused_adaln_layernorm_matmul_into,
    fused_attn_proj_gated_residual_into,
    fused_ffn_gated_residual_into,
    fused_layernorm_modulate,
};
use crate::weight_loading::{AlignedBuffer, RawTensor, WeightError};

pub use super::LoadStats;
use super::vit::{ViTConfig, ViTModel};

/// Configuration for a LeWorldModel.
#[derive(Debug, Clone)]
pub struct LeWMConfig {
    pub image_size: usize,
    pub patch_size: usize,
    pub channels: usize,
    pub encoder_hidden: usize,
    pub encoder_layers: usize,
    pub encoder_heads: usize,
    pub encoder_inter: usize,
    pub predictor_hidden: usize,
    pub predictor_layers: usize,
    pub predictor_heads: usize,
    pub predictor_inner_dim: usize,
    pub predictor_inter: usize,
    pub action_dim: usize,
    pub latent_dim: usize,
}

impl LeWMConfig {
    /// Default configuration matching the PushT checkpoint.
    pub fn pusht() -> Self {
        LeWMConfig {
            image_size: 224,
            patch_size: 14,
            channels: 3,
            encoder_hidden: 192,
            encoder_layers: 6,
            encoder_heads: 3,
            encoder_inter: 768,
            predictor_hidden: 192,
            predictor_layers: 6,
            predictor_heads: 16,
            predictor_inner_dim: 1024,
            predictor_inter: 2048,
            action_dim: 10,
            latent_dim: 192,
        }
    }

    /// Slim configuration: 96d latent, 4 encoder layers, 4 predictor layers.
    /// Predictor internals stay at 192d; uses input_proj/cond_proj to bridge.
    pub fn slim() -> Self {
        LeWMConfig {
            image_size: 224,
            patch_size: 14,
            channels: 3,
            encoder_hidden: 192,
            encoder_layers: 4,
            encoder_heads: 3,
            encoder_inter: 768,
            predictor_hidden: 192,
            predictor_layers: 4,
            predictor_heads: 16,
            predictor_inner_dim: 1024,
            predictor_inter: 2048,
            action_dim: 10,
            latent_dim: 96,
        }
    }

    /// Load config from a JSON file (written by convert_lewm_ckpt.py).
    pub fn from_json(path: &std::path::Path) -> Result<Self, String> {
        let data = std::fs::read_to_string(path)
            .map_err(|e| format!("Failed to read config: {e}"))?;
        let v: serde_json::Value = serde_json::from_str(&data)
            .map_err(|e| format!("Failed to parse config JSON: {e}"))?;
        Ok(LeWMConfig {
            image_size: v["image_size"].as_u64().unwrap_or(224) as usize,
            patch_size: v["patch_size"].as_u64().unwrap_or(14) as usize,
            channels: v["channels"].as_u64().unwrap_or(3) as usize,
            encoder_hidden: v["encoder_hidden"].as_u64().unwrap_or(192) as usize,
            encoder_layers: v["encoder_layers"].as_u64().unwrap_or(6) as usize,
            encoder_heads: v["encoder_heads"].as_u64().unwrap_or(3) as usize,
            encoder_inter: v["encoder_inter"].as_u64().unwrap_or(768) as usize,
            predictor_hidden: v["predictor_hidden"].as_u64().unwrap_or(192) as usize,
            predictor_layers: v["predictor_layers"].as_u64().unwrap_or(6) as usize,
            predictor_heads: v["predictor_heads"].as_u64().unwrap_or(16) as usize,
            predictor_inner_dim: v["predictor_inner_dim"].as_u64().unwrap_or(1024) as usize,
            predictor_inter: v["predictor_inter"].as_u64().unwrap_or(2048) as usize,
            action_dim: v["action_dim"].as_u64().unwrap_or(10) as usize,
            latent_dim: v["latent_dim"].as_u64().unwrap_or(192) as usize,
        })
    }

    /// Whether this config has a latent bottleneck (latent_dim != predictor_hidden).
    pub fn has_projection(&self) -> bool {
        self.latent_dim != self.predictor_hidden
    }

    pub fn predictor_head_dim(&self) -> usize {
        self.predictor_inner_dim / self.predictor_heads
    }
}

/// Pre-allocated buffers for fused LEWM inference.
/// Reused across layers and rollout steps, reducing per-step heap allocations
/// from ~84 to ~8 reusable buffers.
pub struct LeWMBuffers {
    pub seq: Vec<f32>,        // [seq_len * hidden]
    pub mod_params: Vec<f32>, // [6 * hidden]
    pub normed: Vec<f32>,     // [seq_len * hidden]
    pub qkv: Vec<f32>,        // [seq_len * 3 * inner_dim]
    pub attn_out: Vec<f32>,   // [seq_len * inner_dim]
    pub proj: Vec<f32>,       // [seq_len * hidden]
    pub ffn_inter: Vec<f32>,  // [seq_len * inter]
    pub ffn_out: Vec<f32>,    // [seq_len * hidden]
    // Arena buffers: eliminate per-layer allocations in predict_next_fused
    pub q_split: Vec<f32>,    // [seq_len * inner_dim] — Q after QKV split
    pub k_split: Vec<f32>,    // [seq_len * inner_dim] — K after QKV split
    pub v_split: Vec<f32>,    // [seq_len * inner_dim] — V after QKV split
    pub mod_copy: Vec<f32>,   // [6 * hidden] — copy of mod_params for gate extraction
    pub latent_seq: Vec<f32>, // [seq_len * latent_dim] — bottleneck input sequence
    pub cond: Vec<f32>,       // [hidden] — conditioning vector
    // Fused rollout buffers (used by lewm_rollout_fused FFI path)
    pub scores_buf: Vec<f32>, // [seq_len * seq_len] — attention scores for dynamic attention
    pub packed_a: Vec<f32>,   // GEMM packing scratch A
    pub packed_b: Vec<f32>,   // GEMM packing scratch B
}

impl LeWMBuffers {
    pub fn new(config: &LeWMConfig) -> Self {
        let seq_len = 3; // predict_next always uses 3 tokens
        let h = config.predictor_hidden;
        let inner = config.predictor_inner_dim;
        let inter = config.predictor_inter;
        let latent = config.latent_dim;
        LeWMBuffers {
            seq: vec![0.0; seq_len * h],
            mod_params: vec![0.0; 6 * h],
            normed: vec![0.0; seq_len * h],
            qkv: vec![0.0; seq_len * 3 * inner],
            attn_out: vec![0.0; seq_len * inner],
            proj: vec![0.0; seq_len * h],
            ffn_inter: vec![0.0; seq_len * inter],
            ffn_out: vec![0.0; seq_len * h],
            q_split: vec![0.0; seq_len * inner],
            k_split: vec![0.0; seq_len * inner],
            v_split: vec![0.0; seq_len * inner],
            mod_copy: vec![0.0; 6 * h],
            latent_seq: vec![0.0; seq_len * latent],
            cond: vec![0.0; h],
            scores_buf: vec![0.0; seq_len * seq_len], // 3×3 = 9 for single step
            packed_a: Vec::new(),
            packed_b: Vec::new(),
        }
    }
}

/// A single DiT-style transformer layer with adaptive layer normalization (adaLN).
///
/// adaLN modulation produces 6 vectors from the conditioning signal:
/// `(scale1, shift1, gate1, scale2, shift2, gate2)` for attention and MLP sub-layers.
pub struct AdaLNTransformerLayer {
    // adaLN modulation: linear from hidden → 6 * hidden
    pub adaln_weight: AlignedBuffer,
    pub adaln_bias: AlignedBuffer,
    // Fused QKV attention
    pub to_qkv: AlignedBuffer,
    pub attn_out_weight: AlignedBuffer,
    pub attn_out_bias: AlignedBuffer,
    // QK norm
    pub attn_norm_weight: AlignedBuffer,
    pub attn_norm_bias: AlignedBuffer,
    // MLP
    pub mlp_norm_weight: AlignedBuffer,
    pub mlp_norm_bias: AlignedBuffer,
    pub mlp_up_weight: AlignedBuffer,
    pub mlp_up_bias: AlignedBuffer,
    pub mlp_down_weight: AlignedBuffer,
    pub mlp_down_bias: AlignedBuffer,
}

impl AdaLNTransformerLayer {
    /// Create a layer with zeroed weights.
    pub fn new_zeroed() -> Self {
        AdaLNTransformerLayer {
            adaln_weight: AlignedBuffer::new_zeroed(0),
            adaln_bias: AlignedBuffer::new_zeroed(0),
            to_qkv: AlignedBuffer::new_zeroed(0),
            attn_out_weight: AlignedBuffer::new_zeroed(0),
            attn_out_bias: AlignedBuffer::new_zeroed(0),
            attn_norm_weight: AlignedBuffer::new_zeroed(0),
            attn_norm_bias: AlignedBuffer::new_zeroed(0),
            mlp_norm_weight: AlignedBuffer::new_zeroed(0),
            mlp_norm_bias: AlignedBuffer::new_zeroed(0),
            mlp_up_weight: AlignedBuffer::new_zeroed(0),
            mlp_up_bias: AlignedBuffer::new_zeroed(0),
            mlp_down_weight: AlignedBuffer::new_zeroed(0),
            mlp_down_bias: AlignedBuffer::new_zeroed(0),
        }
    }

    /// Forward pass for one DiT adaLN layer.
    ///
    /// `x`: `[seq_len, hidden]` — input token sequence (flat).
    /// `conditioning`: `[hidden]` — action embedding used for adaLN modulation.
    /// `hidden`: hidden dimension of the predictor (e.g. 192).
    /// `num_heads`: number of attention heads (e.g. 16).
    /// `inner_dim`: attention inner dimension = num_heads * head_dim (e.g. 1024).
    /// `inter`: MLP intermediate size (e.g. 2048).
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

        // 1. Compute adaLN modulation: conditioning [hidden] → mod_vec [6*hidden]
        let mut mod_vec = matmul_t(conditioning, &self.adaln_weight, 1, hidden, mod_dim);
        if !self.adaln_bias.is_empty() {
            for j in 0..mod_dim {
                mod_vec[j] += self.adaln_bias[j];
            }
        }
        // Split into 6 vectors of [hidden]: scale1, shift1, gate1, scale2, shift2, gate2
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

        // 3. Fused QKV attention
        //    modulated: [seq_len, hidden] → qkv: [seq_len, 3*inner_dim]
        let qkv = matmul_t(&modulated, &self.to_qkv, seq_len, hidden, 3 * inner_dim);

        // Split into Q, K, V each [seq_len, inner_dim]
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

        // Bidirectional multi-head attention
        let attn_out = bidirectional_attention(&q, &k, &v, seq_len, num_heads, head_dim);

        // Output projection: [seq_len, inner_dim] → [seq_len, hidden]
        let mut proj = matmul_t(&attn_out, &self.attn_out_weight, seq_len, inner_dim, hidden);
        if !self.attn_out_bias.is_empty() {
            for t in 0..seq_len {
                for j in 0..hidden {
                    proj[t * hidden + j] += self.attn_out_bias[j];
                }
            }
        }

        // 4. Gated residual: x = x + gate1 * attn_out
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

        // 6. MLP: up → GELU → down
        let mut up = matmul_t(&modulated2, &self.mlp_up_weight, seq_len, hidden, inter);
        if !self.mlp_up_bias.is_empty() {
            for t in 0..seq_len {
                for j in 0..inter {
                    up[t * inter + j] += self.mlp_up_bias[j];
                }
            }
        }
        for v in up.iter_mut() {
            *v = gelu(*v);
        }
        let mut down = matmul_t(&up, &self.mlp_down_weight, seq_len, inter, hidden);
        if !self.mlp_down_bias.is_empty() {
            for t in 0..seq_len {
                for j in 0..hidden {
                    down[t * hidden + j] += self.mlp_down_bias[j];
                }
            }
        }

        // 7. Gated residual: x = x + gate2 * mlp_out
        for t in 0..seq_len {
            for j in 0..hidden {
                let idx = t * hidden + j;
                residual[idx] += gate2[j] * down[idx];
            }
        }

        residual
    }
}

/// 3-layer MLP projection head: hidden → inter → inter → hidden.
pub struct ProjectionHead {
    pub layers: Vec<(AlignedBuffer, AlignedBuffer)>,
}

impl Clone for ProjectionHead {
    fn clone(&self) -> Self {
        ProjectionHead {
            layers: self
                .layers
                .iter()
                .map(|(w, b)| (AlignedBuffer::from_slice(w), AlignedBuffer::from_slice(b)))
                .collect(),
        }
    }
}

impl ProjectionHead {
    /// Create a projection head with zeroed weights.
    pub fn new_zeroed(num_layers: usize) -> Self {
        let mut layers = Vec::with_capacity(num_layers);
        for _ in 0..num_layers {
            layers.push((AlignedBuffer::new_zeroed(0), AlignedBuffer::new_zeroed(0)));
        }
        ProjectionHead { layers }
    }

    /// Forward pass through the projection MLP.
    ///
    /// Applies GELU activation between layers (not after the last).
    pub fn forward(&self, x: &[f32]) -> Vec<f32> {
        let mut current = x.to_vec();
        for (i, (weight, bias)) in self.layers.iter().enumerate() {
            if weight.is_empty() {
                continue;
            }
            let in_dim = current.len();
            let out_dim = weight.len() / in_dim;
            let mut out = matmul_t(&current, weight, 1, in_dim, out_dim);
            if !bias.is_empty() {
                for j in 0..out_dim {
                    out[j] += bias[j];
                }
            }
            // GELU between layers, not after the last
            if i < self.layers.len() - 1 {
                for v in out.iter_mut() {
                    *v = gelu(*v);
                }
            }
            current = out;
        }
        current
    }
}

/// LeWorldModel: ViT encoder → projector → DiT predictor (adaLN) → pred_proj.
///
/// Encodes observations to latent states, then predicts future latent states
/// conditioned on actions using the DiT-style predictor with adaLN modulation.
pub struct LeWorldModel {
    pub config: LeWMConfig,
    pub encoder: ViTModel,
    pub predictor_layers: Vec<AdaLNTransformerLayer>,
    pub predictor_pos_embed: AlignedBuffer,
    pub predictor_norm_weight: AlignedBuffer,
    pub predictor_norm_bias: AlignedBuffer,
    // Action encoder
    pub action_conv_weight: AlignedBuffer,
    pub action_conv_bias: AlignedBuffer,
    pub action_mlp1_weight: AlignedBuffer,
    pub action_mlp1_bias: AlignedBuffer,
    pub action_mlp2_weight: AlignedBuffer,
    pub action_mlp2_bias: AlignedBuffer,
    // Predictor input/conditioning projections (latent_dim → predictor_hidden).
    // Empty when latent_dim == predictor_hidden (baseline, no bottleneck).
    pub input_proj_weight: AlignedBuffer,
    pub input_proj_bias: AlignedBuffer,
    pub cond_proj_weight: AlignedBuffer,
    pub cond_proj_bias: AlignedBuffer,
    // Projector (encoder → predictor space)
    pub projector: ProjectionHead,
    // Pred_proj (predictor → output space)
    pub pred_proj: ProjectionHead,
    /// Fuse mode for Zig FFI predictor layers.
    /// 0 = standard (separate loops), 1 = ESP-fused (single-pass bias+GELU/residual loops).
    pub fuse_mode: u8,
}

impl LeWorldModel {
    /// Build a LeWorldModel from config with zeroed weights.
    pub fn from_config(config: &LeWMConfig) -> Self {
        // Build encoder as standard ViT (no classifier head)
        let vit_config = ViTConfig {
            image_size: config.image_size,
            patch_size: config.patch_size,
            channels: config.channels,
            hidden_size: config.encoder_hidden,
            num_layers: config.encoder_layers,
            num_heads: config.encoder_heads,
            intermediate_size: config.encoder_inter,
            num_classes: 0,
        };
        let encoder = ViTModel::from_config(&vit_config);

        let mut predictor_layers = Vec::with_capacity(config.predictor_layers);
        for _ in 0..config.predictor_layers {
            predictor_layers.push(AdaLNTransformerLayer::new_zeroed());
        }

        LeWorldModel {
            config: config.clone(),
            encoder,
            predictor_layers,
            predictor_pos_embed: AlignedBuffer::new_zeroed(0),
            predictor_norm_weight: AlignedBuffer::new_zeroed(0),
            predictor_norm_bias: AlignedBuffer::new_zeroed(0),
            action_conv_weight: AlignedBuffer::new_zeroed(0),
            action_conv_bias: AlignedBuffer::new_zeroed(0),
            action_mlp1_weight: AlignedBuffer::new_zeroed(0),
            action_mlp1_bias: AlignedBuffer::new_zeroed(0),
            action_mlp2_weight: AlignedBuffer::new_zeroed(0),
            action_mlp2_bias: AlignedBuffer::new_zeroed(0),
            input_proj_weight: AlignedBuffer::new_zeroed(0),
            input_proj_bias: AlignedBuffer::new_zeroed(0),
            cond_proj_weight: AlignedBuffer::new_zeroed(0),
            cond_proj_bias: AlignedBuffer::new_zeroed(0),
            projector: ProjectionHead::new_zeroed(3),
            pred_proj: ProjectionHead::new_zeroed(3),
            fuse_mode: 0,
        }
    }

    /// Set the fuse mode for Zig FFI predictor layers.
    /// 0 = standard (separate loops), 1 = ESP-fused (single-pass bias+GELU/residual loops).
    pub fn set_fuse_mode(&mut self, mode: u8) {
        self.fuse_mode = mode;
    }

    /// Encode an observation image to a latent state in predictor space.
    ///
    /// `image`: flat `[H * W * C]` pixel data.
    /// Returns `[latent_dim]` latent embedding.
    pub fn encode(&self, image: &[f32], h: usize, w: usize) -> Vec<f32> {
        // ViT encoder → CLS embedding [encoder_hidden]
        let vit_out = self.encoder.forward_image(image, h, w);
        // Project to predictor space
        self.projector.forward(&vit_out.embeddings)
    }

    /// Encode an action vector to an action embedding.
    ///
    /// `action`: `[action_dim]` (e.g. `[10]`).
    /// Returns `[latent_dim]` (== `predictor_hidden` when no projection).
    fn encode_action(&self, action: &[f32]) -> Vec<f32> {
        let act_dim = self.config.action_dim;
        let hidden = self.config.latent_dim;

        // 1. 1D conv with kernel_size=1 is equivalent to a linear layer
        let mut conv_out = vec![0.0f32; act_dim];
        if !self.action_conv_weight.is_empty() {
            // action_conv_weight: [out_channels=10, kernel_size=10, 1] reshaped as [10, 10]
            // This is a 1D conv with kernel_size=1: each output channel is dot(weight[c], input)
            // But the shape [10, 10, 1] means it's a grouped/depthwise-like operation.
            // Actually kernel shape [10, 10, 1] for a 1D conv means:
            //   out_channels=10, in_channels_per_group=10, kernel_size=1
            // This is just a [10, 10] matmul (the trailing 1 is kernel_size=1).
            let weight_elems = act_dim * act_dim;
            if self.action_conv_weight.len() >= weight_elems {
                conv_out = matmul_t(action, &self.action_conv_weight, 1, act_dim, act_dim);
            }
            if !self.action_conv_bias.is_empty() {
                for j in 0..act_dim {
                    conv_out[j] += self.action_conv_bias[j];
                }
            }
        } else {
            conv_out.copy_from_slice(action);
        }

        // 2. MLP: [act_dim] → [inter=768] (GELU) → [hidden=192]
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
        if !self.action_mlp1_bias.is_empty() {
            for j in 0..inter {
                h1[j] += self.action_mlp1_bias[j];
            }
        }
        for v in h1.iter_mut() {
            *v = gelu(*v);
        }

        let mut out = if !self.action_mlp2_weight.is_empty() {
            matmul_t(&h1, &self.action_mlp2_weight, 1, inter, hidden)
        } else {
            vec![0.0f32; hidden]
        };
        if !self.action_mlp2_bias.is_empty() {
            for j in 0..hidden {
                out[j] += self.action_mlp2_bias[j];
            }
        }

        out
    }

    /// Apply input_proj: `[seq_len, latent_dim]` → `[seq_len, predictor_hidden]`.
    fn apply_input_proj(&self, seq: &[f32], seq_len: usize) -> Vec<f32> {
        let latent = self.config.latent_dim;
        let hidden = self.config.predictor_hidden;
        let mut out = vec![0.0f32; seq_len * hidden];
        for t in 0..seq_len {
            let row = matmul_t(
                &seq[t * latent..(t + 1) * latent],
                &self.input_proj_weight,
                1,
                latent,
                hidden,
            );
            out[t * hidden..(t + 1) * hidden].copy_from_slice(&row);
        }
        if !self.input_proj_bias.is_empty() {
            for t in 0..seq_len {
                for j in 0..hidden {
                    out[t * hidden + j] += self.input_proj_bias[j];
                }
            }
        }
        out
    }

    /// Apply cond_proj: `[latent_dim]` → `[predictor_hidden]`.
    fn apply_cond_proj(&self, cond: &[f32]) -> Vec<f32> {
        let latent = self.config.latent_dim;
        let hidden = self.config.predictor_hidden;
        let mut out = matmul_t(cond, &self.cond_proj_weight, 1, latent, hidden);
        if !self.cond_proj_bias.is_empty() {
            for j in 0..hidden {
                out[j] += self.cond_proj_bias[j];
            }
        }
        out
    }

    /// Predict the next latent state given current latent and action.
    ///
    /// `z_t`: `[latent_dim]` — current latent state.
    /// `action`: `[action_dim]` — action to condition on.
    /// Returns `[latent_dim]` — predicted next latent state.
    pub fn predict_next(&self, z_t: &[f32], action: &[f32]) -> Vec<f32> {
        let hidden = self.config.predictor_hidden;
        let latent = self.config.latent_dim;
        let num_heads = self.config.predictor_heads;
        let inner_dim = self.config.predictor_inner_dim;
        let inter = self.config.predictor_inter;
        // Projection is needed when latent != predictor_hidden (bottleneck architecture).
        let has_proj = !self.input_proj_weight.is_empty();

        // 1. Encode action → [latent_dim]
        let a_embed = self.encode_action(action);

        // 2. Build input sequence at latent_dim: [z_t, a_embed, zeros]
        let seq_len = 3;
        let seq_dim = if has_proj { latent } else { hidden };
        let mut seq = vec![0.0f32; seq_len * seq_dim];
        seq[..seq_dim].copy_from_slice(z_t);
        seq[seq_dim..2 * seq_dim].copy_from_slice(&a_embed);

        // 3. Add positional embeddings (at latent_dim or predictor_hidden)
        if !self.predictor_pos_embed.is_empty() {
            let pos_len = self.predictor_pos_embed.len().min(seq.len());
            for i in 0..pos_len {
                seq[i] += self.predictor_pos_embed[i];
            }
        }

        // 4. Apply projections if bottleneck architecture
        let (mut seq, conditioning) = if has_proj {
            let projected_seq = self.apply_input_proj(&seq, seq_len);
            let projected_cond = self.apply_cond_proj(&a_embed);
            (projected_seq, projected_cond)
        } else {
            (seq, a_embed)
        };

        // 5. Run through predictor layers (conditioning = projected action embedding)
        for layer in &self.predictor_layers {
            seq = layer.forward(&seq, &conditioning, seq_len, hidden, num_heads, inner_dim, inter);
        }

        // 6. Final norm
        let normed = layernorm(&seq, &self.predictor_norm_weight, 1e-6, hidden);
        let mut normed = normed;
        if !self.predictor_norm_bias.is_empty() {
            for t in 0..seq_len {
                for j in 0..hidden {
                    normed[t * hidden + j] += self.predictor_norm_bias[j];
                }
            }
        }

        // 7. Extract target position (index 2) → [hidden]
        let target = &normed[2 * hidden..3 * hidden];

        // 8. Project back through pred_proj → [latent_dim]
        self.pred_proj.forward(target)
    }

    /// Multi-step rollout: predict a sequence of future latent states.
    ///
    /// `z_start`: `[latent_dim]` — initial latent state.
    /// `actions`: slice of action vectors, each `[action_dim]`.
    /// Returns one predicted latent state per action.
    pub fn rollout(&self, z_start: &[f32], actions: &[Vec<f32>]) -> Vec<Vec<f32>> {
        let mut states = Vec::with_capacity(actions.len());
        let mut z = z_start.to_vec();
        for action in actions {
            z = self.predict_next(&z, action);
            states.push(z.clone());
        }
        states
    }

    /// Predict the next latent state using fused kernels and pre-allocated buffers.
    ///
    /// Functionally identical to `predict_next`, but eliminates ~76 heap allocations
    /// per step by reusing `LeWMBuffers` and fused kernels that write into pre-allocated
    /// output buffers.
    ///
    /// `z_t`: `[latent_dim]` -- current latent state.
    /// `action`: `[action_dim]` -- action to condition on.
    /// `bufs`: pre-allocated buffers (reused across calls).
    /// Returns `[latent_dim]` -- predicted next latent state.
    pub fn predict_next_fused(
        &self,
        z_t: &[f32],
        action: &[f32],
        bufs: &mut LeWMBuffers,
    ) -> Vec<f32> {
        let hidden = self.config.predictor_hidden;
        let latent = self.config.latent_dim;
        let num_heads = self.config.predictor_heads;
        let inner_dim = self.config.predictor_inner_dim;
        let inter = self.config.predictor_inter;
        let head_dim = inner_dim / num_heads;
        let has_proj = !self.input_proj_weight.is_empty();

        // 1. Encode action -> [latent_dim]
        let a_embed = self.encode_action(action);

        // 2. Build input sequence and apply projections
        let seq_len = 3;
        if has_proj {
            // Build sequence at latent_dim in arena, then project
            let seq_dim = latent;
            let ls = &mut bufs.latent_seq[..seq_len * seq_dim];
            ls[..seq_dim].copy_from_slice(z_t);
            ls[seq_dim..2 * seq_dim].copy_from_slice(&a_embed);
            for j in 2 * seq_dim..3 * seq_dim { ls[j] = 0.0; }
            if !self.predictor_pos_embed.is_empty() {
                let pos_len = self.predictor_pos_embed.len().min(seq_len * seq_dim);
                for i in 0..pos_len {
                    bufs.latent_seq[i] += self.predictor_pos_embed[i];
                }
            }
            let projected = self.apply_input_proj(&bufs.latent_seq[..seq_len * seq_dim], seq_len);
            bufs.seq[..seq_len * hidden].copy_from_slice(&projected);
            let cond_proj = self.apply_cond_proj(&a_embed);
            bufs.cond[..hidden].copy_from_slice(&cond_proj);
        } else {
            bufs.seq[..hidden].copy_from_slice(z_t);
            bufs.seq[hidden..2 * hidden].copy_from_slice(&a_embed);
            for j in 2 * hidden..3 * hidden {
                bufs.seq[j] = 0.0;
            }
            if !self.predictor_pos_embed.is_empty() {
                let pos_len = self.predictor_pos_embed.len().min(bufs.seq.len());
                for i in 0..pos_len {
                    bufs.seq[i] += self.predictor_pos_embed[i];
                }
            }
            bufs.cond[..hidden].copy_from_slice(&a_embed);
        };
        let conditioning = &bufs.cond[..hidden];

        // 3. Run through predictor layers — single Zig FFI call per layer
        #[cfg(feature = "zig-ffi")]
        {
            // Scratch sizing: normed needs max(seq*hidden, seq*inter), proj same
            let normed_size = (seq_len * hidden).max(seq_len * inter);
            let proj_size = normed_size;
            if bufs.normed.len() < normed_size { bufs.normed.resize(normed_size, 0.0); }
            if bufs.proj.len() < proj_size { bufs.proj.resize(proj_size, 0.0); }

            for layer in &self.predictor_layers {
                synapse_core::lewm_predict_layer_v2(
                    &mut bufs.seq,
                    conditioning,
                    seq_len, hidden, num_heads, inner_dim, inter,
                    &layer.adaln_weight, &layer.adaln_bias,
                    &layer.attn_norm_weight,
                    &layer.to_qkv,
                    &layer.attn_out_weight, &layer.attn_out_bias,
                    &layer.mlp_norm_weight,
                    &layer.mlp_up_weight, &layer.mlp_up_bias,
                    &layer.mlp_down_weight, &layer.mlp_down_bias,
                    &mut bufs.mod_params,
                    &mut bufs.normed,
                    &mut bufs.qkv,
                    &mut bufs.attn_out,
                    &mut bufs.proj,
                    self.fuse_mode,
                ).expect("lewm_predict_layer_v2 failed");
            }
        }

        // Fallback: pure Rust fused ops (slower but correct)
        #[cfg(not(feature = "zig-ffi"))]
        for layer in &self.predictor_layers {
            let mod_dim = 6 * hidden;
            let mod_result = matmul_t(conditioning, &layer.adaln_weight, 1, hidden, mod_dim);
            bufs.mod_params[..mod_dim].copy_from_slice(&mod_result);
            if !layer.adaln_bias.is_empty() {
                for j in 0..mod_dim { bufs.mod_params[j] += layer.adaln_bias[j]; }
            }
            bufs.mod_copy[..mod_dim].copy_from_slice(&bufs.mod_params[..mod_dim]);

            fused_adaln_layernorm_matmul_into(
                &bufs.seq, &layer.attn_norm_weight,
                &bufs.mod_copy[0..hidden], &bufs.mod_copy[hidden..2*hidden],
                &layer.to_qkv, seq_len, hidden, 3 * inner_dim, 1e-6, &mut bufs.qkv,
            );

            let qi = seq_len * inner_dim;
            for t in 0..seq_len {
                let qkv_off = t * 3 * inner_dim;
                let off = t * inner_dim;
                bufs.q_split[off..off+inner_dim].copy_from_slice(&bufs.qkv[qkv_off..qkv_off+inner_dim]);
                bufs.k_split[off..off+inner_dim].copy_from_slice(&bufs.qkv[qkv_off+inner_dim..qkv_off+2*inner_dim]);
                bufs.v_split[off..off+inner_dim].copy_from_slice(&bufs.qkv[qkv_off+2*inner_dim..qkv_off+3*inner_dim]);
            }
            let attn_result = bidirectional_attention(
                &bufs.q_split[..qi], &bufs.k_split[..qi], &bufs.v_split[..qi],
                seq_len, num_heads, head_dim,
            );
            bufs.attn_out[..qi].copy_from_slice(&attn_result);

            fused_attn_proj_gated_residual_into(
                &bufs.attn_out, &layer.attn_out_weight, &layer.attn_out_bias,
                &bufs.mod_copy[2*hidden..3*hidden], seq_len, inner_dim, hidden, &mut bufs.seq,
            );
            let modulated = fused_layernorm_modulate(
                &bufs.seq, &layer.mlp_norm_weight,
                &bufs.mod_copy[3*hidden..4*hidden], &bufs.mod_copy[4*hidden..5*hidden],
                seq_len, hidden, 1e-6,
            );
            bufs.normed[..seq_len*hidden].copy_from_slice(&modulated);
            fused_ffn_gated_residual_into(
                &bufs.normed, &layer.mlp_up_weight, &layer.mlp_up_bias,
                &layer.mlp_down_weight, &layer.mlp_down_bias,
                &bufs.mod_copy[5*hidden..6*hidden], seq_len, hidden, inter,
                &mut bufs.seq, &mut bufs.ffn_inter,
            );
        }

        // 5. Final norm
        let normed = layernorm(&bufs.seq, &self.predictor_norm_weight, 1e-6, hidden);
        let mut normed = normed;
        if !self.predictor_norm_bias.is_empty() {
            for t in 0..seq_len {
                for j in 0..hidden {
                    normed[t * hidden + j] += self.predictor_norm_bias[j];
                }
            }
        }

        // 6. Extract target position (index 2) -> [hidden]
        let target = &normed[2 * hidden..3 * hidden];

        // 7. Project back through pred_proj
        self.pred_proj.forward(target)
    }

    /// Multi-step rollout using fused kernels.
    ///
    /// Allocates `LeWMBuffers` once and reuses across all steps.
    pub fn rollout_fused(&self, z_start: &[f32], actions: &[Vec<f32>]) -> Vec<Vec<f32>> {
        let mut bufs = LeWMBuffers::new(&self.config);
        let mut states = Vec::with_capacity(actions.len());
        let mut z = z_start.to_vec();
        for action in actions {
            z = self.predict_next_fused(&z, action, &mut bufs);
            states.push(z.clone());
        }
        states
    }

    /// Fused multi-step predictor: runs all predictor layers once over an N×3-token
    /// sequence, where N = actions.len().
    ///
    /// Constructs the sequence as: `[z_start, a_0, zeros, z_start, a_1, zeros, ...]`.
    /// Same `z_start` for all positions — produces **parallel hypothesis futures**, not
    /// a sequential autoregressive chain. Step 0 output matches sequential rollout step 0;
    /// steps 1 and 2 differ because fused sees all 3 action embeddings simultaneously.
    ///
    /// Bidirectional attention naturally attends across all step tokens, giving the
    /// model cross-step context for free. Saves ~2× predictor layer cost vs calling
    /// `predict_next_fused` N times sequentially.
    ///
    /// `z_start`: `[latent_dim]` — initial latent state.
    /// `actions`: slice of action vectors, each `[action_dim]`.
    /// `bufs`: pre-allocated buffers. Will be resized if needed for seq_len = N*3.
    /// Returns `Vec<Vec<f32>>` — one predicted latent state per action.
    #[allow(clippy::too_many_arguments)]
    pub fn predict_rollout_fused(
        &self,
        z_start: &[f32],
        actions: &[Vec<f32>],
        bufs: &mut LeWMBuffers,
    ) -> Vec<Vec<f32>> {
        let hidden = self.config.predictor_hidden;
        let latent = self.config.latent_dim;
        let num_heads = self.config.predictor_heads;
        let inner_dim = self.config.predictor_inner_dim;
        let inter = self.config.predictor_inter;
        let num_steps = actions.len();
        let fused_seq_len = num_steps * 3; // e.g. 9 for 3 actions

        // 1. Encode all actions upfront.
        let action_embeds: Vec<Vec<f32>> = actions
            .iter()
            .map(|a| self.encode_action(a))
            .collect();

        // Projection is needed when input_proj weights are loaded.
        // For zeroed test models, latent_dim == predictor_hidden (no projection needed).
        let has_proj = !self.input_proj_weight.is_empty();

        // 2. Build fused sequence and ensure buffers are large enough.
        //    Buffer sizing: seq needs fused_seq_len*hidden; intermediate bufs need
        //    max(fused_seq_len*hidden, fused_seq_len*inter).
        let seq_size = fused_seq_len * hidden;
        let scratch_size = seq_size.max(fused_seq_len * inter);
        let mod_size = 6 * hidden;

        if bufs.seq.len() < seq_size {
            bufs.seq.resize(seq_size, 0.0);
        }
        if bufs.normed.len() < scratch_size {
            bufs.normed.resize(scratch_size, 0.0);
        }
        if bufs.proj.len() < scratch_size {
            bufs.proj.resize(scratch_size, 0.0);
        }
        if bufs.mod_params.len() < mod_size {
            bufs.mod_params.resize(mod_size, 0.0);
        }
        if bufs.mod_copy.len() < mod_size {
            bufs.mod_copy.resize(mod_size, 0.0);
        }
        // QKV: 3 components × inner_dim each → resize for full [fused_seq_len * 3 * inner_dim]
        if bufs.qkv.len() < fused_seq_len * 3 * inner_dim {
            bufs.qkv.resize(fused_seq_len * 3 * inner_dim, 0.0);
        }
        if bufs.attn_out.len() < fused_seq_len * inner_dim {
            bufs.attn_out.resize(fused_seq_len * inner_dim, 0.0);
        }
        if bufs.q_split.len() < fused_seq_len * inner_dim {
            bufs.q_split.resize(fused_seq_len * inner_dim, 0.0);
        }
        if bufs.k_split.len() < fused_seq_len * inner_dim {
            bufs.k_split.resize(fused_seq_len * inner_dim, 0.0);
        }
        if bufs.v_split.len() < fused_seq_len * inner_dim {
            bufs.v_split.resize(fused_seq_len * inner_dim, 0.0);
        }
        if bufs.ffn_inter.len() < fused_seq_len * inter {
            bufs.ffn_inter.resize(fused_seq_len * inter, 0.0);
        }
        if bufs.ffn_out.len() < fused_seq_len * hidden {
            bufs.ffn_out.resize(fused_seq_len * hidden, 0.0);
        }
        if bufs.latent_seq.len() < num_steps * 3 * latent {
            bufs.latent_seq.resize(num_steps * 3 * latent, 0.0);
        }
        // Ensure cond buffer is large enough: for slim models (latent < hidden),
        // cond must hold hidden elements for the conditioning vector.
        if bufs.cond.len() < hidden {
            bufs.cond.resize(hidden, 0.0);
        }

        let conditioning = if has_proj {
            // Build latent-space sequence: [z_start, a_i, zeros] per step
            let ls = &mut bufs.latent_seq[..num_steps * 3 * latent];
            for step in 0..num_steps {
                let off = step * 3 * latent;
                ls[off..off + latent].copy_from_slice(z_start);
                ls[off + latent..off + 2 * latent].copy_from_slice(&action_embeds[step]);
                // zeros already zeroed by resize
            }

            // Add positional embeddings (cycle through 3 pattern positions)
            if !self.predictor_pos_embed.is_empty() {
                for step in 0..num_steps {
                    for pos in 0..3 {
                        let embed_off = pos * latent;
                        let seq_off = (step * 3 + pos) * latent;
                        let count = latent.min(self.predictor_pos_embed.len() - embed_off);
                        for j in 0..count {
                            ls[seq_off + j] += self.predictor_pos_embed[embed_off + j];
                        }
                    }
                }
            }

            // Project sequence: [N*3, latent] -> [N*3, hidden]
            let projected = self.apply_input_proj(ls, num_steps * 3);
            bufs.seq[..num_steps * 3 * hidden].copy_from_slice(&projected);

            // Conditioning: use first action embed projected to hidden
            let cond_proj = self.apply_cond_proj(&action_embeds[0]);
            let cond_len = cond_proj.len().min(hidden);
            bufs.cond[..cond_len].copy_from_slice(&cond_proj[..cond_len]);
            &bufs.cond[..hidden]
        } else {
            // No bottleneck: latent == hidden
            for step in 0..num_steps {
                let off = step * 3 * hidden;
                bufs.seq[off..off + hidden].copy_from_slice(z_start);
                bufs.seq[off + hidden..off + 2 * hidden].copy_from_slice(&action_embeds[step]);
                // zeros already zeroed by resize
            }

            // Add positional embeddings
            if !self.predictor_pos_embed.is_empty() {
                for step in 0..num_steps {
                    for pos in 0..3 {
                        let embed_off = pos * hidden;
                        let seq_off = (step * 3 + pos) * hidden;
                        let count = hidden.min(self.predictor_pos_embed.len() - embed_off);
                        for j in 0..count {
                            bufs.seq[seq_off + j] += self.predictor_pos_embed[embed_off + j];
                        }
                    }
                }
            }

            // Conditioning: first action embed
            let cond_len = action_embeds[0].len().min(hidden);
            bufs.cond[..cond_len].copy_from_slice(&action_embeds[0][..cond_len]);
            &bufs.cond[..hidden]
        };

        // 3. Run all predictor layers once over the fused sequence.
        #[cfg(feature = "zig-ffi")]
        {
            let normed_size = scratch_size;
            let proj_size = scratch_size;
            if bufs.normed.len() < normed_size {
                bufs.normed.resize(normed_size, 0.0);
            }
            if bufs.proj.len() < proj_size {
                bufs.proj.resize(proj_size, 0.0);
            }

            if (self.fuse_mode as u32) & 0x01 != 0 {
                // --- Fused rollout path: single FFI call for all layers ---
                let nl = self.predictor_layers.len();

                // Build per-layer weight pointer arrays
                let mut adaln_ws: Vec<*const f32> = Vec::with_capacity(nl);
                let mut adaln_bs: Vec<*const f32> = Vec::with_capacity(nl);
                let mut attn_norm_ws: Vec<*const f32> = Vec::with_capacity(nl);
                let mut to_qkvs: Vec<*const f32> = Vec::with_capacity(nl);
                let mut attn_out_ws: Vec<*const f32> = Vec::with_capacity(nl);
                let mut attn_out_bs: Vec<*const f32> = Vec::with_capacity(nl);
                let mut mlp_norm_ws: Vec<*const f32> = Vec::with_capacity(nl);
                let mut mlp_up_ws: Vec<*const f32> = Vec::with_capacity(nl);
                let mut mlp_up_bs: Vec<*const f32> = Vec::with_capacity(nl);
                let mut mlp_down_ws: Vec<*const f32> = Vec::with_capacity(nl);
                let mut mlp_down_bs: Vec<*const f32> = Vec::with_capacity(nl);

                for layer in &self.predictor_layers {
                    adaln_ws.push(layer.adaln_weight.as_ptr());
                    adaln_bs.push(if layer.adaln_bias.is_empty() { layer.adaln_weight.as_ptr() } else { layer.adaln_bias.as_ptr() });
                    attn_norm_ws.push(layer.attn_norm_weight.as_ptr());
                    to_qkvs.push(layer.to_qkv.as_ptr());
                    attn_out_ws.push(layer.attn_out_weight.as_ptr());
                    attn_out_bs.push(if layer.attn_out_bias.is_empty() { layer.attn_out_weight.as_ptr() } else { layer.attn_out_bias.as_ptr() });
                    mlp_norm_ws.push(layer.mlp_norm_weight.as_ptr());
                    mlp_up_ws.push(layer.mlp_up_weight.as_ptr());
                    mlp_up_bs.push(if layer.mlp_up_bias.is_empty() { layer.mlp_up_weight.as_ptr() } else { layer.mlp_up_bias.as_ptr() });
                    mlp_down_ws.push(layer.mlp_down_weight.as_ptr());
                    mlp_down_bs.push(if layer.mlp_down_bias.is_empty() { layer.mlp_down_weight.as_ptr() } else { layer.mlp_down_bias.as_ptr() });
                }

                // Resize scores_buf for dynamic attention: seq_len * seq_len
                let scores_size = fused_seq_len * fused_seq_len;
                if bufs.scores_buf.len() < scores_size {
                    bufs.scores_buf.resize(scores_size, 0.0);
                }

                // Resize GEMM packing buffers (conservative upper bound)
                // Max GEMM dimensions in the rollout: M=fused_seq_len, N=max(6*hidden, 3*inner_dim, inter), K=max(hidden, inner_dim, inter)
                let max_n = (6 * hidden).max(3 * inner_dim).max(inter);
                let max_k = hidden.max(inner_dim).max(inter);
                // MR=8, NR=8, MC=64, KC=256, NC=256 (from Zig matmul tiling constants)
                let mc = 64usize.min(fused_seq_len);
                let kc = 256usize.min(max_k);
                let nc = 256usize.min(max_n);
                let mr = 8usize;
                let nr = 8usize;
                let pa_size = ((mc + mr - 1) / mr) * mr * kc;
                let pb_size = ((nc + nr - 1) / nr) * nr * kc;
                if bufs.packed_a.len() < pa_size {
                    bufs.packed_a.resize(pa_size, 0.0);
                }
                if bufs.packed_b.len() < pb_size {
                    bufs.packed_b.resize(pb_size, 0.0);
                }

                synapse_core::lewm_rollout_fused(
                    &mut bufs.seq[..seq_size],
                    conditioning,
                    num_steps, hidden, num_heads, inner_dim, inter, nl,
                    &adaln_ws, &adaln_bs, &attn_norm_ws, &to_qkvs,
                    &attn_out_ws, &attn_out_bs, &mlp_norm_ws,
                    &mlp_up_ws, &mlp_up_bs, &mlp_down_ws, &mlp_down_bs,
                    &mut bufs.mod_params,
                    &mut bufs.normed,
                    &mut bufs.qkv,
                    &mut bufs.attn_out,
                    &mut bufs.proj,
                    &mut bufs.scores_buf,
                    &mut bufs.packed_a,
                    &mut bufs.packed_b,
                    self.fuse_mode as u32,
                ).expect("lewm_rollout_fused failed");
            } else {
                // --- Per-layer path (fuse_mode bit 0 not set) ---
                for layer in &self.predictor_layers {
                    synapse_core::lewm_predict_layer_v2(
                        &mut bufs.seq[..seq_size],
                        conditioning,
                        fused_seq_len, hidden, num_heads, inner_dim, inter,
                        &layer.adaln_weight, &layer.adaln_bias,
                        &layer.attn_norm_weight,
                        &layer.to_qkv,
                        &layer.attn_out_weight, &layer.attn_out_bias,
                        &layer.mlp_norm_weight,
                        &layer.mlp_up_weight, &layer.mlp_up_bias,
                        &layer.mlp_down_weight, &layer.mlp_down_bias,
                        &mut bufs.mod_params,
                        &mut bufs.normed,
                        &mut bufs.qkv,
                        &mut bufs.attn_out,
                        &mut bufs.proj,
                        self.fuse_mode,
                    ).expect("lewm_predict_layer_v2 failed");
                }
            }
        }

        // Pure Rust path (fallback — used by tests, ESP32 host harness)
        // Uses only: matmul_t, layernorm, bidirectional_attention, gelu.
        #[cfg(not(feature = "zig-ffi"))]
        for layer in &self.predictor_layers {
            let mod_dim = 6 * hidden;

            // adaLN modulation
            let mod_result = matmul_t(conditioning, &layer.adaln_weight, 1, hidden, mod_dim);
            bufs.mod_params[..mod_dim].copy_from_slice(&mod_result);
            if !layer.adaln_bias.is_empty() {
                for j in 0..mod_dim {
                    bufs.mod_params[j] += layer.adaln_bias[j];
                }
            }
            bufs.mod_copy[..mod_dim].copy_from_slice(&bufs.mod_params[..mod_dim]);

            // Pre-attention: layernorm + modulate (fused kernel, single pass)
            let modulated = fused_layernorm_modulate(
                &bufs.seq[..seq_size],
                &layer.attn_norm_weight,
                &bufs.mod_copy[..hidden],          // scale1
                &bufs.mod_copy[hidden..2 * hidden], // shift1
                fused_seq_len,
                hidden,
                1e-6,
            );
            let head_dim = inner_dim / num_heads;
            let modulated_len = fused_seq_len * hidden;

            // QKV matmul
            let qkv = matmul_t(&modulated, &layer.to_qkv, fused_seq_len, hidden, 3 * inner_dim);
            bufs.qkv[..fused_seq_len * 3 * inner_dim].copy_from_slice(&qkv);

            // Split QKV and run bidirectional attention
            let qi = fused_seq_len * inner_dim;
            for t in 0..fused_seq_len {
                let qkv_off = t * 3 * inner_dim;
                let off = t * inner_dim;
                bufs.q_split[off..off + inner_dim]
                    .copy_from_slice(&bufs.qkv[qkv_off..qkv_off + inner_dim]);
                bufs.k_split[off..off + inner_dim]
                    .copy_from_slice(&bufs.qkv[qkv_off + inner_dim..qkv_off + 2 * inner_dim]);
                bufs.v_split[off..off + inner_dim]
                    .copy_from_slice(&bufs.qkv[qkv_off + 2 * inner_dim..qkv_off + 3 * inner_dim]);
            }
            let attn_result = bidirectional_attention(
                &bufs.q_split[..qi],
                &bufs.k_split[..qi],
                &bufs.v_split[..qi],
                fused_seq_len, num_heads, head_dim,
            );
            bufs.attn_out[..qi].copy_from_slice(&attn_result);

            // Attention output projection + gated residual
            let proj = matmul_t(&bufs.attn_out[..qi], &layer.attn_out_weight, fused_seq_len, inner_dim, hidden);
            for t in 0..fused_seq_len {
                for j in 0..hidden {
                    let idx = t * hidden + j;
                    bufs.seq[idx] += bufs.mod_copy[2 * hidden + j] * proj[idx];
                }
            }
            if !layer.attn_out_bias.is_empty() {
                for t in 0..fused_seq_len {
                    for j in 0..hidden {
                        bufs.seq[t * hidden + j] += layer.attn_out_bias[j];
                    }
                }
            }

            // Pre-FFN: layernorm + modulate (fused kernel, single pass)
            let modulated2 = fused_layernorm_modulate(
                &bufs.seq[..seq_size],
                &layer.mlp_norm_weight,
                &bufs.mod_copy[3 * hidden..4 * hidden], // scale2
                &bufs.mod_copy[4 * hidden..5 * hidden], // shift2
                fused_seq_len,
                hidden,
                1e-6,
            );

            // FFN: fused up + bias + GELU + down + gated residual
            // Uses ffn_inter as scratch [fused_seq_len * inter]; residual (bufs.seq) updated in-place.
            fused_ffn_gated_residual_into(
                &modulated2,
                &layer.mlp_up_weight,
                &layer.mlp_up_bias,
                &layer.mlp_down_weight,
                &layer.mlp_down_bias,
                &bufs.mod_copy[5 * hidden..6 * hidden], // gate2
                fused_seq_len,
                hidden,
                inter,
                &mut bufs.seq[..seq_size],
                &mut bufs.ffn_inter[..fused_seq_len * inter],
            );
        }

        // 4. Final norm
        let normed = layernorm(&bufs.seq[..seq_size], &self.predictor_norm_weight, 1e-6, hidden);
        let mut normed_out = normed;
        if !self.predictor_norm_bias.is_empty() {
            for t in 0..fused_seq_len {
                for j in 0..hidden {
                    normed_out[t * hidden + j] += self.predictor_norm_bias[j];
                }
            }
        }

        // 5. Extract targets at positions 2, 5, 8, ... (index 2 of each 3-token group)
        //    and project each through pred_proj.
        let mut outputs = Vec::with_capacity(num_steps);
        for step in 0..num_steps {
            let target_off = step * 3 * hidden + 2 * hidden;
            let target = &normed_out[target_off..target_off + hidden];
            outputs.push(self.pred_proj.forward(target));
        }
        outputs
    }

    /// Predict the next latent state using Metal GPU acceleration.
    ///
    /// Encodes all 6 predictor layers into a single Metal command buffer with
    /// zero CPU-GPU synchronization between layers. Requires pre-uploaded GPU
    /// weights via `MetalLeWMState::from_model()`.
    ///
    /// `z_t`: `[latent_dim]` -- current latent state.
    /// `action`: `[action_dim]` -- action to condition on.
    /// `state`: pre-uploaded GPU weights and scratch buffers.
    /// `backend`: Metal backend with compiled pipelines.
    /// Returns `[latent_dim]` -- predicted next latent state.
    #[cfg(feature = "metal")]
    pub fn predict_next_metal(
        &self,
        z_t: &[f32],
        action: &[f32],
        state: &crate::metal::lewm_forward::MetalLeWMState,
        backend: &crate::metal::MetalBackend,
    ) -> Vec<f32> {
        let hidden = self.config.predictor_hidden;
        let latent = self.config.latent_dim;
        let has_proj = !self.input_proj_weight.is_empty();

        // 1. Encode action on CPU (tiny, not worth GPU)
        let a_embed = self.encode_action(action);

        // 2. Build input sequence and apply projections on CPU
        let seq_len = 3;
        let (seq, conditioning) = if has_proj {
            let mut latent_seq = vec![0.0f32; seq_len * latent];
            latent_seq[..latent].copy_from_slice(z_t);
            latent_seq[latent..2 * latent].copy_from_slice(&a_embed);
            if !self.predictor_pos_embed.is_empty() {
                let pos_len = self.predictor_pos_embed.len().min(latent_seq.len());
                for i in 0..pos_len {
                    latent_seq[i] += self.predictor_pos_embed[i];
                }
            }
            let projected_seq = self.apply_input_proj(&latent_seq, seq_len);
            let projected_cond = self.apply_cond_proj(&a_embed);
            (projected_seq, projected_cond)
        } else {
            let mut seq = vec![0.0f32; seq_len * hidden];
            seq[..hidden].copy_from_slice(z_t);
            seq[hidden..2 * hidden].copy_from_slice(&a_embed);
            if !self.predictor_pos_embed.is_empty() {
                let pos_len = self.predictor_pos_embed.len().min(seq.len());
                for i in 0..pos_len {
                    seq[i] += self.predictor_pos_embed[i];
                }
            }
            (seq, a_embed)
        };

        // 3. Run predictor layers on GPU (single command buffer)
        let output = crate::metal::lewm_forward::lewm_predict_metal(
            &seq,
            &conditioning,
            state,
            backend,
        );

        // 4. Final layernorm on CPU (small, [3, hidden])
        let normed = layernorm(&output, &self.predictor_norm_weight, 1e-6, hidden);
        let mut normed = normed;
        if !self.predictor_norm_bias.is_empty() {
            for t in 0..seq_len {
                for j in 0..hidden {
                    normed[t * hidden + j] += self.predictor_norm_bias[j];
                }
            }
        }

        // 5. Extract target position (index 2) and project
        let target = &normed[2 * hidden..3 * hidden];
        self.pred_proj.forward(target)
    }

    /// Fused Metal predict: ONE kernel dispatch per layer.
    /// Uses the monolithic `adaln_layer_fused` shader.
    #[cfg(feature = "metal")]
    pub fn predict_next_metal_fused(
        &self,
        z_t: &[f32],
        action: &[f32],
        state: &crate::metal::lewm_forward::MetalLeWMState,
        backend: &crate::metal::MetalBackend,
    ) -> Vec<f32> {
        let hidden = self.config.predictor_hidden;
        let latent = self.config.latent_dim;
        let has_proj = !self.input_proj_weight.is_empty();
        let a_embed = self.encode_action(action);
        let seq_len = 3;
        let (seq, conditioning) = if has_proj {
            let mut ls = vec![0.0f32; seq_len * latent];
            ls[..latent].copy_from_slice(z_t);
            ls[latent..2 * latent].copy_from_slice(&a_embed);
            if !self.predictor_pos_embed.is_empty() {
                let pl = self.predictor_pos_embed.len().min(ls.len());
                for i in 0..pl { ls[i] += self.predictor_pos_embed[i]; }
            }
            (self.apply_input_proj(&ls, seq_len), self.apply_cond_proj(&a_embed))
        } else {
            let mut seq = vec![0.0f32; seq_len * hidden];
            seq[..hidden].copy_from_slice(z_t);
            seq[hidden..2 * hidden].copy_from_slice(&a_embed);
            if !self.predictor_pos_embed.is_empty() {
                let pl = self.predictor_pos_embed.len().min(seq.len());
                for i in 0..pl { seq[i] += self.predictor_pos_embed[i]; }
            }
            (seq, a_embed)
        };
        let output = crate::metal::lewm_forward::lewm_predict_metal_fused(
            &seq, &conditioning, state, backend,
        );
        let normed = layernorm(&output, &self.predictor_norm_weight, 1e-6, hidden);
        let mut normed = normed;
        if !self.predictor_norm_bias.is_empty() {
            for t in 0..seq_len {
                for j in 0..hidden { normed[t * hidden + j] += self.predictor_norm_bias[j]; }
            }
        }
        let target = &normed[2 * hidden..3 * hidden];
        self.pred_proj.forward(target)
    }

    /// V3: Fused Metal with vectorized float4 dot products.
    #[cfg(feature = "metal")]
    pub fn predict_next_metal_v3(
        &self,
        z_t: &[f32],
        action: &[f32],
        state: &crate::metal::lewm_forward::MetalLeWMState,
        backend: &crate::metal::MetalBackend,
    ) -> Vec<f32> {
        let hidden = self.config.predictor_hidden;
        let latent = self.config.latent_dim;
        let has_proj = !self.input_proj_weight.is_empty();
        let a_embed = self.encode_action(action);
        let seq_len = 3;
        let (seq, conditioning) = if has_proj {
            let mut ls = vec![0.0f32; seq_len * latent];
            ls[..latent].copy_from_slice(z_t);
            ls[latent..2 * latent].copy_from_slice(&a_embed);
            if !self.predictor_pos_embed.is_empty() {
                let pl = self.predictor_pos_embed.len().min(ls.len());
                for i in 0..pl { ls[i] += self.predictor_pos_embed[i]; }
            }
            (self.apply_input_proj(&ls, seq_len), self.apply_cond_proj(&a_embed))
        } else {
            let mut seq = vec![0.0f32; seq_len * hidden];
            seq[..hidden].copy_from_slice(z_t);
            seq[hidden..2 * hidden].copy_from_slice(&a_embed);
            if !self.predictor_pos_embed.is_empty() {
                let pl = self.predictor_pos_embed.len().min(seq.len());
                for i in 0..pl { seq[i] += self.predictor_pos_embed[i]; }
            }
            (seq, a_embed)
        };
        let output = crate::metal::lewm_forward::lewm_predict_metal_fused_v3(
            &seq, &conditioning, state, backend,
        );
        let normed = layernorm(&output, &self.predictor_norm_weight, 1e-6, hidden);
        let mut normed = normed;
        if !self.predictor_norm_bias.is_empty() {
            for t in 0..seq_len {
                for j in 0..hidden { normed[t * hidden + j] += self.predictor_norm_bias[j]; }
            }
        }
        let target = &normed[2 * hidden..3 * hidden];
        self.pred_proj.forward(target)
    }

    /// Multi-step rollout using Metal GPU acceleration.
    ///
    /// Uploads weights once, then reuses GPU state across all steps.
    #[cfg(feature = "metal")]
    pub fn rollout_metal(
        &self,
        z_start: &[f32],
        actions: &[Vec<f32>],
        state: &crate::metal::lewm_forward::MetalLeWMState,
        backend: &crate::metal::MetalBackend,
    ) -> Vec<Vec<f32>> {
        let mut states = Vec::with_capacity(actions.len());
        let mut z = z_start.to_vec();
        for action in actions {
            z = self.predict_next_metal(&z, action, state, backend);
            states.push(z.clone());
        }
        states
    }

    /// Load weights from a safetensors HashMap.
    ///
    /// Handles the LeWM checkpoint naming directly (custom prefix matching,
    /// not through the generic WeightMapper pattern).
    pub fn load_weights(
        &mut self,
        weights: HashMap<String, RawTensor>,
    ) -> Result<LoadStats, WeightError> {
        let mut loaded = 0usize;
        let mut skipped = Vec::new();

        for (name, tensor) in &weights {
            let ok = self.set_weight(name, tensor);
            if ok {
                loaded += 1;
            } else {
                skipped.push(name.clone());
            }
        }

        Ok(LoadStats { loaded, skipped })
    }

    /// Map a single checkpoint key to the appropriate internal buffer.
    /// Returns `true` if the key was recognized and loaded.
    fn set_weight(&mut self, key: &str, tensor: &RawTensor) -> bool {
        // ── Encoder weights ──────────────────────────────────────────
        // Checkpoint: encoder.embeddings.* → ViT embeddings
        if let Some(rest) = key.strip_prefix("encoder.") {
            return self.set_encoder_weight(rest, tensor);
        }

        // ── Predictor weights ────────────────────────────────────────
        if key == "predictor.pos_embedding" {
            self.predictor_pos_embed = tensor.data.clone();
            return true;
        }
        if key == "predictor.transformer.norm.weight" {
            self.predictor_norm_weight = tensor.data.clone();
            return true;
        }
        if key == "predictor.transformer.norm.bias" {
            self.predictor_norm_bias = tensor.data.clone();
            return true;
        }
        if let Some(rest) = key.strip_prefix("predictor.transformer.layers.") {
            return self.set_predictor_layer_weight(rest, tensor);
        }

        // ── Input/conditioning projections (slim bottleneck models) ──
        if key == "predictor.transformer.input_proj.weight" {
            self.input_proj_weight = tensor.data.clone();
            return true;
        }
        if key == "predictor.transformer.input_proj.bias" {
            self.input_proj_bias = tensor.data.clone();
            return true;
        }
        if key == "predictor.transformer.cond_proj.weight" {
            self.cond_proj_weight = tensor.data.clone();
            return true;
        }
        if key == "predictor.transformer.cond_proj.bias" {
            self.cond_proj_bias = tensor.data.clone();
            return true;
        }

        // ── Action encoder ───────────────────────────────────────────
        if key == "action_encoder.patch_embed.weight" {
            self.action_conv_weight = tensor.data.clone();
            return true;
        }
        if key == "action_encoder.patch_embed.bias" {
            self.action_conv_bias = tensor.data.clone();
            return true;
        }
        if key == "action_encoder.embed.0.weight" {
            self.action_mlp1_weight = tensor.data.clone();
            return true;
        }
        if key == "action_encoder.embed.0.bias" {
            self.action_mlp1_bias = tensor.data.clone();
            return true;
        }
        if key == "action_encoder.embed.2.weight" {
            self.action_mlp2_weight = tensor.data.clone();
            return true;
        }
        if key == "action_encoder.embed.2.bias" {
            self.action_mlp2_bias = tensor.data.clone();
            return true;
        }

        // ── Projector ────────────────────────────────────────────────
        if let Some(rest) = key.strip_prefix("projector.") {
            return Self::set_projection_weight_on(&mut self.projector, rest, tensor);
        }

        // ── Pred_proj ────────────────────────────────────────────────
        if let Some(rest) = key.strip_prefix("pred_proj.") {
            return Self::set_projection_weight_on(&mut self.pred_proj, rest, tensor);
        }

        false
    }

    fn set_projection_weight_on(head: &mut ProjectionHead, rest: &str, tensor: &RawTensor) -> bool {
        // Parse "N.weight", "N.bias", or "net.N.weight", "net.N.bias"
        // Handle both "0.weight" and "net.0.weight" checkpoint naming
        let rest = rest.strip_prefix("net.").unwrap_or(rest);
        let parts: Vec<&str> = rest.splitn(2, '.').collect();
        if parts.len() != 2 {
            return false;
        }
        let layer_idx: usize = match parts[0].parse() {
            Ok(i) => i,
            Err(_) => return false,
        };
        let field = parts[1];

        // Checkpoint sequential: net.0=Linear, net.1=BatchNorm, net.2=GELU, net.3=Linear
        // We only store the Linear layers. Skip BN (index 1) and GELU (index 2).
        // Map: 0→layers[0], 3→layers[1], (4→layers[2] if exists)
        let our_idx = match layer_idx {
            0 => 0,
            3 => 1,
            4 => 2, // third linear if present
            _ => return false, // skip BN (1), GELU (2), etc.
        };
        if our_idx >= head.layers.len() {
            return false;
        }

        match field {
            "weight" => {
                head.layers[our_idx].0 = tensor.data.clone();
                true
            }
            "bias" => {
                head.layers[our_idx].1 = tensor.data.clone();
                true
            }
            _ => false,
        }
    }

    /// Map encoder sub-key (after stripping "encoder.") to ViT weights.
    fn set_encoder_weight(&mut self, rest: &str, tensor: &RawTensor) -> bool {
        // Checkpoint has "encoder.embeddings.patch_embeddings.projection.weight"
        // which maps to ViT patch_proj.
        match rest {
            "embeddings.patch_embeddings.projection.weight" => {
                self.encoder.patch_proj = tensor.data.clone();
                true
            }
            "embeddings.patch_embeddings.projection.bias" => {
                self.encoder.patch_proj_bias = tensor.data.clone();
                true
            }
            "embeddings.cls_token" => {
                self.encoder.cls_token = tensor.data.clone();
                true
            }
            "embeddings.position_embeddings" => {
                self.encoder.pos_embed = tensor.data.clone();
                true
            }
            "layernorm.weight" => {
                self.encoder.final_norm_weight = tensor.data.clone();
                true
            }
            "layernorm.bias" => {
                self.encoder.final_norm_bias = tensor.data.clone();
                true
            }
            // Hybrid encoder extras
            "meta_token" => {
                let h = self.config.encoder_hidden;
                let total = tensor.data.len();
                self.encoder.num_meta_tokens = total / h;
                self.encoder.meta_token = tensor.data.clone();
                true
            }
            "proj.0.weight" => {
                self.encoder.enc_proj_weight = tensor.data.clone();
                true
            }
            "proj.0.bias" => {
                self.encoder.enc_proj_bias = tensor.data.clone();
                true
            }
            _ if rest.starts_with("proj.") => {
                // Skip BN params (proj.1.weight, proj.1.bias, etc.)
                true
            }
            _ if rest.starts_with("encoder.layer.") => {
                // "encoder.layer.{i}.attention.attention.query.weight" etc.
                // (note: double "encoder" prefix — outer is stripped, inner remains)
                self.set_encoder_layer_weight(rest, tensor)
            }
            _ => false,
        }
    }

    /// Map encoder layer sub-key (e.g. "encoder.layer.0.attention.attention.query.weight").
    fn set_encoder_layer_weight(&mut self, rest: &str, tensor: &RawTensor) -> bool {
        let rest = match rest.strip_prefix("encoder.layer.") {
            Some(r) => r,
            None => return false,
        };

        // Parse layer index
        let dot = match rest.find('.') {
            Some(p) => p,
            None => return false,
        };
        let idx: usize = match rest[..dot].parse() {
            Ok(i) => i,
            Err(_) => return false,
        };
        let field = &rest[dot + 1..];

        let layer = match self.encoder.layers.get_mut(idx) {
            Some(l) => l,
            None => return false,
        };

        // Standard ViT HuggingFace naming
        match field {
            "attention.attention.query.weight" => {
                layer.w_q = tensor.data.clone();
                true
            }
            "attention.attention.query.bias" => {
                layer.q_bias = tensor.data.clone();
                true
            }
            "attention.attention.key.weight" => {
                layer.w_k = tensor.data.clone();
                true
            }
            "attention.attention.key.bias" => {
                layer.k_bias = tensor.data.clone();
                true
            }
            "attention.attention.value.weight" => {
                layer.w_v = tensor.data.clone();
                true
            }
            "attention.attention.value.bias" => {
                layer.v_bias = tensor.data.clone();
                true
            }
            "attention.output.dense.weight" => {
                layer.w_o = tensor.data.clone();
                true
            }
            "attention.output.dense.bias" => {
                layer.o_bias = tensor.data.clone();
                true
            }
            "intermediate.dense.weight" => {
                layer.ffn_up = tensor.data.clone();
                true
            }
            "intermediate.dense.bias" => {
                layer.ffn_up_bias = tensor.data.clone();
                true
            }
            "output.dense.weight" => {
                layer.ffn_down = tensor.data.clone();
                true
            }
            "output.dense.bias" => {
                layer.ffn_down_bias = tensor.data.clone();
                true
            }
            // LayerNorm — ViT uses layernorm_before/norm1 for attn, layernorm_after/norm2 for FFN
            "layernorm_before.weight" | "norm1.weight" => {
                layer.attn_norm_weight = tensor.data.clone();
                true
            }
            "layernorm_before.bias" | "norm1.bias" => {
                layer.attn_norm_bias = tensor.data.clone();
                true
            }
            "layernorm_after.weight" | "norm2.weight" => {
                layer.ffn_norm_weight = tensor.data.clone();
                true
            }
            "layernorm_after.bias" | "norm2.bias" => {
                layer.ffn_norm_bias = tensor.data.clone();
                true
            }
            _ => false,
        }
    }

    /// Map predictor layer sub-key (e.g. "0.adaLN_modulation.1.weight").
    fn set_predictor_layer_weight(&mut self, rest: &str, tensor: &RawTensor) -> bool {
        // Parse layer index from "N.field"
        let dot = match rest.find('.') {
            Some(p) => p,
            None => return false,
        };
        let idx: usize = match rest[..dot].parse() {
            Ok(i) => i,
            Err(_) => return false,
        };
        let field = &rest[dot + 1..];

        let layer = match self.predictor_layers.get_mut(idx) {
            Some(l) => l,
            None => return false,
        };

        match field {
            "adaLN_modulation.1.weight" => {
                layer.adaln_weight = tensor.data.clone();
                true
            }
            "adaLN_modulation.1.bias" => {
                layer.adaln_bias = tensor.data.clone();
                true
            }
            "attn.to_qkv.weight" => {
                layer.to_qkv = tensor.data.clone();
                true
            }
            "attn.to_out.0.weight" => {
                layer.attn_out_weight = tensor.data.clone();
                true
            }
            "attn.to_out.0.bias" => {
                layer.attn_out_bias = tensor.data.clone();
                true
            }
            "attn.norm.weight" => {
                layer.attn_norm_weight = tensor.data.clone();
                true
            }
            "attn.norm.bias" => {
                layer.attn_norm_bias = tensor.data.clone();
                true
            }
            "mlp.net.0.weight" => {
                layer.mlp_norm_weight = tensor.data.clone();
                true
            }
            "mlp.net.0.bias" => {
                layer.mlp_norm_bias = tensor.data.clone();
                true
            }
            "mlp.net.1.weight" => {
                layer.mlp_up_weight = tensor.data.clone();
                true
            }
            "mlp.net.1.bias" => {
                layer.mlp_up_bias = tensor.data.clone();
                true
            }
            "mlp.net.4.weight" => {
                layer.mlp_down_weight = tensor.data.clone();
                true
            }
            "mlp.net.4.bias" => {
                layer.mlp_down_bias = tensor.data.clone();
                true
            }
            _ => false,
        }
    }
}


#[cfg(test)]
mod tests {
    use super::*;

    fn gen_weights(len: usize, seed: u32) -> Vec<f32> {
        (0..len)
            .map(|i| {
                let x = ((i as u32).wrapping_mul(2654435761).wrapping_add(seed)) as f32;
                (x / u32::MAX as f32) * 0.36 - 0.18
            })
            .collect()
    }

    fn small_config() -> LeWMConfig {
        LeWMConfig {
            image_size: 8,
            patch_size: 4,
            channels: 3,
            encoder_hidden: 16,
            encoder_layers: 2,
            encoder_heads: 2,
            encoder_inter: 32,
            predictor_hidden: 16,
            predictor_layers: 2,
            predictor_heads: 2,
            predictor_inner_dim: 16,
            predictor_inter: 32,
            action_dim: 4,
            latent_dim: 16,
        }
    }

    fn build_test_lewm(cfg: &LeWMConfig) -> LeWorldModel {
        let h = cfg.encoder_hidden;
        let pred_h = cfg.predictor_hidden;
        let pred_inner = cfg.predictor_inner_dim;
        let pred_inter = cfg.predictor_inter;
        let act_dim = cfg.action_dim;
        let patch_dim = cfg.patch_size * cfg.patch_size * cfg.channels;
        let num_patches = (cfg.image_size / cfg.patch_size).pow(2);
        let enc_seq_len = num_patches + 1;
        let enc_inter = cfg.encoder_inter;

        let mut model = LeWorldModel::from_config(cfg);

        // Encoder weights
        model.encoder.patch_proj = AlignedBuffer::from_slice(&gen_weights(h * patch_dim, 1));
        model.encoder.cls_token = AlignedBuffer::from_slice(&gen_weights(h, 2));
        model.encoder.pos_embed = AlignedBuffer::from_slice(&gen_weights(enc_seq_len * h, 3));
        model.encoder.final_norm_weight = AlignedBuffer::from_slice(&vec![1.0f32; h]);

        for (i, layer) in model.encoder.layers.iter_mut().enumerate() {
            let s = (i as u32 + 1) * 100;
            layer.attn_norm_weight = AlignedBuffer::from_slice(&vec![1.0f32; h]);
            layer.w_q = AlignedBuffer::from_slice(&gen_weights(h * h, s + 1));
            layer.w_k = AlignedBuffer::from_slice(&gen_weights(h * h, s + 2));
            layer.w_v = AlignedBuffer::from_slice(&gen_weights(h * h, s + 3));
            layer.w_o = AlignedBuffer::from_slice(&gen_weights(h * h, s + 4));
            layer.ffn_norm_weight = AlignedBuffer::from_slice(&vec![1.0f32; h]);
            layer.ffn_up = AlignedBuffer::from_slice(&gen_weights(enc_inter * h, s + 5));
            layer.ffn_down = AlignedBuffer::from_slice(&gen_weights(h * enc_inter, s + 6));
        }

        // Projector: [h] → [enc_inter] → [enc_inter] → [pred_h]
        model.projector.layers[0].0 = AlignedBuffer::from_slice(&gen_weights(enc_inter * h, 400));
        model.projector.layers[0].1 = AlignedBuffer::from_slice(&gen_weights(enc_inter, 401));
        model.projector.layers[1].0 =
            AlignedBuffer::from_slice(&gen_weights(enc_inter * enc_inter, 402));
        model.projector.layers[1].1 = AlignedBuffer::from_slice(&gen_weights(enc_inter, 403));
        model.projector.layers[2].0 =
            AlignedBuffer::from_slice(&gen_weights(pred_h * enc_inter, 404));
        model.projector.layers[2].1 = AlignedBuffer::from_slice(&gen_weights(pred_h, 405));

        // Pred_proj: same structure
        model.pred_proj.layers[0].0 =
            AlignedBuffer::from_slice(&gen_weights(enc_inter * pred_h, 500));
        model.pred_proj.layers[0].1 = AlignedBuffer::from_slice(&gen_weights(enc_inter, 501));
        model.pred_proj.layers[1].0 =
            AlignedBuffer::from_slice(&gen_weights(enc_inter * enc_inter, 502));
        model.pred_proj.layers[1].1 = AlignedBuffer::from_slice(&gen_weights(enc_inter, 503));
        model.pred_proj.layers[2].0 =
            AlignedBuffer::from_slice(&gen_weights(pred_h * enc_inter, 504));
        model.pred_proj.layers[2].1 = AlignedBuffer::from_slice(&gen_weights(pred_h, 505));

        // Action encoder
        model.action_conv_weight = AlignedBuffer::from_slice(&gen_weights(act_dim * act_dim, 600));
        model.action_conv_bias = AlignedBuffer::from_slice(&gen_weights(act_dim, 601));
        model.action_mlp1_weight =
            AlignedBuffer::from_slice(&gen_weights(enc_inter * act_dim, 602));
        model.action_mlp1_bias = AlignedBuffer::from_slice(&gen_weights(enc_inter, 603));
        model.action_mlp2_weight = AlignedBuffer::from_slice(&gen_weights(pred_h * enc_inter, 604));
        model.action_mlp2_bias = AlignedBuffer::from_slice(&gen_weights(pred_h, 605));

        // Predictor pos embed: [3, pred_h]
        model.predictor_pos_embed = AlignedBuffer::from_slice(&gen_weights(3 * pred_h, 700));

        // Predictor norm
        model.predictor_norm_weight = AlignedBuffer::from_slice(&vec![1.0f32; pred_h]);
        model.predictor_norm_bias = AlignedBuffer::from_slice(&vec![0.0f32; pred_h]);

        // Predictor layers (adaLN DiT)
        for (i, layer) in model.predictor_layers.iter_mut().enumerate() {
            let s = (i as u32 + 1) * 1000;
            // adaLN: [6*pred_h, pred_h]
            layer.adaln_weight =
                AlignedBuffer::from_slice(&gen_weights(6 * pred_h * pred_h, s + 1));
            layer.adaln_bias = AlignedBuffer::from_slice(&gen_weights(6 * pred_h, s + 2));
            // Fused QKV: [3*inner_dim, pred_h]
            layer.to_qkv = AlignedBuffer::from_slice(&gen_weights(3 * pred_inner * pred_h, s + 3));
            // Attn out: [pred_h, inner_dim]
            layer.attn_out_weight =
                AlignedBuffer::from_slice(&gen_weights(pred_h * pred_inner, s + 4));
            layer.attn_out_bias = AlignedBuffer::from_slice(&gen_weights(pred_h, s + 5));
            // Attn norm (used for pre-attention LN in adaLN)
            layer.attn_norm_weight = AlignedBuffer::from_slice(&vec![1.0f32; pred_h]);
            layer.attn_norm_bias = AlignedBuffer::from_slice(&vec![0.0f32; pred_h]);
            // MLP norm
            layer.mlp_norm_weight = AlignedBuffer::from_slice(&vec![1.0f32; pred_h]);
            layer.mlp_norm_bias = AlignedBuffer::from_slice(&vec![0.0f32; pred_h]);
            // MLP up: [pred_inter, pred_h]
            layer.mlp_up_weight =
                AlignedBuffer::from_slice(&gen_weights(pred_inter * pred_h, s + 10));
            layer.mlp_up_bias = AlignedBuffer::from_slice(&gen_weights(pred_inter, s + 11));
            // MLP down: [pred_h, pred_inter]
            layer.mlp_down_weight =
                AlignedBuffer::from_slice(&gen_weights(pred_h * pred_inter, s + 12));
            layer.mlp_down_bias = AlignedBuffer::from_slice(&gen_weights(pred_h, s + 13));
        }

        model
    }

    #[test]
    fn test_lewm_encode_produces_finite_embeddings() {
        let cfg = small_config();
        let model = build_test_lewm(&cfg);

        let image: Vec<f32> = (0..cfg.image_size * cfg.image_size * cfg.channels)
            .map(|i| (i as f32) / 255.0)
            .collect();

        let z = model.encode(&image, cfg.image_size, cfg.image_size);

        assert_eq!(
            z.len(),
            cfg.latent_dim,
            "Encoded latent should have latent_dim elements"
        );
        assert!(
            z.iter().all(|v| v.is_finite()),
            "LeWM encode produced non-finite values"
        );
    }

    #[test]
    fn test_lewm_predict_next_produces_finite() {
        let cfg = small_config();
        let model = build_test_lewm(&cfg);

        let z = gen_weights(cfg.latent_dim, 42);
        let action = gen_weights(cfg.action_dim, 43);

        let z_next = model.predict_next(&z, &action);

        assert_eq!(
            z_next.len(),
            cfg.latent_dim,
            "Predicted latent should have latent_dim elements"
        );
        assert!(
            z_next.iter().all(|v| v.is_finite()),
            "LeWM predict_next produced non-finite values"
        );
    }

    #[test]
    fn test_lewm_rollout_correct_length() {
        let cfg = small_config();
        let model = build_test_lewm(&cfg);

        let z = gen_weights(cfg.latent_dim, 42);
        let actions: Vec<Vec<f32>> = (0..10)
            .map(|i| gen_weights(cfg.action_dim, 100 + i))
            .collect();

        let trajectory = model.rollout(&z, &actions);

        assert_eq!(
            trajectory.len(),
            10,
            "Rollout with 10 actions should produce 10 states"
        );
        for (i, state) in trajectory.iter().enumerate() {
            assert_eq!(
                state.len(),
                cfg.latent_dim,
                "State {i} should have latent_dim elements"
            );
            assert!(
                state.iter().all(|v| v.is_finite()),
                "State {i} contains non-finite values"
            );
        }
    }

    #[test]
    fn test_lewm_action_encoder_produces_finite() {
        let cfg = small_config();
        let model = build_test_lewm(&cfg);

        let action = gen_weights(cfg.action_dim, 42);
        let a_embed = model.encode_action(&action);

        assert_eq!(
            a_embed.len(),
            cfg.predictor_hidden,
            "Action embedding should have predictor_hidden elements"
        );
        assert!(
            a_embed.iter().all(|v| v.is_finite()),
            "Action encoder produced non-finite values"
        );
    }

    #[test]
    fn test_lewm_projection_head_roundtrip_shape() {
        let cfg = small_config();
        let model = build_test_lewm(&cfg);

        let x = gen_weights(cfg.encoder_hidden, 42);
        let projected = model.projector.forward(&x);
        assert_eq!(
            projected.len(),
            cfg.predictor_hidden,
            "Projector output should have predictor_hidden elements"
        );
        assert!(
            projected.iter().all(|v| v.is_finite()),
            "Projector produced non-finite values"
        );
    }

    #[test]
    fn test_lewm_adaln_layer_produces_finite() {
        let cfg = small_config();
        let model = build_test_lewm(&cfg);

        let seq_len = 3;
        let hidden = cfg.predictor_hidden;
        let x = gen_weights(seq_len * hidden, 42);
        let cond = gen_weights(hidden, 43);

        let out = model.predictor_layers[0].forward(
            &x,
            &cond,
            seq_len,
            hidden,
            cfg.predictor_heads,
            cfg.predictor_inner_dim,
            cfg.predictor_inter,
        );

        assert_eq!(out.len(), seq_len * hidden);
        assert!(
            out.iter().all(|v| v.is_finite()),
            "AdaLN layer produced non-finite values"
        );
    }

    #[test]
    fn test_lewm_load_weights_recognizes_keys() {
        let cfg = small_config();
        let mut model = LeWorldModel::from_config(&cfg);
        let h = cfg.encoder_hidden;
        let pred_h = cfg.predictor_hidden;
        let act_dim = cfg.action_dim;
        let enc_inter = cfg.encoder_inter;
        let pred_inner = cfg.predictor_inner_dim;
        let _pred_inter = cfg.predictor_inter;
        let patch_dim = cfg.patch_size * cfg.patch_size * cfg.channels;
        let enc_seq_len = (cfg.image_size / cfg.patch_size).pow(2) + 1;

        let rt = |len: usize, seed: u32| -> RawTensor {
            RawTensor {
                data: AlignedBuffer::from_slice(&gen_weights(len, seed)),
                shape: vec![len],
            }
        };

        let mut weights: HashMap<String, RawTensor> = HashMap::new();

        // Encoder
        weights.insert(
            "encoder.embeddings.patch_embeddings.projection.weight".into(),
            rt(h * patch_dim, 1),
        );
        weights.insert("encoder.embeddings.cls_token".into(), rt(h, 2));
        weights.insert(
            "encoder.embeddings.position_embeddings".into(),
            rt(enc_seq_len * h, 3),
        );
        weights.insert("encoder.layernorm.weight".into(), rt(h, 4));
        weights.insert("encoder.layernorm.bias".into(), rt(h, 5));
        weights.insert(
            "encoder.encoder.layer.0.attention.attention.query.weight".into(),
            rt(h * h, 10),
        );

        // Predictor
        weights.insert("predictor.pos_embedding".into(), rt(3 * pred_h, 20));
        weights.insert("predictor.transformer.norm.weight".into(), rt(pred_h, 21));
        weights.insert(
            "predictor.transformer.layers.0.adaLN_modulation.1.weight".into(),
            rt(6 * pred_h * pred_h, 30),
        );
        weights.insert(
            "predictor.transformer.layers.0.attn.to_qkv.weight".into(),
            rt(3 * pred_inner * pred_h, 31),
        );

        // Action encoder
        weights.insert(
            "action_encoder.patch_embed.weight".into(),
            rt(act_dim * act_dim, 40),
        );
        weights.insert(
            "action_encoder.embed.0.weight".into(),
            rt(enc_inter * act_dim, 41),
        );

        // Projector
        weights.insert("projector.0.weight".into(), rt(enc_inter * h, 50));
        weights.insert("projector.0.bias".into(), rt(enc_inter, 51));

        // Pred_proj
        weights.insert("pred_proj.0.weight".into(), rt(enc_inter * pred_h, 60));

        let stats = model.load_weights(weights).expect("load_weights failed");

        assert!(stats.loaded > 0, "Should have loaded some weights");
        assert!(
            !model.encoder.patch_proj.is_empty(),
            "Encoder patch_proj should be loaded"
        );
        assert!(
            !model.predictor_pos_embed.is_empty(),
            "Predictor pos_embed should be loaded"
        );
        assert!(
            !model.predictor_layers[0].adaln_weight.is_empty(),
            "Predictor layer 0 adaln_weight should be loaded"
        );
        assert!(
            !model.action_conv_weight.is_empty(),
            "Action conv weight should be loaded"
        );
        assert!(
            !model.projector.layers[0].0.is_empty(),
            "Projector layer 0 weight should be loaded"
        );
        assert!(
            !model.pred_proj.layers[0].0.is_empty(),
            "Pred_proj layer 0 weight should be loaded"
        );
    }

    #[test]
    fn test_lewm_different_actions_produce_different_predictions() {
        let cfg = small_config();
        let model = build_test_lewm(&cfg);

        // Use non-trivial initial latent so the target token picks up signal
        let z = gen_weights(cfg.latent_dim, 42);
        let mut action1 = vec![0.0f32; cfg.action_dim];
        let mut action2 = vec![0.0f32; cfg.action_dim];
        action1[0] = 1.0;
        action2[0] = -1.0;

        let z1 = model.predict_next(&z, &action1);
        let z2 = model.predict_next(&z, &action2);

        // Different actions should produce different predictions
        let diff: f32 = z1.iter().zip(z2.iter()).map(|(a, b)| (a - b).abs()).sum();
        assert!(
            diff > 1e-6,
            "Different actions should produce different predictions, got diff={diff}"
        );
    }

    #[test]
    fn test_lewm_from_config_dimensions() {
        let cfg = LeWMConfig::pusht();
        let model = LeWorldModel::from_config(&cfg);

        assert_eq!(model.encoder.config.hidden_size, 192);
        assert_eq!(model.encoder.config.num_layers, 6);
        assert_eq!(model.encoder.config.num_heads, 3);
        assert_eq!(model.predictor_layers.len(), 6);
        assert_eq!(model.projector.layers.len(), 3);
        assert_eq!(model.pred_proj.layers.len(), 3);
    }

    #[test]
    fn fused_predict_matches_original() {
        let cfg = small_config();
        let model = build_test_lewm(&cfg);
        let mut bufs = LeWMBuffers::new(&cfg);

        let z = gen_weights(cfg.latent_dim, 42);
        let action = gen_weights(cfg.action_dim, 43);

        let original = model.predict_next(&z, &action);
        let fused = model.predict_next_fused(&z, &action, &mut bufs);

        assert_eq!(original.len(), fused.len());
        for (i, (a, b)) in original.iter().zip(&fused).enumerate() {
            assert!(
                (a - b).abs() < 1e-4,
                "mismatch at index {}: {} vs {} (diff={})",
                i,
                a,
                b,
                (a - b).abs()
            );
        }
    }

    #[test]
    fn fused_rollout_matches_original() {
        let cfg = small_config();
        let model = build_test_lewm(&cfg);

        let z = gen_weights(cfg.latent_dim, 42);
        let actions: Vec<Vec<f32>> = (0..5).map(|i| gen_weights(cfg.action_dim, 100 + i)).collect();

        let original = model.rollout(&z, &actions);
        let fused = model.rollout_fused(&z, &actions);

        assert_eq!(original.len(), fused.len());
        for (step, (orig_step, fused_step)) in original.iter().zip(&fused).enumerate() {
            assert_eq!(orig_step.len(), fused_step.len());
            for (i, (a, b)) in orig_step.iter().zip(fused_step).enumerate() {
                assert!(
                    (a - b).abs() < 1e-4,
                    "step {} index {}: {} vs {} (diff={})",
                    step,
                    i,
                    a,
                    b,
                    (a - b).abs()
                );
            }
        }
    }

    #[test]
    fn lewm_buffers_correct_sizes() {
        let cfg = LeWMConfig::pusht();
        let bufs = LeWMBuffers::new(&cfg);
        assert_eq!(bufs.seq.len(), 3 * 192);
        assert_eq!(bufs.mod_params.len(), 6 * 192);
        assert_eq!(bufs.qkv.len(), 3 * 3 * 1024);
        assert_eq!(bufs.attn_out.len(), 3 * 1024);
        assert_eq!(bufs.ffn_inter.len(), 3 * 2048);
        assert_eq!(bufs.proj.len(), 3 * 192);
        assert_eq!(bufs.ffn_out.len(), 3 * 192);
    }

    #[test]
    fn fused_predict_buffers_reusable() {
        let cfg = small_config();
        let model = build_test_lewm(&cfg);
        let mut bufs = LeWMBuffers::new(&cfg);

        let z = gen_weights(cfg.latent_dim, 42);
        let mut action1 = vec![0.0f32; cfg.action_dim];
        let mut action2 = vec![0.0f32; cfg.action_dim];
        action1[0] = 1.0;
        action2[0] = -1.0;

        // Use buffers for two different predictions
        let r1 = model.predict_next_fused(&z, &action1, &mut bufs);
        let r2 = model.predict_next_fused(&z, &action2, &mut bufs);

        // Both should produce finite values
        assert!(r1.iter().all(|v| v.is_finite()), "First call produced non-finite");
        assert!(r2.iter().all(|v| v.is_finite()), "Second call produced non-finite");

        // Different actions should give different results
        let diff: f32 = r1.iter().zip(r2.iter()).map(|(a, b)| (a - b).abs()).sum();
        assert!(
            diff > 1e-6,
            "Different actions should produce different predictions with reused buffers"
        );
    }

    /// Verify predict_rollout_fused returns valid outputs with 1 step.
    #[test]
    fn lewm_predict_rollout_fused_one_step_produces_finite() {
        let cfg = small_config();
        let model = build_test_lewm(&cfg);
        let mut bufs = LeWMBuffers::new(&cfg);

        let z_start = gen_weights(cfg.latent_dim, 42);
        let actions = vec![gen_weights(cfg.action_dim, 100)];

        // Single-step fused should work the same as predict_next
        let outputs = model.predict_rollout_fused(&z_start, &actions, &mut bufs);

        assert_eq!(outputs.len(), 1, "Should produce 1 output for 1 action");
        assert_eq!(outputs[0].len(), cfg.latent_dim, "Output should have latent_dim elements");
        assert!(outputs[0].iter().all(|v| v.is_finite()), "Output contains non-finite values");
    }

    /// Verify predict_rollout_fused returns valid outputs.
    #[test]
    fn lewm_predict_rollout_fused_produces_finite() {
        let cfg = small_config();
        let model = build_test_lewm(&cfg);
        let mut bufs = LeWMBuffers::new(&cfg);

        let z_start = gen_weights(cfg.latent_dim, 42);
        let actions: Vec<Vec<f32>> = (0..3)
            .map(|i| gen_weights(cfg.action_dim, 100 + i as u32))
            .collect();

        let outputs = model.predict_rollout_fused(&z_start, &actions, &mut bufs);

        // Must return 3 outputs
        assert_eq!(outputs.len(), 3, "Should produce 3 outputs for 3 actions");
        for (i, out) in outputs.iter().enumerate() {
            assert_eq!(
                out.len(),
                cfg.latent_dim,
                "Output {} should have latent_dim elements",
                i
            );
            assert!(
                out.iter().all(|v| v.is_finite()),
                "Output {} contains non-finite values",
                i
            );
        }
    }

    /// Verify predict_rollout_fused step-0 matches sequential predict_next step-0.
    /// Steps 1 and 2 differ by design (parallel futures vs autoregressive).
    #[test]
    fn lewm_rollout_fused_step0_matches_sequential() {
        let cfg = small_config();
        let model = build_test_lewm(&cfg);

        let z = gen_weights(cfg.latent_dim, 42);
        let a0 = gen_weights(cfg.action_dim, 100);
        let a1 = gen_weights(cfg.action_dim, 101);
        let a2 = gen_weights(cfg.action_dim, 102);
        let actions = vec![a0.clone(), a1.clone(), a2.clone()];

        // Sequential: step 0 via predict_next
        let seq0 = model.predict_next(&z, &a0);

        // Fused: first step
        let mut bufs = LeWMBuffers::new(&cfg);
        let fused = model.predict_rollout_fused(&z, &actions, &mut bufs);

        // Step 0 must match exactly (same z_start, same a0)
        assert_eq!(fused[0].len(), seq0.len());
        let cos_sim = cosine_sim(&fused[0], &seq0);
        assert!(
            cos_sim >= 0.999,
            "Fused step-0 cosine sim vs sequential step-0: {} (want >= 0.999)",
            cos_sim
        );
        // All 3 outputs must be finite
        for (i, out) in fused.iter().enumerate() {
            assert!(
                out.iter().all(|v| v.is_finite()),
                "Fused output {} is not finite",
                i
            );
        }
    }

    fn cosine_sim(a: &[f32], b: &[f32]) -> f32 {
        let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
        if na > 0.0 && nb > 0.0 {
            dot / (na * nb)
        } else {
            0.0
        }
    }

    /// Test: rollout_fused matches sequential rollout for pushT config (192d).
    /// Uses seeded weights (not zeroed) so matmul asserts pass.
    /// Fused[0] vs sequential[0] cosine sim should be ≥ 0.999.
    /// Fused[1] and fused[2] differ from sequential by design (parallel futures vs autoregressive).
    #[test]
    fn lewm_rollout_fused_vs_sequential_pushT() {
        // Build pushT model with seeded weights (replicate build_test_lewm pattern)
        let cfg = LeWMConfig::pusht();
        let model = build_test_lewm_pushT(&cfg);
        let mut bufs = LeWMBuffers::new(&cfg);

        let z = gen_weights(cfg.latent_dim, 42);
        let actions: Vec<Vec<f32>> = (0..3).map(|i| gen_weights(cfg.action_dim, 100 + i)).collect();

        let seq_out = model.rollout(&z, &actions);
        let fused_out = model.predict_rollout_fused(&z, &actions, &mut bufs);

        assert_eq!(seq_out.len(), fused_out.len());

        // Step 0: must match (same z_start + a0, same computation path)
        let cos0 = cosine_sim(&seq_out[0], &fused_out[0]);
        assert!(
            cos0 >= 0.999,
            "Step 0 cosine sim {} should be ≥ 0.999 (same z_start + a0)",
            cos0
        );

        // Steps 1-2: differ by design (see docstring)
        let cos1 = cosine_sim(&seq_out[1], &fused_out[1]);
        let cos2 = cosine_sim(&seq_out[2], &fused_out[2]);
        assert!(
            seq_out[1].iter().all(|v| v.is_finite()),
            "sequential[1] must be finite"
        );
        assert!(
            fused_out[1].iter().all(|v| v.is_finite()),
            "fused[1] must be finite"
        );
        assert!(
            fused_out[2].iter().all(|v| v.is_finite()),
            "fused[2] must be finite"
        );
        // Document that cos1/cos2 may differ (intentional architectural difference)
        println!("Note: cos1={:.4}, cos2={:.4} (expected lower than cos0 — parallel vs autoregressive)", cos1, cos2);
    }

    /// Build a seeded pushT LeWorldModel (mirrors build_test_lewm for 192d pushT config).
    fn build_test_lewm_pushT(cfg: &LeWMConfig) -> LeWorldModel {
        let h = cfg.encoder_hidden; // 192
        let pred_h = cfg.predictor_hidden; // 192
        let pred_inner = cfg.predictor_inner_dim; // 192
        let pred_inter = cfg.predictor_inter; // 768
        let act_dim = cfg.action_dim; // 10
        let patch_dim = cfg.patch_size * cfg.patch_size * cfg.channels; // 588
        let num_patches = (cfg.image_size / cfg.patch_size).pow(2); // 256
        let enc_seq_len = num_patches + 1; // 257
        let enc_inter = cfg.encoder_inter; // 768

        let mut model = LeWorldModel::from_config(cfg);

        // Encoder weights
        model.encoder.patch_proj = AlignedBuffer::from_slice(&gen_weights(h * patch_dim, 1));
        model.encoder.cls_token = AlignedBuffer::from_slice(&gen_weights(h, 2));
        model.encoder.pos_embed = AlignedBuffer::from_slice(&gen_weights(enc_seq_len * h, 3));
        model.encoder.final_norm_weight = AlignedBuffer::from_slice(&vec![1.0f32; h]);

        for (i, layer) in model.encoder.layers.iter_mut().enumerate() {
            let s = (i as u32 + 1) * 100;
            layer.attn_norm_weight = AlignedBuffer::from_slice(&vec![1.0f32; h]);
            layer.w_q = AlignedBuffer::from_slice(&gen_weights(h * h, s + 1));
            layer.w_k = AlignedBuffer::from_slice(&gen_weights(h * h, s + 2));
            layer.w_v = AlignedBuffer::from_slice(&gen_weights(h * h, s + 3));
            layer.w_o = AlignedBuffer::from_slice(&gen_weights(h * h, s + 4));
            layer.ffn_norm_weight = AlignedBuffer::from_slice(&vec![1.0f32; h]);
            layer.ffn_up = AlignedBuffer::from_slice(&gen_weights(enc_inter * h, s + 5));
            layer.ffn_down = AlignedBuffer::from_slice(&gen_weights(h * enc_inter, s + 6));
        }

        // Projector: [h] → [enc_inter] → [enc_inter] → [pred_h]
        model.projector.layers[0].0 = AlignedBuffer::from_slice(&gen_weights(enc_inter * h, 400));
        model.projector.layers[0].1 = AlignedBuffer::from_slice(&gen_weights(enc_inter, 401));
        model.projector.layers[1].0 = AlignedBuffer::from_slice(&gen_weights(enc_inter * enc_inter, 402));
        model.projector.layers[1].1 = AlignedBuffer::from_slice(&gen_weights(enc_inter, 403));
        model.projector.layers[2].0 = AlignedBuffer::from_slice(&gen_weights(pred_h * enc_inter, 404));
        model.projector.layers[2].1 = AlignedBuffer::from_slice(&gen_weights(pred_h, 405));

        // Pred_proj: same structure
        model.pred_proj.layers[0].0 = AlignedBuffer::from_slice(&gen_weights(enc_inter * pred_h, 500));
        model.pred_proj.layers[0].1 = AlignedBuffer::from_slice(&gen_weights(enc_inter, 501));
        model.pred_proj.layers[1].0 = AlignedBuffer::from_slice(&gen_weights(enc_inter * enc_inter, 502));
        model.pred_proj.layers[1].1 = AlignedBuffer::from_slice(&gen_weights(enc_inter, 503));
        model.pred_proj.layers[2].0 = AlignedBuffer::from_slice(&gen_weights(enc_inter * pred_h, 504));
        model.pred_proj.layers[2].1 = AlignedBuffer::from_slice(&gen_weights(pred_h, 505));

        // Action encoder
        model.action_conv_weight = AlignedBuffer::from_slice(&gen_weights(act_dim * act_dim, 600));
        model.action_conv_bias = AlignedBuffer::from_slice(&gen_weights(act_dim, 601));
        model.action_mlp1_weight = AlignedBuffer::from_slice(&gen_weights(enc_inter * act_dim, 602));
        model.action_mlp1_bias = AlignedBuffer::from_slice(&gen_weights(enc_inter, 603));
        model.action_mlp2_weight = AlignedBuffer::from_slice(&gen_weights(pred_h * enc_inter, 604));
        model.action_mlp2_bias = AlignedBuffer::from_slice(&gen_weights(pred_h, 605));

        // Predictor pos embed: [3, pred_h]
        model.predictor_pos_embed = AlignedBuffer::from_slice(&gen_weights(3 * pred_h, 700));

        // Predictor norm
        model.predictor_norm_weight = AlignedBuffer::from_slice(&vec![1.0f32; pred_h]);
        model.predictor_norm_bias = AlignedBuffer::from_slice(&vec![0.0f32; pred_h]);

        // Predictor layers (adaLN DiT)
        for (i, layer) in model.predictor_layers.iter_mut().enumerate() {
            let s = (i as u32 + 1) * 1000;
            layer.adaln_weight = AlignedBuffer::from_slice(&gen_weights(6 * pred_h * pred_h, s + 1));
            layer.adaln_bias = AlignedBuffer::from_slice(&gen_weights(6 * pred_h, s + 2));
            layer.to_qkv = AlignedBuffer::from_slice(&gen_weights(3 * pred_inner * pred_h, s + 3));
            layer.attn_out_weight = AlignedBuffer::from_slice(&gen_weights(pred_h * pred_inner, s + 4));
            layer.attn_out_bias = AlignedBuffer::from_slice(&gen_weights(pred_h, s + 5));
            layer.attn_norm_weight = AlignedBuffer::from_slice(&vec![1.0f32; pred_h]);
            layer.attn_norm_bias = AlignedBuffer::from_slice(&vec![0.0f32; pred_h]);
            layer.mlp_norm_weight = AlignedBuffer::from_slice(&vec![1.0f32; pred_h]);
            layer.mlp_norm_bias = AlignedBuffer::from_slice(&vec![0.0f32; pred_h]);
            layer.mlp_up_weight = AlignedBuffer::from_slice(&gen_weights(pred_inter * pred_h, s + 10));
            layer.mlp_up_bias = AlignedBuffer::from_slice(&gen_weights(pred_inter, s + 11));
            layer.mlp_down_weight = AlignedBuffer::from_slice(&gen_weights(pred_h * pred_inter, s + 12));
            layer.mlp_down_bias = AlignedBuffer::from_slice(&gen_weights(pred_h, s + 13));
        }

        model
    }
}
