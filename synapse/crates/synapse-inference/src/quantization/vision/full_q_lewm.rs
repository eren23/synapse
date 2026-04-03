//! Fully quantized LEWM: INT8 ViT encoder + Q4 predictor.
//!
//! This is the most aggressive practical compression for LEWM:
//! - ViT encoder: INT8 projections (~4x compression on ~2.8M params)
//! - Predictor: Q4 projections (~6.4x compression on ~10.8M params)
//! - Total: ~9MB (from ~52MB f32)
//!
//! Target deployment: ESP32-P4 (32MB PSRAM) and WASM browser.

use super::q4_lewm::{
    lq40_config_from_json, lq40_read_f32, lq40_read_projection_head, lq40_read_q4_adaln_layer,
    lq40_read_u32,
};
use crate::models::vision::lewm::{AdaLNTransformerLayer, LeWMConfig, LeWorldModel, ProjectionHead};
use crate::models::vision::vit::ViTConfig;
use crate::ops::activation::gelu;
use crate::ops::attention::bidirectional_attention;
use crate::ops::matmul::matmul_t;
use crate::ops::norm::layernorm;
use crate::ops::patch_embed::patch_embed;
use crate::ops::vector::{add_vecs, add_vecs_inplace};
use crate::quantization::Q4Linear;
use crate::quantization::QuantizedLinear;
use crate::quantization::QuantizedQ4AdaLNLayer;

/// INT8-quantized ViT encoder layer.
pub struct QuantizedEncoderLayer {
    pub hidden_size: usize,
    pub num_heads: usize,
    pub head_dim: usize,
    // INT8 projections
    pub w_q: QuantizedLinear,
    pub w_k: QuantizedLinear,
    pub w_v: QuantizedLinear,
    pub w_o: QuantizedLinear,
    pub ffn_up: QuantizedLinear,
    pub ffn_down: QuantizedLinear,
    // f32 biases + norms
    pub q_bias: Vec<f32>,
    pub k_bias: Vec<f32>,
    pub v_bias: Vec<f32>,
    pub o_bias: Vec<f32>,
    pub ffn_up_bias: Vec<f32>,
    pub ffn_down_bias: Vec<f32>,
    pub attn_norm_weight: Vec<f32>,
    pub attn_norm_bias: Vec<f32>,
    pub ffn_norm_weight: Vec<f32>,
    pub ffn_norm_bias: Vec<f32>,
}

impl QuantizedEncoderLayer {
    fn add_bias(x: &mut [f32], bias: &[f32], m: usize, n: usize) {
        if bias.is_empty() {
            return;
        }
        for row in 0..m {
            for col in 0..n.min(bias.len()) {
                x[row * n + col] += bias[col];
            }
        }
    }

    pub fn forward(&self, x: &[f32], seq_len: usize) -> Vec<f32> {
        let h = self.hidden_size;
        let num_heads = self.num_heads;
        let head_dim = self.head_dim;

        // 1. Attention: norm → Q/K/V via INT8 → bidirectional attention → O via INT8
        let mut normed = layernorm(x, &self.attn_norm_weight, 1e-6, h);
        Self::add_bias(&mut normed, &self.attn_norm_bias, seq_len, h);

        let mut q = self.w_q.forward(&normed, seq_len);
        Self::add_bias(&mut q, &self.q_bias, seq_len, num_heads * head_dim);
        let mut k = self.w_k.forward(&normed, seq_len);
        Self::add_bias(&mut k, &self.k_bias, seq_len, num_heads * head_dim);
        let mut v = self.w_v.forward(&normed, seq_len);
        Self::add_bias(&mut v, &self.v_bias, seq_len, num_heads * head_dim);

        let attn_out = bidirectional_attention(&q, &k, &v, seq_len, num_heads, head_dim);

        let mut proj = self.w_o.forward(&attn_out, seq_len);
        Self::add_bias(&mut proj, &self.o_bias, seq_len, h);
        let mut residual = add_vecs(x, &proj);

        // 2. FFN: norm → up via INT8 → GELU → down via INT8
        let mut normed2 = layernorm(&residual, &self.ffn_norm_weight, 1e-6, h);
        Self::add_bias(&mut normed2, &self.ffn_norm_bias, seq_len, h);
        let inter = self.ffn_up.out_features;

        let mut up = self.ffn_up.forward(&normed2, seq_len);
        Self::add_bias(&mut up, &self.ffn_up_bias, seq_len, inter);
        for val in up.iter_mut() {
            *val = gelu(*val);
        }

        let mut down = self.ffn_down.forward(&up, seq_len);
        Self::add_bias(&mut down, &self.ffn_down_bias, seq_len, h);
        add_vecs_inplace(&mut residual, &down);

        residual
    }

    pub fn memory_bytes(&self) -> usize {
        self.w_q.memory_bytes()
            + self.w_k.memory_bytes()
            + self.w_v.memory_bytes()
            + self.w_o.memory_bytes()
            + self.ffn_up.memory_bytes()
            + self.ffn_down.memory_bytes()
            + (self.q_bias.len()
                + self.k_bias.len()
                + self.v_bias.len()
                + self.o_bias.len()
                + self.ffn_up_bias.len()
                + self.ffn_down_bias.len()
                + self.attn_norm_weight.len()
                + self.attn_norm_bias.len()
                + self.ffn_norm_weight.len()
                + self.ffn_norm_bias.len())
                * 4
    }
}

/// Fully quantized LEWM: INT8 encoder + Q4 predictor.
pub struct FullyQuantizedLeWM {
    pub config: LeWMConfig,
    // INT8 ViT encoder
    pub encoder_layers: Vec<QuantizedEncoderLayer>,
    pub patch_proj: Vec<f32>,
    pub patch_proj_bias: Vec<f32>,
    pub cls_token: Vec<f32>,
    pub pos_embed: Vec<f32>,
    pub final_norm_weight: Vec<f32>,
    pub final_norm_bias: Vec<f32>,
    pub vit_config: ViTConfig,
    // Q4 predictor (reuse existing)
    pub predictor_layers: Vec<QuantizedQ4AdaLNLayer>,
    pub predictor_pos_embed: Vec<f32>,
    pub predictor_norm_weight: Vec<f32>,
    pub predictor_norm_bias: Vec<f32>,
    // Action encoder (f32, small)
    pub action_conv_weight: Vec<f32>,
    pub action_conv_bias: Vec<f32>,
    pub action_mlp1_weight: Vec<f32>,
    pub action_mlp1_bias: Vec<f32>,
    pub action_mlp2_weight: Vec<f32>,
    pub action_mlp2_bias: Vec<f32>,
    // Projectors (f32, small)
    pub projector: ProjectionHead,
    pub pred_proj: ProjectionHead,
    // Input/conditioning projections (latent_dim → predictor_hidden bottleneck)
    pub input_proj_weight: Vec<f32>,
    pub input_proj_bias: Vec<f32>,
    pub cond_proj_weight: Vec<f32>,
    pub cond_proj_bias: Vec<f32>,
    // Hybrid encoder extras (empty for non-hybrid models)
    pub meta_token: Vec<f32>,
    pub num_meta_tokens: usize,
    pub enc_proj_weight: Vec<f32>,
    pub enc_proj_bias: Vec<f32>,
}

impl FullyQuantizedLeWM {
    /// Load from LQ40 binary format (as produced by `export_lewm_q4 --mode full`).
    pub fn from_lq40_bytes(data: &[u8]) -> Result<Self, String> {
        if data.len() < 8 {
            return Err("LQ40 data too short".into());
        }
        if &data[0..4] != b"LQ40" {
            return Err("Not LQ40 format".into());
        }

        let config_len = lq40_read_u32(data, 4) as usize;
        if data.len() < 8 + config_len {
            return Err("LQ40 config truncated".into());
        }
        let config_json: serde_json::Value = serde_json::from_slice(&data[8..8 + config_len])
            .map_err(|e| format!("LQ40 config parse error: {}", e))?;
        match config_json.get("mode").and_then(|v| v.as_str()) {
            Some("full") => {}
            Some(other) => {
                return Err(format!(
                    "LQ40 mode '{other}' is not supported by FullyQuantizedLeWM"
                ))
            }
            None => return Err("LQ40 config missing mode".into()),
        }

        let config = lq40_config_from_json(&config_json)?;
        let mut off = 8 + config_len;
        let vit_config = ViTConfig {
            image_size: config.image_size,
            patch_size: config.patch_size,
            hidden_size: config.encoder_hidden,
            num_layers: config.encoder_layers,
            num_heads: config.encoder_heads,
            intermediate_size: config.encoder_inter,
            channels: config.channels,
            num_classes: 0,
        };

        let patch_proj = lq40_read_f32(data, &mut off);
        let patch_proj_bias = lq40_read_f32(data, &mut off);
        let cls_token = lq40_read_f32(data, &mut off);
        let pos_embed = lq40_read_f32(data, &mut off);

        let mut encoder_layers = Vec::with_capacity(config.encoder_layers);
        for _ in 0..config.encoder_layers {
            encoder_layers.push(QuantizedEncoderLayer {
                hidden_size: vit_config.hidden_size,
                num_heads: vit_config.num_heads,
                head_dim: vit_config.head_dim(),
                attn_norm_weight: lq40_read_f32(data, &mut off),
                attn_norm_bias: lq40_read_f32(data, &mut off),
                w_q: lq40_read_quantized_linear(data, &mut off)?,
                q_bias: lq40_read_f32(data, &mut off),
                w_k: lq40_read_quantized_linear(data, &mut off)?,
                k_bias: lq40_read_f32(data, &mut off),
                w_v: lq40_read_quantized_linear(data, &mut off)?,
                v_bias: lq40_read_f32(data, &mut off),
                w_o: lq40_read_quantized_linear(data, &mut off)?,
                o_bias: lq40_read_f32(data, &mut off),
                ffn_norm_weight: lq40_read_f32(data, &mut off),
                ffn_norm_bias: lq40_read_f32(data, &mut off),
                ffn_up: lq40_read_quantized_linear(data, &mut off)?,
                ffn_up_bias: lq40_read_f32(data, &mut off),
                ffn_down: lq40_read_quantized_linear(data, &mut off)?,
                ffn_down_bias: lq40_read_f32(data, &mut off),
            });
        }

        let final_norm_weight = lq40_read_f32(data, &mut off);
        let final_norm_bias = lq40_read_f32(data, &mut off);
        let predictor_pos_embed = lq40_read_f32(data, &mut off);

        let mut predictor_layers = Vec::with_capacity(config.predictor_layers);
        for _ in 0..config.predictor_layers {
            predictor_layers.push(lq40_read_q4_adaln_layer(data, &mut off)?);
        }

        let predictor_norm_weight = lq40_read_f32(data, &mut off);
        let predictor_norm_bias = lq40_read_f32(data, &mut off);
        let action_conv_weight = lq40_read_f32(data, &mut off);
        let action_conv_bias = lq40_read_f32(data, &mut off);
        let action_mlp1_weight = lq40_read_f32(data, &mut off);
        let action_mlp1_bias = lq40_read_f32(data, &mut off);
        let action_mlp2_weight = lq40_read_f32(data, &mut off);
        let action_mlp2_bias = lq40_read_f32(data, &mut off);
        let projector = lq40_read_projection_head(data, &mut off);
        let pred_proj = lq40_read_projection_head(data, &mut off);
        let input_proj_weight = if off < data.len() {
            lq40_read_f32(data, &mut off)
        } else {
            vec![]
        };
        let input_proj_bias = if off < data.len() {
            lq40_read_f32(data, &mut off)
        } else {
            vec![]
        };
        let cond_proj_weight = if off < data.len() {
            lq40_read_f32(data, &mut off)
        } else {
            vec![]
        };
        let cond_proj_bias = if off < data.len() {
            lq40_read_f32(data, &mut off)
        } else {
            vec![]
        };

        let mut result = Ok(FullyQuantizedLeWM {
            config,
            encoder_layers,
            patch_proj,
            patch_proj_bias,
            cls_token,
            pos_embed,
            final_norm_weight,
            final_norm_bias,
            vit_config,
            predictor_layers,
            predictor_pos_embed,
            predictor_norm_weight,
            predictor_norm_bias,
            action_conv_weight,
            action_conv_bias,
            action_mlp1_weight,
            action_mlp1_bias,
            action_mlp2_weight,
            action_mlp2_bias,
            projector,
            pred_proj,
            input_proj_weight,
            input_proj_bias,
            cond_proj_weight,
            cond_proj_bias,
            // Hybrid extras loaded after input/cond proj (optional, may be empty)
            meta_token: vec![],
            num_meta_tokens: 0,
            enc_proj_weight: vec![],
            enc_proj_bias: vec![],
        });

        // Load hybrid extras if present
        if let Ok(ref mut m) = result {
            if off < data.len() {
                m.meta_token = lq40_read_f32(data, &mut off);
            }
            if off < data.len() {
                m.enc_proj_weight = lq40_read_f32(data, &mut off);
            }
            if off < data.len() {
                m.enc_proj_bias = lq40_read_f32(data, &mut off);
            }
            let h = m.vit_config.hidden_size;
            if h > 0 && !m.meta_token.is_empty() {
                m.num_meta_tokens = m.meta_token.len() / h;
            }
        }

        result
    }

    /// Encode an image using the INT8 ViT encoder.
    pub fn encode(&self, image: &[f32], h: usize, w: usize) -> Vec<f32> {
        let cfg = &self.vit_config;
        let hidden = cfg.hidden_size;
        let seq_len = cfg.seq_len();

        // Patch embedding (f32, small computation)
        let patches = patch_embed(
            image,
            h,
            w,
            cfg.channels,
            cfg.patch_size,
            &self.patch_proj,
            hidden,
        );
        let num_patches = cfg.num_patches();

        // Prepend CLS + add pos embed
        let mut x = vec![0.0f32; seq_len * hidden];
        x[..hidden].copy_from_slice(&self.cls_token);
        x[hidden..hidden + num_patches * hidden].copy_from_slice(&patches);
        for i in 0..seq_len * hidden {
            x[i] += self.pos_embed[i];
        }

        // Run through INT8 encoder layers
        for layer in &self.encoder_layers {
            x = layer.forward(&x, seq_len);
        }

        // Final norm on CLS token
        let cls = &x[..hidden];
        let normed = layernorm(cls, &self.final_norm_weight, 1e-6, hidden);

        // Apply encoder output projection if present (hybrid architecture)
        let enc_out = if !self.enc_proj_weight.is_empty() {
            let out_dim = if !self.enc_proj_bias.is_empty() {
                self.enc_proj_bias.len()
            } else {
                hidden
            };
            let mut proj = vec![0.0f32; out_dim];
            for j in 0..out_dim {
                let mut sum = 0.0f32;
                for k in 0..hidden {
                    sum += normed[k] * self.enc_proj_weight[j * hidden + k];
                }
                if j < self.enc_proj_bias.len() {
                    sum += self.enc_proj_bias[j];
                }
                proj[j] = sum;
            }
            proj
        } else {
            normed
        };

        // Project to predictor space
        self.projector.forward(&enc_out)
    }

    /// Predict next latent using Q4 predictor.
    pub fn predict_next(&self, z_t: &[f32], action: &[f32]) -> Vec<f32> {
        let hidden = self.config.predictor_hidden;
        let latent = self.config.latent_dim;
        let num_heads = self.config.predictor_heads;
        let inner_dim = self.config.predictor_inner_dim;
        let inter = self.config.predictor_inter;
        let has_proj = !self.input_proj_weight.is_empty();

        let a_embed = self.encode_action(action);

        // Build input sequence: [z_t, a_embed, target_token]
        let seq_len = 3;
        let seq_dim = if has_proj { latent } else { hidden };
        let mut seq = vec![0.0f32; seq_len * seq_dim];
        seq[..seq_dim].copy_from_slice(z_t);
        seq[seq_dim..2 * seq_dim].copy_from_slice(&a_embed);

        // Add positional embeddings
        if !self.predictor_pos_embed.is_empty() {
            let pos_len = self.predictor_pos_embed.len().min(seq.len());
            for i in 0..pos_len {
                seq[i] += self.predictor_pos_embed[i];
            }
        }

        // Apply projections if bottleneck architecture
        let (mut x, conditioning) = if has_proj {
            let projected_seq = super::apply_input_proj(
                &self.input_proj_weight,
                &self.input_proj_bias,
                &seq,
                seq_len,
                latent,
                hidden,
            );
            let projected_cond = super::apply_cond_proj(
                &self.cond_proj_weight,
                &self.cond_proj_bias,
                &a_embed,
                latent,
                hidden,
            );
            (projected_seq, projected_cond)
        } else {
            (seq, a_embed)
        };

        // Q4 predictor layers
        for layer in &self.predictor_layers {
            x = layer.forward(
                &x,
                &conditioning,
                seq_len,
                hidden,
                num_heads,
                inner_dim,
                inter,
            );
        }

        let normed = layernorm(&x, &self.predictor_norm_weight, 1e-6, hidden);
        let target = &normed[2 * hidden..3 * hidden];
        self.pred_proj.forward(target)
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

    fn encode_action(&self, action: &[f32]) -> Vec<f32> {
        let act_dim = self.config.action_dim;
        let hidden = self.config.latent_dim;

        let conv_out = if !self.action_conv_weight.is_empty() {
            let out = matmul_t(action, &self.action_conv_weight, 1, act_dim, act_dim);
            let mut out = out;
            for j in 0..act_dim.min(self.action_conv_bias.len()) {
                out[j] += self.action_conv_bias[j];
            }
            out
        } else {
            action.to_vec()
        };

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
        for v in h1.iter_mut() {
            *v = gelu(*v);
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

    /// Total model size in bytes.
    pub fn model_size_bytes(&self) -> usize {
        let enc: usize = self.encoder_layers.iter().map(|l| l.memory_bytes()).sum();
        let enc_misc = (self.patch_proj.len()
            + self.patch_proj_bias.len()
            + self.cls_token.len()
            + self.pos_embed.len()
            + self.final_norm_weight.len()
            + self.final_norm_bias.len())
            * 4;

        let pred: usize = self
            .predictor_layers
            .iter()
            .map(|l| {
                l.adaln_linear.memory_bytes()
                    + l.to_qkv.memory_bytes()
                    + l.attn_out.memory_bytes()
                    + l.mlp_up.memory_bytes()
                    + l.mlp_down.memory_bytes()
                    + (l.adaln_bias.len()
                        + l.attn_out_bias.len()
                        + l.attn_norm_weight.len()
                        + l.attn_norm_bias.len()
                        + l.mlp_norm_weight.len()
                        + l.mlp_norm_bias.len()
                        + l.mlp_up_bias.len()
                        + l.mlp_down_bias.len())
                        * 4
            })
            .sum();
        let pred_misc = (self.predictor_pos_embed.len()
            + self.predictor_norm_weight.len()
            + self.predictor_norm_bias.len())
            * 4;

        let action = (self.action_conv_weight.len()
            + self.action_conv_bias.len()
            + self.action_mlp1_weight.len()
            + self.action_mlp1_bias.len()
            + self.action_mlp2_weight.len()
            + self.action_mlp2_bias.len())
            * 4;

        enc + enc_misc + pred + pred_misc + action
    }
}

fn lq40_read_quantized_linear(data: &[u8], off: &mut usize) -> Result<QuantizedLinear, String> {
    let out_features = lq40_read_u32(data, *off) as usize;
    *off += 4;
    let in_features = lq40_read_u32(data, *off) as usize;
    *off += 4;

    let weights_len = lq40_read_u32(data, *off) as usize;
    *off += 4;
    if *off + weights_len > data.len() {
        return Err("LQ40 INT8 weights truncated".into());
    }
    let weights_int8 = data[*off..*off + weights_len]
        .iter()
        .map(|&b| b as i8)
        .collect();
    *off += weights_len;

    let scales_len = lq40_read_u32(data, *off) as usize;
    *off += 4;
    let mut scales = Vec::with_capacity(scales_len);
    for _ in 0..scales_len {
        let base = *off;
        if base + 4 > data.len() {
            return Err("LQ40 INT8 scales truncated".into());
        }
        scales.push(f32::from_le_bytes([
            data[base],
            data[base + 1],
            data[base + 2],
            data[base + 3],
        ]));
        *off += 4;
    }

    Ok(QuantizedLinear {
        weights_int8,
        scales,
        out_features,
        in_features,
    })
}

impl FullyQuantizedLeWM {
    /// Dequantize the predictor to f32, returning a `LeWorldModel`.
    ///
    /// This enables fused f32 rollout on Full (INT8+Q4) models by converting the Q4
    /// predictor weights to f32 upfront. The INT8 encoder is not needed for rollout
    /// (action encoding uses separate f32 action encoder).
    pub fn dequantize(&self) -> LeWorldModel {
        use crate::weight_loading::AlignedBuffer;

        let hidden = self.config.predictor_hidden;
        let inner_dim = self.config.predictor_inner_dim;
        let inter = self.config.predictor_inter;

        let predictor_layers: Vec<AdaLNTransformerLayer> = self
            .predictor_layers
            .iter()
            .map(|layer| layer.dequantize(hidden, inner_dim, inter))
            .collect();

        // Create a minimal LeWorldModel with empty encoder (not used for rollout)
        let mut model = LeWorldModel::from_config(&self.config);

        // Fill in the predictor parts from dequantized layers
        model.predictor_layers = predictor_layers;
        model.predictor_pos_embed = AlignedBuffer::from_vec(self.predictor_pos_embed.clone());
        model.predictor_norm_weight = AlignedBuffer::from_vec(self.predictor_norm_weight.clone());
        model.predictor_norm_bias = AlignedBuffer::from_vec(self.predictor_norm_bias.clone());

        // Action encoder (f32 in both Full and LeWorldModel)
        model.action_conv_weight = AlignedBuffer::from_vec(self.action_conv_weight.clone());
        model.action_conv_bias = AlignedBuffer::from_vec(self.action_conv_bias.clone());
        model.action_mlp1_weight = AlignedBuffer::from_vec(self.action_mlp1_weight.clone());
        model.action_mlp1_bias = AlignedBuffer::from_vec(self.action_mlp1_bias.clone());
        model.action_mlp2_weight = AlignedBuffer::from_vec(self.action_mlp2_weight.clone());
        model.action_mlp2_bias = AlignedBuffer::from_vec(self.action_mlp2_bias.clone());

        // Input/conditioning projections
        model.input_proj_weight = AlignedBuffer::from_vec(self.input_proj_weight.clone());
        model.input_proj_bias = AlignedBuffer::from_vec(self.input_proj_bias.clone());
        model.cond_proj_weight = AlignedBuffer::from_vec(self.cond_proj_weight.clone());
        model.cond_proj_bias = AlignedBuffer::from_vec(self.cond_proj_bias.clone());

        // Projectors
        model.projector = self.projector.clone();
        model.pred_proj = self.pred_proj.clone();

        model
    }
}
pub fn quantize_lewm_full(model: &LeWorldModel) -> FullyQuantizedLeWM {
    let cfg = &model.config;
    let hidden = cfg.predictor_hidden;
    let inner_dim = cfg.predictor_inner_dim;
    let inter = cfg.predictor_inter;

    // Quantize ViT encoder layers to INT8
    let vit_cfg = &model.encoder.config;
    let enc_h = vit_cfg.hidden_size;
    let enc_heads = vit_cfg.num_heads;
    let enc_head_dim = vit_cfg.head_dim();
    let enc_inter = vit_cfg.intermediate_size;

    let encoder_layers: Vec<QuantizedEncoderLayer> = model
        .encoder
        .layers
        .iter()
        .map(|layer| QuantizedEncoderLayer {
            hidden_size: enc_h,
            num_heads: enc_heads,
            head_dim: enc_head_dim,
            w_q: QuantizedLinear::from_f32(&layer.w_q, enc_heads * enc_head_dim, enc_h),
            w_k: QuantizedLinear::from_f32(&layer.w_k, enc_heads * enc_head_dim, enc_h),
            w_v: QuantizedLinear::from_f32(&layer.w_v, enc_heads * enc_head_dim, enc_h),
            w_o: QuantizedLinear::from_f32(&layer.w_o, enc_h, enc_heads * enc_head_dim),
            ffn_up: QuantizedLinear::from_f32(&layer.ffn_up, enc_inter, enc_h),
            ffn_down: QuantizedLinear::from_f32(&layer.ffn_down, enc_h, enc_inter),
            q_bias: layer.q_bias.to_vec(),
            k_bias: layer.k_bias.to_vec(),
            v_bias: layer.v_bias.to_vec(),
            o_bias: layer.o_bias.to_vec(),
            ffn_up_bias: layer.ffn_up_bias.to_vec(),
            ffn_down_bias: layer.ffn_down_bias.to_vec(),
            attn_norm_weight: layer.attn_norm_weight.to_vec(),
            attn_norm_bias: layer.attn_norm_bias.to_vec(),
            ffn_norm_weight: layer.ffn_norm_weight.to_vec(),
            ffn_norm_bias: layer.ffn_norm_bias.to_vec(),
        })
        .collect();

    // Quantize predictor layers to Q4 (reuse existing quantize_lewm_q4 logic)
    let predictor_layers: Vec<QuantizedQ4AdaLNLayer> = model
        .predictor_layers
        .iter()
        .map(|layer| QuantizedQ4AdaLNLayer {
            adaln_linear: Q4Linear::from_f32(&layer.adaln_weight, 6 * hidden, hidden),
            adaln_bias: layer.adaln_bias.to_vec(),
            to_qkv: Q4Linear::from_f32(&layer.to_qkv, 3 * inner_dim, hidden),
            attn_out: Q4Linear::from_f32(&layer.attn_out_weight, hidden, inner_dim),
            attn_out_bias: layer.attn_out_bias.to_vec(),
            attn_norm_weight: layer.attn_norm_weight.to_vec(),
            attn_norm_bias: layer.attn_norm_bias.to_vec(),
            mlp_norm_weight: layer.mlp_norm_weight.to_vec(),
            mlp_norm_bias: layer.mlp_norm_bias.to_vec(),
            mlp_up: Q4Linear::from_f32(&layer.mlp_up_weight, inter, hidden),
            mlp_up_bias: layer.mlp_up_bias.to_vec(),
            mlp_down: Q4Linear::from_f32(&layer.mlp_down_weight, hidden, inter),
            mlp_down_bias: layer.mlp_down_bias.to_vec(),
        })
        .collect();

    use super::int8_lewm::clone_projection_head;

    FullyQuantizedLeWM {
        config: cfg.clone(),
        encoder_layers,
        patch_proj: model.encoder.patch_proj.to_vec(),
        patch_proj_bias: model.encoder.patch_proj_bias.to_vec(),
        cls_token: model.encoder.cls_token.to_vec(),
        pos_embed: model.encoder.pos_embed.to_vec(),
        final_norm_weight: model.encoder.final_norm_weight.to_vec(),
        final_norm_bias: model.encoder.final_norm_bias.to_vec(),
        vit_config: vit_cfg.clone(),
        predictor_layers,
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
        input_proj_weight: model.input_proj_weight.to_vec(),
        input_proj_bias: model.input_proj_bias.to_vec(),
        cond_proj_weight: model.cond_proj_weight.to_vec(),
        cond_proj_bias: model.cond_proj_bias.to_vec(),
        meta_token: model.encoder.meta_token.to_vec(),
        num_meta_tokens: model.encoder.num_meta_tokens,
        enc_proj_weight: model.encoder.enc_proj_weight.to_vec(),
        enc_proj_bias: model.encoder.enc_proj_bias.to_vec(),
    }
}

// ── Q4 encoder layer ──────────────────────────────────────────────

/// Q4-quantized ViT encoder layer (more aggressive than INT8).
pub struct Q4EncoderLayer {
    pub hidden_size: usize,
    pub num_heads: usize,
    pub head_dim: usize,
    pub w_q: Q4Linear,
    pub w_k: Q4Linear,
    pub w_v: Q4Linear,
    pub w_o: Q4Linear,
    pub ffn_up: Q4Linear,
    pub ffn_down: Q4Linear,
    pub q_bias: Vec<f32>,
    pub k_bias: Vec<f32>,
    pub v_bias: Vec<f32>,
    pub o_bias: Vec<f32>,
    pub ffn_up_bias: Vec<f32>,
    pub ffn_down_bias: Vec<f32>,
    pub attn_norm_weight: Vec<f32>,
    pub attn_norm_bias: Vec<f32>,
    pub ffn_norm_weight: Vec<f32>,
    pub ffn_norm_bias: Vec<f32>,
}

impl Q4EncoderLayer {
    fn add_bias(x: &mut [f32], bias: &[f32], m: usize, n: usize) {
        if bias.is_empty() {
            return;
        }
        for row in 0..m {
            for col in 0..n.min(bias.len()) {
                x[row * n + col] += bias[col];
            }
        }
    }

    pub fn forward(&self, x: &[f32], seq_len: usize) -> Vec<f32> {
        let h = self.hidden_size;
        let num_heads = self.num_heads;
        let head_dim = self.head_dim;

        let mut normed = layernorm(x, &self.attn_norm_weight, 1e-6, h);
        Self::add_bias(&mut normed, &self.attn_norm_bias, seq_len, h);

        let mut q = self.w_q.forward(&normed, seq_len);
        Self::add_bias(&mut q, &self.q_bias, seq_len, num_heads * head_dim);
        let mut k = self.w_k.forward(&normed, seq_len);
        Self::add_bias(&mut k, &self.k_bias, seq_len, num_heads * head_dim);
        let mut v = self.w_v.forward(&normed, seq_len);
        Self::add_bias(&mut v, &self.v_bias, seq_len, num_heads * head_dim);

        let attn_out = bidirectional_attention(&q, &k, &v, seq_len, num_heads, head_dim);

        let mut proj = self.w_o.forward(&attn_out, seq_len);
        Self::add_bias(&mut proj, &self.o_bias, seq_len, h);
        let mut residual = add_vecs(x, &proj);

        let mut normed2 = layernorm(&residual, &self.ffn_norm_weight, 1e-6, h);
        Self::add_bias(&mut normed2, &self.ffn_norm_bias, seq_len, h);
        let inter = self.ffn_up.out_features;

        let mut up = self.ffn_up.forward(&normed2, seq_len);
        Self::add_bias(&mut up, &self.ffn_up_bias, seq_len, inter);
        for val in up.iter_mut() {
            *val = gelu(*val);
        }

        let mut down = self.ffn_down.forward(&up, seq_len);
        Self::add_bias(&mut down, &self.ffn_down_bias, seq_len, h);
        add_vecs_inplace(&mut residual, &down);

        residual
    }

    pub fn memory_bytes(&self) -> usize {
        self.w_q.memory_bytes()
            + self.w_k.memory_bytes()
            + self.w_v.memory_bytes()
            + self.w_o.memory_bytes()
            + self.ffn_up.memory_bytes()
            + self.ffn_down.memory_bytes()
            + (self.q_bias.len()
                + self.k_bias.len()
                + self.v_bias.len()
                + self.o_bias.len()
                + self.ffn_up_bias.len()
                + self.ffn_down_bias.len()
                + self.attn_norm_weight.len()
                + self.attn_norm_bias.len()
                + self.ffn_norm_weight.len()
                + self.ffn_norm_bias.len())
                * 4
    }
}

/// Fully Q4 LEWM: Q4 encoder + Q4 predictor (~8MB runtime).
pub struct Q4FullLeWM {
    pub config: LeWMConfig,
    pub encoder_layers: Vec<Q4EncoderLayer>,
    pub patch_proj: Vec<f32>,
    pub patch_proj_bias: Vec<f32>,
    pub cls_token: Vec<f32>,
    pub pos_embed: Vec<f32>,
    pub final_norm_weight: Vec<f32>,
    pub final_norm_bias: Vec<f32>,
    pub vit_config: ViTConfig,
    pub predictor_layers: Vec<QuantizedQ4AdaLNLayer>,
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
    // Input/conditioning projections (latent_dim → predictor_hidden bottleneck)
    pub input_proj_weight: Vec<f32>,
    pub input_proj_bias: Vec<f32>,
    pub cond_proj_weight: Vec<f32>,
    pub cond_proj_bias: Vec<f32>,
}

impl Q4FullLeWM {
    pub fn encode(&self, image: &[f32], h: usize, w: usize) -> Vec<f32> {
        let cfg = &self.vit_config;
        let hidden = cfg.hidden_size;
        let seq_len = cfg.seq_len();
        let patches = patch_embed(
            image,
            h,
            w,
            cfg.channels,
            cfg.patch_size,
            &self.patch_proj,
            hidden,
        );
        let num_patches = cfg.num_patches();
        let mut x = vec![0.0f32; seq_len * hidden];
        x[..hidden].copy_from_slice(&self.cls_token);
        x[hidden..hidden + num_patches * hidden].copy_from_slice(&patches);
        for i in 0..seq_len * hidden {
            x[i] += self.pos_embed[i];
        }
        for layer in &self.encoder_layers {
            x = layer.forward(&x, seq_len);
        }
        let cls = &x[..hidden];
        let normed = layernorm(cls, &self.final_norm_weight, 1e-6, hidden);
        // Note: Q4FullLeWM does not have enc_proj fields — skip for this path
        self.projector.forward(&normed)
    }

    pub fn predict_next(&self, z_t: &[f32], action: &[f32]) -> Vec<f32> {
        let hidden = self.config.predictor_hidden;
        let latent = self.config.latent_dim;
        let num_heads = self.config.predictor_heads;
        let inner_dim = self.config.predictor_inner_dim;
        let inter = self.config.predictor_inter;
        let has_proj = !self.input_proj_weight.is_empty();

        let a_embed = self.encode_action(action, self.config.latent_dim);

        // Build input sequence: [z_t, a_embed, target_token]
        let seq_len = 3;
        let seq_dim = if has_proj { latent } else { hidden };
        let mut seq = vec![0.0f32; seq_len * seq_dim];
        seq[..seq_dim].copy_from_slice(z_t);
        seq[seq_dim..2 * seq_dim].copy_from_slice(&a_embed);

        // Add positional embeddings
        if !self.predictor_pos_embed.is_empty() {
            let pos_len = self.predictor_pos_embed.len().min(seq.len());
            for i in 0..pos_len {
                seq[i] += self.predictor_pos_embed[i];
            }
        }

        // Apply projections if bottleneck architecture
        let (mut x, conditioning) = if has_proj {
            let projected_seq = super::apply_input_proj(
                &self.input_proj_weight,
                &self.input_proj_bias,
                &seq,
                seq_len,
                latent,
                hidden,
            );
            let projected_cond = super::apply_cond_proj(
                &self.cond_proj_weight,
                &self.cond_proj_bias,
                &a_embed,
                latent,
                hidden,
            );
            (projected_seq, projected_cond)
        } else {
            (seq, a_embed)
        };

        for layer in &self.predictor_layers {
            x = layer.forward(
                &x,
                &conditioning,
                seq_len,
                hidden,
                num_heads,
                inner_dim,
                inter,
            );
        }
        let normed = layernorm(&x, &self.predictor_norm_weight, 1e-6, hidden);
        let target = &normed[2 * hidden..3 * hidden];
        self.pred_proj.forward(target)
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
        let mut conv_out = matmul_t(action, &self.action_conv_weight, 1, act_dim, act_dim);
        for j in 0..act_dim.min(self.action_conv_bias.len()) {
            conv_out[j] += self.action_conv_bias[j];
        }
        let inter = if !self.action_mlp1_weight.is_empty() {
            self.action_mlp1_weight.len() / act_dim
        } else {
            hidden * 4
        };
        let mut h1 = matmul_t(&conv_out, &self.action_mlp1_weight, 1, act_dim, inter);
        for j in 0..inter.min(self.action_mlp1_bias.len()) {
            h1[j] += self.action_mlp1_bias[j];
        }
        for v in h1.iter_mut() {
            *v = gelu(*v);
        }
        let mut out = matmul_t(&h1, &self.action_mlp2_weight, 1, inter, hidden);
        for j in 0..hidden.min(self.action_mlp2_bias.len()) {
            out[j] += self.action_mlp2_bias[j];
        }
        out
    }

    pub fn model_size_bytes(&self) -> usize {
        let enc: usize = self.encoder_layers.iter().map(|l| l.memory_bytes()).sum();
        let enc_misc = (self.patch_proj.len()
            + self.patch_proj_bias.len()
            + self.cls_token.len()
            + self.pos_embed.len()
            + self.final_norm_weight.len()
            + self.final_norm_bias.len())
            * 4;
        let pred: usize = self
            .predictor_layers
            .iter()
            .map(|l| {
                l.adaln_linear.memory_bytes()
                    + l.to_qkv.memory_bytes()
                    + l.attn_out.memory_bytes()
                    + l.mlp_up.memory_bytes()
                    + l.mlp_down.memory_bytes()
                    + (l.adaln_bias.len()
                        + l.attn_out_bias.len()
                        + l.attn_norm_weight.len()
                        + l.attn_norm_bias.len()
                        + l.mlp_norm_weight.len()
                        + l.mlp_norm_bias.len()
                        + l.mlp_up_bias.len()
                        + l.mlp_down_bias.len())
                        * 4
            })
            .sum();
        let pred_misc = (self.predictor_pos_embed.len()
            + self.predictor_norm_weight.len()
            + self.predictor_norm_bias.len())
            * 4;
        let action = (self.action_conv_weight.len()
            + self.action_conv_bias.len()
            + self.action_mlp1_weight.len()
            + self.action_mlp1_bias.len()
            + self.action_mlp2_weight.len()
            + self.action_mlp2_bias.len())
            * 4;
        enc + enc_misc + pred + pred_misc + action
    }
}

/// Quantize to Q4 encoder + Q4 predictor.
pub fn quantize_lewm_q4_full(model: &LeWorldModel) -> Q4FullLeWM {
    let cfg = &model.config;
    let hidden = cfg.predictor_hidden;
    let inner_dim = cfg.predictor_inner_dim;
    let inter = cfg.predictor_inter;
    let vit_cfg = &model.encoder.config;
    let enc_h = vit_cfg.hidden_size;
    let enc_heads = vit_cfg.num_heads;
    let enc_head_dim = vit_cfg.head_dim();
    let enc_inter = vit_cfg.intermediate_size;

    let encoder_layers: Vec<Q4EncoderLayer> = model
        .encoder
        .layers
        .iter()
        .map(|layer| Q4EncoderLayer {
            hidden_size: enc_h,
            num_heads: enc_heads,
            head_dim: enc_head_dim,
            w_q: Q4Linear::from_f32(&layer.w_q, enc_heads * enc_head_dim, enc_h),
            w_k: Q4Linear::from_f32(&layer.w_k, enc_heads * enc_head_dim, enc_h),
            w_v: Q4Linear::from_f32(&layer.w_v, enc_heads * enc_head_dim, enc_h),
            w_o: Q4Linear::from_f32(&layer.w_o, enc_h, enc_heads * enc_head_dim),
            ffn_up: Q4Linear::from_f32(&layer.ffn_up, enc_inter, enc_h),
            ffn_down: Q4Linear::from_f32(&layer.ffn_down, enc_h, enc_inter),
            q_bias: layer.q_bias.to_vec(),
            k_bias: layer.k_bias.to_vec(),
            v_bias: layer.v_bias.to_vec(),
            o_bias: layer.o_bias.to_vec(),
            ffn_up_bias: layer.ffn_up_bias.to_vec(),
            ffn_down_bias: layer.ffn_down_bias.to_vec(),
            attn_norm_weight: layer.attn_norm_weight.to_vec(),
            attn_norm_bias: layer.attn_norm_bias.to_vec(),
            ffn_norm_weight: layer.ffn_norm_weight.to_vec(),
            ffn_norm_bias: layer.ffn_norm_bias.to_vec(),
        })
        .collect();

    let predictor_layers = model
        .predictor_layers
        .iter()
        .map(|layer| QuantizedQ4AdaLNLayer {
            adaln_linear: Q4Linear::from_f32(&layer.adaln_weight, 6 * hidden, hidden),
            adaln_bias: layer.adaln_bias.to_vec(),
            to_qkv: Q4Linear::from_f32(&layer.to_qkv, 3 * inner_dim, hidden),
            attn_out: Q4Linear::from_f32(&layer.attn_out_weight, hidden, inner_dim),
            attn_out_bias: layer.attn_out_bias.to_vec(),
            attn_norm_weight: layer.attn_norm_weight.to_vec(),
            attn_norm_bias: layer.attn_norm_bias.to_vec(),
            mlp_norm_weight: layer.mlp_norm_weight.to_vec(),
            mlp_norm_bias: layer.mlp_norm_bias.to_vec(),
            mlp_up: Q4Linear::from_f32(&layer.mlp_up_weight, inter, hidden),
            mlp_up_bias: layer.mlp_up_bias.to_vec(),
            mlp_down: Q4Linear::from_f32(&layer.mlp_down_weight, hidden, inter),
            mlp_down_bias: layer.mlp_down_bias.to_vec(),
        })
        .collect();

    use super::int8_lewm::clone_projection_head;

    Q4FullLeWM {
        config: cfg.clone(),
        encoder_layers,
        patch_proj: model.encoder.patch_proj.to_vec(),
        patch_proj_bias: model.encoder.patch_proj_bias.to_vec(),
        cls_token: model.encoder.cls_token.to_vec(),
        pos_embed: model.encoder.pos_embed.to_vec(),
        final_norm_weight: model.encoder.final_norm_weight.to_vec(),
        final_norm_bias: model.encoder.final_norm_bias.to_vec(),
        vit_config: vit_cfg.clone(),
        predictor_layers,
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
        input_proj_weight: model.input_proj_weight.to_vec(),
        input_proj_bias: model.input_proj_bias.to_vec(),
        cond_proj_weight: model.cond_proj_weight.to_vec(),
        cond_proj_bias: model.cond_proj_bias.to_vec(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quantized_encoder_layer_forward() {
        let h = 16;
        let heads = 2;
        let head_dim = 8;
        let inter = 32;
        let seq_len = 4;

        let layer = QuantizedEncoderLayer {
            hidden_size: h,
            num_heads: heads,
            head_dim,
            w_q: QuantizedLinear::from_f32(&vec![0.01; h * h], h, h),
            w_k: QuantizedLinear::from_f32(&vec![0.01; h * h], h, h),
            w_v: QuantizedLinear::from_f32(&vec![0.01; h * h], h, h),
            w_o: QuantizedLinear::from_f32(&vec![0.01; h * h], h, h),
            ffn_up: QuantizedLinear::from_f32(&vec![0.01; inter * h], inter, h),
            ffn_down: QuantizedLinear::from_f32(&vec![0.01; h * inter], h, inter),
            q_bias: vec![0.0; h],
            k_bias: vec![0.0; h],
            v_bias: vec![0.0; h],
            o_bias: vec![0.0; h],
            ffn_up_bias: vec![0.0; inter],
            ffn_down_bias: vec![0.0; h],
            attn_norm_weight: vec![1.0; h],
            attn_norm_bias: vec![0.0; h],
            ffn_norm_weight: vec![1.0; h],
            ffn_norm_bias: vec![0.0; h],
        };

        let x = vec![0.1f32; seq_len * h];
        let out = layer.forward(&x, seq_len);

        assert_eq!(out.len(), seq_len * h);
        assert!(out.iter().all(|v| v.is_finite()));
    }
}
