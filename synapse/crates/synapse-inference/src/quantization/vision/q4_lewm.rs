//! Q4_0-quantized LeWorldModel variants.

use crate::models::vision::lewm::{LeWMConfig, LeWorldModel, ProjectionHead};
use crate::models::vision::vit::ViTModel;
use crate::ops::activation::gelu;
use crate::ops::attention::bidirectional_attention;
use crate::ops::norm::layernorm;

use crate::quantization::Q4Linear;

// ---------------------------------------------------------------------------
// Q4-quantized DiT adaLN layer and full LeWM
// ---------------------------------------------------------------------------

/// A Q4_0-quantized DiT-style adaLN transformer layer.
///
/// Same forward logic as [`QuantizedAdaLNLayer`](super::QuantizedAdaLNLayer)
/// but uses [`Q4Linear`] instead of [`QuantizedLinear`](super::QuantizedLinear).
pub struct QuantizedQ4AdaLNLayer {
    // adaLN modulation: [hidden, 6*hidden]
    pub adaln_linear: Q4Linear,
    pub adaln_bias: Vec<f32>,
    // Fused QKV: [hidden, 3*inner_dim]
    pub to_qkv: Q4Linear,
    // Output projection: [inner_dim, hidden]
    pub attn_out: Q4Linear,
    pub attn_out_bias: Vec<f32>,
    // Norms stay f32
    pub attn_norm_weight: Vec<f32>,
    pub attn_norm_bias: Vec<f32>,
    pub mlp_norm_weight: Vec<f32>,
    pub mlp_norm_bias: Vec<f32>,
    // MLP up: [hidden, inter]
    pub mlp_up: Q4Linear,
    pub mlp_up_bias: Vec<f32>,
    // MLP down: [inter, hidden]
    pub mlp_down: Q4Linear,
    pub mlp_down_bias: Vec<f32>,
}

impl QuantizedQ4AdaLNLayer {
    /// Forward pass for one Q4-quantized DiT adaLN layer.
    ///
    /// Same logic as the INT8 [`QuantizedAdaLNLayer::forward()`] but dispatches
    /// through `Q4Linear::forward()`.
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

        // 1. Compute adaLN modulation: conditioning [hidden] -> mod_vec [6*hidden]
        let mut mod_vec = self.adaln_linear.forward(conditioning, 1);
        debug_assert_eq!(mod_vec.len(), mod_dim);
        for j in 0..mod_dim.min(self.adaln_bias.len()) {
            mod_vec[j] += self.adaln_bias[j];
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
        //    modulated: [seq_len, hidden] -> qkv: [seq_len, 3*inner_dim]
        let qkv = self.to_qkv.forward(&modulated, seq_len);
        debug_assert_eq!(qkv.len(), seq_len * 3 * inner_dim);

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

        // Output projection: [seq_len, inner_dim] -> [seq_len, hidden]
        let mut proj = self.attn_out.forward(&attn_out, seq_len);
        for t in 0..seq_len {
            for j in 0..hidden.min(self.attn_out_bias.len()) {
                proj[t * hidden + j] += self.attn_out_bias[j];
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

        // 6. MLP: up -> GELU -> down
        let mut up = self.mlp_up.forward(&modulated2, seq_len);
        for t in 0..seq_len {
            for j in 0..inter.min(self.mlp_up_bias.len()) {
                up[t * inter + j] += self.mlp_up_bias[j];
            }
        }
        for val in up.iter_mut() {
            *val = gelu(*val);
        }
        let mut down = self.mlp_down.forward(&up, seq_len);
        for t in 0..seq_len {
            for j in 0..hidden.min(self.mlp_down_bias.len()) {
                down[t * hidden + j] += self.mlp_down_bias[j];
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

/// Q4_0-quantized LeWorldModel.
///
/// The predictor's adaLN transformer layers use Q4_0 weights via [`Q4Linear`],
/// reducing predictor memory by ~6.4x vs f32 (and ~2x vs INT8). The encoder,
/// action encoder, and projection heads remain in f32.
pub struct QuantizedQ4LeWM {
    pub config: LeWMConfig,
    /// ViT encoder stays f32 (~2.8M params, not worth quantizing yet).
    pub encoder: ViTModel,
    /// Q4-quantized predictor layers.
    pub predictor_layers: Vec<QuantizedQ4AdaLNLayer>,
    pub predictor_pos_embed: Vec<f32>,
    pub predictor_norm_weight: Vec<f32>,
    pub predictor_norm_bias: Vec<f32>,
    // Action encoder -- small, keep f32
    pub action_conv_weight: Vec<f32>,
    pub action_conv_bias: Vec<f32>,
    pub action_mlp1_weight: Vec<f32>,
    pub action_mlp1_bias: Vec<f32>,
    pub action_mlp2_weight: Vec<f32>,
    pub action_mlp2_bias: Vec<f32>,
    // Projection heads -- small, keep f32
    pub projector: ProjectionHead,
    pub pred_proj: ProjectionHead,
    // Input/conditioning projections (latent_dim → predictor_hidden bottleneck)
    pub input_proj_weight: Vec<f32>,
    pub input_proj_bias: Vec<f32>,
    pub cond_proj_weight: Vec<f32>,
    pub cond_proj_bias: Vec<f32>,
}

impl QuantizedQ4LeWM {
    /// Encode an observation image to a latent state in predictor space.
    ///
    /// Delegates to the f32 ViT encoder (not quantized).
    pub fn encode(&self, image: &[f32], h: usize, w: usize) -> Vec<f32> {
        let vit_out = self.encoder.forward_image(image, h, w);
        self.projector.forward(&vit_out.embeddings)
    }

    /// Encode an action vector to an action embedding (f32 path).
    fn encode_action(&self, action: &[f32]) -> Vec<f32> {
        let act_dim = self.config.action_dim;
        let hidden = self.config.latent_dim;

        // 1. 1D conv with kernel_size=1 (equivalent to linear layer)
        let mut conv_out = vec![0.0f32; act_dim];
        if !self.action_conv_weight.is_empty() {
            let weight_elems = act_dim * act_dim;
            if self.action_conv_weight.len() >= weight_elems {
                conv_out = crate::ops::matmul::matmul_t(
                    action,
                    &self.action_conv_weight,
                    1,
                    act_dim,
                    act_dim,
                );
            }
            for j in 0..act_dim.min(self.action_conv_bias.len()) {
                conv_out[j] += self.action_conv_bias[j];
            }
        } else {
            conv_out.copy_from_slice(action);
        }

        // 2. MLP: [act_dim] -> [inter] (GELU) -> [hidden]
        let inter = if !self.action_mlp1_weight.is_empty() {
            self.action_mlp1_weight.len() / act_dim
        } else {
            hidden * 4
        };

        let mut h1 = if !self.action_mlp1_weight.is_empty() {
            crate::ops::matmul::matmul_t(&conv_out, &self.action_mlp1_weight, 1, act_dim, inter)
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
            crate::ops::matmul::matmul_t(&h1, &self.action_mlp2_weight, 1, inter, hidden)
        } else {
            vec![0.0f32; hidden]
        };
        for j in 0..hidden.min(self.action_mlp2_bias.len()) {
            out[j] += self.action_mlp2_bias[j];
        }

        out
    }

    /// Predict the next latent state given current latent and action.
    ///
    /// Uses the Q4-quantized predictor layers for the heavy DiT forward pass.
    pub fn predict_next(&self, z_t: &[f32], action: &[f32]) -> Vec<f32> {
        let hidden = self.config.predictor_hidden;
        let latent = self.config.latent_dim;
        let num_heads = self.config.predictor_heads;
        let inner_dim = self.config.predictor_inner_dim;
        let inter = self.config.predictor_inter;
        let has_proj = !self.input_proj_weight.is_empty();

        // 1. Encode action -> [latent_dim]
        let a_embed = self.encode_action(action);

        // 2. Build input sequence at latent_dim or predictor_hidden
        let seq_len = 3;
        let seq_dim = if has_proj { latent } else { hidden };
        let mut seq = vec![0.0f32; seq_len * seq_dim];
        seq[..seq_dim].copy_from_slice(z_t);
        seq[seq_dim..2 * seq_dim].copy_from_slice(&a_embed);
        // seq[2*seq_dim..3*seq_dim] = zeros (target position to be predicted)

        // 3. Add positional embeddings
        if !self.predictor_pos_embed.is_empty() {
            let pos_len = self.predictor_pos_embed.len().min(seq.len());
            for i in 0..pos_len {
                seq[i] += self.predictor_pos_embed[i];
            }
        }

        // 4. Apply projections if bottleneck architecture
        let (mut seq, conditioning) = if has_proj {
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

        // 5. Run through Q4-quantized predictor layers
        for layer in &self.predictor_layers {
            seq = layer.forward(
                &seq,
                &conditioning,
                seq_len,
                hidden,
                num_heads,
                inner_dim,
                inter,
            );
        }

        // 6. Final norm
        let mut normed = layernorm(&seq, &self.predictor_norm_weight, 1e-6, hidden);
        if !self.predictor_norm_bias.is_empty() {
            for t in 0..seq_len {
                for j in 0..hidden {
                    normed[t * hidden + j] += self.predictor_norm_bias[j];
                }
            }
        }

        // 7. Extract target position (index 2) -> [hidden]
        let target = &normed[2 * hidden..3 * hidden];

        // 8. Project back through pred_proj (f32)
        self.pred_proj.forward(target)
    }

    /// Multi-step rollout: predict a sequence of future latent states.
    pub fn rollout(&self, z_start: &[f32], actions: &[Vec<f32>]) -> Vec<Vec<f32>> {
        let mut states = Vec::with_capacity(actions.len());
        let mut z = z_start.to_vec();
        for action in actions {
            z = self.predict_next(&z, action);
            states.push(z.clone());
        }
        states
    }

    /// Load from LQ40 binary format (as produced by `export_lewm_q4`).
    ///
    /// Format: `[4B "LQ40"][4B config_len][JSON config][sequential weight data]`
    ///
    /// Parses the `q4-pred` mode binary: f32 ViT encoder + Q4 predictor layers.
    /// Automatically supports slim models when the config has `latent_dim != predictor_hidden`.
    pub fn from_lq40_bytes(data: &[u8]) -> Result<Self, String> {
        use crate::weight_loading::AlignedBuffer;

        if data.len() < 8 {
            return Err("LQ40 data too short".into());
        }
        if &data[0..4] != b"LQ40" {
            return Err("Not LQ40 format".into());
        }

        let config_len = u32::from_le_bytes([data[4], data[5], data[6], data[7]]) as usize;
        if data.len() < 8 + config_len {
            return Err("LQ40 config truncated".into());
        }
        let config_json: serde_json::Value = serde_json::from_slice(&data[8..8 + config_len])
            .map_err(|e| format!("LQ40 config parse error: {}", e))?;

        let config = lq40_config_from_json(&config_json)?;
        let mut off = 8 + config_len;

        // --- f32 Encoder (use from_config + populate weights) ---
        let vit_config = crate::models::vision::vit::ViTConfig {
            image_size: config.image_size,
            patch_size: config.patch_size,
            hidden_size: config.encoder_hidden,
            num_layers: config.encoder_layers,
            num_heads: config.encoder_heads,
            intermediate_size: config.encoder_inter,
            channels: config.channels,
            num_classes: 0,
        };
        let mut encoder = ViTModel::from_config(&vit_config);

        encoder.patch_proj = AlignedBuffer::from_slice(&lq40_read_f32(data, &mut off));
        encoder.patch_proj_bias = AlignedBuffer::from_slice(&lq40_read_f32(data, &mut off));
        encoder.cls_token = AlignedBuffer::from_slice(&lq40_read_f32(data, &mut off));
        encoder.pos_embed = AlignedBuffer::from_slice(&lq40_read_f32(data, &mut off));

        for layer in encoder.layers.iter_mut() {
            layer.attn_norm_weight = AlignedBuffer::from_slice(&lq40_read_f32(data, &mut off));
            layer.attn_norm_bias = AlignedBuffer::from_slice(&lq40_read_f32(data, &mut off));
            layer.w_q = AlignedBuffer::from_slice(&lq40_read_f32(data, &mut off));
            layer.q_bias = AlignedBuffer::from_slice(&lq40_read_f32(data, &mut off));
            layer.w_k = AlignedBuffer::from_slice(&lq40_read_f32(data, &mut off));
            layer.k_bias = AlignedBuffer::from_slice(&lq40_read_f32(data, &mut off));
            layer.w_v = AlignedBuffer::from_slice(&lq40_read_f32(data, &mut off));
            layer.v_bias = AlignedBuffer::from_slice(&lq40_read_f32(data, &mut off));
            layer.w_o = AlignedBuffer::from_slice(&lq40_read_f32(data, &mut off));
            layer.o_bias = AlignedBuffer::from_slice(&lq40_read_f32(data, &mut off));
            layer.ffn_norm_weight = AlignedBuffer::from_slice(&lq40_read_f32(data, &mut off));
            layer.ffn_norm_bias = AlignedBuffer::from_slice(&lq40_read_f32(data, &mut off));
            layer.ffn_up = AlignedBuffer::from_slice(&lq40_read_f32(data, &mut off));
            layer.ffn_up_bias = AlignedBuffer::from_slice(&lq40_read_f32(data, &mut off));
            layer.ffn_down = AlignedBuffer::from_slice(&lq40_read_f32(data, &mut off));
            layer.ffn_down_bias = AlignedBuffer::from_slice(&lq40_read_f32(data, &mut off));
        }
        encoder.final_norm_weight = AlignedBuffer::from_slice(&lq40_read_f32(data, &mut off));
        encoder.final_norm_bias = AlignedBuffer::from_slice(&lq40_read_f32(data, &mut off));

        // --- Q4 Predictor ---
        let predictor_pos_embed = lq40_read_f32(data, &mut off);
        let mut predictor_layers = Vec::with_capacity(config.predictor_layers);
        for _ in 0..config.predictor_layers {
            let layer = lq40_read_q4_adaln_layer(data, &mut off)?;
            predictor_layers.push(layer);
        }
        let predictor_norm_weight = lq40_read_f32(data, &mut off);
        let predictor_norm_bias = lq40_read_f32(data, &mut off);

        // --- Action encoder (f32) ---
        let action_conv_weight = lq40_read_f32(data, &mut off);
        let action_conv_bias = lq40_read_f32(data, &mut off);
        let action_mlp1_weight = lq40_read_f32(data, &mut off);
        let action_mlp1_bias = lq40_read_f32(data, &mut off);
        let action_mlp2_weight = lq40_read_f32(data, &mut off);
        let action_mlp2_bias = lq40_read_f32(data, &mut off);

        // --- Projectors (f32) ---
        let projector = lq40_read_projection_head(data, &mut off);
        let pred_proj = lq40_read_projection_head(data, &mut off);

        // --- Input/Cond projections (optional, for slim models) ---
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

        Ok(QuantizedQ4LeWM {
            config,
            encoder,
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
        })
    }
}

/// Quantize a LeWorldModel to Q4_0.
///
/// Converts the predictor's adaLN transformer layers from f32 to Q4_0 blocks.
/// The encoder, action encoder, and projection heads are copied as-is (f32).
pub fn quantize_lewm_q4(model: &LeWorldModel) -> QuantizedQ4LeWM {
    let cfg = &model.config;
    let hidden = cfg.predictor_hidden;
    let inner_dim = cfg.predictor_inner_dim;
    let inter = cfg.predictor_inter;

    let predictor_layers = model
        .predictor_layers
        .iter()
        .map(|layer| {
            // adaLN modulation: [6*hidden, hidden]
            let adaln_linear = Q4Linear::from_f32(&layer.adaln_weight, 6 * hidden, hidden);
            // Fused QKV: [3*inner_dim, hidden]
            let to_qkv = Q4Linear::from_f32(&layer.to_qkv, 3 * inner_dim, hidden);
            // Output projection: [hidden, inner_dim]
            let attn_out = Q4Linear::from_f32(&layer.attn_out_weight, hidden, inner_dim);
            // MLP up: [inter, hidden]
            let mlp_up = Q4Linear::from_f32(&layer.mlp_up_weight, inter, hidden);
            // MLP down: [hidden, inter]
            let mlp_down = Q4Linear::from_f32(&layer.mlp_down_weight, hidden, inter);

            QuantizedQ4AdaLNLayer {
                adaln_linear,
                adaln_bias: layer.adaln_bias.to_vec(),
                to_qkv,
                attn_out,
                attn_out_bias: layer.attn_out_bias.to_vec(),
                attn_norm_weight: layer.attn_norm_weight.to_vec(),
                attn_norm_bias: layer.attn_norm_bias.to_vec(),
                mlp_norm_weight: layer.mlp_norm_weight.to_vec(),
                mlp_norm_bias: layer.mlp_norm_bias.to_vec(),
                mlp_up,
                mlp_up_bias: layer.mlp_up_bias.to_vec(),
                mlp_down,
                mlp_down_bias: layer.mlp_down_bias.to_vec(),
            }
        })
        .collect();

    QuantizedQ4LeWM {
        config: cfg.clone(),
        encoder: clone_vit_encoder(&model.encoder),
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

// ---------------------------------------------------------------------------
// CachedQ4Linear: dequant-at-load for BLAS-speed Q4 inference
// ---------------------------------------------------------------------------

/// Q4 storage with f32 compute cache.
///
/// Weights quantized to Q4 for compression (7MB on disk), then dequantized
/// to f32 at load time for fast inference via platform BLAS (Accelerate on Mac).
/// Separation of concerns: Q4 is the storage format, f32 is the compute format.
pub struct CachedQ4Linear {
    f32_weights: Vec<f32>, // [out_features, in_features] dequantized
    pub out_features: usize,
    pub in_features: usize,
}

impl CachedQ4Linear {
    /// Create from an existing Q4Linear by dequantizing to f32.
    pub fn from_q4(q4: &Q4Linear) -> Self {
        CachedQ4Linear {
            f32_weights: q4.dequantize(),
            out_features: q4.out_features,
            in_features: q4.in_features,
        }
    }

    /// Create from f32 weights: quantize to Q4, then immediately dequant.
    /// This gives Q4-precision weights with f32 compute speed.
    pub fn from_f32(weights: &[f32], out_features: usize, in_features: usize) -> Self {
        let q4 = Q4Linear::from_f32(weights, out_features, in_features);
        Self::from_q4(&q4)
    }

    /// Forward: x [m, in_features] → [m, out_features]
    /// Uses platform-optimal matmul (Accelerate on Mac, Zig on Linux, pure-Rust on WASM).
    pub fn forward(&self, x: &[f32], m: usize) -> Vec<f32> {
        crate::ops::matmul::matmul_t(x, &self.f32_weights, m, self.in_features, self.out_features)
    }

    pub fn empty() -> Self {
        CachedQ4Linear {
            f32_weights: Vec::new(),
            out_features: 0,
            in_features: 0,
        }
    }
}

// ---------------------------------------------------------------------------
// CachedQ4AdaLNLayer: same structure as QuantizedQ4AdaLNLayer, CachedQ4Linear
// ---------------------------------------------------------------------------

/// A DiT-style adaLN transformer layer backed by [`CachedQ4Linear`].
///
/// Same forward logic as [`QuantizedQ4AdaLNLayer`] but dispatches through
/// `CachedQ4Linear::forward()` for platform-BLAS speed.
pub struct CachedQ4AdaLNLayer {
    pub adaln_linear: CachedQ4Linear,
    pub adaln_bias: Vec<f32>,
    pub to_qkv: CachedQ4Linear,
    pub attn_out: CachedQ4Linear,
    pub attn_out_bias: Vec<f32>,
    pub attn_norm_weight: Vec<f32>,
    pub attn_norm_bias: Vec<f32>,
    pub mlp_norm_weight: Vec<f32>,
    pub mlp_norm_bias: Vec<f32>,
    pub mlp_up: CachedQ4Linear,
    pub mlp_up_bias: Vec<f32>,
    pub mlp_down: CachedQ4Linear,
    pub mlp_down_bias: Vec<f32>,
}

impl CachedQ4AdaLNLayer {
    /// Forward pass for one CachedQ4 DiT adaLN layer.
    ///
    /// Identical logic to [`QuantizedQ4AdaLNLayer::forward()`] but uses
    /// `CachedQ4Linear::forward` instead of `Q4Linear::forward`.
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

        // 1. Compute adaLN modulation: conditioning [hidden] -> mod_vec [6*hidden]
        let mut mod_vec = self.adaln_linear.forward(conditioning, 1);
        debug_assert_eq!(mod_vec.len(), mod_dim);
        for j in 0..mod_dim.min(self.adaln_bias.len()) {
            mod_vec[j] += self.adaln_bias[j];
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
        //    modulated: [seq_len, hidden] -> qkv: [seq_len, 3*inner_dim]
        let qkv = self.to_qkv.forward(&modulated, seq_len);
        debug_assert_eq!(qkv.len(), seq_len * 3 * inner_dim);

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

        // Output projection: [seq_len, inner_dim] -> [seq_len, hidden]
        let mut proj = self.attn_out.forward(&attn_out, seq_len);
        for t in 0..seq_len {
            for j in 0..hidden.min(self.attn_out_bias.len()) {
                proj[t * hidden + j] += self.attn_out_bias[j];
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

        // 6. MLP: up -> GELU -> down
        let mut up = self.mlp_up.forward(&modulated2, seq_len);
        for t in 0..seq_len {
            for j in 0..inter.min(self.mlp_up_bias.len()) {
                up[t * inter + j] += self.mlp_up_bias[j];
            }
        }
        for val in up.iter_mut() {
            *val = gelu(*val);
        }
        let mut down = self.mlp_down.forward(&up, seq_len);
        for t in 0..seq_len {
            for j in 0..hidden.min(self.mlp_down_bias.len()) {
                down[t * hidden + j] += self.mlp_down_bias[j];
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

// ---------------------------------------------------------------------------
// CachedQ4LeWM: same as QuantizedQ4LeWM but with CachedQ4AdaLNLayer
// ---------------------------------------------------------------------------

/// CachedQ4 LeWorldModel.
///
/// Predictor adaLN layers use [`CachedQ4Linear`]: Q4 compression at rest,
/// f32/BLAS speed at compute. Encoder, action encoder, and projection heads
/// remain in f32.
pub struct CachedQ4LeWM {
    pub config: LeWMConfig,
    /// ViT encoder stays f32 (~2.8M params, not worth quantizing yet).
    pub encoder: ViTModel,
    /// CachedQ4 predictor layers.
    pub predictor_layers: Vec<CachedQ4AdaLNLayer>,
    pub predictor_pos_embed: Vec<f32>,
    pub predictor_norm_weight: Vec<f32>,
    pub predictor_norm_bias: Vec<f32>,
    // Action encoder -- small, keep f32
    pub action_conv_weight: Vec<f32>,
    pub action_conv_bias: Vec<f32>,
    pub action_mlp1_weight: Vec<f32>,
    pub action_mlp1_bias: Vec<f32>,
    pub action_mlp2_weight: Vec<f32>,
    pub action_mlp2_bias: Vec<f32>,
    // Projection heads -- small, keep f32
    pub projector: ProjectionHead,
    pub pred_proj: ProjectionHead,
    // Input/conditioning projections (latent_dim → predictor_hidden bottleneck)
    pub input_proj_weight: Vec<f32>,
    pub input_proj_bias: Vec<f32>,
    pub cond_proj_weight: Vec<f32>,
    pub cond_proj_bias: Vec<f32>,
}

impl CachedQ4LeWM {
    /// Encode an observation image to a latent state in predictor space.
    ///
    /// Delegates to the f32 ViT encoder (not quantized).
    pub fn encode(&self, image: &[f32], h: usize, w: usize) -> Vec<f32> {
        let vit_out = self.encoder.forward_image(image, h, w);
        self.projector.forward(&vit_out.embeddings)
    }

    /// Encode an action vector to an action embedding (f32 path).
    fn encode_action(&self, action: &[f32]) -> Vec<f32> {
        let act_dim = self.config.action_dim;
        let hidden = self.config.latent_dim;

        // 1. 1D conv with kernel_size=1 (equivalent to linear layer)
        let mut conv_out = vec![0.0f32; act_dim];
        if !self.action_conv_weight.is_empty() {
            let weight_elems = act_dim * act_dim;
            if self.action_conv_weight.len() >= weight_elems {
                conv_out = crate::ops::matmul::matmul_t(
                    action,
                    &self.action_conv_weight,
                    1,
                    act_dim,
                    act_dim,
                );
            }
            for j in 0..act_dim.min(self.action_conv_bias.len()) {
                conv_out[j] += self.action_conv_bias[j];
            }
        } else {
            conv_out.copy_from_slice(action);
        }

        // 2. MLP: [act_dim] -> [inter] (GELU) -> [hidden]
        let inter = if !self.action_mlp1_weight.is_empty() {
            self.action_mlp1_weight.len() / act_dim
        } else {
            hidden * 4
        };

        let mut h1 = if !self.action_mlp1_weight.is_empty() {
            crate::ops::matmul::matmul_t(&conv_out, &self.action_mlp1_weight, 1, act_dim, inter)
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
            crate::ops::matmul::matmul_t(&h1, &self.action_mlp2_weight, 1, inter, hidden)
        } else {
            vec![0.0f32; hidden]
        };
        for j in 0..hidden.min(self.action_mlp2_bias.len()) {
            out[j] += self.action_mlp2_bias[j];
        }

        out
    }

    /// Predict the next latent state given current latent and action.
    ///
    /// Uses the CachedQ4 predictor layers for the heavy DiT forward pass.
    pub fn predict_next(&self, z_t: &[f32], action: &[f32]) -> Vec<f32> {
        let hidden = self.config.predictor_hidden;
        let latent = self.config.latent_dim;
        let num_heads = self.config.predictor_heads;
        let inner_dim = self.config.predictor_inner_dim;
        let inter = self.config.predictor_inter;
        let has_proj = !self.input_proj_weight.is_empty();

        // 1. Encode action -> [latent_dim]
        let a_embed = self.encode_action(action);

        // 2. Build input sequence at latent_dim or predictor_hidden
        let seq_len = 3;
        let seq_dim = if has_proj { latent } else { hidden };
        let mut seq = vec![0.0f32; seq_len * seq_dim];
        seq[..seq_dim].copy_from_slice(z_t);
        seq[seq_dim..2 * seq_dim].copy_from_slice(&a_embed);
        // seq[2*seq_dim..3*seq_dim] = zeros (target position to be predicted)

        // 3. Add positional embeddings
        if !self.predictor_pos_embed.is_empty() {
            let pos_len = self.predictor_pos_embed.len().min(seq.len());
            for i in 0..pos_len {
                seq[i] += self.predictor_pos_embed[i];
            }
        }

        // 4. Apply projections if bottleneck architecture
        let (mut seq, conditioning) = if has_proj {
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

        // 5. Run through CachedQ4 predictor layers
        for layer in &self.predictor_layers {
            seq = layer.forward(
                &seq,
                &conditioning,
                seq_len,
                hidden,
                num_heads,
                inner_dim,
                inter,
            );
        }

        // 6. Final norm
        let mut normed = layernorm(&seq, &self.predictor_norm_weight, 1e-6, hidden);
        if !self.predictor_norm_bias.is_empty() {
            for t in 0..seq_len {
                for j in 0..hidden {
                    normed[t * hidden + j] += self.predictor_norm_bias[j];
                }
            }
        }

        // 7. Extract target position (index 2) -> [hidden]
        let target = &normed[2 * hidden..3 * hidden];

        // 8. Project back through pred_proj (f32)
        self.pred_proj.forward(target)
    }

    /// Multi-step rollout: predict a sequence of future latent states.
    pub fn rollout(&self, z_start: &[f32], actions: &[Vec<f32>]) -> Vec<Vec<f32>> {
        let mut states = Vec::with_capacity(actions.len());
        let mut z = z_start.to_vec();
        for action in actions {
            z = self.predict_next(&z, action);
            states.push(z.clone());
        }
        states
    }
}

/// Quantize a LeWorldModel to CachedQ4 (dequant-at-load).
///
/// Converts the predictor's adaLN transformer layers from f32 to Q4_0 then
/// immediately back to f32 for BLAS-speed inference. The encoder, action
/// encoder, and projection heads are copied as-is (f32).
pub fn cached_q4_lewm(model: &LeWorldModel) -> CachedQ4LeWM {
    let cfg = &model.config;
    let hidden = cfg.predictor_hidden;
    let inner_dim = cfg.predictor_inner_dim;
    let inter = cfg.predictor_inter;

    let predictor_layers = model
        .predictor_layers
        .iter()
        .map(|layer| {
            // adaLN modulation: [6*hidden, hidden]
            let adaln_linear = CachedQ4Linear::from_f32(&layer.adaln_weight, 6 * hidden, hidden);
            // Fused QKV: [3*inner_dim, hidden]
            let to_qkv = CachedQ4Linear::from_f32(&layer.to_qkv, 3 * inner_dim, hidden);
            // Output projection: [hidden, inner_dim]
            let attn_out = CachedQ4Linear::from_f32(&layer.attn_out_weight, hidden, inner_dim);
            // MLP up: [inter, hidden]
            let mlp_up = CachedQ4Linear::from_f32(&layer.mlp_up_weight, inter, hidden);
            // MLP down: [hidden, inter]
            let mlp_down = CachedQ4Linear::from_f32(&layer.mlp_down_weight, hidden, inter);

            CachedQ4AdaLNLayer {
                adaln_linear,
                adaln_bias: layer.adaln_bias.to_vec(),
                to_qkv,
                attn_out,
                attn_out_bias: layer.attn_out_bias.to_vec(),
                attn_norm_weight: layer.attn_norm_weight.to_vec(),
                attn_norm_bias: layer.attn_norm_bias.to_vec(),
                mlp_norm_weight: layer.mlp_norm_weight.to_vec(),
                mlp_norm_bias: layer.mlp_norm_bias.to_vec(),
                mlp_up,
                mlp_up_bias: layer.mlp_up_bias.to_vec(),
                mlp_down,
                mlp_down_bias: layer.mlp_down_bias.to_vec(),
            }
        })
        .collect();

    CachedQ4LeWM {
        config: cfg.clone(),
        encoder: clone_vit_encoder(&model.encoder),
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

/// Clone a ViT encoder by re-creating it from its config and copying all weights.
fn clone_vit_encoder(src: &ViTModel) -> ViTModel {
    use crate::weight_loading::AlignedBuffer;

    let mut dst = ViTModel::from_config(&src.config);

    dst.patch_proj = AlignedBuffer::from_slice(&src.patch_proj);
    dst.patch_proj_bias = AlignedBuffer::from_slice(&src.patch_proj_bias);
    dst.cls_token = AlignedBuffer::from_slice(&src.cls_token);
    dst.pos_embed = AlignedBuffer::from_slice(&src.pos_embed);
    dst.final_norm_weight = AlignedBuffer::from_slice(&src.final_norm_weight);
    dst.final_norm_bias = AlignedBuffer::from_slice(&src.final_norm_bias);
    dst.classifier_head = src
        .classifier_head
        .as_ref()
        .map(|b| AlignedBuffer::from_slice(b));
    dst.classifier_bias = src
        .classifier_bias
        .as_ref()
        .map(|b| AlignedBuffer::from_slice(b));
    dst.class_labels = src.class_labels.clone();

    for (d, s) in dst.layers.iter_mut().zip(src.layers.iter()) {
        d.attn_norm_weight = AlignedBuffer::from_slice(&s.attn_norm_weight);
        d.attn_norm_bias = AlignedBuffer::from_slice(&s.attn_norm_bias);
        d.w_q = AlignedBuffer::from_slice(&s.w_q);
        d.q_bias = AlignedBuffer::from_slice(&s.q_bias);
        d.w_k = AlignedBuffer::from_slice(&s.w_k);
        d.k_bias = AlignedBuffer::from_slice(&s.k_bias);
        d.w_v = AlignedBuffer::from_slice(&s.w_v);
        d.v_bias = AlignedBuffer::from_slice(&s.v_bias);
        d.w_o = AlignedBuffer::from_slice(&s.w_o);
        d.o_bias = AlignedBuffer::from_slice(&s.o_bias);
        d.ffn_norm_weight = AlignedBuffer::from_slice(&s.ffn_norm_weight);
        d.ffn_norm_bias = AlignedBuffer::from_slice(&s.ffn_norm_bias);
        d.ffn_up = AlignedBuffer::from_slice(&s.ffn_up);
        d.ffn_up_bias = AlignedBuffer::from_slice(&s.ffn_up_bias);
        d.ffn_down = AlignedBuffer::from_slice(&s.ffn_down);
        d.ffn_down_bias = AlignedBuffer::from_slice(&s.ffn_down_bias);
    }

    dst
}

/// Clone a projection head by copying all layer weights.
fn clone_projection_head(src: &ProjectionHead) -> ProjectionHead {
    use crate::weight_loading::AlignedBuffer;

    let layers = src
        .layers
        .iter()
        .map(|(w, b)| (AlignedBuffer::from_slice(w), AlignedBuffer::from_slice(b)))
        .collect();
    ProjectionHead { layers }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::vision::lewm::LeWMConfig;
    use crate::weight_loading::AlignedBuffer;

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

        // Projector: [h] -> [enc_inter] -> [enc_inter] -> [pred_h]
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
            layer.adaln_weight =
                AlignedBuffer::from_slice(&gen_weights(6 * pred_h * pred_h, s + 1));
            layer.adaln_bias = AlignedBuffer::from_slice(&gen_weights(6 * pred_h, s + 2));
            layer.to_qkv = AlignedBuffer::from_slice(&gen_weights(3 * pred_inner * pred_h, s + 3));
            layer.attn_out_weight =
                AlignedBuffer::from_slice(&gen_weights(pred_h * pred_inner, s + 4));
            layer.attn_out_bias = AlignedBuffer::from_slice(&gen_weights(pred_h, s + 5));
            layer.attn_norm_weight = AlignedBuffer::from_slice(&vec![1.0f32; pred_h]);
            layer.attn_norm_bias = AlignedBuffer::from_slice(&vec![0.0f32; pred_h]);
            layer.mlp_norm_weight = AlignedBuffer::from_slice(&vec![1.0f32; pred_h]);
            layer.mlp_norm_bias = AlignedBuffer::from_slice(&vec![0.0f32; pred_h]);
            layer.mlp_up_weight =
                AlignedBuffer::from_slice(&gen_weights(pred_inter * pred_h, s + 10));
            layer.mlp_up_bias = AlignedBuffer::from_slice(&gen_weights(pred_inter, s + 11));
            layer.mlp_down_weight =
                AlignedBuffer::from_slice(&gen_weights(pred_h * pred_inter, s + 12));
            layer.mlp_down_bias = AlignedBuffer::from_slice(&gen_weights(pred_h, s + 13));
        }

        model
    }

    #[test]
    fn q4_lewm_predict_produces_finite() {
        let cfg = small_config();
        let model = build_test_lewm(&cfg);
        let quantized = quantize_lewm_q4(&model);
        let z = gen_weights(cfg.latent_dim, 42);
        let action = gen_weights(cfg.action_dim, 43);
        let result = quantized.predict_next(&z, &action);
        assert_eq!(result.len(), cfg.latent_dim);
        assert!(
            result.iter().all(|v| v.is_finite()),
            "Q4 predict_next produced non-finite values"
        );
    }

    #[test]
    fn q4_lewm_rollout_correct_length() {
        let cfg = small_config();
        let model = build_test_lewm(&cfg);
        let quantized = quantize_lewm_q4(&model);
        let z = gen_weights(cfg.latent_dim, 42);
        let actions: Vec<Vec<f32>> = (0..5)
            .map(|i| gen_weights(cfg.action_dim, 100 + i))
            .collect();
        let trajectory = quantized.rollout(&z, &actions);
        assert_eq!(trajectory.len(), 5);
        for (i, state) in trajectory.iter().enumerate() {
            assert_eq!(state.len(), cfg.latent_dim);
            assert!(
                state.iter().all(|v| v.is_finite()),
                "State {i} contains non-finite values"
            );
        }
    }

    #[test]
    fn q4_lewm_encode_matches_f32() {
        // Encoder is NOT quantized, output should be identical
        let cfg = small_config();
        let model = build_test_lewm(&cfg);
        let quantized = quantize_lewm_q4(&model);

        let image: Vec<f32> = (0..cfg.image_size * cfg.image_size * cfg.channels)
            .map(|i| (i as f32) / 255.0)
            .collect();

        let f32_latent = model.encode(&image, cfg.image_size, cfg.image_size);
        let q_latent = quantized.encode(&image, cfg.image_size, cfg.image_size);

        let dot: f32 = f32_latent.iter().zip(&q_latent).map(|(a, b)| a * b).sum();
        let norm_a: f32 = f32_latent.iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm_b: f32 = q_latent.iter().map(|x| x * x).sum::<f32>().sqrt();
        let cos_sim = if norm_a > 0.0 && norm_b > 0.0 {
            dot / (norm_a * norm_b)
        } else {
            0.0
        };
        assert!(
            (cos_sim - 1.0).abs() < 1e-5,
            "Encoder output should be identical (not quantized), cos_sim={cos_sim}"
        );
    }

    #[test]
    fn q4_reduces_predictor_memory_vs_f32() {
        let cfg = small_config();
        let model = build_test_lewm(&cfg);
        let quantized = quantize_lewm_q4(&model);

        let f32_predictor_weight_bytes: usize = model
            .predictor_layers
            .iter()
            .map(|l| {
                (l.adaln_weight.len()
                    + l.to_qkv.len()
                    + l.attn_out_weight.len()
                    + l.mlp_up_weight.len()
                    + l.mlp_down_weight.len())
                    * 4 // f32 = 4 bytes
            })
            .sum();

        let q4_predictor_weight_bytes: usize = quantized
            .predictor_layers
            .iter()
            .map(|l| {
                l.adaln_linear.memory_bytes()
                    + l.to_qkv.memory_bytes()
                    + l.attn_out.memory_bytes()
                    + l.mlp_up.memory_bytes()
                    + l.mlp_down.memory_bytes()
            })
            .sum();

        assert!(
            q4_predictor_weight_bytes < f32_predictor_weight_bytes,
            "Q4 weights ({q4_predictor_weight_bytes} bytes) should be smaller than f32 ({f32_predictor_weight_bytes} bytes)"
        );
        let ratio = f32_predictor_weight_bytes as f64 / q4_predictor_weight_bytes as f64;
        assert!(
            ratio > 3.0,
            "Memory reduction ratio {ratio:.2}x is too low (expected >3x for Q4)"
        );
    }

    #[test]
    fn cached_q4_forward_matches_dequant_matmul() {
        let weights: Vec<f32> = (0..256 * 128)
            .map(|i| ((i % 100) as f32 - 50.0) * 0.01)
            .collect();
        let q4 = Q4Linear::from_f32(&weights, 256, 128);
        let cached = CachedQ4Linear::from_q4(&q4);
        let x: Vec<f32> = (0..3 * 128).map(|i| (i as f32) * 0.001).collect();

        // CachedQ4 should produce same result as matmul_t with dequantized weights
        let expected = crate::ops::matmul::matmul_t(&x, &q4.dequantize(), 3, 128, 256);
        let got = cached.forward(&x, 3);
        let max_diff: f32 = expected
            .iter()
            .zip(&got)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(
            max_diff < 1e-5,
            "CachedQ4 forward mismatch: max_diff={max_diff}"
        );
    }

    #[test]
    fn cached_q4_lewm_predict_produces_finite() {
        let cfg = small_config();
        let model = build_test_lewm(&cfg);
        let cached = cached_q4_lewm(&model);
        let z = gen_weights(cfg.latent_dim, 42);
        let action = gen_weights(cfg.action_dim, 43);
        let result = cached.predict_next(&z, &action);
        assert_eq!(result.len(), cfg.latent_dim);
        assert!(result.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn cached_q4_lewm_rollout_correct_length() {
        let cfg = small_config();
        let model = build_test_lewm(&cfg);
        let cached = cached_q4_lewm(&model);
        let z = gen_weights(cfg.latent_dim, 42);
        let actions: Vec<Vec<f32>> = (0..5)
            .map(|i| gen_weights(cfg.action_dim, 100 + i))
            .collect();
        let traj = cached.rollout(&z, &actions);
        assert_eq!(traj.len(), 5);
    }

    #[test]
    fn lq40_rejects_invalid_magic() {
        let result = QuantizedQ4LeWM::from_lq40_bytes(b"XXXX\x00\x00\x00\x00");
        match result {
            Err(e) => assert!(e.contains("Not LQ40"), "unexpected error: {}", e),
            Ok(_) => panic!("expected error for invalid magic"),
        }
    }

    #[test]
    fn lq40_rejects_short_data() {
        let result = QuantizedQ4LeWM::from_lq40_bytes(b"LQ40");
        match result {
            Err(e) => assert!(e.contains("too short"), "unexpected error: {}", e),
            Ok(_) => panic!("expected error for short data"),
        }
    }
}

// ---------------------------------------------------------------------------
// LQ40 binary reader helpers
// ---------------------------------------------------------------------------

pub(crate) fn lq40_read_u32(data: &[u8], off: usize) -> u32 {
    u32::from_le_bytes([data[off], data[off + 1], data[off + 2], data[off + 3]])
}

/// Read a length-prefixed f32 vector.
pub(crate) fn lq40_read_f32(data: &[u8], off: &mut usize) -> Vec<f32> {
    let len = lq40_read_u32(data, *off) as usize;
    *off += 4;
    let mut v = Vec::with_capacity(len);
    for i in 0..len {
        let base = *off + i * 4;
        v.push(f32::from_le_bytes([
            data[base],
            data[base + 1],
            data[base + 2],
            data[base + 3],
        ]));
    }
    *off += len * 4;
    v
}

/// Read a Q4Linear from LQ40 binary (sparse format with bitmap).
pub(crate) fn lq40_read_q4_linear(data: &[u8], off: &mut usize) -> Result<Q4Linear, String> {
    use crate::quantization::Q4Block;

    let out_features = lq40_read_u32(data, *off) as usize;
    *off += 4;
    let in_features = lq40_read_u32(data, *off) as usize;
    *off += 4;
    let total_blocks = lq40_read_u32(data, *off) as usize;
    *off += 4;
    let nonzero_count = lq40_read_u32(data, *off) as usize;
    *off += 4;

    let bitmap_bytes = (total_blocks + 7) / 8;
    let bitmap = &data[*off..*off + bitmap_bytes];
    *off += bitmap_bytes;

    // Read only non-zero blocks, reconstruct full block array
    let mut blocks = Vec::with_capacity(total_blocks);
    let mut nz_idx = 0;
    for i in 0..total_blocks {
        let is_nonzero = (bitmap[i / 8] >> (i % 8)) & 1 == 1;
        if is_nonzero {
            let base = *off + nz_idx * 20;
            let scale =
                f32::from_le_bytes([data[base], data[base + 1], data[base + 2], data[base + 3]]);
            let mut nibbles = [0u8; 16];
            nibbles.copy_from_slice(&data[base + 4..base + 20]);
            blocks.push(Q4Block { scale, nibbles });
            nz_idx += 1;
        } else {
            blocks.push(Q4Block {
                scale: 0.0,
                nibbles: [0u8; 16],
            });
        }
    }
    *off += nonzero_count * 20;

    Ok(Q4Linear {
        blocks,
        out_features,
        in_features,
        packed_zig_cache: std::cell::RefCell::new(None),
    })
}

/// Read a Q4 adaLN predictor layer from LQ40 format.
pub(crate) fn lq40_read_q4_adaln_layer(
    data: &[u8],
    off: &mut usize,
) -> Result<QuantizedQ4AdaLNLayer, String> {
    let adaln_linear = lq40_read_q4_linear(data, off)?;
    let adaln_bias = lq40_read_f32(data, off);
    let to_qkv = lq40_read_q4_linear(data, off)?;
    let attn_out = lq40_read_q4_linear(data, off)?;
    let attn_out_bias = lq40_read_f32(data, off);
    let attn_norm_weight = lq40_read_f32(data, off);
    let attn_norm_bias = lq40_read_f32(data, off);
    let mlp_norm_weight = lq40_read_f32(data, off);
    let mlp_norm_bias = lq40_read_f32(data, off);
    let mlp_up = lq40_read_q4_linear(data, off)?;
    let mlp_up_bias = lq40_read_f32(data, off);
    let mlp_down = lq40_read_q4_linear(data, off)?;
    let mlp_down_bias = lq40_read_f32(data, off);

    Ok(QuantizedQ4AdaLNLayer {
        adaln_linear,
        adaln_bias,
        to_qkv,
        attn_out,
        attn_out_bias,
        attn_norm_weight,
        attn_norm_bias,
        mlp_norm_weight,
        mlp_norm_bias,
        mlp_up,
        mlp_up_bias,
        mlp_down,
        mlp_down_bias,
    })
}

/// Read a ProjectionHead: [u32 num_layers] then (weight, bias) f32 pairs.
pub(crate) fn lq40_read_projection_head(data: &[u8], off: &mut usize) -> ProjectionHead {
    use crate::weight_loading::AlignedBuffer;
    let n = lq40_read_u32(data, *off) as usize;
    *off += 4;
    let mut layers = Vec::with_capacity(n);
    for _ in 0..n {
        let w = lq40_read_f32(data, off);
        let b = lq40_read_f32(data, off);
        layers.push((AlignedBuffer::from_slice(&w), AlignedBuffer::from_slice(&b)));
    }
    ProjectionHead { layers }
}

/// Parse LeWMConfig from LQ40 JSON header.
pub(crate) fn lq40_config_from_json(v: &serde_json::Value) -> Result<LeWMConfig, String> {
    let get_usize = |key: &str| -> Result<usize, String> {
        v.get(key)
            .and_then(|v| v.as_u64())
            .map(|v| v as usize)
            .ok_or_else(|| format!("Missing '{}' in LQ40 config", key))
    };

    Ok(LeWMConfig {
        image_size: get_usize("image_size")?,
        patch_size: get_usize("patch_size")?,
        encoder_hidden: get_usize("encoder_hidden")?,
        encoder_layers: get_usize("encoder_layers")?,
        encoder_heads: get_usize("encoder_heads")?,
        encoder_inter: get_usize("encoder_inter")?,
        predictor_hidden: get_usize("predictor_hidden")?,
        predictor_layers: get_usize("predictor_layers")?,
        predictor_heads: get_usize("predictor_heads")?,
        predictor_inner_dim: get_usize("predictor_inner_dim")?,
        predictor_inter: get_usize("predictor_inter")?,
        action_dim: get_usize("action_dim")?,
        latent_dim: get_usize("latent_dim")?,
        channels: get_usize("channels")?,
    })
}
