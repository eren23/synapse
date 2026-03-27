//! Q4_0 (4-bit) quantization for the LeWorldModel (LeWM).
//!
//! Q4_0 block format: 32 elements per block, each block stores 1 f32 scale +
//! 16 bytes of nibble pairs = 20 bytes per block. This gives ~6.4x compression
//! vs f32. Predictor weights shrink from ~43MB (f32) to ~7MB (Q4), fitting in
//! ESP32-P4's 32MB PSRAM.

use crate::model::lewm::{LeWMConfig, LeWorldModel, ProjectionHead};
use crate::model::vit::ViTModel;
use crate::ops::activation::gelu;
use crate::ops::attention::bidirectional_attention;
use crate::ops::norm::layernorm;

/// A single Q4_0 block: 32 elements quantized to 4-bit with one f32 scale.
///
/// Each nibble pair packs two signed 4-bit values offset by +8 into a byte:
///   low nibble  = (v0 + 8), range [0, 15]
///   high nibble = (v1 + 8), range [0, 15]
///
/// Dequantization: value = (nibble - 8) * scale
#[repr(C)]
pub struct Q4Block {
    pub scale: f32,
    pub nibbles: [u8; 16], // 32 values packed as nibble pairs
}

/// A linear layer with Q4_0 quantized weights.
///
/// Weights are stored as a flat array of [`Q4Block`]s in row-major order:
/// `blocks[row * blocks_per_row + b]` covers columns `[b*32 .. (b+1)*32)` of row `row`.
pub struct Q4Linear {
    pub blocks: Vec<Q4Block>,
    pub out_features: usize,
    pub in_features: usize,
}

impl Q4Linear {
    /// Quantize an f32 weight matrix `[out_features, in_features]` to Q4_0 blocks.
    ///
    /// `in_features` is padded to a multiple of 32 internally (zero-padded columns).
    pub fn from_f32(weights: &[f32], out_features: usize, in_features: usize) -> Self {
        assert_eq!(weights.len(), out_features * in_features);
        let padded_k = (in_features + 31) / 32 * 32;
        let blocks_per_row = padded_k / 32;
        let mut blocks = Vec::with_capacity(out_features * blocks_per_row);

        for row in 0..out_features {
            for b in 0..blocks_per_row {
                let mut vals = [0.0f32; 32];
                for i in 0..32 {
                    let col = b * 32 + i;
                    if col < in_features {
                        vals[i] = weights[row * in_features + col];
                    }
                }
                let max_abs = vals.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
                let scale = if max_abs == 0.0 { 0.0 } else { max_abs / 7.0 };
                let inv_scale = if scale == 0.0 { 0.0 } else { 1.0 / scale };

                let mut nibbles = [0u8; 16];
                for i in 0..16 {
                    let v0 = (vals[2 * i] * inv_scale).round().clamp(-8.0, 7.0) as i8;
                    let v1 = (vals[2 * i + 1] * inv_scale).round().clamp(-8.0, 7.0) as i8;
                    // Pack: low nibble = v0 + 8, high nibble = v1 + 8
                    nibbles[i] = ((v0 + 8) as u8) | (((v1 + 8) as u8) << 4);
                }
                blocks.push(Q4Block { scale, nibbles });
            }
        }

        Q4Linear {
            blocks,
            out_features,
            in_features,
        }
    }

    /// Forward pass: `x [m, in_features]` -> `[m, out_features]`.
    ///
    /// Dequantizes weights on-the-fly and accumulates dot products. This is a
    /// pure-Rust GEMV suitable for embedded targets without SIMD requirements.
    pub fn forward(&self, x: &[f32], m: usize) -> Vec<f32> {
        let k = self.in_features;
        let n = self.out_features;
        let padded_k = (k + 31) / 32 * 32;
        let blocks_per_row = padded_k / 32;
        let mut out = vec![0.0f32; m * n];

        for i in 0..m {
            for j in 0..n {
                let mut sum = 0.0f32;
                for b in 0..blocks_per_row {
                    let block = &self.blocks[j * blocks_per_row + b];
                    let scale = block.scale;
                    for ni in 0..16 {
                        let byte = block.nibbles[ni];
                        let v0 = ((byte & 0x0F) as i8 - 8) as f32 * scale;
                        let v1 = ((byte >> 4) as i8 - 8) as f32 * scale;
                        let col0 = b * 32 + 2 * ni;
                        let col1 = col0 + 1;
                        if col0 < k {
                            sum += x[i * k + col0] * v0;
                        }
                        if col1 < k {
                            sum += x[i * k + col1] * v1;
                        }
                    }
                }
                out[i * n + j] = sum;
            }
        }
        out
    }

    /// Memory in bytes for the Q4_0 block storage.
    pub fn memory_bytes(&self) -> usize {
        self.blocks.len() * std::mem::size_of::<Q4Block>()
    }

    /// Dequantize all Q4 blocks back to f32.
    /// Returns [out_features, in_features] row-major weights.
    pub fn dequantize(&self) -> Vec<f32> {
        let mut weights = vec![0.0f32; self.out_features * self.in_features];
        let padded_k = (self.in_features + 31) / 32 * 32;
        let blocks_per_row = padded_k / 32;
        for row in 0..self.out_features {
            for b in 0..blocks_per_row {
                let block = &self.blocks[row * blocks_per_row + b];
                for ni in 0..16 {
                    let byte = block.nibbles[ni];
                    let v0 = ((byte & 0x0F) as i8 - 8) as f32 * block.scale;
                    let v1 = ((byte >> 4) as i8 - 8) as f32 * block.scale;
                    let col0 = b * 32 + 2 * ni;
                    let col1 = col0 + 1;
                    if col0 < self.in_features {
                        weights[row * self.in_features + col0] = v0;
                    }
                    if col1 < self.in_features {
                        weights[row * self.in_features + col1] = v1;
                    }
                }
            }
        }
        weights
    }

    /// Create an empty (zero-sized) Q4Linear, used for absent weights.
    pub fn empty() -> Self {
        Q4Linear {
            blocks: Vec::new(),
            out_features: 0,
            in_features: 0,
        }
    }
}

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
        let hidden = self.config.predictor_hidden;

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
        let num_heads = self.config.predictor_heads;
        let inner_dim = self.config.predictor_inner_dim;
        let inter = self.config.predictor_inter;

        // 1. Encode action -> [hidden]
        let a_embed = self.encode_action(action);

        // 2. Build input sequence: [z_t, a_embed, target_token] each [hidden]
        let seq_len = 3;
        let mut seq = vec![0.0f32; seq_len * hidden];
        seq[..hidden].copy_from_slice(z_t);
        seq[hidden..2 * hidden].copy_from_slice(&a_embed);
        // seq[2*hidden..3*hidden] = zeros (target position to be predicted)

        // 3. Add positional embeddings
        if !self.predictor_pos_embed.is_empty() {
            let pos_len = self.predictor_pos_embed.len().min(seq.len());
            for i in 0..pos_len {
                seq[i] += self.predictor_pos_embed[i];
            }
        }

        // 4. Run through Q4-quantized predictor layers
        for layer in &self.predictor_layers {
            seq = layer.forward(&seq, &a_embed, seq_len, hidden, num_heads, inner_dim, inter);
        }

        // 5. Final norm
        let mut normed = layernorm(&seq, &self.predictor_norm_weight, 1e-6, hidden);
        if !self.predictor_norm_bias.is_empty() {
            for t in 0..seq_len {
                for j in 0..hidden {
                    normed[t * hidden + j] += self.predictor_norm_bias[j];
                }
            }
        }

        // 6. Extract target position (index 2) -> [hidden]
        let target = &normed[2 * hidden..3 * hidden];

        // 7. Project back through pred_proj (f32)
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
        let hidden = self.config.predictor_hidden;

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
        let num_heads = self.config.predictor_heads;
        let inner_dim = self.config.predictor_inner_dim;
        let inter = self.config.predictor_inter;

        // 1. Encode action -> [hidden]
        let a_embed = self.encode_action(action);

        // 2. Build input sequence: [z_t, a_embed, target_token] each [hidden]
        let seq_len = 3;
        let mut seq = vec![0.0f32; seq_len * hidden];
        seq[..hidden].copy_from_slice(z_t);
        seq[hidden..2 * hidden].copy_from_slice(&a_embed);
        // seq[2*hidden..3*hidden] = zeros (target position to be predicted)

        // 3. Add positional embeddings
        if !self.predictor_pos_embed.is_empty() {
            let pos_len = self.predictor_pos_embed.len().min(seq.len());
            for i in 0..pos_len {
                seq[i] += self.predictor_pos_embed[i];
            }
        }

        // 4. Run through CachedQ4 predictor layers
        for layer in &self.predictor_layers {
            seq = layer.forward(&seq, &a_embed, seq_len, hidden, num_heads, inner_dim, inter);
        }

        // 5. Final norm
        let mut normed = layernorm(&seq, &self.predictor_norm_weight, 1e-6, hidden);
        if !self.predictor_norm_bias.is_empty() {
            for t in 0..seq_len {
                for j in 0..hidden {
                    normed[t * hidden + j] += self.predictor_norm_bias[j];
                }
            }
        }

        // 6. Extract target position (index 2) -> [hidden]
        let target = &normed[2 * hidden..3 * hidden];

        // 7. Project back through pred_proj (f32)
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
            let adaln_linear =
                CachedQ4Linear::from_f32(&layer.adaln_weight, 6 * hidden, hidden);
            // Fused QKV: [3*inner_dim, hidden]
            let to_qkv = CachedQ4Linear::from_f32(&layer.to_qkv, 3 * inner_dim, hidden);
            // Output projection: [hidden, inner_dim]
            let attn_out =
                CachedQ4Linear::from_f32(&layer.attn_out_weight, hidden, inner_dim);
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
    use crate::model::lewm::LeWMConfig;
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
        model.action_conv_weight =
            AlignedBuffer::from_slice(&gen_weights(act_dim * act_dim, 600));
        model.action_conv_bias = AlignedBuffer::from_slice(&gen_weights(act_dim, 601));
        model.action_mlp1_weight =
            AlignedBuffer::from_slice(&gen_weights(enc_inter * act_dim, 602));
        model.action_mlp1_bias = AlignedBuffer::from_slice(&gen_weights(enc_inter, 603));
        model.action_mlp2_weight =
            AlignedBuffer::from_slice(&gen_weights(pred_h * enc_inter, 604));
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
            layer.to_qkv =
                AlignedBuffer::from_slice(&gen_weights(3 * pred_inner * pred_h, s + 3));
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
    fn q4_linear_from_f32_roundtrip() {
        let weights: Vec<f32> = vec![
            1.0, -1.0, 0.5, -0.5, 0.25, -0.25, 0.1, -0.1, 0.0, 0.3, -0.3, 0.7, -0.7, 0.9,
            -0.9, 0.4, -0.4, 0.6, -0.6, 0.8, -0.8, 0.2, -0.2, 0.15, -0.15, 0.35, -0.35, 0.45,
            -0.45, 0.55, -0.55, 0.65,
        ]; // [1, 32]
        let q = Q4Linear::from_f32(&weights, 1, 32);
        assert_eq!(q.blocks.len(), 1);
        assert!(q.blocks[0].scale > 0.0);
    }

    #[test]
    fn q4_linear_forward_produces_finite() {
        let weights: Vec<f32> = (0..64).map(|i| (i as f32 - 32.0) * 0.01).collect();
        let q = Q4Linear::from_f32(&weights, 2, 32);
        let x = vec![1.0f32; 32];
        let out = q.forward(&x, 1);
        assert_eq!(out.len(), 2);
        assert!(out.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn q4_linear_memory_smaller_than_f32() {
        let weights: Vec<f32> = vec![0.1; 1024 * 512]; // [1024, 512]
        let q = Q4Linear::from_f32(&weights, 1024, 512);
        let f32_bytes = 1024 * 512 * 4;
        assert!(
            q.memory_bytes() < f32_bytes / 3,
            "Q4 should be >3x smaller than f32"
        );
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
        let actions: Vec<Vec<f32>> = (0..5).map(|i| gen_weights(cfg.action_dim, 100 + i)).collect();
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
    fn q4_reduces_memory_vs_int8() {
        // Use a realistically-sized layer where Q4 wins over INT8.
        // At small dimensions (e.g. in_features=16), Q4 block overhead dominates
        // because padding to 32 wastes nibble slots and the f32 scale costs 4 bytes
        // per block. With in_features >= 64, Q4 consistently beats INT8.
        let out = 128;
        let inf = 256;
        let weights: Vec<f32> = gen_weights(out * inf, 999);
        let q4 = Q4Linear::from_f32(&weights, out, inf);
        let q4_bytes = q4.memory_bytes();
        let int8_would_be = out * inf; // 1 byte per weight for INT8
        assert!(
            q4_bytes < int8_would_be,
            "Q4 ({q4_bytes} bytes) should use less memory than INT8 ({int8_would_be} bytes)"
        );
        // Q4 with 256-wide rows: 256/32 = 8 blocks/row, 8*20 = 160 bytes/row
        // INT8: 256 bytes/row. So Q4 is ~1.6x smaller.
        let ratio = int8_would_be as f64 / q4_bytes as f64;
        assert!(
            ratio > 1.4,
            "Q4 should be at least 1.4x smaller than INT8, got {ratio:.2}x"
        );
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
    fn q4_dequantize_roundtrip_accuracy() {
        // Quantize f32 → Q4 → dequant → check max error is within Q4 tolerance
        let weights: Vec<f32> = (0..1024).map(|i| (i as f32 - 512.0) * 0.01).collect();
        let q4 = Q4Linear::from_f32(&weights, 32, 32);
        let dequant = q4.dequantize();
        let max_err: f32 = weights
            .iter()
            .zip(&dequant)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        // Q4 with 4-bit resolution: max error should be < scale/2 ≈ max_abs/7
        assert!(max_err < 1.0, "Q4 roundtrip error too large: {max_err}");
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
        let actions: Vec<Vec<f32>> = (0..5).map(|i| gen_weights(cfg.action_dim, 100 + i)).collect();
        let traj = cached.rollout(&z, &actions);
        assert_eq!(traj.len(), 5);
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
}
