//! Synapse WASM: Real LeWorldModel inference in the browser.
//!
//! Loads the actual LeWM checkpoint (69MB f32 binary) and runs:
//! - ViT encoder (12 layers, 192 hidden, 3 heads)
//! - DiT predictor with adaLN modulation (6 layers, 16 heads, 1024 inner_dim)
//! - Action encoder (conv1d + MLP)
//! - Projector and pred_proj MLPs
//!
//! All ops in PURE RUST — no Zig FFI, fully WASM-compatible.

use serde_json::{json, Value};
use std::collections::HashMap;
use wasm_bindgen::prelude::*;

use synapse_inference::ops::pure_rust_ops::{
    matmul_t, gelu, silu, softmax, bidirectional_attention,
    layernorm_with_bias, layernorm as layernorm_no_bias_inner,
    quantize_per_channel_int8, qgemm_int8,
};

// ── Constants ────────────────────────────────────────────────────────

const STATUS_MANIFEST_JSON: &str = include_str!("../../status/public_status.json");

const HIDDEN: usize = 192;
const ENCODER_LAYERS: usize = 12;
const ENCODER_HEADS: usize = 3;
const ENCODER_INTER: usize = 768;
const PREDICTOR_LAYERS: usize = 6;
const PREDICTOR_HEADS: usize = 16;
const PREDICTOR_INNER_DIM: usize = 1024;
const PREDICTOR_HEAD_DIM: usize = 64; // 1024 / 16
const PREDICTOR_INTER: usize = 2048;
const IMAGE_SIZE: usize = 224;
const PATCH_SIZE: usize = 14;
const CHANNELS: usize = 3;
const NUM_PATCHES: usize = (IMAGE_SIZE / PATCH_SIZE) * (IMAGE_SIZE / PATCH_SIZE); // 256
const SEQ_LEN_VIT: usize = NUM_PATCHES + 1; // 257 (with CLS)
const PATCH_DIM: usize = PATCH_SIZE * PATCH_SIZE * CHANNELS; // 588
const ACTION_DIM: usize = 10;
const LATENT_DIM: usize = HIDDEN; // 192

// ── Local helpers (thin wrappers) ───────────────────────────────────

/// Layer normalization with weight and bias using hardcoded eps=1e-6.
fn layernorm(x: &[f32], weight: &[f32], bias: &[f32], n: usize) -> Vec<f32> {
    layernorm_with_bias(x, weight, bias, 1e-6, n)
}

/// Layer normalization with weight only (no bias) using hardcoded eps=1e-6.
fn layernorm_no_bias(x: &[f32], weight: &[f32], n: usize) -> Vec<f32> {
    layernorm_no_bias_inner(x, weight, 1e-6, n)
}

/// Add per-column bias to a row-major matrix [m, n] in place.
fn add_bias(x: &mut [f32], bias: &[f32], m: usize, n: usize) {
    for row in 0..m {
        for col in 0..n {
            x[row * n + col] += bias[col];
        }
    }
}

/// Extract 14x14 patches from a 224x224x3 image (HWC layout) in CHW order,
/// then project via matmul to get [num_patches, embed_dim].
fn patch_embed(
    image: &[f32],
    height: usize,
    width: usize,
    projection: &[f32],
    proj_bias: &[f32],
) -> Vec<f32> {
    let patches_h = height / PATCH_SIZE;
    let patches_w = width / PATCH_SIZE;
    let num_patches = patches_h * patches_w;

    // Extract patches in CHW order: index = c * P * P + py * P + px
    // Image is stored HWC: image[(y * W + x) * C + c]
    let mut patches = vec![0.0f32; num_patches * PATCH_DIM];
    for ph in 0..patches_h {
        for pw in 0..patches_w {
            let patch_idx = ph * patches_w + pw;
            for c in 0..CHANNELS {
                for py in 0..PATCH_SIZE {
                    for px in 0..PATCH_SIZE {
                        let img_y = ph * PATCH_SIZE + py;
                        let img_x = pw * PATCH_SIZE + px;
                        let img_idx = (img_y * width + img_x) * CHANNELS + c;
                        let patch_pixel = c * PATCH_SIZE * PATCH_SIZE + py * PATCH_SIZE + px;
                        patches[patch_idx * PATCH_DIM + patch_pixel] = image[img_idx];
                    }
                }
            }
        }
    }

    // Project: [num_patches, PATCH_DIM] @ [HIDDEN, PATCH_DIM]^T = [num_patches, HIDDEN]
    let mut result = matmul_t(&patches, projection, num_patches, PATCH_DIM, HIDDEN);

    // Add projection bias
    if !proj_bias.is_empty() {
        add_bias(&mut result, proj_bias, num_patches, HIDDEN);
    }

    result
}

#[wasm_bindgen]
pub fn capability_report_json() -> String {
    let manifest: Value =
        serde_json::from_str(STATUS_MANIFEST_JSON).expect("status manifest should parse");
    let profile = manifest["runtime_profiles"]
        .as_array()
        .and_then(|profiles| {
            profiles
                .iter()
                .find(|profile| profile["id"].as_str() == Some("wasm_portable"))
        })
        .cloned()
        .expect("wasm runtime profile should exist in status manifest");

    serde_json::to_string_pretty(&json!({
        "manifest_version": manifest["manifest_version"],
        "last_verified": manifest["last_verified"],
        "runtime_profile": "wasm_portable",
        "target": "wasm32-unknown-unknown",
        "summary": manifest["positioning"]["wasm_runtime"],
        "backends": profile["backends"],
        "quantization": profile["quantization"],
        "loaded_model": Value::Null,
        "model_families": manifest["model_families"],
        "features": manifest["features"],
        "artifact_budgets": manifest["artifact_budgets"],
        "native_kernel": Value::Null,
    }))
    .expect("capability report should serialize")
}

// ── Weight structures ───────────────────────────────────────────────

struct ViTLayerWeights {
    q_w: Vec<f32>,
    q_b: Vec<f32>,
    k_w: Vec<f32>,
    k_b: Vec<f32>,
    v_w: Vec<f32>,
    v_b: Vec<f32>,
    o_w: Vec<f32>,
    o_b: Vec<f32>,
    ffn_up_w: Vec<f32>,
    ffn_up_b: Vec<f32>,
    ffn_down_w: Vec<f32>,
    ffn_down_b: Vec<f32>,
    norm1_w: Vec<f32>,
    norm1_b: Vec<f32>,
    norm2_w: Vec<f32>,
    norm2_b: Vec<f32>,
}

struct AdaLNWeights {
    adaln_w: Vec<f32>,     // [1152, 192]
    adaln_b: Vec<f32>,     // [1152]
    to_qkv_w: Vec<f32>,    // [3072, 192]
    attn_out_w: Vec<f32>,  // [192, 1024]
    attn_out_b: Vec<f32>,  // [192]
    attn_norm_w: Vec<f32>, // [192]
    attn_norm_b: Vec<f32>, // [192]
    mlp_norm_w: Vec<f32>,  // [192]
    mlp_norm_b: Vec<f32>,  // [192]
    mlp_up_w: Vec<f32>,    // [2048, 192]
    mlp_up_b: Vec<f32>,    // [2048]
    mlp_down_w: Vec<f32>,  // [192, 2048]
    mlp_down_b: Vec<f32>,  // [192]
}

// ── RealLeWM ────────────────────────────────────────────────────────

#[wasm_bindgen]
pub struct RealLeWM {
    // Encoder (ViT)
    encoder_patch_proj: Vec<f32>, // [HIDDEN, PATCH_DIM] = [192, 588]
    encoder_patch_proj_bias: Vec<f32>, // [192]
    encoder_cls_token: Vec<f32>,  // [192]
    encoder_pos_embed: Vec<f32>,  // [257, 192]
    encoder_layers: Vec<ViTLayerWeights>,
    encoder_norm_w: Vec<f32>, // [192]
    encoder_norm_b: Vec<f32>, // [192]

    // Predictor (6 adaLN layers)
    predictor_pos_embed: Vec<f32>, // [3, 192]
    predictor_layers: Vec<AdaLNWeights>,
    predictor_norm_w: Vec<f32>, // [192]
    predictor_norm_b: Vec<f32>, // [192]

    // Action encoder
    action_conv_w: Vec<f32>, // [10, 10, 1] = [100] (1D conv)
    action_conv_b: Vec<f32>, // [10]
    action_mlp1_w: Vec<f32>, // [768, 10]
    action_mlp1_b: Vec<f32>, // [768]
    action_mlp2_w: Vec<f32>, // [192, 768]
    action_mlp2_b: Vec<f32>, // [192]

    // Projector (encoder -> predictor space): 3 linear layers
    // net.0: [2048, 192], net.1: BatchNorm(2048), net.3: [192, 2048]
    projector_layers: Vec<(Vec<f32>, Vec<f32>)>,
    // BatchNorm params for projector
    projector_bn_weight: Vec<f32>, // [2048]
    projector_bn_bias: Vec<f32>,   // [2048]
    projector_bn_mean: Vec<f32>,   // [2048]
    projector_bn_var: Vec<f32>,    // [2048]

    // Pred_proj (predictor -> output space): 3 linear layers
    // net.0: [2048, 192], net.1: BatchNorm(2048), net.3: [192, 2048]
    pred_proj_layers: Vec<(Vec<f32>, Vec<f32>)>,
    // BatchNorm params for pred_proj
    pred_proj_bn_weight: Vec<f32>,
    pred_proj_bn_bias: Vec<f32>,
    pred_proj_bn_mean: Vec<f32>,
    pred_proj_bn_var: Vec<f32>,
}

// ── JSON header types ───────────────────────────────────────────────

#[derive(serde::Deserialize)]
struct TensorInfo {
    #[allow(dead_code)]
    shape: Vec<usize>,
    offset: usize,
    len: usize,
}

// ── Implementation ──────────────────────────────────────────────────

impl RealLeWM {
    /// Extract f32 slice from data blob at given offset/len.
    fn extract_f32(data: &[u8], data_start: usize, info: &TensorInfo) -> Vec<f32> {
        let byte_offset = data_start + info.offset * 4;
        let byte_len = info.len * 4;
        let bytes = &data[byte_offset..byte_offset + byte_len];
        bytes
            .chunks_exact(4)
            .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
            .collect()
    }

    /// Get a tensor from the header map, returning an error if missing.
    fn get_tensor(
        header: &HashMap<String, TensorInfo>,
        data: &[u8],
        data_start: usize,
        name: &str,
    ) -> Result<Vec<f32>, JsError> {
        let info = header
            .get(name)
            .ok_or_else(|| JsError::new(&format!("Missing tensor: {}", name)))?;
        Ok(Self::extract_f32(data, data_start, info))
    }

    /// Get a tensor, returning empty vec if not found (for optional tensors).
    fn get_tensor_opt(
        header: &HashMap<String, TensorInfo>,
        data: &[u8],
        data_start: usize,
        name: &str,
    ) -> Vec<f32> {
        header
            .get(name)
            .map(|info| Self::extract_f32(data, data_start, info))
            .unwrap_or_default()
    }

    // ── ViT Encoder Forward ─────────────────────────────────────────

    fn vit_layer_forward(&self, x: &[f32], layer: &ViTLayerWeights, seq_len: usize) -> Vec<f32> {
        let h = HIDDEN;
        let num_heads = ENCODER_HEADS;
        let head_dim = h / num_heads; // 64
        let q_dim = num_heads * head_dim;
        let inter = ENCODER_INTER;

        // 1. Pre-attention LayerNorm
        let normed = layernorm(x, &layer.norm1_w, &layer.norm1_b, h);

        // 2. Q/K/V projections
        let mut q = matmul_t(&normed, &layer.q_w, seq_len, h, q_dim);
        add_bias(&mut q, &layer.q_b, seq_len, q_dim);
        let mut k = matmul_t(&normed, &layer.k_w, seq_len, h, q_dim);
        add_bias(&mut k, &layer.k_b, seq_len, q_dim);
        let mut v = matmul_t(&normed, &layer.v_w, seq_len, h, q_dim);
        add_bias(&mut v, &layer.v_b, seq_len, q_dim);

        // 3. Bidirectional attention
        let attn_out = bidirectional_attention(&q, &k, &v, seq_len, num_heads, head_dim);

        // 4. Output projection
        let mut proj = matmul_t(&attn_out, &layer.o_w, seq_len, q_dim, h);
        add_bias(&mut proj, &layer.o_b, seq_len, h);

        // 5. Residual connection
        let mut residual = vec![0.0f32; seq_len * h];
        for i in 0..seq_len * h {
            residual[i] = x[i] + proj[i];
        }

        // 6. Pre-FFN LayerNorm
        let normed2 = layernorm(&residual, &layer.norm2_w, &layer.norm2_b, h);

        // 7. FFN: up -> GELU -> down
        let mut up = matmul_t(&normed2, &layer.ffn_up_w, seq_len, h, inter);
        add_bias(&mut up, &layer.ffn_up_b, seq_len, inter);
        for v in up.iter_mut() {
            *v = gelu(*v);
        }
        let mut down = matmul_t(&up, &layer.ffn_down_w, seq_len, inter, h);
        add_bias(&mut down, &layer.ffn_down_b, seq_len, h);

        // 8. Residual connection
        for i in 0..seq_len * h {
            residual[i] += down[i];
        }

        residual
    }

    fn encode_image_inner(&self, pixels: &[f32], height: usize, width: usize) -> Vec<f32> {
        // 1. Patch embedding: image -> [num_patches, HIDDEN]
        let patch_embeddings = patch_embed(
            pixels,
            height,
            width,
            &self.encoder_patch_proj,
            &self.encoder_patch_proj_bias,
        );

        let seq_len = SEQ_LEN_VIT; // 257

        // 2. Prepend CLS token: [seq_len, HIDDEN]
        let mut x = vec![0.0f32; seq_len * HIDDEN];
        x[..HIDDEN].copy_from_slice(&self.encoder_cls_token);
        x[HIDDEN..].copy_from_slice(&patch_embeddings);

        // 3. Add positional embeddings
        let pos_len = self.encoder_pos_embed.len().min(x.len());
        for i in 0..pos_len {
            x[i] += self.encoder_pos_embed[i];
        }

        // 4. Encoder layers
        for layer in &self.encoder_layers {
            x = self.vit_layer_forward(&x, layer, seq_len);
        }

        // 5. Final norm on CLS token (position 0)
        let cls_hidden = &x[..HIDDEN];
        let embeddings = layernorm(
            cls_hidden,
            &self.encoder_norm_w,
            &self.encoder_norm_b,
            HIDDEN,
        );

        embeddings
    }

    // ── Action Encoder ──────────────────────────────────────────────

    fn encode_action(&self, action: &[f32]) -> Vec<f32> {
        // 1. 1D conv with kernel_size=1: equivalent to [10, 10] matmul
        let conv_out = if !self.action_conv_w.is_empty() {
            let mut out = matmul_t(action, &self.action_conv_w, 1, ACTION_DIM, ACTION_DIM);
            if !self.action_conv_b.is_empty() {
                for j in 0..ACTION_DIM {
                    out[j] += self.action_conv_b[j];
                }
            }
            out
        } else {
            action.to_vec()
        };

        // 2. MLP: [10] -> [768] (GELU) -> [192]
        let inter = if !self.action_mlp1_w.is_empty() {
            self.action_mlp1_w.len() / ACTION_DIM
        } else {
            ENCODER_INTER
        };

        let mut h1 = matmul_t(&conv_out, &self.action_mlp1_w, 1, ACTION_DIM, inter);
        if !self.action_mlp1_b.is_empty() {
            for j in 0..inter {
                h1[j] += self.action_mlp1_b[j];
            }
        }
        for v in h1.iter_mut() {
            *v = gelu(*v);
        }

        let mut out = matmul_t(&h1, &self.action_mlp2_w, 1, inter, HIDDEN);
        if !self.action_mlp2_b.is_empty() {
            for j in 0..HIDDEN {
                out[j] += self.action_mlp2_b[j];
            }
        }

        out
    }

    // ── Projector / Pred_proj Forward ───────────────────────────────

    /// Forward through projection MLP with BatchNorm.
    /// Architecture: Linear -> BatchNorm -> GELU -> Linear
    fn projection_forward(
        x: &[f32],
        layers: &[(Vec<f32>, Vec<f32>)],
        bn_weight: &[f32],
        bn_bias: &[f32],
        bn_mean: &[f32],
        bn_var: &[f32],
    ) -> Vec<f32> {
        // Layer 0: Linear [192] -> [2048]
        let (ref w0, ref b0) = layers[0];
        let in_dim = x.len();
        let out_dim = w0.len() / in_dim;
        let mut h = matmul_t(x, w0, 1, in_dim, out_dim);
        for j in 0..out_dim {
            h[j] += b0[j];
        }

        // BatchNorm1d (in eval mode: (x - mean) / sqrt(var + eps) * weight + bias)
        if !bn_weight.is_empty() {
            for j in 0..out_dim {
                let normed = (h[j] - bn_mean[j]) / (bn_var[j] + 1e-5).sqrt();
                h[j] = normed * bn_weight[j] + bn_bias[j];
            }
        }

        // GELU activation
        for v in h.iter_mut() {
            *v = gelu(*v);
        }

        // Layer 1: Linear [2048] -> [192]
        let (ref w1, ref b1) = layers[1];
        let inter = h.len();
        let final_dim = w1.len() / inter;
        let mut out = matmul_t(&h, w1, 1, inter, final_dim);
        for j in 0..final_dim {
            out[j] += b1[j];
        }

        out
    }

    // ── adaLN Predictor Layer Forward ───────────────────────────────

    fn adaln_layer_forward(
        &self,
        x: &[f32],
        conditioning: &[f32],
        layer: &AdaLNWeights,
        seq_len: usize,
    ) -> Vec<f32> {
        let h = HIDDEN;
        let num_heads = PREDICTOR_HEADS;
        let inner_dim = PREDICTOR_INNER_DIM;
        let head_dim = PREDICTOR_HEAD_DIM;
        let inter = PREDICTOR_INTER;
        let mod_dim = 6 * h;

        // 1. Compute adaLN modulation: conditioning [h] -> mod_vec [6*h]
        let mut mod_vec = matmul_t(conditioning, &layer.adaln_w, 1, h, mod_dim);
        for j in 0..mod_dim {
            mod_vec[j] += layer.adaln_b[j];
        }
        // Split into 6 vectors: scale1, shift1, gate1, scale2, shift2, gate2
        let scale1 = &mod_vec[0..h];
        let shift1 = &mod_vec[h..2 * h];
        let gate1 = &mod_vec[2 * h..3 * h];
        let scale2 = &mod_vec[3 * h..4 * h];
        let shift2 = &mod_vec[4 * h..5 * h];
        let gate2 = &mod_vec[5 * h..6 * h];

        let mut residual = x.to_vec();

        // 2. Pre-attention: LayerNorm + modulate
        let normed = layernorm_no_bias(x, &layer.attn_norm_w, h);
        // Add bias from attn_norm_b, then apply adaLN modulation
        let mut modulated = vec![0.0f32; seq_len * h];
        for t in 0..seq_len {
            for j in 0..h {
                let idx = t * h + j;
                let val = normed[idx] + layer.attn_norm_b[j];
                modulated[idx] = val * (1.0 + scale1[j]) + shift1[j];
            }
        }

        // 3. Fused QKV: [seq_len, h] -> [seq_len, 3*inner_dim]
        let qkv = matmul_t(&modulated, &layer.to_qkv_w, seq_len, h, 3 * inner_dim);

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

        // 4. Bidirectional attention
        let attn_out = bidirectional_attention(&q, &k, &v, seq_len, num_heads, head_dim);

        // 5. Output projection: [seq_len, inner_dim] -> [seq_len, h]
        let mut proj = matmul_t(&attn_out, &layer.attn_out_w, seq_len, inner_dim, h);
        add_bias(&mut proj, &layer.attn_out_b, seq_len, h);

        // 6. Gated residual: x = x + gate1 * attn_out
        for t in 0..seq_len {
            for j in 0..h {
                let idx = t * h + j;
                residual[idx] += gate1[j] * proj[idx];
            }
        }

        // 7. Pre-FFN: LayerNorm + modulate
        let normed2 = layernorm_no_bias(&residual, &layer.mlp_norm_w, h);
        let mut modulated2 = vec![0.0f32; seq_len * h];
        for t in 0..seq_len {
            for j in 0..h {
                let idx = t * h + j;
                let val = normed2[idx] + layer.mlp_norm_b[j];
                modulated2[idx] = val * (1.0 + scale2[j]) + shift2[j];
            }
        }

        // 8. MLP: up -> GELU -> down
        let mut up = matmul_t(&modulated2, &layer.mlp_up_w, seq_len, h, inter);
        add_bias(&mut up, &layer.mlp_up_b, seq_len, inter);
        for val in up.iter_mut() {
            *val = gelu(*val);
        }
        let mut down = matmul_t(&up, &layer.mlp_down_w, seq_len, inter, h);
        add_bias(&mut down, &layer.mlp_down_b, seq_len, h);

        // 9. Gated residual: x = x + gate2 * mlp_out
        for t in 0..seq_len {
            for j in 0..h {
                let idx = t * h + j;
                residual[idx] += gate2[j] * down[idx];
            }
        }

        residual
    }

    // ── Predictor Forward ───────────────────────────────────────────

    fn predict_next_inner(&self, z_t: &[f32], action: &[f32]) -> Vec<f32> {
        // 1. Encode action -> [HIDDEN]
        let a_embed = self.encode_action(action);

        // 2. Project state through projector: [192] -> [2048] -> BN -> GELU -> [192]
        let z_projected = Self::projection_forward(
            z_t,
            &self.projector_layers,
            &self.projector_bn_weight,
            &self.projector_bn_bias,
            &self.projector_bn_mean,
            &self.projector_bn_var,
        );

        // 3. Build sequence: [z_projected, action_embed, zeros] + pos_embed
        let seq_len = 3;
        let mut seq = vec![0.0f32; seq_len * HIDDEN];
        seq[..HIDDEN].copy_from_slice(&z_projected);
        seq[HIDDEN..2 * HIDDEN].copy_from_slice(&a_embed);
        // seq[2*HIDDEN..3*HIDDEN] = zeros (target position)

        // Add positional embeddings
        if !self.predictor_pos_embed.is_empty() {
            let pos_len = self.predictor_pos_embed.len().min(seq.len());
            for i in 0..pos_len {
                seq[i] += self.predictor_pos_embed[i];
            }
        }

        // 4. Run through predictor layers (conditioning = action embedding)
        for layer in &self.predictor_layers {
            seq = self.adaln_layer_forward(&seq, &a_embed, layer, seq_len);
        }

        // 5. Final norm
        let normed = layernorm(&seq, &self.predictor_norm_w, &self.predictor_norm_b, HIDDEN);

        // 6. Extract target position (index 2)
        let target = &normed[2 * HIDDEN..3 * HIDDEN];

        // 7. Project through pred_proj
        Self::projection_forward(
            target,
            &self.pred_proj_layers,
            &self.pred_proj_bn_weight,
            &self.pred_proj_bn_bias,
            &self.pred_proj_bn_mean,
            &self.pred_proj_bn_var,
        )
    }
}

// ── WASM Exports ────────────────────────────────────────────────────

#[wasm_bindgen]
impl RealLeWM {
    /// Load model from compact binary format.
    ///
    /// Format: [u32 header_len][JSON header][raw f32 data]
    #[wasm_bindgen(constructor)]
    pub fn load_from_bytes(data: &[u8]) -> Result<RealLeWM, JsError> {
        if data.len() < 4 {
            return Err(JsError::new("Data too short for header length"));
        }

        // 1. Read header length
        let header_len = u32::from_le_bytes([data[0], data[1], data[2], data[3]]) as usize;

        if data.len() < 4 + header_len {
            return Err(JsError::new("Data too short for header"));
        }

        // 2. Parse JSON header
        let header_bytes = &data[4..4 + header_len];
        let header: HashMap<String, TensorInfo> = serde_json::from_slice(header_bytes)
            .map_err(|e| JsError::new(&format!("Failed to parse header: {}", e)))?;

        let data_start = 4 + header_len;

        // Helper closure
        let get = |name: &str| -> Result<Vec<f32>, JsError> {
            Self::get_tensor(&header, data, data_start, name)
        };
        let get_opt =
            |name: &str| -> Vec<f32> { Self::get_tensor_opt(&header, data, data_start, name) };

        // 3. Load encoder weights
        let encoder_patch_proj = get("encoder.embeddings.patch_embeddings.projection.weight")?;
        let encoder_patch_proj_bias = get("encoder.embeddings.patch_embeddings.projection.bias")?;
        let encoder_cls_token = get("encoder.embeddings.cls_token")?;
        let encoder_pos_embed = get("encoder.embeddings.position_embeddings")?;
        let encoder_norm_w = get("encoder.layernorm.weight")?;
        let encoder_norm_b = get("encoder.layernorm.bias")?;

        let mut encoder_layers = Vec::with_capacity(ENCODER_LAYERS);
        for i in 0..ENCODER_LAYERS {
            let prefix = format!("encoder.encoder.layer.{}", i);
            encoder_layers.push(ViTLayerWeights {
                q_w: get(&format!("{}.attention.attention.query.weight", prefix))?,
                q_b: get(&format!("{}.attention.attention.query.bias", prefix))?,
                k_w: get(&format!("{}.attention.attention.key.weight", prefix))?,
                k_b: get(&format!("{}.attention.attention.key.bias", prefix))?,
                v_w: get(&format!("{}.attention.attention.value.weight", prefix))?,
                v_b: get(&format!("{}.attention.attention.value.bias", prefix))?,
                o_w: get(&format!("{}.attention.output.dense.weight", prefix))?,
                o_b: get(&format!("{}.attention.output.dense.bias", prefix))?,
                ffn_up_w: get(&format!("{}.intermediate.dense.weight", prefix))?,
                ffn_up_b: get(&format!("{}.intermediate.dense.bias", prefix))?,
                ffn_down_w: get(&format!("{}.output.dense.weight", prefix))?,
                ffn_down_b: get(&format!("{}.output.dense.bias", prefix))?,
                norm1_w: get(&format!("{}.layernorm_before.weight", prefix))?,
                norm1_b: get(&format!("{}.layernorm_before.bias", prefix))?,
                norm2_w: get(&format!("{}.layernorm_after.weight", prefix))?,
                norm2_b: get(&format!("{}.layernorm_after.bias", prefix))?,
            });
        }

        // 4. Load predictor weights
        let predictor_pos_embed = get("predictor.pos_embedding")?;
        let predictor_norm_w = get("predictor.transformer.norm.weight")?;
        let predictor_norm_b = get("predictor.transformer.norm.bias")?;

        let mut predictor_layers = Vec::with_capacity(PREDICTOR_LAYERS);
        for i in 0..PREDICTOR_LAYERS {
            let prefix = format!("predictor.transformer.layers.{}", i);
            predictor_layers.push(AdaLNWeights {
                adaln_w: get(&format!("{}.adaLN_modulation.1.weight", prefix))?,
                adaln_b: get(&format!("{}.adaLN_modulation.1.bias", prefix))?,
                to_qkv_w: get(&format!("{}.attn.to_qkv.weight", prefix))?,
                attn_out_w: get(&format!("{}.attn.to_out.0.weight", prefix))?,
                attn_out_b: get(&format!("{}.attn.to_out.0.bias", prefix))?,
                attn_norm_w: get(&format!("{}.attn.norm.weight", prefix))?,
                attn_norm_b: get(&format!("{}.attn.norm.bias", prefix))?,
                mlp_norm_w: get(&format!("{}.mlp.net.0.weight", prefix))?,
                mlp_norm_b: get(&format!("{}.mlp.net.0.bias", prefix))?,
                mlp_up_w: get(&format!("{}.mlp.net.1.weight", prefix))?,
                mlp_up_b: get(&format!("{}.mlp.net.1.bias", prefix))?,
                mlp_down_w: get(&format!("{}.mlp.net.4.weight", prefix))?,
                mlp_down_b: get(&format!("{}.mlp.net.4.bias", prefix))?,
            });
        }

        // 5. Load action encoder weights
        let action_conv_w = get("action_encoder.patch_embed.weight")?;
        let action_conv_b = get("action_encoder.patch_embed.bias")?;
        let action_mlp1_w = get("action_encoder.embed.0.weight")?;
        let action_mlp1_b = get("action_encoder.embed.0.bias")?;
        let action_mlp2_w = get("action_encoder.embed.2.weight")?;
        let action_mlp2_b = get("action_encoder.embed.2.bias")?;

        // 6. Load projector weights (net.0 = linear, net.1 = BN, net.3 = linear)
        let projector_layers = vec![
            (get("projector.net.0.weight")?, get("projector.net.0.bias")?),
            (get("projector.net.3.weight")?, get("projector.net.3.bias")?),
        ];
        let projector_bn_weight = get_opt("projector.net.1.weight");
        let projector_bn_bias = get_opt("projector.net.1.bias");
        let projector_bn_mean = get_opt("projector.net.1.running_mean");
        let projector_bn_var = get_opt("projector.net.1.running_var");

        // 7. Load pred_proj weights
        let pred_proj_layers = vec![
            (get("pred_proj.net.0.weight")?, get("pred_proj.net.0.bias")?),
            (get("pred_proj.net.3.weight")?, get("pred_proj.net.3.bias")?),
        ];
        let pred_proj_bn_weight = get_opt("pred_proj.net.1.weight");
        let pred_proj_bn_bias = get_opt("pred_proj.net.1.bias");
        let pred_proj_bn_mean = get_opt("pred_proj.net.1.running_mean");
        let pred_proj_bn_var = get_opt("pred_proj.net.1.running_var");

        Ok(RealLeWM {
            encoder_patch_proj,
            encoder_patch_proj_bias,
            encoder_cls_token,
            encoder_pos_embed,
            encoder_layers,
            encoder_norm_w,
            encoder_norm_b,
            predictor_pos_embed,
            predictor_layers,
            predictor_norm_w,
            predictor_norm_b,
            action_conv_w,
            action_conv_b,
            action_mlp1_w,
            action_mlp1_b,
            action_mlp2_w,
            action_mlp2_b,
            projector_layers,
            projector_bn_weight,
            projector_bn_bias,
            projector_bn_mean,
            projector_bn_var,
            pred_proj_layers,
            pred_proj_bn_weight,
            pred_proj_bn_bias,
            pred_proj_bn_mean,
            pred_proj_bn_var,
        })
    }

    /// Encode a 224x224x3 image (HWC, flat f32 array) to a latent state [192].
    pub fn encode_image(&self, pixels: &[f32], height: usize, width: usize) -> Vec<f32> {
        self.encode_image_inner(pixels, height, width)
    }

    /// Predict next latent state given current state [192] and action [10].
    pub fn predict_next(&self, state: &[f32], action: &[f32]) -> Vec<f32> {
        self.predict_next_inner(state, action)
    }

    /// Multi-step rollout. Returns flattened array [num_steps * 192].
    pub fn rollout(&self, state: &[f32], actions: &[f32], num_steps: usize) -> Vec<f32> {
        let mut states = Vec::with_capacity(num_steps * LATENT_DIM);
        let mut current = state.to_vec();
        for step in 0..num_steps {
            let action = &actions[step * ACTION_DIM..(step + 1) * ACTION_DIM];
            current = self.predict_next_inner(&current, action);
            states.extend_from_slice(&current);
        }
        states
    }

    /// Returns the latent dimension (192).
    pub fn latent_dim(&self) -> usize {
        LATENT_DIM
    }
}

// ── QuantizedLinearWasm ────────────────────────────────────────────
// Uses quantize_per_channel_int8 and qgemm_int8 from synapse-inference

struct QuantizedLinearWasm {
    weights_int8: Vec<i8>,  // [in_features, out_features] transposed layout
    scales: Vec<f32>,       // [out_features]
    out_features: usize,
    in_features: usize,
}

impl QuantizedLinearWasm {
    fn from_f32(weights: &[f32], out_features: usize, in_features: usize) -> Self {
        // Quantize per output-channel (row), then transpose for GEMM
        let mut scales = vec![0.0f32; out_features];
        let mut weights_row = vec![0i8; out_features * in_features];
        for ch in 0..out_features {
            let row = &weights[ch * in_features..(ch + 1) * in_features];
            let max_abs = row.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
            let scale = if max_abs == 0.0 { 1.0 } else { max_abs / 127.0 };
            scales[ch] = scale;
            let inv = 1.0 / scale;
            for j in 0..in_features {
                weights_row[ch * in_features + j] = (row[j] * inv).round().clamp(-128.0, 127.0) as i8;
            }
        }
        // Transpose to [in_features, out_features] for efficient GEMM
        let mut weights_int8 = vec![0i8; out_features * in_features];
        for r in 0..out_features {
            for c in 0..in_features {
                weights_int8[c * out_features + r] = weights_row[r * in_features + c];
            }
        }
        QuantizedLinearWasm { weights_int8, scales, out_features, in_features }
    }

    fn forward(&self, x: &[f32], m: usize) -> Vec<f32> {
        let k = self.in_features;
        let n = self.out_features;
        let (x_int8, scales_x) = quantize_per_channel_int8(x, m, k);
        qgemm_int8(m, n, k, &x_int8, &self.weights_int8, &scales_x, &self.scales)
    }
}

// ── QuantizedAdaLNWasm ─────────────────────────────────────────────

struct QuantizedAdaLNWasm {
    // Heavy matrices quantized to INT8
    adaln_linear: QuantizedLinearWasm,
    to_qkv: QuantizedLinearWasm,
    attn_out: QuantizedLinearWasm,
    mlp_up: QuantizedLinearWasm,
    mlp_down: QuantizedLinearWasm,
    // Biases and norm weights remain f32
    adaln_b: Vec<f32>,
    attn_out_b: Vec<f32>,
    attn_norm_w: Vec<f32>,
    attn_norm_b: Vec<f32>,
    mlp_norm_w: Vec<f32>,
    mlp_norm_b: Vec<f32>,
    mlp_up_b: Vec<f32>,
    mlp_down_b: Vec<f32>,
}

impl QuantizedAdaLNWasm {
    fn from_adaln(layer: &AdaLNWeights) -> Self {
        let h = HIDDEN;
        let inner_dim = PREDICTOR_INNER_DIM;
        let inter = PREDICTOR_INTER;
        let mod_dim = 6 * h;

        QuantizedAdaLNWasm {
            adaln_linear: QuantizedLinearWasm::from_f32(&layer.adaln_w, mod_dim, h),
            to_qkv: QuantizedLinearWasm::from_f32(&layer.to_qkv_w, 3 * inner_dim, h),
            attn_out: QuantizedLinearWasm::from_f32(&layer.attn_out_w, h, inner_dim),
            mlp_up: QuantizedLinearWasm::from_f32(&layer.mlp_up_w, inter, h),
            mlp_down: QuantizedLinearWasm::from_f32(&layer.mlp_down_w, h, inter),
            adaln_b: layer.adaln_b.clone(),
            attn_out_b: layer.attn_out_b.clone(),
            attn_norm_w: layer.attn_norm_w.clone(),
            attn_norm_b: layer.attn_norm_b.clone(),
            mlp_norm_w: layer.mlp_norm_w.clone(),
            mlp_norm_b: layer.mlp_norm_b.clone(),
            mlp_up_b: layer.mlp_up_b.clone(),
            mlp_down_b: layer.mlp_down_b.clone(),
        }
    }
}

// ── RealLeWMInt8 ───────────────────────────────────────────────────

#[wasm_bindgen]
pub struct RealLeWMInt8 {
    // Encoder stays f32 (same as RealLeWM)
    encoder_patch_proj: Vec<f32>,
    encoder_patch_proj_bias: Vec<f32>,
    encoder_cls_token: Vec<f32>,
    encoder_pos_embed: Vec<f32>,
    encoder_layers: Vec<ViTLayerWeights>,
    encoder_norm_w: Vec<f32>,
    encoder_norm_b: Vec<f32>,

    // Predictor uses quantized layers
    predictor_pos_embed: Vec<f32>,
    predictor_layers: Vec<QuantizedAdaLNWasm>,
    predictor_norm_w: Vec<f32>,
    predictor_norm_b: Vec<f32>,

    // Action encoder (f32, small)
    action_conv_w: Vec<f32>,
    action_conv_b: Vec<f32>,
    action_mlp1_w: Vec<f32>,
    action_mlp1_b: Vec<f32>,
    action_mlp2_w: Vec<f32>,
    action_mlp2_b: Vec<f32>,

    // Projector (f32)
    projector_layers: Vec<(Vec<f32>, Vec<f32>)>,
    projector_bn_weight: Vec<f32>,
    projector_bn_bias: Vec<f32>,
    projector_bn_mean: Vec<f32>,
    projector_bn_var: Vec<f32>,

    // Pred_proj (f32)
    pred_proj_layers: Vec<(Vec<f32>, Vec<f32>)>,
    pred_proj_bn_weight: Vec<f32>,
    pred_proj_bn_bias: Vec<f32>,
    pred_proj_bn_mean: Vec<f32>,
    pred_proj_bn_var: Vec<f32>,
}

impl RealLeWMInt8 {
    // ── ViT Encoder Forward (same as RealLeWM) ─────────────────────

    fn vit_layer_forward(&self, x: &[f32], layer: &ViTLayerWeights, seq_len: usize) -> Vec<f32> {
        let h = HIDDEN;
        let num_heads = ENCODER_HEADS;
        let head_dim = h / num_heads;
        let q_dim = num_heads * head_dim;
        let inter = ENCODER_INTER;

        let normed = layernorm(x, &layer.norm1_w, &layer.norm1_b, h);

        let mut q = matmul_t(&normed, &layer.q_w, seq_len, h, q_dim);
        add_bias(&mut q, &layer.q_b, seq_len, q_dim);
        let mut k = matmul_t(&normed, &layer.k_w, seq_len, h, q_dim);
        add_bias(&mut k, &layer.k_b, seq_len, q_dim);
        let mut v = matmul_t(&normed, &layer.v_w, seq_len, h, q_dim);
        add_bias(&mut v, &layer.v_b, seq_len, q_dim);

        let attn_out = bidirectional_attention(&q, &k, &v, seq_len, num_heads, head_dim);

        let mut proj = matmul_t(&attn_out, &layer.o_w, seq_len, q_dim, h);
        add_bias(&mut proj, &layer.o_b, seq_len, h);

        let mut residual = vec![0.0f32; seq_len * h];
        for i in 0..seq_len * h {
            residual[i] = x[i] + proj[i];
        }

        let normed2 = layernorm(&residual, &layer.norm2_w, &layer.norm2_b, h);

        let mut up = matmul_t(&normed2, &layer.ffn_up_w, seq_len, h, inter);
        add_bias(&mut up, &layer.ffn_up_b, seq_len, inter);
        for val in up.iter_mut() {
            *val = gelu(*val);
        }
        let mut down = matmul_t(&up, &layer.ffn_down_w, seq_len, inter, h);
        add_bias(&mut down, &layer.ffn_down_b, seq_len, h);

        for i in 0..seq_len * h {
            residual[i] += down[i];
        }

        residual
    }

    fn encode_image_inner(&self, pixels: &[f32], height: usize, width: usize) -> Vec<f32> {
        let patch_embeddings = patch_embed(
            pixels,
            height,
            width,
            &self.encoder_patch_proj,
            &self.encoder_patch_proj_bias,
        );

        let seq_len = SEQ_LEN_VIT;

        let mut x = vec![0.0f32; seq_len * HIDDEN];
        x[..HIDDEN].copy_from_slice(&self.encoder_cls_token);
        x[HIDDEN..].copy_from_slice(&patch_embeddings);

        let pos_len = self.encoder_pos_embed.len().min(x.len());
        for i in 0..pos_len {
            x[i] += self.encoder_pos_embed[i];
        }

        for layer in &self.encoder_layers {
            x = self.vit_layer_forward(&x, layer, seq_len);
        }

        let cls_hidden = &x[..HIDDEN];
        layernorm(cls_hidden, &self.encoder_norm_w, &self.encoder_norm_b, HIDDEN)
    }

    // ── Action Encoder (same as RealLeWM) ──────────────────────────

    fn encode_action(&self, action: &[f32]) -> Vec<f32> {
        let conv_out = if !self.action_conv_w.is_empty() {
            let mut out = matmul_t(action, &self.action_conv_w, 1, ACTION_DIM, ACTION_DIM);
            if !self.action_conv_b.is_empty() {
                for j in 0..ACTION_DIM {
                    out[j] += self.action_conv_b[j];
                }
            }
            out
        } else {
            action.to_vec()
        };

        let inter = if !self.action_mlp1_w.is_empty() {
            self.action_mlp1_w.len() / ACTION_DIM
        } else {
            ENCODER_INTER
        };

        let mut h1 = matmul_t(&conv_out, &self.action_mlp1_w, 1, ACTION_DIM, inter);
        if !self.action_mlp1_b.is_empty() {
            for j in 0..inter {
                h1[j] += self.action_mlp1_b[j];
            }
        }
        for v in h1.iter_mut() {
            *v = gelu(*v);
        }

        let mut out = matmul_t(&h1, &self.action_mlp2_w, 1, inter, HIDDEN);
        if !self.action_mlp2_b.is_empty() {
            for j in 0..HIDDEN {
                out[j] += self.action_mlp2_b[j];
            }
        }

        out
    }

    // ── Quantized adaLN Predictor Layer Forward ────────────────────

    fn adaln_layer_forward_q(
        &self,
        x: &[f32],
        conditioning: &[f32],
        layer: &QuantizedAdaLNWasm,
        seq_len: usize,
    ) -> Vec<f32> {
        let h = HIDDEN;
        let num_heads = PREDICTOR_HEADS;
        let inner_dim = PREDICTOR_INNER_DIM;
        let head_dim = PREDICTOR_HEAD_DIM;
        let inter = PREDICTOR_INTER;
        let mod_dim = 6 * h;

        // 1. Compute adaLN modulation using quantized linear
        let mut mod_vec = layer.adaln_linear.forward(conditioning, 1);
        for j in 0..mod_dim {
            mod_vec[j] += layer.adaln_b[j];
        }
        let scale1 = &mod_vec[0..h];
        let shift1 = &mod_vec[h..2 * h];
        let gate1 = &mod_vec[2 * h..3 * h];
        let scale2 = &mod_vec[3 * h..4 * h];
        let shift2 = &mod_vec[4 * h..5 * h];
        let gate2 = &mod_vec[5 * h..6 * h];

        let mut residual = x.to_vec();

        // 2. Pre-attention: LayerNorm + modulate
        let normed = layernorm_no_bias(x, &layer.attn_norm_w, h);
        let mut modulated = vec![0.0f32; seq_len * h];
        for t in 0..seq_len {
            for j in 0..h {
                let idx = t * h + j;
                let val = normed[idx] + layer.attn_norm_b[j];
                modulated[idx] = val * (1.0 + scale1[j]) + shift1[j];
            }
        }

        // 3. Fused QKV using quantized linear
        let qkv = layer.to_qkv.forward(&modulated, seq_len);

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

        // 4. Bidirectional attention
        let attn_out = bidirectional_attention(&q, &k, &v, seq_len, num_heads, head_dim);

        // 5. Output projection using quantized linear
        let mut proj = layer.attn_out.forward(&attn_out, seq_len);
        add_bias(&mut proj, &layer.attn_out_b, seq_len, h);

        // 6. Gated residual
        for t in 0..seq_len {
            for j in 0..h {
                let idx = t * h + j;
                residual[idx] += gate1[j] * proj[idx];
            }
        }

        // 7. Pre-FFN: LayerNorm + modulate
        let normed2 = layernorm_no_bias(&residual, &layer.mlp_norm_w, h);
        let mut modulated2 = vec![0.0f32; seq_len * h];
        for t in 0..seq_len {
            for j in 0..h {
                let idx = t * h + j;
                let val = normed2[idx] + layer.mlp_norm_b[j];
                modulated2[idx] = val * (1.0 + scale2[j]) + shift2[j];
            }
        }

        // 8. MLP using quantized linears
        let mut up = layer.mlp_up.forward(&modulated2, seq_len);
        add_bias(&mut up, &layer.mlp_up_b, seq_len, inter);
        for val in up.iter_mut() {
            *val = gelu(*val);
        }
        let mut down = layer.mlp_down.forward(&up, seq_len);
        add_bias(&mut down, &layer.mlp_down_b, seq_len, h);

        // 9. Gated residual
        for t in 0..seq_len {
            for j in 0..h {
                let idx = t * h + j;
                residual[idx] += gate2[j] * down[idx];
            }
        }

        residual
    }

    // ── Predictor Forward ──────────────────────────────────────────

    fn predict_next_inner(&self, z_t: &[f32], action: &[f32]) -> Vec<f32> {
        let a_embed = self.encode_action(action);

        let z_projected = RealLeWM::projection_forward(
            z_t,
            &self.projector_layers,
            &self.projector_bn_weight,
            &self.projector_bn_bias,
            &self.projector_bn_mean,
            &self.projector_bn_var,
        );

        let seq_len = 3;
        let mut seq = vec![0.0f32; seq_len * HIDDEN];
        seq[..HIDDEN].copy_from_slice(&z_projected);
        seq[HIDDEN..2 * HIDDEN].copy_from_slice(&a_embed);

        if !self.predictor_pos_embed.is_empty() {
            let pos_len = self.predictor_pos_embed.len().min(seq.len());
            for i in 0..pos_len {
                seq[i] += self.predictor_pos_embed[i];
            }
        }

        for layer in &self.predictor_layers {
            seq = self.adaln_layer_forward_q(&seq, &a_embed, layer, seq_len);
        }

        let normed = layernorm(&seq, &self.predictor_norm_w, &self.predictor_norm_b, HIDDEN);

        let target = &normed[2 * HIDDEN..3 * HIDDEN];

        RealLeWM::projection_forward(
            target,
            &self.pred_proj_layers,
            &self.pred_proj_bn_weight,
            &self.pred_proj_bn_bias,
            &self.pred_proj_bn_mean,
            &self.pred_proj_bn_var,
        )
    }
}

#[wasm_bindgen]
impl RealLeWMInt8 {
    /// Load from f32 binary (same format as RealLeWM) and quantize predictor
    /// layers to INT8 in-browser. Same 69MB download, ~4x less runtime memory
    /// for predictor inference.
    pub fn from_f32_data(data: &[u8]) -> Result<RealLeWMInt8, JsError> {
        // First, parse the same format as RealLeWM
        if data.len() < 4 {
            return Err(JsError::new("Data too short for header length"));
        }

        let header_len = u32::from_le_bytes([data[0], data[1], data[2], data[3]]) as usize;

        if data.len() < 4 + header_len {
            return Err(JsError::new("Data too short for header"));
        }

        let header_bytes = &data[4..4 + header_len];
        let header: HashMap<String, TensorInfo> = serde_json::from_slice(header_bytes)
            .map_err(|e| JsError::new(&format!("Failed to parse header: {}", e)))?;

        let data_start = 4 + header_len;

        let get = |name: &str| -> Result<Vec<f32>, JsError> {
            RealLeWM::get_tensor(&header, data, data_start, name)
        };
        let get_opt =
            |name: &str| -> Vec<f32> { RealLeWM::get_tensor_opt(&header, data, data_start, name) };

        // Load encoder weights (f32, same as RealLeWM)
        let encoder_patch_proj = get("encoder.embeddings.patch_embeddings.projection.weight")?;
        let encoder_patch_proj_bias = get("encoder.embeddings.patch_embeddings.projection.bias")?;
        let encoder_cls_token = get("encoder.embeddings.cls_token")?;
        let encoder_pos_embed = get("encoder.embeddings.position_embeddings")?;
        let encoder_norm_w = get("encoder.layernorm.weight")?;
        let encoder_norm_b = get("encoder.layernorm.bias")?;

        let mut encoder_layers = Vec::with_capacity(ENCODER_LAYERS);
        for i in 0..ENCODER_LAYERS {
            let prefix = format!("encoder.encoder.layer.{}", i);
            encoder_layers.push(ViTLayerWeights {
                q_w: get(&format!("{}.attention.attention.query.weight", prefix))?,
                q_b: get(&format!("{}.attention.attention.query.bias", prefix))?,
                k_w: get(&format!("{}.attention.attention.key.weight", prefix))?,
                k_b: get(&format!("{}.attention.attention.key.bias", prefix))?,
                v_w: get(&format!("{}.attention.attention.value.weight", prefix))?,
                v_b: get(&format!("{}.attention.attention.value.bias", prefix))?,
                o_w: get(&format!("{}.attention.output.dense.weight", prefix))?,
                o_b: get(&format!("{}.attention.output.dense.bias", prefix))?,
                ffn_up_w: get(&format!("{}.intermediate.dense.weight", prefix))?,
                ffn_up_b: get(&format!("{}.intermediate.dense.bias", prefix))?,
                ffn_down_w: get(&format!("{}.output.dense.weight", prefix))?,
                ffn_down_b: get(&format!("{}.output.dense.bias", prefix))?,
                norm1_w: get(&format!("{}.layernorm_before.weight", prefix))?,
                norm1_b: get(&format!("{}.layernorm_before.bias", prefix))?,
                norm2_w: get(&format!("{}.layernorm_after.weight", prefix))?,
                norm2_b: get(&format!("{}.layernorm_after.bias", prefix))?,
            });
        }

        // Load predictor weights as f32 first, then quantize
        let predictor_pos_embed = get("predictor.pos_embedding")?;
        let predictor_norm_w = get("predictor.transformer.norm.weight")?;
        let predictor_norm_b = get("predictor.transformer.norm.bias")?;

        let mut predictor_layers = Vec::with_capacity(PREDICTOR_LAYERS);
        for i in 0..PREDICTOR_LAYERS {
            let prefix = format!("predictor.transformer.layers.{}", i);
            let f32_layer = AdaLNWeights {
                adaln_w: get(&format!("{}.adaLN_modulation.1.weight", prefix))?,
                adaln_b: get(&format!("{}.adaLN_modulation.1.bias", prefix))?,
                to_qkv_w: get(&format!("{}.attn.to_qkv.weight", prefix))?,
                attn_out_w: get(&format!("{}.attn.to_out.0.weight", prefix))?,
                attn_out_b: get(&format!("{}.attn.to_out.0.bias", prefix))?,
                attn_norm_w: get(&format!("{}.attn.norm.weight", prefix))?,
                attn_norm_b: get(&format!("{}.attn.norm.bias", prefix))?,
                mlp_norm_w: get(&format!("{}.mlp.net.0.weight", prefix))?,
                mlp_norm_b: get(&format!("{}.mlp.net.0.bias", prefix))?,
                mlp_up_w: get(&format!("{}.mlp.net.1.weight", prefix))?,
                mlp_up_b: get(&format!("{}.mlp.net.1.bias", prefix))?,
                mlp_down_w: get(&format!("{}.mlp.net.4.weight", prefix))?,
                mlp_down_b: get(&format!("{}.mlp.net.4.bias", prefix))?,
            };
            // Quantize to INT8
            predictor_layers.push(QuantizedAdaLNWasm::from_adaln(&f32_layer));
        }

        // Action encoder (f32)
        let action_conv_w = get("action_encoder.patch_embed.weight")?;
        let action_conv_b = get("action_encoder.patch_embed.bias")?;
        let action_mlp1_w = get("action_encoder.embed.0.weight")?;
        let action_mlp1_b = get("action_encoder.embed.0.bias")?;
        let action_mlp2_w = get("action_encoder.embed.2.weight")?;
        let action_mlp2_b = get("action_encoder.embed.2.bias")?;

        // Projector (f32)
        let projector_layers = vec![
            (get("projector.net.0.weight")?, get("projector.net.0.bias")?),
            (get("projector.net.3.weight")?, get("projector.net.3.bias")?),
        ];
        let projector_bn_weight = get_opt("projector.net.1.weight");
        let projector_bn_bias = get_opt("projector.net.1.bias");
        let projector_bn_mean = get_opt("projector.net.1.running_mean");
        let projector_bn_var = get_opt("projector.net.1.running_var");

        // Pred_proj (f32)
        let pred_proj_layers = vec![
            (get("pred_proj.net.0.weight")?, get("pred_proj.net.0.bias")?),
            (get("pred_proj.net.3.weight")?, get("pred_proj.net.3.bias")?),
        ];
        let pred_proj_bn_weight = get_opt("pred_proj.net.1.weight");
        let pred_proj_bn_bias = get_opt("pred_proj.net.1.bias");
        let pred_proj_bn_mean = get_opt("pred_proj.net.1.running_mean");
        let pred_proj_bn_var = get_opt("pred_proj.net.1.running_var");

        Ok(RealLeWMInt8 {
            encoder_patch_proj,
            encoder_patch_proj_bias,
            encoder_cls_token,
            encoder_pos_embed,
            encoder_layers,
            encoder_norm_w,
            encoder_norm_b,
            predictor_pos_embed,
            predictor_layers,
            predictor_norm_w,
            predictor_norm_b,
            action_conv_w,
            action_conv_b,
            action_mlp1_w,
            action_mlp1_b,
            action_mlp2_w,
            action_mlp2_b,
            projector_layers,
            projector_bn_weight,
            projector_bn_bias,
            projector_bn_mean,
            projector_bn_var,
            pred_proj_layers,
            pred_proj_bn_weight,
            pred_proj_bn_bias,
            pred_proj_bn_mean,
            pred_proj_bn_var,
        })
    }

    /// Encode a 224x224x3 image (HWC, flat f32 array) to a latent state [192].
    /// Same as RealLeWM — encoder is f32.
    pub fn encode_image(&self, pixels: &[f32], height: usize, width: usize) -> Vec<f32> {
        self.encode_image_inner(pixels, height, width)
    }

    /// Predict next latent state using INT8 quantized predictor layers.
    pub fn predict_next(&self, state: &[f32], action: &[f32]) -> Vec<f32> {
        self.predict_next_inner(state, action)
    }

    /// Multi-step rollout using INT8 quantized predictor layers.
    /// Returns flattened array [num_steps * 192].
    pub fn rollout(&self, state: &[f32], actions: &[f32], num_steps: usize) -> Vec<f32> {
        let mut states = Vec::with_capacity(num_steps * LATENT_DIM);
        let mut current = state.to_vec();
        for step in 0..num_steps {
            let action = &actions[step * ACTION_DIM..(step + 1) * ACTION_DIM];
            current = self.predict_next_inner(&current, action);
            states.extend_from_slice(&current);
        }
        states
    }

    /// Returns the latent dimension (192).
    pub fn latent_dim(&self) -> usize {
        LATENT_DIM
    }

    /// Returns the action dimension (10).
    pub fn action_dim(&self) -> usize {
        ACTION_DIM
    }
}

// ── Keep the old WasmWorldModel for backward compat ─────────────────

/// A minimal world model dynamics predictor for WASM (demo with random weights).
#[wasm_bindgen]
pub struct WasmWorldModel {
    latent_dim: usize,
    action_dim: usize,
    hidden_dim: usize,
    num_layers: usize,
    num_heads: usize,
    action_proj_w: Vec<f32>,
    action_proj_b: Vec<f32>,
    layers_qkv_w: Vec<Vec<f32>>,
    layers_out_w: Vec<Vec<f32>>,
    layers_out_b: Vec<Vec<f32>>,
    layers_norm1_w: Vec<Vec<f32>>,
    layers_norm1_b: Vec<Vec<f32>>,
    layers_mlp_up_w: Vec<Vec<f32>>,
    layers_mlp_up_b: Vec<Vec<f32>>,
    layers_mlp_down_w: Vec<Vec<f32>>,
    layers_mlp_down_b: Vec<Vec<f32>>,
    layers_norm2_w: Vec<Vec<f32>>,
    layers_norm2_b: Vec<Vec<f32>>,
    final_norm_w: Vec<f32>,
    final_norm_b: Vec<f32>,
    output_proj_w: Vec<f32>,
    output_proj_b: Vec<f32>,
}

#[wasm_bindgen]
impl WasmWorldModel {
    #[wasm_bindgen(constructor)]
    pub fn new(
        latent_dim: usize,
        action_dim: usize,
        hidden_dim: usize,
        num_layers: usize,
        num_heads: usize,
    ) -> Self {
        let inter = hidden_dim * 4;
        let gen = |len: usize, seed: u32| -> Vec<f32> {
            (0..len)
                .map(|i| {
                    let x = ((i as u32).wrapping_mul(2654435761).wrapping_add(seed)) as f32;
                    (x / u32::MAX as f32) * 0.1 - 0.05
                })
                .collect()
        };

        let mut layers_qkv_w = Vec::new();
        let mut layers_out_w = Vec::new();
        let mut layers_out_b = Vec::new();
        let mut layers_norm1_w = Vec::new();
        let mut layers_norm1_b = Vec::new();
        let mut layers_mlp_up_w = Vec::new();
        let mut layers_mlp_up_b = Vec::new();
        let mut layers_mlp_down_w = Vec::new();
        let mut layers_mlp_down_b = Vec::new();
        let mut layers_norm2_w = Vec::new();
        let mut layers_norm2_b = Vec::new();

        for i in 0..num_layers {
            let s = (i as u32 + 1) * 100;
            layers_qkv_w.push(gen(3 * hidden_dim * hidden_dim, s + 1));
            layers_out_w.push(gen(hidden_dim * hidden_dim, s + 2));
            layers_out_b.push(vec![0.0; hidden_dim]);
            layers_norm1_w.push(vec![1.0; hidden_dim]);
            layers_norm1_b.push(vec![0.0; hidden_dim]);
            layers_mlp_up_w.push(gen(inter * hidden_dim, s + 5));
            layers_mlp_up_b.push(vec![0.0; inter]);
            layers_mlp_down_w.push(gen(hidden_dim * inter, s + 6));
            layers_mlp_down_b.push(vec![0.0; hidden_dim]);
            layers_norm2_w.push(vec![1.0; hidden_dim]);
            layers_norm2_b.push(vec![0.0; hidden_dim]);
        }

        Self {
            latent_dim,
            action_dim,
            hidden_dim,
            num_layers,
            num_heads,
            action_proj_w: gen(hidden_dim * action_dim, 1),
            action_proj_b: vec![0.0; hidden_dim],
            layers_qkv_w,
            layers_out_w,
            layers_out_b,
            layers_norm1_w,
            layers_norm1_b,
            layers_mlp_up_w,
            layers_mlp_up_b,
            layers_mlp_down_w,
            layers_mlp_down_b,
            layers_norm2_w,
            layers_norm2_b,
            final_norm_w: vec![1.0; hidden_dim],
            final_norm_b: vec![0.0; hidden_dim],
            output_proj_w: gen(latent_dim * hidden_dim, 99),
            output_proj_b: vec![0.0; latent_dim],
        }
    }

    pub fn predict_next(&self, state: &[f32], action: &[f32]) -> Vec<f32> {
        let h = self.hidden_dim;
        let inter = h * 4;
        let heads = self.num_heads;
        let head_dim = h / heads;

        let mut a_hidden = matmul_t(action, &self.action_proj_w, 1, self.action_dim, h);
        for i in 0..h {
            a_hidden[i] += self.action_proj_b[i];
        }

        let seq_len = 2;
        let mut x = vec![0.0f32; seq_len * h];
        x[..h].copy_from_slice(&state[..h.min(self.latent_dim)]);
        x[h..2 * h].copy_from_slice(&a_hidden);

        for l in 0..self.num_layers {
            let mut normed = vec![0.0f32; seq_len * h];
            for t in 0..seq_len {
                let n = layernorm(
                    &x[t * h..(t + 1) * h],
                    &self.layers_norm1_w[l],
                    &self.layers_norm1_b[l],
                    h,
                );
                normed[t * h..(t + 1) * h].copy_from_slice(&n);
            }

            let qkv = matmul_t(&normed, &self.layers_qkv_w[l], seq_len, h, 3 * h);
            let mut q_vec = vec![0.0f32; seq_len * h];
            let mut k_vec = vec![0.0f32; seq_len * h];
            let mut v_vec = vec![0.0f32; seq_len * h];
            for t in 0..seq_len {
                q_vec[t * h..(t + 1) * h].copy_from_slice(&qkv[t * 3 * h..t * 3 * h + h]);
                k_vec[t * h..(t + 1) * h].copy_from_slice(&qkv[t * 3 * h + h..t * 3 * h + 2 * h]);
                v_vec[t * h..(t + 1) * h]
                    .copy_from_slice(&qkv[t * 3 * h + 2 * h..t * 3 * h + 3 * h]);
            }

            let attn_out =
                bidirectional_attention(&q_vec, &k_vec, &v_vec, seq_len, heads, head_dim);
            let projected = matmul_t(&attn_out, &self.layers_out_w[l], seq_len, h, h);

            for i in 0..seq_len * h {
                x[i] += projected[i] + self.layers_out_b[l][i % h];
            }

            let mut normed2 = vec![0.0f32; seq_len * h];
            for t in 0..seq_len {
                let n = layernorm(
                    &x[t * h..(t + 1) * h],
                    &self.layers_norm2_w[l],
                    &self.layers_norm2_b[l],
                    h,
                );
                normed2[t * h..(t + 1) * h].copy_from_slice(&n);
            }

            let up = matmul_t(&normed2, &self.layers_mlp_up_w[l], seq_len, h, inter);
            let mut activated = vec![0.0f32; seq_len * inter];
            for i in 0..seq_len * inter {
                activated[i] = gelu(up[i] + self.layers_mlp_up_b[l][i % inter]);
            }
            let down = matmul_t(&activated, &self.layers_mlp_down_w[l], seq_len, inter, h);

            for i in 0..seq_len * h {
                x[i] += down[i] + self.layers_mlp_down_b[l][i % h];
            }
        }

        let final_hidden = layernorm(&x[..h], &self.final_norm_w, &self.final_norm_b, h);
        let mut out = matmul_t(&final_hidden, &self.output_proj_w, 1, h, self.latent_dim);
        for i in 0..self.latent_dim {
            out[i] += self.output_proj_b[i];
        }
        out
    }

    pub fn rollout(&self, initial_state: &[f32], actions: &[f32], num_steps: usize) -> Vec<f32> {
        let mut states = Vec::with_capacity(num_steps * self.latent_dim);
        let mut state = initial_state.to_vec();
        for step in 0..num_steps {
            let action = &actions[step * self.action_dim..(step + 1) * self.action_dim];
            state = self.predict_next(&state, action);
            states.extend_from_slice(&state);
        }
        states
    }

    pub fn latent_dim(&self) -> usize {
        self.latent_dim
    }
    pub fn action_dim(&self) -> usize {
        self.action_dim
    }
}

/// Geometric attention — also available in WASM.
#[wasm_bindgen]
pub fn geometric_attention_wasm(
    n: usize,
    d: usize,
    q: &[f32],
    k: &[f32],
    v: &[f32],
    positions: &[f32],
    sigma: f32,
) -> Vec<f32> {
    let scale = 1.0 / (d as f32).sqrt();
    let inv_2sigma2 = 1.0 / (2.0 * sigma * sigma);
    let mut out = vec![0.0f32; n * d];

    for i in 0..n {
        let mut scores = vec![0.0f32; n];
        let mut max_score = f32::NEG_INFINITY;

        for j in 0..n {
            let mut dot = 0.0f32;
            for dim in 0..d {
                dot += q[i * d + dim] * k[j * d + dim];
            }

            let mut dist_sq = 0.0f32;
            for dim in 0..3 {
                let diff = positions[i * 3 + dim] - positions[j * 3 + dim];
                dist_sq += diff * diff;
            }

            let score = dot * scale + (-dist_sq * inv_2sigma2).exp();
            scores[j] = score;
            if score > max_score {
                max_score = score;
            }
        }

        softmax(&mut scores);

        for dim in 0..d {
            let mut sum = 0.0f32;
            for j in 0..n {
                sum += scores[j] * v[j * d + dim];
            }
            out[i * d + dim] = sum;
        }
    }
    out
}

// ══════════════════════════════════════════════════════════════════════
// Neo-Unify: Tiny Generative Model (Flow Matching + MoT)
// Architecture from github.com/eren23/neo-unify
// 16x16 RGB, 6 classes, 2.4M params, 128 hidden, 4 heads, 6 blocks
// ══════════════════════════════════════════════════════════════════════

const NU_HIDDEN: usize = 128;
const NU_HEADS: usize = 4;
const NU_HEAD_DIM: usize = NU_HIDDEN / NU_HEADS; // 32
const NU_INTER: usize = 512; // 4 * 128
const NU_BLOCKS: usize = 6;
const NU_PATCH_SIZE: usize = 4;
const NU_IMAGE_SIZE: usize = 16;
const NU_CHANNELS: usize = 3;
const NU_NUM_PATCHES: usize = (NU_IMAGE_SIZE / NU_PATCH_SIZE) * (NU_IMAGE_SIZE / NU_PATCH_SIZE); // 16
const NU_PATCH_DIM: usize = NU_PATCH_SIZE * NU_PATCH_SIZE * NU_CHANNELS; // 48
const NU_NUM_CLASSES: usize = 6;

struct NuBlock {
    ln_attn_w: Vec<f32>,
    ln_attn_b: Vec<f32>,
    qkv_w: Vec<f32>,
    qkv_b: Vec<f32>,
    proj_w: Vec<f32>,
    proj_b: Vec<f32>,
    // Understanding FFN
    ln_und_w: Vec<f32>,
    ln_und_b: Vec<f32>,
    ffn_und_up_w: Vec<f32>,
    ffn_und_up_b: Vec<f32>,
    ffn_und_down_w: Vec<f32>,
    ffn_und_down_b: Vec<f32>,
    // Generation FFN + modulation
    ffn_gen_up_w: Vec<f32>,
    ffn_gen_up_b: Vec<f32>,
    ffn_gen_down_w: Vec<f32>,
    ffn_gen_down_b: Vec<f32>,
    gen_mod_w: Vec<f32>,
    gen_mod_b: Vec<f32>,
}

#[wasm_bindgen]
pub struct NeoUnify {
    patch_proj_w: Vec<f32>,
    patch_proj_b: Vec<f32>,
    pos_emb: Vec<f32>,
    time_mlp_up_w: Vec<f32>,
    time_mlp_up_b: Vec<f32>,
    time_mlp_down_w: Vec<f32>,
    time_mlp_down_b: Vec<f32>,
    class_emb: Vec<f32>,
    blocks: Vec<NuBlock>,
    gen_ln_w: Vec<f32>,
    gen_ln_b: Vec<f32>,
    gen_head_w: Vec<f32>,
    gen_head_b: Vec<f32>,
    und_ln_w: Vec<f32>,
    und_ln_b: Vec<f32>,
    und_head_w: Vec<f32>,
    und_head_b: Vec<f32>,
}

#[wasm_bindgen]
impl NeoUnify {
    #[wasm_bindgen(constructor)]
    pub fn new(data: &[u8]) -> Result<NeoUnify, JsError> {
        // Parse compact binary: [num_tensors:u32] then per tensor: [name_len:u32][name][ndims:u32][shape...][data_len:u32][data_f32]
        let mut weights: HashMap<String, Vec<f32>> = HashMap::new();
        let mut off = 0;

        let read_u32 = |o: &mut usize| -> u32 {
            let v = u32::from_le_bytes([data[*o], data[*o + 1], data[*o + 2], data[*o + 3]]);
            *o += 4;
            v
        };

        let num_tensors = read_u32(&mut off) as usize;
        for _ in 0..num_tensors {
            let name_len = read_u32(&mut off) as usize;
            let name = std::str::from_utf8(&data[off..off + name_len])
                .map_err(|e| JsError::new(&format!("Invalid UTF-8: {}", e)))?
                .to_string();
            off += name_len;

            let ndims = read_u32(&mut off) as usize;
            for _ in 0..ndims {
                let _ = read_u32(&mut off);
            } // skip shape

            let data_len = read_u32(&mut off) as usize;
            let num_floats = data_len / 4;
            let mut tensor = vec![0.0f32; num_floats];
            for i in 0..num_floats {
                tensor[i] = f32::from_le_bytes([
                    data[off + i * 4],
                    data[off + i * 4 + 1],
                    data[off + i * 4 + 2],
                    data[off + i * 4 + 3],
                ]);
            }
            off += data_len;
            weights.insert(name, tensor);
        }

        let get = |name: &str| -> Result<Vec<f32>, JsError> {
            weights
                .get(name)
                .cloned()
                .ok_or_else(|| JsError::new(&format!("Missing weight: {}", name)))
        };

        let mut blocks = Vec::new();
        for i in 0..NU_BLOCKS {
            blocks.push(NuBlock {
                ln_attn_w: get(&format!("blocks.{}.ln_attn.weight", i))?,
                ln_attn_b: get(&format!("blocks.{}.ln_attn.bias", i))?,
                qkv_w: get(&format!("blocks.{}.attn.qkv.weight", i))?,
                qkv_b: get(&format!("blocks.{}.attn.qkv.bias", i))?,
                proj_w: get(&format!("blocks.{}.attn.proj.weight", i))?,
                proj_b: get(&format!("blocks.{}.attn.proj.bias", i))?,
                ln_und_w: get(&format!("blocks.{}.ln_und.weight", i))?,
                ln_und_b: get(&format!("blocks.{}.ln_und.bias", i))?,
                ffn_und_up_w: get(&format!("blocks.{}.ffn_und.0.weight", i))?,
                ffn_und_up_b: get(&format!("blocks.{}.ffn_und.0.bias", i))?,
                ffn_und_down_w: get(&format!("blocks.{}.ffn_und.2.weight", i))?,
                ffn_und_down_b: get(&format!("blocks.{}.ffn_und.2.bias", i))?,
                ffn_gen_up_w: get(&format!("blocks.{}.ffn_gen.0.weight", i))?,
                ffn_gen_up_b: get(&format!("blocks.{}.ffn_gen.0.bias", i))?,
                ffn_gen_down_w: get(&format!("blocks.{}.ffn_gen.2.weight", i))?,
                ffn_gen_down_b: get(&format!("blocks.{}.ffn_gen.2.bias", i))?,
                gen_mod_w: get(&format!("blocks.{}.gen_modulation.1.weight", i))?,
                gen_mod_b: get(&format!("blocks.{}.gen_modulation.1.bias", i))?,
            });
        }

        Ok(NeoUnify {
            patch_proj_w: get("patch_proj.weight")?,
            patch_proj_b: get("patch_proj.bias")?,
            pos_emb: get("pos_emb.weight")?,
            time_mlp_up_w: get("time_emb.mlp.0.weight")?,
            time_mlp_up_b: get("time_emb.mlp.0.bias")?,
            time_mlp_down_w: get("time_emb.mlp.2.weight")?,
            time_mlp_down_b: get("time_emb.mlp.2.bias")?,
            class_emb: get("class_emb.weight")?,
            blocks,
            gen_ln_w: get("gen_ln.weight")?,
            gen_ln_b: get("gen_ln.bias")?,
            gen_head_w: get("gen_head.weight")?,
            gen_head_b: get("gen_head.bias")?,
            und_ln_w: get("und_ln.weight")?,
            und_ln_b: get("und_ln.bias")?,
            und_head_w: get("und_head.weight")?,
            und_head_b: get("und_head.bias")?,
        })
    }

    /// Sinusoidal time embedding + MLP
    fn time_embed(&self, t: f32) -> Vec<f32> {
        let half = NU_HIDDEN / 2;
        let mut emb = vec![0.0f32; NU_HIDDEN];
        for i in 0..half {
            let freq = (-(10000.0f32.ln()) * i as f32 / half as f32).exp();
            emb[i] = (t * freq).sin();
            emb[half + i] = (t * freq).cos();
        }
        // MLP: [128] -> [512] (GELU) -> [128]
        let mut h = matmul_t(&emb, &self.time_mlp_up_w, 1, NU_HIDDEN, NU_INTER);
        for j in 0..NU_INTER {
            h[j] += self.time_mlp_up_b[j];
            h[j] = gelu(h[j]);
        }
        let mut out = matmul_t(&h, &self.time_mlp_down_w, 1, NU_INTER, NU_HIDDEN);
        for j in 0..NU_HIDDEN {
            out[j] += self.time_mlp_down_b[j];
        }
        out
    }

    /// Patchify: [C, H, W] -> [num_patches, patch_dim]
    /// Matches PyTorch: permute(0, 2, 4, 3, 5, 1) → patch order is (py, px, c)
    fn patchify(&self, image: &[f32]) -> Vec<f32> {
        let p = NU_PATCH_SIZE;
        let h = NU_IMAGE_SIZE / p;
        let w = NU_IMAGE_SIZE / p;
        let mut patches = vec![0.0f32; NU_NUM_PATCHES * NU_PATCH_DIM];
        for ph in 0..h {
            for pw in 0..w {
                let patch_idx = ph * w + pw;
                let mut off = 0;
                for py in 0..p {
                    for px in 0..p {
                        for c in 0..NU_CHANNELS {
                            let img_y = ph * p + py;
                            let img_x = pw * p + px;
                            // CHW layout input
                            let img_idx =
                                c * NU_IMAGE_SIZE * NU_IMAGE_SIZE + img_y * NU_IMAGE_SIZE + img_x;
                            patches[patch_idx * NU_PATCH_DIM + off] = if img_idx < image.len() {
                                image[img_idx]
                            } else {
                                0.0
                            };
                            off += 1;
                        }
                    }
                }
            }
        }
        patches
    }

    /// Unpatchify: [num_patches, patch_dim] -> [C, H, W]
    /// Matches PyTorch: permute(0, 5, 1, 3, 2, 4) → patch order is (py, px, c)
    fn unpatchify(&self, patches: &[f32]) -> Vec<f32> {
        let p = NU_PATCH_SIZE;
        let h = NU_IMAGE_SIZE / p;
        let w = NU_IMAGE_SIZE / p;
        let mut image = vec![0.0f32; NU_CHANNELS * NU_IMAGE_SIZE * NU_IMAGE_SIZE];
        for ph in 0..h {
            for pw in 0..w {
                let patch_idx = ph * w + pw;
                let mut off = 0;
                for py in 0..p {
                    for px in 0..p {
                        for c in 0..NU_CHANNELS {
                            let img_y = ph * p + py;
                            let img_x = pw * p + px;
                            let img_idx =
                                c * NU_IMAGE_SIZE * NU_IMAGE_SIZE + img_y * NU_IMAGE_SIZE + img_x;
                            image[img_idx] = patches[patch_idx * NU_PATCH_DIM + off];
                            off += 1;
                        }
                    }
                }
            }
        }
        image
    }

    /// MoT block forward (generation mode with conditioning)
    fn block_forward_gen(&self, x: &[f32], block: &NuBlock, cond: &[f32]) -> Vec<f32> {
        let seq = NU_NUM_PATCHES;
        let d = NU_HIDDEN;

        // 1. Attention: LN -> QKV -> attention -> proj -> residual
        let ln_x = layernorm(x, &block.ln_attn_w, &block.ln_attn_b, d);
        let qkv = {
            let mut q = matmul_t(&ln_x, &block.qkv_w, seq, d, 3 * d);
            add_bias(&mut q, &block.qkv_b, seq, 3 * d);
            q
        };
        let q_slice = &qkv[..seq * d];
        let k_slice = &qkv[seq * d..2 * seq * d];
        let v_slice = &qkv[2 * seq * d..3 * seq * d];
        let attn_out =
            bidirectional_attention(q_slice, k_slice, v_slice, seq, NU_HEADS, NU_HEAD_DIM);
        let mut proj = matmul_t(&attn_out, &block.proj_w, seq, d, d);
        add_bias(&mut proj, &block.proj_b, seq, d);

        // Residual
        let mut h: Vec<f32> = x.iter().zip(proj.iter()).map(|(a, b)| a + b).collect();

        // 2. Generation FFN with adaLN modulation
        // Modulation: SiLU -> Linear -> [shift, scale, gate]
        let mut mod_in = cond.to_vec();
        for v in mod_in.iter_mut() {
            *v = silu(*v);
        }
        let mut mod_out = matmul_t(&mod_in, &block.gen_mod_w, 1, d, 3 * d);
        for j in 0..3 * d {
            mod_out[j] += block.gen_mod_b[j];
        }
        let shift = &mod_out[..d];
        let scale = &mod_out[d..2 * d];
        let gate = &mod_out[2 * d..3 * d];

        // LN (no affine) + modulate
        let ln_h = layernorm_no_bias(&h, &vec![1.0f32; d], d);
        let mut modulated = vec![0.0f32; seq * d];
        for t in 0..seq {
            for j in 0..d {
                modulated[t * d + j] = ln_h[t * d + j] * (1.0 + scale[j]) + shift[j];
            }
        }

        // FFN: up -> GELU -> down
        let mut up = matmul_t(&modulated, &block.ffn_gen_up_w, seq, d, NU_INTER);
        add_bias(&mut up, &block.ffn_gen_up_b, seq, NU_INTER);
        for v in up.iter_mut() {
            *v = gelu(*v);
        }
        let mut down = matmul_t(&up, &block.ffn_gen_down_w, seq, NU_INTER, d);
        add_bias(&mut down, &block.ffn_gen_down_b, seq, d);

        // Gated residual
        for t in 0..seq {
            for j in 0..d {
                h[t * d + j] += gate[j] * down[t * d + j];
            }
        }

        h
    }

    /// Forward generation: noisy image + time + class -> velocity field
    fn forward_generate(&self, x_t: &[f32], t: f32, class_label: usize) -> Vec<f32> {
        let seq = NU_NUM_PATCHES;
        let d = NU_HIDDEN;

        // Patchify + project
        let patches = self.patchify(x_t);
        let mut x = matmul_t(&patches, &self.patch_proj_w, seq, NU_PATCH_DIM, d);
        add_bias(&mut x, &self.patch_proj_b, seq, d);

        // Add position embeddings
        for t_idx in 0..seq {
            for j in 0..d {
                x[t_idx * d + j] += self.pos_emb[t_idx * d + j];
            }
        }

        // Conditioning: time_emb + class_emb
        let t_emb = self.time_embed(t);
        let c_start = class_label * d;
        let mut cond = vec![0.0f32; d];
        for j in 0..d {
            cond[j] = t_emb[j] + self.class_emb[c_start + j];
        }

        // 6 MoT blocks
        for block in &self.blocks {
            x = self.block_forward_gen(&x, block, &cond);
        }

        // Gen head: LN -> Linear -> unpatchify
        let ln_out = layernorm(&x, &self.gen_ln_w, &self.gen_ln_b, d);
        let mut v_patches = matmul_t(&ln_out, &self.gen_head_w, seq, d, NU_PATCH_DIM);
        add_bias(&mut v_patches, &self.gen_head_b, seq, NU_PATCH_DIM);

        self.unpatchify(&v_patches)
    }

    /// Generate an image using RK2 ODE solver with classifier-free guidance.
    /// Returns CHW f32 array [3, 16, 16] = 768 floats.
    pub fn generate(
        &self,
        class_label: usize,
        guidance_scale: f32,
        num_steps: usize,
        seed: u32,
    ) -> Vec<f32> {
        let size = NU_CHANNELS * NU_IMAGE_SIZE * NU_IMAGE_SIZE; // 768
        let null_class = NU_NUM_CLASSES; // 6 = null class for CFG

        // Start from noise (simple PRNG)
        let mut x = vec![0.0f32; size];
        let mut rng_state = seed;
        for v in x.iter_mut() {
            // Box-Muller transform for Gaussian noise
            rng_state = rng_state.wrapping_mul(1664525).wrapping_add(1013904223);
            let u1 = (rng_state as f32 / u32::MAX as f32).max(1e-7);
            rng_state = rng_state.wrapping_mul(1664525).wrapping_add(1013904223);
            let u2 = rng_state as f32 / u32::MAX as f32;
            *v = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos();
        }

        let dt = 1.0 / num_steps as f32;

        for step in 0..num_steps {
            let t = step as f32 * dt;

            // Conditional velocity
            let v_cond = self.forward_generate(&x, t, class_label);
            // Unconditional velocity
            let v_uncond = self.forward_generate(&x, t, null_class);

            // CFG: v = v_uncond + guidance * (v_cond - v_uncond)
            let mut v1 = vec![0.0f32; size];
            for i in 0..size {
                v1[i] = v_uncond[i] + guidance_scale * (v_cond[i] - v_uncond[i]);
            }

            // RK2 midpoint
            let mut x_mid = vec![0.0f32; size];
            for i in 0..size {
                x_mid[i] = x[i] + v1[i] * dt * 0.5;
            }

            let t_mid = t + dt * 0.5;
            let v_cond2 = self.forward_generate(&x_mid, t_mid, class_label);
            let v_uncond2 = self.forward_generate(&x_mid, t_mid, null_class);

            let mut v2 = vec![0.0f32; size];
            for i in 0..size {
                v2[i] = v_uncond2[i] + guidance_scale * (v_cond2[i] - v_uncond2[i]);
            }

            // Update x
            for i in 0..size {
                x[i] += v2[i] * dt;
            }
        }

        // Clamp to [0, 1]
        for v in x.iter_mut() {
            *v = v.clamp(0.0, 1.0);
        }

        x
    }

    /// Generate and return all intermediate steps for visualization.
    /// Returns [num_steps+1, 3, 16, 16] flattened.
    pub fn generate_with_steps(
        &self,
        class_label: usize,
        guidance_scale: f32,
        num_steps: usize,
        seed: u32,
    ) -> Vec<f32> {
        let size = NU_CHANNELS * NU_IMAGE_SIZE * NU_IMAGE_SIZE;
        let null_class = NU_NUM_CLASSES;

        let mut x = vec![0.0f32; size];
        let mut rng_state = seed;
        for v in x.iter_mut() {
            rng_state = rng_state.wrapping_mul(1664525).wrapping_add(1013904223);
            let u1 = (rng_state as f32 / u32::MAX as f32).max(1e-7);
            rng_state = rng_state.wrapping_mul(1664525).wrapping_add(1013904223);
            let u2 = rng_state as f32 / u32::MAX as f32;
            *v = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos();
        }

        let mut all_steps = Vec::with_capacity((num_steps + 1) * size);
        // Save noise
        all_steps.extend_from_slice(&x.iter().map(|v| v.clamp(0.0, 1.0)).collect::<Vec<f32>>());

        let dt = 1.0 / num_steps as f32;

        for step in 0..num_steps {
            let t = step as f32 * dt;

            let v_cond = self.forward_generate(&x, t, class_label);
            let v_uncond = self.forward_generate(&x, t, null_class);

            let mut v1 = vec![0.0f32; size];
            for i in 0..size {
                v1[i] = v_uncond[i] + guidance_scale * (v_cond[i] - v_uncond[i]);
            }

            let mut x_mid = vec![0.0f32; size];
            for i in 0..size {
                x_mid[i] = x[i] + v1[i] * dt * 0.5;
            }

            let t_mid = t + dt * 0.5;
            let v_cond2 = self.forward_generate(&x_mid, t_mid, class_label);
            let v_uncond2 = self.forward_generate(&x_mid, t_mid, null_class);

            for i in 0..size {
                let v2 = v_uncond2[i] + guidance_scale * (v_cond2[i] - v_uncond2[i]);
                x[i] += v2 * dt;
            }

            // Save this step (clamped for visualization)
            all_steps.extend_from_slice(&x.iter().map(|v| v.clamp(0.0, 1.0)).collect::<Vec<f32>>());
        }

        all_steps
    }

    pub fn image_size(&self) -> usize {
        NU_IMAGE_SIZE
    }
    pub fn num_classes(&self) -> usize {
        NU_NUM_CLASSES
    }
}

#[cfg(test)]
mod tests {
    use super::capability_report_json;
    use serde_json::Value;

    #[test]
    fn capability_report_json_identifies_wasm_runtime() {
        let report: Value =
            serde_json::from_str(&capability_report_json()).expect("report should be valid json");
        assert_eq!(report["runtime_profile"], "wasm_portable");
        assert_eq!(report["target"], "wasm32-unknown-unknown");
        assert_eq!(report["native_kernel"], Value::Null);
        assert!(report["backends"]
            .as_array()
            .expect("backends should be an array")
            .iter()
            .any(|backend| backend.as_str() == Some("pure_rust_wasm")));
    }
}
