//! JEPA (Joint Embedding Predictive Architecture) model.
//!
//! Architecture: ViT encoder + narrow predictor transformer.
//! The predictor takes context embeddings and predicts target embeddings
//! in embedding space — no decoder, no token sampling.

use std::collections::HashMap;

use crate::config::{AttentionConfig, FFNConfig, NormConfig};
use crate::ops::matmul::matmul_t;
use crate::ops::norm::apply_norm;
use crate::registry::{create_attention, create_ffn, create_norm, NormVariant};
use crate::weight_loading::{AlignedBuffer, RawTensor, WeightError, WeightMapper};

use super::vit::{EncoderLayer, ViTConfig, ViTModel};

/// Configuration for a JEPA model.
#[derive(Debug, Clone)]
pub struct JEPAConfig {
    /// ViT encoder configuration.
    pub encoder: ViTConfig,
    /// Narrow predictor hidden size (e.g., 384 vs encoder's 768).
    pub predictor_hidden_size: usize,
    /// Shallow predictor depth (e.g., 6 vs encoder's 12).
    pub predictor_num_layers: usize,
    /// Number of attention heads in the predictor.
    pub predictor_num_heads: usize,
}

impl JEPAConfig {
    /// Predictor head dimension.
    pub fn predictor_head_dim(&self) -> usize {
        self.predictor_hidden_size / self.predictor_num_heads
    }

    /// Predictor intermediate (FFN) size — 4x hidden by default.
    pub fn predictor_intermediate_size(&self) -> usize {
        self.predictor_hidden_size * 4
    }
}

/// Output from a JEPA forward pass.
pub struct JEPAOutput {
    /// Context patch embeddings: `[num_context_patches * embed_dim]` (flat).
    pub context_embeddings: Vec<f32>,
    /// Predicted target patch embeddings: `[num_target_patches * embed_dim]` (flat).
    pub predicted_embeddings: Vec<f32>,
}

/// JEPA model: ViT encoder + narrow predictor transformer.
pub struct JEPAModel {
    pub config: JEPAConfig,
    pub encoder: ViTModel,
    pub predictor_layers: Vec<EncoderLayer>,
    pub predictor_norm: Box<dyn NormVariant>,
    pub predictor_norm_weight: AlignedBuffer,
    /// Project encoder embeddings down to predictor dim: `[encoder_dim, predictor_dim]`.
    pub predictor_embed_proj: AlignedBuffer,
    /// Project predictor output back to encoder dim: `[predictor_dim, encoder_dim]`.
    pub predictor_output_proj: AlignedBuffer,
}

impl JEPAModel {
    /// Build a JEPA model from config with zeroed weights.
    pub fn from_config(config: &JEPAConfig) -> Self {
        let norm_config = NormConfig::LayerNorm { eps: 1e-6 };
        let pred_h = config.predictor_hidden_size;
        let pred_head_dim = config.predictor_head_dim();
        let pred_inter = config.predictor_intermediate_size();

        let attn_config = AttentionConfig::Bidirectional {
            num_heads: config.predictor_num_heads,
            head_dim: pred_head_dim,
        };
        let ffn_config = FFNConfig::GELU {
            intermediate_size: pred_inter,
        };

        let mut predictor_layers = Vec::with_capacity(config.predictor_num_layers);
        for _ in 0..config.predictor_num_layers {
            predictor_layers.push(EncoderLayer {
                attn_norm: create_norm(&norm_config),
                attention: create_attention(&attn_config),
                ffn_norm: create_norm(&norm_config),
                ffn: create_ffn(&ffn_config),
                hidden_size: pred_h,
                attn_norm_weight: AlignedBuffer::new_zeroed(0),
                w_q: AlignedBuffer::new_zeroed(0),
                w_k: AlignedBuffer::new_zeroed(0),
                w_v: AlignedBuffer::new_zeroed(0),
                w_o: AlignedBuffer::new_zeroed(0),
                ffn_norm_weight: AlignedBuffer::new_zeroed(0),
                ffn_up: AlignedBuffer::new_zeroed(0),
                ffn_down: AlignedBuffer::new_zeroed(0),
                q_bias: AlignedBuffer::new_zeroed(0),
                k_bias: AlignedBuffer::new_zeroed(0),
                v_bias: AlignedBuffer::new_zeroed(0),
                o_bias: AlignedBuffer::new_zeroed(0),
                ffn_up_bias: AlignedBuffer::new_zeroed(0),
                ffn_down_bias: AlignedBuffer::new_zeroed(0),
                attn_norm_bias: AlignedBuffer::new_zeroed(0),
                ffn_norm_bias: AlignedBuffer::new_zeroed(0),
            });
        }

        // Build ViT encoder with num_classes=0 (no classifier head).
        let mut enc_config = config.encoder.clone();
        enc_config.num_classes = 0;
        let encoder = ViTModel::from_config(&enc_config);

        JEPAModel {
            config: config.clone(),
            encoder,
            predictor_layers,
            predictor_norm: create_norm(&norm_config),
            predictor_norm_weight: AlignedBuffer::new_zeroed(0),
            predictor_embed_proj: AlignedBuffer::new_zeroed(0),
            predictor_output_proj: AlignedBuffer::new_zeroed(0),
        }
    }

    /// Load weights into the ViT encoder from source tensors using a name mapper.
    ///
    /// Delegates to `ViTModel::load_weights()` which already handles the full
    /// ViT weight loading pattern. Use `WeightMapper::dinov2()` for DINOv2
    /// or `WeightMapper::vit()` for standard ViT checkpoints.
    pub fn load_encoder_weights(
        &mut self,
        weights: HashMap<String, RawTensor>,
        mapper: &WeightMapper,
    ) -> Result<crate::models::lm::LoadResult, WeightError> {
        self.encoder.load_weights(weights, mapper)
    }

    /// Forward pass: encode image, then predict target embeddings from context patches.
    ///
    /// `image`: flat `[H * W * C]` pixel data.
    /// `context_mask`: `[num_patches]` — `true` for context patches.
    /// `target_mask`: `[num_patches]` — `true` for target patches.
    ///
    /// Returns context and predicted target embeddings.
    pub fn forward(
        &self,
        image: &[f32],
        h: usize,
        w: usize,
        context_mask: &[bool],
        target_mask: &[bool],
    ) -> JEPAOutput {
        let cfg = &self.config;
        let enc_dim = cfg.encoder.hidden_size;
        let pred_dim = cfg.predictor_hidden_size;

        // 1. Encode full image with ViT → get all patch embeddings [seq_len, enc_dim]
        //    We run the full encoder forward which gives us CLS + patches.
        //    For JEPA we need the patch-level embeddings, not just CLS.
        let all_embeddings = self.encode_patches(image, h, w);
        let num_patches = cfg.encoder.num_patches();

        // 2. Select context patches → [num_context, enc_dim]
        let mut context_embeds = Vec::new();
        for (i, &is_ctx) in context_mask.iter().enumerate().take(num_patches) {
            if is_ctx {
                let start = i * enc_dim;
                context_embeds.extend_from_slice(&all_embeddings[start..start + enc_dim]);
            }
        }
        let num_context = context_embeds.len() / enc_dim;

        // Count target patches
        let num_target = target_mask.iter().take(num_patches).filter(|&&b| b).count();

        // Save context embeddings (in encoder dim) for output
        let context_output = context_embeds.clone();

        // 3. Project context to predictor dimension: [num_context, pred_dim]
        let ctx_projected = if !self.predictor_embed_proj.is_empty() {
            matmul_t(
                &context_embeds,
                &self.predictor_embed_proj,
                num_context,
                enc_dim,
                pred_dim,
            )
        } else {
            // Fallback: truncate or zero-pad
            let mut proj = vec![0.0f32; num_context * pred_dim];
            let copy_dim = enc_dim.min(pred_dim);
            for i in 0..num_context {
                for j in 0..copy_dim {
                    proj[i * pred_dim + j] = context_embeds[i * enc_dim + j];
                }
            }
            proj
        };

        // 4. Build predictor input: context tokens + learnable target query tokens
        //    For simplicity, target queries are zero-initialized (real model would use pos-embed).
        let total_seq = num_context + num_target;
        let mut predictor_input = vec![0.0f32; total_seq * pred_dim];
        // Copy context projections
        predictor_input[..num_context * pred_dim].copy_from_slice(&ctx_projected);
        // Target positions remain zero (positional embeddings would go here in a trained model)

        // 5. Run predictor transformer layers (bidirectional attention)
        let mut x = predictor_input;
        for layer in &self.predictor_layers {
            x = layer.forward(&x, total_seq);
        }

        // 6. Apply final predictor norm
        let normed = apply_norm(
            &x,
            &self.predictor_norm_weight,
            &*self.predictor_norm,
            pred_dim,
        );

        // 7. Extract target positions and project back to encoder dim
        let target_tokens = &normed[num_context * pred_dim..];
        let predicted = if !self.predictor_output_proj.is_empty() {
            matmul_t(
                target_tokens,
                &self.predictor_output_proj,
                num_target,
                pred_dim,
                enc_dim,
            )
        } else {
            // Fallback: zero-pad
            let mut proj = vec![0.0f32; num_target * enc_dim];
            let copy_dim = pred_dim.min(enc_dim);
            for i in 0..num_target {
                for j in 0..copy_dim {
                    proj[i * enc_dim + j] = target_tokens[i * pred_dim + j];
                }
            }
            proj
        };

        JEPAOutput {
            context_embeddings: context_output,
            predicted_embeddings: predicted,
        }
    }

    /// Run the ViT encoder and return patch-level embeddings (without CLS).
    ///
    /// Returns `[num_patches, hidden_size]` (flat).
    fn encode_patches(&self, image: &[f32], height: usize, width: usize) -> Vec<f32> {
        use crate::ops::patch_embed::patch_embed;

        let cfg = &self.config.encoder;
        let enc_dim = cfg.hidden_size;

        // 1. Patch embedding
        let patch_embeddings = patch_embed(
            image,
            height,
            width,
            cfg.channels,
            cfg.patch_size,
            &self.encoder.patch_proj,
            enc_dim,
        );
        let num_patches = cfg.num_patches();
        let seq_len = num_patches + 1; // +1 for CLS

        // 2. Prepend CLS, add pos embed
        let mut x = vec![0.0f32; seq_len * enc_dim];
        if !self.encoder.cls_token.is_empty() {
            x[..enc_dim].copy_from_slice(&self.encoder.cls_token);
        }
        x[enc_dim..].copy_from_slice(&patch_embeddings);

        if !self.encoder.pos_embed.is_empty() {
            let pos_len = self.encoder.pos_embed.len().min(x.len());
            for i in 0..pos_len {
                x[i] += self.encoder.pos_embed[i];
            }
        }

        // 3. Encoder layers
        for layer in &self.encoder.layers {
            x = layer.forward(&x, seq_len);
        }

        // 4. Apply final norm to all tokens
        let normed = apply_norm(
            &x,
            &self.encoder.final_norm_weight,
            &*self.encoder.final_norm,
            enc_dim,
        );

        // 5. Return only patch embeddings (skip CLS at position 0)
        normed[enc_dim..].to_vec()
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

    fn test_jepa_config() -> JEPAConfig {
        JEPAConfig {
            encoder: ViTConfig {
                image_size: 8,
                patch_size: 4,
                channels: 3,
                hidden_size: 32,
                num_layers: 2,
                num_heads: 4,
                intermediate_size: 64,
                num_classes: 0,
            },
            predictor_hidden_size: 16,
            predictor_num_layers: 2,
            predictor_num_heads: 4,
        }
    }

    fn build_test_jepa(cfg: &JEPAConfig) -> JEPAModel {
        let enc_h = cfg.encoder.hidden_size;
        let pred_h = cfg.predictor_hidden_size;
        let pred_inter = cfg.predictor_intermediate_size();
        let patch_dim = cfg.encoder.patch_size * cfg.encoder.patch_size * cfg.encoder.channels;
        let enc_seq_len = cfg.encoder.seq_len();

        let mut model = JEPAModel::from_config(cfg);

        // Set encoder weights
        model.encoder.patch_proj = AlignedBuffer::from_slice(&gen_weights(enc_h * patch_dim, 1));
        model.encoder.cls_token = AlignedBuffer::from_slice(&gen_weights(enc_h, 2));
        model.encoder.pos_embed = AlignedBuffer::from_slice(&gen_weights(enc_seq_len * enc_h, 3));
        model.encoder.final_norm_weight = AlignedBuffer::from_slice(&vec![1.0f32; enc_h]);

        for (i, layer) in model.encoder.layers.iter_mut().enumerate() {
            let s = (i as u32 + 1) * 100;
            layer.attn_norm_weight = AlignedBuffer::from_slice(&vec![1.0f32; enc_h]);
            layer.w_q = AlignedBuffer::from_slice(&gen_weights(enc_h * enc_h, s + 1));
            layer.w_k = AlignedBuffer::from_slice(&gen_weights(enc_h * enc_h, s + 2));
            layer.w_v = AlignedBuffer::from_slice(&gen_weights(enc_h * enc_h, s + 3));
            layer.w_o = AlignedBuffer::from_slice(&gen_weights(enc_h * enc_h, s + 4));
            layer.ffn_norm_weight = AlignedBuffer::from_slice(&vec![1.0f32; enc_h]);
            layer.ffn_up = AlignedBuffer::from_slice(&gen_weights(
                pred_inter.max(cfg.encoder.intermediate_size) * enc_h,
                s + 5,
            ));
            layer.ffn_down = AlignedBuffer::from_slice(&gen_weights(
                enc_h * pred_inter.max(cfg.encoder.intermediate_size),
                s + 6,
            ));
        }

        // Fix encoder FFN sizes to match encoder config
        for (i, layer) in model.encoder.layers.iter_mut().enumerate() {
            let s = (i as u32 + 1) * 100;
            let enc_inter = cfg.encoder.intermediate_size;
            layer.ffn_up = AlignedBuffer::from_slice(&gen_weights(enc_inter * enc_h, s + 5));
            layer.ffn_down = AlignedBuffer::from_slice(&gen_weights(enc_h * enc_inter, s + 6));
        }

        // Set predictor weights
        model.predictor_embed_proj = AlignedBuffer::from_slice(&gen_weights(enc_h * pred_h, 500));
        model.predictor_output_proj = AlignedBuffer::from_slice(&gen_weights(pred_h * enc_h, 501));
        model.predictor_norm_weight = AlignedBuffer::from_slice(&vec![1.0f32; pred_h]);

        for (i, layer) in model.predictor_layers.iter_mut().enumerate() {
            let s = (i as u32 + 10) * 200;
            layer.attn_norm_weight = AlignedBuffer::from_slice(&vec![1.0f32; pred_h]);
            layer.w_q = AlignedBuffer::from_slice(&gen_weights(pred_h * pred_h, s + 1));
            layer.w_k = AlignedBuffer::from_slice(&gen_weights(pred_h * pred_h, s + 2));
            layer.w_v = AlignedBuffer::from_slice(&gen_weights(pred_h * pred_h, s + 3));
            layer.w_o = AlignedBuffer::from_slice(&gen_weights(pred_h * pred_h, s + 4));
            layer.ffn_norm_weight = AlignedBuffer::from_slice(&vec![1.0f32; pred_h]);
            layer.ffn_up = AlignedBuffer::from_slice(&gen_weights(pred_inter * pred_h, s + 5));
            layer.ffn_down = AlignedBuffer::from_slice(&gen_weights(pred_h * pred_inter, s + 6));
        }

        model
    }

    #[test]
    fn test_jepa_forward_produces_finite_embeddings() {
        let cfg = test_jepa_config();
        let model = build_test_jepa(&cfg);

        let num_patches = cfg.encoder.num_patches(); // 4 patches (8/4)^2
        let image: Vec<f32> =
            (0..cfg.encoder.image_size * cfg.encoder.image_size * cfg.encoder.channels)
                .map(|i| (i as f32) / 255.0)
                .collect();

        // Context: first 2 patches, target: last 2 patches
        let mut context_mask = vec![false; num_patches];
        let mut target_mask = vec![false; num_patches];
        context_mask[0] = true;
        context_mask[1] = true;
        target_mask[2] = true;
        target_mask[3] = true;

        let output = model.forward(
            &image,
            cfg.encoder.image_size,
            cfg.encoder.image_size,
            &context_mask,
            &target_mask,
        );

        // Context embeddings: 2 patches * encoder hidden_size
        assert_eq!(
            output.context_embeddings.len(),
            2 * cfg.encoder.hidden_size,
            "Context embeddings should have 2 * encoder_dim elements"
        );
        assert!(
            output.context_embeddings.iter().all(|v| v.is_finite()),
            "JEPA context embeddings contain non-finite values"
        );

        // Predicted embeddings: 2 target patches * encoder hidden_size
        assert_eq!(
            output.predicted_embeddings.len(),
            2 * cfg.encoder.hidden_size,
            "Predicted embeddings should have 2 * encoder_dim elements"
        );
        assert!(
            output.predicted_embeddings.iter().all(|v| v.is_finite()),
            "JEPA predicted embeddings contain non-finite values"
        );
    }

    #[test]
    fn test_jepa_load_encoder_weights() {
        use crate::weight_loading::{RawTensor, WeightMapper};

        let cfg = test_jepa_config();
        let mut model = JEPAModel::from_config(&cfg);
        let enc_h = cfg.encoder.hidden_size;
        let enc_inter = cfg.encoder.intermediate_size;
        let patch_dim = cfg.encoder.patch_size * cfg.encoder.patch_size * cfg.encoder.channels;
        let enc_seq_len = cfg.encoder.seq_len();

        // Build fake weight dict with DINOv2 naming
        let mut weights: HashMap<String, RawTensor> = HashMap::new();

        let rt = |len: usize, seed: u32| -> RawTensor {
            RawTensor {
                data: AlignedBuffer::from_slice(&gen_weights(len, seed)),
                shape: vec![len],
            }
        };
        let ones = |len: usize| -> RawTensor {
            RawTensor {
                data: AlignedBuffer::from_slice(&vec![1.0f32; len]),
                shape: vec![len],
            }
        };

        weights.insert(
            "embeddings.patch_embeddings.projection.weight".into(),
            rt(enc_h * patch_dim, 1),
        );
        weights.insert(
            "embeddings.patch_embeddings.projection.bias".into(),
            rt(enc_h, 2),
        );
        weights.insert("embeddings.cls_token".into(), rt(enc_h, 3));
        weights.insert(
            "embeddings.position_embeddings".into(),
            rt(enc_seq_len * enc_h, 4),
        );
        weights.insert("layernorm.weight".into(), ones(enc_h));
        weights.insert("layernorm.bias".into(), rt(enc_h, 6));

        for i in 0..cfg.encoder.num_layers {
            let s = (i as u32 + 1) * 100;
            weights.insert(
                format!("encoder.layer.{i}.attention.attention.query.weight"),
                rt(enc_h * enc_h, s + 1),
            );
            weights.insert(
                format!("encoder.layer.{i}.attention.attention.query.bias"),
                rt(enc_h, s + 2),
            );
            weights.insert(
                format!("encoder.layer.{i}.attention.attention.key.weight"),
                rt(enc_h * enc_h, s + 3),
            );
            weights.insert(
                format!("encoder.layer.{i}.attention.attention.key.bias"),
                rt(enc_h, s + 4),
            );
            weights.insert(
                format!("encoder.layer.{i}.attention.attention.value.weight"),
                rt(enc_h * enc_h, s + 5),
            );
            weights.insert(
                format!("encoder.layer.{i}.attention.attention.value.bias"),
                rt(enc_h, s + 6),
            );
            weights.insert(
                format!("encoder.layer.{i}.attention.output.dense.weight"),
                rt(enc_h * enc_h, s + 7),
            );
            weights.insert(
                format!("encoder.layer.{i}.attention.output.dense.bias"),
                rt(enc_h, s + 8),
            );
            weights.insert(
                format!("encoder.layer.{i}.intermediate.dense.weight"),
                rt(enc_inter * enc_h, s + 9),
            );
            weights.insert(
                format!("encoder.layer.{i}.intermediate.dense.bias"),
                rt(enc_inter, s + 10),
            );
            weights.insert(
                format!("encoder.layer.{i}.output.dense.weight"),
                rt(enc_h * enc_inter, s + 11),
            );
            weights.insert(
                format!("encoder.layer.{i}.output.dense.bias"),
                rt(enc_h, s + 12),
            );
            weights.insert(format!("encoder.layer.{i}.norm1.weight"), ones(enc_h));
            weights.insert(format!("encoder.layer.{i}.norm1.bias"), rt(enc_h, s + 14));
            weights.insert(format!("encoder.layer.{i}.norm2.weight"), ones(enc_h));
            weights.insert(format!("encoder.layer.{i}.norm2.bias"), rt(enc_h, s + 16));
        }

        let mapper = WeightMapper::dinov2();
        let result = model
            .load_encoder_weights(weights, &mapper)
            .expect("load failed");

        // Verify encoder weights loaded
        assert!(
            !model.encoder.patch_proj.is_empty(),
            "encoder patch_proj should be loaded"
        );
        assert!(
            !model.encoder.cls_token.is_empty(),
            "encoder cls_token should be loaded"
        );
        assert!(
            !model.encoder.pos_embed.is_empty(),
            "encoder pos_embed should be loaded"
        );
        assert!(
            !model.encoder.final_norm_weight.is_empty(),
            "encoder final_norm should be loaded"
        );
        assert!(
            !model.encoder.layers[0].w_q.is_empty(),
            "encoder layer 0 w_q should be loaded"
        );

        // Some keys will be missing (patch_proj_bias is provided but classifier is not expected)
        assert!(
            result.unexpected.is_empty(),
            "Should have no unmapped keys, got: {:?}",
            result.unexpected
        );
    }
}
