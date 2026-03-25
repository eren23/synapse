//! CLIP (Contrastive Language-Image Pre-training) dual-encoder model.
//!
//! Architecture: ViT image encoder + bidirectional text encoder,
//! aligned via learned projection into a shared embedding space.
//! Outputs paired embeddings for image-text similarity scoring.

use crate::config::{AttentionConfig, FFNConfig, NormConfig};
use crate::ops::matmul::matmul_t;
use crate::ops::norm::apply_norm;
use crate::registry::{create_attention, create_ffn, create_norm, NormVariant};
use crate::weight_loading::AlignedBuffer;

use super::vit::{EncoderLayer, ViTConfig, ViTModel};

/// Configuration for a CLIP dual-encoder model.
#[derive(Debug, Clone)]
pub struct CLIPConfig {
    /// Vision encoder (ViT) configuration.
    pub vision: ViTConfig,
    /// Text encoder hidden size.
    pub text_hidden_size: usize,
    /// Number of text encoder layers.
    pub text_num_layers: usize,
    /// Number of text encoder attention heads.
    pub text_num_heads: usize,
    /// Text encoder FFN intermediate size.
    pub text_intermediate_size: usize,
    /// Maximum text sequence length.
    pub text_max_position: usize,
    /// Vocabulary size.
    pub vocab_size: usize,
    /// Shared embedding dimension for alignment.
    pub embed_dim: usize,
}

impl CLIPConfig {
    /// Text encoder head dimension.
    pub fn text_head_dim(&self) -> usize {
        self.text_hidden_size / self.text_num_heads
    }
}

/// Output from a CLIP forward pass.
pub struct CLIPOutput {
    /// Normalized image embedding: `[embed_dim]`.
    pub image_embedding: Vec<f32>,
    /// Normalized text embedding: `[embed_dim]`.
    pub text_embedding: Vec<f32>,
    /// Cosine similarity between image and text embeddings.
    pub similarity: f32,
}

/// CLIP dual-encoder model: ViT image encoder + bidirectional text encoder.
pub struct CLIPModel {
    pub config: CLIPConfig,
    /// ViT vision encoder.
    pub vision_model: ViTModel,
    /// Bidirectional text encoder layers.
    pub text_encoder_layers: Vec<EncoderLayer>,
    /// Token embeddings: `[vocab_size, text_hidden]`.
    pub text_embeddings: AlignedBuffer,
    /// Positional embeddings: `[max_pos, text_hidden]`.
    pub text_pos_embed: AlignedBuffer,
    /// Text encoder final norm.
    pub text_norm: Box<dyn NormVariant>,
    /// Text encoder final norm weight.
    pub text_norm_weight: AlignedBuffer,
    /// Vision projection: `[vision_hidden, embed_dim]`.
    pub vision_proj: AlignedBuffer,
    /// Text projection: `[text_hidden, embed_dim]`.
    pub text_proj: AlignedBuffer,
}

impl CLIPModel {
    /// Build a CLIP model from config with zeroed weights.
    pub fn from_config(config: &CLIPConfig) -> Self {
        let norm_config = NormConfig::LayerNorm { eps: 1e-6 };
        let text_head_dim = config.text_head_dim();

        let attn_config = AttentionConfig::Bidirectional {
            num_heads: config.text_num_heads,
            head_dim: text_head_dim,
        };
        let ffn_config = FFNConfig::GELU {
            intermediate_size: config.text_intermediate_size,
        };

        let mut text_encoder_layers = Vec::with_capacity(config.text_num_layers);
        for _ in 0..config.text_num_layers {
            text_encoder_layers.push(EncoderLayer {
                attn_norm: create_norm(&norm_config),
                attention: create_attention(&attn_config),
                ffn_norm: create_norm(&norm_config),
                ffn: create_ffn(&ffn_config),
                hidden_size: config.text_hidden_size,
                attn_norm_weight: AlignedBuffer::new_zeroed(0),
                w_q: AlignedBuffer::new_zeroed(0),
                w_k: AlignedBuffer::new_zeroed(0),
                w_v: AlignedBuffer::new_zeroed(0),
                w_o: AlignedBuffer::new_zeroed(0),
                ffn_norm_weight: AlignedBuffer::new_zeroed(0),
                ffn_up: AlignedBuffer::new_zeroed(0),
                ffn_down: AlignedBuffer::new_zeroed(0),
            });
        }

        // Vision encoder with no classifier head
        let mut vision_config = config.vision.clone();
        vision_config.num_classes = 0;
        let vision_model = ViTModel::from_config(&vision_config);

        CLIPModel {
            config: config.clone(),
            vision_model,
            text_encoder_layers,
            text_embeddings: AlignedBuffer::new_zeroed(0),
            text_pos_embed: AlignedBuffer::new_zeroed(0),
            text_norm: create_norm(&norm_config),
            text_norm_weight: AlignedBuffer::new_zeroed(0),
            vision_proj: AlignedBuffer::new_zeroed(0),
            text_proj: AlignedBuffer::new_zeroed(0),
        }
    }

    /// Encode an image and return a normalized embedding vector `[embed_dim]`.
    pub fn encode_image(&self, image: &[f32], h: usize, w: usize) -> Vec<f32> {
        let cfg = &self.config;

        // Run ViT forward to get CLS embedding
        let vit_output = self.vision_model.forward_image(image, h, w);
        let vision_hidden = &vit_output.embeddings; // [vision_hidden_size]
        let vh = cfg.vision.hidden_size;
        let ed = cfg.embed_dim;

        // Project to shared embedding space
        let projected = if !self.vision_proj.is_empty() {
            matmul_t(vision_hidden, &self.vision_proj, 1, vh, ed)
        } else {
            // Fallback: truncate or zero-pad
            let mut proj = vec![0.0f32; ed];
            let copy_dim = vh.min(ed);
            proj[..copy_dim].copy_from_slice(&vision_hidden[..copy_dim]);
            proj
        };

        // L2 normalize
        l2_normalize(&projected)
    }

    /// Encode text token IDs and return a normalized embedding vector `[embed_dim]`.
    pub fn encode_text(&self, token_ids: &[u32]) -> Vec<f32> {
        let cfg = &self.config;
        let th = cfg.text_hidden_size;
        let seq_len = token_ids.len();
        let ed = cfg.embed_dim;

        // 1. Token embeddings: lookup each token
        let mut x = vec![0.0f32; seq_len * th];
        if !self.text_embeddings.is_empty() {
            for (t, &tid) in token_ids.iter().enumerate() {
                let idx = tid as usize;
                if idx < cfg.vocab_size {
                    let src_start = idx * th;
                    let src_end = src_start + th;
                    if src_end <= self.text_embeddings.len() {
                        x[t * th..(t + 1) * th].copy_from_slice(&self.text_embeddings[src_start..src_end]);
                    }
                }
            }
        }

        // 2. Add positional embeddings
        if !self.text_pos_embed.is_empty() {
            let pos_len = (seq_len * th).min(self.text_pos_embed.len());
            for i in 0..pos_len {
                x[i] += self.text_pos_embed[i];
            }
        }

        // 3. Run through text encoder layers (bidirectional)
        for layer in &self.text_encoder_layers {
            x = layer.forward(&x, seq_len);
        }

        // 4. Apply final norm
        let normed = apply_norm(&x, &self.text_norm_weight, &*self.text_norm, th);

        // 5. Take EOS token (last token in sequence)
        let eos_start = (seq_len - 1) * th;
        let eos_hidden = &normed[eos_start..eos_start + th];

        // 6. Project to shared embedding space
        let projected = if !self.text_proj.is_empty() {
            matmul_t(eos_hidden, &self.text_proj, 1, th, ed)
        } else {
            let mut proj = vec![0.0f32; ed];
            let copy_dim = th.min(ed);
            proj[..copy_dim].copy_from_slice(&eos_hidden[..copy_dim]);
            proj
        };

        // 7. L2 normalize
        l2_normalize(&projected)
    }

    /// Compute image-text similarity using both encoders.
    pub fn similarity(&self, image: &[f32], h: usize, w: usize, token_ids: &[u32]) -> CLIPOutput {
        let image_embedding = self.encode_image(image, h, w);
        let text_embedding = self.encode_text(token_ids);

        // Cosine similarity (embeddings are already L2-normalized)
        let sim: f32 = image_embedding
            .iter()
            .zip(text_embedding.iter())
            .map(|(a, b)| a * b)
            .sum();

        CLIPOutput {
            image_embedding,
            text_embedding,
            similarity: sim,
        }
    }
}

/// L2-normalize a vector in place and return it.
fn l2_normalize(v: &[f32]) -> Vec<f32> {
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 1e-12 {
        v.iter().map(|x| x / norm).collect()
    } else {
        v.to_vec()
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

    fn test_clip_config() -> CLIPConfig {
        CLIPConfig {
            vision: ViTConfig {
                image_size: 8,
                patch_size: 4,
                channels: 3,
                hidden_size: 32,
                num_layers: 2,
                num_heads: 4,
                intermediate_size: 64,
                num_classes: 0,
            },
            text_hidden_size: 32,
            text_num_layers: 2,
            text_num_heads: 4,
            text_intermediate_size: 64,
            text_max_position: 16,
            vocab_size: 50,
            embed_dim: 16,
        }
    }

    fn build_test_clip(cfg: &CLIPConfig) -> CLIPModel {
        let vh = cfg.vision.hidden_size;
        let th = cfg.text_hidden_size;
        let ed = cfg.embed_dim;
        let v_inter = cfg.vision.intermediate_size;
        let t_inter = cfg.text_intermediate_size;
        let patch_dim = cfg.vision.patch_size * cfg.vision.patch_size * cfg.vision.channels;
        let v_seq_len = cfg.vision.seq_len();

        let mut model = CLIPModel::from_config(cfg);

        // Vision encoder weights
        model.vision_model.patch_proj = AlignedBuffer::from_slice(&gen_weights(vh * patch_dim, 1));
        model.vision_model.cls_token = AlignedBuffer::from_slice(&gen_weights(vh, 2));
        model.vision_model.pos_embed = AlignedBuffer::from_slice(&gen_weights(v_seq_len * vh, 3));
        model.vision_model.final_norm_weight = AlignedBuffer::from_slice(&vec![1.0f32; vh]);

        for (i, layer) in model.vision_model.layers.iter_mut().enumerate() {
            let s = (i as u32 + 1) * 100;
            layer.attn_norm_weight = AlignedBuffer::from_slice(&vec![1.0f32; vh]);
            layer.w_q = AlignedBuffer::from_slice(&gen_weights(vh * vh, s + 1));
            layer.w_k = AlignedBuffer::from_slice(&gen_weights(vh * vh, s + 2));
            layer.w_v = AlignedBuffer::from_slice(&gen_weights(vh * vh, s + 3));
            layer.w_o = AlignedBuffer::from_slice(&gen_weights(vh * vh, s + 4));
            layer.ffn_norm_weight = AlignedBuffer::from_slice(&vec![1.0f32; vh]);
            layer.ffn_up = AlignedBuffer::from_slice(&gen_weights(v_inter * vh, s + 5));
            layer.ffn_down = AlignedBuffer::from_slice(&gen_weights(vh * v_inter, s + 6));
        }

        // Text encoder weights
        model.text_embeddings = AlignedBuffer::from_slice(&gen_weights(cfg.vocab_size * th, 400));
        model.text_pos_embed = AlignedBuffer::from_slice(&gen_weights(cfg.text_max_position * th, 401));
        model.text_norm_weight = AlignedBuffer::from_slice(&vec![1.0f32; th]);

        for (i, layer) in model.text_encoder_layers.iter_mut().enumerate() {
            let s = (i as u32 + 10) * 200;
            layer.attn_norm_weight = AlignedBuffer::from_slice(&vec![1.0f32; th]);
            layer.w_q = AlignedBuffer::from_slice(&gen_weights(th * th, s + 1));
            layer.w_k = AlignedBuffer::from_slice(&gen_weights(th * th, s + 2));
            layer.w_v = AlignedBuffer::from_slice(&gen_weights(th * th, s + 3));
            layer.w_o = AlignedBuffer::from_slice(&gen_weights(th * th, s + 4));
            layer.ffn_norm_weight = AlignedBuffer::from_slice(&vec![1.0f32; th]);
            layer.ffn_up = AlignedBuffer::from_slice(&gen_weights(t_inter * th, s + 5));
            layer.ffn_down = AlignedBuffer::from_slice(&gen_weights(th * t_inter, s + 6));
        }

        // Projection weights
        model.vision_proj = AlignedBuffer::from_slice(&gen_weights(vh * ed, 600));
        model.text_proj = AlignedBuffer::from_slice(&gen_weights(th * ed, 601));

        model
    }

    #[test]
    fn test_clip_image_embedding_finite() {
        let cfg = test_clip_config();
        let model = build_test_clip(&cfg);

        let image: Vec<f32> = (0..cfg.vision.image_size * cfg.vision.image_size * cfg.vision.channels)
            .map(|i| (i as f32) / 255.0)
            .collect();

        let embedding = model.encode_image(&image, cfg.vision.image_size, cfg.vision.image_size);

        assert_eq!(
            embedding.len(),
            cfg.embed_dim,
            "Image embedding should have embed_dim elements"
        );
        assert!(
            embedding.iter().all(|v| v.is_finite()),
            "CLIP image embedding contains non-finite values"
        );

        // Should be L2-normalized (norm ~= 1.0)
        let norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!(
            (norm - 1.0).abs() < 1e-4,
            "Image embedding should be L2-normalized, got norm = {norm}"
        );
    }

    #[test]
    fn test_clip_text_embedding_finite() {
        let cfg = test_clip_config();
        let model = build_test_clip(&cfg);

        let token_ids: Vec<u32> = vec![1, 5, 10, 20, 3]; // fake token sequence

        let embedding = model.encode_text(&token_ids);

        assert_eq!(
            embedding.len(),
            cfg.embed_dim,
            "Text embedding should have embed_dim elements"
        );
        assert!(
            embedding.iter().all(|v| v.is_finite()),
            "CLIP text embedding contains non-finite values"
        );

        // Should be L2-normalized
        let norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!(
            (norm - 1.0).abs() < 1e-4,
            "Text embedding should be L2-normalized, got norm = {norm}"
        );
    }

    #[test]
    fn test_clip_similarity_in_range() {
        let cfg = test_clip_config();
        let model = build_test_clip(&cfg);

        let image: Vec<f32> = (0..cfg.vision.image_size * cfg.vision.image_size * cfg.vision.channels)
            .map(|i| (i as f32) / 255.0)
            .collect();
        let token_ids: Vec<u32> = vec![1, 5, 10, 20, 3];

        let output = model.similarity(&image, cfg.vision.image_size, cfg.vision.image_size, &token_ids);

        assert!(
            output.similarity.is_finite(),
            "Similarity should be finite"
        );
        assert!(
            output.similarity >= -1.0 && output.similarity <= 1.0,
            "Cosine similarity should be in [-1, 1], got {}",
            output.similarity
        );
    }
}
