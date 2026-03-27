//! CLIP (Contrastive Language-Image Pre-training) dual-encoder model.
//!
//! Architecture: ViT image encoder + bidirectional text encoder,
//! aligned via learned projection into a shared embedding space.
//! Outputs paired embeddings for image-text similarity scoring.

use std::collections::{HashMap, HashSet};

use crate::config::{AttentionConfig, FFNConfig, NormConfig};
use crate::ops::matmul::matmul_t;
use crate::ops::norm::apply_norm;
use crate::registry::{create_attention, create_ffn, create_norm, NormVariant};
use crate::weight_loading::{AlignedBuffer, RawTensor, WeightError, WeightMapper};

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
    /// Text encoder final norm bias.
    pub text_norm_bias: AlignedBuffer,
    /// Vision pre-layernorm weight (CLIP applies LN before the encoder).
    pub vision_pre_norm_weight: AlignedBuffer,
    /// Vision pre-layernorm bias.
    pub vision_pre_norm_bias: AlignedBuffer,
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
            text_norm_bias: AlignedBuffer::new_zeroed(0),
            vision_pre_norm_weight: AlignedBuffer::new_zeroed(0),
            vision_pre_norm_bias: AlignedBuffer::new_zeroed(0),
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
                        x[t * th..(t + 1) * th]
                            .copy_from_slice(&self.text_embeddings[src_start..src_end]);
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
        let mut normed = apply_norm(&x, &self.text_norm_weight, &*self.text_norm, th);
        if !self.text_norm_bias.is_empty() {
            for t in 0..seq_len {
                for j in 0..th {
                    normed[t * th + j] += self.text_norm_bias[j];
                }
            }
        }

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

    /// All target weight keys this model expects (Synapse naming).
    pub fn expected_weight_keys(&self) -> Vec<String> {
        let mut keys = Vec::new();

        // Vision encoder keys (prefixed with "vision.")
        keys.push("vision.patch_proj".to_string());
        keys.push("vision.cls_token".to_string());
        keys.push("vision.pos_embed".to_string());
        keys.push("vision.pre_norm.weight".to_string());
        keys.push("vision.pre_norm.bias".to_string());
        for (i, layer) in self.vision_model.layers.iter().enumerate() {
            for k in layer.weight_keys(i) {
                keys.push(format!("vision.{k}"));
            }
        }
        keys.push("vision.norm.weight".to_string());
        keys.push("vision.norm.bias".to_string());

        // Text encoder keys (prefixed with "text.")
        keys.push("text.embeddings".to_string());
        keys.push("text.pos_embed".to_string());
        for (i, layer) in self.text_encoder_layers.iter().enumerate() {
            for k in layer.weight_keys(i) {
                keys.push(format!("text.{k}"));
            }
        }
        keys.push("text.norm.weight".to_string());
        keys.push("text.norm.bias".to_string());

        // Projection weights
        keys.push("vision_proj".to_string());
        keys.push("text_proj".to_string());

        keys
    }

    /// Load weights from source tensors using a name mapper.
    ///
    /// Follows the same pattern as `ViTModel::load_weights()`.
    pub fn load_weights(
        &mut self,
        weights: HashMap<String, RawTensor>,
        mapper: &WeightMapper,
    ) -> Result<super::LoadResult, WeightError> {
        let source_keys: Vec<String> = weights.keys().cloned().collect();
        let mapping = mapper.map_keys(&source_keys);

        let expected: HashSet<String> = self.expected_weight_keys().into_iter().collect();
        let mut loaded = HashSet::new();

        for (source, target) in &mapping.mapping {
            if let Some(raw) = weights.get(source) {
                match self.set_weight(target, raw) {
                    Ok(()) => {
                        if expected.contains(target) {
                            loaded.insert(target.clone());
                        }
                    }
                    Err(e) => eprintln!("Warning: failed to set {target}: {e}"),
                }
            }
        }

        let missing: Vec<String> = expected.difference(&loaded).cloned().collect();
        let unexpected = mapping.unmapped;

        Ok(super::LoadResult {
            missing,
            unexpected,
        })
    }

    /// Assign a weight by its Synapse target key.
    fn set_weight(&mut self, key: &str, tensor: &RawTensor) -> Result<(), WeightError> {
        if let Some(rest) = key.strip_prefix("vision.") {
            self.set_vision_weight(rest, tensor)
        } else if let Some(rest) = key.strip_prefix("text.") {
            self.set_text_weight(rest, tensor)
        } else {
            match key {
                "vision_proj" => {
                    self.vision_proj = tensor.data.clone();
                    Ok(())
                }
                "text_proj" => {
                    self.text_proj = tensor.data.clone();
                    Ok(())
                }
                _ => Err(WeightError::InvalidFormat(format!(
                    "Unknown CLIP key: {key}"
                ))),
            }
        }
    }

    /// Set a vision encoder weight. `key` is after stripping the `"vision."` prefix.
    fn set_vision_weight(&mut self, key: &str, tensor: &RawTensor) -> Result<(), WeightError> {
        match key {
            "patch_proj" => self.vision_model.patch_proj = tensor.data.clone(),
            "cls_token" => self.vision_model.cls_token = tensor.data.clone(),
            "pos_embed" => self.vision_model.pos_embed = tensor.data.clone(),
            "pre_norm.weight" => self.vision_pre_norm_weight = tensor.data.clone(),
            "pre_norm.bias" => self.vision_pre_norm_bias = tensor.data.clone(),
            "norm.weight" => self.vision_model.final_norm_weight = tensor.data.clone(),
            "norm.bias" => self.vision_model.final_norm_bias = tensor.data.clone(),
            _ if key.starts_with("layers[") => {
                if let Some((idx, field)) = parse_layer_key(key) {
                    if let Some(layer) = self.vision_model.layers.get_mut(idx) {
                        layer.set_weight(field, tensor)?;
                    }
                }
            }
            _ => {}
        }
        Ok(())
    }

    /// Set a text encoder weight. `key` is after stripping the `"text."` prefix.
    fn set_text_weight(&mut self, key: &str, tensor: &RawTensor) -> Result<(), WeightError> {
        match key {
            "embeddings" => self.text_embeddings = tensor.data.clone(),
            "pos_embed" => self.text_pos_embed = tensor.data.clone(),
            "norm.weight" => self.text_norm_weight = tensor.data.clone(),
            "norm.bias" => self.text_norm_bias = tensor.data.clone(),
            _ if key.starts_with("layers[") => {
                if let Some((idx, field)) = parse_layer_key(key) {
                    if let Some(layer) = self.text_encoder_layers.get_mut(idx) {
                        layer.set_weight(field, tensor)?;
                    }
                }
            }
            _ => {}
        }
        Ok(())
    }
}

/// Parse `"layers[5].attention.w_q"` → `(5, "attention.w_q")`.
fn parse_layer_key(key: &str) -> Option<(usize, &str)> {
    let rest = key.strip_prefix("layers[")?;
    let bracket = rest.find(']')?;
    let idx: usize = rest[..bracket].parse().ok()?;
    let field = rest[bracket + 1..].strip_prefix('.')?;
    Some((idx, field))
}

/// Parse a HuggingFace CLIP `config.json` into a [`CLIPConfig`].
///
/// Reads `vision_config` and `text_config` sub-objects to build both encoders.
pub fn parse_clip_config(path: &std::path::Path) -> Result<CLIPConfig, Box<dyn std::error::Error>> {
    let json_str = std::fs::read_to_string(path)?;
    parse_clip_config_json(&json_str)
}

/// Parse a CLIP config from a JSON string.
pub fn parse_clip_config_json(json: &str) -> Result<CLIPConfig, Box<dyn std::error::Error>> {
    let v: serde_json::Value = serde_json::from_str(json)?;

    let vc = &v["vision_config"];
    let tc = &v["text_config"];

    let vision_hidden = vc["hidden_size"].as_u64().unwrap_or(768) as usize;
    let vision_layers = vc["num_hidden_layers"].as_u64().unwrap_or(12) as usize;
    let vision_heads = vc["num_attention_heads"].as_u64().unwrap_or(12) as usize;
    let vision_inter = vc["intermediate_size"].as_u64().unwrap_or(3072) as usize;
    let image_size = vc["image_size"].as_u64().unwrap_or(224) as usize;
    let patch_size = vc["patch_size"].as_u64().unwrap_or(32) as usize;
    let channels = vc["num_channels"].as_u64().unwrap_or(3) as usize;

    let text_hidden = tc["hidden_size"].as_u64().unwrap_or(512) as usize;
    let text_layers = tc["num_hidden_layers"].as_u64().unwrap_or(12) as usize;
    let text_heads = tc["num_attention_heads"].as_u64().unwrap_or(8) as usize;
    let text_inter = tc["intermediate_size"].as_u64().unwrap_or(2048) as usize;
    let text_max_pos = tc["max_position_embeddings"].as_u64().unwrap_or(77) as usize;
    let vocab_size = tc["vocab_size"].as_u64().unwrap_or(49408) as usize;

    let embed_dim = v["projection_dim"].as_u64().unwrap_or(512) as usize;

    Ok(CLIPConfig {
        vision: ViTConfig {
            image_size,
            patch_size,
            channels,
            hidden_size: vision_hidden,
            num_layers: vision_layers,
            num_heads: vision_heads,
            intermediate_size: vision_inter,
            num_classes: 0,
        },
        text_hidden_size: text_hidden,
        text_num_layers: text_layers,
        text_num_heads: text_heads,
        text_intermediate_size: text_inter,
        text_max_position: text_max_pos,
        vocab_size,
        embed_dim,
    })
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
        model.text_pos_embed =
            AlignedBuffer::from_slice(&gen_weights(cfg.text_max_position * th, 401));
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

        let image: Vec<f32> =
            (0..cfg.vision.image_size * cfg.vision.image_size * cfg.vision.channels)
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

        let image: Vec<f32> =
            (0..cfg.vision.image_size * cfg.vision.image_size * cfg.vision.channels)
                .map(|i| (i as f32) / 255.0)
                .collect();
        let token_ids: Vec<u32> = vec![1, 5, 10, 20, 3];

        let output = model.similarity(
            &image,
            cfg.vision.image_size,
            cfg.vision.image_size,
            &token_ids,
        );

        assert!(output.similarity.is_finite(), "Similarity should be finite");
        assert!(
            output.similarity >= -1.0 && output.similarity <= 1.0,
            "Cosine similarity should be in [-1, 1], got {}",
            output.similarity
        );
    }

    #[test]
    fn test_clip_load_weights_via_mapper() {
        use crate::weight_loading::{RawTensor, WeightMapper};

        let cfg = test_clip_config();
        let mut model = CLIPModel::from_config(&cfg);
        let vh = cfg.vision.hidden_size;
        let th = cfg.text_hidden_size;
        let ed = cfg.embed_dim;
        let v_inter = cfg.vision.intermediate_size;
        let t_inter = cfg.text_intermediate_size;
        let patch_dim = cfg.vision.patch_size * cfg.vision.patch_size * cfg.vision.channels;
        let v_seq_len = cfg.vision.seq_len();

        // Build a fake weight dict with HuggingFace CLIP naming
        let mut weights: HashMap<String, RawTensor> = HashMap::new();

        let rt = |len: usize, seed: u32| -> RawTensor {
            RawTensor {
                data: AlignedBuffer::from_slice(&gen_weights(len, seed)),
                shape: vec![len],
            }
        };

        // Vision global weights
        weights.insert(
            "vision_model.embeddings.patch_embedding.weight".into(),
            rt(vh * patch_dim, 1),
        );
        weights.insert("vision_model.embeddings.class_embedding".into(), rt(vh, 2));
        weights.insert(
            "vision_model.embeddings.position_embedding.weight".into(),
            rt(v_seq_len * vh, 3),
        );
        weights.insert("vision_model.pre_layernorm.weight".into(), rt(vh, 4));
        weights.insert("vision_model.pre_layernorm.bias".into(), rt(vh, 5));
        weights.insert("vision_model.post_layernorm.weight".into(), rt(vh, 6));
        weights.insert("vision_model.post_layernorm.bias".into(), rt(vh, 7));

        // Vision layer weights
        for i in 0..cfg.vision.num_layers {
            let s = (i as u32 + 1) * 100;
            weights.insert(
                format!("vision_model.encoder.layers.{i}.self_attn.q_proj.weight"),
                rt(vh * vh, s + 1),
            );
            weights.insert(
                format!("vision_model.encoder.layers.{i}.self_attn.q_proj.bias"),
                rt(vh, s + 2),
            );
            weights.insert(
                format!("vision_model.encoder.layers.{i}.self_attn.k_proj.weight"),
                rt(vh * vh, s + 3),
            );
            weights.insert(
                format!("vision_model.encoder.layers.{i}.self_attn.k_proj.bias"),
                rt(vh, s + 4),
            );
            weights.insert(
                format!("vision_model.encoder.layers.{i}.self_attn.v_proj.weight"),
                rt(vh * vh, s + 5),
            );
            weights.insert(
                format!("vision_model.encoder.layers.{i}.self_attn.v_proj.bias"),
                rt(vh, s + 6),
            );
            weights.insert(
                format!("vision_model.encoder.layers.{i}.self_attn.out_proj.weight"),
                rt(vh * vh, s + 7),
            );
            weights.insert(
                format!("vision_model.encoder.layers.{i}.self_attn.out_proj.bias"),
                rt(vh, s + 8),
            );
            weights.insert(
                format!("vision_model.encoder.layers.{i}.mlp.fc1.weight"),
                rt(v_inter * vh, s + 9),
            );
            weights.insert(
                format!("vision_model.encoder.layers.{i}.mlp.fc1.bias"),
                rt(v_inter, s + 10),
            );
            weights.insert(
                format!("vision_model.encoder.layers.{i}.mlp.fc2.weight"),
                rt(vh * v_inter, s + 11),
            );
            weights.insert(
                format!("vision_model.encoder.layers.{i}.mlp.fc2.bias"),
                rt(vh, s + 12),
            );
            weights.insert(
                format!("vision_model.encoder.layers.{i}.layer_norm1.weight"),
                rt(vh, s + 13),
            );
            weights.insert(
                format!("vision_model.encoder.layers.{i}.layer_norm1.bias"),
                rt(vh, s + 14),
            );
            weights.insert(
                format!("vision_model.encoder.layers.{i}.layer_norm2.weight"),
                rt(vh, s + 15),
            );
            weights.insert(
                format!("vision_model.encoder.layers.{i}.layer_norm2.bias"),
                rt(vh, s + 16),
            );
        }

        // Text global weights
        weights.insert(
            "text_model.embeddings.token_embedding.weight".into(),
            rt(cfg.vocab_size * th, 400),
        );
        weights.insert(
            "text_model.embeddings.position_embedding.weight".into(),
            rt(cfg.text_max_position * th, 401),
        );
        weights.insert("text_model.final_layer_norm.weight".into(), rt(th, 402));
        weights.insert("text_model.final_layer_norm.bias".into(), rt(th, 403));

        // Text layer weights
        for i in 0..cfg.text_num_layers {
            let s = (i as u32 + 10) * 200;
            weights.insert(
                format!("text_model.encoder.layers.{i}.self_attn.q_proj.weight"),
                rt(th * th, s + 1),
            );
            weights.insert(
                format!("text_model.encoder.layers.{i}.self_attn.q_proj.bias"),
                rt(th, s + 2),
            );
            weights.insert(
                format!("text_model.encoder.layers.{i}.self_attn.k_proj.weight"),
                rt(th * th, s + 3),
            );
            weights.insert(
                format!("text_model.encoder.layers.{i}.self_attn.k_proj.bias"),
                rt(th, s + 4),
            );
            weights.insert(
                format!("text_model.encoder.layers.{i}.self_attn.v_proj.weight"),
                rt(th * th, s + 5),
            );
            weights.insert(
                format!("text_model.encoder.layers.{i}.self_attn.v_proj.bias"),
                rt(th, s + 6),
            );
            weights.insert(
                format!("text_model.encoder.layers.{i}.self_attn.out_proj.weight"),
                rt(th * th, s + 7),
            );
            weights.insert(
                format!("text_model.encoder.layers.{i}.self_attn.out_proj.bias"),
                rt(th, s + 8),
            );
            weights.insert(
                format!("text_model.encoder.layers.{i}.mlp.fc1.weight"),
                rt(t_inter * th, s + 9),
            );
            weights.insert(
                format!("text_model.encoder.layers.{i}.mlp.fc1.bias"),
                rt(t_inter, s + 10),
            );
            weights.insert(
                format!("text_model.encoder.layers.{i}.mlp.fc2.weight"),
                rt(th * t_inter, s + 11),
            );
            weights.insert(
                format!("text_model.encoder.layers.{i}.mlp.fc2.bias"),
                rt(th, s + 12),
            );
            weights.insert(
                format!("text_model.encoder.layers.{i}.layer_norm1.weight"),
                rt(th, s + 13),
            );
            weights.insert(
                format!("text_model.encoder.layers.{i}.layer_norm1.bias"),
                rt(th, s + 14),
            );
            weights.insert(
                format!("text_model.encoder.layers.{i}.layer_norm2.weight"),
                rt(th, s + 15),
            );
            weights.insert(
                format!("text_model.encoder.layers.{i}.layer_norm2.bias"),
                rt(th, s + 16),
            );
        }

        // Projections
        weights.insert("visual_projection.weight".into(), rt(vh * ed, 600));
        weights.insert("text_projection.weight".into(), rt(th * ed, 601));

        let mapper = WeightMapper::clip();
        let result = model
            .load_weights(weights, &mapper)
            .expect("load_weights failed");

        // Verify key weights were loaded (non-empty)
        assert!(
            !model.vision_model.patch_proj.is_empty(),
            "vision patch_proj should be loaded"
        );
        assert!(
            !model.vision_model.cls_token.is_empty(),
            "vision cls_token should be loaded"
        );
        assert!(
            !model.vision_model.pos_embed.is_empty(),
            "vision pos_embed should be loaded"
        );
        assert!(
            !model.vision_pre_norm_weight.is_empty(),
            "vision pre_norm_weight should be loaded"
        );
        assert!(
            !model.vision_model.final_norm_weight.is_empty(),
            "vision norm weight should be loaded"
        );
        assert!(
            !model.text_embeddings.is_empty(),
            "text embeddings should be loaded"
        );
        assert!(
            !model.text_pos_embed.is_empty(),
            "text pos_embed should be loaded"
        );
        assert!(
            !model.text_norm_weight.is_empty(),
            "text norm weight should be loaded"
        );
        assert!(
            !model.text_norm_bias.is_empty(),
            "text norm bias should be loaded"
        );
        assert!(
            !model.vision_proj.is_empty(),
            "vision_proj should be loaded"
        );
        assert!(!model.text_proj.is_empty(), "text_proj should be loaded");
        assert!(
            !model.vision_model.layers[0].w_q.is_empty(),
            "vision layer 0 w_q should be loaded"
        );
        assert!(
            !model.text_encoder_layers[0].w_q.is_empty(),
            "text layer 0 w_q should be loaded"
        );

        // Should have no unmapped keys (everything is CLIP naming)
        assert!(
            result.unexpected.is_empty(),
            "Should have no unmapped keys, got: {:?}",
            result.unexpected
        );
    }

    #[test]
    fn test_clip_load_weights_then_forward() {
        use crate::weight_loading::{RawTensor, WeightMapper};

        let cfg = test_clip_config();
        let mut model = CLIPModel::from_config(&cfg);
        let vh = cfg.vision.hidden_size;
        let th = cfg.text_hidden_size;
        let ed = cfg.embed_dim;
        let v_inter = cfg.vision.intermediate_size;
        let t_inter = cfg.text_intermediate_size;
        let patch_dim = cfg.vision.patch_size * cfg.vision.patch_size * cfg.vision.channels;
        let v_seq_len = cfg.vision.seq_len();

        let mut weights: HashMap<String, RawTensor> = HashMap::new();

        let rt = |len: usize, seed: u32| -> RawTensor {
            RawTensor {
                data: AlignedBuffer::from_slice(&gen_weights(len, seed)),
                shape: vec![len],
            }
        };

        // All required weights (same as above, using unit norm weights for stability)
        let ones = |len: usize| -> RawTensor {
            RawTensor {
                data: AlignedBuffer::from_slice(&vec![1.0f32; len]),
                shape: vec![len],
            }
        };

        weights.insert(
            "vision_model.embeddings.patch_embedding.weight".into(),
            rt(vh * patch_dim, 1),
        );
        weights.insert("vision_model.embeddings.class_embedding".into(), rt(vh, 2));
        weights.insert(
            "vision_model.embeddings.position_embedding.weight".into(),
            rt(v_seq_len * vh, 3),
        );
        weights.insert("vision_model.pre_layernorm.weight".into(), ones(vh));
        weights.insert("vision_model.pre_layernorm.bias".into(), rt(vh, 5));
        weights.insert("vision_model.post_layernorm.weight".into(), ones(vh));
        weights.insert("vision_model.post_layernorm.bias".into(), rt(vh, 7));

        for i in 0..cfg.vision.num_layers {
            let s = (i as u32 + 1) * 100;
            weights.insert(
                format!("vision_model.encoder.layers.{i}.self_attn.q_proj.weight"),
                rt(vh * vh, s + 1),
            );
            weights.insert(
                format!("vision_model.encoder.layers.{i}.self_attn.q_proj.bias"),
                rt(vh, s + 2),
            );
            weights.insert(
                format!("vision_model.encoder.layers.{i}.self_attn.k_proj.weight"),
                rt(vh * vh, s + 3),
            );
            weights.insert(
                format!("vision_model.encoder.layers.{i}.self_attn.k_proj.bias"),
                rt(vh, s + 4),
            );
            weights.insert(
                format!("vision_model.encoder.layers.{i}.self_attn.v_proj.weight"),
                rt(vh * vh, s + 5),
            );
            weights.insert(
                format!("vision_model.encoder.layers.{i}.self_attn.v_proj.bias"),
                rt(vh, s + 6),
            );
            weights.insert(
                format!("vision_model.encoder.layers.{i}.self_attn.out_proj.weight"),
                rt(vh * vh, s + 7),
            );
            weights.insert(
                format!("vision_model.encoder.layers.{i}.self_attn.out_proj.bias"),
                rt(vh, s + 8),
            );
            weights.insert(
                format!("vision_model.encoder.layers.{i}.mlp.fc1.weight"),
                rt(v_inter * vh, s + 9),
            );
            weights.insert(
                format!("vision_model.encoder.layers.{i}.mlp.fc1.bias"),
                rt(v_inter, s + 10),
            );
            weights.insert(
                format!("vision_model.encoder.layers.{i}.mlp.fc2.weight"),
                rt(vh * v_inter, s + 11),
            );
            weights.insert(
                format!("vision_model.encoder.layers.{i}.mlp.fc2.bias"),
                rt(vh, s + 12),
            );
            weights.insert(
                format!("vision_model.encoder.layers.{i}.layer_norm1.weight"),
                ones(vh),
            );
            weights.insert(
                format!("vision_model.encoder.layers.{i}.layer_norm1.bias"),
                rt(vh, s + 14),
            );
            weights.insert(
                format!("vision_model.encoder.layers.{i}.layer_norm2.weight"),
                ones(vh),
            );
            weights.insert(
                format!("vision_model.encoder.layers.{i}.layer_norm2.bias"),
                rt(vh, s + 16),
            );
        }

        weights.insert(
            "text_model.embeddings.token_embedding.weight".into(),
            rt(cfg.vocab_size * th, 400),
        );
        weights.insert(
            "text_model.embeddings.position_embedding.weight".into(),
            rt(cfg.text_max_position * th, 401),
        );
        weights.insert("text_model.final_layer_norm.weight".into(), ones(th));
        weights.insert("text_model.final_layer_norm.bias".into(), rt(th, 403));

        for i in 0..cfg.text_num_layers {
            let s = (i as u32 + 10) * 200;
            weights.insert(
                format!("text_model.encoder.layers.{i}.self_attn.q_proj.weight"),
                rt(th * th, s + 1),
            );
            weights.insert(
                format!("text_model.encoder.layers.{i}.self_attn.q_proj.bias"),
                rt(th, s + 2),
            );
            weights.insert(
                format!("text_model.encoder.layers.{i}.self_attn.k_proj.weight"),
                rt(th * th, s + 3),
            );
            weights.insert(
                format!("text_model.encoder.layers.{i}.self_attn.k_proj.bias"),
                rt(th, s + 4),
            );
            weights.insert(
                format!("text_model.encoder.layers.{i}.self_attn.v_proj.weight"),
                rt(th * th, s + 5),
            );
            weights.insert(
                format!("text_model.encoder.layers.{i}.self_attn.v_proj.bias"),
                rt(th, s + 6),
            );
            weights.insert(
                format!("text_model.encoder.layers.{i}.self_attn.out_proj.weight"),
                rt(th * th, s + 7),
            );
            weights.insert(
                format!("text_model.encoder.layers.{i}.self_attn.out_proj.bias"),
                rt(th, s + 8),
            );
            weights.insert(
                format!("text_model.encoder.layers.{i}.mlp.fc1.weight"),
                rt(t_inter * th, s + 9),
            );
            weights.insert(
                format!("text_model.encoder.layers.{i}.mlp.fc1.bias"),
                rt(t_inter, s + 10),
            );
            weights.insert(
                format!("text_model.encoder.layers.{i}.mlp.fc2.weight"),
                rt(th * t_inter, s + 11),
            );
            weights.insert(
                format!("text_model.encoder.layers.{i}.mlp.fc2.bias"),
                rt(th, s + 12),
            );
            weights.insert(
                format!("text_model.encoder.layers.{i}.layer_norm1.weight"),
                ones(th),
            );
            weights.insert(
                format!("text_model.encoder.layers.{i}.layer_norm1.bias"),
                rt(th, s + 14),
            );
            weights.insert(
                format!("text_model.encoder.layers.{i}.layer_norm2.weight"),
                ones(th),
            );
            weights.insert(
                format!("text_model.encoder.layers.{i}.layer_norm2.bias"),
                rt(th, s + 16),
            );
        }

        weights.insert("visual_projection.weight".into(), rt(vh * ed, 600));
        weights.insert("text_projection.weight".into(), rt(th * ed, 601));

        let mapper = WeightMapper::clip();
        model
            .load_weights(weights, &mapper)
            .expect("load_weights failed");

        // Run forward pass to verify loaded weights produce finite output
        let image: Vec<f32> =
            (0..cfg.vision.image_size * cfg.vision.image_size * cfg.vision.channels)
                .map(|i| (i as f32) / 255.0)
                .collect();
        let token_ids: Vec<u32> = vec![1, 5, 10, 20, 3];

        let output = model.similarity(
            &image,
            cfg.vision.image_size,
            cfg.vision.image_size,
            &token_ids,
        );

        assert!(
            output.image_embedding.iter().all(|v| v.is_finite()),
            "Image embedding should be finite after weight loading"
        );
        assert!(
            output.text_embedding.iter().all(|v| v.is_finite()),
            "Text embedding should be finite after weight loading"
        );
        assert!(
            output.similarity.is_finite(),
            "Similarity should be finite after weight loading"
        );
    }

    #[test]
    fn test_clip_expected_weight_keys() {
        let cfg = test_clip_config();
        let model = CLIPModel::from_config(&cfg);
        let keys = model.expected_weight_keys();

        // Should have vision keys
        assert!(keys.iter().any(|k| k == "vision.patch_proj"));
        assert!(keys.iter().any(|k| k == "vision.cls_token"));
        assert!(keys.iter().any(|k| k == "vision.pos_embed"));
        assert!(keys.iter().any(|k| k == "vision.pre_norm.weight"));
        assert!(keys.iter().any(|k| k == "vision.norm.weight"));
        assert!(keys.iter().any(|k| k == "vision.layers[0].attention.w_q"));

        // Should have text keys
        assert!(keys.iter().any(|k| k == "text.embeddings"));
        assert!(keys.iter().any(|k| k == "text.pos_embed"));
        assert!(keys.iter().any(|k| k == "text.norm.weight"));
        assert!(keys.iter().any(|k| k == "text.layers[0].attention.w_q"));

        // Should have projection keys
        assert!(keys.iter().any(|k| k == "vision_proj"));
        assert!(keys.iter().any(|k| k == "text_proj"));
    }

    #[test]
    fn test_parse_clip_config_json() {
        let json = r#"{
            "projection_dim": 512,
            "vision_config": {
                "hidden_size": 768,
                "num_hidden_layers": 12,
                "num_attention_heads": 12,
                "intermediate_size": 3072,
                "image_size": 224,
                "patch_size": 32,
                "num_channels": 3
            },
            "text_config": {
                "hidden_size": 512,
                "num_hidden_layers": 12,
                "num_attention_heads": 8,
                "intermediate_size": 2048,
                "max_position_embeddings": 77,
                "vocab_size": 49408
            }
        }"#;

        let cfg = parse_clip_config_json(json).expect("Failed to parse CLIP config JSON");
        assert_eq!(cfg.embed_dim, 512);
        assert_eq!(cfg.vision.hidden_size, 768);
        assert_eq!(cfg.vision.num_layers, 12);
        assert_eq!(cfg.vision.num_heads, 12);
        assert_eq!(cfg.vision.intermediate_size, 3072);
        assert_eq!(cfg.vision.image_size, 224);
        assert_eq!(cfg.vision.patch_size, 32);
        assert_eq!(cfg.text_hidden_size, 512);
        assert_eq!(cfg.text_num_layers, 12);
        assert_eq!(cfg.text_num_heads, 8);
        assert_eq!(cfg.text_intermediate_size, 2048);
        assert_eq!(cfg.text_max_position, 77);
        assert_eq!(cfg.vocab_size, 49408);
    }
}
