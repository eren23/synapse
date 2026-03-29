//! Vision Transformer (ViT) model for image classification and embedding extraction.
//!
//! Implements the standard ViT architecture:
//! patch_embed → prepend CLS → add pos_embed → N × EncoderLayer → final norm → optional classifier.

use std::collections::{HashMap, HashSet};

use crate::ops::activation::gelu;
use crate::ops::attention::bidirectional_attention;
use crate::ops::matmul::matmul_t;
use crate::ops::norm::apply_norm;
use crate::ops::patch_embed::patch_embed;
use crate::ops::vector::{add_vecs, add_vecs_inplace};
use crate::registry::{AttentionVariant, FFNVariant, NormVariant};
use crate::weight_loading::{AlignedBuffer, RawTensor, WeightError, WeightMapper};

/// Configuration for a Vision Transformer model.
#[derive(Debug, Clone)]
pub struct ViTConfig {
    pub image_size: usize,
    pub patch_size: usize,
    pub channels: usize,
    pub hidden_size: usize,
    pub num_layers: usize,
    pub num_heads: usize,
    pub intermediate_size: usize,
    pub num_classes: usize,
}

impl ViTConfig {
    pub fn head_dim(&self) -> usize {
        self.hidden_size / self.num_heads
    }

    pub fn num_patches(&self) -> usize {
        let patches_per_side = self.image_size / self.patch_size;
        patches_per_side * patches_per_side
    }

    /// Total sequence length including CLS token.
    pub fn seq_len(&self) -> usize {
        self.num_patches() + 1
    }
}

/// Output from a ViT forward pass.
pub struct ViTOutput {
    /// CLS token embedding after final norm: [hidden_size].
    pub embeddings: Vec<f32>,
    /// Optional classification logits: [num_classes] (only if classifier head is present).
    pub logits: Option<Vec<f32>>,
}

/// A single ViT encoder layer (pre-norm architecture).
///
/// Forward: norm → bidirectional_attention → residual → norm → FFN → residual.
pub struct EncoderLayer {
    pub attn_norm: Box<dyn NormVariant>,
    pub attention: Box<dyn AttentionVariant>,
    pub ffn_norm: Box<dyn NormVariant>,
    pub ffn: Box<dyn FFNVariant>,
    pub hidden_size: usize,

    // ── Weights ──────────────────────────────────────────────────────
    pub attn_norm_weight: AlignedBuffer,
    pub w_q: AlignedBuffer,
    pub w_k: AlignedBuffer,
    pub w_v: AlignedBuffer,
    pub w_o: AlignedBuffer,
    pub ffn_norm_weight: AlignedBuffer,
    pub ffn_up: AlignedBuffer,
    pub ffn_down: AlignedBuffer,

    // ── Biases (empty if model doesn't use them) ─────────────────────
    pub q_bias: AlignedBuffer,
    pub k_bias: AlignedBuffer,
    pub v_bias: AlignedBuffer,
    pub o_bias: AlignedBuffer,
    pub ffn_up_bias: AlignedBuffer,
    pub ffn_down_bias: AlignedBuffer,
    pub attn_norm_bias: AlignedBuffer,
    pub ffn_norm_bias: AlignedBuffer,
}

impl EncoderLayer {
    /// Add a per-column bias to a row-major matrix `[m, n]` in place.
    /// No-op when `bias` is empty (model has no biases).
    fn add_bias(x: &mut [f32], bias: &[f32], m: usize, n: usize) {
        if bias.is_empty() {
            return;
        }
        for row in 0..m {
            for col in 0..n {
                x[row * n + col] += bias[col];
            }
        }
    }

    /// Pre-norm encoder forward: norm→bidirectional_attention→residual→norm→FFN→residual.
    ///
    /// `x` is `[seq_len, hidden_size]` (flat). Returns same shape.
    pub fn forward(&self, x: &[f32], seq_len: usize) -> Vec<f32> {
        let h = self.hidden_size;

        // 1. Attention sub-layer
        let mut normed = apply_norm(x, &self.attn_norm_weight, &*self.attn_norm, h);
        Self::add_bias(&mut normed, &self.attn_norm_bias, seq_len, h);
        let attn_out = self.apply_attention(&normed, seq_len);
        let mut residual = add_vecs(x, &attn_out);

        // 2. FFN sub-layer
        let mut normed = apply_norm(&residual, &self.ffn_norm_weight, &*self.ffn_norm, h);
        Self::add_bias(&mut normed, &self.ffn_norm_bias, seq_len, h);
        let ffn_out = self.apply_ffn(&normed, seq_len);
        add_vecs_inplace(&mut residual, &ffn_out);

        residual
    }

    fn apply_attention(&self, x: &[f32], seq_len: usize) -> Vec<f32> {
        let h = self.hidden_size;
        let num_heads = self.attention.num_heads();
        let head_dim = self.attention.head_dim();
        let q_dim = num_heads * head_dim;

        // Q, K, V projections: x is [seq_len, h]
        let mut q = matmul_t(x, &self.w_q, seq_len, h, q_dim);
        Self::add_bias(&mut q, &self.q_bias, seq_len, q_dim);
        let mut k = matmul_t(x, &self.w_k, seq_len, h, q_dim);
        Self::add_bias(&mut k, &self.k_bias, seq_len, q_dim);
        let mut v = matmul_t(x, &self.w_v, seq_len, h, q_dim);
        Self::add_bias(&mut v, &self.v_bias, seq_len, q_dim);

        // Bidirectional attention (no causal mask, no RoPE)
        let attn_out = bidirectional_attention(&q, &k, &v, seq_len, num_heads, head_dim);

        // Output projection
        let mut out = matmul_t(&attn_out, &self.w_o, seq_len, q_dim, h);
        Self::add_bias(&mut out, &self.o_bias, seq_len, h);
        out
    }

    fn apply_ffn(&self, x: &[f32], seq_len: usize) -> Vec<f32> {
        let h = self.hidden_size;
        let inter = self.ffn.intermediate_size();

        // GELU FFN: y = gelu(x @ up^T + up_bias) @ down^T + down_bias
        let mut activated = matmul_t(x, &self.ffn_up, seq_len, h, inter);
        Self::add_bias(&mut activated, &self.ffn_up_bias, seq_len, inter);
        for v in activated.iter_mut() {
            *v = gelu(*v);
        }
        let mut out = matmul_t(&activated, &self.ffn_down, seq_len, inter, h);
        Self::add_bias(&mut out, &self.ffn_down_bias, seq_len, h);
        out
    }

    /// Weight keys this layer expects (relative to `layers[i].`).
    pub fn weight_keys(&self, layer_idx: usize) -> Vec<String> {
        let i = layer_idx;
        vec![
            format!("layers[{i}].attn_norm.weight"),
            format!("layers[{i}].attn_norm.bias"),
            format!("layers[{i}].attention.w_q"),
            format!("layers[{i}].attention.q_bias"),
            format!("layers[{i}].attention.w_k"),
            format!("layers[{i}].attention.k_bias"),
            format!("layers[{i}].attention.w_v"),
            format!("layers[{i}].attention.v_bias"),
            format!("layers[{i}].attention.w_o"),
            format!("layers[{i}].attention.o_bias"),
            format!("layers[{i}].ffn_norm.weight"),
            format!("layers[{i}].ffn_norm.bias"),
            format!("layers[{i}].ffn.w_up"),
            format!("layers[{i}].ffn.up_bias"),
            format!("layers[{i}].ffn.w_down"),
            format!("layers[{i}].ffn.down_bias"),
        ]
    }

    /// Assign a weight by its field name (e.g. "attention.w_q").
    pub fn set_weight(&mut self, field: &str, tensor: &RawTensor) -> Result<(), WeightError> {
        match field {
            "attn_norm.weight" => self.attn_norm_weight = tensor.data.clone(),
            "attn_norm.bias" => self.attn_norm_bias = tensor.data.clone(),
            "attention.w_q" => self.w_q = tensor.data.clone(),
            "attention.q_bias" => self.q_bias = tensor.data.clone(),
            "attention.w_k" => self.w_k = tensor.data.clone(),
            "attention.k_bias" => self.k_bias = tensor.data.clone(),
            "attention.w_v" => self.w_v = tensor.data.clone(),
            "attention.v_bias" => self.v_bias = tensor.data.clone(),
            "attention.w_o" => self.w_o = tensor.data.clone(),
            "attention.o_bias" => self.o_bias = tensor.data.clone(),
            "ffn_norm.weight" => self.ffn_norm_weight = tensor.data.clone(),
            "ffn_norm.bias" => self.ffn_norm_bias = tensor.data.clone(),
            "ffn.w_up" => self.ffn_up = tensor.data.clone(),
            "ffn.up_bias" => self.ffn_up_bias = tensor.data.clone(),
            "ffn.w_down" => self.ffn_down = tensor.data.clone(),
            "ffn.down_bias" => self.ffn_down_bias = tensor.data.clone(),
            _ => {}
        }
        Ok(())
    }
}

/// A Vision Transformer model: patch_embed → CLS + pos → N × EncoderLayer → norm → classifier.
pub struct ViTModel {
    pub config: ViTConfig,
    pub layers: Vec<EncoderLayer>,
    pub final_norm: Box<dyn NormVariant>,

    // ── Weights ──────────────────────────────────────────────────────
    /// Patch projection weight: [embed_dim, patch_dim] where patch_dim = P*P*C.
    pub patch_proj: AlignedBuffer,
    /// Patch projection bias: [embed_dim].
    pub patch_proj_bias: AlignedBuffer,
    /// Positional embeddings: [seq_len, hidden_size] where seq_len = 1 + num_patches.
    pub pos_embed: AlignedBuffer,
    /// CLS token embedding: [hidden_size].
    pub cls_token: AlignedBuffer,
    /// Final layer norm weight: [hidden_size].
    pub final_norm_weight: AlignedBuffer,
    /// Final layer norm bias: [hidden_size].
    pub final_norm_bias: AlignedBuffer,
    /// Optional classifier weight: [num_classes, hidden_size].
    pub classifier_head: Option<AlignedBuffer>,
    /// Optional classifier bias: [num_classes].
    pub classifier_bias: Option<AlignedBuffer>,
    /// ImageNet class labels (optional, for display).
    pub class_labels: Vec<String>,
}

impl ViTModel {
    /// Build a ViT model from config with zeroed weights.
    pub fn from_config(config: &ViTConfig) -> Self {
        use crate::config::{AttentionConfig, FFNConfig, NormConfig};
        use crate::registry::{create_attention, create_ffn, create_norm};

        let norm_config = NormConfig::LayerNorm { eps: 1e-6 };
        let attn_config = AttentionConfig::Bidirectional {
            num_heads: config.num_heads,
            head_dim: config.head_dim(),
        };
        let ffn_config = FFNConfig::GELU {
            intermediate_size: config.intermediate_size,
        };

        let mut layers = Vec::with_capacity(config.num_layers);
        for _ in 0..config.num_layers {
            layers.push(EncoderLayer {
                attn_norm: create_norm(&norm_config),
                attention: create_attention(&attn_config),
                ffn_norm: create_norm(&norm_config),
                ffn: create_ffn(&ffn_config),
                hidden_size: config.hidden_size,
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

        let classifier_head = if config.num_classes > 0 {
            Some(AlignedBuffer::new_zeroed(0))
        } else {
            None
        };

        let classifier_bias = if config.num_classes > 0 {
            Some(AlignedBuffer::new_zeroed(0))
        } else {
            None
        };

        ViTModel {
            config: config.clone(),
            layers,
            final_norm: create_norm(&norm_config),
            patch_proj: AlignedBuffer::new_zeroed(0),
            patch_proj_bias: AlignedBuffer::new_zeroed(0),
            pos_embed: AlignedBuffer::new_zeroed(0),
            cls_token: AlignedBuffer::new_zeroed(0),
            final_norm_weight: AlignedBuffer::new_zeroed(0),
            final_norm_bias: AlignedBuffer::new_zeroed(0),
            classifier_head,
            classifier_bias,
            class_labels: Vec::new(),
        }
    }

    /// Forward pass: image → embeddings (+ optional logits).
    ///
    /// `image` is `[height * width * channels]` (flat, HWC layout).
    pub fn forward_image(&self, image: &[f32], height: usize, width: usize) -> ViTOutput {
        let cfg = &self.config;
        let h = cfg.hidden_size;

        // 1. Patch embedding: image → [num_patches, hidden_size]
        let mut patch_embeddings = patch_embed(
            image,
            height,
            width,
            cfg.channels,
            cfg.patch_size,
            &self.patch_proj,
            h,
        );
        let num_patches = cfg.num_patches();

        // Add patch projection bias
        if !self.patch_proj_bias.is_empty() {
            for i in 0..num_patches {
                for j in 0..h {
                    patch_embeddings[i * h + j] += self.patch_proj_bias[j];
                }
            }
        }

        let seq_len = num_patches + 1; // +1 for CLS token

        // 2. Prepend CLS token to get [seq_len, hidden_size]
        let mut x = vec![0.0f32; seq_len * h];
        // CLS token at position 0
        if !self.cls_token.is_empty() {
            x[..h].copy_from_slice(&self.cls_token);
        }
        // Patch embeddings at positions 1..
        x[h..].copy_from_slice(&patch_embeddings);

        // 3. Add positional embeddings element-wise
        if !self.pos_embed.is_empty() {
            let pos_len = self.pos_embed.len().min(x.len());
            for i in 0..pos_len {
                x[i] += self.pos_embed[i];
            }
        }

        // 4. Encoder layers
        for layer in &self.layers {
            x = layer.forward(&x, seq_len);
        }

        // 5. Take CLS token (position 0), apply final norm
        let cls_hidden = &x[..h];
        let mut embeddings = apply_norm(cls_hidden, &self.final_norm_weight, &*self.final_norm, h);
        // Add final norm bias
        if !self.final_norm_bias.is_empty() {
            for j in 0..h {
                embeddings[j] += self.final_norm_bias[j];
            }
        }

        // 6. Optional classifier head
        let logits = self.classifier_head.as_ref().map(|head| {
            if head.is_empty() {
                return vec![];
            }
            let mut out = matmul_t(&embeddings, head, 1, h, self.config.num_classes);
            // Add classifier bias
            if let Some(ref bias) = self.classifier_bias {
                if !bias.is_empty() {
                    for j in 0..self.config.num_classes {
                        out[j] += bias[j];
                    }
                }
            }
            out
        });

        ViTOutput { embeddings, logits }
    }

    /// All target weight keys this model expects (Synapse naming).
    pub fn expected_weight_keys(&self) -> Vec<String> {
        let mut keys = vec![
            "patch_proj".to_string(),
            "patch_proj_bias".to_string(),
            "cls_token".to_string(),
            "pos_embed".to_string(),
        ];
        for (i, layer) in self.layers.iter().enumerate() {
            keys.extend(layer.weight_keys(i));
        }
        keys.push("norm.weight".to_string());
        keys.push("norm.bias".to_string());
        if self.classifier_head.is_some() {
            keys.push("classifier.weight".to_string());
            keys.push("classifier.bias".to_string());
        }
        keys
    }

    /// Load weights from source tensors using a name mapper.
    pub fn load_weights(
        &mut self,
        weights: HashMap<String, RawTensor>,
        mapper: &WeightMapper,
    ) -> Result<crate::models::lm::LoadResult, WeightError> {
        let source_keys: Vec<String> = weights.keys().cloned().collect();
        let mapping = mapper.map_keys(&source_keys);

        let expected: HashSet<String> = self.expected_weight_keys().into_iter().collect();
        let mut loaded = HashSet::new();

        for (source, target) in &mapping.mapping {
            if let Some(raw) = weights.get(source) {
                self.set_weight(target, raw)?;
                if expected.contains(target) {
                    loaded.insert(target.clone());
                }
            }
        }

        let missing: Vec<String> = expected.difference(&loaded).cloned().collect();
        let unexpected = mapping.unmapped;

        Ok(crate::models::lm::LoadResult {
            missing,
            unexpected,
        })
    }

    fn set_weight(&mut self, key: &str, tensor: &RawTensor) -> Result<(), WeightError> {
        match key {
            "patch_proj" => self.patch_proj = tensor.data.clone(),
            "patch_proj_bias" => self.patch_proj_bias = tensor.data.clone(),
            "cls_token" => self.cls_token = tensor.data.clone(),
            "pos_embed" => self.pos_embed = tensor.data.clone(),
            "norm.weight" => self.final_norm_weight = tensor.data.clone(),
            "norm.bias" => self.final_norm_bias = tensor.data.clone(),
            "classifier.weight" => {
                if self.classifier_head.is_some() {
                    self.classifier_head = Some(tensor.data.clone());
                }
            }
            "classifier.bias" => {
                if self.classifier_bias.is_some() {
                    self.classifier_bias = Some(tensor.data.clone());
                }
            }
            _ if key.starts_with("layers[") => {
                if let Some((idx, field)) = parse_layer_key(key) {
                    if let Some(layer) = self.layers.get_mut(idx) {
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

/// Parse a HuggingFace ViT `config.json` into a [`ViTConfig`].
///
/// Reads fields like `hidden_size`, `num_hidden_layers`, `num_attention_heads`,
/// `intermediate_size`, `image_size`, `patch_size`, `num_channels`.
/// Also extracts `id2label` for ImageNet class names if present.
pub fn parse_vit_config(path: &std::path::Path) -> Result<ViTConfig, Box<dyn std::error::Error>> {
    let json_str = std::fs::read_to_string(path)?;
    parse_vit_config_json(&json_str)
}

/// Parse a ViT config from a JSON string.
pub fn parse_vit_config_json(json: &str) -> Result<ViTConfig, Box<dyn std::error::Error>> {
    let v: serde_json::Value = serde_json::from_str(json)?;

    let hidden_size = v["hidden_size"].as_u64().unwrap_or(768) as usize;
    let num_layers = v["num_hidden_layers"].as_u64().unwrap_or(12) as usize;
    let num_heads = v["num_attention_heads"].as_u64().unwrap_or(12) as usize;
    let intermediate_size = v["intermediate_size"].as_u64().unwrap_or(3072) as usize;
    let image_size = v["image_size"].as_u64().unwrap_or(224) as usize;
    let patch_size = v["patch_size"].as_u64().unwrap_or(16) as usize;
    let channels = v["num_channels"].as_u64().unwrap_or(3) as usize;

    // Determine num_classes from id2label if available, else fall back to num_labels
    let num_classes = if let Some(id2label) = v.get("id2label").and_then(|v| v.as_object()) {
        id2label.len()
    } else {
        v["num_labels"].as_u64().unwrap_or(1000) as usize
    };

    Ok(ViTConfig {
        image_size,
        patch_size,
        channels,
        hidden_size,
        num_layers,
        num_heads,
        intermediate_size,
        num_classes,
    })
}

/// Extract ImageNet class labels from the HF config `id2label` field.
///
/// Returns a Vec<String> of length num_classes, ordered by integer key.
pub fn parse_vit_labels(path: &std::path::Path) -> Result<Vec<String>, Box<dyn std::error::Error>> {
    let json_str = std::fs::read_to_string(path)?;
    parse_vit_labels_json(&json_str)
}

/// Extract class labels from a JSON string.
pub fn parse_vit_labels_json(json: &str) -> Result<Vec<String>, Box<dyn std::error::Error>> {
    let v: serde_json::Value = serde_json::from_str(json)?;
    let id2label = v.get("id2label").and_then(|v| v.as_object());
    match id2label {
        Some(map) => {
            let mut labels: Vec<(usize, String)> = map
                .iter()
                .filter_map(|(k, v)| {
                    let idx: usize = k.parse().ok()?;
                    let label = v.as_str()?.to_string();
                    Some((idx, label))
                })
                .collect();
            labels.sort_by_key(|(idx, _)| *idx);
            Ok(labels.into_iter().map(|(_, label)| label).collect())
        }
        None => Ok(Vec::new()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_vit_config() -> ViTConfig {
        ViTConfig {
            image_size: 8,
            patch_size: 4,
            channels: 3,
            hidden_size: 32,
            num_layers: 2,
            num_heads: 4,
            intermediate_size: 64,
            num_classes: 10,
        }
    }

    fn gen_weights(len: usize, seed: u32) -> Vec<f32> {
        (0..len)
            .map(|i| {
                let x = ((i as u32).wrapping_mul(2654435761).wrapping_add(seed)) as f32;
                (x / u32::MAX as f32) * 0.36 - 0.18
            })
            .collect()
    }

    fn build_test_vit(cfg: &ViTConfig) -> ViTModel {
        let h = cfg.hidden_size;
        let patch_dim = cfg.patch_size * cfg.patch_size * cfg.channels;
        let inter = cfg.intermediate_size;
        let seq_len = cfg.seq_len();

        let mut model = ViTModel::from_config(cfg);

        // Set weights directly (bypass weight mapper for unit tests)
        model.patch_proj = AlignedBuffer::from_slice(&gen_weights(h * patch_dim, 1));
        model.cls_token = AlignedBuffer::from_slice(&gen_weights(h, 2));
        model.pos_embed = AlignedBuffer::from_slice(&gen_weights(seq_len * h, 3));
        model.final_norm_weight = AlignedBuffer::from_slice(&vec![1.0f32; h]);

        for (i, layer) in model.layers.iter_mut().enumerate() {
            let s = (i as u32 + 1) * 100;
            layer.attn_norm_weight = AlignedBuffer::from_slice(&vec![1.0f32; h]);
            layer.w_q = AlignedBuffer::from_slice(&gen_weights(h * h, s + 1));
            layer.w_k = AlignedBuffer::from_slice(&gen_weights(h * h, s + 2));
            layer.w_v = AlignedBuffer::from_slice(&gen_weights(h * h, s + 3));
            layer.w_o = AlignedBuffer::from_slice(&gen_weights(h * h, s + 4));
            layer.ffn_norm_weight = AlignedBuffer::from_slice(&vec![1.0f32; h]);
            layer.ffn_up = AlignedBuffer::from_slice(&gen_weights(inter * h, s + 5));
            layer.ffn_down = AlignedBuffer::from_slice(&gen_weights(h * inter, s + 6));
        }

        if cfg.num_classes > 0 {
            model.classifier_head = Some(AlignedBuffer::from_slice(&gen_weights(
                cfg.num_classes * h,
                999,
            )));
        }

        model
    }

    #[test]
    fn test_vit_forward_produces_finite_embeddings() {
        let cfg = test_vit_config();
        let model = build_test_vit(&cfg);

        let image: Vec<f32> = (0..cfg.image_size * cfg.image_size * cfg.channels)
            .map(|i| (i as f32) / 255.0)
            .collect();

        let output = model.forward_image(&image, cfg.image_size, cfg.image_size);

        // Embeddings should have correct size and be finite
        assert_eq!(output.embeddings.len(), cfg.hidden_size);
        assert!(
            output.embeddings.iter().all(|v| v.is_finite()),
            "ViT forward produced non-finite embeddings"
        );

        // Logits should exist (num_classes > 0) and be finite
        let logits = output
            .logits
            .as_ref()
            .expect("expected logits for classification model");
        assert_eq!(logits.len(), cfg.num_classes);
        assert!(
            logits.iter().all(|v| v.is_finite()),
            "ViT forward produced non-finite logits"
        );
    }

    #[test]
    fn test_vit_patch_embed_shape() {
        let cfg = test_vit_config();
        let h = cfg.hidden_size;
        let patch_dim = cfg.patch_size * cfg.patch_size * cfg.channels;

        let image: Vec<f32> = (0..cfg.image_size * cfg.image_size * cfg.channels)
            .map(|i| (i as f32) / 255.0)
            .collect();

        let projection = gen_weights(h * patch_dim, 42);
        let result = crate::ops::patch_embed::patch_embed(
            &image,
            cfg.image_size,
            cfg.image_size,
            cfg.channels,
            cfg.patch_size,
            &projection,
            h,
        );

        let expected_patches = cfg.num_patches();
        assert_eq!(
            result.len(),
            expected_patches * h,
            "Expected {} patches * {} hidden = {} elements, got {}",
            expected_patches,
            h,
            expected_patches * h,
            result.len()
        );
    }
}
