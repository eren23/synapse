use std::collections::{HashMap, HashSet};

use super::WeightError;

/// A rule mapping a source name pattern to a target name pattern.
///
/// Supports `{i}` as a placeholder for layer indices.
#[derive(Debug, Clone)]
pub struct MappingRule {
    pub source: String,
    pub target: String,
}

fn rule(source: &str, target: &str) -> MappingRule {
    MappingRule {
        source: source.to_string(),
        target: target.to_string(),
    }
}

/// Result of mapping weight names.
#[derive(Debug)]
pub struct MappingResult {
    /// source name → target name
    pub mapping: HashMap<String, String>,
    /// Source keys that didn't match any rule.
    pub unmapped: Vec<String>,
}

/// Maps HuggingFace weight names to Synapse internal paths via pattern rules.
#[derive(Debug)]
pub struct WeightMapper {
    rules: Vec<MappingRule>,
}

impl WeightMapper {
    pub fn new(rules: Vec<MappingRule>) -> Self {
        WeightMapper { rules }
    }

    /// Create a mapper for Qwen3 weight names.
    pub fn qwen3() -> Self {
        WeightMapper::new(vec![
            rule("model.embed_tokens.weight", "embed_tokens.weight"),
            rule(
                "model.layers.{i}.self_attn.q_proj.weight",
                "layers[{i}].attention.w_q",
            ),
            rule(
                "model.layers.{i}.self_attn.k_proj.weight",
                "layers[{i}].attention.w_k",
            ),
            rule(
                "model.layers.{i}.self_attn.v_proj.weight",
                "layers[{i}].attention.w_v",
            ),
            rule(
                "model.layers.{i}.self_attn.o_proj.weight",
                "layers[{i}].attention.w_o",
            ),
            rule(
                "model.layers.{i}.self_attn.q_proj.bias",
                "layers[{i}].attention.q_bias",
            ),
            rule(
                "model.layers.{i}.self_attn.k_proj.bias",
                "layers[{i}].attention.k_bias",
            ),
            rule(
                "model.layers.{i}.self_attn.v_proj.bias",
                "layers[{i}].attention.v_bias",
            ),
            rule(
                "model.layers.{i}.self_attn.q_norm.weight",
                "layers[{i}].attention.q_norm",
            ),
            rule(
                "model.layers.{i}.self_attn.k_norm.weight",
                "layers[{i}].attention.k_norm",
            ),
            rule(
                "model.layers.{i}.mlp.gate_proj.weight",
                "layers[{i}].ffn.w_gate",
            ),
            rule(
                "model.layers.{i}.mlp.up_proj.weight",
                "layers[{i}].ffn.w_up",
            ),
            rule(
                "model.layers.{i}.mlp.down_proj.weight",
                "layers[{i}].ffn.w_down",
            ),
            rule(
                "model.layers.{i}.input_layernorm.weight",
                "layers[{i}].attn_norm.weight",
            ),
            rule(
                "model.layers.{i}.post_attention_layernorm.weight",
                "layers[{i}].ffn_norm.weight",
            ),
            rule("model.norm.weight", "norm.weight"),
            rule("lm_head.weight", "lm_head.weight"),
        ])
    }

    /// Create a mapper for the given HuggingFace model type.
    ///
    /// Supported: `"qwen3"`, `"llama"`, `"mistral"`, `"phi"` / `"phi3"`,
    /// `"gemma"` / `"gemma2"`, `"vit"`, `"clip"`, `"dinov2"`.
    pub fn from_model_type(model_type: &str) -> Result<Self, WeightError> {
        match model_type {
            "qwen3" => Ok(Self::qwen3()),
            "qwen2" | "qwen2.5" => Ok(Self::llama()), // Same naming as LLaMA (no q_norm/k_norm)
            "llama" => Ok(Self::llama()),
            "mistral" => Ok(Self::mistral()),
            "phi" | "phi3" => Ok(Self::phi()),
            "gemma" | "gemma2" => Ok(Self::gemma()),
            "vit" => Ok(Self::vit()),
            "clip" => Ok(Self::clip()),
            "dinov2" => Ok(Self::dinov2()),
            _ => Err(WeightError::InvalidFormat(format!(
                "Unsupported model type: {model_type}"
            ))),
        }
    }

    /// Create a mapper for LLaMA weight names.
    ///
    /// Identical to Qwen3 but without q_norm/k_norm rules (LLaMA doesn't
    /// have per-head norms).
    pub fn llama() -> Self {
        WeightMapper::new(vec![
            rule("model.embed_tokens.weight", "embed_tokens.weight"),
            rule(
                "model.layers.{i}.self_attn.q_proj.weight",
                "layers[{i}].attention.w_q",
            ),
            rule(
                "model.layers.{i}.self_attn.k_proj.weight",
                "layers[{i}].attention.w_k",
            ),
            rule(
                "model.layers.{i}.self_attn.v_proj.weight",
                "layers[{i}].attention.w_v",
            ),
            rule(
                "model.layers.{i}.self_attn.o_proj.weight",
                "layers[{i}].attention.w_o",
            ),
            rule(
                "model.layers.{i}.self_attn.q_proj.bias",
                "layers[{i}].attention.q_bias",
            ),
            rule(
                "model.layers.{i}.self_attn.k_proj.bias",
                "layers[{i}].attention.k_bias",
            ),
            rule(
                "model.layers.{i}.self_attn.v_proj.bias",
                "layers[{i}].attention.v_bias",
            ),
            rule(
                "model.layers.{i}.mlp.gate_proj.weight",
                "layers[{i}].ffn.w_gate",
            ),
            rule(
                "model.layers.{i}.mlp.up_proj.weight",
                "layers[{i}].ffn.w_up",
            ),
            rule(
                "model.layers.{i}.mlp.down_proj.weight",
                "layers[{i}].ffn.w_down",
            ),
            rule(
                "model.layers.{i}.input_layernorm.weight",
                "layers[{i}].attn_norm.weight",
            ),
            rule(
                "model.layers.{i}.post_attention_layernorm.weight",
                "layers[{i}].ffn_norm.weight",
            ),
            rule("model.norm.weight", "norm.weight"),
            rule("lm_head.weight", "lm_head.weight"),
        ])
    }

    /// Create a mapper for Mistral weight names.
    ///
    /// Mistral has identical weight naming to LLaMA (no per-head norms).
    pub fn mistral() -> Self {
        Self::llama()
    }

    /// Create a mapper for Phi-3/Phi-4 weight names (separate projections).
    ///
    /// When projections are stored as separate q_proj/k_proj/v_proj and
    /// gate_proj/up_proj, Phi uses the same HF naming convention as LLaMA.
    ///
    /// NOTE: Some Phi checkpoints use fused `qkv_proj` (shape `[3*hidden, hidden]`)
    /// and fused `gate_up_proj` (shape `[2*intermediate, hidden]`). Splitting those
    /// into individual Q/K/V and gate/up tensors is not yet implemented and will
    /// require a dedicated pre-processing step before weight loading.
    pub fn phi() -> Self {
        Self::llama()
    }

    /// Create a mapper for Gemma / Gemma-2 weight names.
    ///
    /// Gemma uses the same HF naming convention as LLaMA (no per-head norms).
    /// Gemma-1 ties embeddings (no separate lm_head), but the mapper still
    /// includes the lm_head rule — unused rules are harmless and Gemma-2 does
    /// have a separate lm_head.
    pub fn gemma() -> Self {
        Self::llama()
    }

    /// Create a mapper for HuggingFace CLIP (openai/clip-vit-base-patch32) weight names.
    ///
    /// CLIP has TWO encoder prefixes: `vision_model.` for the ViT image encoder
    /// and `text_model.` for the bidirectional text encoder, plus global projection
    /// weights `visual_projection.weight` and `text_projection.weight`.
    pub fn clip() -> Self {
        WeightMapper::new(vec![
            // ── Vision side ─────────────────────────────────────────
            rule(
                "vision_model.embeddings.patch_embedding.weight",
                "vision.patch_proj",
            ),
            rule(
                "vision_model.embeddings.class_embedding",
                "vision.cls_token",
            ),
            rule(
                "vision_model.embeddings.position_embedding.weight",
                "vision.pos_embed",
            ),
            rule(
                "vision_model.encoder.layers.{i}.self_attn.q_proj.weight",
                "vision.layers[{i}].attention.w_q",
            ),
            rule(
                "vision_model.encoder.layers.{i}.self_attn.q_proj.bias",
                "vision.layers[{i}].attention.q_bias",
            ),
            rule(
                "vision_model.encoder.layers.{i}.self_attn.k_proj.weight",
                "vision.layers[{i}].attention.w_k",
            ),
            rule(
                "vision_model.encoder.layers.{i}.self_attn.k_proj.bias",
                "vision.layers[{i}].attention.k_bias",
            ),
            rule(
                "vision_model.encoder.layers.{i}.self_attn.v_proj.weight",
                "vision.layers[{i}].attention.w_v",
            ),
            rule(
                "vision_model.encoder.layers.{i}.self_attn.v_proj.bias",
                "vision.layers[{i}].attention.v_bias",
            ),
            rule(
                "vision_model.encoder.layers.{i}.self_attn.out_proj.weight",
                "vision.layers[{i}].attention.w_o",
            ),
            rule(
                "vision_model.encoder.layers.{i}.self_attn.out_proj.bias",
                "vision.layers[{i}].attention.o_bias",
            ),
            rule(
                "vision_model.encoder.layers.{i}.mlp.fc1.weight",
                "vision.layers[{i}].ffn.w_up",
            ),
            rule(
                "vision_model.encoder.layers.{i}.mlp.fc1.bias",
                "vision.layers[{i}].ffn.up_bias",
            ),
            rule(
                "vision_model.encoder.layers.{i}.mlp.fc2.weight",
                "vision.layers[{i}].ffn.w_down",
            ),
            rule(
                "vision_model.encoder.layers.{i}.mlp.fc2.bias",
                "vision.layers[{i}].ffn.down_bias",
            ),
            rule(
                "vision_model.encoder.layers.{i}.layer_norm1.weight",
                "vision.layers[{i}].attn_norm.weight",
            ),
            rule(
                "vision_model.encoder.layers.{i}.layer_norm1.bias",
                "vision.layers[{i}].attn_norm.bias",
            ),
            rule(
                "vision_model.encoder.layers.{i}.layer_norm2.weight",
                "vision.layers[{i}].ffn_norm.weight",
            ),
            rule(
                "vision_model.encoder.layers.{i}.layer_norm2.bias",
                "vision.layers[{i}].ffn_norm.bias",
            ),
            rule("vision_model.pre_layernorm.weight", "vision.pre_norm.weight"),
            rule("vision_model.pre_layernorm.bias", "vision.pre_norm.bias"),
            rule("vision_model.post_layernorm.weight", "vision.norm.weight"),
            rule("vision_model.post_layernorm.bias", "vision.norm.bias"),
            // ── Text side ───────────────────────────────────────────
            rule(
                "text_model.embeddings.token_embedding.weight",
                "text.embeddings",
            ),
            rule(
                "text_model.embeddings.position_embedding.weight",
                "text.pos_embed",
            ),
            rule(
                "text_model.encoder.layers.{i}.self_attn.q_proj.weight",
                "text.layers[{i}].attention.w_q",
            ),
            rule(
                "text_model.encoder.layers.{i}.self_attn.q_proj.bias",
                "text.layers[{i}].attention.q_bias",
            ),
            rule(
                "text_model.encoder.layers.{i}.self_attn.k_proj.weight",
                "text.layers[{i}].attention.w_k",
            ),
            rule(
                "text_model.encoder.layers.{i}.self_attn.k_proj.bias",
                "text.layers[{i}].attention.k_bias",
            ),
            rule(
                "text_model.encoder.layers.{i}.self_attn.v_proj.weight",
                "text.layers[{i}].attention.w_v",
            ),
            rule(
                "text_model.encoder.layers.{i}.self_attn.v_proj.bias",
                "text.layers[{i}].attention.v_bias",
            ),
            rule(
                "text_model.encoder.layers.{i}.self_attn.out_proj.weight",
                "text.layers[{i}].attention.w_o",
            ),
            rule(
                "text_model.encoder.layers.{i}.self_attn.out_proj.bias",
                "text.layers[{i}].attention.o_bias",
            ),
            rule(
                "text_model.encoder.layers.{i}.mlp.fc1.weight",
                "text.layers[{i}].ffn.w_up",
            ),
            rule(
                "text_model.encoder.layers.{i}.mlp.fc1.bias",
                "text.layers[{i}].ffn.up_bias",
            ),
            rule(
                "text_model.encoder.layers.{i}.mlp.fc2.weight",
                "text.layers[{i}].ffn.w_down",
            ),
            rule(
                "text_model.encoder.layers.{i}.mlp.fc2.bias",
                "text.layers[{i}].ffn.down_bias",
            ),
            rule(
                "text_model.encoder.layers.{i}.layer_norm1.weight",
                "text.layers[{i}].attn_norm.weight",
            ),
            rule(
                "text_model.encoder.layers.{i}.layer_norm1.bias",
                "text.layers[{i}].attn_norm.bias",
            ),
            rule(
                "text_model.encoder.layers.{i}.layer_norm2.weight",
                "text.layers[{i}].ffn_norm.weight",
            ),
            rule(
                "text_model.encoder.layers.{i}.layer_norm2.bias",
                "text.layers[{i}].ffn_norm.bias",
            ),
            rule("text_model.final_layer_norm.weight", "text.norm.weight"),
            rule("text_model.final_layer_norm.bias", "text.norm.bias"),
            // ── Projections ─────────────────────────────────────────
            rule("visual_projection.weight", "vision_proj"),
            rule("text_projection.weight", "text_proj"),
        ])
    }

    /// Create a mapper for HuggingFace DINOv2 (facebook/dinov2-base) weight names.
    ///
    /// DINOv2 uses standard ViT architecture without prefix and
    /// the same layer naming convention as HuggingFace ViT models.
    pub fn dinov2() -> Self {
        WeightMapper::new(vec![
            rule(
                "embeddings.patch_embeddings.projection.weight",
                "patch_proj",
            ),
            rule(
                "embeddings.patch_embeddings.projection.bias",
                "patch_proj_bias",
            ),
            rule("embeddings.cls_token", "cls_token"),
            rule("embeddings.position_embeddings", "pos_embed"),
            // Attention weights
            rule(
                "encoder.layer.{i}.attention.attention.query.weight",
                "layers[{i}].attention.w_q",
            ),
            rule(
                "encoder.layer.{i}.attention.attention.query.bias",
                "layers[{i}].attention.q_bias",
            ),
            rule(
                "encoder.layer.{i}.attention.attention.key.weight",
                "layers[{i}].attention.w_k",
            ),
            rule(
                "encoder.layer.{i}.attention.attention.key.bias",
                "layers[{i}].attention.k_bias",
            ),
            rule(
                "encoder.layer.{i}.attention.attention.value.weight",
                "layers[{i}].attention.w_v",
            ),
            rule(
                "encoder.layer.{i}.attention.attention.value.bias",
                "layers[{i}].attention.v_bias",
            ),
            rule(
                "encoder.layer.{i}.attention.output.dense.weight",
                "layers[{i}].attention.w_o",
            ),
            rule(
                "encoder.layer.{i}.attention.output.dense.bias",
                "layers[{i}].attention.o_bias",
            ),
            // FFN weights
            rule(
                "encoder.layer.{i}.intermediate.dense.weight",
                "layers[{i}].ffn.w_up",
            ),
            rule(
                "encoder.layer.{i}.intermediate.dense.bias",
                "layers[{i}].ffn.up_bias",
            ),
            rule(
                "encoder.layer.{i}.output.dense.weight",
                "layers[{i}].ffn.w_down",
            ),
            rule(
                "encoder.layer.{i}.output.dense.bias",
                "layers[{i}].ffn.down_bias",
            ),
            // LayerNorm weights and biases
            rule(
                "encoder.layer.{i}.norm1.weight",
                "layers[{i}].attn_norm.weight",
            ),
            rule(
                "encoder.layer.{i}.norm1.bias",
                "layers[{i}].attn_norm.bias",
            ),
            rule(
                "encoder.layer.{i}.norm2.weight",
                "layers[{i}].ffn_norm.weight",
            ),
            rule(
                "encoder.layer.{i}.norm2.bias",
                "layers[{i}].ffn_norm.bias",
            ),
            // Final norm
            rule("layernorm.weight", "norm.weight"),
            rule("layernorm.bias", "norm.bias"),
        ])
    }

    /// Create a mapper for HuggingFace ViT (Vision Transformer) weight names.
    ///
    /// Maps `google/vit-base-patch16-224` style naming to Synapse internal paths.
    /// ViT uses LayerNorm (with bias), bidirectional attention, and GELU FFN (non-gated).
    pub fn vit() -> Self {
        WeightMapper::new(vec![
            rule(
                "vit.embeddings.patch_embeddings.projection.weight",
                "patch_proj",
            ),
            rule(
                "vit.embeddings.patch_embeddings.projection.bias",
                "patch_proj_bias",
            ),
            rule("vit.embeddings.cls_token", "cls_token"),
            rule("vit.embeddings.position_embeddings", "pos_embed"),
            // Attention weights
            rule(
                "vit.encoder.layer.{i}.attention.attention.query.weight",
                "layers[{i}].attention.w_q",
            ),
            rule(
                "vit.encoder.layer.{i}.attention.attention.query.bias",
                "layers[{i}].attention.q_bias",
            ),
            rule(
                "vit.encoder.layer.{i}.attention.attention.key.weight",
                "layers[{i}].attention.w_k",
            ),
            rule(
                "vit.encoder.layer.{i}.attention.attention.key.bias",
                "layers[{i}].attention.k_bias",
            ),
            rule(
                "vit.encoder.layer.{i}.attention.attention.value.weight",
                "layers[{i}].attention.w_v",
            ),
            rule(
                "vit.encoder.layer.{i}.attention.attention.value.bias",
                "layers[{i}].attention.v_bias",
            ),
            rule(
                "vit.encoder.layer.{i}.attention.output.dense.weight",
                "layers[{i}].attention.w_o",
            ),
            rule(
                "vit.encoder.layer.{i}.attention.output.dense.bias",
                "layers[{i}].attention.o_bias",
            ),
            // FFN weights
            rule(
                "vit.encoder.layer.{i}.intermediate.dense.weight",
                "layers[{i}].ffn.w_up",
            ),
            rule(
                "vit.encoder.layer.{i}.intermediate.dense.bias",
                "layers[{i}].ffn.up_bias",
            ),
            rule(
                "vit.encoder.layer.{i}.output.dense.weight",
                "layers[{i}].ffn.w_down",
            ),
            rule(
                "vit.encoder.layer.{i}.output.dense.bias",
                "layers[{i}].ffn.down_bias",
            ),
            // LayerNorm weights and biases
            rule(
                "vit.encoder.layer.{i}.layernorm_before.weight",
                "layers[{i}].attn_norm.weight",
            ),
            rule(
                "vit.encoder.layer.{i}.layernorm_before.bias",
                "layers[{i}].attn_norm.bias",
            ),
            rule(
                "vit.encoder.layer.{i}.layernorm_after.weight",
                "layers[{i}].ffn_norm.weight",
            ),
            rule(
                "vit.encoder.layer.{i}.layernorm_after.bias",
                "layers[{i}].ffn_norm.bias",
            ),
            // Final norm
            rule("vit.layernorm.weight", "norm.weight"),
            rule("vit.layernorm.bias", "norm.bias"),
            // Classifier head
            rule("classifier.weight", "classifier.weight"),
            rule("classifier.bias", "classifier.bias"),
        ])
    }

    /// Map a single source name. Returns `None` if no rule matches.
    pub fn map_name(&self, source: &str) -> Option<String> {
        for rule in &self.rules {
            if let Some(mapped) = try_match_and_replace(source, &rule.source, &rule.target) {
                return Some(mapped);
            }
        }
        None
    }

    /// Map all source keys. Returns the mapping and any unmapped keys.
    pub fn map_keys(&self, source_keys: &[String]) -> MappingResult {
        let mut mapping = HashMap::new();
        let mut unmapped = Vec::new();

        for key in source_keys {
            match self.map_name(key) {
                Some(target) => {
                    mapping.insert(key.clone(), target);
                }
                None => {
                    unmapped.push(key.clone());
                }
            }
        }

        MappingResult { mapping, unmapped }
    }

    /// Validate that all expected target keys are produced and no source keys are unmapped.
    ///
    /// - Unmapped source keys → `Err(WeightError::UnexpectedKeys(...))`
    /// - Missing expected target keys → `Err(WeightError::MissingKeys(...))`
    pub fn validate(
        &self,
        source_keys: &[String],
        expected_targets: &[String],
    ) -> Result<MappingResult, WeightError> {
        let result = self.map_keys(source_keys);

        if !result.unmapped.is_empty() {
            return Err(WeightError::UnexpectedKeys(result.unmapped.clone()));
        }

        let mapped_targets: HashSet<&str> = result.mapping.values().map(|s| s.as_str()).collect();
        let missing: Vec<String> = expected_targets
            .iter()
            .filter(|t| !mapped_targets.contains(t.as_str()))
            .cloned()
            .collect();

        if !missing.is_empty() {
            return Err(WeightError::MissingKeys(missing));
        }

        Ok(result)
    }
}

/// Match `source` against `source_pattern` (with optional `{i}` placeholder),
/// extracting the captured layer index and substituting it into `target_pattern`.
fn try_match_and_replace(source: &str, source_pattern: &str, target_pattern: &str) -> Option<String> {
    if !source_pattern.contains("{i}") {
        // Exact match, no placeholder
        if source == source_pattern {
            return Some(target_pattern.to_string());
        }
        return None;
    }

    let (prefix, suffix) = source_pattern.split_once("{i}").unwrap();

    if !source.starts_with(prefix) {
        return None;
    }
    if !source.ends_with(suffix) {
        return None;
    }
    // Guard against prefix+suffix overlapping for very short inputs
    if source.len() < prefix.len() + suffix.len() {
        return None;
    }

    let captured = if suffix.is_empty() {
        &source[prefix.len()..]
    } else {
        &source[prefix.len()..source.len() - suffix.len()]
    };

    // Must be a valid non-negative integer (layer index)
    if captured.is_empty() || !captured.chars().all(|c| c.is_ascii_digit()) {
        return None;
    }

    Some(target_pattern.replace("{i}", captured))
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── Pattern matching ─────────────────────────────────────────

    #[test]
    fn exact_match_no_placeholder() {
        let result = try_match_and_replace("model.norm.weight", "model.norm.weight", "norm.weight");
        assert_eq!(result, Some("norm.weight".to_string()));
    }

    #[test]
    fn no_match_returns_none() {
        let result = try_match_and_replace("foo.bar", "model.norm.weight", "norm.weight");
        assert_eq!(result, None);
    }

    #[test]
    fn placeholder_extracts_layer_index() {
        let result = try_match_and_replace(
            "model.layers.12.self_attn.q_proj.weight",
            "model.layers.{i}.self_attn.q_proj.weight",
            "layers[{i}].attention.w_q",
        );
        assert_eq!(result, Some("layers[12].attention.w_q".to_string()));
    }

    #[test]
    fn placeholder_rejects_non_digit() {
        let result = try_match_and_replace(
            "model.layers.abc.self_attn.q_proj.weight",
            "model.layers.{i}.self_attn.q_proj.weight",
            "layers[{i}].attention.w_q",
        );
        assert_eq!(result, None);
    }

    // ── Qwen3 full mapping ───────────────────────────────────────

    fn generate_qwen3_hf_keys(num_layers: usize) -> Vec<String> {
        let mut keys = vec!["model.embed_tokens.weight".to_string()];
        for i in 0..num_layers {
            keys.push(format!("model.layers.{i}.self_attn.q_proj.weight"));
            keys.push(format!("model.layers.{i}.self_attn.k_proj.weight"));
            keys.push(format!("model.layers.{i}.self_attn.v_proj.weight"));
            keys.push(format!("model.layers.{i}.self_attn.o_proj.weight"));
            keys.push(format!("model.layers.{i}.self_attn.q_norm.weight"));
            keys.push(format!("model.layers.{i}.self_attn.k_norm.weight"));
            keys.push(format!("model.layers.{i}.mlp.gate_proj.weight"));
            keys.push(format!("model.layers.{i}.mlp.up_proj.weight"));
            keys.push(format!("model.layers.{i}.mlp.down_proj.weight"));
            keys.push(format!("model.layers.{i}.input_layernorm.weight"));
            keys.push(format!("model.layers.{i}.post_attention_layernorm.weight"));
        }
        keys.push("model.norm.weight".to_string());
        keys.push("lm_head.weight".to_string());
        keys
    }

    fn generate_qwen3_synapse_keys(num_layers: usize) -> Vec<String> {
        let mut keys = vec!["embed_tokens.weight".to_string()];
        for i in 0..num_layers {
            keys.push(format!("layers[{i}].attention.w_q"));
            keys.push(format!("layers[{i}].attention.w_k"));
            keys.push(format!("layers[{i}].attention.w_v"));
            keys.push(format!("layers[{i}].attention.w_o"));
            keys.push(format!("layers[{i}].attention.q_norm"));
            keys.push(format!("layers[{i}].attention.k_norm"));
            keys.push(format!("layers[{i}].ffn.w_gate"));
            keys.push(format!("layers[{i}].ffn.w_up"));
            keys.push(format!("layers[{i}].ffn.w_down"));
            keys.push(format!("layers[{i}].attn_norm.weight"));
            keys.push(format!("layers[{i}].ffn_norm.weight"));
        }
        keys.push("norm.weight".to_string());
        keys.push("lm_head.weight".to_string());
        keys
    }

    #[test]
    fn qwen3_maps_all_28_layers() {
        let num_layers = 28; // Qwen3-0.6B
        let mapper = WeightMapper::qwen3();
        let hf_keys = generate_qwen3_hf_keys(num_layers);
        let synapse_keys = generate_qwen3_synapse_keys(num_layers);

        let result = mapper.validate(&hf_keys, &synapse_keys).unwrap();

        assert!(
            result.unmapped.is_empty(),
            "Unexpected keys: {:?}",
            result.unmapped
        );
        assert_eq!(
            result.mapping.len(),
            hf_keys.len(),
            "All HF keys should be mapped"
        );

        // Verify total: 28 layers × 11 per-layer + 3 global = 311
        assert_eq!(hf_keys.len(), 28 * 11 + 3);
        assert_eq!(result.mapping.len(), 311);
    }

    #[test]
    fn qwen3_individual_mappings_correct() {
        let mapper = WeightMapper::qwen3();

        assert_eq!(
            mapper.map_name("model.embed_tokens.weight"),
            Some("embed_tokens.weight".to_string())
        );
        assert_eq!(
            mapper.map_name("model.layers.0.self_attn.q_proj.weight"),
            Some("layers[0].attention.w_q".to_string())
        );
        assert_eq!(
            mapper.map_name("model.layers.27.mlp.down_proj.weight"),
            Some("layers[27].ffn.w_down".to_string())
        );
        assert_eq!(
            mapper.map_name("model.layers.5.input_layernorm.weight"),
            Some("layers[5].attn_norm.weight".to_string())
        );
        assert_eq!(
            mapper.map_name("model.layers.5.self_attn.q_norm.weight"),
            Some("layers[5].attention.q_norm".to_string())
        );
        assert_eq!(
            mapper.map_name("lm_head.weight"),
            Some("lm_head.weight".to_string())
        );
    }

    // ── Error handling ───────────────────────────────────────────

    #[test]
    fn missing_target_key_produces_error() {
        let mapper = WeightMapper::qwen3();
        // Provide keys for 0 layers — all per-layer targets will be missing
        let source_keys = vec![
            "model.embed_tokens.weight".to_string(),
            "model.norm.weight".to_string(),
            "lm_head.weight".to_string(),
        ];
        let expected_targets = generate_qwen3_synapse_keys(1); // Expect 1 layer

        let result = mapper.validate(&source_keys, &expected_targets);
        assert!(matches!(result, Err(WeightError::MissingKeys(ref keys)) if !keys.is_empty()));
    }

    #[test]
    fn extra_key_produces_unexpected_error() {
        let mapper = WeightMapper::qwen3();
        let mut keys = generate_qwen3_hf_keys(1);
        keys.push("some.unknown.weight".to_string()); // extra key

        let synapse_keys = generate_qwen3_synapse_keys(1);
        let result = mapper.validate(&keys, &synapse_keys);
        assert!(
            matches!(result, Err(WeightError::UnexpectedKeys(ref k)) if k == &["some.unknown.weight"]),
            "Expected UnexpectedKeys error, got: {result:?}"
        );
    }

    #[test]
    fn map_keys_reports_unmapped() {
        let mapper = WeightMapper::qwen3();
        let keys = vec![
            "model.embed_tokens.weight".to_string(),
            "totally.unknown.key".to_string(),
        ];
        let result = mapper.map_keys(&keys);
        assert_eq!(result.mapping.len(), 1);
        assert_eq!(result.unmapped, vec!["totally.unknown.key"]);
    }

    // ── LLaMA / Mistral weight mapping ─────────────────────────────

    /// Generate HF-style keys for LLaMA/Mistral (no q_norm/k_norm).
    fn generate_llama_hf_keys(num_layers: usize) -> Vec<String> {
        let mut keys = vec!["model.embed_tokens.weight".to_string()];
        for i in 0..num_layers {
            keys.push(format!("model.layers.{i}.self_attn.q_proj.weight"));
            keys.push(format!("model.layers.{i}.self_attn.k_proj.weight"));
            keys.push(format!("model.layers.{i}.self_attn.v_proj.weight"));
            keys.push(format!("model.layers.{i}.self_attn.o_proj.weight"));
            keys.push(format!("model.layers.{i}.mlp.gate_proj.weight"));
            keys.push(format!("model.layers.{i}.mlp.up_proj.weight"));
            keys.push(format!("model.layers.{i}.mlp.down_proj.weight"));
            keys.push(format!("model.layers.{i}.input_layernorm.weight"));
            keys.push(format!("model.layers.{i}.post_attention_layernorm.weight"));
        }
        keys.push("model.norm.weight".to_string());
        keys.push("lm_head.weight".to_string());
        keys
    }

    /// Generate Synapse-side keys for LLaMA/Mistral (no q_norm/k_norm).
    fn generate_llama_synapse_keys(num_layers: usize) -> Vec<String> {
        let mut keys = vec!["embed_tokens.weight".to_string()];
        for i in 0..num_layers {
            keys.push(format!("layers[{i}].attention.w_q"));
            keys.push(format!("layers[{i}].attention.w_k"));
            keys.push(format!("layers[{i}].attention.w_v"));
            keys.push(format!("layers[{i}].attention.w_o"));
            keys.push(format!("layers[{i}].ffn.w_gate"));
            keys.push(format!("layers[{i}].ffn.w_up"));
            keys.push(format!("layers[{i}].ffn.w_down"));
            keys.push(format!("layers[{i}].attn_norm.weight"));
            keys.push(format!("layers[{i}].ffn_norm.weight"));
        }
        keys.push("norm.weight".to_string());
        keys.push("lm_head.weight".to_string());
        keys
    }

    #[test]
    fn llama_maps_all_32_layers() {
        let num_layers = 32; // LLaMA-7B
        let mapper = WeightMapper::llama();
        let hf_keys = generate_llama_hf_keys(num_layers);
        let synapse_keys = generate_llama_synapse_keys(num_layers);

        let result = mapper.validate(&hf_keys, &synapse_keys).unwrap();

        assert!(
            result.unmapped.is_empty(),
            "Unexpected keys: {:?}",
            result.unmapped
        );
        assert_eq!(
            result.mapping.len(),
            hf_keys.len(),
            "All HF keys should be mapped"
        );

        // Verify total: 32 layers x 9 per-layer + 3 global = 291
        assert_eq!(hf_keys.len(), 32 * 9 + 3);
        assert_eq!(result.mapping.len(), 291);
    }

    #[test]
    fn llama_individual_mappings_correct() {
        let mapper = WeightMapper::llama();

        assert_eq!(
            mapper.map_name("model.embed_tokens.weight"),
            Some("embed_tokens.weight".to_string())
        );
        assert_eq!(
            mapper.map_name("model.layers.0.self_attn.q_proj.weight"),
            Some("layers[0].attention.w_q".to_string())
        );
        assert_eq!(
            mapper.map_name("model.layers.31.mlp.down_proj.weight"),
            Some("layers[31].ffn.w_down".to_string())
        );
        assert_eq!(
            mapper.map_name("model.layers.5.input_layernorm.weight"),
            Some("layers[5].attn_norm.weight".to_string())
        );
        // LLaMA should NOT map q_norm/k_norm
        assert_eq!(
            mapper.map_name("model.layers.5.self_attn.q_norm.weight"),
            None
        );
        assert_eq!(
            mapper.map_name("model.layers.5.self_attn.k_norm.weight"),
            None
        );
        assert_eq!(
            mapper.map_name("lm_head.weight"),
            Some("lm_head.weight".to_string())
        );
    }

    #[test]
    fn mistral_maps_all_32_layers() {
        let num_layers = 32; // Mistral-7B
        let mapper = WeightMapper::mistral();
        let hf_keys = generate_llama_hf_keys(num_layers);
        let synapse_keys = generate_llama_synapse_keys(num_layers);

        let result = mapper.validate(&hf_keys, &synapse_keys).unwrap();

        assert!(
            result.unmapped.is_empty(),
            "Unexpected keys: {:?}",
            result.unmapped
        );
        assert_eq!(
            result.mapping.len(),
            hf_keys.len(),
            "All HF keys should be mapped"
        );
    }

    #[test]
    fn mistral_individual_mappings_correct() {
        let mapper = WeightMapper::mistral();

        assert_eq!(
            mapper.map_name("model.embed_tokens.weight"),
            Some("embed_tokens.weight".to_string())
        );
        assert_eq!(
            mapper.map_name("model.layers.0.self_attn.q_proj.weight"),
            Some("layers[0].attention.w_q".to_string())
        );
        assert_eq!(
            mapper.map_name("model.layers.15.self_attn.v_proj.weight"),
            Some("layers[15].attention.w_v".to_string())
        );
        // Mistral should NOT map q_norm/k_norm
        assert_eq!(
            mapper.map_name("model.layers.0.self_attn.q_norm.weight"),
            None
        );
        assert_eq!(
            mapper.map_name("model.layers.0.self_attn.k_norm.weight"),
            None
        );
        assert_eq!(
            mapper.map_name("model.norm.weight"),
            Some("norm.weight".to_string())
        );
    }

    // ── from_model_type auto-detection ─────────────────────────────

    #[test]
    fn from_model_type_selects_qwen3() {
        let mapper = WeightMapper::from_model_type("qwen3").unwrap();
        // Qwen3 mapper should handle q_norm
        assert_eq!(
            mapper.map_name("model.layers.0.self_attn.q_norm.weight"),
            Some("layers[0].attention.q_norm".to_string())
        );
    }

    #[test]
    fn from_model_type_selects_llama() {
        let mapper = WeightMapper::from_model_type("llama").unwrap();
        // LLaMA mapper should NOT handle q_norm
        assert_eq!(
            mapper.map_name("model.layers.0.self_attn.q_norm.weight"),
            None
        );
    }

    #[test]
    fn from_model_type_selects_mistral() {
        let mapper = WeightMapper::from_model_type("mistral").unwrap();
        // Mistral mapper should NOT handle q_norm
        assert_eq!(
            mapper.map_name("model.layers.0.self_attn.q_norm.weight"),
            None
        );
    }

    #[test]
    fn from_model_type_rejects_unknown() {
        let result = WeightMapper::from_model_type("gpt2");
        assert!(
            matches!(result, Err(WeightError::InvalidFormat(ref msg)) if msg.contains("gpt2")),
            "Expected InvalidFormat error for unsupported model type, got: {result:?}"
        );
    }

    // ── Phi weight mapping ──────────────────────────────────────────

    #[test]
    fn phi_individual_mappings_correct() {
        let mapper = WeightMapper::phi();

        assert_eq!(
            mapper.map_name("model.embed_tokens.weight"),
            Some("embed_tokens.weight".to_string())
        );
        assert_eq!(
            mapper.map_name("model.layers.0.self_attn.q_proj.weight"),
            Some("layers[0].attention.w_q".to_string())
        );
        assert_eq!(
            mapper.map_name("model.layers.0.self_attn.k_proj.weight"),
            Some("layers[0].attention.w_k".to_string())
        );
        assert_eq!(
            mapper.map_name("model.layers.0.self_attn.v_proj.weight"),
            Some("layers[0].attention.w_v".to_string())
        );
        assert_eq!(
            mapper.map_name("model.layers.0.self_attn.o_proj.weight"),
            Some("layers[0].attention.w_o".to_string())
        );
        assert_eq!(
            mapper.map_name("model.layers.15.mlp.gate_proj.weight"),
            Some("layers[15].ffn.w_gate".to_string())
        );
        assert_eq!(
            mapper.map_name("model.layers.15.mlp.up_proj.weight"),
            Some("layers[15].ffn.w_up".to_string())
        );
        assert_eq!(
            mapper.map_name("model.layers.15.mlp.down_proj.weight"),
            Some("layers[15].ffn.w_down".to_string())
        );
        assert_eq!(
            mapper.map_name("model.layers.5.input_layernorm.weight"),
            Some("layers[5].attn_norm.weight".to_string())
        );
        assert_eq!(
            mapper.map_name("model.layers.5.post_attention_layernorm.weight"),
            Some("layers[5].ffn_norm.weight".to_string())
        );
        assert_eq!(
            mapper.map_name("model.norm.weight"),
            Some("norm.weight".to_string())
        );
        assert_eq!(
            mapper.map_name("lm_head.weight"),
            Some("lm_head.weight".to_string())
        );
        // Phi should NOT map q_norm/k_norm
        assert_eq!(
            mapper.map_name("model.layers.0.self_attn.q_norm.weight"),
            None
        );
        assert_eq!(
            mapper.map_name("model.layers.0.self_attn.k_norm.weight"),
            None
        );
    }

    // ── Gemma weight mapping ────────────────────────────────────────

    #[test]
    fn gemma_individual_mappings_correct() {
        let mapper = WeightMapper::gemma();

        assert_eq!(
            mapper.map_name("model.embed_tokens.weight"),
            Some("embed_tokens.weight".to_string())
        );
        assert_eq!(
            mapper.map_name("model.layers.0.self_attn.q_proj.weight"),
            Some("layers[0].attention.w_q".to_string())
        );
        assert_eq!(
            mapper.map_name("model.layers.0.self_attn.k_proj.weight"),
            Some("layers[0].attention.w_k".to_string())
        );
        assert_eq!(
            mapper.map_name("model.layers.0.self_attn.v_proj.weight"),
            Some("layers[0].attention.w_v".to_string())
        );
        assert_eq!(
            mapper.map_name("model.layers.0.self_attn.o_proj.weight"),
            Some("layers[0].attention.w_o".to_string())
        );
        assert_eq!(
            mapper.map_name("model.layers.17.mlp.gate_proj.weight"),
            Some("layers[17].ffn.w_gate".to_string())
        );
        assert_eq!(
            mapper.map_name("model.layers.17.mlp.up_proj.weight"),
            Some("layers[17].ffn.w_up".to_string())
        );
        assert_eq!(
            mapper.map_name("model.layers.17.mlp.down_proj.weight"),
            Some("layers[17].ffn.w_down".to_string())
        );
        assert_eq!(
            mapper.map_name("model.layers.5.input_layernorm.weight"),
            Some("layers[5].attn_norm.weight".to_string())
        );
        assert_eq!(
            mapper.map_name("model.layers.5.post_attention_layernorm.weight"),
            Some("layers[5].ffn_norm.weight".to_string())
        );
        assert_eq!(
            mapper.map_name("model.norm.weight"),
            Some("norm.weight".to_string())
        );
        assert_eq!(
            mapper.map_name("lm_head.weight"),
            Some("lm_head.weight".to_string())
        );
        // Gemma should NOT map q_norm/k_norm
        assert_eq!(
            mapper.map_name("model.layers.0.self_attn.q_norm.weight"),
            None
        );
        assert_eq!(
            mapper.map_name("model.layers.0.self_attn.k_norm.weight"),
            None
        );
    }

    // ── from_model_type for phi / gemma ─────────────────────────────

    #[test]
    fn from_model_type_selects_phi() {
        let mapper = WeightMapper::from_model_type("phi3").unwrap();
        // Phi mapper should handle standard projections
        assert_eq!(
            mapper.map_name("model.layers.0.self_attn.q_proj.weight"),
            Some("layers[0].attention.w_q".to_string())
        );
        // But NOT q_norm/k_norm
        assert_eq!(
            mapper.map_name("model.layers.0.self_attn.q_norm.weight"),
            None
        );

        // Also works with bare "phi" alias
        let mapper2 = WeightMapper::from_model_type("phi").unwrap();
        assert_eq!(
            mapper2.map_name("model.layers.0.self_attn.q_proj.weight"),
            Some("layers[0].attention.w_q".to_string())
        );
    }

    #[test]
    fn from_model_type_selects_gemma() {
        let mapper = WeightMapper::from_model_type("gemma2").unwrap();
        // Gemma mapper should handle standard projections
        assert_eq!(
            mapper.map_name("model.layers.0.self_attn.q_proj.weight"),
            Some("layers[0].attention.w_q".to_string())
        );
        // But NOT q_norm/k_norm
        assert_eq!(
            mapper.map_name("model.layers.0.self_attn.q_norm.weight"),
            None
        );

        // Also works with bare "gemma" alias
        let mapper2 = WeightMapper::from_model_type("gemma").unwrap();
        assert_eq!(
            mapper2.map_name("model.layers.0.self_attn.q_proj.weight"),
            Some("layers[0].attention.w_q".to_string())
        );
    }

    // ── CLIP weight mapping ───────────────────────────────────────────

    #[test]
    fn clip_vision_mappings_correct() {
        let mapper = WeightMapper::clip();

        assert_eq!(
            mapper.map_name("vision_model.embeddings.patch_embedding.weight"),
            Some("vision.patch_proj".to_string())
        );
        assert_eq!(
            mapper.map_name("vision_model.embeddings.class_embedding"),
            Some("vision.cls_token".to_string())
        );
        assert_eq!(
            mapper.map_name("vision_model.embeddings.position_embedding.weight"),
            Some("vision.pos_embed".to_string())
        );
        assert_eq!(
            mapper.map_name("vision_model.encoder.layers.0.self_attn.q_proj.weight"),
            Some("vision.layers[0].attention.w_q".to_string())
        );
        assert_eq!(
            mapper.map_name("vision_model.encoder.layers.5.self_attn.out_proj.bias"),
            Some("vision.layers[5].attention.o_bias".to_string())
        );
        assert_eq!(
            mapper.map_name("vision_model.encoder.layers.11.mlp.fc1.weight"),
            Some("vision.layers[11].ffn.w_up".to_string())
        );
        assert_eq!(
            mapper.map_name("vision_model.encoder.layers.3.layer_norm1.weight"),
            Some("vision.layers[3].attn_norm.weight".to_string())
        );
        assert_eq!(
            mapper.map_name("vision_model.pre_layernorm.weight"),
            Some("vision.pre_norm.weight".to_string())
        );
        assert_eq!(
            mapper.map_name("vision_model.post_layernorm.weight"),
            Some("vision.norm.weight".to_string())
        );
    }

    #[test]
    fn clip_text_mappings_correct() {
        let mapper = WeightMapper::clip();

        assert_eq!(
            mapper.map_name("text_model.embeddings.token_embedding.weight"),
            Some("text.embeddings".to_string())
        );
        assert_eq!(
            mapper.map_name("text_model.embeddings.position_embedding.weight"),
            Some("text.pos_embed".to_string())
        );
        assert_eq!(
            mapper.map_name("text_model.encoder.layers.0.self_attn.q_proj.weight"),
            Some("text.layers[0].attention.w_q".to_string())
        );
        assert_eq!(
            mapper.map_name("text_model.encoder.layers.7.mlp.fc2.bias"),
            Some("text.layers[7].ffn.down_bias".to_string())
        );
        assert_eq!(
            mapper.map_name("text_model.encoder.layers.2.layer_norm2.weight"),
            Some("text.layers[2].ffn_norm.weight".to_string())
        );
        assert_eq!(
            mapper.map_name("text_model.final_layer_norm.weight"),
            Some("text.norm.weight".to_string())
        );
        assert_eq!(
            mapper.map_name("text_model.final_layer_norm.bias"),
            Some("text.norm.bias".to_string())
        );
    }

    #[test]
    fn clip_projection_mappings_correct() {
        let mapper = WeightMapper::clip();

        assert_eq!(
            mapper.map_name("visual_projection.weight"),
            Some("vision_proj".to_string())
        );
        assert_eq!(
            mapper.map_name("text_projection.weight"),
            Some("text_proj".to_string())
        );
    }

    #[test]
    fn from_model_type_selects_clip() {
        let mapper = WeightMapper::from_model_type("clip").unwrap();
        assert_eq!(
            mapper.map_name("vision_model.encoder.layers.0.self_attn.q_proj.weight"),
            Some("vision.layers[0].attention.w_q".to_string())
        );
        assert_eq!(
            mapper.map_name("text_model.encoder.layers.0.self_attn.q_proj.weight"),
            Some("text.layers[0].attention.w_q".to_string())
        );
    }

    // ── DINOv2 weight mapping ─────────────────────────────────────────

    #[test]
    fn dinov2_individual_mappings_correct() {
        let mapper = WeightMapper::dinov2();

        assert_eq!(
            mapper.map_name("embeddings.patch_embeddings.projection.weight"),
            Some("patch_proj".to_string())
        );
        assert_eq!(
            mapper.map_name("embeddings.patch_embeddings.projection.bias"),
            Some("patch_proj_bias".to_string())
        );
        assert_eq!(
            mapper.map_name("embeddings.cls_token"),
            Some("cls_token".to_string())
        );
        assert_eq!(
            mapper.map_name("embeddings.position_embeddings"),
            Some("pos_embed".to_string())
        );
        assert_eq!(
            mapper.map_name("encoder.layer.0.attention.attention.query.weight"),
            Some("layers[0].attention.w_q".to_string())
        );
        assert_eq!(
            mapper.map_name("encoder.layer.5.attention.attention.value.bias"),
            Some("layers[5].attention.v_bias".to_string())
        );
        assert_eq!(
            mapper.map_name("encoder.layer.11.intermediate.dense.weight"),
            Some("layers[11].ffn.w_up".to_string())
        );
        assert_eq!(
            mapper.map_name("encoder.layer.3.norm1.weight"),
            Some("layers[3].attn_norm.weight".to_string())
        );
        assert_eq!(
            mapper.map_name("encoder.layer.3.norm2.bias"),
            Some("layers[3].ffn_norm.bias".to_string())
        );
        assert_eq!(
            mapper.map_name("layernorm.weight"),
            Some("norm.weight".to_string())
        );
        assert_eq!(
            mapper.map_name("layernorm.bias"),
            Some("norm.bias".to_string())
        );
    }

    #[test]
    fn from_model_type_selects_dinov2() {
        let mapper = WeightMapper::from_model_type("dinov2").unwrap();
        assert_eq!(
            mapper.map_name("encoder.layer.0.attention.attention.query.weight"),
            Some("layers[0].attention.w_q".to_string())
        );
    }
}
