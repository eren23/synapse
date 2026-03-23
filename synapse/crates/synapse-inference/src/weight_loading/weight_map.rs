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

        // Verify total: 28 layers × 9 per-layer + 3 global = 255
        assert_eq!(hf_keys.len(), 28 * 9 + 3);
        assert_eq!(result.mapping.len(), 255);
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
}
