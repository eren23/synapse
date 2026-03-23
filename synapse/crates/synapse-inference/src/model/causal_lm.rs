use std::collections::{HashMap, HashSet};

use crate::config::ModelConfig;
use crate::model::decoder_layer::{apply_norm, matmul_t};
use crate::registry::NormVariant;
use crate::weight_loading::{RawTensor, WeightMapper};

use super::DecoderLayer;

/// Output from a forward pass.
pub struct ModelOutput {
    /// Flat logits: `[seq_len * vocab_size]`.
    pub logits: Vec<f32>,
    /// Logical shape `[batch, seq_len, vocab_size]`.
    pub shape: [usize; 3],
}

/// Result of weight loading: lists of missing and unexpected keys.
pub struct LoadResult {
    pub missing: Vec<String>,
    pub unexpected: Vec<String>,
}

/// A causal language model: embedding → N × DecoderLayer → norm → lm_head.
pub struct CausalLM {
    pub config: ModelConfig,
    pub layers: Vec<DecoderLayer>,
    pub final_norm: Box<dyn NormVariant>,

    // ── Weights ──────────────────────────────────────────────────────
    pub embed_tokens: Vec<f32>,
    pub final_norm_weight: Vec<f32>,
    /// `None` when `tie_word_embeddings` is true (reuses `embed_tokens`).
    pub lm_head_weight: Option<Vec<f32>>,
}

impl CausalLM {
    /// Total number of unique trainable parameters.
    pub fn param_count(&self) -> usize {
        let arch = &self.config.architecture;
        let h = arch.hidden_size;

        let embed = arch.vocab_size * h;
        let layers: usize = self.layers.iter().map(|l| l.param_count()).sum();
        let norm = h;
        let lm_head = if arch.tie_word_embeddings { 0 } else { arch.vocab_size * h };

        embed + layers + norm + lm_head
    }

    /// All target weight keys this model expects (Synapse naming).
    pub fn expected_weight_keys(&self) -> Vec<String> {
        let mut keys = vec!["embed_tokens.weight".to_string()];
        for (i, layer) in self.layers.iter().enumerate() {
            keys.extend(layer.weight_keys(i));
        }
        keys.push("norm.weight".to_string());
        keys.push("lm_head.weight".to_string());
        keys
    }

    /// Load weights from source tensors using a name mapper.
    ///
    /// Returns missing (expected but not provided) and unexpected (provided but
    /// not expected) keys. Both should be empty for a correct checkpoint.
    pub fn load_weights(
        &mut self,
        weights: HashMap<String, RawTensor>,
        mapper: &WeightMapper,
    ) -> LoadResult {
        let source_keys: Vec<String> = weights.keys().cloned().collect();
        let mapping = mapper.map_keys(&source_keys);

        let expected: HashSet<String> = self.expected_weight_keys().into_iter().collect();
        let mut loaded = HashSet::new();

        for (source, target) in &mapping.mapping {
            if expected.contains(target) {
                if let Some(raw) = weights.get(source) {
                    self.set_weight(target, &raw.data);
                    loaded.insert(target.clone());
                }
            }
        }

        let missing: Vec<String> = expected.difference(&loaded).cloned().collect();
        let unexpected = mapping.unmapped;

        LoadResult { missing, unexpected }
    }

    /// Forward pass: token_ids → logits.
    ///
    /// `token_ids` is a 1-D slice of token indices for a single sequence.
    /// Returns `ModelOutput` with shape `[1, seq_len, vocab_size]`.
    pub fn forward(&self, token_ids: &[u32]) -> ModelOutput {
        let seq_len = token_ids.len();
        let h = self.config.architecture.hidden_size;
        let vocab = self.config.architecture.vocab_size;

        // 1. Embedding lookup → [seq_len, h]
        let mut x = vec![0.0f32; seq_len * h];
        for (t, &id) in token_ids.iter().enumerate() {
            let id = id as usize;
            if id < vocab {
                let src = &self.embed_tokens[id * h..(id + 1) * h];
                x[t * h..(t + 1) * h].copy_from_slice(src);
            }
        }

        // 2. Decoder layers
        for layer in &self.layers {
            x = layer.forward(&x, seq_len);
        }

        // 3. Final norm
        x = apply_norm(&x, &self.final_norm_weight, &*self.final_norm, h);

        // 4. LM head projection → [seq_len, vocab]
        let lm_weight = self.lm_head_weight.as_ref().unwrap_or(&self.embed_tokens);
        let logits = matmul_t(&x, lm_weight, seq_len, h, vocab);

        ModelOutput {
            logits,
            shape: [1, seq_len, vocab],
        }
    }

    // ── Internal ─────────────────────────────────────────────────────

    fn set_weight(&mut self, key: &str, data: &[f32]) {
        match key {
            "embed_tokens.weight" => self.embed_tokens = data.to_vec(),
            "norm.weight" => self.final_norm_weight = data.to_vec(),
            "lm_head.weight" => {
                if !self.config.architecture.tie_word_embeddings {
                    self.lm_head_weight = Some(data.to_vec());
                }
                // For tied models, accept the key but reuse embed_tokens.
            }
            _ if key.starts_with("layers[") => {
                if let Some((idx, field)) = parse_layer_key(key) {
                    if let Some(layer) = self.layers.get_mut(idx) {
                        layer.set_weight(field, data);
                    }
                }
            }
            _ => {}
        }
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
