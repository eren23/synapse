use crate::config::ModelConfig;
use crate::registry::{create_attention, create_ffn, create_norm};

use super::causal_lm::CausalLM;
use super::decoder_layer::DecoderLayer;

/// Assembles a [`CausalLM`] from a [`ModelConfig`].
///
/// The returned model has the correct structure (layers, norms, attention
/// geometry) but weights are uninitialized (empty `Vec`s). Call
/// [`CausalLM::load_weights`] to populate them before inference.
pub struct ModelBuilder;

impl ModelBuilder {
    /// Build a `CausalLM` from a model configuration.
    ///
    /// Creates embedding + N decoder layers + final norm + lm_head.
    /// Handles `tie_word_embeddings`: when true, `lm_head_weight` is `None`.
    pub fn from_config(config: &ModelConfig) -> CausalLM {
        let arch = &config.architecture;
        let num_layers = arch.num_layers;

        let mut layers = Vec::with_capacity(num_layers);
        for _ in 0..num_layers {
            layers.push(DecoderLayer {
                attn_norm: create_norm(&config.norm),
                attention: create_attention(&config.attention),
                ffn_norm: create_norm(&config.norm),
                ffn: create_ffn(&config.ffn),
                hidden_size: arch.hidden_size,
                attn_norm_weight: Vec::new(),
                w_q: Vec::new(),
                w_k: Vec::new(),
                w_v: Vec::new(),
                w_o: Vec::new(),
                ffn_norm_weight: Vec::new(),
                ffn_gate: Vec::new(),
                ffn_up: Vec::new(),
                ffn_down: Vec::new(),
            });
        }

        CausalLM {
            final_norm: create_norm(&config.norm),
            layers,
            embed_tokens: Vec::new(),
            final_norm_weight: Vec::new(),
            lm_head_weight: if arch.tie_word_embeddings {
                None
            } else {
                Some(Vec::new())
            },
            config: config.clone(),
        }
    }
}
