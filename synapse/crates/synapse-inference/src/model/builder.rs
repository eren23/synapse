use crate::config::ModelConfig;
use crate::registry::{create_attention, create_ffn, create_norm};
use crate::weight_loading::AlignedBuffer;

use super::causal_lm::CausalLM;
use super::decoder_layer::DecoderLayer;

/// Assembles a [`CausalLM`] from a [`ModelConfig`].
///
/// The returned model has the correct structure (layers, norms, attention
/// geometry) but weights are uninitialized (empty buffers). Call
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
                attn_norm_weight: AlignedBuffer::new_zeroed(0),
                w_q: AlignedBuffer::new_zeroed(0),
                w_k: AlignedBuffer::new_zeroed(0),
                w_v: AlignedBuffer::new_zeroed(0),
                w_o: AlignedBuffer::new_zeroed(0),
                q_norm_weight: AlignedBuffer::new_zeroed(0),
                k_norm_weight: AlignedBuffer::new_zeroed(0),
                ffn_norm_weight: AlignedBuffer::new_zeroed(0),
                ffn_gate: AlignedBuffer::new_zeroed(0),
                ffn_up: AlignedBuffer::new_zeroed(0),
                ffn_down: AlignedBuffer::new_zeroed(0),
            });
        }

        CausalLM {
            final_norm: create_norm(&config.norm),
            layers,
            embed_tokens: AlignedBuffer::new_zeroed(0),
            final_norm_weight: AlignedBuffer::new_zeroed(0),
            lm_head_weight: if arch.tie_word_embeddings {
                None
            } else {
                Some(AlignedBuffer::new_zeroed(0))
            },
            config: config.clone(),
        }
    }
}
