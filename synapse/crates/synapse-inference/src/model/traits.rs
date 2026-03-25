use crate::config::ModelConfig;
use crate::kv_cache::KVCache;
use super::causal_lm::ModelOutput;

/// Trait for models that can run forward passes for inference.
///
/// Implemented by both [`CausalLM`](super::CausalLM) (f32) and
/// [`QuantizedCausalLM`](crate::quantization::QuantizedCausalLM) (INT8),
/// allowing the generation pipeline to be generic over model precision.
pub trait Model {
    /// Full forward pass (no cache, recomputes everything).
    fn forward(&self, token_ids: &[u32]) -> ModelOutput;

    /// Prefill: process all prompt tokens, populate KV cache, return last logits.
    fn forward_prefill(&self, token_ids: &[u32], cache: &mut KVCache) -> ModelOutput;

    /// Single-token decode using KV cache.
    fn forward_one(&self, token: u32, cache: &mut KVCache) -> ModelOutput;

    /// Draft forward with fewer layers (for speculative decoding).
    /// Default: falls back to full forward_one.
    fn forward_one_draft(&self, token: u32, cache: &mut KVCache, _n_layers: usize) -> ModelOutput {
        self.forward_one(token, cache)
    }

    /// Number of decoder layers.
    fn num_layers(&self) -> usize;

    /// Model configuration.
    fn config(&self) -> &ModelConfig;
}
