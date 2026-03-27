use super::causal_lm::ModelOutput;
use crate::config::ModelConfig;
use crate::kv_cache::KVCache;

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

    /// Prefill with backend dispatch.
    /// Default: falls back to CPU forward_prefill.
    #[cfg(feature = "metal")]
    fn forward_prefill_gpu(
        &self,
        token_ids: &[u32],
        cache: &mut KVCache,
        backend: &crate::metal::ComputeBackend,
    ) -> ModelOutput {
        let _ = backend;
        self.forward_prefill(token_ids, cache)
    }

    /// Single-token decode using KV cache.
    fn forward_one(&self, token: u32, cache: &mut KVCache) -> ModelOutput;

    /// Draft forward with fewer layers (for speculative decoding).
    /// Default: falls back to full forward_one.
    fn forward_one_draft(&self, token: u32, cache: &mut KVCache, _n_layers: usize) -> ModelOutput {
        self.forward_one(token, cache)
    }

    /// Single-token decode with Metal GPU backend.
    /// Default: falls back to CPU forward_one.
    #[cfg(feature = "metal")]
    fn forward_one_gpu(
        &self,
        token: u32,
        cache: &mut KVCache,
        backend: &crate::metal::ComputeBackend,
    ) -> ModelOutput {
        let _ = backend;
        self.forward_one(token, cache)
    }

    /// GPU-resident single-token decode: all layers in one command buffer.
    /// Default: falls back to forward_one_gpu (per-layer dispatch).
    #[cfg(feature = "metal")]
    fn forward_one_gpu_resident(
        &self,
        token: u32,
        model_bufs: &mut crate::metal::gpu_buffers::MetalModelBuffers,
        backend: &crate::metal::MetalBackend,
    ) -> ModelOutput {
        let _ = (token, model_bufs, backend);
        unimplemented!("GPU-resident forward not supported for this model type")
    }

    /// Number of decoder layers.
    fn num_layers(&self) -> usize;

    /// Model configuration.
    fn config(&self) -> &ModelConfig;
}
