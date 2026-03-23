/// Trait for KV-cache allocation strategies.
///
/// Current implementation: `PreAllocatedStrategy` — allocates all memory
/// up-front at init, zero allocations during inference.
///
/// Future strategies:
/// - `PagedStrategy`: PagedAttention-style block allocation for dynamic batching.
/// - `SlidingWindowStrategy`: Fixed window with eviction for long-context models
///   (e.g. Mistral's sliding-window attention).
pub trait CacheStrategy {
    fn name(&self) -> &str;
    fn max_seq_len(&self) -> usize;
}

/// Pre-allocates contiguous K/V buffers for `max_seq_len` tokens per layer.
/// All memory is allocated at construction; append/slice/reset are zero-alloc.
#[derive(Debug)]
pub struct PreAllocatedStrategy {
    max_seq_len: usize,
}

impl PreAllocatedStrategy {
    pub fn new(max_seq_len: usize) -> Self {
        Self { max_seq_len }
    }
}

impl CacheStrategy for PreAllocatedStrategy {
    fn name(&self) -> &str {
        "PreAllocated"
    }

    fn max_seq_len(&self) -> usize {
        self.max_seq_len
    }
}
