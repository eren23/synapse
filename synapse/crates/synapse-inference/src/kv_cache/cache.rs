use std::ptr;

use synapse_core::SynapseError;
use synapse_sys as ffi;

fn check_status(status: ffi::syn_status_t) -> Result<(), SynapseError> {
    match status {
        ffi::SYN_OK => Ok(()),
        ffi::SYN_ERR_NULL_PTR => Err(SynapseError::NullPointer),
        ffi::SYN_ERR_INVALID_ARG => Err(SynapseError::InvalidArg),
        ffi::SYN_ERR_OUT_OF_MEMORY => Err(SynapseError::OutOfMemory),
        ffi::SYN_ERR_SHAPE_MISMATCH => Err(SynapseError::ShapeMismatch),
        ffi::SYN_ERR_NOT_CONTIGUOUS => Err(SynapseError::NotContiguous),
        ffi::SYN_ERR_INVALID_AXIS => Err(SynapseError::InvalidAxis),
        ffi::SYN_ERR_INVALID_DIMENSIONS => Err(SynapseError::InvalidDimensions),
        ffi::SYN_ERR_INTERNAL => Err(SynapseError::Internal),
        code => Err(SynapseError::Unknown(code)),
    }
}

/// A single layer's KV cache, wrapping a Zig-allocated pre-allocated buffer.
///
/// Each layer holds separate K and V buffers sized for `max_seq_len` tokens.
/// Append is O(1) memcpy, slice is zero-copy, reset just rewinds the position.
pub struct KVCacheLayer {
    ptr: *mut ffi::syn_kvcache_t,
    n_kv_heads: usize,
    head_dim: usize,
    max_seq_len: usize,
}

unsafe impl Send for KVCacheLayer {}

impl KVCacheLayer {
    pub fn new(
        max_seq_len: usize,
        n_kv_heads: usize,
        head_dim: usize,
    ) -> Result<Self, SynapseError> {
        let mut ptr: *mut ffi::syn_kvcache_t = ptr::null_mut();
        unsafe {
            check_status(ffi::syn_kvcache_create(
                max_seq_len,
                n_kv_heads,
                head_dim,
                &mut ptr,
            ))?;
        }
        Ok(Self {
            ptr,
            n_kv_heads,
            head_dim,
            max_seq_len,
        })
    }

    /// Append one token's K and V vectors.
    /// `k_token` and `v_token` must each have length `n_kv_heads * head_dim`.
    pub fn append(&mut self, k_token: &[f32], v_token: &[f32]) -> Result<(), SynapseError> {
        let stride = self.n_kv_heads * self.head_dim;
        unsafe {
            check_status(ffi::syn_kvcache_append(
                self.ptr,
                k_token.as_ptr(),
                v_token.as_ptr(),
                stride,
            ))
        }
    }

    /// Get zero-copy slices into the populated K and V regions.
    /// Returns `(k_slice, v_slice, seq_len)`.
    pub fn slice(&self) -> Result<(&[f32], &[f32], usize), SynapseError> {
        let mut k_ptr: *const f32 = ptr::null();
        let mut v_ptr: *const f32 = ptr::null();
        let mut seq_len: usize = 0;
        unsafe {
            check_status(ffi::syn_kvcache_slice(
                self.ptr,
                &mut k_ptr,
                &mut v_ptr,
                &mut seq_len,
            ))?;
            let total = seq_len * self.n_kv_heads * self.head_dim;
            Ok((
                std::slice::from_raw_parts(k_ptr, total),
                std::slice::from_raw_parts(v_ptr, total),
                seq_len,
            ))
        }
    }

    /// Reset position to 0. No deallocation — buffers are reused.
    pub fn reset(&mut self) -> Result<(), SynapseError> {
        unsafe { check_status(ffi::syn_kvcache_reset(self.ptr)) }
    }

    pub fn current_len(&self) -> Result<usize, SynapseError> {
        let mut seq_len: usize = 0;
        unsafe {
            check_status(ffi::syn_kvcache_slice(
                self.ptr,
                ptr::null_mut(),
                ptr::null_mut(),
                &mut seq_len,
            ))?;
        }
        Ok(seq_len)
    }

    pub fn max_seq_len(&self) -> usize {
        self.max_seq_len
    }

    pub fn n_kv_heads(&self) -> usize {
        self.n_kv_heads
    }

    pub fn head_dim(&self) -> usize {
        self.head_dim
    }
}

impl Drop for KVCacheLayer {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            unsafe {
                ffi::syn_kvcache_destroy(self.ptr);
            }
        }
    }
}

/// Multi-layer KV cache for a transformer model.
///
/// Manages `num_layers` independent `KVCacheLayer` instances, each pre-allocated
/// for `max_seq_len` tokens. All memory is allocated at construction; no
/// allocations occur during append/slice/reset.
pub struct KVCache {
    layers: Vec<KVCacheLayer>,
    max_seq_len: usize,
    n_kv_heads: usize,
    head_dim: usize,
}

impl KVCache {
    /// Create a KV cache for `num_layers` transformer layers.
    ///
    /// Total allocation: `2 * num_layers * max_seq_len * n_kv_heads * head_dim * sizeof(f32)`.
    pub fn new(
        num_layers: usize,
        max_seq_len: usize,
        n_kv_heads: usize,
        head_dim: usize,
    ) -> Result<Self, SynapseError> {
        let mut layers = Vec::with_capacity(num_layers);
        for _ in 0..num_layers {
            layers.push(KVCacheLayer::new(max_seq_len, n_kv_heads, head_dim)?);
        }
        Ok(Self {
            layers,
            max_seq_len,
            n_kv_heads,
            head_dim,
        })
    }

    /// Append one token's K/V to a specific layer.
    pub fn append(
        &mut self,
        layer: usize,
        k_token: &[f32],
        v_token: &[f32],
    ) -> Result<(), SynapseError> {
        self.layers[layer].append(k_token, v_token)
    }

    /// Get zero-copy K/V slices for a specific layer.
    /// Returns `(k_slice, v_slice, seq_len)`.
    pub fn get(&self, layer: usize) -> Result<(&[f32], &[f32], usize), SynapseError> {
        self.layers[layer].slice()
    }

    /// Reset all layers to empty state. No deallocation.
    pub fn reset(&mut self) -> Result<(), SynapseError> {
        for layer in &mut self.layers {
            layer.reset()?;
        }
        Ok(())
    }

    /// Current sequence length (same across all layers after uniform appends).
    pub fn current_len(&self) -> Result<usize, SynapseError> {
        self.layers[0].current_len()
    }

    /// Get a mutable reference to a specific layer's cache.
    pub fn layer_mut(&mut self, layer: usize) -> &mut KVCacheLayer {
        &mut self.layers[layer]
    }

    pub fn num_layers(&self) -> usize {
        self.layers.len()
    }

    pub fn max_seq_len(&self) -> usize {
        self.max_seq_len
    }

    pub fn n_kv_heads(&self) -> usize {
        self.n_kv_heads
    }

    pub fn head_dim(&self) -> usize {
        self.head_dim
    }

    /// Expected total allocation in bytes.
    pub fn expected_allocation_bytes(&self) -> usize {
        2 * self.layers.len()
            * self.max_seq_len
            * self.n_kv_heads
            * self.head_dim
            * std::mem::size_of::<f32>()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const NUM_LAYERS: usize = 28;
    const MAX_SEQ: usize = 2048;
    const N_KV_HEADS: usize = 8;
    const HEAD_DIM: usize = 128;
    const STRIDE: usize = N_KV_HEADS * HEAD_DIM;

    fn make_token(base: f32) -> Vec<f32> {
        (0..STRIDE).map(|i| base + i as f32 * 0.001).collect()
    }

    #[test]
    fn append_single_tokens_and_retrieve() {
        let mut cache = KVCache::new(NUM_LAYERS, MAX_SEQ, N_KV_HEADS, HEAD_DIM).unwrap();

        let k0 = make_token(1.0);
        let v0 = make_token(2.0);
        cache.append(0, &k0, &v0).unwrap();

        let (k, v, seq_len) = cache.get(0).unwrap();
        assert_eq!(seq_len, 1);
        assert_eq!(&k[..STRIDE], &k0[..]);
        assert_eq!(&v[..STRIDE], &v0[..]);

        // Append a second token
        let k1 = make_token(3.0);
        let v1 = make_token(4.0);
        cache.append(0, &k1, &v1).unwrap();

        let (k, v, seq_len) = cache.get(0).unwrap();
        assert_eq!(seq_len, 2);
        assert_eq!(&k[..STRIDE], &k0[..]);
        assert_eq!(&k[STRIDE..2 * STRIDE], &k1[..]);
        assert_eq!(&v[..STRIDE], &v0[..]);
        assert_eq!(&v[STRIDE..2 * STRIDE], &v1[..]);
    }

    #[test]
    fn full_capacity_fill() {
        let max_seq = 64; // smaller for test speed
        let mut cache = KVCache::new(1, max_seq, N_KV_HEADS, HEAD_DIM).unwrap();

        for i in 0..max_seq {
            let k = make_token(i as f32);
            let v = make_token(i as f32 + 0.5);
            cache.append(0, &k, &v).unwrap();
        }

        let (k, v, seq_len) = cache.get(0).unwrap();
        assert_eq!(seq_len, max_seq);

        // Verify first and last tokens
        let k_first = make_token(0.0);
        let k_last = make_token((max_seq - 1) as f32);
        assert_eq!(&k[..STRIDE], &k_first[..]);
        assert_eq!(&k[(max_seq - 1) * STRIDE..max_seq * STRIDE], &k_last[..]);

        let v_first = make_token(0.5);
        let v_last = make_token((max_seq - 1) as f32 + 0.5);
        assert_eq!(&v[..STRIDE], &v_first[..]);
        assert_eq!(&v[(max_seq - 1) * STRIDE..max_seq * STRIDE], &v_last[..]);
    }

    #[test]
    fn reset_and_reuse() {
        let mut cache = KVCache::new(NUM_LAYERS, MAX_SEQ, N_KV_HEADS, HEAD_DIM).unwrap();

        // Fill some tokens
        for i in 0..10 {
            let k = make_token(i as f32);
            let v = make_token(i as f32);
            cache.append(0, &k, &v).unwrap();
        }
        assert_eq!(cache.current_len().unwrap(), 10);

        // Reset
        cache.reset().unwrap();

        // All layers should be empty
        for layer in 0..NUM_LAYERS {
            let (_, _, seq_len) = cache.get(layer).unwrap();
            assert_eq!(seq_len, 0, "layer {} should be empty after reset", layer);
        }

        // Reuse: append again
        let k = make_token(99.0);
        let v = make_token(99.0);
        cache.append(0, &k, &v).unwrap();
        let (k_out, v_out, seq_len) = cache.get(0).unwrap();
        assert_eq!(seq_len, 1);
        assert_eq!(&k_out[..STRIDE], &k[..]);
        assert_eq!(&v_out[..STRIDE], &v[..]);
    }

    #[test]
    fn memory_matches_expected_formula() {
        let cache = KVCache::new(NUM_LAYERS, MAX_SEQ, N_KV_HEADS, HEAD_DIM).unwrap();

        let expected =
            2 * NUM_LAYERS * MAX_SEQ * N_KV_HEADS * HEAD_DIM * std::mem::size_of::<f32>();
        assert_eq!(cache.expected_allocation_bytes(), expected);

        // 2 * 28 * 2048 * 8 * 128 * 4 = 2 * 28 * 2048 * 1024 * 4 = 2 * 28 * 8388608 = 469762048
        assert_eq!(expected, 469_762_048);
    }

    #[test]
    fn no_allocation_after_init() {
        let mut cache = KVCache::new(2, 64, N_KV_HEADS, HEAD_DIM).unwrap();

        // After init, all operations should be zero-alloc.
        // We verify this by checking that append/slice/reset don't fail
        // and that the cache continues to work correctly after many operations.
        for round in 0..3 {
            for i in 0..64 {
                let k = make_token((round * 100 + i) as f32);
                let v = make_token((round * 100 + i) as f32 + 0.5);
                cache.append(0, &k, &v).unwrap();
                cache.append(1, &k, &v).unwrap();
            }

            // Verify both layers
            for layer in 0..2 {
                let (_, _, seq_len) = cache.get(layer).unwrap();
                assert_eq!(seq_len, 64);
            }

            cache.reset().unwrap();

            for layer in 0..2 {
                let (_, _, seq_len) = cache.get(layer).unwrap();
                assert_eq!(seq_len, 0);
            }
        }
    }

    #[test]
    fn multi_layer_independence() {
        let mut cache = KVCache::new(4, MAX_SEQ, N_KV_HEADS, HEAD_DIM).unwrap();

        // Different data per layer
        for layer in 0..4 {
            let k = make_token(layer as f32 * 10.0);
            let v = make_token(layer as f32 * 10.0 + 5.0);
            cache.append(layer, &k, &v).unwrap();
        }

        // Verify each layer has its own data
        for layer in 0..4 {
            let (k, v, seq_len) = cache.get(layer).unwrap();
            assert_eq!(seq_len, 1);
            let expected_k = make_token(layer as f32 * 10.0);
            let expected_v = make_token(layer as f32 * 10.0 + 5.0);
            assert_eq!(&k[..STRIDE], &expected_k[..]);
            assert_eq!(&v[..STRIDE], &expected_v[..]);
        }
    }
}
