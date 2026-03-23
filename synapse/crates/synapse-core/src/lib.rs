//! Safe Rust wrappers around the Zig-backed Synapse FFI tensor library.
//!
//! Provides RAII-managed [`Tensor`] handles and [`Result`]-based error handling
//! over the raw C ABI in `synapse-sys`.

use std::fmt;
use std::ptr;

use synapse_sys as ffi;

// ------------------------------------------------------------------
// Error types
// ------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SynapseError {
    NullPointer,
    InvalidArg,
    OutOfMemory,
    ShapeMismatch,
    NotContiguous,
    InvalidAxis,
    InvalidDimensions,
    Internal,
    Unknown(i32),
}

impl fmt::Display for SynapseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SynapseError::NullPointer => write!(f, "null pointer"),
            SynapseError::InvalidArg => write!(f, "invalid argument"),
            SynapseError::OutOfMemory => write!(f, "out of memory"),
            SynapseError::ShapeMismatch => write!(f, "shape mismatch"),
            SynapseError::NotContiguous => write!(f, "tensor not contiguous"),
            SynapseError::InvalidAxis => write!(f, "invalid axis"),
            SynapseError::InvalidDimensions => write!(f, "invalid dimensions"),
            SynapseError::Internal => write!(f, "internal error"),
            SynapseError::Unknown(code) => write!(f, "unknown error (code {})", code),
        }
    }
}

impl std::error::Error for SynapseError {}

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

// ------------------------------------------------------------------
// Tensor (RAII wrapper over opaque FFI handle)
// ------------------------------------------------------------------

/// An owned, RAII-managed tensor backed by the Zig runtime.
///
/// Automatically destroys the underlying FFI handle on drop.
/// Not `Clone` — use explicit operations to produce new tensors.
pub struct Tensor {
    ptr: *mut ffi::syn_tensor_t,
}

// Tensor handles are heap-allocated in the Zig allocator; the pointer
// itself is safe to move across threads (no TLS dependency).
unsafe impl Send for Tensor {}

impl Drop for Tensor {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            unsafe { ffi::syn_tensor_destroy(self.ptr) };
        }
    }
}

impl fmt::Debug for Tensor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if let Ok(shape) = self.shape() {
            write!(f, "Tensor(shape={:?})", shape)
        } else {
            write!(f, "Tensor(<invalid>)")
        }
    }
}

impl Tensor {
    // ------------------------------------------------------------------
    // Construction
    // ------------------------------------------------------------------

    /// Create a tensor from flat f32 data and a shape.
    pub fn from_data(data: &[f32], shape: &[usize]) -> Result<Self, SynapseError> {
        let numel: usize = shape.iter().product();
        if data.len() != numel {
            return Err(SynapseError::ShapeMismatch);
        }

        unsafe {
            let mut storage: *mut ffi::syn_storage_t = ptr::null_mut();
            check_status(ffi::syn_storage_create(numel, &mut storage))?;

            // Copy data into storage.
            let mut sdata: *mut f32 = ptr::null_mut();
            let status = ffi::syn_storage_data(storage, &mut sdata);
            if status != ffi::SYN_OK {
                ffi::syn_storage_release(storage);
                return Err(check_status(status).unwrap_err());
            }
            ptr::copy_nonoverlapping(data.as_ptr(), sdata, numel);

            // Create tensor over storage.
            let mut tensor: *mut ffi::syn_tensor_t = ptr::null_mut();
            let status = ffi::syn_tensor_create(storage, shape.as_ptr(), shape.len(), &mut tensor);
            // Release our storage ref (tensor holds its own).
            ffi::syn_storage_release(storage);
            check_status(status)?;

            Ok(Tensor { ptr: tensor })
        }
    }

    /// Create a zero-filled tensor with the given shape.
    pub fn zeros(shape: &[usize]) -> Result<Self, SynapseError> {
        let numel: usize = shape.iter().product();
        let data = vec![0.0f32; numel];
        Self::from_data(&data, shape)
    }

    /// Wrap a raw FFI tensor pointer. Takes ownership.
    ///
    /// # Safety
    /// `ptr` must be a valid, owned FFI tensor handle.
    pub unsafe fn from_raw(ptr: *mut ffi::syn_tensor_t) -> Self {
        Tensor { ptr }
    }

    /// Return the raw FFI pointer without dropping.
    pub fn into_raw(self) -> *mut ffi::syn_tensor_t {
        let ptr = self.ptr;
        std::mem::forget(self);
        ptr
    }

    pub fn as_ptr(&self) -> *mut ffi::syn_tensor_t {
        self.ptr
    }

    // ------------------------------------------------------------------
    // Accessors
    // ------------------------------------------------------------------

    pub fn shape(&self) -> Result<Vec<usize>, SynapseError> {
        let mut dims = [0usize; 8];
        let mut ndim: usize = 0;
        unsafe {
            check_status(ffi::syn_tensor_shape(self.ptr, dims.as_mut_ptr(), &mut ndim))?;
        }
        Ok(dims[..ndim].to_vec())
    }

    pub fn ndim(&self) -> Result<usize, SynapseError> {
        let mut ndim: usize = 0;
        unsafe { check_status(ffi::syn_tensor_ndim(self.ptr, &mut ndim))? };
        Ok(ndim)
    }

    pub fn numel(&self) -> Result<usize, SynapseError> {
        Ok(self.shape()?.iter().product())
    }

    /// Read tensor data into a Vec.
    pub fn to_vec(&self) -> Result<Vec<f32>, SynapseError> {
        let n = self.numel()?;
        unsafe {
            let mut dptr: *mut f32 = ptr::null_mut();
            check_status(ffi::syn_tensor_data_ptr(self.ptr, &mut dptr))?;
            Ok(std::slice::from_raw_parts(dptr, n).to_vec())
        }
    }

    // ------------------------------------------------------------------
    // Transformer ops — safe wrappers
    // ------------------------------------------------------------------

    /// Layer normalization over trailing dimensions.
    ///
    /// `gamma` and `beta` are 1-D affine parameters sized to the product of
    /// the last `normalized_dim` dimensions of `self`.
    pub fn layernorm(
        &self,
        gamma: &Tensor,
        beta: &Tensor,
        normalized_dim: usize,
        eps: f32,
    ) -> Result<Tensor, SynapseError> {
        let mut out: *mut ffi::syn_tensor_t = ptr::null_mut();
        unsafe {
            check_status(ffi::syn_layernorm_forward(
                &mut out,
                self.ptr,
                gamma.ptr,
                beta.ptr,
                normalized_dim,
                eps,
            ))?;
            Ok(Tensor::from_raw(out))
        }
    }

    /// Scaled dot-product attention.
    ///
    /// `self` is the query tensor `[batch, heads, seq_q, d_head]`.
    /// Returns `(output, Option<attn_weights>)`.
    pub fn scaled_dot_product_attention(
        &self,
        key: &Tensor,
        value: &Tensor,
        scale: f32,
        causal: bool,
    ) -> Result<(Tensor, Option<Tensor>), SynapseError> {
        let mut out: *mut ffi::syn_tensor_t = ptr::null_mut();
        let mut weights: *mut ffi::syn_tensor_t = ptr::null_mut();
        unsafe {
            check_status(ffi::syn_scaled_dot_product_attention(
                &mut out,
                &mut weights,
                self.ptr,
                key.ptr,
                value.ptr,
                scale,
                causal as i32,
            ))?;
            let w = if weights.is_null() {
                None
            } else {
                Some(Tensor::from_raw(weights))
            };
            Ok((Tensor::from_raw(out), w))
        }
    }

    /// Rotary positional embedding (RoPE).
    ///
    /// Input must be 4-D `[batch, heads, seq, d_head]` with even `d_head`.
    /// `cos_table` and `sin_table` are precomputed `[max_seq, d_head/2]`.
    pub fn rope(
        &self,
        cos_table: &Tensor,
        sin_table: &Tensor,
        offset: usize,
    ) -> Result<Tensor, SynapseError> {
        let mut out: *mut ffi::syn_tensor_t = ptr::null_mut();
        unsafe {
            check_status(ffi::syn_rope_forward(
                &mut out,
                self.ptr,
                cos_table.ptr,
                sin_table.ptr,
                offset,
            ))?;
            Ok(Tensor::from_raw(out))
        }
    }

    /// Generate an additive causal mask `[seq_len, seq_len]`.
    ///
    /// `mask[i][j] = 0` if `j <= i`, `-inf` otherwise.
    pub fn causal_mask(seq_len: usize) -> Result<Tensor, SynapseError> {
        let mut out: *mut ffi::syn_tensor_t = ptr::null_mut();
        unsafe {
            check_status(ffi::syn_causal_mask(&mut out, seq_len))?;
            Ok(Tensor::from_raw(out))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ================================================================
    // Helpers
    // ================================================================

    fn approx_eq(a: &[f32], b: &[f32], tol: f32) -> bool {
        a.len() == b.len() && a.iter().zip(b).all(|(x, y)| (x - y).abs() < tol)
    }

    /// Build precomputed cos/sin tables for RoPE tests.
    /// Shape: [max_seq, half_d]
    fn build_rope_tables(max_seq: usize, half_d: usize) -> (Vec<f32>, Vec<f32>) {
        let mut cos_data = vec![0.0f32; max_seq * half_d];
        let mut sin_data = vec![0.0f32; max_seq * half_d];
        for pos in 0..max_seq {
            for i in 0..half_d {
                let freq = 1.0 / (10000.0f32).powf(2.0 * i as f32 / (2 * half_d) as f32);
                let angle = pos as f32 * freq;
                cos_data[pos * half_d + i] = angle.cos();
                sin_data[pos * half_d + i] = angle.sin();
            }
        }
        (cos_data, sin_data)
    }

    // ================================================================
    // FFI roundtrip: layernorm
    // ================================================================

    #[test]
    fn test_layernorm_roundtrip() {
        // input: [2, 4], normalize over last dim (normalized_dim=1)
        let input = Tensor::from_data(
            &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            &[2, 4],
        )
        .unwrap();

        let gamma = Tensor::from_data(&[1.0, 1.0, 1.0, 1.0], &[4]).unwrap();
        let beta = Tensor::from_data(&[0.0, 0.0, 0.0, 0.0], &[4]).unwrap();

        let result = input.layernorm(&gamma, &beta, 1, 1e-5).unwrap();
        let shape = result.shape().unwrap();
        assert_eq!(shape, &[2, 4]);

        let data = result.to_vec().unwrap();
        // Each row should be zero-mean with unit variance (approximately).
        let row0 = &data[0..4];
        let mean: f32 = row0.iter().sum::<f32>() / 4.0;
        assert!(mean.abs() < 1e-5, "row mean should be ~0, got {}", mean);
    }

    // ================================================================
    // FFI roundtrip: scaled dot-product attention
    // ================================================================

    #[test]
    fn test_attention_roundtrip() {
        // [batch=1, heads=1, seq=2, d_head=4]
        let q = Tensor::from_data(
            &[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
            &[1, 1, 2, 4],
        )
        .unwrap();
        let k = q.clone_tensor().unwrap();
        let v = Tensor::from_data(
            &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            &[1, 1, 2, 4],
        )
        .unwrap();

        let (output, weights) = q
            .scaled_dot_product_attention(&k, &v, 0.5, false)
            .unwrap();

        assert_eq!(output.shape().unwrap(), &[1, 1, 2, 4]);
        assert!(weights.is_some());
        let w = weights.unwrap();
        assert_eq!(w.shape().unwrap(), &[1, 1, 2, 2]);
    }

    // ================================================================
    // FFI roundtrip: RoPE
    // ================================================================

    #[test]
    fn test_rope_roundtrip() {
        let batch = 1;
        let heads = 1;
        let seq = 2;
        let d_head = 4;
        let half_d = d_head / 2;
        let max_seq = 8;

        let input_data: Vec<f32> = (0..batch * heads * seq * d_head)
            .map(|i| i as f32 + 1.0)
            .collect();
        let input = Tensor::from_data(&input_data, &[batch, heads, seq, d_head]).unwrap();

        let (cos_data, sin_data) = build_rope_tables(max_seq, half_d);
        let cos_table = Tensor::from_data(&cos_data, &[max_seq, half_d]).unwrap();
        let sin_table = Tensor::from_data(&sin_data, &[max_seq, half_d]).unwrap();

        let result = input.rope(&cos_table, &sin_table, 0).unwrap();
        assert_eq!(result.shape().unwrap(), &[batch, heads, seq, d_head]);

        // At position 0 all angles are 0, so cos=1, sin=0 → output == input for first token.
        let result_data = result.to_vec().unwrap();
        assert!(
            approx_eq(&result_data[0..d_head], &input_data[0..d_head], 1e-4),
            "position-0 token should be unchanged by RoPE"
        );
    }

    // ================================================================
    // FFI roundtrip: causal mask
    // ================================================================

    #[test]
    fn test_causal_mask_roundtrip() {
        let mask = Tensor::causal_mask(3).unwrap();
        assert_eq!(mask.shape().unwrap(), &[3, 3]);

        let data = mask.to_vec().unwrap();
        // Row 0: [0, -inf, -inf]
        assert_eq!(data[0], 0.0);
        assert!(data[1].is_infinite() && data[1] < 0.0);
        assert!(data[2].is_infinite() && data[2] < 0.0);
        // Row 1: [0, 0, -inf]
        assert_eq!(data[3], 0.0);
        assert_eq!(data[4], 0.0);
        assert!(data[5].is_infinite() && data[5] < 0.0);
        // Row 2: [0, 0, 0]
        assert_eq!(data[6], 0.0);
        assert_eq!(data[7], 0.0);
        assert_eq!(data[8], 0.0);
    }

    // ================================================================
    // Error propagation: shape mismatches
    // ================================================================

    #[test]
    fn test_layernorm_shape_mismatch() {
        let input = Tensor::from_data(&[1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
        // gamma has wrong size (3 instead of 2)
        let gamma = Tensor::from_data(&[1.0, 1.0, 1.0], &[3]).unwrap();
        let beta = Tensor::from_data(&[0.0, 0.0], &[2]).unwrap();

        let err = input.layernorm(&gamma, &beta, 1, 1e-5);
        assert!(err.is_err());
        assert_eq!(err.unwrap_err(), SynapseError::ShapeMismatch);
    }

    #[test]
    fn test_attention_dimension_error() {
        // 3D tensor instead of required 4D
        let q = Tensor::from_data(&[1.0, 2.0, 3.0, 4.0], &[1, 2, 2]).unwrap();
        let k = Tensor::from_data(&[1.0, 2.0, 3.0, 4.0], &[1, 2, 2]).unwrap();
        let v = Tensor::from_data(&[1.0, 2.0, 3.0, 4.0], &[1, 2, 2]).unwrap();

        let err = q.scaled_dot_product_attention(&k, &v, 1.0, false);
        assert!(err.is_err());
    }

    #[test]
    fn test_rope_dimension_error() {
        // 2D tensor instead of required 4D
        let input = Tensor::from_data(&[1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
        let cos_table = Tensor::from_data(&[1.0], &[1]).unwrap();
        let sin_table = Tensor::from_data(&[0.0], &[1]).unwrap();

        let err = input.rope(&cos_table, &sin_table, 0);
        assert!(err.is_err());
        assert_eq!(err.unwrap_err(), SynapseError::InvalidDimensions);
    }

    #[test]
    fn test_causal_mask_zero_seq_len() {
        let err = Tensor::causal_mask(0);
        assert!(err.is_err());
        assert_eq!(err.unwrap_err(), SynapseError::InvalidArg);
    }

    // ================================================================
    // Memory safety: no leaks in create/call/destroy cycles (10K iters)
    // ================================================================

    #[test]
    fn test_layernorm_memory_safety_10k() {
        for _ in 0..10_000 {
            let input = Tensor::from_data(&[1.0, 2.0, 3.0, 4.0], &[1, 4]).unwrap();
            let gamma = Tensor::from_data(&[1.0, 1.0, 1.0, 1.0], &[4]).unwrap();
            let beta = Tensor::from_data(&[0.0, 0.0, 0.0, 0.0], &[4]).unwrap();
            let _result = input.layernorm(&gamma, &beta, 1, 1e-5).unwrap();
            // All tensors dropped here via RAII.
        }
    }

    #[test]
    fn test_attention_memory_safety_10k() {
        for _ in 0..10_000 {
            let q = Tensor::from_data(&[1.0, 0.0, 0.0, 1.0], &[1, 1, 1, 4]).unwrap();
            let k = Tensor::from_data(&[1.0, 0.0, 0.0, 1.0], &[1, 1, 1, 4]).unwrap();
            let v = Tensor::from_data(&[1.0, 2.0, 3.0, 4.0], &[1, 1, 1, 4]).unwrap();
            let (_out, _w) = q
                .scaled_dot_product_attention(&k, &v, 0.5, false)
                .unwrap();
        }
    }

    #[test]
    fn test_rope_memory_safety_10k() {
        let half_d = 2;
        let max_seq = 4;
        let (cos_data, sin_data) = build_rope_tables(max_seq, half_d);

        for _ in 0..10_000 {
            let input = Tensor::from_data(&[1.0, 2.0, 3.0, 4.0], &[1, 1, 1, 4]).unwrap();
            let cos_table = Tensor::from_data(&cos_data, &[max_seq, half_d]).unwrap();
            let sin_table = Tensor::from_data(&sin_data, &[max_seq, half_d]).unwrap();
            let _result = input.rope(&cos_table, &sin_table, 0).unwrap();
        }
    }

    #[test]
    fn test_causal_mask_memory_safety_10k() {
        for _ in 0..10_000 {
            let _mask = Tensor::causal_mask(4).unwrap();
        }
    }
}

// ------------------------------------------------------------------
// Helper: clone a tensor by reading its data and recreating it
// ------------------------------------------------------------------
impl Tensor {
    /// Create a new tensor with the same data and shape (deep copy).
    pub fn clone_tensor(&self) -> Result<Tensor, SynapseError> {
        let data = self.to_vec()?;
        let shape = self.shape()?;
        Tensor::from_data(&data, &shape)
    }
}
