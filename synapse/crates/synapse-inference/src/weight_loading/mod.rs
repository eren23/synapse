pub mod converter;
#[cfg(not(target_os = "espidf"))]
pub mod gguf;
#[cfg(not(target_os = "espidf"))]
pub mod safetensors;
pub mod weight_map;

pub use converter::{bf16_to_f32, f16_to_f32, transpose};
#[cfg(not(target_os = "espidf"))]
pub use gguf::{load_gguf, parse_gguf};
#[cfg(not(target_os = "espidf"))]
pub use safetensors::{load_safetensors, load_safetensors_sharded, parse_safetensors};
pub use weight_map::WeightMapper;

use std::alloc::{alloc_zeroed, dealloc, Layout};
use std::fmt;
use std::ops::{Deref, DerefMut};

/// Buffer alignment in bytes — matches cache line size for SIMD.
const ALIGN: usize = 64;

/// A 64-byte-aligned f32 buffer for SIMD-friendly weight storage.
///
/// Avoids the double-allocation pattern in weight loading: data is copied
/// directly from mmap into an aligned buffer in a single pass.
pub struct AlignedBuffer {
    ptr: *mut f32,
    len: usize,
}

// SAFETY: AlignedBuffer fully owns its allocation; no shared mutable state.
unsafe impl Send for AlignedBuffer {}
unsafe impl Sync for AlignedBuffer {}

impl AlignedBuffer {
    /// Allocate a zeroed, 64-byte-aligned buffer for `len` f32 elements.
    pub fn new_zeroed(len: usize) -> Self {
        if len == 0 {
            return Self {
                ptr: ALIGN as *mut f32,
                len: 0,
            };
        }
        let layout = Self::layout_for(len);
        let ptr = unsafe { alloc_zeroed(layout) as *mut f32 };
        if ptr.is_null() {
            std::alloc::handle_alloc_error(layout);
        }
        Self { ptr, len }
    }

    /// Copy raw little-endian f32 bytes directly into an aligned buffer.
    ///
    /// This is the fast path for F32 safetensors — single allocation, single memcpy.
    /// Requires a little-endian target (x86, ARM).
    #[cfg(target_endian = "little")]
    pub fn from_f32_bytes(bytes: &[u8]) -> Self {
        assert!(bytes.len() % 4 == 0, "byte length must be multiple of 4");
        let count = bytes.len() / 4;
        if count == 0 {
            return Self::new_zeroed(0);
        }
        let mut buf = Self::new_zeroed(count);
        unsafe {
            std::ptr::copy_nonoverlapping(bytes.as_ptr(), buf.as_mut_ptr() as *mut u8, bytes.len());
        }
        buf
    }

    /// Create an aligned buffer by copying data from a slice.
    pub fn from_slice(data: &[f32]) -> Self {
        if data.is_empty() {
            return Self::new_zeroed(0);
        }
        let mut buf = Self::new_zeroed(data.len());
        unsafe {
            std::ptr::copy_nonoverlapping(data.as_ptr(), buf.as_mut_ptr(), data.len());
        }
        buf
    }

    /// Create an aligned buffer by copying data from a Vec.
    pub fn from_vec(data: Vec<f32>) -> Self {
        Self::from_slice(&data)
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Returns true if the buffer pointer is 64-byte aligned.
    pub fn is_aligned(&self) -> bool {
        self.len == 0 || self.ptr as usize % ALIGN == 0
    }

    pub fn as_ptr(&self) -> *const f32 {
        self.ptr
    }

    pub fn as_mut_ptr(&mut self) -> *mut f32 {
        self.ptr
    }

    fn layout_for(len: usize) -> Layout {
        Layout::from_size_align(len * std::mem::size_of::<f32>(), ALIGN).unwrap()
    }
}

impl Drop for AlignedBuffer {
    fn drop(&mut self) {
        if self.len > 0 {
            unsafe { dealloc(self.ptr as *mut u8, Self::layout_for(self.len)) }
        }
    }
}

impl Clone for AlignedBuffer {
    fn clone(&self) -> Self {
        Self::from_slice(self)
    }
}

impl Deref for AlignedBuffer {
    type Target = [f32];
    fn deref(&self) -> &[f32] {
        if self.len == 0 {
            &[]
        } else {
            unsafe { std::slice::from_raw_parts(self.ptr, self.len) }
        }
    }
}

impl DerefMut for AlignedBuffer {
    fn deref_mut(&mut self) -> &mut [f32] {
        if self.len == 0 {
            &mut []
        } else {
            unsafe { std::slice::from_raw_parts_mut(self.ptr, self.len) }
        }
    }
}

impl fmt::Debug for AlignedBuffer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Debug::fmt(&**self, f)
    }
}

impl PartialEq for AlignedBuffer {
    fn eq(&self, other: &Self) -> bool {
        **self == **other
    }
}

impl PartialEq<Vec<f32>> for AlignedBuffer {
    fn eq(&self, other: &Vec<f32>) -> bool {
        **self == **other
    }
}

impl PartialEq<AlignedBuffer> for Vec<f32> {
    fn eq(&self, other: &AlignedBuffer) -> bool {
        **self == **other
    }
}

impl FromIterator<f32> for AlignedBuffer {
    fn from_iter<I: IntoIterator<Item = f32>>(iter: I) -> Self {
        let v: Vec<f32> = iter.into_iter().collect();
        Self::from_slice(&v)
    }
}

/// Raw tensor data before wrapping in a synapse-core Tensor.
#[derive(Debug, Clone)]
pub struct RawTensor {
    pub data: AlignedBuffer,
    pub shape: Vec<usize>,
}

/// Errors from weight loading operations.
#[derive(Debug)]
pub enum WeightError {
    Io(std::io::Error),
    InvalidFormat(String),
    UnsupportedDtype(String),
    ShapeMismatch(String),
    MissingKeys(Vec<String>),
    UnexpectedKeys(Vec<String>),
    #[cfg(feature = "zig-ffi")]
    TensorError(synapse_core::SynapseError),
    #[cfg(not(feature = "zig-ffi"))]
    TensorError(String),
}

impl fmt::Display for WeightError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            WeightError::Io(e) => write!(f, "IO error: {e}"),
            WeightError::InvalidFormat(msg) => write!(f, "Invalid format: {msg}"),
            WeightError::UnsupportedDtype(dtype) => write!(f, "Unsupported dtype: {dtype}"),
            WeightError::ShapeMismatch(msg) => write!(f, "Shape mismatch: {msg}"),
            WeightError::MissingKeys(keys) => write!(f, "Missing keys: {keys:?}"),
            WeightError::UnexpectedKeys(keys) => write!(f, "Unexpected keys: {keys:?}"),
            WeightError::TensorError(e) => write!(f, "Tensor error: {e}"),
        }
    }
}

impl std::error::Error for WeightError {}

#[cfg(test)]
mod tests {
    use super::AlignedBuffer;

    #[test]
    fn aligned_buffer_is_64_byte_aligned() {
        for &len in &[1, 16, 256, 1024, 65536] {
            let buf = AlignedBuffer::new_zeroed(len);
            assert!(
                buf.is_aligned(),
                "Buffer of len {len} not 64-byte aligned: ptr={:?}",
                buf.as_ptr()
            );
        }
    }

    #[test]
    fn aligned_buffer_empty() {
        let buf = AlignedBuffer::new_zeroed(0);
        assert!(buf.is_empty());
        assert_eq!(buf.len(), 0);
        assert!(buf.is_aligned());
    }

    #[test]
    fn aligned_buffer_from_slice_roundtrip() {
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let buf = AlignedBuffer::from_slice(&data);
        assert!(buf.is_aligned());
        assert_eq!(buf, data);
    }

    #[cfg(target_endian = "little")]
    #[test]
    fn aligned_buffer_from_f32_bytes_bit_exact() {
        let original = vec![1.0f32, -2.5, 3.14, 0.0];
        let bytes: Vec<u8> = original.iter().flat_map(|f| f.to_le_bytes()).collect();
        let buf = AlignedBuffer::from_f32_bytes(&bytes);
        assert!(buf.is_aligned());
        assert_eq!(buf, original);
    }

    #[test]
    fn aligned_buffer_clone_preserves_alignment() {
        let buf = AlignedBuffer::from_slice(&[1.0, 2.0, 3.0]);
        let cloned = buf.clone();
        assert!(cloned.is_aligned());
        assert_eq!(buf, cloned);
    }

    #[test]
    fn aligned_buffer_from_iter() {
        let buf: AlignedBuffer = (0..10).map(|i| i as f32).collect();
        assert!(buf.is_aligned());
        assert_eq!(buf.len(), 10);
        assert_eq!(buf[0], 0.0);
        assert_eq!(buf[9], 9.0);
    }
}
