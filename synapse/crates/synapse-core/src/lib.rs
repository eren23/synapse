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
    InvalidCapabilityValue(&'static str, u32),
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
            SynapseError::InvalidCapabilityValue(field, value) => {
                write!(f, "invalid capability value for {field}: {value}")
            }
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TargetArch {
    Unknown,
    Aarch64,
    X86_64,
    Wasm32,
}

impl TargetArch {
    fn from_ffi(value: u32) -> Result<Self, SynapseError> {
        match value {
            ffi::SYN_ARCH_UNKNOWN => Ok(Self::Unknown),
            ffi::SYN_ARCH_AARCH64 => Ok(Self::Aarch64),
            ffi::SYN_ARCH_X86_64 => Ok(Self::X86_64),
            ffi::SYN_ARCH_WASM32 => Ok(Self::Wasm32),
            _ => Err(SynapseError::InvalidCapabilityValue("target_arch", value)),
        }
    }

    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Unknown => "unknown",
            Self::Aarch64 => "aarch64",
            Self::X86_64 => "x86_64",
            Self::Wasm32 => "wasm32",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TargetOs {
    Unknown,
    Macos,
    Linux,
    Windows,
    Wasm,
}

impl TargetOs {
    fn from_ffi(value: u32) -> Result<Self, SynapseError> {
        match value {
            ffi::SYN_OS_UNKNOWN => Ok(Self::Unknown),
            ffi::SYN_OS_MACOS => Ok(Self::Macos),
            ffi::SYN_OS_LINUX => Ok(Self::Linux),
            ffi::SYN_OS_WINDOWS => Ok(Self::Windows),
            ffi::SYN_OS_WASM => Ok(Self::Wasm),
            _ => Err(SynapseError::InvalidCapabilityValue("target_os", value)),
        }
    }

    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Unknown => "unknown",
            Self::Macos => "macos",
            Self::Linux => "linux",
            Self::Windows => "windows",
            Self::Wasm => "wasm",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SimdBackend {
    Scalar,
    Neon,
    Avx2,
}

impl SimdBackend {
    fn from_ffi(value: u32) -> Result<Self, SynapseError> {
        match value {
            ffi::SYN_BACKEND_SCALAR => Ok(Self::Scalar),
            ffi::SYN_BACKEND_NEON => Ok(Self::Neon),
            ffi::SYN_BACKEND_AVX2 => Ok(Self::Avx2),
            _ => Err(SynapseError::InvalidCapabilityValue("simd_backend", value)),
        }
    }

    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Scalar => "scalar",
            Self::Neon => "neon",
            Self::Avx2 => "avx2",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CapabilityRuntimeProfile {
    NativePerf,
    ArmCompact,
    WasmPortable,
}

impl CapabilityRuntimeProfile {
    fn from_ffi(value: u32) -> Result<Self, SynapseError> {
        match value {
            ffi::SYN_RUNTIME_NATIVE_PERF => Ok(Self::NativePerf),
            ffi::SYN_RUNTIME_ARM_COMPACT => Ok(Self::ArmCompact),
            ffi::SYN_RUNTIME_WASM_PORTABLE => Ok(Self::WasmPortable),
            _ => Err(SynapseError::InvalidCapabilityValue(
                "runtime_profile",
                value,
            )),
        }
    }

    pub fn as_str(&self) -> &'static str {
        match self {
            Self::NativePerf => "native_perf",
            Self::ArmCompact => "arm_compact",
            Self::WasmPortable => "wasm_portable",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CapabilitySupportLevel {
    Stable,
    Beta,
    Experimental,
}

impl CapabilitySupportLevel {
    fn from_ffi(value: u32) -> Result<Self, SynapseError> {
        match value {
            ffi::SYN_SUPPORT_STABLE => Ok(Self::Stable),
            ffi::SYN_SUPPORT_BETA => Ok(Self::Beta),
            ffi::SYN_SUPPORT_EXPERIMENTAL => Ok(Self::Experimental),
            _ => Err(SynapseError::InvalidCapabilityValue("support_level", value)),
        }
    }

    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Stable => "stable",
            Self::Beta => "beta",
            Self::Experimental => "experimental",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CapabilitySummary {
    pub abi_version: u32,
    pub target_arch: TargetArch,
    pub target_os: TargetOs,
    pub simd_backend: SimdBackend,
    pub runtime_profile: CapabilityRuntimeProfile,
    pub support_level: CapabilitySupportLevel,
    pub feature_bits: u64,
}

impl CapabilitySummary {
    pub fn has_feature(&self, feature_bit: u64) -> bool {
        self.feature_bits & feature_bit != 0
    }

    pub fn feature_names(&self) -> Vec<&'static str> {
        let mut names = Vec::new();
        if self.has_feature(ffi::SYN_FEATURE_SGEMM) {
            names.push("sgemm");
        }
        if self.has_feature(ffi::SYN_FEATURE_LAYERNORM) {
            names.push("layernorm");
        }
        if self.has_feature(ffi::SYN_FEATURE_RMSNORM) {
            names.push("rmsnorm");
        }
        if self.has_feature(ffi::SYN_FEATURE_FUSED_ATTENTION) {
            names.push("fused_attention");
        }
        if self.has_feature(ffi::SYN_FEATURE_INT8_QUANT) {
            names.push("int8_quant");
        }
        if self.has_feature(ffi::SYN_FEATURE_Q4_0_GEMV) {
            names.push("q4_0_gemv");
        }
        if self.has_feature(ffi::SYN_FEATURE_KV_CACHE) {
            names.push("kvcache");
        }
        if self.has_feature(ffi::SYN_FEATURE_GEOMETRIC_ATTENTION) {
            names.push("geometric_attention");
        }
        names
    }
}

pub fn capability_summary() -> Result<CapabilitySummary, SynapseError> {
    let mut raw = ffi::syn_capability_summary_t {
        abi_version: 0,
        target_arch: 0,
        target_os: 0,
        simd_backend: 0,
        runtime_profile: 0,
        support_level: 0,
        feature_bits: 0,
    };
    unsafe {
        check_status(ffi::syn_capability_summary(&mut raw))?;
    }
    Ok(CapabilitySummary {
        abi_version: raw.abi_version,
        target_arch: TargetArch::from_ffi(raw.target_arch)?,
        target_os: TargetOs::from_ffi(raw.target_os)?,
        simd_backend: SimdBackend::from_ffi(raw.simd_backend)?,
        runtime_profile: CapabilityRuntimeProfile::from_ffi(raw.runtime_profile)?,
        support_level: CapabilitySupportLevel::from_ffi(raw.support_level)?,
        feature_bits: raw.feature_bits,
    })
}

pub fn runtime_capabilities_json() -> Result<String, SynapseError> {
    struct CapabilityJsonGuard {
        ptr: *mut u8,
        len: usize,
    }

    impl Drop for CapabilityJsonGuard {
        fn drop(&mut self) {
            if !self.ptr.is_null() {
                unsafe {
                    let _ = ffi::syn_runtime_capabilities_free(self.ptr, self.len);
                }
            }
        }
    }

    let mut ptr_out: *mut u8 = ptr::null_mut();
    let mut len_out: usize = 0;
    unsafe {
        check_status(ffi::syn_runtime_capabilities_json(
            &mut ptr_out,
            &mut len_out,
        ))?;
        if ptr_out.is_null() {
            return if len_out == 0 {
                Ok(String::new())
            } else {
                Err(SynapseError::NullPointer)
            };
        }

        let json = CapabilityJsonGuard {
            ptr: ptr_out,
            len: len_out,
        };
        let bytes = std::slice::from_raw_parts(json.ptr, json.len);
        Ok(std::str::from_utf8(bytes)
            .map_err(|_| SynapseError::InvalidArg)?
            .to_owned())
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
            check_status(ffi::syn_tensor_shape(
                self.ptr,
                dims.as_mut_ptr(),
                &mut ndim,
            ))?;
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

    /// RMS normalization over trailing dimensions.
    ///
    /// `gamma` is a 1-D affine parameter sized to the product of
    /// the last `num_norm_dims` dimensions of `self`.
    pub fn rmsnorm(
        &self,
        gamma: &Tensor,
        num_norm_dims: usize,
        eps: f32,
    ) -> Result<Tensor, SynapseError> {
        let mut out: *mut ffi::syn_tensor_t = ptr::null_mut();
        unsafe {
            check_status(ffi::syn_rmsnorm_forward(
                &mut out,
                self.ptr,
                gamma.ptr,
                num_norm_dims,
                eps,
            ))?;
            Ok(Tensor::from_raw(out))
        }
    }
}

// ------------------------------------------------------------------
// Flat-array activation helpers
// ------------------------------------------------------------------

/// In-place SiLU activation: `dst[i] = src[i] / (1 + exp(-src[i]))`.
pub fn silu(dst: &mut [f32], src: &[f32]) -> Result<(), SynapseError> {
    assert_eq!(dst.len(), src.len());
    unsafe { check_status(ffi::syn_silu(dst.as_mut_ptr(), src.as_ptr(), src.len())) }
}

/// Fused SwiGLU: `dst[i] = silu(gate[i]) * up[i]`.
pub fn swiglu(dst: &mut [f32], gate: &[f32], up: &[f32]) -> Result<(), SynapseError> {
    assert_eq!(dst.len(), gate.len());
    assert_eq!(gate.len(), up.len());
    unsafe {
        check_status(ffi::syn_swiglu(
            dst.as_mut_ptr(),
            gate.as_ptr(),
            up.as_ptr(),
            gate.len(),
        ))
    }
}

// ------------------------------------------------------------------
// INT8 quantization helpers
// ------------------------------------------------------------------

/// Quantize a `[channels, channel_size]` f32 matrix to per-channel INT8.
///
/// Returns `(quantized_data, scales)`.
pub fn quantize_per_channel_int8(
    data: &[f32],
    channels: usize,
    channel_size: usize,
) -> Result<(Vec<i8>, Vec<f32>), SynapseError> {
    if channels == 0 || channel_size == 0 {
        return Err(SynapseError::InvalidArg);
    }
    if data.len() != channels * channel_size {
        return Err(SynapseError::ShapeMismatch);
    }
    let mut out = vec![0i8; channels * channel_size];
    let mut scales = vec![0.0f32; channels];
    unsafe {
        check_status(ffi::syn_quantize_per_channel_int8(
            data.as_ptr(),
            channels,
            channel_size,
            out.as_mut_ptr(),
            scales.as_mut_ptr(),
        ))?;
    }
    Ok((out, scales))
}

/// Dequantize a `[channels, channel_size]` INT8 matrix back to f32.
pub fn dequantize_per_channel_int8(
    data: &[i8],
    channels: usize,
    channel_size: usize,
    scales: &[f32],
) -> Result<Vec<f32>, SynapseError> {
    if channels == 0 || channel_size == 0 {
        return Err(SynapseError::InvalidArg);
    }
    if data.len() != channels * channel_size || scales.len() != channels {
        return Err(SynapseError::ShapeMismatch);
    }
    let mut out = vec![0.0f32; channels * channel_size];
    unsafe {
        check_status(ffi::syn_dequantize_per_channel_int8(
            data.as_ptr(),
            channels,
            channel_size,
            out.as_mut_ptr(),
            scales.as_ptr(),
        ))?;
    }
    Ok(out)
}

/// INT8 quantized GEMM: `C[m,n] = diag(scales_a) * (A_i8 * B_i8) * diag(scales_b)`.
pub fn qgemm_int8(
    m: usize,
    n: usize,
    k: usize,
    a: &[i8],
    b: &[i8],
    scales_a: &[f32],
    scales_b: &[f32],
) -> Result<Vec<f32>, SynapseError> {
    if m == 0 || n == 0 || k == 0 {
        return Ok(vec![0.0f32; m * n]);
    }
    if a.len() < m * k || b.len() < k * n {
        return Err(SynapseError::ShapeMismatch);
    }
    if scales_a.len() < m || scales_b.len() < n {
        return Err(SynapseError::ShapeMismatch);
    }
    let mut c = vec![0.0f32; m * n];
    unsafe {
        check_status(ffi::syn_qgemm_int8(
            m,
            n,
            k,
            a.as_ptr(),
            k,
            b.as_ptr(),
            n,
            c.as_mut_ptr(),
            n,
            scales_a.as_ptr(),
            scales_b.as_ptr(),
        ))?;
    }
    Ok(c)
}

/// Fused causal attention for a single head.
///
/// Q: `[seq_q, d_head]`, K: `[seq_k, d_head]`, V: `[seq_k, d_head]`.
/// Returns output `[seq_q, d_head]` with causal masking and online softmax.
pub fn fused_attention(
    seq_q: usize,
    seq_k: usize,
    d_head: usize,
    q: &[f32],
    k: &[f32],
    v: &[f32],
) -> Result<Vec<f32>, SynapseError> {
    if seq_q == 0 || seq_k == 0 || d_head == 0 {
        return Ok(vec![0.0f32; seq_q * d_head]);
    }
    let mut out = vec![0.0f32; seq_q * d_head];
    unsafe {
        check_status(ffi::syn_fused_attention(
            seq_q,
            seq_k,
            d_head,
            q.as_ptr(),
            k.as_ptr(),
            v.as_ptr(),
            out.as_mut_ptr(),
        ))?;
    }
    Ok(out)
}

/// Bidirectional fused attention (no causal mask).
/// Q: [seq_q, d_head], K: [seq_k, d_head], V: [seq_k, d_head].
/// Returns [seq_q, d_head]. Uses tiled SIMD with online softmax.
pub fn fused_attention_bidi(
    seq_q: usize,
    seq_k: usize,
    d_head: usize,
    q: &[f32],
    k: &[f32],
    v: &[f32],
) -> Result<Vec<f32>, SynapseError> {
    if seq_q == 0 || seq_k == 0 || d_head == 0 {
        return Ok(vec![0.0f32; seq_q * d_head]);
    }
    let mut out = vec![0.0f32; seq_q * d_head];
    unsafe {
        check_status(ffi::syn_fused_attention_bidi(
            seq_q,
            seq_k,
            d_head,
            q.as_ptr(),
            k.as_ptr(),
            v.as_ptr(),
            out.as_mut_ptr(),
        ))?;
    }
    Ok(out)
}

/// Q4_0 matrix-vector multiply: C[1,N] = A_f32[1,K] @ dequant(B_q4[N,K]).
///
/// `a` is `[K]` f32 input, `b_q4` is raw Q4_0 block data for `[N, K]` matrix,
/// returns `[N]` f32 output. K must be a multiple of 32.
pub fn q4_0_gemv(n: usize, k: usize, a: &[f32], b_q4: &[u8]) -> Result<Vec<f32>, SynapseError> {
    if n == 0 || k == 0 {
        return Ok(vec![0.0f32; n]);
    }
    if k % 32 != 0 {
        return Err(SynapseError::ShapeMismatch);
    }
    if a.len() < k {
        return Err(SynapseError::ShapeMismatch);
    }
    let mut c = vec![0.0f32; n];
    unsafe {
        check_status(ffi::syn_q4_0_gemv(
            n,
            k,
            a.as_ptr(),
            b_q4.as_ptr(),
            c.as_mut_ptr(),
        ))?;
    }
    Ok(c)
}

/// Fused LEWM predictor layer: runs one full adaLN transformer layer in Zig SIMD.
///
/// Modifies `seq` in-place. All scratch buffers must be pre-allocated.
/// This is a single FFI call that replaces 12+ individual matmul/norm/attention calls.
#[allow(clippy::too_many_arguments)]
pub fn lewm_predict_layer(
    seq: &mut [f32],
    conditioning: &[f32],
    seq_len: usize,
    hidden: usize,
    num_heads: usize,
    inner_dim: usize,
    inter: usize,
    adaln_weight: &[f32],
    adaln_bias: &[f32],
    attn_norm_weight: &[f32],
    to_qkv: &[f32],
    attn_out_weight: &[f32],
    attn_out_bias: &[f32],
    mlp_norm_weight: &[f32],
    mlp_up_weight: &[f32],
    mlp_up_bias: &[f32],
    mlp_down_weight: &[f32],
    mlp_down_bias: &[f32],
    mod_buf: &mut [f32],
    normed_buf: &mut [f32],
    qkv_buf: &mut [f32],
    attn_buf: &mut [f32],
    proj_buf: &mut [f32],
) -> Result<(), SynapseError> {
    unsafe {
        check_status(ffi::syn_lewm_predict_layer(
            seq.as_mut_ptr(),
            conditioning.as_ptr(),
            seq_len,
            hidden,
            num_heads,
            inner_dim,
            inter,
            adaln_weight.as_ptr(),
            if adaln_bias.is_empty() { std::ptr::null() } else { adaln_bias.as_ptr() },
            attn_norm_weight.as_ptr(),
            to_qkv.as_ptr(),
            attn_out_weight.as_ptr(),
            if attn_out_bias.is_empty() { std::ptr::null() } else { attn_out_bias.as_ptr() },
            mlp_norm_weight.as_ptr(),
            mlp_up_weight.as_ptr(),
            if mlp_up_bias.is_empty() { std::ptr::null() } else { mlp_up_bias.as_ptr() },
            mlp_down_weight.as_ptr(),
            if mlp_down_bias.is_empty() { std::ptr::null() } else { mlp_down_bias.as_ptr() },
            mod_buf.as_mut_ptr(),
            normed_buf.as_mut_ptr(),
            qkv_buf.as_mut_ptr(),
            attn_buf.as_mut_ptr(),
            proj_buf.as_mut_ptr(),
        ))
    }
}

/// V2 fused LEWM predictor layer with mode selection.
/// mode=0: standard (separate loops), mode=1: ESP-fused (single-pass bias+GELU/residual loops).
#[allow(clippy::too_many_arguments)]
pub fn lewm_predict_layer_v2(
    seq: &mut [f32],
    conditioning: &[f32],
    seq_len: usize,
    hidden: usize,
    num_heads: usize,
    inner_dim: usize,
    inter: usize,
    adaln_weight: &[f32],
    adaln_bias: &[f32],
    attn_norm_weight: &[f32],
    to_qkv: &[f32],
    attn_out_weight: &[f32],
    attn_out_bias: &[f32],
    mlp_norm_weight: &[f32],
    mlp_up_weight: &[f32],
    mlp_up_bias: &[f32],
    mlp_down_weight: &[f32],
    mlp_down_bias: &[f32],
    mod_buf: &mut [f32],
    normed_buf: &mut [f32],
    qkv_buf: &mut [f32],
    attn_buf: &mut [f32],
    proj_buf: &mut [f32],
    mode: u8,
) -> Result<(), SynapseError> {
    unsafe {
        check_status(ffi::syn_lewm_predict_layer_v2(
            seq.as_mut_ptr(),
            conditioning.as_ptr(),
            seq_len,
            hidden,
            num_heads,
            inner_dim,
            inter,
            adaln_weight.as_ptr(),
            if adaln_bias.is_empty() { std::ptr::null() } else { adaln_bias.as_ptr() },
            attn_norm_weight.as_ptr(),
            to_qkv.as_ptr(),
            attn_out_weight.as_ptr(),
            if attn_out_bias.is_empty() { std::ptr::null() } else { attn_out_bias.as_ptr() },
            mlp_norm_weight.as_ptr(),
            mlp_up_weight.as_ptr(),
            if mlp_up_bias.is_empty() { std::ptr::null() } else { mlp_up_bias.as_ptr() },
            mlp_down_weight.as_ptr(),
            if mlp_down_bias.is_empty() { std::ptr::null() } else { mlp_down_bias.as_ptr() },
            mod_buf.as_mut_ptr(),
            normed_buf.as_mut_ptr(),
            qkv_buf.as_mut_ptr(),
            attn_buf.as_mut_ptr(),
            proj_buf.as_mut_ptr(),
            mode,
        ))
    }
}

/// Fused LEWM rollout: process all N rollout steps through all layers in a single call.
///
/// Per-layer weight pointer arrays (`adaln_ws`, etc.) are slices of `*const f32`, one entry
/// per layer. Scratch buffers must be pre-sized for `fused_seq_len = num_steps * 3`.
/// `mode` is a u32 bitfield (FUSED_ROLLOUT=0x01, ESP_FUSED=0x02, BLAS=0x08, etc.).
#[allow(clippy::too_many_arguments)]
pub fn lewm_rollout_fused(
    seq: &mut [f32],
    conditioning: &[f32],
    num_steps: usize,
    hidden: usize,
    num_heads: usize,
    inner_dim: usize,
    inter: usize,
    num_layers: usize,
    adaln_ws: &[*const f32],
    adaln_bs: &[*const f32],
    attn_norm_ws: &[*const f32],
    to_qkvs: &[*const f32],
    attn_out_ws: &[*const f32],
    attn_out_bs: &[*const f32],
    mlp_norm_ws: &[*const f32],
    mlp_up_ws: &[*const f32],
    mlp_up_bs: &[*const f32],
    mlp_down_ws: &[*const f32],
    mlp_down_bs: &[*const f32],
    mod_buf: &mut [f32],
    normed_buf: &mut [f32],
    qkv_buf: &mut [f32],
    attn_buf: &mut [f32],
    proj_buf: &mut [f32],
    scores_buf: &mut [f32],
    packed_a: &mut [f32],
    packed_b: &mut [f32],
    mode: u32,
) -> Result<(), SynapseError> {
    unsafe {
        check_status(ffi::syn_lewm_rollout_fused(
            seq.as_mut_ptr(),
            conditioning.as_ptr(),
            num_steps,
            hidden,
            num_heads,
            inner_dim,
            inter,
            num_layers,
            adaln_ws.as_ptr(),
            adaln_bs.as_ptr(),
            attn_norm_ws.as_ptr(),
            to_qkvs.as_ptr(),
            attn_out_ws.as_ptr(),
            attn_out_bs.as_ptr(),
            mlp_norm_ws.as_ptr(),
            mlp_up_ws.as_ptr(),
            mlp_up_bs.as_ptr(),
            mlp_down_ws.as_ptr(),
            mlp_down_bs.as_ptr(),
            mod_buf.as_mut_ptr(),
            normed_buf.as_mut_ptr(),
            qkv_buf.as_mut_ptr(),
            attn_buf.as_mut_ptr(),
            proj_buf.as_mut_ptr(),
            scores_buf.as_mut_ptr(),
            packed_a.as_mut_ptr(),
            packed_b.as_mut_ptr(),
            mode,
        ))
    }
}

/// Projection GEMV with fused bias: output[m,n] = input[m,k] * weight[n,k]^T + bias[n].
///
/// `input` is `[m * k]`, `weight` is `[n * k]` (row-major, each row = one output neuron),
/// `bias` is `[n]` (or empty for no bias). Returns `[m * n]` f32 output.
pub fn projection_gemv_bias(
    m: usize,
    n: usize,
    k: usize,
    input: &[f32],
    weight: &[f32],
    bias: &[f32],
) -> Result<Vec<f32>, SynapseError> {
    if m == 0 || n == 0 || k == 0 {
        return Ok(vec![0.0f32; m * n]);
    }
    if input.len() < m * k {
        return Err(SynapseError::ShapeMismatch);
    }
    if weight.len() < n * k {
        return Err(SynapseError::ShapeMismatch);
    }
    if !bias.is_empty() && bias.len() < n {
        return Err(SynapseError::ShapeMismatch);
    }
    let mut out = vec![0.0f32; m * n];
    let bias_ptr = if bias.is_empty() {
        std::ptr::null()
    } else {
        bias.as_ptr()
    };
    unsafe {
        check_status(ffi::syn_projection_gemv_bias(
            m,
            n,
            k,
            input.as_ptr(),
            weight.as_ptr(),
            bias_ptr,
            out.as_mut_ptr(),
        ))?;
    }
    Ok(out)
}

// ------------------------------------------------------------------
// KV-Cache (RAII wrapper over opaque FFI handle)
// ------------------------------------------------------------------

/// An owned, RAII-managed KV-cache backed by the Zig runtime.
pub struct KvCache {
    ptr: *mut ffi::syn_kvcache_t,
    stride: usize,
}

unsafe impl Send for KvCache {}

impl Drop for KvCache {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            unsafe { ffi::syn_kvcache_destroy(self.ptr) };
        }
    }
}

impl fmt::Debug for KvCache {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "KvCache(stride={})", self.stride)
    }
}

impl KvCache {
    /// Create a KV-cache with pre-allocated buffers.
    pub fn new(max_seq: usize, n_kv_heads: usize, head_dim: usize) -> Result<Self, SynapseError> {
        let mut ptr: *mut ffi::syn_kvcache_t = ptr::null_mut();
        unsafe {
            check_status(ffi::syn_kvcache_create(
                max_seq, n_kv_heads, head_dim, &mut ptr,
            ))?;
        }
        Ok(KvCache {
            ptr,
            stride: n_kv_heads * head_dim,
        })
    }

    /// Append K/V vectors for a single token.
    ///
    /// Both slices must have exactly `n_kv_heads * head_dim` elements.
    pub fn append(&mut self, k_token: &[f32], v_token: &[f32]) -> Result<(), SynapseError> {
        if k_token.len() != self.stride || v_token.len() != self.stride {
            return Err(SynapseError::ShapeMismatch);
        }
        unsafe {
            check_status(ffi::syn_kvcache_append(
                self.ptr,
                k_token.as_ptr(),
                v_token.as_ptr(),
                self.stride,
            ))
        }
    }

    /// Get the current sequence length and pointers into the populated region.
    pub fn seq_len(&self) -> Result<usize, SynapseError> {
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

    /// Get zero-copy slices of the populated K and V data.
    ///
    /// The returned slices are valid until the next `append` or `reset`.
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
            let total = seq_len * self.stride;
            let k = if total > 0 {
                std::slice::from_raw_parts(k_ptr, total)
            } else {
                &[]
            };
            let v = if total > 0 {
                std::slice::from_raw_parts(v_ptr, total)
            } else {
                &[]
            };
            Ok((k, v, seq_len))
        }
    }

    /// Reset the position counter to 0. No deallocation.
    pub fn reset(&mut self) -> Result<(), SynapseError> {
        unsafe { check_status(ffi::syn_kvcache_reset(self.ptr)) }
    }

    /// Truncate to a given position. Used by speculative decoding to roll back
    /// rejected draft tokens. Only the position counter is updated.
    pub fn truncate_to(&mut self, new_len: usize) -> Result<(), SynapseError> {
        unsafe { check_status(ffi::syn_kvcache_truncate(self.ptr, new_len)) }
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
        let input = Tensor::from_data(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], &[2, 4]).unwrap();

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
        let q =
            Tensor::from_data(&[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0], &[1, 1, 2, 4]).unwrap();
        let k = q.clone_tensor().unwrap();
        let v =
            Tensor::from_data(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], &[1, 1, 2, 4]).unwrap();

        let (output, weights) = q.scaled_dot_product_attention(&k, &v, 0.5, false).unwrap();

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
            let (_out, _w) = q.scaled_dot_product_attention(&k, &v, 0.5, false).unwrap();
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

    // ================================================================
    // FFI roundtrip: RMS normalization
    // ================================================================

    #[test]
    fn test_rmsnorm_roundtrip() {
        // input: [2, 4], normalize over last dim
        let input = Tensor::from_data(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], &[2, 4]).unwrap();
        let gamma = Tensor::from_data(&[1.0, 1.0, 1.0, 1.0], &[4]).unwrap();

        let result = input.rmsnorm(&gamma, 1, 1e-5).unwrap();
        assert_eq!(result.shape().unwrap(), &[2, 4]);

        let data = result.to_vec().unwrap();
        // RMS norm preserves relative ordering within each row
        assert!(data[0] < data[1] && data[1] < data[2] && data[2] < data[3]);
        // Verify RMS normalization property: sqrt(mean(out^2)) ≈ 1 (with gamma=1)
        let row0 = &data[0..4];
        let rms: f32 = (row0.iter().map(|x| x * x).sum::<f32>() / 4.0).sqrt();
        assert!((rms - 1.0).abs() < 0.1, "RMS should be ~1, got {}", rms);
    }

    #[test]
    fn test_rmsnorm_shape_mismatch() {
        let input = Tensor::from_data(&[1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
        // gamma has wrong size (3 instead of 2)
        let gamma = Tensor::from_data(&[1.0, 1.0, 1.0], &[3]).unwrap();
        let err = input.rmsnorm(&gamma, 1, 1e-5);
        assert!(err.is_err());
        assert_eq!(err.unwrap_err(), SynapseError::ShapeMismatch);
    }

    #[test]
    fn test_rmsnorm_invalid_norm_dims() {
        let input = Tensor::from_data(&[1.0, 2.0], &[2]).unwrap();
        let gamma = Tensor::from_data(&[1.0, 1.0], &[2]).unwrap();
        // num_norm_dims=0 is invalid
        let err = input.rmsnorm(&gamma, 0, 1e-5);
        assert!(err.is_err());
        assert_eq!(err.unwrap_err(), SynapseError::InvalidArg);
    }

    #[test]
    fn test_rmsnorm_memory_safety_10k() {
        for _ in 0..10_000 {
            let input = Tensor::from_data(&[1.0, 2.0, 3.0, 4.0], &[1, 4]).unwrap();
            let gamma = Tensor::from_data(&[1.0, 1.0, 1.0, 1.0], &[4]).unwrap();
            let _result = input.rmsnorm(&gamma, 1, 1e-5).unwrap();
        }
    }

    // ================================================================
    // FFI roundtrip: SiLU activation
    // ================================================================

    #[test]
    fn test_silu_roundtrip() {
        let src = [0.0, 1.0, -1.0, 2.0];
        let mut dst = [0.0f32; 4];
        silu(&mut dst, &src).unwrap();

        // silu(0) = 0
        assert!(dst[0].abs() < 1e-6);
        // silu(x) > 0 for x > 0
        assert!(dst[1] > 0.0);
        // silu(x) < 0 for x < 0
        assert!(dst[2] < 0.0);
        // silu(2) ≈ 2 * sigmoid(2) ≈ 2 * 0.8808 ≈ 1.7616
        assert!((dst[3] - 1.7616).abs() < 0.01);
    }

    #[test]
    fn test_silu_null_ptr() {
        let mut dst = [0.0f32; 4];
        // Should succeed with zero length even though src is arbitrary
        let result = silu(&mut dst[..0], &[]);
        assert!(result.is_ok());
    }

    // ================================================================
    // FFI roundtrip: SwiGLU
    // ================================================================

    #[test]
    fn test_swiglu_roundtrip() {
        let gate = [0.0, 1.0, 2.0, -1.0];
        let up = [1.0, 2.0, 3.0, 4.0];
        let mut dst = [0.0f32; 4];
        swiglu(&mut dst, &gate, &up).unwrap();

        // swiglu(0, x) = silu(0) * x = 0
        assert!(dst[0].abs() < 1e-6);
        // swiglu(g, u) = silu(g) * u
        assert!(dst[1] > 0.0); // silu(1) * 2
        assert!(dst[2] > 0.0); // silu(2) * 3
        assert!(dst[3] < 0.0); // silu(-1) * 4 < 0
    }

    // ================================================================
    // FFI roundtrip: quantize / dequantize
    // ================================================================

    #[test]
    fn test_quantize_dequantize_roundtrip() {
        let data = vec![1.0, -2.0, 3.0, -4.0, 0.5, -0.5, 1.5, -1.5];
        let channels = 2;
        let channel_size = 4;

        let (quantized, scales) = quantize_per_channel_int8(&data, channels, channel_size).unwrap();
        assert_eq!(quantized.len(), 8);
        assert_eq!(scales.len(), 2);

        let dequantized =
            dequantize_per_channel_int8(&quantized, channels, channel_size, &scales).unwrap();
        assert_eq!(dequantized.len(), 8);

        // Dequantized should be close to original (quantization error < scale)
        for (orig, deq) in data.iter().zip(dequantized.iter()) {
            let err = (orig - deq).abs();
            assert!(
                err < 0.1,
                "quantization error too large: {} vs {}",
                orig,
                deq
            );
        }
    }

    #[test]
    fn test_quantize_invalid_dimensions() {
        let result = quantize_per_channel_int8(&[], 0, 0);
        assert!(result.is_err());
        assert_eq!(result.unwrap_err(), SynapseError::InvalidArg);
    }

    #[test]
    fn test_quantize_shape_mismatch() {
        // data has 4 elements but channels * channel_size = 6
        let result = quantize_per_channel_int8(&[1.0, 2.0, 3.0, 4.0], 2, 3);
        assert!(result.is_err());
        assert_eq!(result.unwrap_err(), SynapseError::ShapeMismatch);
    }

    // ================================================================
    // FFI roundtrip: INT8 quantized GEMM
    // ================================================================

    #[test]
    fn test_qgemm_roundtrip() {
        // Simple 2x3 * 3x2 = 2x2 with identity-like scales
        let a: Vec<i8> = vec![1, 2, 3, 4, 5, 6];
        let b: Vec<i8> = vec![7, 8, 9, 10, 11, 12];
        let scales_a = vec![1.0f32, 1.0];
        let scales_b = vec![1.0f32, 1.0];

        let c = qgemm_int8(2, 2, 3, &a, &b, &scales_a, &scales_b).unwrap();
        assert_eq!(c.len(), 4);

        // Verify: C[0,0] = 1*7 + 2*9 + 3*11 = 7+18+33 = 58
        assert!((c[0] - 58.0).abs() < 1e-3, "C[0,0] = {}", c[0]);
        // C[0,1] = 1*8 + 2*10 + 3*12 = 8+20+36 = 64
        assert!((c[1] - 64.0).abs() < 1e-3, "C[0,1] = {}", c[1]);
    }

    #[test]
    fn test_qgemm_with_scales() {
        let a: Vec<i8> = vec![127, 0, 0, 127];
        let b: Vec<i8> = vec![127, 0, 0, 127];
        let scales_a = vec![2.0f32, 3.0];
        let scales_b = vec![0.5f32, 1.0];

        let c = qgemm_int8(2, 2, 2, &a, &b, &scales_a, &scales_b).unwrap();
        // C[0,0] = scales_a[0] * scales_b[0] * (127*127) = 2*0.5*16129 = 16129.0
        assert!((c[0] - 16129.0).abs() < 1e-1, "C[0,0] = {}", c[0]);
    }

    #[test]
    fn test_qgemm_zero_dimensions() {
        let c = qgemm_int8(0, 0, 0, &[], &[], &[], &[]).unwrap();
        assert!(c.is_empty());
    }

    // ================================================================
    // FFI roundtrip: KV-Cache
    // ================================================================

    #[test]
    fn test_kvcache_create_and_slice() {
        let cache = KvCache::new(16, 2, 4).unwrap();
        let seq_len = cache.seq_len().unwrap();
        assert_eq!(seq_len, 0);
    }

    #[test]
    fn test_kvcache_append_and_slice() {
        let mut cache = KvCache::new(16, 2, 4).unwrap(); // stride = 8
        let k_token = vec![1.0f32; 8];
        let v_token = vec![2.0f32; 8];

        cache.append(&k_token, &v_token).unwrap();
        let (k, v, seq_len) = cache.slice().unwrap();

        assert_eq!(seq_len, 1);
        assert_eq!(k.len(), 8);
        assert_eq!(v.len(), 8);
        assert!(approx_eq(k, &k_token, 1e-6));
        assert!(approx_eq(v, &v_token, 1e-6));
    }

    #[test]
    fn test_kvcache_multiple_appends() {
        let mut cache = KvCache::new(16, 1, 4).unwrap(); // stride = 4

        for i in 0..5 {
            let val = i as f32;
            cache.append(&[val; 4], &[val + 10.0; 4]).unwrap();
        }

        let (k, v, seq_len) = cache.slice().unwrap();
        assert_eq!(seq_len, 5);
        assert_eq!(k.len(), 20);
        assert_eq!(v.len(), 20);

        // Verify first token
        assert!(approx_eq(&k[0..4], &[0.0; 4], 1e-6));
        assert!(approx_eq(&v[0..4], &[10.0; 4], 1e-6));
    }

    #[test]
    fn test_kvcache_reset() {
        let mut cache = KvCache::new(16, 1, 4).unwrap();
        cache.append(&[1.0; 4], &[2.0; 4]).unwrap();
        assert_eq!(cache.seq_len().unwrap(), 1);

        cache.reset().unwrap();
        assert_eq!(cache.seq_len().unwrap(), 0);

        // Can append again after reset
        cache.append(&[3.0; 4], &[4.0; 4]).unwrap();
        assert_eq!(cache.seq_len().unwrap(), 1);
    }

    #[test]
    fn test_kvcache_shape_mismatch() {
        let mut cache = KvCache::new(16, 2, 4).unwrap(); // stride = 8
                                                         // Wrong stride (4 instead of 8)
        let err = cache.append(&[1.0; 4], &[2.0; 4]);
        assert!(err.is_err());
        assert_eq!(err.unwrap_err(), SynapseError::ShapeMismatch);
    }

    #[test]
    fn test_kvcache_invalid_dimensions() {
        let err = KvCache::new(0, 2, 4);
        assert!(err.is_err());
        assert_eq!(err.unwrap_err(), SynapseError::InvalidDimensions);
    }

    #[test]
    fn test_kvcache_memory_safety_10k() {
        for _ in 0..10_000 {
            let mut cache = KvCache::new(4, 1, 4).unwrap();
            cache.append(&[1.0; 4], &[2.0; 4]).unwrap();
            let _ = cache.slice().unwrap();
            cache.reset().unwrap();
            // Cache dropped via RAII.
        }
    }

    // ================================================================
    // Memory safety: SiLU/SwiGLU/quantize 10K cycles
    // ================================================================

    #[test]
    fn test_silu_memory_safety_10k() {
        for _ in 0..10_000 {
            let mut dst = [0.0f32; 4];
            silu(&mut dst, &[1.0, 2.0, 3.0, 4.0]).unwrap();
        }
    }

    #[test]
    fn test_quantize_dequantize_memory_safety_10k() {
        for _ in 0..10_000 {
            let data = vec![1.0, -2.0, 3.0, -4.0];
            let (q, s) = quantize_per_channel_int8(&data, 1, 4).unwrap();
            let _d = dequantize_per_channel_int8(&q, 1, 4, &s).unwrap();
        }
    }

    #[test]
    fn test_qgemm_memory_safety_10k() {
        let a: Vec<i8> = vec![1, 2, 3, 4];
        let b: Vec<i8> = vec![5, 6, 7, 8];
        let sa = vec![1.0f32, 1.0];
        let sb = vec![1.0f32, 1.0];
        for _ in 0..10_000 {
            let _c = qgemm_int8(2, 2, 2, &a, &b, &sa, &sb).unwrap();
        }
    }

    #[test]
    fn test_capability_summary_roundtrip() {
        let summary = capability_summary().expect("capability summary should succeed");
        assert_eq!(summary.abi_version, ffi::SYN_CAPABILITY_ABI_VERSION);
        assert!(summary.has_feature(ffi::SYN_FEATURE_SGEMM));
        assert!(!summary.feature_names().is_empty());
    }

    #[test]
    fn test_runtime_capabilities_json_roundtrip() {
        let json = runtime_capabilities_json().expect("json capability report should succeed");
        let parsed: serde_json::Value =
            serde_json::from_str(&json).expect("json capability report should parse");
        assert_eq!(parsed["abi_version"].as_u64(), Some(1));
        assert!(parsed["simd_backend"].as_str().is_some());
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
