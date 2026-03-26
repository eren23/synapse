//! Raw FFI bindings to the Zig-backed Synapse tensor library (`libsynapse_zig.a`).
//!
//! All functions return `syn_status_t` (i32) error codes.
//! Handles are opaque pointers — do not dereference or free them directly;
//! use the corresponding release/destroy functions.

#![allow(non_camel_case_types)]

use std::os::raw::c_int;

// ------------------------------------------------------------------
// Status codes
// ------------------------------------------------------------------
pub const SYN_OK: c_int = 0;
pub const SYN_ERR_NULL_PTR: c_int = 1;
pub const SYN_ERR_INVALID_ARG: c_int = 2;
pub const SYN_ERR_OUT_OF_MEMORY: c_int = 3;
pub const SYN_ERR_SHAPE_MISMATCH: c_int = 4;
pub const SYN_ERR_NOT_CONTIGUOUS: c_int = 5;
pub const SYN_ERR_INVALID_AXIS: c_int = 6;
pub const SYN_ERR_INVALID_DIMENSIONS: c_int = 7;
pub const SYN_ERR_INTERNAL: c_int = 8;

// ------------------------------------------------------------------
// Opaque handle types
// ------------------------------------------------------------------
pub type syn_storage_t = std::ffi::c_void;
pub type syn_tensor_t = std::ffi::c_void;
pub type syn_arena_t = std::ffi::c_void;
pub type syn_pool_t = std::ffi::c_void;
pub type syn_kvcache_t = std::ffi::c_void;

pub type syn_status_t = c_int;

extern "C" {
    // ------------------------------------------------------------------
    // Storage
    // ------------------------------------------------------------------
    pub fn syn_storage_create(count: usize, out: *mut *mut syn_storage_t) -> syn_status_t;
    pub fn syn_storage_retain(s: *mut syn_storage_t) -> syn_status_t;
    pub fn syn_storage_release(s: *mut syn_storage_t) -> syn_status_t;
    pub fn syn_storage_data(s: *mut syn_storage_t, out: *mut *mut f32) -> syn_status_t;
    pub fn syn_storage_len(s: *mut syn_storage_t, out: *mut usize) -> syn_status_t;

    // ------------------------------------------------------------------
    // Tensor
    // ------------------------------------------------------------------
    pub fn syn_tensor_create(
        storage: *mut syn_storage_t,
        dims: *const usize,
        ndim: usize,
        out: *mut *mut syn_tensor_t,
    ) -> syn_status_t;

    pub fn syn_tensor_destroy(t: *mut syn_tensor_t) -> syn_status_t;

    pub fn syn_tensor_shape(
        t: *mut syn_tensor_t,
        out_dims: *mut usize,
        out_ndim: *mut usize,
    ) -> syn_status_t;

    pub fn syn_tensor_ndim(t: *mut syn_tensor_t, out: *mut usize) -> syn_status_t;

    pub fn syn_tensor_data_ptr(t: *mut syn_tensor_t, out: *mut *mut f32) -> syn_status_t;

    pub fn syn_tensor_is_contiguous(t: *mut syn_tensor_t, out: *mut i32) -> syn_status_t;

    pub fn syn_tensor_contiguous(
        t: *mut syn_tensor_t,
        out: *mut *mut syn_tensor_t,
    ) -> syn_status_t;

    // ------------------------------------------------------------------
    // Arena allocator
    // ------------------------------------------------------------------
    pub fn syn_arena_create(region_capacity: usize, out: *mut *mut syn_arena_t) -> syn_status_t;
    pub fn syn_arena_reset(a: *mut syn_arena_t) -> syn_status_t;
    pub fn syn_arena_destroy(a: *mut syn_arena_t) -> syn_status_t;

    // ------------------------------------------------------------------
    // Pool allocator
    // ------------------------------------------------------------------
    pub fn syn_pool_create(count: usize, out: *mut *mut syn_pool_t) -> syn_status_t;
    pub fn syn_pool_destroy(p: *mut syn_pool_t) -> syn_status_t;

    // ------------------------------------------------------------------
    // SGEMM
    // ------------------------------------------------------------------
    pub fn syn_sgemm(
        m: usize, n: usize, k: usize,
        a: *const f32, lda: usize, trans_a: c_int,
        b: *const f32, ldb: usize, trans_b: c_int,
        c: *mut f32, ldc: usize,
    ) -> syn_status_t;

    // ------------------------------------------------------------------
    // Element-wise ops
    // ------------------------------------------------------------------
    pub fn syn_add(dst: *mut f32, a: *const f32, b: *const f32, len: usize) -> syn_status_t;
    pub fn syn_sub(dst: *mut f32, a: *const f32, b: *const f32, len: usize) -> syn_status_t;
    pub fn syn_mul(dst: *mut f32, a: *const f32, b: *const f32, len: usize) -> syn_status_t;
    pub fn syn_div(dst: *mut f32, a: *const f32, b: *const f32, len: usize) -> syn_status_t;

    // ------------------------------------------------------------------
    // Activations
    // ------------------------------------------------------------------
    pub fn syn_relu(dst: *mut f32, src: *const f32, len: usize) -> syn_status_t;
    pub fn syn_sigmoid(dst: *mut f32, src: *const f32, len: usize) -> syn_status_t;
    pub fn syn_tanh_act(dst: *mut f32, src: *const f32, len: usize) -> syn_status_t;
    pub fn syn_gelu(dst: *mut f32, src: *const f32, len: usize) -> syn_status_t;

    // ------------------------------------------------------------------
    // Tensor reductions
    // ------------------------------------------------------------------
    pub fn syn_reduce_sum(
        t: *mut syn_tensor_t, axis: usize, keepdim: c_int,
        out: *mut *mut syn_tensor_t,
    ) -> syn_status_t;

    pub fn syn_reduce_max(
        t: *mut syn_tensor_t, axis: usize, keepdim: c_int,
        out: *mut *mut syn_tensor_t,
    ) -> syn_status_t;

    pub fn syn_reduce_mean(
        t: *mut syn_tensor_t, axis: usize, keepdim: c_int,
        out: *mut *mut syn_tensor_t,
    ) -> syn_status_t;

    // ------------------------------------------------------------------
    // Softmax
    // ------------------------------------------------------------------
    pub fn syn_softmax(
        input: *mut syn_tensor_t, axis: usize,
        out: *mut *mut syn_tensor_t,
    ) -> syn_status_t;

    // ------------------------------------------------------------------
    // Batch normalization
    // ------------------------------------------------------------------
    pub fn syn_batchnorm(
        input: *mut syn_tensor_t, num_features: usize, eps: f32,
        out: *mut *mut syn_tensor_t,
    ) -> syn_status_t;

    // ------------------------------------------------------------------
    // Conv2d
    // ------------------------------------------------------------------
    pub fn syn_conv2d(
        input: *mut syn_tensor_t, kernel: *mut syn_tensor_t,
        stride_h: usize, stride_w: usize,
        pad_h: usize, pad_w: usize,
        out: *mut *mut syn_tensor_t,
    ) -> syn_status_t;

    // ------------------------------------------------------------------
    // Pooling
    // ------------------------------------------------------------------
    pub fn syn_maxpool2d(
        input: *mut syn_tensor_t,
        kernel_h: usize, kernel_w: usize,
        stride_h: usize, stride_w: usize,
        out: *mut *mut syn_tensor_t,
    ) -> syn_status_t;

    pub fn syn_avgpool2d(
        input: *mut syn_tensor_t,
        kernel_h: usize, kernel_w: usize,
        stride_h: usize, stride_w: usize,
        out: *mut *mut syn_tensor_t,
    ) -> syn_status_t;

    // ------------------------------------------------------------------
    // Transpose
    // ------------------------------------------------------------------
    pub fn syn_transpose(
        input: *mut syn_tensor_t,
        out: *mut *mut syn_tensor_t,
    ) -> syn_status_t;

    // ------------------------------------------------------------------
    // Raw SIMD vector ops
    // ------------------------------------------------------------------
    pub fn syn_vadd(dst: *mut f32, a: *const f32, b: *const f32, len: usize) -> syn_status_t;
    pub fn syn_vmul(dst: *mut f32, a: *const f32, b: *const f32, len: usize) -> syn_status_t;
    pub fn syn_vfma(
        dst: *mut f32, a: *const f32, b: *const f32, c: *const f32, len: usize,
    ) -> syn_status_t;
    pub fn syn_vreduce_sum(src: *const f32, len: usize, out: *mut f32) -> syn_status_t;
    pub fn syn_vreduce_max(src: *const f32, len: usize, out: *mut f32) -> syn_status_t;

    // ------------------------------------------------------------------
    // Layer normalization
    // ------------------------------------------------------------------
    pub fn syn_layernorm_forward(
        out: *mut *mut syn_tensor_t,
        input: *mut syn_tensor_t,
        gamma: *mut syn_tensor_t,
        beta: *mut syn_tensor_t,
        normalized_dim: usize,
        eps: f32,
    ) -> syn_status_t;

    // ------------------------------------------------------------------
    // Scaled dot-product attention
    // ------------------------------------------------------------------
    pub fn syn_scaled_dot_product_attention(
        out: *mut *mut syn_tensor_t,
        attn_weights: *mut *mut syn_tensor_t,
        query: *mut syn_tensor_t,
        key: *mut syn_tensor_t,
        value: *mut syn_tensor_t,
        scale: f32,
        causal: i32,
    ) -> syn_status_t;

    // ------------------------------------------------------------------
    // Rotary positional embedding (RoPE)
    // ------------------------------------------------------------------
    pub fn syn_rope_forward(
        out: *mut *mut syn_tensor_t,
        input: *mut syn_tensor_t,
        cos_table: *mut syn_tensor_t,
        sin_table: *mut syn_tensor_t,
        offset: usize,
    ) -> syn_status_t;

    // ------------------------------------------------------------------
    // Causal attention mask
    // ------------------------------------------------------------------
    pub fn syn_causal_mask(out: *mut *mut syn_tensor_t, seq_len: usize) -> syn_status_t;

    // ------------------------------------------------------------------
    // RMS normalization
    // ------------------------------------------------------------------
    pub fn syn_rmsnorm_forward(
        out: *mut *mut syn_tensor_t,
        input: *mut syn_tensor_t,
        gamma: *mut syn_tensor_t,
        num_norm_dims: usize,
        eps: f32,
    ) -> syn_status_t;

    // ------------------------------------------------------------------
    // SiLU activation
    // ------------------------------------------------------------------
    pub fn syn_silu(dst: *mut f32, src: *const f32, len: usize) -> syn_status_t;

    // ------------------------------------------------------------------
    // Fused SwiGLU
    // ------------------------------------------------------------------
    pub fn syn_swiglu(
        dst: *mut f32, gate: *const f32, up: *const f32, len: usize,
    ) -> syn_status_t;

    // ------------------------------------------------------------------
    // Per-channel INT8 quantization / dequantization
    // ------------------------------------------------------------------
    pub fn syn_quantize_per_channel_int8(
        data: *const f32,
        channels: usize,
        channel_size: usize,
        out: *mut i8,
        scales: *mut f32,
    ) -> syn_status_t;

    pub fn syn_dequantize_per_channel_int8(
        data: *const i8,
        channels: usize,
        channel_size: usize,
        out: *mut f32,
        scales: *const f32,
    ) -> syn_status_t;

    // ------------------------------------------------------------------
    // INT8 quantized GEMM
    // ------------------------------------------------------------------
    pub fn syn_qgemm_int8(
        m: usize, n: usize, k: usize,
        a: *const i8, lda: usize,
        b: *const i8, ldb: usize,
        c: *mut f32, ldc: usize,
        scales_a: *const f32,
        scales_b: *const f32,
    ) -> syn_status_t;

    /// Fused causal attention for a single head on flat arrays.
    /// Q: [seq_q, d_head], K: [seq_k, d_head], V: [seq_k, d_head]
    /// Output: [seq_q, d_head]
    pub fn syn_fused_attention(
        seq_q: usize,
        seq_k: usize,
        d_head: usize,
        q: *const f32,
        k: *const f32,
        v: *const f32,
        out: *mut f32,
    ) -> syn_status_t;

    /// Bidirectional fused attention (no causal mask). For ViT/JEPA/CLIP encoders.
    pub fn syn_fused_attention_bidi(
        seq_q: usize, seq_k: usize, d_head: usize,
        q: *const f32, k: *const f32, v: *const f32,
        out: *mut f32,
    ) -> syn_status_t;

    /// Q4_0 matrix-vector multiply: C[1,N] = A_f32[1,K] @ dequant(B_q4[N,K]).
    /// K must be a multiple of 32.
    /// Geometric attention: distance-biased attention for 3D point clouds.
    pub fn syn_geometric_attention(
        n: usize, d: usize, pos_dim: usize,
        q: *const f32, k: *const f32, v: *const f32,
        positions: *const f32, out: *mut f32,
        sigma: f32,
    ) -> syn_status_t;

    pub fn syn_q4_0_gemv(
        n: usize, k: usize,
        a: *const f32,
        b_q4: *const u8,
        c: *mut f32,
    ) -> syn_status_t;

    // ------------------------------------------------------------------
    // KV-Cache
    // ------------------------------------------------------------------
    pub fn syn_kvcache_create(
        max_seq: usize, n_kv_heads: usize, head_dim: usize,
        out: *mut *mut syn_kvcache_t,
    ) -> syn_status_t;

    pub fn syn_kvcache_destroy(cache: *mut syn_kvcache_t) -> syn_status_t;

    pub fn syn_kvcache_append(
        cache: *mut syn_kvcache_t,
        k_token: *const f32,
        v_token: *const f32,
        stride: usize,
    ) -> syn_status_t;

    pub fn syn_kvcache_slice(
        cache: *mut syn_kvcache_t,
        k_out: *mut *const f32,
        v_out: *mut *const f32,
        seq_len_out: *mut usize,
    ) -> syn_status_t;

    pub fn syn_kvcache_reset(cache: *mut syn_kvcache_t) -> syn_status_t;
    pub fn syn_kvcache_truncate(cache: *mut syn_kvcache_t, new_len: usize) -> syn_status_t;
}
