/**
 * synapse.h -- C API for the Synapse tensor library (backed by Zig).
 *
 * All functions return syn_status_t (int32_t) error codes.
 * Handles are opaque pointers -- do not dereference or free them directly;
 * use the corresponding release/destroy functions.
 *
 * Build: zig build          (produces libsynapse_zig.a in zig-out/lib/)
 * Cross: zig build -Dtarget=aarch64-linux-gnu
 *        zig build -Dtarget=x86_64-linux-gnu
 */
#ifndef SYNAPSE_H
#define SYNAPSE_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ------------------------------------------------------------------ */
/* Status codes                                                        */
/* ------------------------------------------------------------------ */
typedef int32_t syn_status_t;

#define SYN_OK                     0
#define SYN_ERR_NULL_PTR           1
#define SYN_ERR_INVALID_ARG        2
#define SYN_ERR_OUT_OF_MEMORY      3
#define SYN_ERR_SHAPE_MISMATCH     4
#define SYN_ERR_NOT_CONTIGUOUS     5
#define SYN_ERR_INVALID_AXIS       6
#define SYN_ERR_INVALID_DIMENSIONS 7
#define SYN_ERR_INTERNAL           8

/* ------------------------------------------------------------------ */
/* Opaque handle types                                                 */
/* ------------------------------------------------------------------ */
typedef struct syn_storage_s syn_storage_t;
typedef struct syn_tensor_s  syn_tensor_t;
typedef struct syn_arena_s    syn_arena_t;
typedef struct syn_pool_s     syn_pool_t;
typedef struct syn_kvcache_s  syn_kvcache_t;

/* ------------------------------------------------------------------ */
/* Storage (ref-counted, 64-byte aligned f32 buffer)                   */
/* ------------------------------------------------------------------ */

/** Allocate zero-initialized storage for `count` floats. */
syn_status_t syn_storage_create(size_t count, syn_storage_t **out);

/** Increment the storage reference count. */
syn_status_t syn_storage_retain(syn_storage_t *s);

/** Decrement the reference count; frees when it reaches zero. */
syn_status_t syn_storage_release(syn_storage_t *s);

/** Get a float pointer to the raw data. */
syn_status_t syn_storage_data(syn_storage_t *s, float **out);

/** Get the number of float elements. */
syn_status_t syn_storage_len(syn_storage_t *s, size_t *out);

/* ------------------------------------------------------------------ */
/* Tensor (N-D view over storage)                                      */
/* ------------------------------------------------------------------ */

/** Create a tensor from storage + shape.  Storage refcount is incremented. */
syn_status_t syn_tensor_create(syn_storage_t *storage,
                               const size_t *dims, size_t ndim,
                               syn_tensor_t **out);

/** Destroy the tensor handle and release its storage reference. */
syn_status_t syn_tensor_destroy(syn_tensor_t *t);

/** Write shape dims and ndim.  Either pointer may be NULL. */
syn_status_t syn_tensor_shape(syn_tensor_t *t,
                              size_t *out_dims, size_t *out_ndim);

/** Get the number of dimensions. */
syn_status_t syn_tensor_ndim(syn_tensor_t *t, size_t *out);

/** Get a raw float pointer to the tensor data at its offset. */
syn_status_t syn_tensor_data_ptr(syn_tensor_t *t, float **out);

/** Check contiguity.  Writes 1 (contiguous) or 0 (strided). */
syn_status_t syn_tensor_is_contiguous(syn_tensor_t *t, int32_t *out);

/** Return a contiguous copy (or a lightweight alias if already contiguous). */
syn_status_t syn_tensor_contiguous(syn_tensor_t *t, syn_tensor_t **out);

/* ------------------------------------------------------------------ */
/* Arena allocator                                                     */
/* ------------------------------------------------------------------ */

/** Create a region-based arena with the given per-region capacity (bytes). */
syn_status_t syn_arena_create(size_t region_capacity, syn_arena_t **out);

/** O(1) reset -- rewind to the start; all prior allocations are invalid. */
syn_status_t syn_arena_reset(syn_arena_t *a);

/** Free all backing memory and destroy the arena. */
syn_status_t syn_arena_destroy(syn_arena_t *a);

/* ------------------------------------------------------------------ */
/* Pool allocator (128-byte fixed-size slots)                          */
/* ------------------------------------------------------------------ */

/** Create a pool with `count` pre-allocated 128-byte slots. */
syn_status_t syn_pool_create(size_t count, syn_pool_t **out);

/** Destroy the pool and free backing memory. */
syn_status_t syn_pool_destroy(syn_pool_t *p);

/* ------------------------------------------------------------------ */
/* SGEMM: C = op(A) * op(B),  row-major layout                        */
/*   trans=0: use matrix as-is; trans!=0: use transpose.               */
/*   C must be zero-initialized before the first call.                 */
/* ------------------------------------------------------------------ */
syn_status_t syn_sgemm(size_t m, size_t n, size_t k,
                       const float *a, size_t lda, int trans_a,
                       const float *b, size_t ldb, int trans_b,
                       float *c, size_t ldc);

/* ------------------------------------------------------------------ */
/* Element-wise operations on flat f32 arrays                          */
/* ------------------------------------------------------------------ */
syn_status_t syn_add(float *dst, const float *a, const float *b, size_t len);
syn_status_t syn_sub(float *dst, const float *a, const float *b, size_t len);
syn_status_t syn_mul(float *dst, const float *a, const float *b, size_t len);
syn_status_t syn_div(float *dst, const float *a, const float *b, size_t len);

/* ------------------------------------------------------------------ */
/* Activation functions on flat f32 arrays                             */
/* ------------------------------------------------------------------ */
syn_status_t syn_relu(float *dst, const float *src, size_t len);
syn_status_t syn_sigmoid(float *dst, const float *src, size_t len);
syn_status_t syn_tanh_act(float *dst, const float *src, size_t len);
syn_status_t syn_gelu(float *dst, const float *src, size_t len);

/* ------------------------------------------------------------------ */
/* Tensor reductions (keepdim: 0=squeeze, non-zero=keep)               */
/* ------------------------------------------------------------------ */
syn_status_t syn_reduce_sum(syn_tensor_t *t, size_t axis, int keepdim,
                            syn_tensor_t **out);
syn_status_t syn_reduce_max(syn_tensor_t *t, size_t axis, int keepdim,
                            syn_tensor_t **out);
syn_status_t syn_reduce_mean(syn_tensor_t *t, size_t axis, int keepdim,
                             syn_tensor_t **out);

/* ------------------------------------------------------------------ */
/* Softmax along an axis                                               */
/* ------------------------------------------------------------------ */
syn_status_t syn_softmax(syn_tensor_t *input, size_t axis,
                         syn_tensor_t **out);

/* ------------------------------------------------------------------ */
/* Batch normalization (inference, default gamma=1 beta=0)             */
/*   Input must be 2-D [N, C] where C == num_features.                */
/* ------------------------------------------------------------------ */
syn_status_t syn_batchnorm(syn_tensor_t *input, size_t num_features,
                           float eps, syn_tensor_t **out);

/* ------------------------------------------------------------------ */
/* Conv2d  --  NCHW layout                                             */
/*   input:  [N, C_in,  H,  W]                                        */
/*   kernel: [C_out, C_in, KH, KW]                                    */
/* ------------------------------------------------------------------ */
syn_status_t syn_conv2d(syn_tensor_t *input, syn_tensor_t *kernel,
                        size_t stride_h, size_t stride_w,
                        size_t pad_h, size_t pad_w,
                        syn_tensor_t **out);

/* ------------------------------------------------------------------ */
/* Pooling  --  NCHW layout                                            */
/* ------------------------------------------------------------------ */
syn_status_t syn_maxpool2d(syn_tensor_t *input,
                           size_t kernel_h, size_t kernel_w,
                           size_t stride_h, size_t stride_w,
                           syn_tensor_t **out);

syn_status_t syn_avgpool2d(syn_tensor_t *input,
                           size_t kernel_h, size_t kernel_w,
                           size_t stride_h, size_t stride_w,
                           syn_tensor_t **out);

/* ------------------------------------------------------------------ */
/* Transpose (2-D only): result[j,i] = input[i,j]                     */
/* ------------------------------------------------------------------ */
syn_status_t syn_transpose(syn_tensor_t *input, syn_tensor_t **out);

/* ------------------------------------------------------------------ */
/* Raw SIMD vector operations (auto-dispatched backend)                */
/* ------------------------------------------------------------------ */
syn_status_t syn_vadd(float *dst, const float *a, const float *b, size_t len);
syn_status_t syn_vmul(float *dst, const float *a, const float *b, size_t len);
syn_status_t syn_vfma(float *dst, const float *a, const float *b,
                      const float *c, size_t len);
syn_status_t syn_vreduce_sum(const float *src, size_t len, float *out);
syn_status_t syn_vreduce_max(const float *src, size_t len, float *out);

/* ------------------------------------------------------------------ */
/* Layer normalization (trailing dims)                                  */
/*   input:  N-D tensor                                                */
/*   gamma, beta: 1-D tensors with size = product of trailing dims     */
/*   normalized_dim: number of trailing dimensions to normalize over   */
/* ------------------------------------------------------------------ */
syn_status_t syn_layernorm_forward(syn_tensor_t **out,
                                   syn_tensor_t *input,
                                   syn_tensor_t *gamma,
                                   syn_tensor_t *beta,
                                   size_t normalized_dim,
                                   float eps);

/* ------------------------------------------------------------------ */
/* Scaled dot-product attention                                        */
/*   Q: [batch, heads, seq_q, d_head]                                  */
/*   K: [batch, heads, seq_k, d_head]                                  */
/*   V: [batch, heads, seq_k, d_head]                                  */
/*   attn_weights: optional [batch, heads, seq_q, seq_k]; NULL to skip */
/*   scale: accepted for API compat; internally derived from d_head    */
/* ------------------------------------------------------------------ */
syn_status_t syn_scaled_dot_product_attention(syn_tensor_t **out,
                                              syn_tensor_t **attn_weights,
                                              syn_tensor_t *query,
                                              syn_tensor_t *key,
                                              syn_tensor_t *value,
                                              float scale,
                                              int32_t causal);

/* ------------------------------------------------------------------ */
/* Rotary positional embedding (RoPE)                                  */
/*   input: [batch, heads, seq, d_head]  (d_head must be even)         */
/*   cos_table, sin_table: precomputed [max_seq, d_head/2]             */
/*   offset: position offset for KV-cache scenarios                    */
/* ------------------------------------------------------------------ */
syn_status_t syn_rope_forward(syn_tensor_t **out,
                              syn_tensor_t *input,
                              syn_tensor_t *cos_table,
                              syn_tensor_t *sin_table,
                              size_t offset);

/* ------------------------------------------------------------------ */
/* Causal attention mask: [seq_len, seq_len]                           */
/*   mask[i][j] = 0 if j <= i, -inf otherwise (additive mask)         */
/* ------------------------------------------------------------------ */
syn_status_t syn_causal_mask(syn_tensor_t **out, size_t seq_len);

/* ------------------------------------------------------------------ */
/* RMS normalization (trailing dims)                                    */
/*   input:  N-D tensor                                                */
/*   gamma: 1-D tensor with size = product of trailing dims            */
/*   num_norm_dims: number of trailing dimensions to normalize over    */
/* ------------------------------------------------------------------ */
syn_status_t syn_rmsnorm_forward(syn_tensor_t **out,
                                  syn_tensor_t *input,
                                  syn_tensor_t *gamma,
                                  size_t num_norm_dims,
                                  float eps);

/* ------------------------------------------------------------------ */
/* SiLU activation on flat f32 arrays                                  */
/*   dst[i] = src[i] / (1 + exp(-src[i]))                             */
/* ------------------------------------------------------------------ */
syn_status_t syn_silu(float *dst, const float *src, size_t len);

/* ------------------------------------------------------------------ */
/* Fused SwiGLU on flat f32 arrays                                     */
/*   dst[i] = silu(gate[i]) * up[i]                                    */
/* ------------------------------------------------------------------ */
syn_status_t syn_swiglu(float *dst, const float *gate, const float *up,
                         size_t len);

/* ------------------------------------------------------------------ */
/* Per-channel INT8 quantization                                       */
/*   data: [channels × channel_size] row-major f32 input               */
/*   out:  [channels × channel_size] int8 output                       */
/*   scales: [channels] per-channel scale factors (written)             */
/* ------------------------------------------------------------------ */
syn_status_t syn_quantize_per_channel_int8(const float *data,
                                            size_t channels,
                                            size_t channel_size,
                                            int8_t *out,
                                            float *scales);

/* ------------------------------------------------------------------ */
/* Per-channel INT8 dequantization                                     */
/*   data: [channels × channel_size] int8 input                        */
/*   out:  [channels × channel_size] f32 output                        */
/*   scales: [channels] per-channel scale factors (read)                */
/* ------------------------------------------------------------------ */
syn_status_t syn_dequantize_per_channel_int8(const int8_t *data,
                                              size_t channels,
                                              size_t channel_size,
                                              float *out,
                                              const float *scales);

/* ------------------------------------------------------------------ */
/* INT8 quantized GEMM: C = diag(scales_a) * (A_i8 * B_i8) * diag(sb) */
/*   C must be pre-allocated [m × n], zeroed internally.               */
/* ------------------------------------------------------------------ */
syn_status_t syn_qgemm_int8(size_t m, size_t n, size_t k,
                              const int8_t *a, size_t lda,
                              const int8_t *b, size_t ldb,
                              float *c, size_t ldc,
                              const float *scales_a,
                              const float *scales_b);

/* ------------------------------------------------------------------ */
/* KV-Cache for autoregressive inference                               */
/*   Manages pre-allocated K/V buffers [max_seq, n_kv_heads, head_dim] */
/* ------------------------------------------------------------------ */

/** Create a KV-cache with pre-allocated buffers. */
syn_status_t syn_kvcache_create(size_t max_seq, size_t n_kv_heads,
                                 size_t head_dim, syn_kvcache_t **out);

/** Destroy the KV-cache and free backing buffers. */
syn_status_t syn_kvcache_destroy(syn_kvcache_t *cache);

/** Append K/V vectors for one token.  stride = n_kv_heads * head_dim. */
syn_status_t syn_kvcache_append(syn_kvcache_t *cache,
                                 const float *k_token,
                                 const float *v_token,
                                 size_t stride);

/** Get zero-copy views into populated region. Any out-pointer may be NULL. */
syn_status_t syn_kvcache_slice(syn_kvcache_t *cache,
                                const float **k_out,
                                const float **v_out,
                                size_t *seq_len_out);

/** Reset position counter to 0.  No deallocation. */
syn_status_t syn_kvcache_reset(syn_kvcache_t *cache);

#ifdef __cplusplus
}
#endif

#endif /* SYNAPSE_H */
