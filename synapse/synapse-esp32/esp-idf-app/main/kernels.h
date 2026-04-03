/*
 * kernels.h -- Math operations: softmax, layernorm, GELU LUT, quantize,
 *              matmul, fused ops, allocators, attention kernels.
 */
#pragma once

#include "lewm_types.h"

/* ------------------------------------------------------------------ */
/* Allocators (PSRAM fallback + logging)                               */
/* ------------------------------------------------------------------ */

void *lewm_alloc(size_t size, uint32_t caps);
void *lewm_calloc(size_t count, size_t size, uint32_t caps);
void *lewm_alloc_aligned(size_t alignment, size_t size, uint32_t caps);

/* ------------------------------------------------------------------ */
/* Float buffer helpers                                                */
/* ------------------------------------------------------------------ */

bool alloc_float_buffer(FloatBuffer *buffer, size_t len, uint32_t caps);
bool copy_f32_payload(FloatBuffer *buffer, const uint8_t *src, size_t len, uint32_t caps);

/* ------------------------------------------------------------------ */
/* Cosine similarity                                                   */
/* ------------------------------------------------------------------ */

float cosine_similarity(const float *a, const float *b, size_t len);

/* ------------------------------------------------------------------ */
/* Matrix / vector operations                                          */
/* ------------------------------------------------------------------ */

void matmul_t_into(const float *a, const float *b, size_t m, size_t k, size_t n, float *out);
void add_bias_inplace(float *x, const float *bias, size_t m, size_t n);
void layernorm_into(const float *x, const float *weight, size_t rows, size_t hidden, float *out);

/* ------------------------------------------------------------------ */
/* GELU LUT                                                            */
/* ------------------------------------------------------------------ */

void gelu_lut_init(void);
float gelu_scalar(float x);

/* ------------------------------------------------------------------ */
/* Exp LUT (optimized softmax)                                         */
/* ------------------------------------------------------------------ */

void exp_lut_init(void);
float exp_lut_scalar(float x);

/* ------------------------------------------------------------------ */
/* Softmax / normalization                                             */
/* ------------------------------------------------------------------ */

void softmax_inplace(float *x, size_t len);
void l1_normalize_inplace(float *x, size_t len);
float elu_plus1(float x);

/* ------------------------------------------------------------------ */
/* Quantization                                                        */
/* ------------------------------------------------------------------ */

void quantize_row_int8(const float *input, size_t len, int8_t *out, float *scale_out);

/* ------------------------------------------------------------------ */
/* Linear layer forward passes                                         */
/* ------------------------------------------------------------------ */

void q4linear_forward_into(const Q4LinearRef *linear, const float *x, size_t m, float *out);

void int8linear_forward_into(
    const Int8LinearRef *linear, const float *x, size_t m,
    int8_t *row_quant, float *out);

void int8linear_forward_prequant(
    const Int8LinearRef *linear, const int8_t *all_i8, const float *scales,
    size_t m, size_t in_pad, float *out);

/* ------------------------------------------------------------------ */
/* Attention kernels                                                   */
/* ------------------------------------------------------------------ */

void bidirectional_attention_from_qkv(
    const float *qkv, size_t seq_len, size_t num_heads, size_t head_dim, float *out);

void bidirectional_attention_separate(
    const float *q, const float *k, const float *v,
    size_t seq_len, size_t num_heads, size_t head_dim,
    float *scores, float *out, bool linear_attn);

void linear_attention_kernel_trick(
    const float *q, const float *k, const float *v,
    size_t seq_len, size_t num_heads, size_t head_dim, float *out);

/* ------------------------------------------------------------------ */
/* Logging helpers                                                     */
/* ------------------------------------------------------------------ */

void log_vector_preview(const char *label, const float *values, size_t len);
void log_vector_stats(const char *label, const float *values, size_t len);
