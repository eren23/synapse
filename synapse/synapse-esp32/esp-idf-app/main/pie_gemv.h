#pragma once

#include <stddef.h>
#include <stdint.h>

/**
 * INT8 dot product using PIE SIMD (16-wide esp.vmac.s8).
 * Both a and b must be contiguous int8_t arrays of length `len`.
 * `len` must be a multiple of 16 (caller pads if needed).
 * Returns the int32 accumulator result.
 */
int32_t pie_dot_int8(const int8_t *a, const int8_t *b, size_t len);

/**
 * INT8 GEMV: out[j] = sum_k(row_quant[k] * weights_t[j * in + k]) for each j.
 * `weights_t` is transposed: [out_features][in_features], contiguous in k.
 * `row_quant` is [in_features].
 * `out_i32` is [out_features] accumulator output.
 * `in_features` must be a multiple of 16.
 */
void pie_int8_gemv(
    const int8_t *row_quant,
    const int8_t *weights_t,
    size_t out_features,
    size_t in_features,
    int32_t *out_i32
);

/**
 * Transpose INT8 weight matrix from [in][out] to [out][in].
 * Caller must allocate `out_t` with size out_features * in_features_padded bytes.
 * `in_features_padded` is in_features rounded up to multiple of 16.
 */
void transpose_int8_weights(
    const int8_t *src,
    size_t in_features,
    size_t out_features,
    size_t in_features_padded,
    int8_t *out_t
);

/**
 * Q4 block dot product using PIE.
 *
 * Unpacks 16 nibble bytes into 32 INT8 weights [-8..7],
 * quantizes 32 input floats to INT8, and computes the dot product
 * via PIE SIMD (2 iterations of 16-wide MAC).
 *
 * Returns: dot_product * input_scale * weight_block_scale
 */
/**
 * Self-test: run PIE dot product against scalar reference on known vectors.
 * Returns 0 on success, -1 on mismatch.
 * Logs results via ESP_LOGI/ESP_LOGE.
 */
int pie_self_test(void);

float pie_q4_block_dot(
    const uint8_t *nibbles,    /* 16 bytes of packed Q4 nibbles */
    float weight_scale,        /* per-block Q4 scale */
    const float *input,        /* 32 input floats starting at block column offset */
    size_t valid_count         /* how many of the 32 elements are valid (<=in_features) */
);
