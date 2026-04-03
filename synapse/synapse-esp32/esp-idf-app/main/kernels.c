/*
 * kernels.c -- Math operations, quantization, attention kernels, allocators.
 *
 * Extracted from app_main.c -- preserves ALL optimized code:
 *   - Exp LUT (s_exp_lut, exp_lut_init, exp_lut_scalar) for softmax
 *   - Reciprocal multiply (inv_sum) in softmax instead of division
 *   - Aligned PIE alloc with PSRAM fallback in q4linear_forward_into
 *   - Nested (token,col) fused bias+GELU / bias+residual loops
 */

#include "kernels.h"
#include "dual_core.h"
#include "pie_gemv.h"

#include <alloca.h>
#include <math.h>

#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

#include "esp_heap_caps.h"
#include "esp_log.h"

static const char *TAG = "kernels";

/* ================================================================== */
/* Allocators                                                          */
/* ================================================================== */

void *lewm_alloc(size_t size, uint32_t caps) {
    if (size == 0) return NULL;
    void *ptr = heap_caps_malloc(size, caps);
    if (!ptr) {
        ptr = malloc(size);
        if (ptr) ESP_LOGW(TAG, "Fallback to default heap for %zu bytes", size);
    }
    return ptr;
}

void *lewm_calloc(size_t count, size_t size, uint32_t caps) {
    if (count == 0 || size == 0) return NULL;
    void *ptr = heap_caps_calloc(count, size, caps);
    if (!ptr) {
        ptr = calloc(count, size);
        if (ptr) ESP_LOGW(TAG, "Fallback to default heap for %zu * %zu bytes", count, size);
    }
    return ptr;
}

void *lewm_alloc_aligned(size_t alignment, size_t size, uint32_t caps) {
    if (size == 0) return NULL;
    void *ptr = heap_caps_aligned_alloc(alignment, size, caps);
    if (!ptr) {
        /* PSRAM fallback */
        ptr = heap_caps_aligned_alloc(alignment, size, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
        if (ptr) ESP_LOGW(TAG, "Aligned alloc fallback to PSRAM for %zu bytes", size);
    }
    return ptr;
}

/* ================================================================== */
/* Float buffer helpers                                                */
/* ================================================================== */

bool alloc_float_buffer(FloatBuffer *buffer, size_t len, uint32_t caps) {
    buffer->len = len;
    if (len == 0) {
        buffer->data = NULL;
        return true;
    }
    buffer->data = (float *)lewm_alloc(len * sizeof(float), caps);
    return buffer->data != NULL;
}

bool copy_f32_payload(FloatBuffer *buffer, const uint8_t *src, size_t len, uint32_t caps) {
    if (!alloc_float_buffer(buffer, len, caps)) {
        return false;
    }
    for (size_t i = 0; i < len; ++i) {
        uint32_t raw = ((uint32_t)src[i * 4]) |
                       ((uint32_t)src[i * 4 + 1] << 8) |
                       ((uint32_t)src[i * 4 + 2] << 16) |
                       ((uint32_t)src[i * 4 + 3] << 24);
        memcpy(&buffer->data[i], &raw, sizeof(float));
    }
    return true;
}

/* ================================================================== */
/* Cosine similarity                                                   */
/* ================================================================== */

float cosine_similarity(const float *a, const float *b, size_t len) {
    float dot = 0.0f, na = 0.0f, nb = 0.0f;
    for (size_t i = 0; i < len; i++) {
        dot += a[i] * b[i];
        na += a[i] * a[i];
        nb += b[i] * b[i];
    }
    float denom = sqrtf(na) * sqrtf(nb);
    return denom > 0.0f ? dot / denom : 0.0f;
}

/* ================================================================== */
/* Matrix / vector operations                                          */
/* ================================================================== */

void matmul_t_into(
    const float *a,
    const float *b,
    size_t m,
    size_t k,
    size_t n,
    float *out
) {
    for (size_t i = 0; i < m; ++i) {
        for (size_t j = 0; j < n; ++j) {
            float sum = 0.0f;
            for (size_t p = 0; p < k; ++p) {
                sum += a[i * k + p] * b[j * k + p];
            }
            out[i * n + j] = sum;
        }
    }
}

void add_bias_inplace(float *x, const float *bias, size_t m, size_t n) {
    if (bias == NULL) {
        return;
    }
    for (size_t row = 0; row < m; ++row) {
        for (size_t col = 0; col < n; ++col) {
            x[row * n + col] += bias[col];
        }
    }
}

void layernorm_into(
    const float *x,
    const float *weight,
    size_t rows,
    size_t hidden,
    float *out
) {
    for (size_t row = 0; row < rows; ++row) {
        const float *src = x + row * hidden;
        float mean = 0.0f;
        for (size_t i = 0; i < hidden; ++i) {
            mean += src[i];
        }
        mean /= (float)hidden;

        float var = 0.0f;
        for (size_t i = 0; i < hidden; ++i) {
            float diff = src[i] - mean;
            var += diff * diff;
        }
        var /= (float)hidden;
        float scale = 1.0f / sqrtf(var + 1e-6f);

        for (size_t i = 0; i < hidden; ++i) {
            out[row * hidden + i] = (src[i] - mean) * scale * weight[i];
        }
    }
}

/* ================================================================== */
/* GELU LUT                                                            */
/* ================================================================== */

#define GELU_LUT_SIZE 1024
#define GELU_LUT_MIN  (-8.0f)
#define GELU_LUT_MAX  (8.0f)
#define GELU_LUT_STEP ((GELU_LUT_MAX - GELU_LUT_MIN) / (float)(GELU_LUT_SIZE - 1))

static float s_gelu_lut[GELU_LUT_SIZE];
static bool s_gelu_lut_ready = false;

void gelu_lut_init(void) {
    if (s_gelu_lut_ready) return;
    const float pi = 3.14159265358979323846f;
    const float alpha = sqrtf(2.0f / pi);
    for (int i = 0; i < GELU_LUT_SIZE; i++) {
        float x = GELU_LUT_MIN + (float)i * GELU_LUT_STEP;
        s_gelu_lut[i] = 0.5f * x * (1.0f + tanhf(alpha * (x + 0.044715f * x * x * x)));
    }
    s_gelu_lut_ready = true;
}

float gelu_scalar(float x) {
    if (x <= GELU_LUT_MIN) return 0.0f;
    if (x >= GELU_LUT_MAX) return x;

    float idx_f = (x - GELU_LUT_MIN) / GELU_LUT_STEP;
    int idx = (int)idx_f;
    if (idx >= GELU_LUT_SIZE - 1) return s_gelu_lut[GELU_LUT_SIZE - 1];
    float frac = idx_f - (float)idx;
    return s_gelu_lut[idx] + frac * (s_gelu_lut[idx + 1] - s_gelu_lut[idx]);
}

/* ================================================================== */
/* Exp LUT for softmax (OPTIMIZED: avoids expf() per element)          */
/* Range [-8, 8], 256 entries, linear interpolation.                   */
/* ================================================================== */

#define EXP_LUT_SIZE  256
#define EXP_LUT_MIN   (-8.0f)
#define EXP_LUT_MAX   (8.0f)
#define EXP_LUT_STEP  ((EXP_LUT_MAX - EXP_LUT_MIN) / (float)(EXP_LUT_SIZE - 1))

static float s_exp_lut[EXP_LUT_SIZE];
static bool s_exp_lut_ready = false;

void exp_lut_init(void) {
    if (s_exp_lut_ready) return;
    for (int i = 0; i < EXP_LUT_SIZE; i++) {
        float x = EXP_LUT_MIN + (float)i * EXP_LUT_STEP;
        s_exp_lut[i] = expf(x);
    }
    s_exp_lut_ready = true;
}

/* Clamped exp lookup with linear interpolation: x clamped to [-8, 8] */
float exp_lut_scalar(float x) {
    if (x <= EXP_LUT_MIN) return 0.0f;
    if (x >= EXP_LUT_MAX) return s_exp_lut[EXP_LUT_SIZE - 1];
    float idx_f = (x - EXP_LUT_MIN) / EXP_LUT_STEP;
    int idx = (int)idx_f;
    if (idx >= EXP_LUT_SIZE - 1) return s_exp_lut[EXP_LUT_SIZE - 1];
    float frac = idx_f - (float)idx;
    return s_exp_lut[idx] + frac * (s_exp_lut[idx + 1] - s_exp_lut[idx]);
}

/* ================================================================== */
/* Softmax / normalization                                             */
/* OPTIMIZED: uses exp_lut_scalar + reciprocal multiply (inv_sum)      */
/* ================================================================== */

void softmax_inplace(float *x, size_t len) {
    float max_value = -INFINITY;
    for (size_t i = 0; i < len; ++i) {
        if (x[i] > max_value) {
            max_value = x[i];
        }
    }

    float sum = 0.0f;
    for (size_t i = 0; i < len; ++i) {
        x[i] = exp_lut_scalar(x[i] - max_value);
        sum += x[i];
    }

    if (sum > 0.0f) {
        float inv_sum = 1.0f / sum;
        for (size_t i = 0; i < len; ++i) {
            x[i] *= inv_sum;
        }
    }
}

void l1_normalize_inplace(float *x, size_t len) {
    float abs_sum = 0.0f;
    for (size_t i = 0; i < len; ++i) abs_sum += fabsf(x[i]);
    if (abs_sum > 1e-12f) {
        float inv_sum = 1.0f / abs_sum;
        for (size_t i = 0; i < len; ++i) x[i] *= inv_sum;
    }
}

/* ELU+1 feature map for kernel-trick linear attention */
float elu_plus1(float x) {
    return x >= 0.0f ? x + 1.0f : exp_lut_scalar(x);
}

/* ================================================================== */
/* Quantization                                                        */
/* ================================================================== */

void quantize_row_int8(
    const float *input,
    size_t len,
    int8_t *out,
    float *scale_out
) {
    float max_abs = 0.0f;
    for (size_t i = 0; i < len; ++i) {
        float abs_value = fabsf(input[i]);
        if (abs_value > max_abs) {
            max_abs = abs_value;
        }
    }
    float scale = max_abs == 0.0f ? 1.0f : max_abs / 127.0f;
    float inv_scale = 1.0f / scale;
    for (size_t i = 0; i < len; ++i) {
        float value = roundf(input[i] * inv_scale);
        if (value > 127.0f) {
            value = 127.0f;
        } else if (value < -128.0f) {
            value = -128.0f;
        }
        out[i] = (int8_t)value;
    }
    *scale_out = scale;
}

/* ================================================================== */
/* Q4 linear forward                                                   */
/* OPTIMIZED: aligned PIE alloc with PSRAM fallback                    */
/* ================================================================== */

static bool bitmap_get(const uint8_t *bitmap, size_t bit_index) {
    return ((bitmap[bit_index / 8] >> (bit_index % 8)) & 1U) != 0;
}

static float read_f32_le(const uint8_t *ptr) {
    float value = 0.0f;
    uint32_t raw = ((uint32_t)ptr[0]) |
                   ((uint32_t)ptr[1] << 8) |
                   ((uint32_t)ptr[2] << 16) |
                   ((uint32_t)ptr[3] << 24);
    memcpy(&value, &raw, sizeof(value));
    return value;
}

void q4linear_forward_into(
    const Q4LinearRef *linear,
    const float *x,
    size_t m,
    float *out
) {
    memset(out, 0, m * linear->out_features * sizeof(float));

    /* Quantize each input row to INT8 once upfront.
     * Aligned to 16 bytes for PIE SIMD loads (esp.vld.128.ip).
     * Try internal SRAM first, fall back to PSRAM. */
    size_t in_padded = (linear->in_features + 31U) & ~31U;
    int8_t *x_i8 = (int8_t *)heap_caps_aligned_alloc(
        16, m * in_padded, MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);
    if (!x_i8) {
        x_i8 = (int8_t *)heap_caps_aligned_alloc(
            16, m * in_padded, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
    }
    float *x_scales = (float *)malloc(m * sizeof(float));

    if (x_i8 && x_scales) {
        for (size_t batch = 0; batch < m; ++batch) {
            quantize_row_int8(
                x + batch * linear->in_features,
                linear->in_features,
                x_i8 + batch * in_padded,
                &x_scales[batch]);
            /* Zero-pad */
            memset(x_i8 + batch * in_padded + linear->in_features, 0,
                   in_padded - linear->in_features);
        }
    }

    for (size_t row = 0; row < linear->out_features; ++row) {
        uint32_t nz_index = linear->row_nz_starts[row];

        for (size_t block = 0; block < linear->blocks_per_row; ++block) {
            size_t global_block = row * linear->blocks_per_row + block;
            if (!bitmap_get(linear->bitmap, global_block)) {
                continue;
            }

            const uint8_t *block_ptr = linear->blocks_data + (size_t)nz_index * 20U;
            float w_scale = read_f32_le(block_ptr);
            const uint8_t *nibbles = block_ptr + 4U;
            nz_index++;

            if (x_i8 && x_scales) {
                /* PIE path: unpack Q4 -> INT8, dot with pre-quantized input */
                int8_t w_i8[32] __attribute__((aligned(16)));
                for (size_t i = 0; i < 16; i++) {
                    uint8_t packed = nibbles[i];
                    w_i8[i * 2]     = (int8_t)((packed & 0x0FU) - 8U);
                    w_i8[i * 2 + 1] = (int8_t)((packed >> 4U) - 8U);
                }

                size_t col_start = block * 32U;
                for (size_t batch = 0; batch < m; ++batch) {
                    int32_t dot = pie_dot_int8(
                        w_i8, x_i8 + batch * in_padded + col_start, 32);
                    out[batch * linear->out_features + row] +=
                        (float)dot * x_scales[batch] * w_scale;
                }
            } else {
                /* Scalar fallback */
                for (size_t nibble = 0; nibble < 16U; ++nibble) {
                    uint8_t packed = nibbles[nibble];
                    size_t col0 = block * 32U + nibble * 2U;
                    size_t col1 = col0 + 1U;
                    float v0 = ((float)((int)(packed & 0x0FU) - 8)) * w_scale;
                    float v1 = ((float)((int)(packed >> 4U) - 8)) * w_scale;
                    if (col0 < linear->in_features) {
                        for (size_t batch = 0; batch < m; ++batch) {
                            out[batch * linear->out_features + row] +=
                                x[batch * linear->in_features + col0] * v0;
                        }
                    }
                    if (col1 < linear->in_features) {
                        for (size_t batch = 0; batch < m; ++batch) {
                            out[batch * linear->out_features + row] +=
                                x[batch * linear->in_features + col1] * v1;
                        }
                    }
                }
            }
        }

        if ((row & 0xFFU) == 0xFFU) {
            vTaskDelay(pdMS_TO_TICKS(1));
        }
    }

    if (x_i8) free(x_i8);
    if (x_scales) free(x_scales);
}

/* ================================================================== */
/* INT8 linear forward (pre-quantized input path)                      */
/* ================================================================== */

void int8linear_forward_prequant(
    const Int8LinearRef *linear,
    const int8_t *all_i8,
    const float *scales,
    size_t m,
    size_t in_pad,
    float *out
) {
    size_t out_f = linear->out_features;
    memset(out, 0, m * out_f * sizeof(float));

    if (linear->weights_t != NULL && s_core1_start && out_f >= 64) {
        size_t mid_j = out_f / 2;
        static GemvWorkItem s_gemv_pq_c1;
        s_gemv_pq_c1 = (GemvWorkItem){
            .all_i8 = all_i8, .weights_t = linear->weights_t,
            .scales = scales, .w_scales = linear->scales.data,
            .out = out, .m = m, .in_pad = in_pad, .out_f = out_f,
            .j_start = mid_j, .j_end = out_f,
        };
        core1_dispatch(gemv_compute_range, &s_gemv_pq_c1);

        GemvWorkItem c0 = {
            .all_i8 = all_i8, .weights_t = linear->weights_t,
            .scales = scales, .w_scales = linear->scales.data,
            .out = out, .m = m, .in_pad = in_pad, .out_f = out_f,
            .j_start = 0, .j_end = mid_j,
        };
        gemv_compute_range(&c0);
        core1_wait();
    } else if (linear->weights_t != NULL) {
        for (size_t j = 0; j < out_f; ++j) {
            const int8_t *wj = linear->weights_t + j * in_pad;
            float ws = linear->scales.data[j];
            for (size_t row = 0; row < m; ++row) {
                int32_t dot = pie_dot_int8(all_i8 + row * in_pad, wj, in_pad);
                out[row * out_f + j] = (float)dot * scales[row] * ws;
            }
        }
    }
}

/* ================================================================== */
/* INT8 linear forward (float input)                                   */
/* ================================================================== */

void int8linear_forward_into(
    const Int8LinearRef *linear,
    const float *x,
    size_t m,
    int8_t *row_quant,
    float *out
) {
    size_t in_f = linear->in_features;
    size_t out_f = linear->out_features;
    size_t in_pad = linear->in_features_padded;

    memset(out, 0, m * out_f * sizeof(float));

    if (linear->weights_t != NULL && m > 1) {
        /*
         * Tiled PIE path: pre-quantize ALL input rows, then iterate
         * weights outer / rows inner. Each weight row (in_pad bytes)
         * is loaded from PSRAM once and reused for all m input rows.
         */
        int8_t *all_i8 = (int8_t *)heap_caps_aligned_alloc(
            16, m * in_pad, MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);
        float *scales = (float *)malloc(m * sizeof(float));

        if (all_i8 && scales) {
            /* Pre-quantize all input rows */
            for (size_t row = 0; row < m; ++row) {
                quantize_row_int8(x + row * in_f, in_f,
                                  all_i8 + row * in_pad, &scales[row]);
                if (in_pad > in_f) {
                    memset(all_i8 + row * in_pad + in_f, 0, in_pad - in_f);
                }
            }

            /* Dual-core GEMV: split output features across cores */
            if (s_core1_start && out_f >= 64) {
                size_t mid_j = out_f / 2;
                static GemvWorkItem s_gemv_c1;
                s_gemv_c1 = (GemvWorkItem){
                    .all_i8 = all_i8, .weights_t = linear->weights_t,
                    .scales = scales, .w_scales = linear->scales.data,
                    .out = out, .m = m, .in_pad = in_pad, .out_f = out_f,
                    .j_start = mid_j, .j_end = out_f,
                };
                core1_dispatch(gemv_compute_range, &s_gemv_c1);

                /* Core 0: first half */
                GemvWorkItem c0_work = {
                    .all_i8 = all_i8, .weights_t = linear->weights_t,
                    .scales = scales, .w_scales = linear->scales.data,
                    .out = out, .m = m, .in_pad = in_pad, .out_f = out_f,
                    .j_start = 0, .j_end = mid_j,
                };
                gemv_compute_range(&c0_work);
                core1_wait();
            } else {
                /* Single-core: weights outer, rows inner */
                for (size_t j = 0; j < out_f; ++j) {
                    const int8_t *wj = linear->weights_t + j * in_pad;
                    float w_scale = linear->scales.data[j];
                    for (size_t row = 0; row < m; ++row) {
                        int32_t dot = pie_dot_int8(
                            all_i8 + row * in_pad, wj, in_pad);
                        out[row * out_f + j] =
                            (float)dot * scales[row] * w_scale;
                    }
                }
            }

            free(all_i8);
            free(scales);
            return;
        }
        /* Fall through to per-row path if alloc failed */
        free(all_i8);
        free(scales);
    }

    /* Per-row path (m==1 or no transposed weights or alloc failed) */
    for (size_t row = 0; row < m; ++row) {
        float row_scale = 1.0f;
        quantize_row_int8(x + row * in_f, in_f, row_quant, &row_scale);

        if (linear->weights_t != NULL) {
            int32_t acc_buf[768];
            pie_int8_gemv(row_quant, linear->weights_t, out_f, in_pad, acc_buf);
            for (size_t j = 0; j < out_f; ++j) {
                out[row * out_f + j] =
                    (float)acc_buf[j] * row_scale * linear->scales.data[j];
            }
        } else {
            for (size_t j = 0; j < out_f; ++j) {
                int32_t acc = 0;
                for (size_t k = 0; k < in_f; ++k) {
                    acc += (int32_t)row_quant[k] *
                           (int32_t)linear->weights_data[k * out_f + j];
                }
                out[row * out_f + j] =
                    (float)acc * row_scale * linear->scales.data[j];
            }
        }
    }
}

/* ================================================================== */
/* Attention: bidirectional from interleaved QKV (Q4 predictor)        */
/* ================================================================== */

void bidirectional_attention_from_qkv(
    const float *qkv,
    size_t seq_len,
    size_t num_heads,
    size_t head_dim,
    float *out
) {
    size_t inner_dim = num_heads * head_dim;
    float *scores = (float *)malloc(seq_len * sizeof(float));
    if (!scores) {
        memset(out, 0, seq_len * inner_dim * sizeof(float));
        return;
    }

    memset(out, 0, seq_len * inner_dim * sizeof(float));
    float inv_sqrt_hd = 1.0f / sqrtf((float)head_dim);

    for (size_t head = 0; head < num_heads; ++head) {
        for (size_t q_token = 0; q_token < seq_len; ++q_token) {
            for (size_t k_token = 0; k_token < seq_len; ++k_token) {
                float dot = 0.0f;
                for (size_t d = 0; d < head_dim; ++d) {
                    size_t q_idx = q_token * 3U * inner_dim + head * head_dim + d;
                    size_t k_idx = k_token * 3U * inner_dim + inner_dim + head * head_dim + d;
                    dot += qkv[q_idx] * qkv[k_idx];
                }
                scores[k_token] = dot * inv_sqrt_hd;
            }

            softmax_inplace(scores, seq_len);

            for (size_t d = 0; d < head_dim; ++d) {
                float value = 0.0f;
                for (size_t k_token = 0; k_token < seq_len; ++k_token) {
                    size_t v_idx =
                        k_token * 3U * inner_dim + 2U * inner_dim + head * head_dim + d;
                    value += scores[k_token] * qkv[v_idx];
                }
                out[q_token * inner_dim + head * head_dim + d] = value;
            }
        }
    }
    free(scores);
}

/* ================================================================== */
/* Attention: separate Q/K/V with INT8/PIE acceleration (encoder)      */
/* ================================================================== */

void bidirectional_attention_separate(
    const float *q,
    const float *k,
    const float *v,
    size_t seq_len,
    size_t num_heads,
    size_t head_dim,
    float *scores,
    float *out,
    bool linear_attn
) {
    size_t inner_dim = num_heads * head_dim;
    size_t head_dim_padded = (head_dim + 15U) & ~15U; /* round up to 16 for PIE */
    float inv_sqrt_hd = 1.0f / sqrtf((float)head_dim);

    memset(out, 0, seq_len * inner_dim * sizeof(float));

    /* Pre-quantize Q and K to INT8 for PIE-accelerated dot products.
     * Layout: [seq_len][num_heads][head_dim_padded] INT8 + per-row scales. */
    size_t row_bytes = num_heads * head_dim_padded;
    int8_t *q_i8 = (int8_t *)heap_caps_aligned_alloc(
        16, seq_len * row_bytes, MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);
    int8_t *k_i8 = (int8_t *)heap_caps_aligned_alloc(
        16, seq_len * row_bytes, MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);
    float *q_scales = (float *)malloc(seq_len * num_heads * sizeof(float));
    float *k_scales = (float *)malloc(seq_len * num_heads * sizeof(float));

    bool use_pie = q_i8 && k_i8 && q_scales && k_scales;

    if (use_pie) {
        /* Quantize each (token, head) slice to INT8 */
        for (size_t token = 0; token < seq_len; ++token) {
            for (size_t head = 0; head < num_heads; ++head) {
                const float *q_src = q + token * inner_dim + head * head_dim;
                const float *k_src = k + token * inner_dim + head * head_dim;
                int8_t *q_dst = q_i8 + token * row_bytes + head * head_dim_padded;
                int8_t *k_dst = k_i8 + token * row_bytes + head * head_dim_padded;

                quantize_row_int8(q_src, head_dim, q_dst,
                                  &q_scales[token * num_heads + head]);
                quantize_row_int8(k_src, head_dim, k_dst,
                                  &k_scales[token * num_heads + head]);
                /* Zero pad */
                memset(q_dst + head_dim, 0, head_dim_padded - head_dim);
                memset(k_dst + head_dim, 0, head_dim_padded - head_dim);
            }
        }

        /* PIE attention: QK^T via INT8 dot products.
         * Split query tokens across Core 0 and Core 1 for parallelism. */
        if (s_core1_start && seq_len > 16) {
            size_t mid = seq_len / 2;
            /* Core 1 scores buffer (separate from core 0's) */
            float *scores1 = (float *)malloc(seq_len * sizeof(float));
            if (scores1) {
                /* Set up Core 1 work: second half of q_tokens */
                s_core1_attn_work = (AttnWorkItem){
                    .q_i8 = q_i8, .k_i8 = k_i8,
                    .q_scales = q_scales, .k_scales = k_scales,
                    .v = v, .out = out,
                    .inv_sqrt_hd = inv_sqrt_hd,
                    .seq_len = seq_len, .num_heads = num_heads,
                    .head_dim = head_dim, .head_dim_padded = head_dim_padded,
                    .inner_dim = inner_dim, .row_bytes = row_bytes,
                    .q_start = mid, .q_end = seq_len,
                    .scores = scores1,
                    .linear_attn = linear_attn,
                };
                core1_dispatch(attn_compute_range_wrapper,
                               (void *)&s_core1_attn_work);

                /* Core 0: first half of q_tokens */
                AttnWorkItem w0 = {
                    .q_i8 = q_i8, .k_i8 = k_i8,
                    .q_scales = q_scales, .k_scales = k_scales,
                    .v = v, .out = out,
                    .inv_sqrt_hd = inv_sqrt_hd,
                    .seq_len = seq_len, .num_heads = num_heads,
                    .head_dim = head_dim, .head_dim_padded = head_dim_padded,
                    .inner_dim = inner_dim, .row_bytes = row_bytes,
                    .q_start = 0, .q_end = mid,
                    .scores = scores,
                    .linear_attn = linear_attn,
                };
                attn_compute_range(&w0);

                /* Wait for Core 1 */
                core1_wait();
                free(scores1);
            } else {
                /* Fallback: single-core */
                AttnWorkItem w = {
                    .q_i8 = q_i8, .k_i8 = k_i8,
                    .q_scales = q_scales, .k_scales = k_scales,
                    .v = v, .out = out,
                    .inv_sqrt_hd = inv_sqrt_hd,
                    .seq_len = seq_len, .num_heads = num_heads,
                    .head_dim = head_dim, .head_dim_padded = head_dim_padded,
                    .inner_dim = inner_dim, .row_bytes = row_bytes,
                    .q_start = 0, .q_end = seq_len,
                    .scores = scores,
                    .linear_attn = linear_attn,
                };
                attn_compute_range(&w);
            }
        } else {
            /* Small seq_len: single-core is fine */
            AttnWorkItem w = {
                .q_i8 = q_i8, .k_i8 = k_i8,
                .q_scales = q_scales, .k_scales = k_scales,
                .v = v, .out = out,
                .inv_sqrt_hd = inv_sqrt_hd,
                .seq_len = seq_len, .num_heads = num_heads,
                .head_dim = head_dim, .head_dim_padded = head_dim_padded,
                .inner_dim = inner_dim, .row_bytes = row_bytes,
                .q_start = 0, .q_end = seq_len,
                .scores = scores,
                .linear_attn = linear_attn,
            };
            attn_compute_range(&w);
        }
    } else {
        /* Scalar fallback */
        for (size_t head = 0; head < num_heads; ++head) {
            for (size_t q_token = 0; q_token < seq_len; ++q_token) {
                for (size_t k_token = 0; k_token < seq_len; ++k_token) {
                    float dot = 0.0f;
                    for (size_t d = 0; d < head_dim; ++d) {
                        dot += q[q_token * inner_dim + head * head_dim + d] *
                               k[k_token * inner_dim + head * head_dim + d];
                    }
                    scores[k_token] = dot * inv_sqrt_hd;
                }
                if (linear_attn) {
                    l1_normalize_inplace(scores, seq_len);
                } else {
                    softmax_inplace(scores, seq_len);
                }
                for (size_t d = 0; d < head_dim; ++d) {
                    float value = 0.0f;
                    for (size_t k_token = 0; k_token < seq_len; ++k_token) {
                        value += scores[k_token] *
                                 v[k_token * inner_dim + head * head_dim + d];
                    }
                    out[q_token * inner_dim + head * head_dim + d] = value;
                }
            }
        }
    }

    free(q_i8);
    free(k_i8);
    free(q_scales);
    free(k_scales);
}

/* ================================================================== */
/* Kernel-trick linear attention: O(nd^2) instead of O(n^2 d)         */
/* ================================================================== */

void linear_attention_kernel_trick(
    const float *q, const float *k, const float *v,
    size_t seq_len, size_t num_heads, size_t head_dim,
    float *out
) {
    size_t inner_dim = num_heads * head_dim;
    /* Heap-allocate KV matrix (too large for ESP32 stack) */
    float *kv_matrix = (float *)heap_caps_calloc(head_dim * head_dim, sizeof(float), MALLOC_CAP_INTERNAL);
    float *k_sum = (float *)calloc(head_dim, sizeof(float));
    float *phi_q = (float *)calloc(head_dim, sizeof(float));
    if (!kv_matrix || !k_sum || !phi_q) {
        free(kv_matrix); free(k_sum); free(phi_q);
        /* Fallback: zero output */
        memset(out, 0, seq_len * inner_dim * sizeof(float));
        return;
    }

    for (size_t head = 0; head < num_heads; ++head) {
        size_t d = head_dim;

        /* 1. Compute KV = phi(K)^T @ V and k_sum = sum phi(K) */
        memset(kv_matrix, 0, d * d * sizeof(float));
        memset(k_sum, 0, d * sizeof(float));

        for (size_t kt = 0; kt < seq_len; ++kt) {
            const float *k_row = k + kt * inner_dim + head * d;
            const float *v_row = v + kt * inner_dim + head * d;

            for (size_t i = 0; i < d; ++i) {
                float phi_k = elu_plus1(k_row[i]);
                k_sum[i] += phi_k;
                for (size_t j = 0; j < d; ++j) {
                    kv_matrix[i * d + j] += phi_k * v_row[j];
                }
            }
        }

        /* 2. For each query: out = phi(Q) @ KV / (phi(Q) . k_sum) */
        for (size_t qt = 0; qt < seq_len; ++qt) {
            const float *q_row = q + qt * inner_dim + head * d;
            float *out_row = out + qt * inner_dim + head * d;

            float z = 0.0f;
            for (size_t i = 0; i < d; ++i) {
                phi_q[i] = elu_plus1(q_row[i]);
                z += phi_q[i] * k_sum[i];
            }
            float inv_z = (z > 1e-12f) ? 1.0f / z : 0.0f;

            for (size_t j = 0; j < d; ++j) {
                float sum = 0.0f;
                for (size_t i = 0; i < d; ++i) {
                    sum += phi_q[i] * kv_matrix[i * d + j];
                }
                out_row[j] = sum * inv_z;
            }
        }
    }
    free(kv_matrix);
    free(k_sum);
    free(phi_q);
}

/* ================================================================== */
/* Logging helpers                                                     */
/* ================================================================== */

void log_vector_preview(const char *label, const float *values, size_t len) {
    char line[256];
    int written = snprintf(line, sizeof(line), "%s len=%lu first=[", label, (unsigned long)len);
    size_t preview = len < 8U ? len : 8U;
    for (size_t i = 0; i < preview && written > 0 && (size_t)written < sizeof(line); ++i) {
        written += snprintf(
            line + written,
            sizeof(line) - (size_t)written,
            "%s%.6f",
            i == 0U ? "" : ", ",
            values[i]
        );
    }
    snprintf(line + written, sizeof(line) - (size_t)written, "]");
    ESP_LOGI(TAG, "%s", line);
}

void log_vector_stats(const char *label, const float *values, size_t len) {
    float sum = 0.0f;
    float sq_sum = 0.0f;
    float max_abs = 0.0f;
    for (size_t i = 0; i < len; ++i) {
        float value = values[i];
        sum += value;
        sq_sum += value * value;
        float abs_value = fabsf(value);
        if (abs_value > max_abs) {
            max_abs = abs_value;
        }
    }
    ESP_LOGI(
        TAG,
        "%s stats: sum=%.6f l2=%.6f max_abs=%.6f",
        label,
        sum,
        sqrtf(sq_sum),
        max_abs
    );
}
