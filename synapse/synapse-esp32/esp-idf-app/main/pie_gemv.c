/*
 * PIE SIMD kernels for INT8 GEMV on ESP32-P4.
 *
 * Uses the PIE (Processor Instruction Extensions) 128-bit SIMD unit:
 *   - QR0-QR7: 128-bit vector registers (16x int8)
 *   - XACC: 40-bit scalar accumulator
 *   - esp.vld.128.ip: load 16x int8
 *   - esp.vmulas.s8.xacc: 16-wide signed int8 multiply-accumulate, sum into XACC
 *   - esp.zero.xacc: clear scalar accumulator
 *   - esp.movx.r.xacc.l: read XACC low 32 bits to GPR
 *
 * The key insight: esp.vmulas.s8.xacc does a full 16-element dot product
 * step in one instruction, accumulating the sum of all 16 products into
 * the scalar XACC register. This means we get a complete dot product
 * result without needing to extract and sum individual QACC lanes.
 */

#include "pie_gemv.h"
#include <string.h>
#include "esp_heap_caps.h"

/* ------------------------------------------------------------------ */
/* Scalar fallback (for non-P4 builds or host testing)                 */
/* ------------------------------------------------------------------ */

#if !CONFIG_IDF_TARGET_ESP32P4

int32_t pie_dot_int8(const int8_t *a, const int8_t *b, size_t len) {
    int32_t acc = 0;
    for (size_t i = 0; i < len; i++) {
        acc += (int32_t)a[i] * (int32_t)b[i];
    }
    return acc;
}

#else

/* ------------------------------------------------------------------ */
/* PIE implementation for ESP32-P4                                     */
/* ------------------------------------------------------------------ */

/*
 * Process up to CHUNK_ELEMS elements per XACC session to avoid
 * 40-bit accumulator overflow. 256 elements × 127 × 127 = 4.1M,
 * safely within 2^39 = 549B.
 */
#define PIE_CHUNK_ELEMS 256

static int32_t pie_dot_chunk(const int8_t *a, const int8_t *b, size_t len16) {
    /* len16 must be a multiple of 16 and <= PIE_CHUNK_ELEMS */
    int32_t partial;
    const int8_t *a_ptr = a;
    const int8_t *b_ptr = b;
    size_t count = len16;

    asm volatile(
        "esp.zero.xacc\n"
        "esp.vld.128.ip q0, %[ap], 16\n"
        "esp.vld.128.ip q1, %[bp], 16\n"
        "addi %[n], %[n], -16\n"
        "beqz %[n], 2f\n"
        "1:\n"
        "esp.vmulas.s8.xacc.ld.ip q0, %[ap], 16, q0, q1\n"
        "esp.vld.128.ip q1, %[bp], 16\n"
        "addi %[n], %[n], -16\n"
        "bnez %[n], 1b\n"
        "2:\n"
        "esp.vmulas.s8.xacc q0, q1\n"
        "esp.movx.r.xacc.l %[res]\n"
        : [ap] "+r" (a_ptr),
          [bp] "+r" (b_ptr),
          [n] "+r" (count),
          [res] "=r" (partial)
        :
        : "memory"
    );
    return partial;
}

int32_t pie_dot_int8(const int8_t *a, const int8_t *b, size_t len) {
    int32_t result = 0;
    size_t done = 0;

    /* Process in chunks to avoid XACC overflow */
    while (done + 16 <= len) {
        size_t remaining = len - done;
        size_t chunk = remaining & ~15U; /* round down to 16 */
        if (chunk > PIE_CHUNK_ELEMS) chunk = PIE_CHUNK_ELEMS;
        result += pie_dot_chunk(a + done, b + done, chunk);
        done += chunk;
    }

    /* Handle remainder (< 16 elements) */
    for (size_t i = done; i < len; i++) {
        result += (int32_t)a[i] * (int32_t)b[i];
    }

    return result;
}

#endif /* CONFIG_IDF_TARGET_ESP32P4 */

/* ------------------------------------------------------------------ */
/* GEMV using pie_dot_int8                                             */
/* ------------------------------------------------------------------ */

void pie_int8_gemv(
    const int8_t *row_quant,
    const int8_t *weights_t,
    size_t out_features,
    size_t in_features,
    int32_t *out_i32
) {
    for (size_t j = 0; j < out_features; j++) {
        out_i32[j] = pie_dot_int8(
            row_quant,
            weights_t + j * in_features,
            in_features
        );
    }
}

/* ------------------------------------------------------------------ */
/* Weight transpose: [in][out] -> [out][in_padded]                     */
/* ------------------------------------------------------------------ */

void transpose_int8_weights(
    const int8_t *src,
    size_t in_features,
    size_t out_features,
    size_t in_features_padded,
    int8_t *out_t
) {
    memset(out_t, 0, out_features * in_features_padded);
    for (size_t k = 0; k < in_features; k++) {
        for (size_t j = 0; j < out_features; j++) {
            out_t[j * in_features_padded + k] = src[k * out_features + j];
        }
    }
}

/* ------------------------------------------------------------------ */
/* Self-test                                                           */
/* ------------------------------------------------------------------ */

#include "esp_log.h"

static int32_t scalar_dot_int8(const int8_t *a, const int8_t *b, size_t len) {
    int32_t acc = 0;
    for (size_t i = 0; i < len; i++) {
        acc += (int32_t)a[i] * (int32_t)b[i];
    }
    return acc;
}

int pie_self_test(void) {
    static const char *TAG = "pie-test";
    int failures = 0;

    /* Test 1: 32 elements (2 PIE iterations) */
    int8_t a32[32] __attribute__((aligned(16)));
    int8_t b32[32] __attribute__((aligned(16)));
    for (int i = 0; i < 32; i++) {
        a32[i] = (int8_t)(i - 16);      /* -16..15 */
        b32[i] = (int8_t)(31 - i - 16); /* 15..-16 */
    }
    int32_t ref32 = scalar_dot_int8(a32, b32, 32);
    int32_t pie32 = pie_dot_int8(a32, b32, 32);
    if (ref32 != pie32) {
        ESP_LOGE(TAG, "FAIL test_32: scalar=%ld pie=%ld", (long)ref32, (long)pie32);
        failures++;
    } else {
        ESP_LOGI(TAG, "OK test_32: result=%ld", (long)pie32);
    }

    /* Test 2: 192 elements (typical hidden dim, 12 PIE iterations) */
    int8_t a192[192] __attribute__((aligned(16)));
    int8_t b192[192] __attribute__((aligned(16)));
    for (int i = 0; i < 192; i++) {
        a192[i] = (int8_t)((i * 7 + 3) % 255 - 127);
        b192[i] = (int8_t)((i * 13 + 5) % 255 - 127);
    }
    int32_t ref192 = scalar_dot_int8(a192, b192, 192);
    int32_t pie192 = pie_dot_int8(a192, b192, 192);
    if (ref192 != pie192) {
        ESP_LOGE(TAG, "FAIL test_192: scalar=%ld pie=%ld", (long)ref192, (long)pie192);
        failures++;
    } else {
        ESP_LOGI(TAG, "OK test_192: result=%ld", (long)pie192);
    }

    /* Test 3: 768 elements (encoder inter dim) -- must be 16-byte aligned for PIE */
    int8_t *a768 = (int8_t *)heap_caps_aligned_alloc(16, 768, MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);
    int8_t *b768 = (int8_t *)heap_caps_aligned_alloc(16, 768, MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);
    if (a768 && b768) {
        for (int i = 0; i < 768; i++) {
            a768[i] = (int8_t)((i * 11 + 7) % 255 - 127);
            b768[i] = (int8_t)((i * 17 + 13) % 255 - 127);
        }
        int32_t ref768 = scalar_dot_int8(a768, b768, 768);
        int32_t pie768 = pie_dot_int8(a768, b768, 768);
        if (ref768 != pie768) {
            ESP_LOGE(TAG, "FAIL test_768: scalar=%ld pie=%ld", (long)ref768, (long)pie768);
            failures++;
        } else {
            ESP_LOGI(TAG, "OK test_768: result=%ld", (long)pie768);
        }
    }
    free(a768);
    free(b768);

    /* Test 4: Q4 block dot product */
    uint8_t nibbles[16];
    for (int i = 0; i < 16; i++) nibbles[i] = ((i + 3) & 0xF) | (((15 - i) & 0xF) << 4);
    float input[32];
    for (int i = 0; i < 32; i++) input[i] = (float)(i - 16) * 0.1f;

    float pie_q4 = pie_q4_block_dot(nibbles, 0.5f, input, 32);
    /* Compute reference */
    float ref_q4 = 0.0f;
    for (int i = 0; i < 16; i++) {
        int8_t lo = (nibbles[i] & 0xF) - 8;
        int8_t hi = (nibbles[i] >> 4) - 8;
        ref_q4 += (float)lo * 0.5f * input[i * 2];
        ref_q4 += (float)hi * 0.5f * input[i * 2 + 1];
    }
    float q4_err = pie_q4 - ref_q4;
    if (q4_err < 0) q4_err = -q4_err;
    float q4_tol = (ref_q4 < 0 ? -ref_q4 : ref_q4) * 0.05f + 0.1f; /* 5% + epsilon */
    if (q4_err > q4_tol) {
        ESP_LOGE(TAG, "FAIL test_q4: ref=%.4f pie=%.4f err=%.4f", ref_q4, pie_q4, q4_err);
        failures++;
    } else {
        ESP_LOGI(TAG, "OK test_q4: ref=%.4f pie=%.4f err=%.4f", ref_q4, pie_q4, q4_err);
    }

    if (failures == 0) {
        ESP_LOGI(TAG, "All PIE self-tests passed");
    } else {
        ESP_LOGE(TAG, "%d PIE self-test(s) FAILED", failures);
    }
    return failures == 0 ? 0 : -1;
}

/* ------------------------------------------------------------------ */
/* Q4 block dot product with PIE                                       */
/* ------------------------------------------------------------------ */

float pie_q4_block_dot(
    const uint8_t *nibbles,
    float weight_scale,
    const float *input,
    size_t valid_count
) {
    /* Unpack 16 nibble bytes → 32 INT8 weights in [-8, 7] */
    int8_t w_i8[32] __attribute__((aligned(16)));
    for (size_t i = 0; i < 16; i++) {
        uint8_t packed = nibbles[i];
        w_i8[i * 2]     = (int8_t)((packed & 0x0FU) - 8U);
        w_i8[i * 2 + 1] = (int8_t)((packed >> 4U) - 8U);
    }

    /* Quantize 32 input floats → INT8 */
    int8_t x_i8[32] __attribute__((aligned(16)));
    float max_abs = 0.0f;
    size_t count = valid_count < 32 ? valid_count : 32;
    for (size_t i = 0; i < count; i++) {
        float a = input[i] < 0 ? -input[i] : input[i];
        if (a > max_abs) max_abs = a;
    }
    float x_scale = max_abs > 0.0f ? max_abs / 127.0f : 1.0f;
    float inv_scale = 1.0f / x_scale;
    for (size_t i = 0; i < count; i++) {
        float v = input[i] * inv_scale;
        if (v > 127.0f) v = 127.0f;
        if (v < -128.0f) v = -128.0f;
        x_i8[i] = (int8_t)(v + (v >= 0 ? 0.5f : -0.5f));
    }
    /* Zero-pad if valid_count < 32 */
    for (size_t i = count; i < 32; i++) {
        w_i8[i] = 0;
        x_i8[i] = 0;
    }

    /* PIE dot product: 32 elements = 2 iterations of 16 */
    int32_t dot = pie_dot_int8(w_i8, x_i8, 32);

    return (float)dot * x_scale * weight_scale;
}
