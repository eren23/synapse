#include <ctype.h>
#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

#include "esp_heap_caps.h"
#include "esp_log.h"
#include "esp_system.h"
#include "esp_task_wdt.h"
#include "esp_timer.h"

#include "pie_gemv.h"
#include "freertos/semphr.h"

static const char *TAG = "synapse-lewm";

/* Forward declarations for functions used in dual-core code */
static void softmax_inplace(float *x, size_t len);

/* ------------------------------------------------------------------ */
/* Dual-core attention: Core 1 worker for parallel attention compute    */
/* ------------------------------------------------------------------ */

typedef struct {
    /* Shared read-only inputs */
    const int8_t *q_i8;
    const int8_t *k_i8;
    const float *q_scales;
    const float *k_scales;
    const float *v;
    float *out;
    float inv_sqrt_hd;
    size_t seq_len;
    size_t num_heads;
    size_t head_dim;
    size_t head_dim_padded;
    size_t inner_dim;
    size_t row_bytes;
    /* Per-core range */
    size_t q_start;
    size_t q_end;
    /* Per-core scratch (scores buffer) */
    float *scores;
    /* Linear attention: skip softmax, use L1 normalization */
    bool linear_attn;
} AttnWorkItem;

/* Generic Core 1 work dispatch: function pointer + argument */
typedef void (*core1_fn_t)(void *arg);
static volatile core1_fn_t s_core1_fn = NULL;
static volatile void *s_core1_arg = NULL;
static SemaphoreHandle_t s_core1_start = NULL;
static SemaphoreHandle_t s_core1_done = NULL;

/* Dispatch a function to Core 1 and wait for completion */
static void core1_dispatch(core1_fn_t fn, void *arg) {
    s_core1_fn = fn;
    s_core1_arg = arg;
    xSemaphoreGive(s_core1_start);
}
static void core1_wait(void) {
    xSemaphoreTake(s_core1_done, portMAX_DELAY);
}

/* INT8 GEMV work item for dual-core FFN */
typedef struct {
    const int8_t *all_i8;
    const int8_t *weights_t;
    const float *scales;
    const float *w_scales;
    float *out;
    size_t m;
    size_t in_pad;
    size_t out_f;
    size_t j_start;
    size_t j_end;
} GemvWorkItem;

static void gemv_compute_range(void *arg) {
    const GemvWorkItem *w = (const GemvWorkItem *)arg;
    for (size_t j = w->j_start; j < w->j_end; ++j) {
        const int8_t *wj = w->weights_t + j * w->in_pad;
        float ws = w->w_scales[j];
        for (size_t row = 0; row < w->m; ++row) {
            int32_t dot = pie_dot_int8(
                w->all_i8 + row * w->in_pad, wj, w->in_pad);
            w->out[row * w->out_f + j] = (float)dot * w->scales[row] * ws;
        }
    }
}

static volatile AttnWorkItem s_core1_attn_work;

static void l1_normalize_inplace(float *x, size_t len) {
    float abs_sum = 0.0f;
    for (size_t i = 0; i < len; ++i) abs_sum += fabsf(x[i]);
    if (abs_sum > 1e-12f) {
        float inv = 1.0f / abs_sum;
        for (size_t i = 0; i < len; ++i) x[i] *= inv;
    }
}

/* ELU+1 feature map for kernel-trick linear attention: φ(x) = elu(x) + 1 */
static inline float elu_plus1(float x) {
    return x >= 0.0f ? x + 1.0f : expf(x);
}

/*
 * Kernel-trick linear attention: O(nd²) instead of O(n²d).
 *
 * Instead of building the full [n,n] score matrix:
 *   1. Apply φ to K → φK [n, d]
 *   2. Compute KV = φK^T @ V → [d, d]     (once per head)
 *   3. Compute k_sum = Σ φK[k] → [d]       (once per head)
 *   4. For each query: φQ @ KV / (φQ · k_sum) → [d]
 *
 * Uses f32 compute (φ breaks INT8 quantization).
 */
static void linear_attention_kernel_trick(
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

        /* 1. Compute KV = φ(K)^T @ V and k_sum = Σ φ(K) */
        memset(kv_matrix, 0, d * d * sizeof(float));
        memset(k_sum, 0, d * sizeof(float));

        for (size_t kt = 0; kt < seq_len; ++kt) {
            const float *k_row = k + kt * inner_dim + head * d;
            const float *v_row = v + kt * inner_dim + head * d;

            for (size_t i = 0; i < d; ++i) {
                float phi_k = elu_plus1(k_row[i]);
                k_sum[i] += phi_k;
                /* KV[i][j] += φ(K[kt][i]) * V[kt][j] */
                for (size_t j = 0; j < d; ++j) {
                    kv_matrix[i * d + j] += phi_k * v_row[j];
                }
            }
        }

        /* 2. For each query: out = φ(Q) @ KV / (φ(Q) · k_sum) */
        for (size_t qt = 0; qt < seq_len; ++qt) {
            const float *q_row = q + qt * inner_dim + head * d;
            float *out_row = out + qt * inner_dim + head * d;

            /* Compute φ(Q) and normalization Z = φ(Q) · k_sum */
            float z = 0.0f;
            for (size_t i = 0; i < d; ++i) {
                phi_q[i] = elu_plus1(q_row[i]);
                z += phi_q[i] * k_sum[i];
            }
            float inv_z = (z > 1e-12f) ? 1.0f / z : 0.0f;

            /* out = φ(Q) @ KV * inv_z */
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

static void attn_compute_range(const AttnWorkItem *w) {
    for (size_t head = 0; head < w->num_heads; ++head) {
        for (size_t q_token = w->q_start; q_token < w->q_end; ++q_token) {
            const int8_t *qi = w->q_i8 + q_token * w->row_bytes + head * w->head_dim_padded;
            float q_sc = w->q_scales[q_token * w->num_heads + head];

            for (size_t k_token = 0; k_token < w->seq_len; ++k_token) {
                const int8_t *ki = w->k_i8 + k_token * w->row_bytes + head * w->head_dim_padded;
                float k_sc = w->k_scales[k_token * w->num_heads + head];
                int32_t dot_i32 = pie_dot_int8(qi, ki, w->head_dim_padded);
                w->scores[k_token] = (float)dot_i32 * q_sc * k_sc * w->inv_sqrt_hd;
            }

            if (w->linear_attn) {
                l1_normalize_inplace(w->scores, w->seq_len);
            } else {
                softmax_inplace(w->scores, w->seq_len);
            }

            float *out_head = w->out + q_token * w->inner_dim + head * w->head_dim;
            memset(out_head, 0, w->head_dim * sizeof(float));
            for (size_t k_token = 0; k_token < w->seq_len; ++k_token) {
                float s = w->scores[k_token];
                const float *v_row = w->v + k_token * w->inner_dim + head * w->head_dim;
                for (size_t d = 0; d < w->head_dim; ++d) {
                    out_head[d] += s * v_row[d];
                }
            }
        }
    }
}

static void attn_compute_range_wrapper(void *arg) {
    attn_compute_range((const AttnWorkItem *)arg);
}

static void core1_worker(void *arg) {
    (void)arg;
    for (;;) {
        xSemaphoreTake(s_core1_start, portMAX_DELAY);
        if (s_core1_fn) {
            s_core1_fn((void *)s_core1_arg);
        }
        xSemaphoreGive(s_core1_done);
    }
}

static void dual_core_init(void) {
    if (s_core1_start) return;
    s_core1_start = xSemaphoreCreateBinary();
    s_core1_done = xSemaphoreCreateBinary();
    xTaskCreatePinnedToCore(core1_worker, "core1", 8192, NULL, 5, NULL, 1);
    ESP_LOGI(TAG, "Dual-core worker started on Core 1");
}

extern const uint8_t _binary_model_bin_start[] asm("_binary_model_bin_start");
extern const uint8_t _binary_model_bin_end[] asm("_binary_model_bin_end");

typedef struct {
    size_t len;
    float *data;
} FloatBuffer;

typedef struct {
    FloatBuffer weight;
    FloatBuffer bias;
} ProjectionLayer;

typedef struct {
    size_t num_layers;
    ProjectionLayer *layers;
    size_t max_dim;
} ProjectionHead;

typedef struct {
    size_t out_features;
    size_t in_features;
    size_t blocks_per_row;
    size_t total_blocks;
    const uint8_t *bitmap;
    const uint8_t *blocks_data;
    uint32_t *row_nz_starts;
} Q4LinearRef;

typedef struct {
    size_t out_features;
    size_t in_features;
    size_t in_features_padded; /* rounded up to multiple of 16 for PIE */
    const int8_t *weights_data;
    int8_t *weights_t;         /* transposed: [out][in_padded], PIE-friendly */
    FloatBuffer scales;
} Int8LinearRef;

typedef struct {
    Q4LinearRef adaln_linear;
    FloatBuffer adaln_bias;
    Q4LinearRef to_qkv;
    Q4LinearRef attn_out;
    FloatBuffer attn_out_bias;
    FloatBuffer attn_norm_weight;
    FloatBuffer attn_norm_bias;
    FloatBuffer mlp_norm_weight;
    FloatBuffer mlp_norm_bias;
    Q4LinearRef mlp_up;
    FloatBuffer mlp_up_bias;
    Q4LinearRef mlp_down;
    FloatBuffer mlp_down_bias;
} PredictorLayer;

typedef struct {
    Int8LinearRef w_q;
    Int8LinearRef w_k;
    Int8LinearRef w_v;
    Int8LinearRef w_o;
    Int8LinearRef ffn_up;
    Int8LinearRef ffn_down;
    FloatBuffer q_bias;
    FloatBuffer k_bias;
    FloatBuffer v_bias;
    FloatBuffer o_bias;
    FloatBuffer ffn_up_bias;
    FloatBuffer ffn_down_bias;
    FloatBuffer attn_norm_weight;
    FloatBuffer attn_norm_bias;
    FloatBuffer ffn_norm_weight;
    FloatBuffer ffn_norm_bias;
} EncoderLayer;

typedef struct {
    float *seq_raw;
    float *seq;
    float *conditioning;
    float *action_hidden;
    float *mod_vec;
    float *normed;
    float *modulated;
    float *qkv;
    float *attn_out;
    float *proj;
    float *ffn_inter;
    float *proj_tmp_a;
    float *proj_tmp_b;
} PredictorScratch;

typedef struct {
    float *x;
    float *normed;
    float *q;
    float *k;
    float *v;
    float *attn_out;
    float *proj;
    float *ffn_inter;
    float *patch;
    float *scores;
    float *cls_norm;
    int8_t *row_quant;
} EncoderScratch;

static void log_vector_preview(const char *label, const float *values, size_t len);
static void log_vector_stats(const char *label, const float *values, size_t len);

/* Cosine similarity for parity comparison */
static float cosine_similarity(const float *a, const float *b, size_t len) {
    float dot = 0.0f, na = 0.0f, nb = 0.0f;
    for (size_t i = 0; i < len; i++) {
        dot += a[i] * b[i];
        na += a[i] * a[i];
        nb += b[i] * b[i];
    }
    float denom = sqrtf(na) * sqrtf(nb);
    return denom > 0.0f ? dot / denom : 0.0f;
}

typedef struct {
    char mode[32];
    bool apply_final_norm_bias;
    bool has_full_encoder;
    size_t image_size;
    size_t patch_size;
    size_t channels;
    size_t encoder_hidden;
    size_t encoder_layers;
    size_t encoder_heads;
    size_t encoder_inter;
    size_t encoder_seq_len;
    size_t latent_dim;
    size_t action_dim;
    size_t predictor_hidden;
    size_t predictor_layers;
    size_t predictor_heads;
    size_t predictor_inner_dim;
    size_t predictor_inter;
    FloatBuffer predictor_pos_embed;
    FloatBuffer predictor_norm_weight;
    FloatBuffer predictor_norm_bias;
    FloatBuffer action_conv_weight;
    FloatBuffer action_conv_bias;
    FloatBuffer action_mlp1_weight;
    FloatBuffer action_mlp1_bias;
    FloatBuffer action_mlp2_weight;
    FloatBuffer action_mlp2_bias;
    FloatBuffer input_proj_weight;
    FloatBuffer input_proj_bias;
    FloatBuffer cond_proj_weight;
    FloatBuffer cond_proj_bias;
    FloatBuffer patch_proj;
    FloatBuffer patch_proj_bias;
    /* INT8-quantized patch_proj for PIE-accelerated batch embedding */
    Int8LinearRef patch_proj_i8;
    float *all_patches;   /* scratch: [num_patches * patch_dim] for batch extract */
    FloatBuffer cls_token;
    FloatBuffer pos_embed;
    FloatBuffer final_norm_weight;
    FloatBuffer final_norm_bias;
    /* Hybrid encoder extras */
    size_t meta_tokens;
    FloatBuffer meta_token;
    FloatBuffer enc_proj_weight;
    FloatBuffer enc_proj_bias;
    ProjectionHead projector;
    ProjectionHead pred_proj;
    EncoderLayer *encoder;
    PredictorLayer *layers;
    EncoderScratch encoder_scratch;
    PredictorScratch scratch;
} PredictorModel;

static uint32_t read_u32_le(const uint8_t *ptr) {
    return ((uint32_t)ptr[0]) |
           ((uint32_t)ptr[1] << 8) |
           ((uint32_t)ptr[2] << 16) |
           ((uint32_t)ptr[3] << 24);
}

static float read_f32_le(const uint8_t *ptr) {
    float value = 0.0f;
    uint32_t raw = read_u32_le(ptr);
    memcpy(&value, &raw, sizeof(value));
    return value;
}

static const char *skip_ws(const char *ptr) {
    while (*ptr && isspace((unsigned char)*ptr)) {
        ptr++;
    }
    return ptr;
}

static bool json_extract_string(const char *json, const char *key, char *out, size_t out_len) {
    char needle[64];
    snprintf(needle, sizeof(needle), "\"%s\"", key);

    const char *start = strstr(json, needle);
    if (start == NULL) {
        return false;
    }

    start = strchr(start, ':');
    if (start == NULL) {
        return false;
    }
    start = skip_ws(start + 1);
    if (*start != '"') {
        return false;
    }
    start++;

    const char *end = strchr(start, '"');
    if (end == NULL) {
        return false;
    }

    size_t copy_len = (size_t)(end - start);
    if (copy_len + 1 > out_len) {
        return false;
    }

    memcpy(out, start, copy_len);
    out[copy_len] = '\0';
    return true;
}

static bool json_extract_u32(const char *json, const char *key, uint32_t *value) {
    char needle[64];
    snprintf(needle, sizeof(needle), "\"%s\"", key);

    const char *start = strstr(json, needle);
    if (start == NULL) {
        return false;
    }

    start = strchr(start, ':');
    if (start == NULL) {
        return false;
    }
    start = skip_ws(start + 1);

    char *end = NULL;
    unsigned long parsed = strtoul(start, &end, 10);
    if (end == start) {
        return false;
    }

    *value = (uint32_t)parsed;
    return true;
}

static void *alloc_caps(size_t size, uint32_t caps) {
    if (size == 0) {
        return NULL;
    }
    void *ptr = heap_caps_malloc(size, caps);
    if (ptr == NULL) {
        ptr = malloc(size);
    }
    return ptr;
}

static void *calloc_caps(size_t count, size_t size, uint32_t caps) {
    if (count == 0 || size == 0) {
        return NULL;
    }
    void *ptr = heap_caps_calloc(count, size, caps);
    if (ptr == NULL) {
        ptr = calloc(count, size);
    }
    return ptr;
}

static bool alloc_float_buffer(FloatBuffer *buffer, size_t len, uint32_t caps) {
    buffer->len = len;
    if (len == 0) {
        buffer->data = NULL;
        return true;
    }
    buffer->data = (float *)alloc_caps(len * sizeof(float), caps);
    return buffer->data != NULL;
}

static bool copy_f32_payload(FloatBuffer *buffer, const uint8_t *src, size_t len, uint32_t caps) {
    if (!alloc_float_buffer(buffer, len, caps)) {
        return false;
    }
    for (size_t i = 0; i < len; ++i) {
        buffer->data[i] = read_f32_le(src + i * 4);
    }
    return true;
}

static bool cursor_read_f32_vector(
    const uint8_t *data,
    size_t data_len,
    size_t *off,
    FloatBuffer *buffer,
    uint32_t caps
) {
    if (*off + 4 > data_len) {
        return false;
    }
    uint32_t len = read_u32_le(data + *off);
    *off += 4;
    if (*off + (size_t)len * 4 > data_len) {
        return false;
    }
    bool ok = copy_f32_payload(buffer, data + *off, len, caps);
    *off += (size_t)len * 4;
    return ok;
}

static bool cursor_skip_f32_vector(const uint8_t *data, size_t data_len, size_t *off) {
    if (*off + 4 > data_len) {
        return false;
    }
    uint32_t len = read_u32_le(data + *off);
    *off += 4;
    if (*off + (size_t)len * 4 > data_len) {
        return false;
    }
    *off += (size_t)len * 4;
    return true;
}

static bool bitmap_get(const uint8_t *bitmap, size_t bit_index) {
    return ((bitmap[bit_index / 8] >> (bit_index % 8)) & 1U) != 0;
}

static bool cursor_read_q4_linear(
    const uint8_t *data,
    size_t data_len,
    size_t *off,
    Q4LinearRef *linear
) {
    if (*off + 16 > data_len) {
        return false;
    }

    linear->out_features = read_u32_le(data + *off);
    *off += 4;
    linear->in_features = read_u32_le(data + *off);
    *off += 4;
    linear->total_blocks = read_u32_le(data + *off);
    *off += 4;
    uint32_t nonzero_count = read_u32_le(data + *off);
    *off += 4;

    linear->blocks_per_row = (linear->in_features + 31U) / 32U;
    if (linear->out_features * linear->blocks_per_row != linear->total_blocks) {
        return false;
    }

    size_t bitmap_bytes = (linear->total_blocks + 7U) / 8U;
    if (*off + bitmap_bytes > data_len) {
        return false;
    }
    linear->bitmap = data + *off;
    *off += bitmap_bytes;

    size_t block_bytes = (size_t)nonzero_count * 20U;
    if (*off + block_bytes > data_len) {
        return false;
    }
    linear->blocks_data = data + *off;
    *off += block_bytes;

    linear->row_nz_starts =
        (uint32_t *)calloc_caps(linear->out_features + 1U, sizeof(uint32_t), MALLOC_CAP_8BIT);
    if (linear->row_nz_starts == NULL) {
        return false;
    }

    uint32_t nz_running = 0;
    linear->row_nz_starts[0] = 0;
    for (size_t row = 0; row < linear->out_features; ++row) {
        for (size_t block = 0; block < linear->blocks_per_row; ++block) {
            size_t global_block = row * linear->blocks_per_row + block;
            if (bitmap_get(linear->bitmap, global_block)) {
                nz_running++;
            }
        }
        linear->row_nz_starts[row + 1U] = nz_running;
    }

    return nz_running == nonzero_count;
}

static bool cursor_skip_int8_linear(const uint8_t *data, size_t data_len, size_t *off) {
    if (*off + 12 > data_len) {
        ESP_LOGE(TAG, "INT8 linear header truncated at off=%lu len=%lu",
                 (unsigned long)*off, (unsigned long)data_len);
        return false;
    }
    uint32_t out_features = read_u32_le(data + *off);
    *off += 4;
    uint32_t in_features = read_u32_le(data + *off);
    *off += 4;
    uint32_t weights_len = read_u32_le(data + *off);
    *off += 4;
    if (*off + weights_len > data_len) {
        ESP_LOGE(TAG,
                 "INT8 weights truncated at off=%lu out=%lu in=%lu weights_len=%lu len=%lu",
                 (unsigned long)*off,
                 (unsigned long)out_features,
                 (unsigned long)in_features,
                 (unsigned long)weights_len,
                 (unsigned long)data_len);
        return false;
    }
    *off += weights_len;
    if (*off + 4 > data_len) {
        ESP_LOGE(TAG, "INT8 scales header truncated at off=%lu len=%lu",
                 (unsigned long)*off, (unsigned long)data_len);
        return false;
    }
    uint32_t scales_len = read_u32_le(data + *off);
    *off += 4;
    if (*off + (size_t)scales_len * 4 > data_len) {
        ESP_LOGE(TAG,
                 "INT8 scales truncated at off=%lu out=%lu in=%lu scales_len=%lu len=%lu",
                 (unsigned long)*off,
                 (unsigned long)out_features,
                 (unsigned long)in_features,
                 (unsigned long)scales_len,
                 (unsigned long)data_len);
        return false;
    }
    *off += (size_t)scales_len * 4;
    return true;
}

static bool cursor_read_int8_linear(
    const uint8_t *data,
    size_t data_len,
    size_t *off,
    Int8LinearRef *linear,
    uint32_t caps
) {
    if (*off + 12 > data_len) {
        return false;
    }

    linear->out_features = read_u32_le(data + *off);
    *off += 4;
    linear->in_features = read_u32_le(data + *off);
    *off += 4;
    uint32_t weights_len = read_u32_le(data + *off);
    *off += 4;
    if (*off + weights_len > data_len) {
        return false;
    }
    linear->weights_data = (const int8_t *)(data + *off);
    *off += weights_len;

    if (*off + 4 > data_len) {
        return false;
    }
    uint32_t scales_len = read_u32_le(data + *off);
    *off += 4;
    if (*off + (size_t)scales_len * 4 > data_len) {
        return false;
    }
    bool ok = copy_f32_payload(&linear->scales, data + *off, scales_len, caps);
    *off += (size_t)scales_len * 4;
    if (!ok) return false;

    /* Transpose weights for PIE SIMD: [in][out] -> [out][in_padded] */
    linear->in_features_padded = (linear->in_features + 15U) & ~15U;
    size_t t_size = linear->out_features * linear->in_features_padded;
    linear->weights_t = (int8_t *)heap_caps_malloc(
        t_size, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
    if (linear->weights_t) {
        transpose_int8_weights(
            linear->weights_data,
            linear->in_features,
            linear->out_features,
            linear->in_features_padded,
            linear->weights_t);
    }

    return true;
}

static bool cursor_skip_projection_head(const uint8_t *data, size_t data_len, size_t *off) {
    if (*off + 4 > data_len) {
        return false;
    }
    uint32_t num_layers = read_u32_le(data + *off);
    *off += 4;
    for (uint32_t i = 0; i < num_layers; ++i) {
        if (!cursor_skip_f32_vector(data, data_len, off)) {
            return false;
        }
        if (!cursor_skip_f32_vector(data, data_len, off)) {
            return false;
        }
    }
    return true;
}

static bool cursor_read_projection_head(
    const uint8_t *data,
    size_t data_len,
    size_t *off,
    ProjectionHead *head,
    size_t input_dim,
    uint32_t caps
) {
    if (*off + 4 > data_len) {
        return false;
    }
    head->num_layers = read_u32_le(data + *off);
    *off += 4;
    head->max_dim = input_dim;
    if (head->num_layers == 0) {
        head->layers = NULL;
        return true;
    }

    head->layers =
        (ProjectionLayer *)calloc_caps(head->num_layers, sizeof(ProjectionLayer), MALLOC_CAP_8BIT);
    if (head->layers == NULL) {
        return false;
    }

    size_t current_dim = input_dim;
    for (size_t i = 0; i < head->num_layers; ++i) {
        if (!cursor_read_f32_vector(data, data_len, off, &head->layers[i].weight, caps)) {
            return false;
        }
        if (!cursor_read_f32_vector(data, data_len, off, &head->layers[i].bias, caps)) {
            return false;
        }
        if (head->layers[i].weight.len == 0 || head->layers[i].weight.data == NULL) {
            continue;
        }
        if (current_dim == 0 || head->layers[i].weight.len % current_dim != 0) {
            return false;
        }
        current_dim = head->layers[i].weight.len / current_dim;
        if (current_dim > head->max_dim) {
            head->max_dim = current_dim;
        }
    }

    return true;
}

static bool cursor_read_q4_predictor_layer(
    const uint8_t *data,
    size_t data_len,
    size_t *off,
    PredictorLayer *layer,
    uint32_t float_caps
) {
    if (!cursor_read_q4_linear(data, data_len, off, &layer->adaln_linear)) {
        return false;
    }
    if (!cursor_read_f32_vector(data, data_len, off, &layer->adaln_bias, float_caps)) {
        return false;
    }
    if (!cursor_read_q4_linear(data, data_len, off, &layer->to_qkv)) {
        return false;
    }
    if (!cursor_read_q4_linear(data, data_len, off, &layer->attn_out)) {
        return false;
    }
    if (!cursor_read_f32_vector(data, data_len, off, &layer->attn_out_bias, float_caps)) {
        return false;
    }
    if (!cursor_read_f32_vector(data, data_len, off, &layer->attn_norm_weight, float_caps)) {
        return false;
    }
    if (!cursor_read_f32_vector(data, data_len, off, &layer->attn_norm_bias, float_caps)) {
        return false;
    }
    if (!cursor_read_f32_vector(data, data_len, off, &layer->mlp_norm_weight, float_caps)) {
        return false;
    }
    if (!cursor_read_f32_vector(data, data_len, off, &layer->mlp_norm_bias, float_caps)) {
        return false;
    }
    if (!cursor_read_q4_linear(data, data_len, off, &layer->mlp_up)) {
        return false;
    }
    if (!cursor_read_f32_vector(data, data_len, off, &layer->mlp_up_bias, float_caps)) {
        return false;
    }
    if (!cursor_read_q4_linear(data, data_len, off, &layer->mlp_down)) {
        return false;
    }
    return cursor_read_f32_vector(data, data_len, off, &layer->mlp_down_bias, float_caps);
}

static bool cursor_read_encoder_layer(
    const uint8_t *data,
    size_t data_len,
    size_t *off,
    EncoderLayer *layer,
    uint32_t float_caps
) {
    if (!cursor_read_f32_vector(data, data_len, off, &layer->attn_norm_weight, float_caps)) {
        return false;
    }
    if (!cursor_read_f32_vector(data, data_len, off, &layer->attn_norm_bias, float_caps)) {
        return false;
    }
    if (!cursor_read_int8_linear(data, data_len, off, &layer->w_q, float_caps)) {
        return false;
    }
    if (!cursor_read_f32_vector(data, data_len, off, &layer->q_bias, float_caps)) {
        return false;
    }
    if (!cursor_read_int8_linear(data, data_len, off, &layer->w_k, float_caps)) {
        return false;
    }
    if (!cursor_read_f32_vector(data, data_len, off, &layer->k_bias, float_caps)) {
        return false;
    }
    if (!cursor_read_int8_linear(data, data_len, off, &layer->w_v, float_caps)) {
        return false;
    }
    if (!cursor_read_f32_vector(data, data_len, off, &layer->v_bias, float_caps)) {
        return false;
    }
    if (!cursor_read_int8_linear(data, data_len, off, &layer->w_o, float_caps)) {
        return false;
    }
    if (!cursor_read_f32_vector(data, data_len, off, &layer->o_bias, float_caps)) {
        return false;
    }
    if (!cursor_read_f32_vector(data, data_len, off, &layer->ffn_norm_weight, float_caps)) {
        return false;
    }
    if (!cursor_read_f32_vector(data, data_len, off, &layer->ffn_norm_bias, float_caps)) {
        return false;
    }
    if (!cursor_read_int8_linear(data, data_len, off, &layer->ffn_up, float_caps)) {
        return false;
    }
    if (!cursor_read_f32_vector(data, data_len, off, &layer->ffn_up_bias, float_caps)) {
        return false;
    }
    if (!cursor_read_int8_linear(data, data_len, off, &layer->ffn_down, float_caps)) {
        return false;
    }
    return cursor_read_f32_vector(data, data_len, off, &layer->ffn_down_bias, float_caps);
}

static bool skip_q4_pred_encoder(
    const uint8_t *data,
    size_t data_len,
    size_t *off,
    size_t encoder_layers
) {
    if (!cursor_skip_f32_vector(data, data_len, off) ||
        !cursor_skip_f32_vector(data, data_len, off) ||
        !cursor_skip_f32_vector(data, data_len, off) ||
        !cursor_skip_f32_vector(data, data_len, off)) {
        return false;
    }

    for (size_t i = 0; i < encoder_layers; ++i) {
        for (size_t j = 0; j < 16; ++j) {
            if (!cursor_skip_f32_vector(data, data_len, off)) {
                return false;
            }
        }
    }

    return cursor_skip_f32_vector(data, data_len, off) &&
           cursor_skip_f32_vector(data, data_len, off);
}

__attribute__((unused))
static bool skip_full_encoder(
    const uint8_t *data,
    size_t data_len,
    size_t *off,
    size_t encoder_layers
) {
    size_t step_off = *off;
    if (!cursor_skip_f32_vector(data, data_len, off)) {
        ESP_LOGE(TAG, "full skip failed at patch_proj off=%lu", (unsigned long)step_off);
        return false;
    }
    step_off = *off;
    if (!cursor_skip_f32_vector(data, data_len, off)) {
        ESP_LOGE(TAG, "full skip failed at patch_proj_bias off=%lu", (unsigned long)step_off);
        return false;
    }
    step_off = *off;
    if (!cursor_skip_f32_vector(data, data_len, off)) {
        ESP_LOGE(TAG, "full skip failed at cls_token off=%lu", (unsigned long)step_off);
        return false;
    }
    step_off = *off;
    if (!cursor_skip_f32_vector(data, data_len, off)) {
        ESP_LOGE(TAG, "full skip failed at pos_embed off=%lu", (unsigned long)step_off);
        return false;
    }

    for (size_t i = 0; i < encoder_layers; ++i) {
        step_off = *off;
        if (!cursor_skip_f32_vector(data, data_len, off)) {
            ESP_LOGE(TAG, "full skip failed at layer %lu attn_norm_weight off=%lu",
                     (unsigned long)i, (unsigned long)step_off);
            return false;
        }
        step_off = *off;
        if (!cursor_skip_f32_vector(data, data_len, off)) {
            ESP_LOGE(TAG, "full skip failed at layer %lu attn_norm_bias off=%lu",
                     (unsigned long)i, (unsigned long)step_off);
            return false;
        }
        step_off = *off;
        if (!cursor_skip_int8_linear(data, data_len, off)) {
            ESP_LOGE(TAG, "full skip failed at layer %lu w_q off=%lu",
                     (unsigned long)i, (unsigned long)step_off);
            return false;
        }
        step_off = *off;
        if (!cursor_skip_f32_vector(data, data_len, off)) {
            ESP_LOGE(TAG, "full skip failed at layer %lu q_bias off=%lu",
                     (unsigned long)i, (unsigned long)step_off);
            return false;
        }
        step_off = *off;
        if (!cursor_skip_int8_linear(data, data_len, off)) {
            ESP_LOGE(TAG, "full skip failed at layer %lu w_k off=%lu",
                     (unsigned long)i, (unsigned long)step_off);
            return false;
        }
        step_off = *off;
        if (!cursor_skip_f32_vector(data, data_len, off)) {
            ESP_LOGE(TAG, "full skip failed at layer %lu k_bias off=%lu",
                     (unsigned long)i, (unsigned long)step_off);
            return false;
        }
        step_off = *off;
        if (!cursor_skip_int8_linear(data, data_len, off)) {
            ESP_LOGE(TAG, "full skip failed at layer %lu w_v off=%lu",
                     (unsigned long)i, (unsigned long)step_off);
            return false;
        }
        step_off = *off;
        if (!cursor_skip_f32_vector(data, data_len, off)) {
            ESP_LOGE(TAG, "full skip failed at layer %lu v_bias off=%lu",
                     (unsigned long)i, (unsigned long)step_off);
            return false;
        }
        step_off = *off;
        if (!cursor_skip_int8_linear(data, data_len, off)) {
            ESP_LOGE(TAG, "full skip failed at layer %lu w_o off=%lu",
                     (unsigned long)i, (unsigned long)step_off);
            return false;
        }
        step_off = *off;
        if (!cursor_skip_f32_vector(data, data_len, off)) {
            ESP_LOGE(TAG, "full skip failed at layer %lu o_bias off=%lu",
                     (unsigned long)i, (unsigned long)step_off);
            return false;
        }
        step_off = *off;
        if (!cursor_skip_f32_vector(data, data_len, off)) {
            ESP_LOGE(TAG, "full skip failed at layer %lu ffn_norm_weight off=%lu",
                     (unsigned long)i, (unsigned long)step_off);
            return false;
        }
        step_off = *off;
        if (!cursor_skip_f32_vector(data, data_len, off)) {
            ESP_LOGE(TAG, "full skip failed at layer %lu ffn_norm_bias off=%lu",
                     (unsigned long)i, (unsigned long)step_off);
            return false;
        }
        step_off = *off;
        if (!cursor_skip_int8_linear(data, data_len, off)) {
            ESP_LOGE(TAG, "full skip failed at layer %lu ffn_up off=%lu",
                     (unsigned long)i, (unsigned long)step_off);
            return false;
        }
        step_off = *off;
        if (!cursor_skip_f32_vector(data, data_len, off)) {
            ESP_LOGE(TAG, "full skip failed at layer %lu ffn_up_bias off=%lu",
                     (unsigned long)i, (unsigned long)step_off);
            return false;
        }
        step_off = *off;
        if (!cursor_skip_int8_linear(data, data_len, off)) {
            ESP_LOGE(TAG, "full skip failed at layer %lu ffn_down off=%lu",
                     (unsigned long)i, (unsigned long)step_off);
            return false;
        }
        step_off = *off;
        if (!cursor_skip_f32_vector(data, data_len, off)) {
            ESP_LOGE(TAG, "full skip failed at layer %lu ffn_down_bias off=%lu",
                     (unsigned long)i, (unsigned long)step_off);
            return false;
        }
    }

    step_off = *off;
    if (!cursor_skip_f32_vector(data, data_len, off)) {
        ESP_LOGE(TAG, "full skip failed at final_norm_weight off=%lu", (unsigned long)step_off);
        return false;
    }
    step_off = *off;
    if (!cursor_skip_f32_vector(data, data_len, off)) {
        ESP_LOGE(TAG, "full skip failed at final_norm_bias off=%lu", (unsigned long)step_off);
        return false;
    }
    return true;
}

static void matmul_t_into(
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

static void add_bias_inplace(float *x, const float *bias, size_t m, size_t n) {
    if (bias == NULL) {
        return;
    }
    for (size_t row = 0; row < m; ++row) {
        for (size_t col = 0; col < n; ++col) {
            x[row * n + col] += bias[col];
        }
    }
}

static void layernorm_into(
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

/*
 * GELU lookup table: 1024 entries over [-8, 8].
 * gelu(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
 *
 * For |x| > 8, gelu(x) ≈ x (positive) or 0 (negative).
 * Linear interpolation between table entries gives <0.001 max error.
 *
 * Built once at startup, stored in internal SRAM for fast access.
 */
#define GELU_LUT_SIZE 1024
#define GELU_LUT_MIN  (-8.0f)
#define GELU_LUT_MAX  (8.0f)
#define GELU_LUT_STEP ((GELU_LUT_MAX - GELU_LUT_MIN) / (float)(GELU_LUT_SIZE - 1))

static float s_gelu_lut[GELU_LUT_SIZE];
static bool s_gelu_lut_ready = false;

static void gelu_lut_init(void) {
    if (s_gelu_lut_ready) return;
    const float pi = 3.14159265358979323846f;
    const float alpha = sqrtf(2.0f / pi);
    for (int i = 0; i < GELU_LUT_SIZE; i++) {
        float x = GELU_LUT_MIN + (float)i * GELU_LUT_STEP;
        s_gelu_lut[i] = 0.5f * x * (1.0f + tanhf(alpha * (x + 0.044715f * x * x * x)));
    }
    s_gelu_lut_ready = true;
}

static inline float gelu_scalar(float x) {
    if (x <= GELU_LUT_MIN) return 0.0f;
    if (x >= GELU_LUT_MAX) return x;

    float idx_f = (x - GELU_LUT_MIN) / GELU_LUT_STEP;
    int idx = (int)idx_f;
    if (idx >= GELU_LUT_SIZE - 1) return s_gelu_lut[GELU_LUT_SIZE - 1];
    float frac = idx_f - (float)idx;
    return s_gelu_lut[idx] + frac * (s_gelu_lut[idx + 1] - s_gelu_lut[idx]);
}

static void quantize_row_int8(const float *input, size_t len, int8_t *out, float *scale_out);

static void softmax_inplace(float *x, size_t len) {
    float max_value = -INFINITY;
    for (size_t i = 0; i < len; ++i) {
        if (x[i] > max_value) {
            max_value = x[i];
        }
    }

    float sum = 0.0f;
    for (size_t i = 0; i < len; ++i) {
        x[i] = expf(x[i] - max_value);
        sum += x[i];
    }

    if (sum > 0.0f) {
        for (size_t i = 0; i < len; ++i) {
            x[i] /= sum;
        }
    }
}

static void q4linear_forward_into(
    const Q4LinearRef *linear,
    const float *x,
    size_t m,
    float *out
) {
    memset(out, 0, m * linear->out_features * sizeof(float));

    /* Quantize each input row to INT8 once upfront */
    size_t in_padded = (linear->in_features + 31U) & ~31U;
    int8_t *x_i8 = (int8_t *)heap_caps_malloc(
        m * in_padded, MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);
    float *x_scales = (float *)heap_caps_malloc(
        m * sizeof(float), MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);

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
                /* PIE path: unpack Q4 → INT8, dot with pre-quantized input */
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

static void bidirectional_attention_from_qkv(
    const float *qkv,
    size_t seq_len,
    size_t num_heads,
    size_t head_dim,
    float *out
) {
    size_t inner_dim = num_heads * head_dim;
    float scores[3] = {0.0f, 0.0f, 0.0f};

    for (size_t token = 0; token < seq_len * inner_dim; ++token) {
        out[token] = 0.0f;
    }

    for (size_t head = 0; head < num_heads; ++head) {
        for (size_t q_token = 0; q_token < seq_len; ++q_token) {
            for (size_t k_token = 0; k_token < seq_len; ++k_token) {
                float dot = 0.0f;
                for (size_t d = 0; d < head_dim; ++d) {
                    size_t q_idx = q_token * 3U * inner_dim + head * head_dim + d;
                    size_t k_idx = k_token * 3U * inner_dim + inner_dim + head * head_dim + d;
                    dot += qkv[q_idx] * qkv[k_idx];
                }
                scores[k_token] = dot / sqrtf((float)head_dim);
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
}

static void quantize_row_int8(
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

/* INT8 GEMV with pre-quantized input (avoids re-quantizing for shared inputs like QKV) */
static void int8linear_forward_prequant(
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

static void int8linear_forward_into(
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
         *
         * Memory reads: out_f * in_pad (weight matrix once)
         *             + m * in_pad (input rows, tiny, stays in cache)
         * vs old path: m * out_f * in_pad (weight matrix per row!)
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

static void bidirectional_attention_separate(
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
     * Layout: [seq_len][num_heads][head_dim_padded] INT8 + per-row scales.
     * Total: seq_len * num_heads * head_dim_padded * 2 (Q + K) bytes. */
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

/* Maximum sequence length for the predictor (30 = 10 actions × 3 tokens each).
 * This is the maximum N for predict_rollout_fused (N=10 steps → 30 tokens).
 * Scratch buffers are allocated for this size so the fused function never reallocates. */
#define MAX_PREDICTOR_SEQ_LEN  30U

static bool allocate_scratch(PredictorModel *model) {
    PredictorScratch *scratch = &model->scratch;
    EncoderScratch *encoder_scratch = &model->encoder_scratch;
    size_t max_seq_len = MAX_PREDICTOR_SEQ_LEN;
    size_t hidden = model->predictor_hidden;
    size_t inner = model->predictor_inner_dim;
    size_t inter = model->predictor_inter;
    // seq_raw always stores hidden elements per token (needed for projected action embeds
    // and positional embeddings in has_proj path). Token layout: [z_start, a_proj, zeros].
    size_t seq_raw_dim = hidden;
    size_t proj_tmp_dim = model->pred_proj.max_dim > model->projector.max_dim ?
                          model->pred_proj.max_dim :
                          model->projector.max_dim;

    scratch->seq_raw =
        (float *)calloc_caps(max_seq_len * seq_raw_dim, sizeof(float), MALLOC_CAP_INTERNAL);
    scratch->seq = (float *)calloc_caps(max_seq_len * hidden, sizeof(float), MALLOC_CAP_INTERNAL);
    scratch->conditioning = (float *)calloc_caps(hidden, sizeof(float), MALLOC_CAP_INTERNAL);
    scratch->action_hidden = (float *)calloc_caps(inter, sizeof(float), MALLOC_CAP_INTERNAL);
    scratch->mod_vec = (float *)calloc_caps(6U * hidden, sizeof(float), MALLOC_CAP_INTERNAL);
    scratch->normed = (float *)calloc_caps(max_seq_len * hidden, sizeof(float), MALLOC_CAP_INTERNAL);
    scratch->modulated =
        (float *)calloc_caps(max_seq_len * hidden, sizeof(float), MALLOC_CAP_INTERNAL);
    scratch->qkv =
        (float *)calloc_caps(max_seq_len * 3U * inner, sizeof(float), MALLOC_CAP_INTERNAL);
    scratch->attn_out =
        (float *)calloc_caps(max_seq_len * inner, sizeof(float), MALLOC_CAP_INTERNAL);
    scratch->proj = (float *)calloc_caps(max_seq_len * hidden, sizeof(float), MALLOC_CAP_INTERNAL);
    scratch->ffn_inter =
        (float *)calloc_caps(max_seq_len * inter, sizeof(float), MALLOC_CAP_INTERNAL);
    scratch->proj_tmp_a =
        (float *)calloc_caps(proj_tmp_dim, sizeof(float), MALLOC_CAP_INTERNAL);
    scratch->proj_tmp_b =
        (float *)calloc_caps(proj_tmp_dim, sizeof(float), MALLOC_CAP_INTERNAL);

    bool predictor_ok = scratch->seq_raw != NULL &&
           scratch->seq != NULL &&
           scratch->conditioning != NULL &&
           scratch->action_hidden != NULL &&
           scratch->mod_vec != NULL &&
           scratch->normed != NULL &&
           scratch->modulated != NULL &&
           scratch->qkv != NULL &&
           scratch->attn_out != NULL &&
           scratch->proj != NULL &&
           scratch->ffn_inter != NULL &&
           scratch->proj_tmp_a != NULL &&
           scratch->proj_tmp_b != NULL;

    if (!predictor_ok) {
        return false;
    }

    if (!model->has_full_encoder) {
        return true;
    }

    size_t enc_seq_len = model->encoder_seq_len;
    size_t enc_hidden = model->encoder_hidden;
    size_t enc_inner = model->encoder_hidden;
    size_t enc_inter = model->encoder_inter;
    size_t patch_dim = model->patch_size * model->patch_size * model->channels;
    size_t row_quant_dim = patch_dim;
    if (enc_hidden > row_quant_dim) {
        row_quant_dim = enc_hidden;
    }
    if (enc_inter > row_quant_dim) {
        row_quant_dim = enc_inter;
    }

    uint32_t enc_caps = MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT;
    encoder_scratch->x =
        (float *)calloc_caps(enc_seq_len * enc_hidden, sizeof(float), enc_caps);
    encoder_scratch->normed =
        (float *)calloc_caps(enc_seq_len * enc_hidden, sizeof(float), enc_caps);
    encoder_scratch->q =
        (float *)calloc_caps(enc_seq_len * enc_inner, sizeof(float), enc_caps);
    encoder_scratch->k =
        (float *)calloc_caps(enc_seq_len * enc_inner, sizeof(float), enc_caps);
    encoder_scratch->v =
        (float *)calloc_caps(enc_seq_len * enc_inner, sizeof(float), enc_caps);
    encoder_scratch->attn_out =
        (float *)calloc_caps(enc_seq_len * enc_inner, sizeof(float), enc_caps);
    encoder_scratch->proj =
        (float *)calloc_caps(enc_seq_len * enc_hidden, sizeof(float), enc_caps);
    encoder_scratch->ffn_inter =
        (float *)calloc_caps(enc_seq_len * enc_inter, sizeof(float), enc_caps);
    encoder_scratch->patch = (float *)calloc_caps(patch_dim, sizeof(float), enc_caps);
    encoder_scratch->scores =
        (float *)calloc_caps(enc_seq_len, sizeof(float), MALLOC_CAP_INTERNAL);
    encoder_scratch->cls_norm =
        (float *)calloc_caps(enc_hidden, sizeof(float), MALLOC_CAP_INTERNAL);
    encoder_scratch->row_quant =
        (int8_t *)calloc_caps(row_quant_dim, sizeof(int8_t), MALLOC_CAP_INTERNAL);

    return encoder_scratch->x != NULL &&
           encoder_scratch->normed != NULL &&
           encoder_scratch->q != NULL &&
           encoder_scratch->k != NULL &&
           encoder_scratch->v != NULL &&
           encoder_scratch->attn_out != NULL &&
           encoder_scratch->proj != NULL &&
           encoder_scratch->ffn_inter != NULL &&
           encoder_scratch->patch != NULL &&
           encoder_scratch->scores != NULL &&
           encoder_scratch->cls_norm != NULL &&
           encoder_scratch->row_quant != NULL;
}

static bool quantize_f32_to_int8linear(const float *, size_t, size_t, Int8LinearRef *);

static bool parse_predictor_model(
    const uint8_t *model_data,
    size_t model_len,
    const char *config_json,
    PredictorModel *model
) {
    uint32_t encoder_layers = 0;
    uint32_t predictor_layers = 0;
    uint32_t predictor_hidden = 0;
    uint32_t predictor_heads = 0;
    uint32_t predictor_inner_dim = 0;
    uint32_t predictor_inter = 0;
    uint32_t latent_dim = 0;
    uint32_t action_dim = 0;
    uint32_t image_size = 0;
    uint32_t patch_size = 0;
    uint32_t channels = 0;
    uint32_t encoder_hidden = 0;
    uint32_t encoder_heads = 0;
    uint32_t encoder_inter = 0;

    if (!json_extract_string(config_json, "mode", model->mode, sizeof(model->mode)) ||
        !json_extract_u32(config_json, "encoder_layers", &encoder_layers) ||
        !json_extract_u32(config_json, "predictor_layers", &predictor_layers) ||
        !json_extract_u32(config_json, "predictor_hidden", &predictor_hidden) ||
        !json_extract_u32(config_json, "predictor_heads", &predictor_heads) ||
        !json_extract_u32(config_json, "predictor_inner_dim", &predictor_inner_dim) ||
        !json_extract_u32(config_json, "predictor_inter", &predictor_inter) ||
        !json_extract_u32(config_json, "latent_dim", &latent_dim) ||
        !json_extract_u32(config_json, "action_dim", &action_dim) ||
        !json_extract_u32(config_json, "image_size", &image_size) ||
        !json_extract_u32(config_json, "patch_size", &patch_size) ||
        !json_extract_u32(config_json, "channels", &channels) ||
        !json_extract_u32(config_json, "encoder_hidden", &encoder_hidden) ||
        !json_extract_u32(config_json, "encoder_heads", &encoder_heads) ||
        !json_extract_u32(config_json, "encoder_inter", &encoder_inter)) {
        return false;
    }

    model->has_full_encoder = strcmp(model->mode, "full") == 0;
    model->apply_final_norm_bias = strcmp(model->mode, "full") != 0;
    model->image_size = image_size;
    model->patch_size = patch_size;
    model->channels = channels;
    model->encoder_hidden = encoder_hidden;
    model->encoder_layers = encoder_layers;
    model->encoder_heads = encoder_heads;
    model->encoder_inter = encoder_inter;
    /* Parse optional meta_tokens count (hybrid encoder) */
    uint32_t meta_tokens_count = 0;
    json_extract_u32(config_json, "meta_tokens", &meta_tokens_count);
    model->meta_tokens = (size_t)meta_tokens_count;
    model->encoder_seq_len = (image_size / patch_size) * (image_size / patch_size)
                             + 1U + model->meta_tokens;
    model->latent_dim = latent_dim;
    model->action_dim = action_dim;
    model->predictor_hidden = predictor_hidden;
    model->predictor_layers = predictor_layers;
    model->predictor_heads = predictor_heads;
    model->predictor_inner_dim = predictor_inner_dim;
    model->predictor_inter = predictor_inter;

    size_t config_len = (size_t)read_u32_le(model_data + 4U);
    size_t off = 8U + config_len;
    uint32_t float_caps = MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT;

    if (model->has_full_encoder) {
        if (!cursor_read_f32_vector(model_data, model_len, &off, &model->patch_proj, float_caps) ||
            !cursor_read_f32_vector(model_data, model_len, &off, &model->patch_proj_bias, float_caps) ||
            !cursor_read_f32_vector(model_data, model_len, &off, &model->cls_token, float_caps) ||
            !cursor_read_f32_vector(model_data, model_len, &off, &model->pos_embed, float_caps)) {
            return false;
        }
        model->encoder = (EncoderLayer *)calloc_caps(
            model->encoder_layers,
            sizeof(EncoderLayer),
            MALLOC_CAP_8BIT
        );
        if (model->encoder == NULL) {
            return false;
        }
        for (size_t i = 0; i < model->encoder_layers; ++i) {
            if (!cursor_read_encoder_layer(
                    model_data,
                    model_len,
                    &off,
                    &model->encoder[i],
                    float_caps
                )) {
                return false;
            }
        }
        if (!cursor_read_f32_vector(model_data, model_len, &off, &model->final_norm_weight, float_caps) ||
            !cursor_read_f32_vector(model_data, model_len, &off, &model->final_norm_bias, float_caps)) {
            return false;
        }
    } else if (strcmp(model->mode, "q4-pred") == 0 ||
               strcmp(model->mode, "wanda20-q4") == 0 ||
               strcmp(model->mode, "wanda40-q4") == 0) {
        if (!skip_q4_pred_encoder(model_data, model_len, &off, encoder_layers)) {
            return false;
        }
    } else {
        return false;
    }

    if (!cursor_read_f32_vector(model_data, model_len, &off, &model->predictor_pos_embed, float_caps)) {
        return false;
    }

    model->layers = (PredictorLayer *)calloc_caps(
        model->predictor_layers,
        sizeof(PredictorLayer),
        MALLOC_CAP_8BIT
    );
    if (model->layers == NULL) {
        return false;
    }
    for (size_t i = 0; i < model->predictor_layers; ++i) {
        if (!cursor_read_q4_predictor_layer(
                model_data,
                model_len,
                &off,
                &model->layers[i],
                float_caps
            )) {
            return false;
        }
    }

    if (!cursor_read_f32_vector(model_data, model_len, &off, &model->predictor_norm_weight, float_caps) ||
        !cursor_read_f32_vector(model_data, model_len, &off, &model->predictor_norm_bias, float_caps) ||
        !cursor_read_f32_vector(model_data, model_len, &off, &model->action_conv_weight, float_caps) ||
        !cursor_read_f32_vector(model_data, model_len, &off, &model->action_conv_bias, float_caps) ||
        !cursor_read_f32_vector(model_data, model_len, &off, &model->action_mlp1_weight, float_caps) ||
        !cursor_read_f32_vector(model_data, model_len, &off, &model->action_mlp1_bias, float_caps) ||
        !cursor_read_f32_vector(model_data, model_len, &off, &model->action_mlp2_weight, float_caps) ||
        !cursor_read_f32_vector(model_data, model_len, &off, &model->action_mlp2_bias, float_caps)) {
        return false;
    }

    if (model->has_full_encoder) {
        if (!cursor_read_projection_head(
                model_data,
                model_len,
                &off,
                &model->projector,
                model->encoder_hidden,
                float_caps
            )) {
            return false;
        }
    } else if (!cursor_skip_projection_head(model_data, model_len, &off)) {
        return false;
    }

    if (
        !cursor_read_projection_head(
            model_data,
            model_len,
            &off,
            &model->pred_proj,
            model->predictor_hidden,
            float_caps
        )) {
        return false;
    }

    if (off < model_len &&
        !cursor_read_f32_vector(model_data, model_len, &off, &model->input_proj_weight, float_caps)) {
        return false;
    }
    if (off < model_len &&
        !cursor_read_f32_vector(model_data, model_len, &off, &model->input_proj_bias, float_caps)) {
        return false;
    }
    if (off < model_len &&
        !cursor_read_f32_vector(model_data, model_len, &off, &model->cond_proj_weight, float_caps)) {
        return false;
    }
    if (off < model_len &&
        !cursor_read_f32_vector(model_data, model_len, &off, &model->cond_proj_bias, float_caps)) {
        return false;
    }

    /* Hybrid encoder extras: meta_token, enc_proj_weight, enc_proj_bias */
    if (off < model_len &&
        !cursor_read_f32_vector(model_data, model_len, &off, &model->meta_token, float_caps)) {
        return false;
    }
    if (off < model_len &&
        !cursor_read_f32_vector(model_data, model_len, &off, &model->enc_proj_weight, float_caps)) {
        return false;
    }
    if (off < model_len &&
        !cursor_read_f32_vector(model_data, model_len, &off, &model->enc_proj_bias, float_caps)) {
        return false;
    }

    if (!allocate_scratch(model)) {
        ESP_LOGE(
            TAG,
            "Scratch allocation failed hidden=%lu inner=%lu inter=%lu pred_proj_max=%lu",
            (unsigned long)model->predictor_hidden,
            (unsigned long)model->predictor_inner_dim,
            (unsigned long)model->predictor_inter,
            (unsigned long)model->pred_proj.max_dim
        );
        return false;
    }

    /* Quantize patch_proj to INT8 for PIE-accelerated batch embedding */
    if (model->has_full_encoder && model->patch_proj.len > 0) {
        size_t patch_dim = model->patch_size * model->patch_size * model->channels;
        size_t enc_h = model->encoder_hidden;
        size_t num_patches = (model->image_size / model->patch_size) *
                             (model->image_size / model->patch_size);

        memset(&model->patch_proj_i8, 0, sizeof(model->patch_proj_i8));
        if (quantize_f32_to_int8linear(model->patch_proj.data, enc_h, patch_dim,
                                       &model->patch_proj_i8)) {
            ESP_LOGI(TAG, "Patch proj quantized to INT8: %lux%lu → %lu padded",
                (unsigned long)enc_h, (unsigned long)patch_dim,
                (unsigned long)model->patch_proj_i8.in_features_padded);
        }

        /* Allocate batch patch buffer in PSRAM */
        model->all_patches = (float *)heap_caps_calloc(
            num_patches * patch_dim, sizeof(float),
            MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
        if (model->all_patches) {
            ESP_LOGI(TAG, "Batch patch buffer: %lu patches × %lu dims = %luKB",
                (unsigned long)num_patches, (unsigned long)patch_dim,
                (unsigned long)(num_patches * patch_dim * 4 / 1024));
        }
    }

    return true;
}

static void projection_head_forward(
    const ProjectionHead *head,
    const float *input,
    size_t input_dim,
    PredictorScratch *scratch,
    float *out
) {
    const float *current = input;
    size_t current_dim = input_dim;
    bool use_a = true;

    for (size_t layer_index = 0; layer_index < head->num_layers; ++layer_index) {
        const ProjectionLayer *layer = &head->layers[layer_index];
        if (layer->weight.len == 0 || layer->weight.data == NULL) {
            continue;
        }
        size_t out_dim = current_dim == 0 ? 0 : layer->weight.len / current_dim;
        float *next = use_a ? scratch->proj_tmp_a : scratch->proj_tmp_b;

        matmul_t_into(current, layer->weight.data, 1U, current_dim, out_dim, next);
        add_bias_inplace(next, layer->bias.data, 1U, out_dim);
        if (layer_index + 1U < head->num_layers) {
            for (size_t i = 0; i < out_dim; ++i) {
                next[i] = gelu_scalar(next[i]);
            }
        }

        current = next;
        current_dim = out_dim;
        use_a = !use_a;
    }

    memcpy(out, current, current_dim * sizeof(float));
}

static void log_projection_head_shape(const ProjectionHead *head, size_t input_dim) {
    size_t current_dim = input_dim;
    ESP_LOGI(
        TAG,
        "pred_proj layers=%lu input_dim=%lu max_dim=%lu",
        (unsigned long)head->num_layers,
        (unsigned long)input_dim,
        (unsigned long)head->max_dim
    );
    for (size_t layer_index = 0; layer_index < head->num_layers; ++layer_index) {
        const ProjectionLayer *layer = &head->layers[layer_index];
        if (layer->weight.len == 0 || layer->weight.data == NULL) {
            ESP_LOGI(
                TAG,
                "pred_proj layer %lu weight_len=%lu bias_len=%lu in=%lu out=skip",
                (unsigned long)layer_index,
                (unsigned long)layer->weight.len,
                (unsigned long)layer->bias.len,
                (unsigned long)current_dim
            );
            continue;
        }
        size_t out_dim = current_dim == 0 ? 0 : layer->weight.len / current_dim;
        ESP_LOGI(
            TAG,
            "pred_proj layer %lu weight_len=%lu bias_len=%lu in=%lu out=%lu",
            (unsigned long)layer_index,
            (unsigned long)layer->weight.len,
            (unsigned long)layer->bias.len,
            (unsigned long)current_dim,
            (unsigned long)out_dim
        );
        current_dim = out_dim;
    }
}

static void encode_action(const PredictorModel *model, const float *action, float *out) {
    size_t act_dim = model->action_dim;
    size_t latent = model->latent_dim;
    size_t inter = model->action_mlp1_weight.len == 0 ? latent * 4U :
                  model->action_mlp1_weight.len / act_dim;

    float *conv_out = model->scratch.proj_tmp_a;
    float *h1 = model->scratch.action_hidden;

    if (model->action_conv_weight.len != 0) {
        matmul_t_into(action, model->action_conv_weight.data, 1U, act_dim, act_dim, conv_out);
        add_bias_inplace(conv_out, model->action_conv_bias.data, 1U, act_dim);
    } else {
        memcpy(conv_out, action, act_dim * sizeof(float));
    }

    if (model->action_mlp1_weight.len != 0) {
        matmul_t_into(conv_out, model->action_mlp1_weight.data, 1U, act_dim, inter, h1);
        add_bias_inplace(h1, model->action_mlp1_bias.data, 1U, inter);
        for (size_t i = 0; i < inter; ++i) {
            h1[i] = gelu_scalar(h1[i]);
        }
    } else {
        memset(h1, 0, inter * sizeof(float));
    }

    if (model->action_mlp2_weight.len != 0) {
        matmul_t_into(h1, model->action_mlp2_weight.data, 1U, inter, latent, out);
        add_bias_inplace(out, model->action_mlp2_bias.data, 1U, latent);
    } else {
        memset(out, 0, latent * sizeof(float));
    }
}

static void apply_input_proj(
    const PredictorModel *model,
    const float *seq,
    size_t seq_len,
    float *out
) {
    size_t latent = model->latent_dim;
    size_t hidden = model->predictor_hidden;
    for (size_t token = 0; token < seq_len; ++token) {
        matmul_t_into(
            seq + token * latent,
            model->input_proj_weight.data,
            1U,
            latent,
            hidden,
            out + token * hidden
        );
    }
    add_bias_inplace(out, model->input_proj_bias.data, seq_len, hidden);
}

static void apply_cond_proj(const PredictorModel *model, const float *cond, float *out) {
    matmul_t_into(
        cond,
        model->cond_proj_weight.data,
        1U,
        model->latent_dim,
        model->predictor_hidden,
        out
    );
    add_bias_inplace(out, model->cond_proj_bias.data, 1U, model->predictor_hidden);
}

/* Quantize f32 weight matrix [out_f, in_f] to INT8 per-channel at runtime. */
static bool quantize_f32_to_int8linear(
    const float *weights, size_t out_f, size_t in_f,
    Int8LinearRef *out_ref
) {
    size_t in_pad = (in_f + 15U) & ~15U;
    out_ref->out_features = out_f;
    out_ref->in_features = in_f;
    out_ref->in_features_padded = in_pad;
    out_ref->weights_data = NULL;

    /* Allocate transposed weights in PSRAM */
    out_ref->weights_t = (int8_t *)heap_caps_aligned_alloc(
        16, out_f * in_pad, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
    out_ref->scales.data = (float *)heap_caps_malloc(out_f * sizeof(float), MALLOC_CAP_SPIRAM);
    out_ref->scales.len = out_f;
    if (!out_ref->weights_t || !out_ref->scales.data) return false;

    /* Per-channel quantization: find max abs per output row, quantize */
    for (size_t j = 0; j < out_f; ++j) {
        float max_abs = 0.0f;
        for (size_t k = 0; k < in_f; ++k) {
            float v = fabsf(weights[j * in_f + k]);
            if (v > max_abs) max_abs = v;
        }
        float scale = max_abs / 127.0f;
        out_ref->scales.data[j] = scale;
        float inv_scale = (scale > 1e-12f) ? 1.0f / scale : 0.0f;

        /* Store transposed: weights_t[j * in_pad + k] */
        for (size_t k = 0; k < in_f; ++k) {
            float v = weights[j * in_f + k] * inv_scale;
            int32_t q = (int32_t)roundf(v);
            if (q > 127) q = 127;
            if (q < -128) q = -128;
            out_ref->weights_t[j * in_pad + k] = (int8_t)q;
        }
        /* Zero pad */
        if (in_pad > in_f) {
            memset(out_ref->weights_t + j * in_pad + in_f, 0, in_pad - in_f);
        }
    }
    return true;
}

static void patch_embed_image_into(
    PredictorModel *model,
    const float *image,
    size_t height,
    size_t width,
    float *out
) {
    EncoderScratch *scratch = &model->encoder_scratch;
    size_t hidden = model->encoder_hidden;
    size_t patch = model->patch_size;
    size_t channels = model->channels;
    size_t patches_h = height / patch;
    size_t patches_w = width / patch;
    size_t patch_dim = patch * patch * channels;
    size_t num_patches = patches_h * patches_w;

    /* Extract ALL patches into the pre-allocated batch buffer */
    float *all_patches = model->all_patches;
    for (size_t ph = 0; ph < patches_h; ++ph) {
        for (size_t pw = 0; pw < patches_w; ++pw) {
            size_t patch_idx = ph * patches_w + pw;
            float *dst = all_patches + patch_idx * patch_dim;
            for (size_t c = 0; c < channels; ++c) {
                for (size_t py = 0; py < patch; ++py) {
                    for (size_t px = 0; px < patch; ++px) {
                        size_t img_y = ph * patch + py;
                        size_t img_x = pw * patch + px;
                        dst[c * patch * patch + py * patch + px] =
                            image[(img_y * width + img_x) * channels + c];
                    }
                }
            }
        }
    }

    /* Batched INT8 GEMM: [num_patches, patch_dim] @ [hidden, patch_dim]^T → [num_patches, hidden] */
    if (model->patch_proj_i8.weights_t != NULL) {
        int8linear_forward_into(
            &model->patch_proj_i8,
            all_patches,
            num_patches,
            scratch->row_quant,
            out
        );
        /* Add patch projection bias */
        if (model->patch_proj_bias.len > 0) {
            add_bias_inplace(out, model->patch_proj_bias.data, num_patches, hidden);
        }
    } else {
        /* Fallback: scalar per-patch */
        for (size_t i = 0; i < num_patches; ++i) {
            matmul_t_into(
                all_patches + i * patch_dim,
                model->patch_proj.data,
                1U, patch_dim, hidden,
                out + i * hidden
            );
        }
        if (model->patch_proj_bias.len > 0) {
            add_bias_inplace(out, model->patch_proj_bias.data, num_patches, hidden);
        }
    }
}

static void encoder_layer_forward(
    PredictorModel *model,
    const EncoderLayer *layer,
    float *seq
) {
    EncoderScratch *scratch = &model->encoder_scratch;
    size_t seq_len = model->encoder_seq_len;
    size_t hidden = model->encoder_hidden;
    size_t head_dim = hidden / model->encoder_heads;
    static int layer_idx = 0;

    int64_t t0 = esp_timer_get_time();

    layernorm_into(seq, layer->attn_norm_weight.data, seq_len, hidden, scratch->normed);
    add_bias_inplace(scratch->normed, layer->attn_norm_bias.data, seq_len, hidden);

    int64_t t_norm1 = esp_timer_get_time();

    /* Pre-quantize normed input ONCE for Q/K/V (saves 2x redundant quantization) */
    {
        size_t in_pad = layer->w_q.in_features_padded;
        int8_t *qkv_i8 = (int8_t *)heap_caps_aligned_alloc(
            16, seq_len * in_pad, MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);
        float *qkv_scales = (float *)malloc(seq_len * sizeof(float));

        if (qkv_i8 && qkv_scales) {
            for (size_t row = 0; row < seq_len; ++row) {
                quantize_row_int8(scratch->normed + row * hidden, hidden,
                                  qkv_i8 + row * in_pad, &qkv_scales[row]);
                if (in_pad > hidden)
                    memset(qkv_i8 + row * in_pad + hidden, 0, in_pad - hidden);
            }
            int8linear_forward_prequant(&layer->w_q, qkv_i8, qkv_scales, seq_len, in_pad, scratch->q);
            add_bias_inplace(scratch->q, layer->q_bias.data, seq_len, hidden);
            int8linear_forward_prequant(&layer->w_k, qkv_i8, qkv_scales, seq_len, in_pad, scratch->k);
            add_bias_inplace(scratch->k, layer->k_bias.data, seq_len, hidden);
            int8linear_forward_prequant(&layer->w_v, qkv_i8, qkv_scales, seq_len, in_pad, scratch->v);
            add_bias_inplace(scratch->v, layer->v_bias.data, seq_len, hidden);
            free(qkv_i8);
            free(qkv_scales);
        } else {
            free(qkv_i8); free(qkv_scales);
            int8linear_forward_into(&layer->w_q, scratch->normed, seq_len, scratch->row_quant, scratch->q);
            add_bias_inplace(scratch->q, layer->q_bias.data, seq_len, hidden);
            int8linear_forward_into(&layer->w_k, scratch->normed, seq_len, scratch->row_quant, scratch->k);
            add_bias_inplace(scratch->k, layer->k_bias.data, seq_len, hidden);
            int8linear_forward_into(&layer->w_v, scratch->normed, seq_len, scratch->row_quant, scratch->v);
            add_bias_inplace(scratch->v, layer->v_bias.data, seq_len, hidden);
        }
    }

    int64_t t_qkv = esp_timer_get_time();

    /* L blocks (linear attention) detected by empty Q bias */
    bool use_linear = (layer->q_bias.len == 0);
    if (use_linear) {
        /* Kernel-trick O(nd²) linear attention — no score matrix */
        linear_attention_kernel_trick(
            scratch->q, scratch->k, scratch->v,
            seq_len, model->encoder_heads, head_dim,
            scratch->attn_out
        );
    } else {
        bidirectional_attention_separate(
            scratch->q, scratch->k, scratch->v,
            seq_len, model->encoder_heads, head_dim,
            scratch->scores, scratch->attn_out, false
        );
    }

    int64_t t_attn = esp_timer_get_time();

    int8linear_forward_into(
        &layer->w_o,
        scratch->attn_out,
        seq_len,
        scratch->row_quant,
        scratch->proj
    );
    add_bias_inplace(scratch->proj, layer->o_bias.data, seq_len, hidden);
    for (size_t i = 0; i < seq_len * hidden; ++i) {
        seq[i] += scratch->proj[i];
    }

    int64_t t_oproj = esp_timer_get_time();

    layernorm_into(seq, layer->ffn_norm_weight.data, seq_len, hidden, scratch->normed);
    add_bias_inplace(scratch->normed, layer->ffn_norm_bias.data, seq_len, hidden);
    int8linear_forward_into(
        &layer->ffn_up,
        scratch->normed,
        seq_len,
        scratch->row_quant,
        scratch->ffn_inter
    );
    add_bias_inplace(scratch->ffn_inter, layer->ffn_up_bias.data, seq_len, model->encoder_inter);
    for (size_t i = 0; i < seq_len * model->encoder_inter; ++i) {
        scratch->ffn_inter[i] = gelu_scalar(scratch->ffn_inter[i]);
    }
    int8linear_forward_into(
        &layer->ffn_down,
        scratch->ffn_inter,
        seq_len,
        scratch->row_quant,
        scratch->proj
    );
    add_bias_inplace(scratch->proj, layer->ffn_down_bias.data, seq_len, hidden);
    for (size_t i = 0; i < seq_len * hidden; ++i) {
        seq[i] += scratch->proj[i];
    }

    int64_t t_ffn = esp_timer_get_time();

    ESP_LOGI(TAG, "enc.layer%d breakdown: norm=%.0fms qkv=%.0fms attn=%.0fms oproj=%.0fms ffn=%.0fms total=%.0fms",
        layer_idx,
        (double)(t_norm1 - t0) / 1000.0,
        (double)(t_qkv - t_norm1) / 1000.0,
        (double)(t_attn - t_qkv) / 1000.0,
        (double)(t_oproj - t_attn) / 1000.0,
        (double)(t_ffn - t_oproj) / 1000.0,
        (double)(t_ffn - t0) / 1000.0);
    layer_idx++;
}

bool encode_image(
    PredictorModel *model,
    const float *image,
    size_t height,
    size_t width,
    float *out
) {
    if (!model->has_full_encoder) {
        return false;
    }

    EncoderScratch *scratch = &model->encoder_scratch;
    size_t hidden = model->encoder_hidden;
    size_t seq_len = model->encoder_seq_len;
    size_t meta = model->meta_tokens;
    size_t patch_offset = (1U + meta) * hidden;

    /* Build sequence: [CLS, meta_tokens..., patches...] */
    memcpy(scratch->x, model->cls_token.data, hidden * sizeof(float));
    if (meta > 0 && model->meta_token.len > 0) {
        memcpy(scratch->x + hidden, model->meta_token.data, meta * hidden * sizeof(float));
    }
    int64_t t_pe_start = esp_timer_get_time();
    patch_embed_image_into(model, image, height, width, scratch->x + patch_offset);
    int64_t t_pe_end = esp_timer_get_time();
    ESP_LOGI(TAG, "enc.patch_embed latency: %.1f ms", (double)(t_pe_end - t_pe_start) / 1000.0);
    log_vector_preview("enc.patch0_embed", scratch->x + patch_offset, hidden);
    log_vector_stats("enc.patch0_embed", scratch->x + patch_offset, hidden);
    for (size_t i = 0; i < seq_len * hidden && i < model->pos_embed.len; ++i) {
        scratch->x[i] += model->pos_embed.data[i];
    }
    log_vector_preview("enc.patch0_with_pos", scratch->x + patch_offset, hidden);
    log_vector_stats("enc.patch0_with_pos", scratch->x + patch_offset, hidden);

    for (size_t layer_index = 0; layer_index < model->encoder_layers; ++layer_index) {
        encoder_layer_forward(model, &model->encoder[layer_index], scratch->x);
        if (layer_index == 0U) {
            log_vector_preview("enc.layer0_cls", scratch->x, hidden);
            log_vector_stats("enc.layer0_cls", scratch->x, hidden);
        }
        if ((layer_index & 1U) == 1U) {
            vTaskDelay(pdMS_TO_TICKS(1));
        }
    }

    layernorm_into(scratch->x, model->final_norm_weight.data, 1U, hidden, scratch->cls_norm);
    log_vector_preview("enc.cls_norm", scratch->cls_norm, hidden);
    log_vector_stats("enc.cls_norm", scratch->cls_norm, hidden);

    /* Hybrid encoder output projection (Linear, no activation) */
    float *enc_out = scratch->cls_norm;
    if (model->enc_proj_weight.len > 0) {
        /* out = cls_norm @ weight^T + bias (reuse scratch->normed as temp) */
        size_t out_dim = model->enc_proj_bias.len > 0 ? model->enc_proj_bias.len : hidden;
        for (size_t j = 0; j < out_dim; ++j) {
            float sum = 0.0f;
            for (size_t k = 0; k < hidden; ++k) {
                sum += scratch->cls_norm[k] * model->enc_proj_weight.data[j * hidden + k];
            }
            if (model->enc_proj_bias.len > 0) {
                sum += model->enc_proj_bias.data[j];
            }
            scratch->normed[j] = sum;
        }
        enc_out = scratch->normed;
    }

    projection_head_forward(&model->projector, enc_out, hidden, &model->scratch, out);
    return true;
}

static void predictor_layer_forward(
    const PredictorLayer *layer,
    PredictorScratch *scratch,
    float *seq,
    const float *conditioning,
    size_t seq_len,
    size_t hidden,
    size_t num_heads,
    size_t inner_dim,
    size_t inter
) {
    q4linear_forward_into(&layer->adaln_linear, conditioning, 1U, scratch->mod_vec);
    add_bias_inplace(scratch->mod_vec, layer->adaln_bias.data, 1U, 6U * hidden);

    const float *scale1 = scratch->mod_vec;
    const float *shift1 = scratch->mod_vec + hidden;
    const float *gate1 = scratch->mod_vec + 2U * hidden;
    const float *scale2 = scratch->mod_vec + 3U * hidden;
    const float *shift2 = scratch->mod_vec + 4U * hidden;
    const float *gate2 = scratch->mod_vec + 5U * hidden;

    layernorm_into(seq, layer->attn_norm_weight.data, seq_len, hidden, scratch->normed);
    for (size_t token = 0; token < seq_len; ++token) {
        for (size_t i = 0; i < hidden; ++i) {
            size_t idx = token * hidden + i;
            scratch->modulated[idx] = scratch->normed[idx] * (1.0f + scale1[i]) + shift1[i];
        }
    }

    q4linear_forward_into(&layer->to_qkv, scratch->modulated, seq_len, scratch->qkv);
    bidirectional_attention_from_qkv(
        scratch->qkv,
        seq_len,
        num_heads,
        inner_dim / num_heads,
        scratch->attn_out
    );
    q4linear_forward_into(&layer->attn_out, scratch->attn_out, seq_len, scratch->proj);
    add_bias_inplace(scratch->proj, layer->attn_out_bias.data, seq_len, hidden);

    for (size_t token = 0; token < seq_len; ++token) {
        for (size_t i = 0; i < hidden; ++i) {
            size_t idx = token * hidden + i;
            seq[idx] += gate1[i] * scratch->proj[idx];
        }
    }

    layernorm_into(seq, layer->mlp_norm_weight.data, seq_len, hidden, scratch->normed);
    for (size_t token = 0; token < seq_len; ++token) {
        for (size_t i = 0; i < hidden; ++i) {
            size_t idx = token * hidden + i;
            scratch->modulated[idx] = scratch->normed[idx] * (1.0f + scale2[i]) + shift2[i];
        }
    }

    q4linear_forward_into(&layer->mlp_up, scratch->modulated, seq_len, scratch->ffn_inter);
    add_bias_inplace(scratch->ffn_inter, layer->mlp_up_bias.data, seq_len, inter);
    for (size_t i = 0; i < seq_len * inter; ++i) {
        scratch->ffn_inter[i] = gelu_scalar(scratch->ffn_inter[i]);
    }

    q4linear_forward_into(&layer->mlp_down, scratch->ffn_inter, seq_len, scratch->proj);
    add_bias_inplace(scratch->proj, layer->mlp_down_bias.data, seq_len, hidden);

    for (size_t token = 0; token < seq_len; ++token) {
        for (size_t i = 0; i < hidden; ++i) {
            size_t idx = token * hidden + i;
            seq[idx] += gate2[i] * scratch->proj[idx];
        }
    }
}

void predict_next(
    PredictorModel *model,
    const float *state,
    const float *action,
    float *out
) {
    PredictorScratch *scratch = &model->scratch;
    size_t seq_len = 3U;
    size_t hidden = model->predictor_hidden;
    size_t latent = model->latent_dim;
    bool has_proj = model->input_proj_weight.len != 0;
    size_t seq_dim = has_proj ? latent : hidden;

    encode_action(model, action, scratch->conditioning);

    memset(scratch->seq_raw, 0, seq_len * seq_dim * sizeof(float));
    memcpy(scratch->seq_raw, state, seq_dim * sizeof(float));
    memcpy(scratch->seq_raw + seq_dim, scratch->conditioning, seq_dim * sizeof(float));

    size_t pos_len = model->predictor_pos_embed.len < seq_len * seq_dim ?
                     model->predictor_pos_embed.len :
                     seq_len * seq_dim;
    for (size_t i = 0; i < pos_len; ++i) {
        scratch->seq_raw[i] += model->predictor_pos_embed.data[i];
    }

    if (has_proj) {
        apply_input_proj(model, scratch->seq_raw, seq_len, scratch->seq);
        apply_cond_proj(model, scratch->conditioning, scratch->proj_tmp_a);
        memcpy(scratch->conditioning, scratch->proj_tmp_a, hidden * sizeof(float));
    } else {
        memcpy(scratch->seq, scratch->seq_raw, seq_len * hidden * sizeof(float));
    }

    for (size_t layer_index = 0; layer_index < model->predictor_layers; ++layer_index) {
        predictor_layer_forward(
            &model->layers[layer_index],
            scratch,
            scratch->seq,
            scratch->conditioning,
            seq_len,
            hidden,
            model->predictor_heads,
            model->predictor_inner_dim,
            model->predictor_inter
        );
    }

    layernorm_into(
        scratch->seq,
        model->predictor_norm_weight.data,
        seq_len,
        hidden,
        scratch->normed
    );
    if (model->apply_final_norm_bias && model->predictor_norm_bias.len != 0) {
        add_bias_inplace(scratch->normed, model->predictor_norm_bias.data, seq_len, hidden);
    }

    projection_head_forward(
        &model->pred_proj,
        scratch->normed + 2U * hidden,
        hidden,
        scratch,
        out
    );
}

/*
 * Fused multi-step rollout: runs all predictor layers once over an N×3-token
 * sequence, where N = num_steps.
 *
 * Constructs the fused sequence as:
 *   [z_start, a_0, zeros, z_start, a_1, zeros, ...]
 *
 * Same z_start for all positions — produces parallel hypothesis futures,
 * not an autoregressive chain. The bidirectional attention in each predictor
 * layer naturally attends across all step tokens.
 *
 * Saves ~2× predictor layer cost vs calling predict_next N times.
 *
 * Arguments:
 *   model      -- loaded LEWM model
 *   z_start    -- [latent_dim] initial latent state
 *   actions    -- [num_steps][action_dim] action vectors
 *   num_steps  -- number of rollout steps (must be <= 10 for MAX_PREDICTOR_SEQ_LEN=30)
 *   out        -- [num_steps][latent_dim] output buffer (caller-allocated)
 */
void predict_rollout_fused(
    PredictorModel *model,
    const float *z_start,
    const float *actions,
    size_t num_steps,
    float *out
) {
    PredictorScratch *scratch = &model->scratch;
    size_t hidden = model->predictor_hidden;
    size_t latent = model->latent_dim;
    size_t inter = model->predictor_inter;
    bool has_proj = model->input_proj_weight.len != 0;
    size_t seq_dim = has_proj ? latent : hidden;
    size_t fused_seq_len = num_steps * 3U;

    /*
     * Allocate scratch buffer for action embeddings on the heap.
     * Size: num_steps * hidden (f32, internal SRAM).
     */
    float *action_embeds = (float *)calloc_caps(
        num_steps * hidden, sizeof(float), MALLOC_CAP_INTERNAL);
    if (!action_embeds) {
        /* Fallback: zero all outputs */
        memset(out, 0, num_steps * latent * sizeof(float));
        return;
    }

    /* 1. Encode all actions upfront. */
    for (size_t s = 0; s < num_steps; ++s) {
        encode_action(model, actions + s * model->action_dim, action_embeds + s * hidden);
    }

    /* 2. Build fused sequence: [z_start, a_s, zeros] per step.
     * seq_raw stores [N*3, hidden] elements per token. Token layout: [z_start, a_proj, zeros].
     * Use scratch->proj_tmp_a as temp buffer for projected action embeds (reused per step). */
    memset(scratch->seq_raw, 0, fused_seq_len * hidden * sizeof(float));
    for (size_t s = 0; s < num_steps; ++s) {
        size_t off = s * 3U * hidden;
        /* Token 0: z_start (use first latent elements when has_proj, since z_start is latent-dim) */
        if (has_proj) {
            memcpy(scratch->seq_raw + off, z_start, latent * sizeof(float));
        } else {
            memcpy(scratch->seq_raw + off, z_start, hidden * sizeof(float));
        }
        /* Token 1: action embedding (projected to hidden dim) */
        if (has_proj) {
            apply_cond_proj(model, action_embeds + s * hidden, scratch->proj_tmp_a);
            memcpy(scratch->seq_raw + off + hidden, scratch->proj_tmp_a, hidden * sizeof(float));
        } else {
            memcpy(scratch->seq_raw + off + hidden,
                   action_embeds + s * hidden, hidden * sizeof(float));
        }
        /* Token 2: zeros (already zeroed above) */
    }

    /* 2b. Add positional embeddings (cycle through 3 pattern positions).
     * predictor_pos_embed is [3 * seq_dim] in the binary.
     * We repeat the 3-position pattern for each step. */
    if (model->predictor_pos_embed.len > 0) {
        for (size_t s = 0; s < num_steps; ++s) {
            for (size_t pos = 0; pos < 3U; ++pos) {
                size_t embed_off = pos * seq_dim;
                size_t seq_off = (s * 3U + pos) * seq_dim;
                size_t count = seq_dim;
                if (embed_off + count > model->predictor_pos_embed.len) {
                    count = model->predictor_pos_embed.len - embed_off;
                }
                if (count == 0) break;
                for (size_t j = 0; j < count; ++j) {
                    scratch->seq_raw[seq_off + j] +=
                        model->predictor_pos_embed.data[embed_off + j];
                }
            }
        }
    }

    /* 3. Apply input_proj if bottleneck architecture. */
    float *seq_ptr;
    if (has_proj) {
        apply_input_proj(model, scratch->seq_raw, fused_seq_len, scratch->seq);
        /* Conditioning: use first action embed projected to hidden */
        apply_cond_proj(model, action_embeds, scratch->proj_tmp_a);
        memcpy(scratch->conditioning, scratch->proj_tmp_a, hidden * sizeof(float));
        seq_ptr = scratch->seq;
    } else {
        memcpy(scratch->seq, scratch->seq_raw, fused_seq_len * hidden * sizeof(float));
        memcpy(scratch->conditioning, action_embeds, hidden * sizeof(float));
        seq_ptr = scratch->seq;
    }

    /* 4. Run all predictor layers once over the fused sequence. */
    for (size_t layer_index = 0; layer_index < model->predictor_layers; ++layer_index) {
        predictor_layer_forward(
            &model->layers[layer_index],
            scratch,
            seq_ptr,
            scratch->conditioning,
            fused_seq_len,
            hidden,
            model->predictor_heads,
            model->predictor_inner_dim,
            inter
        );
    }

    /* 5. Final norm. */
    layernorm_into(
        seq_ptr,
        model->predictor_norm_weight.data,
        fused_seq_len,
        hidden,
        scratch->normed
    );
    if (model->apply_final_norm_bias && model->predictor_norm_bias.len != 0) {
        add_bias_inplace(scratch->normed, model->predictor_norm_bias.data,
                         fused_seq_len, hidden);
    }

    /* 6. Extract targets at positions 2, 5, 8, ... and project each.
     * For step s: target is at seq_ptr[(s*3 + 2) * hidden]. */
    for (size_t s = 0; s < num_steps; ++s) {
        size_t target_off = (s * 3U + 2U) * hidden;
        projection_head_forward(
            &model->pred_proj,
            scratch->normed + target_off,
            hidden,
            scratch,
            out + s * latent
        );
    }

    free(action_embeds);
}

static void fill_deterministic_vector(float *out, size_t len, uint32_t seed) {
    for (size_t idx = 0; idx < len; ++idx) {
        uint32_t mixed = seed * 1664525U + ((uint32_t)idx * 1013904223U);
        int32_t centered = (int32_t)(mixed % 2001U) - 1000;
        out[idx] = (float)centered / 1000.0f;
    }
}

static void log_vector_preview(const char *label, const float *values, size_t len) {
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

static void log_vector_stats(const char *label, const float *values, size_t len) {
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

static void probe_projection_head(
    const ProjectionHead *head,
    const float *input,
    size_t input_dim,
    PredictorScratch *scratch
) {
    const float *current = input;
    size_t current_dim = input_dim;
    bool use_a = true;

    for (size_t layer_index = 0; layer_index < head->num_layers; ++layer_index) {
        const ProjectionLayer *layer = &head->layers[layer_index];
        if (layer->weight.len == 0 || layer->weight.data == NULL) {
            continue;
        }
        size_t out_dim = current_dim == 0 ? 0 : layer->weight.len / current_dim;
        float *next = use_a ? scratch->proj_tmp_a : scratch->proj_tmp_b;
        char label[64];

        matmul_t_into(current, layer->weight.data, 1U, current_dim, out_dim, next);
        add_bias_inplace(next, layer->bias.data, 1U, out_dim);
        snprintf(label, sizeof(label), "probe.pred_proj_layer%lu_pre_gelu", (unsigned long)layer_index);
        log_vector_preview(label, next, out_dim);
        log_vector_stats(label, next, out_dim);

        if (layer_index + 1U < head->num_layers) {
            for (size_t i = 0; i < out_dim; ++i) {
                next[i] = gelu_scalar(next[i]);
            }
            snprintf(label, sizeof(label), "probe.pred_proj_layer%lu_post_gelu", (unsigned long)layer_index);
            log_vector_preview(label, next, out_dim);
            log_vector_stats(label, next, out_dim);
        }

        current = next;
        current_dim = out_dim;
        use_a = !use_a;
    }
}

static void log_boot_state(void) {
    ESP_LOGI(TAG, "=== Synapse LEWM serial bring-up ===");
    ESP_LOGI(TAG, "Free heap: %lu bytes", (unsigned long)esp_get_free_heap_size());
    ESP_LOGI(
        TAG,
        "Free PSRAM: %lu bytes",
        (unsigned long)heap_caps_get_free_size(MALLOC_CAP_SPIRAM)
    );
}

static void relax_task_watchdog_for_smoke(void) {
    esp_task_wdt_config_t twdt_config = {
        .timeout_ms = 15000,
        .idle_core_mask = (1U << 1),
        .trigger_panic = false,
    };
    esp_err_t err = esp_task_wdt_reconfigure(&twdt_config);
    if (err != ESP_OK) {
        ESP_LOGW(TAG, "TWDT reconfigure failed: %s", esp_err_to_name(err));
    } else {
        ESP_LOGI(TAG, "TWDT reconfigured for CPU0-heavy smoke inference.");
    }
}

__attribute__((unused))
static void run_full_encode_smoke(PredictorModel *model) {
    size_t image_len = model->image_size * model->image_size * model->channels;
    float *image = (float *)calloc_caps(image_len, sizeof(float), MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
    float *latent = (float *)calloc_caps(model->latent_dim, sizeof(float), MALLOC_CAP_INTERNAL);
    float *action = (float *)calloc_caps(model->action_dim, sizeof(float), MALLOC_CAP_INTERNAL);
    float *next = (float *)calloc_caps(model->latent_dim, sizeof(float), MALLOC_CAP_INTERNAL);

    if (image == NULL || latent == NULL || action == NULL || next == NULL) {
        ESP_LOGE(TAG, "Failed to allocate full encode smoke-test buffers.");
        free(image);
        free(latent);
        free(action);
        free(next);
        return;
    }

    fill_deterministic_vector(image, image_len, 7U);
    fill_deterministic_vector(action, model->action_dim, 101U);

    int64_t started_us = esp_timer_get_time();
    if (!encode_image(model, image, model->image_size, model->image_size, latent)) {
        ESP_LOGE(TAG, "encode_image failed for full smoke path.");
        free(image);
        free(latent);
        free(action);
        free(next);
        return;
    }
    int64_t elapsed_us = esp_timer_get_time() - started_us;
    ESP_LOGI(TAG, "encode latency: %.3f ms", (double)elapsed_us / 1000.0);
    log_vector_preview("encode", latent, model->latent_dim);
    log_vector_stats("encode", latent, model->latent_dim);
    vTaskDelay(pdMS_TO_TICKS(1));

    started_us = esp_timer_get_time();
    predict_next(model, latent, action, next);
    elapsed_us = esp_timer_get_time() - started_us;
    ESP_LOGI(TAG, "encode+predict_next latency: %.3f ms", (double)elapsed_us / 1000.0);
    log_vector_preview("encode_predict_next", next, model->latent_dim);
    log_vector_stats("encode_predict_next", next, model->latent_dim);
    vTaskDelay(pdMS_TO_TICKS(1));

    free(image);
    free(latent);
    free(action);
    free(next);
}

__attribute__((unused))
static void run_predictor_smoke(PredictorModel *model) {
    float *state = (float *)calloc_caps(model->latent_dim, sizeof(float), MALLOC_CAP_INTERNAL);
    float *action = (float *)calloc_caps(model->action_dim, sizeof(float), MALLOC_CAP_INTERNAL);
    float *next = (float *)calloc_caps(model->latent_dim, sizeof(float), MALLOC_CAP_INTERNAL);
    float *rollout_state =
        (float *)calloc_caps(model->latent_dim, sizeof(float), MALLOC_CAP_INTERNAL);
    float *action_embed =
        (float *)calloc_caps(model->latent_dim, sizeof(float), MALLOC_CAP_INTERNAL);
    float *conditioning =
        (float *)calloc_caps(model->predictor_hidden, sizeof(float), MALLOC_CAP_INTERNAL);
    float *layer0_adaln =
        (float *)calloc_caps(6U * model->predictor_hidden, sizeof(float), MALLOC_CAP_INTERNAL);

    if (state == NULL || action == NULL || next == NULL || rollout_state == NULL ||
        action_embed == NULL || conditioning == NULL || layer0_adaln == NULL) {
        ESP_LOGE(TAG, "Failed to allocate predictor smoke-test buffers.");
        free(state);
        free(action);
        free(next);
        free(rollout_state);
        free(action_embed);
        free(conditioning);
        free(layer0_adaln);
        return;
    }

    fill_deterministic_vector(state, model->latent_dim, 11U);
    fill_deterministic_vector(action, model->action_dim, 101U);

    encode_action(model, action, action_embed);
    log_vector_preview("probe.action_embed", action_embed, model->latent_dim);
    log_vector_stats("probe.action_embed", action_embed, model->latent_dim);

    if (model->cond_proj_weight.len != 0) {
        apply_cond_proj(model, action_embed, conditioning);
    } else {
        memcpy(conditioning, action_embed, model->predictor_hidden * sizeof(float));
    }
    log_vector_preview("probe.conditioning", conditioning, model->predictor_hidden);
    log_vector_stats("probe.conditioning", conditioning, model->predictor_hidden);

    q4linear_forward_into(&model->layers[0].adaln_linear, conditioning, 1U, layer0_adaln);
    add_bias_inplace(layer0_adaln, model->layers[0].adaln_bias.data, 1U, 6U * model->predictor_hidden);
    log_vector_preview("probe.layer0_adaln", layer0_adaln, 6U * model->predictor_hidden);
    log_vector_stats("probe.layer0_adaln", layer0_adaln, 6U * model->predictor_hidden);

    int64_t started_us = esp_timer_get_time();
    predict_next(model, state, action, next);
    int64_t elapsed_us = esp_timer_get_time() - started_us;

    ESP_LOGI(
        TAG,
        "predict_next latency: %.3f ms",
        (double)elapsed_us / 1000.0
    );
    log_vector_preview("probe.final_target", model->scratch.normed + 2U * model->predictor_hidden, model->predictor_hidden);
    log_vector_stats("probe.final_target", model->scratch.normed + 2U * model->predictor_hidden, model->predictor_hidden);
    probe_projection_head(
        &model->pred_proj,
        model->scratch.normed + 2U * model->predictor_hidden,
        model->predictor_hidden,
        &model->scratch
    );
    log_vector_preview("predict_next", next, model->latent_dim);
    log_vector_stats("predict_next", next, model->latent_dim);
    vTaskDelay(pdMS_TO_TICKS(1));

    memcpy(rollout_state, state, model->latent_dim * sizeof(float));
    float *prev_step = (float *)calloc_caps(model->latent_dim, sizeof(float), MALLOC_CAP_INTERNAL);
    for (size_t step = 0; step < 3U; ++step) {
        fill_deterministic_vector(action, model->action_dim, 101U + (uint32_t)step * 17U);
        started_us = esp_timer_get_time();
        predict_next(model, rollout_state, action, next);
        elapsed_us = esp_timer_get_time() - started_us;
        float cos = step > 0 ? cosine_similarity(prev_step, next, model->latent_dim) : 0.0f;
        ESP_LOGI(TAG, "rollout step %lu: %.3f ms cos_vs_prev=%.6f",
            (unsigned long)step, (double)elapsed_us / 1000.0, cos);
        log_vector_stats("rollout", next, model->latent_dim);
        memcpy(prev_step, next, model->latent_dim * sizeof(float));
        memcpy(rollout_state, next, model->latent_dim * sizeof(float));
        vTaskDelay(pdMS_TO_TICKS(1));
    }
    free(prev_step);

    free(state);
    free(action);
    free(next);
    free(rollout_state);
    free(action_embed);
    free(conditioning);
    free(layer0_adaln);
}

/**
 * Load an LQ40 model blob into a PredictorModel.
 * Returns true on success.  The caller must keep model_data alive
 * because Q4 weight refs point directly into it.
 */
static bool load_model(const uint8_t *model_data, size_t model_len,
                        PredictorModel *model) {
    if (model_len < 8U || memcmp(model_data, "LQ40", 4U) != 0) {
        ESP_LOGE(TAG, "Embedded model is missing or not an LQ40 blob.");
        ESP_LOGE(TAG, "Replace main/model.bin with an exported LEWM binary before flashing.");
        return false;
    }

    uint32_t config_len = read_u32_le(model_data + 4U);
    if (model_len < 8U + config_len) {
        ESP_LOGE(TAG, "Embedded LQ40 config is truncated.");
        return false;
    }

    char *config_json = (char *)malloc((size_t)config_len + 1U);
    if (config_json == NULL) {
        ESP_LOGE(TAG, "Failed to allocate config buffer.");
        return false;
    }

    memcpy(config_json, model_data + 8U, config_len);
    config_json[config_len] = '\0';

    memset(model, 0, sizeof(*model));
    if (!parse_predictor_model(model_data, model_len, config_json, model)) {
        ESP_LOGE(TAG, "Failed to parse predictor payload from embedded model.");
        free(config_json);
        return false;
    }

    ESP_LOGI(TAG, "Embedded model bytes: %lu", (unsigned long)model_len);
    ESP_LOGI(TAG, "LQ40 mode: %s", model->mode);
    ESP_LOGI(
        TAG,
        "Predictor config: latent=%lu action=%lu hidden=%lu layers=%lu heads=%lu inner=%lu inter=%lu",
        (unsigned long)model->latent_dim,
        (unsigned long)model->action_dim,
        (unsigned long)model->predictor_hidden,
        (unsigned long)model->predictor_layers,
        (unsigned long)model->predictor_heads,
        (unsigned long)model->predictor_inner_dim,
        (unsigned long)model->predictor_inter
    );
    if (model->has_full_encoder) {
        ESP_LOGI(
            TAG,
            "Encoder config: image=%lux%lu patch=%lu channels=%lu hidden=%lu layers=%lu heads=%lu inter=%lu seq=%lu",
            (unsigned long)model->image_size,
            (unsigned long)model->image_size,
            (unsigned long)model->patch_size,
            (unsigned long)model->channels,
            (unsigned long)model->encoder_hidden,
            (unsigned long)model->encoder_layers,
            (unsigned long)model->encoder_heads,
            (unsigned long)model->encoder_inter,
            (unsigned long)model->encoder_seq_len
        );
    }
    log_projection_head_shape(&model->pred_proj, model->predictor_hidden);
    ESP_LOGI(
        TAG,
        "Scratch footprint: seq=%luB qkv=%luB ffn=%luB",
        (unsigned long)(3U * model->predictor_hidden * sizeof(float)),
        (unsigned long)(3U * 3U * model->predictor_inner_dim * sizeof(float)),
        (unsigned long)(3U * model->predictor_inter * sizeof(float))
    );

    free(config_json);
    return true;
}

/* ------------------------------------------------------------------ */
/* WiFi credentials -- edit before flashing                            */
/* ------------------------------------------------------------------ */
#define WIFI_SSID "FiberHGW_ZY1B56"
#define WIFI_PASS "YvtHq3vPAX9U"

#include "wifi.h"
#include "http_server.h"

void app_main(void) {
    log_boot_state();
    relax_task_watchdog_for_smoke();

    /* 1. Load model from embedded flash blob */
    static PredictorModel model;
    size_t model_len = (size_t)(_binary_model_bin_end - _binary_model_bin_start);
    if (!load_model(_binary_model_bin_start, model_len, &model)) {
        ESP_LOGE(TAG, "Model load failed -- halting.");
        return;
    }

    /* PIE SIMD self-test */
    if (pie_self_test() != 0) {
        ESP_LOGE(TAG, "PIE self-test FAILED -- halting.");
        return;
    }

    /* Initialize dual-core attention worker on Core 1 */
    dual_core_init();

    /* Initialize GELU lookup table */
    gelu_lut_init();
    ESP_LOGI(TAG, "GELU LUT initialized (%d entries, [%.1f, %.1f])",
             GELU_LUT_SIZE, GELU_LUT_MIN, GELU_LUT_MAX);

    /* Run smoke tests to get baseline timing */
    run_predictor_smoke(&model);
    if (model.has_full_encoder) {
        run_full_encode_smoke(&model);
    }

    /* 2. Connect to WiFi */
    ESP_LOGI(TAG, "Connecting to WiFi \"%s\" ...", WIFI_SSID);
    if (wifi_init_sta(WIFI_SSID, WIFI_PASS) != ESP_OK) {
        ESP_LOGE(TAG, "WiFi connection failed -- halting.");
        return;
    }
    ESP_LOGI(TAG, "WiFi connected: %s", wifi_get_ip());

    /* 3. Start HTTP inference server */
    ServerConfig cfg = {
        .model            = &model,
        .latent_dim       = model.latent_dim,
        .action_dim       = model.action_dim,
        .predictor_layers = model.predictor_layers,
        .has_encoder      = model.has_full_encoder,
    };
    memcpy(cfg.mode, model.mode, sizeof(cfg.mode));
    cfg.mode[sizeof(cfg.mode) - 1] = '\0';

    httpd_handle_t server = start_inference_server(&cfg);
    if (!server) {
        ESP_LOGE(TAG, "HTTP server failed to start -- halting.");
        return;
    }

    ESP_LOGI(TAG, "=== Ready: http://%s/ ===", wifi_get_ip());

    /* 4. Main loop: heartbeat + watchdog feed */
    while (1) {
        vTaskDelay(pdMS_TO_TICKS(10000));
        ESP_LOGI(TAG, "alive | heap=%lu psram=%lu",
                 (unsigned long)esp_get_free_heap_size(),
                 (unsigned long)heap_caps_get_free_size(MALLOC_CAP_SPIRAM));
    }
}
