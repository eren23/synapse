/*
 * model_loader.c -- LQ40 binary model parsing and loading.
 *
 * Extracted from app_main.c -- computation logic is unchanged.
 */

#include "model_loader.h"
#include "inference.h"
#include "kernels.h"
#include "pie_gemv.h"

#include <ctype.h>
#include <math.h>
#include <string.h>

#include "esp_heap_caps.h"
#include "esp_log.h"

static const char *TAG = "model-loader";

/* ================================================================== */
/* Binary helpers                                                      */
/* ================================================================== */

static uint32_t read_u32_le(const uint8_t *ptr) {
    return ((uint32_t)ptr[0]) |
           ((uint32_t)ptr[1] << 8) |
           ((uint32_t)ptr[2] << 16) |
           ((uint32_t)ptr[3] << 24);
}

/* ================================================================== */
/* JSON helpers                                                        */
/* ================================================================== */

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
    if (start == NULL) return false;

    start = strchr(start, ':');
    if (start == NULL) return false;
    start = skip_ws(start + 1);
    if (*start != '"') return false;
    start++;

    const char *end = strchr(start, '"');
    if (end == NULL) return false;

    size_t copy_len = (size_t)(end - start);
    if (copy_len + 1 > out_len) return false;

    memcpy(out, start, copy_len);
    out[copy_len] = '\0';
    return true;
}

static bool json_extract_u32(const char *json, const char *key, uint32_t *value) {
    char needle[64];
    snprintf(needle, sizeof(needle), "\"%s\"", key);

    const char *start = strstr(json, needle);
    if (start == NULL) return false;

    start = strchr(start, ':');
    if (start == NULL) return false;
    start = skip_ws(start + 1);

    char *end = NULL;
    unsigned long parsed = strtoul(start, &end, 10);
    if (end == start) return false;

    *value = (uint32_t)parsed;
    return true;
}

/* ================================================================== */
/* Cursor-based binary readers                                         */
/* ================================================================== */

static bool cursor_read_f32_vector(
    const uint8_t *data,
    size_t data_len,
    size_t *off,
    FloatBuffer *buffer,
    uint32_t caps
) {
    if (*off + 4 > data_len) return false;
    uint32_t len = read_u32_le(data + *off);
    *off += 4;
    if (*off + (size_t)len * 4 > data_len) return false;
    bool ok = copy_f32_payload(buffer, data + *off, len, caps);
    *off += (size_t)len * 4;
    return ok;
}

static bool cursor_skip_f32_vector(const uint8_t *data, size_t data_len, size_t *off) {
    if (*off + 4 > data_len) return false;
    uint32_t len = read_u32_le(data + *off);
    *off += 4;
    if (*off + (size_t)len * 4 > data_len) return false;
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
    if (*off + 16 > data_len) return false;

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
    if (*off + bitmap_bytes > data_len) return false;
    linear->bitmap = data + *off;
    *off += bitmap_bytes;

    size_t block_bytes = (size_t)nonzero_count * 20U;
    if (*off + block_bytes > data_len) return false;
    linear->blocks_data = data + *off;
    *off += block_bytes;

    linear->row_nz_starts =
        (uint32_t *)lewm_calloc(linear->out_features + 1U, sizeof(uint32_t), MALLOC_CAP_8BIT);
    if (linear->row_nz_starts == NULL) return false;

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
    /* Suppress unused warnings */
    (void)out_features;
    (void)in_features;
    return true;
}

static bool cursor_read_int8_linear(
    const uint8_t *data,
    size_t data_len,
    size_t *off,
    Int8LinearRef *linear,
    uint32_t caps
) {
    if (*off + 12 > data_len) return false;

    linear->out_features = read_u32_le(data + *off);
    *off += 4;
    linear->in_features = read_u32_le(data + *off);
    *off += 4;
    uint32_t weights_len = read_u32_le(data + *off);
    *off += 4;
    if (*off + weights_len > data_len) return false;
    linear->weights_data = (const int8_t *)(data + *off);
    *off += weights_len;

    if (*off + 4 > data_len) return false;
    uint32_t scales_len = read_u32_le(data + *off);
    *off += 4;
    if (*off + (size_t)scales_len * 4 > data_len) return false;
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
    if (*off + 4 > data_len) return false;
    uint32_t num_layers = read_u32_le(data + *off);
    *off += 4;
    for (uint32_t i = 0; i < num_layers; ++i) {
        if (!cursor_skip_f32_vector(data, data_len, off)) return false;
        if (!cursor_skip_f32_vector(data, data_len, off)) return false;
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
    if (*off + 4 > data_len) return false;
    head->num_layers = read_u32_le(data + *off);
    *off += 4;
    head->max_dim = input_dim;
    if (head->num_layers == 0) {
        head->layers = NULL;
        return true;
    }

    head->layers =
        (ProjectionLayer *)lewm_calloc(head->num_layers, sizeof(ProjectionLayer), MALLOC_CAP_8BIT);
    if (head->layers == NULL) return false;

    size_t current_dim = input_dim;
    for (size_t i = 0; i < head->num_layers; ++i) {
        if (!cursor_read_f32_vector(data, data_len, off, &head->layers[i].weight, caps)) return false;
        if (!cursor_read_f32_vector(data, data_len, off, &head->layers[i].bias, caps)) return false;
        if (head->layers[i].weight.len == 0 || head->layers[i].weight.data == NULL) continue;
        if (current_dim == 0 || head->layers[i].weight.len % current_dim != 0) return false;
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
    if (!cursor_read_q4_linear(data, data_len, off, &layer->adaln_linear)) return false;
    if (!cursor_read_f32_vector(data, data_len, off, &layer->adaln_bias, float_caps)) return false;
    if (!cursor_read_q4_linear(data, data_len, off, &layer->to_qkv)) return false;
    if (!cursor_read_q4_linear(data, data_len, off, &layer->attn_out)) return false;
    if (!cursor_read_f32_vector(data, data_len, off, &layer->attn_out_bias, float_caps)) return false;
    if (!cursor_read_f32_vector(data, data_len, off, &layer->attn_norm_weight, float_caps)) return false;
    if (!cursor_read_f32_vector(data, data_len, off, &layer->attn_norm_bias, float_caps)) return false;
    if (!cursor_read_f32_vector(data, data_len, off, &layer->mlp_norm_weight, float_caps)) return false;
    if (!cursor_read_f32_vector(data, data_len, off, &layer->mlp_norm_bias, float_caps)) return false;
    if (!cursor_read_q4_linear(data, data_len, off, &layer->mlp_up)) return false;
    if (!cursor_read_f32_vector(data, data_len, off, &layer->mlp_up_bias, float_caps)) return false;
    if (!cursor_read_q4_linear(data, data_len, off, &layer->mlp_down)) return false;
    return cursor_read_f32_vector(data, data_len, off, &layer->mlp_down_bias, float_caps);
}

static bool cursor_read_encoder_layer(
    const uint8_t *data,
    size_t data_len,
    size_t *off,
    EncoderLayer *layer,
    uint32_t float_caps
) {
    if (!cursor_read_f32_vector(data, data_len, off, &layer->attn_norm_weight, float_caps)) return false;
    if (!cursor_read_f32_vector(data, data_len, off, &layer->attn_norm_bias, float_caps)) return false;
    if (!cursor_read_int8_linear(data, data_len, off, &layer->w_q, float_caps)) return false;
    if (!cursor_read_f32_vector(data, data_len, off, &layer->q_bias, float_caps)) return false;
    if (!cursor_read_int8_linear(data, data_len, off, &layer->w_k, float_caps)) return false;
    if (!cursor_read_f32_vector(data, data_len, off, &layer->k_bias, float_caps)) return false;
    if (!cursor_read_int8_linear(data, data_len, off, &layer->w_v, float_caps)) return false;
    if (!cursor_read_f32_vector(data, data_len, off, &layer->v_bias, float_caps)) return false;
    if (!cursor_read_int8_linear(data, data_len, off, &layer->w_o, float_caps)) return false;
    if (!cursor_read_f32_vector(data, data_len, off, &layer->o_bias, float_caps)) return false;
    if (!cursor_read_f32_vector(data, data_len, off, &layer->ffn_norm_weight, float_caps)) return false;
    if (!cursor_read_f32_vector(data, data_len, off, &layer->ffn_norm_bias, float_caps)) return false;
    if (!cursor_read_int8_linear(data, data_len, off, &layer->ffn_up, float_caps)) return false;
    if (!cursor_read_f32_vector(data, data_len, off, &layer->ffn_up_bias, float_caps)) return false;
    if (!cursor_read_int8_linear(data, data_len, off, &layer->ffn_down, float_caps)) return false;
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
            if (!cursor_skip_f32_vector(data, data_len, off)) return false;
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

/* ================================================================== */
/* Quantize f32 weights to INT8 at runtime (for patch_proj)            */
/* ================================================================== */

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

/* ================================================================== */
/* Scratch allocation                                                  */
/* ================================================================== */

static bool allocate_scratch(PredictorModel *model) {
    PredictorScratch *scratch = &model->scratch;
    EncoderScratch *encoder_scratch = &model->encoder_scratch;
    size_t max_seq_len = MAX_PREDICTOR_SEQ_LEN;
    size_t hidden = model->predictor_hidden;
    size_t inner = model->predictor_inner_dim;
    size_t inter = model->predictor_inter;
    size_t seq_raw_dim = hidden;
    size_t proj_tmp_dim = model->pred_proj.max_dim > model->projector.max_dim ?
                          model->pred_proj.max_dim :
                          model->projector.max_dim;

    scratch->seq_raw =
        (float *)lewm_calloc(max_seq_len * seq_raw_dim, sizeof(float), MALLOC_CAP_INTERNAL);
    scratch->seq = (float *)lewm_calloc(max_seq_len * hidden, sizeof(float), MALLOC_CAP_INTERNAL);
    scratch->conditioning = (float *)lewm_calloc(hidden, sizeof(float), MALLOC_CAP_INTERNAL);
    scratch->action_hidden = (float *)lewm_calloc(inter, sizeof(float), MALLOC_CAP_INTERNAL);
    scratch->mod_vec = (float *)lewm_calloc(6U * hidden, sizeof(float), MALLOC_CAP_INTERNAL);
    scratch->normed = (float *)lewm_calloc(max_seq_len * hidden, sizeof(float), MALLOC_CAP_INTERNAL);
    scratch->modulated =
        (float *)lewm_calloc(max_seq_len * hidden, sizeof(float), MALLOC_CAP_INTERNAL);
    scratch->qkv =
        (float *)lewm_calloc(max_seq_len * 3U * inner, sizeof(float), MALLOC_CAP_INTERNAL);
    scratch->attn_out =
        (float *)lewm_calloc(max_seq_len * inner, sizeof(float), MALLOC_CAP_INTERNAL);
    scratch->proj = (float *)lewm_calloc(max_seq_len * hidden, sizeof(float), MALLOC_CAP_INTERNAL);
    scratch->ffn_inter =
        (float *)lewm_calloc(max_seq_len * inter, sizeof(float), MALLOC_CAP_INTERNAL);
    scratch->proj_tmp_a =
        (float *)lewm_calloc(proj_tmp_dim, sizeof(float), MALLOC_CAP_INTERNAL);
    scratch->proj_tmp_b =
        (float *)lewm_calloc(proj_tmp_dim, sizeof(float), MALLOC_CAP_INTERNAL);

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

    if (!predictor_ok) return false;

    if (!model->has_full_encoder) return true;

    size_t enc_seq_len = model->encoder_seq_len;
    size_t enc_hidden = model->encoder_hidden;
    size_t enc_inner = model->encoder_hidden;
    size_t enc_inter = model->encoder_inter;
    size_t patch_dim = model->patch_size * model->patch_size * model->channels;
    size_t row_quant_dim = patch_dim;
    if (enc_hidden > row_quant_dim) row_quant_dim = enc_hidden;
    if (enc_inter > row_quant_dim) row_quant_dim = enc_inter;

    uint32_t enc_caps = MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT;
    encoder_scratch->x =
        (float *)lewm_calloc(enc_seq_len * enc_hidden, sizeof(float), enc_caps);
    encoder_scratch->normed =
        (float *)lewm_calloc(enc_seq_len * enc_hidden, sizeof(float), enc_caps);
    encoder_scratch->q =
        (float *)lewm_calloc(enc_seq_len * enc_inner, sizeof(float), enc_caps);
    encoder_scratch->k =
        (float *)lewm_calloc(enc_seq_len * enc_inner, sizeof(float), enc_caps);
    encoder_scratch->v =
        (float *)lewm_calloc(enc_seq_len * enc_inner, sizeof(float), enc_caps);
    encoder_scratch->attn_out =
        (float *)lewm_calloc(enc_seq_len * enc_inner, sizeof(float), enc_caps);
    encoder_scratch->proj =
        (float *)lewm_calloc(enc_seq_len * enc_hidden, sizeof(float), enc_caps);
    encoder_scratch->ffn_inter =
        (float *)lewm_calloc(enc_seq_len * enc_inter, sizeof(float), enc_caps);
    encoder_scratch->patch = (float *)lewm_calloc(patch_dim, sizeof(float), enc_caps);
    encoder_scratch->scores =
        (float *)lewm_calloc(enc_seq_len, sizeof(float), MALLOC_CAP_INTERNAL);
    encoder_scratch->cls_norm =
        (float *)lewm_calloc(enc_hidden, sizeof(float), MALLOC_CAP_INTERNAL);
    encoder_scratch->row_quant =
        (int8_t *)lewm_calloc(row_quant_dim, sizeof(int8_t), MALLOC_CAP_INTERNAL);

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

/* ================================================================== */
/* Parse the full predictor model                                      */
/* ================================================================== */

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
        model->encoder = (EncoderLayer *)lewm_calloc(
            model->encoder_layers, sizeof(EncoderLayer), MALLOC_CAP_8BIT);
        if (model->encoder == NULL) return false;
        for (size_t i = 0; i < model->encoder_layers; ++i) {
            if (!cursor_read_encoder_layer(model_data, model_len, &off, &model->encoder[i], float_caps)) {
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
        if (!skip_q4_pred_encoder(model_data, model_len, &off, encoder_layers)) return false;
    } else {
        return false;
    }

    if (!cursor_read_f32_vector(model_data, model_len, &off, &model->predictor_pos_embed, float_caps)) {
        return false;
    }

    model->layers = (PredictorLayer *)lewm_calloc(
        model->predictor_layers, sizeof(PredictorLayer), MALLOC_CAP_8BIT);
    if (model->layers == NULL) return false;
    for (size_t i = 0; i < model->predictor_layers; ++i) {
        if (!cursor_read_q4_predictor_layer(model_data, model_len, &off, &model->layers[i], float_caps)) {
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
        if (!cursor_read_projection_head(model_data, model_len, &off, &model->projector,
                                         model->encoder_hidden, float_caps)) {
            return false;
        }
    } else if (!cursor_skip_projection_head(model_data, model_len, &off)) {
        return false;
    }

    if (!cursor_read_projection_head(model_data, model_len, &off, &model->pred_proj,
                                     model->predictor_hidden, float_caps)) {
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
        ESP_LOGE(TAG,
            "Scratch allocation failed hidden=%lu inner=%lu inter=%lu pred_proj_max=%lu",
            (unsigned long)model->predictor_hidden,
            (unsigned long)model->predictor_inner_dim,
            (unsigned long)model->predictor_inter,
            (unsigned long)model->pred_proj.max_dim);
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
            ESP_LOGI(TAG, "Patch proj quantized to INT8: %lux%lu -> %lu padded",
                (unsigned long)enc_h, (unsigned long)patch_dim,
                (unsigned long)model->patch_proj_i8.in_features_padded);
        }

        /* Allocate batch patch buffer in PSRAM */
        model->all_patches = (float *)heap_caps_calloc(
            num_patches * patch_dim, sizeof(float),
            MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
        if (model->all_patches) {
            ESP_LOGI(TAG, "Batch patch buffer: %lu patches x %lu dims = %luKB",
                (unsigned long)num_patches, (unsigned long)patch_dim,
                (unsigned long)(num_patches * patch_dim * 4 / 1024));
        }
    }

    return true;
}

/* ================================================================== */
/* Public: load_model                                                  */
/* ================================================================== */

bool load_model(const uint8_t *model_data, size_t model_len,
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
    ESP_LOGI(TAG,
        "Predictor config: latent=%lu action=%lu hidden=%lu layers=%lu heads=%lu inner=%lu inter=%lu",
        (unsigned long)model->latent_dim,
        (unsigned long)model->action_dim,
        (unsigned long)model->predictor_hidden,
        (unsigned long)model->predictor_layers,
        (unsigned long)model->predictor_heads,
        (unsigned long)model->predictor_inner_dim,
        (unsigned long)model->predictor_inter);
    if (model->has_full_encoder) {
        ESP_LOGI(TAG,
            "Encoder config: image=%lux%lu patch=%lu channels=%lu hidden=%lu layers=%lu heads=%lu inter=%lu seq=%lu",
            (unsigned long)model->image_size,
            (unsigned long)model->image_size,
            (unsigned long)model->patch_size,
            (unsigned long)model->channels,
            (unsigned long)model->encoder_hidden,
            (unsigned long)model->encoder_layers,
            (unsigned long)model->encoder_heads,
            (unsigned long)model->encoder_inter,
            (unsigned long)model->encoder_seq_len);
    }
    log_projection_head_shape(&model->pred_proj, model->predictor_hidden);
    ESP_LOGI(TAG,
        "Scratch footprint: seq=%luB qkv=%luB ffn=%luB",
        (unsigned long)(3U * model->predictor_hidden * sizeof(float)),
        (unsigned long)(3U * 3U * model->predictor_inner_dim * sizeof(float)),
        (unsigned long)(3U * model->predictor_inter * sizeof(float)));

    free(config_json);
    return true;
}
