/*
 * inference.c -- predict_next, encode_image, encoder/predictor layer forward.
 *
 * Extracted from app_main.c -- preserves ALL optimized code:
 *   - Nested (token,col) loops in encoder_layer_forward fused bias+residual
 *   - Nested (token,col) loops in predictor_layer_forward fused bias+GELU
 *   - esp_task_wdt_reset instead of vTaskDelay in encoder loop
 *   - encoder_layer_forward takes layer_idx as parameter (correctness fix)
 */

#include "inference.h"
#include "kernels.h"
#include "dual_core.h"
#include "pie_gemv.h"

#include <math.h>

#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

#include "esp_heap_caps.h"
#include "esp_log.h"
#include "esp_task_wdt.h"
#include "esp_timer.h"

static const char *TAG = "inference";

/* ================================================================== */
/* Encode action                                                       */
/* ================================================================== */

void encode_action(const PredictorModel *model, const float *action, float *out) {
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

/* ================================================================== */
/* Input / conditioning projections                                    */
/* ================================================================== */

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

/* ================================================================== */
/* Patch embedding                                                     */
/* ================================================================== */

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

    /* Batched INT8 GEMM: [num_patches, patch_dim] @ [hidden, patch_dim]^T -> [num_patches, hidden] */
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

/* ================================================================== */
/* Encoder layer forward                                               */
/* FIX: layer_idx is now a parameter, not a static variable            */
/* OPTIMIZED: fused bias+residual, fused bias+GELU, nested loops       */
/* ================================================================== */

static void encoder_layer_forward(
    PredictorModel *model,
    const EncoderLayer *layer,
    float *seq,
    int layer_idx
) {
    EncoderScratch *scratch = &model->encoder_scratch;
    size_t seq_len = model->encoder_seq_len;
    size_t hidden = model->encoder_hidden;
    size_t head_dim = hidden / model->encoder_heads;

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
        /* Kernel-trick O(nd^2) linear attention -- no score matrix */
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

    /* Fused w_o output + bias + residual add: single loop, reads proj once + writes seq once. */
    int8linear_forward_into(
        &layer->w_o,
        scratch->attn_out,
        seq_len,
        scratch->row_quant,
        scratch->proj
    );
    {
        const float *bias = layer->o_bias.data;
        for (size_t t = 0; t < seq_len; ++t) {
            size_t off = t * hidden;
            for (size_t col = 0; col < hidden; ++col) {
                seq[off + col] += scratch->proj[off + col] + bias[col];
            }
        }
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
    /* Fused FFN_up bias + GELU: single loop, one PSRAM read + write per element. */
    {
        const float *bias = layer->ffn_up_bias.data;
        size_t inter = model->encoder_inter;
        for (size_t t = 0; t < seq_len; ++t) {
            size_t off = t * inter;
            for (size_t col = 0; col < inter; ++col) {
                float v = scratch->ffn_inter[off + col] + bias[col];
                scratch->ffn_inter[off + col] = gelu_scalar(v);
            }
        }
    }
    int8linear_forward_into(
        &layer->ffn_down,
        scratch->ffn_inter,
        seq_len,
        scratch->row_quant,
        scratch->proj
    );
    /* Fused FFN_down bias + residual add: single loop, reads proj once + writes seq once. */
    {
        const float *bias = layer->ffn_down_bias.data;
        for (size_t t = 0; t < seq_len; ++t) {
            size_t off = t * hidden;
            for (size_t col = 0; col < hidden; ++col) {
                seq[off + col] += scratch->proj[off + col] + bias[col];
            }
        }
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
}

/* ================================================================== */
/* Encode image                                                        */
/* ================================================================== */

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
    for (size_t i = 0; i < seq_len * hidden && i < model->pos_embed.len; ++i) {
        scratch->x[i] += model->pos_embed.data[i];
    }

    for (size_t layer_index = 0; layer_index < model->encoder_layers; ++layer_index) {
        encoder_layer_forward(model, &model->encoder[layer_index], scratch->x, (int)layer_index);
        if ((layer_index & 1U) == 1U) {
            esp_task_wdt_reset();
        }
    }

    layernorm_into(scratch->x, model->final_norm_weight.data, 1U, hidden, scratch->cls_norm);

    /* Hybrid encoder output projection (Linear, no activation) */
    float *enc_out = scratch->cls_norm;
    if (model->enc_proj_weight.len > 0) {
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

/* ================================================================== */
/* Predictor layer forward                                             */
/* OPTIMIZED: nested (token,col) fused bias+GELU and bias+gated resid  */
/* ================================================================== */

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
    /* Fused mlp_up bias + GELU: single loop, reads ffn_inter once + writes once. */
    {
        const float *bias = layer->mlp_up_bias.data;
        for (size_t t = 0; t < seq_len; ++t) {
            size_t off = t * inter;
            for (size_t col = 0; col < inter; ++col) {
                float v = scratch->ffn_inter[off + col] + bias[col];
                scratch->ffn_inter[off + col] = gelu_scalar(v);
            }
        }
    }

    q4linear_forward_into(&layer->mlp_down, scratch->ffn_inter, seq_len, scratch->proj);
    /* Fused mlp_down bias + gated residual: single loop over seq*hidden. */
    {
        const float *bias = layer->mlp_down_bias.data;
        for (size_t token = 0; token < seq_len; ++token) {
            size_t base = token * hidden;
            for (size_t i = 0; i < hidden; ++i) {
                seq[base + i] += gate2[i] * (scratch->proj[base + i] + bias[i]);
            }
        }
    }
}

/* ================================================================== */
/* Projection head forward                                             */
/* ================================================================== */

void projection_head_forward(
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

void log_projection_head_shape(const ProjectionHead *head, size_t input_dim) {
    size_t current_dim = input_dim;
    ESP_LOGI(TAG, "pred_proj layers=%lu input_dim=%lu max_dim=%lu",
        (unsigned long)head->num_layers,
        (unsigned long)input_dim,
        (unsigned long)head->max_dim);
    for (size_t layer_index = 0; layer_index < head->num_layers; ++layer_index) {
        const ProjectionLayer *layer = &head->layers[layer_index];
        if (layer->weight.len == 0 || layer->weight.data == NULL) {
            ESP_LOGI(TAG, "pred_proj layer %lu weight_len=%lu bias_len=%lu in=%lu out=skip",
                (unsigned long)layer_index,
                (unsigned long)layer->weight.len,
                (unsigned long)layer->bias.len,
                (unsigned long)current_dim);
            continue;
        }
        size_t out_dim = current_dim == 0 ? 0 : layer->weight.len / current_dim;
        ESP_LOGI(TAG, "pred_proj layer %lu weight_len=%lu bias_len=%lu in=%lu out=%lu",
            (unsigned long)layer_index,
            (unsigned long)layer->weight.len,
            (unsigned long)layer->bias.len,
            (unsigned long)current_dim,
            (unsigned long)out_dim);
        current_dim = out_dim;
    }
}

/* ================================================================== */
/* predict_next                                                        */
/* ================================================================== */

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

/* ================================================================== */
/* Fused multi-step rollout                                            */
/* ================================================================== */

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

    float *action_embeds = (float *)lewm_calloc(
        num_steps * hidden, sizeof(float), MALLOC_CAP_INTERNAL);
    if (!action_embeds) {
        memset(out, 0, num_steps * latent * sizeof(float));
        return;
    }

    /* 1. Encode all actions upfront. */
    for (size_t s = 0; s < num_steps; ++s) {
        encode_action(model, actions + s * model->action_dim, action_embeds + s * hidden);
    }

    /* 2. Build fused sequence: [z_start, a_s, zeros] per step. */
    memset(scratch->seq_raw, 0, fused_seq_len * hidden * sizeof(float));
    for (size_t s = 0; s < num_steps; ++s) {
        size_t off = s * 3U * hidden;
        if (has_proj) {
            memcpy(scratch->seq_raw + off, z_start, latent * sizeof(float));
        } else {
            memcpy(scratch->seq_raw + off, z_start, hidden * sizeof(float));
        }
        if (has_proj) {
            apply_cond_proj(model, action_embeds + s * hidden, scratch->proj_tmp_a);
            memcpy(scratch->seq_raw + off + hidden, scratch->proj_tmp_a, hidden * sizeof(float));
        } else {
            memcpy(scratch->seq_raw + off + hidden,
                   action_embeds + s * hidden, hidden * sizeof(float));
        }
    }

    /* 2b. Add positional embeddings (cycle through 3 pattern positions). */
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

    /* 6. Extract targets at positions 2, 5, 8, ... and project each. */
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
