/*
 * smoke_tests.c -- Boot-time smoke tests for predictor and encoder.
 *
 * Extracted from app_main.c -- computation logic is unchanged except:
 *   - Uses lewm_calloc for consistent alloc with PSRAM fallback
 *   - Fixed double-free in rollout cleanup (prev_step allocated separately)
 */

#include "smoke_tests.h"
#include "inference.h"
#include "kernels.h"

#include <math.h>

#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

#include "esp_heap_caps.h"
#include "esp_log.h"
#include "esp_task_wdt.h"
#include "esp_timer.h"

static const char *TAG = "smoke-tests";

/* ================================================================== */
/* Helpers                                                             */
/* ================================================================== */

void fill_deterministic_vector(float *out, size_t len, uint32_t seed) {
    for (size_t idx = 0; idx < len; ++idx) {
        uint32_t mixed = seed * 1664525U + ((uint32_t)idx * 1013904223U);
        int32_t centered = (int32_t)(mixed % 2001U) - 1000;
        out[idx] = (float)centered / 1000.0f;
    }
}

void probe_projection_head(
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
}

void log_boot_state(void) {
    ESP_LOGI(TAG, "=== Synapse LEWM serial bring-up ===");
    ESP_LOGI(TAG, "Free heap: %lu bytes", (unsigned long)esp_get_free_heap_size());
    ESP_LOGI(TAG, "Free PSRAM: %lu bytes",
        (unsigned long)heap_caps_get_free_size(MALLOC_CAP_SPIRAM));
}

void relax_task_watchdog_for_smoke(void) {
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

/* ================================================================== */
/* Full encoder + predictor smoke                                      */
/* ================================================================== */

__attribute__((unused))
void run_full_encode_smoke(PredictorModel *model) {
    size_t image_len = model->image_size * model->image_size * model->channels;
    float *image = (float *)lewm_calloc(image_len, sizeof(float), MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
    float *latent = (float *)lewm_calloc(model->latent_dim, sizeof(float), MALLOC_CAP_INTERNAL);
    float *action = (float *)lewm_calloc(model->action_dim, sizeof(float), MALLOC_CAP_INTERNAL);
    float *next = (float *)lewm_calloc(model->latent_dim, sizeof(float), MALLOC_CAP_INTERNAL);

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
    int64_t encode_elapsed_us = esp_timer_get_time() - started_us;
    ESP_LOGI(TAG, "encode latency: %.3f ms", (double)encode_elapsed_us / 1000.0);
    vTaskDelay(pdMS_TO_TICKS(1));

    started_us = esp_timer_get_time();
    predict_next(model, latent, action, next);
    int64_t predict_elapsed_us = esp_timer_get_time() - started_us;
    ESP_LOGI(TAG, "predict_next after encode latency: %.3f ms",
             (double)predict_elapsed_us / 1000.0);
    ESP_LOGI(TAG, "encode + predict_next total: %.3f ms",
             (double)(encode_elapsed_us + predict_elapsed_us) / 1000.0);
    vTaskDelay(pdMS_TO_TICKS(1));

    free(image);
    free(latent);
    free(action);
    free(next);
}

/* ================================================================== */
/* Predictor-only smoke                                                */
/* ================================================================== */

__attribute__((unused))
void run_predictor_smoke(PredictorModel *model) {
    float *state = (float *)lewm_calloc(model->latent_dim, sizeof(float), MALLOC_CAP_INTERNAL);
    float *action = (float *)lewm_calloc(model->action_dim, sizeof(float), MALLOC_CAP_INTERNAL);
    float *next = (float *)lewm_calloc(model->latent_dim, sizeof(float), MALLOC_CAP_INTERNAL);
    float *rollout_state =
        (float *)lewm_calloc(model->latent_dim, sizeof(float), MALLOC_CAP_INTERNAL);
    float *action_embed =
        (float *)lewm_calloc(model->latent_dim, sizeof(float), MALLOC_CAP_INTERNAL);
    float *conditioning =
        (float *)lewm_calloc(model->predictor_hidden, sizeof(float), MALLOC_CAP_INTERNAL);
    float *layer0_adaln =
        (float *)lewm_calloc(6U * model->predictor_hidden, sizeof(float), MALLOC_CAP_INTERNAL);

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

    if (model->cond_proj_weight.len != 0) {
        matmul_t_into(
            action_embed,
            model->cond_proj_weight.data,
            1U,
            model->latent_dim,
            model->predictor_hidden,
            conditioning
        );
        add_bias_inplace(conditioning, model->cond_proj_bias.data, 1U, model->predictor_hidden);
    } else {
        memcpy(conditioning, action_embed, model->predictor_hidden * sizeof(float));
    }

    q4linear_forward_into(&model->layers[0].adaln_linear, conditioning, 1U, layer0_adaln);
    add_bias_inplace(layer0_adaln, model->layers[0].adaln_bias.data, 1U, 6U * model->predictor_hidden);

    int64_t started_us = esp_timer_get_time();
    predict_next(model, state, action, next);
    int64_t elapsed_us = esp_timer_get_time() - started_us;

    ESP_LOGI(TAG, "predict_next latency: %.3f ms", (double)elapsed_us / 1000.0);
    probe_projection_head(
        &model->pred_proj,
        model->scratch.normed + 2U * model->predictor_hidden,
        model->predictor_hidden,
        &model->scratch
    );
    vTaskDelay(pdMS_TO_TICKS(1));

    /* 50-step fused rollout via predict_rollout_fused. */
    {
        static const size_t ROLLOUT_STEPS = 50U;
        float *rollout_actions = (float *)lewm_calloc(
            ROLLOUT_STEPS * model->action_dim, sizeof(float), MALLOC_CAP_INTERNAL);
        float *rollout_outputs = (float *)lewm_calloc(
            ROLLOUT_STEPS * model->latent_dim, sizeof(float), MALLOC_CAP_INTERNAL);
        float *prev_step = (float *)lewm_calloc(
            model->latent_dim, sizeof(float), MALLOC_CAP_INTERNAL);

        if (rollout_actions && rollout_outputs && prev_step) {
            for (size_t s = 0; s < ROLLOUT_STEPS; ++s) {
                fill_deterministic_vector(rollout_actions + s * model->action_dim,
                                         model->action_dim, 101U + (uint32_t)s * 17U);
            }
            started_us = esp_timer_get_time();
            predict_rollout_fused(model, state, rollout_actions,
                                  ROLLOUT_STEPS, rollout_outputs);
            elapsed_us = esp_timer_get_time() - started_us;
            ESP_LOGI(TAG, "fused_rollout %lu steps: %.3f ms total (%.3f ms/step)",
                (unsigned long)ROLLOUT_STEPS,
                (double)elapsed_us / 1000.0,
                (double)elapsed_us / 1000.0 / ROLLOUT_STEPS);

            for (size_t s = 0; s < ROLLOUT_STEPS; ++s) {
                const float *step_out = rollout_outputs + s * model->latent_dim;
                float cos = s > 0 ? cosine_similarity(prev_step, step_out, model->latent_dim) : 0.0f;
                ESP_LOGI(TAG, "fused_rollout step %lu/%lu: cos=%.6f",
                    (unsigned long)(s + 1), (unsigned long)ROLLOUT_STEPS, cos);
                memcpy(prev_step, step_out, model->latent_dim * sizeof(float));
            }
        } else {
            ESP_LOGW(TAG, "Failed to allocate fused rollout buffers -- skipping 50-step test");
        }
        free(rollout_actions);
        free(rollout_outputs);
        free(prev_step);
    }

    free(state);
    free(action);
    free(next);
    free(rollout_state);
    free(action_embed);
    free(conditioning);
    free(layer0_adaln);
}
