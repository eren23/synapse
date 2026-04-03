/*
 * dual_core.c -- Core 1 worker, dispatch, semaphores for ESP32-P4 dual-core.
 *
 * Extracted from app_main.c -- computation logic is unchanged except:
 *   - Memory barriers (__sync_synchronize) added to dispatch/worker
 */

#include "dual_core.h"

#include "freertos/FreeRTOS.h"
#include "freertos/semphr.h"
#include "freertos/task.h"

#include "esp_log.h"
#include "pie_gemv.h"
#include "kernels.h"

static const char *TAG = "dual-core";

/* Generic Core 1 work dispatch: function pointer + argument */
static volatile core1_fn_t s_core1_fn = NULL;
static volatile void *s_core1_arg = NULL;
SemaphoreHandle_t s_core1_start = NULL;
static SemaphoreHandle_t s_core1_done = NULL;

volatile AttnWorkItem s_core1_attn_work;

/* ------------------------------------------------------------------ */
/* Dispatch / wait                                                     */
/* ------------------------------------------------------------------ */

void core1_dispatch(core1_fn_t fn, void *arg) {
    s_core1_fn = fn;
    s_core1_arg = arg;
    __sync_synchronize();
    xSemaphoreGive(s_core1_start);
}

void core1_wait(void) {
    xSemaphoreTake(s_core1_done, portMAX_DELAY);
}

/* ------------------------------------------------------------------ */
/* Work item handlers                                                  */
/* ------------------------------------------------------------------ */

void gemv_compute_range(void *arg) {
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

void attn_compute_range(const AttnWorkItem *w) {
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

void attn_compute_range_wrapper(void *arg) {
    attn_compute_range((const AttnWorkItem *)arg);
}

/* ------------------------------------------------------------------ */
/* Core 1 worker task                                                  */
/* ------------------------------------------------------------------ */

static void core1_worker(void *arg) {
    (void)arg;
    for (;;) {
        xSemaphoreTake(s_core1_start, portMAX_DELAY);
        __sync_synchronize();
        if (s_core1_fn) {
            s_core1_fn((void *)s_core1_arg);
        }
        xSemaphoreGive(s_core1_done);
    }
}

void dual_core_init(void) {
    if (s_core1_start) return;
    s_core1_start = xSemaphoreCreateBinary();
    s_core1_done = xSemaphoreCreateBinary();
    xTaskCreatePinnedToCore(core1_worker, "core1", 8192, NULL, 5, NULL, 1);
    ESP_LOGI(TAG, "Dual-core worker started on Core 1");
}
