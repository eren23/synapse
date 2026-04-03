/*
 * app_main.c -- Boot, WiFi, HTTP server start.
 *
 * All model logic has been extracted into:
 *   lewm_types.h   -- shared struct/typedef definitions
 *   dual_core.c/.h -- Core 1 worker, dispatch, semaphores
 *   kernels.c/.h   -- math ops, quantization, attention kernels, allocators
 *   model_loader.c/.h -- LQ40 parsing, model loading
 *   inference.c/.h -- predict_next, encode_image, layer forward passes
 *   smoke_tests.c/.h -- boot-time smoke tests
 */

#include "lewm_types.h"
#include "dual_core.h"
#include "kernels.h"
#include "model_loader.h"
#include "inference.h"
#include "smoke_tests.h"
#include "pie_gemv.h"
#include "wifi.h"
#include "http_server.h"

#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

#include "esp_heap_caps.h"
#include "esp_log.h"

static const char *TAG = "synapse-lewm";

/* Embedded model binary (linked from model.bin via CMakeLists.txt) */
extern const uint8_t _binary_model_bin_start[] asm("_binary_model_bin_start");
extern const uint8_t _binary_model_bin_end[]   asm("_binary_model_bin_end");

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

    /* Initialize GELU and Exp lookup tables */
    gelu_lut_init();
    exp_lut_init();
    ESP_LOGI(TAG, "GELU + Exp LUTs initialized");

    /* Run smoke tests to get baseline timing */
    run_predictor_smoke(&model);
    if (model.has_full_encoder) {
        run_full_encode_smoke(&model);
    }

    /* 2. Connect to WiFi */
    ESP_LOGI(TAG, "Connecting to WiFi \"%s\" ...", CONFIG_LEWM_WIFI_SSID);
    if (wifi_init_sta(CONFIG_LEWM_WIFI_SSID, CONFIG_LEWM_WIFI_PASS) != ESP_OK) {
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
        .image_size       = model.image_size,
        .channels         = model.channels,
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
