#pragma once

#include <stdbool.h>
#include <stddef.h>

#include "esp_http_server.h"

/**
 * Configuration passed to the HTTP inference server.
 * Uses an opaque model pointer so http_server.c does not need
 * the full PredictorModel struct definition.
 */
typedef struct {
    void       *model;            /* opaque pointer to PredictorModel */
    size_t      latent_dim;
    size_t      action_dim;
    size_t      predictor_layers;
    bool        has_encoder;
    char        mode[32];
} ServerConfig;

/**
 * Start the HTTP inference server on port 80.
 * The config is copied internally; the caller may free it after this returns.
 * Returns the server handle, or NULL on failure.
 */
httpd_handle_t start_inference_server(const ServerConfig *config);

/**
 * Stop the HTTP server and release resources.
 */
void stop_inference_server(httpd_handle_t server);
