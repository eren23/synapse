/*
 * HTTP inference server for LEWM on ESP32-P4.
 *
 * Endpoints:
 *   GET  /         -- serve embedded index.html
 *   GET  /status   -- model & device info (JSON)
 *   POST /predict  -- single-step latent prediction
 *   POST /rollout  -- multi-step trajectory rollout
 *   POST /encode   -- encode raw image into latent (binary body)
 *   OPTIONS (wildcard) -- CORS preflight
 */

#include <stdlib.h>
#include <string.h>

#include "esp_heap_caps.h"
#include "esp_http_server.h"
#include "esp_log.h"
#include "esp_timer.h"
#include "cJSON.h"

#include "http_server.h"
#include "wifi.h"

static const char *TAG = "http-server";

/* ------------------------------------------------------------------ */
/* Forward-declared PredictorModel: we never dereference this pointer  */
/* directly -- all access goes through the extern helpers below.       */
/* ------------------------------------------------------------------ */
typedef struct PredictorModel PredictorModel;

/*
 * These functions are defined in app_main.c but made non-static so the
 * HTTP server can call them.  The PredictorModel pointer is cast from
 * the opaque void* stored in ServerConfig.
 */
extern void predict_next(PredictorModel *model,
                         const float *state,
                         const float *action,
                         float *out);

extern void predict_rollout_fused(PredictorModel *model,
                                  const float *z_start,
                                  const float *actions,
                                  size_t num_steps,
                                  float *out);

extern bool encode_image(PredictorModel *model,
                         const float *image,
                         size_t height,
                         size_t width,
                         float *out);

/* ------------------------------------------------------------------ */
/* Embedded HTML dashboard                                             */
/* ------------------------------------------------------------------ */
extern const uint8_t index_html_start[] asm("_binary_index_html_start");
extern const uint8_t index_html_end[]   asm("_binary_index_html_end");

/* ------------------------------------------------------------------ */
/* Cached server config (set once in start_inference_server)           */
/* ------------------------------------------------------------------ */
static ServerConfig s_cfg;

/* ------------------------------------------------------------------ */
/* Helpers                                                             */
/* ------------------------------------------------------------------ */

static esp_err_t set_cors_headers(httpd_req_t *req)
{
    httpd_resp_set_hdr(req, "Access-Control-Allow-Origin", "*");
    httpd_resp_set_hdr(req, "Access-Control-Allow-Methods",
                       "GET, POST, OPTIONS");
    httpd_resp_set_hdr(req, "Access-Control-Allow-Headers", "Content-Type");
    return ESP_OK;
}

/**
 * Receive the full HTTP body into a heap-allocated buffer (PSRAM).
 * Returns NULL on error or if body exceeds 2 MB.
 */
static char *receive_body(httpd_req_t *req)
{
    int total = req->content_len;
    if (total <= 0 || total > 2 * 1024 * 1024) {
        return NULL;
    }

    char *buf = heap_caps_malloc(total + 1,
                                 MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
    if (!buf) {
        return NULL;
    }

    int received = 0;
    while (received < total) {
        int ret = httpd_req_recv(req, buf + received, total - received);
        if (ret <= 0) {
            free(buf);
            return NULL;
        }
        received += ret;
    }
    buf[total] = '\0';
    return buf;
}

/**
 * Copy a cJSON array of numbers into a float buffer.
 * Returns the number of elements copied.
 */
static size_t cjson_to_float_array(const cJSON *arr, float *out,
                                   size_t max_len)
{
    size_t count = 0;
    const cJSON *item;
    cJSON_ArrayForEach(item, arr) {
        if (count >= max_len) break;
        out[count++] = (float)item->valuedouble;
    }
    return count;
}

/**
 * Serialize a float array to a JSON array string, e.g. "[0.1,0.2,0.3]".
 * Caller must free() the returned buffer.
 */
static char *float_array_to_json(const float *data, size_t len)
{
    /* Estimate: each float ~12 chars, plus brackets and commas */
    size_t buf_size = len * 14 + 4;
    char *buf = heap_caps_malloc(buf_size,
                                 MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
    if (!buf) return NULL;

    size_t pos = 0;
    buf[pos++] = '[';
    for (size_t i = 0; i < len; i++) {
        if (i > 0) buf[pos++] = ',';
        pos += snprintf(buf + pos, buf_size - pos, "%.6f", data[i]);
    }
    buf[pos++] = ']';
    buf[pos] = '\0';
    return buf;
}

/* ================================================================== */
/* URI Handlers                                                       */
/* ================================================================== */

/* GET / ------------------------------------------------------------ */

static esp_err_t index_handler(httpd_req_t *req)
{
    set_cors_headers(req);
    httpd_resp_set_type(req, "text/html");
    httpd_resp_send(req, (const char *)index_html_start,
                    index_html_end - index_html_start);
    return ESP_OK;
}

/* OPTIONS (CORS) --------------------------------------------------- */

static esp_err_t cors_options_handler(httpd_req_t *req)
{
    set_cors_headers(req);
    httpd_resp_set_status(req, "204 No Content");
    httpd_resp_send(req, NULL, 0);
    return ESP_OK;
}

/* GET /status ------------------------------------------------------ */

static esp_err_t status_handler(httpd_req_t *req)
{
    set_cors_headers(req);

    cJSON *root = cJSON_CreateObject();
    cJSON_AddStringToObject(root, "model", "LeWorldModel (PushT)");
    cJSON_AddStringToObject(root, "mode", s_cfg.mode);
    cJSON_AddNumberToObject(root, "latent_dim",
                            (double)s_cfg.latent_dim);
    cJSON_AddNumberToObject(root, "action_dim",
                            (double)s_cfg.action_dim);
    cJSON_AddNumberToObject(root, "predictor_layers",
                            (double)s_cfg.predictor_layers);
    cJSON_AddBoolToObject(root, "has_encoder", s_cfg.has_encoder);
    cJSON_AddStringToObject(root, "backend", "ESP32-P4 scalar");
    cJSON_AddNumberToObject(root, "heap_free",
                            (double)esp_get_free_heap_size());
    cJSON_AddNumberToObject(root, "psram_free",
                            (double)heap_caps_get_free_size(
                                MALLOC_CAP_SPIRAM));
    cJSON_AddStringToObject(root, "wifi_ip", wifi_get_ip());

    char *json = cJSON_Print(root);
    httpd_resp_set_type(req, "application/json");
    httpd_resp_send(req, json, strlen(json));

    free(json);
    cJSON_Delete(root);
    return ESP_OK;
}

/* POST /predict ---------------------------------------------------- */

static esp_err_t predict_handler(httpd_req_t *req)
{
    set_cors_headers(req);

    char *body = receive_body(req);
    if (!body) {
        httpd_resp_send_err(req, HTTPD_400_BAD_REQUEST,
                            "Failed to read request body");
        return ESP_FAIL;
    }

    cJSON *root = cJSON_Parse(body);
    free(body);
    if (!root) {
        httpd_resp_send_err(req, HTTPD_400_BAD_REQUEST,
                            "Invalid JSON");
        return ESP_FAIL;
    }

    const cJSON *latent_arr = cJSON_GetObjectItem(root, "latent");
    const cJSON *action_arr = cJSON_GetObjectItem(root, "action");
    if (!cJSON_IsArray(latent_arr) || !cJSON_IsArray(action_arr)) {
        cJSON_Delete(root);
        httpd_resp_send_err(req, HTTPD_400_BAD_REQUEST,
                            "Missing 'latent' or 'action' arrays");
        return ESP_FAIL;
    }

    size_t ldim = s_cfg.latent_dim;
    size_t adim = s_cfg.action_dim;

    float *latent = heap_caps_malloc(ldim * sizeof(float),
                                     MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
    float *action = heap_caps_malloc(adim * sizeof(float),
                                     MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
    float *output = heap_caps_malloc(ldim * sizeof(float),
                                     MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
    if (!latent || !action || !output) {
        free(latent); free(action); free(output);
        cJSON_Delete(root);
        httpd_resp_send_err(req, HTTPD_500_INTERNAL_SERVER_ERROR,
                            "Out of memory");
        return ESP_FAIL;
    }

    cjson_to_float_array(latent_arr, latent, ldim);
    cjson_to_float_array(action_arr, action, adim);
    cJSON_Delete(root);

    int64_t t0 = esp_timer_get_time();
    predict_next((PredictorModel *)s_cfg.model,
                 latent, action, output);
    int64_t latency_us = esp_timer_get_time() - t0;
    double latency_ms = (double)latency_us / 1000.0;

    char *arr_json = float_array_to_json(output, ldim);
    free(latent);
    free(action);
    free(output);

    if (!arr_json) {
        httpd_resp_send_err(req, HTTPD_500_INTERNAL_SERVER_ERROR,
                            "Failed to serialize output");
        return ESP_FAIL;
    }

    /* Build response: {"next_latent": [...], "latency_ms": N} */
    size_t resp_size = strlen(arr_json) + 64;
    char *resp = heap_caps_malloc(resp_size,
                                  MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
    if (!resp) {
        free(arr_json);
        httpd_resp_send_err(req, HTTPD_500_INTERNAL_SERVER_ERROR,
                            "Out of memory");
        return ESP_FAIL;
    }
    snprintf(resp, resp_size,
             "{\"next_latent\":%s,\"latency_ms\":%.2f}",
             arr_json, latency_ms);
    free(arr_json);

    httpd_resp_set_type(req, "application/json");
    httpd_resp_send(req, resp, strlen(resp));
    free(resp);

    ESP_LOGI(TAG, "predict: %.1f ms", latency_ms);
    return ESP_OK;
}

/* POST /rollout ---------------------------------------------------- */

static esp_err_t rollout_handler(httpd_req_t *req)
{
    set_cors_headers(req);

    char *body = receive_body(req);
    if (!body) {
        httpd_resp_send_err(req, HTTPD_400_BAD_REQUEST,
                            "Failed to read request body");
        return ESP_FAIL;
    }

    cJSON *root = cJSON_Parse(body);
    free(body);
    if (!root) {
        httpd_resp_send_err(req, HTTPD_400_BAD_REQUEST, "Invalid JSON");
        return ESP_FAIL;
    }

    const cJSON *latent_arr  = cJSON_GetObjectItem(root, "latent");
    const cJSON *actions_arr = cJSON_GetObjectItem(root, "actions");
    if (!cJSON_IsArray(latent_arr) || !cJSON_IsArray(actions_arr)) {
        cJSON_Delete(root);
        httpd_resp_send_err(req, HTTPD_400_BAD_REQUEST,
                            "Missing 'latent' or 'actions' arrays");
        return ESP_FAIL;
    }

    size_t ldim = s_cfg.latent_dim;
    size_t adim = s_cfg.action_dim;

    /* Number of rollout steps = length of actions array */
    int steps = cJSON_GetArraySize(actions_arr);
    const cJSON *steps_item = cJSON_GetObjectItem(root, "steps");
    if (cJSON_IsNumber(steps_item) && steps_item->valueint < steps) {
        steps = steps_item->valueint;
    }
    if (steps <= 0 || steps > 1000) {
        cJSON_Delete(root);
        httpd_resp_send_err(req, HTTPD_400_BAD_REQUEST,
                            "Invalid steps count (1..1000)");
        return ESP_FAIL;
    }

    /* Allocate working buffers */
    float *state  = heap_caps_malloc(ldim * sizeof(float),
                                     MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
    float *action = heap_caps_malloc(adim * sizeof(float),
                                     MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
    float *next   = heap_caps_malloc(ldim * sizeof(float),
                                     MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
    /* Trajectory: (steps+1) * ldim floats (initial + one per step) */
    float *traj   = heap_caps_malloc((size_t)(steps + 1) * ldim * sizeof(float),
                                     MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
    if (!state || !action || !next || !traj) {
        free(state); free(action); free(next); free(traj);
        cJSON_Delete(root);
        httpd_resp_send_err(req, HTTPD_500_INTERNAL_SERVER_ERROR,
                            "Out of memory");
        return ESP_FAIL;
    }

    /* Copy initial latent */
    cjson_to_float_array(latent_arr, state, ldim);
    memcpy(traj, state, ldim * sizeof(float));

    int64_t t0 = esp_timer_get_time();

    for (int i = 0; i < steps; i++) {
        const cJSON *act_i = cJSON_GetArrayItem(actions_arr, i);
        if (cJSON_IsArray(act_i)) {
            cjson_to_float_array(act_i, action, adim);
        } else {
            memset(action, 0, adim * sizeof(float));
        }

        predict_next((PredictorModel *)s_cfg.model,
                     state, action, next);

        /* next becomes current state */
        memcpy(state, next, ldim * sizeof(float));
        memcpy(traj + (size_t)(i + 1) * ldim, next,
               ldim * sizeof(float));
    }

    int64_t latency_us = esp_timer_get_time() - t0;
    double latency_ms = (double)latency_us / 1000.0;

    cJSON_Delete(root);
    free(action);
    free(next);

    /*
     * Build response:
     * {"trajectory": [[...], [...], ...], "steps": N, "latency_ms": M}
     *
     * We build the trajectory JSON manually for efficiency.
     */
    /* Estimate size: (steps+1) entries, each ~ldim*14 chars */
    size_t traj_json_size = (size_t)(steps + 1) * ldim * 14 + (size_t)(steps + 1) * 4 + 128;
    char *resp = heap_caps_malloc(traj_json_size,
                                  MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
    if (!resp) {
        free(state); free(traj);
        httpd_resp_send_err(req, HTTPD_500_INTERNAL_SERVER_ERROR,
                            "Out of memory");
        return ESP_FAIL;
    }

    size_t pos = 0;
    pos += snprintf(resp + pos, traj_json_size - pos,
                    "{\"trajectory\":[");

    for (int i = 0; i <= steps; i++) {
        if (i > 0) resp[pos++] = ',';
        char *step_json = float_array_to_json(traj + (size_t)i * ldim,
                                              ldim);
        if (step_json) {
            pos += snprintf(resp + pos, traj_json_size - pos,
                            "%s", step_json);
            free(step_json);
        }
    }

    pos += snprintf(resp + pos, traj_json_size - pos,
                    "],\"steps\":%d,\"latency_ms\":%.2f}",
                    steps, latency_ms);

    free(state);
    free(traj);

    httpd_resp_set_type(req, "application/json");
    httpd_resp_send(req, resp, pos);
    free(resp);

    ESP_LOGI(TAG, "rollout: %d steps in %.1f ms (%.1f ms/step)",
             steps, latency_ms, latency_ms / steps);
    return ESP_OK;
}

/* POST /rollout_fused ----------------------------------------------
 * Fused multi-step rollout: encodes all actions, builds one fused
 * N×3-token sequence, runs predictor layers once.
 * Same z_start for all positions — parallel future hypotheses.
 * Faster than sequential rollout for N>1.
 * ---------------------------------------------------------------- */

static esp_err_t rollout_fused_handler(httpd_req_t *req)
{
    set_cors_headers(req);

    char *body = receive_body(req);
    if (!body) {
        httpd_resp_send_err(req, HTTPD_400_BAD_REQUEST,
                            "Failed to read request body");
        return ESP_FAIL;
    }

    cJSON *root = cJSON_Parse(body);
    free(body);
    if (!root) {
        httpd_resp_send_err(req, HTTPD_400_BAD_REQUEST, "Invalid JSON");
        return ESP_FAIL;
    }

    const cJSON *latent_arr  = cJSON_GetObjectItem(root, "latent");
    const cJSON *actions_arr = cJSON_GetObjectItem(root, "actions");
    if (!cJSON_IsArray(latent_arr) || !cJSON_IsArray(actions_arr)) {
        cJSON_Delete(root);
        httpd_resp_send_err(req, HTTPD_400_BAD_REQUEST,
                            "Missing 'latent' or 'actions' arrays");
        return ESP_FAIL;
    }

    size_t ldim = s_cfg.latent_dim;
    size_t adim = s_cfg.action_dim;

    int num_steps = cJSON_GetArraySize(actions_arr);
    if (num_steps <= 0 || num_steps > 10) {
        /* Hard limit: MAX_PREDICTOR_SEQ_LEN = 30, and num_steps * 3 <= 30 */
        cJSON_Delete(root);
        httpd_resp_send_err(req, HTTPD_400_BAD_REQUEST,
                            "Invalid steps (must be 1..10)");
        return ESP_FAIL;
    }

    /* Allocate working buffers */
    float *z_start = heap_caps_malloc(ldim * sizeof(float),
                                       MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
    float *actions = heap_caps_malloc((size_t)num_steps * adim * sizeof(float),
                                      MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
    float *outputs = heap_caps_malloc((size_t)num_steps * ldim * sizeof(float),
                                      MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
    if (!z_start || !actions || !outputs) {
        free(z_start); free(actions); free(outputs);
        cJSON_Delete(root);
        httpd_resp_send_err(req, HTTPD_500_INTERNAL_SERVER_ERROR,
                            "Out of memory");
        return ESP_FAIL;
    }

    /* Parse latent */
    cjson_to_float_array(latent_arr, z_start, ldim);

    /* Parse actions */
    for (int s = 0; s < num_steps; ++s) {
        const cJSON *act_i = cJSON_GetArrayItem(actions_arr, s);
        if (cJSON_IsArray(act_i)) {
            cjson_to_float_array(act_i, actions + (size_t)s * adim, adim);
        } else {
            memset(actions + (size_t)s * adim, 0, adim * sizeof(float));
        }
    }

    int64_t t0 = esp_timer_get_time();

    /* Call the fused predictor */
    predict_rollout_fused((PredictorModel *)s_cfg.model,
                          z_start, actions, (size_t)num_steps, outputs);

    int64_t latency_us = esp_timer_get_time() - t0;
    double latency_ms = (double)latency_us / 1000.0;

    cJSON_Delete(root);
    free(z_start);
    free(actions);

    /*
     * Build response: {"trajectory": [[...], [...], [...]], "steps": N, "latency_ms": M}
     */
    size_t resp_size = (size_t)num_steps * ldim * 14 + (size_t)num_steps * 4 + 128;
    char *resp = heap_caps_malloc(resp_size,
                                  MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
    if (!resp) {
        free(outputs);
        httpd_resp_send_err(req, HTTPD_500_INTERNAL_SERVER_ERROR,
                            "Out of memory");
        return ESP_FAIL;
    }

    size_t pos = 0;
    pos += snprintf(resp + pos, resp_size - pos, "{\"trajectory\":[");

    for (int s = 0; s < num_steps; ++s) {
        if (s > 0) resp[pos++] = ',';
        char *step_json = float_array_to_json(outputs + (size_t)s * ldim, ldim);
        if (step_json) {
            pos += snprintf(resp + pos, resp_size - pos, "%s", step_json);
            free(step_json);
        }
    }

    pos += snprintf(resp + pos, resp_size - pos,
                    "],\"steps\":%d,\"latency_ms\":%.2f}",
                    num_steps, latency_ms);

    free(outputs);

    httpd_resp_set_type(req, "application/json");
    httpd_resp_send(req, resp, pos);
    free(resp);

    ESP_LOGI(TAG, "rollout_fused: %d steps in %.1f ms (%.1f ms/step)",
             num_steps, latency_ms, latency_ms / num_steps);
    return ESP_OK;
}

/* POST /encode ----------------------------------------------------- */

static esp_err_t encode_handler(httpd_req_t *req)
{
    set_cors_headers(req);

    if (!s_cfg.has_encoder) {
        httpd_resp_send_err(req, HTTPD_400_BAD_REQUEST,
                            "Model does not have a vision encoder");
        return ESP_FAIL;
    }

    /* Read height/width from query params (default 224) */
    size_t height = 224;
    size_t width  = 224;

    char param_buf[16];
    if (httpd_req_get_url_query_str(req, param_buf,
                                     sizeof(param_buf)) == ESP_OK) {
        /* Re-parse with a bigger buffer for safety */
        size_t qlen = httpd_req_get_url_query_len(req) + 1;
        char *qstr = malloc(qlen);
        if (qstr && httpd_req_get_url_query_str(req, qstr, qlen) == ESP_OK) {
            char val[16];
            if (httpd_query_key_value(qstr, "height", val,
                                      sizeof(val)) == ESP_OK) {
                height = (size_t)atoi(val);
            }
            if (httpd_query_key_value(qstr, "width", val,
                                      sizeof(val)) == ESP_OK) {
                width = (size_t)atoi(val);
            }
        }
        free(qstr);
    }

    /* Expected body: height * width * 3 * sizeof(float) raw LE f32 */
    size_t expected = height * width * 3 * sizeof(float);
    int total = req->content_len;
    if (total <= 0 || (size_t)total != expected) {
        ESP_LOGE(TAG, "encode: expected %zu bytes, got %d",
                 expected, total);
        httpd_resp_send_err(req, HTTPD_400_BAD_REQUEST,
                            "Body size does not match height*width*3*4");
        return ESP_FAIL;
    }

    /* Receive raw image into PSRAM */
    float *image = heap_caps_malloc(expected,
                                    MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
    if (!image) {
        httpd_resp_send_err(req, HTTPD_500_INTERNAL_SERVER_ERROR,
                            "Out of PSRAM for image");
        return ESP_FAIL;
    }

    int received = 0;
    char *dst = (char *)image;
    while (received < total) {
        int ret = httpd_req_recv(req, dst + received, total - received);
        if (ret <= 0) {
            free(image);
            httpd_resp_send_err(req, HTTPD_408_REQ_TIMEOUT,
                                "Timed out receiving image data");
            return ESP_FAIL;
        }
        received += ret;
    }

    size_t ldim = s_cfg.latent_dim;
    float *output = heap_caps_malloc(ldim * sizeof(float),
                                     MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
    if (!output) {
        free(image);
        httpd_resp_send_err(req, HTTPD_500_INTERNAL_SERVER_ERROR,
                            "Out of memory");
        return ESP_FAIL;
    }

    int64_t t0 = esp_timer_get_time();
    bool ok = encode_image((PredictorModel *)s_cfg.model,
                           image, height, width, output);
    int64_t latency_us = esp_timer_get_time() - t0;
    double latency_ms = (double)latency_us / 1000.0;

    free(image);

    if (!ok) {
        free(output);
        httpd_resp_send_err(req, HTTPD_500_INTERNAL_SERVER_ERROR,
                            "encode_image() failed");
        return ESP_FAIL;
    }

    char *arr_json = float_array_to_json(output, ldim);
    free(output);

    if (!arr_json) {
        httpd_resp_send_err(req, HTTPD_500_INTERNAL_SERVER_ERROR,
                            "Failed to serialize latent");
        return ESP_FAIL;
    }

    size_t resp_size = strlen(arr_json) + 64;
    char *resp = heap_caps_malloc(resp_size,
                                  MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
    if (!resp) {
        free(arr_json);
        httpd_resp_send_err(req, HTTPD_500_INTERNAL_SERVER_ERROR,
                            "Out of memory");
        return ESP_FAIL;
    }
    snprintf(resp, resp_size,
             "{\"latent\":%s,\"latency_ms\":%.2f}",
             arr_json, latency_ms);
    free(arr_json);

    httpd_resp_set_type(req, "application/json");
    httpd_resp_send(req, resp, strlen(resp));
    free(resp);

    ESP_LOGI(TAG, "encode: %zux%zu -> latent[%zu] in %.1f ms",
             height, width, ldim, latency_ms);
    return ESP_OK;
}

/* ================================================================== */
/* Server lifecycle                                                   */
/* ================================================================== */

httpd_handle_t start_inference_server(const ServerConfig *config)
{
    /* Cache config locally */
    memcpy(&s_cfg, config, sizeof(ServerConfig));

    httpd_config_t http_config = HTTPD_DEFAULT_CONFIG();
    http_config.max_uri_handlers  = 10;
    http_config.stack_size        = 8192;
    http_config.recv_wait_timeout = 120;   /* 120 s for large /encode uploads */
    http_config.send_wait_timeout = 120;

    httpd_handle_t server = NULL;
    esp_err_t err = httpd_start(&server, &http_config);
    if (err != ESP_OK) {
        ESP_LOGE(TAG, "Failed to start HTTP server: %s",
                 esp_err_to_name(err));
        return NULL;
    }

    /* GET / */
    const httpd_uri_t uri_index = {
        .uri      = "/",
        .method   = HTTP_GET,
        .handler  = index_handler,
    };
    httpd_register_uri_handler(server, &uri_index);

    /* GET /status */
    const httpd_uri_t uri_status = {
        .uri      = "/status",
        .method   = HTTP_GET,
        .handler  = status_handler,
    };
    httpd_register_uri_handler(server, &uri_status);

    /* POST /predict */
    const httpd_uri_t uri_predict = {
        .uri      = "/predict",
        .method   = HTTP_POST,
        .handler  = predict_handler,
    };
    httpd_register_uri_handler(server, &uri_predict);

    /* POST /rollout */
    const httpd_uri_t uri_rollout = {
        .uri      = "/rollout",
        .method   = HTTP_POST,
        .handler  = rollout_handler,
    };
    httpd_register_uri_handler(server, &uri_rollout);

    /* POST /rollout_fused */
    const httpd_uri_t uri_rollout_fused = {
        .uri      = "/rollout_fused",
        .method   = HTTP_POST,
        .handler  = rollout_fused_handler,
    };
    httpd_register_uri_handler(server, &uri_rollout_fused);

    /* POST /encode */
    const httpd_uri_t uri_encode = {
        .uri      = "/encode",
        .method   = HTTP_POST,
        .handler  = encode_handler,
    };
    httpd_register_uri_handler(server, &uri_encode);

    /* OPTIONS wildcard (CORS preflight for any path) */
    const httpd_uri_t uri_options = {
        .uri      = "/predict",
        .method   = HTTP_OPTIONS,
        .handler  = cors_options_handler,
    };
    httpd_register_uri_handler(server, &uri_options);

    const httpd_uri_t uri_options_rollout = {
        .uri      = "/rollout",
        .method   = HTTP_OPTIONS,
        .handler  = cors_options_handler,
    };
    httpd_register_uri_handler(server, &uri_options_rollout);

    const httpd_uri_t uri_options_rollout_fused = {
        .uri      = "/rollout_fused",
        .method   = HTTP_OPTIONS,
        .handler  = cors_options_handler,
    };
    httpd_register_uri_handler(server, &uri_options_rollout_fused);

    const httpd_uri_t uri_options_encode = {
        .uri      = "/encode",
        .method   = HTTP_OPTIONS,
        .handler  = cors_options_handler,
    };
    httpd_register_uri_handler(server, &uri_options_encode);

    ESP_LOGI(TAG, "HTTP server started on port 80");
    return server;
}

void stop_inference_server(httpd_handle_t server)
{
    if (server) {
        httpd_stop(server);
        ESP_LOGI(TAG, "HTTP server stopped");
    }
}
