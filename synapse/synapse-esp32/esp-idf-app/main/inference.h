/*
 * inference.h -- predict_next, encode_image, encoder/predictor layer forward.
 */
#pragma once

#include "lewm_types.h"

/**
 * Single-step latent prediction.
 * Non-static: called from http_server.c.
 */
void predict_next(PredictorModel *model,
                  const float *state,
                  const float *action,
                  float *out);

/**
 * Fused multi-step rollout.
 * Non-static: called from http_server.c.
 */
void predict_rollout_fused(PredictorModel *model,
                           const float *z_start,
                           const float *actions,
                           size_t num_steps,
                           float *out);

/**
 * Encode raw image into latent via the vision encoder.
 * Non-static: called from http_server.c.
 */
bool encode_image(PredictorModel *model,
                  const float *image,
                  size_t height,
                  size_t width,
                  float *out);

/**
 * Encode an action vector into action embedding.
 */
void encode_action(const PredictorModel *model, const float *action, float *out);

/**
 * Projection head forward (used by smoke tests too).
 */
void projection_head_forward(
    const ProjectionHead *head,
    const float *input,
    size_t input_dim,
    PredictorScratch *scratch,
    float *out);

/**
 * Log projection head shape info.
 */
void log_projection_head_shape(const ProjectionHead *head, size_t input_dim);
