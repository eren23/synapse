/*
 * smoke_tests.h -- Boot-time smoke tests for predictor and encoder.
 */
#pragma once

#include "lewm_types.h"

/**
 * Fill a vector with deterministic pseudo-random values in [-1, 1].
 */
void fill_deterministic_vector(float *out, size_t len, uint32_t seed);

/**
 * Probe projection head internals (logs per-layer outputs).
 */
void probe_projection_head(
    const ProjectionHead *head,
    const float *input,
    size_t input_dim,
    PredictorScratch *scratch);

/**
 * Log boot state (heap, PSRAM).
 */
void log_boot_state(void);

/**
 * Relax task watchdog for CPU-heavy smoke inference.
 */
void relax_task_watchdog_for_smoke(void);

/**
 * Run full encoder + predictor smoke test (if model has encoder).
 */
void run_full_encode_smoke(PredictorModel *model);

/**
 * Run predictor-only smoke test.
 */
void run_predictor_smoke(PredictorModel *model);
