/*
 * dual_core.h -- Core 1 worker dispatch for parallel compute on ESP32-P4.
 */
#pragma once

#include "lewm_types.h"

#include "freertos/FreeRTOS.h"
#include "freertos/semphr.h"

/**
 * Initialize Core 1 worker task and semaphores.
 * Safe to call multiple times (idempotent).
 */
void dual_core_init(void);

/**
 * Dispatch a function to Core 1 for execution.
 * Non-blocking: returns immediately after posting work.
 */
void core1_dispatch(core1_fn_t fn, void *arg);

/**
 * Block until Core 1 finishes the most recently dispatched work.
 */
void core1_wait(void);

/**
 * GemvWorkItem handler: compute a range of output features for INT8 GEMV.
 * Used as the core1_fn_t for dual-core FFN splits.
 */
void gemv_compute_range(void *arg);

/**
 * AttnWorkItem handler: compute a range of query tokens for INT8 attention.
 */
void attn_compute_range(const AttnWorkItem *w);

/**
 * Wrapper that casts void* to AttnWorkItem* for core1_dispatch.
 */
void attn_compute_range_wrapper(void *arg);

/**
 * Global volatile attention work item for Core 1 dispatch.
 * Must be set before calling core1_dispatch(attn_compute_range_wrapper, ...).
 */
extern volatile AttnWorkItem s_core1_attn_work;

/**
 * Semaphore handle exposed for conditional dual-core logic in kernels.
 */
extern SemaphoreHandle_t s_core1_start;
