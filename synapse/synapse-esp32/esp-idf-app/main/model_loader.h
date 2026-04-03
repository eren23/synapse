/*
 * model_loader.h -- LQ40 binary model parsing and loading.
 */
#pragma once

#include "lewm_types.h"

/**
 * Load an LQ40 model blob into a PredictorModel.
 * Returns true on success.  The caller must keep model_data alive
 * because Q4 weight refs point directly into it.
 */
bool load_model(const uint8_t *model_data, size_t model_len, PredictorModel *model);
