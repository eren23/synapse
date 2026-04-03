/*
 * lewm_types.h -- Shared type definitions for LEWM on ESP32-P4.
 *
 * Every .c file in the project includes this header to get the struct
 * and typedef definitions that were previously local to app_main.c.
 */
#pragma once

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

/* Maximum sequence length for the predictor (150 = 50 actions x 3 tokens each).
 * Scratch buffers are allocated for this size so fused rollout never reallocates. */
#define MAX_PREDICTOR_SEQ_LEN  150U

/* ------------------------------------------------------------------ */
/* Core data structures                                                */
/* ------------------------------------------------------------------ */

typedef struct {
    size_t len;
    float *data;
} FloatBuffer;

typedef struct {
    FloatBuffer weight;
    FloatBuffer bias;
} ProjectionLayer;

typedef struct {
    size_t num_layers;
    ProjectionLayer *layers;
    size_t max_dim;
} ProjectionHead;

typedef struct {
    size_t out_features;
    size_t in_features;
    size_t blocks_per_row;
    size_t total_blocks;
    const uint8_t *bitmap;
    const uint8_t *blocks_data;
    uint32_t *row_nz_starts;
} Q4LinearRef;

typedef struct {
    size_t out_features;
    size_t in_features;
    size_t in_features_padded; /* rounded up to multiple of 16 for PIE */
    const int8_t *weights_data;
    int8_t *weights_t;         /* transposed: [out][in_padded], PIE-friendly */
    FloatBuffer scales;
} Int8LinearRef;

typedef struct {
    Q4LinearRef adaln_linear;
    FloatBuffer adaln_bias;
    Q4LinearRef to_qkv;
    Q4LinearRef attn_out;
    FloatBuffer attn_out_bias;
    FloatBuffer attn_norm_weight;
    FloatBuffer attn_norm_bias;
    FloatBuffer mlp_norm_weight;
    FloatBuffer mlp_norm_bias;
    Q4LinearRef mlp_up;
    FloatBuffer mlp_up_bias;
    Q4LinearRef mlp_down;
    FloatBuffer mlp_down_bias;
} PredictorLayer;

typedef struct {
    Int8LinearRef w_q;
    Int8LinearRef w_k;
    Int8LinearRef w_v;
    Int8LinearRef w_o;
    Int8LinearRef ffn_up;
    Int8LinearRef ffn_down;
    FloatBuffer q_bias;
    FloatBuffer k_bias;
    FloatBuffer v_bias;
    FloatBuffer o_bias;
    FloatBuffer ffn_up_bias;
    FloatBuffer ffn_down_bias;
    FloatBuffer attn_norm_weight;
    FloatBuffer attn_norm_bias;
    FloatBuffer ffn_norm_weight;
    FloatBuffer ffn_norm_bias;
} EncoderLayer;

typedef struct {
    float *seq_raw;
    float *seq;
    float *conditioning;
    float *action_hidden;
    float *mod_vec;
    float *normed;
    float *modulated;
    float *qkv;
    float *attn_out;
    float *proj;
    float *ffn_inter;
    float *proj_tmp_a;
    float *proj_tmp_b;
} PredictorScratch;

typedef struct {
    float *x;
    float *normed;
    float *q;
    float *k;
    float *v;
    float *attn_out;
    float *proj;
    float *ffn_inter;
    float *patch;
    float *scores;
    float *cls_norm;
    int8_t *row_quant;
} EncoderScratch;

typedef struct PredictorModel {
    char mode[32];
    bool apply_final_norm_bias;
    bool has_full_encoder;
    size_t image_size;
    size_t patch_size;
    size_t channels;
    size_t encoder_hidden;
    size_t encoder_layers;
    size_t encoder_heads;
    size_t encoder_inter;
    size_t encoder_seq_len;
    size_t latent_dim;
    size_t action_dim;
    size_t predictor_hidden;
    size_t predictor_layers;
    size_t predictor_heads;
    size_t predictor_inner_dim;
    size_t predictor_inter;
    FloatBuffer predictor_pos_embed;
    FloatBuffer predictor_norm_weight;
    FloatBuffer predictor_norm_bias;
    FloatBuffer action_conv_weight;
    FloatBuffer action_conv_bias;
    FloatBuffer action_mlp1_weight;
    FloatBuffer action_mlp1_bias;
    FloatBuffer action_mlp2_weight;
    FloatBuffer action_mlp2_bias;
    FloatBuffer input_proj_weight;
    FloatBuffer input_proj_bias;
    FloatBuffer cond_proj_weight;
    FloatBuffer cond_proj_bias;
    FloatBuffer patch_proj;
    FloatBuffer patch_proj_bias;
    /* INT8-quantized patch_proj for PIE-accelerated batch embedding */
    Int8LinearRef patch_proj_i8;
    float *all_patches;   /* scratch: [num_patches * patch_dim] for batch extract */
    FloatBuffer cls_token;
    FloatBuffer pos_embed;
    FloatBuffer final_norm_weight;
    FloatBuffer final_norm_bias;
    /* Hybrid encoder extras */
    size_t meta_tokens;
    FloatBuffer meta_token;
    FloatBuffer enc_proj_weight;
    FloatBuffer enc_proj_bias;
    ProjectionHead projector;
    ProjectionHead pred_proj;
    EncoderLayer *encoder;
    PredictorLayer *layers;
    EncoderScratch encoder_scratch;
    PredictorScratch scratch;
} PredictorModel;

/* ------------------------------------------------------------------ */
/* Dual-core work items                                                */
/* ------------------------------------------------------------------ */

typedef void (*core1_fn_t)(void *arg);

typedef struct {
    /* Shared read-only inputs */
    const int8_t *q_i8;
    const int8_t *k_i8;
    const float *q_scales;
    const float *k_scales;
    const float *v;
    float *out;
    float inv_sqrt_hd;
    size_t seq_len;
    size_t num_heads;
    size_t head_dim;
    size_t head_dim_padded;
    size_t inner_dim;
    size_t row_bytes;
    /* Per-core range */
    size_t q_start;
    size_t q_end;
    /* Per-core scratch (scores buffer) */
    float *scores;
    /* Linear attention: skip softmax, use L1 normalization */
    bool linear_attn;
} AttnWorkItem;

typedef struct {
    const int8_t *all_i8;
    const int8_t *weights_t;
    const float *scales;
    const float *w_scales;
    float *out;
    size_t m;
    size_t in_pad;
    size_t out_f;
    size_t j_start;
    size_t j_end;
} GemvWorkItem;
