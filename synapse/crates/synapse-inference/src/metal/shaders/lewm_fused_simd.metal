// lewm_fused_simd.metal
// ====================
// Fused adaLN transformer layer with aggressively vectorized float4 dot products.
// One dispatch per layer, 6 dispatches total.
//
// Key optimization: 3-way unrolled float4 multiply-accumulate in all matmul inner
// loops. Processes 12 floats per iteration (3 × float4) to maximize ALU utilization
// and hide memory latency.

#include <metal_stdlib>
using namespace metal;

// LEWM PushT fixed dimensions
constant uint V3_H = 192;        // hidden
constant uint V3_INNER = 1024;   // attention inner dim
constant uint V3_HEADS = 16;
constant uint V3_HD = 64;        // head dim
constant uint V3_INTER = 2048;   // FFN intermediate
constant uint V3_SEQ = 3;
constant uint V3_MOD_DIM = 1152; // 6 * H

inline float v3_gelu_approx(float x) {
    return 0.5f * x * (1.0f + tanh(sqrt(2.0f / M_PI_F) * (x + 0.044715f * x * x * x)));
}

// ── Main fused kernel with vectorized float4 dot products ──────────

kernel void adaln_layer_fused_simd(
    // Input/output
    device float* seq          [[buffer(0)]],
    device const float* cond   [[buffer(1)]],
    // Layer weights (same layout as adaln_layer_fused)
    device const float* w_adaln      [[buffer(2)]],
    device const float* b_adaln      [[buffer(3)]],
    device const float* w_qkv        [[buffer(4)]],
    device const float* w_attn_out   [[buffer(5)]],
    device const float* b_attn_out   [[buffer(6)]],
    device const float* attn_norm_w  [[buffer(7)]],
    device const float* mlp_norm_w   [[buffer(8)]],
    device const float* w_up         [[buffer(9)]],
    device const float* b_up         [[buffer(10)]],
    device const float* w_down       [[buffer(11)]],
    device const float* b_down       [[buffer(12)]],
    // Scratch buffers
    device float* mod_params [[buffer(13)]],
    device float* normed     [[buffer(14)]],
    device float* qkv_buf    [[buffer(15)]],
    device float* attn_out   [[buffer(16)]],
    device float* ffn_buf    [[buffer(17)]],
    // Padded input for simdgroup (8 rows instead of 3)
    device float* padded_a   [[buffer(18)]],
    // Thread info
    uint tid [[thread_position_in_grid]],
    uint tcount [[threads_per_grid]])
{
    // Pointers into mod_params
    device const float* scale1 = mod_params;
    device const float* shift1 = mod_params + H;
    device const float* gate1  = mod_params + 2 * H;
    device const float* scale2 = mod_params + 3 * H;
    device const float* shift2 = mod_params + 4 * H;
    device const float* gate2  = mod_params + 5 * H;

    // ═══ STEP 1: adaLN modulation [1, H] × [MOD_DIM, H]^T → [1, V3_MOD_DIM] ═══
    // Small (1×192 × 1152×192), use work-loop
    for (uint i = tid; i < V3_MOD_DIM; i += tcount) {
        float sum = 0.0f;
        for (uint k = 0; k < H; k++) {
            sum += cond[k] * w_adaln[i * V3_H + k];
        }
        mod_params[i] = sum + b_adaln[i];
    }
    threadgroup_barrier(mem_flags::mem_device);

    // ═══ STEP 2: LayerNorm + modulate → normed ═══
    for (uint i = tid; i < V3_SEQ * H; i += tcount) {
        uint row = i / H;
        uint col = i % H;
        // Compute mean/var
        float sum = 0.0f, sum_sq = 0.0f;
        for (uint j = 0; j < H; j++) {
            float v = seq[row * V3_H + j];
            sum += v; sum_sq += v * v;
        }
        float mean = sum / float(V3_H);
        float var = sum_sq / float(V3_H) - mean * mean;
        float inv_std = rsqrt(var + 1e-6f);
        float n = (seq[i] - mean) * inv_std * attn_norm_w[col];
        normed[i] = n * (1.0f + scale1[col]) + shift1[col];
    }
    threadgroup_barrier(mem_flags::mem_device);

    // ═══ STEP 3: QKV projection [3, H] × [3*V3_INNER, H]^T → [3, 3*V3_INNER] ═══
    // This is the BIG matmul — use work-loop with vectorized dot products
    uint qkv_total = V3_SEQ * 3 * V3_INNER;
    for (uint i = tid; i < qkv_total; i += tcount) {
        uint row = i / (3 * V3_INNER);
        uint col = i % (3 * V3_INNER);
        // Vectorized dot product
        float4 acc0 = 0, acc1 = 0, acc2 = 0;
        device const float4* nr = (device const float4*)(normed + row * V3_H);
        device const float4* wr = (device const float4*)(w_qkv + col * V3_H);
        for (uint k = 0; k < V3_H/4; k += 3) {
            acc0 += nr[k] * wr[k];
            if (k+1 < V3_H/4) acc1 += nr[k+1] * wr[k+1];
            if (k+2 < V3_H/4) acc2 += nr[k+2] * wr[k+2];
        }
        float4 t = acc0 + acc1 + acc2;
        qkv_buf[i] = t.x + t.y + t.z + t.w;
    }
    threadgroup_barrier(mem_flags::mem_device);

    // ═══ STEP 4: Attention (48 units) ═══
    for (uint i = tid; i < V3_HEADS * V3_SEQ; i += tcount) {
        uint head = i / V3_SEQ;
        uint qi = i % V3_SEQ;
        float atn_scale = rsqrt(float(HD));
        float scores[3];
        float max_s = -1e9f;
        for (uint ki = 0; ki < V3_SEQ; ki++) {
            float dot = 0.0f;
            for (uint d = 0; d < V3_HD; d++) {
                dot += qkv_buf[qi*3*V3_INNER + head*HD + d]
                     * qkv_buf[ki*3*V3_INNER + V3_INNER + head*HD + d];
            }
            scores[ki] = dot * atn_scale;
            max_s = max(max_s, scores[ki]);
        }
        float se = 0.0f;
        for (uint ki = 0; ki < V3_SEQ; ki++) { scores[ki] = exp(scores[ki]-max_s); se += scores[ki]; }
        float is = 1.0f / se;
        for (uint d = 0; d < V3_HD; d++) {
            float v = 0.0f;
            for (uint ki = 0; ki < V3_SEQ; ki++)
                v += scores[ki] * is * qkv_buf[ki*3*V3_INNER + 2*V3_INNER + head*HD + d];
            attn_out[qi*V3_INNER + head*HD + d] = v;
        }
    }
    threadgroup_barrier(mem_flags::mem_device);

    // ═══ STEP 5: Output proj + gated residual ═══
    for (uint i = tid; i < V3_SEQ * H; i += tcount) {
        uint row = i / H;
        uint col = i % H;
        // Vectorized dot
        float4 acc0 = 0, acc1 = 0;
        device const float4* ar = (device const float4*)(attn_out + row * V3_INNER);
        device const float4* wr = (device const float4*)(w_attn_out + col * V3_INNER);
        for (uint k = 0; k < V3_INNER/4; k += 2) {
            acc0 += ar[k] * wr[k];
            if (k+1 < V3_INNER/4) acc1 += ar[k+1] * wr[k+1];
        }
        float4 t = acc0 + acc1;
        float proj = t.x + t.y + t.z + t.w + b_attn_out[col];
        seq[i] += gate1[col] * proj;
    }
    threadgroup_barrier(mem_flags::mem_device);

    // ═══ STEP 6: Pre-FFN norm + modulate ═══
    for (uint i = tid; i < V3_SEQ * H; i += tcount) {
        uint row = i / H;
        uint col = i % H;
        float sum = 0.0f, sum_sq = 0.0f;
        for (uint j = 0; j < H; j++) {
            float v = seq[row * V3_H + j];
            sum += v; sum_sq += v * v;
        }
        float mean = sum / float(V3_H);
        float var = sum_sq / float(V3_H) - mean * mean;
        float inv_std = rsqrt(var + 1e-6f);
        float n = (seq[i] - mean) * inv_std * mlp_norm_w[col];
        normed[i] = n * (1.0f + scale2[col]) + shift2[col];
    }
    threadgroup_barrier(mem_flags::mem_device);

    // ═══ STEP 7: FFN up + GELU ═══
    for (uint i = tid; i < V3_SEQ * V3_INTER; i += tcount) {
        uint row = i / V3_INTER;
        uint col = i % V3_INTER;
        float4 acc0 = 0, acc1 = 0, acc2 = 0;
        device const float4* nr = (device const float4*)(normed + row * V3_H);
        device const float4* wr = (device const float4*)(w_up + col * V3_H);
        for (uint k = 0; k < V3_H/4; k += 3) {
            acc0 += nr[k] * wr[k];
            if (k+1 < V3_H/4) acc1 += nr[k+1] * wr[k+1];
            if (k+2 < V3_H/4) acc2 += nr[k+2] * wr[k+2];
        }
        float4 t = acc0 + acc1 + acc2;
        float val = t.x + t.y + t.z + t.w + b_up[col];
        ffn_buf[i] = v3_gelu_approx(val);
    }
    threadgroup_barrier(mem_flags::mem_device);

    // ═══ STEP 8: FFN down + gated residual ═══
    for (uint i = tid; i < V3_SEQ * H; i += tcount) {
        uint row = i / H;
        uint col = i % H;
        float4 acc0 = 0, acc1 = 0, acc2 = 0, acc3 = 0;
        device const float4* fr = (device const float4*)(ffn_buf + row * V3_INTER);
        device const float4* wr = (device const float4*)(w_down + col * V3_INTER);
        for (uint k = 0; k < V3_INTER/4; k += 4) {
            acc0 += fr[k] * wr[k];
            if (k+1 < V3_INTER/4) acc1 += fr[k+1] * wr[k+1];
            if (k+2 < V3_INTER/4) acc2 += fr[k+2] * wr[k+2];
            if (k+3 < V3_INTER/4) acc3 += fr[k+3] * wr[k+3];
        }
        float4 t = acc0 + acc1 + acc2 + acc3;
        float down = t.x + t.y + t.z + t.w + b_down[col];
        seq[i] += gate2[col] * down;
    }
}
