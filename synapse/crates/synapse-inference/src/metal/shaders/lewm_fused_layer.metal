// lewm_fused_layer.metal
// =====================
// Single monolithic kernel for one complete adaLN transformer layer.
// ONE dispatch per layer = 6 dispatches total (vs ~65 with separate kernels).
//
// Fixed dimensions (LEWM PushT):
//   seq_len=3, hidden=192, inner_dim=1024, num_heads=16, head_dim=64, inter=2048
//
// Thread organization: N_MAX threads (N_MAX = max output dim = 3072 for QKV).
// Threads idle during smaller steps. threadgroup_barrier between steps.
//
// All intermediates in device memory (scratch buffers pre-allocated).
// Weights read from device memory sequentially.

#include <metal_stdlib>
using namespace metal;

// Fixed LEWM PushT dimensions
constant uint H = 192;
constant uint INNER = 1024;
constant uint HEADS = 16;
constant uint HD = 64;     // INNER / HEADS
constant uint INTER = 2048;
constant uint SEQ = 3;
constant uint MOD_DIM = 1152; // 6 * H

// Helpers
inline float gelu_approx(float x) {
    return 0.5f * x * (1.0f + tanh(sqrt(2.0f / M_PI_F) * (x + 0.044715f * x * x * x)));
}

// Compute layernorm mean/var for a row using float4 vectorized loads.
// H=192 = 48 × float4, perfect alignment.
inline float2 row_mean_var(device const float* row, uint n) {
    float4 vsum = float4(0.0f);
    float4 vsum_sq = float4(0.0f);
    uint n4 = n - (n % 4);
    device const float4* row4 = (device const float4*)row;
    for (uint j = 0; j < n4; j += 4) {
        float4 v = row4[j >> 2];
        vsum += v;
        vsum_sq += v * v;
    }
    float sum = vsum.x + vsum.y + vsum.z + vsum.w;
    float sum_sq = vsum_sq.x + vsum_sq.y + vsum_sq.z + vsum_sq.w;
    for (uint j = n4; j < n; j++) {
        float v = row[j];
        sum += v;
        sum_sq += v * v;
    }
    float mean = sum / float(n);
    float var = sum_sq / float(n) - mean * mean;
    return float2(mean, var);
}

// One thread computes one output element of: C[row, col] = dot(A[row, :], B[col, :])
// B is [N, K] row-major (transposed layout).
// Uses float4 vectorized loads (4x bandwidth) and 4-way unrolling to hide load latency.
inline float gemv_dot(device const float* a, device const float* b_row, uint k) {
    float4 acc0 = float4(0.0f);
    float4 acc1 = float4(0.0f);
    float4 acc2 = float4(0.0f);
    float4 acc3 = float4(0.0f);

    // Main loop: process 16 elements per iteration (4 × float4)
    uint i = 0;
    uint k16 = k - (k % 16);
    device const float4* a4 = (device const float4*)a;
    device const float4* b4 = (device const float4*)b_row;
    for (; i < k16; i += 16) {
        uint base = i >> 2;  // i / 4
        acc0 += a4[base]     * b4[base];
        acc1 += a4[base + 1] * b4[base + 1];
        acc2 += a4[base + 2] * b4[base + 2];
        acc3 += a4[base + 3] * b4[base + 3];
    }

    // Reduce 4 accumulators
    float4 total = acc0 + acc1 + acc2 + acc3;
    float sum = total.x + total.y + total.z + total.w;

    // Scalar tail for remaining elements
    for (; i < k; i++) {
        sum += a[i] * b_row[i];
    }
    return sum;
}

kernel void adaln_layer_fused(
    // Input/output
    device float* seq          [[buffer(0)]],  // [SEQ * H] = [576] — residual, modified in-place
    device const float* cond   [[buffer(1)]],  // [H] = [192] — action embedding

    // Layer weights
    device const float* w_adaln      [[buffer(2)]],   // [MOD_DIM, H] = [1152, 192]
    device const float* b_adaln      [[buffer(3)]],   // [MOD_DIM] = [1152]
    device const float* w_qkv        [[buffer(4)]],   // [3*INNER, H] = [3072, 192]
    device const float* w_attn_out   [[buffer(5)]],   // [H, INNER] = [192, 1024]
    device const float* b_attn_out   [[buffer(6)]],   // [H] = [192]
    device const float* attn_norm_w  [[buffer(7)]],   // [H] = [192]
    device const float* mlp_norm_w   [[buffer(8)]],   // [H] = [192]
    device const float* w_up         [[buffer(9)]],   // [INTER, H] = [2048, 192]
    device const float* b_up         [[buffer(10)]],  // [INTER] = [2048]
    device const float* w_down       [[buffer(11)]],  // [H, INTER] = [192, 2048]
    device const float* b_down       [[buffer(12)]],  // [H] = [192]

    // Scratch buffers (pre-allocated, reused)
    device float* mod_params [[buffer(13)]],  // [MOD_DIM] = [1152]
    device float* normed     [[buffer(14)]],  // [SEQ * H] = [576]
    device float* qkv_buf    [[buffer(15)]],  // [SEQ * 3*INNER] = [9216]
    device float* attn_out   [[buffer(16)]],  // [SEQ * INNER] = [3072]
    device float* ffn_buf    [[buffer(17)]],  // [SEQ * INTER] = [6144]

    // Thread info
    uint tid [[thread_position_in_grid]],
    uint tcount [[threads_per_grid]])
{
    // Work-loop pattern: each thread processes multiple output elements.
    // This avoids launching 9216 threads when only 576-6144 are needed per step.

    device const float* scale1 = mod_params;
    device const float* shift1 = mod_params + H;
    device const float* gate1  = mod_params + 2 * H;
    device const float* scale2 = mod_params + 3 * H;
    device const float* shift2 = mod_params + 4 * H;
    device const float* gate2  = mod_params + 5 * H;

    // ═══ STEP 1: adaLN modulation [MOD_DIM=1152 outputs] ═══
    for (uint i = tid; i < MOD_DIM; i += tcount) {
        mod_params[i] = gemv_dot(cond, w_adaln + i * H, H) + b_adaln[i];
    }
    threadgroup_barrier(mem_flags::mem_device);

    // ═══ STEP 2: LayerNorm + modulate [SEQ*H=576 outputs] ═══
    for (uint i = tid; i < SEQ * H; i += tcount) {
        uint row = i / H;
        uint col = i % H;
        float2 mv = row_mean_var(seq + row * H, H);
        float inv_std = rsqrt(mv.y + 1e-6f);
        float n = (seq[i] - mv.x) * inv_std * attn_norm_w[col];
        normed[i] = n * (1.0f + scale1[col]) + shift1[col];
    }
    threadgroup_barrier(mem_flags::mem_device);

    // ═══ STEP 3: QKV projection [SEQ*3*INNER=9216 outputs] ═══
    for (uint i = tid; i < SEQ * 3 * INNER; i += tcount) {
        uint row = i / (3 * INNER);
        uint col = i % (3 * INNER);
        qkv_buf[i] = gemv_dot(normed + row * H, w_qkv + col * H, H);
    }
    threadgroup_barrier(mem_flags::mem_device);

    // ═══ STEP 4: Attention (48 work units = HEADS*SEQ) ═══
    for (uint i = tid; i < HEADS * SEQ; i += tcount) {
        uint head = i / SEQ;
        uint qi = i % SEQ;
        float atn_scale = rsqrt(float(HD));

        float scores[3];
        float max_s = -1e9f;
        for (uint ki = 0; ki < SEQ; ki++) {
            float dot = 0.0f;
            for (uint d = 0; d < HD; d++) {
                dot += qkv_buf[qi * 3 * INNER + head * HD + d]
                     * qkv_buf[ki * 3 * INNER + INNER + head * HD + d];
            }
            scores[ki] = dot * atn_scale;
            max_s = max(max_s, scores[ki]);
        }
        float sum_exp = 0.0f;
        for (uint ki = 0; ki < SEQ; ki++) {
            scores[ki] = exp(scores[ki] - max_s);
            sum_exp += scores[ki];
        }
        float inv_sum = 1.0f / sum_exp;
        for (uint d = 0; d < HD; d++) {
            float val = 0.0f;
            for (uint ki = 0; ki < SEQ; ki++) {
                val += scores[ki] * inv_sum * qkv_buf[ki * 3 * INNER + 2 * INNER + head * HD + d];
            }
            attn_out[qi * INNER + head * HD + d] = val;
        }
    }
    threadgroup_barrier(mem_flags::mem_device);

    // ═══ STEP 5: Output proj + gated residual [SEQ*H=576] ═══
    for (uint i = tid; i < SEQ * H; i += tcount) {
        uint row = i / H;
        uint col = i % H;
        float proj = gemv_dot(attn_out + row * INNER, w_attn_out + col * INNER, INNER) + b_attn_out[col];
        seq[i] += gate1[col] * proj;
    }
    threadgroup_barrier(mem_flags::mem_device);

    // ═══ STEP 6: Pre-FFN LayerNorm + modulate [SEQ*H=576] ═══
    for (uint i = tid; i < SEQ * H; i += tcount) {
        uint row = i / H;
        uint col = i % H;
        float2 mv = row_mean_var(seq + row * H, H);
        float inv_std = rsqrt(mv.y + 1e-6f);
        float n = (seq[i] - mv.x) * inv_std * mlp_norm_w[col];
        normed[i] = n * (1.0f + scale2[col]) + shift2[col];
    }
    threadgroup_barrier(mem_flags::mem_device);

    // ═══ STEP 7: FFN up + GELU [SEQ*INTER=6144] ═══
    for (uint i = tid; i < SEQ * INTER; i += tcount) {
        uint row = i / INTER;
        uint col = i % INTER;
        float val = gemv_dot(normed + row * H, w_up + col * H, H) + b_up[col];
        ffn_buf[i] = gelu_approx(val);
    }
    threadgroup_barrier(mem_flags::mem_device);

    // ═══ STEP 8: FFN down + gated residual [SEQ*H=576] ═══
    for (uint i = tid; i < SEQ * H; i += tcount) {
        uint row = i / H;
        uint col = i % H;
        float down = gemv_dot(ffn_buf + row * INTER, w_down + col * INTER, INTER) + b_down[col];
        seq[i] += gate2[col] * down;
    }
}
