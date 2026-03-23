#include <metal_stdlib>
using namespace metal;

constant uint RMSNORM_THREADS = 256;

/// RMS normalization with parallel threadgroup reduction.
/// out[i] = x[i] * rsqrt(mean(x^2) + eps) * weight[i]
///
/// Each threadgroup processes one vector of length `n`.
/// For batched inputs, dispatch batch_size threadgroups.
///
/// Dispatch: threadgroups = batch_size (or 1), threads_per_threadgroup = 256
kernel void rmsnorm(
    device const float* x [[buffer(0)]],
    device const float* weight [[buffer(1)]],
    device float* out [[buffer(2)]],
    constant uint& n [[buffer(3)]],
    constant float& eps [[buffer(4)]],
    uint tid [[thread_index_in_threadgroup]],
    uint tgid [[threadgroup_position_in_grid]])
{
    threadgroup float shared_sum[RMSNORM_THREADS];

    uint base = tgid * n;

    // Each thread accumulates sum of squares for its strided elements
    float local_sum = 0.0;
    for (uint i = tid; i < n; i += RMSNORM_THREADS) {
        float val = x[base + i];
        local_sum += val * val;
    }
    shared_sum[tid] = local_sum;

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Parallel reduction for sum of squares
    for (uint stride = RMSNORM_THREADS / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_sum[tid] += shared_sum[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Compute reciprocal RMS (shared across all threads via broadcast)
    float rms = rsqrt(shared_sum[0] / float(n) + eps);

    // Normalize and scale
    for (uint i = tid; i < n; i += RMSNORM_THREADS) {
        out[base + i] = x[base + i] * rms * weight[i];
    }
}
