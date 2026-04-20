#include <metal_stdlib>
using namespace metal;

constant uint LN_THREADS = 256;

/// LayerNorm with weight (gamma) and bias (beta).
/// out[i] = (x[i] - mean) / sqrt(var + eps) * gamma[i] + beta[i]
///
/// Each threadgroup processes one vector of length `n`.
/// Dispatch: threadgroups = batch_size, threads_per_threadgroup = 256
kernel void layernorm_wb(
    device const float* x [[buffer(0)]],
    device const float* gamma [[buffer(1)]],
    device const float* beta [[buffer(2)]],
    device float* out [[buffer(3)]],
    constant uint& n [[buffer(4)]],
    constant float& eps [[buffer(5)]],
    uint tid [[thread_index_in_threadgroup]],
    uint tgid [[threadgroup_position_in_grid]])
{
    threadgroup float shared[LN_THREADS];
    uint base = tgid * n;

    // Pass 1: compute mean via parallel sum
    float local_sum = 0.0;
    for (uint i = tid; i < n; i += LN_THREADS) {
        local_sum += x[base + i];
    }
    shared[tid] = local_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = LN_THREADS / 2; s > 0; s >>= 1) {
        if (tid < s) shared[tid] += shared[tid + s];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float mean = shared[0] / float(n);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Pass 2: compute variance via parallel sum of (x - mean)^2
    float local_var = 0.0;
    for (uint i = tid; i < n; i += LN_THREADS) {
        float d = x[base + i] - mean;
        local_var += d * d;
    }
    shared[tid] = local_var;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = LN_THREADS / 2; s > 0; s >>= 1) {
        if (tid < s) shared[tid] += shared[tid + s];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float inv_std = rsqrt(shared[0] / float(n) + eps);

    // Pass 3: normalize + affine
    for (uint i = tid; i < n; i += LN_THREADS) {
        out[base + i] = (x[base + i] - mean) * inv_std * gamma[i] + beta[i];
    }
}
