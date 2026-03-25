// Per-head RMSNorm: normalize each head independently.
// x: [num_heads * head_dim], weight: [head_dim] (shared across heads)
// out: [num_heads * head_dim]
// One threadgroup per head, 256 threads per threadgroup.
// If weight is empty (head_dim_weight == 0), copies x to out unchanged.

constant uint HN_THREADS = 256;

kernel void headwise_rmsnorm(
    device const float* x [[buffer(0)]],
    device const float* weight [[buffer(1)]],
    device float* out [[buffer(2)]],
    constant uint& num_heads [[buffer(3)]],
    constant uint& head_dim [[buffer(4)]],
    constant float& eps [[buffer(5)]],
    constant uint& head_dim_weight [[buffer(6)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]])
{
    if (tgid >= num_heads) return;
    uint base = tgid * head_dim;

    // If no weight (head_dim_weight == 0), identity pass-through
    if (head_dim_weight == 0) {
        for (uint i = tid; i < head_dim; i += HN_THREADS) {
            out[base + i] = x[base + i];
        }
        return;
    }

    // Parallel sum of squares
    threadgroup float shared_sum[HN_THREADS];
    float local_sum = 0.0;
    for (uint i = tid; i < head_dim; i += HN_THREADS) {
        float val = x[base + i];
        local_sum += val * val;
    }
    shared_sum[tid] = local_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = HN_THREADS / 2; s > 0; s >>= 1) {
        if (tid < s) shared_sum[tid] += shared_sum[tid + s];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    float rms = rsqrt(shared_sum[0] / float(head_dim) + eps);

    // Normalize and apply weight
    for (uint i = tid; i < head_dim; i += HN_THREADS) {
        out[base + i] = x[base + i] * rms * weight[i];
    }
}
