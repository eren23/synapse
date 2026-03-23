#include <metal_stdlib>
using namespace metal;

constant uint ATTN_THREADS = 256;

/// Fused scaled dot-product attention with causal masking.
///
/// Q[seq_len, head_dim], K[kv_len, head_dim], V[kv_len, head_dim] -> out[seq_len, head_dim]
/// score = Q * K^T / sqrt(head_dim), causal mask (j > q_pos => -inf), softmax, * V
///
/// Each threadgroup processes one query position.
/// Phase 1-2: threads split over kv_len for numerically-stable softmax (parallel reduction).
/// Phase 3: threads split over head_dim, each computes its output dimension independently.
///
/// Dispatch: threadgroups = seq_len, threads_per_threadgroup = 256
kernel void attention(
    device const float* Q [[buffer(0)]],
    device const float* K [[buffer(1)]],
    device const float* V [[buffer(2)]],
    device float* out [[buffer(3)]],
    constant uint& seq_len [[buffer(4)]],
    constant uint& kv_len [[buffer(5)]],
    constant uint& head_dim [[buffer(6)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]])
{
    uint q_pos = tgid;
    if (q_pos >= seq_len) return;

    float scale = rsqrt(float(head_dim));
    threadgroup float shared[ATTN_THREADS];

    // Effective kv length after causal masking
    uint causal_len = min(q_pos + 1, kv_len);

    // --- Phase 1: Find max attention score (numerical stability) ---
    float local_max = -INFINITY;
    for (uint j = tid; j < causal_len; j += ATTN_THREADS) {
        float score = 0.0;
        for (uint d = 0; d < head_dim; d++) {
            score += Q[q_pos * head_dim + d] * K[j * head_dim + d];
        }
        local_max = max(local_max, score * scale);
    }

    shared[tid] = local_max;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = ATTN_THREADS / 2; s > 0; s >>= 1) {
        if (tid < s) shared[tid] = max(shared[tid], shared[tid + s]);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float global_max = shared[0];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // --- Phase 2: Compute sum(exp(score - max)) for softmax denominator ---
    float local_sum = 0.0;
    for (uint j = tid; j < causal_len; j += ATTN_THREADS) {
        float score = 0.0;
        for (uint d = 0; d < head_dim; d++) {
            score += Q[q_pos * head_dim + d] * K[j * head_dim + d];
        }
        local_sum += exp(score * scale - global_max);
    }

    shared[tid] = local_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = ATTN_THREADS / 2; s > 0; s >>= 1) {
        if (tid < s) shared[tid] += shared[tid + s];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float global_sum = shared[0];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // --- Phase 3: Compute weighted output = softmax(scores) * V ---
    // Threads split over head_dim; each computes one output element independently.
    for (uint d = tid; d < head_dim; d += ATTN_THREADS) {
        float val = 0.0;
        for (uint j = 0; j < causal_len; j++) {
            float score = 0.0;
            for (uint dd = 0; dd < head_dim; dd++) {
                score += Q[q_pos * head_dim + dd] * K[j * head_dim + dd];
            }
            float weight = exp(score * scale - global_max) / global_sum;
            val += weight * V[j * head_dim + d];
        }
        out[q_pos * head_dim + d] = val;
    }
}
