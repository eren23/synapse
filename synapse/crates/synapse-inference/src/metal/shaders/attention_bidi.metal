#include <metal_stdlib>
using namespace metal;

constant uint BIDI_ATTN_THREADS = 256;

/// Bidirectional scaled dot-product attention with strided per-head access.
///
/// Q, K, V pointers are offset to the start of the target head.
/// Elements for query position `t` at dimension `d` are at offset `t * stride + d`.
/// stride = hidden (total width of all heads interleaved).
/// head_dim = width of one head's data within each row.
///
/// out[t * stride + d] = softmax(Q[t,:] · K^T / √head_dim) · V[:,d]
///
/// Dispatch: threadgroups = seq_len, threads_per_threadgroup = 256
kernel void attention_bidi(
    device const float* Q [[buffer(0)]],
    device const float* K [[buffer(1)]],
    device const float* V [[buffer(2)]],
    device float* out [[buffer(3)]],
    constant uint& seq_len [[buffer(4)]],
    constant uint& head_dim [[buffer(5)]],
    constant uint& stride [[buffer(6)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]])
{
    uint q_pos = tgid;
    if (q_pos >= seq_len) return;

    float scale = rsqrt(float(head_dim));
    threadgroup float shared[BIDI_ATTN_THREADS];

    // Phase 1: find max score
    float local_max = -INFINITY;
    for (uint j = tid; j < seq_len; j += BIDI_ATTN_THREADS) {
        float score = 0.0;
        for (uint d = 0; d < head_dim; d++) {
            score += Q[q_pos * stride + d] * K[j * stride + d];
        }
        local_max = max(local_max, score * scale);
    }
    shared[tid] = local_max;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = BIDI_ATTN_THREADS / 2; s > 0; s >>= 1) {
        if (tid < s) shared[tid] = max(shared[tid], shared[tid + s]);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float global_max = shared[0];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Phase 2: softmax denominator
    float local_sum = 0.0;
    for (uint j = tid; j < seq_len; j += BIDI_ATTN_THREADS) {
        float score = 0.0;
        for (uint d = 0; d < head_dim; d++) {
            score += Q[q_pos * stride + d] * K[j * stride + d];
        }
        local_sum += exp(score * scale - global_max);
    }
    shared[tid] = local_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = BIDI_ATTN_THREADS / 2; s > 0; s >>= 1) {
        if (tid < s) shared[tid] += shared[tid + s];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float global_sum = shared[0];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Phase 3: weighted V, threads split over head_dim
    for (uint d = tid; d < head_dim; d += BIDI_ATTN_THREADS) {
        float val = 0.0;
        for (uint j = 0; j < seq_len; j++) {
            float score = 0.0;
            for (uint dd = 0; dd < head_dim; dd++) {
                score += Q[q_pos * stride + dd] * K[j * stride + dd];
            }
            float w = exp(score * scale - global_max) / global_sum;
            val += w * V[j * stride + d];
        }
        out[q_pos * stride + d] = val;
    }
}
