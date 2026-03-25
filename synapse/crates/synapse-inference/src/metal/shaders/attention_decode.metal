// Single-query decode attention with GQA support.
// One threadgroup per query head, 256 threads per threadgroup.
//
// Computes Q·K^T scores ONCE, stores in threadgroup memory, then reuses
// for softmax normalization and V weighted sum.
//
// Q: [q_dim] = [num_heads * head_dim]
// K_cache, V_cache: [max_seq, kv_dim] where kv_dim = num_kv_heads * head_dim
// out: [q_dim]
//
// Threadgroup memory: 8192 floats = 32KB (supports seq_len up to 8192).

constant uint ATTN_DECODE_THREADS = 256;
constant uint MAX_CACHED_SEQ = 4096;

kernel void attention_decode(
    device const float* Q [[buffer(0)]],
    device const float* K_cache [[buffer(1)]],
    device const float* V_cache [[buffer(2)]],
    device float* out [[buffer(3)]],
    constant uint& num_heads [[buffer(4)]],
    constant uint& num_kv_heads [[buffer(5)]],
    constant uint& head_dim [[buffer(6)]],
    constant uint& seq_len [[buffer(7)]],
    constant uint& kv_dim [[buffer(8)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]])
{
    uint head = tgid;
    if (head >= num_heads) return;

    uint groups = num_heads / num_kv_heads;
    uint kv_head = head / groups;
    float scale = rsqrt(float(head_dim));

    // Shared memory for scores (computed once, reused for softmax + V weighting)
    threadgroup float scores[MAX_CACHED_SEQ];
    threadgroup float shared_reduce[ATTN_DECODE_THREADS];

    uint effective_seq = min(seq_len, MAX_CACHED_SEQ);

    // ── Phase 1: Compute all Q·K^T scores (ONCE) ─────────────────
    for (uint j = tid; j < effective_seq; j += ATTN_DECODE_THREADS) {
        float dot = 0.0;
        for (uint d = 0; d < head_dim; d++) {
            dot += Q[head * head_dim + d] * K_cache[j * kv_dim + kv_head * head_dim + d];
        }
        scores[j] = dot * scale;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ── Phase 2: Softmax — find max ──────────────────────────────
    float local_max = -INFINITY;
    for (uint j = tid; j < effective_seq; j += ATTN_DECODE_THREADS) {
        if (scores[j] > local_max) local_max = scores[j];
    }
    shared_reduce[tid] = local_max;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = ATTN_DECODE_THREADS / 2; s > 0; s >>= 1) {
        if (tid < s) shared_reduce[tid] = max(shared_reduce[tid], shared_reduce[tid + s]);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float global_max = shared_reduce[0];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ── Phase 3: Softmax — compute exp and sum ───────────────────
    float local_sum = 0.0;
    for (uint j = tid; j < effective_seq; j += ATTN_DECODE_THREADS) {
        float e = exp(scores[j] - global_max);
        scores[j] = e;  // overwrite score with exp(score - max)
        local_sum += e;
    }
    shared_reduce[tid] = local_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = ATTN_DECODE_THREADS / 2; s > 0; s >>= 1) {
        if (tid < s) shared_reduce[tid] += shared_reduce[tid + s];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float inv_sum = (shared_reduce[0] > 0.0) ? (1.0 / shared_reduce[0]) : 0.0;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ── Phase 4: Normalize scores in-place ───────────────────────
    for (uint j = tid; j < effective_seq; j += ATTN_DECODE_THREADS) {
        scores[j] *= inv_sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ── Phase 5: Weighted V sum ──────────────────────────────────
    // Each thread computes a subset of head_dim output elements
    for (uint d = tid; d < head_dim; d += ATTN_DECODE_THREADS) {
        float val = 0.0;
        for (uint j = 0; j < effective_seq; j++) {
            val += scores[j] * V_cache[j * kv_dim + kv_head * head_dim + d];
        }
        out[head * head_dim + d] = val;
    }
}
