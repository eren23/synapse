// Single-query decode attention with GQA support.
// One threadgroup per query head, 256 threads per threadgroup.
// 3-phase: (1) find max score, (2) compute softmax sum, (3) weighted output.
//
// Q: [q_dim] = [num_heads * head_dim]
// K_cache, V_cache: [max_seq, kv_dim] where kv_dim = num_kv_heads * head_dim
// out: [q_dim]

constant uint ATTN_DECODE_THREADS = 256;

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

    threadgroup float shared[ATTN_DECODE_THREADS];

    // ── Phase 1: max score ──────────────────────────────────────
    float local_max = -INFINITY;
    for (uint j = tid; j < seq_len; j += ATTN_DECODE_THREADS) {
        float dot = 0.0;
        for (uint d = 0; d < head_dim; d++) {
            dot += Q[head * head_dim + d] * K_cache[j * kv_dim + kv_head * head_dim + d];
        }
        float s = dot * scale;
        if (s > local_max) local_max = s;
    }
    shared[tid] = local_max;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = ATTN_DECODE_THREADS / 2; s > 0; s >>= 1) {
        if (tid < s) shared[tid] = max(shared[tid], shared[tid + s]);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float global_max = shared[0];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ── Phase 2: sum of exp(score - max) ────────────────────────
    float local_sum = 0.0;
    for (uint j = tid; j < seq_len; j += ATTN_DECODE_THREADS) {
        float dot = 0.0;
        for (uint d = 0; d < head_dim; d++) {
            dot += Q[head * head_dim + d] * K_cache[j * kv_dim + kv_head * head_dim + d];
        }
        local_sum += exp(dot * scale - global_max);
    }
    shared[tid] = local_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = ATTN_DECODE_THREADS / 2; s > 0; s >>= 1) {
        if (tid < s) shared[tid] += shared[tid + s];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float global_sum = shared[0];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float inv_sum = (global_sum > 0.0) ? (1.0 / global_sum) : 0.0;

    // ── Phase 3: weighted output ────────────────────────────────
    // Each thread computes a subset of head_dim output elements
    for (uint d = tid; d < head_dim; d += ATTN_DECODE_THREADS) {
        float val = 0.0;
        for (uint j = 0; j < seq_len; j++) {
            float dot = 0.0;
            for (uint dd = 0; dd < head_dim; dd++) {
                dot += Q[head * head_dim + dd] * K_cache[j * kv_dim + kv_head * head_dim + dd];
            }
            float w = exp(dot * scale - global_max) * inv_sum;
            val += w * V_cache[j * kv_dim + kv_head * head_dim + d];
        }
        out[head * head_dim + d] = val;
    }
}
