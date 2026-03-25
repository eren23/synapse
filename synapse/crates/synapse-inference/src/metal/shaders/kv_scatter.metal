// Append K and V token vectors at position `pos` into the cache buffers.
// k_cache, v_cache: [max_seq, kv_dim] row-major
// k_token, v_token: [kv_dim]
kernel void kv_cache_scatter(
    device float* k_cache [[buffer(0)]],
    device float* v_cache [[buffer(1)]],
    device const float* k_token [[buffer(2)]],
    device const float* v_token [[buffer(3)]],
    constant uint& pos [[buffer(4)]],
    constant uint& kv_dim [[buffer(5)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid >= kv_dim) return;
    uint offset = pos * kv_dim + tid;
    k_cache[offset] = k_token[tid];
    v_cache[offset] = v_token[tid];
}
