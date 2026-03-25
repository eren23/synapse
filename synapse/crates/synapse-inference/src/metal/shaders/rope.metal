// RoPE rotation (rotate_half convention): pairs (i, i+half_d)
// Operates on a single token's Q or K: [num_heads * head_dim]
// cos_row/sin_row: [half_d] for the given position (pre-indexed by caller)
kernel void rope_rotate_half(
    device float* qk [[buffer(0)]],
    device const float* cos_row [[buffer(1)]],
    device const float* sin_row [[buffer(2)]],
    constant uint& num_heads [[buffer(3)]],
    constant uint& head_dim [[buffer(4)]],
    uint tid [[thread_position_in_grid]])
{
    uint half_d = head_dim / 2;
    uint total_pairs = num_heads * half_d;
    if (tid >= total_pairs) return;

    uint head = tid / half_d;
    uint i = tid % half_d;
    uint base = head * head_dim;

    float c = cos_row[i];
    float s = sin_row[i];
    float x0 = qk[base + i];
    float x1 = qk[base + half_d + i];

    qk[base + i]          = x0 * c - x1 * s;
    qk[base + half_d + i] = x1 * c + x0 * s;
}
