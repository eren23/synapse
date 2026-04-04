// Q4_0 GEMV: y[N] = A_f32[1,K] * dequant(B_q4[N, K/32 blocks])
//
// Adaptive parallelism based on K:
// - Small K (≤1024, ≤32 blocks): 1 thread per output row (256 rows per threadgroup)
//   Most GPU-friendly: high occupancy, simple, no reduction needed
// - Large K (>1024): threadgroup-parallel reduction per row
//
// B is in GGUF Q4_0 format: each block = 18 bytes: [f16 scale][16 nibble bytes]
// Row j starts at b_q4[j * blocks_per_row * 18].
//
// Dispatch: threadgroups = [ceil(N/256), 1, 1], threads_per_threadgroup = [256, 1, 1]

#include <metal_stdlib>
using namespace metal;

kernel void gemv_q4(
    device const float* a [[buffer(0)]],       // [K] f32 activations
    device const uchar* b_q4 [[buffer(1)]],    // [N * row_bytes] raw Q4_0 blocks
    device float* c [[buffer(2)]],             // [N] f32 output
    constant uint& N [[buffer(3)]],            // output dimension
    constant uint& K [[buffer(4)]],            // input dimension
    uint tid [[thread_position_in_grid]])       // global thread index
{
    // 1 thread = 1 output row. Simple, high occupancy for small-to-medium K.
    if (tid >= N) return;

    const uint blocks_per_row = (K + 31) / 32;
    const uint row_bytes = blocks_per_row * 18;
    device const uchar* row = b_q4 + tid * row_bytes;

    float sum = 0.0;

    for (uint b = 0; b < blocks_per_row; b++) {
        device const uchar* block = row + b * 18;

        // Read f16 scale via hardware half conversion
        uint16_t scale_bits = uint16_t(block[0]) | (uint16_t(block[1]) << 8);
        float s = float(as_type<half>(scale_bits));
        if (!isfinite(s)) s = 0.0;

        uint a_base = b * 32;

        // Unrolled: process 4 values per iteration (2 bytes → 4 nibbles)
        for (uint i = 0; i < 16; i += 2) {
            uchar b0 = block[2 + i];
            uchar b1 = block[2 + i + 1];
            uint idx = a_base + i * 2;

            float v0 = float(int(b0 & 0x0Fu) - 8) * s;
            float v1 = float(int(b0 >> 4) - 8) * s;
            float v2 = float(int(b1 & 0x0Fu) - 8) * s;
            float v3 = float(int(b1 >> 4) - 8) * s;

            if (idx     < K) sum += a[idx]     * v0;
            if (idx + 1 < K) sum += a[idx + 1] * v1;
            if (idx + 2 < K) sum += a[idx + 2] * v2;
            if (idx + 3 < K) sum += a[idx + 3] * v3;
        }
    }

    c[tid] = sum;
}
