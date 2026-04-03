// Q4_0 GEMV: y[N] = A_f32[1,K] * dequant(B_q4[N, K/32 blocks])
//
// B is stored in GGUF Q4_0 format: each block = 18 bytes:
//   [f16 scale (2 bytes)][16 nibble bytes (32 values)]
// Nibble decoding: low nibble first, value = (nibble - 8) * scale
//
// Layout: B has N rows, each row has (K/32) blocks = (K/32)*18 bytes.
// Row j starts at b_q4[j * row_bytes].
//
// One thread per output element (N threads). Each thread reads one full
// row of Q4 blocks (K/32 blocks * 18 bytes) and the shared A vector.
// Reads ~K/2 bytes per thread vs K*4 for f32 — 8x bandwidth reduction.

#include <metal_stdlib>
using namespace metal;

// Decode f16 (IEEE 754 half) to float
inline float f16_to_float(uint16_t bits) {
    // Use Metal's built-in half type for hardware conversion
    half h = as_type<half>(bits);
    return float(h);
}

kernel void gemv_q4(
    device const float* a [[buffer(0)]],       // [K] f32 activations
    device const uchar* b_q4 [[buffer(1)]],    // [N * row_bytes] raw Q4_0 blocks
    device float* c [[buffer(2)]],             // [N] f32 output
    constant uint& N [[buffer(3)]],            // output dimension
    constant uint& K [[buffer(4)]],            // input dimension
    uint tid [[thread_position_in_grid]])
{
    if (tid >= N) return;

    const uint blocks_per_row = (K + 31) / 32;
    const uint block_bytes = 18;  // f16 scale (2) + 16 nibble bytes
    const uint row_bytes = blocks_per_row * block_bytes;

    // Pointer to this row's Q4 blocks
    device const uchar* row = b_q4 + tid * row_bytes;

    float sum = 0.0;
    uint a_idx = 0;

    for (uint b = 0; b < blocks_per_row; b++) {
        device const uchar* block = row + b * block_bytes;

        // Read f16 scale (GGUF stores as f16 LE)
        uint16_t scale_bits = uint16_t(block[0]) | (uint16_t(block[1]) << 8);
        float scale = f16_to_float(scale_bits);
        if (!isfinite(scale)) scale = 0.0;  // guard against NaN/Inf from corrupt blocks

        // Process 32 elements from 16 nibble bytes
        for (uint i = 0; i < 16; i++) {
            uchar nibble_byte = block[2 + i];
            // Low nibble: bits [0:3], offset by 8 → signed range [-8, +7]
            int lo_val = int(nibble_byte & 0x0Fu) - 8;
            // High nibble: bits [4:7]
            int hi_val = int(nibble_byte >> 4) - 8;

            if (a_idx < K) {
                sum += a[a_idx] * (float(lo_val) * scale);
                a_idx++;
            }
            if (a_idx < K) {
                sum += a[a_idx] * (float(hi_val) * scale);
                a_idx++;
            }
        }
    }

    c[tid] = sum;
}
