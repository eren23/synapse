// INT8 GEMV: y[N] = sum_k(A_f32[k] * dequant(B_int8[k,N])) * scale[N]
//
// Weights are stored as INT8 with per-column f32 scales.
// One thread per output element. Each thread:
// 1. Reads all K elements of A (f32, cached across threads)
// 2. Reads K int8 weights from its column of B
// 3. Accumulates int32, then converts to f32 and applies scale
//
// This reads 1 byte per weight instead of 4 — 4x bandwidth reduction.

kernel void gemv_int8(
    device const float* a [[buffer(0)]],        // [K] f32 activations
    device const char* b_int8 [[buffer(1)]],    // [K, N] int8 weights (row-major)
    device const float* scales [[buffer(2)]],    // [N] per-column scale
    device float* c [[buffer(3)]],               // [N] f32 output
    constant uint& N [[buffer(4)]],
    constant uint& K [[buffer(5)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid >= N) return;

    // Accumulate in int32 for precision, convert at the end
    float sum = 0.0;

    // Process 4 elements at a time for better ILP
    uint k = 0;
    uint k4 = K & ~3u;  // K rounded down to multiple of 4
    for (; k < k4; k += 4) {
        float a0 = a[k], a1 = a[k+1], a2 = a[k+2], a3 = a[k+3];
        float b0 = float(b_int8[(k+0) * N + tid]);
        float b1 = float(b_int8[(k+1) * N + tid]);
        float b2 = float(b_int8[(k+2) * N + tid]);
        float b3 = float(b_int8[(k+3) * N + tid]);
        sum += a0 * b0 + a1 * b1 + a2 * b2 + a3 * b3;
    }
    // Remainder
    for (; k < K; k++) {
        sum += a[k] * float(b_int8[k * N + tid]);
    }

    c[tid] = sum * scales[tid];
}
