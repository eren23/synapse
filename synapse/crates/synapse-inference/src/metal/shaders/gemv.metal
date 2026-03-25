// GEMV: y[N] = A[1,K] * B[K,N]  (M=1 specialization)
// One thread per output element. All threads read the same A vector (cache-friendly).
// B is stored row-major: B[k][n] = b[k * N + n].
//
// For K=1024, N=1024: 1024 threads, each doing 1024 multiply-adds.
// Much more efficient than the 32×32 tiled matmul which wastes 31/32 rows for M=1.
kernel void gemv(
    device const float* a [[buffer(0)]],
    device const float* b [[buffer(1)]],
    device float* c [[buffer(2)]],
    constant uint& N [[buffer(3)]],
    constant uint& K [[buffer(4)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid >= N) return;
    float sum = 0.0;
    for (uint k = 0; k < K; k++) {
        sum += a[k] * b[k * N + tid];
    }
    c[tid] = sum;
}
