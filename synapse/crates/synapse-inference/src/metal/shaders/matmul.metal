#include <metal_stdlib>
using namespace metal;

constant uint TILE_SIZE = 32;

/// Tiled GEMM: C[M,N] = A[M,K] * B[K,N]
/// Uses 32x32 threadgroup shared memory tiles for coalesced access.
/// Handles arbitrary M, N, K with bounds-checked loads (zero-padded).
///
/// Dispatch: threadgroups = ceil(N/32) x ceil(M/32), threads_per_threadgroup = 32x32
kernel void matmul(
    device const float* a [[buffer(0)]],
    device const float* b [[buffer(1)]],
    device float* c [[buffer(2)]],
    constant uint& M [[buffer(3)]],
    constant uint& N [[buffer(4)]],
    constant uint& K [[buffer(5)]],
    uint2 tid [[thread_position_in_threadgroup]],
    uint2 tgid [[threadgroup_position_in_grid]])
{
    threadgroup float a_tile[TILE_SIZE][TILE_SIZE];
    threadgroup float b_tile[TILE_SIZE][TILE_SIZE];

    uint row = tgid.y * TILE_SIZE + tid.y;
    uint col = tgid.x * TILE_SIZE + tid.x;

    float sum = 0.0;
    uint num_tiles = (K + TILE_SIZE - 1) / TILE_SIZE;

    for (uint t = 0; t < num_tiles; t++) {
        uint a_col = t * TILE_SIZE + tid.x;
        uint b_row = t * TILE_SIZE + tid.y;

        a_tile[tid.y][tid.x] = (row < M && a_col < K) ? a[row * K + a_col] : 0.0;
        b_tile[tid.y][tid.x] = (b_row < K && col < N) ? b[b_row * N + col] : 0.0;

        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint k = 0; k < TILE_SIZE; k++) {
            sum += a_tile[tid.y][k] * b_tile[k][tid.x];
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (row < M && col < N) {
        c[row * N + col] = sum;
    }
}

#if __METAL_VERSION__ >= 300
/// SIMD-group accelerated GEMM using simdgroup_float8x8 matrix operations.
/// Requires Apple GPU family 7+ (M1/A14 and later) with Metal 3.0.
/// Uses 8x8 simdgroup tiles within 32x32 threadgroup tiles for higher throughput.
///
/// Dispatch: same as matmul — threadgroups = ceil(N/32) x ceil(M/32), threads = 32x32
kernel void matmul_simd(
    device const float* a [[buffer(0)]],
    device const float* b [[buffer(1)]],
    device float* c [[buffer(2)]],
    constant uint& M [[buffer(3)]],
    constant uint& N [[buffer(4)]],
    constant uint& K [[buffer(5)]],
    uint2 tid [[thread_position_in_threadgroup]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]])
{
    threadgroup float a_tile[TILE_SIZE][TILE_SIZE];
    threadgroup float b_tile[TILE_SIZE][TILE_SIZE];

    uint row = tgid.y * TILE_SIZE + tid.y;
    uint col = tgid.x * TILE_SIZE + tid.x;

    // Each simdgroup handles a 8x8 sub-tile within the 32x32 threadgroup tile.
    // 32x32 = 1024 threads / 32 per simdgroup = 32 simdgroups.
    // Arrange as 4x8 grid of 8x8 sub-tiles.
    uint sg_row = (simd_group_id / 4) * 8;
    uint sg_col = (simd_group_id % 4) * 8;

    simdgroup_float8x8 acc = simdgroup_float8x8(0);

    uint num_tiles = (K + TILE_SIZE - 1) / TILE_SIZE;

    for (uint t = 0; t < num_tiles; t++) {
        uint a_col = t * TILE_SIZE + tid.x;
        uint b_row = t * TILE_SIZE + tid.y;

        a_tile[tid.y][tid.x] = (row < M && a_col < K) ? a[row * K + a_col] : 0.0;
        b_tile[tid.y][tid.x] = (b_row < K && col < N) ? b[b_row * N + col] : 0.0;

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Multiply 8x8 sub-tiles using simdgroup matrix ops
        for (uint k = 0; k < TILE_SIZE; k += 8) {
            simdgroup_float8x8 ma;
            simdgroup_float8x8 mb;
            simdgroup_load(ma, &a_tile[sg_row][k], TILE_SIZE);
            simdgroup_load(mb, &b_tile[k][sg_col], TILE_SIZE);
            simdgroup_multiply_accumulate(acc, ma, mb, acc);
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Store 8x8 result sub-tile back to C via shared memory
    simdgroup_store(acc, &a_tile[sg_row][sg_col], TILE_SIZE);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (row < M && col < N) {
        c[row * N + col] = a_tile[tid.y][tid.x];
    }
}
#endif
