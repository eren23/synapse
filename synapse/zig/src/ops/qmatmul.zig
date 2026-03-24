//! INT8 Quantized GEMM with per-channel scaling.
//!
//! C_f32[M,N] = diag(scales_a) * (A_int8 * B_int8) * diag(scales_b)
//!            = scales_a[i] * scales_b[j] * sum_k(A[i,k] * B[k,j])
//!
//! Tiling hierarchy (GOTO BLAS approach):
//! - L3: NC=4096 (partition N)
//! - L2: MC=256 (partition M), KC=512 (partition K)
//! - L1: MR=8 x NR=8 micro-kernel with SIMD integer multiply-accumulate
//! - Accumulate int8*int8 → int32 inside micro-kernel
//! - Scale to f32 at tile boundary (end of each KC chunk)

const std = @import("std");

// Tiling parameters
pub const MR: usize = 8;
pub const NR: usize = 8;
pub const MC: usize = 256;
pub const KC: usize = 512;
pub const NC: usize = 4096;

const VEC_LEN: usize = 4;
const I32x4 = @Vector(VEC_LEN, i32);
const F32x4 = @Vector(VEC_LEN, f32);
const VECS_PER_NR: usize = NR / VEC_LEN; // 2

// ================================================================
// Tiled INT8 GEMM
// ================================================================

/// Tiled INT8 GEMM on flat row-major arrays.
/// C[M,N] = scales_a[i] * scales_b[j] * sum_k(A_i8[i,k] * B_i8[k,j])
///
/// C is zeroed internally before accumulation.
/// packed_a: scratch buffer, size >= ceil(min(MC,m)/MR)*MR * min(KC,k)
/// packed_b: scratch buffer, size >= ceil(min(NC,n)/NR)*NR * min(KC,k)
pub fn int8GemmTiled(
    m: usize,
    n: usize,
    k: usize,
    a: [*]const i8,
    lda: usize,
    b: [*]const i8,
    ldb: usize,
    c: [*]f32,
    ldc: usize,
    scales_a: [*]const f32,
    scales_b: [*]const f32,
    packed_a: [*]i8,
    packed_b: [*]i8,
) void {
    // Fast path: M=1 GEMV (single-token decode in LLM inference).
    // Avoids tiling, packing, and the scalar edge fallback.
    if (m == 1) {
        int8GemvRow(n, k, a, ldb, b, ldb, c, scales_a[0], scales_b);
        return;
    }

    // Zero C
    for (0..m) |i| {
        @memset((c + i * ldc)[0..n], @as(f32, 0));
    }

    // L3: partition N
    var jc: usize = 0;
    while (jc < n) : (jc += NC) {
        const nc = @min(NC, n - jc);

        // L2: partition K
        var pc: usize = 0;
        while (pc < k) : (pc += KC) {
            const kc = @min(KC, k - pc);

            // Pack B[pc:pc+kc, jc:jc+nc] into NR-wide column panels
            packB(b, ldb, pc, jc, kc, nc, packed_b);

            // L2: partition M
            var ic: usize = 0;
            while (ic < m) : (ic += MC) {
                const mc = @min(MC, m - ic);

                // Pack A[ic:ic+mc, pc:pc+kc] into MR-wide row panels
                packA(a, lda, ic, pc, mc, kc, packed_a);

                // Macro kernel: multiply packed panels, accumulate scaled results into C
                macroKernel(mc, nc, kc, packed_a, packed_b, c + ic * ldc + jc, ldc, scales_a + ic, scales_b + jc);
            }
        }
    }
}

// ================================================================
// Naive INT8 GEMM (correctness baseline)
// ================================================================

/// Naive triple-loop INT8 GEMM with per-channel scaling. C is overwritten.
pub noinline fn naiveInt8Gemm(
    m: usize,
    n: usize,
    k: usize,
    a: [*]const i8,
    lda: usize,
    b: [*]const i8,
    ldb: usize,
    c: [*]f32,
    ldc: usize,
    scales_a: [*]const f32,
    scales_b: [*]const f32,
) void {
    for (0..m) |i| {
        for (0..n) |j| {
            var acc: i32 = 0;
            for (0..k) |p| {
                acc += @as(i32, a[i * lda + p]) * @as(i32, b[p * ldb + j]);
            }
            c[i * ldc + j] = @as(f32, @floatFromInt(acc)) * scales_a[i] * scales_b[j];
        }
    }
}

// ================================================================
// Naive f32 GEMM (for benchmark comparison)
// ================================================================

/// Naive triple-loop f32 GEMM. C is overwritten.
pub noinline fn naiveF32Gemm(
    m: usize,
    n: usize,
    k: usize,
    a: [*]const f32,
    lda: usize,
    b: [*]const f32,
    ldb: usize,
    c: [*]f32,
    ldc: usize,
) void {
    for (0..m) |i| {
        for (0..n) |j| {
            var sum: f32 = 0;
            for (0..k) |p| {
                sum += a[i * lda + p] * b[p * ldb + j];
            }
            c[i * ldc + j] = sum;
        }
    }
}

// ================================================================
// M=1 GEMV (single-token decode fast path)
// ================================================================

const I8x4 = @Vector(4, i8);

/// Specialized INT8 matrix-vector multiply for M=1.
///
/// C[1,N] = scale_a * diag(scales_b) * (A[1,K] @ B[K,N])
///
/// No packing needed. Processes 32 output columns at a time with
/// 8 × I32x4 accumulators. Uses vector loads + @intCast widening
/// (compiles to NEON sxtl) instead of scalar byte loads.
/// 4-unrolled over K to hide load latency.
fn int8GemvRow(
    n: usize,
    k: usize,
    a: [*]const i8,
    _lda: usize,
    b: [*]const i8,
    ldb: usize,
    c: [*]f32,
    scale_a: f32,
    scales_b: [*]const f32,
) void {
    _ = _lda;
    const GEMV_NR: usize = 32; // Process 32 columns = 8 × I32x4

    // Process 32 columns at a time (8 × I32x4 accumulators)
    var j: usize = 0;
    while (j + GEMV_NR <= n) : (j += GEMV_NR) {
        const zero: I32x4 = @splat(@as(i32, 0));
        var acc: [8]I32x4 = .{zero} ** 8;

        // Main K loop — 4-unrolled for ILP
        var p: usize = 0;
        const k4 = k - (k % 4);
        while (p < k4) : (p += 4) {
            inline for (0..4) |u| {
                const a_val: i32 = a[p + u];
                const a_bcast: I32x4 = @splat(a_val);
                const b_base = b + (p + u) * ldb + j;
                inline for (0..8) |v| {
                    const b_i8: I8x4 = (b_base + v * 4)[0..4].*;
                    const b_i32: I32x4 = @intCast(b_i8);
                    acc[v] += a_bcast * b_i32;
                }
            }
        }
        // K remainder
        while (p < k) : (p += 1) {
            const a_val: i32 = a[p];
            const a_bcast: I32x4 = @splat(a_val);
            const b_base = b + p * ldb + j;
            inline for (0..8) |v| {
                const b_i8: I8x4 = (b_base + v * 4)[0..4].*;
                const b_i32: I32x4 = @intCast(b_i8);
                acc[v] += a_bcast * b_i32;
            }
        }

        // Convert to f32, apply scales, write to C
        const sa: F32x4 = @splat(scale_a);
        inline for (0..8) |v| {
            const f_acc: F32x4 = @floatFromInt(acc[v]);
            const sb: F32x4 = (scales_b + j + v * 4)[0..VEC_LEN].*;
            (c + j + v * 4)[0..VEC_LEN].* = f_acc * sa * sb;
        }
    }

    // Handle remaining columns with 8-wide blocks
    while (j + 8 <= n) : (j += 8) {
        const zero: I32x4 = @splat(@as(i32, 0));
        var acc_lo: I32x4 = zero;
        var acc_hi: I32x4 = zero;

        for (0..k) |p| {
            const a_val: i32 = a[p];
            const a_bcast: I32x4 = @splat(a_val);
            const b_base = b + p * ldb + j;
            const b_lo_i8: I8x4 = b_base[0..4].*;
            const b_hi_i8: I8x4 = (b_base + 4)[0..4].*;
            acc_lo += a_bcast * @as(I32x4, @intCast(b_lo_i8));
            acc_hi += a_bcast * @as(I32x4, @intCast(b_hi_i8));
        }

        const sa: F32x4 = @splat(scale_a);
        (c + j)[0..VEC_LEN].* = @as(F32x4, @floatFromInt(acc_lo)) * sa * (scales_b + j)[0..VEC_LEN].*;
        (c + j + 4)[0..VEC_LEN].* = @as(F32x4, @floatFromInt(acc_hi)) * sa * (scales_b + j + 4)[0..VEC_LEN].*;
    }

    // Scalar tail for remaining columns
    while (j < n) : (j += 1) {
        var acc: i32 = 0;
        for (0..k) |p| {
            acc += @as(i32, a[p]) * @as(i32, b[p * ldb + j]);
        }
        c[j] = @as(f32, @floatFromInt(acc)) * scale_a * scales_b[j];
    }
}

// ================================================================
// Packing routines
// ================================================================

/// Pack an MC×KC block of A (starting at row ic, col pc) into MR-wide row panels.
/// Zero-pads partial panels when mc is not a multiple of MR.
fn packA(
    a: [*]const i8,
    lda: usize,
    ic: usize,
    pc: usize,
    mc: usize,
    kc: usize,
    dst: [*]i8,
) void {
    var idx: usize = 0;
    var ir: usize = 0;
    while (ir < mc) : (ir += MR) {
        const mr = @min(MR, mc - ir);
        for (0..kc) |p| {
            var r: usize = 0;
            while (r < mr) : (r += 1) {
                dst[idx] = a[(ic + ir + r) * lda + (pc + p)];
                idx += 1;
            }
            // Zero-pad remainder to MR boundary
            while (r < MR) : (r += 1) {
                dst[idx] = 0;
                idx += 1;
            }
        }
    }
}

/// Pack a KC×NC block of B (starting at row pc, col jc) into NR-wide column panels.
/// Zero-pads partial panels when nc is not a multiple of NR.
fn packB(
    b: [*]const i8,
    ldb: usize,
    pc: usize,
    jc: usize,
    kc: usize,
    nc: usize,
    dst: [*]i8,
) void {
    var idx: usize = 0;
    var jr: usize = 0;
    while (jr < nc) : (jr += NR) {
        const nr = @min(NR, nc - jr);
        for (0..kc) |p| {
            var col: usize = 0;
            while (col < nr) : (col += 1) {
                dst[idx] = b[(pc + p) * ldb + (jc + jr + col)];
                idx += 1;
            }
            // Zero-pad remainder to NR boundary
            while (col < NR) : (col += 1) {
                dst[idx] = 0;
                idx += 1;
            }
        }
    }
}

// ================================================================
// Macro kernel + micro kernels
// ================================================================

/// Multiply packed A (mc×kc) and packed B (kc×nc), scale and accumulate into C.
fn macroKernel(
    mc: usize,
    nc: usize,
    kc: usize,
    packed_a: [*]const i8,
    packed_b: [*]const i8,
    c: [*]f32,
    ldc: usize,
    scales_a: [*]const f32,
    scales_b: [*]const f32,
) void {
    var jr: usize = 0;
    while (jr < nc) : (jr += NR) {
        const nr = @min(NR, nc - jr);
        const b_panel = packed_b + (jr / NR) * NR * kc;

        var ir: usize = 0;
        while (ir < mc) : (ir += MR) {
            const mr = @min(MR, mc - ir);
            const a_panel = packed_a + (ir / MR) * MR * kc;

            if (mr == MR and nr == NR) {
                microKernel8x8(kc, a_panel, b_panel, c + ir * ldc + jr, ldc, scales_a + ir, scales_b + jr);
            } else {
                microKernelEdge(mr, nr, kc, a_panel, b_panel, c + ir * ldc + jr, ldc, scales_a + ir, scales_b + jr);
            }
        }
    }
}

/// 8×8 INT8 micro-kernel with SIMD integer multiply-accumulate.
/// Accumulates int8×int8 → int32, then scales to f32 and adds to C at tile boundary.
/// Uses 16 I32x4 accumulators (8 rows × 2 groups of 4 columns).
inline fn microKernel8x8(
    kc: usize,
    a: [*]const i8,
    b: [*]const i8,
    c: [*]f32,
    ldc: usize,
    row_scales: [*]const f32,
    col_scales: [*]const f32,
) void {
    // Initialize 8×8 int32 accumulators to zero
    const zero: I32x4 = @splat(@as(i32, 0));
    var c_acc: [MR][VECS_PER_NR]I32x4 = .{.{zero} ** VECS_PER_NR} ** MR;

    // Main accumulation loop over K dimension
    for (0..kc) |p| {
        const a_ptr = a + p * MR;
        const b_ptr = b + p * NR;

        // Load NR int8 values of B, widen to I32x4 vectors
        var b_vec: [VECS_PER_NR]I32x4 = undefined;
        for (0..VECS_PER_NR) |v| {
            const off = v * VEC_LEN;
            b_vec[v] = I32x4{
                @as(i32, b_ptr[off + 0]),
                @as(i32, b_ptr[off + 1]),
                @as(i32, b_ptr[off + 2]),
                @as(i32, b_ptr[off + 3]),
            };
        }

        // Broadcast each A element and multiply-accumulate
        for (0..MR) |i| {
            const a_val: i32 = a_ptr[i];
            const a_bcast: I32x4 = @splat(a_val);
            for (0..VECS_PER_NR) |v| {
                c_acc[i][v] += a_bcast * b_vec[v];
            }
        }
    }

    // Convert to f32, apply per-channel scales, add to C
    for (0..MR) |i| {
        const sa: F32x4 = @splat(row_scales[i]);
        for (0..VECS_PER_NR) |v| {
            const f_acc: F32x4 = @floatFromInt(c_acc[i][v]);
            const sb: F32x4 = (col_scales + v * VEC_LEN)[0..VEC_LEN].*;
            const c_ptr = (c + i * ldc + v * VEC_LEN);
            const existing: F32x4 = c_ptr[0..VEC_LEN].*;
            c_ptr[0..VEC_LEN].* = existing + f_acc * sa * sb;
        }
    }
}

/// Scalar fallback micro-kernel for edge tiles (mr < MR or nr < NR).
fn microKernelEdge(
    mr: usize,
    nr: usize,
    kc: usize,
    a: [*]const i8,
    b: [*]const i8,
    c: [*]f32,
    ldc: usize,
    row_scales: [*]const f32,
    col_scales: [*]const f32,
) void {
    for (0..mr) |i| {
        for (0..nr) |j| {
            var acc: i32 = 0;
            for (0..kc) |p| {
                acc += @as(i32, a[p * MR + i]) * @as(i32, b[p * NR + j]);
            }
            c[i * ldc + j] += @as(f32, @floatFromInt(acc)) * row_scales[i] * col_scales[j];
        }
    }
}
