//! SGEMM (Single-precision General Matrix Multiplication) with tiled micro-kernel.
//!
//! C[M,N] = op(A) * op(B) where op(X) = X or X^T.
//!
//! Tiling hierarchy (GOTO BLAS approach):
//! - L3: NC=4096 (partition N)
//! - L2: MC=256 (partition M), KC=512 (partition K)
//! - L1: MR=8 x NR=8 micro-kernel with FMA
//! - A packed into MR-wide row panels, B packed into NR-wide column panels
//! - Edge cleanup via scalar fallback for non-tile-multiple dimensions

const std = @import("std");
const shape_mod = @import("../tensor/shape.zig");
const tensor_mod = @import("../tensor/tensor.zig");
const storage_mod = @import("../tensor/storage.zig");

const Shape = shape_mod.Shape;
const Tensor = tensor_mod.Tensor;
const Storage = storage_mod.Storage;

// Tiling parameters
pub const MR: usize = 8;
pub const NR: usize = 8;
pub const MC: usize = 256;
pub const KC: usize = 512;
pub const NC: usize = 4096;

const VEC_LEN: usize = 4;
const F32x4 = @Vector(VEC_LEN, f32);
const VECS_PER_NR: usize = NR / VEC_LEN; // 2

// ================================================================
// Tensor-level API
// ================================================================

/// Tiled SGEMM: C = op(A) * op(B) with 8x8 micro-kernel and cache-optimal packing.
pub fn matmul(
    allocator: std.mem.Allocator,
    a: Tensor(f32),
    b: Tensor(f32),
    trans_a: bool,
    trans_b: bool,
) !Tensor(f32) {
    if (a.shape.ndim != 2 or b.shape.ndim != 2) return error.InvalidDimensions;

    const m = if (trans_a) a.shape.dims[1] else a.shape.dims[0];
    const k_a = if (trans_a) a.shape.dims[0] else a.shape.dims[1];
    const k_b = if (trans_b) b.shape.dims[1] else b.shape.dims[0];
    const n = if (trans_b) b.shape.dims[0] else b.shape.dims[1];

    if (k_a != k_b) return error.ShapeMismatch;
    const k = k_a;

    const out_shape = Shape.init(&[_]usize{ m, n });
    const numel = if (m == 0 or n == 0) @as(usize, 1) else m * n;
    const out_storage = try Storage.create(allocator, f32, numel);
    const result = Tensor(f32).init(out_storage, out_shape);
    out_storage.release();

    if (m == 0 or n == 0 or k == 0) return result;

    // Tight packing buffer allocation
    const eff_kc = @min(KC, k);
    const eff_mc = ((@min(MC, m) + MR - 1) / MR) * MR;
    const eff_nc = ((@min(NC, n) + NR - 1) / NR) * NR;
    const packed_a = try allocator.alloc(f32, eff_mc * eff_kc);
    defer allocator.free(packed_a);
    const packed_b = try allocator.alloc(f32, eff_nc * eff_kc);
    defer allocator.free(packed_b);

    sgemmTiled(
        m, n, k,
        a.storage.dataAs(f32).ptr + a.offset, a.strides[0], trans_a,
        b.storage.dataAs(f32).ptr + b.offset, b.strides[0], trans_b,
        result.storage.dataAs(f32).ptr, n,
        packed_a.ptr, packed_b.ptr,
    );

    return result;
}

/// Naive triple-loop SGEMM: C = op(A) * op(B). For correctness baseline.
pub fn naiveMatmul(
    allocator: std.mem.Allocator,
    a: Tensor(f32),
    b: Tensor(f32),
    trans_a: bool,
    trans_b: bool,
) !Tensor(f32) {
    if (a.shape.ndim != 2 or b.shape.ndim != 2) return error.InvalidDimensions;

    const m = if (trans_a) a.shape.dims[1] else a.shape.dims[0];
    const k_a = if (trans_a) a.shape.dims[0] else a.shape.dims[1];
    const k_b = if (trans_b) b.shape.dims[1] else b.shape.dims[0];
    const n = if (trans_b) b.shape.dims[0] else b.shape.dims[1];

    if (k_a != k_b) return error.ShapeMismatch;
    const k = k_a;

    const out_shape = Shape.init(&[_]usize{ m, n });
    const numel = if (m == 0 or n == 0) @as(usize, 1) else m * n;
    const out_storage = try Storage.create(allocator, f32, numel);
    const result = Tensor(f32).init(out_storage, out_shape);
    out_storage.release();

    if (m == 0 or n == 0 or k == 0) return result;

    naiveSgemm(
        m, n, k,
        a.storage.dataAs(f32).ptr + a.offset, a.strides[0], trans_a,
        b.storage.dataAs(f32).ptr + b.offset, b.strides[0], trans_b,
        result.storage.dataAs(f32).ptr, n,
    );

    return result;
}

// ================================================================
// Raw SGEMM API (for benchmarks without tensor overhead)
// ================================================================

/// Tiled SGEMM on flat row-major arrays. C must be zeroed before first call.
/// packed_a must hold >= ceil(min(MC,m)/MR)*MR * min(KC,k) elements.
/// packed_b must hold >= ceil(min(NC,n)/NR)*NR * min(KC,k) elements.
pub fn sgemmTiled(
    m: usize,
    n: usize,
    k: usize,
    a: [*]const f32,
    lda: usize,
    trans_a: bool,
    b: [*]const f32,
    ldb: usize,
    trans_b: bool,
    c: [*]f32,
    ldc: usize,
    packed_a: [*]f32,
    packed_b: [*]f32,
) void {
    // Fast path: M=1 GEMV (single-token decode in LLM inference).
    // Avoids tiling/packing and the scalar microKernelEdge fallback.
    if (m == 1 and !trans_a) {
        f32GemvRow(n, k, a, b, ldb, trans_b, c);
        return;
    }

    // Fast path: skinny GEMM for M=2..16 (LEWM predict, small-batch inference).
    // Calls the GEMV kernel per row — avoids packA/packB overhead that dominates
    // at small M. The GEMV kernel uses 8×F32x4 SIMD accumulators internally.
    if (m <= 16 and !trans_a) {
        for (0..m) |i| {
            f32GemvRow(n, k, a + i * lda, b, ldb, trans_b, c + i * ldc);
        }
        return;
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
            packB(b, ldb, trans_b, pc, jc, kc, nc, packed_b);

            // L2: partition M
            var ic: usize = 0;
            while (ic < m) : (ic += MC) {
                const mc = @min(MC, m - ic);

                // Pack A[ic:ic+mc, pc:pc+kc] into MR-wide row panels
                packA(a, lda, trans_a, ic, pc, mc, kc, packed_a);

                // Macro kernel: multiply packed panels, accumulate into C
                macroKernel(mc, nc, kc, packed_a, packed_b, c + ic * ldc + jc, ldc);
            }
        }
    }
}

/// Naive triple-loop SGEMM on flat row-major arrays. C is overwritten (not accumulated).
pub noinline fn naiveSgemm(
    m: usize,
    n: usize,
    k: usize,
    a: [*]const f32,
    lda: usize,
    trans_a: bool,
    b: [*]const f32,
    ldb: usize,
    trans_b: bool,
    c: [*]f32,
    ldc: usize,
) void {
    for (0..m) |i| {
        for (0..n) |j| {
            var sum: f32 = 0;
            for (0..k) |p| {
                const a_val = if (trans_a) a[p * lda + i] else a[i * lda + p];
                const b_val = if (trans_b) b[j * ldb + p] else b[p * ldb + j];
                sum += a_val * b_val;
            }
            c[i * ldc + j] = sum;
        }
    }
}

// ================================================================
// M=1 GEMV (single-token decode fast path)
// ================================================================

/// Specialized f32 matrix-vector multiply for M=1.
///
/// C[1,N] = A[1,K] x B^T[K,N]  (when trans_b=true, B is stored [N,K])
/// C[1,N] = A[1,K] x B[K,N]    (when trans_b=false, B is stored [K,N])
///
/// Processes 32 output columns at a time with 8 x F32x4 FMA accumulators.
/// 4-unrolled over K to hide load latency.
fn f32GemvRow(
    n: usize,
    k: usize,
    a: [*]const f32,
    b: [*]const f32,
    ldb: usize,
    trans_b: bool,
    c: [*]f32,
) void {
    const GEMV_NR: usize = 32; // 8 × F32x4

    if (trans_b) {
        // B is [N, K] (row j has K elements). dot(A[0..K], B[j, 0..K]) for each j.
        // Process GEMV_NR output columns at a time.
        var j: usize = 0;
        while (j + GEMV_NR <= n) : (j += GEMV_NR) {
            var acc: [8]F32x4 = .{@as(F32x4, @splat(@as(f32, 0)))} ** 8;

            // 4-unrolled K loop
            var p: usize = 0;
            const k4 = k - (k % 4);
            while (p < k4) : (p += 4) {
                inline for (0..4) |u| {
                    const a_bcast: F32x4 = @splat(a[p + u]);
                    inline for (0..8) |v| {
                        // B[j+v*4..j+v*4+4, p+u] — row-major B[N,K], element at row (j+v*4+lane), col (p+u)
                        // For trans_b: B[row, col] = b[row * ldb + col]
                        const b_vec = F32x4{
                            b[(j + v * 4 + 0) * ldb + p + u],
                            b[(j + v * 4 + 1) * ldb + p + u],
                            b[(j + v * 4 + 2) * ldb + p + u],
                            b[(j + v * 4 + 3) * ldb + p + u],
                        };
                        acc[v] = @mulAdd(F32x4, a_bcast, b_vec, acc[v]);
                    }
                }
            }
            // K remainder
            while (p < k) : (p += 1) {
                const a_bcast: F32x4 = @splat(a[p]);
                inline for (0..8) |v| {
                    const b_vec = F32x4{
                        b[(j + v * 4 + 0) * ldb + p],
                        b[(j + v * 4 + 1) * ldb + p],
                        b[(j + v * 4 + 2) * ldb + p],
                        b[(j + v * 4 + 3) * ldb + p],
                    };
                    acc[v] = @mulAdd(F32x4, a_bcast, b_vec, acc[v]);
                }
            }

            inline for (0..8) |v| {
                (c + j + v * 4)[0..VEC_LEN].* = acc[v];
            }
        }

        // 8-wide cleanup
        while (j + 8 <= n) : (j += 8) {
            var acc_lo: F32x4 = @splat(@as(f32, 0));
            var acc_hi: F32x4 = @splat(@as(f32, 0));
            for (0..k) |p| {
                const a_bcast: F32x4 = @splat(a[p]);
                acc_lo = @mulAdd(F32x4, a_bcast, F32x4{
                    b[(j + 0) * ldb + p], b[(j + 1) * ldb + p],
                    b[(j + 2) * ldb + p], b[(j + 3) * ldb + p],
                }, acc_lo);
                acc_hi = @mulAdd(F32x4, a_bcast, F32x4{
                    b[(j + 4) * ldb + p], b[(j + 5) * ldb + p],
                    b[(j + 6) * ldb + p], b[(j + 7) * ldb + p],
                }, acc_hi);
            }
            (c + j)[0..VEC_LEN].* = acc_lo;
            (c + j + 4)[0..VEC_LEN].* = acc_hi;
        }

        // Scalar tail
        while (j < n) : (j += 1) {
            var sum: f32 = 0;
            for (0..k) |p| {
                sum += a[p] * b[j * ldb + p];
            }
            c[j] = sum;
        }
    } else {
        // B is [K, N] (column j spans K rows). Contiguous column access.
        var j: usize = 0;
        while (j + GEMV_NR <= n) : (j += GEMV_NR) {
            var acc: [8]F32x4 = .{@as(F32x4, @splat(@as(f32, 0)))} ** 8;

            var p: usize = 0;
            const k4 = k - (k % 4);
            while (p < k4) : (p += 4) {
                inline for (0..4) |u| {
                    const a_bcast: F32x4 = @splat(a[p + u]);
                    const b_base = b + (p + u) * ldb + j;
                    inline for (0..8) |v| {
                        const b_vec: F32x4 = (b_base + v * 4)[0..VEC_LEN].*;
                        acc[v] = @mulAdd(F32x4, a_bcast, b_vec, acc[v]);
                    }
                }
            }
            while (p < k) : (p += 1) {
                const a_bcast: F32x4 = @splat(a[p]);
                const b_base = b + p * ldb + j;
                inline for (0..8) |v| {
                    const b_vec: F32x4 = (b_base + v * 4)[0..VEC_LEN].*;
                    acc[v] = @mulAdd(F32x4, a_bcast, b_vec, acc[v]);
                }
            }

            inline for (0..8) |v| {
                (c + j + v * 4)[0..VEC_LEN].* = acc[v];
            }
        }

        // Scalar tail
        while (j < n) : (j += 1) {
            var sum: f32 = 0;
            for (0..k) |p| {
                sum += a[p] * b[p * ldb + j];
            }
            c[j] = sum;
        }
    }
}

// ================================================================
// Packing routines
// ================================================================

/// Pack a MC x KC block of A (starting at row ic, col pc) into MR-wide row panels.
/// Zero-pads partial panels when mc is not a multiple of MR.
fn packA(
    a: [*]const f32,
    lda: usize,
    trans_a: bool,
    ic: usize,
    pc: usize,
    mc: usize,
    kc: usize,
    dst: [*]f32,
) void {
    var idx: usize = 0;
    var ir: usize = 0;
    while (ir < mc) : (ir += MR) {
        const mr = @min(MR, mc - ir);
        for (0..kc) |p| {
            var r: usize = 0;
            while (r < mr) : (r += 1) {
                const row = ic + ir + r;
                const col = pc + p;
                dst[idx] = if (trans_a) a[col * lda + row] else a[row * lda + col];
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

/// Pack a KC x NC block of B (starting at row pc, col jc) into NR-wide column panels.
/// Zero-pads partial panels when nc is not a multiple of NR.
fn packB(
    b: [*]const f32,
    ldb: usize,
    trans_b: bool,
    pc: usize,
    jc: usize,
    kc: usize,
    nc: usize,
    dst: [*]f32,
) void {
    var idx: usize = 0;
    var jr: usize = 0;
    while (jr < nc) : (jr += NR) {
        const nr = @min(NR, nc - jr);
        for (0..kc) |p| {
            var col: usize = 0;
            while (col < nr) : (col += 1) {
                const row = pc + p;
                const c_col = jc + jr + col;
                dst[idx] = if (trans_b) b[c_col * ldb + row] else b[row * ldb + c_col];
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

/// Multiply packed A (mc x kc) and packed B (kc x nc), accumulate into C.
fn macroKernel(
    mc: usize,
    nc: usize,
    kc: usize,
    packed_a: [*]const f32,
    packed_b: [*]const f32,
    c: [*]f32,
    ldc: usize,
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
                microKernel8x8(kc, a_panel, b_panel, c + ir * ldc + jr, ldc);
            } else {
                microKernelEdge(mr, nr, kc, a_panel, b_panel, c + ir * ldc + jr, ldc);
            }
        }
    }
}

/// 8x8 FMA micro-kernel. Accumulates C[8,8] += packed_A[8,kc] * packed_B[kc,8].
/// Uses 16 F32x4 accumulators (8 rows x 2 groups of 4 columns).
inline fn microKernel8x8(
    kc: usize,
    a: [*]const f32,
    b: [*]const f32,
    c: [*]f32,
    ldc: usize,
) void {
    // Load current C into SIMD accumulators
    var c_acc: [MR][VECS_PER_NR]F32x4 = undefined;
    for (0..MR) |i| {
        for (0..VECS_PER_NR) |v| {
            c_acc[i][v] = (c + i * ldc + v * VEC_LEN)[0..VEC_LEN].*;
        }
    }

    // Main FMA loop over K dimension
    for (0..kc) |p| {
        const a_ptr = a + p * MR;

        // Load NR elements of B as VECS_PER_NR vectors
        var b_vec: [VECS_PER_NR]F32x4 = undefined;
        for (0..VECS_PER_NR) |v| {
            b_vec[v] = (b + p * NR + v * VEC_LEN)[0..VEC_LEN].*;
        }

        // For each row of the micro-tile: broadcast A element, FMA with B row
        for (0..MR) |i| {
            const a_bcast: F32x4 = @splat(a_ptr[i]);
            for (0..VECS_PER_NR) |v| {
                c_acc[i][v] = @mulAdd(F32x4, a_bcast, b_vec[v], c_acc[i][v]);
            }
        }
    }

    // Store accumulators back to C
    for (0..MR) |i| {
        for (0..VECS_PER_NR) |v| {
            (c + i * ldc + v * VEC_LEN)[0..VEC_LEN].* = c_acc[i][v];
        }
    }
}

/// Scalar fallback micro-kernel for edge tiles (mr < MR or nr < NR).
fn microKernelEdge(
    mr: usize,
    nr: usize,
    kc: usize,
    a: [*]const f32,
    b: [*]const f32,
    c: [*]f32,
    ldc: usize,
) void {
    for (0..mr) |i| {
        for (0..nr) |j| {
            var sum: f32 = c[i * ldc + j];
            for (0..kc) |p| {
                sum += a[p * MR + i] * b[p * NR + j];
            }
            c[i * ldc + j] = sum;
        }
    }
}
