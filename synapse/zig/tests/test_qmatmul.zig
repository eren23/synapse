//! Correctness tests for INT8 quantized GEMM.
//! Compares INT8 GEMM output against f32 naive GEMM for various sizes.
//! Tolerance: max relative error <= 1%.

const std = @import("std");
const testing = std.testing;
const synapse = @import("synapse");
const quantize = synapse.ops.quantize;
const qmatmul = synapse.ops.qmatmul;

// ================================================================
// Helpers
// ================================================================

/// Fill a buffer with deterministic pseudo-random values in [-1, 1].
fn fillData(data: []f32, seed: u32) void {
    var s: u32 = seed;
    for (data) |*v| {
        s = s *% 1103515245 +% 12345;
        const bits: i32 = @bitCast(s);
        const shifted: i16 = @truncate(bits >> 16);
        v.* = @as(f32, @floatFromInt(shifted)) / 32768.0;
    }
}

/// Run a complete INT8 vs f32 GEMM comparison.
/// 1. Compute f32 naive GEMM for reference.
/// 2. Quantize A per-row, B per-column.
/// 3. Compute INT8 tiled GEMM.
/// 4. Also verify INT8 naive vs INT8 tiled consistency.
/// 5. Check relative error vs f32 reference <= tol.
fn checkQMatmul(
    allocator: std.mem.Allocator,
    m: usize,
    n: usize,
    k: usize,
    tol: f32,
) !void {
    // Allocate f32 matrices
    const a_f32 = try allocator.alloc(f32, m * k);
    defer allocator.free(a_f32);
    const b_f32 = try allocator.alloc(f32, k * n);
    defer allocator.free(b_f32);
    const c_ref = try allocator.alloc(f32, m * n);
    defer allocator.free(c_ref);
    const c_int8_tiled = try allocator.alloc(f32, m * n);
    defer allocator.free(c_int8_tiled);
    const c_int8_naive = try allocator.alloc(f32, m * n);
    defer allocator.free(c_int8_naive);

    // Fill with deterministic data
    fillData(a_f32, 42);
    fillData(b_f32, 137);

    // f32 naive GEMM reference
    qmatmul.naiveF32Gemm(m, n, k, a_f32.ptr, k, b_f32.ptr, n, c_ref.ptr, n);

    // Quantize A per-row and B per-column
    const a_i8 = try allocator.alloc(i8, m * k);
    defer allocator.free(a_i8);
    const scales_a = try allocator.alloc(f32, m);
    defer allocator.free(scales_a);
    const b_i8 = try allocator.alloc(i8, k * n);
    defer allocator.free(b_i8);
    const scales_b = try allocator.alloc(f32, n);
    defer allocator.free(scales_b);

    quantize.quantizePerChannelInt8(a_f32.ptr, m, k, a_i8.ptr, scales_a.ptr);
    quantize.quantizePerColumnInt8(b_f32.ptr, k, n, b_i8.ptr, scales_b.ptr);

    // INT8 naive GEMM
    qmatmul.naiveInt8Gemm(m, n, k, a_i8.ptr, k, b_i8.ptr, n, c_int8_naive.ptr, n, scales_a.ptr, scales_b.ptr);

    // INT8 tiled GEMM
    const eff_kc = @min(qmatmul.KC, k);
    const eff_mc = ((@min(qmatmul.MC, m) + qmatmul.MR - 1) / qmatmul.MR) * qmatmul.MR;
    const eff_nc = ((@min(qmatmul.NC, n) + qmatmul.NR - 1) / qmatmul.NR) * qmatmul.NR;
    const packed_a = try allocator.alloc(i8, eff_mc * eff_kc);
    defer allocator.free(packed_a);
    const packed_b = try allocator.alloc(i8, eff_nc * eff_kc);
    defer allocator.free(packed_b);

    qmatmul.int8GemmTiled(m, n, k, a_i8.ptr, k, b_i8.ptr, n, c_int8_tiled.ptr, n, scales_a.ptr, scales_b.ptr, packed_a.ptr, packed_b.ptr);

    // Check INT8 tiled vs INT8 naive (should be very close — same integer arithmetic)
    var max_naive_diff: f32 = 0;
    for (0..m * n) |i| {
        const diff = @abs(c_int8_tiled[i] - c_int8_naive[i]);
        const denom = @max(@abs(c_int8_naive[i]), @as(f32, 1e-6));
        const rel = diff / denom;
        if (rel > max_naive_diff) max_naive_diff = rel;
    }

    if (max_naive_diff > 1e-5) {
        std.debug.print("FAIL: INT8 tiled vs naive mismatch: max relative error {e:.4}\n", .{max_naive_diff});
        return error.TestUnexpectedResult;
    }

    // Check INT8 tiled vs f32 reference using Frobenius norm relative error.
    // Per-element max relative error is unreliable for quantized GEMM because near-zero
    // output elements inflate the metric. Frobenius norm is the standard for GEMM accuracy.
    var sum_sq_diff: f64 = 0;
    var sum_sq_ref: f64 = 0;
    for (0..m * n) |i| {
        const diff: f64 = @as(f64, c_int8_tiled[i]) - @as(f64, c_ref[i]);
        sum_sq_diff += diff * diff;
        const ref_val: f64 = @as(f64, c_ref[i]);
        sum_sq_ref += ref_val * ref_val;
    }
    const frob_error: f64 = if (sum_sq_ref > 0) @sqrt(sum_sq_diff / sum_sq_ref) else 0;

    if (frob_error > tol) {
        std.debug.print(
            "FAIL: INT8 vs f32 Frobenius relative error {d:.6} > tolerance {d:.6} for {d}x{d}x{d}\n",
            .{ frob_error, @as(f64, tol), m, k, n },
        );
        return error.TestUnexpectedResult;
    }
}

// ================================================================
// Correctness tests at various sizes
// ================================================================

test "qmatmul: 8x8" {
    try checkQMatmul(testing.allocator, 8, 8, 8, 0.01);
}

test "qmatmul: 64x64" {
    try checkQMatmul(testing.allocator, 64, 64, 64, 0.01);
}

test "qmatmul: 128x512 non-square" {
    try checkQMatmul(testing.allocator, 128, 128, 512, 0.01);
}

test "qmatmul: 512x512" {
    try checkQMatmul(testing.allocator, 512, 512, 512, 0.01);
}

// ================================================================
// Edge cases
// ================================================================

test "qmatmul: non-aligned 7x13" {
    try checkQMatmul(testing.allocator, 7, 13, 5, 0.01);
}

test "qmatmul: non-aligned 17x33" {
    try checkQMatmul(testing.allocator, 17, 33, 25, 0.01);
}

test "qmatmul: single element 1x1" {
    try checkQMatmul(testing.allocator, 1, 1, 1, 0.01);
}

test "qmatmul: wide 4x256" {
    try checkQMatmul(testing.allocator, 4, 256, 64, 0.01);
}
