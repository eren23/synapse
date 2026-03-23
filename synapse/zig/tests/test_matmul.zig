//! Correctness tests for SGEMM: tiled result vs naive triple-loop for various sizes.
//! Tolerance: 1e-4 relative error for large sizes, 1e-5 for small sizes.

const std = @import("std");
const testing = std.testing;
const synapse = @import("synapse");

const Tensor = synapse.tensor.core.Tensor;
const Shape = synapse.tensor.shape.Shape;
const Storage = synapse.tensor.storage.Storage;
const matmul_mod = synapse.ops.matmul;

// ================================================================
// Helpers
// ================================================================

/// Create a 2D tensor filled with deterministic pseudo-random values in [-1, 1].
fn makeTensor(allocator: std.mem.Allocator, rows: usize, cols: usize, seed: u32) !Tensor(f32) {
    const n = rows * cols;
    const storage = try Storage.create(allocator, f32, n);
    const data = storage.dataAs(f32);
    var s: u32 = seed;
    for (0..n) |i| {
        s = s *% 1103515245 +% 12345;
        // Map upper bits to [-1, 1)
        const bits: i32 = @bitCast(s);
        const shifted: i16 = @truncate(bits >> 16);
        data[i] = @as(f32, @floatFromInt(shifted)) / 32768.0;
    }
    const shape = Shape.init(&[_]usize{ rows, cols });
    const t = Tensor(f32).init(storage, shape);
    storage.release();
    return t;
}

/// Check if two floats are approximately equal within relative + absolute tolerance.
fn isClose(actual: f32, expected: f32, tol: f32) bool {
    const diff = @abs(actual - expected);
    if (diff <= tol) return true; // absolute tolerance
    const denom = @max(@abs(expected), @abs(actual));
    if (denom < 1e-10) return diff <= tol;
    return diff / denom <= tol;
}

/// Run a complete matmul correctness check: compare tiled vs naive.
fn checkMatmul(
    allocator: std.mem.Allocator,
    m: usize,
    n: usize,
    k: usize,
    trans_a: bool,
    trans_b: bool,
    tol: f32,
) !void {
    const a_rows = if (trans_a) k else m;
    const a_cols = if (trans_a) m else k;
    const b_rows = if (trans_b) n else k;
    const b_cols = if (trans_b) k else n;

    const a = try makeTensor(allocator, a_rows, a_cols, 42);
    defer a.release();
    const b = try makeTensor(allocator, b_rows, b_cols, 137);
    defer b.release();

    const naive_result = try matmul_mod.naiveMatmul(allocator, a, b, trans_a, trans_b);
    defer naive_result.release();
    const tiled_result = try matmul_mod.matmul(allocator, a, b, trans_a, trans_b);
    defer tiled_result.release();

    // Verify shapes match
    try testing.expectEqual(naive_result.shape.ndim, tiled_result.shape.ndim);
    try testing.expectEqual(naive_result.shape.dims[0], m);
    try testing.expectEqual(naive_result.shape.dims[1], n);
    try testing.expectEqual(tiled_result.shape.dims[0], m);
    try testing.expectEqual(tiled_result.shape.dims[1], n);

    // Compare element-wise
    const naive_data = naive_result.storage.dataAs(f32);
    const tiled_data = tiled_result.storage.dataAs(f32);
    for (0..m * n) |i| {
        if (!isClose(tiled_data[i], naive_data[i], tol)) {
            std.debug.print(
                "MISMATCH at [{d},{d}] (flat {d}): tiled={d:.8} naive={d:.8} diff={d:.8}\n",
                .{
                    i / n,
                    i % n,
                    i,
                    tiled_data[i],
                    naive_data[i],
                    @abs(tiled_data[i] - naive_data[i]),
                },
            );
            return error.TestUnexpectedResult;
        }
    }
}

// ================================================================
// Square size tests
// ================================================================

test "matmul: 1x1" {
    try checkMatmul(testing.allocator, 1, 1, 1, false, false, 1e-5);
}

test "matmul: 8x8" {
    try checkMatmul(testing.allocator, 8, 8, 8, false, false, 1e-5);
}

test "matmul: 16x16" {
    try checkMatmul(testing.allocator, 16, 16, 16, false, false, 1e-5);
}

test "matmul: 64x64" {
    try checkMatmul(testing.allocator, 64, 64, 64, false, false, 1e-4);
}

test "matmul: 128x128" {
    try checkMatmul(testing.allocator, 128, 128, 128, false, false, 1e-4);
}

test "matmul: 512x512" {
    try checkMatmul(testing.allocator, 512, 512, 512, false, false, 1e-4);
}

test "matmul: 1024x1024" {
    try checkMatmul(testing.allocator, 1024, 1024, 1024, false, false, 1e-4);
}

// ================================================================
// Non-square test
// ================================================================

test "matmul: non-square 32x64 * 64x48" {
    try checkMatmul(testing.allocator, 32, 48, 64, false, false, 1e-4);
}

// ================================================================
// Non-aligned edge cases (exercises edge micro-kernel)
// ================================================================

test "matmul: non-aligned 7x13 * 13x5" {
    try checkMatmul(testing.allocator, 7, 5, 13, false, false, 1e-5);
}

test "matmul: non-aligned 17x33" {
    try checkMatmul(testing.allocator, 17, 33, 25, false, false, 1e-4);
}

// ================================================================
// Transposed variants
// ================================================================

test "matmul: trans_a 8x8" {
    try checkMatmul(testing.allocator, 8, 8, 8, true, false, 1e-5);
}

test "matmul: trans_b 8x8" {
    try checkMatmul(testing.allocator, 8, 8, 8, false, true, 1e-5);
}

test "matmul: trans_a and trans_b 8x8" {
    try checkMatmul(testing.allocator, 8, 8, 8, true, true, 1e-5);
}

test "matmul: trans_a 64x64" {
    try checkMatmul(testing.allocator, 64, 64, 64, true, false, 1e-4);
}

test "matmul: trans_b 64x64" {
    try checkMatmul(testing.allocator, 64, 64, 64, false, true, 1e-4);
}

test "matmul: trans_a and trans_b 64x64" {
    try checkMatmul(testing.allocator, 64, 64, 64, true, true, 1e-4);
}

test "matmul: trans_a non-square 48x32" {
    // A is K x M = 64x32 (trans), B is K x N = 64x48, result is 32x48
    try checkMatmul(testing.allocator, 32, 48, 64, true, false, 1e-4);
}

test "matmul: trans_b non-square 32x48" {
    // A is M x K = 32x64, B is N x K = 48x64 (trans), result is 32x48
    try checkMatmul(testing.allocator, 32, 48, 64, false, true, 1e-4);
}
