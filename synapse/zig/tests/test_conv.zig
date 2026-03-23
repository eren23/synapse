//! Correctness tests for Conv2d: im2col+GEMM result vs naive 4-loop for
//! 3x3, 5x5, 1x1 kernels with various padding/stride combinations.
//! Tolerance: 1e-4 relative error.

const std = @import("std");
const testing = std.testing;
const synapse = @import("synapse");

const Tensor = synapse.tensor.core.Tensor;
const Shape = synapse.tensor.shape.Shape;
const Storage = synapse.tensor.storage.Storage;
const conv_mod = synapse.ops.conv;

// ================================================================
// Helpers
// ================================================================

/// Create a 4D NCHW tensor filled with deterministic pseudo-random values in [-1, 1].
fn makeTensor4d(
    allocator: std.mem.Allocator,
    n: usize,
    c: usize,
    h: usize,
    w: usize,
    seed: u32,
) !Tensor(f32) {
    const numel = n * c * h * w;
    const storage = try Storage.create(allocator, f32, numel);
    const data = storage.dataAs(f32);
    var s: u32 = seed;
    for (0..numel) |i| {
        s = s *% 1103515245 +% 12345;
        const bits: i32 = @bitCast(s);
        const shifted: i16 = @truncate(bits >> 16);
        data[i] = @as(f32, @floatFromInt(shifted)) / 32768.0;
    }
    const shape = Shape.init(&[_]usize{ n, c, h, w });
    const t = Tensor(f32).init(storage, shape);
    storage.release();
    return t;
}

/// Check if two floats are approximately equal within relative + absolute tolerance.
fn isClose(actual: f32, expected: f32, tol: f32) bool {
    const diff = @abs(actual - expected);
    if (diff <= tol) return true;
    const denom = @max(@abs(expected), @abs(actual));
    if (denom < 1e-10) return diff <= tol;
    return diff / denom <= tol;
}

/// Run a complete conv2d correctness check: compare im2col+GEMM vs naive.
fn checkConv(
    allocator: std.mem.Allocator,
    batch: usize,
    c_in: usize,
    c_out: usize,
    h: usize,
    w: usize,
    kh: usize,
    kw: usize,
    stride_h: usize,
    stride_w: usize,
    pad_h: usize,
    pad_w: usize,
    tol: f32,
) !void {
    const input = try makeTensor4d(allocator, batch, c_in, h, w, 42);
    defer input.release();
    const kernel = try makeTensor4d(allocator, c_out, c_in, kh, kw, 137);
    defer kernel.release();

    const naive_result = try conv_mod.conv2dNaive(allocator, input, kernel, stride_h, stride_w, pad_h, pad_w);
    defer naive_result.release();
    const gemm_result = try conv_mod.conv2d(allocator, input, kernel, stride_h, stride_w, pad_h, pad_w);
    defer gemm_result.release();

    // Verify shapes match
    try testing.expectEqual(naive_result.shape.ndim, gemm_result.shape.ndim);
    for (0..naive_result.shape.ndim) |d| {
        try testing.expectEqual(naive_result.shape.dims[d], gemm_result.shape.dims[d]);
    }

    // Compare element-wise
    const naive_data = naive_result.storage.dataAs(f32);
    const gemm_data = gemm_result.storage.dataAs(f32);
    const numel = naive_result.numel();

    for (0..numel) |i| {
        if (!isClose(gemm_data[i], naive_data[i], tol)) {
            std.debug.print(
                "MISMATCH at flat {d}: gemm={d:.8} naive={d:.8} diff={d:.8}\n",
                .{ i, gemm_data[i], naive_data[i], @abs(gemm_data[i] - naive_data[i]) },
            );
            return error.TestUnexpectedResult;
        }
    }
}

// ================================================================
// 3x3 kernel tests
// ================================================================

test "conv2d: 3x3 no padding stride 1" {
    try checkConv(testing.allocator, 1, 3, 8, 16, 16, 3, 3, 1, 1, 0, 0, 1e-4);
}

test "conv2d: 3x3 pad 1 stride 1" {
    try checkConv(testing.allocator, 1, 3, 8, 16, 16, 3, 3, 1, 1, 1, 1, 1e-4);
}

test "conv2d: 3x3 pad 1 stride 2" {
    try checkConv(testing.allocator, 1, 3, 8, 16, 16, 3, 3, 2, 2, 1, 1, 1e-4);
}

test "conv2d: 3x3 batch 2" {
    try checkConv(testing.allocator, 2, 3, 4, 8, 8, 3, 3, 1, 1, 1, 1, 1e-4);
}

test "conv2d: 3x3 large 32x32" {
    try checkConv(testing.allocator, 1, 3, 16, 32, 32, 3, 3, 1, 1, 1, 1, 1e-4);
}

// ================================================================
// 5x5 kernel tests
// ================================================================

test "conv2d: 5x5 no padding stride 1" {
    try checkConv(testing.allocator, 1, 3, 8, 16, 16, 5, 5, 1, 1, 0, 0, 1e-4);
}

test "conv2d: 5x5 pad 2 stride 1" {
    try checkConv(testing.allocator, 1, 3, 8, 16, 16, 5, 5, 1, 1, 2, 2, 1e-4);
}

test "conv2d: 5x5 pad 2 stride 2" {
    try checkConv(testing.allocator, 1, 3, 8, 16, 16, 5, 5, 2, 2, 2, 2, 1e-4);
}

test "conv2d: 5x5 asymmetric padding" {
    try checkConv(testing.allocator, 1, 3, 8, 16, 16, 5, 5, 1, 1, 1, 2, 1e-4);
}

// ================================================================
// 1x1 kernel tests (direct path)
// ================================================================

test "conv2d: 1x1 stride 1 no padding" {
    try checkConv(testing.allocator, 1, 3, 16, 16, 16, 1, 1, 1, 1, 0, 0, 1e-4);
}

test "conv2d: 1x1 stride 1 batch 2" {
    try checkConv(testing.allocator, 2, 3, 16, 8, 8, 1, 1, 1, 1, 0, 0, 1e-4);
}

test "conv2d: 1x1 many channels" {
    try checkConv(testing.allocator, 1, 16, 32, 8, 8, 1, 1, 1, 1, 0, 0, 1e-4);
}

// ================================================================
// Edge cases
// ================================================================

test "conv2d: single output pixel" {
    // Input 3x3, kernel 3x3, no pad -> 1x1 output
    try checkConv(testing.allocator, 1, 1, 1, 3, 3, 3, 3, 1, 1, 0, 0, 1e-4);
}

test "conv2d: asymmetric stride" {
    try checkConv(testing.allocator, 1, 3, 8, 16, 16, 3, 3, 1, 2, 1, 1, 1e-4);
}

test "conv2d: single channel" {
    try checkConv(testing.allocator, 1, 1, 4, 8, 8, 3, 3, 1, 1, 1, 1, 1e-4);
}
