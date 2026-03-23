//! Tests for per-channel INT8 quantization and dequantization.
//! Criteria:
//! - Round-trip error <= 0.5% per channel
//! - Symmetric quantization: q(-x) == -q(x)
//! - All-zeros channel handled gracefully

const std = @import("std");
const testing = std.testing;
const synapse = @import("synapse");
const quantize = synapse.ops.quantize;

// ================================================================
// Helpers
// ================================================================

/// Fill a buffer with deterministic pseudo-random values in [-range, range].
fn fillRandom(data: []f32, seed: u32, range: f32) void {
    var s: u32 = seed;
    for (data) |*v| {
        s = s *% 1103515245 +% 12345;
        const bits: i32 = @bitCast(s);
        const shifted: i16 = @truncate(bits >> 16);
        v.* = @as(f32, @floatFromInt(shifted)) / 32768.0 * range;
    }
}

/// Compute per-channel relative error: max(|orig - reconstructed|) / max(|orig|) per channel.
fn perChannelRelativeError(
    original: []const f32,
    reconstructed: []const f32,
    channels: usize,
    channel_size: usize,
) f32 {
    var worst: f32 = 0;
    for (0..channels) |c| {
        var max_abs: f32 = 0;
        var max_diff: f32 = 0;
        for (0..channel_size) |j| {
            const idx = c * channel_size + j;
            const abs_val = @abs(original[idx]);
            if (abs_val > max_abs) max_abs = abs_val;
            const diff = @abs(original[idx] - reconstructed[idx]);
            if (diff > max_diff) max_diff = diff;
        }
        if (max_abs > 0) {
            const rel = max_diff / max_abs;
            if (rel > worst) worst = rel;
        }
    }
    return worst;
}

// ================================================================
// Round-trip tests
// ================================================================

test "quantize: per-channel round-trip small" {
    const channels = 4;
    const ch_size = 16;
    const n = channels * ch_size;

    var data: [n]f32 = undefined;
    fillRandom(&data, 42, 10.0);

    var q_data: [n]i8 = undefined;
    var scales: [channels]f32 = undefined;
    var reconstructed: [n]f32 = undefined;

    quantize.quantizePerChannelInt8(&data, channels, ch_size, &q_data, &scales);
    quantize.dequantizePerChannelInt8(&q_data, channels, ch_size, &reconstructed, &scales);

    const err = perChannelRelativeError(&data, &reconstructed, channels, ch_size);
    if (err > 0.005) {
        std.debug.print("FAIL: round-trip error {d:.6} > 0.5%\n", .{err});
        return error.TestUnexpectedResult;
    }
}

test "quantize: per-channel round-trip large" {
    const allocator = testing.allocator;
    const channels = 64;
    const ch_size = 512;
    const n = channels * ch_size;

    const data = try allocator.alloc(f32, n);
    defer allocator.free(data);
    fillRandom(data, 137, 100.0);

    const q_data = try allocator.alloc(i8, n);
    defer allocator.free(q_data);
    const scales = try allocator.alloc(f32, channels);
    defer allocator.free(scales);
    const reconstructed = try allocator.alloc(f32, n);
    defer allocator.free(reconstructed);

    quantize.quantizePerChannelInt8(data.ptr, channels, ch_size, q_data.ptr, scales.ptr);
    quantize.dequantizePerChannelInt8(q_data.ptr, channels, ch_size, reconstructed.ptr, scales.ptr);

    const err = perChannelRelativeError(data, reconstructed, channels, ch_size);
    if (err > 0.005) {
        std.debug.print("FAIL: round-trip error {d:.6} > 0.5%\n", .{err});
        return error.TestUnexpectedResult;
    }
}

// ================================================================
// Symmetric quantization test
// ================================================================

test "quantize: symmetric quantization" {
    // For symmetric quantization, q(-x) should equal -q(x) for most values
    const ch_size = 32;
    var data: [ch_size]f32 = undefined;
    fillRandom(&data, 99, 5.0);

    // Quantize positive data
    var q_pos: [ch_size]i8 = undefined;
    var scale_pos: [1]f32 = undefined;
    quantize.quantizePerChannelInt8(&data, 1, ch_size, &q_pos, &scale_pos);

    // Negate data and quantize
    var neg_data: [ch_size]f32 = undefined;
    for (0..ch_size) |j| {
        neg_data[j] = -data[j];
    }
    var q_neg: [ch_size]i8 = undefined;
    var scale_neg: [1]f32 = undefined;
    quantize.quantizePerChannelInt8(&neg_data, 1, ch_size, &q_neg, &scale_neg);

    // Scales should be equal (both depend on max abs)
    try testing.expectApproxEqAbs(scale_pos[0], scale_neg[0], 1e-7);

    // q(-x) should equal -q(x) for each element
    for (0..ch_size) |j| {
        try testing.expectEqual(-q_pos[j], q_neg[j]);
    }
}

// ================================================================
// All-zeros channel test
// ================================================================

test "quantize: all-zeros channel" {
    const ch_size = 16;
    // Two channels: first is normal, second is all zeros
    var data: [2 * ch_size]f32 = undefined;
    fillRandom(data[0..ch_size], 42, 3.0);
    @memset(data[ch_size..], @as(f32, 0));

    var q_data: [2 * ch_size]i8 = undefined;
    var scales: [2]f32 = undefined;

    quantize.quantizePerChannelInt8(&data, 2, ch_size, &q_data, &scales);

    // First channel should have a proper scale
    try testing.expect(scales[0] > 0);

    // Second channel: scale should be 1.0, all outputs should be 0
    try testing.expectEqual(@as(f32, 1.0), scales[1]);
    for (ch_size..2 * ch_size) |j| {
        try testing.expectEqual(@as(i8, 0), q_data[j]);
    }

    // Dequantize should give back zeros for the zero channel
    var reconstructed: [2 * ch_size]f32 = undefined;
    quantize.dequantizePerChannelInt8(&q_data, 2, ch_size, &reconstructed, &scales);
    for (ch_size..2 * ch_size) |j| {
        try testing.expectEqual(@as(f32, 0), reconstructed[j]);
    }
}

// ================================================================
// Per-column quantization test
// ================================================================

test "quantize: per-column round-trip" {
    const rows = 16;
    const cols = 8;
    const n = rows * cols;

    var data: [n]f32 = undefined;
    fillRandom(&data, 77, 8.0);

    var q_data: [n]i8 = undefined;
    var scales: [cols]f32 = undefined;

    quantize.quantizePerColumnInt8(&data, rows, cols, &q_data, &scales);

    // Dequantize manually per-column to verify
    var reconstructed: [n]f32 = undefined;
    for (0..rows) |i| {
        for (0..cols) |j| {
            reconstructed[i * cols + j] = @as(f32, @floatFromInt(q_data[i * cols + j])) * scales[j];
        }
    }

    // Check per-column error
    var worst: f32 = 0;
    for (0..cols) |j| {
        var max_abs: f32 = 0;
        var max_diff: f32 = 0;
        for (0..rows) |i| {
            const idx = i * cols + j;
            const abs_val = @abs(data[idx]);
            if (abs_val > max_abs) max_abs = abs_val;
            const diff = @abs(data[idx] - reconstructed[idx]);
            if (diff > max_diff) max_diff = diff;
        }
        if (max_abs > 0) {
            const rel = max_diff / max_abs;
            if (rel > worst) worst = rel;
        }
    }

    if (worst > 0.005) {
        std.debug.print("FAIL: per-column round-trip error {d:.6} > 0.5%\n", .{worst});
        return error.TestUnexpectedResult;
    }
}

// ================================================================
// Value range test
// ================================================================

test "quantize: values clamped to [-127, 127]" {
    const ch_size = 8;
    var data = [_]f32{ 100.0, -100.0, 50.0, -50.0, 0.0, 127.0, -127.0, 1.0 };
    var q_data: [ch_size]i8 = undefined;
    var scales: [1]f32 = undefined;

    quantize.quantizePerChannelInt8(&data, 1, ch_size, &q_data, &scales);

    // Max abs is 127.0, scale = 127.0/127.0 = 1.0
    try testing.expectApproxEqAbs(@as(f32, 1.0), scales[0], 1e-7);

    // Check all values are in [-127, 127]
    for (&q_data) |v| {
        try testing.expect(v >= -127 and v <= 127);
    }

    // The extreme values (100, -100) should quantize to (100, -100) with scale=1
    try testing.expectEqual(@as(i8, 100), q_data[0]);
    try testing.expectEqual(@as(i8, -100), q_data[1]);
    try testing.expectEqual(@as(i8, 127), q_data[5]);
    try testing.expectEqual(@as(i8, -127), q_data[6]);
}
