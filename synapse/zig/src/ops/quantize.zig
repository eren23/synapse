//! Per-channel INT8 quantization and dequantization.
//!
//! Symmetric quantization: scale = max(|x|) / 127, q = round(x / scale).
//! Quantized values are clamped to [-127, 127] (symmetric range).
//! All-zeros channels get scale = 1.0 to avoid division by zero.

const std = @import("std");

/// Quantize a 2D row-major matrix [channels × channel_size] to per-channel (per-row) INT8.
/// scale[c] = max(|data[c, :]|) / 127
/// out[c, j] = clamp(round(data[c, j] / scale[c]), -127, 127)
pub fn quantizePerChannelInt8(
    data: [*]const f32,
    channels: usize,
    channel_size: usize,
    out: [*]i8,
    scales: [*]f32,
) void {
    for (0..channels) |c| {
        const row = data + c * channel_size;
        const out_row = out + c * channel_size;

        // Find max absolute value in this channel
        var max_abs: f32 = 0;
        for (0..channel_size) |j| {
            const abs_val = @abs(row[j]);
            if (abs_val > max_abs) max_abs = abs_val;
        }

        // Handle all-zeros channel
        if (max_abs == 0) {
            scales[c] = 1.0;
            @memset(out_row[0..channel_size], @as(i8, 0));
            continue;
        }

        const scale = max_abs / 127.0;
        scales[c] = scale;
        const inv_scale = 127.0 / max_abs;

        for (0..channel_size) |j| {
            const scaled = row[j] * inv_scale;
            const rounded = @max(@as(f32, -127.0), @min(@as(f32, 127.0), @round(scaled)));
            out_row[j] = @intFromFloat(rounded);
        }
    }
}

/// Quantize a 2D row-major matrix [rows × cols] with per-column scales.
/// scale[j] = max(|data[:, j]|) / 127
/// out[i, j] = clamp(round(data[i, j] / scale[j]), -127, 127)
pub fn quantizePerColumnInt8(
    data: [*]const f32,
    rows: usize,
    cols: usize,
    out: [*]i8,
    scales: [*]f32,
) void {
    // First pass: find max abs per column (row-major traversal for cache friendliness)
    @memset(scales[0..cols], @as(f32, 0));
    for (0..rows) |i| {
        for (0..cols) |j| {
            const abs_val = @abs(data[i * cols + j]);
            if (abs_val > scales[j]) scales[j] = abs_val;
        }
    }

    // Compute final scales
    for (0..cols) |j| {
        if (scales[j] == 0) {
            scales[j] = 1.0;
        } else {
            scales[j] = scales[j] / 127.0;
        }
    }

    // Second pass: quantize (row-major traversal)
    for (0..rows) |i| {
        for (0..cols) |j| {
            const inv_scale = 127.0 / (scales[j] * 127.0);
            const scaled = data[i * cols + j] * inv_scale;
            const rounded = @max(@as(f32, -127.0), @min(@as(f32, 127.0), @round(scaled)));
            out[i * cols + j] = @intFromFloat(rounded);
        }
    }
}

/// Dequantize per-channel (per-row) INT8 data back to f32.
/// out[c, j] = data[c, j] * scale[c]
pub fn dequantizePerChannelInt8(
    data: [*]const i8,
    channels: usize,
    channel_size: usize,
    out: [*]f32,
    scales: [*]const f32,
) void {
    for (0..channels) |c| {
        const scale = scales[c];
        const in_row = data + c * channel_size;
        const out_row = out + c * channel_size;

        for (0..channel_size) |j| {
            out_row[j] = @as(f32, @floatFromInt(in_row[j])) * scale;
        }
    }
}
