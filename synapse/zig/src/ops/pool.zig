//! Pooling operations: MaxPool2d (with argmax indices for backward) and AvgPool2d.
//!
//! Input layout: NCHW (batch, channels, height, width)
//! Output layout: NCHW (batch, channels, H_out, W_out)
//!
//! H_out = (H - KH) / stride_h + 1
//! W_out = (W - KW) / stride_w + 1

const std = @import("std");
const shape_mod = @import("../tensor/shape.zig");
const tensor_mod = @import("../tensor/tensor.zig");
const storage_mod = @import("../tensor/storage.zig");

const Shape = shape_mod.Shape;
const Tensor = tensor_mod.Tensor;
const Storage = storage_mod.Storage;

/// Result of MaxPool2d: output values and argmax indices for backward pass.
pub const MaxPool2dResult = struct {
    output: Tensor(f32),
    /// Flat indices into the H_in * W_in spatial plane for each output position.
    argmax: []usize,
    allocator: std.mem.Allocator,

    pub fn release(self: MaxPool2dResult) void {
        self.output.release();
        self.allocator.free(self.argmax);
    }
};

/// MaxPool2d with argmax indices for backward pass.
/// input: 4D tensor [N, C, H, W]
/// Returns output tensor [N, C, H_out, W_out] and argmax indices.
pub fn maxPool2d(
    allocator: std.mem.Allocator,
    input: Tensor(f32),
    kernel_h: usize,
    kernel_w: usize,
    stride_h: usize,
    stride_w: usize,
) !MaxPool2dResult {
    if (input.shape.ndim != 4) return error.InvalidDimensions;
    if (kernel_h == 0 or kernel_w == 0) return error.InvalidDimensions;
    if (stride_h == 0 or stride_w == 0) return error.InvalidStride;

    const batch = input.shape.dims[0];
    const channels = input.shape.dims[1];
    const h_in = input.shape.dims[2];
    const w_in = input.shape.dims[3];

    if (h_in < kernel_h or w_in < kernel_w) return error.InvalidDimensions;

    const h_out = (h_in - kernel_h) / stride_h + 1;
    const w_out = (w_in - kernel_w) / stride_w + 1;

    const out_shape = Shape.init(&[_]usize{ batch, channels, h_out, w_out });
    const out_numel = batch * channels * h_out * w_out;
    const safe_numel = if (out_numel == 0) 1 else out_numel;

    const out_storage = try Storage.create(allocator, f32, safe_numel);
    const result = Tensor(f32).init(out_storage, out_shape);
    out_storage.release();

    const argmax = try allocator.alloc(usize, safe_numel);

    if (out_numel == 0) {
        return .{ .output = result, .argmax = argmax, .allocator = allocator };
    }

    const in_data = input.dataPtr();
    const out_data = result.dataPtr();

    for (0..batch) |n| {
        for (0..channels) |c| {
            const in_plane = in_data + n * channels * h_in * w_in + c * h_in * w_in;
            const out_offset = n * channels * h_out * w_out + c * h_out * w_out;

            for (0..h_out) |oh| {
                for (0..w_out) |ow| {
                    var max_val: f32 = -std.math.inf(f32);
                    var max_idx: usize = 0;

                    for (0..kernel_h) |kh| {
                        for (0..kernel_w) |kw| {
                            const ih = oh * stride_h + kh;
                            const iw = ow * stride_w + kw;
                            const flat_idx = ih * w_in + iw;
                            const val = in_plane[flat_idx];
                            if (val > max_val) {
                                max_val = val;
                                max_idx = flat_idx;
                            }
                        }
                    }

                    const out_idx = out_offset + oh * w_out + ow;
                    out_data[out_idx] = max_val;
                    argmax[out_idx] = max_idx;
                }
            }
        }
    }

    return .{ .output = result, .argmax = argmax, .allocator = allocator };
}

/// AvgPool2d forward.
/// input: 4D tensor [N, C, H, W]
/// Returns output tensor [N, C, H_out, W_out].
pub fn avgPool2d(
    allocator: std.mem.Allocator,
    input: Tensor(f32),
    kernel_h: usize,
    kernel_w: usize,
    stride_h: usize,
    stride_w: usize,
) !Tensor(f32) {
    if (input.shape.ndim != 4) return error.InvalidDimensions;
    if (kernel_h == 0 or kernel_w == 0) return error.InvalidDimensions;
    if (stride_h == 0 or stride_w == 0) return error.InvalidStride;

    const batch = input.shape.dims[0];
    const channels = input.shape.dims[1];
    const h_in = input.shape.dims[2];
    const w_in = input.shape.dims[3];

    if (h_in < kernel_h or w_in < kernel_w) return error.InvalidDimensions;

    const h_out = (h_in - kernel_h) / stride_h + 1;
    const w_out = (w_in - kernel_w) / stride_w + 1;

    const out_shape = Shape.init(&[_]usize{ batch, channels, h_out, w_out });
    const out_numel = batch * channels * h_out * w_out;

    const out_storage = try Storage.create(allocator, f32, if (out_numel == 0) 1 else out_numel);
    const result = Tensor(f32).init(out_storage, out_shape);
    out_storage.release();

    if (out_numel == 0) return result;

    const in_data = input.dataPtr();
    const out_data = result.dataPtr();
    const inv_pool_size: f32 = 1.0 / @as(f32, @floatFromInt(kernel_h * kernel_w));

    for (0..batch) |n| {
        for (0..channels) |c| {
            const in_plane = in_data + n * channels * h_in * w_in + c * h_in * w_in;
            const out_offset = n * channels * h_out * w_out + c * h_out * w_out;

            for (0..h_out) |oh| {
                for (0..w_out) |ow| {
                    var sum: f32 = 0;

                    for (0..kernel_h) |kh| {
                        for (0..kernel_w) |kw| {
                            const ih = oh * stride_h + kh;
                            const iw = ow * stride_w + kw;
                            sum += in_plane[ih * w_in + iw];
                        }
                    }

                    out_data[out_offset + oh * w_out + ow] = sum * inv_pool_size;
                }
            }
        }
    }

    return result;
}
