//! Batch normalization with Welford single-pass mean+variance.
//! Supports training mode (compute batch stats, update running stats)
//! and inference mode (use running stats).

const std = @import("std");
const shape_mod = @import("../tensor/shape.zig");
const tensor_mod = @import("../tensor/tensor.zig");
const storage_mod = @import("../tensor/storage.zig");

const Shape = shape_mod.Shape;
const Tensor = tensor_mod.Tensor;
const Storage = storage_mod.Storage;

pub const BatchNorm = struct {
    num_features: usize,
    eps: f32,
    momentum: f32,
    gamma: []f32,
    beta: []f32,
    running_mean: []f32,
    running_var: []f32,
    allocator: std.mem.Allocator,

    /// Create a BatchNorm layer for `num_features` channels.
    /// gamma initialized to 1, beta to 0, running_mean to 0, running_var to 1.
    pub fn init(allocator: std.mem.Allocator, num_features: usize, eps: f32, momentum: f32) !BatchNorm {
        const gamma = try allocator.alloc(f32, num_features);
        for (gamma) |*g| g.* = 1.0;

        const beta = try allocator.alloc(f32, num_features);
        for (beta) |*b| b.* = 0.0;

        const running_mean = try allocator.alloc(f32, num_features);
        for (running_mean) |*m| m.* = 0.0;

        const running_var = try allocator.alloc(f32, num_features);
        for (running_var) |*v| v.* = 1.0;

        return .{
            .num_features = num_features,
            .eps = eps,
            .momentum = momentum,
            .gamma = gamma,
            .beta = beta,
            .running_mean = running_mean,
            .running_var = running_var,
            .allocator = allocator,
        };
    }

    /// Free all owned buffers.
    pub fn deinit(self: *BatchNorm) void {
        self.allocator.free(self.gamma);
        self.allocator.free(self.beta);
        self.allocator.free(self.running_mean);
        self.allocator.free(self.running_var);
    }

    /// Forward pass. Input shape: [N, C] where C == num_features.
    /// In training mode, computes batch mean/var with Welford and updates running stats.
    /// In inference mode, uses running stats.
    pub fn forward(self: *BatchNorm, allocator: std.mem.Allocator, input: Tensor(f32), training: bool) !Tensor(f32) {
        std.debug.assert(input.shape.ndim == 2);
        const batch_size = input.shape.dims[0];
        const num_channels = input.shape.dims[1];
        std.debug.assert(num_channels == self.num_features);

        const out_storage = try Storage.create(allocator, f32, input.numel());
        const out = Tensor(f32).init(out_storage, input.shape);
        out_storage.release();

        const in_data = input.storage.dataAs(f32);
        const out_data = out.storage.dataAs(f32);
        const in_stride0 = input.strides[0];
        const in_stride1 = input.strides[1];

        for (0..num_channels) |c| {
            var used_mean: f32 = undefined;
            var used_var: f32 = undefined;

            if (training) {
                // Welford single-pass mean + variance.
                const result = welfordMeanVar(in_data, input.offset, in_stride0, in_stride1, c, batch_size);
                used_mean = result.mean;
                used_var = result.variance;

                // Update running stats with exponential moving average.
                self.running_mean[c] = (1.0 - self.momentum) * self.running_mean[c] + self.momentum * used_mean;
                self.running_var[c] = (1.0 - self.momentum) * self.running_var[c] + self.momentum * used_var;
            } else {
                used_mean = self.running_mean[c];
                used_var = self.running_var[c];
            }

            const inv_std = 1.0 / @sqrt(used_var + self.eps);
            const g = self.gamma[c];
            const b = self.beta[c];

            for (0..batch_size) |n| {
                const in_idx = input.offset + n * in_stride0 + c * in_stride1;
                const out_idx = n * num_channels + c;
                out_data[out_idx] = g * (in_data[in_idx] - used_mean) * inv_std + b;
            }
        }

        return out;
    }
};

/// Welford single-pass mean + variance for a column of a 2D tensor.
pub fn welfordMeanVar(
    data: []f32,
    offset: usize,
    stride0: usize,
    stride1: usize,
    channel: usize,
    batch_size: usize,
) struct { mean: f32, variance: f32 } {
    var mean: f32 = 0;
    var m2: f32 = 0;

    for (0..batch_size) |n| {
        const x = data[offset + n * stride0 + channel * stride1];
        const count_f: f32 = @floatFromInt(n + 1);
        const delta = x - mean;
        mean += delta / count_f;
        const delta2 = x - mean;
        m2 += delta * delta2;
    }

    const batch_f: f32 = @floatFromInt(batch_size);
    return .{ .mean = mean, .variance = m2 / batch_f };
}

/// Two-pass mean + variance (for benchmark comparison).
pub fn twoPassMeanVar(
    data: []f32,
    offset: usize,
    stride0: usize,
    stride1: usize,
    channel: usize,
    batch_size: usize,
) struct { mean: f32, variance: f32 } {
    const batch_f: f32 = @floatFromInt(batch_size);

    // Pass 1: mean
    var sum: f32 = 0;
    for (0..batch_size) |n| {
        sum += data[offset + n * stride0 + channel * stride1];
    }
    const mean = sum / batch_f;

    // Pass 2: variance
    var sq_sum: f32 = 0;
    for (0..batch_size) |n| {
        const diff = data[offset + n * stride0 + channel * stride1] - mean;
        sq_sum += diff * diff;
    }

    return .{ .mean = mean, .variance = sq_sum / batch_f };
}
