//! Numerically-stable softmax and log-softmax using the online trick.
//! Single-pass max+sum computation followed by a normalization pass.

const std = @import("std");
const shape_mod = @import("../tensor/shape.zig");
const tensor_mod = @import("../tensor/tensor.zig");
const storage_mod = @import("../tensor/storage.zig");

const Shape = shape_mod.Shape;
const MAX_RANK = shape_mod.MAX_RANK;
const Tensor = tensor_mod.Tensor;
const Storage = storage_mod.Storage;

// ============================================================
// Public API
// ============================================================

/// Numerically-stable softmax along the given axis.
/// Uses the online trick: single-pass max+sum_exp, then normalize.
pub fn softmax(allocator: std.mem.Allocator, input: Tensor(f32), axis: usize) !Tensor(f32) {
    return softmaxGeneric(allocator, input, axis, false);
}

/// Numerically-stable log-softmax along the given axis.
/// output[i] = input[i] - max - log(sum(exp(input - max)))
pub fn logSoftmax(allocator: std.mem.Allocator, input: Tensor(f32), axis: usize) !Tensor(f32) {
    return softmaxGeneric(allocator, input, axis, true);
}

// ============================================================
// Implementation
// ============================================================

fn softmaxGeneric(
    allocator: std.mem.Allocator,
    input: Tensor(f32),
    axis: usize,
    log_mode: bool,
) !Tensor(f32) {
    if (axis >= input.shape.ndim) return error.InvalidAxis;

    const out_storage = try Storage.create(allocator, f32, input.numel());
    const out = Tensor(f32).init(out_storage, input.shape);
    out_storage.release();

    const in_data = input.storage.dataAs(f32);
    const out_data = out.storage.dataAs(f32);
    const axis_size = input.shape.dims[axis];
    const out_strides = input.shape.contiguousStrides();

    // Build outer shape (all dims except axis).
    var outer_shape = Shape{};
    var oj: usize = 0;
    for (0..input.shape.ndim) |d| {
        if (d != axis) {
            outer_shape.dims[oj] = input.shape.dims[d];
            oj += 1;
        }
    }
    outer_shape.ndim = oj;
    const outer_strides = outer_shape.contiguousStrides();
    const outer_numel = if (outer_shape.ndim == 0) @as(usize, 1) else outer_shape.numel();

    const in_axis_stride = input.strides[axis];
    const out_axis_stride = out_strides[axis];

    for (0..outer_numel) |outer_idx| {
        // Decompose outer_idx into multi-dim outer indices.
        var outer_indices: [MAX_RANK]usize = [_]usize{0} ** MAX_RANK;
        var remaining = outer_idx;
        for (0..outer_shape.ndim) |d| {
            if (outer_strides[d] > 0) {
                outer_indices[d] = remaining / outer_strides[d];
                remaining %= outer_strides[d];
            }
        }

        // Map to full input/output indices (axis dim = 0).
        var full_indices: [MAX_RANK]usize = [_]usize{0} ** MAX_RANK;
        var j: usize = 0;
        for (0..input.shape.ndim) |d| {
            if (d == axis) {
                full_indices[d] = 0;
            } else {
                full_indices[d] = outer_indices[j];
                j += 1;
            }
        }

        // Compute base offsets.
        var in_base: usize = input.offset;
        var out_base: usize = 0;
        for (0..input.shape.ndim) |d| {
            in_base += full_indices[d] * input.strides[d];
            out_base += full_indices[d] * out_strides[d];
        }

        // --- Pass 1: Online max + sum_exp ---
        var max_val: f32 = -std.math.inf(f32);
        var sum_exp: f32 = 0.0;

        for (0..axis_size) |k| {
            const x = in_data[in_base + k * in_axis_stride];
            if (x > max_val) {
                // Rescale partial sum for the new maximum.
                sum_exp = sum_exp * @exp(max_val - x) + 1.0;
                max_val = x;
            } else {
                sum_exp += @exp(x - max_val);
            }
        }

        // --- Pass 2: Normalize ---
        if (log_mode) {
            const log_sum = @log(sum_exp);
            for (0..axis_size) |k| {
                const x = in_data[in_base + k * in_axis_stride];
                out_data[out_base + k * out_axis_stride] = x - max_val - log_sum;
            }
        } else {
            const inv_sum = 1.0 / sum_exp;
            for (0..axis_size) |k| {
                const x = in_data[in_base + k * in_axis_stride];
                out_data[out_base + k * out_axis_stride] = @exp(x - max_val) * inv_sum;
            }
        }
    }

    return out;
}
