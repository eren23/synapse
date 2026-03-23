//! Tensor reduction operations: sum, mean, max, min, argmax along arbitrary axes.
//! SIMD-accelerated inner loops for contiguous innermost-axis reductions.

const std = @import("std");
const shape_mod = @import("../tensor/shape.zig");
const tensor_mod = @import("../tensor/tensor.zig");
const storage_mod = @import("../tensor/storage.zig");

const Shape = shape_mod.Shape;
const MAX_RANK = shape_mod.MAX_RANK;
const Tensor = tensor_mod.Tensor;
const Storage = storage_mod.Storage;

const VEC_LEN = 4;
const F32x4 = @Vector(VEC_LEN, f32);

const ReduceOp = enum { sum, mean, max, min, argmax };

// ============================================================
// Public API
// ============================================================

/// Reduce sum along the given axis.
pub fn reduceSum(allocator: std.mem.Allocator, t: Tensor(f32), axis: usize, keepdim: bool) !Tensor(f32) {
    return reduceGeneric(allocator, t, axis, keepdim, .sum);
}

/// Reduce mean along the given axis.
pub fn reduceMean(allocator: std.mem.Allocator, t: Tensor(f32), axis: usize, keepdim: bool) !Tensor(f32) {
    return reduceGeneric(allocator, t, axis, keepdim, .mean);
}

/// Reduce max along the given axis.
pub fn reduceMax(allocator: std.mem.Allocator, t: Tensor(f32), axis: usize, keepdim: bool) !Tensor(f32) {
    return reduceGeneric(allocator, t, axis, keepdim, .max);
}

/// Reduce min along the given axis.
pub fn reduceMin(allocator: std.mem.Allocator, t: Tensor(f32), axis: usize, keepdim: bool) !Tensor(f32) {
    return reduceGeneric(allocator, t, axis, keepdim, .min);
}

/// Argmax along the given axis. Returns indices as f32.
pub fn argmax(allocator: std.mem.Allocator, t: Tensor(f32), axis: usize, keepdim: bool) !Tensor(f32) {
    return reduceGeneric(allocator, t, axis, keepdim, .argmax);
}

// ============================================================
// SIMD primitives (exported for benchmarking)
// ============================================================

/// SIMD-accelerated horizontal sum using 4-wide vectors.
pub fn simdSum(slice: []const f32) f32 {
    const len = slice.len;
    var acc: F32x4 = @splat(0.0);
    var i: usize = 0;

    while (i + VEC_LEN <= len) : (i += VEC_LEN) {
        const v: F32x4 = slice[i..][0..VEC_LEN].*;
        acc += v;
    }

    var sum: f32 = @reduce(.Add, acc);
    while (i < len) : (i += 1) {
        sum += slice[i];
    }
    return sum;
}

/// Scalar horizontal sum (baseline for benchmarking).
pub fn scalarSum(ptr: [*]const f32, len: usize) f32 {
    var sum: f32 = 0;
    var i: usize = 0;
    while (i < len) : (i += 1) {
        sum += ptr[i];
    }
    return sum;
}

fn simdMax(slice: []const f32) f32 {
    const len = slice.len;
    if (len == 0) return -std.math.inf(f32);

    var acc: F32x4 = @splat(-std.math.inf(f32));
    var i: usize = 0;

    while (i + VEC_LEN <= len) : (i += VEC_LEN) {
        const v: F32x4 = slice[i..][0..VEC_LEN].*;
        acc = @max(acc, v);
    }

    var max_val: f32 = @reduce(.Max, acc);
    while (i < len) : (i += 1) {
        max_val = @max(max_val, slice[i]);
    }
    return max_val;
}

fn simdMin(slice: []const f32) f32 {
    const len = slice.len;
    if (len == 0) return std.math.inf(f32);

    var acc: F32x4 = @splat(std.math.inf(f32));
    var i: usize = 0;

    while (i + VEC_LEN <= len) : (i += VEC_LEN) {
        const v: F32x4 = slice[i..][0..VEC_LEN].*;
        acc = @min(acc, v);
    }

    var min_val: f32 = @reduce(.Min, acc);
    while (i < len) : (i += 1) {
        min_val = @min(min_val, slice[i]);
    }
    return min_val;
}

// ============================================================
// Core reduction engine
// ============================================================

fn computeOutputShape(input_shape: Shape, axis: usize, keepdim: bool) Shape {
    if (keepdim) {
        var out_shape = input_shape;
        out_shape.dims[axis] = 1;
        return out_shape;
    } else {
        var out_shape = Shape{};
        var j: usize = 0;
        for (0..input_shape.ndim) |i| {
            if (i != axis) {
                out_shape.dims[j] = input_shape.dims[i];
                j += 1;
            }
        }
        out_shape.ndim = j;
        return out_shape;
    }
}

fn reduceGeneric(
    allocator: std.mem.Allocator,
    t: Tensor(f32),
    axis: usize,
    keepdim: bool,
    op: ReduceOp,
) !Tensor(f32) {
    if (axis >= t.shape.ndim) return error.InvalidAxis;

    const out_shape = computeOutputShape(t.shape, axis, keepdim);
    const out_numel = if (out_shape.ndim == 0) @as(usize, 1) else out_shape.numel();
    const out_storage = try Storage.create(allocator, f32, out_numel);
    const out = Tensor(f32).init(out_storage, out_shape);
    out_storage.release();

    const data = t.storage.dataAs(f32);
    const out_data = out.storage.dataAs(f32);

    const reduce_size = t.shape.dims[axis];
    const reduce_stride = t.strides[axis];

    // Fast SIMD path: contiguous tensor, reducing along the last axis (stride=1).
    const use_simd = t.isContiguous() and
        axis == t.shape.ndim - 1 and
        reduce_stride == 1 and
        op != .argmax;

    if (use_simd) {
        const base = t.offset;
        for (0..out_numel) |out_idx| {
            const src_offset = base + out_idx * reduce_size;
            const slice = data[src_offset .. src_offset + reduce_size];
            out_data[out_idx] = switch (op) {
                .sum => simdSum(slice),
                .mean => simdSum(slice) / @as(f32, @floatFromInt(reduce_size)),
                .max => simdMax(slice),
                .min => simdMin(slice),
                .argmax => unreachable,
            };
        }
    } else {
        // General strided path.
        // Build outer shape (all dims except the reduction axis).
        var outer_shape = Shape{};
        var oj: usize = 0;
        for (0..t.shape.ndim) |d| {
            if (d != axis) {
                outer_shape.dims[oj] = t.shape.dims[d];
                oj += 1;
            }
        }
        outer_shape.ndim = oj;
        const outer_strides = outer_shape.contiguousStrides();

        for (0..out_numel) |out_idx| {
            // Decompose flat output index into outer indices.
            var outer_indices: [MAX_RANK]usize = [_]usize{0} ** MAX_RANK;
            var remaining = out_idx;
            for (0..outer_shape.ndim) |d| {
                if (outer_strides[d] > 0) {
                    outer_indices[d] = remaining / outer_strides[d];
                    remaining %= outer_strides[d];
                }
            }

            // Map outer indices to input multi-dim indices (axis dim = 0).
            var in_indices: [MAX_RANK]usize = [_]usize{0} ** MAX_RANK;
            var j: usize = 0;
            for (0..t.shape.ndim) |d| {
                if (d == axis) {
                    in_indices[d] = 0;
                } else {
                    in_indices[d] = outer_indices[j];
                    j += 1;
                }
            }

            // Compute input base offset.
            var base_offset: usize = t.offset;
            for (0..t.shape.ndim) |d| {
                base_offset += in_indices[d] * t.strides[d];
            }

            // Reduce along the axis.
            var acc: f32 = switch (op) {
                .sum, .mean => 0.0,
                .max, .argmax => -std.math.inf(f32),
                .min => std.math.inf(f32),
            };
            var best_idx: f32 = 0.0;

            for (0..reduce_size) |k| {
                const val = data[base_offset + k * reduce_stride];
                switch (op) {
                    .sum, .mean => acc += val,
                    .max => acc = @max(acc, val),
                    .min => acc = @min(acc, val),
                    .argmax => {
                        if (val > acc) {
                            acc = val;
                            best_idx = @floatFromInt(k);
                        }
                    },
                }
            }

            out_data[out_idx] = switch (op) {
                .mean => acc / @as(f32, @floatFromInt(reduce_size)),
                .argmax => best_idx,
                else => acc,
            };
        }
    }

    return out;
}
