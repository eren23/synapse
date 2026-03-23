const std = @import("std");
const shape_mod = @import("shape.zig");
const Shape = shape_mod.Shape;
const MAX_RANK = shape_mod.MAX_RANK;
const tensor_mod = @import("tensor.zig");

pub const ViewError = error{
    ShapeMismatch,
    NotContiguous,
    IndexOutOfBounds,
    InvalidAxis,
};

/// Half-open range [start, end) for slicing a single dimension.
pub const Range = struct {
    start: usize,
    end: usize,
};

/// Zero-copy reshape. Requires the tensor to be contiguous and numel to match.
/// Returns a new view sharing the same storage (ref count incremented).
pub fn reshape(comptime T: type, t: tensor_mod.Tensor(T), new_shape: Shape) ViewError!tensor_mod.Tensor(T) {
    if (t.numel() != new_shape.numel()) return ViewError.ShapeMismatch;
    if (!t.isContiguous()) return ViewError.NotContiguous;

    return .{
        .storage = t.storage.retain(),
        .shape = new_shape,
        .strides = new_shape.contiguousStrides(),
        .offset = t.offset,
    };
}

/// Transpose two axes. Zero-copy: swaps dims and strides.
/// Returns a new view sharing the same storage (ref count incremented).
pub fn transpose(comptime T: type, t: tensor_mod.Tensor(T), dim0: usize, dim1: usize) ViewError!tensor_mod.Tensor(T) {
    if (dim0 >= t.shape.ndim or dim1 >= t.shape.ndim) return ViewError.InvalidAxis;

    var new_shape = t.shape;
    var new_strides = t.strides;

    std.mem.swap(usize, &new_shape.dims[dim0], &new_shape.dims[dim1]);
    std.mem.swap(usize, &new_strides[dim0], &new_strides[dim1]);

    return .{
        .storage = t.storage.retain(),
        .shape = new_shape,
        .strides = new_strides,
        .offset = t.offset,
    };
}

/// Slice: extract a sub-view along each dimension using half-open ranges.
/// ranges.len must equal ndim. Each range is [start, end).
/// Returns a new view sharing the same storage (ref count incremented).
pub fn slice(comptime T: type, t: tensor_mod.Tensor(T), ranges: []const Range) ViewError!tensor_mod.Tensor(T) {
    if (ranges.len != t.shape.ndim) return ViewError.ShapeMismatch;

    var new_shape = t.shape;
    var new_offset = t.offset;

    for (0..t.shape.ndim) |i| {
        if (ranges[i].start >= ranges[i].end or ranges[i].end > t.shape.dims[i]) {
            return ViewError.IndexOutOfBounds;
        }
        new_offset += ranges[i].start * t.strides[i];
        new_shape.dims[i] = ranges[i].end - ranges[i].start;
    }

    return .{
        .storage = t.storage.retain(),
        .shape = new_shape,
        .strides = t.strides,
        .offset = new_offset,
    };
}
