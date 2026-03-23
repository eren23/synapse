const std = @import("std");

pub const MAX_RANK: usize = 8;

pub const ShapeError = error{
    IncompatibleShapes,
    RankTooHigh,
};

pub const Shape = struct {
    dims: [MAX_RANK]usize = [_]usize{0} ** MAX_RANK,
    ndim: usize = 0,

    /// Create a Shape from a slice of dimension sizes.
    pub fn init(dims: []const usize) Shape {
        std.debug.assert(dims.len <= MAX_RANK);
        var shape = Shape{};
        shape.ndim = dims.len;
        @memcpy(shape.dims[0..dims.len], dims);
        return shape;
    }

    /// Total number of elements (product of all dimensions). Scalars (ndim=0) return 1.
    pub fn numel(self: Shape) usize {
        if (self.ndim == 0) return 1;
        var n: usize = 1;
        for (self.dims[0..self.ndim]) |d| {
            n *= d;
        }
        return n;
    }

    /// Compute row-major (C-order) contiguous strides for this shape.
    /// Returns an array where strides[i] = product of dims[i+1..ndim].
    pub fn contiguousStrides(self: Shape) [MAX_RANK]usize {
        var strides = [_]usize{0} ** MAX_RANK;
        if (self.ndim == 0) return strides;
        strides[self.ndim - 1] = 1;
        var i: usize = self.ndim - 1;
        while (i > 0) {
            i -= 1;
            strides[i] = strides[i + 1] * self.dims[i + 1];
        }
        return strides;
    }

    /// Element-wise equality check.
    pub fn eql(self: Shape, other: Shape) bool {
        if (self.ndim != other.ndim) return false;
        return std.mem.eql(usize, self.dims[0..self.ndim], other.dims[0..other.ndim]);
    }
};

/// Compute the broadcast-compatible output shape following NumPy semantics.
/// Shapes are right-aligned; dimensions must be equal or one of them must be 1.
pub fn broadcastShapes(a: Shape, b: Shape) ShapeError!Shape {
    const max_ndim = @max(a.ndim, b.ndim);
    if (max_ndim > MAX_RANK) return ShapeError.RankTooHigh;

    var result = Shape{ .ndim = max_ndim };

    var i: usize = 0;
    while (i < max_ndim) : (i += 1) {
        const a_dim = if (i < a.ndim) a.dims[a.ndim - 1 - i] else 1;
        const b_dim = if (i < b.ndim) b.dims[b.ndim - 1 - i] else 1;

        if (a_dim == b_dim) {
            result.dims[max_ndim - 1 - i] = a_dim;
        } else if (a_dim == 1) {
            result.dims[max_ndim - 1 - i] = b_dim;
        } else if (b_dim == 1) {
            result.dims[max_ndim - 1 - i] = a_dim;
        } else {
            return ShapeError.IncompatibleShapes;
        }
    }

    return result;
}

/// Check whether two shapes are broadcast-compatible.
pub fn isCompatible(a: Shape, b: Shape) bool {
    _ = broadcastShapes(a, b) catch return false;
    return true;
}
