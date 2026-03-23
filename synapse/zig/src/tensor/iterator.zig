const std = @import("std");
const shape_mod = @import("shape.zig");
const MAX_RANK = shape_mod.MAX_RANK;
const tensor_mod = @import("tensor.zig");

/// Strided multi-dimensional iterator over a Tensor.
/// Uses a fast pointer-walk for contiguous tensors and falls back
/// to per-element index computation for non-contiguous (transposed/sliced) views.
pub fn TensorIterator(comptime T: type) type {
    return struct {
        const Self = @This();

        data: []T,
        strides: [MAX_RANK]usize,
        shape_dims: [MAX_RANK]usize,
        ndim: usize,
        offset: usize,
        indices: [MAX_RANK]usize,
        flat_pos: usize,
        total: usize,
        count: usize,
        is_contiguous: bool,

        pub fn init(t: tensor_mod.Tensor(T)) Self {
            return .{
                .data = t.storage.dataAs(T),
                .strides = t.strides,
                .shape_dims = t.shape.dims,
                .ndim = t.shape.ndim,
                .offset = t.offset,
                .indices = [_]usize{0} ** MAX_RANK,
                .flat_pos = 0,
                .total = t.numel(),
                .count = 0,
                .is_contiguous = t.isContiguous(),
            };
        }

        /// Return the next element, or null when exhausted.
        pub fn next(self: *Self) ?T {
            if (self.count >= self.total) return null;
            self.count += 1;

            if (self.is_contiguous) {
                const idx = self.offset + self.flat_pos;
                self.flat_pos += 1;
                return self.data[idx];
            } else {
                var idx: usize = self.offset;
                for (0..self.ndim) |i| {
                    idx += self.indices[i] * self.strides[i];
                }
                self.advanceIndices();
                return self.data[idx];
            }
        }

        /// Advance the multi-dimensional index (odometer style, last dim first).
        fn advanceIndices(self: *Self) void {
            var i = self.ndim;
            while (i > 0) {
                i -= 1;
                self.indices[i] += 1;
                if (self.indices[i] < self.shape_dims[i]) return;
                self.indices[i] = 0;
            }
        }

        /// Reset the iterator to the beginning.
        pub fn reset(self: *Self) void {
            self.indices = [_]usize{0} ** MAX_RANK;
            self.flat_pos = 0;
            self.count = 0;
        }
    };
}
