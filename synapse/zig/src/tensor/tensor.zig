const std = @import("std");
const Storage = @import("storage.zig").Storage;
const shape_mod = @import("shape.zig");
const Shape = shape_mod.Shape;
const MAX_RANK = shape_mod.MAX_RANK;

pub const TensorError = error{
    ShapeMismatch,
    NotContiguous,
    IndexOutOfBounds,
    InvalidAxis,
};

/// Generic N-dimensional tensor wrapping a ref-counted Storage buffer.
/// Element type is determined at comptime. Supports strided access with
/// arbitrary offset for zero-copy views.
pub fn Tensor(comptime T: type) type {
    return struct {
        const Self = @This();

        storage: *Storage,
        shape: Shape,
        strides: [MAX_RANK]usize,
        offset: usize,

        /// Create a tensor wrapping a storage with the given shape.
        /// Storage ref count is incremented; caller must call `release`.
        /// Uses contiguous (row-major) strides with zero offset.
        pub fn init(storage: *Storage, sh: Shape) Self {
            return .{
                .storage = storage.retain(),
                .shape = sh,
                .strides = sh.contiguousStrides(),
                .offset = 0,
            };
        }

        /// Create a tensor with explicit strides and offset (used by view operations).
        /// Storage ref count is incremented; caller must call `release`.
        pub fn initWithStrides(storage: *Storage, sh: Shape, strides: [MAX_RANK]usize, offset: usize) Self {
            return .{
                .storage = storage.retain(),
                .shape = sh,
                .strides = strides,
                .offset = offset,
            };
        }

        /// Total number of elements.
        pub fn numel(self: Self) usize {
            return self.shape.numel();
        }

        /// Compute flat index into the storage for multi-dimensional indices.
        fn flatIndex(self: Self, indices: []const usize) usize {
            std.debug.assert(indices.len == self.shape.ndim);
            var idx = self.offset;
            for (0..self.shape.ndim) |i| {
                std.debug.assert(indices[i] < self.shape.dims[i]);
                idx += indices[i] * self.strides[i];
            }
            return idx;
        }

        /// Read element at the given multi-dimensional index.
        pub fn at(self: Self, indices: []const usize) T {
            return self.storage.dataAs(T)[self.flatIndex(indices)];
        }

        /// Write element at the given multi-dimensional index.
        pub fn set(self: Self, indices: []const usize, value: T) void {
            self.storage.dataAs(T)[self.flatIndex(indices)] = value;
        }

        /// Check if the tensor has standard contiguous (row-major) layout.
        pub fn isContiguous(self: Self) bool {
            if (self.shape.ndim == 0) return true;
            const expected = self.shape.contiguousStrides();
            return std.mem.eql(usize, self.strides[0..self.shape.ndim], expected[0..self.shape.ndim]);
        }

        /// Get the underlying storage pointer (for zero-copy verification).
        pub fn storagePtr(self: Self) *Storage {
            return self.storage;
        }

        /// Get a typed pointer to the data at the tensor's offset.
        /// Only meaningful for contiguous tensors.
        pub fn dataPtr(self: Self) [*]T {
            return self.storage.dataAs(T).ptr + self.offset;
        }

        /// Decrement the storage reference count.
        pub fn release(self: Self) void {
            self.storage.release();
        }
    };
}
