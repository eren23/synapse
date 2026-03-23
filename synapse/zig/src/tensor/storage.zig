const std = @import("std");

/// Dense contiguous byte buffer with 64-byte alignment and atomic reference counting.
pub const Storage = struct {
    allocator: std.mem.Allocator,
    data: []align(64) u8,
    ref_count: std.atomic.Value(u32),

    /// Allocate a new Storage holding `n` elements of type `T`, zero-initialized.
    /// Returned with ref_count = 1. Caller must call `release` when done.
    pub fn create(allocator: std.mem.Allocator, comptime T: type, n: usize) !*Storage {
        const byte_len = n * @sizeOf(T);
        const data = try allocator.alignedAlloc(u8, .@"64", byte_len);
        @memset(data, 0);

        const self = try allocator.create(Storage);
        self.* = .{
            .allocator = allocator,
            .data = data,
            .ref_count = std.atomic.Value(u32).init(1),
        };
        return self;
    }

    /// Increment the reference count. Returns `self` for chaining.
    pub fn retain(self: *Storage) *Storage {
        _ = self.ref_count.fetchAdd(1, .monotonic);
        return self;
    }

    /// Decrement the reference count. Frees the backing memory when it reaches 0.
    pub fn release(self: *Storage) void {
        if (self.ref_count.fetchSub(1, .acq_rel) == 1) {
            self.allocator.free(self.data);
            self.allocator.destroy(self);
        }
    }

    /// Reinterpret the raw byte buffer as a typed slice of `T`.
    pub fn dataAs(self: *Storage, comptime T: type) []T {
        const ptr: [*]T = @ptrCast(@alignCast(self.data.ptr));
        return ptr[0 .. self.data.len / @sizeOf(T)];
    }

    /// Number of bytes in the backing buffer.
    pub fn byteLen(self: *const Storage) usize {
        return self.data.len;
    }

    /// Current reference count (atomic load with acquire ordering).
    pub fn refCount(self: *const Storage) u32 {
        return self.ref_count.load(.acquire);
    }
};
