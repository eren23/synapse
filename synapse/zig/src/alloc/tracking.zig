const std = @import("std");
const Allocator = std.mem.Allocator;
const Alignment = std.mem.Alignment;

/// Debug allocator wrapper that counts allocations/frees and detects leaks.
/// Wraps any backing allocator and maintains counters for diagnostics.
pub const TrackingAllocator = struct {
    backing_allocator: Allocator,
    alloc_count: usize,
    free_count: usize,
    total_bytes_allocated: usize,
    total_bytes_freed: usize,
    active_bytes: isize,

    pub fn init(backing_allocator: Allocator) TrackingAllocator {
        return .{
            .backing_allocator = backing_allocator,
            .alloc_count = 0,
            .free_count = 0,
            .total_bytes_allocated = 0,
            .total_bytes_freed = 0,
            .active_bytes = 0,
        };
    }

    pub fn allocator(self: *TrackingAllocator) Allocator {
        return .{
            .ptr = @ptrCast(self),
            .vtable = &vtable,
        };
    }

    const vtable: Allocator.VTable = .{
        .alloc = alloc,
        .resize = resize,
        .remap = Allocator.noRemap,
        .free = freeFn,
    };

    fn alloc(ctx: *anyopaque, len: usize, ptr_align: Alignment, ret_addr: usize) ?[*]u8 {
        const self: *TrackingAllocator = @ptrCast(@alignCast(ctx));
        const result = self.backing_allocator.rawAlloc(len, ptr_align, ret_addr);
        if (result != null) {
            self.alloc_count += 1;
            self.total_bytes_allocated += len;
            self.active_bytes += @as(isize, @intCast(len));
        }
        return result;
    }

    fn resize(ctx: *anyopaque, memory: []u8, ptr_align: Alignment, new_len: usize, ret_addr: usize) bool {
        const self: *TrackingAllocator = @ptrCast(@alignCast(ctx));
        return self.backing_allocator.rawResize(memory, ptr_align, new_len, ret_addr);
    }

    fn freeFn(ctx: *anyopaque, memory: []u8, ptr_align: Alignment, ret_addr: usize) void {
        const self: *TrackingAllocator = @ptrCast(@alignCast(ctx));
        const len = memory.len;
        self.backing_allocator.rawFree(memory, ptr_align, ret_addr);
        self.free_count += 1;
        self.total_bytes_freed += len;
        self.active_bytes -= @as(isize, @intCast(len));
    }

    /// Returns true if there are outstanding allocations (leak detected).
    pub fn detectLeaks(self: *const TrackingAllocator) bool {
        return self.alloc_count != self.free_count;
    }

    /// Returns the number of currently outstanding (unfreed) allocations.
    pub fn outstandingAllocations(self: *const TrackingAllocator) usize {
        return self.alloc_count - self.free_count;
    }

    /// Print a summary of allocation statistics.
    pub fn dumpStats(self: *const TrackingAllocator) void {
        std.debug.print(
            \\Tracking Allocator Stats:
            \\  Allocations:  {}
            \\  Frees:        {}
            \\  Outstanding:  {}
            \\  Bytes alloc:  {}
            \\  Bytes freed:  {}
            \\  Active bytes: {}
            \\  Leaks:        {}
            \\
        , .{
            self.alloc_count,
            self.free_count,
            self.outstandingAllocations(),
            self.total_bytes_allocated,
            self.total_bytes_freed,
            self.active_bytes,
            self.detectLeaks(),
        });
    }
};
