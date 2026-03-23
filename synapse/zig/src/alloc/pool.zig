const std = @import("std");
const Allocator = std.mem.Allocator;
const Alignment = std.mem.Alignment;

/// Fixed-size slab allocator using a free-list for O(1) acquire/release.
/// Pre-allocates a fixed number of slots of uniform size. Each freed slot
/// is pushed onto an intrusive free list embedded in the slot memory itself.
pub fn PoolAllocator(comptime slot_size: usize) type {
    const raw_size = @max(slot_size, @sizeOf(usize));
    const align_val = @alignOf(usize);
    const effective_slot_size = ((raw_size + align_val - 1) / align_val) * align_val;

    return struct {
        const Self = @This();

        backing: []u8,
        free_head: ?usize, // index into slots
        slot_count: usize,
        backing_allocator: Allocator,

        /// Initialize the pool, pre-allocating `count` slots from `backing_allocator`.
        pub fn init(backing_allocator: Allocator, count: usize) !Self {
            const total = effective_slot_size * count;
            const mem = try backing_allocator.alloc(u8, total);

            // Build free list: each slot stores the index of the next free slot
            var i: usize = 0;
            while (i < count) : (i += 1) {
                const slot_ptr: *usize = @ptrCast(@alignCast(mem.ptr + i * effective_slot_size));
                if (i + 1 < count) {
                    slot_ptr.* = i + 1;
                } else {
                    slot_ptr.* = std.math.maxInt(usize); // sentinel for end
                }
            }

            return .{
                .backing = mem,
                .free_head = if (count > 0) 0 else null,
                .slot_count = count,
                .backing_allocator = backing_allocator,
            };
        }

        /// Acquire a slot from the pool. Returns null if pool is exhausted.
        pub fn acquire(self: *Self) ?[*]u8 {
            const idx = self.free_head orelse return null;
            const ptr = self.backing.ptr + idx * effective_slot_size;
            const next_ptr: *const usize = @ptrCast(@alignCast(ptr));
            const next = next_ptr.*;
            self.free_head = if (next == std.math.maxInt(usize)) null else next;
            return ptr;
        }

        /// Release a slot back to the pool.
        pub fn release(self: *Self, ptr: [*]u8) void {
            const addr = @intFromPtr(ptr);
            const base = @intFromPtr(self.backing.ptr);
            const offset = addr - base;
            const idx = offset / effective_slot_size;

            const slot_ptr: *usize = @ptrCast(@alignCast(ptr));
            slot_ptr.* = if (self.free_head) |h| h else std.math.maxInt(usize);
            self.free_head = idx;
        }

        /// Return a std.mem.Allocator interface backed by this pool.
        /// Only supports allocations of exactly `slot_size` bytes or less.
        pub fn allocator(self: *Self) Allocator {
            return .{
                .ptr = @ptrCast(self),
                .vtable = &vtable,
            };
        }

        const vtable: Allocator.VTable = .{
            .alloc = poolAlloc,
            .resize = Allocator.noResize,
            .remap = Allocator.noRemap,
            .free = poolFree,
        };

        fn poolAlloc(ctx: *anyopaque, len: usize, _: Alignment, _: usize) ?[*]u8 {
            const self: *Self = @ptrCast(@alignCast(ctx));
            if (len > effective_slot_size) return null;
            return self.acquire();
        }

        fn poolFree(ctx: *anyopaque, memory: []u8, _: Alignment, _: usize) void {
            const self: *Self = @ptrCast(@alignCast(ctx));
            self.release(memory.ptr);
        }

        /// Free the backing memory.
        pub fn deinit(self: *Self) void {
            self.backing_allocator.free(self.backing);
            self.backing = &.{};
            self.free_head = null;
            self.slot_count = 0;
        }
    };
}
