const std = @import("std");
const Allocator = std.mem.Allocator;
const Alignment = std.mem.Alignment;

/// Region-based arena allocator with O(1) reset and bump allocation.
/// Allocates from a contiguous backing buffer using a simple bump pointer.
/// When the current region is exhausted, a new region is allocated from the
/// backing allocator. Reset releases all regions back to the initial state.
pub const ArenaAllocator = struct {
    const Region = struct {
        data: []u8,
        next: ?*Region,
    };

    backing_allocator: Allocator,
    first_region: ?*Region,
    current_region: ?*Region,
    offset: usize,
    region_capacity: usize,

    /// Initialize an arena with the given backing allocator and per-region capacity.
    pub fn init(backing_allocator: Allocator, region_capacity: usize) ArenaAllocator {
        return .{
            .backing_allocator = backing_allocator,
            .first_region = null,
            .current_region = null,
            .offset = 0,
            .region_capacity = region_capacity,
        };
    }

    /// Return a std.mem.Allocator interface backed by this arena.
    pub fn allocator(self: *ArenaAllocator) Allocator {
        return .{
            .ptr = @ptrCast(self),
            .vtable = &vtable,
        };
    }

    const vtable: Allocator.VTable = .{
        .alloc = alloc,
        .resize = Allocator.noResize,
        .remap = Allocator.noRemap,
        .free = free,
    };

    fn ensureRegion(self: *ArenaAllocator, min_size: usize) !void {
        if (self.current_region) |region| {
            if (self.offset + min_size <= region.data.len) return;
            // Check if there's a next region we can reuse (after a reset)
            if (region.next) |next| {
                if (min_size <= next.data.len) {
                    self.current_region = next;
                    self.offset = 0;
                    return;
                }
            }
        }

        const cap = @max(self.region_capacity, min_size);
        const total_size = @sizeOf(Region) + cap;
        const raw = self.backing_allocator.rawAlloc(total_size, Alignment.of(Region), @returnAddress()) orelse return error.OutOfMemory;

        const region: *Region = @ptrCast(@alignCast(raw));
        const data_start = raw + @sizeOf(Region);
        region.* = .{
            .data = data_start[0..cap],
            .next = null,
        };

        if (self.current_region) |cur| {
            cur.next = region;
        }

        if (self.first_region == null) {
            self.first_region = region;
        }

        self.current_region = region;
        self.offset = 0;
    }

    fn alloc(ctx: *anyopaque, len: usize, ptr_align: Alignment, _: usize) ?[*]u8 {
        const self: *ArenaAllocator = @ptrCast(@alignCast(ctx));
        const alignment = ptr_align.toByteUnits();
        self.ensureRegion(len + alignment - 1) catch return null;

        const region = self.current_region.?;
        const base_addr = @intFromPtr(region.data.ptr) + self.offset;
        const aligned_addr = std.mem.alignForward(usize, base_addr, alignment);
        const padding = aligned_addr - base_addr;

        if (self.offset + padding + len > region.data.len) {
            // Need a new region
            self.ensureRegion(len + alignment - 1) catch return null;
            const new_region = self.current_region.?;
            const new_base = @intFromPtr(new_region.data.ptr);
            const new_aligned = std.mem.alignForward(usize, new_base, alignment);
            const new_padding = new_aligned - new_base;
            self.offset = new_padding + len;
            return @ptrFromInt(new_aligned);
        }

        self.offset += padding + len;
        return @ptrFromInt(aligned_addr);
    }

    fn free(_: *anyopaque, _: []u8, _: Alignment, _: usize) void {
        // Arena doesn't free individual allocations
    }

    /// O(1) reset: rewind the bump pointer to the start of the first region.
    /// All previously allocated memory becomes invalid.
    pub fn reset(self: *ArenaAllocator) void {
        self.current_region = self.first_region;
        self.offset = 0;
    }

    /// Release all backing memory.
    pub fn deinit(self: *ArenaAllocator) void {
        var maybe_region = self.first_region;
        while (maybe_region) |region| {
            const next = region.next;
            const raw: [*]u8 = @ptrCast(@alignCast(region));
            const full_slice = raw[0 .. @sizeOf(Region) + region.data.len];
            self.backing_allocator.rawFree(full_slice, Alignment.of(Region), @returnAddress());
            maybe_region = next;
        }
        self.first_region = null;
        self.current_region = null;
        self.offset = 0;
    }
};
