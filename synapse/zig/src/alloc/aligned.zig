const std = @import("std");
const Allocator = std.mem.Allocator;
const Alignment = std.mem.Alignment;

/// Wrapper allocator ensuring all allocations are 64-byte aligned for SIMD.
/// Delegates to a backing allocator with forced alignment.
pub const AlignedAllocator = struct {
    backing_allocator: Allocator,

    pub const ALIGNMENT: usize = 64;
    const min_alignment: Alignment = Alignment.fromByteUnits(ALIGNMENT); // @"64" == 6

    pub fn init(backing_allocator: Allocator) AlignedAllocator {
        return .{ .backing_allocator = backing_allocator };
    }

    pub fn allocator(self: *AlignedAllocator) Allocator {
        return .{
            .ptr = @ptrCast(self),
            .vtable = &vtable,
        };
    }

    const vtable: Allocator.VTable = .{
        .alloc = alloc,
        .resize = resize,
        .remap = remap,
        .free = freeFn,
    };

    fn maxAlign(a: Alignment, b: Alignment) Alignment {
        const a_int = @intFromEnum(a);
        const b_int = @intFromEnum(b);
        return if (a_int >= b_int) a else b;
    }

    fn alloc(ctx: *anyopaque, len: usize, ptr_align: Alignment, ret_addr: usize) ?[*]u8 {
        const self: *AlignedAllocator = @ptrCast(@alignCast(ctx));
        const effective_align = maxAlign(ptr_align, min_alignment);
        return self.backing_allocator.rawAlloc(len, effective_align, ret_addr);
    }

    fn resize(ctx: *anyopaque, memory: []u8, ptr_align: Alignment, new_len: usize, ret_addr: usize) bool {
        const self: *AlignedAllocator = @ptrCast(@alignCast(ctx));
        const effective_align = maxAlign(ptr_align, min_alignment);
        return self.backing_allocator.rawResize(memory, effective_align, new_len, ret_addr);
    }

    fn remap(ctx: *anyopaque, memory: []u8, ptr_align: Alignment, new_len: usize, ret_addr: usize) ?[*]u8 {
        const self: *AlignedAllocator = @ptrCast(@alignCast(ctx));
        const effective_align = maxAlign(ptr_align, min_alignment);
        return self.backing_allocator.rawRemap(memory, effective_align, new_len, ret_addr);
    }

    fn freeFn(ctx: *anyopaque, memory: []u8, ptr_align: Alignment, ret_addr: usize) void {
        const self: *AlignedAllocator = @ptrCast(@alignCast(ctx));
        const effective_align = maxAlign(ptr_align, min_alignment);
        self.backing_allocator.rawFree(memory, effective_align, ret_addr);
    }
};
