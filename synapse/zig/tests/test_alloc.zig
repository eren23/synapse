const std = @import("std");
const testing = std.testing;

const ArenaAllocator = @import("arena").ArenaAllocator;
const PoolAllocator = @import("pool").PoolAllocator;
const AlignedAllocator = @import("aligned").AlignedAllocator;
const TrackingAllocator = @import("tracking").TrackingAllocator;

// ============================================================
// Arena Allocator Tests
// ============================================================

test "arena: basic allocation" {
    var arena = ArenaAllocator.init(std.heap.page_allocator, 4096);
    defer arena.deinit();

    const alloc = arena.allocator();

    const buf = try alloc.alloc(u8, 100);
    try testing.expect(buf.len == 100);

    const buf2 = try alloc.alloc(u32, 25);
    try testing.expect(buf2.len == 25);
}

test "arena: reset and re-alloc" {
    var arena = ArenaAllocator.init(std.heap.page_allocator, 4096);
    defer arena.deinit();

    const alloc = arena.allocator();

    // Allocate some memory
    _ = try alloc.alloc(u8, 1000);
    _ = try alloc.alloc(u8, 1000);
    _ = try alloc.alloc(u8, 1000);

    // Reset should be O(1) — just rewinds the pointer
    arena.reset();

    // Should be able to allocate again from the beginning
    const after_reset = try alloc.alloc(u8, 2500);
    try testing.expect(after_reset.len == 2500);
}

test "arena: multiple regions" {
    // Use a small region size to force multiple regions
    var arena = ArenaAllocator.init(std.heap.page_allocator, 64);
    defer arena.deinit();

    const alloc = arena.allocator();

    // Allocate more than one region worth
    var bufs: [20][]u8 = undefined;
    for (&bufs) |*buf| {
        buf.* = try alloc.alloc(u8, 32);
        @memset(buf.*, 0xAB);
    }

    // Verify all buffers are valid
    for (bufs) |buf| {
        for (buf) |byte| {
            try testing.expect(byte == 0xAB);
        }
    }
}

test "arena: large allocation exceeding region capacity" {
    var arena = ArenaAllocator.init(std.heap.page_allocator, 256);
    defer arena.deinit();

    const alloc = arena.allocator();

    // Allocate larger than region capacity
    const big = try alloc.alloc(u8, 1024);
    try testing.expect(big.len == 1024);
    @memset(big, 0xCD);

    for (big) |byte| {
        try testing.expect(byte == 0xCD);
    }
}

// ============================================================
// Pool Allocator Tests
// ============================================================

test "pool: acquire and release" {
    const Pool64 = PoolAllocator(64);
    var pool = try Pool64.init(std.heap.page_allocator, 100);
    defer pool.deinit();

    const slot = pool.acquire() orelse return error.TestFailed;
    const slice: []u8 = slot[0..64];
    @memset(slice, 0xFF);

    for (slice) |byte| {
        try testing.expect(byte == 0xFF);
    }

    pool.release(slot);
}

test "pool: exhaust and reacquire" {
    const Pool32 = PoolAllocator(32);
    var pool = try Pool32.init(std.heap.page_allocator, 5);
    defer pool.deinit();

    // Acquire all 5 slots
    var slots: [5][*]u8 = undefined;
    for (&slots) |*s| {
        s.* = pool.acquire() orelse return error.TestFailed;
    }

    // Pool should be exhausted
    try testing.expect(pool.acquire() == null);

    // Release one and reacquire
    pool.release(slots[2]);
    const reacquired = pool.acquire() orelse return error.TestFailed;
    try testing.expect(reacquired == slots[2]);

    // Should be exhausted again
    try testing.expect(pool.acquire() == null);
}

test "pool: all slots usable after cycling" {
    const Pool16 = PoolAllocator(16);
    var pool = try Pool16.init(std.heap.page_allocator, 10);
    defer pool.deinit();

    // Acquire and release all slots multiple times
    var i: usize = 0;
    while (i < 3) : (i += 1) {
        var slots: [10][*]u8 = undefined;
        for (&slots) |*s| {
            s.* = pool.acquire() orelse return error.TestFailed;
        }
        for (&slots) |*s| {
            pool.release(s.*);
        }
    }

    // Should still be able to acquire all 10
    var count: usize = 0;
    while (pool.acquire() != null) {
        count += 1;
    }
    try testing.expect(count == 10);
}

// ============================================================
// Aligned Allocator Tests
// ============================================================

test "aligned: 64-byte alignment" {
    var aligned = AlignedAllocator.init(std.heap.page_allocator);
    const alloc = aligned.allocator();

    // Allocate various sizes and verify alignment
    const sizes = [_]usize{ 1, 7, 15, 32, 63, 64, 100, 256, 1000 };
    for (sizes) |size| {
        const buf = try alloc.alloc(u8, size);
        defer alloc.free(buf);

        const addr = @intFromPtr(buf.ptr);
        try testing.expect(addr % 64 == 0);
    }
}

test "aligned: multiple allocations all aligned" {
    var aligned = AlignedAllocator.init(std.heap.page_allocator);
    const alloc = aligned.allocator();

    var bufs: [50][]u8 = undefined;
    var allocated: usize = 0;

    for (&bufs, 0..) |*buf, i| {
        buf.* = try alloc.alloc(u8, (i + 1) * 13);
        allocated += 1;

        const addr = @intFromPtr(buf.ptr);
        try testing.expect(addr % 64 == 0);
    }

    // Free all
    for (bufs[0..allocated]) |buf| {
        alloc.free(buf);
    }
}

// ============================================================
// Tracking Allocator Tests
// ============================================================

test "tracking: counts allocs and frees" {
    var tracking = TrackingAllocator.init(std.heap.page_allocator);
    const alloc = tracking.allocator();

    const a = try alloc.alloc(u8, 100);
    const b = try alloc.alloc(u8, 200);

    try testing.expect(tracking.alloc_count == 2);
    try testing.expect(tracking.free_count == 0);
    try testing.expect(tracking.detectLeaks() == true);

    alloc.free(a);
    try testing.expect(tracking.alloc_count == 2);
    try testing.expect(tracking.free_count == 1);
    try testing.expect(tracking.detectLeaks() == true);
    try testing.expect(tracking.outstandingAllocations() == 1);

    alloc.free(b);
    try testing.expect(tracking.alloc_count == 2);
    try testing.expect(tracking.free_count == 2);
    try testing.expect(tracking.detectLeaks() == false);
    try testing.expect(tracking.outstandingAllocations() == 0);
}

test "tracking: leak detection" {
    var tracking = TrackingAllocator.init(std.heap.page_allocator);
    const alloc = tracking.allocator();

    // Allocate without freeing
    _ = try alloc.alloc(u8, 64);
    _ = try alloc.alloc(u8, 128);
    _ = try alloc.alloc(u8, 256);

    try testing.expect(tracking.detectLeaks() == true);
    try testing.expect(tracking.outstandingAllocations() == 3);
    try testing.expect(tracking.total_bytes_allocated == 64 + 128 + 256);
}

test "tracking: byte accounting" {
    var tracking = TrackingAllocator.init(std.heap.page_allocator);
    const alloc = tracking.allocator();

    const a = try alloc.alloc(u8, 100);
    const b = try alloc.alloc(u8, 200);

    try testing.expect(tracking.total_bytes_allocated == 300);
    try testing.expect(tracking.active_bytes == 300);

    alloc.free(a);
    try testing.expect(tracking.total_bytes_freed == 100);
    try testing.expect(tracking.active_bytes == 200);

    alloc.free(b);
    try testing.expect(tracking.total_bytes_freed == 300);
    try testing.expect(tracking.active_bytes == 0);
}

test "tracking: no leaks when all freed" {
    var tracking = TrackingAllocator.init(std.heap.page_allocator);
    const alloc = tracking.allocator();

    var bufs: [10][]u8 = undefined;
    for (&bufs, 0..) |*buf, i| {
        buf.* = try alloc.alloc(u8, (i + 1) * 50);
    }

    try testing.expect(tracking.detectLeaks() == true);

    for (bufs) |buf| {
        alloc.free(buf);
    }

    try testing.expect(tracking.detectLeaks() == false);
    try testing.expect(tracking.outstandingAllocations() == 0);
    try testing.expect(tracking.active_bytes == 0);
}
