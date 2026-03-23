const std = @import("std");
const ArenaAllocator = @import("arena").ArenaAllocator;
const PoolAllocator = @import("pool").PoolAllocator;

const ITERATIONS = 10_000;

// ============================================================
// Helpers
// ============================================================

fn formatDuration(ns: u64) f64 {
    return @as(f64, @floatFromInt(ns)) / 1_000_000.0;
}

// ============================================================
// Arena Benchmark: Arena vs std.heap.page_allocator (10K mixed-size)
// ============================================================

/// Naive baseline: use page_allocator directly for mixed-size allocations.
fn benchNaivePageAllocator() !u64 {
    const alloc = std.heap.page_allocator;
    var timer = try std.time.Timer.start();

    var bufs: [ITERATIONS][]u8 = undefined;

    // Mixed-size allocations: 8, 16, 32, 64, 128, 256, 512, 1024
    const size_table = [_]usize{ 8, 16, 32, 64, 128, 256, 512, 1024 };

    for (0..ITERATIONS) |i| {
        const size = size_table[i % size_table.len];
        bufs[i] = try alloc.alloc(u8, size);
    }

    // Free all
    for (0..ITERATIONS) |i| {
        alloc.free(bufs[i]);
    }

    return timer.read();
}

/// Optimized: use arena allocator for mixed-size allocations.
fn benchArenaAllocator() !u64 {
    var arena = ArenaAllocator.init(std.heap.page_allocator, 1024 * 1024); // 1MB region
    defer arena.deinit();

    const alloc = arena.allocator();
    var timer = try std.time.Timer.start();

    const size_table = [_]usize{ 8, 16, 32, 64, 128, 256, 512, 1024 };

    for (0..ITERATIONS) |i| {
        const size = size_table[i % size_table.len];
        _ = try alloc.alloc(u8, size);
    }

    // Arena reset frees everything in O(1)
    arena.reset();

    return timer.read();
}

// ============================================================
// Pool Benchmark: Pool vs malloc/free (page_allocator) for fixed-size (10K)
// ============================================================

/// Naive baseline: use page_allocator for fixed-size 64-byte allocations.
fn benchNaiveMallocFree() !u64 {
    const alloc = std.heap.page_allocator;
    var timer = try std.time.Timer.start();

    for (0..ITERATIONS) |_| {
        const buf = try alloc.alloc(u8, 64);
        alloc.free(buf);
    }

    return timer.read();
}

/// Optimized: use pool allocator for fixed-size 64-byte acquire/release.
fn benchPoolAllocator() !u64 {
    const Pool64 = PoolAllocator(64);
    var pool = try Pool64.init(std.heap.page_allocator, ITERATIONS);
    defer pool.deinit();

    var timer = try std.time.Timer.start();

    for (0..ITERATIONS) |_| {
        const ptr = pool.acquire() orelse return error.PoolExhausted;
        pool.release(ptr);
    }

    return timer.read();
}

// ============================================================
// Main: run benchmarks and report results
// ============================================================

pub fn main() !void {
    const print = std.debug.print;

    print("=== Memory Allocator Benchmarks ({} iterations) ===\n\n", .{ITERATIONS});

    // --- Arena Benchmark ---
    const naive_arena_ns = try benchNaivePageAllocator();
    const arena_ns = try benchArenaAllocator();
    const arena_speedup = @as(f64, @floatFromInt(naive_arena_ns)) / @as(f64, @floatFromInt(arena_ns));

    print("--- Arena vs page_allocator (mixed-size) ---\n", .{});
    print("  page_allocator: {d:.3} ms\n", .{formatDuration(naive_arena_ns)});
    print("  Arena:          {d:.3} ms\n", .{formatDuration(arena_ns)});
    print("  Speedup:        {d:.1}x\n", .{arena_speedup});
    if (arena_speedup >= 3.0) {
        print("  PASS: Arena >= 3x throughput vs page_allocator\n\n", .{});
    } else {
        print("  FAIL: Arena speedup {d:.1}x < 3x target\n\n", .{arena_speedup});
    }

    // --- Pool Benchmark ---
    const naive_pool_ns = try benchNaiveMallocFree();
    const pool_ns = try benchPoolAllocator();
    const pool_speedup = @as(f64, @floatFromInt(naive_pool_ns)) / @as(f64, @floatFromInt(pool_ns));

    print("--- Pool vs malloc/free (fixed-size 64B) ---\n", .{});
    print("  page_allocator: {d:.3} ms\n", .{formatDuration(naive_pool_ns)});
    print("  Pool:           {d:.3} ms\n", .{formatDuration(pool_ns)});
    print("  Speedup:        {d:.1}x\n", .{pool_speedup});
    if (pool_speedup >= 5.0) {
        print("  PASS: Pool >= 5x throughput vs malloc/free\n\n", .{});
    } else {
        print("  FAIL: Pool speedup {d:.1}x < 5x target\n\n", .{pool_speedup});
    }

    // --- Summary ---
    const all_pass = arena_speedup >= 3.0 and pool_speedup >= 5.0;
    if (all_pass) {
        print("=== ALL BENCHMARKS PASSED ===\n", .{});
    } else {
        print("=== SOME BENCHMARKS FAILED ===\n", .{});
        std.process.exit(1);
    }
}
