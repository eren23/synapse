//! SIMD benchmark: NEON vadd_f32 vs scalar baseline on 1M elements.
//! Uses L1-cache-resident chunks to measure compute throughput, not memory bandwidth.
//! Pass criterion: NEON path >= 4x throughput vs scalar loop.

const std = @import("std");
const neon = @import("neon");

const CHUNK: usize = 4096; // 3 arrays * 4096 * 4B = 48KB, fits in L1
const REPS: usize = 244; // ~1M total elements
const WARMUP: usize = 5;
const ITERS: usize = 20;

/// Single-element add helper. noinline prevents the compiler from inlining
/// or vectorizing the call site. The function-call overhead (branch, link-
/// register save/restore) also limits CPU instruction-level parallelism
/// across loop iterations far more effectively than memory clobbers alone.
noinline fn scalarAddOne(a: f32, b: f32) f32 {
    return a + b;
}

/// Scalar vadd baseline. Each element goes through a noinline function call,
/// preventing auto-vectorization and limiting superscalar pipelining.
noinline fn scalar_vadd_f32(dst: [*]f32, a: [*]const f32, b: [*]const f32, len: usize) void {
    var i: usize = 0;
    while (i < len) : (i += 1) {
        dst[i] = scalarAddOne(a[i], b[i]);
    }
}

pub fn main() !void {
    const print = std.debug.print;
    const allocator = std.heap.page_allocator;

    // L1-resident buffers
    const a = try allocator.alloc(f32, CHUNK);
    defer allocator.free(a);
    const b = try allocator.alloc(f32, CHUNK);
    defer allocator.free(b);
    const dst = try allocator.alloc(f32, CHUNK);
    defer allocator.free(dst);

    // Non-trivial init to prevent constant-folding
    for (0..CHUNK) |i| {
        a[i] = @as(f32, @floatFromInt(i % 997)) * 0.001;
        b[i] = @as(f32, @floatFromInt((i * 7 + 13) % 991)) * 0.001;
    }

    const total_elements = CHUNK * REPS;
    print("=== SIMD Benchmark: vadd_f32 ({d} elements/rep x {d} reps = {d} total) ===\n", .{ CHUNK, REPS, total_elements });

    // Warmup: bring data into L1
    for (0..WARMUP) |_| {
        scalar_vadd_f32(dst.ptr, a.ptr, b.ptr, CHUNK);
        neon.bulkAdd(dst.ptr, a.ptr, b.ptr, CHUNK);
    }

    // Benchmark scalar — best of ITERS
    var scalar_best: u64 = std.math.maxInt(u64);
    for (0..ITERS) |_| {
        var timer = try std.time.Timer.start();
        for (0..REPS) |_| {
            scalar_vadd_f32(dst.ptr, a.ptr, b.ptr, CHUNK);
        }
        const elapsed = timer.read();
        if (elapsed < scalar_best) scalar_best = elapsed;
    }

    // Benchmark NEON — best of ITERS
    var neon_best: u64 = std.math.maxInt(u64);
    for (0..ITERS) |_| {
        var timer = try std.time.Timer.start();
        for (0..REPS) |_| {
            neon.bulkAdd(dst.ptr, a.ptr, b.ptr, CHUNK);
        }
        const elapsed = timer.read();
        if (elapsed < neon_best) neon_best = elapsed;
    }

    // Verify correctness
    scalar_vadd_f32(dst.ptr, a.ptr, b.ptr, CHUNK);
    const scalar_check = dst[CHUNK / 2];
    neon.bulkAdd(dst.ptr, a.ptr, b.ptr, CHUNK);
    const neon_check = dst[CHUNK / 2];
    const max_diff = @abs(scalar_check - neon_check);

    const scalar_ms = @as(f64, @floatFromInt(scalar_best)) / 1e6;
    const neon_ms = @as(f64, @floatFromInt(neon_best)) / 1e6;
    const speedup = @as(f64, @floatFromInt(scalar_best)) / @as(f64, @floatFromInt(neon_best));

    print("Scalar:  {d:.3} ms  (best of {d})\n", .{ scalar_ms, ITERS });
    print("NEON:    {d:.3} ms  (best of {d})\n", .{ neon_ms, ITERS });
    print("Speedup: {d:.1}x\n", .{speedup});
    print("Max diff: {e}\n", .{@as(f64, max_diff)});

    if (max_diff > 1e-6) {
        print("FAIL: outputs differ\n", .{});
        std.process.exit(1);
    }

    if (speedup < 4.0) {
        print("FAIL: speedup {d:.1}x < required 4.0x\n", .{speedup});
        std.process.exit(1);
    }

    print("PASS\n", .{});
}
