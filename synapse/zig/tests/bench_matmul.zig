//! Benchmark: 512x512 SGEMM, 100 iterations.
//! Compares naive triple-loop vs tiled SGEMM with 8x8 micro-kernel.
//! Pass criteria: >= 5x speedup, >= 2 GFLOPS on tiled.

const std = @import("std");
const matmul = @import("synapse").ops.matmul;

const N_SIZE: usize = 512;
const WARMUP: usize = 3;
const ITERS: usize = 100;

// 2 * M * N * K FLOPs per SGEMM (multiply + add)
const FLOPS_PER_ITER: u64 = 2 * N_SIZE * N_SIZE * N_SIZE;
const TOTAL_FLOPS: u64 = FLOPS_PER_ITER * ITERS;

pub fn main() !void {
    const print = std.debug.print;
    const allocator = std.heap.page_allocator;

    // Allocate matrices A[N,N], B[N,N], C[N,N]
    const a = try allocator.alloc(f32, N_SIZE * N_SIZE);
    defer allocator.free(a);
    const b = try allocator.alloc(f32, N_SIZE * N_SIZE);
    defer allocator.free(b);
    const c_naive = try allocator.alloc(f32, N_SIZE * N_SIZE);
    defer allocator.free(c_naive);
    const c_tiled = try allocator.alloc(f32, N_SIZE * N_SIZE);
    defer allocator.free(c_tiled);

    // Packing buffers
    const packed_a = try allocator.alloc(f32, matmul.MC * matmul.KC);
    defer allocator.free(packed_a);
    const packed_b = try allocator.alloc(f32, matmul.KC * N_SIZE);
    defer allocator.free(packed_b);

    // Fill A and B with deterministic data
    fillData(a, 42);
    fillData(b, 137);

    print("=== SGEMM Benchmark: {d}x{d} x {d} iters ===\n", .{ N_SIZE, N_SIZE, ITERS });
    print("FLOPs per iter: {d:.2} M\n", .{@as(f64, @floatFromInt(FLOPS_PER_ITER)) / 1e6});
    print("\n", .{});

    // ---- Warmup ----
    var sink: f32 = 0;
    for (0..WARMUP) |_| {
        zeroFill(c_tiled);
        matmul.sgemmTiled(
            N_SIZE, N_SIZE, N_SIZE,
            a.ptr, N_SIZE, false,
            b.ptr, N_SIZE, false,
            c_tiled.ptr, N_SIZE,
            packed_a.ptr, packed_b.ptr,
        );
        sink += c_tiled[0];
    }

    // ---- Benchmark naive ----
    const naive_start = std.time.nanoTimestamp();
    for (0..ITERS) |_| {
        zeroFill(c_naive);
        matmul.naiveSgemm(
            N_SIZE, N_SIZE, N_SIZE,
            a.ptr, N_SIZE, false,
            b.ptr, N_SIZE, false,
            c_naive.ptr, N_SIZE,
        );
        asm volatile ("" ::: .{ .memory = true });
    }
    const naive_end = std.time.nanoTimestamp();
    const naive_ns: u64 = @intCast(naive_end - naive_start);
    sink += c_naive[0];

    // ---- Benchmark tiled ----
    const tiled_start = std.time.nanoTimestamp();
    for (0..ITERS) |_| {
        zeroFill(c_tiled);
        matmul.sgemmTiled(
            N_SIZE, N_SIZE, N_SIZE,
            a.ptr, N_SIZE, false,
            b.ptr, N_SIZE, false,
            c_tiled.ptr, N_SIZE,
            packed_a.ptr, packed_b.ptr,
        );
        asm volatile ("" ::: .{ .memory = true });
    }
    const tiled_end = std.time.nanoTimestamp();
    const tiled_ns: u64 = @intCast(tiled_end - tiled_start);
    sink += c_tiled[0];

    // ---- Verify correctness (spot check) ----
    zeroFill(c_naive);
    matmul.naiveSgemm(N_SIZE, N_SIZE, N_SIZE, a.ptr, N_SIZE, false, b.ptr, N_SIZE, false, c_naive.ptr, N_SIZE);
    zeroFill(c_tiled);
    matmul.sgemmTiled(N_SIZE, N_SIZE, N_SIZE, a.ptr, N_SIZE, false, b.ptr, N_SIZE, false, c_tiled.ptr, N_SIZE, packed_a.ptr, packed_b.ptr);

    var max_diff: f32 = 0;
    for (0..N_SIZE * N_SIZE) |i| {
        const diff = @abs(c_tiled[i] - c_naive[i]);
        const denom = @max(@abs(c_naive[i]), @as(f32, 1.0));
        const rel = diff / denom;
        if (rel > max_diff) max_diff = rel;
    }

    // ---- Report ----
    const naive_ms = @as(f64, @floatFromInt(naive_ns)) / 1e6;
    const tiled_ms = @as(f64, @floatFromInt(tiled_ns)) / 1e6;
    const speedup = @as(f64, @floatFromInt(naive_ns)) / @as(f64, @floatFromInt(tiled_ns));
    const tiled_gflops = @as(f64, @floatFromInt(TOTAL_FLOPS)) / @as(f64, @floatFromInt(tiled_ns));
    const naive_gflops = @as(f64, @floatFromInt(TOTAL_FLOPS)) / @as(f64, @floatFromInt(naive_ns));

    print("Naive:   {d:.2} ms  ({d:.2} GFLOPS)\n", .{ naive_ms, naive_gflops });
    print("Tiled:   {d:.2} ms  ({d:.2} GFLOPS)\n", .{ tiled_ms, tiled_gflops });
    print("Speedup: {d:.2}x\n", .{speedup});
    print("Max relative error: {e:.2}\n", .{max_diff});
    print("\n", .{});

    // ---- Pass/fail checks ----
    var passed = true;

    if (max_diff > 1e-4) {
        print("FAIL: max relative error {e:.2} > 1e-4\n", .{max_diff});
        passed = false;
    } else {
        print("PASS: correctness within 1e-4\n", .{});
    }

    if (speedup < 5.0) {
        print("FAIL: speedup {d:.2}x < required 5.0x\n", .{speedup});
        passed = false;
    } else {
        print("PASS: speedup {d:.2}x >= 5.0x\n", .{speedup});
    }

    if (tiled_gflops < 2.0) {
        print("FAIL: tiled {d:.2} GFLOPS < required 2.0 GFLOPS\n", .{tiled_gflops});
        passed = false;
    } else {
        print("PASS: tiled {d:.2} GFLOPS >= 2.0 GFLOPS\n", .{tiled_gflops});
    }

    // Prevent sink from being optimized away
    if (sink == 0.0) print("sink: {d}\n", .{sink});

    if (!passed) std.process.exit(1);
}

fn fillData(data: []f32, seed: u32) void {
    var s: u32 = seed;
    for (data) |*v| {
        s = s *% 1103515245 +% 12345;
        const bits: i32 = @bitCast(s);
        const shifted: i16 = @truncate(bits >> 16);
        v.* = @as(f32, @floatFromInt(shifted)) / 32768.0;
    }
}

fn zeroFill(data: []f32) void {
    @memset(data, 0);
}
