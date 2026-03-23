//! Benchmarks:
//! 1. SIMD reduce_sum vs scalar on 1M elements (target: >= 3x speedup).
//! 2. Single-pass batchnorm stats vs two-pass (target: >= 1.5x speedup).

const std = @import("std");

const N_REDUCE: usize = 1_000_000;
const N_BN: usize = 200_000;
const C_BN: usize = 100;
const REDUCE_WARMUP: usize = 5;
const REDUCE_ITERS: usize = 100;
const BN_WARMUP: usize = 3;
const BN_ITERS: usize = 20;

// ============================================================
// SIMD reduce_sum (4-wide vectors)
// ============================================================

const VEC_LEN = 4;
const F32x4 = @Vector(VEC_LEN, f32);

fn simd_reduce_sum(data: []const f32) f32 {
    const len = data.len;
    var acc: F32x4 = @splat(0.0);
    var i: usize = 0;

    while (i + VEC_LEN <= len) : (i += VEC_LEN) {
        const v: F32x4 = data[i..][0..VEC_LEN].*;
        acc += v;
    }

    var sum: f32 = @reduce(.Add, acc);
    while (i < len) : (i += 1) {
        sum += data[i];
    }
    return sum;
}

/// Scalar sum with asm barrier to prevent auto-vectorization.
noinline fn scalar_reduce_sum(ptr: [*]const f32, len: usize) f32 {
    var sum: f32 = 0;
    var i: usize = 0;
    while (i < len) : (i += 1) {
        sum += ptr[i];
        asm volatile ("" ::: .{ .memory = true });
    }
    return sum;
}

// ============================================================
// Single-pass batchnorm stats: computes sum and sum_sq in one pass.
// This reads data once (1 pass) vs two-pass which reads twice.
// ============================================================

noinline fn single_pass_stats(
    data: []const f32,
    n: usize,
    c: usize,
    ch: usize,
) struct { mean: f32, variance: f32 } {
    var sum: f32 = 0;
    var sum_sq: f32 = 0;
    for (0..n) |i| {
        const x = data[i * c + ch];
        sum += x;
        sum_sq += x * x;
    }
    const n_f: f32 = @floatFromInt(n);
    const mean = sum / n_f;
    return .{ .mean = mean, .variance = sum_sq / n_f - mean * mean };
}

// ============================================================
// Two-pass batchnorm stats: compute mean in pass 1, variance in pass 2.
// ============================================================

noinline fn two_pass_stats(
    data: []const f32,
    n: usize,
    c: usize,
    ch: usize,
) struct { mean: f32, variance: f32 } {
    const n_f: f32 = @floatFromInt(n);

    // Pass 1: mean
    var sum: f32 = 0;
    for (0..n) |i| {
        sum += data[i * c + ch];
    }
    const mean = sum / n_f;

    // Pass 2: variance
    var sq_sum: f32 = 0;
    for (0..n) |i| {
        const diff = data[i * c + ch] - mean;
        sq_sum += diff * diff;
    }

    return .{ .mean = mean, .variance = sq_sum / n_f };
}

// ============================================================
// Main
// ============================================================

pub fn main() !void {
    const print = std.debug.print;
    const allocator = std.heap.page_allocator;

    // ---- Benchmark 1: SIMD reduce_sum ----
    const data = try allocator.alloc(f32, N_REDUCE);
    defer allocator.free(data);
    for (0..N_REDUCE) |i| {
        data[i] = @as(f32, @floatFromInt(i)) * 1.0e-4;
    }

    // Warmup
    var sink: f32 = 0;
    for (0..REDUCE_WARMUP) |_| {
        sink += simd_reduce_sum(data);
        sink += scalar_reduce_sum(data.ptr, data.len);
    }

    // Benchmark scalar
    const scalar_start = std.time.nanoTimestamp();
    for (0..REDUCE_ITERS) |_| {
        sink += scalar_reduce_sum(data.ptr, data.len);
        asm volatile ("" ::: .{ .memory = true });
    }
    const scalar_end = std.time.nanoTimestamp();
    const scalar_ns: u64 = @intCast(scalar_end - scalar_start);

    // Benchmark SIMD
    const simd_start = std.time.nanoTimestamp();
    for (0..REDUCE_ITERS) |_| {
        sink += simd_reduce_sum(data);
        asm volatile ("" ::: .{ .memory = true });
    }
    const simd_end = std.time.nanoTimestamp();
    const simd_ns: u64 = @intCast(simd_end - simd_start);

    const reduce_speedup = @as(f64, @floatFromInt(scalar_ns)) / @as(f64, @floatFromInt(simd_ns));

    print("=== Benchmark 1: SIMD reduce_sum on {d} elements x {d} iters ===\n", .{ N_REDUCE, REDUCE_ITERS });
    print("Scalar:  {d:.2} ms\n", .{@as(f64, @floatFromInt(scalar_ns)) / 1e6});
    print("SIMD:    {d:.2} ms\n", .{@as(f64, @floatFromInt(simd_ns)) / 1e6});
    print("Speedup: {d:.2}x\n", .{reduce_speedup});

    if (reduce_speedup < 3.0) {
        print("FAIL: reduce speedup {d:.2}x < required 3.0x\n", .{reduce_speedup});
        std.process.exit(1);
    }
    print("PASS\n\n", .{});

    // ---- Benchmark 2: Single-pass vs two-pass batchnorm stats ----
    const bn_data = try allocator.alloc(f32, N_BN * C_BN);
    defer allocator.free(bn_data);
    for (0..N_BN * C_BN) |i| {
        bn_data[i] = @as(f32, @floatFromInt(i % 997)) * 0.01 - 5.0;
    }

    // Warmup
    for (0..BN_WARMUP) |_| {
        for (0..C_BN) |c| {
            const sp = single_pass_stats(bn_data, N_BN, C_BN, c);
            sink += sp.mean;
            const tp = two_pass_stats(bn_data, N_BN, C_BN, c);
            sink += tp.mean;
        }
    }

    // Benchmark two-pass (3 data reads: 2 for stats + shared overhead)
    const tp_start = std.time.nanoTimestamp();
    for (0..BN_ITERS) |_| {
        for (0..C_BN) |c| {
            const r = two_pass_stats(bn_data, N_BN, C_BN, c);
            sink += r.mean + r.variance;
        }
    }
    const tp_end = std.time.nanoTimestamp();
    const tp_ns: u64 = @intCast(tp_end - tp_start);

    // Benchmark single-pass (1 data read for stats)
    const sp_start = std.time.nanoTimestamp();
    for (0..BN_ITERS) |_| {
        for (0..C_BN) |c| {
            const r = single_pass_stats(bn_data, N_BN, C_BN, c);
            sink += r.mean + r.variance;
        }
    }
    const sp_end = std.time.nanoTimestamp();
    const sp_ns: u64 = @intCast(sp_end - sp_start);

    const bn_speedup = @as(f64, @floatFromInt(tp_ns)) / @as(f64, @floatFromInt(sp_ns));

    print("=== Benchmark 2: Single-pass vs two-pass BN stats [{d},{d}] x {d} iters ===\n", .{ N_BN, C_BN, BN_ITERS });
    print("Two-pass:    {d:.2} ms\n", .{@as(f64, @floatFromInt(tp_ns)) / 1e6});
    print("Single-pass: {d:.2} ms\n", .{@as(f64, @floatFromInt(sp_ns)) / 1e6});
    print("Speedup:     {d:.2}x\n", .{bn_speedup});

    if (bn_speedup < 1.5) {
        print("FAIL: single-pass speedup {d:.2}x < required 1.5x\n", .{bn_speedup});
        std.process.exit(1);
    }
    print("PASS\n", .{});

    // Prevent sink from being optimized away.
    if (sink == 0.0) {
        print("sink: {d}\n", .{sink});
    }
}
