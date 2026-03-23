//! Benchmarks:
//! 1. SIMD LayerNorm vs scalar on [64, 128, 256], 100 iterations (target: >= 4x).
//! 2. Welford single-pass vs two-pass stats on [200K, 100] (target: >= 1.5x).

const std = @import("std");

// Benchmark 1 constants
const B1_OUTER: usize = 64 * 128; // 8192 instances
const B1_NORM: usize = 256; // elements per instance
const B1_TOTAL: usize = B1_OUTER * B1_NORM; // 2,097,152
const B1_WARMUP: usize = 5;
const B1_ITERS: usize = 100;

// Benchmark 2 constants (same layout as bench_reduce.zig batchnorm stats)
const B2_N: usize = 200_000;
const B2_C: usize = 100;
const B2_WARMUP: usize = 3;
const B2_ITERS: usize = 20;

const VEC_LEN = 4;
const F32x4 = @Vector(VEC_LEN, f32);

// ============================================================
// Benchmark 1: SIMD LayerNorm (Welford + normalize)
// ============================================================

fn simd_layernorm(out: []f32, inp: []const f32, gamma: []const f32, beta: []const f32, norm_size: usize, outer_size: usize, eps: f32) void {
    for (0..outer_size) |outer| {
        const base = outer * norm_size;
        const in_slice = inp[base .. base + norm_size];
        const out_slice = out[base .. base + norm_size];

        // SIMD sum+sum_sq accumulation (division-free), 2x unrolled
        var sum_a: F32x4 = @splat(0.0);
        var ssq_a: F32x4 = @splat(0.0);
        var sum_b: F32x4 = @splat(0.0);
        var ssq_b: F32x4 = @splat(0.0);
        var i: usize = 0;
        while (i + 8 <= norm_size) : (i += 8) {
            const x_a: F32x4 = in_slice[i..][0..VEC_LEN].*;
            const x_b: F32x4 = in_slice[i + 4 ..][0..VEC_LEN].*;
            sum_a += x_a;
            ssq_a += x_a * x_a;
            sum_b += x_b;
            ssq_b += x_b * x_b;
        }
        while (i + VEC_LEN <= norm_size) : (i += VEC_LEN) {
            const x: F32x4 = in_slice[i..][0..VEC_LEN].*;
            sum_a += x;
            ssq_a += x * x;
        }
        var total_sum: f32 = @reduce(.Add, sum_a + sum_b);
        var total_ssq: f32 = @reduce(.Add, ssq_a + ssq_b);
        while (i < norm_size) : (i += 1) {
            total_sum += in_slice[i];
            total_ssq += in_slice[i] * in_slice[i];
        }
        const norm_f: f32 = @as(f32, @floatFromInt(norm_size));
        const mean = total_sum / norm_f;
        const variance = @max(total_ssq / norm_f - mean * mean, 0.0);
        const inv_std = 1.0 / @sqrt(variance + eps);

        // SIMD normalize + affine, 2x unrolled
        const mean_v: F32x4 = @splat(mean);
        const inv_std_v: F32x4 = @splat(inv_std);
        var j: usize = 0;
        while (j + 8 <= norm_size) : (j += 8) {
            const x_a: F32x4 = in_slice[j..][0..VEC_LEN].*;
            const g_a: F32x4 = gamma[j..][0..VEC_LEN].*;
            const b_a: F32x4 = beta[j..][0..VEC_LEN].*;
            const x_b: F32x4 = in_slice[j + 4 ..][0..VEC_LEN].*;
            const g_b: F32x4 = gamma[j + 4 ..][0..VEC_LEN].*;
            const b_b: F32x4 = beta[j + 4 ..][0..VEC_LEN].*;
            out_slice[j..][0..VEC_LEN].* = g_a * ((x_a - mean_v) * inv_std_v) + b_a;
            out_slice[j + 4 ..][0..VEC_LEN].* = g_b * ((x_b - mean_v) * inv_std_v) + b_b;
        }
        while (j + VEC_LEN <= norm_size) : (j += VEC_LEN) {
            const x: F32x4 = in_slice[j..][0..VEC_LEN].*;
            const g: F32x4 = gamma[j..][0..VEC_LEN].*;
            const b: F32x4 = beta[j..][0..VEC_LEN].*;
            out_slice[j..][0..VEC_LEN].* = g * ((x - mean_v) * inv_std_v) + b;
        }
        while (j < norm_size) : (j += 1) {
            out_slice[j] = gamma[j] * (in_slice[j] - mean) * inv_std + beta[j];
        }
    }
}

/// Scalar LayerNorm with asm barriers to prevent auto-vectorization.
noinline fn scalar_layernorm(out: [*]f32, inp: [*]const f32, gamma: [*]const f32, beta: [*]const f32, norm_size: usize, outer_size: usize, eps: f32) void {
    for (0..outer_size) |outer| {
        const base = outer * norm_size;

        // Scalar Welford
        var mean: f32 = 0;
        var m2_val: f32 = 0;
        var k: usize = 0;
        while (k < norm_size) : (k += 1) {
            const x = inp[base + k];
            const count_f: f32 = @floatFromInt(k + 1);
            const delta = x - mean;
            mean += delta / count_f;
            const delta2 = x - mean;
            m2_val += delta * delta2;
            asm volatile ("" ::: .{ .memory = true });
        }
        const variance = m2_val / @as(f32, @floatFromInt(norm_size));
        const inv_std = 1.0 / @sqrt(variance + eps);

        var j: usize = 0;
        while (j < norm_size) : (j += 1) {
            out[base + j] = gamma[j] * (inp[base + j] - mean) * inv_std + beta[j];
            asm volatile ("" ::: .{ .memory = true });
        }
    }
}

// ============================================================
// Benchmark 2: Single-pass vs two-pass stats (strided access)
// ============================================================

/// Single-pass stats: sum + sum_sq in one loop (reads data once).
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

/// Two-pass stats: mean first, then variance (reads data twice).
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

    // ---- Allocate Benchmark 1 data ----
    const inp = try allocator.alloc(f32, B1_TOTAL);
    defer allocator.free(inp);
    const out = try allocator.alloc(f32, B1_TOTAL);
    defer allocator.free(out);
    const gamma = try allocator.alloc(f32, B1_NORM);
    defer allocator.free(gamma);
    const beta = try allocator.alloc(f32, B1_NORM);
    defer allocator.free(beta);

    fillData(inp, 42);
    for (gamma) |*g| g.* = 1.0;
    for (beta) |*b| b.* = 0.0;

    // ---- Warmup ----
    var sink: f32 = 0;
    for (0..B1_WARMUP) |_| {
        simd_layernorm(out, inp, gamma, beta, B1_NORM, B1_OUTER, 1e-5);
        sink += out[0];
        scalar_layernorm(out.ptr, inp.ptr, gamma.ptr, beta.ptr, B1_NORM, B1_OUTER, 1e-5);
        sink += out[0];
    }

    // ---- Benchmark 1: Scalar ----
    const scalar_start = std.time.nanoTimestamp();
    for (0..B1_ITERS) |_| {
        scalar_layernorm(out.ptr, inp.ptr, gamma.ptr, beta.ptr, B1_NORM, B1_OUTER, 1e-5);
        asm volatile ("" ::: .{ .memory = true });
    }
    const scalar_end = std.time.nanoTimestamp();
    const scalar_ns: u64 = @intCast(scalar_end - scalar_start);
    sink += out[0];

    // ---- Benchmark 1: SIMD ----
    const simd_start = std.time.nanoTimestamp();
    for (0..B1_ITERS) |_| {
        simd_layernorm(out, inp, gamma, beta, B1_NORM, B1_OUTER, 1e-5);
        asm volatile ("" ::: .{ .memory = true });
    }
    const simd_end = std.time.nanoTimestamp();
    const simd_ns: u64 = @intCast(simd_end - simd_start);
    sink += out[0];

    const simd_speedup = @as(f64, @floatFromInt(scalar_ns)) / @as(f64, @floatFromInt(simd_ns));

    print("=== Benchmark 1: SIMD vs scalar LayerNorm [64,128,256] x {d} iters ===\n", .{B1_ITERS});
    print("Scalar:  {d:.2} ms\n", .{@as(f64, @floatFromInt(scalar_ns)) / 1e6});
    print("SIMD:    {d:.2} ms\n", .{@as(f64, @floatFromInt(simd_ns)) / 1e6});
    print("Speedup: {d:.2}x\n", .{simd_speedup});

    if (simd_speedup < 4.0) {
        print("FAIL: SIMD speedup {d:.2}x < required 4.0x\n", .{simd_speedup});
        std.process.exit(1);
    }
    print("PASS\n\n", .{});

    // ---- Allocate Benchmark 2 data ----
    const bn_data = try allocator.alloc(f32, B2_N * B2_C);
    defer allocator.free(bn_data);
    for (0..B2_N * B2_C) |i| {
        bn_data[i] = @as(f32, @floatFromInt(i % 997)) * 0.01 - 5.0;
    }

    // ---- Warmup ----
    for (0..B2_WARMUP) |_| {
        for (0..B2_C) |c| {
            const sp = single_pass_stats(bn_data, B2_N, B2_C, c);
            sink += sp.mean;
            const tp = two_pass_stats(bn_data, B2_N, B2_C, c);
            sink += tp.mean;
        }
    }

    // ---- Benchmark 2: Two-pass ----
    const tp_start = std.time.nanoTimestamp();
    for (0..B2_ITERS) |_| {
        for (0..B2_C) |c| {
            const r = two_pass_stats(bn_data, B2_N, B2_C, c);
            sink += r.mean + r.variance;
        }
    }
    const tp_end = std.time.nanoTimestamp();
    const tp_ns: u64 = @intCast(tp_end - tp_start);

    // ---- Benchmark 2: Single-pass ----
    const sp_start = std.time.nanoTimestamp();
    for (0..B2_ITERS) |_| {
        for (0..B2_C) |c| {
            const r = single_pass_stats(bn_data, B2_N, B2_C, c);
            sink += r.mean + r.variance;
        }
    }
    const sp_end = std.time.nanoTimestamp();
    const sp_ns: u64 = @intCast(sp_end - sp_start);

    const wp_speedup = @as(f64, @floatFromInt(tp_ns)) / @as(f64, @floatFromInt(sp_ns));

    print("=== Benchmark 2: Single-pass vs two-pass stats [{d},{d}] x {d} iters ===\n", .{ B2_N, B2_C, B2_ITERS });
    print("Two-pass:    {d:.2} ms\n", .{@as(f64, @floatFromInt(tp_ns)) / 1e6});
    print("Single-pass: {d:.2} ms\n", .{@as(f64, @floatFromInt(sp_ns)) / 1e6});
    print("Speedup:     {d:.2}x\n", .{wp_speedup});

    if (wp_speedup < 1.5) {
        print("FAIL: single-pass speedup {d:.2}x < required 1.5x\n", .{wp_speedup});
        std.process.exit(1);
    }
    print("PASS\n", .{});

    // Prevent sink from being optimized away.
    if (sink == 0.0) {
        print("sink: {d}\n", .{sink});
    }
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
