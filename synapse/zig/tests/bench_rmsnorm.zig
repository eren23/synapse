//! Benchmark: SIMD RMSNorm vs scalar on [64, 1024], 200 iterations.
//! Pass criteria: SIMD >= 4x throughput vs scalar.

const std = @import("std");

const B_OUTER: usize = 64;
const B_NORM: usize = 1024;
const B_TOTAL: usize = B_OUTER * B_NORM;
const B_WARMUP: usize = 5;
const B_ITERS: usize = 200;

const VEC_LEN = 4;
const F32x4 = @Vector(VEC_LEN, f32);

// ============================================================
// SIMD RMSNorm
// ============================================================

fn simd_rmsnorm(out: []f32, inp: []const f32, gamma: []const f32, norm_size: usize, outer_size: usize, eps: f32) void {
    for (0..outer_size) |outer| {
        const base = outer * norm_size;
        const in_slice = inp[base .. base + norm_size];
        const out_slice = out[base .. base + norm_size];

        // SIMD sum_sq accumulation, 2x unrolled
        var ssq_a: F32x4 = @splat(0.0);
        var ssq_b: F32x4 = @splat(0.0);
        var i: usize = 0;
        while (i + 8 <= norm_size) : (i += 8) {
            const x_a: F32x4 = in_slice[i..][0..VEC_LEN].*;
            const x_b: F32x4 = in_slice[i + 4 ..][0..VEC_LEN].*;
            ssq_a += x_a * x_a;
            ssq_b += x_b * x_b;
        }
        while (i + VEC_LEN <= norm_size) : (i += VEC_LEN) {
            const x: F32x4 = in_slice[i..][0..VEC_LEN].*;
            ssq_a += x * x;
        }
        var total_ssq: f32 = @reduce(.Add, ssq_a + ssq_b);
        while (i < norm_size) : (i += 1) {
            total_ssq += in_slice[i] * in_slice[i];
        }
        const norm_f: f32 = @as(f32, @floatFromInt(norm_size));
        const rms_inv = 1.0 / @sqrt(total_ssq / norm_f + eps);

        // SIMD scale, 2x unrolled
        const inv_v: F32x4 = @splat(rms_inv);
        var j: usize = 0;
        while (j + 8 <= norm_size) : (j += 8) {
            const x_a: F32x4 = in_slice[j..][0..VEC_LEN].*;
            const g_a: F32x4 = gamma[j..][0..VEC_LEN].*;
            const x_b: F32x4 = in_slice[j + 4 ..][0..VEC_LEN].*;
            const g_b: F32x4 = gamma[j + 4 ..][0..VEC_LEN].*;
            out_slice[j..][0..VEC_LEN].* = g_a * x_a * inv_v;
            out_slice[j + 4 ..][0..VEC_LEN].* = g_b * x_b * inv_v;
        }
        while (j + VEC_LEN <= norm_size) : (j += VEC_LEN) {
            const x: F32x4 = in_slice[j..][0..VEC_LEN].*;
            const g: F32x4 = gamma[j..][0..VEC_LEN].*;
            out_slice[j..][0..VEC_LEN].* = g * x * inv_v;
        }
        while (j < norm_size) : (j += 1) {
            out_slice[j] = gamma[j] * in_slice[j] * rms_inv;
        }
    }
}

// ============================================================
// Scalar RMSNorm (with asm barriers to prevent auto-vectorization)
// ============================================================

noinline fn scalar_rmsnorm(out: [*]f32, inp: [*]const f32, gamma: [*]const f32, norm_size: usize, outer_size: usize, eps: f32) void {
    for (0..outer_size) |outer| {
        const base = outer * norm_size;

        // Scalar sum of squares
        var sum_sq: f32 = 0;
        var k: usize = 0;
        while (k < norm_size) : (k += 1) {
            const x = inp[base + k];
            sum_sq += x * x;
            asm volatile ("" ::: .{ .memory = true });
        }
        const norm_f: f32 = @floatFromInt(norm_size);
        const rms_inv = 1.0 / @sqrt(sum_sq / norm_f + eps);

        var j: usize = 0;
        while (j < norm_size) : (j += 1) {
            out[base + j] = gamma[j] * inp[base + j] * rms_inv;
            asm volatile ("" ::: .{ .memory = true });
        }
    }
}

// ============================================================
// Main
// ============================================================

pub fn main() !void {
    const print = std.debug.print;
    const allocator = std.heap.page_allocator;

    const inp = try allocator.alloc(f32, B_TOTAL);
    defer allocator.free(inp);
    const out = try allocator.alloc(f32, B_TOTAL);
    defer allocator.free(out);
    const gamma = try allocator.alloc(f32, B_NORM);
    defer allocator.free(gamma);

    fillData(inp, 42);
    for (gamma) |*g| g.* = 1.0;

    // Warmup
    var sink: f32 = 0;
    for (0..B_WARMUP) |_| {
        simd_rmsnorm(out, inp, gamma, B_NORM, B_OUTER, 1e-5);
        sink += out[0];
        scalar_rmsnorm(out.ptr, inp.ptr, gamma.ptr, B_NORM, B_OUTER, 1e-5);
        sink += out[0];
    }

    // Correctness check
    simd_rmsnorm(out, inp, gamma, B_NORM, B_OUTER, 1e-5);
    const simd_check = out[B_TOTAL / 2];
    scalar_rmsnorm(out.ptr, inp.ptr, gamma.ptr, B_NORM, B_OUTER, 1e-5);
    const scalar_check = out[B_TOTAL / 2];
    const max_diff = @abs(simd_check - scalar_check);
    if (max_diff > 1e-4) {
        print("FAIL: outputs differ by {e}\n", .{@as(f64, max_diff)});
        std.process.exit(1);
    }

    // Benchmark: Scalar
    const scalar_start = std.time.nanoTimestamp();
    for (0..B_ITERS) |_| {
        scalar_rmsnorm(out.ptr, inp.ptr, gamma.ptr, B_NORM, B_OUTER, 1e-5);
        asm volatile ("" ::: .{ .memory = true });
    }
    const scalar_end = std.time.nanoTimestamp();
    const scalar_ns: u64 = @intCast(scalar_end - scalar_start);
    sink += out[0];

    // Benchmark: SIMD
    const simd_start = std.time.nanoTimestamp();
    for (0..B_ITERS) |_| {
        simd_rmsnorm(out, inp, gamma, B_NORM, B_OUTER, 1e-5);
        asm volatile ("" ::: .{ .memory = true });
    }
    const simd_end = std.time.nanoTimestamp();
    const simd_ns: u64 = @intCast(simd_end - simd_start);
    sink += out[0];

    const speedup = @as(f64, @floatFromInt(scalar_ns)) / @as(f64, @floatFromInt(simd_ns));

    print("=== Benchmark: SIMD vs scalar RMSNorm [{d},{d}] x {d} iters ===\n", .{ B_OUTER, B_NORM, B_ITERS });
    print("Scalar:  {d:.2} ms\n", .{@as(f64, @floatFromInt(scalar_ns)) / 1e6});
    print("SIMD:    {d:.2} ms\n", .{@as(f64, @floatFromInt(simd_ns)) / 1e6});
    print("Speedup: {d:.2}x\n", .{speedup});

    if (speedup < 4.0) {
        print("FAIL: SIMD speedup {d:.2}x < required 4.0x\n", .{speedup});
        std.process.exit(1);
    }
    print("PASS\n", .{});

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
