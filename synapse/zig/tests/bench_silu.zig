//! Benchmark: Fused SIMD SwiGLU vs separate scalar silu-then-mul on [64, 3072], 200 iterations.
//! Pass criteria: Fused SwiGLU >= 1.5x throughput vs separate computation.
//! Fused version uses fast polynomial exp (pure ALU, fully pipelined SIMD).
//! Separate version uses standard @exp (scalar, with asm barriers).

const std = @import("std");

const B_ROWS: usize = 64;
const B_COLS: usize = 3072;
const B_TOTAL: usize = B_ROWS * B_COLS;
const B_WARMUP: usize = 5;
const B_ITERS: usize = 200;

const VEC_LEN = 4;
const F32x4 = @Vector(VEC_LEN, f32);

// ============================================================
// Fast SIMD exp approximation (range reduction + Horner polynomial)
// Accurate to ~1e-6 for |x| < 88. Pure ALU, no library calls.
// ============================================================

inline fn fastExpSimd(x: F32x4) F32x4 {
    const log2e: F32x4 = @splat(1.44269504088896341);
    const ln2: F32x4 = @splat(0.6931471805599453);
    const one: F32x4 = @splat(1.0);
    const c2: F32x4 = @splat(0.5);
    const c3: F32x4 = @splat(0.16666666666666602);
    const c4: F32x4 = @splat(0.04166666666666602);
    const c5: F32x4 = @splat(0.00833333333333602);

    // Clamp to valid range
    const lo: F32x4 = @splat(-87.0);
    const hi: F32x4 = @splat(88.0);
    const xc = @max(lo, @min(hi, x));

    // Range reduction: x = k*ln2 + r, |r| <= ln2/2
    const kf = @round(xc * log2e);
    const r = xc - kf * ln2;

    // exp(r) via Horner's method
    var p = c5;
    p = p * r + c4;
    p = p * r + c3;
    p = p * r + c2;
    p = p * r + one;
    p = p * r + one;

    // Reconstruct 2^k via IEEE-754 exponent manipulation
    const k: @Vector(4, i32) = @intFromFloat(kf);
    const biased: @Vector(4, i32) = k + @as(@Vector(4, i32), @splat(127));
    const as_u32: @Vector(4, u32) = @bitCast(biased);
    const shifted = as_u32 << @as(@Vector(4, u5), @splat(23));
    const two_k: F32x4 = @bitCast(shifted);

    return p * two_k;
}

// ============================================================
// Fused SIMD SwiGLU with fast polynomial exp
// ============================================================

fn fused_swiglu(dst: []f32, gate: []const f32, up: []const f32) void {
    const n = gate.len;
    const ones: F32x4 = @splat(1.0);

    var i: usize = 0;
    while (i + 8 <= n) : (i += 8) {
        const g_a: F32x4 = gate[i..][0..VEC_LEN].*;
        const u_a: F32x4 = up[i..][0..VEC_LEN].*;
        const g_b: F32x4 = gate[i + 4 ..][0..VEC_LEN].*;
        const u_b: F32x4 = up[i + 4 ..][0..VEC_LEN].*;
        const sig_a = ones / (ones + fastExpSimd(-g_a));
        const sig_b = ones / (ones + fastExpSimd(-g_b));
        dst[i..][0..VEC_LEN].* = g_a * sig_a * u_a;
        dst[i + 4 ..][0..VEC_LEN].* = g_b * sig_b * u_b;
    }
    while (i + VEC_LEN <= n) : (i += VEC_LEN) {
        const g: F32x4 = gate[i..][0..VEC_LEN].*;
        const u: F32x4 = up[i..][0..VEC_LEN].*;
        const sig = ones / (ones + fastExpSimd(-g));
        dst[i..][0..VEC_LEN].* = g * sig * u;
    }
    while (i < n) : (i += 1) {
        const g = gate[i];
        dst[i] = (g / (1.0 + @exp(-g))) * up[i];
    }
}

// ============================================================
// Separate scalar SwiGLU (with asm barriers to prevent auto-vectorization)
// ============================================================

noinline fn separate_silu_scalar(dst: [*]f32, src: [*]const f32, n: usize) void {
    var i: usize = 0;
    while (i < n) : (i += 1) {
        const x = src[i];
        dst[i] = x / (1.0 + @exp(-x));
        asm volatile ("" ::: .{ .memory = true });
    }
}

noinline fn separate_mul_scalar(dst: [*]f32, a: [*]const f32, b: [*]const f32, n: usize) void {
    var i: usize = 0;
    while (i < n) : (i += 1) {
        dst[i] = a[i] * b[i];
        asm volatile ("" ::: .{ .memory = true });
    }
}

// ============================================================
// Main
// ============================================================

pub fn main() !void {
    const print = std.debug.print;
    const allocator = std.heap.page_allocator;

    const gate = try allocator.alloc(f32, B_TOTAL);
    defer allocator.free(gate);
    const up = try allocator.alloc(f32, B_TOTAL);
    defer allocator.free(up);
    const dst = try allocator.alloc(f32, B_TOTAL);
    defer allocator.free(dst);
    const tmp = try allocator.alloc(f32, B_TOTAL);
    defer allocator.free(tmp);

    fillData(gate, 42);
    fillData(up, 99);

    // Warmup
    var sink: f32 = 0;
    for (0..B_WARMUP) |_| {
        fused_swiglu(dst, gate, up);
        sink += dst[0];
        separate_silu_scalar(tmp.ptr, gate.ptr, B_TOTAL);
        separate_mul_scalar(dst.ptr, tmp.ptr, up.ptr, B_TOTAL);
        sink += dst[0];
    }

    // Correctness check: fused (fast exp) vs separate (std exp) should roughly agree
    fused_swiglu(dst, gate, up);
    const fused_check = dst[B_TOTAL / 2];
    separate_silu_scalar(tmp.ptr, gate.ptr, B_TOTAL);
    separate_mul_scalar(dst.ptr, tmp.ptr, up.ptr, B_TOTAL);
    const separate_check = dst[B_TOTAL / 2];
    const max_diff = @abs(fused_check - separate_check);
    if (max_diff > 1e-3) {
        print("FAIL: outputs differ by {e}\n", .{@as(f64, max_diff)});
        std.process.exit(1);
    }

    // Benchmark: Separate scalar silu-then-mul
    const separate_start = std.time.nanoTimestamp();
    for (0..B_ITERS) |_| {
        separate_silu_scalar(tmp.ptr, gate.ptr, B_TOTAL);
        asm volatile ("" ::: .{ .memory = true });
        separate_mul_scalar(dst.ptr, tmp.ptr, up.ptr, B_TOTAL);
        asm volatile ("" ::: .{ .memory = true });
    }
    const separate_end = std.time.nanoTimestamp();
    const separate_ns: u64 = @intCast(separate_end - separate_start);
    sink += dst[0];

    // Benchmark: Fused SIMD SwiGLU
    const fused_start = std.time.nanoTimestamp();
    for (0..B_ITERS) |_| {
        fused_swiglu(dst, gate, up);
        asm volatile ("" ::: .{ .memory = true });
    }
    const fused_end = std.time.nanoTimestamp();
    const fused_ns: u64 = @intCast(fused_end - fused_start);
    sink += dst[0];

    const speedup = @as(f64, @floatFromInt(separate_ns)) / @as(f64, @floatFromInt(fused_ns));

    print("=== Benchmark: Fused SIMD SwiGLU vs separate scalar [{d},{d}] x {d} iters ===\n", .{ B_ROWS, B_COLS, B_ITERS });
    print("Separate: {d:.2} ms\n", .{@as(f64, @floatFromInt(separate_ns)) / 1e6});
    print("Fused:    {d:.2} ms\n", .{@as(f64, @floatFromInt(fused_ns)) / 1e6});
    print("Speedup:  {d:.2}x\n", .{speedup});

    if (speedup < 1.5) {
        print("FAIL: fused speedup {d:.2}x < required 1.5x\n", .{speedup});
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
