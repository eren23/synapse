//! ARM NEON SIMD operations for f32x4 vectors.
//! Provides primitive vector operations and bulk array operations
//! using 128-bit (4-wide) float32 vectors. On aarch64 these map
//! directly to NEON instructions; on other architectures Zig's
//! @Vector still produces correct (auto-vectorized) code.

const std = @import("std");

pub const F32x4 = @Vector(4, f32);

// ============================================================
// Primitive f32x4 operations
// ============================================================

pub inline fn load(ptr: [*]const f32) F32x4 {
    return ptr[0..4].*;
}

pub inline fn store(ptr: [*]f32, v: F32x4) void {
    ptr[0..4].* = v;
}

pub inline fn splat(val: f32) F32x4 {
    return @splat(val);
}

pub inline fn add(a: F32x4, b: F32x4) F32x4 {
    return a + b;
}

pub inline fn mul(a: F32x4, b: F32x4) F32x4 {
    return a * b;
}

/// Fused multiply-add: a * b + c
pub inline fn fma(a: F32x4, b: F32x4, c: F32x4) F32x4 {
    return @mulAdd(F32x4, a, b, c);
}

/// Horizontal sum: reduce all 4 lanes to a single f32.
/// On aarch64, compiles to NEON faddp (pairwise add).
pub inline fn hsum(v: F32x4) f32 {
    return @reduce(.Add, v);
}

/// Horizontal max: reduce all 4 lanes to a single f32.
/// On aarch64, compiles to NEON fmaxp (pairwise max).
pub inline fn hmax(v: F32x4) f32 {
    return @reduce(.Max, v);
}

/// Fast vectorized exp(x) using Cephes-style range reduction + degree-6 polynomial.
/// Uses Cahan split for ln2 to minimize range-reduction error.
/// Handles NaN, +inf, -inf correctly.
pub inline fn expVec(x: F32x4) F32x4 {
    const log2e_v: F32x4 = @splat(1.44269504088896341);
    const ln2_c1: F32x4 = @splat(0.693359375); // Cephes C1, exact in f32
    const ln2_c2: F32x4 = @splat(-2.12194440e-4); // Cephes C2
    const clamp_hi: F32x4 = @splat(88.72283935546875);
    const clamp_lo: F32x4 = @splat(-87.3365478515625);
    const one: F32x4 = @splat(1.0);
    const zero: F32x4 = @splat(0.0);

    // Detect special values to avoid UB in @intFromFloat
    const nan_mask = x != x;
    const pos_inf_mask = x == @as(F32x4, @splat(std.math.inf(f32)));
    const neg_inf_mask = x == @as(F32x4, @splat(-std.math.inf(f32)));

    // Replace NaN/inf with 0 for safe computation
    var safe_x = @select(f32, nan_mask, zero, x);
    safe_x = @select(f32, pos_inf_mask, zero, safe_x);
    safe_x = @select(f32, neg_inf_mask, zero, safe_x);

    const clamped = @min(@max(safe_x, clamp_lo), clamp_hi);

    // Range reduction: n = round(x / ln2)
    const n_float = @round(clamped * log2e_v);

    // Reduced argument: r = x - n*ln2 (Cahan split for precision)
    const r = clamped - n_float * ln2_c1 - n_float * ln2_c2;

    // Degree-6 Horner: 1 + r(1 + r(1/2 + r(1/6 + r(1/24 + r(1/120 + r/720)))))
    var p: F32x4 = @splat(@as(f32, 1.0 / 720.0));
    p = @mulAdd(F32x4, p, r, @as(F32x4, @splat(@as(f32, 1.0 / 120.0))));
    p = @mulAdd(F32x4, p, r, @as(F32x4, @splat(@as(f32, 1.0 / 24.0))));
    p = @mulAdd(F32x4, p, r, @as(F32x4, @splat(@as(f32, 1.0 / 6.0))));
    p = @mulAdd(F32x4, p, r, @as(F32x4, @splat(@as(f32, 0.5))));
    p = @mulAdd(F32x4, p, r, one);
    p = @mulAdd(F32x4, p, r, one);

    // Reconstruct: exp(x) = 2^n * exp(r) via IEEE 754 exponent bits
    const n_int: @Vector(4, i32) = @intFromFloat(n_float);
    const biased: @Vector(4, i32) = n_int + @as(@Vector(4, i32), @splat(@as(i32, 127)));
    const biased_u: @Vector(4, u32) = @bitCast(biased);
    const shift: @Vector(4, u5) = @splat(23);
    const pow2: F32x4 = @bitCast(biased_u << shift);

    var result = p * pow2;

    // Restore correct outputs for special inputs
    result = @select(f32, nan_mask, @as(F32x4, @splat(std.math.nan(f32))), result);
    result = @select(f32, pos_inf_mask, @as(F32x4, @splat(std.math.inf(f32))), result);
    result = @select(f32, neg_inf_mask, zero, result);

    return result;
}

/// Vectorized tanh(x) = (exp(2x) - 1) / (exp(2x) + 1)
/// Handles NaN propagation and clamps to avoid inf/inf.
pub inline fn tanhVec(x: F32x4) F32x4 {
    const one: F32x4 = @splat(1.0);
    const max_val: F32x4 = @splat(10.0);
    const min_val: F32x4 = @splat(-10.0);
    const nan_mask = x != x;
    const safe_x = @select(f32, nan_mask, @as(F32x4, @splat(0.0)), x);
    const clamped = @max(@min(safe_x, max_val), min_val);
    const exp2x = expVec(clamped + clamped);
    var result = (exp2x - one) / (exp2x + one);
    result = @select(f32, nan_mask, @as(F32x4, @splat(std.math.nan(f32))), result);
    return result;
}

/// Vectorized sigmoid(x) = 1 / (1 + exp(-x))
pub inline fn sigmoidVec(x: F32x4) F32x4 {
    const one: F32x4 = @splat(1.0);
    return one / (one + expVec(-x));
}

// ============================================================
// Bulk operations on arrays (dispatch-compatible signatures)
// ============================================================

pub fn bulkAdd(dst: [*]f32, a: [*]const f32, b: [*]const f32, len: usize) void {
    var i: usize = 0;
    while (i + 4 <= len) : (i += 4) {
        store(dst + i, add(load(a + i), load(b + i)));
    }
    while (i < len) : (i += 1) {
        dst[i] = a[i] + b[i];
    }
}

pub fn bulkMul(dst: [*]f32, a: [*]const f32, b: [*]const f32, len: usize) void {
    var i: usize = 0;
    while (i + 4 <= len) : (i += 4) {
        store(dst + i, mul(load(a + i), load(b + i)));
    }
    while (i < len) : (i += 1) {
        dst[i] = a[i] * b[i];
    }
}

pub fn bulkFma(dst: [*]f32, a: [*]const f32, b: [*]const f32, c: [*]const f32, len: usize) void {
    var i: usize = 0;
    while (i + 4 <= len) : (i += 4) {
        store(dst + i, fma(load(a + i), load(b + i), load(c + i)));
    }
    while (i < len) : (i += 1) {
        dst[i] = @mulAdd(f32, a[i], b[i], c[i]);
    }
}

pub fn bulkExp(dst: [*]f32, src: [*]const f32, len: usize) void {
    var i: usize = 0;
    while (i + 4 <= len) : (i += 4) {
        store(dst + i, expVec(load(src + i)));
    }
    while (i < len) : (i += 1) {
        dst[i] = scalarExp(src[i]);
    }
}

pub fn bulkTanh(dst: [*]f32, src: [*]const f32, len: usize) void {
    var i: usize = 0;
    while (i + 4 <= len) : (i += 4) {
        store(dst + i, tanhVec(load(src + i)));
    }
    while (i < len) : (i += 1) {
        dst[i] = scalarTanh(src[i]);
    }
}

pub fn bulkSigmoid(dst: [*]f32, src: [*]const f32, len: usize) void {
    var i: usize = 0;
    while (i + 4 <= len) : (i += 4) {
        store(dst + i, sigmoidVec(load(src + i)));
    }
    while (i < len) : (i += 1) {
        dst[i] = scalarSigmoid(src[i]);
    }
}

pub fn bulkHsum(src: [*]const f32, len: usize) f32 {
    var acc: F32x4 = @splat(0.0);
    var i: usize = 0;
    while (i + 4 <= len) : (i += 4) {
        acc += load(src + i);
    }
    var sum = hsum(acc);
    while (i < len) : (i += 1) {
        sum += src[i];
    }
    return sum;
}

pub fn bulkHmax(src: [*]const f32, len: usize) f32 {
    if (len == 0) return -std.math.inf(f32);
    var acc: F32x4 = @splat(-std.math.inf(f32));
    var i: usize = 0;
    while (i + 4 <= len) : (i += 4) {
        acc = @max(acc, load(src + i));
    }
    var max_val = hmax(acc);
    while (i < len) : (i += 1) {
        if (src[i] > max_val) max_val = src[i];
    }
    return max_val;
}

// ============================================================
// Scalar helpers for remainder elements
// ============================================================

inline fn scalarExp(x: f32) f32 {
    if (std.math.isNan(x)) return std.math.nan(f32);
    if (x == std.math.inf(f32)) return std.math.inf(f32);
    if (x == -std.math.inf(f32)) return 0.0;

    const clamp_hi: f32 = 88.72283935546875;
    const clamp_lo: f32 = -87.3365478515625;
    const log2e: f32 = 1.44269504088896341;
    const ln2_c1: f32 = 0.693359375;
    const ln2_c2: f32 = -2.12194440e-4;

    const clamped = @max(@min(x, clamp_hi), clamp_lo);
    const n_float = @round(clamped * log2e);
    const r = clamped - n_float * ln2_c1 - n_float * ln2_c2;

    var p: f32 = 1.0 / 720.0;
    p = @mulAdd(f32, p, r, 1.0 / 120.0);
    p = @mulAdd(f32, p, r, 1.0 / 24.0);
    p = @mulAdd(f32, p, r, 1.0 / 6.0);
    p = @mulAdd(f32, p, r, 0.5);
    p = @mulAdd(f32, p, r, 1.0);
    p = @mulAdd(f32, p, r, 1.0);

    const n_int: i32 = @intFromFloat(n_float);
    const biased: u32 = @bitCast(n_int + @as(i32, 127));
    const pow2: f32 = @bitCast(biased << 23);
    return p * pow2;
}

inline fn scalarTanh(x: f32) f32 {
    if (std.math.isNan(x)) return std.math.nan(f32);
    const clamped = @max(@min(x, @as(f32, 10.0)), @as(f32, -10.0));
    const exp2x = scalarExp(clamped + clamped);
    return (exp2x - 1.0) / (exp2x + 1.0);
}

inline fn scalarSigmoid(x: f32) f32 {
    if (std.math.isNan(x)) return std.math.nan(f32);
    return 1.0 / (1.0 + scalarExp(-x));
}

// ============================================================
// Slice-based API (used by tests and benchmarks)
// ============================================================

pub fn vadd_f32(dst: []f32, a: []const f32, b: []const f32) void {
    const len = @min(dst.len, @min(a.len, b.len));
    bulkAdd(dst.ptr, a.ptr, b.ptr, len);
}

pub fn vmul_f32(dst: []f32, a: []const f32, b: []const f32) void {
    const len = @min(dst.len, @min(a.len, b.len));
    bulkMul(dst.ptr, a.ptr, b.ptr, len);
}

pub fn vfma_f32(dst: []f32, a: []const f32, b: []const f32, c: []const f32) void {
    const len = @min(dst.len, @min(a.len, @min(b.len, c.len)));
    bulkFma(dst.ptr, a.ptr, b.ptr, c.ptr, len);
}

pub fn vexp_f32(dst: []f32, src: []const f32) void {
    const len = @min(dst.len, src.len);
    bulkExp(dst.ptr, src.ptr, len);
}

pub fn vtanh_f32(dst: []f32, src: []const f32) void {
    const len = @min(dst.len, src.len);
    bulkTanh(dst.ptr, src.ptr, len);
}

pub fn vsigmoid_f32(dst: []f32, src: []const f32) void {
    const len = @min(dst.len, src.len);
    bulkSigmoid(dst.ptr, src.ptr, len);
}
