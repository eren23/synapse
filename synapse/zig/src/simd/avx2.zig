//! AVX2 SIMD operations for f32x8 vectors.
//! Provides primitive vector operations and bulk array operations
//! using 256-bit (8-wide) float32 vectors.

const std = @import("std");

pub const F32x8 = @Vector(8, f32);

// ============================================================
// Primitive f32x8 operations
// ============================================================

pub inline fn load(ptr: [*]const f32) F32x8 {
    return ptr[0..8].*;
}

pub inline fn store(ptr: [*]f32, v: F32x8) void {
    ptr[0..8].* = v;
}

pub inline fn splat(val: f32) F32x8 {
    return @splat(val);
}

pub inline fn add(a: F32x8, b: F32x8) F32x8 {
    return a + b;
}

pub inline fn mul(a: F32x8, b: F32x8) F32x8 {
    return a * b;
}

/// Fused multiply-add: a * b + c
pub inline fn fma(a: F32x8, b: F32x8, c: F32x8) F32x8 {
    return @mulAdd(F32x8, a, b, c);
}

/// Horizontal sum: reduce all 8 lanes to a single f32.
pub inline fn hsum(v: F32x8) f32 {
    return @reduce(.Add, v);
}

/// Horizontal max: reduce all 8 lanes to a single f32.
pub inline fn hmax(v: F32x8) f32 {
    return @reduce(.Max, v);
}

/// Fast vectorized exp(x) using Cody-Waite range reduction + degree-5 polynomial.
/// Accurate to ~2.4e-6 absolute error for f32.
pub inline fn expVec(x: F32x8) F32x8 {
    const ln2: F32x8 = @splat(0.6931471805599453);
    const ln2_inv: F32x8 = @splat(1.4426950408889634);
    const one: F32x8 = @splat(1.0);

    // Clamp to prevent overflow/underflow
    const max_val: F32x8 = @splat(88.0);
    const min_val: F32x8 = @splat(-88.0);
    const clamped = @max(@min(x, max_val), min_val);

    // Range reduction: x = n*ln2 + r, where |r| <= ln2/2
    const n_float: F32x8 = @round(clamped * ln2_inv);
    const r: F32x8 = clamped - n_float * ln2;

    // Horner's form: 1 + r(1 + r(1/2 + r(1/6 + r(1/24 + r/120))))
    var p: F32x8 = @splat(@as(f32, 1.0 / 120.0));
    p = @mulAdd(F32x8, p, r, @as(F32x8, @splat(@as(f32, 1.0 / 24.0))));
    p = @mulAdd(F32x8, p, r, @as(F32x8, @splat(@as(f32, 1.0 / 6.0))));
    p = @mulAdd(F32x8, p, r, @as(F32x8, @splat(@as(f32, 0.5))));
    p = @mulAdd(F32x8, p, r, one);
    p = @mulAdd(F32x8, p, r, one);

    // Reconstruct: exp(x) = 2^n * exp(r)
    const n_int: @Vector(8, i32) = @intFromFloat(n_float);
    const biased: @Vector(8, i32) = n_int + @as(@Vector(8, i32), @splat(@as(i32, 127)));
    const biased_u: @Vector(8, u32) = @bitCast(biased);
    const shift: @Vector(8, u5) = @splat(23);
    const pow2: F32x8 = @bitCast(biased_u << shift);

    return p * pow2;
}

/// Vectorized tanh(x) = (exp(2x) - 1) / (exp(2x) + 1)
pub inline fn tanhVec(x: F32x8) F32x8 {
    const one: F32x8 = @splat(1.0);
    const max_val: F32x8 = @splat(10.0);
    const min_val: F32x8 = @splat(-10.0);
    const clamped = @max(@min(x, max_val), min_val);
    const exp2x = expVec(clamped + clamped);
    return (exp2x - one) / (exp2x + one);
}

/// Vectorized sigmoid(x) = 1 / (1 + exp(-x))
pub inline fn sigmoidVec(x: F32x8) F32x8 {
    const one: F32x8 = @splat(1.0);
    return one / (one + expVec(-x));
}

// ============================================================
// Bulk operations on arrays
// ============================================================

pub fn bulkAdd(dst: [*]f32, a: [*]const f32, b: [*]const f32, len: usize) void {
    var i: usize = 0;
    while (i + 8 <= len) : (i += 8) {
        store(dst + i, add(load(a + i), load(b + i)));
    }
    while (i < len) : (i += 1) {
        dst[i] = a[i] + b[i];
    }
}

pub fn bulkMul(dst: [*]f32, a: [*]const f32, b: [*]const f32, len: usize) void {
    var i: usize = 0;
    while (i + 8 <= len) : (i += 8) {
        store(dst + i, mul(load(a + i), load(b + i)));
    }
    while (i < len) : (i += 1) {
        dst[i] = a[i] * b[i];
    }
}

pub fn bulkFma(dst: [*]f32, a: [*]const f32, b: [*]const f32, c: [*]const f32, len: usize) void {
    var i: usize = 0;
    while (i + 8 <= len) : (i += 8) {
        store(dst + i, fma(load(a + i), load(b + i), load(c + i)));
    }
    while (i < len) : (i += 1) {
        dst[i] = @mulAdd(f32, a[i], b[i], c[i]);
    }
}

pub fn bulkExp(dst: [*]f32, src: [*]const f32, len: usize) void {
    var i: usize = 0;
    while (i + 8 <= len) : (i += 8) {
        store(dst + i, expVec(load(src + i)));
    }
    while (i < len) : (i += 1) {
        dst[i] = scalarExp(src[i]);
    }
}

pub fn bulkTanh(dst: [*]f32, src: [*]const f32, len: usize) void {
    var i: usize = 0;
    while (i + 8 <= len) : (i += 8) {
        store(dst + i, tanhVec(load(src + i)));
    }
    while (i < len) : (i += 1) {
        dst[i] = scalarTanh(src[i]);
    }
}

pub fn bulkSigmoid(dst: [*]f32, src: [*]const f32, len: usize) void {
    var i: usize = 0;
    while (i + 8 <= len) : (i += 8) {
        store(dst + i, sigmoidVec(load(src + i)));
    }
    while (i < len) : (i += 1) {
        dst[i] = scalarSigmoid(src[i]);
    }
}

pub fn bulkHsum(src: [*]const f32, len: usize) f32 {
    var acc: F32x8 = @splat(0.0);
    var i: usize = 0;
    while (i + 8 <= len) : (i += 8) {
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
    var acc: F32x8 = @splat(-std.math.inf(f32));
    var i: usize = 0;
    while (i + 8 <= len) : (i += 8) {
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
    const clamped = @max(@min(x, @as(f32, 88.0)), @as(f32, -88.0));
    const ln2: f32 = 0.6931471805599453;
    const ln2_inv: f32 = 1.4426950408889634;
    const n_float = @round(clamped * ln2_inv);
    const r = clamped - n_float * ln2;

    var p: f32 = 1.0 / 120.0;
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
    const clamped = @max(@min(x, @as(f32, 10.0)), @as(f32, -10.0));
    const exp2x = scalarExp(clamped + clamped);
    return (exp2x - 1.0) / (exp2x + 1.0);
}

inline fn scalarSigmoid(x: f32) f32 {
    return 1.0 / (1.0 + scalarExp(-x));
}
