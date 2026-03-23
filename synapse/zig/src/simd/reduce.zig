//! Horizontal reduction operations with NEON-optimized paths.
//! Provides horizontal sum and horizontal max using NEON pairwise
//! instructions on aarch64, with scalar fallback on other architectures.

const std = @import("std");
const builtin = @import("builtin");

const F32x4 = @Vector(4, f32);
const F32x2 = @Vector(2, f32);

// ============================================================
// Single-vector pairwise reductions (NEON primitives)
// ============================================================

/// Pairwise horizontal sum of a f32x4 vector.
/// On aarch64, maps to two NEON faddp instructions.
/// [a, b, c, d] -> a + b + c + d
pub inline fn pairwiseSum(v: F32x4) f32 {
    // Pairwise add: extract odd/even lanes, add pairs, then sum
    const lo: F32x2 = @shuffle(f32, v, undefined, [2]i32{ 0, 2 });
    const hi: F32x2 = @shuffle(f32, v, undefined, [2]i32{ 1, 3 });
    const pair_sums = lo + hi; // [a+b, c+d]
    return pair_sums[0] + pair_sums[1];
}

/// Pairwise horizontal max of a f32x4 vector.
/// On aarch64, maps to two NEON fmaxp instructions.
/// [a, b, c, d] -> max(a, b, c, d)
pub inline fn pairwiseMax(v: F32x4) f32 {
    const lo: F32x2 = @shuffle(f32, v, undefined, [2]i32{ 0, 2 });
    const hi: F32x2 = @shuffle(f32, v, undefined, [2]i32{ 1, 3 });
    const pair_maxs = @max(lo, hi); // [max(a,b), max(c,d)]
    return @max(pair_maxs[0], pair_maxs[1]);
}

// ============================================================
// Bulk horizontal reductions with architecture dispatch
// ============================================================

/// Horizontal sum over an array.
/// Uses NEON f32x4 accumulation and pairwise reduction on aarch64.
/// Handles tail elements (non-multiple-of-4 lengths).
pub fn horizontalSum(src: [*]const f32, len: usize) f32 {
    if (comptime builtin.cpu.arch == .aarch64) {
        return neonHorizontalSum(src, len);
    }
    return scalarHorizontalSum(src, len);
}

/// Horizontal max over an array.
/// Uses NEON f32x4 accumulation and pairwise reduction on aarch64.
/// Handles tail elements (non-multiple-of-4 lengths).
pub fn horizontalMax(src: [*]const f32, len: usize) f32 {
    if (comptime builtin.cpu.arch == .aarch64) {
        return neonHorizontalMax(src, len);
    }
    return scalarHorizontalMax(src, len);
}

// ============================================================
// NEON-optimized implementations
// ============================================================

fn neonHorizontalSum(src: [*]const f32, len: usize) f32 {
    var acc: F32x4 = @splat(0.0);
    var i: usize = 0;

    // Accumulate 4 elements at a time into NEON register
    while (i + 4 <= len) : (i += 4) {
        const v: F32x4 = (src + i)[0..4].*;
        acc += v;
    }

    // Pairwise reduce the accumulator
    var sum = pairwiseSum(acc);

    // Handle tail elements
    while (i < len) : (i += 1) {
        sum += src[i];
    }
    return sum;
}

fn neonHorizontalMax(src: [*]const f32, len: usize) f32 {
    if (len == 0) return -std.math.inf(f32);

    var acc: F32x4 = @splat(-std.math.inf(f32));
    var i: usize = 0;

    // Accumulate max of 4 elements at a time
    while (i + 4 <= len) : (i += 4) {
        const v: F32x4 = (src + i)[0..4].*;
        acc = @max(acc, v);
    }

    // Pairwise reduce the accumulator
    var max_val = pairwiseMax(acc);

    // Handle tail elements
    while (i < len) : (i += 1) {
        if (src[i] > max_val) max_val = src[i];
    }
    return max_val;
}

// ============================================================
// Scalar fallback implementations
// ============================================================

fn scalarHorizontalSum(src: [*]const f32, len: usize) f32 {
    var sum: f32 = 0;
    var i: usize = 0;
    while (i < len) : (i += 1) {
        sum += src[i];
    }
    return sum;
}

fn scalarHorizontalMax(src: [*]const f32, len: usize) f32 {
    if (len == 0) return -std.math.inf(f32);
    var max_val: f32 = src[0];
    var i: usize = 1;
    while (i < len) : (i += 1) {
        if (src[i] > max_val) max_val = src[i];
    }
    return max_val;
}

// ============================================================
// Slice-based API (used by tests)
// ============================================================

pub fn horizontal_sum_f32(src: []const f32) f32 {
    return horizontalSum(src.ptr, src.len);
}

pub fn horizontal_max_f32(src: []const f32) f32 {
    return horizontalMax(src.ptr, src.len);
}
