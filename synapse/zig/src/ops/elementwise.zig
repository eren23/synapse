//! Vectorized element-wise operations, fused ops, broadcasting, and activation functions.
//! All operations use portable @Vector SIMD for automatic hardware acceleration
//! (NEON on aarch64, SSE/AVX on x86_64).

const std = @import("std");
const shape_mod = @import("shape");
const Shape = shape_mod.Shape;
const ShapeError = shape_mod.ShapeError;
const broadcastShapes = shape_mod.broadcastShapes;
const MAX_RANK = shape_mod.MAX_RANK;

const VEC_LEN = 4;
const F32x4 = @Vector(VEC_LEN, f32);

// ============================================================
// Basic element-wise operations
// ============================================================

/// Element-wise addition: dst[i] = a[i] + b[i]
pub fn add(dst: []f32, a: []const f32, b: []const f32) void {
    binaryOp(dst, a, b, .add);
}

/// Element-wise subtraction: dst[i] = a[i] - b[i]
pub fn sub(dst: []f32, a: []const f32, b: []const f32) void {
    binaryOp(dst, a, b, .sub);
}

/// Element-wise multiplication: dst[i] = a[i] * b[i]
pub fn mul(dst: []f32, a: []const f32, b: []const f32) void {
    binaryOp(dst, a, b, .mul);
}

/// Element-wise division: dst[i] = a[i] / b[i]
pub fn div(dst: []f32, a: []const f32, b: []const f32) void {
    binaryOp(dst, a, b, .div);
}

const BinOp = enum { add, sub, mul, div };

inline fn applyScalarOp(comptime op: BinOp, a: f32, b: f32) f32 {
    return switch (op) {
        .add => a + b,
        .sub => a - b,
        .mul => a * b,
        .div => a / b,
    };
}

inline fn applyVecOp(comptime op: BinOp, a: F32x4, b: F32x4) F32x4 {
    return switch (op) {
        .add => a + b,
        .sub => a - b,
        .mul => a * b,
        .div => a / b,
    };
}

fn binaryOp(dst: []f32, a: []const f32, b: []const f32, comptime op: BinOp) void {
    const len = dst.len;
    std.debug.assert(a.len >= len);
    std.debug.assert(b.len >= len);

    var i: usize = 0;
    while (i + VEC_LEN <= len) : (i += VEC_LEN) {
        const va: F32x4 = a[i..][0..VEC_LEN].*;
        const vb: F32x4 = b[i..][0..VEC_LEN].*;
        (dst.ptr + i)[0..VEC_LEN].* = applyVecOp(op, va, vb);
    }
    while (i < len) : (i += 1) {
        dst[i] = applyScalarOp(op, a[i], b[i]);
    }
}

// ============================================================
// Fused operations
// ============================================================

/// Fused add + ReLU: dst[i] = max(a[i] + b[i], 0)
/// Single pass over memory — ~2x faster than separate add then relu.
pub fn addRelu(dst: []f32, a: []const f32, b: []const f32) void {
    const len = dst.len;
    std.debug.assert(a.len >= len);
    std.debug.assert(b.len >= len);

    const zero: F32x4 = @splat(0.0);
    var i: usize = 0;
    while (i + VEC_LEN <= len) : (i += VEC_LEN) {
        const va: F32x4 = a[i..][0..VEC_LEN].*;
        const vb: F32x4 = b[i..][0..VEC_LEN].*;
        (dst.ptr + i)[0..VEC_LEN].* = @max(va + vb, zero);
    }
    while (i < len) : (i += 1) {
        dst[i] = @max(a[i] + b[i], 0.0);
    }
}

/// Fused multiply-add: dst[i] = a[i] * b[i] + c[i]
pub fn mulAdd(dst: []f32, a: []const f32, b: []const f32, c: []const f32) void {
    const len = dst.len;
    std.debug.assert(a.len >= len);
    std.debug.assert(b.len >= len);
    std.debug.assert(c.len >= len);

    var i: usize = 0;
    while (i + VEC_LEN <= len) : (i += VEC_LEN) {
        const va: F32x4 = a[i..][0..VEC_LEN].*;
        const vb: F32x4 = b[i..][0..VEC_LEN].*;
        const vc: F32x4 = c[i..][0..VEC_LEN].*;
        (dst.ptr + i)[0..VEC_LEN].* = @mulAdd(F32x4, va, vb, vc);
    }
    while (i < len) : (i += 1) {
        dst[i] = @mulAdd(f32, a[i], b[i], c[i]);
    }
}

// ============================================================
// Activation functions
// ============================================================

/// ReLU: dst[i] = max(src[i], 0)
pub fn relu(dst: []f32, src: []const f32) void {
    const len = dst.len;
    std.debug.assert(src.len >= len);

    const zero: F32x4 = @splat(0.0);
    var i: usize = 0;
    while (i + VEC_LEN <= len) : (i += VEC_LEN) {
        const v: F32x4 = src[i..][0..VEC_LEN].*;
        (dst.ptr + i)[0..VEC_LEN].* = @max(v, zero);
    }
    while (i < len) : (i += 1) {
        dst[i] = @max(src[i], 0.0);
    }
}

/// Leaky ReLU: dst[i] = src[i] > 0 ? src[i] : alpha * src[i]
pub fn leakyRelu(dst: []f32, src: []const f32, alpha: f32) void {
    const len = dst.len;
    std.debug.assert(src.len >= len);

    const alpha_v: F32x4 = @splat(alpha);
    const zero: F32x4 = @splat(0.0);
    var i: usize = 0;
    while (i + VEC_LEN <= len) : (i += VEC_LEN) {
        const v: F32x4 = src[i..][0..VEC_LEN].*;
        const scaled = v * alpha_v;
        const mask = v > zero;
        (dst.ptr + i)[0..VEC_LEN].* = @select(f32, mask, v, scaled);
    }
    while (i < len) : (i += 1) {
        dst[i] = if (src[i] > 0.0) src[i] else alpha * src[i];
    }
}

/// Sigmoid: dst[i] = 1 / (1 + exp(-src[i]))
pub fn sigmoid(dst: []f32, src: []const f32) void {
    const len = dst.len;
    std.debug.assert(src.len >= len);

    const one: F32x4 = @splat(1.0);
    var i: usize = 0;
    while (i + VEC_LEN <= len) : (i += VEC_LEN) {
        const v: F32x4 = src[i..][0..VEC_LEN].*;
        (dst.ptr + i)[0..VEC_LEN].* = one / (one + expVec(-v));
    }
    while (i < len) : (i += 1) {
        dst[i] = 1.0 / (1.0 + scalarExp(-src[i]));
    }
}

/// Tanh: dst[i] = tanh(src[i])
pub fn tanh(dst: []f32, src: []const f32) void {
    const len = dst.len;
    std.debug.assert(src.len >= len);

    var i: usize = 0;
    while (i + VEC_LEN <= len) : (i += VEC_LEN) {
        const v: F32x4 = src[i..][0..VEC_LEN].*;
        (dst.ptr + i)[0..VEC_LEN].* = tanhVec(v);
    }
    while (i < len) : (i += 1) {
        dst[i] = scalarTanh(src[i]);
    }
}

/// GELU (approximate): dst[i] = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
pub fn gelu(dst: []f32, src: []const f32) void {
    const len = dst.len;
    std.debug.assert(src.len >= len);

    const half: F32x4 = @splat(0.5);
    const one: F32x4 = @splat(1.0);
    const sqrt_2_over_pi: F32x4 = @splat(0.7978845608028654);
    const coeff: F32x4 = @splat(0.044715);

    var i: usize = 0;
    while (i + VEC_LEN <= len) : (i += VEC_LEN) {
        const x: F32x4 = src[i..][0..VEC_LEN].*;
        const x3 = x * x * x;
        const inner = sqrt_2_over_pi * (x + coeff * x3);
        (dst.ptr + i)[0..VEC_LEN].* = half * x * (one + tanhVec(inner));
    }
    while (i < len) : (i += 1) {
        const x = src[i];
        const inner = 0.7978845608028654 * (x + 0.044715 * x * x * x);
        dst[i] = 0.5 * x * (1.0 + scalarTanh(inner));
    }
}

// ============================================================
// Broadcasting operations
// ============================================================

pub const BroadcastError = ShapeError;

/// Broadcast element-wise addition.
pub fn broadcastAdd(dst: []f32, a: []const f32, a_shape: Shape, b: []const f32, b_shape: Shape) BroadcastError!void {
    return broadcastBinaryOp(dst, a, a_shape, b, b_shape, .add);
}

/// Broadcast element-wise subtraction.
pub fn broadcastSub(dst: []f32, a: []const f32, a_shape: Shape, b: []const f32, b_shape: Shape) BroadcastError!void {
    return broadcastBinaryOp(dst, a, a_shape, b, b_shape, .sub);
}

/// Broadcast element-wise multiplication.
pub fn broadcastMul(dst: []f32, a: []const f32, a_shape: Shape, b: []const f32, b_shape: Shape) BroadcastError!void {
    return broadcastBinaryOp(dst, a, a_shape, b, b_shape, .mul);
}

/// Broadcast element-wise division.
pub fn broadcastDiv(dst: []f32, a: []const f32, a_shape: Shape, b: []const f32, b_shape: Shape) BroadcastError!void {
    return broadcastBinaryOp(dst, a, a_shape, b, b_shape, .div);
}

fn broadcastBinaryOp(
    dst: []f32,
    a: []const f32,
    a_shape: Shape,
    b: []const f32,
    b_shape: Shape,
    comptime op: BinOp,
) BroadcastError!void {
    const out_shape = try broadcastShapes(a_shape, b_shape);
    const out_numel = out_shape.numel();
    std.debug.assert(dst.len >= out_numel);

    // Same shape: flat SIMD operation
    if (a_shape.eql(b_shape)) {
        binaryOp(dst[0..out_numel], a[0..out_numel], b[0..out_numel], op);
        return;
    }

    // Scalar broadcast
    if (a_shape.numel() == 1) {
        scalarLeftOp(dst[0..out_numel], a[0], b[0..out_numel], op);
        return;
    }
    if (b_shape.numel() == 1) {
        scalarRightOp(dst[0..out_numel], a[0..out_numel], b[0], op);
        return;
    }

    // Compute broadcast strides for optimized paths
    const a_bstrides = computeBroadcastStrides(a_shape, out_shape);
    const b_bstrides = computeBroadcastStrides(b_shape, out_shape);
    const ndim = out_shape.ndim;

    // 2D+ optimized row/column broadcast
    if (ndim >= 2) {
        const inner_dim = out_shape.dims[ndim - 1];
        const outer_count = out_numel / inner_dim;

        // Both have stride in inner dim: SIMD per row
        // Handles [1,N]+[M,N] and similar
        if (a_bstrides[ndim - 1] != 0 and b_bstrides[ndim - 1] != 0) {
            var row: usize = 0;
            while (row < outer_count) : (row += 1) {
                const dst_off = row * inner_dim;
                const a_off = computeRowOffset(row, out_shape, a_bstrides);
                const b_off = computeRowOffset(row, out_shape, b_bstrides);
                binaryOp(
                    dst[dst_off..][0..inner_dim],
                    a[a_off..][0..inner_dim],
                    b[b_off..][0..inner_dim],
                    op,
                );
            }
            return;
        }

        // A has stride 0 in inner dim: column broadcast from A
        // Handles [M,1]+[M,N]
        if (a_bstrides[ndim - 1] == 0 and b_bstrides[ndim - 1] != 0) {
            var row: usize = 0;
            while (row < outer_count) : (row += 1) {
                const dst_off = row * inner_dim;
                const a_off = computeRowOffset(row, out_shape, a_bstrides);
                const b_off = computeRowOffset(row, out_shape, b_bstrides);
                scalarLeftOp(
                    dst[dst_off..][0..inner_dim],
                    a[a_off],
                    b[b_off..][0..inner_dim],
                    op,
                );
            }
            return;
        }

        // B has stride 0 in inner dim: column broadcast from B
        // Handles [M,N]+[M,1]
        if (a_bstrides[ndim - 1] != 0 and b_bstrides[ndim - 1] == 0) {
            var row: usize = 0;
            while (row < outer_count) : (row += 1) {
                const dst_off = row * inner_dim;
                const a_off = computeRowOffset(row, out_shape, a_bstrides);
                const b_off = computeRowOffset(row, out_shape, b_bstrides);
                scalarRightOp(
                    dst[dst_off..][0..inner_dim],
                    a[a_off..][0..inner_dim],
                    b[b_off],
                    op,
                );
            }
            return;
        }
    }

    // General fallback: iterate with index computation
    generalBroadcastOp(dst, a, a_bstrides, b, b_bstrides, out_shape, op);
}

fn computeBroadcastStrides(in_shape: Shape, out_shape: Shape) [MAX_RANK]usize {
    var strides = [_]usize{0} ** MAX_RANK;
    if (in_shape.ndim == 0) return strides;

    const in_strides = in_shape.contiguousStrides();
    var i: usize = 0;
    while (i < in_shape.ndim) : (i += 1) {
        const out_idx = out_shape.ndim - 1 - i;
        const in_idx = in_shape.ndim - 1 - i;
        if (in_shape.dims[in_idx] == out_shape.dims[out_idx]) {
            strides[out_idx] = in_strides[in_idx];
        }
        // else: broadcast dimension, stride stays 0
    }
    return strides;
}

fn computeRowOffset(row: usize, out_shape: Shape, bstrides: [MAX_RANK]usize) usize {
    const ndim = out_shape.ndim;
    if (ndim <= 1) return 0;

    var remaining = row;
    var offset: usize = 0;
    var d: usize = 0;
    while (d < ndim - 1) : (d += 1) {
        var block: usize = 1;
        var dd: usize = d + 1;
        while (dd < ndim - 1) : (dd += 1) {
            block *= out_shape.dims[dd];
        }
        const idx = remaining / block;
        remaining %= block;
        offset += idx * bstrides[d];
    }
    return offset;
}

fn scalarLeftOp(dst: []f32, scalar: f32, data: []const f32, comptime op: BinOp) void {
    const len = dst.len;
    const s: F32x4 = @splat(scalar);
    var i: usize = 0;
    while (i + VEC_LEN <= len) : (i += VEC_LEN) {
        const v: F32x4 = data[i..][0..VEC_LEN].*;
        (dst.ptr + i)[0..VEC_LEN].* = applyVecOp(op, s, v);
    }
    while (i < len) : (i += 1) {
        dst[i] = applyScalarOp(op, scalar, data[i]);
    }
}

fn scalarRightOp(dst: []f32, data: []const f32, scalar: f32, comptime op: BinOp) void {
    const len = dst.len;
    const s: F32x4 = @splat(scalar);
    var i: usize = 0;
    while (i + VEC_LEN <= len) : (i += VEC_LEN) {
        const v: F32x4 = data[i..][0..VEC_LEN].*;
        (dst.ptr + i)[0..VEC_LEN].* = applyVecOp(op, v, s);
    }
    while (i < len) : (i += 1) {
        dst[i] = applyScalarOp(op, data[i], scalar);
    }
}

fn generalBroadcastOp(
    dst: []f32,
    a: []const f32,
    a_strides: [MAX_RANK]usize,
    b: []const f32,
    b_strides: [MAX_RANK]usize,
    out_shape: Shape,
    comptime op: BinOp,
) void {
    const ndim = out_shape.ndim;
    const total = out_shape.numel();
    var indices = [_]usize{0} ** MAX_RANK;

    for (0..total) |flat_idx| {
        var a_idx: usize = 0;
        var b_idx: usize = 0;
        for (0..ndim) |d| {
            a_idx += indices[d] * a_strides[d];
            b_idx += indices[d] * b_strides[d];
        }
        dst[flat_idx] = applyScalarOp(op, a[a_idx], b[b_idx]);

        // Increment indices (rightmost first)
        var d: usize = ndim;
        while (d > 0) {
            d -= 1;
            indices[d] += 1;
            if (indices[d] < out_shape.dims[d]) break;
            indices[d] = 0;
        }
    }
}

// ============================================================
// Vectorized math helpers (portable @Vector SIMD)
// ============================================================

/// Fast vectorized exp(x) using Cody-Waite range reduction + degree-5 polynomial.
inline fn expVec(x: F32x4) F32x4 {
    const ln2: F32x4 = @splat(0.6931471805599453);
    const ln2_inv: F32x4 = @splat(1.4426950408889634);
    const one: F32x4 = @splat(1.0);
    const max_val: F32x4 = @splat(88.0);
    const min_val: F32x4 = @splat(-88.0);

    const clamped = @max(@min(x, max_val), min_val);
    const n_float: F32x4 = @round(clamped * ln2_inv);
    const r: F32x4 = clamped - n_float * ln2;

    var p: F32x4 = @splat(@as(f32, 1.0 / 120.0));
    p = @mulAdd(F32x4, p, r, @as(F32x4, @splat(@as(f32, 1.0 / 24.0))));
    p = @mulAdd(F32x4, p, r, @as(F32x4, @splat(@as(f32, 1.0 / 6.0))));
    p = @mulAdd(F32x4, p, r, @as(F32x4, @splat(@as(f32, 0.5))));
    p = @mulAdd(F32x4, p, r, one);
    p = @mulAdd(F32x4, p, r, one);

    const n_int: @Vector(VEC_LEN, i32) = @intFromFloat(n_float);
    const biased: @Vector(VEC_LEN, i32) = n_int + @as(@Vector(VEC_LEN, i32), @splat(@as(i32, 127)));
    const biased_u: @Vector(VEC_LEN, u32) = @bitCast(biased);
    const shift: @Vector(VEC_LEN, u5) = @splat(23);
    const pow2: F32x4 = @bitCast(biased_u << shift);

    return p * pow2;
}

/// Vectorized tanh(x) = (exp(2x) - 1) / (exp(2x) + 1)
inline fn tanhVec(x: F32x4) F32x4 {
    const one: F32x4 = @splat(1.0);
    const max_val: F32x4 = @splat(10.0);
    const min_val: F32x4 = @splat(-10.0);
    const clamped = @max(@min(x, max_val), min_val);
    const exp2x = expVec(clamped + clamped);
    return (exp2x - one) / (exp2x + one);
}

/// Scalar exp for tail elements.
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

/// Scalar tanh for tail elements.
inline fn scalarTanh(x: f32) f32 {
    const clamped = @max(@min(x, @as(f32, 10.0)), @as(f32, -10.0));
    const exp2x = scalarExp(clamped + clamped);
    return (exp2x - 1.0) / (exp2x + 1.0);
}
