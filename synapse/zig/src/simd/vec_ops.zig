//! Public vectorized operations API.
//! All operations dispatch through the runtime-detected optimal backend
//! (AVX2, NEON, or scalar fallback).

const std = @import("std");
const dispatch = @import("dispatch");

pub const Backend = dispatch.Backend;

/// Returns the active SIMD backend.
pub fn activeBackend() Backend {
    return dispatch.getOps().backend;
}

/// Element-wise addition: dst[i] = a[i] + b[i]
pub fn add(dst: []f32, a: []const f32, b: []const f32) void {
    const len = dst.len;
    std.debug.assert(a.len >= len);
    std.debug.assert(b.len >= len);
    dispatch.getOps().addFn(dst.ptr, a.ptr, b.ptr, len);
}

/// Element-wise multiplication: dst[i] = a[i] * b[i]
pub fn mul(dst: []f32, a: []const f32, b: []const f32) void {
    const len = dst.len;
    std.debug.assert(a.len >= len);
    std.debug.assert(b.len >= len);
    dispatch.getOps().mulFn(dst.ptr, a.ptr, b.ptr, len);
}

/// Fused multiply-add: dst[i] = a[i] * b[i] + c[i]
pub fn fma(dst: []f32, a: []const f32, b: []const f32, c: []const f32) void {
    const len = dst.len;
    std.debug.assert(a.len >= len);
    std.debug.assert(b.len >= len);
    std.debug.assert(c.len >= len);
    dispatch.getOps().fmaFn(dst.ptr, a.ptr, b.ptr, c.ptr, len);
}

/// Element-wise exp: dst[i] = exp(src[i])
pub fn exp(dst: []f32, src: []const f32) void {
    const len = dst.len;
    std.debug.assert(src.len >= len);
    dispatch.getOps().expFn(dst.ptr, src.ptr, len);
}

/// Element-wise tanh: dst[i] = tanh(src[i])
pub fn tanh(dst: []f32, src: []const f32) void {
    const len = dst.len;
    std.debug.assert(src.len >= len);
    dispatch.getOps().tanhFn(dst.ptr, src.ptr, len);
}

/// Element-wise sigmoid: dst[i] = 1 / (1 + exp(-src[i]))
pub fn sigmoid(dst: []f32, src: []const f32) void {
    const len = dst.len;
    std.debug.assert(src.len >= len);
    dispatch.getOps().sigmoidFn(dst.ptr, src.ptr, len);
}

/// Horizontal sum: returns sum of all elements.
pub fn hsum(src: []const f32) f32 {
    return dispatch.getOps().hsumFn(src.ptr, src.len);
}

/// Horizontal max: returns max of all elements.
pub fn hmax(src: []const f32) f32 {
    return dispatch.getOps().hmaxFn(src.ptr, src.len);
}
