//! SiLU activation and fused SwiGLU kernel with SIMD vectorization.
//! SiLU(x) = x * sigmoid(x) = x / (1 + exp(-x))
//! SwiGLU(gate, up) = SiLU(gate) * up
//! Two SwiGLU implementations: fused (primary), separate (benchmark baseline).

const std = @import("std");

const VEC_LEN = 4;
const F32x4 = @Vector(VEC_LEN, f32);

// ============================================================
// Public API
// ============================================================

/// SIMD-vectorized SiLU activation: dst[i] = src[i] / (1 + exp(-src[i]))
/// Single pass, 2x-unrolled.
pub fn silu(dst: []f32, src: []const f32) void {
    std.debug.assert(dst.len == src.len);
    const n = src.len;
    const ones: F32x4 = @splat(1.0);

    var i: usize = 0;
    // 2x unrolled: 8 elements per iteration
    while (i + 8 <= n) : (i += 8) {
        const x_a: F32x4 = src[i..][0..VEC_LEN].*;
        const x_b: F32x4 = src[i + 4 ..][0..VEC_LEN].*;
        const sig_a = ones / (ones + @exp(-x_a));
        const sig_b = ones / (ones + @exp(-x_b));
        dst[i..][0..VEC_LEN].* = x_a * sig_a;
        dst[i + 4 ..][0..VEC_LEN].* = x_b * sig_b;
    }
    while (i + VEC_LEN <= n) : (i += VEC_LEN) {
        const x: F32x4 = src[i..][0..VEC_LEN].*;
        const sig = ones / (ones + @exp(-x));
        dst[i..][0..VEC_LEN].* = x * sig;
    }
    // Scalar tail
    while (i < n) : (i += 1) {
        const x = src[i];
        dst[i] = x / (1.0 + @exp(-x));
    }
}

/// Scalar reference SiLU (correctness baseline).
pub fn siluScalar(dst: []f32, src: []const f32) void {
    std.debug.assert(dst.len == src.len);
    for (src, dst) |x, *d| {
        d.* = x / (1.0 + @exp(-x));
    }
}

/// Fused SwiGLU: dst[i] = silu(gate[i]) * up[i]
/// Single pass, no intermediate allocation. 2x-unrolled SIMD.
pub fn swigluFused(dst: []f32, gate: []const f32, up: []const f32) void {
    std.debug.assert(dst.len == gate.len);
    std.debug.assert(gate.len == up.len);
    const n = gate.len;
    const ones: F32x4 = @splat(1.0);

    var i: usize = 0;
    // 2x unrolled
    while (i + 8 <= n) : (i += 8) {
        const g_a: F32x4 = gate[i..][0..VEC_LEN].*;
        const u_a: F32x4 = up[i..][0..VEC_LEN].*;
        const g_b: F32x4 = gate[i + 4 ..][0..VEC_LEN].*;
        const u_b: F32x4 = up[i + 4 ..][0..VEC_LEN].*;
        const sig_a = ones / (ones + @exp(-g_a));
        const sig_b = ones / (ones + @exp(-g_b));
        dst[i..][0..VEC_LEN].* = g_a * sig_a * u_a;
        dst[i + 4 ..][0..VEC_LEN].* = g_b * sig_b * u_b;
    }
    while (i + VEC_LEN <= n) : (i += VEC_LEN) {
        const g: F32x4 = gate[i..][0..VEC_LEN].*;
        const u: F32x4 = up[i..][0..VEC_LEN].*;
        const sig = ones / (ones + @exp(-g));
        dst[i..][0..VEC_LEN].* = g * sig * u;
    }
    // Scalar tail
    while (i < n) : (i += 1) {
        const g = gate[i];
        dst[i] = (g / (1.0 + @exp(-g))) * up[i];
    }
}

/// Separate (non-fused) SwiGLU: silu(gate) into tmp, then tmp * up into dst.
/// Two-pass implementation used as benchmark baseline.
/// Caller must provide pre-allocated tmp buffer of same length.
pub fn swigluSeparate(dst: []f32, gate: []const f32, up: []const f32, tmp: []f32) void {
    std.debug.assert(dst.len == gate.len);
    std.debug.assert(gate.len == up.len);
    std.debug.assert(tmp.len == gate.len);

    // Pass 1: silu(gate) -> tmp
    silu(tmp, gate);

    // Pass 2: tmp * up -> dst
    simdMul(dst, tmp, up);
}

// ============================================================
// Internal helpers
// ============================================================

/// SIMD element-wise multiply: dst[i] = a[i] * b[i]
fn simdMul(dst: []f32, a: []const f32, b: []const f32) void {
    const n = a.len;
    var i: usize = 0;
    while (i + 8 <= n) : (i += 8) {
        const va: F32x4 = a[i..][0..VEC_LEN].*;
        const vb: F32x4 = b[i..][0..VEC_LEN].*;
        const va2: F32x4 = a[i + 4 ..][0..VEC_LEN].*;
        const vb2: F32x4 = b[i + 4 ..][0..VEC_LEN].*;
        dst[i..][0..VEC_LEN].* = va * vb;
        dst[i + 4 ..][0..VEC_LEN].* = va2 * vb2;
    }
    while (i + VEC_LEN <= n) : (i += VEC_LEN) {
        const va: F32x4 = a[i..][0..VEC_LEN].*;
        const vb: F32x4 = b[i..][0..VEC_LEN].*;
        dst[i..][0..VEC_LEN].* = va * vb;
    }
    while (i < n) : (i += 1) {
        dst[i] = a[i] * b[i];
    }
}
