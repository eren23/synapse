//! RMS normalization with SIMD-vectorized squared-sum reduction.
//! Formula: output = x * rsqrt(mean(x²) + eps) * gamma
//! No mean subtraction (unlike LayerNorm).
//! Two implementations: SIMD (primary), scalar reference.

const std = @import("std");
const shape_mod = @import("../tensor/shape.zig");
const tensor_mod = @import("../tensor/tensor.zig");
const storage_mod = @import("../tensor/storage.zig");

const Shape = shape_mod.Shape;
const Tensor = tensor_mod.Tensor;
const Storage = storage_mod.Storage;

const VEC_LEN = 4;
const F32x4 = @Vector(VEC_LEN, f32);

// ============================================================
// Helpers
// ============================================================

/// Product of the last `num_norm_dims` dimensions.
fn computeNormSize(shape: Shape, num_norm_dims: usize) usize {
    var size: usize = 1;
    const start = shape.ndim - num_norm_dims;
    for (start..shape.ndim) |d| {
        size *= shape.dims[d];
    }
    return size;
}

// ============================================================
// Public API
// ============================================================

/// SIMD RMS normalization (primary implementation).
/// Normalizes over the last `num_norm_dims` trailing dimensions.
/// gamma must have length equal to the product of trailing dims.
pub fn rmsNorm(
    allocator: std.mem.Allocator,
    input: Tensor(f32),
    num_norm_dims: usize,
    gamma: []const f32,
    eps: f32,
) !Tensor(f32) {
    if (num_norm_dims == 0 or num_norm_dims > input.shape.ndim) return error.InvalidNormDims;
    std.debug.assert(input.isContiguous());

    const norm_size = computeNormSize(input.shape, num_norm_dims);
    const outer_size = input.numel() / norm_size;
    std.debug.assert(gamma.len == norm_size);

    const out_storage = try Storage.create(allocator, f32, input.numel());
    const out = Tensor(f32).init(out_storage, input.shape);
    out_storage.release();

    const in_data = input.storage.dataAs(f32);
    const out_data = out.storage.dataAs(f32);

    for (0..outer_size) |outer| {
        const base = input.offset + outer * norm_size;
        const in_slice = in_data[base .. base + norm_size];
        const out_slice = out_data[base .. base + norm_size];

        const rms_inv = simdRmsInv(in_slice, eps);
        simdScaleGamma(out_slice, in_slice, gamma, rms_inv);
    }

    return out;
}

/// Scalar reference RMS normalization (correctness baseline).
pub fn rmsNormScalar(
    allocator: std.mem.Allocator,
    input: Tensor(f32),
    num_norm_dims: usize,
    gamma: []const f32,
    eps: f32,
) !Tensor(f32) {
    if (num_norm_dims == 0 or num_norm_dims > input.shape.ndim) return error.InvalidNormDims;
    std.debug.assert(input.isContiguous());

    const norm_size = computeNormSize(input.shape, num_norm_dims);
    const outer_size = input.numel() / norm_size;
    std.debug.assert(gamma.len == norm_size);

    const out_storage = try Storage.create(allocator, f32, input.numel());
    const out = Tensor(f32).init(out_storage, input.shape);
    out_storage.release();

    const in_data = input.storage.dataAs(f32);
    const out_data = out.storage.dataAs(f32);
    const norm_f: f32 = @floatFromInt(norm_size);

    for (0..outer_size) |outer| {
        const base = input.offset + outer * norm_size;

        // Scalar sum of squares
        var sum_sq: f32 = 0;
        for (0..norm_size) |k| {
            const x = in_data[base + k];
            sum_sq += x * x;
        }
        const rms_inv = 1.0 / @sqrt(sum_sq / norm_f + eps);

        // Scalar normalize + scale
        for (0..norm_size) |j| {
            out_data[base + j] = gamma[j] * in_data[base + j] * rms_inv;
        }
    }

    return out;
}

// ============================================================
// SIMD primitives (pub for benchmarking)
// ============================================================

/// SIMD rsqrt(mean(x²) + eps) over a contiguous f32 slice.
/// Uses 2x-unrolled sum_sq accumulation (division-free inner loop).
pub fn simdRmsInv(data: []const f32, eps: f32) f32 {
    const n = data.len;
    if (n == 0) return 0;

    var ssq_a: F32x4 = @splat(0.0);
    var ssq_b: F32x4 = @splat(0.0);

    var i: usize = 0;
    // 2x unrolled: 8 elements per iteration
    while (i + 8 <= n) : (i += 8) {
        const x_a: F32x4 = data[i..][0..VEC_LEN].*;
        const x_b: F32x4 = data[i + 4 ..][0..VEC_LEN].*;
        ssq_a += x_a * x_a;
        ssq_b += x_b * x_b;
    }
    // Remaining full vector
    while (i + VEC_LEN <= n) : (i += VEC_LEN) {
        const x: F32x4 = data[i..][0..VEC_LEN].*;
        ssq_a += x * x;
    }

    var total_ssq: f32 = @reduce(.Add, ssq_a + ssq_b);

    // Scalar tail
    while (i < n) : (i += 1) {
        total_ssq += data[i] * data[i];
    }

    const n_f: f32 = @floatFromInt(n);
    return 1.0 / @sqrt(total_ssq / n_f + eps);
}

/// SIMD scale: dst[i] = gamma[i] * src[i] * rms_inv
pub fn simdScaleGamma(
    dst: []f32,
    src: []const f32,
    gamma: []const f32,
    rms_inv: f32,
) void {
    const n = src.len;
    const inv_vec: F32x4 = @splat(rms_inv);

    var i: usize = 0;
    // 2x unrolled: 8 elements per iteration
    while (i + 8 <= n) : (i += 8) {
        const x_a: F32x4 = src[i..][0..VEC_LEN].*;
        const g_a: F32x4 = gamma[i..][0..VEC_LEN].*;
        const x_b: F32x4 = src[i + 4 ..][0..VEC_LEN].*;
        const g_b: F32x4 = gamma[i + 4 ..][0..VEC_LEN].*;
        dst[i..][0..VEC_LEN].* = g_a * x_a * inv_vec;
        dst[i + 4 ..][0..VEC_LEN].* = g_b * x_b * inv_vec;
    }
    while (i + VEC_LEN <= n) : (i += VEC_LEN) {
        const x: F32x4 = src[i..][0..VEC_LEN].*;
        const g: F32x4 = gamma[i..][0..VEC_LEN].*;
        dst[i..][0..VEC_LEN].* = g * x * inv_vec;
    }

    // Scalar tail
    while (i < n) : (i += 1) {
        dst[i] = gamma[i] * src[i] * rms_inv;
    }
}
