//! Layer normalization with SIMD-vectorized Welford single-pass algorithm.
//! Normalizes over trailing dimensions. Applies affine transform (gamma*x+beta).
//! Three implementations: SIMD Welford (primary), naive two-pass, scalar reference.

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

/// SIMD Welford single-pass layer normalization (primary implementation).
/// Normalizes over the last `num_norm_dims` trailing dimensions.
/// gamma and beta must have length equal to the product of trailing dims.
pub fn layerNorm(
    allocator: std.mem.Allocator,
    input: Tensor(f32),
    num_norm_dims: usize,
    gamma: []const f32,
    beta: []const f32,
    eps: f32,
) !Tensor(f32) {
    if (num_norm_dims == 0 or num_norm_dims > input.shape.ndim) return error.InvalidNormDims;
    std.debug.assert(input.isContiguous());

    const norm_size = computeNormSize(input.shape, num_norm_dims);
    const outer_size = input.numel() / norm_size;
    std.debug.assert(gamma.len == norm_size);
    std.debug.assert(beta.len == norm_size);

    const out_storage = try Storage.create(allocator, f32, input.numel());
    const out = Tensor(f32).init(out_storage, input.shape);
    out_storage.release();

    const in_data = input.storage.dataAs(f32);
    const out_data = out.storage.dataAs(f32);

    for (0..outer_size) |outer| {
        const base = input.offset + outer * norm_size;
        const in_slice = in_data[base .. base + norm_size];
        const out_slice = out_data[base .. base + norm_size];

        const stats = simdWelfordMeanVar(in_slice);
        const inv_std = 1.0 / @sqrt(stats.variance + eps);

        simdNormalizeAffine(out_slice, in_slice, gamma, beta, stats.mean, inv_std);
    }

    return out;
}

/// Naive two-pass layer normalization (benchmark baseline).
/// Pass 1: compute mean. Pass 2: compute variance.
pub fn layerNormTwoPass(
    allocator: std.mem.Allocator,
    input: Tensor(f32),
    num_norm_dims: usize,
    gamma: []const f32,
    beta: []const f32,
    eps: f32,
) !Tensor(f32) {
    if (num_norm_dims == 0 or num_norm_dims > input.shape.ndim) return error.InvalidNormDims;
    std.debug.assert(input.isContiguous());

    const norm_size = computeNormSize(input.shape, num_norm_dims);
    const outer_size = input.numel() / norm_size;

    const out_storage = try Storage.create(allocator, f32, input.numel());
    const out = Tensor(f32).init(out_storage, input.shape);
    out_storage.release();

    const in_data = input.storage.dataAs(f32);
    const out_data = out.storage.dataAs(f32);
    const norm_f: f32 = @floatFromInt(norm_size);

    for (0..outer_size) |outer| {
        const base = input.offset + outer * norm_size;
        const in_slice = in_data[base .. base + norm_size];
        const out_slice = out_data[base .. base + norm_size];

        // Pass 1: mean
        var sum: f32 = 0;
        for (in_slice) |x| sum += x;
        const mean = sum / norm_f;

        // Pass 2: variance
        var sq_sum: f32 = 0;
        for (in_slice) |x| {
            const d = x - mean;
            sq_sum += d * d;
        }
        const variance = sq_sum / norm_f;
        const inv_std = 1.0 / @sqrt(variance + eps);

        // Normalize + affine
        for (0..norm_size) |j| {
            out_slice[j] = gamma[j] * (in_slice[j] - mean) * inv_std + beta[j];
        }
    }

    return out;
}

/// Scalar reference layer normalization (correctness baseline).
/// Uses Welford's algorithm without SIMD.
pub fn layerNormScalar(
    allocator: std.mem.Allocator,
    input: Tensor(f32),
    num_norm_dims: usize,
    gamma: []const f32,
    beta: []const f32,
    eps: f32,
) !Tensor(f32) {
    if (num_norm_dims == 0 or num_norm_dims > input.shape.ndim) return error.InvalidNormDims;
    std.debug.assert(input.isContiguous());

    const norm_size = computeNormSize(input.shape, num_norm_dims);
    const outer_size = input.numel() / norm_size;

    const out_storage = try Storage.create(allocator, f32, input.numel());
    const out = Tensor(f32).init(out_storage, input.shape);
    out_storage.release();

    const in_data = input.storage.dataAs(f32);
    const out_data = out.storage.dataAs(f32);
    const norm_f: f32 = @floatFromInt(norm_size);

    for (0..outer_size) |outer| {
        const base = input.offset + outer * norm_size;

        // Scalar Welford single-pass
        var mean: f32 = 0;
        var m2: f32 = 0;
        for (0..norm_size) |k| {
            const x = in_data[base + k];
            const count_f: f32 = @floatFromInt(k + 1);
            const delta = x - mean;
            mean += delta / count_f;
            const delta2 = x - mean;
            m2 += delta * delta2;
        }
        const variance = m2 / norm_f;
        const inv_std = 1.0 / @sqrt(variance + eps);

        // Scalar normalize + affine
        for (0..norm_size) |j| {
            out_data[base + j] = gamma[j] * (in_data[base + j] - mean) * inv_std + beta[j];
        }
    }

    return out;
}

// ============================================================
// SIMD primitives (pub for benchmarking)
// ============================================================

/// SIMD single-pass mean+variance over a contiguous f32 slice.
/// Uses 2x-unrolled sum+sum_sq accumulation (division-free inner loop).
pub fn simdWelfordMeanVar(data: []const f32) struct { mean: f32, variance: f32 } {
    const n = data.len;
    if (n == 0) return .{ .mean = 0, .variance = 0 };

    var sum_a: F32x4 = @splat(0.0);
    var ssq_a: F32x4 = @splat(0.0);
    var sum_b: F32x4 = @splat(0.0);
    var ssq_b: F32x4 = @splat(0.0);

    var i: usize = 0;
    // 2x unrolled: 8 elements per iteration
    while (i + 8 <= n) : (i += 8) {
        const x_a: F32x4 = data[i..][0..VEC_LEN].*;
        const x_b: F32x4 = data[i + 4 ..][0..VEC_LEN].*;
        sum_a += x_a;
        ssq_a += x_a * x_a;
        sum_b += x_b;
        ssq_b += x_b * x_b;
    }
    // Remaining full vector
    while (i + VEC_LEN <= n) : (i += VEC_LEN) {
        const x: F32x4 = data[i..][0..VEC_LEN].*;
        sum_a += x;
        ssq_a += x * x;
    }

    var total_sum: f32 = @reduce(.Add, sum_a + sum_b);
    var total_ssq: f32 = @reduce(.Add, ssq_a + ssq_b);

    // Scalar tail
    while (i < n) : (i += 1) {
        total_sum += data[i];
        total_ssq += data[i] * data[i];
    }

    const n_f: f32 = @as(f32, @floatFromInt(n));
    const mean = total_sum / n_f;
    return .{ .mean = mean, .variance = @max(total_ssq / n_f - mean * mean, 0.0) };
}

/// SIMD normalize + affine: dst[i] = gamma[i] * (src[i] - mean) * inv_std + beta[i]
pub fn simdNormalizeAffine(
    dst: []f32,
    src: []const f32,
    gamma: []const f32,
    beta: []const f32,
    mean: f32,
    inv_std: f32,
) void {
    const n = src.len;
    const mean_vec: F32x4 = @splat(mean);
    const inv_std_vec: F32x4 = @splat(inv_std);

    var i: usize = 0;
    // 2x unrolled: 8 elements per iteration
    while (i + 8 <= n) : (i += 8) {
        const x_a: F32x4 = src[i..][0..VEC_LEN].*;
        const g_a: F32x4 = gamma[i..][0..VEC_LEN].*;
        const b_a: F32x4 = beta[i..][0..VEC_LEN].*;
        const x_b: F32x4 = src[i + 4 ..][0..VEC_LEN].*;
        const g_b: F32x4 = gamma[i + 4 ..][0..VEC_LEN].*;
        const b_b: F32x4 = beta[i + 4 ..][0..VEC_LEN].*;
        dst[i..][0..VEC_LEN].* = g_a * ((x_a - mean_vec) * inv_std_vec) + b_a;
        dst[i + 4 ..][0..VEC_LEN].* = g_b * ((x_b - mean_vec) * inv_std_vec) + b_b;
    }
    while (i + VEC_LEN <= n) : (i += VEC_LEN) {
        const x: F32x4 = src[i..][0..VEC_LEN].*;
        const g: F32x4 = gamma[i..][0..VEC_LEN].*;
        const b: F32x4 = beta[i..][0..VEC_LEN].*;
        dst[i..][0..VEC_LEN].* = g * ((x - mean_vec) * inv_std_vec) + b;
    }

    // Scalar tail
    while (i < n) : (i += 1) {
        dst[i] = gamma[i] * (src[i] - mean) * inv_std + beta[i];
    }
}
