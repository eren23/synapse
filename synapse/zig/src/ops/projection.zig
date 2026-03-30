//! Projection GEMV with fused bias for small-K linear layers.
//!
//! Optimized for LEWM input_proj/cond_proj: M in {1,3}, N=192, K in [48,192].
//! Avoids packing overhead of the general SGEMM path.

const std = @import("std");

const VEC_LEN = 4;
const Vec = @Vector(VEC_LEN, f32);

/// Projection GEMV: output[m,n] = input[m,k] * weight[n,k]^T + bias[n]
///
/// weight is stored row-major as [n, k] (each row is one output neuron's weights).
/// bias may be null (no bias added). Output is [m, n] row-major.
pub fn projectionGemvBias(
    m: usize,
    n: usize,
    k: usize,
    input: [*]const f32,
    weight: [*]const f32,
    bias: ?[*]const f32,
    output: [*]f32,
) void {
    const k_vec = k - (k % VEC_LEN);

    for (0..m) |row| {
        const in_row = input + row * k;

        for (0..n) |col| {
            const w_row = weight + col * k;

            // Vectorized dot product
            var acc: Vec = @splat(0.0);
            var p: usize = 0;
            while (p < k_vec) : (p += VEC_LEN) {
                const a: Vec = in_row[p..][0..VEC_LEN].*;
                const b: Vec = w_row[p..][0..VEC_LEN].*;
                acc = @mulAdd(Vec, a, b, acc);
            }
            var dot: f32 = @reduce(.Add, acc);

            // Scalar tail
            while (p < k) : (p += 1) {
                dot += in_row[p] * w_row[p];
            }

            // Fused bias
            if (bias) |b| {
                dot += b[col];
            }

            output[row * n + col] = dot;
        }
    }
}

// ============================================================
// Tests
// ============================================================

test "projection_gemv_bias identity" {
    // 2x2 identity weight, zero bias => output == input
    const input = [_]f32{ 1.0, 2.0, 3.0, 4.0 }; // [2, 2]
    const weight = [_]f32{ 1.0, 0.0, 0.0, 1.0 }; // [2, 2] identity
    const bias = [_]f32{ 0.0, 0.0 };
    var output: [4]f32 = undefined;

    projectionGemvBias(2, 2, 2, &input, &weight, &bias, &output);

    try std.testing.expectApproxEqAbs(@as(f32, 1.0), output[0], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 2.0), output[1], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 3.0), output[2], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 4.0), output[3], 1e-5);
}

test "projection_gemv_bias with bias" {
    // input [1, 3] * weight [2, 3]^T + bias [2]
    const input = [_]f32{ 1.0, 2.0, 3.0 }; // [1, 3]
    const weight = [_]f32{
        1.0, 0.0, 0.0, // row 0: picks input[0]
        0.0, 1.0, 0.0, // row 1: picks input[1]
    }; // [2, 3]
    const bias = [_]f32{ 10.0, 20.0 };
    var output: [2]f32 = undefined;

    projectionGemvBias(1, 2, 3, &input, &weight, &bias, &output);

    try std.testing.expectApproxEqAbs(@as(f32, 11.0), output[0], 1e-5); // 1.0 + 10.0
    try std.testing.expectApproxEqAbs(@as(f32, 22.0), output[1], 1e-5); // 2.0 + 20.0
}

test "projection_gemv_bias null bias" {
    const input = [_]f32{ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0 }; // [2, 3]
    const weight = [_]f32{
        1.0, 1.0, 1.0, // row 0: sum of input row
        0.0, 0.0, 1.0, // row 1: input[2] only
    }; // [2, 3]
    var output: [4]f32 = undefined;

    projectionGemvBias(2, 2, 3, &input, &weight, null, &output);

    try std.testing.expectApproxEqAbs(@as(f32, 6.0), output[0], 1e-5); // 1+2+3
    try std.testing.expectApproxEqAbs(@as(f32, 3.0), output[1], 1e-5); // 3
    try std.testing.expectApproxEqAbs(@as(f32, 15.0), output[2], 1e-5); // 4+5+6
    try std.testing.expectApproxEqAbs(@as(f32, 6.0), output[3], 1e-5); // 6
}

test "projection_gemv_bias vectorized path" {
    // K=8 to exercise the SIMD path (VEC_LEN=4, so 2 full vector iterations)
    const k = 8;
    var input: [k]f32 = undefined;
    var weight: [k]f32 = undefined;
    for (0..k) |i| {
        input[i] = @as(f32, @floatFromInt(i + 1)); // 1,2,...,8
        weight[i] = 1.0; // all ones => dot = sum(1..8) = 36
    }
    const bias = [_]f32{5.0};
    var output: [1]f32 = undefined;

    projectionGemvBias(1, 1, k, &input, &weight, &bias, &output);

    try std.testing.expectApproxEqAbs(@as(f32, 41.0), output[0], 1e-5); // 36 + 5
}
