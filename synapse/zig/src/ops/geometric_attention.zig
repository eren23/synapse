//! Geometric Attention: distance-aware attention for 3D point clouds and molecules.
//!
//! score[i,j] = softmax(Q[i]·K[j] / sqrt(d) + distance_bias(pos[i], pos[j]))
//!
//! Standard attention ignores spatial position. Geometric attention adds a
//! distance-dependent Gaussian bias so closer points attend more to each other.
//! Used in: PointNet++, AlphaFold pairwise attention, robotics 3D manipulation.
//!
//! This is an op that PyTorch/MLX don't have optimized kernels for.

const std = @import("std");

const VEC_LEN: usize = 4;
const F32x4 = @Vector(VEC_LEN, f32);

/// Geometric attention with distance bias.
///
/// Q, K, V: [n, d] row-major (n points, d embedding dimension)
/// positions: [n, pos_dim] spatial coordinates (typically 3D)
/// out: [n, d] output embeddings
/// sigma: bandwidth of the Gaussian distance kernel
///
/// For each query point i:
///   score[j] = (Q[i] · K[j]) / sqrt(d) + exp(-||pos_i - pos_j||² / (2σ²))
///   out[i] = softmax(scores) · V
pub fn geometricAttention(
    n: usize,
    d: usize,
    pos_dim: usize,
    q: [*]const f32,
    k: [*]const f32,
    v: [*]const f32,
    positions: [*]const f32,
    out: [*]f32,
    sigma: f32,
) void {
    const scale: f32 = 1.0 / @sqrt(@as(f32, @floatFromInt(d)));
    const inv_2sigma2: f32 = 1.0 / (2.0 * sigma * sigma);

    for (0..n) |i| {
        // ── Phase 1: Compute scores with distance bias ──────────────

        // Use stack allocation for scores (max 4096 points, then heap fallback)
        var score_buf: [4096]f32 = undefined;
        const scores: [*]f32 = if (n <= 4096) &score_buf else unreachable;
        // TODO: for n > 4096, allocate from scratch buffer

        var max_score: f32 = -std.math.inf(f32);

        for (0..n) |j| {
            // Q·K dot product with SIMD
            var dot: f32 = 0;
            var dim: usize = 0;
            const d4 = d - (d % VEC_LEN);
            while (dim < d4) : (dim += VEC_LEN) {
                const q_vec: F32x4 = (q + i * d + dim)[0..VEC_LEN].*;
                const k_vec: F32x4 = (k + j * d + dim)[0..VEC_LEN].*;
                const prod = q_vec * k_vec;
                dot += @reduce(.Add, prod);
            }
            // Scalar tail
            while (dim < d) : (dim += 1) {
                dot += q[i * d + dim] * k[j * d + dim];
            }

            // Distance bias: exp(-||pos_i - pos_j||² / (2σ²))
            var dist_sq: f32 = 0;
            for (0..pos_dim) |p| {
                const diff = positions[i * pos_dim + p] - positions[j * pos_dim + p];
                dist_sq += diff * diff;
            }
            const dist_bias = @exp(-dist_sq * inv_2sigma2);

            const score = dot * scale + dist_bias;
            scores[j] = score;
            if (score > max_score) max_score = score;
        }

        // ── Phase 2: Softmax ────────────────────────────────────────

        var sum_exp: f32 = 0;
        for (0..n) |j| {
            const e = @exp(scores[j] - max_score);
            scores[j] = e;
            sum_exp += e;
        }
        const inv_sum = if (sum_exp > 0) 1.0 / sum_exp else 0.0;
        for (0..n) |j| {
            scores[j] *= inv_sum;
        }

        // ── Phase 3: Weighted V sum with SIMD ───────────────────────

        // Zero output row
        @memset((out + i * d)[0..d], @as(f32, 0));

        for (0..n) |j| {
            const w = scores[j];
            if (w < 1e-8) continue; // Skip negligible weights

            const w_splat: F32x4 = @splat(w);
            var dim2: usize = 0;
            const d4_2 = d - (d % VEC_LEN);
            while (dim2 < d4_2) : (dim2 += VEC_LEN) {
                const v_vec: F32x4 = (v + j * d + dim2)[0..VEC_LEN].*;
                const out_vec: F32x4 = (out + i * d + dim2)[0..VEC_LEN].*;
                (out + i * d + dim2)[0..VEC_LEN].* = out_vec + w_splat * v_vec;
            }
            while (dim2 < d) : (dim2 += 1) {
                out[i * d + dim2] += w * v[j * d + dim2];
            }
        }
    }
}
