//! Fused LEWM adaLN predictor layer — one function call per layer.
//!
//! Performs the complete DiT-style adaLN transformer layer in a single call,
//! using Zig SIMD matmul internally. Zero FFI round-trips, zero allocations.
//!
//! Optimized for seq_len <= 16 (LEWM predict uses seq_len=3).

const std = @import("std");
const matmul_ops = @import("matmul.zig");

// ================================================================
// Inline helpers
// ================================================================

fn layernorm_into(x: [*]const f32, gamma: [*]const f32, seq_len: usize, hidden: usize, out: [*]f32) void {
    const eps: f32 = 1e-6;
    for (0..seq_len) |t| {
        const row = x + t * hidden;
        const dst = out + t * hidden;
        var sum: f32 = 0;
        for (0..hidden) |j| sum += row[j];
        const mean = sum / @as(f32, @floatFromInt(hidden));
        var var_sum: f32 = 0;
        for (0..hidden) |j| {
            const d = row[j] - mean;
            var_sum += d * d;
        }
        const inv_std = 1.0 / @sqrt(var_sum / @as(f32, @floatFromInt(hidden)) + eps);
        for (0..hidden) |j| dst[j] = (row[j] - mean) * inv_std * gamma[j];
    }
}

fn modulate_inplace(buf: [*]f32, scale: [*]const f32, shift: [*]const f32, seq_len: usize, hidden: usize) void {
    for (0..seq_len) |t| {
        for (0..hidden) |j| {
            const idx = t * hidden + j;
            buf[idx] = buf[idx] * (1.0 + scale[j]) + shift[j];
        }
    }
}

fn add_bias(x: [*]f32, bias: [*]const f32, seq_len: usize, dim: usize) void {
    for (0..seq_len) |t| {
        for (0..dim) |j| x[t * dim + j] += bias[j];
    }
}

fn gelu_inplace(x: [*]f32, len: usize) void {
    const c: f32 = 0.7978845608028654;
    for (0..len) |i| {
        const v = x[i];
        x[i] = 0.5 * v * (1.0 + std.math.tanh(c * (v + 0.044715 * v * v * v)));
    }
}

fn gated_residual(seq: [*]f32, gate: [*]const f32, proj: [*]const f32, seq_len: usize, hidden: usize) void {
    for (0..seq_len) |t| {
        for (0..hidden) |j| {
            seq[t * hidden + j] += gate[j] * proj[t * hidden + j];
        }
    }
}

/// GEMV/skinny-GEMM via the tiled matmul kernel (M<=16 uses per-row GEMV fast path).
/// C[m,n] = A[m,k] @ B[n,k]^T (B is in row-major [n,k], transposed internally).
fn gemm_t(m: usize, n: usize, k: usize, a: [*]const f32, b: [*]const f32, c: [*]f32) void {
    var dummy: [1]f32 = .{0};
    matmul_ops.sgemmTiled(m, n, k, a, k, false, b, k, true, c, n, &dummy, &dummy);
}

/// Inline bidirectional attention for small seq_len (<=16).
/// Q, K, V: [seq_len * inner_dim] interleaved across heads.
/// Output: [seq_len * inner_dim].
fn small_bidirectional_attention(
    q: [*]const f32,
    k_in: [*]const f32,
    v_in: [*]const f32,
    out: [*]f32,
    seq_len: usize,
    num_heads: usize,
    head_dim: usize,
) void {
    const inner_dim = num_heads * head_dim;
    const inv_sqrt = 1.0 / @sqrt(@as(f32, @floatFromInt(head_dim)));

    for (0..num_heads) |head| {
        for (0..seq_len) |qi| {
            // Compute attention scores for this query
            var scores: [16]f32 = undefined;
            var max_s: f32 = -1e30;
            for (0..seq_len) |ki| {
                var dot: f32 = 0;
                const q_off = qi * inner_dim + head * head_dim;
                const k_off = ki * inner_dim + head * head_dim;
                for (0..head_dim) |d| {
                    dot += q[q_off + d] * k_in[k_off + d];
                }
                scores[ki] = dot * inv_sqrt;
                if (scores[ki] > max_s) max_s = scores[ki];
            }
            // Softmax
            var exp_sum: f32 = 0;
            for (0..seq_len) |ki| {
                scores[ki] = @exp(scores[ki] - max_s);
                exp_sum += scores[ki];
            }
            const inv_sum = if (exp_sum > 1e-12) 1.0 / exp_sum else 0.0;
            // Weighted V aggregation
            const out_off = qi * inner_dim + head * head_dim;
            for (0..head_dim) |d| {
                var val: f32 = 0;
                for (0..seq_len) |ki| {
                    val += scores[ki] * v_in[ki * inner_dim + head * head_dim + d];
                }
                out[out_off + d] = val * inv_sum;
            }
        }
    }
}

// ================================================================
// Main fused layer
// ================================================================

pub fn lewmPredictorLayer(
    seq: [*]f32, // [seq_len * hidden], in-place
    conditioning: [*]const f32, // [hidden]
    seq_len: usize,
    hidden: usize,
    num_heads: usize,
    inner_dim: usize,
    inter: usize,
    // Weights
    adaln_weight: [*]const f32,
    adaln_bias: ?[*]const f32,
    attn_norm_weight: [*]const f32,
    to_qkv: [*]const f32,
    attn_out_weight: [*]const f32,
    attn_out_bias: ?[*]const f32,
    mlp_norm_weight: [*]const f32,
    mlp_up_weight: [*]const f32,
    mlp_up_bias: ?[*]const f32,
    mlp_down_weight: [*]const f32,
    mlp_down_bias: ?[*]const f32,
    // Scratch
    mod_buf: [*]f32, // [6 * hidden]
    normed_buf: [*]f32, // [max(seq_len * hidden, seq_len * inter)]
    qkv_buf: [*]f32, // [seq_len * 3 * inner_dim]
    attn_buf: [*]f32, // [seq_len * inner_dim]
    proj_buf: [*]f32, // [max(seq_len * hidden, seq_len * inter)]
) void {
    const mod_dim = 6 * hidden;

    // 1. adaLN modulation
    gemm_t(1, mod_dim, hidden, conditioning, adaln_weight, mod_buf);
    if (adaln_bias) |bias| {
        for (0..mod_dim) |j| mod_buf[j] += bias[j];
    }

    // 2. LayerNorm + modulate + QKV
    layernorm_into(seq, attn_norm_weight, seq_len, hidden, normed_buf);
    modulate_inplace(normed_buf, mod_buf, mod_buf + hidden, seq_len, hidden);
    gemm_t(seq_len, 3 * inner_dim, hidden, normed_buf, to_qkv, qkv_buf);

    // 3. Attention (works directly on interleaved QKV — no split needed for small seq)
    // Split QKV into separate buffers for the attention kernel
    for (0..seq_len) |t| {
        const qkv_off = t * 3 * inner_dim;
        const off = t * inner_dim;
        @memcpy((normed_buf + off)[0..inner_dim], (qkv_buf + qkv_off)[0..inner_dim]); // Q
        @memcpy((proj_buf + off)[0..inner_dim], (qkv_buf + qkv_off + inner_dim)[0..inner_dim]); // K
        @memcpy((qkv_buf + off)[0..inner_dim], (qkv_buf + qkv_off + 2 * inner_dim)[0..inner_dim]); // V (reuse qkv_buf start)
    }
    // Now: normed_buf has Q, proj_buf has K, qkv_buf[0..seq*inner] has V
    small_bidirectional_attention(normed_buf, proj_buf, qkv_buf, attn_buf, seq_len, num_heads, inner_dim / num_heads);

    // 4. Output projection + gated residual
    gemm_t(seq_len, hidden, inner_dim, attn_buf, attn_out_weight, proj_buf);
    if (attn_out_bias) |bias| add_bias(proj_buf, bias, seq_len, hidden);
    gated_residual(seq, mod_buf + 2 * hidden, proj_buf, seq_len, hidden); // gate1

    // 5. LayerNorm + modulate for FFN
    layernorm_into(seq, mlp_norm_weight, seq_len, hidden, normed_buf);
    modulate_inplace(normed_buf, mod_buf + 3 * hidden, mod_buf + 4 * hidden, seq_len, hidden);

    // 6. FFN: up → GELU → down
    gemm_t(seq_len, inter, hidden, normed_buf, mlp_up_weight, proj_buf);
    if (mlp_up_bias) |bias| add_bias(proj_buf, bias, seq_len, inter);
    gelu_inplace(proj_buf, seq_len * inter);
    gemm_t(seq_len, hidden, inter, proj_buf, mlp_down_weight, normed_buf);
    if (mlp_down_bias) |bias| add_bias(normed_buf, bias, seq_len, hidden);

    // 7. Gated residual
    gated_residual(seq, mod_buf + 5 * hidden, normed_buf, seq_len, hidden); // gate2
}
