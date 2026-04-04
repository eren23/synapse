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

pub fn layernorm_into(x: [*]const f32, gamma: [*]const f32, seq_len: usize, hidden: usize, out: [*]f32) void {
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

pub fn modulate_inplace(buf: [*]f32, scale: [*]const f32, shift: [*]const f32, seq_len: usize, hidden: usize) void {
    for (0..seq_len) |t| {
        for (0..hidden) |j| {
            const idx = t * hidden + j;
            buf[idx] = buf[idx] * (1.0 + scale[j]) + shift[j];
        }
    }
}

pub fn add_bias(x: [*]f32, bias: [*]const f32, seq_len: usize, dim: usize) void {
    for (0..seq_len) |t| {
        for (0..dim) |j| x[t * dim + j] += bias[j];
    }
}

pub fn gelu_inplace(x: [*]f32, len: usize) void {
    const c: f32 = 0.7978845608028654;
    for (0..len) |i| {
        const v = x[i];
        x[i] = 0.5 * v * (1.0 + std.math.tanh(c * (v + 0.044715 * v * v * v)));
    }
}

pub fn gated_residual(seq: [*]f32, gate: [*]const f32, proj: [*]const f32, seq_len: usize, hidden: usize) void {
    for (0..seq_len) |t| {
        for (0..hidden) |j| {
            seq[t * hidden + j] += gate[j] * proj[t * hidden + j];
        }
    }
}

// ================================================================
// ESP-style fused helpers — single-pass loops that eliminate
// intermediate memory round-trips (mirrors inference.c fusions).
// ================================================================

/// Fused bias + GELU: x[i] = gelu(x[i] + bias[i]).
/// Replaces separate add_bias + gelu_inplace (2 loops → 1).
pub fn bias_gelu_inplace(x: [*]f32, bias: [*]const f32, seq_len: usize, dim: usize) void {
    const c: f32 = 0.7978845608028654;
    for (0..seq_len) |t| {
        for (0..dim) |j| {
            const idx = t * dim + j;
            const v = x[idx] + bias[j];
            x[idx] = 0.5 * v * (1.0 + std.math.tanh(c * (v + 0.044715 * v * v * v)));
        }
    }
}

/// Fused bias + gated residual: seq[i] += gate[i] * (proj[i] + bias[i]).
/// Replaces separate add_bias + gated_residual (2 loops → 1).
pub fn bias_gated_residual(seq: [*]f32, gate: [*]const f32, proj: [*]const f32, bias: [*]const f32, seq_len: usize, dim: usize) void {
    for (0..seq_len) |t| {
        for (0..dim) |j| {
            const idx = t * dim + j;
            seq[idx] += gate[j] * (proj[idx] + bias[j]);
        }
    }
}

/// GEMV/skinny-GEMM via the tiled matmul kernel (M<=16 uses per-row GEMV fast path).
/// C[m,n] = A[m,k] @ B[n,k]^T (B is in row-major [n,k], transposed internally).
pub fn gemm_t(m: usize, n: usize, k: usize, a: [*]const f32, b: [*]const f32, c: [*]f32) void {
    var dummy: [1]f32 = .{0};
    matmul_ops.sgemmTiled(m, n, k, a, k, false, b, k, true, c, n, &dummy, &dummy);
}

/// Inline bidirectional attention for small seq_len (<=16).
/// Q, K, V: [seq_len * inner_dim] interleaved across heads.
/// Output: [seq_len * inner_dim].
pub fn small_bidirectional_attention(
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

/// Dynamic bidirectional attention for arbitrary seq_len.
/// Same algorithm as `small_bidirectional_attention` but uses a caller-provided
/// `scores_buf` (sized seq_len * seq_len) instead of a stack-allocated [16]f32.
/// This removes the seq_len<=16 limit, enabling fused rollout with seq_len=150+.
pub fn bidirectional_attention_dynamic(
    q: [*]const f32,
    k_in: [*]const f32,
    v_in: [*]const f32,
    out: [*]f32,
    scores_buf: [*]f32,
    seq_len: usize,
    num_heads: usize,
    head_dim: usize,
) void {
    const inner_dim = num_heads * head_dim;
    const inv_sqrt = 1.0 / @sqrt(@as(f32, @floatFromInt(head_dim)));

    for (0..num_heads) |head| {
        for (0..seq_len) |qi| {
            var max_s: f32 = -1e30;
            for (0..seq_len) |ki| {
                var dot: f32 = 0;
                const q_off = qi * inner_dim + head * head_dim;
                const k_off = ki * inner_dim + head * head_dim;
                for (0..head_dim) |d| {
                    dot += q[q_off + d] * k_in[k_off + d];
                }
                const score = dot * inv_sqrt;
                scores_buf[qi * seq_len + ki] = score;
                if (score > max_s) max_s = score;
            }
            var exp_sum: f32 = 0;
            for (0..seq_len) |ki| {
                const idx = qi * seq_len + ki;
                scores_buf[idx] = @exp(scores_buf[idx] - max_s);
                exp_sum += scores_buf[idx];
            }
            const inv_sum = if (exp_sum > 1e-12) 1.0 / exp_sum else 0.0;
            const out_off = qi * inner_dim + head * head_dim;
            for (0..head_dim) |d| {
                var val: f32 = 0;
                for (0..seq_len) |ki| {
                    val += scores_buf[qi * seq_len + ki] * v_in[ki * inner_dim + head * head_dim + d];
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

/// ESP-fused predictor layer — same semantics as `lewmPredictorLayer` but with
/// 3 inner fusions that eliminate extra memory passes:
///   1. attn out: bias + gated residual in one loop
///   2. FFN up:   bias + GELU in one loop
///   3. FFN down: bias + gated residual in one loop
pub fn lewmPredictorLayerEspFused(
    seq: [*]f32,
    conditioning: [*]const f32,
    seq_len: usize,
    hidden: usize,
    num_heads: usize,
    inner_dim: usize,
    inter: usize,
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
    mod_buf: [*]f32,
    normed_buf: [*]f32,
    qkv_buf: [*]f32,
    attn_buf: [*]f32,
    proj_buf: [*]f32,
) void {
    const mod_dim = 6 * hidden;

    // 1. adaLN modulation (same as standard)
    gemm_t(1, mod_dim, hidden, conditioning, adaln_weight, mod_buf);
    if (adaln_bias) |bias| {
        for (0..mod_dim) |j| mod_buf[j] += bias[j];
    }

    // 2. LayerNorm + modulate + QKV (same as standard)
    layernorm_into(seq, attn_norm_weight, seq_len, hidden, normed_buf);
    modulate_inplace(normed_buf, mod_buf, mod_buf + hidden, seq_len, hidden);
    gemm_t(seq_len, 3 * inner_dim, hidden, normed_buf, to_qkv, qkv_buf);

    // 3. Attention (same as standard)
    for (0..seq_len) |t| {
        const qkv_off = t * 3 * inner_dim;
        const off = t * inner_dim;
        @memcpy((normed_buf + off)[0..inner_dim], (qkv_buf + qkv_off)[0..inner_dim]);
        @memcpy((proj_buf + off)[0..inner_dim], (qkv_buf + qkv_off + inner_dim)[0..inner_dim]);
        @memcpy((qkv_buf + off)[0..inner_dim], (qkv_buf + qkv_off + 2 * inner_dim)[0..inner_dim]);
    }
    small_bidirectional_attention(normed_buf, proj_buf, qkv_buf, attn_buf, seq_len, num_heads, inner_dim / num_heads);

    // 4. Output projection + FUSED bias+gated residual (ESP fusion #1)
    gemm_t(seq_len, hidden, inner_dim, attn_buf, attn_out_weight, proj_buf);
    if (attn_out_bias) |bias| {
        bias_gated_residual(seq, mod_buf + 2 * hidden, proj_buf, bias, seq_len, hidden);
    } else {
        gated_residual(seq, mod_buf + 2 * hidden, proj_buf, seq_len, hidden);
    }

    // 5. LayerNorm + modulate for FFN (same as standard)
    layernorm_into(seq, mlp_norm_weight, seq_len, hidden, normed_buf);
    modulate_inplace(normed_buf, mod_buf + 3 * hidden, mod_buf + 4 * hidden, seq_len, hidden);

    // 6. FFN: up → FUSED bias+GELU → down (ESP fusion #2)
    gemm_t(seq_len, inter, hidden, normed_buf, mlp_up_weight, proj_buf);
    if (mlp_up_bias) |bias| {
        bias_gelu_inplace(proj_buf, bias, seq_len, inter);
    } else {
        gelu_inplace(proj_buf, seq_len * inter);
    }

    // 7. FFN down + FUSED bias+gated residual (ESP fusion #3)
    gemm_t(seq_len, hidden, inter, proj_buf, mlp_down_weight, normed_buf);
    if (mlp_down_bias) |bias| {
        bias_gated_residual(seq, mod_buf + 5 * hidden, normed_buf, bias, seq_len, hidden);
    } else {
        gated_residual(seq, mod_buf + 5 * hidden, normed_buf, seq_len, hidden);
    }
}

/// Dispatch between standard and ESP-fused predictor layer.
/// mode: 0 = standard (separate loops), 1 = ESP-fused (single-pass loops).
pub fn lewmPredictorLayerV2(
    seq: [*]f32,
    conditioning: [*]const f32,
    seq_len: usize,
    hidden: usize,
    num_heads: usize,
    inner_dim: usize,
    inter: usize,
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
    mod_buf: [*]f32,
    normed_buf: [*]f32,
    qkv_buf: [*]f32,
    attn_buf: [*]f32,
    proj_buf: [*]f32,
    mode: u8,
) void {
    if (mode == 1) {
        lewmPredictorLayerEspFused(
            seq, conditioning, seq_len, hidden, num_heads, inner_dim, inter,
            adaln_weight, adaln_bias, attn_norm_weight, to_qkv, attn_out_weight, attn_out_bias,
            mlp_norm_weight, mlp_up_weight, mlp_up_bias, mlp_down_weight, mlp_down_bias,
            mod_buf, normed_buf, qkv_buf, attn_buf, proj_buf,
        );
    } else {
        lewmPredictorLayer(
            seq, conditioning, seq_len, hidden, num_heads, inner_dim, inter,
            adaln_weight, adaln_bias, attn_norm_weight, to_qkv, attn_out_weight, attn_out_bias,
            mlp_norm_weight, mlp_up_weight, mlp_up_bias, mlp_down_weight, mlp_down_bias,
            mod_buf, normed_buf, qkv_buf, attn_buf, proj_buf,
        );
    }
}

// ================================================================
// Tests
// ================================================================

test "bidirectional_attention_dynamic matches small_bidirectional_attention" {
    const seq_len = 3;
    const num_heads = 2;
    const head_dim = 4;
    const inner_dim = num_heads * head_dim; // 8

    // Fill Q, K, V with deterministic LCG pseudo-random data
    var q_buf: [seq_len * inner_dim]f32 = undefined;
    var k_buf: [seq_len * inner_dim]f32 = undefined;
    var v_buf: [seq_len * inner_dim]f32 = undefined;

    var seed: i32 = 12345;
    const a: i32 = 1103515245;
    const c: i32 = 12345;
    const m: i32 = 1 << 16;

    for (&q_buf) |*p| {
        seed = @rem(seed *% a +% c, m);
        p.* = @as(f32, @floatFromInt(seed)) / @as(f32, @floatFromInt(m)) - 0.5;
    }
    for (&k_buf) |*p| {
        seed = @rem(seed *% a +% c, m);
        p.* = @as(f32, @floatFromInt(seed)) / @as(f32, @floatFromInt(m)) - 0.5;
    }
    for (&v_buf) |*p| {
        seed = @rem(seed *% a +% c, m);
        p.* = @as(f32, @floatFromInt(seed)) / @as(f32, @floatFromInt(m)) - 0.5;
    }

    // Reference: small_bidirectional_attention (stack-based, seq_len<=16)
    var out_ref: [seq_len * inner_dim]f32 = undefined;
    small_bidirectional_attention(&q_buf, &k_buf, &v_buf, &out_ref, seq_len, num_heads, head_dim);

    // Dynamic: bidirectional_attention_dynamic (heap-style scores buffer)
    var out_dyn: [seq_len * inner_dim]f32 = undefined;
    var scores_buf: [seq_len * seq_len]f32 = undefined;
    bidirectional_attention_dynamic(&q_buf, &k_buf, &v_buf, &out_dyn, &scores_buf, seq_len, num_heads, head_dim);

    // Assert all elements match within 1e-5
    const tol: f32 = 1e-5;
    for (0..seq_len * inner_dim) |i| {
        const diff = @abs(out_ref[i] - out_dyn[i]);
        if (diff > tol) {
            std.debug.print("Mismatch at index {}: ref={d:.8} dyn={d:.8} diff={d:.8}\n", .{ i, out_ref[i], out_dyn[i], diff });
            return error.TestUnexpectedResult;
        }
    }
}
