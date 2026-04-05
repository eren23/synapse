//! Fused Code WM encoder — weight-shared pre-norm transformer loops.
//!
//! Unlike LEWM's fused_lewm_rollout (which handles adaLN modulation + gated
//! residuals + per-layer weights), Code WM uses vanilla pre-norm blocks:
//!     PreNorm → MHA → +res → PreNorm → MLP(GELU_erf) → +res
//! with a single weight-shared block iterated encoder_loops times.
//!
//! Key differences from LEWM kernel:
//!   - LayerNorm with bias (gamma + beta)  — PyTorch nn.LayerNorm default
//!   - Exact erf GELU (not tanh approximation) — matches nn.GELU() default
//!   - Plain residuals (no gating)
//!   - Single shared weight set (loops reuse the same norm/attn/mlp weights)
//!
//! **Status**: correctness verified (cos ≈ 1.0 vs sequential path) but
//! currently SLOWER than the Rust sequential path because the attention
//! uses fused_lewm_layer's scalar implementation. The sequential Rust path
//! already dispatches to SIMD-tiled `syn_fused_attention_bidi` per-head,
//! which dominates the speedup budget. Use this kernel as scaffolding for
//! future specialized paths (quantized fused, Metal GPU, etc.) — NOT as
//! a drop-in replacement for `encode()`.

const std = @import("std");
const builtin = @import("builtin");
const layer_ops = @import("fused_lewm_layer.zig");
const matmul_ops = @import("matmul.zig");
const rollout = @import("fused_lewm_rollout.zig");

const is_macos = builtin.os.tag == .macos;

// ================================================================
// Code WM specific helpers (LayerNorm with bias, exact-erf GELU)
// ================================================================

/// LayerNorm with both gamma (weight) and beta (bias). Eps matches PyTorch default.
pub fn layernorm_wb_into(
    x: [*]const f32,
    gamma: [*]const f32,
    beta: [*]const f32,
    seq_len: usize,
    hidden: usize,
    out: [*]f32,
) void {
    const eps: f32 = 1e-5;
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
        for (0..hidden) |j| dst[j] = (row[j] - mean) * inv_std * gamma[j] + beta[j];
    }
}

/// Abramowitz & Stegun 7.1.26 erf approximation (matches the Rust code_wm.rs port).
inline fn erf_f32(x: f32) f32 {
    const sign: f32 = if (x < 0) -1.0 else 1.0;
    const ax = @abs(x);
    const t = 1.0 / (1.0 + 0.3275911 * ax);
    const poly = (((1.061405429 * t - 1.453152027) * t + 1.421413741) * t - 0.284496736) * t + 0.254829592;
    const y = 1.0 - poly * t * @exp(-ax * ax);
    return sign * y;
}

/// Exact erf GELU — matches PyTorch nn.GELU() default (approximate='none').
/// y = 0.5 * x * (1 + erf(x / sqrt(2)))
pub fn gelu_erf_inplace(x: [*]f32, len: usize) void {
    const inv_sqrt2: f32 = 0.70710678118654752440;
    for (0..len) |i| {
        const v = x[i];
        x[i] = 0.5 * v * (1.0 + erf_f32(v * inv_sqrt2));
    }
}

/// Fused bias + GELU-erf (single pass).
pub fn bias_gelu_erf_inplace(x: [*]f32, bias: [*]const f32, seq_len: usize, dim: usize) void {
    const inv_sqrt2: f32 = 0.70710678118654752440;
    for (0..seq_len) |t| {
        for (0..dim) |j| {
            const idx = t * dim + j;
            const v = x[idx] + bias[j];
            x[idx] = 0.5 * v * (1.0 + erf_f32(v * inv_sqrt2));
        }
    }
}

/// Plain residual add: seq[i] += (proj[i] + bias[j]) for each element.
pub fn bias_residual(seq: [*]f32, proj: [*]const f32, bias: [*]const f32, seq_len: usize, dim: usize) void {
    for (0..seq_len) |t| {
        for (0..dim) |j| {
            const idx = t * dim + j;
            seq[idx] += proj[idx] + bias[j];
        }
    }
}

// ================================================================
// Fused Code WM encoder (weight-shared loops)
// ================================================================

/// Apply the Code WM encoder block `num_loops` times in-place.
///
/// Layout of the block's weights (single shared block, all f32):
///   norm1_w, norm1_b:        [hidden]
///   attn_in_w:               [3*hidden, hidden]  (fused QKV)
///   attn_in_b:               [3*hidden]
///   attn_out_w:              [hidden, hidden]
///   attn_out_b:              [hidden]
///   norm2_w, norm2_b:        [hidden]
///   mlp_up_w:                [mlp_hidden, hidden]
///   mlp_up_b:                [mlp_hidden]
///   mlp_down_w:              [hidden, mlp_hidden]
///   mlp_down_b:              [hidden]
///
/// Scratch buffers (caller-allocated):
///   normed_buf: [max(seq_len*hidden, seq_len*mlp_hidden)]
///   qkv_buf:    [seq_len * 3 * hidden]
///   attn_buf:   [seq_len * hidden]
///   proj_buf:   [max(seq_len*hidden, seq_len*mlp_hidden)]
///   scores_buf: [seq_len * seq_len]
///   packed_a, packed_b: GEMM packing (sizes from rollout.packBufSizes)
///
/// `seq` is modified in-place: the final [seq_len, hidden] sequence after num_loops iterations.
pub fn codeWmEncoderFused(
    seq: [*]f32,
    seq_len: usize,
    hidden: usize,
    num_heads: usize,
    mlp_hidden: usize,
    num_loops: usize,
    // Shared block weights (not per-loop — encoder weights are weight-shared)
    norm1_w: [*]const f32,
    norm1_b: [*]const f32,
    attn_in_w: [*]const f32,
    attn_in_b: [*]const f32,
    attn_out_w: [*]const f32,
    attn_out_b: [*]const f32,
    norm2_w: [*]const f32,
    norm2_b: [*]const f32,
    mlp_up_w: [*]const f32,
    mlp_up_b: [*]const f32,
    mlp_down_w: [*]const f32,
    mlp_down_b: [*]const f32,
    // Scratch buffers
    normed_buf: [*]f32,
    qkv_buf: [*]f32,
    attn_buf: [*]f32,
    proj_buf: [*]f32,
    scores_buf: [*]f32,
    packed_a: [*]f32,
    packed_b: [*]f32,
    // Mode flags (BLAS_ACCELERATE etc.)
    mode: u32,
) void {
    const head_dim = hidden / num_heads;

    var loop: usize = 0;
    while (loop < num_loops) : (loop += 1) {
        // -- a. LayerNorm with bias (pre-attn) --
        layernorm_wb_into(seq, norm1_w, norm1_b, seq_len, hidden, normed_buf);

        // -- b. Fused QKV projection: [seq_len, hidden] → [seq_len, 3*hidden] --
        rollout.gemm_dispatch(seq_len, 3 * hidden, hidden, normed_buf, attn_in_w, qkv_buf, packed_a, packed_b, mode);
        // Add bias
        for (0..seq_len) |t| {
            for (0..3 * hidden) |j| qkv_buf[t * 3 * hidden + j] += attn_in_b[j];
        }

        // -- c. Split QKV into contiguous per-head buffers and run attention --
        // normed_buf = Q, proj_buf = K, qkv_buf (tail offset) = V  (reuse buffers)
        for (0..seq_len) |t| {
            const src = t * 3 * hidden;
            const dst = t * hidden;
            @memcpy((normed_buf + dst)[0..hidden], (qkv_buf + src)[0..hidden]);            // Q
            @memcpy((proj_buf + dst)[0..hidden], (qkv_buf + src + hidden)[0..hidden]);     // K
        }
        // Copy V into attn_buf so we don't overwrite with attention output yet
        for (0..seq_len) |t| {
            const src = t * 3 * hidden;
            const dst = t * hidden;
            @memcpy((attn_buf + dst)[0..hidden], (qkv_buf + src + 2 * hidden)[0..hidden]); // V
        }
        // Attention: Q=normed_buf, K=proj_buf, V=attn_buf → output → qkv_buf (reuse)
        if (seq_len <= 16) {
            layer_ops.small_bidirectional_attention(normed_buf, proj_buf, attn_buf, qkv_buf, seq_len, num_heads, head_dim);
        } else {
            layer_ops.bidirectional_attention_dynamic(normed_buf, proj_buf, attn_buf, qkv_buf, scores_buf, seq_len, num_heads, head_dim);
        }

        // -- d. Output projection + plain residual --
        rollout.gemm_dispatch(seq_len, hidden, hidden, qkv_buf, attn_out_w, proj_buf, packed_a, packed_b, mode);
        bias_residual(seq, proj_buf, attn_out_b, seq_len, hidden);

        // -- e. LayerNorm with bias (pre-MLP) --
        layernorm_wb_into(seq, norm2_w, norm2_b, seq_len, hidden, normed_buf);

        // -- f. MLP up + GELU-erf --
        rollout.gemm_dispatch(seq_len, mlp_hidden, hidden, normed_buf, mlp_up_w, proj_buf, packed_a, packed_b, mode);
        bias_gelu_erf_inplace(proj_buf, mlp_up_b, seq_len, mlp_hidden);

        // -- g. MLP down + plain residual --
        rollout.gemm_dispatch(seq_len, hidden, mlp_hidden, proj_buf, mlp_down_w, normed_buf, packed_a, packed_b, mode);
        bias_residual(seq, normed_buf, mlp_down_b, seq_len, hidden);
    }
}
