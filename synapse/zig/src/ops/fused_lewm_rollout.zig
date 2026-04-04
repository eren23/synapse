//! Fused LEWM rollout — process all N rollout steps in one pass per layer.
//!
//! Instead of calling predict_next N times (each with seq_len=3 through 6 layers =
//! 1,200 tiny M=3 GEMMs), this batches everything: seq_len=N*3, through 6 layers =
//! 24 M=N*3 GEMMs. Dramatic speedup from better GEMM tiling utilization.
//!
//! Uses a u32 bitfield flag system for dispatch configuration:
//!   FUSED_ROLLOUT | ESP_FUSED | PREPACK_WEIGHTS | BLAS_ACCELERATE | SHARED_ADALN | QUANT_*

const std = @import("std");
const builtin = @import("builtin");
const layer_ops = @import("fused_lewm_layer.zig");
const matmul_ops = @import("matmul.zig");

const is_macos = builtin.os.tag == .macos;

const CblasRowMajor: c_int = 101;
const CblasNoTrans: c_int = 111;
const CblasTrans: c_int = 112;

extern "c" fn cblas_sgemm(
    order: c_int,
    transA: c_int,
    transB: c_int,
    m: c_int,
    n: c_int,
    k: c_int,
    alpha: f32,
    a: [*]const f32,
    lda: c_int,
    b: [*]const f32,
    ldb: c_int,
    beta: f32,
    c_out: [*]f32,
    ldc: c_int,
) void;

// ================================================================
// Flag constants (u32 bitfield)
// ================================================================

pub const FUSED_ROLLOUT: u32 = 0x01;
pub const ESP_FUSED: u32 = 0x02;
pub const PREPACK_WEIGHTS: u32 = 0x04;
pub const BLAS_ACCELERATE: u32 = 0x08;
pub const SHARED_ADALN: u32 = 0x10;
pub const QUANT_INT8: u32 = 0x20;
pub const QUANT_Q4: u32 = 0x40;
pub const MODE_DEFAULT: u32 = 0x00;
pub const MODE_PORTABLE: u32 = FUSED_ROLLOUT | ESP_FUSED | PREPACK_WEIGHTS | SHARED_ADALN;
pub const MODE_FAST_MAC: u32 = MODE_PORTABLE | BLAS_ACCELERATE;

// ================================================================
// Helpers
// ================================================================

pub fn hasFlag(mode: u32, flag: u32) bool {
    return (mode & flag) != 0;
}

/// GEMM dispatch for fused rollout — handles M>16 via tiled SGEMM with caller-provided
/// packing buffers. Falls through to per-row GEMV for M<=16 (no packing needed).
/// BLAS/Accelerate and quantized paths will be wired in later tasks.
///
/// C[m,n] = A[m,k] @ B[n,k]^T  (B stored row-major [n,k], transposed internally).
pub fn gemm_dispatch(
    m: usize,
    n: usize,
    k: usize,
    a: [*]const f32,
    b: [*]const f32,
    c: [*]f32,
    packed_a: [*]f32,
    packed_b: [*]f32,
    mode: u32,
) void {
    // BLAS Accelerate path (macOS only)
    if (comptime is_macos) {
        if (hasFlag(mode, BLAS_ACCELERATE)) {
            cblas_sgemm(
                CblasRowMajor,
                CblasNoTrans,
                CblasTrans,
                @intCast(m),
                @intCast(n),
                @intCast(k),
                1.0,
                a,
                @intCast(k),
                b,
                @intCast(k),
                0.0,
                c,
                @intCast(n),
            );
            return;
        }
    }
    // Zero output before tiled SGEMM (required: sgemmTiled accumulates into C)
    const total = m * n;
    for (0..total) |i| c[i] = 0;
    matmul_ops.sgemmTiled(m, n, k, a, k, false, b, k, true, c, n, packed_a, packed_b);
}

/// Compute the packing buffer sizes needed for sgemmTiled given max dimensions.
/// Returns .{ packed_a_size, packed_b_size } in number of f32 elements.
pub fn packBufSizes(max_m: usize, max_n: usize, max_k: usize) struct { a: usize, b: usize } {
    const MR = matmul_ops.MR;
    const NR = matmul_ops.NR;
    const MC = matmul_ops.MC;
    const KC = matmul_ops.KC;
    const NC = matmul_ops.NC;

    const mc = @min(MC, max_m);
    const kc = @min(KC, max_k);
    const nc = @min(NC, max_n);

    const packed_a_size = ((mc + MR - 1) / MR) * MR * kc;
    const packed_b_size = ((nc + NR - 1) / NR) * NR * kc;
    return .{ .a = packed_a_size, .b = packed_b_size };
}

// ================================================================
// Core fused rollout
// ================================================================

/// Process all rollout steps through all layers in a single batched pass.
///
/// seq:          [num_steps * 3 * hidden] in-place sequence buffer
/// conditioning: [hidden] shared conditioning vector
/// num_steps:    number of rollout steps (e.g. 50)
/// hidden:       hidden dimension
/// num_heads:    number of attention heads
/// inner_dim:    attention inner dimension (num_heads * head_dim)
/// inter:        FFN intermediate dimension
/// num_layers:   number of transformer layers (e.g. 6)
///
/// Per-layer weight arrays: each is [num_layers] pointers to the weight for that layer.
///   adaln_ws, adaln_bs, attn_norm_ws, to_qkvs, attn_out_ws, attn_out_bs,
///   mlp_norm_ws, mlp_up_ws, mlp_up_bs, mlp_down_ws, mlp_down_bs
///
/// Scratch buffers:
///   mod_buf:    [6 * hidden]
///   normed_buf: [max(seq_len * hidden, seq_len * inter, seq_len * inner_dim)]
///   qkv_buf:    [seq_len * 3 * inner_dim]
///   attn_buf:   [seq_len * inner_dim]
///   proj_buf:   [max(seq_len * hidden, seq_len * inter, seq_len * inner_dim)]
///   scores_buf: [seq_len * seq_len]  (for dynamic attention)
///   packed_a:   [packBufSizes(...).a]  (GEMM packing scratch)
///   packed_b:   [packBufSizes(...).b]  (GEMM packing scratch)
///
/// mode: u32 bitfield of FUSED_ROLLOUT | ESP_FUSED | SHARED_ADALN | ...
pub fn lewmRolloutFused(
    seq: [*]f32,
    conditioning: [*]const f32,
    num_steps: usize,
    hidden: usize,
    num_heads: usize,
    inner_dim: usize,
    inter: usize,
    num_layers: usize,
    // Per-layer weight pointer arrays
    adaln_ws: [*]const [*]const f32,
    adaln_bs: [*]const [*]const f32,
    attn_norm_ws: [*]const [*]const f32,
    to_qkvs: [*]const [*]const f32,
    attn_out_ws: [*]const [*]const f32,
    attn_out_bs: [*]const [*]const f32,
    mlp_norm_ws: [*]const [*]const f32,
    mlp_up_ws: [*]const [*]const f32,
    mlp_up_bs: [*]const [*]const f32,
    mlp_down_ws: [*]const [*]const f32,
    mlp_down_bs: [*]const [*]const f32,
    // Scratch buffers
    mod_buf: [*]f32,
    normed_buf: [*]f32,
    qkv_buf: [*]f32,
    attn_buf: [*]f32,
    proj_buf: [*]f32,
    scores_buf: [*]f32,
    packed_a: [*]f32,
    packed_b: [*]f32,
    // Mode flags
    mode: u32,
) void {
    const seq_len = num_steps * 3;
    const mod_dim = 6 * hidden;
    const use_esp = hasFlag(mode, ESP_FUSED);
    const head_dim = inner_dim / num_heads;

    for (0..num_layers) |layer| {
        // -- a. adaLN modulation (GEMV: 1 x 6*hidden from hidden) --
        // SHARED_ADALN: computed once per layer (conditioning is shared across steps).
        // Both the SHARED_ADALN path and the non-SHARED_ADALN path do the same thing
        // here because conditioning is already shared. Flag kept for API consistency.
        gemm_dispatch(1, mod_dim, hidden, conditioning, adaln_ws[layer], mod_buf, packed_a, packed_b, mode);
        // Add bias
        const adaln_bias = adaln_bs[layer];
        for (0..mod_dim) |j| mod_buf[j] += adaln_bias[j];

        // -- b. LayerNorm + modulate for FULL seq_len --
        layer_ops.layernorm_into(seq, attn_norm_ws[layer], seq_len, hidden, normed_buf);
        layer_ops.modulate_inplace(normed_buf, mod_buf, mod_buf + hidden, seq_len, hidden);

        // -- c. QKV projection (big GEMM: seq_len x 3*inner_dim from hidden) --
        gemm_dispatch(seq_len, 3 * inner_dim, hidden, normed_buf, to_qkvs[layer], qkv_buf, packed_a, packed_b, mode);

        // -- d. Split QKV and run attention --
        for (0..seq_len) |t| {
            const qkv_off = t * 3 * inner_dim;
            const off = t * inner_dim;
            @memcpy((normed_buf + off)[0..inner_dim], (qkv_buf + qkv_off)[0..inner_dim]); // Q
            @memcpy((proj_buf + off)[0..inner_dim], (qkv_buf + qkv_off + inner_dim)[0..inner_dim]); // K
            @memcpy((qkv_buf + off)[0..inner_dim], (qkv_buf + qkv_off + 2 * inner_dim)[0..inner_dim]); // V
        }
        // Dynamic attention for large seq_len, small for <=16
        if (seq_len <= 16) {
            layer_ops.small_bidirectional_attention(normed_buf, proj_buf, qkv_buf, attn_buf, seq_len, num_heads, head_dim);
        } else {
            layer_ops.bidirectional_attention_dynamic(normed_buf, proj_buf, qkv_buf, attn_buf, scores_buf, seq_len, num_heads, head_dim);
        }

        // -- e. Output projection + gated residual --
        gemm_dispatch(seq_len, hidden, inner_dim, attn_buf, attn_out_ws[layer], proj_buf, packed_a, packed_b, mode);
        if (use_esp) {
            layer_ops.bias_gated_residual(seq, mod_buf + 2 * hidden, proj_buf, attn_out_bs[layer], seq_len, hidden);
        } else {
            layer_ops.add_bias(proj_buf, attn_out_bs[layer], seq_len, hidden);
            layer_ops.gated_residual(seq, mod_buf + 2 * hidden, proj_buf, seq_len, hidden);
        }

        // -- f. FFN norm + modulate --
        layer_ops.layernorm_into(seq, mlp_norm_ws[layer], seq_len, hidden, normed_buf);
        layer_ops.modulate_inplace(normed_buf, mod_buf + 3 * hidden, mod_buf + 4 * hidden, seq_len, hidden);

        // -- g. FFN up + GELU --
        gemm_dispatch(seq_len, inter, hidden, normed_buf, mlp_up_ws[layer], proj_buf, packed_a, packed_b, mode);
        if (use_esp) {
            layer_ops.bias_gelu_inplace(proj_buf, mlp_up_bs[layer], seq_len, inter);
        } else {
            layer_ops.add_bias(proj_buf, mlp_up_bs[layer], seq_len, inter);
            layer_ops.gelu_inplace(proj_buf, seq_len * inter);
        }

        // -- h. FFN down + gated residual --
        gemm_dispatch(seq_len, hidden, inter, proj_buf, mlp_down_ws[layer], normed_buf, packed_a, packed_b, mode);
        if (use_esp) {
            layer_ops.bias_gated_residual(seq, mod_buf + 5 * hidden, normed_buf, mlp_down_bs[layer], seq_len, hidden);
        } else {
            layer_ops.add_bias(normed_buf, mlp_down_bs[layer], seq_len, hidden);
            layer_ops.gated_residual(seq, mod_buf + 5 * hidden, normed_buf, seq_len, hidden);
        }
    }
}
