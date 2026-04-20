//! Fused Code WM encoder — weight-shared pre-norm transformer loops.
//!
//! Vanilla pre-norm blocks: PreNorm → MHA → +res → PreNorm → MLP(GELU_erf) → +res
//! with a single weight-shared block iterated encoder_loops times.
//!
//! Optimizations:
//!   - BLAS Accelerate dispatch for all GEMMs (macOS)
//!   - Tiled SGEMM attention with per-head strided access + alpha=scale fusion
//!   - Fused QKV bias+split in one pass (4 passes → 1)
//!   - SIMD F32x4 LayerNorm (3-pass vectorized sum/var/normalize)
//!   - SIMD F32x4 erf-GELU with fused bias addition
//!   - Online softmax (single-pass max+exp+normalize)

const std = @import("std");
const builtin = @import("builtin");
const layer_ops = @import("fused_lewm_layer.zig");
const matmul_ops = @import("matmul.zig");
const rollout = @import("fused_lewm_rollout.zig");

const is_macos = builtin.os.tag == .macos;

const blas_int = c_int;
const CblasRowMajor: blas_int = 101;
const CblasNoTrans: blas_int = 111;
const CblasTrans: blas_int = 112;

extern "c" fn cblas_sgemm(
    order: blas_int,
    transA: blas_int,
    transB: blas_int,
    m: blas_int,
    n: blas_int,
    k: blas_int,
    alpha: f32,
    a: [*]const f32,
    lda: blas_int,
    b: [*]const f32,
    ldb: blas_int,
    beta: f32,
    c_out: [*]f32,
    ldc: blas_int,
) void;

const VEC_LEN: usize = 4;
const F32x4 = @Vector(VEC_LEN, f32);

// ================================================================
// SIMD LayerNorm with bias (3-pass vectorized)
// ================================================================

/// LayerNorm with gamma + beta, vectorized with F32x4.
/// 3 passes per row: sum→mean, var→inv_std, normalize+affine.
pub fn layernorm_wb_into(
    x: [*]const f32,
    gamma: [*]const f32,
    beta: [*]const f32,
    seq_len: usize,
    hidden: usize,
    out: [*]f32,
) void {
    const eps: f32 = 1e-5;
    const vec_end = hidden - (hidden % VEC_LEN);

    for (0..seq_len) |t| {
        const row = x + t * hidden;
        const dst = out + t * hidden;

        // Pass 1: sum (SIMD)
        var sum_v: F32x4 = @splat(@as(f32, 0));
        var j: usize = 0;
        while (j < vec_end) : (j += VEC_LEN) {
            sum_v += @as(F32x4, (row + j)[0..VEC_LEN].*);
        }
        var sum: f32 = @reduce(.Add, sum_v);
        while (j < hidden) : (j += 1) sum += row[j];
        const mean = sum / @as(f32, @floatFromInt(hidden));

        // Pass 2: variance (SIMD)
        const mean_v: F32x4 = @splat(mean);
        var var_v: F32x4 = @splat(@as(f32, 0));
        j = 0;
        while (j < vec_end) : (j += VEC_LEN) {
            const d = @as(F32x4, (row + j)[0..VEC_LEN].*) - mean_v;
            var_v = @mulAdd(F32x4, d, d, var_v);
        }
        var var_sum: f32 = @reduce(.Add, var_v);
        while (j < hidden) : (j += 1) {
            const d = row[j] - mean;
            var_sum += d * d;
        }
        const inv_std = 1.0 / @sqrt(var_sum / @as(f32, @floatFromInt(hidden)) + eps);

        // Pass 3: normalize + affine (SIMD)
        const inv_v: F32x4 = @splat(inv_std);
        j = 0;
        while (j < vec_end) : (j += VEC_LEN) {
            const val = (@as(F32x4, (row + j)[0..VEC_LEN].*) - mean_v) * inv_v;
            const g: F32x4 = @as(F32x4, (gamma + j)[0..VEC_LEN].*);
            const b: F32x4 = @as(F32x4, (beta + j)[0..VEC_LEN].*);
            (dst + j)[0..VEC_LEN].* = @mulAdd(F32x4, val, g, b);
        }
        while (j < hidden) : (j += 1) {
            dst[j] = (row[j] - mean) * inv_std * gamma[j] + beta[j];
        }
    }
}

// ================================================================
// SIMD erf + GELU
// ================================================================

/// Scalar erf — Abramowitz & Stegun 7.1.26.
inline fn erf_f32(x: f32) f32 {
    const sign: f32 = if (x < 0) -1.0 else 1.0;
    const ax = @abs(x);
    const t = 1.0 / (1.0 + 0.3275911 * ax);
    const poly = (((1.061405429 * t - 1.453152027) * t + 1.421413741) * t - 0.284496736) * t + 0.254829592;
    const y = 1.0 - poly * t * @exp(-ax * ax);
    return sign * y;
}

/// SIMD erf on F32x4 — same polynomial, 4 lanes.
inline fn erf_f32x4(x: F32x4) F32x4 {
    const zero: F32x4 = @splat(0);
    const one: F32x4 = @splat(1.0);
    const neg_one: F32x4 = @splat(-1.0);
    const sign: F32x4 = @select(f32, x < zero, neg_one, one);
    const ax = @abs(x);
    const t = one / (one + @as(F32x4, @splat(0.3275911)) * ax);
    // Horner: ((((a5*t + a4)*t + a3)*t + a2)*t + a1)
    var poly = @as(F32x4, @splat(1.061405429)) * t + @as(F32x4, @splat(-1.453152027));
    poly = poly * t + @as(F32x4, @splat(1.421413741));
    poly = poly * t + @as(F32x4, @splat(-0.284496736));
    poly = poly * t + @as(F32x4, @splat(0.254829592));
    const y = one - poly * t * @exp(-ax * ax);
    return sign * y;
}

/// GELU-erf: y = 0.5 * x * (1 + erf(x / sqrt(2)))
pub fn gelu_erf_inplace(x: [*]f32, len: usize) void {
    const inv_sqrt2: F32x4 = @splat(0.70710678118654752440);
    const half: F32x4 = @splat(0.5);
    const one: F32x4 = @splat(1.0);
    const vec_end = len - (len % VEC_LEN);
    var i: usize = 0;
    while (i < vec_end) : (i += VEC_LEN) {
        const v: F32x4 = @as(F32x4, (x + i)[0..VEC_LEN].*);
        (x + i)[0..VEC_LEN].* = half * v * (one + erf_f32x4(v * inv_sqrt2));
    }
    while (i < len) : (i += 1) {
        const v = x[i];
        x[i] = 0.5 * v * (1.0 + erf_f32(v * 0.70710678118654752440));
    }
}

/// Fused bias + GELU-erf in one pass, SIMD vectorized.
pub fn bias_gelu_erf_inplace(x: [*]f32, bias: [*]const f32, seq_len: usize, dim: usize) void {
    const inv_sqrt2: F32x4 = @splat(0.70710678118654752440);
    const half: F32x4 = @splat(0.5);
    const one: F32x4 = @splat(1.0);
    const vec_end = dim - (dim % VEC_LEN);

    for (0..seq_len) |t| {
        const row = x + t * dim;
        var j: usize = 0;
        while (j < vec_end) : (j += VEC_LEN) {
            const v: F32x4 = @as(F32x4, (row + j)[0..VEC_LEN].*) + @as(F32x4, (bias + j)[0..VEC_LEN].*);
            (row + j)[0..VEC_LEN].* = half * v * (one + erf_f32x4(v * inv_sqrt2));
        }
        while (j < dim) : (j += 1) {
            const v = row[j] + bias[j];
            row[j] = 0.5 * v * (1.0 + erf_f32(v * 0.70710678118654752440));
        }
    }
}

/// Plain residual: seq[i] += proj[i] + bias[j], SIMD vectorized.
pub fn bias_residual(seq: [*]f32, proj: [*]const f32, bias: [*]const f32, seq_len: usize, dim: usize) void {
    const vec_end = dim - (dim % VEC_LEN);
    for (0..seq_len) |t| {
        const s = seq + t * dim;
        const p = proj + t * dim;
        var j: usize = 0;
        while (j < vec_end) : (j += VEC_LEN) {
            (s + j)[0..VEC_LEN].* = @as(F32x4, (s + j)[0..VEC_LEN].*) + @as(F32x4, (p + j)[0..VEC_LEN].*) + @as(F32x4, (bias + j)[0..VEC_LEN].*);
        }
        while (j < dim) : (j += 1) {
            s[j] += p[j] + bias[j];
        }
    }
}

// ================================================================
// Strided GEMM dispatch (custom lda/ldb/ldc for attention)
// ================================================================

/// Strided GEMM with explicit strides and alpha/beta.
/// C = alpha * op(A) @ op(B) + beta * C
fn gemm_strided(
    m: usize,
    n: usize,
    k: usize,
    alpha: f32,
    a: [*]const f32,
    lda: usize,
    trans_a: bool,
    b: [*]const f32,
    ldb: usize,
    trans_b: bool,
    beta: f32,
    c_out: [*]f32,
    ldc: usize,
    packed_a: [*]f32,
    packed_b: [*]f32,
    mode: u32,
) void {
    if (comptime is_macos) {
        if (rollout.hasFlag(mode, rollout.BLAS_ACCELERATE)) {
            cblas_sgemm(
                CblasRowMajor,
                if (trans_a) CblasTrans else CblasNoTrans,
                if (trans_b) CblasTrans else CblasNoTrans,
                @intCast(m),
                @intCast(n),
                @intCast(k),
                alpha,
                a,
                @intCast(lda),
                b,
                @intCast(ldb),
                beta,
                c_out,
                @intCast(ldc),
            );
            return;
        }
    }
    // Fallback: zero C if beta=0 (the common case), then tiled SGEMM with alpha=1.
    // Note: sgemmTiled always accumulates (C += A@B), so we zero first.
    // Alpha != 1.0 is handled as a post-scale when not using BLAS.
    if (beta == 0.0) {
        for (0..m) |i| @memset((c_out + i * ldc)[0..n], 0);
    }
    matmul_ops.sgemmTiled(m, n, k, a, lda, trans_a, b, ldb, trans_b, c_out, ldc, packed_a, packed_b);
    if (alpha != 1.0) {
        for (0..m) |i| {
            const row = c_out + i * ldc;
            for (0..n) |j| row[j] *= alpha;
        }
    }
}

// ================================================================
// Tiled bidirectional attention with alpha=scale fusion
// ================================================================

/// Fast bidirectional multi-head attention using tiled GEMM / BLAS.
/// Fuses the 1/sqrt(d) scale into the Q@K^T GEMM alpha parameter
/// (saves one pass over seq_len² scores).
///
/// Q, K, V: [seq_len, hidden]  (all heads interleaved)
/// out:     [seq_len, hidden]
/// scores_buf: [seq_len * seq_len] scratch (reused per head)
pub fn fast_bidirectional_attention(
    q: [*]const f32,
    k_in: [*]const f32,
    v_in: [*]const f32,
    out: [*]f32,
    scores_buf: [*]f32,
    seq_len: usize,
    num_heads: usize,
    head_dim: usize,
    packed_a: [*]f32,
    packed_b: [*]f32,
    mode: u32,
) void {
    const hidden = num_heads * head_dim;
    const scale: f32 = 1.0 / @sqrt(@as(f32, @floatFromInt(head_dim)));

    for (0..num_heads) |head| {
        const h_off = head * head_dim;

        // S[S,S] = (1/√d) * Q_h[S,d] @ K_h^T[d,S]  — scale fused into alpha
        gemm_strided(
            seq_len, seq_len, head_dim,
            scale,                                // alpha = 1/√d (fused scale)
            q + h_off, hidden, false,
            k_in + h_off, hidden, true,
            0.0,                                  // beta = 0 (overwrite)
            scores_buf, seq_len,
            packed_a, packed_b, mode,
        );

        // Online softmax per row (single-pass max + exp + normalize)
        for (0..seq_len) |qi| {
            const rb = qi * seq_len;
            var mx: f32 = -std.math.inf(f32);
            var se: f32 = 0.0;
            for (0..seq_len) |ki| {
                const x = scores_buf[rb + ki];
                if (x > mx) {
                    se = se * @exp(mx - x) + 1.0;
                    mx = x;
                } else {
                    se += @exp(x - mx);
                }
            }
            const inv = 1.0 / se;
            for (0..seq_len) |ki| {
                scores_buf[rb + ki] = @exp(scores_buf[rb + ki] - mx) * inv;
            }
        }

        // O_h[S,d] = S[S,S] @ V_h[S,d]
        gemm_strided(
            seq_len, head_dim, seq_len,
            1.0,                                  // alpha = 1
            scores_buf, seq_len, false,
            v_in + h_off, hidden, false,
            0.0,                                  // beta = 0
            out + h_off, hidden,
            packed_a, packed_b, mode,
        );
    }
}

// ================================================================
// Fused QKV bias + split (4 passes → 1)
// ================================================================

/// Read fused QKV [seq_len, 3*hidden], add bias, split into separate
/// Q/K/V buffers [seq_len, hidden] — all in a single pass.
fn qkv_bias_split(
    qkv: [*]const f32,
    bias: [*]const f32,
    q_out: [*]f32,
    k_out: [*]f32,
    v_out: [*]f32,
    seq_len: usize,
    hidden: usize,
) void {
    const h3 = 3 * hidden;
    const vec_end = hidden - (hidden % VEC_LEN);

    for (0..seq_len) |t| {
        const src = qkv + t * h3;
        const dst_q = q_out + t * hidden;
        const dst_k = k_out + t * hidden;
        const dst_v = v_out + t * hidden;

        // Q: src[0..hidden] + bias[0..hidden]
        var j: usize = 0;
        while (j < vec_end) : (j += VEC_LEN) {
            (dst_q + j)[0..VEC_LEN].* = @as(F32x4, (src + j)[0..VEC_LEN].*) + @as(F32x4, (bias + j)[0..VEC_LEN].*);
        }
        while (j < hidden) : (j += 1) dst_q[j] = src[j] + bias[j];

        // K: src[hidden..2*hidden] + bias[hidden..2*hidden]
        j = 0;
        while (j < vec_end) : (j += VEC_LEN) {
            (dst_k + j)[0..VEC_LEN].* = @as(F32x4, (src + hidden + j)[0..VEC_LEN].*) + @as(F32x4, (bias + hidden + j)[0..VEC_LEN].*);
        }
        while (j < hidden) : (j += 1) dst_k[j] = src[hidden + j] + bias[hidden + j];

        // V: src[2*hidden..3*hidden] + bias[2*hidden..3*hidden]
        j = 0;
        while (j < vec_end) : (j += VEC_LEN) {
            (dst_v + j)[0..VEC_LEN].* = @as(F32x4, (src + 2 * hidden + j)[0..VEC_LEN].*) + @as(F32x4, (bias + 2 * hidden + j)[0..VEC_LEN].*);
        }
        while (j < hidden) : (j += 1) dst_v[j] = src[2 * hidden + j] + bias[2 * hidden + j];
    }
}

// ================================================================
// Fused Code WM encoder (weight-shared loops)
// ================================================================

/// Apply the Code WM encoder block `num_loops` times in-place.
///
/// Scratch buffers (caller-allocated):
///   normed_buf: [max(seq_len*hidden, seq_len*mlp_hidden)]
///   qkv_buf:    [seq_len * 3 * hidden]
///   attn_buf:   [seq_len * hidden]
///   proj_buf:   [max(seq_len*hidden, seq_len*mlp_hidden)]
///   scores_buf: [seq_len * seq_len]
///   packed_a, packed_b: GEMM packing (sizes from rollout.packBufSizes)
pub fn codeWmEncoderFused(
    seq: [*]f32,
    seq_len: usize,
    hidden: usize,
    num_heads: usize,
    mlp_hidden: usize,
    num_loops: usize,
    // Shared block weights
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
        // -- a. LayerNorm with bias (pre-attn) -- [SIMD vectorized]
        layernorm_wb_into(seq, norm1_w, norm1_b, seq_len, hidden, normed_buf);

        // -- b. Fused QKV projection: [seq_len, hidden] → [seq_len, 3*hidden] --
        rollout.gemm_dispatch(seq_len, 3 * hidden, hidden, normed_buf, attn_in_w, qkv_buf, packed_a, packed_b, mode);

        // -- c. Fused QKV bias + split (1 pass instead of 4) -- [SIMD vectorized]
        //    Q → normed_buf, K → proj_buf, V → attn_buf
        qkv_bias_split(qkv_buf, attn_in_b, normed_buf, proj_buf, attn_buf, seq_len, hidden);

        // -- d. Attention with fused scale (alpha=1/√d into GEMM) --
        fast_bidirectional_attention(
            normed_buf, proj_buf, attn_buf, qkv_buf, scores_buf,
            seq_len, num_heads, head_dim,
            packed_a, packed_b, mode,
        );

        // -- e. Output projection + residual -- [SIMD vectorized residual]
        rollout.gemm_dispatch(seq_len, hidden, hidden, qkv_buf, attn_out_w, proj_buf, packed_a, packed_b, mode);
        bias_residual(seq, proj_buf, attn_out_b, seq_len, hidden);

        // -- f. LayerNorm with bias (pre-MLP) -- [SIMD vectorized]
        layernorm_wb_into(seq, norm2_w, norm2_b, seq_len, hidden, normed_buf);

        // -- g. MLP up + fused bias+GELU-erf -- [SIMD vectorized]
        rollout.gemm_dispatch(seq_len, mlp_hidden, hidden, normed_buf, mlp_up_w, proj_buf, packed_a, packed_b, mode);
        bias_gelu_erf_inplace(proj_buf, mlp_up_b, seq_len, mlp_hidden);

        // -- h. MLP down + residual -- [SIMD vectorized residual]
        rollout.gemm_dispatch(seq_len, hidden, mlp_hidden, proj_buf, mlp_down_w, normed_buf, packed_a, packed_b, mode);
        bias_residual(seq, normed_buf, mlp_down_b, seq_len, hidden);
    }
}
