//! SIMD-vectorized RWKV-7 WKV recurrence kernel.
//!
//! Per head, per token:
//!   ka[j] = k[j] * a[j]
//!   sdk[d] = sum_l(state[d,l] * k[l])        — matvec: state @ k
//!   state[d,j] = w[d]*state[d,j] - sdk[d]*ka[j] + k[d]*v[j]
//!   output[d] = sum_j(state[d,j] * r[j])     — matvec: state @ r
//!
//! The hot path is O(N²) per token per head (N=head_size, typically 64).
//! State matrix is N×N = 4096 floats = 16KB → fits in L1 cache.
//! Vectorized over j (columns) using F32x4.

const std = @import("std");

const VEC_LEN = 4;
const F32x4 = @Vector(VEC_LEN, f32);

// ============================================================
// Public API
// ============================================================

/// Single-step WKV7 for one head.
///
/// - r:     [N] receptance (query)
/// - k:     [N] normalized key (L2-normed with k_k)
/// - v:     [N] value
/// - w:     [N] per-channel decay in (0,1), NOT log space
/// - a:     [N] alpha gate in (0,1)
/// - state: [N*N] row-major, updated in place
/// - out:   [N] output, written
/// - n:     head_size
pub fn wkv7Step(
    r: [*]const f32,
    k: [*]const f32,
    v: [*]const f32,
    w: [*]const f32,
    a: [*]const f32,
    state: [*]f32,
    out: [*]f32,
    n: usize,
) void {
    // 1. Precompute ka[j] = k[j] * a[j]
    // Stack-allocate for typical head_size <= 128
    var ka_buf: [128]f32 = undefined;
    const ka = ka_buf[0..n];
    for (0..n) |j| {
        ka[j] = k[j] * a[j];
    }

    // 2. Compute sdk[d] = state[d,:] · k   (matvec)
    var sdk_buf: [128]f32 = undefined;
    const sdk = sdk_buf[0..n];
    for (0..n) |d| {
        const row = state + d * n;
        var dot: f32 = 0.0;

        var j: usize = 0;
        while (j + VEC_LEN <= n) : (j += VEC_LEN) {
            const sv: F32x4 = row[j..][0..VEC_LEN].*;
            const kv: F32x4 = k[j..][0..VEC_LEN].*;
            dot += @reduce(.Add, sv * kv);
        }
        while (j < n) : (j += 1) {
            dot += row[j] * k[j];
        }
        sdk[d] = dot;
    }

    // 3. State update + output in one pass over rows
    for (0..n) |d| {
        const row = state + d * n;
        const w_d = w[d];
        const neg_sdk: F32x4 = @splat(-sdk[d]);
        const w_splat: F32x4 = @splat(w_d);
        const kd_splat: F32x4 = @splat(k[d]);
        var o_acc: f32 = 0.0;

        // Vectorized over columns
        var j: usize = 0;
        while (j + VEC_LEN <= n) : (j += VEC_LEN) {
            const h: F32x4 = row[j..][0..VEC_LEN].*;
            const ka_v: F32x4 = ka[j..][0..VEC_LEN].*;
            const vv: F32x4 = v[j..][0..VEC_LEN].*;
            const rv: F32x4 = r[j..][0..VEC_LEN].*;

            // state[d,j] = w[d]*h + (-sdk[d])*ka[j] + k[d]*v[j]
            const h_new = w_splat * h + neg_sdk * ka_v + kd_splat * vv;
            row[j..][0..VEC_LEN].* = h_new;

            // output[d] += state[d,j] * r[j]
            o_acc += @reduce(.Add, h_new * rv);
        }

        // Scalar tail
        while (j < n) : (j += 1) {
            const h_new = w_d * row[j] - sdk[d] * ka[j] + k[d] * v[j];
            row[j] = h_new;
            o_acc += h_new * r[j];
        }

        out[d] = o_acc;
    }
}

/// Sequential WKV7 for a sequence of tokens on one head.
///
/// All per-token inputs are flat arrays of shape [seq_len, N].
pub fn wkv7Seq(
    r: [*]const f32,
    k: [*]const f32,
    v: [*]const f32,
    w: [*]const f32,
    a: [*]const f32,
    state: [*]f32,
    out: [*]f32,
    seq_len: usize,
    n: usize,
) void {
    for (0..seq_len) |t| {
        wkv7Step(
            r + t * n,
            k + t * n,
            v + t * n,
            w + t * n,
            a + t * n,
            state,
            out + t * n,
            n,
        );
    }
}
