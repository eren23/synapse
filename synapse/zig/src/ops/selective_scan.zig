//! SIMD-vectorized Mamba selective scan kernel.
//!
//! Recurrence per channel i, state dimension j:
//!   A = -exp(a_log[i,j])
//!   h[i,j] = exp(delta[i] * A) * h[i,j] + delta[i] * B[j] * x[i]
//!   y[i]   = sum_j(C[j] * h[i,j]) + D[i] * x[i]
//!
//! Vectorized over j (d_state dimension, typically 16) using F32x4.

const std = @import("std");

const VEC_LEN = 4;
const F32x4 = @Vector(VEC_LEN, f32);

// ============================================================
// Public API
// ============================================================

/// Single-step selective scan for all channels.
///
/// Processes one token: updates `state` in-place, writes output to `y`.
/// - x:      [d_inner]
/// - delta:  [d_inner] (positive, after softplus)
/// - a_log:  [d_inner * d_state] (raw log values; A = -exp(a_log))
/// - b:      [d_state] (input-dependent)
/// - c:      [d_state] (input-dependent)
/// - d_skip: [d_inner] (skip connection)
/// - state:  [d_inner * d_state] (updated in place)
/// - y:      [d_inner] (output, written)
pub fn selectiveScanStep(
    x: [*]const f32,
    delta: [*]const f32,
    a_log: [*]const f32,
    b: [*]const f32,
    c: [*]const f32,
    d_skip: [*]const f32,
    state: [*]f32,
    y: [*]f32,
    d_inner: usize,
    d_state: usize,
) void {
    // For each channel i, process all d_state elements vectorized
    for (0..d_inner) |i| {
        const dt_i = delta[i];
        const x_i = x[i];
        const dt_x: F32x4 = @splat(dt_i * x_i);
        const dt_splat: F32x4 = @splat(dt_i);

        var yi: f32 = 0.0;
        const state_row = state + i * d_state;
        const a_row = a_log + i * d_state;

        // Vectorized inner loop over d_state
        var j: usize = 0;
        while (j + VEC_LEN <= d_state) : (j += VEC_LEN) {
            // Load state, a_log, b, c
            const h: F32x4 = state_row[j..][0..VEC_LEN].*;
            const al: F32x4 = a_row[j..][0..VEC_LEN].*;
            const bv: F32x4 = b[j..][0..VEC_LEN].*;
            const cv: F32x4 = c[j..][0..VEC_LEN].*;

            // A = -exp(a_log), discretized = exp(delta * A) = exp(-delta * exp(a_log))
            const a_disc = @exp(-dt_splat * @exp(al));

            // h_new = a_disc * h + delta * b * x
            const h_new = a_disc * h + dt_x * bv;
            state_row[j..][0..VEC_LEN].* = h_new;

            // y[i] += sum(c * h_new)
            const contrib = cv * h_new;
            yi += @reduce(.Add, contrib);
        }

        // Scalar tail
        while (j < d_state) : (j += 1) {
            const a = -@exp(a_row[j]);
            const a_disc = @exp(dt_i * a);
            const h_new = a_disc * state_row[j] + dt_i * b[j] * x_i;
            state_row[j] = h_new;
            yi += c[j] * h_new;
        }

        // Skip connection
        y[i] = yi + d_skip[i] * x_i;
    }
}

/// Sequential selective scan for a sequence of tokens (prefill).
///
/// Calls selectiveScanStep for each token in the sequence.
/// - xs:     [seq_len * d_inner]
/// - deltas: [seq_len * d_inner]
/// - a_log:  [d_inner * d_state] (shared across time)
/// - bs:     [seq_len * d_state]
/// - cs:     [seq_len * d_state]
/// - d_skip: [d_inner]
/// - state:  [d_inner * d_state] (updated in place)
/// - ys:     [seq_len * d_inner] (output)
pub fn selectiveScanSeq(
    xs: [*]const f32,
    deltas: [*]const f32,
    a_log: [*]const f32,
    bs: [*]const f32,
    cs: [*]const f32,
    d_skip: [*]const f32,
    state: [*]f32,
    ys: [*]f32,
    seq_len: usize,
    d_inner: usize,
    d_state: usize,
) void {
    for (0..seq_len) |t| {
        selectiveScanStep(
            xs + t * d_inner,
            deltas + t * d_inner,
            a_log,
            bs + t * d_state,
            cs + t * d_state,
            d_skip,
            state,
            ys + t * d_inner,
            d_inner,
            d_state,
        );
    }
}
