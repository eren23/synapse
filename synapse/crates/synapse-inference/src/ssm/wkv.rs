//! WKV7 kernel for RWKV-7 "Goose" inference.
//!
//! Implements the RWKV-7 linear attention recurrence where the state matrix
//! accumulates outer products of keys and values with learnable decay and
//! a feedback term. This is NOT standard softmax attention.
//!
//! Recurrence per head per token:
//!   vk = outer(k, v)                    — new info: value-key outer product
//!   ab = -outer(kk, kk * a)             — feedback: state self-correction
//!   state = w * state + state @ ab + vk — decay + feedback + new info
//!   output = state @ r                  — read state with receptance

/// RWKV-7 single-step WKV computation for one attention head.
///
/// # Parameters
/// - `r`     — receptance `[head_size]` (query vector)
/// - `k`     — normalized key `[head_size]` (already L2-normed with k_k)
/// - `v`     — value `[head_size]`
/// - `w`     — per-channel decay `[head_size]` (already in `(0,1)` range, NOT log space)
/// - `a`     — alpha gate `[head_size]` (in `(0,1)` range, controls feedback strength)
/// - `state` — `[head_size * head_size]`, updated in place
/// - `head_size` — dimension per head
///
/// # Returns
/// Output vector `[head_size]`.
pub fn wkv7_step(
    r: &[f32],
    k: &[f32],
    v: &[f32],
    w: &[f32],
    a: &[f32],
    state: &mut [f32],
    head_size: usize,
) -> Vec<f32> {
    let n = head_size;

    // Zig SIMD fast path
    #[cfg(feature = "zig-ffi")]
    {
        let mut output = vec![0.0f32; n];
        unsafe {
            synapse_sys::syn_wkv7_step(
                r.as_ptr(), k.as_ptr(), v.as_ptr(),
                w.as_ptr(), a.as_ptr(),
                state.as_mut_ptr(), output.as_mut_ptr(),
                n,
            );
        }
        return output;
    }

    // Pure-Rust fallback
    #[cfg(not(feature = "zig-ffi"))]
    {
        let ka: Vec<f32> = (0..n).map(|j| k[j] * a[j]).collect();

        let mut state_dot_k = vec![0.0f32; n];
        for d in 0..n {
            let mut dot = 0.0f32;
            for l in 0..n {
                dot += state[d * n + l] * k[l];
            }
            state_dot_k[d] = dot;
        }

        for d in 0..n {
            let w_d = w[d];
            let sdk = state_dot_k[d];
            let k_d = k[d];
            for j in 0..n {
                state[d * n + j] = w_d * state[d * n + j]
                    - sdk * ka[j]
                    + k_d * v[j];
            }
        }

        let mut output = vec![0.0f32; n];
        for d in 0..n {
            let mut sum = 0.0f32;
            for j in 0..n {
                sum += state[d * n + j] * r[j];
            }
            output[d] = sum;
        }

        output
    }
}

/// RWKV-7 WKV computation for a sequence (prefill) on a single head.
///
/// Processes `seq_len` timesteps sequentially, updating state at each step.
/// All per-token inputs (r, k, v, w, a) are provided as flat arrays
/// of shape `[seq_len, head_size]`.
pub fn wkv7_seq(
    r: &[f32],
    k: &[f32],
    v: &[f32],
    w: &[f32],
    a: &[f32],
    state: &mut [f32],
    seq_len: usize,
    head_size: usize,
) -> Vec<f32> {
    let mut output = vec![0.0f32; seq_len * head_size];
    for t in 0..seq_len {
        let off = t * head_size;
        let r_t = &r[off..off + head_size];
        let k_t = &k[off..off + head_size];
        let v_t = &v[off..off + head_size];
        let w_t = &w[off..off + head_size];
        let a_t = &a[off..off + head_size];
        let o_t = wkv7_step(r_t, k_t, v_t, w_t, a_t, state, head_size);
        output[off..off + head_size].copy_from_slice(&o_t);
    }
    output
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wkv7_step_produces_finite_output() {
        let n = 4;
        let r = vec![0.1, 0.2, 0.3, 0.4];
        let k = vec![0.5, 0.6, 0.7, 0.8];
        let v = vec![0.9, 1.0, 1.1, 1.2];
        let w = vec![0.9, 0.85, 0.8, 0.75]; // decay in (0,1)
        let a = vec![0.3, 0.4, 0.5, 0.6];   // alpha in (0,1)
        let mut state = vec![0.0f32; n * n];

        let out = wkv7_step(&r, &k, &v, &w, &a, &mut state, n);
        assert_eq!(out.len(), n);
        for (i, &val) in out.iter().enumerate() {
            assert!(val.is_finite(), "output[{i}] = {val} is not finite");
        }
    }

    #[test]
    fn wkv7_step_zero_state_first_output_is_zero() {
        let n = 4;
        let r = vec![0.1, 0.2, 0.3, 0.4];
        let k = vec![0.5, 0.6, 0.7, 0.8];
        let v = vec![0.9, 1.0, 1.1, 1.2];
        let w = vec![0.9, 0.85, 0.8, 0.75];
        let a = vec![0.3, 0.4, 0.5, 0.6];
        let mut state = vec![0.0f32; n * n];

        // Output is computed AFTER state update, so first step
        // will have output from the just-inserted vk
        let out = wkv7_step(&r, &k, &v, &w, &a, &mut state, n);

        // State should be nonzero (contains k outer v)
        let state_sum: f32 = state.iter().map(|v| v.abs()).sum();
        assert!(state_sum > 0.0, "state should be nonzero after first step");

        // Output should also be nonzero (state @ r after update)
        let out_sum: f32 = out.iter().map(|v| v.abs()).sum();
        assert!(out_sum > 0.0, "output should be nonzero after first step");
    }

    #[test]
    fn wkv7_step_updates_state() {
        let n = 4;
        let r = vec![0.1, 0.2, 0.3, 0.4];
        let k = vec![0.5, 0.6, 0.7, 0.8];
        let v = vec![0.9, 1.0, 1.1, 1.2];
        let w = vec![0.9, 0.85, 0.8, 0.75];
        let a = vec![0.3, 0.4, 0.5, 0.6];
        let mut state = vec![0.0f32; n * n];

        let _out1 = wkv7_step(&r, &k, &v, &w, &a, &mut state, n);
        let state_after_1 = state.clone();

        let _out2 = wkv7_step(&r, &k, &v, &w, &a, &mut state, n);
        let state_changed = state.iter().zip(state_after_1.iter())
            .any(|(&a, &b)| (a - b).abs() > 1e-10);
        assert!(state_changed, "state should change after second step");
    }

    #[test]
    fn wkv7_seq_matches_individual_steps() {
        let n = 4;
        let seq_len = 3;

        let r = vec![
            0.1, 0.2, 0.3, 0.4,
            0.5, 0.6, 0.7, 0.8,
            0.9, 1.0, 1.1, 1.2,
        ];
        let k = vec![
            0.2, 0.3, 0.4, 0.5,
            0.6, 0.7, 0.8, 0.9,
            1.0, 1.1, 1.2, 1.3,
        ];
        let v = vec![
            0.3, 0.4, 0.5, 0.6,
            0.7, 0.8, 0.9, 1.0,
            1.1, 1.2, 1.3, 1.4,
        ];
        let w = vec![
            0.9, 0.85, 0.8, 0.75,
            0.9, 0.85, 0.8, 0.75,
            0.9, 0.85, 0.8, 0.75,
        ];
        let a = vec![
            0.3, 0.4, 0.5, 0.6,
            0.3, 0.4, 0.5, 0.6,
            0.3, 0.4, 0.5, 0.6,
        ];

        // Run wkv7_seq
        let mut state_seq = vec![0.0f32; n * n];
        let seq_out = wkv7_seq(&r, &k, &v, &w, &a, &mut state_seq, seq_len, n);

        // Run individual steps
        let mut state_step = vec![0.0f32; n * n];
        let mut step_out = Vec::with_capacity(seq_len * n);
        for t in 0..seq_len {
            let off = t * n;
            let o_t = wkv7_step(
                &r[off..off+n], &k[off..off+n], &v[off..off+n],
                &w[off..off+n], &a[off..off+n],
                &mut state_step, n,
            );
            step_out.extend_from_slice(&o_t);
        }

        assert_eq!(seq_out.len(), step_out.len());
        for (i, (&a, &b)) in seq_out.iter().zip(step_out.iter()).enumerate() {
            assert!(
                (a - b).abs() < 1e-6,
                "seq vs step output[{i}] mismatch: {a} vs {b}"
            );
        }
    }

    #[test]
    fn wkv7_decay_shrinks_state() {
        let n = 4;
        let r = vec![0.0f32; n];
        let k = vec![0.0f32; n]; // zero key = no new info, no feedback
        let v = vec![0.0f32; n];
        let w = vec![0.5, 0.5, 0.5, 0.5]; // strong decay
        let a = vec![0.0f32; n]; // zero alpha = no feedback

        let mut state = vec![1.0f32; n * n];
        let norm_before: f32 = state.iter().map(|v| v * v).sum();

        for _ in 0..5 {
            let _ = wkv7_step(&r, &k, &v, &w, &a, &mut state, n);
        }

        let norm_after: f32 = state.iter().map(|v| v * v).sum();
        assert!(
            norm_after < norm_before,
            "state norm should decrease with decay: before={norm_before}, after={norm_after}"
        );
    }

    #[test]
    fn wkv7_seq_output_length() {
        let n = 4;
        let seq_len = 5;
        let r = vec![0.1f32; seq_len * n];
        let k = vec![0.2f32; seq_len * n];
        let v = vec![0.3f32; seq_len * n];
        let w = vec![0.9f32; seq_len * n];
        let a = vec![0.5f32; seq_len * n];
        let mut state = vec![0.0f32; n * n];

        let out = wkv7_seq(&r, &k, &v, &w, &a, &mut state, seq_len, n);
        assert_eq!(out.len(), seq_len * n);
        for (i, &val) in out.iter().enumerate() {
            assert!(val.is_finite(), "seq output[{i}] = {val} is not finite");
        }
    }
}
