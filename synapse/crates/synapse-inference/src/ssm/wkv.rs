//! WKV (Weighted Key-Value) kernel for RWKV inference.
//!
//! Provides single-step and sequence WKV computation used by RWKV's time mixing.
//! The WKV state is a per-head matrix of shape `[head_size, head_size]` that
//! accumulates decayed outer products of keys and values.

/// Single-step WKV computation for one attention head.
///
/// State: `wkv_state [head_size, head_size]` -- accumulated outer products with decay.
///
/// Output:
///   `output[d] = sum_j( wkv_state[d][j] * r[j] )`  (query state with receptance)
///
/// State update:
///   `wkv_state[d][j] = exp(w[d]) * wkv_state[d][j] + k[d] * v[j]`
///
/// This implements: `wkv_state = diag(exp(w)) @ wkv_state + k outer v`, `output = wkv_state @ r`.
pub fn wkv_step(
    r: &[f32],              // receptance [head_size]
    k: &[f32],              // key [head_size]
    v: &[f32],              // value [head_size]
    w: &[f32],              // decay weights [head_size] (in log space, negative)
    wkv_state: &mut [f32],  // [head_size, head_size]
    head_size: usize,
) -> Vec<f32> {
    let mut output = vec![0.0f32; head_size];

    // Output: o = state @ r
    for d in 0..head_size {
        let mut sum = 0.0f32;
        for j in 0..head_size {
            sum += wkv_state[d * head_size + j] * r[j];
        }
        output[d] = sum;
    }

    // State update: state[d][j] = exp(w[d]) * state[d][j] + k[d] * v[j]
    for d in 0..head_size {
        let decay = w[d].exp();
        for j in 0..head_size {
            wkv_state[d * head_size + j] =
                decay * wkv_state[d * head_size + j] + k[d] * v[j];
        }
    }

    output
}

/// WKV computation for a sequence (prefill) on a single head.
///
/// Processes `seq_len` timesteps sequentially, updating state at each step.
pub fn wkv_seq(
    r: &[f32],              // [seq_len, head_size]
    k: &[f32],              // [seq_len, head_size]
    v: &[f32],              // [seq_len, head_size]
    w: &[f32],              // [head_size] decay weights
    wkv_state: &mut [f32],  // [head_size, head_size]
    seq_len: usize,
    head_size: usize,
) -> Vec<f32> {
    let mut output = vec![0.0f32; seq_len * head_size];
    for t in 0..seq_len {
        let r_t = &r[t * head_size..(t + 1) * head_size];
        let k_t = &k[t * head_size..(t + 1) * head_size];
        let v_t = &v[t * head_size..(t + 1) * head_size];
        let o_t = wkv_step(r_t, k_t, v_t, w, wkv_state, head_size);
        output[t * head_size..(t + 1) * head_size].copy_from_slice(&o_t);
    }
    output
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wkv_step_produces_finite_output() {
        let head_size = 4;
        let r = vec![0.1, 0.2, 0.3, 0.4];
        let k = vec![0.5, 0.6, 0.7, 0.8];
        let v = vec![0.9, 1.0, 1.1, 1.2];
        let w = vec![-0.1, -0.2, -0.3, -0.4]; // negative log-space decay
        let mut state = vec![0.0f32; head_size * head_size];

        let out = wkv_step(&r, &k, &v, &w, &mut state, head_size);

        assert_eq!(out.len(), head_size);
        for (i, &val) in out.iter().enumerate() {
            assert!(val.is_finite(), "output[{i}] = {val} is not finite");
        }
    }

    #[test]
    fn wkv_step_zero_state_first_output_is_zero() {
        // With zero state, output = state @ r = 0, then state gets updated
        let head_size = 4;
        let r = vec![0.1, 0.2, 0.3, 0.4];
        let k = vec![0.5, 0.6, 0.7, 0.8];
        let v = vec![0.9, 1.0, 1.1, 1.2];
        let w = vec![-0.1, -0.2, -0.3, -0.4];
        let mut state = vec![0.0f32; head_size * head_size];

        let out = wkv_step(&r, &k, &v, &w, &mut state, head_size);

        // First step from zero state: output should be all zeros
        for (i, &val) in out.iter().enumerate() {
            assert_eq!(val, 0.0, "first step output[{i}] should be 0.0, got {val}");
        }

        // But state should now be nonzero (k outer v)
        let state_sum: f32 = state.iter().map(|v| v.abs()).sum();
        assert!(state_sum > 0.0, "state should be nonzero after first step");
    }

    #[test]
    fn wkv_step_updates_state() {
        let head_size = 4;
        let r = vec![0.1, 0.2, 0.3, 0.4];
        let k = vec![0.5, 0.6, 0.7, 0.8];
        let v = vec![0.9, 1.0, 1.1, 1.2];
        let w = vec![-0.1, -0.2, -0.3, -0.4];
        let mut state = vec![0.0f32; head_size * head_size];

        // First step: state goes from zero to k outer v
        let _out1 = wkv_step(&r, &k, &v, &w, &mut state, head_size);
        let state_after_1 = state.clone();

        // Second step: state should change (decay + new outer product)
        let _out2 = wkv_step(&r, &k, &v, &w, &mut state, head_size);
        let state_changed = state
            .iter()
            .zip(state_after_1.iter())
            .any(|(&a, &b)| (a - b).abs() > 1e-10);
        assert!(state_changed, "state should change after second step");
    }

    #[test]
    fn wkv_step_second_output_nonzero() {
        let head_size = 4;
        let r = vec![0.1, 0.2, 0.3, 0.4];
        let k = vec![0.5, 0.6, 0.7, 0.8];
        let v = vec![0.9, 1.0, 1.1, 1.2];
        let w = vec![-0.1, -0.2, -0.3, -0.4];
        let mut state = vec![0.0f32; head_size * head_size];

        // First step populates state
        let _out1 = wkv_step(&r, &k, &v, &w, &mut state, head_size);

        // Second step should have nonzero output (state is now nonzero)
        let out2 = wkv_step(&r, &k, &v, &w, &mut state, head_size);
        let out2_sum: f32 = out2.iter().map(|v| v.abs()).sum();
        assert!(out2_sum > 0.0, "second step output should be nonzero");
    }

    #[test]
    fn wkv_seq_matches_individual_steps() {
        let head_size = 4;
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
        let w = vec![-0.1, -0.2, -0.3, -0.4];

        // Run wkv_seq
        let mut state_seq = vec![0.0f32; head_size * head_size];
        let seq_out = wkv_seq(&r, &k, &v, &w, &mut state_seq, seq_len, head_size);

        // Run individual steps
        let mut state_step = vec![0.0f32; head_size * head_size];
        let mut step_out = Vec::with_capacity(seq_len * head_size);
        for t in 0..seq_len {
            let r_t = &r[t * head_size..(t + 1) * head_size];
            let k_t = &k[t * head_size..(t + 1) * head_size];
            let v_t = &v[t * head_size..(t + 1) * head_size];
            let o_t = wkv_step(r_t, k_t, v_t, &w, &mut state_step, head_size);
            step_out.extend_from_slice(&o_t);
        }

        // Outputs should match exactly
        assert_eq!(seq_out.len(), step_out.len());
        for (i, (&a, &b)) in seq_out.iter().zip(step_out.iter()).enumerate() {
            assert!(
                (a - b).abs() < 1e-6,
                "seq vs step output[{i}] mismatch: {a} vs {b}"
            );
        }

        // States should match exactly
        for (i, (&a, &b)) in state_seq.iter().zip(state_step.iter()).enumerate() {
            assert!(
                (a - b).abs() < 1e-6,
                "seq vs step state[{i}] mismatch: {a} vs {b}"
            );
        }
    }

    #[test]
    fn wkv_seq_output_length() {
        let head_size = 4;
        let seq_len = 5;
        let r = vec![0.1f32; seq_len * head_size];
        let k = vec![0.2f32; seq_len * head_size];
        let v = vec![0.3f32; seq_len * head_size];
        let w = vec![-0.1f32; head_size];
        let mut state = vec![0.0f32; head_size * head_size];

        let out = wkv_seq(&r, &k, &v, &w, &mut state, seq_len, head_size);
        assert_eq!(out.len(), seq_len * head_size);
        for (i, &val) in out.iter().enumerate() {
            assert!(val.is_finite(), "seq output[{i}] = {val} is not finite");
        }
    }

    #[test]
    fn wkv_decay_shrinks_state() {
        // With no new input, repeatedly applying decay should shrink state
        let head_size = 4;
        let r = vec![0.0f32; head_size]; // zero receptance
        let k = vec![0.0f32; head_size]; // zero key (no new info)
        let v = vec![0.0f32; head_size]; // zero value
        let w = vec![-0.5, -0.5, -0.5, -0.5]; // strong decay

        // Seed state with some values
        let mut state = vec![1.0f32; head_size * head_size];

        let state_norm_before: f32 = state.iter().map(|v| v * v).sum();

        // Apply several steps with zero input
        for _ in 0..5 {
            let _ = wkv_step(&r, &k, &v, &w, &mut state, head_size);
        }

        let state_norm_after: f32 = state.iter().map(|v| v * v).sum();
        assert!(
            state_norm_after < state_norm_before,
            "state norm should decrease with decay: before={state_norm_before}, after={state_norm_after}"
        );
    }
}
