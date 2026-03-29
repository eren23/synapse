/// Softplus activation used to ensure `delta > 0`.
///
/// `softplus(x) = log(1 + exp(x))`
pub fn compute_delta(raw_dt: &[f32]) -> Vec<f32> {
    raw_dt
        .iter()
        .map(|&x| {
            if x >= 20.0 {
                x // numerical stability: log(1+exp(x)) ≈ x for large x
            } else {
                (1.0 + x.exp()).ln()
            }
        })
        .collect()
}

/// Single-step selective scan (for autoregressive decode).
///
/// Applies one step of the discretised SSM recurrence:
/// ```text
/// h[i, j] = exp(delta[i] * A[i, j]) * h[i, j] + delta[i] * B[j] * x[i]
/// y[i]    = sum_j(C[j] * h[i, j]) + D[i] * x[i]
/// ```
///
/// # Parameters
/// - `x`         — input `[d_inner]`
/// - `delta`     — discretisation steps `[d_inner]`, must be positive (use [`compute_delta`])
/// - `a_log`     — log of A matrix `[d_inner * d_state]`, stored as negative values for stability
/// - `b`         — input-dependent B `[d_state]`
/// - `c`         — input-dependent C `[d_state]`
/// - `d`         — skip-connection weights `[d_inner]`
/// - `ssm_state` — `[d_inner * d_state]`, updated in place
///
/// # Returns
/// Output vector `[d_inner]`.
pub fn selective_scan_step(
    x: &[f32],
    delta: &[f32],
    a_log: &[f32],
    b: &[f32],
    c: &[f32],
    d: &[f32],
    ssm_state: &mut [f32],
) -> Vec<f32> {
    let d_inner = x.len();
    let d_state = b.len();

    debug_assert_eq!(delta.len(), d_inner);
    debug_assert_eq!(a_log.len(), d_inner * d_state);
    debug_assert_eq!(c.len(), d_state);
    debug_assert_eq!(d.len(), d_inner);
    debug_assert_eq!(ssm_state.len(), d_inner * d_state);

    // Zig SIMD fast path
    #[cfg(feature = "zig-ffi")]
    {
        let mut y = vec![0.0f32; d_inner];
        unsafe {
            synapse_sys::syn_selective_scan_step(
                x.as_ptr(), delta.as_ptr(), a_log.as_ptr(),
                b.as_ptr(), c.as_ptr(), d.as_ptr(),
                ssm_state.as_mut_ptr(), y.as_mut_ptr(),
                d_inner, d_state,
            );
        }
        return y;
    }

    // Pure-Rust fallback
    #[cfg(not(feature = "zig-ffi"))]
    {
        let mut y = vec![0.0f32; d_inner];

        for i in 0..d_inner {
            let dt_i = delta[i];
            let x_i = x[i];
            let mut yi = 0.0f32;

            for j in 0..d_state {
                let idx = i * d_state + j;
                // A = -exp(a_log) (always negative for stability)
                // Discretise: exp(delta * A) = exp(-delta * exp(a_log))
                let a = -(a_log[idx].exp());
                let a_disc = (dt_i * a).exp();
                // Update hidden state
                let h_new = a_disc * ssm_state[idx] + dt_i * b[j] * x_i;
                ssm_state[idx] = h_new;
                yi += c[j] * h_new;
            }

            // Skip connection
            y[i] = yi + d[i] * x_i;
        }

        y
    }
}

/// Sequential selective scan for prefill (processes a sequence of tokens).
///
/// Calls [`selective_scan_step`] for every token in the sequence, accumulating
/// the hidden state across steps.
///
/// # Parameters
/// - `xs`        — token inputs, shape `[seq_len, d_inner]` (row-major)
/// - `deltas`    — per-token delta, shape `[seq_len, d_inner]`
/// - `a_log`     — `[d_inner * d_state]` (shared across time steps)
/// - `bs`        — per-token B, shape `[seq_len, d_state]`
/// - `cs`        — per-token C, shape `[seq_len, d_state]`
/// - `d`         — skip-connection `[d_inner]`
/// - `ssm_state` — `[d_inner * d_state]`, updated in place
///
/// # Returns
/// Output tensor `[seq_len, d_inner]` (row-major).
pub fn selective_scan_seq(
    xs: &[f32],
    deltas: &[f32],
    a_log: &[f32],
    bs: &[f32],
    cs: &[f32],
    d: &[f32],
    ssm_state: &mut [f32],
) -> Vec<f32> {
    let d_inner = d.len();
    let d_state = a_log.len() / d_inner;
    let seq_len = xs.len() / d_inner;

    debug_assert_eq!(xs.len(), seq_len * d_inner);
    debug_assert_eq!(deltas.len(), seq_len * d_inner);
    debug_assert_eq!(bs.len(), seq_len * d_state);
    debug_assert_eq!(cs.len(), seq_len * d_state);

    let mut outputs = Vec::with_capacity(seq_len * d_inner);

    for t in 0..seq_len {
        let x_t = &xs[t * d_inner..(t + 1) * d_inner];
        let delta_t = &deltas[t * d_inner..(t + 1) * d_inner];
        let b_t = &bs[t * d_state..(t + 1) * d_state];
        let c_t = &cs[t * d_state..(t + 1) * d_state];

        let y_t = selective_scan_step(x_t, delta_t, a_log, b_t, c_t, d, ssm_state);
        outputs.extend_from_slice(&y_t);
    }

    outputs
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_test_inputs(d_inner: usize, d_state: usize) -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>) {
        let x = vec![0.5f32; d_inner];
        let delta = vec![0.1f32; d_inner];
        // a_log: negative values (log of eigenvalues < 1 for stability)
        let a_log = vec![-0.5f32; d_inner * d_state];
        let b = vec![0.3f32; d_state];
        let c = vec![0.2f32; d_state];
        let d_skip = vec![1.0f32; d_inner];
        let ssm_state = vec![0.0f32; d_inner * d_state];
        (x, delta, a_log, b, c, d_skip, ssm_state)
    }

    #[test]
    fn test_selective_scan_step_produces_finite_output() {
        let d_inner = 8;
        let d_state = 4;
        let (x, delta, a_log, b, c, d_skip, mut ssm_state) =
            make_test_inputs(d_inner, d_state);

        let y = selective_scan_step(&x, &delta, &a_log, &b, &c, &d_skip, &mut ssm_state);

        assert_eq!(y.len(), d_inner);
        for (i, &v) in y.iter().enumerate() {
            assert!(v.is_finite(), "output[{i}] = {v} is not finite");
        }
    }

    #[test]
    fn test_selective_scan_step_updates_state() {
        let d_inner = 4;
        let d_state = 2;
        let (x, delta, a_log, b, c, d_skip, mut ssm_state) =
            make_test_inputs(d_inner, d_state);

        // State should be all zeros before
        assert!(ssm_state.iter().all(|&v| v == 0.0));

        selective_scan_step(&x, &delta, &a_log, &b, &c, &d_skip, &mut ssm_state);

        // After one step with nonzero input, state must be nonzero
        let any_nonzero = ssm_state.iter().any(|&v| v != 0.0);
        assert!(any_nonzero, "SSM state should be updated after a step with nonzero input");

        // All values should remain finite
        for (i, &v) in ssm_state.iter().enumerate() {
            assert!(v.is_finite(), "ssm_state[{i}] = {v} is not finite after step");
        }
    }

    #[test]
    fn test_selective_scan_sequence() {
        let d_inner = 6;
        let d_state = 3;
        let seq_len = 5;

        let xs = vec![0.4f32; seq_len * d_inner];
        let deltas = vec![0.05f32; seq_len * d_inner];
        let a_log = vec![-0.3f32; d_inner * d_state];
        let bs = vec![0.2f32; seq_len * d_state];
        let cs = vec![0.1f32; seq_len * d_state];
        let d_skip = vec![1.0f32; d_inner];
        let mut ssm_state = vec![0.0f32; d_inner * d_state];

        let outputs = selective_scan_seq(
            &xs, &deltas, &a_log, &bs, &cs, &d_skip, &mut ssm_state,
        );

        // Output shape check
        assert_eq!(outputs.len(), seq_len * d_inner);

        // All outputs finite
        for (i, &v) in outputs.iter().enumerate() {
            assert!(v.is_finite(), "outputs[{i}] = {v} is not finite");
        }

        // Later timesteps should differ from earlier ones because state accumulates
        let t0 = &outputs[0..d_inner];
        let t4 = &outputs[4 * d_inner..5 * d_inner];
        let different = t0.iter().zip(t4.iter()).any(|(a, b)| (a - b).abs() > 1e-6);
        assert!(different, "outputs at t=0 and t=4 should differ as state accumulates");

        // Final state should be nonzero
        let any_nonzero = ssm_state.iter().any(|&v| v.abs() > 1e-9);
        assert!(any_nonzero, "SSM state should be nonzero after processing a sequence");
    }

    #[test]
    fn compute_delta_is_positive() {
        let raw = vec![-5.0f32, -1.0, 0.0, 1.0, 5.0, 25.0];
        let delta = compute_delta(&raw);
        for (i, &v) in delta.iter().enumerate() {
            assert!(v > 0.0, "delta[{i}] = {v} should be positive");
            assert!(v.is_finite(), "delta[{i}] = {v} should be finite");
        }
    }

    #[test]
    fn compute_delta_large_input_is_stable() {
        // For very large inputs, softplus(x) ≈ x
        let raw = vec![50.0f32, 100.0];
        let delta = compute_delta(&raw);
        assert!((delta[0] - 50.0).abs() < 1.0, "softplus(50) should ≈ 50, got {}", delta[0]);
        assert!((delta[1] - 100.0).abs() < 1.0, "softplus(100) should ≈ 100, got {}", delta[1]);
    }
}
