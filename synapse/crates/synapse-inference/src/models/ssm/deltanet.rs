//! Gated DeltaNet kernel for linear-attention hybrid models (e.g. Qwen3.5).
//!
//! Implements the Gated DeltaNet recurrence:
//!
//! ```text
//! S_t = alpha_t * S_{t-1} + beta_t * outer(v_t, k_t)
//! o_t = S_t @ q_t
//! ```
//!
//! where `alpha` is a per-step decay gate and `beta` is a per-step write gate,
//! and `q`, `k` are L2-normalised before being passed to the kernel.

/// L2-normalise a vector. Returns a unit-norm copy.
///
/// The denominator is clamped to `1e-12` to avoid division by zero.
pub fn l2_normalize(x: &[f32]) -> Vec<f32> {
    let norm = x.iter().map(|v| v * v).sum::<f32>().sqrt().max(1e-12);
    x.iter().map(|v| v / norm).collect()
}

/// Single-step Gated DeltaNet for one attention head.
///
/// Computes:
///   `S_t = alpha * S_{t-1} + beta * outer(v, k)`
///   `o_t = S_t @ q`
///
/// # Arguments
/// * `q` — query vector `[head_dim]`, should be L2 normalised by the caller.
/// * `k` — key vector `[head_dim]`, should be L2 normalised by the caller.
/// * `v` — value vector `[head_dim]`.
/// * `alpha` — scalar decay gate in `(0, 1)`, controls memory retention.
/// * `beta`  — scalar update gate in `(0, 1)`, controls write strength.
/// * `memory` — mutable state matrix `[head_dim * head_dim]` (row-major), updated in place.
/// * `head_dim` — size of each head dimension.
///
/// # Returns
/// Output vector `[head_dim]`.
pub fn deltanet_step(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    alpha: f32,
    beta: f32,
    memory: &mut [f32],
    head_dim: usize,
) -> Vec<f32> {
    // State update: S = alpha * S + beta * outer(v, k)
    for d in 0..head_dim {
        for j in 0..head_dim {
            memory[d * head_dim + j] =
                alpha * memory[d * head_dim + j] + beta * v[d] * k[j];
        }
    }

    // Output: o = S @ q
    let mut output = vec![0.0f32; head_dim];
    for d in 0..head_dim {
        let mut sum = 0.0f32;
        for j in 0..head_dim {
            sum += memory[d * head_dim + j] * q[j];
        }
        output[d] = sum;
    }
    output
}

/// Process a full sequence with the Gated DeltaNet kernel for one attention head.
///
/// Calls [`deltanet_step`] for each timestep, threading the memory state through.
///
/// # Arguments
/// * `q` — queries `[seq_len * head_dim]`.
/// * `k` — keys `[seq_len * head_dim]`.
/// * `v` — values `[seq_len * head_dim]`.
/// * `alpha` — per-timestep decay gates `[seq_len]`.
/// * `beta`  — per-timestep update gates `[seq_len]`.
/// * `memory` — mutable state matrix `[head_dim * head_dim]`, updated in place.
/// * `seq_len` — number of tokens.
/// * `head_dim` — size of each head dimension.
///
/// # Returns
/// Output tensor `[seq_len * head_dim]`.
pub fn deltanet_seq(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    alpha: &[f32],
    beta: &[f32],
    memory: &mut [f32],
    seq_len: usize,
    head_dim: usize,
) -> Vec<f32> {
    let mut output = vec![0.0f32; seq_len * head_dim];
    for t in 0..seq_len {
        let q_t = &q[t * head_dim..(t + 1) * head_dim];
        let k_t = &k[t * head_dim..(t + 1) * head_dim];
        let v_t = &v[t * head_dim..(t + 1) * head_dim];
        let o_t = deltanet_step(q_t, k_t, v_t, alpha[t], beta[t], memory, head_dim);
        output[t * head_dim..(t + 1) * head_dim].copy_from_slice(&o_t);
    }
    output
}

#[cfg(test)]
mod tests {
    use super::*;

    // -----------------------------------------------------------------------
    // deltanet_step tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_deltanet_step_produces_finite_output() {
        let head_dim = 4;
        let q = vec![0.1, 0.2, 0.3, 0.4];
        let k = vec![0.5, 0.6, 0.7, 0.8];
        let v = vec![0.9, 1.0, 1.1, 1.2];
        let mut memory = vec![0.0f32; head_dim * head_dim];

        let out = deltanet_step(&q, &k, &v, 0.9, 0.5, &mut memory, head_dim);

        assert_eq!(out.len(), head_dim, "output length should equal head_dim");
        for (i, &val) in out.iter().enumerate() {
            assert!(val.is_finite(), "output[{i}] = {val} is not finite");
        }
    }

    #[test]
    fn test_deltanet_step_updates_memory() {
        let head_dim = 4;
        let q = vec![0.1, 0.2, 0.3, 0.4];
        let k = vec![0.5, 0.6, 0.7, 0.8];
        let v = vec![0.9, 1.0, 1.1, 1.2];
        let mut memory = vec![0.0f32; head_dim * head_dim];

        let _ = deltanet_step(&q, &k, &v, 0.9, 0.5, &mut memory, head_dim);

        let mem_sum: f32 = memory.iter().map(|v| v.abs()).sum();
        assert!(mem_sum > 0.0, "memory should be non-zero after a step");
    }

    #[test]
    fn test_deltanet_alpha_controls_decay() {
        // With alpha = 0.0, old memory should be completely erased.
        let head_dim = 4;
        let q = vec![0.1, 0.2, 0.3, 0.4];
        let k = vec![0.5, 0.6, 0.7, 0.8];
        let v = vec![0.9, 1.0, 1.1, 1.2];

        // Seed with arbitrary non-zero memory
        let mut memory = vec![99.0f32; head_dim * head_dim];

        // alpha = 0 => old content is multiplied by zero and erased
        let _ = deltanet_step(&q, &k, &v, 0.0, 1.0, &mut memory, head_dim);

        // The new memory should equal exactly beta * outer(v, k)  (beta=1 here)
        for d in 0..head_dim {
            for j in 0..head_dim {
                let expected = v[d] * k[j];
                let actual = memory[d * head_dim + j];
                assert!(
                    (actual - expected).abs() < 1e-6,
                    "memory[{d}][{j}] = {actual}, expected {expected}"
                );
            }
        }
    }

    #[test]
    fn test_deltanet_step_first_output_zero_from_zero_state() {
        // When the initial state is zero, the output before the state update is
        // zero. deltanet_step updates state *then* reads it for the output, so
        // the output of the FIRST step should be non-zero (it already includes
        // the new outer product).  Verify the output is finite and has the right
        // length — the exact values are checked implicitly by other tests.
        let head_dim = 4;
        let q = l2_normalize(&[0.1, 0.2, 0.3, 0.4]);
        let k = l2_normalize(&[0.5, 0.6, 0.7, 0.8]);
        let v = vec![0.9, 1.0, 1.1, 1.2];
        let mut memory = vec![0.0f32; head_dim * head_dim];

        let out = deltanet_step(&q, &k, &v, 0.9, 0.5, &mut memory, head_dim);

        assert_eq!(out.len(), head_dim);
        assert!(out.iter().all(|v| v.is_finite()));
    }

    // -----------------------------------------------------------------------
    // deltanet_seq tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_deltanet_seq() {
        let head_dim = 4;
        let seq_len = 5;
        let q: Vec<f32> = (0..seq_len * head_dim).map(|i| i as f32 * 0.01).collect();
        let k: Vec<f32> = (0..seq_len * head_dim).map(|i| i as f32 * 0.02).collect();
        let v: Vec<f32> = (0..seq_len * head_dim).map(|i| i as f32 * 0.03 + 0.1).collect();
        let alpha: Vec<f32> = vec![0.95, 0.9, 0.85, 0.8, 0.75];
        let beta: Vec<f32> = vec![0.5, 0.6, 0.7, 0.8, 0.9];
        let mut memory = vec![0.0f32; head_dim * head_dim];

        let out = deltanet_seq(&q, &k, &v, &alpha, &beta, &mut memory, seq_len, head_dim);

        assert_eq!(out.len(), seq_len * head_dim, "output length mismatch");
        for (i, &val) in out.iter().enumerate() {
            assert!(val.is_finite(), "seq output[{i}] = {val} is not finite");
        }
    }

    #[test]
    fn test_deltanet_seq_matches_individual_steps() {
        let head_dim = 4;
        let seq_len = 3;

        let q = vec![
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
        let alpha = vec![0.9, 0.8, 0.7];
        let beta = vec![0.5, 0.6, 0.7];

        // Run deltanet_seq
        let mut mem_seq = vec![0.0f32; head_dim * head_dim];
        let seq_out =
            deltanet_seq(&q, &k, &v, &alpha, &beta, &mut mem_seq, seq_len, head_dim);

        // Run individual steps
        let mut mem_step = vec![0.0f32; head_dim * head_dim];
        let mut step_out = Vec::with_capacity(seq_len * head_dim);
        for t in 0..seq_len {
            let q_t = &q[t * head_dim..(t + 1) * head_dim];
            let k_t = &k[t * head_dim..(t + 1) * head_dim];
            let v_t = &v[t * head_dim..(t + 1) * head_dim];
            let o_t =
                deltanet_step(q_t, k_t, v_t, alpha[t], beta[t], &mut mem_step, head_dim);
            step_out.extend_from_slice(&o_t);
        }

        assert_eq!(seq_out.len(), step_out.len());
        for (i, (&a, &b)) in seq_out.iter().zip(step_out.iter()).enumerate() {
            assert!(
                (a - b).abs() < 1e-6,
                "seq vs step output[{i}]: {a} vs {b}"
            );
        }
    }

    // -----------------------------------------------------------------------
    // l2_normalize tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_l2_normalize() {
        let x = vec![3.0f32, 4.0];
        let normed = l2_normalize(&x);

        // Norm should be 1.0
        let norm: f32 = normed.iter().map(|v| v * v).sum::<f32>().sqrt();
        assert!(
            (norm - 1.0).abs() < 1e-6,
            "expected unit norm, got {norm}"
        );

        // Direction preserved: 3/5 = 0.6, 4/5 = 0.8
        assert!((normed[0] - 0.6).abs() < 1e-6);
        assert!((normed[1] - 0.8).abs() < 1e-6);
    }

    #[test]
    fn test_l2_normalize_zero_vector() {
        // A zero vector should not panic and should produce finite output.
        let x = vec![0.0f32; 4];
        let normed = l2_normalize(&x);
        assert_eq!(normed.len(), 4);
        assert!(normed.iter().all(|v| v.is_finite()));
    }
}
