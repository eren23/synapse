//! Geometric attention: distance-aware attention for 3D point clouds and molecules.
//!
//! An op that PyTorch/MLX don't have optimized SIMD kernels for.
//! Hand-tuned in Zig with NEON/AVX2 vectorization.
//!
//! ```text
//! score[i,j] = softmax(Q[i]·K[j]/√d + exp(-||pos_i - pos_j||² / 2σ²))
//! out[i] = Σ_j score[i,j] * V[j]
//! ```

/// Geometric attention with Gaussian distance bias.
///
/// Combines standard dot-product attention with a distance-dependent bias
/// so spatially close points attend more to each other.
///
/// # Arguments
/// - `q`, `k`, `v`: `[n, d]` query, key, value embeddings
/// - `positions`: `[n, pos_dim]` spatial coordinates (e.g., 3D xyz)
/// - `sigma`: bandwidth of the Gaussian distance kernel
///
/// # Returns
/// `[n, d]` output embeddings
pub fn geometric_attention(
    n: usize,
    d: usize,
    pos_dim: usize,
    q: &[f32],
    k: &[f32],
    v: &[f32],
    positions: &[f32],
    sigma: f32,
) -> Vec<f32> {
    assert!(n <= 4096, "geometric_attention: n > 4096 not yet supported");
    let mut out = vec![0.0f32; n * d];
    let status = unsafe {
        synapse_sys::syn_geometric_attention(
            n,
            d,
            pos_dim,
            q.as_ptr(),
            k.as_ptr(),
            v.as_ptr(),
            positions.as_ptr(),
            out.as_mut_ptr(),
            sigma,
        )
    };
    debug_assert_eq!(status, synapse_sys::SYN_OK);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_geometric_attention_basic() {
        // 4 points in 3D, 8-dim embeddings
        let n = 4;
        let d = 8;

        // Points at corners of a square
        let positions = vec![
            0.0, 0.0, 0.0, // point 0: origin
            1.0, 0.0, 0.0, // point 1: right
            0.0, 1.0, 0.0, // point 2: up
            10.0, 10.0, 0.0, // point 3: far away
        ];

        // Random-ish Q, K, V
        let q: Vec<f32> = (0..n * d).map(|i| (i as f32 * 0.1).sin()).collect();
        let k: Vec<f32> = (0..n * d).map(|i| (i as f32 * 0.13).cos()).collect();
        let v: Vec<f32> = (0..n * d).map(|i| (i as f32 * 0.07 + 1.0).sin()).collect();

        let sigma = 1.0;
        let out = geometric_attention(n, d, 3, &q, &k, &v, &positions, sigma);

        assert_eq!(out.len(), n * d);
        assert!(out.iter().all(|v| v.is_finite()), "output should be finite");

        // Point 3 is far from others — its attention to points 0,1,2 should be
        // more dominated by the dot-product (distance bias ≈ 0 for far points).
        // Points 0,1,2 are close — their mutual attention should have significant
        // distance bias contribution.
        // We just verify the output is non-trivial (not all zeros, not all same)
        let norms: Vec<f32> = (0..n)
            .map(|i| {
                out[i * d..(i + 1) * d]
                    .iter()
                    .map(|x| x * x)
                    .sum::<f32>()
                    .sqrt()
            })
            .collect();
        assert!(
            norms.iter().all(|n| *n > 0.0),
            "all output norms should be positive"
        );
    }

    #[test]
    fn test_geometric_attention_distance_matters() {
        // Two identical points at different distances should get different attention patterns
        let n = 3;
        let d = 4;

        let q = vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0];
        let k = q.clone();
        let v: Vec<f32> = (0..n * d).map(|i| (i + 1) as f32).collect();

        // Config 1: all points close together
        let pos_close = vec![0.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0, 0.1, 0.0];
        let out_close = geometric_attention(n, d, 3, &q, &k, &v, &pos_close, 1.0);

        // Config 2: one point far away
        let pos_far = vec![0.0, 0.0, 0.0, 0.1, 0.0, 0.0, 100.0, 100.0, 0.0];
        let out_far = geometric_attention(n, d, 3, &q, &k, &v, &pos_far, 1.0);

        // Outputs should differ because distance changes attention distribution
        assert_ne!(
            out_close, out_far,
            "different distances should produce different outputs"
        );
    }
}
