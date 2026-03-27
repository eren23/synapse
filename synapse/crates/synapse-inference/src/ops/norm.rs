use crate::registry::NormVariant;

/// RMS normalization over the last dimension (SIMD via Zig FFI).
///
/// Uses `syn_vmul` / `syn_vreduce_sum` for zero-copy SIMD on each row,
/// avoiding tensor-handle allocation overhead that dominates at small sizes.
#[cfg(feature = "zig-ffi")]
pub(crate) fn rmsnorm(x: &[f32], weight: &[f32], eps: f32, hidden_size: usize) -> Vec<f32> {
    let n = x.len() / hidden_size;
    let mut out = vec![0.0f32; x.len()];

    unsafe {
        for i in 0..n {
            let off = i * hidden_size;
            let row_ptr = x.as_ptr().add(off);
            let out_ptr = out.as_mut_ptr().add(off);

            // SIMD: out = x * x  (reuse output as scratch for squared values)
            synapse_sys::syn_vmul(out_ptr, row_ptr, row_ptr, hidden_size);

            // SIMD: ms = sum(x^2)
            let mut sum_sq = 0.0f32;
            synapse_sys::syn_vreduce_sum(out_ptr, hidden_size, &mut sum_sq);

            let scale = 1.0 / (sum_sq / hidden_size as f32 + eps).sqrt();

            // SIMD: out = x * weight
            synapse_sys::syn_vmul(out_ptr, row_ptr, weight.as_ptr(), hidden_size);

            // Scale by normalization factor.  At 1024 elements this is auto-
            // vectorized by LLVM and negligible relative to the SIMD ops above.
            for j in 0..hidden_size {
                *out_ptr.add(j) *= scale;
            }
        }
    }
    out
}

#[cfg(not(feature = "zig-ffi"))]
pub(crate) fn rmsnorm(x: &[f32], weight: &[f32], eps: f32, hidden_size: usize) -> Vec<f32> {
    super::pure_rust_ops::rmsnorm(x, weight, eps, hidden_size)
}

/// Layer normalization over the last dimension (gamma only, no beta).
pub(crate) fn layernorm(x: &[f32], weight: &[f32], eps: f32, hidden_size: usize) -> Vec<f32> {
    let n = x.len() / hidden_size;
    let mut out = vec![0.0f32; x.len()];
    for i in 0..n {
        let off = i * hidden_size;
        let slice = &x[off..off + hidden_size];
        let mean: f32 = slice.iter().sum::<f32>() / hidden_size as f32;
        let var: f32 =
            slice.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / hidden_size as f32;
        let scale = 1.0 / (var + eps).sqrt();
        for j in 0..hidden_size {
            out[off + j] = (slice[j] - mean) * scale * weight[j];
        }
    }
    out
}

/// Apply normalization (dispatch on variant name).
pub(crate) fn apply_norm(
    x: &[f32],
    weight: &[f32],
    norm: &dyn NormVariant,
    hidden_size: usize,
) -> Vec<f32> {
    let eps = norm.eps() as f32;
    match norm.name() {
        "RMSNorm" => rmsnorm(x, weight, eps, hidden_size),
        "LayerNorm" => layernorm(x, weight, eps, hidden_size),
        _ => x.to_vec(),
    }
}

pub(crate) fn apply_headwise_rmsnorm(
    x: &[f32],
    weight: &[f32],
    _rows: usize,
    _heads: usize,
    head_dim: usize,
    eps: f32,
) -> Vec<f32> {
    if weight.is_empty() {
        return x.to_vec();
    }

    // Data is already contiguous per-head: [rows * heads, head_dim].
    // Delegate to SIMD rmsnorm which normalizes over the last dimension.
    rmsnorm(x, weight, eps, head_dim)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rms(v: &[f32]) -> f32 {
        (v.iter().map(|x| x * x).sum::<f32>() / v.len() as f32).sqrt()
    }

    #[test]
    fn rmsnorm_output_has_unit_rms() {
        let x = vec![1.0f32, 2.0, 3.0, 4.0];
        let weight = vec![1.0f32; 4];
        let out = rmsnorm(&x, &weight, 1e-8, 4);
        let r = rms(&out);
        assert!((r - 1.0).abs() < 0.01, "RMS of rmsnorm output should be ~1.0, got {r}");
    }

    #[test]
    fn rmsnorm_with_ones_weight_is_identity_scale() {
        // With weight=[1,...], rmsnorm just rescales direction not orientation.
        // Confirm the output has the same direction as x/||x||_rms.
        let x = vec![3.0f32, 4.0, 0.0, 0.0];
        let weight = vec![1.0f32; 4];
        let out = rmsnorm(&x, &weight, 1e-8, 4);
        // output[0]/output[1] should equal x[0]/x[1] = 3/4
        let ratio_out = out[0] / out[1];
        let ratio_in = x[0] / x[1];
        assert!(
            (ratio_out - ratio_in).abs() < 1e-5,
            "rmsnorm with ones weight should preserve direction: got ratio {ratio_out}, expected {ratio_in}"
        );
    }

    #[test]
    fn layernorm_output_has_zero_mean() {
        let x = vec![1.0f32, 2.0, 3.0, 4.0];
        let weight = vec![1.0f32; 4];
        let out = layernorm(&x, &weight, 1e-8, 4);
        let mean: f32 = out.iter().sum::<f32>() / out.len() as f32;
        assert!(mean.abs() < 1e-5, "layernorm output mean should be ~0.0, got {mean}");
    }

    #[test]
    fn layernorm_batched_independent() {
        // Two different rows; each should normalize independently (mean ~ 0).
        // Row 0 and row 1 should NOT produce identical outputs.
        let x = vec![
            1.0f32, 2.0, 3.0, 4.0,   // row 0: ascending
            10.0f32, 10.0, 20.0, 20.0, // row 1: different distribution
        ];
        let weight = vec![1.0f32; 4];
        let out = layernorm(&x, &weight, 1e-8, 4);
        assert_eq!(out.len(), 8);

        // Each row independently normalized: both means should be ~0
        let mean0: f32 = out[0..4].iter().sum::<f32>() / 4.0;
        let mean1: f32 = out[4..8].iter().sum::<f32>() / 4.0;
        assert!(mean0.abs() < 1e-5, "row 0 mean should be 0, got {mean0}");
        assert!(mean1.abs() < 1e-5, "row 1 mean should be 0, got {mean1}");

        // The two rows should NOT be identical (different input distributions)
        let diff: f32 = out[0..4].iter().zip(out[4..8].iter()).map(|(a, b)| (a - b).abs()).sum();
        assert!(diff > 1e-3, "rows with different inputs should produce different normalized outputs, diff={diff}");
    }
}
