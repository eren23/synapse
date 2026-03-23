use std::mem;

use crate::quantization::calibration::MinMaxCalibration;

/// A linear layer with INT8 quantized weights and f32 per-channel scales.
///
/// Weights are stored in transposed layout `[in_features, out_features]` for
/// direct use by the SIMD GEMM kernel. Forward pass quantizes activations
/// on-the-fly and dispatches to `syn_qgemm_int8`.
pub struct QuantizedLinear {
    /// INT8 weights in GEMM-ready layout: `[in_features, out_features]`.
    /// This is the transpose of the logical `[out_features, in_features]` matrix.
    pub weights_int8: Vec<i8>,
    /// Per-channel scale factors `[out_features]`.
    pub scales: Vec<f32>,
    pub out_features: usize,
    pub in_features: usize,
}

/// Transpose a row-major `[rows, cols]` i8 matrix to `[cols, rows]`.
fn transpose_i8(src: &[i8], rows: usize, cols: usize) -> Vec<i8> {
    let mut dst = vec![0i8; rows * cols];
    for r in 0..rows {
        for c in 0..cols {
            dst[c * rows + r] = src[r * cols + c];
        }
    }
    dst
}

impl QuantizedLinear {
    /// Quantize an f32 weight matrix to INT8 using min/max calibration.
    ///
    /// `weights` is `[out_features, in_features]` row-major.
    pub fn from_f32(weights: &[f32], out_features: usize, in_features: usize) -> Self {
        assert_eq!(weights.len(), out_features * in_features);
        let scales = MinMaxCalibration::compute_scales(weights, out_features, in_features);
        Self::from_f32_with_scales(weights, &scales, out_features, in_features)
    }

    /// Quantize an f32 weight matrix to INT8 using pre-computed scales.
    pub fn from_f32_with_scales(
        weights: &[f32],
        scales: &[f32],
        out_features: usize,
        in_features: usize,
    ) -> Self {
        assert_eq!(weights.len(), out_features * in_features);
        assert_eq!(scales.len(), out_features);
        let mut weights_row_major = vec![0i8; out_features * in_features];
        for ch in 0..out_features {
            let s = scales[ch];
            if s == 0.0 {
                continue;
            }
            let inv_s = 1.0 / s;
            for j in 0..in_features {
                let val = weights[ch * in_features + j] * inv_s;
                weights_row_major[ch * in_features + j] = val.round().clamp(-128.0, 127.0) as i8;
            }
        }
        // Store in transposed layout [in_features, out_features] for GEMM.
        let weights_int8 = transpose_i8(&weights_row_major, out_features, in_features);
        QuantizedLinear {
            weights_int8,
            scales: scales.to_vec(),
            out_features,
            in_features,
        }
    }

    /// Create an empty (zero-sized) QuantizedLinear, used for absent gate weights.
    pub fn empty() -> Self {
        QuantizedLinear {
            weights_int8: Vec::new(),
            scales: Vec::new(),
            out_features: 0,
            in_features: 0,
        }
    }

    /// Forward pass: `x [m, in_features]` → `[m, out_features]`.
    ///
    /// Quantizes input activations to INT8 per-row, then dispatches to
    /// the Zig SIMD `syn_qgemm_int8` kernel:
    ///   `Y = diag(scales_x) * (x_i8 @ W_i8^T) * diag(scales_w)`
    pub fn forward(&self, x: &[f32], m: usize) -> Vec<f32> {
        let k = self.in_features;
        let n = self.out_features;
        debug_assert_eq!(
            x.len(),
            m * k,
            "QuantizedLinear::forward: x.len() != m * in_features"
        );

        if m == 0 || k == 0 || n == 0 {
            return vec![0.0f32; m * n];
        }

        // Quantize input activations per-row to INT8.
        let (x_int8, scales_x) = synapse_core::quantize_per_channel_int8(x, m, k)
            .expect("quantize_per_channel_int8 failed");

        // GEMM: C[m,n] = diag(scales_x) * (x_i8[m,k] @ W_i8^T[k,n]) * diag(scales_w)
        synapse_core::qgemm_int8(m, n, k, &x_int8, &self.weights_int8, &scales_x, &self.scales)
            .expect("qgemm_int8 failed")
    }

    /// Forward pass with pre-quantized input activations.
    ///
    /// Use this when the same input is projected through multiple linear layers
    /// (e.g. Q/K/V projections) to avoid redundant quantization of x.
    pub fn forward_pre_quantized(
        &self,
        x_int8: &[i8],
        scales_x: &[f32],
        m: usize,
    ) -> Vec<f32> {
        let k = self.in_features;
        let n = self.out_features;

        if m == 0 || k == 0 || n == 0 {
            return vec![0.0f32; m * n];
        }

        synapse_core::qgemm_int8(m, n, k, x_int8, &self.weights_int8, scales_x, &self.scales)
            .expect("qgemm_int8 failed")
    }

    /// Memory in bytes for INT8 weights + f32 scales.
    pub fn memory_bytes(&self) -> usize {
        self.weights_int8.len() * mem::size_of::<i8>()
            + self.scales.len() * mem::size_of::<f32>()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Reference f32 matmul: Y = X @ W^T, where W is [n, k] and X is [m, k].
    fn matmul_f32_ref(x: &[f32], w: &[f32], m: usize, n: usize, k: usize) -> Vec<f32> {
        let mut out = vec![0.0f32; m * n];
        for i in 0..m {
            for j in 0..n {
                let mut sum = 0.0f32;
                for d in 0..k {
                    sum += x[i * k + d] * w[j * k + d];
                }
                out[i * n + j] = sum;
            }
        }
        out
    }

    /// Frobenius-norm-based relative error: ||a - b||_F / ||b||_F.
    /// Standard metric for matrix approximation quality (matches Zig test suite).
    fn frobenius_relative_error(a: &[f32], b: &[f32]) -> f32 {
        assert_eq!(a.len(), b.len());
        let mut diff_sq = 0.0f64;
        let mut ref_sq = 0.0f64;
        for (&va, &vb) in a.iter().zip(b.iter()) {
            diff_sq += ((va - vb) as f64).powi(2);
            ref_sq += (vb as f64).powi(2);
        }
        if ref_sq == 0.0 {
            return 0.0;
        }
        (diff_sq / ref_sq).sqrt() as f32
    }

    /// Max absolute error between two vectors.
    fn max_absolute_error(a: &[f32], b: &[f32]) -> f32 {
        assert_eq!(a.len(), b.len());
        a.iter()
            .zip(b.iter())
            .map(|(&va, &vb)| (va - vb).abs())
            .fold(0.0f32, f32::max)
    }

    /// Simple deterministic pseudo-random f32 in [-1, 1].
    fn pseudo_random_vec(len: usize, seed: u64) -> Vec<f32> {
        let mut state = seed;
        (0..len)
            .map(|_| {
                // xorshift64
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                // Map to [-1, 1]
                (state as f32 / u64::MAX as f32) * 2.0 - 1.0
            })
            .collect()
    }

    #[test]
    fn test_int8_matmul_vs_f32_small() {
        let m = 4;
        let k = 16;
        let n = 8;

        let x = pseudo_random_vec(m * k, 42);
        let w = pseudo_random_vec(n * k, 123);

        let ql = QuantizedLinear::from_f32(&w, n, k);
        let y_int8 = ql.forward(&x, m);
        let y_ref = matmul_f32_ref(&x, &w, m, n, k);

        let rel_err = frobenius_relative_error(&y_int8, &y_ref);
        assert!(
            rel_err < 0.1,
            "small matmul: Frobenius relative error {rel_err:.4} >= 0.1"
        );
    }

    #[test]
    fn test_int8_matmul_vs_f32_medium() {
        let m = 32;
        let k = 128;
        let n = 64;

        let x = pseudo_random_vec(m * k, 777);
        let w = pseudo_random_vec(n * k, 999);

        let ql = QuantizedLinear::from_f32(&w, n, k);
        let y_int8 = ql.forward(&x, m);
        let y_ref = matmul_f32_ref(&x, &w, m, n, k);

        let rel_err = frobenius_relative_error(&y_int8, &y_ref);
        assert!(
            rel_err < 0.1,
            "medium matmul: Frobenius relative error {rel_err:.4} >= 0.1"
        );
    }

    #[test]
    fn test_int8_matmul_vs_f32_large() {
        let m = 1;
        let k = 512;
        let n = 512;

        let x = pseudo_random_vec(m * k, 314);
        let w = pseudo_random_vec(n * k, 159);

        let ql = QuantizedLinear::from_f32(&w, n, k);
        let y_int8 = ql.forward(&x, m);
        let y_ref = matmul_f32_ref(&x, &w, m, n, k);

        let rel_err = frobenius_relative_error(&y_int8, &y_ref);
        let abs_err = max_absolute_error(&y_int8, &y_ref);
        assert!(
            rel_err < 0.1,
            "large matmul: Frobenius relative error {rel_err:.4} >= 0.1"
        );
        assert!(
            abs_err < 0.5,
            "large matmul: max absolute error {abs_err:.4} >= 0.5"
        );
    }

    #[test]
    fn test_int8_matmul_max_absolute_logit_error() {
        // Simulate logit-like output: larger weight matrix.
        let m = 1;
        let k = 256;
        let n = 256;

        let x = pseudo_random_vec(m * k, 2025);
        let w = pseudo_random_vec(n * k, 2026);

        let ql = QuantizedLinear::from_f32(&w, n, k);
        let y_int8 = ql.forward(&x, m);
        let y_ref = matmul_f32_ref(&x, &w, m, n, k);

        let abs_err = max_absolute_error(&y_int8, &y_ref);
        assert!(
            abs_err <= 0.5,
            "logit error: max absolute error {abs_err:.4} > 0.5"
        );
    }

    #[test]
    fn test_pre_quantized_matches_forward() {
        let m = 8;
        let k = 64;
        let n = 32;

        let x = pseudo_random_vec(m * k, 555);
        let w = pseudo_random_vec(n * k, 666);

        let ql = QuantizedLinear::from_f32(&w, n, k);

        let y_normal = ql.forward(&x, m);

        // Pre-quantize x and use forward_pre_quantized.
        let (x_int8, scales_x) = synapse_core::quantize_per_channel_int8(&x, m, k).unwrap();
        let y_pre = ql.forward_pre_quantized(&x_int8, &scales_x, m);

        // Should be bit-identical since they use the same quantized input.
        assert_eq!(y_normal.len(), y_pre.len());
        let abs_err = max_absolute_error(&y_normal, &y_pre);
        assert!(
            abs_err < 1e-6,
            "pre-quantized path diverges: max abs error {abs_err}"
        );
    }

    #[test]
    fn test_empty_forward() {
        let ql = QuantizedLinear::empty();
        let result = ql.forward(&[], 0);
        assert!(result.is_empty());
    }

    #[test]
    fn test_single_element() {
        let w = vec![2.0f32];
        let x = vec![3.0f32];
        let ql = QuantizedLinear::from_f32(&w, 1, 1);
        let y = ql.forward(&x, 1);
        assert_eq!(y.len(), 1);
        let rel_err = ((y[0] - 6.0).abs()) / 6.0;
        assert!(rel_err < 0.1, "1x1 matmul: rel error {rel_err:.4}");
    }

    #[test]
    fn test_memory_bytes() {
        let w = pseudo_random_vec(64 * 128, 42);
        let ql = QuantizedLinear::from_f32(&w, 64, 128);
        let mem = ql.memory_bytes();
        // weights_int8 (transposed): 64*128 bytes + scales: 64*4 bytes
        let expected = 64 * 128 + 64 * 4;
        assert_eq!(mem, expected);
    }
}
