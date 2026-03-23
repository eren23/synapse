use std::mem;

use crate::quantization::calibration::MinMaxCalibration;

/// A linear layer with INT8 quantized weights and f32 per-channel scales.
///
/// Stores weights as `[out_features, in_features]` in INT8 with one scale
/// per output channel. Forward pass dequantizes on-the-fly during matmul,
/// keeping activations in f32 throughout.
pub struct QuantizedLinear {
    /// INT8 weights, row-major `[out_features, in_features]`.
    pub weights_int8: Vec<i8>,
    /// Per-channel scale factors `[out_features]`.
    pub scales: Vec<f32>,
    pub out_features: usize,
    pub in_features: usize,
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
        let mut weights_int8 = vec![0i8; out_features * in_features];
        for ch in 0..out_features {
            let s = scales[ch];
            if s == 0.0 {
                continue;
            }
            let inv_s = 1.0 / s;
            for j in 0..in_features {
                let val = weights[ch * in_features + j] * inv_s;
                weights_int8[ch * in_features + j] = val.round().clamp(-128.0, 127.0) as i8;
            }
        }
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
    /// Computes `y = x * dequant(W)^T` by dequantizing on-the-fly.
    /// The scale is factored out of the inner loop: `y[i,j] = scale[j] * Σ_d x[i,d] * w_i8[j,d]`.
    pub fn forward(&self, x: &[f32], m: usize) -> Vec<f32> {
        let k = self.in_features;
        let n = self.out_features;
        debug_assert_eq!(
            x.len(),
            m * k,
            "QuantizedLinear::forward: x.len() != m * in_features"
        );

        let mut out = vec![0.0f32; m * n];
        for i in 0..m {
            let x_row = &x[i * k..(i + 1) * k];
            for j in 0..n {
                let w_row = &self.weights_int8[j * k..(j + 1) * k];
                let mut sum = 0.0f32;
                for d in 0..k {
                    sum += x_row[d] * w_row[d] as f32;
                }
                out[i * n + j] = sum * self.scales[j];
            }
        }
        out
    }

    /// Memory in bytes for INT8 weights + f32 scales.
    pub fn memory_bytes(&self) -> usize {
        self.weights_int8.len() * mem::size_of::<i8>()
            + self.scales.len() * mem::size_of::<f32>()
    }
}
