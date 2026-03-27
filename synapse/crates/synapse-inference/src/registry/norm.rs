use super::NormVariant;

/// RMSNorm: output = gamma * x * rsqrt(mean(x^2) + eps)
///
/// The production forward path calls Zig `syn_rmsnorm_forward` via FFI.
/// A pure-Rust reference forward is provided for testing.
#[derive(Debug, Clone)]
pub struct RMSNorm {
    eps: f64,
    hidden_size: usize,
    gamma: Vec<f32>,
}

impl RMSNorm {
    pub fn new(eps: f64, hidden_size: usize) -> Self {
        Self {
            eps,
            hidden_size,
            gamma: vec![1.0; hidden_size],
        }
    }

    pub fn with_weights(eps: f64, gamma: Vec<f32>) -> Self {
        let hidden_size = gamma.len();
        Self {
            eps,
            hidden_size,
            gamma,
        }
    }

    pub fn hidden_size(&self) -> usize {
        self.hidden_size
    }

    pub fn gamma(&self) -> &[f32] {
        &self.gamma
    }

    pub fn param_count(&self) -> usize {
        self.hidden_size
    }

    pub fn output_shape(&self, input_shape: &[usize]) -> Vec<usize> {
        input_shape.to_vec()
    }

    /// Reference forward pass (pure Rust).
    /// Input is a flat buffer with the given shape; last dim must equal hidden_size.
    pub fn forward(&self, input: &[f32], shape: &[usize]) -> Vec<f32> {
        let total: usize = shape.iter().product();
        assert_eq!(input.len(), total);
        let norm_dim = *shape.last().unwrap();
        assert_eq!(norm_dim, self.hidden_size);

        let num_vectors = total / self.hidden_size;
        let mut output = vec![0.0f32; total];

        for v in 0..num_vectors {
            let off = v * self.hidden_size;
            let x = &input[off..off + self.hidden_size];
            let o = &mut output[off..off + self.hidden_size];

            let mean_sq: f32 = x.iter().map(|&xi| xi * xi).sum::<f32>() / self.hidden_size as f32;
            let rms_inv = 1.0 / (mean_sq + self.eps as f32).sqrt();

            for i in 0..self.hidden_size {
                o[i] = self.gamma[i] * x[i] * rms_inv;
            }
        }
        output
    }

    /// Forward pass using Zig FFI (`syn_rmsnorm_forward`).
    ///
    /// # Safety
    /// Requires the synapse Zig library to be linked.
    pub unsafe fn forward_ffi(&self, input: &[f32], shape: &[usize]) -> Vec<f32> {
        use synapse_sys::*;

        let total: usize = shape.iter().product();
        assert_eq!(input.len(), total);

        let mut in_storage = std::ptr::null_mut();
        assert_eq!(syn_storage_create(total, &mut in_storage), SYN_OK);
        let mut in_data = std::ptr::null_mut();
        syn_storage_data(in_storage, &mut in_data);
        std::ptr::copy_nonoverlapping(input.as_ptr(), in_data, total);
        let mut in_tensor = std::ptr::null_mut();
        syn_tensor_create(in_storage, shape.as_ptr(), shape.len(), &mut in_tensor);

        let mut g_storage = std::ptr::null_mut();
        assert_eq!(syn_storage_create(self.hidden_size, &mut g_storage), SYN_OK);
        let mut g_data = std::ptr::null_mut();
        syn_storage_data(g_storage, &mut g_data);
        std::ptr::copy_nonoverlapping(self.gamma.as_ptr(), g_data, self.hidden_size);
        let g_shape = [self.hidden_size];
        let mut g_tensor = std::ptr::null_mut();
        syn_tensor_create(g_storage, g_shape.as_ptr(), 1, &mut g_tensor);

        let mut out_tensor = std::ptr::null_mut();
        assert_eq!(
            syn_rmsnorm_forward(&mut out_tensor, in_tensor, g_tensor, 1, self.eps as f32),
            SYN_OK,
        );

        let mut out_data = std::ptr::null_mut();
        syn_tensor_data_ptr(out_tensor, &mut out_data);
        let output = std::slice::from_raw_parts(out_data, total).to_vec();

        syn_tensor_destroy(out_tensor);
        syn_tensor_destroy(in_tensor);
        syn_tensor_destroy(g_tensor);
        syn_storage_release(in_storage);
        syn_storage_release(g_storage);

        output
    }
}

impl NormVariant for RMSNorm {
    fn eps(&self) -> f64 {
        self.eps
    }
    fn name(&self) -> &str {
        "RMSNorm"
    }
}

// ─────────────────────────────────────────────────────────────────────

/// LayerNorm (inference): output = gamma * (x - mean) / sqrt(var + eps) + beta
///
/// The production forward path calls Zig `syn_layernorm_forward` via FFI.
#[derive(Debug, Clone)]
pub struct LayerNormInfer {
    eps: f64,
    hidden_size: usize,
    gamma: Vec<f32>,
    beta: Vec<f32>,
}

impl LayerNormInfer {
    pub fn new(eps: f64, hidden_size: usize) -> Self {
        Self {
            eps,
            hidden_size,
            gamma: vec![1.0; hidden_size],
            beta: vec![0.0; hidden_size],
        }
    }

    pub fn with_weights(eps: f64, gamma: Vec<f32>, beta: Vec<f32>) -> Self {
        assert_eq!(gamma.len(), beta.len());
        let hidden_size = gamma.len();
        Self {
            eps,
            hidden_size,
            gamma,
            beta,
        }
    }

    pub fn hidden_size(&self) -> usize {
        self.hidden_size
    }

    pub fn param_count(&self) -> usize {
        self.hidden_size * 2 // gamma + beta
    }

    pub fn output_shape(&self, input_shape: &[usize]) -> Vec<usize> {
        input_shape.to_vec()
    }

    /// Reference forward pass (pure Rust).
    pub fn forward(&self, input: &[f32], shape: &[usize]) -> Vec<f32> {
        let total: usize = shape.iter().product();
        assert_eq!(input.len(), total);
        let norm_dim = *shape.last().unwrap();
        assert_eq!(norm_dim, self.hidden_size);

        let num_vectors = total / self.hidden_size;
        let mut output = vec![0.0f32; total];

        for v in 0..num_vectors {
            let off = v * self.hidden_size;
            let x = &input[off..off + self.hidden_size];
            let o = &mut output[off..off + self.hidden_size];

            let mean: f32 = x.iter().sum::<f32>() / self.hidden_size as f32;
            let var: f32 = x.iter().map(|&xi| (xi - mean) * (xi - mean)).sum::<f32>()
                / self.hidden_size as f32;
            let inv_std = 1.0 / (var + self.eps as f32).sqrt();

            for i in 0..self.hidden_size {
                o[i] = self.gamma[i] * (x[i] - mean) * inv_std + self.beta[i];
            }
        }
        output
    }

    /// Forward pass using Zig FFI (`syn_layernorm_forward`).
    ///
    /// # Safety
    /// Requires the synapse Zig library to be linked.
    pub unsafe fn forward_ffi(&self, input: &[f32], shape: &[usize]) -> Vec<f32> {
        use synapse_sys::*;

        let total: usize = shape.iter().product();
        assert_eq!(input.len(), total);

        let mut in_storage = std::ptr::null_mut();
        assert_eq!(syn_storage_create(total, &mut in_storage), SYN_OK);
        let mut in_data = std::ptr::null_mut();
        syn_storage_data(in_storage, &mut in_data);
        std::ptr::copy_nonoverlapping(input.as_ptr(), in_data, total);
        let mut in_tensor = std::ptr::null_mut();
        syn_tensor_create(in_storage, shape.as_ptr(), shape.len(), &mut in_tensor);

        let param_shape = [self.hidden_size];

        let mut g_storage = std::ptr::null_mut();
        assert_eq!(syn_storage_create(self.hidden_size, &mut g_storage), SYN_OK);
        let mut g_data = std::ptr::null_mut();
        syn_storage_data(g_storage, &mut g_data);
        std::ptr::copy_nonoverlapping(self.gamma.as_ptr(), g_data, self.hidden_size);
        let mut g_tensor = std::ptr::null_mut();
        syn_tensor_create(g_storage, param_shape.as_ptr(), 1, &mut g_tensor);

        let mut b_storage = std::ptr::null_mut();
        assert_eq!(syn_storage_create(self.hidden_size, &mut b_storage), SYN_OK);
        let mut b_data = std::ptr::null_mut();
        syn_storage_data(b_storage, &mut b_data);
        std::ptr::copy_nonoverlapping(self.beta.as_ptr(), b_data, self.hidden_size);
        let mut b_tensor = std::ptr::null_mut();
        syn_tensor_create(b_storage, param_shape.as_ptr(), 1, &mut b_tensor);

        let mut out_tensor = std::ptr::null_mut();
        // normalized_dim = number of trailing dims to normalize (1 = last dim)
        assert_eq!(
            syn_layernorm_forward(
                &mut out_tensor,
                in_tensor,
                g_tensor,
                b_tensor,
                1,
                self.eps as f32,
            ),
            SYN_OK,
        );

        let mut out_data = std::ptr::null_mut();
        syn_tensor_data_ptr(out_tensor, &mut out_data);
        let output = std::slice::from_raw_parts(out_data, total).to_vec();

        syn_tensor_destroy(out_tensor);
        syn_tensor_destroy(in_tensor);
        syn_tensor_destroy(g_tensor);
        syn_tensor_destroy(b_tensor);
        syn_storage_release(in_storage);
        syn_storage_release(g_storage);
        syn_storage_release(b_storage);

        output
    }
}

impl NormVariant for LayerNormInfer {
    fn eps(&self) -> f64 {
        self.eps
    }
    fn name(&self) -> &str {
        "LayerNorm"
    }
}

// ─────────────────────────────────────────────────────────────────────
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rmsnorm_output_shape_matches_input() {
        let norm = RMSNorm::new(1e-6, 8);
        let shapes: &[&[usize]] = &[&[8], &[2, 8], &[2, 3, 8]];
        for shape in shapes {
            assert_eq!(norm.output_shape(shape), shape.to_vec());
        }
    }

    #[test]
    fn rmsnorm_output_approx_unit_rms() {
        let norm = RMSNorm::new(1e-8, 4);
        let input = vec![1.0, 2.0, 3.0, 4.0];
        let out = norm.forward(&input, &[4]);

        // RMS of output should be close to 1 (with gamma=1, eps~0).
        let rms: f32 = (out.iter().map(|x| x * x).sum::<f32>() / 4.0).sqrt();
        assert!(
            (rms - 1.0).abs() < 0.01,
            "RMS of output should be ~1.0, got {rms}"
        );
    }

    #[test]
    fn rmsnorm_weights_scale_output() {
        let gamma = vec![2.0, 0.5, 3.0, 1.0];
        let norm_unit = RMSNorm::new(1e-8, 4);
        let norm_scaled = RMSNorm::with_weights(1e-8, gamma.clone());

        let input = vec![1.0, 2.0, 3.0, 4.0];
        let out_unit = norm_unit.forward(&input, &[4]);
        let out_scaled = norm_scaled.forward(&input, &[4]);

        for i in 0..4 {
            let expected = out_unit[i] * gamma[i];
            assert!(
                (out_scaled[i] - expected).abs() < 1e-6,
                "Scaled output[{i}] = {}, expected {}",
                out_scaled[i],
                expected
            );
        }
    }

    #[test]
    fn rmsnorm_param_count() {
        let norm = RMSNorm::new(1e-6, 512);
        assert_eq!(norm.param_count(), 512);
    }

    #[test]
    fn layernorm_output_shape_matches_input() {
        let norm = LayerNormInfer::new(1e-5, 16);
        let shapes: &[&[usize]] = &[&[16], &[4, 16], &[2, 3, 16]];
        for shape in shapes {
            assert_eq!(norm.output_shape(shape), shape.to_vec());
        }
    }

    #[test]
    fn layernorm_param_count() {
        let norm = LayerNormInfer::new(1e-5, 256);
        assert_eq!(norm.param_count(), 512); // gamma + beta
    }

    #[test]
    fn rmsnorm_and_layernorm_produce_different_output() {
        let hidden = 8;
        let rms = RMSNorm::new(1e-6, hidden);
        let ln = LayerNormInfer::new(1e-6, hidden);

        let input: Vec<f32> = (1..=hidden as u32).map(|x| x as f32 * 0.3).collect();
        let shape = [hidden];

        let out_rms = rms.forward(&input, &shape);
        let out_ln = ln.forward(&input, &shape);

        // They should NOT be identical (different formulas).
        let diff: f32 = out_rms
            .iter()
            .zip(out_ln.iter())
            .map(|(a, b)| (a - b).abs())
            .sum();
        assert!(
            diff > 1e-4,
            "RMSNorm and LayerNorm should produce different outputs, total diff = {diff}"
        );
    }

    #[test]
    fn norm_trait_dispatch() {
        let variants: Vec<Box<dyn NormVariant>> = vec![
            Box::new(RMSNorm::new(1e-6, 64)),
            Box::new(LayerNormInfer::new(1e-5, 64)),
        ];

        assert_eq!(variants[0].name(), "RMSNorm");
        assert!((variants[0].eps() - 1e-6).abs() < 1e-12);

        assert_eq!(variants[1].name(), "LayerNorm");
        assert!((variants[1].eps() - 1e-5).abs() < 1e-12);
    }

    #[test]
    fn rmsnorm_batched_forward() {
        let norm = RMSNorm::new(1e-6, 4);
        let input = vec![
            1.0, 2.0, 3.0, 4.0, // vector 0
            4.0, 3.0, 2.0, 1.0, // vector 1
        ];
        let out = norm.forward(&input, &[2, 4]);
        assert_eq!(out.len(), 8);

        // Each vector should have its own normalization.
        let rms0: f32 = (out[0..4].iter().map(|x| x * x).sum::<f32>() / 4.0).sqrt();
        let rms1: f32 = (out[4..8].iter().map(|x| x * x).sum::<f32>() / 4.0).sqrt();
        assert!((rms0 - 1.0).abs() < 0.01);
        assert!((rms1 - 1.0).abs() < 0.01);
    }

    #[test]
    fn rmsnorm_ffi_matches_reference() {
        let norm = RMSNorm::new(1e-6, 4);
        let input = vec![1.0, 2.0, 3.0, 4.0];
        let shape = [4usize];

        let ref_out = norm.forward(&input, &shape);
        let ffi_out = unsafe { norm.forward_ffi(&input, &shape) };

        for i in 0..4 {
            assert!(
                (ref_out[i] - ffi_out[i]).abs() < 1e-5,
                "Mismatch at {i}: ref={}, ffi={}",
                ref_out[i],
                ffi_out[i]
            );
        }
    }

    #[test]
    fn layernorm_ffi_matches_reference() {
        let norm = LayerNormInfer::new(1e-5, 4);
        let input = vec![1.0, 2.0, 3.0, 4.0];
        let shape = [4usize];

        let ref_out = norm.forward(&input, &shape);
        let ffi_out = unsafe { norm.forward_ffi(&input, &shape) };

        for i in 0..4 {
            assert!(
                (ref_out[i] - ffi_out[i]).abs() < 1e-4,
                "Mismatch at {i}: ref={}, ffi={}",
                ref_out[i],
                ffi_out[i]
            );
        }
    }
}
