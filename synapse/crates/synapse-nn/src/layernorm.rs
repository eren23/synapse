//! Layer normalization module.

use synapse_autograd::Tensor;

use crate::module::Module;

/// Layer normalization over the last N dimensions.
///
/// Normalizes input over the dimensions specified by `normalized_shape`,
/// then applies an affine transform: `output = gamma * normalized + beta`.
pub struct LayerNorm {
    pub weight: Tensor,
    pub bias: Tensor,
    pub normalized_shape: Vec<usize>,
    pub eps: f32,
    training: bool,
}

impl LayerNorm {
    /// Create a new LayerNorm. Weight initialized to ones, bias to zeros.
    pub fn new(normalized_shape: &[usize]) -> Result<Self, String> {
        if normalized_shape.is_empty() {
            return Err("normalized_shape must not be empty".into());
        }
        let total: usize = normalized_shape.iter().product();
        if total == 0 {
            return Err("normalized_shape dimensions must be > 0".into());
        }
        Ok(LayerNorm {
            weight: Tensor::ones(&[total]),
            bias: Tensor::zeros(&[total]),
            normalized_shape: normalized_shape.to_vec(),
            eps: 1e-5,
            training: true,
        })
    }

    /// Set epsilon value (builder pattern).
    pub fn with_eps(mut self, eps: f32) -> Self {
        self.eps = eps;
        self
    }
}

impl Module for LayerNorm {
    fn forward(&self, input: &Tensor) -> Tensor {
        let ndim = input.shape.len();
        let norm_ndim = self.normalized_shape.len();
        assert!(
            ndim >= norm_ndim,
            "input rank {} must be >= normalized_shape rank {}",
            ndim,
            norm_ndim
        );

        // Verify trailing dimensions match normalized_shape
        for i in 0..norm_ndim {
            assert_eq!(
                input.shape[ndim - norm_ndim + i],
                self.normalized_shape[i],
                "input shape mismatch at dim {}",
                ndim - norm_ndim + i
            );
        }

        let norm_size: usize = self.normalized_shape.iter().product();
        let outer = input.numel() / norm_size;

        let mut output = vec![0.0f32; input.numel()];

        for i in 0..outer {
            let base = i * norm_size;
            let slice = &input.data[base..base + norm_size];

            // Compute mean
            let mean: f32 = slice.iter().sum::<f32>() / norm_size as f32;

            // Compute variance
            let var: f32 = slice.iter().map(|&x| (x - mean) * (x - mean)).sum::<f32>()
                / norm_size as f32;

            let inv_std = 1.0 / (var + self.eps).sqrt();

            // Normalize and apply affine transform
            for j in 0..norm_size {
                let x_hat = (slice[j] - mean) * inv_std;
                output[base + j] = self.weight.data[j] * x_hat + self.bias.data[j];
            }
        }

        Tensor::new(output, input.shape.clone())
    }

    fn parameters(&self) -> Vec<&Tensor> {
        vec![&self.weight, &self.bias]
    }

    fn parameters_mut(&mut self) -> Vec<&mut Tensor> {
        vec![&mut self.weight, &mut self.bias]
    }

    fn set_training(&mut self, training: bool) {
        self.training = training;
    }

    fn is_training(&self) -> bool {
        self.training
    }

    fn name(&self) -> &str {
        "LayerNorm"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_tensor(shape: &[usize], seed: u32) -> Tensor {
        let n: usize = shape.iter().product();
        let mut state = seed.wrapping_mul(2654435761);
        let data: Vec<f32> = (0..n)
            .map(|_| {
                state = state.wrapping_mul(1664525).wrapping_add(1013904223);
                (state as f32 / u32::MAX as f32) - 0.5
            })
            .collect();
        Tensor::new(data, shape.to_vec())
    }

    #[test]
    fn test_output_shape_preserved() {
        let ln = LayerNorm::new(&[16]).unwrap();
        let input = make_tensor(&[2, 4, 16], 1);
        let output = ln.forward(&input);
        assert_eq!(output.shape, vec![2, 4, 16]);
    }

    #[test]
    fn test_normalized_mean_near_zero() {
        let d = 64;
        let ln = LayerNorm::new(&[d]).unwrap();
        let input = make_tensor(&[4, 8, d], 42);
        let output = ln.forward(&input);

        // Check that each normalized slice has mean ≈ 0
        let outer = 4 * 8;
        for i in 0..outer {
            let base = i * d;
            let slice = &output.data[base..base + d];
            let mean: f32 = slice.iter().sum::<f32>() / d as f32;
            assert!(
                mean.abs() < 1e-4,
                "mean {} at slice {} exceeds tolerance",
                mean,
                i
            );
        }
    }

    #[test]
    fn test_normalized_variance_near_one() {
        let d = 64;
        let ln = LayerNorm::new(&[d]).unwrap();
        let input = make_tensor(&[4, 8, d], 42);
        let output = ln.forward(&input);

        let outer = 4 * 8;
        for i in 0..outer {
            let base = i * d;
            let slice = &output.data[base..base + d];
            let mean: f32 = slice.iter().sum::<f32>() / d as f32;
            let var: f32 =
                slice.iter().map(|&x| (x - mean) * (x - mean)).sum::<f32>() / d as f32;
            assert!(
                (var - 1.0).abs() < 5e-4,
                "variance {} at slice {} exceeds tolerance",
                var,
                i
            );
        }
    }

    #[test]
    fn test_parameter_count() {
        let shape = &[8, 16];
        let ln = LayerNorm::new(shape).unwrap();
        let params = ln.parameters();
        let total: usize = params.iter().map(|p| p.numel()).sum();
        assert_eq!(total, 2 * 8 * 16);
    }

    #[test]
    fn test_weight_ones_bias_zeros() {
        let ln = LayerNorm::new(&[32]).unwrap();
        assert!(ln.weight.data.iter().all(|&v| v == 1.0));
        assert!(ln.bias.data.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn test_gradient_flows() {
        // Verify gradients can flow through by using autograd graph
        use synapse_autograd::backward::backward;
        use synapse_autograd::graph::Graph;

        let d = 16;
        let input_data = make_tensor(&[4, d], 1);
        let weight_data = Tensor::ones(&[d]);
        let bias_data = Tensor::zeros(&[d]);

        let mut g = Graph::new();
        let vi = g.variable(input_data, true);
        let vw = g.variable(weight_data, true);
        let vb = g.variable(bias_data, true);
        let out = g.layer_norm(vi, vw, vb, 1e-5);

        backward(&mut g, out);

        assert!(g.grad(vi).is_some(), "input gradient should exist");
        assert!(g.grad(vw).is_some(), "weight gradient should exist");
        assert!(g.grad(vb).is_some(), "bias gradient should exist");

        // Gradients should be finite
        for &v in &[vi, vw, vb] {
            let grad = g.grad(v).unwrap();
            for &val in &grad.data {
                assert!(val.is_finite(), "gradient contains inf or nan");
            }
        }
    }

    #[test]
    fn test_constant_input_edge_case() {
        let d = 32;
        let ln = LayerNorm::new(&[d]).unwrap();
        // All elements identical → near-zero variance
        let input = Tensor::new(vec![5.0; 8 * d], vec![8, d]);
        let output = ln.forward(&input);

        // Output should be bias (0.0) since normalized = 0 and weight = 1
        for &val in &output.data {
            assert!(val.abs() < 1e-3, "constant input should normalize to ~bias");
        }
        // No inf/nan
        for &val in &output.data {
            assert!(val.is_finite());
        }
    }

    #[test]
    fn test_with_eps() {
        let ln = LayerNorm::new(&[16]).unwrap().with_eps(1e-3);
        assert_eq!(ln.eps, 1e-3);
    }

    #[test]
    fn test_module_trait() {
        let mut ln = LayerNorm::new(&[16]).unwrap();
        assert_eq!(ln.name(), "LayerNorm");
        assert!(ln.is_training());
        ln.set_training(false);
        assert!(!ln.is_training());
    }
}
