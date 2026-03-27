//! Batch normalization layers: BatchNorm1d, BatchNorm2d.

use synapse_autograd::Tensor;

use crate::module::Module;

// ── BatchNorm1d ───────────────────────────────────────────────────────

/// Batch normalization over a 2D input [N, C] or 3D input [N, C, L].
/// Normalizes over the batch (and spatial) dimensions, per-channel.
pub struct BatchNorm1d {
    pub num_features: usize,
    pub gamma: Tensor,        // scale, shape [C]
    pub beta: Tensor,         // shift, shape [C]
    pub running_mean: Tensor, // [C]
    pub running_var: Tensor,  // [C]
    pub eps: f32,
    pub momentum: f32,
    training: bool,
}

impl BatchNorm1d {
    pub fn new(num_features: usize) -> Self {
        BatchNorm1d {
            num_features,
            gamma: Tensor::ones(&[num_features]),
            beta: Tensor::zeros(&[num_features]),
            running_mean: Tensor::zeros(&[num_features]),
            running_var: Tensor::ones(&[num_features]),
            eps: 1e-5,
            momentum: 0.1,
            training: true,
        }
    }
}

impl Module for BatchNorm1d {
    /// Forward: input [N, C] or [N, C, L] -> same shape
    fn forward(&self, input: &Tensor) -> Tensor {
        let ndim = input.shape.len();
        assert!(ndim == 2 || ndim == 3, "BatchNorm1d expects 2D or 3D input");
        let c = input.shape[1];
        assert_eq!(c, self.num_features);
        let batch = input.shape[0];
        let spatial: usize = if ndim == 3 { input.shape[2] } else { 1 };

        let mut output = vec![0.0f32; input.numel()];

        for ci in 0..c {
            let (mean, var) = if self.training {
                // Compute mean and variance over batch and spatial dims
                let mut sum = 0.0f32;
                let count = (batch * spatial) as f32;
                for n in 0..batch {
                    for s in 0..spatial {
                        let idx = n * c * spatial + ci * spatial + s;
                        sum += input.data[idx];
                    }
                }
                let mean = sum / count;
                let mut var_sum = 0.0f32;
                for n in 0..batch {
                    for s in 0..spatial {
                        let idx = n * c * spatial + ci * spatial + s;
                        let diff = input.data[idx] - mean;
                        var_sum += diff * diff;
                    }
                }
                let var = var_sum / count;
                (mean, var)
            } else {
                (self.running_mean.data[ci], self.running_var.data[ci])
            };

            let inv_std = 1.0 / (var + self.eps).sqrt();
            let g = self.gamma.data[ci];
            let b = self.beta.data[ci];

            for n in 0..batch {
                for s in 0..spatial {
                    let idx = n * c * spatial + ci * spatial + s;
                    output[idx] = g * (input.data[idx] - mean) * inv_std + b;
                }
            }
        }

        Tensor::new(output, input.shape.clone())
    }

    fn parameters(&self) -> Vec<&Tensor> {
        vec![&self.gamma, &self.beta]
    }

    fn parameters_mut(&mut self) -> Vec<&mut Tensor> {
        vec![&mut self.gamma, &mut self.beta]
    }

    fn set_training(&mut self, training: bool) {
        self.training = training;
    }

    fn is_training(&self) -> bool {
        self.training
    }

    fn name(&self) -> &str {
        "BatchNorm1d"
    }
}

// ── BatchNorm2d ───────────────────────────────────────────────────────

/// Batch normalization over a 4D input [N, C, H, W].
/// Normalizes over N, H, W dimensions per channel.
pub struct BatchNorm2d {
    pub num_features: usize,
    pub gamma: Tensor,        // [C]
    pub beta: Tensor,         // [C]
    pub running_mean: Tensor, // [C]
    pub running_var: Tensor,  // [C]
    pub eps: f32,
    pub momentum: f32,
    training: bool,
}

impl BatchNorm2d {
    pub fn new(num_features: usize) -> Self {
        BatchNorm2d {
            num_features,
            gamma: Tensor::ones(&[num_features]),
            beta: Tensor::zeros(&[num_features]),
            running_mean: Tensor::zeros(&[num_features]),
            running_var: Tensor::ones(&[num_features]),
            eps: 1e-5,
            momentum: 0.1,
            training: true,
        }
    }
}

impl Module for BatchNorm2d {
    /// Forward: input [N, C, H, W] -> same shape
    fn forward(&self, input: &Tensor) -> Tensor {
        assert_eq!(
            input.shape.len(),
            4,
            "BatchNorm2d expects 4D input [N, C, H, W]"
        );
        let batch = input.shape[0];
        let c = input.shape[1];
        let h = input.shape[2];
        let w = input.shape[3];
        assert_eq!(c, self.num_features);

        let hw = h * w;
        let mut output = vec![0.0f32; input.numel()];

        for ci in 0..c {
            let (mean, var) = if self.training {
                let count = (batch * hw) as f32;
                let mut sum = 0.0f32;
                for n in 0..batch {
                    let base = n * c * hw + ci * hw;
                    for j in 0..hw {
                        sum += input.data[base + j];
                    }
                }
                let mean = sum / count;
                let mut var_sum = 0.0f32;
                for n in 0..batch {
                    let base = n * c * hw + ci * hw;
                    for j in 0..hw {
                        let diff = input.data[base + j] - mean;
                        var_sum += diff * diff;
                    }
                }
                let var = var_sum / count;
                (mean, var)
            } else {
                (self.running_mean.data[ci], self.running_var.data[ci])
            };

            let inv_std = 1.0 / (var + self.eps).sqrt();
            let g = self.gamma.data[ci];
            let b = self.beta.data[ci];

            for n in 0..batch {
                let base = n * c * hw + ci * hw;
                for j in 0..hw {
                    output[base + j] = g * (input.data[base + j] - mean) * inv_std + b;
                }
            }
        }

        Tensor::new(output, input.shape.clone())
    }

    fn parameters(&self) -> Vec<&Tensor> {
        vec![&self.gamma, &self.beta]
    }

    fn parameters_mut(&mut self) -> Vec<&mut Tensor> {
        vec![&mut self.gamma, &mut self.beta]
    }

    fn set_training(&mut self, training: bool) {
        self.training = training;
    }

    fn is_training(&self) -> bool {
        self.training
    }

    fn name(&self) -> &str {
        "BatchNorm2d"
    }
}
