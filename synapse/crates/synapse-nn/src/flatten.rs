//! Flatten layer: reshapes a contiguous range of dimensions into one.

use synapse_autograd::Tensor;

use crate::module::Module;

pub struct Flatten {
    pub start_dim: usize,
    pub end_dim: isize, // -1 means last dim
    training: bool,
}

impl Flatten {
    /// Create a Flatten layer that flattens dimensions [start_dim, end_dim].
    /// end_dim can be negative (-1 = last dimension).
    pub fn new(start_dim: usize, end_dim: isize) -> Self {
        Flatten {
            start_dim,
            end_dim,
            training: true,
        }
    }
}

impl Default for Flatten {
    /// Default: flatten all dims except batch (start_dim=1, end_dim=-1).
    fn default() -> Self {
        Flatten::new(1, -1)
    }
}

impl Module for Flatten {
    fn forward(&self, input: &Tensor) -> Tensor {
        let ndim = input.shape.len();
        let end = if self.end_dim < 0 {
            (ndim as isize + self.end_dim) as usize
        } else {
            self.end_dim as usize
        };
        assert!(self.start_dim <= end && end < ndim);

        // Compute new shape
        let mut new_shape = Vec::new();
        // Dims before start_dim
        for i in 0..self.start_dim {
            new_shape.push(input.shape[i]);
        }
        // Flattened dims
        let flat: usize = input.shape[self.start_dim..=end].iter().product();
        new_shape.push(flat);
        // Dims after end_dim
        for i in (end + 1)..ndim {
            new_shape.push(input.shape[i]);
        }

        input.reshape(&new_shape)
    }

    fn parameters(&self) -> Vec<&Tensor> {
        vec![]
    }

    fn parameters_mut(&mut self) -> Vec<&mut Tensor> {
        vec![]
    }

    fn set_training(&mut self, training: bool) {
        self.training = training;
    }

    fn is_training(&self) -> bool {
        self.training
    }

    fn name(&self) -> &str {
        "Flatten"
    }
}
