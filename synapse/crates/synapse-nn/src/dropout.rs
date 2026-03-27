//! Dropout layer: randomly zeros elements during training.

use rand::Rng;
use synapse_autograd::Tensor;

use crate::module::Module;

pub struct Dropout {
    pub p: f32, // probability of dropping
    training: bool,
}

impl Dropout {
    /// Create a Dropout layer with drop probability `p`.
    pub fn new(p: f32) -> Self {
        assert!((0.0..1.0).contains(&p), "dropout p must be in [0, 1)");
        Dropout { p, training: true }
    }
}

impl Module for Dropout {
    /// During training: randomly zero elements with probability p, scale remaining by 1/(1-p).
    /// During inference: identity (pass through unchanged).
    fn forward(&self, input: &Tensor) -> Tensor {
        if !self.training || self.p == 0.0 {
            return input.clone();
        }

        let mut rng = rand::thread_rng();
        let scale = 1.0 / (1.0 - self.p);
        let data: Vec<f32> = input
            .data
            .iter()
            .map(|&x| {
                if rng.gen::<f32>() < self.p {
                    0.0
                } else {
                    x * scale
                }
            })
            .collect();

        Tensor::new(data, input.shape.clone())
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
        "Dropout"
    }
}
