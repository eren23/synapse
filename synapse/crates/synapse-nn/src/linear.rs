//! Fully-connected (dense) linear layer: y = xW^T + b

use synapse_autograd::Tensor;

use crate::init::xavier_uniform;
use crate::module::Module;

pub struct Linear {
    pub weight: Tensor,       // [out_features, in_features]
    pub bias: Option<Tensor>, // [out_features]
    training: bool,
}

impl Linear {
    /// Create a new Linear layer with Xavier uniform initialization.
    /// weight shape: [out_features, in_features]
    /// bias shape: [out_features] (if bias=true)
    pub fn new(in_features: usize, out_features: usize, bias: bool) -> Self {
        let weight = xavier_uniform(&[out_features, in_features]);
        let bias_tensor = if bias {
            Some(Tensor::zeros(&[out_features]))
        } else {
            None
        };
        Linear {
            weight,
            bias: bias_tensor,
            training: true,
        }
    }

    pub fn in_features(&self) -> usize {
        self.weight.shape[1]
    }

    pub fn out_features(&self) -> usize {
        self.weight.shape[0]
    }
}

impl Module for Linear {
    /// Forward: input [batch, in_features] -> output [batch, out_features]
    /// Computes: output = input @ weight^T + bias
    fn forward(&self, input: &Tensor) -> Tensor {
        assert_eq!(
            input.shape.len(),
            2,
            "Linear expects 2D input [batch, in_features]"
        );
        assert_eq!(
            input.shape[1],
            self.in_features(),
            "input features {} != expected {}",
            input.shape[1],
            self.in_features()
        );

        // input [B, in] @ weight^T [in, out] = [B, out]
        let wt = self.weight.transpose_2d();
        let mut output = input.matmul(&wt);

        if let Some(ref bias) = self.bias {
            // Broadcast bias [out] to [B, out]
            let bias_2d = bias.reshape(&[1, self.out_features()]);
            output = output.add_broadcast(&bias_2d);
        }

        output
    }

    fn parameters(&self) -> Vec<&Tensor> {
        let mut params = vec![&self.weight];
        if let Some(ref b) = self.bias {
            params.push(b);
        }
        params
    }

    fn parameters_mut(&mut self) -> Vec<&mut Tensor> {
        let mut params = vec![&mut self.weight];
        if let Some(ref mut b) = self.bias {
            params.push(b);
        }
        params
    }

    fn set_training(&mut self, training: bool) {
        self.training = training;
    }

    fn is_training(&self) -> bool {
        self.training
    }

    fn name(&self) -> &str {
        "Linear"
    }
}
