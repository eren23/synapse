//! Sequential container: chains modules in order.

use synapse_autograd::Tensor;

use crate::module::Module;

pub struct Sequential {
    layers: Vec<Box<dyn Module>>,
    training: bool,
}

impl Sequential {
    pub fn new() -> Self {
        Sequential {
            layers: Vec::new(),
            training: true,
        }
    }

    /// Add a layer to the end of the sequential chain.
    pub fn add(mut self, layer: Box<dyn Module>) -> Self {
        self.layers.push(layer);
        self
    }

    /// Push a layer (non-builder pattern).
    pub fn push(&mut self, layer: Box<dyn Module>) {
        self.layers.push(layer);
    }

    pub fn len(&self) -> usize {
        self.layers.len()
    }

    pub fn is_empty(&self) -> bool {
        self.layers.is_empty()
    }
}

impl Default for Sequential {
    fn default() -> Self {
        Self::new()
    }
}

impl Module for Sequential {
    fn forward(&self, input: &Tensor) -> Tensor {
        let mut x = input.clone();
        for layer in &self.layers {
            x = layer.forward(&x);
        }
        x
    }

    fn parameters(&self) -> Vec<&Tensor> {
        self.layers.iter().flat_map(|l| l.parameters()).collect()
    }

    fn parameters_mut(&mut self) -> Vec<&mut Tensor> {
        self.layers
            .iter_mut()
            .flat_map(|l| l.parameters_mut())
            .collect()
    }

    fn set_training(&mut self, training: bool) {
        self.training = training;
        for layer in &mut self.layers {
            layer.set_training(training);
        }
    }

    fn is_training(&self) -> bool {
        self.training
    }

    fn name(&self) -> &str {
        "Sequential"
    }
}
