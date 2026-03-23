//! Module trait and ModuleList container.

use synapse_autograd::Tensor;

/// Core trait for all neural network layers.
pub trait Module {
    /// Compute the forward pass.
    fn forward(&self, input: &Tensor) -> Tensor;

    /// Return references to all learnable parameters.
    fn parameters(&self) -> Vec<&Tensor>;

    /// Return mutable references to all learnable parameters.
    fn parameters_mut(&mut self) -> Vec<&mut Tensor>;

    /// Set training mode (affects dropout, batchnorm, etc.).
    fn set_training(&mut self, training: bool);

    /// Check if the module is in training mode.
    fn is_training(&self) -> bool;

    /// Return the module name.
    fn name(&self) -> &str;
}

/// A list of modules, useful for holding sub-modules.
pub struct ModuleList {
    modules: Vec<Box<dyn Module>>,
}

impl ModuleList {
    pub fn new() -> Self {
        ModuleList {
            modules: Vec::new(),
        }
    }

    pub fn push(&mut self, module: Box<dyn Module>) {
        self.modules.push(module);
    }

    pub fn len(&self) -> usize {
        self.modules.len()
    }

    pub fn is_empty(&self) -> bool {
        self.modules.is_empty()
    }

    pub fn get(&self, idx: usize) -> Option<&dyn Module> {
        self.modules.get(idx).map(|m| m.as_ref())
    }

    pub fn get_mut(&mut self, idx: usize) -> Option<&mut Box<dyn Module>> {
        self.modules.get_mut(idx)
    }

    pub fn iter(&self) -> impl Iterator<Item = &dyn Module> {
        self.modules.iter().map(|m| m.as_ref())
    }
}

impl Default for ModuleList {
    fn default() -> Self {
        Self::new()
    }
}

impl Module for ModuleList {
    fn forward(&self, input: &Tensor) -> Tensor {
        let mut x = input.clone();
        for module in &self.modules {
            x = module.forward(&x);
        }
        x
    }

    fn parameters(&self) -> Vec<&Tensor> {
        self.modules.iter().flat_map(|m| m.parameters()).collect()
    }

    fn parameters_mut(&mut self) -> Vec<&mut Tensor> {
        self.modules
            .iter_mut()
            .flat_map(|m| m.parameters_mut())
            .collect()
    }

    fn set_training(&mut self, training: bool) {
        for module in &mut self.modules {
            module.set_training(training);
        }
    }

    fn is_training(&self) -> bool {
        self.modules.first().map_or(false, |m| m.is_training())
    }

    fn name(&self) -> &str {
        "ModuleList"
    }
}
