//! Activation function modules: ReLU, Sigmoid, Tanh, GELU, Softmax.

use synapse_autograd::Tensor;

use crate::module::Module;

// ── ReLU ──────────────────────────────────────────────────────────────

pub struct ReLU {
    training: bool,
}

impl ReLU {
    pub fn new() -> Self {
        ReLU { training: true }
    }
}

impl Default for ReLU {
    fn default() -> Self {
        Self::new()
    }
}

impl Module for ReLU {
    fn forward(&self, input: &Tensor) -> Tensor {
        input.relu()
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
        "ReLU"
    }
}

// ── Sigmoid ───────────────────────────────────────────────────────────

pub struct Sigmoid {
    training: bool,
}

impl Sigmoid {
    pub fn new() -> Self {
        Sigmoid { training: true }
    }
}

impl Default for Sigmoid {
    fn default() -> Self {
        Self::new()
    }
}

impl Module for Sigmoid {
    fn forward(&self, input: &Tensor) -> Tensor {
        input.sigmoid()
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
        "Sigmoid"
    }
}

// ── Tanh ──────────────────────────────────────────────────────────────

pub struct Tanh {
    training: bool,
}

impl Tanh {
    pub fn new() -> Self {
        Tanh { training: true }
    }
}

impl Default for Tanh {
    fn default() -> Self {
        Self::new()
    }
}

impl Module for Tanh {
    fn forward(&self, input: &Tensor) -> Tensor {
        input.tanh_act()
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
        "Tanh"
    }
}

// ── GELU ──────────────────────────────────────────────────────────────

pub struct GELU {
    training: bool,
}

impl GELU {
    pub fn new() -> Self {
        GELU { training: true }
    }
}

impl Default for GELU {
    fn default() -> Self {
        Self::new()
    }
}

impl Module for GELU {
    fn forward(&self, input: &Tensor) -> Tensor {
        input.gelu()
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
        "GELU"
    }
}

// ── Softmax ───────────────────────────────────────────────────────────

pub struct Softmax {
    dim: usize,
    training: bool,
}

impl Softmax {
    pub fn new(dim: usize) -> Self {
        Softmax {
            dim,
            training: true,
        }
    }
}

impl Module for Softmax {
    fn forward(&self, input: &Tensor) -> Tensor {
        input.softmax_axis(self.dim)
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
        "Softmax"
    }
}
