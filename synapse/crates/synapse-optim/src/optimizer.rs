use std::collections::HashMap;

/// A trainable parameter holding data and an optional gradient.
///
/// Optimizers read `.grad` and update `.data` in-place during `step()`.
#[derive(Clone, Debug)]
pub struct Param {
    pub data: Vec<f32>,
    pub grad: Option<Vec<f32>>,
}

impl Param {
    pub fn new(data: Vec<f32>) -> Self {
        Param { data, grad: None }
    }

    pub fn with_grad(data: Vec<f32>, grad: Vec<f32>) -> Self {
        assert_eq!(data.len(), grad.len(), "data and grad length mismatch");
        Param {
            data,
            grad: Some(grad),
        }
    }

    pub fn numel(&self) -> usize {
        self.data.len()
    }
}

/// A named group of parameter indices with per-group hyperparameters.
#[derive(Clone, Debug)]
pub struct ParamGroup {
    /// Indices into the parameter slice passed to `step()`.
    pub params: Vec<usize>,
    /// Hyperparameters for this group (keys are optimizer-specific).
    pub hyper: HashMap<String, f32>,
}

impl ParamGroup {
    pub fn new(params: Vec<usize>) -> Self {
        ParamGroup {
            params,
            hyper: HashMap::new(),
        }
    }

    pub fn with_hyper(params: Vec<usize>, hyper: HashMap<String, f32>) -> Self {
        ParamGroup { params, hyper }
    }
}

/// Serialisable optimizer state: maps string keys to vectors of f32.
pub type StateDict = HashMap<String, Vec<f32>>;

/// Common interface implemented by all optimizers.
pub trait Optimizer {
    /// Perform a single optimization step, updating parameter data in-place.
    fn step(&mut self, params: &mut [Param]);

    /// Zero out all gradients.
    fn zero_grad(&self, params: &mut [Param]) {
        for p in params.iter_mut() {
            if let Some(ref mut g) = p.grad {
                for v in g.iter_mut() {
                    *v = 0.0;
                }
            }
        }
    }

    /// Register an additional parameter group with per-group hyperparameters.
    fn add_param_group(&mut self, group: ParamGroup);

    /// Export internal optimizer state (momentum buffers, step counts, etc.).
    fn state_dict(&self) -> StateDict;

    /// Restore optimizer state from a previously exported state dict.
    fn load_state_dict(&mut self, state: &StateDict);
}
