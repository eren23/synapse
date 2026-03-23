use crate::tensor::Tensor;
use crate::variable::VariableId;

/// Backward function for a computation graph node.
pub trait GradFn {
    /// Compute gradients w.r.t. each input given the output gradient.
    fn backward(&self, grad_output: &Tensor) -> Vec<Option<Tensor>>;
    /// Return the variable IDs of inputs to this operation.
    fn inputs(&self) -> &[VariableId];
}
