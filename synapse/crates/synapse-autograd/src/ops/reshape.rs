use crate::function::GradFn;
use crate::graph::Graph;
use crate::tensor::Tensor;
use crate::variable::VariableId;

// ── Reshape ────────────────────────────────────────────────────────

pub struct ReshapeBackward {
    input_ids: Vec<VariableId>,
    input_shape: Vec<usize>,
}

impl GradFn for ReshapeBackward {
    fn backward(&self, grad_output: &Tensor) -> Vec<Option<Tensor>> {
        vec![Some(grad_output.reshape(&self.input_shape))]
    }
    fn inputs(&self) -> &[VariableId] {
        &self.input_ids
    }
}

// ── Transpose ──────────────────────────────────────────────────────

pub struct TransposeBackward {
    input_ids: Vec<VariableId>,
    dim0: usize,
    dim1: usize,
}

impl GradFn for TransposeBackward {
    fn backward(&self, grad_output: &Tensor) -> Vec<Option<Tensor>> {
        // Transpose is its own inverse
        vec![Some(grad_output.transpose_dims(self.dim0, self.dim1))]
    }
    fn inputs(&self) -> &[VariableId] {
        &self.input_ids
    }
}

// ── Graph methods ──────────────────────────────────────────────────

impl Graph {
    pub fn reshape(&mut self, a: VariableId, shape: &[usize]) -> VariableId {
        let input_shape = self.variables[&a].data.shape.clone();
        let output = self.variables[&a].data.reshape(shape);
        if !self.should_track(&[a]) {
            return self.untracked(output);
        }
        self.record_op(
            Box::new(ReshapeBackward {
                input_ids: vec![a],
                input_shape,
            }),
            &[a],
            output,
        )
    }

    pub fn transpose(&mut self, a: VariableId, dim0: usize, dim1: usize) -> VariableId {
        let output = self.variables[&a].data.transpose_dims(dim0, dim1);
        if !self.should_track(&[a]) {
            return self.untracked(output);
        }
        self.record_op(
            Box::new(TransposeBackward {
                input_ids: vec![a],
                dim0,
                dim1,
            }),
            &[a],
            output,
        )
    }
}
