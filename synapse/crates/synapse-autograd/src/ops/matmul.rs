use crate::function::GradFn;
use crate::graph::Graph;
use crate::tensor::Tensor;
use crate::variable::VariableId;

pub struct MatMulBackward {
    input_ids: Vec<VariableId>,
    a_data: Tensor,
    b_data: Tensor,
}

impl GradFn for MatMulBackward {
    fn backward(&self, grad_output: &Tensor) -> Vec<Option<Tensor>> {
        // C = A @ B  →  dA = dC @ B^T,  dB = A^T @ dC
        let grad_a = grad_output.matmul(&self.b_data.transpose_2d());
        let grad_b = self.a_data.transpose_2d().matmul(grad_output);
        vec![Some(grad_a), Some(grad_b)]
    }
    fn inputs(&self) -> &[VariableId] {
        &self.input_ids
    }
}

impl Graph {
    pub fn matmul(&mut self, a: VariableId, b: VariableId) -> VariableId {
        let a_data = self.variables[&a].data.clone();
        let b_data = self.variables[&b].data.clone();
        let output = a_data.matmul(&b_data);
        if !self.should_track(&[a, b]) {
            return self.untracked(output);
        }
        self.record_op(
            Box::new(MatMulBackward { input_ids: vec![a, b], a_data, b_data }),
            &[a, b],
            output,
        )
    }
}
