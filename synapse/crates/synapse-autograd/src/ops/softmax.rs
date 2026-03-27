use crate::function::GradFn;
use crate::graph::Graph;
use crate::tensor::Tensor;
use crate::variable::VariableId;

// ── Softmax ────────────────────────────────────────────────────────

pub struct SoftmaxBackward {
    input_ids: Vec<VariableId>,
    output_data: Tensor,
    axis: usize,
}

impl GradFn for SoftmaxBackward {
    fn backward(&self, grad_output: &Tensor) -> Vec<Option<Tensor>> {
        // dx = s * (dout - sum(dout * s, axis, keepdim=true))
        let s = &self.output_data;
        let ds = grad_output.mul(s);
        let sum_ds = ds.sum_axis(self.axis, true).broadcast_to(&s.shape);
        let grad = s.mul(&grad_output.sub(&sum_ds));
        vec![Some(grad)]
    }
    fn inputs(&self) -> &[VariableId] {
        &self.input_ids
    }
}

// ── LogSoftmax ─────────────────────────────────────────────────────

pub struct LogSoftmaxBackward {
    input_ids: Vec<VariableId>,
    softmax_data: Tensor, // softmax(input), not log_softmax
    axis: usize,
}

impl GradFn for LogSoftmaxBackward {
    fn backward(&self, grad_output: &Tensor) -> Vec<Option<Tensor>> {
        // dx = dout - softmax * sum(dout, axis, keepdim)
        let sum_grad = grad_output
            .sum_axis(self.axis, true)
            .broadcast_to(&self.softmax_data.shape);
        let grad = grad_output.sub(&self.softmax_data.mul(&sum_grad));
        vec![Some(grad)]
    }
    fn inputs(&self) -> &[VariableId] {
        &self.input_ids
    }
}

// ── Graph methods ──────────────────────────────────────────────────

impl Graph {
    pub fn softmax(&mut self, a: VariableId, axis: usize) -> VariableId {
        let output = self.variables[&a].data.softmax_axis(axis);
        if !self.should_track(&[a]) {
            return self.untracked(output.clone());
        }
        let output_data = output.clone();
        self.record_op(
            Box::new(SoftmaxBackward {
                input_ids: vec![a],
                output_data,
                axis,
            }),
            &[a],
            output,
        )
    }

    pub fn log_softmax(&mut self, a: VariableId, axis: usize) -> VariableId {
        let softmax_data = self.variables[&a].data.softmax_axis(axis);
        let output = self.variables[&a].data.log_softmax_axis(axis);
        if !self.should_track(&[a]) {
            return self.untracked(output);
        }
        self.record_op(
            Box::new(LogSoftmaxBackward {
                input_ids: vec![a],
                softmax_data,
                axis,
            }),
            &[a],
            output,
        )
    }
}
