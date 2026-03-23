use crate::function::GradFn;
use crate::graph::Graph;
use crate::tensor::Tensor;
use crate::variable::VariableId;

// ── Sum all ────────────────────────────────────────────────────────

pub struct SumAllBackward {
    input_ids: Vec<VariableId>,
    input_shape: Vec<usize>,
}

impl GradFn for SumAllBackward {
    fn backward(&self, grad_output: &Tensor) -> Vec<Option<Tensor>> {
        // d(sum(x))/dx = ones * grad_output[0]
        let n: usize = self.input_shape.iter().product();
        let data = vec![grad_output.data[0]; n];
        vec![Some(Tensor::new(data, self.input_shape.clone()))]
    }
    fn inputs(&self) -> &[VariableId] {
        &self.input_ids
    }
}

// ── Sum axis ───────────────────────────────────────────────────────

pub struct SumAxisBackward {
    input_ids: Vec<VariableId>,
    input_shape: Vec<usize>,
    axis: usize,
    keepdim: bool,
}

impl GradFn for SumAxisBackward {
    fn backward(&self, grad_output: &Tensor) -> Vec<Option<Tensor>> {
        // Expand grad_output back to input shape by broadcasting along summed axis
        let grad = if self.keepdim {
            grad_output.broadcast_to(&self.input_shape)
        } else {
            // Insert the removed axis back as size 1, then broadcast
            let mut expanded_shape = self.input_shape.clone();
            expanded_shape[self.axis] = 1;
            let reshaped = grad_output.reshape(&expanded_shape);
            reshaped.broadcast_to(&self.input_shape)
        };
        vec![Some(grad)]
    }
    fn inputs(&self) -> &[VariableId] {
        &self.input_ids
    }
}

// ── Mean all ───────────────────────────────────────────────────────

pub struct MeanAllBackward {
    input_ids: Vec<VariableId>,
    input_shape: Vec<usize>,
}

impl GradFn for MeanAllBackward {
    fn backward(&self, grad_output: &Tensor) -> Vec<Option<Tensor>> {
        let n: usize = self.input_shape.iter().product();
        let val = grad_output.data[0] / n as f32;
        vec![Some(Tensor::new(vec![val; n], self.input_shape.clone()))]
    }
    fn inputs(&self) -> &[VariableId] {
        &self.input_ids
    }
}

// ── Mean axis ──────────────────────────────────────────────────────

pub struct MeanAxisBackward {
    input_ids: Vec<VariableId>,
    input_shape: Vec<usize>,
    axis: usize,
    keepdim: bool,
}

impl GradFn for MeanAxisBackward {
    fn backward(&self, grad_output: &Tensor) -> Vec<Option<Tensor>> {
        let n = self.input_shape[self.axis] as f32;
        let scaled = grad_output.scale(1.0 / n);
        let grad = if self.keepdim {
            scaled.broadcast_to(&self.input_shape)
        } else {
            let mut expanded_shape = self.input_shape.clone();
            expanded_shape[self.axis] = 1;
            scaled.reshape(&expanded_shape).broadcast_to(&self.input_shape)
        };
        vec![Some(grad)]
    }
    fn inputs(&self) -> &[VariableId] {
        &self.input_ids
    }
}

// ── Graph methods ──────────────────────────────────────────────────

impl Graph {
    pub fn sum_all(&mut self, a: VariableId) -> VariableId {
        let input_shape = self.variables[&a].data.shape.clone();
        let output = self.variables[&a].data.sum_all();
        if !self.should_track(&[a]) {
            return self.untracked(output);
        }
        self.record_op(
            Box::new(SumAllBackward { input_ids: vec![a], input_shape }),
            &[a],
            output,
        )
    }

    pub fn sum_axis(&mut self, a: VariableId, axis: usize, keepdim: bool) -> VariableId {
        let input_shape = self.variables[&a].data.shape.clone();
        let output = self.variables[&a].data.sum_axis(axis, keepdim);
        if !self.should_track(&[a]) {
            return self.untracked(output);
        }
        self.record_op(
            Box::new(SumAxisBackward { input_ids: vec![a], input_shape, axis, keepdim }),
            &[a],
            output,
        )
    }

    pub fn mean_all(&mut self, a: VariableId) -> VariableId {
        let input_shape = self.variables[&a].data.shape.clone();
        let output = self.variables[&a].data.mean_all();
        if !self.should_track(&[a]) {
            return self.untracked(output);
        }
        self.record_op(
            Box::new(MeanAllBackward { input_ids: vec![a], input_shape }),
            &[a],
            output,
        )
    }

    pub fn mean_axis(&mut self, a: VariableId, axis: usize, keepdim: bool) -> VariableId {
        let input_shape = self.variables[&a].data.shape.clone();
        let output = self.variables[&a].data.mean_axis(axis, keepdim);
        if !self.should_track(&[a]) {
            return self.untracked(output);
        }
        self.record_op(
            Box::new(MeanAxisBackward { input_ids: vec![a], input_shape, axis, keepdim }),
            &[a],
            output,
        )
    }
}
