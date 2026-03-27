use crate::function::GradFn;
use crate::graph::Graph;
use crate::tensor::Tensor;
use crate::variable::VariableId;

// ── Add (broadcasting) ─────────────────────────────────────────────

pub struct AddBackward {
    input_ids: Vec<VariableId>,
    a_shape: Vec<usize>,
    b_shape: Vec<usize>,
}

impl GradFn for AddBackward {
    fn backward(&self, grad_output: &Tensor) -> Vec<Option<Tensor>> {
        vec![
            Some(grad_output.reduce_sum_to(&self.a_shape)),
            Some(grad_output.reduce_sum_to(&self.b_shape)),
        ]
    }
    fn inputs(&self) -> &[VariableId] {
        &self.input_ids
    }
}

// ── Sub (broadcasting) ─────────────────────────────────────────────

pub struct SubBackward {
    input_ids: Vec<VariableId>,
    a_shape: Vec<usize>,
    b_shape: Vec<usize>,
}

impl GradFn for SubBackward {
    fn backward(&self, grad_output: &Tensor) -> Vec<Option<Tensor>> {
        vec![
            Some(grad_output.reduce_sum_to(&self.a_shape)),
            Some(grad_output.neg().reduce_sum_to(&self.b_shape)),
        ]
    }
    fn inputs(&self) -> &[VariableId] {
        &self.input_ids
    }
}

// ── Mul (broadcasting) ─────────────────────────────────────────────

pub struct MulBackward {
    input_ids: Vec<VariableId>,
    a_data: Tensor,
    b_data: Tensor,
    a_shape: Vec<usize>,
    b_shape: Vec<usize>,
}

impl GradFn for MulBackward {
    fn backward(&self, grad_output: &Tensor) -> Vec<Option<Tensor>> {
        let out_shape = &grad_output.shape;
        let b_broad = self.b_data.broadcast_to(out_shape);
        let a_broad = self.a_data.broadcast_to(out_shape);
        vec![
            Some(grad_output.mul(&b_broad).reduce_sum_to(&self.a_shape)),
            Some(grad_output.mul(&a_broad).reduce_sum_to(&self.b_shape)),
        ]
    }
    fn inputs(&self) -> &[VariableId] {
        &self.input_ids
    }
}

// ── Div (broadcasting) ─────────────────────────────────────────────

pub struct DivBackward {
    input_ids: Vec<VariableId>,
    a_data: Tensor,
    b_data: Tensor,
    a_shape: Vec<usize>,
    b_shape: Vec<usize>,
}

impl GradFn for DivBackward {
    fn backward(&self, grad_output: &Tensor) -> Vec<Option<Tensor>> {
        let out_shape = &grad_output.shape;
        let b_broad = self.b_data.broadcast_to(out_shape);
        let a_broad = self.a_data.broadcast_to(out_shape);
        let grad_a = grad_output.div(&b_broad).reduce_sum_to(&self.a_shape);
        let b_sq = b_broad.mul(&b_broad);
        let grad_b = grad_output
            .neg()
            .mul(&a_broad)
            .div(&b_sq)
            .reduce_sum_to(&self.b_shape);
        vec![Some(grad_a), Some(grad_b)]
    }
    fn inputs(&self) -> &[VariableId] {
        &self.input_ids
    }
}

// ── Neg ────────────────────────────────────────────────────────────

pub struct NegBackward {
    input_ids: Vec<VariableId>,
}

impl GradFn for NegBackward {
    fn backward(&self, grad_output: &Tensor) -> Vec<Option<Tensor>> {
        vec![Some(grad_output.neg())]
    }
    fn inputs(&self) -> &[VariableId] {
        &self.input_ids
    }
}

// ── Graph methods ──────────────────────────────────────────────────

impl Graph {
    pub fn add(&mut self, a: VariableId, b: VariableId) -> VariableId {
        let a_shape = self.variables[&a].data.shape.clone();
        let b_shape = self.variables[&b].data.shape.clone();
        let output = self.variables[&a]
            .data
            .add_broadcast(&self.variables[&b].data);
        if !self.should_track(&[a, b]) {
            return self.untracked(output);
        }
        self.record_op(
            Box::new(AddBackward {
                input_ids: vec![a, b],
                a_shape,
                b_shape,
            }),
            &[a, b],
            output,
        )
    }

    pub fn sub(&mut self, a: VariableId, b: VariableId) -> VariableId {
        let a_shape = self.variables[&a].data.shape.clone();
        let b_shape = self.variables[&b].data.shape.clone();
        let output = self.variables[&a]
            .data
            .sub_broadcast(&self.variables[&b].data);
        if !self.should_track(&[a, b]) {
            return self.untracked(output);
        }
        self.record_op(
            Box::new(SubBackward {
                input_ids: vec![a, b],
                a_shape,
                b_shape,
            }),
            &[a, b],
            output,
        )
    }

    pub fn mul(&mut self, a: VariableId, b: VariableId) -> VariableId {
        let a_data = self.variables[&a].data.clone();
        let b_data = self.variables[&b].data.clone();
        let output = a_data.mul_broadcast(&b_data);
        if !self.should_track(&[a, b]) {
            return self.untracked(output);
        }
        let a_shape = a_data.shape.clone();
        let b_shape = b_data.shape.clone();
        self.record_op(
            Box::new(MulBackward {
                input_ids: vec![a, b],
                a_data,
                b_data,
                a_shape,
                b_shape,
            }),
            &[a, b],
            output,
        )
    }

    pub fn div(&mut self, a: VariableId, b: VariableId) -> VariableId {
        let a_data = self.variables[&a].data.clone();
        let b_data = self.variables[&b].data.clone();
        let output = a_data.div_broadcast(&b_data);
        if !self.should_track(&[a, b]) {
            return self.untracked(output);
        }
        let a_shape = a_data.shape.clone();
        let b_shape = b_data.shape.clone();
        self.record_op(
            Box::new(DivBackward {
                input_ids: vec![a, b],
                a_data,
                b_data,
                a_shape,
                b_shape,
            }),
            &[a, b],
            output,
        )
    }

    pub fn neg(&mut self, a: VariableId) -> VariableId {
        let output = self.variables[&a].data.neg();
        if !self.should_track(&[a]) {
            return self.untracked(output);
        }
        self.record_op(Box::new(NegBackward { input_ids: vec![a] }), &[a], output)
    }
}
