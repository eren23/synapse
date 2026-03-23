use crate::function::GradFn;
use crate::graph::Graph;
use crate::tensor::Tensor;
use crate::variable::VariableId;

// ── ReLU ───────────────────────────────────────────────────────────

pub struct ReluBackward {
    input_ids: Vec<VariableId>,
    input_data: Tensor,
}

impl GradFn for ReluBackward {
    fn backward(&self, grad_output: &Tensor) -> Vec<Option<Tensor>> {
        let data: Vec<f32> = grad_output
            .data
            .iter()
            .zip(&self.input_data.data)
            .map(|(&g, &x)| if x > 0.0 { g } else { 0.0 })
            .collect();
        vec![Some(Tensor::new(data, grad_output.shape.clone()))]
    }
    fn inputs(&self) -> &[VariableId] {
        &self.input_ids
    }
}

// ── Sigmoid ────────────────────────────────────────────────────────

pub struct SigmoidBackward {
    input_ids: Vec<VariableId>,
    output_data: Tensor, // sigmoid(x)
}

impl GradFn for SigmoidBackward {
    fn backward(&self, grad_output: &Tensor) -> Vec<Option<Tensor>> {
        // d(sigmoid)/dx = sigmoid * (1 - sigmoid)
        let data: Vec<f32> = grad_output
            .data
            .iter()
            .zip(&self.output_data.data)
            .map(|(&g, &s)| g * s * (1.0 - s))
            .collect();
        vec![Some(Tensor::new(data, grad_output.shape.clone()))]
    }
    fn inputs(&self) -> &[VariableId] {
        &self.input_ids
    }
}

// ── Tanh ───────────────────────────────────────────────────────────

pub struct TanhBackward {
    input_ids: Vec<VariableId>,
    output_data: Tensor, // tanh(x)
}

impl GradFn for TanhBackward {
    fn backward(&self, grad_output: &Tensor) -> Vec<Option<Tensor>> {
        // d(tanh)/dx = 1 - tanh^2
        let data: Vec<f32> = grad_output
            .data
            .iter()
            .zip(&self.output_data.data)
            .map(|(&g, &t)| g * (1.0 - t * t))
            .collect();
        vec![Some(Tensor::new(data, grad_output.shape.clone()))]
    }
    fn inputs(&self) -> &[VariableId] {
        &self.input_ids
    }
}

// ── GELU ───────────────────────────────────────────────────────────

pub struct GeluBackward {
    input_ids: Vec<VariableId>,
    input_data: Tensor,
}

impl GradFn for GeluBackward {
    fn backward(&self, grad_output: &Tensor) -> Vec<Option<Tensor>> {
        let c = (2.0f32 / std::f32::consts::PI).sqrt();
        let data: Vec<f32> = grad_output
            .data
            .iter()
            .zip(&self.input_data.data)
            .map(|(&g, &x)| {
                let inner = c * (x + 0.044715 * x * x * x);
                let tanh_val = inner.tanh();
                let sech2 = 1.0 - tanh_val * tanh_val;
                let d_inner = c * (1.0 + 3.0 * 0.044715 * x * x);
                // gelu'(x) = 0.5*(1+tanh) + 0.5*x*sech^2*d_inner
                g * (0.5 * (1.0 + tanh_val) + 0.5 * x * sech2 * d_inner)
            })
            .collect();
        vec![Some(Tensor::new(data, grad_output.shape.clone()))]
    }
    fn inputs(&self) -> &[VariableId] {
        &self.input_ids
    }
}

// ── Graph methods ──────────────────────────────────────────────────

impl Graph {
    pub fn relu(&mut self, a: VariableId) -> VariableId {
        let input_data = self.variables[&a].data.clone();
        let output = input_data.relu();
        if !self.should_track(&[a]) {
            return self.untracked(output);
        }
        self.record_op(
            Box::new(ReluBackward { input_ids: vec![a], input_data }),
            &[a],
            output,
        )
    }

    pub fn sigmoid(&mut self, a: VariableId) -> VariableId {
        let output = self.variables[&a].data.sigmoid();
        if !self.should_track(&[a]) {
            return self.untracked(output.clone());
        }
        let output_data = output.clone();
        self.record_op(
            Box::new(SigmoidBackward { input_ids: vec![a], output_data }),
            &[a],
            output,
        )
    }

    pub fn tanh_op(&mut self, a: VariableId) -> VariableId {
        let output = self.variables[&a].data.tanh_act();
        if !self.should_track(&[a]) {
            return self.untracked(output.clone());
        }
        let output_data = output.clone();
        self.record_op(
            Box::new(TanhBackward { input_ids: vec![a], output_data }),
            &[a],
            output,
        )
    }

    pub fn gelu(&mut self, a: VariableId) -> VariableId {
        let input_data = self.variables[&a].data.clone();
        let output = input_data.gelu();
        if !self.should_track(&[a]) {
            return self.untracked(output);
        }
        self.record_op(
            Box::new(GeluBackward { input_ids: vec![a], input_data }),
            &[a],
            output,
        )
    }
}
