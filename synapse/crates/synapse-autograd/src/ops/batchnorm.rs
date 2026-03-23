use crate::function::GradFn;
use crate::graph::Graph;
use crate::tensor::Tensor;
use crate::variable::VariableId;

pub struct BatchNormBackward {
    input_ids: Vec<VariableId>, // [input, gamma, beta]
    x_hat: Tensor,              // normalized
    inv_std: Tensor,            // [C]
    gamma_data: Tensor,         // [C]
    n_batch: usize,
}

impl GradFn for BatchNormBackward {
    fn backward(&self, grad_output: &Tensor) -> Vec<Option<Tensor>> {
        let n = self.n_batch;
        let c = self.gamma_data.numel();
        let nf = n as f32;

        // dbeta = sum(dout, axis=0) → [C]
        let grad_beta = grad_output.sum_axis(0, false);

        // dgamma = sum(dout * x_hat, axis=0) → [C]
        let grad_gamma = grad_output.mul(&self.x_hat).sum_axis(0, false);

        // dx using efficient formula:
        // dx_hat = dout * gamma  [N, C]
        let gamma_broad = self.gamma_data.reshape(&[1, c]).broadcast_to(&[n, c]);
        let dx_hat = grad_output.mul(&gamma_broad);

        // dx = inv_std/N * (N*dx_hat - sum(dx_hat) - x_hat*sum(dx_hat*x_hat))
        let inv_std_broad = self.inv_std.reshape(&[1, c]).broadcast_to(&[n, c]);
        let sum_dx_hat = dx_hat.sum_axis(0, true).broadcast_to(&[n, c]);
        let sum_dx_hat_xhat = dx_hat.mul(&self.x_hat).sum_axis(0, true).broadcast_to(&[n, c]);

        // N*dx_hat
        let term1 = dx_hat.scale(nf);
        // - sum(dx_hat)
        let term2 = term1.sub(&sum_dx_hat);
        // - x_hat * sum(dx_hat * x_hat)
        let term3 = term2.sub(&self.x_hat.mul(&sum_dx_hat_xhat));
        // inv_std / N
        let grad_input = inv_std_broad.scale(1.0 / nf).mul(&term3);

        vec![Some(grad_input), Some(grad_gamma), Some(grad_beta)]
    }
    fn inputs(&self) -> &[VariableId] {
        &self.input_ids
    }
}

impl Graph {
    /// Batch normalization: input [N,C], gamma [C], beta [C]
    pub fn batch_norm(
        &mut self, input: VariableId, gamma: VariableId, beta: VariableId, eps: f32,
    ) -> VariableId {
        let input_data = self.variables[&input].data.clone();
        let gamma_data = self.variables[&gamma].data.clone();
        let beta_data = self.variables[&beta].data.clone();

        let n = input_data.shape[0];
        let c = input_data.shape[1];

        // mean [C], var [C]
        let mean = input_data.mean_axis(0, false);
        let mean_broad = mean.reshape(&[1, c]).broadcast_to(&[n, c]);
        let diff = input_data.sub(&mean_broad);
        let var = diff.mul(&diff).mean_axis(0, false);

        // inv_std [C]
        let inv_std_data: Vec<f32> = var.data.iter().map(|&v| 1.0 / (v + eps).sqrt()).collect();
        let inv_std = Tensor::new(inv_std_data, vec![c]);

        // x_hat = (x - mean) * inv_std  [N, C]
        let inv_std_broad = inv_std.reshape(&[1, c]).broadcast_to(&[n, c]);
        let x_hat = diff.mul(&inv_std_broad);

        // y = gamma * x_hat + beta  [N, C]
        let gamma_broad = gamma_data.reshape(&[1, c]).broadcast_to(&[n, c]);
        let beta_broad = beta_data.reshape(&[1, c]).broadcast_to(&[n, c]);
        let output = x_hat.mul(&gamma_broad).add(&beta_broad);

        if !self.should_track(&[input, gamma, beta]) {
            return self.untracked(output);
        }
        self.record_op(
            Box::new(BatchNormBackward {
                input_ids: vec![input, gamma, beta],
                x_hat,
                inv_std,
                gamma_data,
                n_batch: n,
            }),
            &[input, gamma, beta],
            output,
        )
    }
}
