use crate::function::GradFn;
use crate::graph::Graph;
use crate::tensor::Tensor;
use crate::variable::VariableId;

// ── MSE Loss ───────────────────────────────────────────────────────

pub struct MseLossBackward {
    input_ids: Vec<VariableId>,
    pred_data: Tensor,
    target_data: Tensor,
}

impl GradFn for MseLossBackward {
    fn backward(&self, grad_output: &Tensor) -> Vec<Option<Tensor>> {
        let n = self.pred_data.numel() as f32;
        let diff = self.pred_data.sub(&self.target_data);
        let scale = 2.0 * grad_output.data[0] / n;
        let grad_pred = diff.scale(scale);
        let grad_target = grad_pred.neg();
        vec![Some(grad_pred), Some(grad_target)]
    }
    fn inputs(&self) -> &[VariableId] {
        &self.input_ids
    }
}

// ── Cross-Entropy Loss ─────────────────────────────────────────────

pub struct CrossEntropyLossBackward {
    input_ids: Vec<VariableId>,
    softmax_data: Tensor, // softmax(pred)
    target_data: Tensor,
    log_softmax_data: Tensor,
    n_batch: usize,
}

impl GradFn for CrossEntropyLossBackward {
    fn backward(&self, grad_output: &Tensor) -> Vec<Option<Tensor>> {
        let n = self.n_batch as f32;
        let scale = grad_output.data[0] / n;
        // d/dpred = (softmax - target) / N
        let grad_pred = self.softmax_data.sub(&self.target_data).scale(scale);
        // d/dtarget = -log_softmax / N
        let grad_target = self.log_softmax_data.neg().scale(scale);
        vec![Some(grad_pred), Some(grad_target)]
    }
    fn inputs(&self) -> &[VariableId] {
        &self.input_ids
    }
}

// ── Graph methods ──────────────────────────────────────────────────

impl Graph {
    /// MSE loss: mean((pred - target)^2)
    pub fn mse_loss(&mut self, pred: VariableId, target: VariableId) -> VariableId {
        let pred_data = self.variables[&pred].data.clone();
        let target_data = self.variables[&target].data.clone();
        let diff = pred_data.sub(&target_data);
        let output = diff.mul(&diff).mean_all();
        if !self.should_track(&[pred, target]) {
            return self.untracked(output);
        }
        self.record_op(
            Box::new(MseLossBackward { input_ids: vec![pred, target], pred_data, target_data }),
            &[pred, target],
            output,
        )
    }

    /// Cross-entropy loss: -mean(sum(target * log_softmax(pred), axis=1))
    /// pred: [N, C] logits, target: [N, C] probability distribution
    pub fn cross_entropy_loss(&mut self, pred: VariableId, target: VariableId) -> VariableId {
        let pred_data = self.variables[&pred].data.clone();
        let target_data = self.variables[&target].data.clone();
        let n = pred_data.shape[0];

        let log_sm = pred_data.log_softmax_axis(1);
        let softmax_data = pred_data.softmax_axis(1);
        // loss = -(1/N) * sum(target * log_softmax)
        let per_elem = target_data.mul(&log_sm);
        let loss_val = -(per_elem.data.iter().sum::<f32>()) / n as f32;
        let output = Tensor::scalar(loss_val);

        if !self.should_track(&[pred, target]) {
            return self.untracked(output);
        }
        self.record_op(
            Box::new(CrossEntropyLossBackward {
                input_ids: vec![pred, target],
                softmax_data,
                target_data,
                log_softmax_data: log_sm,
                n_batch: n,
            }),
            &[pred, target],
            output,
        )
    }
}
