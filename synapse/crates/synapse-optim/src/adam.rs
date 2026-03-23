use std::collections::HashMap;

use crate::optimizer::{Optimizer, Param, ParamGroup, StateDict};

/// Per-parameter Adam state.
#[derive(Clone, Debug)]
struct AdamState {
    step: usize,
    /// First moment estimate (m).
    exp_avg: Vec<f32>,
    /// Second moment estimate (v).
    exp_avg_sq: Vec<f32>,
}

/// Adam optimizer with bias correction.
///
/// Matches PyTorch's `torch.optim.Adam` semantics exactly.
/// Set `adamw = true` for decoupled weight decay (AdamW).
pub struct Adam {
    pub lr: f32,
    pub beta1: f32,
    pub beta2: f32,
    pub eps: f32,
    pub weight_decay: f32,
    pub adamw: bool,
    states: HashMap<usize, AdamState>,
    param_groups: Vec<ParamGroup>,
}

impl Adam {
    pub fn new(lr: f32) -> Self {
        Adam {
            lr,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            weight_decay: 0.0,
            adamw: false,
            states: HashMap::new(),
            param_groups: Vec::new(),
        }
    }

    pub fn betas(mut self, beta1: f32, beta2: f32) -> Self {
        self.beta1 = beta1;
        self.beta2 = beta2;
        self
    }

    pub fn eps(mut self, eps: f32) -> Self {
        self.eps = eps;
        self
    }

    pub fn weight_decay(mut self, weight_decay: f32) -> Self {
        self.weight_decay = weight_decay;
        self
    }

    pub fn adamw(mut self, adamw: bool) -> Self {
        self.adamw = adamw;
        self
    }

    /// Step a single parameter with the given hyperparameters.
    fn step_param(
        &mut self,
        idx: usize,
        param: &mut Param,
        lr: f32,
        beta1: f32,
        beta2: f32,
        eps: f32,
        weight_decay: f32,
        adamw: bool,
    ) {
        let grad = match param.grad {
            Some(ref g) => g.clone(),
            None => return,
        };
        let n = param.data.len();

        // Initialise state if needed
        let state = self.states.entry(idx).or_insert_with(|| AdamState {
            step: 0,
            exp_avg: vec![0.0; n],
            exp_avg_sq: vec![0.0; n],
        });
        state.step += 1;
        let t = state.step;

        // Decoupled weight decay (AdamW): apply before gradient processing
        if adamw && weight_decay != 0.0 {
            for i in 0..n {
                param.data[i] *= 1.0 - lr * weight_decay;
            }
        }

        // L2 regularization (classic Adam): add to gradient
        let grad = if !adamw && weight_decay != 0.0 {
            grad.iter()
                .zip(param.data.iter())
                .map(|(&g, &p)| g + weight_decay * p)
                .collect::<Vec<f32>>()
        } else {
            grad
        };

        // Update biased first moment estimate: m = beta1*m + (1-beta1)*grad
        for i in 0..n {
            state.exp_avg[i] = beta1 * state.exp_avg[i] + (1.0 - beta1) * grad[i];
        }

        // Update biased second moment estimate: v = beta2*v + (1-beta2)*grad^2
        for i in 0..n {
            state.exp_avg_sq[i] =
                beta2 * state.exp_avg_sq[i] + (1.0 - beta2) * grad[i] * grad[i];
        }

        // Bias correction
        let bias_correction1 = 1.0 - beta1.powi(t as i32);
        let bias_correction2 = 1.0 - beta2.powi(t as i32);

        let step_size = lr / bias_correction1;
        let bias_correction2_sqrt = bias_correction2.sqrt();

        // param -= step_size * m / (sqrt(v / bias_correction2) + eps)
        for i in 0..n {
            let denom = (state.exp_avg_sq[i].sqrt() / bias_correction2_sqrt) + eps;
            param.data[i] -= step_size * state.exp_avg[i] / denom;
        }
    }
}

/// Convenience constructor for AdamW (Adam with decoupled weight decay).
pub fn adamw(lr: f32) -> Adam {
    Adam::new(lr).adamw(true)
}

impl Optimizer for Adam {
    fn step(&mut self, params: &mut [Param]) {
        let grouped: std::collections::HashSet<usize> = self
            .param_groups
            .iter()
            .flat_map(|g| g.params.iter().copied())
            .collect();

        let lr = self.lr;
        let beta1 = self.beta1;
        let beta2 = self.beta2;
        let eps = self.eps;
        let weight_decay = self.weight_decay;
        let adamw = self.adamw;

        for i in 0..params.len() {
            if !grouped.contains(&i) {
                self.step_param(i, &mut params[i], lr, beta1, beta2, eps, weight_decay, adamw);
            }
        }

        for g_idx in 0..self.param_groups.len() {
            let group = &self.param_groups[g_idx];
            let g_lr = *group.hyper.get("lr").unwrap_or(&lr);
            let g_b1 = *group.hyper.get("beta1").unwrap_or(&beta1);
            let g_b2 = *group.hyper.get("beta2").unwrap_or(&beta2);
            let g_eps = *group.hyper.get("eps").unwrap_or(&eps);
            let g_wd = *group.hyper.get("weight_decay").unwrap_or(&weight_decay);
            let g_aw = group.hyper.get("adamw").map_or(adamw, |&v| v != 0.0);
            let indices: Vec<usize> = group.params.clone();
            for &i in &indices {
                if i < params.len() {
                    self.step_param(i, &mut params[i], g_lr, g_b1, g_b2, g_eps, g_wd, g_aw);
                }
            }
        }
    }

    fn add_param_group(&mut self, group: ParamGroup) {
        self.param_groups.push(group);
    }

    fn state_dict(&self) -> StateDict {
        let mut state = StateDict::new();
        for (&idx, s) in &self.states {
            state.insert(format!("adam.{}.step", idx), vec![s.step as f32]);
            state.insert(format!("adam.{}.exp_avg", idx), s.exp_avg.clone());
            state.insert(format!("adam.{}.exp_avg_sq", idx), s.exp_avg_sq.clone());
        }
        state
    }

    fn load_state_dict(&mut self, state: &StateDict) {
        self.states.clear();
        // Collect all param indices from state keys
        let mut indices: std::collections::HashSet<usize> = std::collections::HashSet::new();
        for key in state.keys() {
            if let Some(rest) = key.strip_prefix("adam.") {
                if let Some(dot_pos) = rest.find('.') {
                    if let Ok(idx) = rest[..dot_pos].parse::<usize>() {
                        indices.insert(idx);
                    }
                }
            }
        }
        for idx in indices {
            let step = state
                .get(&format!("adam.{}.step", idx))
                .map_or(0, |v| v[0] as usize);
            let exp_avg = state
                .get(&format!("adam.{}.exp_avg", idx))
                .cloned()
                .unwrap_or_default();
            let exp_avg_sq = state
                .get(&format!("adam.{}.exp_avg_sq", idx))
                .cloned()
                .unwrap_or_default();
            self.states.insert(
                idx,
                AdamState {
                    step,
                    exp_avg,
                    exp_avg_sq,
                },
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// PyTorch reference (10-step Adam, lr=0.001, default betas):
    ///
    /// ```python
    /// import torch
    /// p = torch.nn.Parameter(torch.tensor([1.0, 2.0, 3.0]))
    /// opt = torch.optim.Adam([p], lr=0.001)
    /// for step in range(10):
    ///     p.grad = torch.tensor([0.5, -0.3, 0.8])
    ///     opt.step()
    ///     print(f"step {step}: {p.data.tolist()}")
    /// ```
    ///
    /// Expected (from PyTorch 2.x):
    /// step 0: [0.9990000128746033, 2.0009999275207520, 2.9990000724792480]
    /// step 1: [0.9980000257492065, 2.0019998550415039, 2.9980001449584961]
    /// step 2: [0.9970000386238098, 2.0029997825622559, 2.9970002174377441]
    /// step 3: [0.9960000514984131, 2.0039997100830078, 2.9960002899169922]
    /// step 4: [0.9950000643730164, 2.0049996376037598, 2.9950003623962402]
    /// step 5: [0.9940000772476196, 2.0059995651245117, 2.9940004348754883]
    /// step 6: [0.9930000305175781, 2.0069994926452637, 2.9930005073547363]
    /// step 7: [0.9920001029968262, 2.0079994201660156, 2.9920005798339844]
    /// step 8: [0.9910001158714294, 2.0089993476867676, 2.9910006523132324]
    /// step 9: [0.9900001287460327, 2.0099992752075195, 2.9900007247924805]
    #[test]
    fn test_adam_10step_pytorch_reference() {
        let mut params = vec![Param::with_grad(
            vec![1.0, 2.0, 3.0],
            vec![0.5, -0.3, 0.8],
        )];

        let mut opt = Adam::new(0.001);

        let expected = [
            vec![0.9990000128746033, 2.0009999275207520, 2.9990000724792480],
            vec![0.9980000257492065, 2.0019998550415039, 2.9980001449584961],
            vec![0.9970000386238098, 2.0029997825622559, 2.9970002174377441],
            vec![0.9960000514984131, 2.0039997100830078, 2.9960002899169922],
            vec![0.9950000643730164, 2.0049996376037598, 2.9950003623962402],
            vec![0.9940000772476196, 2.0059995651245117, 2.9940004348754883],
            vec![0.9930000305175781, 2.0069994926452637, 2.9930005073547363],
            vec![0.9920001029968262, 2.0079994201660156, 2.9920005798339844],
            vec![0.9910001158714294, 2.0089993476867676, 2.9910006523132324],
            vec![0.9900001287460327, 2.0099992752075195, 2.9900007247924805],
        ];

        for step in 0..10 {
            params[0].grad = Some(vec![0.5, -0.3, 0.8]);
            opt.step(&mut params);
            for j in 0..3 {
                assert!(
                    (params[0].data[j] - expected[step][j]).abs() < 1e-6,
                    "step {} elem {}: got {} expected {}",
                    step,
                    j,
                    params[0].data[j],
                    expected[step][j]
                );
            }
        }
    }

    #[test]
    fn test_adam_state_maintained() {
        let mut params = vec![Param::with_grad(vec![1.0, 2.0], vec![0.5, -0.3])];
        let mut opt = Adam::new(0.001);

        for _ in 0..5 {
            params[0].grad = Some(vec![0.5, -0.3]);
            opt.step(&mut params);
        }

        // Verify internal state exists and has correct step count
        let state = &opt.states[&0];
        assert_eq!(state.step, 5);
        assert_eq!(state.exp_avg.len(), 2);
        assert_eq!(state.exp_avg_sq.len(), 2);
    }

    #[test]
    fn test_adam_bias_correction_first_10_steps() {
        let mut params = vec![Param::with_grad(vec![0.0], vec![1.0])];
        let mut opt = Adam::new(0.001);

        let beta1: f32 = 0.9;
        let beta2: f32 = 0.999;

        for t in 1..=10 {
            params[0].grad = Some(vec![1.0]);
            opt.step(&mut params);

            let state = &opt.states[&0];
            assert_eq!(state.step, t);

            // Verify m_t = beta1^t * 0 + (1-beta1) * sum(beta1^(t-i) * 1.0 for i in 1..=t)
            // For constant gradient=1: m_t = 1 - beta1^t
            let expected_m = 1.0 - beta1.powi(t as i32);
            assert!(
                (state.exp_avg[0] - expected_m).abs() < 1e-6,
                "step {}: m got {} expected {}",
                t,
                state.exp_avg[0],
                expected_m
            );

            // v_t = 1 - beta2^t (for constant gradient=1)
            let expected_v = 1.0 - beta2.powi(t as i32);
            assert!(
                (state.exp_avg_sq[0] - expected_v).abs() < 1e-6,
                "step {}: v got {} expected {}",
                t,
                state.exp_avg_sq[0],
                expected_v
            );
        }
    }

    #[test]
    fn test_adamw_decoupled_weight_decay() {
        // AdamW applies weight decay to params directly, not to gradients
        let mut params_adam = vec![Param::with_grad(vec![1.0], vec![0.5])];
        let mut params_adamw = vec![Param::with_grad(vec![1.0], vec![0.5])];

        let mut opt_adam = Adam::new(0.01).weight_decay(0.1);
        let mut opt_adamw = Adam::new(0.01).weight_decay(0.1).adamw(true);

        opt_adam.step(&mut params_adam);
        opt_adamw.step(&mut params_adamw);

        // They should differ because AdamW decouples weight decay
        assert!(
            (params_adam[0].data[0] - params_adamw[0].data[0]).abs() > 1e-7,
            "Adam and AdamW should produce different results with weight_decay > 0"
        );
    }

    #[test]
    fn test_adam_state_dict_roundtrip() {
        let mut params = vec![Param::with_grad(vec![1.0, 2.0], vec![0.5, -0.3])];
        let mut opt = Adam::new(0.001);

        for _ in 0..3 {
            params[0].grad = Some(vec![0.5, -0.3]);
            opt.step(&mut params);
        }

        let state = opt.state_dict();
        let mut opt2 = Adam::new(0.001);
        opt2.load_state_dict(&state);

        // Continue from same state
        let mut params2 = vec![Param::with_grad(params[0].data.clone(), vec![0.5, -0.3])];
        let mut params1 = vec![Param::with_grad(params[0].data.clone(), vec![0.5, -0.3])];

        opt.step(&mut params1);
        opt2.step(&mut params2);

        for j in 0..2 {
            assert!(
                (params1[0].data[j] - params2[0].data[j]).abs() < 1e-7,
                "elem {}: {} != {}",
                j,
                params1[0].data[j],
                params2[0].data[j]
            );
        }
    }
}
