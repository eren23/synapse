use std::collections::HashMap;

use crate::optimizer::{Optimizer, Param, ParamGroup, StateDict};

/// Stochastic Gradient Descent with optional momentum, dampening, weight decay,
/// and Nesterov acceleration.
///
/// Matches PyTorch's `torch.optim.SGD` semantics exactly.
pub struct SGD {
    pub lr: f32,
    pub momentum: f32,
    pub dampening: f32,
    pub weight_decay: f32,
    pub nesterov: bool,
    /// Per-parameter momentum buffers, keyed by param index.
    buffers: HashMap<usize, Vec<f32>>,
    /// Whether each parameter has been seen (for first-step dampening bypass).
    first_step: HashMap<usize, bool>,
    /// Additional parameter groups with per-group hyperparameters.
    param_groups: Vec<ParamGroup>,
}

impl SGD {
    pub fn new(lr: f32) -> Self {
        SGD {
            lr,
            momentum: 0.0,
            dampening: 0.0,
            weight_decay: 0.0,
            nesterov: false,
            buffers: HashMap::new(),
            first_step: HashMap::new(),
            param_groups: Vec::new(),
        }
    }

    pub fn momentum(mut self, momentum: f32) -> Self {
        self.momentum = momentum;
        self
    }

    pub fn dampening(mut self, dampening: f32) -> Self {
        self.dampening = dampening;
        self
    }

    pub fn weight_decay(mut self, weight_decay: f32) -> Self {
        self.weight_decay = weight_decay;
        self
    }

    pub fn nesterov(mut self, nesterov: bool) -> Self {
        self.nesterov = nesterov;
        self
    }

    /// Step a single parameter with the given hyperparameters.
    fn step_param(&mut self, idx: usize, param: &mut Param, lr: f32, momentum: f32, dampening: f32, weight_decay: f32, nesterov: bool) {
        let grad = match param.grad {
            Some(ref g) => g.clone(),
            None => return,
        };
        let n = param.data.len();

        // Apply L2 weight decay: d_p = grad + weight_decay * param
        let mut d_p: Vec<f32> = if weight_decay != 0.0 {
            grad.iter()
                .zip(param.data.iter())
                .map(|(&g, &p)| g + weight_decay * p)
                .collect()
        } else {
            grad
        };

        // Apply momentum
        if momentum != 0.0 {
            let is_first = *self.first_step.get(&idx).unwrap_or(&true);
            if is_first {
                // First step: buf = d_p (no dampening applied)
                self.buffers.insert(idx, d_p.clone());
                self.first_step.insert(idx, false);
            } else {
                // buf = momentum * buf + (1 - dampening) * d_p
                let buf = self.buffers.get_mut(&idx).unwrap();
                for i in 0..n {
                    buf[i] = momentum * buf[i] + (1.0 - dampening) * d_p[i];
                }
            }

            let buf = &self.buffers[&idx];
            if nesterov {
                // d_p = d_p + momentum * buf
                for i in 0..n {
                    d_p[i] += momentum * buf[i];
                }
            } else {
                d_p = buf.clone();
            }
        }

        // param = param - lr * d_p
        for i in 0..n {
            param.data[i] -= lr * d_p[i];
        }
    }
}

impl Optimizer for SGD {
    fn step(&mut self, params: &mut [Param]) {
        // Step default group (all params not in explicit groups)
        let grouped: std::collections::HashSet<usize> = self
            .param_groups
            .iter()
            .flat_map(|g| g.params.iter().copied())
            .collect();

        let lr = self.lr;
        let momentum = self.momentum;
        let dampening = self.dampening;
        let weight_decay = self.weight_decay;
        let nesterov = self.nesterov;

        for i in 0..params.len() {
            if !grouped.contains(&i) {
                self.step_param(i, &mut params[i], lr, momentum, dampening, weight_decay, nesterov);
            }
        }

        // Step explicit param groups
        for g_idx in 0..self.param_groups.len() {
            let group = &self.param_groups[g_idx];
            let g_lr = *group.hyper.get("lr").unwrap_or(&lr);
            let g_mom = *group.hyper.get("momentum").unwrap_or(&momentum);
            let g_damp = *group.hyper.get("dampening").unwrap_or(&dampening);
            let g_wd = *group.hyper.get("weight_decay").unwrap_or(&weight_decay);
            let g_nest = group.hyper.get("nesterov").map_or(nesterov, |&v| v != 0.0);
            let indices: Vec<usize> = group.params.clone();
            for &i in &indices {
                if i < params.len() {
                    self.step_param(i, &mut params[i], g_lr, g_mom, g_damp, g_wd, g_nest);
                }
            }
        }
    }

    fn add_param_group(&mut self, group: ParamGroup) {
        self.param_groups.push(group);
    }

    fn state_dict(&self) -> StateDict {
        let mut state = StateDict::new();
        for (&idx, buf) in &self.buffers {
            state.insert(format!("sgd.{}.momentum_buffer", idx), buf.clone());
        }
        for (&idx, &is_first) in &self.first_step {
            state.insert(
                format!("sgd.{}.first_step", idx),
                vec![if is_first { 1.0 } else { 0.0 }],
            );
        }
        state
    }

    fn load_state_dict(&mut self, state: &StateDict) {
        self.buffers.clear();
        self.first_step.clear();
        for (key, val) in state {
            if let Some(rest) = key.strip_prefix("sgd.") {
                if let Some(idx_str) = rest.strip_suffix(".momentum_buffer") {
                    if let Ok(idx) = idx_str.parse::<usize>() {
                        self.buffers.insert(idx, val.clone());
                    }
                } else if let Some(idx_str) = rest.strip_suffix(".first_step") {
                    if let Ok(idx) = idx_str.parse::<usize>() {
                        self.first_step.insert(idx, val[0] != 0.0);
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// PyTorch reference (5-step SGD, lr=0.1, momentum=0.9, weight_decay=0.01):
    ///
    /// ```python
    /// import torch
    /// p = torch.tensor([1.0, 2.0, 3.0], requires_grad=False)
    /// p = torch.nn.Parameter(p)
    /// opt = torch.optim.SGD([p], lr=0.1, momentum=0.9, weight_decay=0.01)
    /// for step in range(5):
    ///     p.grad = torch.tensor([0.5, -0.3, 0.8])
    ///     opt.step()
    ///     print(f"step {step}: {p.data.tolist()}")
    /// ```
    ///
    /// Expected outputs:
    /// step 0: [0.949, 2.028, 2.917]
    /// step 1: [0.852151, 2.081172, 2.759383]
    /// step 2: [0.714134749, 2.156945628, 2.534768317]
    /// step 3: [0.539205988351, 2.252984947572, 2.250080333983]
    /// step 4: [0.331230897779, 2.367167350239, 1.911611068934]
    #[test]
    fn test_sgd_5step_pytorch_reference() {
        let mut params = vec![Param::with_grad(
            vec![1.0, 2.0, 3.0],
            vec![0.5, -0.3, 0.8],
        )];

        let mut opt = SGD::new(0.1).momentum(0.9).weight_decay(0.01);

        let expected = [
            vec![0.949, 2.028, 2.917],
            vec![0.8521510, 2.081172, 2.759383],
            vec![0.714134749, 2.156945628, 2.534768317],
            vec![0.539205988351, 2.252984947572, 2.250080333983],
            vec![0.331230897779, 2.367167350239, 1.911611068934],
        ];

        for step in 0..5 {
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
    fn test_sgd_vanilla() {
        // No momentum, no weight decay: param -= lr * grad
        let mut params = vec![Param::with_grad(vec![1.0, 2.0], vec![0.5, -0.3])];
        let mut opt = SGD::new(0.1);
        opt.step(&mut params);
        assert!((params[0].data[0] - 0.95).abs() < 1e-7);
        assert!((params[0].data[1] - 2.03).abs() < 1e-7);
    }

    #[test]
    fn test_sgd_nesterov() {
        let mut params = vec![Param::with_grad(vec![1.0], vec![1.0])];
        let mut opt = SGD::new(0.1).momentum(0.9).nesterov(true);

        // Step 1: buf=1.0, d_p = 1.0 + 0.9*1.0 = 1.9, param = 1.0 - 0.1*1.9 = 0.81
        opt.step(&mut params);
        assert!((params[0].data[0] - 0.81).abs() < 1e-6, "got {}", params[0].data[0]);
    }

    #[test]
    fn test_sgd_zero_grad() {
        let mut params = vec![Param::with_grad(vec![1.0, 2.0], vec![0.5, -0.3])];
        let opt = SGD::new(0.1);
        opt.zero_grad(&mut params);
        assert_eq!(params[0].grad.as_ref().unwrap(), &vec![0.0, 0.0]);
    }

    #[test]
    fn test_sgd_state_dict_roundtrip() {
        let mut params = vec![Param::with_grad(vec![1.0], vec![1.0])];
        let mut opt = SGD::new(0.1).momentum(0.9);
        opt.step(&mut params);

        let state = opt.state_dict();
        let mut opt2 = SGD::new(0.1).momentum(0.9);
        opt2.load_state_dict(&state);

        // Both optimizers should produce the same result on the next step
        let mut params2 = vec![Param::with_grad(params[0].data.clone(), vec![1.0])];
        let mut params1 = vec![Param::with_grad(params[0].data.clone(), vec![1.0])];
        params[0].grad = Some(vec![1.0]);

        opt.step(&mut params1);
        opt2.step(&mut params2);

        assert!((params1[0].data[0] - params2[0].data[0]).abs() < 1e-7);
    }
}
