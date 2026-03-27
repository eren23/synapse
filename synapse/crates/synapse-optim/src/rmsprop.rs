use std::collections::HashMap;

use crate::optimizer::{Optimizer, Param, ParamGroup, StateDict};

/// Per-parameter RMSProp state.
#[derive(Clone, Debug)]
struct RMSPropState {
    square_avg: Vec<f32>,
    grad_avg: Option<Vec<f32>>,
    momentum_buffer: Option<Vec<f32>>,
}

/// RMSProp optimizer with optional centering and momentum.
///
/// Matches PyTorch's `torch.optim.RMSprop` semantics exactly.
pub struct RMSProp {
    pub lr: f32,
    pub alpha: f32,
    pub eps: f32,
    pub weight_decay: f32,
    pub momentum: f32,
    pub centered: bool,
    states: HashMap<usize, RMSPropState>,
    param_groups: Vec<ParamGroup>,
}

impl RMSProp {
    pub fn new(lr: f32) -> Self {
        RMSProp {
            lr,
            alpha: 0.99,
            eps: 1e-8,
            weight_decay: 0.0,
            momentum: 0.0,
            centered: false,
            states: HashMap::new(),
            param_groups: Vec::new(),
        }
    }

    pub fn alpha(mut self, alpha: f32) -> Self {
        self.alpha = alpha;
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

    pub fn momentum(mut self, momentum: f32) -> Self {
        self.momentum = momentum;
        self
    }

    pub fn centered(mut self, centered: bool) -> Self {
        self.centered = centered;
        self
    }

    fn step_param(
        &mut self,
        idx: usize,
        param: &mut Param,
        lr: f32,
        alpha: f32,
        eps: f32,
        weight_decay: f32,
        momentum: f32,
        centered: bool,
    ) {
        let grad = match param.grad {
            Some(ref g) => g.clone(),
            None => return,
        };
        let n = param.data.len();

        // L2 weight decay
        let grad = if weight_decay != 0.0 {
            grad.iter()
                .zip(param.data.iter())
                .map(|(&g, &p)| g + weight_decay * p)
                .collect::<Vec<f32>>()
        } else {
            grad
        };

        // Initialise state
        let state = self.states.entry(idx).or_insert_with(|| RMSPropState {
            square_avg: vec![0.0; n],
            grad_avg: if centered { Some(vec![0.0; n]) } else { None },
            momentum_buffer: if momentum > 0.0 {
                Some(vec![0.0; n])
            } else {
                None
            },
        });

        // Ensure centered/momentum buffers exist if config changed
        if centered && state.grad_avg.is_none() {
            state.grad_avg = Some(vec![0.0; n]);
        }
        if momentum > 0.0 && state.momentum_buffer.is_none() {
            state.momentum_buffer = Some(vec![0.0; n]);
        }

        // square_avg = alpha * square_avg + (1 - alpha) * grad^2
        for i in 0..n {
            state.square_avg[i] = alpha * state.square_avg[i] + (1.0 - alpha) * grad[i] * grad[i];
        }

        // Compute denominator
        let avg: Vec<f32> = if centered {
            let grad_avg = state.grad_avg.as_mut().unwrap();
            // grad_avg = alpha * grad_avg + (1 - alpha) * grad
            for i in 0..n {
                grad_avg[i] = alpha * grad_avg[i] + (1.0 - alpha) * grad[i];
            }
            // avg = square_avg - grad_avg^2
            state
                .square_avg
                .iter()
                .zip(grad_avg.iter())
                .map(|(&sq, &ga)| (sq - ga * ga).sqrt() + eps)
                .collect()
        } else {
            state.square_avg.iter().map(|&sq| sq.sqrt() + eps).collect()
        };

        if momentum > 0.0 {
            let buf = state.momentum_buffer.as_mut().unwrap();
            // buf = momentum * buf + grad / avg
            for i in 0..n {
                buf[i] = momentum * buf[i] + grad[i] / avg[i];
            }
            for i in 0..n {
                param.data[i] -= lr * buf[i];
            }
        } else {
            for i in 0..n {
                param.data[i] -= lr * grad[i] / avg[i];
            }
        }
    }
}

impl Optimizer for RMSProp {
    fn step(&mut self, params: &mut [Param]) {
        let grouped: std::collections::HashSet<usize> = self
            .param_groups
            .iter()
            .flat_map(|g| g.params.iter().copied())
            .collect();

        let lr = self.lr;
        let alpha = self.alpha;
        let eps = self.eps;
        let weight_decay = self.weight_decay;
        let momentum = self.momentum;
        let centered = self.centered;

        for i in 0..params.len() {
            if !grouped.contains(&i) {
                self.step_param(
                    i,
                    &mut params[i],
                    lr,
                    alpha,
                    eps,
                    weight_decay,
                    momentum,
                    centered,
                );
            }
        }

        for g_idx in 0..self.param_groups.len() {
            let group = &self.param_groups[g_idx];
            let g_lr = *group.hyper.get("lr").unwrap_or(&lr);
            let g_alpha = *group.hyper.get("alpha").unwrap_or(&alpha);
            let g_eps = *group.hyper.get("eps").unwrap_or(&eps);
            let g_wd = *group.hyper.get("weight_decay").unwrap_or(&weight_decay);
            let g_mom = *group.hyper.get("momentum").unwrap_or(&momentum);
            let g_cen = group.hyper.get("centered").map_or(centered, |&v| v != 0.0);
            let indices: Vec<usize> = group.params.clone();
            for &i in &indices {
                if i < params.len() {
                    self.step_param(i, &mut params[i], g_lr, g_alpha, g_eps, g_wd, g_mom, g_cen);
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
            state.insert(format!("rmsprop.{}.square_avg", idx), s.square_avg.clone());
            if let Some(ref ga) = s.grad_avg {
                state.insert(format!("rmsprop.{}.grad_avg", idx), ga.clone());
            }
            if let Some(ref mb) = s.momentum_buffer {
                state.insert(format!("rmsprop.{}.momentum_buffer", idx), mb.clone());
            }
        }
        state
    }

    fn load_state_dict(&mut self, state: &StateDict) {
        self.states.clear();
        let mut indices: std::collections::HashSet<usize> = std::collections::HashSet::new();
        for key in state.keys() {
            if let Some(rest) = key.strip_prefix("rmsprop.") {
                if let Some(dot_pos) = rest.find('.') {
                    if let Ok(idx) = rest[..dot_pos].parse::<usize>() {
                        indices.insert(idx);
                    }
                }
            }
        }
        for idx in indices {
            let square_avg = state
                .get(&format!("rmsprop.{}.square_avg", idx))
                .cloned()
                .unwrap_or_default();
            let grad_avg = state.get(&format!("rmsprop.{}.grad_avg", idx)).cloned();
            let momentum_buffer = state
                .get(&format!("rmsprop.{}.momentum_buffer", idx))
                .cloned();
            self.states.insert(
                idx,
                RMSPropState {
                    square_avg,
                    grad_avg,
                    momentum_buffer,
                },
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rmsprop_basic() {
        // Single step, no momentum, no centering
        // square_avg = (1-alpha)*grad^2 = 0.01 * 0.25 = 0.0025
        // param -= lr * grad / (sqrt(square_avg) + eps)
        //       -= 0.01 * 0.5 / (sqrt(0.0025) + 1e-8)
        //       -= 0.01 * 0.5 / 0.05
        //       -= 0.1
        // param = 1.0 - 0.1 = 0.9
        let mut params = vec![Param::with_grad(vec![1.0], vec![0.5])];
        let mut opt = RMSProp::new(0.01).alpha(0.99);
        opt.step(&mut params);
        assert!(
            (params[0].data[0] - 0.9).abs() < 1e-6,
            "got {}",
            params[0].data[0]
        );
    }

    #[test]
    fn test_rmsprop_centered() {
        let mut params = vec![Param::with_grad(vec![1.0, 2.0], vec![0.5, -0.3])];
        let mut opt = RMSProp::new(0.01).centered(true);

        // Run a few steps to exercise centered variant
        for _ in 0..5 {
            params[0].grad = Some(vec![0.5, -0.3]);
            opt.step(&mut params);
        }

        // Just verify it runs and produces reasonable values
        assert!(params[0].data[0] < 1.0); // moved from 1.0 in negative grad direction
        assert!(params[0].data[1] > 2.0); // moved from 2.0 in positive direction
    }

    #[test]
    fn test_rmsprop_with_momentum() {
        let mut params = vec![Param::with_grad(vec![1.0], vec![1.0])];
        let mut opt = RMSProp::new(0.01).momentum(0.9);

        opt.step(&mut params);
        let after_step1 = params[0].data[0];

        params[0].grad = Some(vec![1.0]);
        opt.step(&mut params);
        let after_step2 = params[0].data[0];

        // With momentum, second step should move further than first
        let delta1 = 1.0 - after_step1;
        let delta2 = after_step1 - after_step2;
        assert!(
            delta2 > delta1,
            "momentum should accelerate: delta1={}, delta2={}",
            delta1,
            delta2
        );
    }

    #[test]
    fn test_rmsprop_state_dict_roundtrip() {
        let mut params = vec![Param::with_grad(vec![1.0], vec![0.5])];
        let mut opt = RMSProp::new(0.01).momentum(0.9).centered(true);
        opt.step(&mut params);

        let state = opt.state_dict();
        let mut opt2 = RMSProp::new(0.01).momentum(0.9).centered(true);
        opt2.load_state_dict(&state);

        let mut p1 = vec![Param::with_grad(params[0].data.clone(), vec![0.5])];
        let mut p2 = vec![Param::with_grad(params[0].data.clone(), vec![0.5])];

        opt.step(&mut p1);
        opt2.step(&mut p2);

        assert!((p1[0].data[0] - p2[0].data[0]).abs() < 1e-7);
    }
}
