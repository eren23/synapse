use std::f32::consts::PI;

/// Learning rate scheduler that decays the LR by `gamma` every `step_size` epochs.
///
/// Matches PyTorch's `torch.optim.lr_scheduler.StepLR`.
pub struct StepLR {
    pub base_lr: f32,
    pub step_size: usize,
    pub gamma: f32,
    epoch: usize,
}

impl StepLR {
    pub fn new(base_lr: f32, step_size: usize, gamma: f32) -> Self {
        StepLR {
            base_lr,
            step_size,
            gamma,
            epoch: 0,
        }
    }

    /// Advance one epoch and return the new learning rate.
    pub fn step(&mut self) -> f32 {
        self.epoch += 1;
        self.get_lr()
    }

    /// Current learning rate without advancing.
    pub fn get_lr(&self) -> f32 {
        let num_decays = self.epoch / self.step_size;
        self.base_lr * self.gamma.powi(num_decays as i32)
    }

    pub fn epoch(&self) -> usize {
        self.epoch
    }
}

/// Cosine annealing learning rate schedule.
///
/// Matches PyTorch's `torch.optim.lr_scheduler.CosineAnnealingLR`.
///
/// `lr = eta_min + (base_lr - eta_min) * (1 + cos(pi * epoch / T_max)) / 2`
pub struct CosineAnnealingLR {
    pub base_lr: f32,
    pub t_max: usize,
    pub eta_min: f32,
    epoch: usize,
}

impl CosineAnnealingLR {
    pub fn new(base_lr: f32, t_max: usize) -> Self {
        CosineAnnealingLR {
            base_lr,
            t_max,
            eta_min: 0.0,
            epoch: 0,
        }
    }

    pub fn eta_min(mut self, eta_min: f32) -> Self {
        self.eta_min = eta_min;
        self
    }

    pub fn step(&mut self) -> f32 {
        self.epoch += 1;
        self.get_lr()
    }

    pub fn get_lr(&self) -> f32 {
        self.eta_min
            + (self.base_lr - self.eta_min)
                * (1.0 + (PI * self.epoch as f32 / self.t_max as f32).cos())
                / 2.0
    }

    pub fn epoch(&self) -> usize {
        self.epoch
    }
}

/// Linear warmup scheduler: linearly ramps LR from 0 to base_lr over `warmup_steps`.
///
/// After warmup completes, LR stays at `base_lr`.
pub struct LinearWarmup {
    pub base_lr: f32,
    pub warmup_steps: usize,
    step_count: usize,
}

impl LinearWarmup {
    pub fn new(base_lr: f32, warmup_steps: usize) -> Self {
        assert!(warmup_steps > 0, "warmup_steps must be > 0");
        LinearWarmup {
            base_lr,
            warmup_steps,
            step_count: 0,
        }
    }

    pub fn step(&mut self) -> f32 {
        self.step_count += 1;
        self.get_lr()
    }

    pub fn get_lr(&self) -> f32 {
        let ratio = (self.step_count as f32 / self.warmup_steps as f32).min(1.0);
        self.base_lr * ratio
    }

    pub fn step_count(&self) -> usize {
        self.step_count
    }
}

/// Reduce learning rate when a metric stops improving.
///
/// Matches PyTorch's `torch.optim.lr_scheduler.ReduceLROnPlateau` (mode="min").
pub struct ReduceLROnPlateau {
    pub lr: f32,
    pub factor: f32,
    pub patience: usize,
    pub min_lr: f32,
    pub threshold: f32,
    best: f32,
    num_bad_epochs: usize,
    mode_min: bool,
}

impl ReduceLROnPlateau {
    pub fn new(lr: f32) -> Self {
        ReduceLROnPlateau {
            lr,
            factor: 0.1,
            patience: 10,
            min_lr: 0.0,
            threshold: 1e-4,
            best: f32::INFINITY,
            num_bad_epochs: 0,
            mode_min: true,
        }
    }

    pub fn factor(mut self, factor: f32) -> Self {
        assert!(factor < 1.0, "factor must be < 1.0");
        self.factor = factor;
        self
    }

    pub fn patience(mut self, patience: usize) -> Self {
        self.patience = patience;
        self
    }

    pub fn min_lr(mut self, min_lr: f32) -> Self {
        self.min_lr = min_lr;
        self
    }

    pub fn threshold(mut self, threshold: f32) -> Self {
        self.threshold = threshold;
        self
    }

    /// Set mode to "max" (default is "min").
    pub fn mode_max(mut self) -> Self {
        self.mode_min = false;
        self.best = f32::NEG_INFINITY;
        self
    }

    /// Report a metric value. Returns the (possibly reduced) learning rate.
    pub fn step(&mut self, metric: f32) -> f32 {
        let improved = if self.mode_min {
            metric < self.best * (1.0 - self.threshold)
        } else {
            metric > self.best * (1.0 + self.threshold)
        };

        if improved {
            self.best = metric;
            self.num_bad_epochs = 0;
        } else {
            self.num_bad_epochs += 1;
        }

        if self.num_bad_epochs > self.patience {
            self.lr = (self.lr * self.factor).max(self.min_lr);
            self.num_bad_epochs = 0;
        }

        self.lr
    }

    pub fn get_lr(&self) -> f32 {
        self.lr
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── StepLR ──────────────────────────────────────────────────────────

    #[test]
    fn test_step_lr_values() {
        let mut sched = StepLR::new(0.1, 3, 0.5);
        // epoch 0: lr = 0.1
        assert!((sched.get_lr() - 0.1).abs() < 1e-7);

        // epochs 1,2,3: still in first interval
        assert!((sched.step() - 0.1).abs() < 1e-7); // epoch 1
        assert!((sched.step() - 0.1).abs() < 1e-7); // epoch 2
        assert!((sched.step() - 0.05).abs() < 1e-7); // epoch 3: decay

        assert!((sched.step() - 0.05).abs() < 1e-7); // epoch 4
        assert!((sched.step() - 0.05).abs() < 1e-7); // epoch 5
        assert!((sched.step() - 0.025).abs() < 1e-7); // epoch 6: decay again
    }

    #[test]
    fn test_step_lr_no_decay_before_step_size() {
        let mut sched = StepLR::new(0.01, 10, 0.1);
        for _ in 0..9 {
            sched.step();
        }
        assert!((sched.get_lr() - 0.01).abs() < 1e-8);
    }

    // ── CosineAnnealingLR ───────────────────────────────────────────────

    #[test]
    fn test_cosine_annealing_values() {
        let mut sched = CosineAnnealingLR::new(0.1, 10);

        // At epoch 0: cos(0) = 1, lr = 0.1*(1+1)/2 = 0.1
        assert!((sched.get_lr() - 0.1).abs() < 1e-7);

        // At epoch 5 (half cycle): cos(pi/2) = 0, lr = 0.1*(1+0)/2 = 0.05
        for _ in 0..5 {
            sched.step();
        }
        assert!(
            (sched.get_lr() - 0.05).abs() < 1e-6,
            "got {}",
            sched.get_lr()
        );

        // At epoch 10 (full cycle): cos(pi) = -1, lr = 0.1*(1-1)/2 = 0
        for _ in 0..5 {
            sched.step();
        }
        assert!(
            sched.get_lr().abs() < 1e-6,
            "got {}",
            sched.get_lr()
        );
    }

    #[test]
    fn test_cosine_annealing_with_eta_min() {
        let mut sched = CosineAnnealingLR::new(0.1, 10).eta_min(0.01);

        // epoch 0: lr = 0.01 + (0.1-0.01)*(1+cos(0))/2 = 0.01 + 0.09 = 0.1
        assert!((sched.get_lr() - 0.1).abs() < 1e-7);

        // epoch 10: lr = 0.01 + 0.09*(1+cos(pi))/2 = 0.01
        for _ in 0..10 {
            sched.step();
        }
        assert!(
            (sched.get_lr() - 0.01).abs() < 1e-6,
            "got {}",
            sched.get_lr()
        );
    }

    // ── LinearWarmup ────────────────────────────────────────────────────

    #[test]
    fn test_linear_warmup_values() {
        let mut sched = LinearWarmup::new(0.1, 5);

        // step 0: lr = 0
        assert!(sched.get_lr().abs() < 1e-7);

        // Linearly increasing
        assert!((sched.step() - 0.02).abs() < 1e-7); // step 1: 0.1 * 1/5
        assert!((sched.step() - 0.04).abs() < 1e-7); // step 2: 0.1 * 2/5
        assert!((sched.step() - 0.06).abs() < 1e-7); // step 3: 0.1 * 3/5
        assert!((sched.step() - 0.08).abs() < 1e-7); // step 4: 0.1 * 4/5
        assert!((sched.step() - 0.1).abs() < 1e-7);  // step 5: 0.1 * 5/5

        // After warmup, stays at base_lr
        assert!((sched.step() - 0.1).abs() < 1e-7);
        assert!((sched.step() - 0.1).abs() < 1e-7);
    }

    // ── ReduceLROnPlateau ───────────────────────────────────────────────

    #[test]
    fn test_reduce_lr_on_plateau() {
        let mut sched = ReduceLROnPlateau::new(0.1)
            .factor(0.5)
            .patience(2);

        // Improving metric
        sched.step(1.0);
        assert!((sched.get_lr() - 0.1).abs() < 1e-7);

        sched.step(0.9);
        assert!((sched.get_lr() - 0.1).abs() < 1e-7);

        // Stagnating: 3 bad epochs (patience=2, reduce after >patience)
        sched.step(0.9);
        assert!((sched.get_lr() - 0.1).abs() < 1e-7);

        sched.step(0.9);
        assert!((sched.get_lr() - 0.1).abs() < 1e-7);

        sched.step(0.9); // bad_epochs=3 > patience=2 => reduce
        assert!(
            (sched.get_lr() - 0.05).abs() < 1e-7,
            "got {}",
            sched.get_lr()
        );
    }

    #[test]
    fn test_reduce_lr_on_plateau_min_lr() {
        let mut sched = ReduceLROnPlateau::new(0.01)
            .factor(0.1)
            .patience(0)
            .min_lr(0.001);

        // First call sets best, second triggers reduction
        sched.step(1.0);
        sched.step(1.0); // bad_epoch=1 > patience=0
        assert!(
            (sched.get_lr() - 0.001).abs() < 1e-7,
            "got {}",
            sched.get_lr()
        );

        // Should not go below min_lr
        sched.step(1.0);
        sched.step(1.0);
        assert!(sched.get_lr() >= 0.001);
    }

    #[test]
    fn test_reduce_lr_on_plateau_mode_max() {
        let mut sched = ReduceLROnPlateau::new(0.1)
            .factor(0.5)
            .patience(1)
            .mode_max();

        sched.step(0.5); // new best
        sched.step(0.8); // new best
        sched.step(0.7); // bad
        sched.step(0.7); // bad (2 > patience=1)
        assert!(
            (sched.get_lr() - 0.05).abs() < 1e-7,
            "got {}",
            sched.get_lr()
        );
    }
}
