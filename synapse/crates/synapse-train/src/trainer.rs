use std::time::Instant;

use synapse_autograd::Tensor;

use crate::callback::{CallbackAction, TrainerCallback};
use crate::metrics::RunningMean;
use crate::progress::ProgressTracker;

/// Configuration for the training loop.
pub struct TrainerConfig {
    pub epochs: usize,
}

/// Result of a single training epoch.
pub struct EpochResult {
    pub epoch: usize,
    pub train_loss: f32,
    pub val_loss: Option<f32>,
    pub duration_secs: f32,
}

/// Accumulated training history.
pub struct TrainHistory {
    pub epochs: Vec<EpochResult>,
}

impl TrainHistory {
    pub fn new() -> Self {
        TrainHistory { epochs: vec![] }
    }

    pub fn best_val_loss(&self) -> Option<f32> {
        self.epochs
            .iter()
            .filter_map(|e| e.val_loss)
            .min_by(|a, b| a.partial_cmp(b).unwrap())
    }

    pub fn last_train_loss(&self) -> Option<f32> {
        self.epochs.last().map(|e| e.train_loss)
    }
}

impl Default for TrainHistory {
    fn default() -> Self {
        Self::new()
    }
}

/// Trait that models implement to plug into the Trainer loop.
///
/// A single `&mut self` avoids borrow conflicts between train/validate closures.
pub trait TrainLoop {
    /// Return training batches for one epoch as (input, target) pairs.
    fn train_batches(&self) -> Vec<(Tensor, Tensor)>;

    /// Execute one training step on a batch. Returns the scalar loss.
    fn train_step(&mut self, input: &Tensor, target: &Tensor) -> f32;

    /// Optionally run validation at end of epoch. Returns validation loss.
    fn validate(&mut self) -> Option<f32> {
        None
    }
}

/// Orchestrates the epoch/batch training loop with callbacks.
pub struct Trainer {
    pub config: TrainerConfig,
    callbacks: Vec<Box<dyn TrainerCallback>>,
}

impl Trainer {
    pub fn new(config: TrainerConfig) -> Self {
        Trainer {
            config,
            callbacks: vec![],
        }
    }

    pub fn add_callback(&mut self, cb: Box<dyn TrainerCallback>) {
        self.callbacks.push(cb);
    }

    /// Run the full training loop.
    pub fn fit(&mut self, model: &mut dyn TrainLoop) -> TrainHistory {
        let mut history = TrainHistory::new();
        let mut progress = ProgressTracker::new(self.config.epochs);

        for epoch in 0..self.config.epochs {
            let epoch_start = Instant::now();

            for cb in &mut self.callbacks {
                cb.on_epoch_start(epoch);
            }

            let batches = model.train_batches();
            let n_batches = batches.len();
            progress.start_epoch(epoch, n_batches);

            let mut loss_tracker = RunningMean::new();

            for (i, (input, target)) in batches.iter().enumerate() {
                let loss = model.train_step(input, target);
                loss_tracker.update(loss);
                progress.update_batch(i);

                for cb in &mut self.callbacks {
                    cb.on_batch_end(epoch, i, loss);
                }
            }

            let train_loss = loss_tracker.mean();
            let val_loss = model.validate();
            let duration_secs = epoch_start.elapsed().as_secs_f32();

            let result = EpochResult {
                epoch,
                train_loss,
                val_loss,
                duration_secs,
            };

            let mut should_stop = false;
            for cb in &mut self.callbacks {
                if let CallbackAction::Stop = cb.on_epoch_end(&result) {
                    should_stop = true;
                }
            }

            history.epochs.push(result);

            if should_stop {
                break;
            }
        }

        for cb in &mut self.callbacks {
            cb.on_train_end(&history);
        }

        history
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct DummyModel {
        step_count: usize,
    }

    impl TrainLoop for DummyModel {
        fn train_batches(&self) -> Vec<(Tensor, Tensor)> {
            vec![
                (
                    Tensor::new(vec![1.0, 2.0], vec![1, 2]),
                    Tensor::new(vec![1.0], vec![1, 1]),
                );
                4
            ]
        }

        fn train_step(&mut self, _input: &Tensor, _target: &Tensor) -> f32 {
            self.step_count += 1;
            1.0 / self.step_count as f32
        }
    }

    #[test]
    fn trainer_runs_one_epoch() {
        let mut trainer = Trainer::new(TrainerConfig { epochs: 1 });
        let mut model = DummyModel { step_count: 0 };
        let history = trainer.fit(&mut model);
        assert_eq!(history.epochs.len(), 1);
        assert!(history.epochs[0].train_loss > 0.0);
        assert_eq!(model.step_count, 4);
    }

    #[test]
    fn trainer_runs_multiple_epochs() {
        let mut trainer = Trainer::new(TrainerConfig { epochs: 5 });
        let mut model = DummyModel { step_count: 0 };
        let history = trainer.fit(&mut model);
        assert_eq!(history.epochs.len(), 5);
        assert_eq!(model.step_count, 20);
    }
}
