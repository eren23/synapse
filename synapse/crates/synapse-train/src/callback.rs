use crate::trainer::{EpochResult, TrainHistory};

/// Action returned by callbacks to control the training loop.
pub enum CallbackAction {
    Continue,
    Stop,
}

/// Trait for hooks into the training loop.
pub trait TrainerCallback {
    fn on_epoch_start(&mut self, _epoch: usize) {}
    fn on_epoch_end(&mut self, _result: &EpochResult) -> CallbackAction {
        CallbackAction::Continue
    }
    fn on_batch_end(&mut self, _epoch: usize, _batch: usize, _loss: f32) {}
    fn on_train_end(&mut self, _history: &TrainHistory) {}
}

/// Stops training when the monitored loss stops improving.
pub struct EarlyStopping {
    patience: usize,
    min_delta: f32,
    best_loss: f32,
    counter: usize,
    stopped_epoch: Option<usize>,
}

impl EarlyStopping {
    pub fn new(patience: usize, min_delta: f32) -> Self {
        EarlyStopping {
            patience,
            min_delta,
            best_loss: f32::INFINITY,
            counter: 0,
            stopped_epoch: None,
        }
    }

    pub fn stopped_epoch(&self) -> Option<usize> {
        self.stopped_epoch
    }

    pub fn best_loss(&self) -> f32 {
        self.best_loss
    }

    pub fn counter(&self) -> usize {
        self.counter
    }
}

impl TrainerCallback for EarlyStopping {
    fn on_epoch_end(&mut self, result: &EpochResult) -> CallbackAction {
        let loss = result.val_loss.unwrap_or(result.train_loss);
        if loss < self.best_loss - self.min_delta {
            self.best_loss = loss;
            self.counter = 0;
        } else {
            self.counter += 1;
        }
        if self.counter >= self.patience {
            self.stopped_epoch = Some(result.epoch);
            CallbackAction::Stop
        } else {
            CallbackAction::Continue
        }
    }
}

/// Tracks the best model checkpoint (by validation or training loss).
pub struct ModelCheckpoint {
    best_loss: f32,
    best_epoch: Option<usize>,
}

impl ModelCheckpoint {
    pub fn new() -> Self {
        ModelCheckpoint {
            best_loss: f32::INFINITY,
            best_epoch: None,
        }
    }

    pub fn best_loss(&self) -> f32 {
        self.best_loss
    }

    pub fn best_epoch(&self) -> Option<usize> {
        self.best_epoch
    }

    pub fn is_best(&self, loss: f32) -> bool {
        loss < self.best_loss
    }
}

impl Default for ModelCheckpoint {
    fn default() -> Self {
        Self::new()
    }
}

impl TrainerCallback for ModelCheckpoint {
    fn on_epoch_end(&mut self, result: &EpochResult) -> CallbackAction {
        let loss = result.val_loss.unwrap_or(result.train_loss);
        if loss < self.best_loss {
            self.best_loss = loss;
            self.best_epoch = Some(result.epoch);
        }
        CallbackAction::Continue
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn early_stopping_triggers_after_patience() {
        let mut es = EarlyStopping::new(3, 0.0);

        // Improving losses
        let actions: Vec<CallbackAction> = (0..3)
            .map(|i| {
                es.on_epoch_end(&EpochResult {
                    epoch: i,
                    train_loss: 1.0 - i as f32 * 0.1,
                    val_loss: None,
                    duration_secs: 0.0,
                })
            })
            .collect();
        assert!(actions.iter().all(|a| matches!(a, CallbackAction::Continue)));
        assert_eq!(es.counter(), 0);

        // Non-improving losses: patience=3, so need 3 non-improving epochs
        for i in 3..5 {
            let action = es.on_epoch_end(&EpochResult {
                epoch: i,
                train_loss: 1.0,
                val_loss: None,
                duration_secs: 0.0,
            });
            assert!(matches!(action, CallbackAction::Continue));
        }
        assert_eq!(es.counter(), 2);

        // Third non-improving epoch triggers stop
        let action = es.on_epoch_end(&EpochResult {
            epoch: 5,
            train_loss: 1.0,
            val_loss: None,
            duration_secs: 0.0,
        });
        assert!(matches!(action, CallbackAction::Stop));
        assert_eq!(es.stopped_epoch(), Some(5));
    }

    #[test]
    fn early_stopping_resets_on_improvement() {
        let mut es = EarlyStopping::new(2, 0.0);

        es.on_epoch_end(&EpochResult {
            epoch: 0,
            train_loss: 1.0,
            val_loss: None,
            duration_secs: 0.0,
        });
        es.on_epoch_end(&EpochResult {
            epoch: 1,
            train_loss: 1.1,
            val_loss: None,
            duration_secs: 0.0,
        });
        assert_eq!(es.counter(), 1);

        // Improvement resets counter
        es.on_epoch_end(&EpochResult {
            epoch: 2,
            train_loss: 0.5,
            val_loss: None,
            duration_secs: 0.0,
        });
        assert_eq!(es.counter(), 0);
    }

    #[test]
    fn model_checkpoint_tracks_best() {
        let mut mc = ModelCheckpoint::new();

        mc.on_epoch_end(&EpochResult {
            epoch: 0,
            train_loss: 1.0,
            val_loss: Some(0.9),
            duration_secs: 0.0,
        });
        assert_eq!(mc.best_epoch(), Some(0));
        assert!((mc.best_loss() - 0.9).abs() < 1e-6);

        mc.on_epoch_end(&EpochResult {
            epoch: 1,
            train_loss: 0.8,
            val_loss: Some(0.7),
            duration_secs: 0.0,
        });
        assert_eq!(mc.best_epoch(), Some(1));

        mc.on_epoch_end(&EpochResult {
            epoch: 2,
            train_loss: 0.5,
            val_loss: Some(0.8),
            duration_secs: 0.0,
        });
        assert_eq!(mc.best_epoch(), Some(1)); // unchanged
    }
}
