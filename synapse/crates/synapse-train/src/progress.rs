use std::time::{Duration, Instant};

/// Tracks progress through epochs and batches, providing ETA estimates.
pub struct ProgressTracker {
    total_epochs: usize,
    current_epoch: usize,
    total_batches: usize,
    current_batch: usize,
    start_time: Instant,
    epoch_start: Instant,
}

impl ProgressTracker {
    pub fn new(total_epochs: usize) -> Self {
        let now = Instant::now();
        ProgressTracker {
            total_epochs,
            current_epoch: 0,
            total_batches: 0,
            current_batch: 0,
            start_time: now,
            epoch_start: now,
        }
    }

    pub fn start_epoch(&mut self, epoch: usize, num_batches: usize) {
        self.current_epoch = epoch;
        self.total_batches = num_batches;
        self.current_batch = 0;
        self.epoch_start = Instant::now();
    }

    pub fn update_batch(&mut self, batch: usize) {
        self.current_batch = batch + 1;
    }

    pub fn epoch_elapsed(&self) -> Duration {
        self.epoch_start.elapsed()
    }

    pub fn total_elapsed(&self) -> Duration {
        self.start_time.elapsed()
    }

    /// Estimated time remaining for the current epoch.
    pub fn batch_eta(&self) -> Duration {
        if self.current_batch == 0 {
            return Duration::ZERO;
        }
        let elapsed = self.epoch_start.elapsed();
        let per_batch = elapsed / self.current_batch as u32;
        let remaining = self.total_batches.saturating_sub(self.current_batch);
        per_batch * remaining as u32
    }

    /// Estimated time remaining for all epochs.
    pub fn epoch_eta(&self) -> Duration {
        let completed = self.current_epoch + 1;
        if completed == 0 {
            return Duration::ZERO;
        }
        let elapsed = self.start_time.elapsed();
        let per_epoch = elapsed / completed as u32;
        let remaining = self.total_epochs.saturating_sub(completed);
        per_epoch * remaining as u32
    }

    pub fn format_batch_progress(&self) -> String {
        let pct = if self.total_batches == 0 {
            100.0
        } else {
            self.current_batch as f64 / self.total_batches as f64 * 100.0
        };
        let eta = self.batch_eta();
        format!(
            "Epoch {}/{} | Batch {}/{} ({:.0}%) | ETA: {:.1}s",
            self.current_epoch + 1,
            self.total_epochs,
            self.current_batch,
            self.total_batches,
            pct,
            eta.as_secs_f64()
        )
    }

    pub fn format_epoch_progress(&self, train_loss: f32, val_loss: Option<f32>) -> String {
        let elapsed = self.epoch_elapsed();
        let eta = self.epoch_eta();
        let val_str = match val_loss {
            Some(v) => format!(" | val_loss: {:.4}", v),
            None => String::new(),
        };
        format!(
            "Epoch {}/{} | loss: {:.4}{} | {:.1}s | ETA: {:.1}s",
            self.current_epoch + 1,
            self.total_epochs,
            train_loss,
            val_str,
            elapsed.as_secs_f64(),
            eta.as_secs_f64()
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn progress_format() {
        let mut p = ProgressTracker::new(10);
        p.start_epoch(0, 100);
        p.update_batch(49);
        let s = p.format_batch_progress();
        assert!(s.contains("Epoch 1/10"));
        assert!(s.contains("Batch 50/100"));
    }
}
