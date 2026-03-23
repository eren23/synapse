/// Tracks a running mean of scalar values.
pub struct RunningMean {
    sum: f64,
    count: usize,
}

impl RunningMean {
    pub fn new() -> Self {
        RunningMean { sum: 0.0, count: 0 }
    }

    pub fn update(&mut self, value: f32) {
        self.sum += value as f64;
        self.count += 1;
    }

    pub fn mean(&self) -> f32 {
        if self.count == 0 {
            0.0
        } else {
            (self.sum / self.count as f64) as f32
        }
    }

    pub fn count(&self) -> usize {
        self.count
    }

    pub fn reset(&mut self) {
        self.sum = 0.0;
        self.count = 0;
    }
}

impl Default for RunningMean {
    fn default() -> Self {
        Self::new()
    }
}

/// Tracks classification accuracy (top-1 and top-5).
pub struct Accuracy {
    correct: usize,
    total: usize,
}

impl Accuracy {
    pub fn new() -> Self {
        Accuracy {
            correct: 0,
            total: 0,
        }
    }

    /// Update with logits `[batch, num_classes]` and integer class targets `[batch]`.
    pub fn update_top1(&mut self, logits: &[f32], targets: &[usize], num_classes: usize) {
        let batch = targets.len();
        for i in 0..batch {
            let start = i * num_classes;
            let end = start + num_classes;
            let pred = logits[start..end]
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .map(|(idx, _)| idx)
                .unwrap();
            if pred == targets[i] {
                self.correct += 1;
            }
            self.total += 1;
        }
    }

    /// Update with top-5 accuracy.
    pub fn update_top5(&mut self, logits: &[f32], targets: &[usize], num_classes: usize) {
        let batch = targets.len();
        for i in 0..batch {
            let start = i * num_classes;
            let end = start + num_classes;
            let mut indexed: Vec<(usize, f32)> = logits[start..end]
                .iter()
                .enumerate()
                .map(|(idx, &val)| (idx, val))
                .collect();
            indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
            let top5: Vec<usize> = indexed.iter().take(5).map(|(idx, _)| *idx).collect();
            if top5.contains(&targets[i]) {
                self.correct += 1;
            }
            self.total += 1;
        }
    }

    pub fn accuracy(&self) -> f32 {
        if self.total == 0 {
            0.0
        } else {
            self.correct as f32 / self.total as f32
        }
    }

    pub fn correct(&self) -> usize {
        self.correct
    }

    pub fn total(&self) -> usize {
        self.total
    }

    pub fn reset(&mut self) {
        self.correct = 0;
        self.total = 0;
    }
}

impl Default for Accuracy {
    fn default() -> Self {
        Self::new()
    }
}

/// Confusion matrix for multi-class classification.
pub struct ConfusionMatrix {
    matrix: Vec<Vec<usize>>,
    num_classes: usize,
}

impl ConfusionMatrix {
    pub fn new(num_classes: usize) -> Self {
        ConfusionMatrix {
            matrix: vec![vec![0; num_classes]; num_classes],
            num_classes,
        }
    }

    /// Record a prediction: row = actual, column = predicted.
    pub fn update(&mut self, predicted: usize, actual: usize) {
        assert!(predicted < self.num_classes && actual < self.num_classes);
        self.matrix[actual][predicted] += 1;
    }

    pub fn matrix(&self) -> &Vec<Vec<usize>> {
        &self.matrix
    }

    pub fn num_classes(&self) -> usize {
        self.num_classes
    }

    /// Precision for a single class: TP / (TP + FP).
    pub fn precision(&self, class: usize) -> f32 {
        let tp = self.matrix[class][class] as f32;
        let col_sum: f32 = (0..self.num_classes)
            .map(|i| self.matrix[i][class] as f32)
            .sum();
        if col_sum == 0.0 {
            0.0
        } else {
            tp / col_sum
        }
    }

    /// Recall for a single class: TP / (TP + FN).
    pub fn recall(&self, class: usize) -> f32 {
        let tp = self.matrix[class][class] as f32;
        let row_sum: f32 = self.matrix[class].iter().sum::<usize>() as f32;
        if row_sum == 0.0 {
            0.0
        } else {
            tp / row_sum
        }
    }

    /// Overall accuracy: sum of diagonal / total.
    pub fn accuracy(&self) -> f32 {
        let correct: usize = (0..self.num_classes).map(|i| self.matrix[i][i]).sum();
        let total: usize = self.matrix.iter().flat_map(|row| row.iter()).sum();
        if total == 0 {
            0.0
        } else {
            correct as f32 / total as f32
        }
    }

    pub fn reset(&mut self) {
        for row in &mut self.matrix {
            for v in row {
                *v = 0;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn running_mean_tracks_correctly() {
        let mut m = RunningMean::new();
        m.update(1.0);
        m.update(3.0);
        m.update(5.0);
        assert!((m.mean() - 3.0).abs() < 1e-6);
        assert_eq!(m.count(), 3);
    }

    #[test]
    fn accuracy_top1() {
        let mut acc = Accuracy::new();
        // 3 samples, 4 classes
        // logits: sample 0 predicts class 2, sample 1 predicts class 0, sample 2 predicts class 1
        let logits = vec![
            0.1, 0.2, 0.9, 0.1, // pred=2
            0.8, 0.1, 0.05, 0.05, // pred=0
            0.1, 0.7, 0.1, 0.1, // pred=1
        ];
        let targets = vec![2, 0, 3]; // correct, correct, wrong
        acc.update_top1(&logits, &targets, 4);
        assert!((acc.accuracy() - 2.0 / 3.0).abs() < 1e-6);
    }

    #[test]
    fn accuracy_top5() {
        let mut acc = Accuracy::new();
        // 10 classes, target=7, prediction has 7 in top-5
        let mut logits = vec![0.0; 10];
        logits[0] = 0.9;
        logits[1] = 0.8;
        logits[2] = 0.7;
        logits[3] = 0.6;
        logits[7] = 0.5; // 5th highest
        acc.update_top5(&logits, &[7], 10);
        assert!((acc.accuracy() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn confusion_matrix_metrics() {
        let mut cm = ConfusionMatrix::new(3);
        // Class 0: 5 correct, 1 misclassified as 1
        for _ in 0..5 {
            cm.update(0, 0);
        }
        cm.update(1, 0);
        // Class 1: 3 correct
        for _ in 0..3 {
            cm.update(1, 1);
        }
        // Class 2: 2 correct, 1 misclassified as 0
        for _ in 0..2 {
            cm.update(2, 2);
        }
        cm.update(0, 2);

        assert!((cm.accuracy() - 10.0 / 12.0).abs() < 1e-6);
        // Class 0 precision: 5 / (5+1) = 5/6  (col 0: actual0->pred0=5, actual2->pred0=1)
        assert!((cm.precision(0) - 5.0 / 6.0).abs() < 1e-6);
        // Class 0 recall: 5 / (5+1) = 5/6  (row 0: pred0=5, pred1=1)
        assert!((cm.recall(0) - 5.0 / 6.0).abs() < 1e-6);
    }
}
