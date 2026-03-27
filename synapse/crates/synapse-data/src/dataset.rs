use crate::Tensor;

/// A dataset provides indexed access to samples.
/// Each sample is a `Vec<Tensor>` (e.g., [features, labels]).
pub trait Dataset: Send + Sync {
    fn len(&self) -> usize;
    fn get(&self, index: usize) -> Vec<Tensor>;
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// A dataset that stores all samples in memory.
pub struct InMemoryDataset {
    samples: Vec<Vec<Tensor>>,
}

impl InMemoryDataset {
    pub fn new(samples: Vec<Vec<Tensor>>) -> Self {
        Self { samples }
    }
}

impl Dataset for InMemoryDataset {
    fn len(&self) -> usize {
        self.samples.len()
    }

    fn get(&self, index: usize) -> Vec<Tensor> {
        self.samples[index].clone()
    }
}

/// A dataset wrapping a features tensor and a labels tensor.
/// Features shape: [N, ...], Labels shape: [N, ...].
/// Each sample returns [features_row, labels_row].
pub struct TensorDataset {
    features: Tensor,
    labels: Tensor,
}

impl TensorDataset {
    pub fn new(features: Tensor, labels: Tensor) -> Self {
        assert_eq!(
            features.shape()[0],
            labels.shape()[0],
            "features and labels must have the same number of samples: {} vs {}",
            features.shape()[0],
            labels.shape()[0]
        );
        Self { features, labels }
    }
}

impl Dataset for TensorDataset {
    fn len(&self) -> usize {
        self.features.shape()[0]
    }

    fn get(&self, index: usize) -> Vec<Tensor> {
        vec![self.features.select(index), self.labels.select(index)]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_in_memory_dataset() {
        let samples = vec![
            vec![Tensor::new(vec![1.0, 2.0], vec![2])],
            vec![Tensor::new(vec![3.0, 4.0], vec![2])],
            vec![Tensor::new(vec![5.0, 6.0], vec![2])],
        ];
        let ds = InMemoryDataset::new(samples);
        assert_eq!(ds.len(), 3);
        assert!(!ds.is_empty());
        assert_eq!(ds.get(0)[0].data(), &[1.0, 2.0]);
        assert_eq!(ds.get(2)[0].data(), &[5.0, 6.0]);
    }

    #[test]
    fn test_tensor_dataset() {
        let features = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![3, 2]);
        let labels = Tensor::new(vec![0.0, 1.0, 2.0], vec![3]);
        let ds = TensorDataset::new(features, labels);

        assert_eq!(ds.len(), 3);

        let sample = ds.get(1);
        assert_eq!(sample.len(), 2);
        assert_eq!(sample[0].data(), &[3.0, 4.0]); // features row 1
        assert_eq!(sample[1].data(), &[1.0]); // label row 1 (scalar wrapped)
    }

    #[test]
    #[should_panic(expected = "features and labels must have the same number of samples")]
    fn test_tensor_dataset_mismatched_sizes() {
        let features = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let labels = Tensor::new(vec![0.0, 1.0, 2.0], vec![3]);
        TensorDataset::new(features, labels);
    }
}
