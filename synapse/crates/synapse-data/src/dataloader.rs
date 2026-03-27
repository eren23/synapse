use std::sync::mpsc::{sync_channel, Receiver};
use std::sync::Arc;
use std::thread::{self, JoinHandle};

use crate::collate::default_collate;
use crate::dataset::Dataset;
use crate::sampler::{RandomSampler, Sampler, SequentialSampler};
use crate::Tensor;

/// A configurable data loader that batches dataset samples with optional shuffling,
/// drop-last semantics, and multi-threaded prefetching with double-buffering.
pub struct DataLoader<D: Dataset> {
    dataset: Arc<D>,
    batch_size: usize,
    shuffle: bool,
    drop_last: bool,
    num_workers: usize,
    seed: u64,
    epoch: u64,
}

impl<D: Dataset + 'static> DataLoader<D> {
    pub fn new(dataset: D, batch_size: usize) -> Self {
        assert!(batch_size > 0, "batch_size must be > 0");
        Self {
            dataset: Arc::new(dataset),
            batch_size,
            shuffle: false,
            drop_last: false,
            num_workers: 0,
            seed: 0,
            epoch: 0,
        }
    }

    pub fn shuffle(mut self, shuffle: bool) -> Self {
        self.shuffle = shuffle;
        self
    }

    pub fn drop_last(mut self, drop_last: bool) -> Self {
        self.drop_last = drop_last;
        self
    }

    pub fn num_workers(mut self, num_workers: usize) -> Self {
        self.num_workers = num_workers;
        self
    }

    pub fn seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }

    /// Expected number of batches per epoch.
    pub fn num_batches(&self) -> usize {
        let n = self.dataset.len();
        if self.drop_last {
            n / self.batch_size
        } else {
            (n + self.batch_size - 1) / self.batch_size
        }
    }

    /// Create an iterator over batches for one epoch.
    /// Each batch is a `Vec<Tensor>` where each tensor has a leading batch dimension.
    pub fn iter(&mut self) -> DataLoaderIter<D> {
        let indices = self.generate_indices();
        let batches = self.split_into_batches(indices);
        let total_batches = batches.len();

        self.epoch += 1;

        if self.num_workers == 0 {
            // Single-threaded: no prefetching
            DataLoaderIter::SingleThreaded {
                dataset: Arc::clone(&self.dataset),
                batches,
                pos: 0,
            }
        } else {
            // Multi-threaded prefetching with double-buffering (channel capacity = 2)
            let dataset = Arc::clone(&self.dataset);
            let num_workers = self.num_workers;

            let (tx, rx) = sync_channel(2); // double-buffer

            let handle = thread::spawn(move || {
                for batch_indices in batches {
                    let samples = load_samples_parallel(&dataset, &batch_indices, num_workers);
                    let batch = default_collate(&samples);
                    if tx.send(batch).is_err() {
                        break;
                    }
                }
            });

            DataLoaderIter::Prefetched {
                rx,
                _handle: handle,
                total_batches,
                yielded: 0,
            }
        }
    }

    fn generate_indices(&mut self) -> Vec<usize> {
        let n = self.dataset.len();
        if self.shuffle {
            // Different seed each epoch for different ordering
            let epoch_seed = self.seed.wrapping_add(self.epoch);
            let mut sampler = RandomSampler::new(n, epoch_seed);
            sampler.indices()
        } else {
            let mut sampler = SequentialSampler::new(n);
            sampler.indices()
        }
    }

    fn split_into_batches(&self, indices: Vec<usize>) -> Vec<Vec<usize>> {
        let mut batches: Vec<Vec<usize>> = indices
            .chunks(self.batch_size)
            .map(|chunk| chunk.to_vec())
            .collect();

        if self.drop_last {
            if let Some(last) = batches.last() {
                if last.len() < self.batch_size {
                    batches.pop();
                }
            }
        }

        batches
    }
}

/// Load samples in parallel using scoped threads.
fn load_samples_parallel<D: Dataset>(
    dataset: &Arc<D>,
    indices: &[usize],
    num_workers: usize,
) -> Vec<Vec<Tensor>> {
    if num_workers <= 1 || indices.len() <= 1 {
        return indices.iter().map(|&i| dataset.get(i)).collect();
    }

    let chunk_size = (indices.len() + num_workers - 1) / num_workers;
    let mut all_samples = Vec::with_capacity(indices.len());

    thread::scope(|s| {
        let handles: Vec<_> = indices
            .chunks(chunk_size.max(1))
            .map(|chunk| {
                let ds = &dataset;
                s.spawn(move || chunk.iter().map(|&idx| ds.get(idx)).collect::<Vec<_>>())
            })
            .collect();

        for h in handles {
            all_samples.extend(h.join().unwrap());
        }
    });

    all_samples
}

/// Iterator over batches produced by a DataLoader.
pub enum DataLoaderIter<D: Dataset> {
    SingleThreaded {
        dataset: Arc<D>,
        batches: Vec<Vec<usize>>,
        pos: usize,
    },
    Prefetched {
        rx: Receiver<Vec<Tensor>>,
        _handle: JoinHandle<()>,
        total_batches: usize,
        yielded: usize,
    },
}

impl<D: Dataset> Iterator for DataLoaderIter<D> {
    type Item = Vec<Tensor>;

    fn next(&mut self) -> Option<Self::Item> {
        match self {
            DataLoaderIter::SingleThreaded {
                dataset,
                batches,
                pos,
            } => {
                if *pos >= batches.len() {
                    return None;
                }
                let batch_indices = &batches[*pos];
                let samples: Vec<Vec<Tensor>> =
                    batch_indices.iter().map(|&i| dataset.get(i)).collect();
                *pos += 1;
                Some(default_collate(&samples))
            }
            DataLoaderIter::Prefetched {
                rx,
                total_batches,
                yielded,
                ..
            } => {
                if *yielded >= *total_batches {
                    return None;
                }
                match rx.recv() {
                    Ok(batch) => {
                        *yielded += 1;
                        Some(batch)
                    }
                    Err(_) => None,
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dataset::{InMemoryDataset, TensorDataset};

    fn make_simple_dataset(n: usize) -> InMemoryDataset {
        let samples: Vec<Vec<Tensor>> = (0..n)
            .map(|i| vec![Tensor::new(vec![i as f32], vec![1])])
            .collect();
        InMemoryDataset::new(samples)
    }

    #[test]
    fn test_batch_count_no_drop() {
        let ds = make_simple_dataset(10);
        let mut loader = DataLoader::new(ds, 3);
        let batches: Vec<_> = loader.iter().collect();
        // ceil(10/3) = 4 batches: [3, 3, 3, 1]
        assert_eq!(batches.len(), 4);
    }

    #[test]
    fn test_batch_count_drop_last() {
        let ds = make_simple_dataset(10);
        let mut loader = DataLoader::new(ds, 3).drop_last(true);
        let batches: Vec<_> = loader.iter().collect();
        // floor(10/3) = 3 batches: [3, 3, 3]
        assert_eq!(batches.len(), 3);
    }

    #[test]
    fn test_batch_count_exact_division() {
        let ds = make_simple_dataset(12);
        let mut loader = DataLoader::new(ds, 4);
        let batches: Vec<_> = loader.iter().collect();
        assert_eq!(batches.len(), 3);

        let ds = make_simple_dataset(12);
        let mut loader = DataLoader::new(ds, 4).drop_last(true);
        let batches: Vec<_> = loader.iter().collect();
        assert_eq!(batches.len(), 3);
    }

    #[test]
    fn test_batch_shapes() {
        let features = Tensor::new((0..20).map(|i| i as f32).collect(), vec![10, 2]);
        let labels = Tensor::new((0..10).map(|i| i as f32).collect(), vec![10]);
        let ds = TensorDataset::new(features, labels);

        let mut loader = DataLoader::new(ds, 4);
        let batches: Vec<_> = loader.iter().collect();
        assert_eq!(batches.len(), 3); // ceil(10/4)

        // First batch: [4, 2] features, [4, 1] labels
        assert_eq!(batches[0][0].shape(), &[4, 2]);
        assert_eq!(batches[0][1].shape(), &[4, 1]);

        // Last batch (only 2 samples): [2, 2] features, [2, 1] labels
        assert_eq!(batches[2][0].shape(), &[2, 2]);
        assert_eq!(batches[2][1].shape(), &[2, 1]);
    }

    #[test]
    fn test_shuffle_covers_all_elements() {
        let ds = make_simple_dataset(20);
        let mut loader = DataLoader::new(ds, 5).shuffle(true).seed(42);

        let batches: Vec<_> = loader.iter().collect();
        let mut all_values: Vec<f32> = batches.iter().flat_map(|b| b[0].data().to_vec()).collect();
        all_values.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let expected: Vec<f32> = (0..20).map(|i| i as f32).collect();
        assert_eq!(all_values, expected);
    }

    #[test]
    fn test_shuffle_different_order_each_epoch() {
        let ds = make_simple_dataset(20);
        let mut loader = DataLoader::new(ds, 20).shuffle(true).seed(42);

        let epoch1: Vec<_> = loader.iter().collect();
        let epoch2: Vec<_> = loader.iter().collect();

        // Both epochs have all elements
        let vals1 = epoch1[0][0].data().to_vec();
        let vals2 = epoch2[0][0].data().to_vec();

        let mut sorted1 = vals1.clone();
        let mut sorted2 = vals2.clone();
        sorted1.sort_by(|a, b| a.partial_cmp(b).unwrap());
        sorted2.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let expected: Vec<f32> = (0..20).map(|i| i as f32).collect();
        assert_eq!(sorted1, expected);
        assert_eq!(sorted2, expected);

        // Order should differ (extremely unlikely to match with 20 elements)
        assert_ne!(vals1, vals2);
    }

    #[test]
    fn test_no_shuffle_sequential_order() {
        let ds = make_simple_dataset(10);
        let mut loader = DataLoader::new(ds, 10);
        let batches: Vec<_> = loader.iter().collect();
        let values: Vec<f32> = batches[0][0].data().to_vec();
        let expected: Vec<f32> = (0..10).map(|i| i as f32).collect();
        assert_eq!(values, expected);
    }

    #[test]
    fn test_prefetched_dataloader() {
        let ds = make_simple_dataset(10);
        let mut loader = DataLoader::new(ds, 3).num_workers(2);
        let batches: Vec<_> = loader.iter().collect();
        assert_eq!(batches.len(), 4); // ceil(10/3) = 4

        // Verify all elements present
        let mut all_values: Vec<f32> = batches.iter().flat_map(|b| b[0].data().to_vec()).collect();
        all_values.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let expected: Vec<f32> = (0..10).map(|i| i as f32).collect();
        assert_eq!(all_values, expected);
    }

    #[test]
    fn test_prefetched_dataloader_with_shuffle() {
        let ds = make_simple_dataset(20);
        let mut loader = DataLoader::new(ds, 5).shuffle(true).seed(99).num_workers(4);
        let batches: Vec<_> = loader.iter().collect();
        assert_eq!(batches.len(), 4);

        let mut all_values: Vec<f32> = batches.iter().flat_map(|b| b[0].data().to_vec()).collect();
        all_values.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let expected: Vec<f32> = (0..20).map(|i| i as f32).collect();
        assert_eq!(all_values, expected);
    }

    #[test]
    fn test_prefetched_drop_last() {
        let ds = make_simple_dataset(10);
        let mut loader = DataLoader::new(ds, 3).drop_last(true).num_workers(2);
        let batches: Vec<_> = loader.iter().collect();
        assert_eq!(batches.len(), 3);
        // All batches should have exactly batch_size samples
        for b in &batches {
            assert_eq!(b[0].shape()[0], 3);
        }
    }

    #[test]
    fn test_single_element_batch() {
        let ds = make_simple_dataset(1);
        let mut loader = DataLoader::new(ds, 1);
        let batches: Vec<_> = loader.iter().collect();
        assert_eq!(batches.len(), 1);
        assert_eq!(batches[0][0].shape(), &[1, 1]);
    }

    #[test]
    fn test_batch_size_larger_than_dataset() {
        let ds = make_simple_dataset(3);
        let mut loader = DataLoader::new(ds, 10);
        let batches: Vec<_> = loader.iter().collect();
        assert_eq!(batches.len(), 1); // single batch with 3 elements
        assert_eq!(batches[0][0].shape()[0], 3);
    }

    #[test]
    fn test_batch_size_larger_drop_last() {
        let ds = make_simple_dataset(3);
        let mut loader = DataLoader::new(ds, 10).drop_last(true);
        let batches: Vec<_> = loader.iter().collect();
        assert_eq!(batches.len(), 0); // 3 < 10, so no full batch
    }

    /// Benchmark: DataLoader throughput with prefetch threads.
    /// This is a basic throughput measurement, not a rigorous benchmark.
    #[test]
    fn bench_dataloader_throughput() {
        use std::time::Instant;

        let n = 1000;
        let feat_dim = 128;
        let features = Tensor::new(vec![1.0f32; n * feat_dim], vec![n, feat_dim]);
        let labels = Tensor::new(vec![0.0f32; n], vec![n]);
        let ds = TensorDataset::new(features, labels);

        let batch_size = 32;
        let mut loader = DataLoader::new(ds, batch_size).num_workers(4);

        let start = Instant::now();
        let batch_count: usize = loader.iter().count();
        let elapsed = start.elapsed();

        let batches_per_sec = batch_count as f64 / elapsed.as_secs_f64();
        eprintln!(
            "Prefetched DataLoader: {} batches in {:.3}ms ({:.0} batches/sec)",
            batch_count,
            elapsed.as_secs_f64() * 1000.0,
            batches_per_sec
        );
        assert_eq!(batch_count, (n + batch_size - 1) / batch_size);
    }
}
