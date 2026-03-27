use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::{Rng, SeedableRng};

/// A sampler produces a sequence of dataset indices.
pub trait Sampler: Send {
    /// Reset the sampler for a new epoch, returning an iterator over indices.
    fn indices(&mut self) -> Vec<usize>;
    fn len(&self) -> usize;
}

/// Yields indices 0, 1, 2, ..., N-1 in order.
pub struct SequentialSampler {
    size: usize,
}

impl SequentialSampler {
    pub fn new(size: usize) -> Self {
        Self { size }
    }
}

impl Sampler for SequentialSampler {
    fn indices(&mut self) -> Vec<usize> {
        (0..self.size).collect()
    }

    fn len(&self) -> usize {
        self.size
    }
}

/// Yields all N indices in a uniformly random permutation.
pub struct RandomSampler {
    size: usize,
    rng: StdRng,
}

impl RandomSampler {
    pub fn new(size: usize, seed: u64) -> Self {
        Self {
            size,
            rng: StdRng::seed_from_u64(seed),
        }
    }
}

impl Sampler for RandomSampler {
    fn indices(&mut self) -> Vec<usize> {
        let mut idx: Vec<usize> = (0..self.size).collect();
        idx.shuffle(&mut self.rng);
        idx
    }

    fn len(&self) -> usize {
        self.size
    }
}

/// Yields indices sampled with replacement according to given weights.
/// The number of samples produced equals the dataset size.
pub struct WeightedRandomSampler {
    weights: Vec<f64>,
    cumulative: Vec<f64>,
    num_samples: usize,
    rng: StdRng,
}

impl WeightedRandomSampler {
    pub fn new(weights: Vec<f64>, num_samples: usize, seed: u64) -> Self {
        assert!(!weights.is_empty(), "weights must not be empty");
        assert!(
            weights.iter().all(|&w| w >= 0.0),
            "weights must be non-negative"
        );
        let total: f64 = weights.iter().sum();
        assert!(total > 0.0, "total weight must be positive");

        // Build cumulative distribution
        let mut cumulative = Vec::with_capacity(weights.len());
        let mut acc = 0.0;
        for &w in &weights {
            acc += w / total;
            cumulative.push(acc);
        }
        // Ensure last element is exactly 1.0 to avoid floating-point edge cases
        if let Some(last) = cumulative.last_mut() {
            *last = 1.0;
        }

        Self {
            weights,
            cumulative,
            num_samples,
            rng: StdRng::seed_from_u64(seed),
        }
    }

    pub fn weights(&self) -> &[f64] {
        &self.weights
    }
}

impl Sampler for WeightedRandomSampler {
    fn indices(&mut self) -> Vec<usize> {
        let mut result = Vec::with_capacity(self.num_samples);
        for _ in 0..self.num_samples {
            let u: f64 = self.rng.gen();
            // Binary search for the bucket
            let idx = match self
                .cumulative
                .binary_search_by(|c| c.partial_cmp(&u).unwrap_or(std::cmp::Ordering::Equal))
            {
                Ok(i) => i,
                Err(i) => i,
            };
            result.push(idx.min(self.cumulative.len() - 1));
        }
        result
    }

    fn len(&self) -> usize {
        self.num_samples
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sequential_sampler() {
        let mut s = SequentialSampler::new(5);
        assert_eq!(s.len(), 5);
        assert_eq!(s.indices(), vec![0, 1, 2, 3, 4]);
        // Stable across calls
        assert_eq!(s.indices(), vec![0, 1, 2, 3, 4]);
    }

    #[test]
    fn test_random_sampler_covers_all() {
        let mut s = RandomSampler::new(10, 42);
        let idx = s.indices();
        assert_eq!(idx.len(), 10);
        let mut sorted = idx.clone();
        sorted.sort();
        assert_eq!(sorted, (0..10).collect::<Vec<_>>());
    }

    #[test]
    fn test_random_sampler_different_each_epoch() {
        let mut s = RandomSampler::new(20, 42);
        let epoch1 = s.indices();
        let epoch2 = s.indices();
        // Extremely unlikely to be the same with 20 elements
        assert_ne!(epoch1, epoch2);
        // But both cover all elements
        let mut s1 = epoch1.clone();
        let mut s2 = epoch2.clone();
        s1.sort();
        s2.sort();
        let expected: Vec<usize> = (0..20).collect();
        assert_eq!(s1, expected);
        assert_eq!(s2, expected);
    }

    #[test]
    fn test_random_sampler_seed_reproducibility() {
        let mut s1 = RandomSampler::new(10, 123);
        let mut s2 = RandomSampler::new(10, 123);
        assert_eq!(s1.indices(), s2.indices());
    }

    #[test]
    fn test_weighted_random_sampler() {
        // Weight index 2 very heavily
        let weights = vec![0.0, 0.0, 1.0, 0.0, 0.0];
        let mut s = WeightedRandomSampler::new(weights, 10, 42);
        let idx = s.indices();
        assert_eq!(idx.len(), 10);
        // All samples should be index 2
        assert!(idx.iter().all(|&i| i == 2));
    }

    #[test]
    fn test_weighted_random_sampler_distribution() {
        // Weight index 0 at 90%, index 1 at 10%
        let weights = vec![9.0, 1.0];
        let mut s = WeightedRandomSampler::new(weights, 1000, 42);
        let idx = s.indices();
        let count_0 = idx.iter().filter(|&&i| i == 0).count();
        // Should be roughly 900 +/- some variance
        assert!(count_0 > 800, "expected ~900, got {}", count_0);
        assert!(count_0 < 980, "expected ~900, got {}", count_0);
    }

    #[test]
    #[should_panic(expected = "weights must not be empty")]
    fn test_weighted_sampler_empty_weights() {
        WeightedRandomSampler::new(vec![], 10, 42);
    }

    #[test]
    #[should_panic(expected = "total weight must be positive")]
    fn test_weighted_sampler_zero_weights() {
        WeightedRandomSampler::new(vec![0.0, 0.0], 10, 42);
    }
}
