use crate::Tensor;
use rand::Rng;

/// A transform modifies tensor data.
pub trait Transform: Send + Sync {
    fn apply(&self, tensor: &Tensor) -> Tensor;
}

/// Normalize: `(x - mean) / std` element-wise.
pub struct Normalize {
    pub mean: f32,
    pub std: f32,
}

impl Normalize {
    pub fn new(mean: f32, std: f32) -> Self {
        assert!(std != 0.0, "std must be non-zero");
        Self { mean, std }
    }
}

impl Transform for Normalize {
    fn apply(&self, tensor: &Tensor) -> Tensor {
        let data: Vec<f32> = tensor.data().iter().map(|&x| (x - self.mean) / self.std).collect();
        Tensor::new(data, tensor.shape().to_vec())
    }
}

/// Randomly flip tensor data along the last dimension with probability `p`.
/// Useful for data augmentation on image-like tensors.
pub struct RandomHorizontalFlip {
    pub p: f32,
}

impl RandomHorizontalFlip {
    pub fn new(p: f32) -> Self {
        assert!((0.0..=1.0).contains(&p), "p must be in [0, 1]");
        Self { p }
    }
}

impl Transform for RandomHorizontalFlip {
    fn apply(&self, tensor: &Tensor) -> Tensor {
        let mut rng = rand::thread_rng();
        if rng.gen::<f32>() >= self.p {
            return tensor.clone();
        }

        let shape = tensor.shape();
        if shape.is_empty() {
            return tensor.clone();
        }

        let last_dim = *shape.last().unwrap();
        if last_dim <= 1 {
            return tensor.clone();
        }

        let mut data = tensor.data().to_vec();
        // Flip along the last dimension: process in chunks of `last_dim`
        for chunk in data.chunks_mut(last_dim) {
            chunk.reverse();
        }

        Tensor::new(data, shape.to_vec())
    }
}

/// Identity transform that converts raw f32 data into a Tensor.
/// In our framework tensors are already f32, so this is effectively a clone/identity,
/// but it serves as a composable pipeline element.
pub struct ToTensor;

impl Transform for ToTensor {
    fn apply(&self, tensor: &Tensor) -> Tensor {
        tensor.clone()
    }
}

/// Compose multiple transforms into a single sequential pipeline.
pub struct Compose {
    transforms: Vec<Box<dyn Transform>>,
}

impl Compose {
    pub fn new(transforms: Vec<Box<dyn Transform>>) -> Self {
        Self { transforms }
    }
}

impl Transform for Compose {
    fn apply(&self, tensor: &Tensor) -> Tensor {
        let mut result = tensor.clone();
        for t in &self.transforms {
            result = t.apply(&result);
        }
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_normalize() {
        let t = Tensor::new(vec![2.0, 4.0, 6.0, 8.0], vec![4]);
        let norm = Normalize::new(5.0, 2.0);
        let result = norm.apply(&t);
        assert_eq!(result.shape(), &[4]);
        let expected: Vec<f32> = vec![-1.5, -0.5, 0.5, 1.5];
        for (a, b) in result.data().iter().zip(expected.iter()) {
            assert!((a - b).abs() < 1e-6, "{} != {}", a, b);
        }
    }

    #[test]
    fn test_normalize_preserves_shape() {
        let t = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let norm = Normalize::new(0.0, 1.0);
        let result = norm.apply(&t);
        assert_eq!(result.shape(), &[2, 2]);
        assert_eq!(result.data(), t.data());
    }

    #[test]
    fn test_random_horizontal_flip_always() {
        // p=1.0 means always flip
        let t = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
        let flip = RandomHorizontalFlip::new(1.0);
        let result = flip.apply(&t);
        assert_eq!(result.shape(), &[2, 3]);
        assert_eq!(result.data(), &[3.0, 2.0, 1.0, 6.0, 5.0, 4.0]);
    }

    #[test]
    fn test_random_horizontal_flip_never() {
        let t = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let flip = RandomHorizontalFlip::new(0.0);
        let result = flip.apply(&t);
        assert_eq!(result.data(), t.data());
    }

    #[test]
    fn test_to_tensor() {
        let t = Tensor::new(vec![1.0, 2.0, 3.0], vec![3]);
        let tt = ToTensor;
        let result = tt.apply(&t);
        assert_eq!(result.data(), t.data());
        assert_eq!(result.shape(), t.shape());
    }

    #[test]
    fn test_compose() {
        let t = Tensor::new(vec![10.0, 20.0, 30.0, 40.0], vec![4]);
        let pipeline = Compose::new(vec![
            Box::new(Normalize::new(25.0, 10.0)),
            Box::new(ToTensor),
        ]);
        let result = pipeline.apply(&t);
        let expected: Vec<f32> = vec![-1.5, -0.5, 0.5, 1.5];
        for (a, b) in result.data().iter().zip(expected.iter()) {
            assert!((a - b).abs() < 1e-6);
        }
    }

    #[test]
    #[should_panic(expected = "std must be non-zero")]
    fn test_normalize_zero_std() {
        Normalize::new(0.0, 0.0);
    }
}
