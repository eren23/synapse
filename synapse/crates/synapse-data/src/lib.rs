pub mod collate;
pub mod dataloader;
pub mod dataset;
pub mod sampler;
pub mod transform;

use std::fmt;

/// A simple N-dimensional tensor backed by contiguous f32 data in row-major order.
#[derive(Clone)]
pub struct Tensor {
    data: Vec<f32>,
    shape: Vec<usize>,
}

impl fmt::Debug for Tensor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Tensor")
            .field("shape", &self.shape)
            .field("numel", &self.data.len())
            .finish()
    }
}

impl Tensor {
    /// Create a tensor from flat data and shape. Panics if data length != product of shape.
    pub fn new(data: Vec<f32>, shape: Vec<usize>) -> Self {
        let numel: usize = shape.iter().product();
        assert_eq!(
            data.len(),
            numel,
            "data length {} does not match shape {:?} (expected {})",
            data.len(),
            shape,
            numel
        );
        Self { data, shape }
    }

    /// Create a tensor filled with zeros.
    pub fn zeros(shape: Vec<usize>) -> Self {
        let numel: usize = shape.iter().product();
        Self {
            data: vec![0.0; numel],
            shape,
        }
    }

    /// Create a tensor filled with a constant value.
    pub fn full(shape: Vec<usize>, value: f32) -> Self {
        let numel: usize = shape.iter().product();
        Self {
            data: vec![value; numel],
            shape,
        }
    }

    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    pub fn ndim(&self) -> usize {
        self.shape.len()
    }

    pub fn numel(&self) -> usize {
        self.data.len()
    }

    pub fn data(&self) -> &[f32] {
        &self.data
    }

    pub fn data_mut(&mut self) -> &mut [f32] {
        &mut self.data
    }

    /// Select a single element along the first dimension (row), returning a tensor
    /// with one fewer dimension. For a tensor of shape [N, D1, D2, ...], returns shape [D1, D2, ...].
    /// For a 1-D tensor of shape [N], returns a scalar tensor of shape [1].
    pub fn select(&self, index: usize) -> Self {
        assert!(
            !self.shape.is_empty(),
            "cannot select from a scalar tensor"
        );
        assert!(
            index < self.shape[0],
            "index {} out of bounds for dimension 0 with size {}",
            index,
            self.shape[0]
        );

        let inner_numel: usize = self.shape[1..].iter().product::<usize>().max(1);
        let start = index * inner_numel;
        let end = start + inner_numel;
        let data = self.data[start..end].to_vec();

        let new_shape = if self.shape.len() == 1 {
            vec![1]
        } else {
            self.shape[1..].to_vec()
        };

        Self {
            data,
            shape: new_shape,
        }
    }

    /// Stack multiple tensors along a new leading dimension (dim 0).
    /// All tensors must have the same shape.
    /// Result shape: [N, ...original_shape].
    pub fn stack(tensors: &[Tensor]) -> Self {
        assert!(!tensors.is_empty(), "cannot stack empty tensor list");
        let expected_shape = &tensors[0].shape;
        for (i, t) in tensors.iter().enumerate().skip(1) {
            assert_eq!(
                &t.shape, expected_shape,
                "shape mismatch at index {}: {:?} vs {:?}",
                i, t.shape, expected_shape
            );
        }

        let n = tensors.len();
        let elem_numel = tensors[0].numel();
        let mut data = Vec::with_capacity(n * elem_numel);
        for t in tensors {
            data.extend_from_slice(&t.data);
        }

        let mut shape = Vec::with_capacity(expected_shape.len() + 1);
        shape.push(n);
        shape.extend_from_slice(expected_shape);

        Self { data, shape }
    }
}

impl PartialEq for Tensor {
    fn eq(&self, other: &Self) -> bool {
        self.shape == other.shape && self.data == other.data
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tensor_new() {
        let t = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
        assert_eq!(t.shape(), &[2, 3]);
        assert_eq!(t.numel(), 6);
        assert_eq!(t.ndim(), 2);
    }

    #[test]
    fn test_tensor_zeros() {
        let t = Tensor::zeros(vec![3, 4]);
        assert_eq!(t.shape(), &[3, 4]);
        assert!(t.data().iter().all(|&v| v == 0.0));
    }

    #[test]
    fn test_tensor_select() {
        let t = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
        let row0 = t.select(0);
        assert_eq!(row0.shape(), &[3]);
        assert_eq!(row0.data(), &[1.0, 2.0, 3.0]);

        let row1 = t.select(1);
        assert_eq!(row1.shape(), &[3]);
        assert_eq!(row1.data(), &[4.0, 5.0, 6.0]);
    }

    #[test]
    fn test_tensor_select_1d() {
        let t = Tensor::new(vec![10.0, 20.0, 30.0], vec![3]);
        let s = t.select(1);
        assert_eq!(s.shape(), &[1]);
        assert_eq!(s.data(), &[20.0]);
    }

    #[test]
    fn test_tensor_stack() {
        let a = Tensor::new(vec![1.0, 2.0], vec![2]);
        let b = Tensor::new(vec![3.0, 4.0], vec![2]);
        let c = Tensor::new(vec![5.0, 6.0], vec![2]);
        let stacked = Tensor::stack(&[a, b, c]);
        assert_eq!(stacked.shape(), &[3, 2]);
        assert_eq!(stacked.data(), &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    }

    #[test]
    #[should_panic]
    fn test_tensor_new_shape_mismatch() {
        Tensor::new(vec![1.0, 2.0], vec![3]);
    }

    #[test]
    #[should_panic]
    fn test_tensor_select_out_of_bounds() {
        let t = Tensor::new(vec![1.0, 2.0], vec![2]);
        t.select(5);
    }
}
