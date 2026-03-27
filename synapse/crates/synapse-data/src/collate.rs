use crate::Tensor;

/// Default collate function: given a batch of samples (each a `Vec<Tensor>`),
/// stack corresponding tensors along a new leading batch dimension.
///
/// For N samples each containing K tensors, produces K tensors each with
/// a new leading dimension of size N.
///
/// Example: 4 samples of [features_shape=[3], label_shape=[1]]
/// → [features_shape=[4,3], label_shape=[4,1]]
pub fn default_collate(samples: &[Vec<Tensor>]) -> Vec<Tensor> {
    assert!(!samples.is_empty(), "cannot collate empty batch");

    let num_fields = samples[0].len();
    assert!(
        samples.iter().all(|s| s.len() == num_fields),
        "all samples must have the same number of tensors"
    );

    (0..num_fields)
        .map(|field_idx| {
            let tensors: Vec<&Tensor> = samples.iter().map(|s| &s[field_idx]).collect();
            stack_tensors(&tensors)
        })
        .collect()
}

/// Stack a slice of tensor references along a new leading (dim 0) axis.
/// All tensors must have the same shape.
fn stack_tensors(tensors: &[&Tensor]) -> Tensor {
    assert!(!tensors.is_empty(), "cannot stack empty tensor list");
    let shape = tensors[0].shape();
    for (i, t) in tensors.iter().enumerate().skip(1) {
        assert_eq!(
            t.shape(),
            shape,
            "shape mismatch at index {}: {:?} vs {:?}",
            i,
            t.shape(),
            shape
        );
    }

    let n = tensors.len();
    let elem_numel = tensors[0].numel();
    let mut data = Vec::with_capacity(n * elem_numel);
    for t in tensors {
        data.extend_from_slice(t.data());
    }

    let mut new_shape = Vec::with_capacity(shape.len() + 1);
    new_shape.push(n);
    new_shape.extend_from_slice(shape);
    Tensor::new(data, new_shape)
}

/// Pad a batch of variable-length 1-D tensors to the maximum length, filling
/// with `pad_value`. Returns a single 2-D tensor of shape [N, max_len].
pub fn pad_sequences(tensors: &[Tensor], pad_value: f32) -> Tensor {
    assert!(!tensors.is_empty(), "cannot pad empty sequence list");
    for t in tensors {
        assert_eq!(
            t.ndim(),
            1,
            "pad_sequences expects 1-D tensors, got shape {:?}",
            t.shape()
        );
    }

    let max_len = tensors.iter().map(|t| t.shape()[0]).max().unwrap();
    let n = tensors.len();
    let mut data = vec![pad_value; n * max_len];

    for (i, t) in tensors.iter().enumerate() {
        let len = t.shape()[0];
        data[i * max_len..i * max_len + len].copy_from_slice(t.data());
    }

    Tensor::new(data, vec![n, max_len])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_collate_shapes() {
        let samples = vec![
            vec![
                Tensor::new(vec![1.0, 2.0, 3.0], vec![3]),
                Tensor::new(vec![0.0], vec![1]),
            ],
            vec![
                Tensor::new(vec![4.0, 5.0, 6.0], vec![3]),
                Tensor::new(vec![1.0], vec![1]),
            ],
            vec![
                Tensor::new(vec![7.0, 8.0, 9.0], vec![3]),
                Tensor::new(vec![2.0], vec![1]),
            ],
        ];

        let batch = default_collate(&samples);
        assert_eq!(batch.len(), 2);
        assert_eq!(batch[0].shape(), &[3, 3]); // [batch=3, features=3]
        assert_eq!(batch[1].shape(), &[3, 1]); // [batch=3, label=1]
        assert_eq!(
            batch[0].data(),
            &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]
        );
        assert_eq!(batch[1].data(), &[0.0, 1.0, 2.0]);
    }

    #[test]
    fn test_default_collate_2d_features() {
        let samples = vec![
            vec![Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2])],
            vec![Tensor::new(vec![5.0, 6.0, 7.0, 8.0], vec![2, 2])],
        ];

        let batch = default_collate(&samples);
        assert_eq!(batch.len(), 1);
        assert_eq!(batch[0].shape(), &[2, 2, 2]); // [batch=2, 2, 2]
    }

    #[test]
    fn test_pad_sequences() {
        let seqs = vec![
            Tensor::new(vec![1.0, 2.0], vec![2]),
            Tensor::new(vec![3.0, 4.0, 5.0], vec![3]),
            Tensor::new(vec![6.0], vec![1]),
        ];

        let padded = pad_sequences(&seqs, 0.0);
        assert_eq!(padded.shape(), &[3, 3]); // [N=3, max_len=3]
        assert_eq!(
            padded.data(),
            &[
                1.0, 2.0, 0.0, // seq 0, padded
                3.0, 4.0, 5.0, // seq 1, no padding
                6.0, 0.0, 0.0, // seq 2, padded
            ]
        );
    }

    #[test]
    fn test_pad_sequences_custom_pad_value() {
        let seqs = vec![
            Tensor::new(vec![1.0], vec![1]),
            Tensor::new(vec![2.0, 3.0], vec![2]),
        ];
        let padded = pad_sequences(&seqs, -1.0);
        assert_eq!(padded.data(), &[1.0, -1.0, 2.0, 3.0]);
    }

    #[test]
    #[should_panic(expected = "cannot collate empty batch")]
    fn test_collate_empty() {
        default_collate(&[]);
    }

    #[test]
    #[should_panic(expected = "cannot pad empty sequence list")]
    fn test_pad_empty() {
        pad_sequences(&[], 0.0);
    }
}
