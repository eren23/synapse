//! Embedding layer: lookup table for dense vectors.

use synapse_autograd::Tensor;

use crate::init::randn;
use crate::module::Module;

pub struct Embedding {
    pub weight: Tensor, // [num_embeddings, embedding_dim]
    pub num_embeddings: usize,
    pub embedding_dim: usize,
    /// Indices accessed in the last forward pass (for sparse gradient).
    pub last_indices: Vec<usize>,
    training: bool,
}

impl Embedding {
    /// Create an Embedding layer with normally-distributed weights.
    pub fn new(num_embeddings: usize, embedding_dim: usize) -> Self {
        let weight = randn(&[num_embeddings, embedding_dim], 1.0);
        Embedding {
            weight,
            num_embeddings,
            embedding_dim,
            last_indices: Vec::new(),
            training: true,
        }
    }
}

impl Module for Embedding {
    /// Forward: input tensor of indices (f32, cast to usize) of any shape [*]
    /// Output: [*, embedding_dim]
    ///
    /// The input values are interpreted as integer indices into the embedding table.
    fn forward(&self, input: &Tensor) -> Tensor {
        let n = input.numel();
        let mut output = vec![0.0f32; n * self.embedding_dim];

        for i in 0..n {
            let idx = input.data[i] as usize;
            assert!(
                idx < self.num_embeddings,
                "embedding index {} out of range [0, {})",
                idx,
                self.num_embeddings
            );
            let src_start = idx * self.embedding_dim;
            let dst_start = i * self.embedding_dim;
            output[dst_start..dst_start + self.embedding_dim]
                .copy_from_slice(&self.weight.data[src_start..src_start + self.embedding_dim]);
        }

        // Output shape: input_shape + [embedding_dim]
        let mut out_shape = input.shape.clone();
        out_shape.push(self.embedding_dim);

        Tensor::new(output, out_shape)
    }

    fn parameters(&self) -> Vec<&Tensor> {
        vec![&self.weight]
    }

    fn parameters_mut(&mut self) -> Vec<&mut Tensor> {
        vec![&mut self.weight]
    }

    fn set_training(&mut self, training: bool) {
        self.training = training;
    }

    fn is_training(&self) -> bool {
        self.training
    }

    fn name(&self) -> &str {
        "Embedding"
    }
}
