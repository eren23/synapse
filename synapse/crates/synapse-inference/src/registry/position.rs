//! Positional encoding implementations: RoPE and Learned embeddings.

use super::PositionVariant;
#[cfg(feature = "zig-ffi")]
use synapse_core::{SynapseError, Tensor};

// ── RoPE ─────────────────────────────────────────────────────────────

/// Rotary Positional Embedding (RoPE).
///
/// Precomputes cos/sin caches for the full position range. Rotation is applied
/// to pairs of dimensions in the head vector, encoding relative position
/// information that the model can use in attention computation.
#[derive(Debug)]
pub struct RoPEPosition {
    base: f64,
    max_position_embeddings: usize,
    head_dim: usize,
    #[allow(dead_code)]
    cos_data: Vec<f32>, // [max_pos, head_dim/2]
    #[allow(dead_code)]
    sin_data: Vec<f32>, // [max_pos, head_dim/2]
}

impl RoPEPosition {
    /// Create RoPE tables for the given base frequency and dimensions.
    ///
    /// `head_dim` must be even (pairs of dimensions are rotated together).
    pub fn new(base: f64, max_position_embeddings: usize, head_dim: usize) -> Self {
        assert!(head_dim % 2 == 0, "head_dim must be even for RoPE");
        let half_d = head_dim / 2;
        let mut cos_data = vec![0.0f32; max_position_embeddings * half_d];
        let mut sin_data = vec![0.0f32; max_position_embeddings * half_d];

        for pos in 0..max_position_embeddings {
            for i in 0..half_d {
                let freq = 1.0 / (base as f32).powf(2.0 * i as f32 / head_dim as f32);
                let angle = pos as f32 * freq;
                cos_data[pos * half_d + i] = angle.cos();
                sin_data[pos * half_d + i] = angle.sin();
            }
        }

        Self {
            base,
            max_position_embeddings,
            head_dim,
            cos_data,
            sin_data,
        }
    }

    /// Apply RoPE to a 4-D tensor `[batch, heads, seq, head_dim]`.
    ///
    /// `offset` is the starting position index (for KV-cache decode).
    #[cfg(feature = "zig-ffi")]
    pub fn apply(&self, input: &Tensor, offset: usize) -> Result<Tensor, SynapseError> {
        let half_d = self.head_dim / 2;
        let cos = Tensor::from_data(&self.cos_data, &[self.max_position_embeddings, half_d])?;
        let sin = Tensor::from_data(&self.sin_data, &[self.max_position_embeddings, half_d])?;
        input.rope(&cos, &sin, offset)
    }

    /// Get cos/sin cache tensors for use with attention layers directly.
    #[cfg(feature = "zig-ffi")]
    pub fn cos_sin_tensors(&self) -> Result<(Tensor, Tensor), SynapseError> {
        let half_d = self.head_dim / 2;
        Ok((
            Tensor::from_data(&self.cos_data, &[self.max_position_embeddings, half_d])?,
            Tensor::from_data(&self.sin_data, &[self.max_position_embeddings, half_d])?,
        ))
    }

    pub fn head_dim(&self) -> usize {
        self.head_dim
    }
}

impl PositionVariant for RoPEPosition {
    fn max_position_embeddings(&self) -> usize {
        self.max_position_embeddings
    }
    fn base(&self) -> Option<f64> {
        Some(self.base)
    }
    fn name(&self) -> &str {
        "RoPE"
    }
}

// ── Learned Positional Embeddings ────────────────────────────────────

/// Learned positional embeddings.
///
/// Stores a trainable embedding table `[max_positions, hidden_size]` and adds
/// the appropriate row to each token's hidden state.
#[derive(Debug)]
pub struct LearnedPosition {
    max_position_embeddings: usize,
    hidden_size: usize,
    embeddings: Vec<f32>, // [max_pos, hidden_size]
}

impl LearnedPosition {
    pub fn new(max_position_embeddings: usize, hidden_size: usize) -> Self {
        Self {
            max_position_embeddings,
            hidden_size,
            embeddings: vec![0.0; max_position_embeddings * hidden_size],
        }
    }

    /// Set the embedding table. Must have `max_position_embeddings * hidden_size` elements.
    pub fn set_embeddings(&mut self, embeddings: Vec<f32>) {
        assert_eq!(
            embeddings.len(),
            self.max_position_embeddings * self.hidden_size
        );
        self.embeddings = embeddings;
    }

    /// Add positional embeddings to input `[batch, seq_len, hidden_size]` in-place.
    ///
    /// `offset` is the starting position index (for KV-cache decode).
    pub fn apply(
        &self,
        input: &mut [f32],
        batch: usize,
        seq_len: usize,
        offset: usize,
    ) -> Result<(), &'static str> {
        if offset + seq_len > self.max_position_embeddings {
            return Err("offset + seq_len exceeds max_position_embeddings");
        }
        for b in 0..batch {
            for s in 0..seq_len {
                let pos = offset + s;
                let inp = (b * seq_len + s) * self.hidden_size;
                let emb = pos * self.hidden_size;
                for h in 0..self.hidden_size {
                    input[inp + h] += self.embeddings[emb + h];
                }
            }
        }
        Ok(())
    }

    pub fn hidden_size(&self) -> usize {
        self.hidden_size
    }
}

impl PositionVariant for LearnedPosition {
    fn max_position_embeddings(&self) -> usize {
        self.max_position_embeddings
    }
    fn name(&self) -> &str {
        "Learned"
    }
}

// ── Tests ────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rope_precompute_shape() {
        let rope = RoPEPosition::new(10000.0, 128, 8);
        assert_eq!(rope.cos_data.len(), 128 * 4); // max_pos * half_d
        assert_eq!(rope.sin_data.len(), 128 * 4);
    }

    #[test]
    #[cfg(feature = "zig-ffi")]
    fn rope_position_zero_identity() {
        let rope = RoPEPosition::new(10000.0, 16, 8);
        let data: Vec<f32> = (0..8).map(|i| i as f32 + 1.0).collect();
        let input = Tensor::from_data(&data, &[1, 1, 1, 8]).unwrap();

        let output = rope.apply(&input, 0).unwrap();
        let out_data = output.to_vec().unwrap();

        // At position 0 all angles are 0 → cos=1, sin=0 → output == input
        for i in 0..8 {
            assert!(
                (out_data[i] - data[i]).abs() < 1e-4,
                "pos-0 should be identity: got {} vs {}",
                out_data[i],
                data[i]
            );
        }
    }

    #[test]
    #[cfg(feature = "zig-ffi")]
    fn rope_positions_affect_output() {
        let rope = RoPEPosition::new(10000.0, 128, 8);
        let data: Vec<f32> = (0..8).map(|i| i as f32 + 1.0).collect();
        let input = Tensor::from_data(&data, &[1, 1, 1, 8]).unwrap();

        let out0 = rope.apply(&input, 0).unwrap().to_vec().unwrap();
        let out5 = rope.apply(&input, 5).unwrap().to_vec().unwrap();

        let differs = out0
            .iter()
            .zip(out5.iter())
            .any(|(a, b)| (a - b).abs() > 1e-6);
        assert!(
            differs,
            "Different positions should produce different outputs"
        );
    }

    #[test]
    #[cfg(feature = "zig-ffi")]
    fn rope_cos_sin_tensors_shape() {
        let rope = RoPEPosition::new(10000.0, 64, 16);
        let (cos, sin) = rope.cos_sin_tensors().unwrap();
        assert_eq!(cos.shape().unwrap(), &[64, 8]); // max_pos, head_dim/2
        assert_eq!(sin.shape().unwrap(), &[64, 8]);
    }

    #[test]
    fn learned_adds_embeddings() {
        let mut pos = LearnedPosition::new(16, 4);
        let emb: Vec<f32> = (0..64).map(|i| i as f32 * 0.01).collect();
        pos.set_embeddings(emb.clone());

        let mut input = vec![1.0f32; 2 * 3 * 4]; // batch=2, seq=3, hidden=4
        pos.apply(&mut input, 2, 3, 0).unwrap();

        // batch 0, seq 0: should be 1.0 + emb[0..4]
        for i in 0..4 {
            assert!((input[i] - (1.0 + emb[i])).abs() < 1e-6);
        }
        // batch 0, seq 2: should be 1.0 + emb[8..12]
        for i in 0..4 {
            assert!((input[2 * 4 + i] - (1.0 + emb[2 * 4 + i])).abs() < 1e-6);
        }
    }

    #[test]
    fn learned_offset() {
        let mut pos = LearnedPosition::new(16, 4);
        let emb: Vec<f32> = (0..64).map(|i| i as f32 * 0.01).collect();
        pos.set_embeddings(emb.clone());

        let mut input = vec![0.0f32; 4]; // batch=1, seq=1, hidden=4
        pos.apply(&mut input, 1, 1, 5).unwrap();

        for i in 0..4 {
            assert!((input[i] - emb[5 * 4 + i]).abs() < 1e-6);
        }
    }

    #[test]
    fn learned_out_of_bounds() {
        let pos = LearnedPosition::new(8, 4);
        let mut input = vec![0.0f32; 4];
        assert!(pos.apply(&mut input, 1, 1, 8).is_err()); // offset 8 + seq 1 > max 8
    }

    #[test]
    fn rope_trait_accessors() {
        let rope = RoPEPosition::new(500000.0, 131072, 64);
        assert_eq!(rope.max_position_embeddings(), 131072);
        assert_eq!(rope.base(), Some(500000.0));
        assert_eq!(rope.name(), "RoPE");
    }

    #[test]
    fn learned_trait_accessors() {
        let learned = LearnedPosition::new(2048, 512);
        assert_eq!(learned.max_position_embeddings(), 2048);
        assert_eq!(learned.base(), None);
        assert_eq!(learned.name(), "Learned");
    }
}
