//! Positional encoding modules for sequence models.

use synapse_autograd::Tensor;

use crate::embedding::Embedding;
use crate::module::Module;

// ═══════════════════════════════════════════════════════════════════════
// SinusoidalPositionalEncoding
// ═══════════════════════════════════════════════════════════════════════

/// Fixed sinusoidal positional encoding (Vaswani et al., 2017).
///
/// Precomputes a `[max_len, d_model]` table using:
///   PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
///   PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
///
/// Forward adds the encoding to input via broadcast.
/// Input: `[B, S, D]` → Output: `[B, S, D]`
pub struct SinusoidalPositionalEncoding {
    pub table: Tensor,
    pub max_len: usize,
    pub d_model: usize,
    training: bool,
}

impl SinusoidalPositionalEncoding {
    pub fn new(max_len: usize, d_model: usize) -> Self {
        let mut data = vec![0.0f32; max_len * d_model];
        for pos in 0..max_len {
            for i in 0..(d_model / 2) {
                let freq = 10000.0f32.powf(2.0 * i as f32 / d_model as f32);
                let theta = pos as f32 / freq;
                data[pos * d_model + 2 * i] = theta.sin();
                data[pos * d_model + 2 * i + 1] = theta.cos();
            }
            if d_model % 2 == 1 {
                let i = d_model / 2;
                let freq = 10000.0f32.powf(2.0 * i as f32 / d_model as f32);
                let theta = pos as f32 / freq;
                data[pos * d_model + d_model - 1] = theta.sin();
            }
        }
        SinusoidalPositionalEncoding {
            table: Tensor::new(data, vec![max_len, d_model]),
            max_len,
            d_model,
            training: false,
        }
    }
}

impl Module for SinusoidalPositionalEncoding {
    fn forward(&self, input: &Tensor) -> Tensor {
        assert_eq!(input.shape.len(), 3, "expected [B, S, D] input");
        let seq_len = input.shape[1];
        assert!(
            seq_len <= self.max_len,
            "sequence length {} exceeds max_len {}",
            seq_len,
            self.max_len
        );
        assert_eq!(input.shape[2], self.d_model, "d_model mismatch");

        let pe_data: Vec<f32> = self.table.data[..seq_len * self.d_model].to_vec();
        let pe = Tensor::new(pe_data, vec![seq_len, self.d_model]);
        let pe = pe.unsqueeze(0); // [1, S, D]
        input.add_broadcast(&pe)
    }

    fn parameters(&self) -> Vec<&Tensor> {
        vec![]
    }

    fn parameters_mut(&mut self) -> Vec<&mut Tensor> {
        vec![]
    }

    fn set_training(&mut self, training: bool) {
        self.training = training;
    }

    fn is_training(&self) -> bool {
        self.training
    }

    fn name(&self) -> &str {
        "SinusoidalPositionalEncoding"
    }
}

// ═══════════════════════════════════════════════════════════════════════
// LearnablePositionalEmbedding
// ═══════════════════════════════════════════════════════════════════════

/// Learnable positional embedding backed by an `Embedding(max_len, d_model)`.
///
/// Generates position indices `[0, 1, …, seq_len-1]` automatically and
/// adds the looked-up embeddings to the input.
/// Input: `[B, S, D]` → Output: `[B, S, D]`
pub struct LearnablePositionalEmbedding {
    pub embedding: Embedding,
    pub max_len: usize,
    pub d_model: usize,
    training: bool,
}

impl LearnablePositionalEmbedding {
    pub fn new(max_len: usize, d_model: usize) -> Self {
        LearnablePositionalEmbedding {
            embedding: Embedding::new(max_len, d_model),
            max_len,
            d_model,
            training: true,
        }
    }
}

impl Module for LearnablePositionalEmbedding {
    fn forward(&self, input: &Tensor) -> Tensor {
        assert_eq!(input.shape.len(), 3, "expected [B, S, D] input");
        let seq_len = input.shape[1];
        assert!(
            seq_len <= self.max_len,
            "sequence length {} exceeds max_len {}",
            seq_len,
            self.max_len
        );
        assert_eq!(input.shape[2], self.d_model, "d_model mismatch");

        let indices_data: Vec<f32> = (0..seq_len).map(|i| i as f32).collect();
        let indices = Tensor::new(indices_data, vec![seq_len]);
        let pe = self.embedding.forward(&indices); // [S, D]
        let pe = pe.unsqueeze(0); // [1, S, D]
        input.add_broadcast(&pe)
    }

    fn parameters(&self) -> Vec<&Tensor> {
        vec![&self.embedding.weight]
    }

    fn parameters_mut(&mut self) -> Vec<&mut Tensor> {
        vec![&mut self.embedding.weight]
    }

    fn set_training(&mut self, training: bool) {
        self.training = training;
    }

    fn is_training(&self) -> bool {
        self.training
    }

    fn name(&self) -> &str {
        "LearnablePositionalEmbedding"
    }
}

// ═══════════════════════════════════════════════════════════════════════
// RotaryPositionalEmbedding
// ═══════════════════════════════════════════════════════════════════════

/// Rotary Position Embedding (Su et al., 2021).
///
/// Precomputes cos/sin tables of shape `[max_len, d_head/2]` using:
///   theta(pos, i) = pos / 10000^(2i/d_head)
///
/// Applied to query/key tensors of shape `[B, H, S, D]` via `apply`.
pub struct RotaryPositionalEmbedding {
    pub cos_table: Tensor,
    pub sin_table: Tensor,
    pub max_len: usize,
    pub d_head: usize,
}

impl RotaryPositionalEmbedding {
    pub fn new(max_len: usize, d_head: usize) -> Self {
        let half_d = d_head / 2;
        let mut cos_data = vec![0.0f32; max_len * half_d];
        let mut sin_data = vec![0.0f32; max_len * half_d];
        for s in 0..max_len {
            for i in 0..half_d {
                let theta = (s as f32) / 10000.0f32.powf(2.0 * i as f32 / d_head as f32);
                cos_data[s * half_d + i] = theta.cos();
                sin_data[s * half_d + i] = theta.sin();
            }
        }
        RotaryPositionalEmbedding {
            cos_table: Tensor::new(cos_data, vec![max_len, half_d]),
            sin_table: Tensor::new(sin_data, vec![max_len, half_d]),
            max_len,
            d_head,
        }
    }

    /// Apply RoPE rotation to an input tensor.
    ///
    /// * `input` – shape `[B, H, S, D]`
    /// * `offset` – position offset (for incremental decoding)
    ///
    /// Returns the rotated tensor with the same shape.
    pub fn apply(&self, input: &Tensor, offset: usize) -> Tensor {
        let shape = &input.shape;
        assert_eq!(shape.len(), 4, "expected [B, H, S, D] input");
        let (batch, heads, seq, d_head) = (shape[0], shape[1], shape[2], shape[3]);
        assert_eq!(d_head, self.d_head, "d_head mismatch");
        assert!(
            offset + seq <= self.max_len,
            "offset + seq_len exceeds max_len"
        );
        let half_d = d_head / 2;

        let mut output = vec![0.0f32; input.numel()];

        for b in 0..batch {
            for h in 0..heads {
                for s in 0..seq {
                    let base = ((b * heads + h) * seq + s) * d_head;
                    let table_pos = offset + s;
                    for i in 0..half_d {
                        let cos_val = self.cos_table.data[table_pos * half_d + i];
                        let sin_val = self.sin_table.data[table_pos * half_d + i];
                        let x_even = input.data[base + 2 * i];
                        let x_odd = input.data[base + 2 * i + 1];

                        output[base + 2 * i] = x_even * cos_val - x_odd * sin_val;
                        output[base + 2 * i + 1] = x_even * sin_val + x_odd * cos_val;
                    }
                }
            }
        }

        Tensor::new(output, shape.clone())
    }
}

// ═══════════════════════════════════════════════════════════════════════
// MeanPool1d
// ═══════════════════════════════════════════════════════════════════════

/// Average pooling over the sequence dimension.
///
/// Reduces `[B, S, D]` → `[B, D]` by averaging over dim 1.
pub struct MeanPool1d {
    training: bool,
}

impl MeanPool1d {
    pub fn new() -> Self {
        MeanPool1d { training: false }
    }
}

impl Default for MeanPool1d {
    fn default() -> Self {
        Self::new()
    }
}

impl Module for MeanPool1d {
    fn forward(&self, input: &Tensor) -> Tensor {
        assert_eq!(input.shape.len(), 3, "expected [B, S, D] input");
        input.mean_axis(1, false)
    }

    fn parameters(&self) -> Vec<&Tensor> {
        vec![]
    }

    fn parameters_mut(&mut self) -> Vec<&mut Tensor> {
        vec![]
    }

    fn set_training(&mut self, training: bool) {
        self.training = training;
    }

    fn is_training(&self) -> bool {
        self.training
    }

    fn name(&self) -> &str {
        "MeanPool1d"
    }
}

// ═══════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    // ── SinusoidalPositionalEncoding ─────────────────────────────────

    #[test]
    fn test_sinusoidal_output_shape() {
        let pe = SinusoidalPositionalEncoding::new(128, 64);
        let input = Tensor::zeros(&[2, 16, 64]);
        let output = pe.forward(&input);
        assert_eq!(output.shape, vec![2, 16, 64]);
    }

    #[test]
    fn test_sinusoidal_values_match_reference() {
        let d_model = 8;
        let pe = SinusoidalPositionalEncoding::new(4, d_model);

        // Hand-compute first few entries:
        // pos=0: all theta=0 => sin(0)=0, cos(0)=1
        // pos=1, i=0: theta = 1/10000^(0/8) = 1/1 = 1
        //   sin(1) ≈ 0.8414709848, cos(1) ≈ 0.5403023059
        // pos=1, i=1: theta = 1/10000^(2/8) = 1/10
        //   sin(0.1) ≈ 0.09983341665, cos(0.1) ≈ 0.99500416527
        // pos=2, i=0: theta = 2/1 = 2
        //   sin(2) ≈ 0.9092974268, cos(2) ≈ -0.4161468365

        // pos=0: all sin values are 0, all cos values are 1
        for i in 0..(d_model / 2) {
            assert!(
                pe.table.data[0 * d_model + 2 * i].abs() < 1e-6,
                "pos=0, sin dim {} should be 0",
                2 * i
            );
            assert!(
                (pe.table.data[0 * d_model + 2 * i + 1] - 1.0).abs() < 1e-6,
                "pos=0, cos dim {} should be 1",
                2 * i + 1
            );
        }

        // pos=1, i=0
        let theta_1_0 = 1.0f32 / 10000.0f32.powf(0.0 / d_model as f32);
        assert!(
            (pe.table.data[1 * d_model + 0] - theta_1_0.sin()).abs() < 1e-6,
            "pos=1, dim=0 sin mismatch"
        );
        assert!(
            (pe.table.data[1 * d_model + 1] - theta_1_0.cos()).abs() < 1e-6,
            "pos=1, dim=1 cos mismatch"
        );

        // pos=1, i=1
        let theta_1_1 = 1.0f32 / 10000.0f32.powf(2.0 / d_model as f32);
        assert!(
            (pe.table.data[1 * d_model + 2] - theta_1_1.sin()).abs() < 1e-6,
            "pos=1, dim=2 sin mismatch"
        );
        assert!(
            (pe.table.data[1 * d_model + 3] - theta_1_1.cos()).abs() < 1e-6,
            "pos=1, dim=3 cos mismatch"
        );

        // pos=2, i=0
        let theta_2_0 = 2.0f32 / 10000.0f32.powf(0.0 / d_model as f32);
        assert!(
            (pe.table.data[2 * d_model + 0] - theta_2_0.sin()).abs() < 1e-6,
            "pos=2, dim=0 sin mismatch"
        );
        assert!(
            (pe.table.data[2 * d_model + 1] - theta_2_0.cos()).abs() < 1e-6,
            "pos=2, dim=1 cos mismatch"
        );
    }

    #[test]
    fn test_sinusoidal_forward_adds_encoding() {
        let pe = SinusoidalPositionalEncoding::new(8, 4);
        let input = Tensor::zeros(&[1, 4, 4]);
        let output = pe.forward(&input);
        // With zero input, output should equal the PE table values
        for s in 0..4 {
            for d in 0..4 {
                assert!(
                    (output.data[s * 4 + d] - pe.table.data[s * 4 + d]).abs() < 1e-6,
                    "zero input + PE should equal PE table"
                );
            }
        }
    }

    #[test]
    fn test_sinusoidal_no_parameters() {
        let pe = SinusoidalPositionalEncoding::new(16, 8);
        assert!(pe.parameters().is_empty());
        assert_eq!(pe.name(), "SinusoidalPositionalEncoding");
    }

    // ── LearnablePositionalEmbedding ─────────────────────────────────

    #[test]
    fn test_learnable_output_shape() {
        let pe = LearnablePositionalEmbedding::new(64, 32);
        let input = Tensor::zeros(&[2, 16, 32]);
        let output = pe.forward(&input);
        assert_eq!(output.shape, vec![2, 16, 32]);
    }

    #[test]
    fn test_learnable_parameter_count() {
        let max_len = 64;
        let d_model = 32;
        let pe = LearnablePositionalEmbedding::new(max_len, d_model);
        let params = pe.parameters();
        assert_eq!(params.len(), 1);
        assert_eq!(params[0].numel(), max_len * d_model);
        assert_eq!(params[0].shape, vec![max_len, d_model]);
    }

    #[test]
    fn test_learnable_module_trait() {
        let mut pe = LearnablePositionalEmbedding::new(16, 8);
        assert!(pe.is_training());
        pe.set_training(false);
        assert!(!pe.is_training());
        assert_eq!(pe.name(), "LearnablePositionalEmbedding");
    }

    // ── RotaryPositionalEmbedding ────────────────────────────────────

    #[test]
    fn test_rope_table_shapes() {
        let max_len = 128;
        let d_head = 64;
        let rope = RotaryPositionalEmbedding::new(max_len, d_head);
        assert_eq!(rope.cos_table.shape, vec![128, 32]);
        assert_eq!(rope.sin_table.shape, vec![128, 32]);
    }

    #[test]
    fn test_rope_tables_match_theta_formula() {
        let max_len = 8;
        let d_head = 16;
        let half_d = d_head / 2;
        let rope = RotaryPositionalEmbedding::new(max_len, d_head);

        for s in 0..max_len {
            for i in 0..half_d {
                let theta = (s as f32) / 10000.0f32.powf(2.0 * i as f32 / d_head as f32);
                assert!(
                    (rope.cos_table.data[s * half_d + i] - theta.cos()).abs() < 1e-6,
                    "cos mismatch at pos={}, i={}",
                    s,
                    i
                );
                assert!(
                    (rope.sin_table.data[s * half_d + i] - theta.sin()).abs() < 1e-6,
                    "sin mismatch at pos={}, i={}",
                    s,
                    i
                );
            }
        }
    }

    #[test]
    fn test_rope_apply_output_shape() {
        let rope = RotaryPositionalEmbedding::new(64, 16);
        let input = Tensor::ones(&[2, 4, 8, 16]);
        let output = rope.apply(&input, 0);
        assert_eq!(output.shape, vec![2, 4, 8, 16]);
    }

    #[test]
    fn test_rope_different_positions_different_dots() {
        // Vectors at different positions should produce different dot products,
        // demonstrating that relative position matters.
        let rope = RotaryPositionalEmbedding::new(64, 8);
        let d = 8;

        // Same vector at two different positions
        let v: Vec<f32> = (0..d).map(|i| (i as f32 + 1.0) * 0.1).collect();
        let input_pos0 = Tensor::new(v.clone(), vec![1, 1, 1, d]);
        let input_pos5 = Tensor::new(v.clone(), vec![1, 1, 1, d]);

        let rotated_pos0 = rope.apply(&input_pos0, 0);
        let rotated_pos5 = rope.apply(&input_pos5, 5);

        // Dot product of same vector at same position vs different positions
        let dot_same: f32 = rotated_pos0
            .data
            .iter()
            .zip(&rotated_pos0.data)
            .map(|(a, b)| a * b)
            .sum();
        let dot_diff: f32 = rotated_pos0
            .data
            .iter()
            .zip(&rotated_pos5.data)
            .map(|(a, b)| a * b)
            .sum();

        assert!(
            (dot_same - dot_diff).abs() > 1e-4,
            "dot products should differ for different positions: same={}, diff={}",
            dot_same,
            dot_diff
        );
    }

    #[test]
    fn test_rope_with_offset() {
        let rope = RotaryPositionalEmbedding::new(64, 8);
        let v: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let input = Tensor::new(v, vec![1, 1, 1, 8]);

        let out_offset0 = rope.apply(&input, 0);
        let out_offset3 = rope.apply(&input, 3);

        // Different offsets should produce different outputs
        assert_ne!(out_offset0.data, out_offset3.data);
    }

    // ── MeanPool1d ───────────────────────────────────────────────────

    #[test]
    fn test_meanpool_shape_reduction() {
        let pool = MeanPool1d::new();
        let input = Tensor::ones(&[4, 16, 64]);
        let output = pool.forward(&input);
        assert_eq!(output.shape, vec![4, 64]);
    }

    #[test]
    fn test_meanpool_values_correct() {
        let pool = MeanPool1d::new();
        // [1, 3, 2] with known values
        let data = vec![
            1.0, 2.0, // seq pos 0
            3.0, 4.0, // seq pos 1
            5.0, 6.0, // seq pos 2
        ];
        let input = Tensor::new(data, vec![1, 3, 2]);
        let output = pool.forward(&input);

        assert_eq!(output.shape, vec![1, 2]);
        // mean over seq: dim0 = (1+3+5)/3 = 3.0, dim1 = (2+4+6)/3 = 4.0
        assert!((output.data[0] - 3.0).abs() < 1e-6);
        assert!((output.data[1] - 4.0).abs() < 1e-6);
    }

    #[test]
    fn test_meanpool_batch() {
        let pool = MeanPool1d::new();
        // [2, 2, 3] - two batches
        let data = vec![
            1.0, 2.0, 3.0, // b0, s0
            4.0, 5.0, 6.0, // b0, s1
            7.0, 8.0, 9.0, // b1, s0
            10.0, 11.0, 12.0, // b1, s1
        ];
        let input = Tensor::new(data, vec![2, 2, 3]);
        let output = pool.forward(&input);

        assert_eq!(output.shape, vec![2, 3]);
        // b0: mean = (1+4)/2=2.5, (2+5)/2=3.5, (3+6)/2=4.5
        assert!((output.data[0] - 2.5).abs() < 1e-6);
        assert!((output.data[1] - 3.5).abs() < 1e-6);
        assert!((output.data[2] - 4.5).abs() < 1e-6);
        // b1: mean = (7+10)/2=8.5, (8+11)/2=9.5, (9+12)/2=10.5
        assert!((output.data[3] - 8.5).abs() < 1e-6);
        assert!((output.data[4] - 9.5).abs() < 1e-6);
        assert!((output.data[5] - 10.5).abs() < 1e-6);
    }

    #[test]
    fn test_meanpool_no_parameters() {
        let pool = MeanPool1d::new();
        assert!(pool.parameters().is_empty());
        assert_eq!(pool.name(), "MeanPool1d");
    }

    // ── Module trait compliance ──────────────────────────────────────

    #[test]
    fn test_all_modules_implement_module_trait() {
        // Verify each module can be used as a trait object
        let modules: Vec<Box<dyn Module>> = vec![
            Box::new(SinusoidalPositionalEncoding::new(16, 8)),
            Box::new(LearnablePositionalEmbedding::new(16, 8)),
            Box::new(MeanPool1d::new()),
        ];

        let input = Tensor::ones(&[1, 4, 8]);
        for module in &modules {
            let output = module.forward(&input);
            assert!(
                !output.data.is_empty(),
                "{} produced empty output",
                module.name()
            );
        }
    }

    #[test]
    fn test_training_mode_toggle() {
        let mut sinusoidal = SinusoidalPositionalEncoding::new(8, 4);
        assert!(!sinusoidal.is_training());
        sinusoidal.set_training(true);
        assert!(sinusoidal.is_training());

        let mut learnable = LearnablePositionalEmbedding::new(8, 4);
        assert!(learnable.is_training());
        learnable.set_training(false);
        assert!(!learnable.is_training());

        let mut pool = MeanPool1d::new();
        assert!(!pool.is_training());
        pool.set_training(true);
        assert!(pool.is_training());
    }
}
