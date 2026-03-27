//! Transformer encoder and decoder blocks (pre-norm architecture).

use synapse_autograd::Tensor;

use crate::attention::MultiHeadAttention;
use crate::dropout::Dropout;
use crate::layernorm::LayerNorm;
use crate::linear::Linear;
use crate::module::Module;

// ── Activation enum ─────────────────────────────────────────────

/// Configurable activation function for the feed-forward network.
#[derive(Clone, Copy, Debug)]
pub enum Activation {
    ReLU,
    GELU,
}

impl Activation {
    fn apply(&self, input: &Tensor) -> Tensor {
        match self {
            Activation::ReLU => input.relu(),
            Activation::GELU => input.gelu(),
        }
    }
}

// ── Config structs ──────────────────────────────────────────────

#[derive(Clone)]
pub struct TransformerEncoderConfig {
    pub d_model: usize,
    pub n_heads: usize,
    pub d_ff: usize,
    pub n_layers: usize,
    pub dropout: f32,
    pub activation: Activation,
}

#[derive(Clone)]
pub struct TransformerDecoderConfig {
    pub d_model: usize,
    pub n_heads: usize,
    pub d_ff: usize,
    pub n_layers: usize,
    pub dropout: f32,
    pub activation: Activation,
}

// ── Helpers ─────────────────────────────────────────────────────

/// Apply a Linear layer to a 3D input `[B, S, D_in]` → `[B, S, D_out]`.
fn linear_3d(linear: &Linear, input: &Tensor) -> Tensor {
    let batch = input.shape[0];
    let seq = input.shape[1];
    let d_in = input.shape[2];
    let flat = input.reshape(&[batch * seq, d_in]);
    let out = linear.forward(&flat);
    out.reshape(&[batch, seq, linear.out_features()])
}

// ── TransformerEncoderLayer ────────────────────────────────────

/// Single transformer encoder layer with pre-norm architecture.
///
/// ```text
/// x = x + self_attn(norm1(x))
/// x = x + ff2(dropout(activation(ff1(norm2(x)))))
/// ```
pub struct TransformerEncoderLayer {
    pub self_attn: MultiHeadAttention,
    pub norm1: LayerNorm,
    pub norm2: LayerNorm,
    pub ff1: Linear,
    pub ff2: Linear,
    pub dropout: Dropout,
    activation: Activation,
    training: bool,
}

impl TransformerEncoderLayer {
    pub fn new(config: &TransformerEncoderConfig) -> Self {
        TransformerEncoderLayer {
            self_attn: MultiHeadAttention::new(config.d_model, config.n_heads, config.dropout),
            norm1: LayerNorm::new(&[config.d_model]).unwrap(),
            norm2: LayerNorm::new(&[config.d_model]).unwrap(),
            ff1: Linear::new(config.d_model, config.d_ff, true),
            ff2: Linear::new(config.d_ff, config.d_model, true),
            dropout: Dropout::new(config.dropout),
            activation: config.activation,
            training: true,
        }
    }
}

impl Module for TransformerEncoderLayer {
    fn forward(&self, input: &Tensor) -> Tensor {
        // Self-attention sub-layer with pre-norm
        let normed = self.norm1.forward(input);
        let attn_out = self.self_attn.forward(&normed);
        let x = input.add_broadcast(&attn_out);

        // Feed-forward sub-layer with pre-norm
        let normed = self.norm2.forward(&x);
        let ff_out = linear_3d(&self.ff1, &normed);
        let ff_out = self.activation.apply(&ff_out);
        let ff_out = self.dropout.forward(&ff_out);
        let ff_out = linear_3d(&self.ff2, &ff_out);
        x.add_broadcast(&ff_out)
    }

    fn parameters(&self) -> Vec<&Tensor> {
        let mut params = Vec::new();
        params.extend(self.self_attn.parameters());
        params.extend(self.norm1.parameters());
        params.extend(self.norm2.parameters());
        params.extend(self.ff1.parameters());
        params.extend(self.ff2.parameters());
        params
    }

    fn parameters_mut(&mut self) -> Vec<&mut Tensor> {
        let mut params = Vec::new();
        params.extend(self.self_attn.parameters_mut());
        params.extend(self.norm1.parameters_mut());
        params.extend(self.norm2.parameters_mut());
        params.extend(self.ff1.parameters_mut());
        params.extend(self.ff2.parameters_mut());
        params
    }

    fn set_training(&mut self, training: bool) {
        self.training = training;
        self.self_attn.set_training(training);
        self.norm1.set_training(training);
        self.norm2.set_training(training);
        self.ff1.set_training(training);
        self.ff2.set_training(training);
        self.dropout.set_training(training);
    }

    fn is_training(&self) -> bool {
        self.training
    }

    fn name(&self) -> &str {
        "TransformerEncoderLayer"
    }
}

// ── TransformerEncoder ─────────────────────────────────────────

/// Stack of N `TransformerEncoderLayer`s followed by a final LayerNorm.
pub struct TransformerEncoder {
    pub layers: Vec<TransformerEncoderLayer>,
    pub final_norm: LayerNorm,
    training: bool,
}

impl TransformerEncoder {
    pub fn new(config: &TransformerEncoderConfig) -> Self {
        let layers = (0..config.n_layers)
            .map(|_| TransformerEncoderLayer::new(config))
            .collect();
        TransformerEncoder {
            layers,
            final_norm: LayerNorm::new(&[config.d_model]).unwrap(),
            training: true,
        }
    }
}

impl Module for TransformerEncoder {
    fn forward(&self, input: &Tensor) -> Tensor {
        let mut x = input.clone();
        for layer in &self.layers {
            x = layer.forward(&x);
        }
        self.final_norm.forward(&x)
    }

    fn parameters(&self) -> Vec<&Tensor> {
        let mut params = Vec::new();
        for layer in &self.layers {
            params.extend(layer.parameters());
        }
        params.extend(self.final_norm.parameters());
        params
    }

    fn parameters_mut(&mut self) -> Vec<&mut Tensor> {
        let mut params = Vec::new();
        for layer in &mut self.layers {
            params.extend(layer.parameters_mut());
        }
        params.extend(self.final_norm.parameters_mut());
        params
    }

    fn set_training(&mut self, training: bool) {
        self.training = training;
        for layer in &mut self.layers {
            layer.set_training(training);
        }
        self.final_norm.set_training(training);
    }

    fn is_training(&self) -> bool {
        self.training
    }

    fn name(&self) -> &str {
        "TransformerEncoder"
    }
}

// ── TransformerDecoderLayer ────────────────────────────────────

/// Single transformer decoder layer with pre-norm architecture.
///
/// ```text
/// x = x + self_attn(norm1(x))                          [causal]
/// x = x + cross_attn(norm2(x), memory, memory)
/// x = x + ff2(dropout(activation(ff1(norm3(x)))))
/// ```
pub struct TransformerDecoderLayer {
    pub self_attn: MultiHeadAttention,
    pub cross_attn: MultiHeadAttention,
    pub norm1: LayerNorm,
    pub norm2: LayerNorm,
    pub norm3: LayerNorm,
    pub ff1: Linear,
    pub ff2: Linear,
    pub dropout: Dropout,
    activation: Activation,
    training: bool,
}

impl TransformerDecoderLayer {
    pub fn new(config: &TransformerDecoderConfig) -> Self {
        TransformerDecoderLayer {
            self_attn: MultiHeadAttention::new(config.d_model, config.n_heads, config.dropout),
            cross_attn: MultiHeadAttention::new(config.d_model, config.n_heads, config.dropout),
            norm1: LayerNorm::new(&[config.d_model]).unwrap(),
            norm2: LayerNorm::new(&[config.d_model]).unwrap(),
            norm3: LayerNorm::new(&[config.d_model]).unwrap(),
            ff1: Linear::new(config.d_model, config.d_ff, true),
            ff2: Linear::new(config.d_ff, config.d_model, true),
            dropout: Dropout::new(config.dropout),
            activation: config.activation,
            training: true,
        }
    }

    /// Forward with encoder memory and optional causal masking on self-attention.
    ///
    /// * `tgt` – decoder input `[B, Tgt_S, D]`
    /// * `memory` – encoder output `[B, Src_S, D]`
    /// * `causal` – apply causal mask to self-attention
    ///
    /// Returns `[B, Tgt_S, D]`.
    pub fn forward_with_memory(&self, tgt: &Tensor, memory: &Tensor, causal: bool) -> Tensor {
        // Causal self-attention with pre-norm
        let normed = self.norm1.forward(tgt);
        let self_attn_out = self
            .self_attn
            .forward_with_mask(&normed, &normed, &normed, causal);
        let x = tgt.add_broadcast(&self_attn_out);

        // Cross-attention with pre-norm (query=decoder, key/value=encoder memory)
        let normed = self.norm2.forward(&x);
        let cross_attn_out = self
            .cross_attn
            .forward_with_mask(&normed, memory, memory, false);
        let x = x.add_broadcast(&cross_attn_out);

        // Feed-forward with pre-norm
        let normed = self.norm3.forward(&x);
        let ff_out = linear_3d(&self.ff1, &normed);
        let ff_out = self.activation.apply(&ff_out);
        let ff_out = self.dropout.forward(&ff_out);
        let ff_out = linear_3d(&self.ff2, &ff_out);
        x.add_broadcast(&ff_out)
    }
}

impl Module for TransformerDecoderLayer {
    /// Default forward uses causal self-attention without cross-attention.
    /// For the full decoder layer with cross-attention, use `forward_with_memory`.
    fn forward(&self, input: &Tensor) -> Tensor {
        let normed = self.norm1.forward(input);
        let self_attn_out = self
            .self_attn
            .forward_with_mask(&normed, &normed, &normed, true);
        let x = input.add_broadcast(&self_attn_out);

        // Skip cross-attention (no memory available)

        let normed = self.norm3.forward(&x);
        let ff_out = linear_3d(&self.ff1, &normed);
        let ff_out = self.activation.apply(&ff_out);
        let ff_out = self.dropout.forward(&ff_out);
        let ff_out = linear_3d(&self.ff2, &ff_out);
        x.add_broadcast(&ff_out)
    }

    fn parameters(&self) -> Vec<&Tensor> {
        let mut params = Vec::new();
        params.extend(self.self_attn.parameters());
        params.extend(self.cross_attn.parameters());
        params.extend(self.norm1.parameters());
        params.extend(self.norm2.parameters());
        params.extend(self.norm3.parameters());
        params.extend(self.ff1.parameters());
        params.extend(self.ff2.parameters());
        params
    }

    fn parameters_mut(&mut self) -> Vec<&mut Tensor> {
        let mut params = Vec::new();
        params.extend(self.self_attn.parameters_mut());
        params.extend(self.cross_attn.parameters_mut());
        params.extend(self.norm1.parameters_mut());
        params.extend(self.norm2.parameters_mut());
        params.extend(self.norm3.parameters_mut());
        params.extend(self.ff1.parameters_mut());
        params.extend(self.ff2.parameters_mut());
        params
    }

    fn set_training(&mut self, training: bool) {
        self.training = training;
        self.self_attn.set_training(training);
        self.cross_attn.set_training(training);
        self.norm1.set_training(training);
        self.norm2.set_training(training);
        self.norm3.set_training(training);
        self.ff1.set_training(training);
        self.ff2.set_training(training);
        self.dropout.set_training(training);
    }

    fn is_training(&self) -> bool {
        self.training
    }

    fn name(&self) -> &str {
        "TransformerDecoderLayer"
    }
}

// ── TransformerDecoder ─────────────────────────────────────────

/// Stack of N `TransformerDecoderLayer`s followed by a final LayerNorm.
pub struct TransformerDecoder {
    pub layers: Vec<TransformerDecoderLayer>,
    pub final_norm: LayerNorm,
    training: bool,
}

impl TransformerDecoder {
    pub fn new(config: &TransformerDecoderConfig) -> Self {
        let layers = (0..config.n_layers)
            .map(|_| TransformerDecoderLayer::new(config))
            .collect();
        TransformerDecoder {
            layers,
            final_norm: LayerNorm::new(&[config.d_model]).unwrap(),
            training: true,
        }
    }

    /// Forward pass with encoder memory.
    ///
    /// * `tgt` – decoder input `[B, Tgt_S, D]`
    /// * `memory` – encoder output `[B, Src_S, D]`
    /// * `causal` – apply causal mask to self-attention
    ///
    /// Returns `[B, Tgt_S, D]`.
    pub fn forward_with_memory(&self, tgt: &Tensor, memory: &Tensor, causal: bool) -> Tensor {
        let mut x = tgt.clone();
        for layer in &self.layers {
            x = layer.forward_with_memory(&x, memory, causal);
        }
        self.final_norm.forward(&x)
    }
}

impl Module for TransformerDecoder {
    fn forward(&self, input: &Tensor) -> Tensor {
        let mut x = input.clone();
        for layer in &self.layers {
            x = layer.forward(&x);
        }
        self.final_norm.forward(&x)
    }

    fn parameters(&self) -> Vec<&Tensor> {
        let mut params = Vec::new();
        for layer in &self.layers {
            params.extend(layer.parameters());
        }
        params.extend(self.final_norm.parameters());
        params
    }

    fn parameters_mut(&mut self) -> Vec<&mut Tensor> {
        let mut params = Vec::new();
        for layer in &mut self.layers {
            params.extend(layer.parameters_mut());
        }
        params.extend(self.final_norm.parameters_mut());
        params
    }

    fn set_training(&mut self, training: bool) {
        self.training = training;
        for layer in &mut self.layers {
            layer.set_training(training);
        }
        self.final_norm.set_training(training);
    }

    fn is_training(&self) -> bool {
        self.training
    }

    fn name(&self) -> &str {
        "TransformerDecoder"
    }
}

// ── Tests ──────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_tensor(shape: &[usize], seed: u32) -> Tensor {
        let n: usize = shape.iter().product();
        let mut state = seed.wrapping_mul(2654435761);
        let data: Vec<f32> = (0..n)
            .map(|_| {
                state = state.wrapping_mul(1664525).wrapping_add(1013904223);
                (state as f32 / u32::MAX as f32) - 0.5
            })
            .collect();
        Tensor::new(data, shape.to_vec())
    }

    fn encoder_config(
        d_model: usize,
        n_heads: usize,
        d_ff: usize,
        n_layers: usize,
    ) -> TransformerEncoderConfig {
        TransformerEncoderConfig {
            d_model,
            n_heads,
            d_ff,
            n_layers,
            dropout: 0.0,
            activation: Activation::ReLU,
        }
    }

    fn decoder_config(
        d_model: usize,
        n_heads: usize,
        d_ff: usize,
        n_layers: usize,
    ) -> TransformerDecoderConfig {
        TransformerDecoderConfig {
            d_model,
            n_heads,
            d_ff,
            n_layers,
            dropout: 0.0,
            activation: Activation::ReLU,
        }
    }

    // ── Output shapes ─────────────────────────────────────────

    #[test]
    fn test_encoder_output_shape() {
        let config = encoder_config(32, 4, 64, 2);
        let mut enc = TransformerEncoder::new(&config);
        enc.set_training(false);
        let input = make_tensor(&[2, 8, 32], 1);
        let output = enc.forward(&input);
        assert_eq!(output.shape, vec![2, 8, 32]);
    }

    #[test]
    fn test_encoder_layer_output_shape() {
        let config = encoder_config(64, 8, 128, 1);
        let mut layer = TransformerEncoderLayer::new(&config);
        layer.set_training(false);
        let input = make_tensor(&[1, 4, 64], 42);
        let output = layer.forward(&input);
        assert_eq!(output.shape, vec![1, 4, 64]);
    }

    #[test]
    fn test_decoder_output_shape() {
        let config = decoder_config(32, 4, 64, 2);
        let mut dec = TransformerDecoder::new(&config);
        dec.set_training(false);
        let tgt = make_tensor(&[2, 6, 32], 1);
        let memory = make_tensor(&[2, 10, 32], 2);
        let output = dec.forward_with_memory(&tgt, &memory, true);
        assert_eq!(output.shape, vec![2, 6, 32]);
    }

    #[test]
    fn test_decoder_layer_output_shape() {
        let config = decoder_config(64, 8, 128, 1);
        let mut layer = TransformerDecoderLayer::new(&config);
        layer.set_training(false);
        let tgt = make_tensor(&[1, 5, 64], 1);
        let memory = make_tensor(&[1, 10, 64], 2);
        let output = layer.forward_with_memory(&tgt, &memory, true);
        assert_eq!(output.shape, vec![1, 5, 64]);
    }

    // ── Parameter counts ──────────────────────────────────────

    #[test]
    fn test_encoder_layer_param_count() {
        let d = 64usize;
        let d_ff = 256usize;
        let config = encoder_config(d, 4, d_ff, 1);
        let layer = TransformerEncoderLayer::new(&config);
        let total: usize = layer.parameters().iter().map(|p| p.numel()).sum();
        // MHA: 4*D*D + 4*D
        // norm1: D + D, norm2: D + D
        // ff1: D*Dff + Dff, ff2: Dff*D + D
        let expected = 4 * d * d + 4 * d + 2 * d + 2 * d + d * d_ff + d_ff + d_ff * d + d;
        assert_eq!(
            total, expected,
            "encoder layer param count mismatch: got {} expected {}",
            total, expected
        );
    }

    #[test]
    fn test_decoder_layer_param_count() {
        let d = 64usize;
        let d_ff = 256usize;
        let config = decoder_config(d, 4, d_ff, 1);
        let layer = TransformerDecoderLayer::new(&config);
        let total: usize = layer.parameters().iter().map(|p| p.numel()).sum();
        // 2x MHA: 2*(4*D*D + 4*D)
        // norm1,2,3: 3*(D + D)
        // ff1: D*Dff + Dff, ff2: Dff*D + D
        let expected = 2 * (4 * d * d + 4 * d) + 3 * 2 * d + d * d_ff + d_ff + d_ff * d + d;
        assert_eq!(
            total, expected,
            "decoder layer param count mismatch: got {} expected {}",
            total, expected
        );
    }

    #[test]
    fn test_encoder_total_param_count() {
        let d = 32usize;
        let d_ff = 64usize;
        let n_layers = 3;
        let config = encoder_config(d, 4, d_ff, n_layers);
        let enc = TransformerEncoder::new(&config);
        let total: usize = enc.parameters().iter().map(|p| p.numel()).sum();
        let per_layer = 4 * d * d + 4 * d + 2 * d + 2 * d + d * d_ff + d_ff + d_ff * d + d;
        let final_norm = 2 * d;
        assert_eq!(total, n_layers * per_layer + final_norm);
    }

    // ── Training mode propagation ─────────────────────────────

    #[test]
    fn test_encoder_training_mode_propagates() {
        let config = encoder_config(32, 4, 64, 2);
        let mut enc = TransformerEncoder::new(&config);
        assert!(enc.is_training());

        enc.set_training(false);
        assert!(!enc.is_training());
        for layer in &enc.layers {
            assert!(!layer.is_training());
            assert!(!layer.self_attn.is_training());
            assert!(!layer.norm1.is_training());
            assert!(!layer.norm2.is_training());
            assert!(!layer.ff1.is_training());
            assert!(!layer.ff2.is_training());
            assert!(!layer.dropout.is_training());
        }
        assert!(!enc.final_norm.is_training());

        enc.set_training(true);
        assert!(enc.is_training());
        for layer in &enc.layers {
            assert!(layer.is_training());
        }
    }

    #[test]
    fn test_decoder_training_mode_propagates() {
        let config = decoder_config(32, 4, 64, 2);
        let mut dec = TransformerDecoder::new(&config);
        assert!(dec.is_training());

        dec.set_training(false);
        assert!(!dec.is_training());
        for layer in &dec.layers {
            assert!(!layer.is_training());
            assert!(!layer.self_attn.is_training());
            assert!(!layer.cross_attn.is_training());
            assert!(!layer.norm1.is_training());
            assert!(!layer.norm2.is_training());
            assert!(!layer.norm3.is_training());
            assert!(!layer.ff1.is_training());
            assert!(!layer.ff2.is_training());
            assert!(!layer.dropout.is_training());
        }
        assert!(!dec.final_norm.is_training());
    }

    // ── 4-layer forward + gradient connectivity ───────────────

    #[test]
    fn test_encoder_4_layers_forward_and_gradient() {
        let d = 16;
        let d_ff = 32;
        let config = encoder_config(d, 2, d_ff, 4);
        let mut enc = TransformerEncoder::new(&config);
        enc.set_training(false);

        let input = make_tensor(&[1, 4, d], 1);
        let base_out = enc.forward(&input);
        assert_eq!(base_out.shape, vec![1, 4, d]);

        // Verify gradient connectivity: perturb each parameter and check output changes
        let eps = 0.1;
        let n_params = enc.parameters().len();
        for pi in 0..n_params {
            // Perturb parameter pi
            {
                let param = &mut enc.parameters_mut()[pi];
                for val in param.data.iter_mut() {
                    *val += eps;
                }
            }
            let perturbed_out = enc.forward(&input);
            let max_diff: f32 = base_out
                .data
                .iter()
                .zip(&perturbed_out.data)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max);
            assert!(
                max_diff > 1e-7,
                "encoder: parameter {} has no gradient connectivity (max_diff={})",
                pi,
                max_diff
            );
            // Restore
            {
                let param = &mut enc.parameters_mut()[pi];
                for val in param.data.iter_mut() {
                    *val -= eps;
                }
            }
        }
    }

    #[test]
    fn test_decoder_4_layers_forward_and_gradient() {
        let d = 16;
        let d_ff = 32;
        let config = decoder_config(d, 2, d_ff, 4);
        let mut dec = TransformerDecoder::new(&config);
        dec.set_training(false);

        let tgt = make_tensor(&[1, 4, d], 1);
        let memory = make_tensor(&[1, 6, d], 2);
        let base_out = dec.forward_with_memory(&tgt, &memory, true);
        assert_eq!(base_out.shape, vec![1, 4, d]);

        let eps = 0.1;
        let n_params = dec.parameters().len();
        for pi in 0..n_params {
            {
                let param = &mut dec.parameters_mut()[pi];
                for val in param.data.iter_mut() {
                    *val += eps;
                }
            }
            let perturbed_out = dec.forward_with_memory(&tgt, &memory, true);
            let max_diff: f32 = base_out
                .data
                .iter()
                .zip(&perturbed_out.data)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max);
            assert!(
                max_diff > 1e-7,
                "decoder: parameter {} has no gradient connectivity (max_diff={})",
                pi,
                max_diff
            );
            {
                let param = &mut dec.parameters_mut()[pi];
                for val in param.data.iter_mut() {
                    *val -= eps;
                }
            }
        }
    }

    // ── Residual path verification ────────────────────────────

    #[test]
    fn test_encoder_layer_residual_path() {
        // With zero-initialized attention and FFN weights, residual path gives identity.
        let config = encoder_config(16, 2, 32, 1);
        let mut layer = TransformerEncoderLayer::new(&config);
        layer.set_training(false);

        // Zero all MHA weights and biases
        zero_mha(&mut layer.self_attn);
        // Zero FFN weights and biases
        zero_linear(&mut layer.ff1);
        zero_linear(&mut layer.ff2);

        let input = make_tensor(&[2, 4, 16], 42);
        let output = layer.forward(&input);

        // Output should equal input (residual connections pass through)
        for (i, (a, b)) in input.data.iter().zip(&output.data).enumerate() {
            assert!(
                (a - b).abs() < 1e-5,
                "encoder residual: mismatch at index {} (input={}, output={})",
                i,
                a,
                b
            );
        }
    }

    #[test]
    fn test_decoder_layer_residual_path() {
        let config = decoder_config(16, 2, 32, 1);
        let mut layer = TransformerDecoderLayer::new(&config);
        layer.set_training(false);

        zero_mha(&mut layer.self_attn);
        zero_mha(&mut layer.cross_attn);
        zero_linear(&mut layer.ff1);
        zero_linear(&mut layer.ff2);

        let tgt = make_tensor(&[2, 4, 16], 42);
        let memory = make_tensor(&[2, 6, 16], 99);
        let output = layer.forward_with_memory(&tgt, &memory, true);

        for (i, (a, b)) in tgt.data.iter().zip(&output.data).enumerate() {
            assert!(
                (a - b).abs() < 1e-5,
                "decoder residual: mismatch at index {} (input={}, output={})",
                i,
                a,
                b
            );
        }
    }

    // ── Activation variants ───────────────────────────────────

    #[test]
    fn test_encoder_gelu_activation() {
        let config = TransformerEncoderConfig {
            d_model: 16,
            n_heads: 2,
            d_ff: 32,
            n_layers: 1,
            dropout: 0.0,
            activation: Activation::GELU,
        };
        let mut enc = TransformerEncoder::new(&config);
        enc.set_training(false);
        let input = make_tensor(&[1, 4, 16], 1);
        let output = enc.forward(&input);
        assert_eq!(output.shape, vec![1, 4, 16]);
    }

    #[test]
    fn test_decoder_gelu_activation() {
        let config = TransformerDecoderConfig {
            d_model: 16,
            n_heads: 2,
            d_ff: 32,
            n_layers: 1,
            dropout: 0.0,
            activation: Activation::GELU,
        };
        let mut dec = TransformerDecoder::new(&config);
        dec.set_training(false);
        let tgt = make_tensor(&[1, 4, 16], 1);
        let memory = make_tensor(&[1, 6, 16], 2);
        let output = dec.forward_with_memory(&tgt, &memory, true);
        assert_eq!(output.shape, vec![1, 4, 16]);
    }

    // ── Module trait ──────────────────────────────────────────

    #[test]
    fn test_module_names() {
        let enc_config = encoder_config(16, 2, 32, 1);
        let dec_config = decoder_config(16, 2, 32, 1);

        let enc_layer = TransformerEncoderLayer::new(&enc_config);
        assert_eq!(enc_layer.name(), "TransformerEncoderLayer");

        let enc = TransformerEncoder::new(&enc_config);
        assert_eq!(enc.name(), "TransformerEncoder");

        let dec_layer = TransformerDecoderLayer::new(&dec_config);
        assert_eq!(dec_layer.name(), "TransformerDecoderLayer");

        let dec = TransformerDecoder::new(&dec_config);
        assert_eq!(dec.name(), "TransformerDecoder");
    }

    #[test]
    fn test_encoder_as_trait_object() {
        let config = encoder_config(16, 2, 32, 1);
        let mut enc: Box<dyn Module> = Box::new(TransformerEncoder::new(&config));
        enc.set_training(false);
        let input = make_tensor(&[1, 4, 16], 1);
        let output = enc.forward(&input);
        assert_eq!(output.shape, vec![1, 4, 16]);
    }

    // ── Inference determinism ─────────────────────────────────

    #[test]
    fn test_encoder_inference_deterministic() {
        let config = encoder_config(16, 2, 32, 1);
        let mut enc = TransformerEncoder::new(&config);
        enc.set_training(false);
        let input = make_tensor(&[1, 4, 16], 1);
        let out1 = enc.forward(&input);
        let out2 = enc.forward(&input);
        assert_eq!(out1.data, out2.data, "inference should be deterministic");
    }

    #[test]
    fn test_decoder_inference_deterministic() {
        let config = decoder_config(16, 2, 32, 1);
        let mut dec = TransformerDecoder::new(&config);
        dec.set_training(false);
        let tgt = make_tensor(&[1, 4, 16], 1);
        let memory = make_tensor(&[1, 6, 16], 2);
        let out1 = dec.forward_with_memory(&tgt, &memory, true);
        let out2 = dec.forward_with_memory(&tgt, &memory, true);
        assert_eq!(out1.data, out2.data, "inference should be deterministic");
    }

    // ── Helpers ───────────────────────────────────────────────

    fn zero_mha(mha: &mut MultiHeadAttention) {
        for val in mha.w_q.weight.data.iter_mut() {
            *val = 0.0;
        }
        if let Some(ref mut b) = mha.w_q.bias {
            for val in b.data.iter_mut() {
                *val = 0.0;
            }
        }
        for val in mha.w_k.weight.data.iter_mut() {
            *val = 0.0;
        }
        if let Some(ref mut b) = mha.w_k.bias {
            for val in b.data.iter_mut() {
                *val = 0.0;
            }
        }
        for val in mha.w_v.weight.data.iter_mut() {
            *val = 0.0;
        }
        if let Some(ref mut b) = mha.w_v.bias {
            for val in b.data.iter_mut() {
                *val = 0.0;
            }
        }
        for val in mha.w_o.weight.data.iter_mut() {
            *val = 0.0;
        }
        if let Some(ref mut b) = mha.w_o.bias {
            for val in b.data.iter_mut() {
                *val = 0.0;
            }
        }
    }

    fn zero_linear(linear: &mut Linear) {
        for val in linear.weight.data.iter_mut() {
            *val = 0.0;
        }
        if let Some(ref mut b) = linear.bias {
            for val in b.data.iter_mut() {
                *val = 0.0;
            }
        }
    }
}
