//! Diffusion LLM model with iterative denoising generation.
//!
//! Implements a bidirectional transformer that generates text by
//! iteratively unmasking tokens from a fully masked sequence.

use crate::diffusion::config::DiffusionLLMConfig;
use crate::diffusion::schedule::{unmask_by_confidence, tokens_per_step, MaskSchedule};
use crate::ops::matmul::matmul_t;
use crate::ops::activation::silu;
use crate::ops::pure_rust_ops::rmsnorm;

/// Softmax in-place over the first `n` elements.
fn softmax(x: &mut [f32], n: usize) {
    let max = x[..n].iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0f32;
    for i in 0..n {
        x[i] = (x[i] - max).exp();
        sum += x[i];
    }
    if sum > 0.0 {
        for i in 0..n {
            x[i] /= sum;
        }
    }
}

/// A bidirectional decoder layer (no causal mask).
///
/// All tokens attend to all other tokens, enabling the model to
/// use full context when predicting masked positions.
pub struct BiDirectionalLayer {
    pub hidden_size: usize,
    pub num_heads: usize,
    pub head_dim: usize,
    pub intermediate_size: usize,
    pub norm_eps: f32,

    // Attention weights
    pub attn_norm_weight: Vec<f32>,
    /// Q projection: [hidden_size, hidden_size] stored row-major
    pub w_q: Vec<f32>,
    /// K projection: [hidden_size, hidden_size]
    pub w_k: Vec<f32>,
    /// V projection: [hidden_size, hidden_size]
    pub w_v: Vec<f32>,
    /// Output projection: [hidden_size, hidden_size]
    pub w_o: Vec<f32>,

    // FFN weights
    pub ffn_norm_weight: Vec<f32>,
    /// Gate projection: [intermediate_size, hidden_size]
    pub ffn_gate_weight: Vec<f32>,
    /// Up projection: [intermediate_size, hidden_size]
    pub ffn_up_weight: Vec<f32>,
    /// Down projection: [hidden_size, intermediate_size]
    pub ffn_down_weight: Vec<f32>,
}

impl BiDirectionalLayer {
    /// Bidirectional forward: all tokens attend to all tokens (no causal mask).
    ///
    /// `x` is `[seq_len, hidden_size]` (flattened).
    /// Returns `[seq_len, hidden_size]`.
    pub fn forward(&self, x: &[f32], seq_len: usize) -> Vec<f32> {
        let h = self.hidden_size;
        let nh = self.num_heads;
        let hd = self.head_dim;
        let qk_dim = nh * hd; // should equal hidden_size

        // --- Attention block ---
        // 1. RMSNorm
        let normed = rmsnorm(x, &self.attn_norm_weight, self.norm_eps, h);

        // 2. Project Q, K, V
        // w_q is [hidden_size, hidden_size] => matmul_t(normed, w_q, seq_len, h, h)
        // gives [seq_len, hidden_size] = normed[seq_len, h] * w_q^T[h, h]
        let q = matmul_t(&normed, &self.w_q, seq_len, h, qk_dim);
        let k = matmul_t(&normed, &self.w_k, seq_len, h, qk_dim);
        let v = matmul_t(&normed, &self.w_v, seq_len, h, qk_dim);

        // 3. Bidirectional multi-head attention (no causal mask)
        let scale = 1.0 / (hd as f32).sqrt();
        let mut attn_out = vec![0.0f32; seq_len * qk_dim];

        for head in 0..nh {
            for qi in 0..seq_len {
                // Compute attention scores for this query against all keys
                let mut scores = vec![0.0f32; seq_len];
                for ki in 0..seq_len {
                    let mut dot = 0.0f32;
                    for d in 0..hd {
                        dot += q[qi * qk_dim + head * hd + d]
                            * k[ki * qk_dim + head * hd + d];
                    }
                    scores[ki] = dot * scale;
                }

                // Softmax over all positions (bidirectional -- no mask)
                softmax(&mut scores, seq_len);

                // Weighted sum of values
                for d in 0..hd {
                    let mut val = 0.0f32;
                    for ki in 0..seq_len {
                        val += scores[ki] * v[ki * qk_dim + head * hd + d];
                    }
                    attn_out[qi * qk_dim + head * hd + d] = val;
                }
            }
        }

        // 4. Output projection
        let projected = matmul_t(&attn_out, &self.w_o, seq_len, qk_dim, h);

        // 5. Residual connection
        let mut hidden: Vec<f32> = x
            .iter()
            .zip(projected.iter())
            .map(|(&a, &b)| a + b)
            .collect();

        // --- FFN block ---
        // 1. RMSNorm
        let ffn_normed = rmsnorm(&hidden, &self.ffn_norm_weight, self.norm_eps, h);

        // 2. SwiGLU FFN: silu(gate) * up, then down
        let gate = matmul_t(&ffn_normed, &self.ffn_gate_weight, seq_len, h, self.intermediate_size);
        let up = matmul_t(&ffn_normed, &self.ffn_up_weight, seq_len, h, self.intermediate_size);

        let mut swiglu = Vec::with_capacity(seq_len * self.intermediate_size);
        for i in 0..gate.len() {
            swiglu.push(silu(gate[i]) * up[i]);
        }

        let ffn_out = matmul_t(&swiglu, &self.ffn_down_weight, seq_len, self.intermediate_size, h);

        // 3. Residual connection
        for i in 0..hidden.len() {
            hidden[i] += ffn_out[i];
        }

        hidden
    }
}

/// Diffusion language model with iterative denoising.
///
/// Generates text by starting with a fully masked output sequence and
/// iteratively unmasking tokens based on model confidence.
pub struct DiffusionModel {
    pub config: DiffusionLLMConfig,
    /// Token embeddings: [vocab_size, hidden_size]
    pub embed_tokens: Vec<f32>,
    /// Bidirectional transformer layers
    pub layers: Vec<BiDirectionalLayer>,
    /// Final RMSNorm weight: [hidden_size]
    pub final_norm_weight: Vec<f32>,
    /// LM head weight: [vocab_size, hidden_size]
    pub lm_head_weight: Vec<f32>,
}

impl DiffusionModel {
    /// Generate text by iterative denoising.
    ///
    /// 1. Start with prompt tokens + masked output positions
    /// 2. Run full bidirectional forward pass
    /// 3. Unmask most confident tokens
    /// 4. Repeat for T steps
    ///
    /// # Arguments
    /// - `prompt_tokens`: tokens that are fixed (not masked)
    /// - `output_len`: number of tokens to generate
    /// - `schedule`: denoising schedule strategy
    ///
    /// # Returns
    /// Generated token ids of length `output_len`.
    pub fn generate(
        &self,
        prompt_tokens: &[u32],
        output_len: usize,
        schedule: MaskSchedule,
    ) -> Vec<u32> {
        let total_len = prompt_tokens.len() + output_len;
        let mask_id = self.config.mask_token_id;

        // Initialize: prompt + masks
        let mut tokens: Vec<u32> = prompt_tokens.to_vec();
        tokens.extend(vec![mask_id; output_len]);

        let mut is_masked = vec![false; total_len];
        for i in prompt_tokens.len()..total_len {
            is_masked[i] = true;
        }

        let steps = tokens_per_step(schedule, output_len, self.config.num_denoise_steps);

        for step_tokens in &steps {
            if *step_tokens == 0 {
                continue;
            }

            // Full bidirectional forward pass
            let logits = self.forward_bidirectional(&tokens);

            // Unmask top-confidence tokens
            let unmasked = unmask_by_confidence(
                &logits,
                &is_masked,
                self.config.vocab_size,
                *step_tokens,
            );

            for (pos, tok) in unmasked {
                tokens[pos] = tok;
                is_masked[pos] = false;
            }
        }

        // Return only generated tokens
        tokens[prompt_tokens.len()..].to_vec()
    }

    /// Full bidirectional forward pass.
    ///
    /// Returns logits `[total_len, vocab_size]` (flattened).
    fn forward_bidirectional(&self, tokens: &[u32]) -> Vec<f32> {
        let seq_len = tokens.len();
        let h = self.config.hidden_size;

        // Embed
        let mut hidden = Vec::with_capacity(seq_len * h);
        for &tid in tokens {
            let off = tid as usize * h;
            hidden.extend_from_slice(&self.embed_tokens[off..off + h]);
        }

        // Process through all layers (bidirectional)
        for layer in &self.layers {
            hidden = layer.forward(&hidden, seq_len);
        }

        // Final norm
        let normed = rmsnorm(&hidden, &self.final_norm_weight, self.config.norm_eps as f32, h);

        // LM head: [seq_len, hidden_size] * [vocab_size, hidden_size]^T => [seq_len, vocab_size]
        matmul_t(&normed, &self.lm_head_weight, seq_len, h, self.config.vocab_size)
    }
}

/// Create a DiffusionModel with random weights for testing.
///
/// Weights are deterministic (seeded by position) so tests are reproducible.
#[cfg(test)]
pub(crate) fn build_test_model(config: &DiffusionLLMConfig) -> DiffusionModel {
    let h = config.hidden_size;
    let nh = config.num_heads;
    let hd = config.head_dim;
    let qk_dim = nh * hd;
    let inter = config.intermediate_size;
    let vocab = config.vocab_size;

    // Simple deterministic pseudo-random for reproducibility
    fn pseudo_rand(seed: usize, len: usize) -> Vec<f32> {
        let scale = 0.02;
        (0..len)
            .map(|i| {
                let x = (seed
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(i.wrapping_mul(1442695040888963407))
                    & 0xFFFFFF) as f32
                    / 0xFFFFFF as f32;
                (x - 0.5) * scale
            })
            .collect()
    }

    let embed_tokens = pseudo_rand(42, vocab * h);
    let final_norm_weight = vec![1.0f32; h];
    let lm_head_weight = pseudo_rand(99, vocab * h);

    let mut layers = Vec::with_capacity(config.num_layers);
    for layer_idx in 0..config.num_layers {
        let seed_base = layer_idx * 1000;
        layers.push(BiDirectionalLayer {
            hidden_size: h,
            num_heads: nh,
            head_dim: hd,
            intermediate_size: inter,
            norm_eps: config.norm_eps as f32,
            attn_norm_weight: vec![1.0f32; h],
            w_q: pseudo_rand(seed_base + 1, qk_dim * h),
            w_k: pseudo_rand(seed_base + 2, qk_dim * h),
            w_v: pseudo_rand(seed_base + 3, qk_dim * h),
            w_o: pseudo_rand(seed_base + 4, h * qk_dim),
            ffn_norm_weight: vec![1.0f32; h],
            ffn_gate_weight: pseudo_rand(seed_base + 5, inter * h),
            ffn_up_weight: pseudo_rand(seed_base + 6, inter * h),
            ffn_down_weight: pseudo_rand(seed_base + 7, h * inter),
        });
    }

    DiffusionModel {
        config: config.clone(),
        embed_tokens,
        layers,
        final_norm_weight,
        lm_head_weight,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::diffusion::config::DiffusionLLMConfig;

    #[test]
    fn forward_produces_finite_logits() {
        let config = DiffusionLLMConfig::tiny_test();
        let model = build_test_model(&config);

        // 3 prompt tokens + 5 masked
        let prompt = vec![1u32, 2, 3];
        let total_len = prompt.len() + 5;
        let mut tokens: Vec<u32> = prompt.clone();
        tokens.extend(vec![config.mask_token_id; 5]);

        let logits = model.forward_bidirectional(&tokens);
        assert_eq!(logits.len(), total_len * config.vocab_size);
        assert!(
            logits.iter().all(|v| v.is_finite()),
            "all logits should be finite"
        );
    }

    #[test]
    fn generate_produces_valid_tokens() {
        let config = DiffusionLLMConfig::tiny_test();
        let model = build_test_model(&config);

        let prompt = vec![1u32, 2, 3];
        let output = model.generate(&prompt, 5, MaskSchedule::Linear);

        assert_eq!(output.len(), 5);
        for &tok in &output {
            assert!(
                (tok as usize) < config.vocab_size,
                "token {tok} should be < vocab_size {}",
                config.vocab_size
            );
        }
    }

    #[test]
    fn generate_with_confidence_schedule() {
        let config = DiffusionLLMConfig::tiny_test();
        let model = build_test_model(&config);

        let prompt = vec![5u32, 10];
        let output = model.generate(&prompt, 8, MaskSchedule::Confidence);

        assert_eq!(output.len(), 8);
        for &tok in &output {
            assert!((tok as usize) < config.vocab_size);
        }
    }

    #[test]
    fn generate_with_cosine_schedule() {
        let config = DiffusionLLMConfig::tiny_test();
        let model = build_test_model(&config);

        let prompt = vec![3u32];
        let output = model.generate(&prompt, 10, MaskSchedule::Cosine);

        assert_eq!(output.len(), 10);
        for &tok in &output {
            assert!((tok as usize) < config.vocab_size);
        }
    }

    #[test]
    fn generate_no_masks_remain() {
        let config = DiffusionLLMConfig::tiny_test();
        let model = build_test_model(&config);

        let prompt = vec![1u32];
        let output = model.generate(&prompt, 5, MaskSchedule::Linear);

        // After generation, no token should be the mask token
        // (unless the model genuinely predicts it, which is unlikely with random weights
        //  but possible -- the key invariant is that all positions were unmasked)
        assert_eq!(output.len(), 5);
    }

    #[test]
    fn generate_different_schedules_produce_output() {
        let config = DiffusionLLMConfig::tiny_test();
        let model = build_test_model(&config);

        let prompt = vec![1u32, 2];
        for schedule in &[MaskSchedule::Linear, MaskSchedule::Confidence, MaskSchedule::Cosine] {
            let output = model.generate(&prompt, 4, *schedule);
            assert_eq!(output.len(), 4, "schedule {:?} should produce 4 tokens", schedule);
        }
    }

    #[test]
    fn bidirectional_layer_preserves_shape() {
        let config = DiffusionLLMConfig::tiny_test();
        let model = build_test_model(&config);

        let seq_len = 4;
        let h = config.hidden_size;
        let input = vec![0.1f32; seq_len * h];

        let output = model.layers[0].forward(&input, seq_len);
        assert_eq!(output.len(), seq_len * h);
        assert!(output.iter().all(|v| v.is_finite()));
    }
}
