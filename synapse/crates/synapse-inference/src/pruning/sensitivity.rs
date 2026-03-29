//! Sensitivity analysis: measure layer importance by output divergence.
//!
//! For each layer, we compare the model's output with that layer active vs.
//! skipped. Layers whose removal causes minimal divergence are candidates for
//! pruning or removal.

use crate::model::causal_lm::ModelOutput;
use crate::ops::matmul::matmul_t;
use crate::ops::pure_rust_ops::rmsnorm;

/// Importance score for a single layer.
#[derive(Debug, Clone)]
pub struct LayerImportance {
    pub layer_idx: usize,
    /// Cosine similarity between full-model logits and skip-this-layer logits.
    /// Values near 1.0 mean the layer is redundant.
    pub cosine_similarity: f32,
    /// KL divergence of softmax distributions (full vs skip).
    pub kl_divergence: f32,
    /// Whether this layer is safe to remove (cos > threshold).
    pub removable: bool,
}

impl LayerImportance {
    /// The divergence score: lower = more redundant = safer to remove.
    pub fn divergence(&self) -> f32 {
        1.0 - self.cosine_similarity
    }
}

/// Sensitivity analyzer for Mamba models.
///
/// Runs forward passes with each layer individually skipped to measure
/// how much each layer contributes to the final output.
pub struct SensitivityAnalyzer {
    /// Cosine similarity threshold: layers above this are considered removable.
    pub removable_threshold: f32,
}

impl Default for SensitivityAnalyzer {
    fn default() -> Self {
        Self {
            removable_threshold: 0.99,
        }
    }
}

impl SensitivityAnalyzer {
    pub fn new(removable_threshold: f32) -> Self {
        Self { removable_threshold }
    }

    /// Analyze a Mamba model's layer sensitivity.
    ///
    /// Returns importance scores sorted by divergence (most redundant first).
    pub fn analyze_mamba(
        &self,
        config: &crate::ssm::config::MambaConfig,
        embed_tokens: &[f32],
        blocks: &[crate::ssm::mamba_block::MambaBlock],
        final_norm_weight: &[f32],
        lm_head_weight: &[f32],
        calibration_tokens: &[u32],
    ) -> Vec<LayerImportance> {
        let _d_model = config.d_model;
        let _vocab = config.vocab_size;
        let _norm_eps = config.norm_eps as f32;

        // Get baseline logits (all layers active)
        let baseline = self.mamba_forward_skip(
            config, embed_tokens, blocks, final_norm_weight, lm_head_weight,
            calibration_tokens, None,
        );

        let num_layers = blocks.len();
        let mut importances = Vec::with_capacity(num_layers);

        for skip_idx in 0..num_layers {
            let skipped = self.mamba_forward_skip(
                config, embed_tokens, blocks, final_norm_weight, lm_head_weight,
                calibration_tokens, Some(skip_idx),
            );

            let cos = cosine_similarity(&baseline.logits, &skipped.logits);
            let kl = kl_divergence_softmax(&baseline.logits, &skipped.logits);

            importances.push(LayerImportance {
                layer_idx: skip_idx,
                cosine_similarity: cos,
                kl_divergence: kl,
                removable: cos > self.removable_threshold,
            });
        }

        // Sort by cosine similarity descending (most redundant first)
        importances.sort_by(|a, b| b.cosine_similarity.partial_cmp(&a.cosine_similarity).unwrap());
        importances
    }

    /// Analyze an RWKV model's layer sensitivity.
    pub fn analyze_rwkv(
        &self,
        config: &crate::ssm::rwkv_config::RwkvConfig,
        embed_tokens: &[f32],
        pre_ln_weight: Option<&[f32]>,
        pre_ln_bias: Option<&[f32]>,
        blocks: &[crate::ssm::rwkv_block::RwkvBlock],
        final_norm_weight: &[f32],
        final_norm_bias: &[f32],
        lm_head_weight: &[f32],
        calibration_tokens: &[u32],
    ) -> Vec<LayerImportance> {
        // Get baseline logits
        let baseline = self.rwkv_forward_skip(
            config, embed_tokens, pre_ln_weight, pre_ln_bias,
            blocks, final_norm_weight, final_norm_bias, lm_head_weight,
            calibration_tokens, None,
        );

        let num_layers = blocks.len();
        let mut importances = Vec::with_capacity(num_layers);

        for skip_idx in 0..num_layers {
            let skipped = self.rwkv_forward_skip(
                config, embed_tokens, pre_ln_weight, pre_ln_bias,
                blocks, final_norm_weight, final_norm_bias, lm_head_weight,
                calibration_tokens, Some(skip_idx),
            );

            let cos = cosine_similarity(&baseline.logits, &skipped.logits);
            let kl = kl_divergence_softmax(&baseline.logits, &skipped.logits);

            importances.push(LayerImportance {
                layer_idx: skip_idx,
                cosine_similarity: cos,
                kl_divergence: kl,
                removable: cos > self.removable_threshold,
            });
        }

        importances.sort_by(|a, b| b.cosine_similarity.partial_cmp(&a.cosine_similarity).unwrap());
        importances
    }

    /// Forward pass through Mamba model, optionally skipping one layer.
    fn mamba_forward_skip(
        &self,
        config: &crate::ssm::config::MambaConfig,
        embed_tokens: &[f32],
        blocks: &[crate::ssm::mamba_block::MambaBlock],
        final_norm_weight: &[f32],
        lm_head_weight: &[f32],
        token_ids: &[u32],
        skip_layer: Option<usize>,
    ) -> ModelOutput {
        let d_model = config.d_model;
        let vocab = config.vocab_size;
        let norm_eps = config.norm_eps as f32;

        // Create fresh state for each analysis run
        let mut state = crate::ssm::state::RecurrentState::new(
            config.num_layers,
            config.d_inner(),
            config.d_state,
            config.d_conv,
        );

        // Embed last token only (for efficiency, we only need final logits)
        let last_token = *token_ids.last().unwrap_or(&0);
        let id = last_token as usize;
        let mut hidden = vec![0.0f32; d_model];
        if id < vocab {
            hidden.copy_from_slice(&embed_tokens[id * d_model..(id + 1) * d_model]);
        }

        // Process through all tokens to build state, then get final hidden
        // For calibration, prefill all tokens
        let seq_len = token_ids.len();
        let mut all_hidden = vec![0.0f32; seq_len * d_model];
        for (t, &tid) in token_ids.iter().enumerate() {
            let tid = tid as usize;
            if tid < vocab {
                let src = &embed_tokens[tid * d_model..(tid + 1) * d_model];
                all_hidden[t * d_model..(t + 1) * d_model].copy_from_slice(src);
            }
        }

        for (i, block) in blocks.iter().enumerate() {
            if Some(i) == skip_layer {
                // Skip this layer: hidden passes through unchanged
                continue;
            }
            all_hidden = block.forward_seq(&all_hidden, seq_len, &mut state.layers[i]);
        }

        // Final norm + LM head on last token
        let last_hidden = &all_hidden[(seq_len - 1) * d_model..seq_len * d_model];
        let normed = rmsnorm(last_hidden, final_norm_weight, norm_eps, d_model);
        let logits = matmul_t(&normed, lm_head_weight, 1, d_model, vocab);

        ModelOutput {
            logits,
            shape: [1, 1, vocab],
        }
    }

    /// Forward pass through RWKV model, optionally skipping one layer.
    fn rwkv_forward_skip(
        &self,
        config: &crate::ssm::rwkv_config::RwkvConfig,
        embed_tokens: &[f32],
        pre_ln_weight: Option<&[f32]>,
        pre_ln_bias: Option<&[f32]>,
        blocks: &[crate::ssm::rwkv_block::RwkvBlock],
        final_norm_weight: &[f32],
        final_norm_bias: &[f32],
        lm_head_weight: &[f32],
        token_ids: &[u32],
        skip_layer: Option<usize>,
    ) -> ModelOutput {
        let hidden_size = config.hidden_size;
        let vocab = config.vocab_size;

        // Create fresh RWKV state
        let mut state = crate::ssm::rwkv_state::RwkvState::new(
            config.num_layers,
            config.hidden_size,
            config.num_heads,
            config.head_size,
        );

        // Embed tokens
        let seq_len = token_ids.len();
        let mut hidden = vec![0.0f32; seq_len * hidden_size];
        for (t, &tid) in token_ids.iter().enumerate() {
            let tid = tid as usize;
            if tid < vocab {
                let src = &embed_tokens[tid * hidden_size..(tid + 1) * hidden_size];
                hidden[t * hidden_size..(t + 1) * hidden_size].copy_from_slice(src);
            }
        }

        // Pre-LayerNorm if present
        if let (Some(w), Some(b)) = (pre_ln_weight, pre_ln_bias) {
            hidden = layernorm_simple(&hidden, w, b, hidden_size);
        }

        // Process through blocks, skipping specified layer
        let mut v_first: Option<Vec<f32>> = None;
        for (i, block) in blocks.iter().enumerate() {
            if Some(i) == skip_layer {
                continue;
            }
            let (new_hidden, v0_out) = block.forward_seq(
                &hidden, seq_len, &mut state.layers[i], v_first.as_deref(),
            );
            hidden = new_hidden;
            if v_first.is_none() {
                v_first = Some(v0_out);
            }
        }

        // Final LayerNorm + LM head on last token
        let last_hidden = &hidden[(seq_len - 1) * hidden_size..seq_len * hidden_size];
        let normed = layernorm_simple(
            last_hidden, final_norm_weight, final_norm_bias, hidden_size,
        );
        let logits = matmul_t(&normed, lm_head_weight, 1, hidden_size, vocab);

        ModelOutput {
            logits,
            shape: [1, 1, vocab],
        }
    }
}

// ── Utility functions ──────────────────────────────────────────────

/// Cosine similarity between two vectors.
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    let mut dot = 0.0f32;
    let mut norm_a = 0.0f32;
    let mut norm_b = 0.0f32;
    for i in 0..a.len() {
        dot += a[i] * b[i];
        norm_a += a[i] * a[i];
        norm_b += b[i] * b[i];
    }
    let denom = norm_a.sqrt() * norm_b.sqrt();
    if denom < 1e-12 {
        0.0
    } else {
        dot / denom
    }
}

/// KL divergence between two logit vectors (applies softmax internally).
pub fn kl_divergence_softmax(logits_p: &[f32], logits_q: &[f32]) -> f32 {
    assert_eq!(logits_p.len(), logits_q.len());
    let n = logits_p.len();
    if n == 0 {
        return 0.0;
    }

    // Numerically stable softmax
    let max_p = logits_p.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let max_q = logits_q.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

    let exp_p: Vec<f32> = logits_p.iter().map(|&x| (x - max_p).exp()).collect();
    let exp_q: Vec<f32> = logits_q.iter().map(|&x| (x - max_q).exp()).collect();
    let sum_p: f32 = exp_p.iter().sum();
    let sum_q: f32 = exp_q.iter().sum();

    let mut kl = 0.0f32;
    for i in 0..n {
        let p = exp_p[i] / sum_p;
        let q = (exp_q[i] / sum_q).max(1e-10);
        if p > 1e-10 {
            kl += p * (p / q).ln();
        }
    }
    kl
}

/// Simple LayerNorm (for RWKV analysis).
fn layernorm_simple(x: &[f32], weight: &[f32], bias: &[f32], hidden_size: usize) -> Vec<f32> {
    let n = x.len() / hidden_size;
    let mut out = vec![0.0f32; x.len()];
    for i in 0..n {
        let off = i * hidden_size;
        let row = &x[off..off + hidden_size];
        let mean: f32 = row.iter().sum::<f32>() / hidden_size as f32;
        let var: f32 = row.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / hidden_size as f32;
        let inv_std = 1.0 / (var + 1e-5_f32).sqrt();
        for j in 0..hidden_size {
            out[off + j] = (row[j] - mean) * inv_std * weight[j] + bias[j];
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cosine_similarity_identical() {
        let a = vec![1.0, 2.0, 3.0, 4.0];
        assert!((cosine_similarity(&a, &a) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_cosine_similarity_orthogonal() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![0.0, 1.0, 0.0];
        assert!(cosine_similarity(&a, &b).abs() < 1e-6);
    }

    #[test]
    fn test_cosine_similarity_opposite() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![-1.0, -2.0, -3.0];
        assert!((cosine_similarity(&a, &b) + 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_kl_divergence_identical() {
        let a = vec![1.0, 2.0, 3.0, 4.0];
        assert!(kl_divergence_softmax(&a, &a) < 1e-6);
    }

    #[test]
    fn test_kl_divergence_different() {
        let a = vec![10.0, 0.0, 0.0, 0.0];
        let b = vec![0.0, 0.0, 0.0, 10.0];
        assert!(kl_divergence_softmax(&a, &b) > 1.0);
    }

    #[test]
    fn test_sensitivity_analyzer_creation() {
        let analyzer = SensitivityAnalyzer::default();
        assert_eq!(analyzer.removable_threshold, 0.99);

        let custom = SensitivityAnalyzer::new(0.95);
        assert_eq!(custom.removable_threshold, 0.95);
    }

    #[test]
    fn test_layer_importance_divergence() {
        let imp = LayerImportance {
            layer_idx: 3,
            cosine_similarity: 0.98,
            kl_divergence: 0.05,
            removable: false,
        };
        assert!((imp.divergence() - 0.02).abs() < 1e-6);
    }

    #[test]
    fn test_sensitivity_mamba_synthetic() {
        use crate::ssm::config::MambaConfig;
        use crate::ssm::mamba_block::MambaBlock;

        let d_model = 16;
        let d_inner = 32;
        let d_state = 4;
        let d_conv = 4;
        let dt_rank = 4;
        let vocab = 32;
        let num_layers = 3;

        let config = MambaConfig {
            d_model,
            d_state,
            d_conv,
            expand: 2,
            dt_rank,
            vocab_size: vocab,
            num_layers,
            norm_eps: 1e-5,
        };

        // Create blocks with small random-ish weights
        let blocks: Vec<MambaBlock> = (0..num_layers)
            .map(|i| {
                let seed = (i + 1) as f32;
                MambaBlock {
                    d_model,
                    d_inner,
                    d_state,
                    d_conv,
                    dt_rank,
                    norm_weight: vec![1.0; d_model],
                    norm_eps: 1e-5,
                    in_proj_weight: (0..2 * d_inner * d_model)
                        .map(|j| (j as f32 * seed * 0.01).sin() * 0.1)
                        .collect(),
                    in_proj_bias: vec![],
                    conv1d_weight: vec![0.1; d_inner * d_conv],
                    conv1d_bias: vec![0.0; d_inner],
                    x_proj_weight: vec![0.01; (dt_rank + 2 * d_state) * d_inner],
                    dt_proj_weight: vec![0.01; d_inner * dt_rank],
                    dt_proj_bias: vec![0.5; d_inner], // positive bias for stability
                    a_log: vec![-1.0; d_inner * d_state],
                    d_param: vec![1.0; d_inner],
                    out_proj_weight: (0..d_model * d_inner)
                        .map(|j| (j as f32 * seed * 0.02).cos() * 0.1)
                        .collect(),
                    out_proj_bias: vec![],
                }
            })
            .collect();

        let embed_tokens: Vec<f32> = (0..vocab * d_model)
            .map(|i| (i as f32 * 0.01).sin())
            .collect();
        let final_norm_weight = vec![1.0; d_model];
        let lm_head_weight: Vec<f32> = (0..vocab * d_model)
            .map(|i| (i as f32 * 0.02).cos())
            .collect();

        let calibration = vec![1u32, 5, 10, 3, 7];

        let analyzer = SensitivityAnalyzer::new(0.99);
        let importances = analyzer.analyze_mamba(
            &config,
            &embed_tokens,
            &blocks,
            &final_norm_weight,
            &lm_head_weight,
            &calibration,
        );

        assert_eq!(importances.len(), num_layers);
        // Each importance should have valid cosine similarity
        for imp in &importances {
            assert!(imp.cosine_similarity >= -1.0 && imp.cosine_similarity <= 1.0);
            assert!(imp.kl_divergence >= 0.0);
        }
        // Should be sorted by cosine similarity descending
        for w in importances.windows(2) {
            assert!(w[0].cosine_similarity >= w[1].cosine_similarity);
        }
    }
}
