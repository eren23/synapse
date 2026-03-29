//! Layer removal: drop near-identity layers from SSM models.
//!
//! Based on ShortGPT/Block Influence Score: layers where cos(input, output) ≈ 1.0
//! are near-identity transforms and can be removed with minimal quality loss.

use crate::ssm::mamba_block::MambaBlock;
use crate::ssm::mamba_model::MambaModel;
use crate::ssm::rwkv_block::RwkvBlock;
use crate::ssm::rwkv_model::RwkvModel;
use super::sensitivity::{SensitivityAnalyzer, cosine_similarity};
use crate::model::traits::Model;

/// Result of a layer removal operation.
#[derive(Debug)]
pub struct RemovalResult {
    /// Indices of removed layers (from original model).
    pub removed_layers: Vec<usize>,
    /// Cosine similarity of pruned model output vs original.
    pub output_similarity: f32,
    /// Number of remaining layers.
    pub remaining_layers: usize,
}

/// Removes redundant layers from models based on sensitivity analysis.
pub struct LayerRemover {
    /// Minimum cosine similarity to maintain vs original output.
    pub quality_threshold: f32,
    /// Maximum number of layers to remove.
    pub max_removals: usize,
}

impl Default for LayerRemover {
    fn default() -> Self {
        Self {
            quality_threshold: 0.95,
            max_removals: usize::MAX,
        }
    }
}

impl LayerRemover {
    pub fn new(quality_threshold: f32, max_removals: usize) -> Self {
        Self {
            quality_threshold,
            max_removals,
        }
    }

    /// Remove redundant layers from a Mamba model.
    ///
    /// Uses greedy removal: analyzes sensitivity, removes the most redundant layer,
    /// re-checks quality, repeats until threshold is hit.
    ///
    /// Returns the pruned model and a report of what was removed.
    pub fn remove_mamba_layers(
        &self,
        model: MambaModel,
        calibration_tokens: &[u32],
    ) -> (MambaModel, RemovalResult) {
        let analyzer = SensitivityAnalyzer::new(0.999);

        // Get original model output for quality comparison
        let original_output = model.forward(calibration_tokens);

        let mut config = model.config.clone();
        let embed_tokens = model.embed_tokens.clone();
        let final_norm_weight = model.final_norm_weight.clone();
        let lm_head_weight = model.lm_head_weight.clone();
        let mut blocks = model.blocks;

        let mut removed_layers = Vec::new();
        let mut last_similarity = 1.0f32;

        for _ in 0..self.max_removals {
            if blocks.len() <= 1 {
                break; // Don't remove the last layer
            }

            // Analyze current model
            let importances = analyzer.analyze_mamba(
                &config,
                &embed_tokens,
                &blocks,
                &final_norm_weight,
                &lm_head_weight,
                calibration_tokens,
            );

            // Find most redundant layer (highest cosine similarity = lowest divergence)
            let candidate = match importances.first() {
                Some(imp) if imp.cosine_similarity > 0.9 => imp.clone(),
                _ => break, // No good candidates
            };

            // Trial removal: build model without this layer
            let make_trial_blocks = || -> Vec<MambaBlock> {
                blocks
                    .iter()
                    .enumerate()
                    .filter(|(i, _)| *i != candidate.layer_idx)
                    .map(|(_, b)| clone_mamba_block(b))
                    .collect()
            };

            let mut trial_config = config.clone();
            trial_config.num_layers = blocks.len() - 1;

            let trial_model = MambaModel::new(
                trial_config.clone(),
                embed_tokens.clone(),
                make_trial_blocks(),
                final_norm_weight.clone(),
                lm_head_weight.clone(),
            );

            let trial_output = trial_model.forward(calibration_tokens);
            let similarity = cosine_similarity(&original_output.logits, &trial_output.logits);

            if similarity < self.quality_threshold {
                break; // Would drop quality too much
            }

            // Accept the removal
            removed_layers.push(candidate.layer_idx);
            blocks = make_trial_blocks();
            config = trial_config;
            last_similarity = similarity;
        }

        let remaining = blocks.len();
        config.num_layers = remaining;

        let pruned_model = MambaModel::new(
            config,
            embed_tokens,
            blocks,
            final_norm_weight,
            lm_head_weight,
        );

        let result = RemovalResult {
            removed_layers,
            output_similarity: last_similarity,
            remaining_layers: remaining,
        };

        (pruned_model, result)
    }

    /// Remove redundant layers from an RWKV model.
    pub fn remove_rwkv_layers(
        &self,
        model: RwkvModel,
        calibration_tokens: &[u32],
    ) -> (RwkvModel, RemovalResult) {
        let analyzer = SensitivityAnalyzer::new(0.999);

        let original_output = model.forward(calibration_tokens);

        let mut config = model.config.clone();
        let embed_tokens = model.embed_tokens.clone();
        let pre_ln_weight = model.pre_ln_weight.clone();
        let pre_ln_bias = model.pre_ln_bias.clone();
        let final_norm_weight = model.final_norm_weight.clone();
        let final_norm_bias = model.final_norm_bias.clone();
        let lm_head_weight = model.lm_head_weight.clone();
        let mut blocks = model.blocks;

        let mut removed_layers = Vec::new();
        let mut last_similarity = 1.0f32;

        for _ in 0..self.max_removals {
            if blocks.len() <= 1 {
                break;
            }

            let importances = analyzer.analyze_rwkv(
                &config,
                &embed_tokens,
                pre_ln_weight.as_deref(),
                pre_ln_bias.as_deref(),
                &blocks,
                &final_norm_weight,
                &final_norm_bias,
                &lm_head_weight,
                calibration_tokens,
            );

            let candidate = match importances.first() {
                Some(imp) if imp.cosine_similarity > 0.9 => imp.clone(),
                _ => break,
            };

            let make_trial_blocks = || -> Vec<RwkvBlock> {
                blocks
                    .iter()
                    .enumerate()
                    .filter(|(i, _)| *i != candidate.layer_idx)
                    .map(|(_, b)| clone_rwkv_block(b))
                    .collect()
            };

            let mut trial_config = config.clone();
            trial_config.num_layers = blocks.len() - 1;

            let mut trial_model = RwkvModel::new(
                trial_config.clone(),
                embed_tokens.clone(),
                make_trial_blocks(),
                final_norm_weight.clone(),
                final_norm_bias.clone(),
                lm_head_weight.clone(),
            );
            trial_model.pre_ln_weight = pre_ln_weight.clone();
            trial_model.pre_ln_bias = pre_ln_bias.clone();

            let trial_output = trial_model.forward(calibration_tokens);
            let similarity = cosine_similarity(&original_output.logits, &trial_output.logits);

            if similarity < self.quality_threshold {
                break;
            }

            removed_layers.push(candidate.layer_idx);
            blocks = make_trial_blocks();
            config = trial_config;
            last_similarity = similarity;
        }

        let remaining = blocks.len();
        config.num_layers = remaining;

        let mut pruned_model = RwkvModel::new(
            config,
            embed_tokens,
            blocks,
            final_norm_weight,
            final_norm_bias,
            lm_head_weight,
        );
        pruned_model.pre_ln_weight = pre_ln_weight;
        pruned_model.pre_ln_bias = pre_ln_bias;

        let result = RemovalResult {
            removed_layers,
            output_similarity: last_similarity,
            remaining_layers: remaining,
        };

        (pruned_model, result)
    }
}

/// Clone a MambaBlock (MambaBlock doesn't derive Clone, so we do it manually).
pub(crate) fn clone_mamba_block(b: &MambaBlock) -> MambaBlock {
    MambaBlock {
        d_model: b.d_model,
        d_inner: b.d_inner,
        d_state: b.d_state,
        d_conv: b.d_conv,
        dt_rank: b.dt_rank,
        norm_weight: b.norm_weight.clone(),
        norm_eps: b.norm_eps,
        in_proj_weight: b.in_proj_weight.clone(),
        in_proj_bias: b.in_proj_bias.clone(),
        conv1d_weight: b.conv1d_weight.clone(),
        conv1d_bias: b.conv1d_bias.clone(),
        x_proj_weight: b.x_proj_weight.clone(),
        dt_proj_weight: b.dt_proj_weight.clone(),
        dt_proj_bias: b.dt_proj_bias.clone(),
        a_log: b.a_log.clone(),
        d_param: b.d_param.clone(),
        out_proj_weight: b.out_proj_weight.clone(),
        out_proj_bias: b.out_proj_bias.clone(),
    }
}

/// Clone an RwkvBlock.
pub(crate) fn clone_rwkv_block(b: &RwkvBlock) -> RwkvBlock {
    RwkvBlock {
        hidden_size: b.hidden_size,
        num_heads: b.num_heads,
        head_size: b.head_size,
        intermediate_size: b.intermediate_size,
        decay_rank: b.decay_rank,
        alpha_rank: b.alpha_rank,
        gate_rank: b.gate_rank,
        norm_eps: b.norm_eps,
        ln1_weight: b.ln1_weight.clone(),
        ln1_bias: b.ln1_bias.clone(),
        x_r: b.x_r.clone(),
        x_k: b.x_k.clone(),
        x_v: b.x_v.clone(),
        x_w: b.x_w.clone(),
        x_a: b.x_a.clone(),
        x_g: b.x_g.clone(),
        r_proj: b.r_proj.clone(),
        k_proj: b.k_proj.clone(),
        v_proj: b.v_proj.clone(),
        o_proj: b.o_proj.clone(),
        w0: b.w0.clone(),
        w1: b.w1.clone(),
        w2: b.w2.clone(),
        a0: b.a0.clone(),
        a1: b.a1.clone(),
        a2: b.a2.clone(),
        g1: b.g1.clone(),
        g2: b.g2.clone(),
        k_k: b.k_k.clone(),
        k_a: b.k_a.clone(),
        r_k: b.r_k.clone(),
        g_norm_weight: b.g_norm_weight.clone(),
        g_norm_bias: b.g_norm_bias.clone(),
        ln2_weight: b.ln2_weight.clone(),
        ln2_bias: b.ln2_bias.clone(),
        ffn_x_k: b.ffn_x_k.clone(),
        v_rank: b.v_rank,
        v0: b.v0.clone(),
        v1: b.v1.clone(),
        v2: b.v2.clone(),
        ffn_key_weight: b.ffn_key_weight.clone(),
        ffn_value_weight: b.ffn_value_weight.clone(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ssm::config::MambaConfig;

    fn make_test_mamba_model(num_layers: usize) -> (MambaModel, MambaConfig) {
        let d_model = 16;
        let d_inner = 32;
        let d_state = 4;
        let d_conv = 4;
        let dt_rank = 4;
        let vocab = 32;

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
                    dt_proj_bias: vec![0.5; d_inner],
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

        let model = MambaModel::new(
            config.clone(),
            embed_tokens,
            blocks,
            final_norm_weight,
            lm_head_weight,
        );

        (model, config)
    }

    #[test]
    fn test_layer_remover_creation() {
        let remover = LayerRemover::default();
        assert_eq!(remover.quality_threshold, 0.95);

        let custom = LayerRemover::new(0.90, 5);
        assert_eq!(custom.quality_threshold, 0.90);
        assert_eq!(custom.max_removals, 5);
    }

    #[test]
    fn test_removal_result() {
        let result = RemovalResult {
            removed_layers: vec![2, 5],
            output_similarity: 0.98,
            remaining_layers: 10,
        };
        assert_eq!(result.removed_layers.len(), 2);
        assert_eq!(result.remaining_layers, 10);
    }

    #[test]
    fn test_mamba_layer_removal_preserves_at_least_one() {
        let (model, _) = make_test_mamba_model(2);
        let calibration = vec![1u32, 5, 10, 3, 7];

        let remover = LayerRemover::new(0.0, 100); // very aggressive
        let (pruned, result) = remover.remove_mamba_layers(model, &calibration);

        // Should always keep at least 1 layer
        assert!(result.remaining_layers >= 1);
        assert_eq!(pruned.blocks.len(), result.remaining_layers);
    }

    #[test]
    fn test_mamba_layer_removal_conservative() {
        let (model, _) = make_test_mamba_model(4);
        let calibration = vec![1u32, 5, 10, 3, 7];

        // Very conservative: require near-perfect similarity
        let remover = LayerRemover::new(0.9999, 1);
        let (_pruned, result) = remover.remove_mamba_layers(model, &calibration);

        // Conservative pruning: may or may not remove layers depending on model
        assert!(result.remaining_layers >= 3); // at most 1 removed
    }

    #[test]
    fn test_clone_mamba_block() {
        let block = MambaBlock {
            d_model: 16,
            d_inner: 32,
            d_state: 4,
            d_conv: 4,
            dt_rank: 4,
            norm_weight: vec![1.0; 16],
            norm_eps: 1e-5,
            in_proj_weight: vec![0.1; 64 * 16],
            in_proj_bias: vec![],
            conv1d_weight: vec![0.1; 32 * 4],
            conv1d_bias: vec![0.0; 32],
            x_proj_weight: vec![0.01; 12 * 32],
            dt_proj_weight: vec![0.01; 32 * 4],
            dt_proj_bias: vec![0.5; 32],
            a_log: vec![-1.0; 32 * 4],
            d_param: vec![1.0; 32],
            out_proj_weight: vec![0.1; 16 * 32],
            out_proj_bias: vec![],
        };

        let cloned = clone_mamba_block(&block);
        assert_eq!(block.d_model, cloned.d_model);
        assert_eq!(block.in_proj_weight, cloned.in_proj_weight);
        assert_eq!(block.out_proj_weight, cloned.out_proj_weight);
    }
}
