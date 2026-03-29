//! Surgery pipeline: orchestrates analyze → prune → validate → export.
//!
//! Combines sensitivity analysis, layer removal, Wanda weight pruning,
//! and SSM-aware channel pruning into a single configurable pipeline.

use crate::model::traits::Model;
use crate::ssm::mamba_model::MambaModel;
use super::layer_removal::LayerRemover;
use super::sensitivity::cosine_similarity;
use super::wanda::WandaPruner;
use super::ssm_pruning::MambaChannelPruner;

/// A pruning strategy to apply.
#[derive(Debug, Clone)]
pub enum PruningStrategy {
    /// Remove redundant layers. Value = max layers to remove.
    LayerRemoval { max_layers: usize },
    /// Wanda weight pruning. Value = sparsity ratio (0.0-1.0).
    WandaPruning { sparsity: f32 },
    /// Mamba channel pruning. Value = target d_inner.
    MambaChannelPruning { target_d_inner: usize },
}

/// Report from a surgery operation.
#[derive(Debug)]
pub struct SurgeryReport {
    /// Steps that were applied.
    pub steps: Vec<SurgeryStep>,
    /// Final cosine similarity vs original model output.
    pub final_similarity: f32,
    /// Original model parameter count.
    pub original_params: usize,
    /// Pruned model parameter count.
    pub pruned_params: usize,
    /// Compression ratio.
    pub compression_ratio: f32,
}

/// A single step in the surgery pipeline.
#[derive(Debug)]
pub struct SurgeryStep {
    pub strategy: String,
    pub details: String,
    pub similarity_after: f32,
}

/// Configurable surgery pipeline.
pub struct SurgeonPipeline {
    /// Ordered list of pruning strategies to apply.
    pub strategies: Vec<PruningStrategy>,
    /// Maximum acceptable quality loss (cosine distance from original).
    pub max_quality_loss: f32,
}

impl SurgeonPipeline {
    pub fn new(strategies: Vec<PruningStrategy>, max_quality_loss: f32) -> Self {
        Self {
            strategies,
            max_quality_loss,
        }
    }

    /// Run the full surgery pipeline on a Mamba model.
    ///
    /// Applies strategies in order, checking quality after each step.
    /// Stops early if quality drops below threshold.
    pub fn run_mamba(
        &self,
        model: MambaModel,
        calibration_tokens: &[u32],
    ) -> (MambaModel, SurgeryReport) {
        let original_output = model.forward(calibration_tokens);
        let original_params = count_mamba_params(&model);

        let mut current = model;
        let mut steps = Vec::new();
        let quality_threshold = 1.0 - self.max_quality_loss;

        for strategy in &self.strategies {
            match strategy {
                PruningStrategy::LayerRemoval { max_layers } => {
                    let remover = LayerRemover::new(quality_threshold, *max_layers);
                    let (pruned, result) = remover.remove_mamba_layers(current, calibration_tokens);

                    let sim = cosine_similarity(
                        &original_output.logits,
                        &pruned.forward(calibration_tokens).logits,
                    );

                    steps.push(SurgeryStep {
                        strategy: "LayerRemoval".into(),
                        details: format!(
                            "removed {} layers (kept {})",
                            result.removed_layers.len(),
                            result.remaining_layers,
                        ),
                        similarity_after: sim,
                    });

                    current = pruned;
                    if sim < quality_threshold {
                        break;
                    }
                }

                PruningStrategy::WandaPruning { sparsity } => {
                    let pruner = WandaPruner::new(*sparsity);
                    let total = pruner.prune_mamba_calibrated(
                        &current.config,
                        &current.embed_tokens,
                        &mut current.blocks,
                        calibration_tokens,
                    );

                    let sim = cosine_similarity(
                        &original_output.logits,
                        &current.forward(calibration_tokens).logits,
                    );

                    steps.push(SurgeryStep {
                        strategy: "WandaPruning".into(),
                        details: format!("pruned {} weights at {:.0}% sparsity", total, sparsity * 100.0),
                        similarity_after: sim,
                    });

                    if sim < quality_threshold {
                        break;
                    }
                }

                PruningStrategy::MambaChannelPruning { target_d_inner } => {
                    let pruner = MambaChannelPruner::new(*target_d_inner);
                    let mut new_blocks = Vec::with_capacity(current.blocks.len());
                    let mut total_removed = 0;

                    for block in &current.blocks {
                        let (pruned_block, result) = pruner.prune_block(block);
                        total_removed += result.channels_removed;
                        new_blocks.push(pruned_block);
                    }

                    // Update config
                    let new_config = current.config.clone();
                    // expand ratio adjusts with new d_inner
                    // d_inner = d_model * expand, so we keep the config but blocks have smaller d_inner

                    let pruned = MambaModel::new(
                        new_config,
                        current.embed_tokens.clone(),
                        new_blocks,
                        current.final_norm_weight.clone(),
                        current.lm_head_weight.clone(),
                    );

                    let sim = cosine_similarity(
                        &original_output.logits,
                        &pruned.forward(calibration_tokens).logits,
                    );

                    steps.push(SurgeryStep {
                        strategy: "MambaChannelPruning".into(),
                        details: format!(
                            "removed {} channels per block (target d_inner={})",
                            total_removed / current.blocks.len().max(1),
                            target_d_inner,
                        ),
                        similarity_after: sim,
                    });

                    current = pruned;
                    if sim < quality_threshold {
                        break;
                    }
                }
            }
        }

        let final_sim = cosine_similarity(
            &original_output.logits,
            &current.forward(calibration_tokens).logits,
        );
        let pruned_params = count_mamba_params(&current);

        let report = SurgeryReport {
            steps,
            final_similarity: final_sim,
            original_params,
            pruned_params,
            compression_ratio: original_params as f32 / pruned_params.max(1) as f32,
        };

        (current, report)
    }
}

/// Count total parameters in a Mamba model.
fn count_mamba_params(model: &MambaModel) -> usize {
    let mut count = model.embed_tokens.len()
        + model.final_norm_weight.len()
        + model.lm_head_weight.len();

    for block in &model.blocks {
        count += block.in_proj_weight.len()
            + block.in_proj_bias.len()
            + block.conv1d_weight.len()
            + block.conv1d_bias.len()
            + block.x_proj_weight.len()
            + block.dt_proj_weight.len()
            + block.dt_proj_bias.len()
            + block.a_log.len()
            + block.d_param.len()
            + block.out_proj_weight.len()
            + block.out_proj_bias.len()
            + block.norm_weight.len();
    }
    count
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ssm::config::MambaConfig;
    use crate::ssm::mamba_block::MambaBlock;

    fn make_test_model() -> MambaModel {
        let d_model = 16;
        let d_inner = 32;
        let d_state = 4;
        let d_conv = 4;
        let dt_rank = 4;
        let vocab = 32;
        let num_layers = 4;

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

        MambaModel::new(config, embed_tokens, blocks, final_norm_weight, lm_head_weight)
    }

    #[test]
    fn test_pipeline_layer_removal_only() {
        let model = make_test_model();
        let tokens = vec![1u32, 5, 10, 3, 7];

        let pipeline = SurgeonPipeline::new(
            vec![PruningStrategy::LayerRemoval { max_layers: 2 }],
            0.5, // very permissive
        );

        let (_pruned, report) = pipeline.run_mamba(model, &tokens);

        assert!(!report.steps.is_empty());
        assert!(report.final_similarity > 0.0);
        assert!(report.pruned_params <= report.original_params);
        assert!(report.compression_ratio >= 1.0);
    }

    #[test]
    fn test_pipeline_wanda_only() {
        let model = make_test_model();
        let tokens = vec![1u32, 5, 10];

        let pipeline = SurgeonPipeline::new(
            vec![PruningStrategy::WandaPruning { sparsity: 0.3 }],
            0.5,
        );

        let (_pruned, report) = pipeline.run_mamba(model, &tokens);

        assert_eq!(report.steps.len(), 1);
        assert!(report.steps[0].strategy.contains("Wanda"));
    }

    #[test]
    fn test_pipeline_combined() {
        let model = make_test_model();
        let tokens = vec![1u32, 5, 10, 3, 7];

        let pipeline = SurgeonPipeline::new(
            vec![
                PruningStrategy::LayerRemoval { max_layers: 1 },
                PruningStrategy::WandaPruning { sparsity: 0.2 },
            ],
            0.5,
        );

        let (_pruned, report) = pipeline.run_mamba(model, &tokens);

        assert!(report.steps.len() >= 1);
        assert!(report.final_similarity > 0.0);
    }

    #[test]
    fn test_count_mamba_params() {
        let model = make_test_model();
        let params = count_mamba_params(&model);
        // embed (32*16) + lm_head (32*16) + norm (16) + 4 blocks
        assert!(params > 0);
        assert!(params > 1000); // reasonable for a 4-layer model
    }

    #[test]
    fn test_surgery_report_format() {
        let report = SurgeryReport {
            steps: vec![SurgeryStep {
                strategy: "test".into(),
                details: "removed 2 layers".into(),
                similarity_after: 0.95,
            }],
            final_similarity: 0.95,
            original_params: 10000,
            pruned_params: 7000,
            compression_ratio: 10000.0 / 7000.0,
        };

        assert_eq!(report.steps.len(), 1);
        assert!(report.compression_ratio > 1.0);
    }
}
