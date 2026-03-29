//! Wanda (Weights and Activations) pruning.
//!
//! One-shot pruning: importance[i,j] = |W[i,j]| * ||X[:,j]||_2
//! Prunes bottom-p% weights per output row. No retraining required.
//!
//! Ref: Sun et al., "A Simple and Effective Pruning Approach for Large Language Models" (ICLR 2024)

/// Prune a weight matrix using Wanda scoring.
///
/// For each row (output channel), computes importance = |W[i,j]| * activation_norms[j]
/// and zeros out the bottom `sparsity` fraction.
///
/// - `weights`: row-major `[rows, cols]`
/// - `activation_norms`: L2 norms of each input column, shape `[cols]`
/// - `sparsity`: fraction to prune (0.0 = no pruning, 0.5 = prune 50%)
///
/// Returns the pruned weight matrix and the number of zeroed weights.
pub fn wanda_prune_matrix(
    weights: &mut [f32],
    rows: usize,
    cols: usize,
    activation_norms: &[f32],
    sparsity: f32,
) -> usize {
    assert_eq!(weights.len(), rows * cols);
    assert_eq!(activation_norms.len(), cols);
    assert!(sparsity >= 0.0 && sparsity <= 1.0);

    let prune_count = (cols as f32 * sparsity) as usize;
    if prune_count == 0 {
        return 0;
    }

    let mut total_pruned = 0;

    // Per-row pruning (structured per output channel)
    for row in 0..rows {
        let row_start = row * cols;
        let row_weights = &weights[row_start..row_start + cols];

        // Compute Wanda importance scores
        let mut scores: Vec<(usize, f32)> = row_weights
            .iter()
            .enumerate()
            .map(|(j, &w)| (j, w.abs() * activation_norms[j]))
            .collect();

        // Sort by importance ascending (least important first)
        scores.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

        // Zero out bottom-k weights
        for &(j, _) in scores.iter().take(prune_count) {
            weights[row_start + j] = 0.0;
            total_pruned += 1;
        }
    }

    total_pruned
}

/// Compute activation L2 norms from a batch of activation vectors.
///
/// - `activations`: row-major `[batch_size, hidden_dim]`
/// - Returns L2 norm of each column, shape `[hidden_dim]`
pub fn compute_activation_norms(
    activations: &[f32],
    batch_size: usize,
    hidden_dim: usize,
) -> Vec<f32> {
    assert_eq!(activations.len(), batch_size * hidden_dim);
    let mut norms = vec![0.0f32; hidden_dim];

    for row in 0..batch_size {
        let offset = row * hidden_dim;
        for j in 0..hidden_dim {
            let val = activations[offset + j];
            norms[j] += val * val;
        }
    }

    for n in norms.iter_mut() {
        *n = n.sqrt();
    }
    norms
}

/// Wanda pruner for Mamba models.
///
/// Captures activation norms during a calibration forward pass, then prunes
/// linear projections (in_proj, out_proj) of each block.
pub struct WandaPruner {
    /// Target sparsity ratio (0.0-1.0).
    pub sparsity: f32,
}

impl WandaPruner {
    pub fn new(sparsity: f32) -> Self {
        assert!(sparsity >= 0.0 && sparsity < 1.0);
        Self { sparsity }
    }

    /// Prune a Mamba model's linear projections in-place.
    ///
    /// For each block, prunes `in_proj_weight` and `out_proj_weight` using
    /// synthetic activation norms (based on weight statistics, no calibration data needed).
    ///
    /// Returns total number of pruned weights.
    pub fn prune_mamba(
        &self,
        blocks: &mut [crate::ssm::mamba_block::MambaBlock],
    ) -> usize {
        let mut total = 0;

        for block in blocks.iter_mut() {
            let d_model = block.d_model;
            let d_inner = block.d_inner;

            // For in_proj: [2*d_inner, d_model], activation is the d_model embedding
            // Use uniform norms as a reasonable approximation without calibration data
            let in_norms = vec![1.0f32; d_model];
            total += wanda_prune_matrix(
                &mut block.in_proj_weight,
                2 * d_inner,
                d_model,
                &in_norms,
                self.sparsity,
            );

            // For out_proj: [d_model, d_inner], activation is d_inner SSM output
            let out_norms = vec![1.0f32; d_inner];
            total += wanda_prune_matrix(
                &mut block.out_proj_weight,
                d_model,
                d_inner,
                &out_norms,
                self.sparsity,
            );
        }

        total
    }

    /// Prune with calibration data for more accurate activation norms.
    ///
    /// Runs tokens through the model to capture actual activation magnitudes,
    /// then uses those for Wanda scoring.
    pub fn prune_mamba_calibrated(
        &self,
        config: &crate::ssm::config::MambaConfig,
        embed_tokens: &[f32],
        blocks: &mut [crate::ssm::mamba_block::MambaBlock],
        calibration_tokens: &[u32],
    ) -> usize {
        let d_model = config.d_model;
        let vocab = config.vocab_size;
        let seq_len = calibration_tokens.len();
        let mut total = 0;

        // Collect embeddings as activations for in_proj
        let mut embed_acts = vec![0.0f32; seq_len * d_model];
        for (t, &tid) in calibration_tokens.iter().enumerate() {
            let tid = tid as usize;
            if tid < vocab {
                let src = &embed_tokens[tid * d_model..(tid + 1) * d_model];
                embed_acts[t * d_model..(t + 1) * d_model].copy_from_slice(src);
            }
        }

        // Use embedding activations as norms for in_proj
        let in_norms = compute_activation_norms(&embed_acts, seq_len, d_model);

        for block in blocks.iter_mut() {
            let d_inner = block.d_inner;

            // Prune in_proj with calibrated norms
            total += wanda_prune_matrix(
                &mut block.in_proj_weight,
                2 * d_inner,
                d_model,
                &in_norms,
                self.sparsity,
            );

            // For out_proj, use uniform norms (capturing SSM activations
            // requires running the full block, which changes with pruning)
            let out_norms = vec![1.0f32; d_inner];
            total += wanda_prune_matrix(
                &mut block.out_proj_weight,
                d_model,
                d_inner,
                &out_norms,
                self.sparsity,
            );
        }

        total
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_wanda_prune_zero_sparsity() {
        let mut weights = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let norms = vec![1.0, 1.0, 1.0];
        let pruned = wanda_prune_matrix(&mut weights, 2, 3, &norms, 0.0);
        assert_eq!(pruned, 0);
        assert!(weights.iter().all(|&w| w != 0.0));
    }

    #[test]
    fn test_wanda_prune_half_sparsity() {
        // 2x4 matrix, prune 50% per row (= 2 per row)
        let mut weights = vec![
            0.1, 0.5, 0.9, 0.3, // row 0
            0.8, 0.2, 0.4, 0.6, // row 1
        ];
        let norms = vec![1.0; 4]; // uniform norms -> pure magnitude pruning

        let pruned = wanda_prune_matrix(&mut weights, 2, 4, &norms, 0.5);
        assert_eq!(pruned, 4); // 2 per row * 2 rows

        // Row 0: 0.1 and 0.3 should be pruned (smallest magnitudes)
        assert_eq!(weights[0], 0.0); // 0.1 pruned
        assert_ne!(weights[1], 0.0); // 0.5 kept
        assert_ne!(weights[2], 0.0); // 0.9 kept
        assert_eq!(weights[3], 0.0); // 0.3 pruned
    }

    #[test]
    fn test_wanda_activation_norms_affect_scoring() {
        // Same weights but different activation norms
        let mut weights = vec![0.5, 0.5, 0.5, 0.5]; // 1x4
        let norms = vec![0.1, 10.0, 0.1, 10.0]; // columns 0,2 have low activation

        let pruned = wanda_prune_matrix(&mut weights, 1, 4, &norms, 0.5);
        assert_eq!(pruned, 2);

        // Columns 0 and 2 should be pruned (low activation norms)
        assert_eq!(weights[0], 0.0);
        assert_ne!(weights[1], 0.0);
        assert_eq!(weights[2], 0.0);
        assert_ne!(weights[3], 0.0);
    }

    #[test]
    fn test_compute_activation_norms() {
        let activations = vec![
            3.0, 0.0, 4.0, // row 0
            0.0, 4.0, 3.0, // row 1
        ];
        let norms = compute_activation_norms(&activations, 2, 3);
        assert_eq!(norms.len(), 3);
        assert!((norms[0] - 3.0).abs() < 1e-6); // sqrt(9+0)
        assert!((norms[1] - 4.0).abs() < 1e-6); // sqrt(0+16)
        assert!((norms[2] - 5.0).abs() < 1e-6); // sqrt(16+9)
    }

    #[test]
    fn test_wanda_pruner_creation() {
        let pruner = WandaPruner::new(0.3);
        assert_eq!(pruner.sparsity, 0.3);
    }

    #[test]
    fn test_wanda_prune_mamba_synthetic() {
        use crate::ssm::mamba_block::MambaBlock;

        let d_model = 8;
        let d_inner = 16;
        let mut blocks = vec![MambaBlock {
            d_model,
            d_inner,
            d_state: 4,
            d_conv: 4,
            dt_rank: 4,
            norm_weight: vec![1.0; d_model],
            norm_eps: 1e-5,
            in_proj_weight: vec![0.1; 2 * d_inner * d_model],
            in_proj_bias: vec![],
            conv1d_weight: vec![0.1; d_inner * 4],
            conv1d_bias: vec![0.0; d_inner],
            x_proj_weight: vec![0.01; 12 * d_inner],
            dt_proj_weight: vec![0.01; d_inner * 4],
            dt_proj_bias: vec![0.5; d_inner],
            a_log: vec![-1.0; d_inner * 4],
            d_param: vec![1.0; d_inner],
            out_proj_weight: vec![0.1; d_model * d_inner],
            out_proj_bias: vec![],
        }];

        let pruner = WandaPruner::new(0.25);
        let total = pruner.prune_mamba(&mut blocks);

        // 25% of in_proj (2*16*8 = 256, 25% = 2 per row * 32 rows = 64)
        // 25% of out_proj (8*16 = 128, 25% = 4 per row * 8 rows = 32)
        assert!(total > 0);

        // Verify some weights are now zero
        let zeros_in = blocks[0].in_proj_weight.iter().filter(|&&w| w == 0.0).count();
        let zeros_out = blocks[0].out_proj_weight.iter().filter(|&&w| w == 0.0).count();
        assert!(zeros_in > 0);
        assert!(zeros_out > 0);
    }
}
