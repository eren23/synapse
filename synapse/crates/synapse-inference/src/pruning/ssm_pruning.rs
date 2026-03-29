//! SSM-aware structured pruning for Mamba and RWKV models.
//!
//! Inspired by Mamba-Shedder (NAACL 2025):
//! - Channel pruning: reduce d_inner by removing low-importance channels
//! - Head pruning: reduce num_heads in RWKV by importance
//!
//! These are structured pruning methods that actually reduce matrix dimensions,
//! yielding real speedups without sparse kernel support.

use crate::ssm::mamba_block::MambaBlock;
use crate::ssm::rwkv_block::RwkvBlock;

/// Result of channel pruning on a Mamba model.
#[derive(Debug)]
pub struct ChannelPruneResult {
    pub original_d_inner: usize,
    pub pruned_d_inner: usize,
    pub channels_removed: usize,
}

/// Result of head pruning on an RWKV model.
#[derive(Debug)]
pub struct HeadPruneResult {
    pub original_num_heads: usize,
    pub pruned_num_heads: usize,
    pub heads_removed: usize,
}

/// Mamba channel pruner: reduces d_inner by removing channels with lowest importance.
///
/// Channel importance is measured by the L2 norm of the corresponding out_proj column,
/// since out_proj maps d_inner -> d_model. Low-norm columns contribute less to output.
pub struct MambaChannelPruner {
    /// Target d_inner after pruning.
    pub target_d_inner: usize,
}

impl MambaChannelPruner {
    pub fn new(target_d_inner: usize) -> Self {
        Self { target_d_inner }
    }

    /// Compute channel importance scores for a block.
    ///
    /// Returns importance score for each of the d_inner channels,
    /// based on out_proj column L2 norms.
    pub fn channel_importance(block: &MambaBlock) -> Vec<f32> {
        let d_model = block.d_model;
        let d_inner = block.d_inner;

        // out_proj: [d_model, d_inner] row-major
        // Column j importance = L2 norm of column j
        let mut scores = vec![0.0f32; d_inner];
        for row in 0..d_model {
            for col in 0..d_inner {
                let w = block.out_proj_weight[row * d_inner + col];
                scores[col] += w * w;
            }
        }
        for s in scores.iter_mut() {
            *s = s.sqrt();
        }
        scores
    }

    /// Prune channels from a Mamba block, returning a new block with reduced d_inner.
    ///
    /// Keeps the top-k channels by importance.
    pub fn prune_block(&self, block: &MambaBlock) -> (MambaBlock, ChannelPruneResult) {
        let old_d_inner = block.d_inner;
        let new_d_inner = self.target_d_inner.min(old_d_inner);

        if new_d_inner >= old_d_inner {
            return (
                super::layer_removal::clone_mamba_block(block),
                ChannelPruneResult {
                    original_d_inner: old_d_inner,
                    pruned_d_inner: old_d_inner,
                    channels_removed: 0,
                },
            );
        }

        let scores = Self::channel_importance(block);

        // Get indices of top-k channels sorted by importance (descending)
        let mut indexed: Vec<(usize, f32)> = scores.into_iter().enumerate().collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        let mut keep_indices: Vec<usize> = indexed.iter().take(new_d_inner).map(|&(i, _)| i).collect();
        keep_indices.sort(); // Keep original order for consistency

        let d_model = block.d_model;
        let d_state = block.d_state;
        let d_conv = block.d_conv;
        let dt_rank = block.dt_rank;

        // Prune in_proj: [2*d_inner, d_model] -> keep rows for kept channels
        // First d_inner rows are for x, next d_inner for z
        let mut new_in_proj = Vec::with_capacity(2 * new_d_inner * d_model);
        for &idx in &keep_indices {
            let start = idx * d_model;
            new_in_proj.extend_from_slice(&block.in_proj_weight[start..start + d_model]);
        }
        for &idx in &keep_indices {
            let start = (old_d_inner + idx) * d_model;
            new_in_proj.extend_from_slice(&block.in_proj_weight[start..start + d_model]);
        }

        // Prune conv1d: [d_inner, d_conv] -> keep rows for kept channels
        let new_conv1d_weight = select_rows(&block.conv1d_weight, old_d_inner, d_conv, &keep_indices);
        let new_conv1d_bias = select_elements(&block.conv1d_bias, &keep_indices);

        // Prune x_proj: [dt_rank + 2*d_state, d_inner] -> keep columns for kept channels
        let x_proj_rows = dt_rank + 2 * d_state;
        let new_x_proj = select_cols(&block.x_proj_weight, x_proj_rows, old_d_inner, &keep_indices);

        // Prune dt_proj: [d_inner, dt_rank] -> keep rows for kept channels
        let new_dt_proj = select_rows(&block.dt_proj_weight, old_d_inner, dt_rank, &keep_indices);
        let new_dt_proj_bias = select_elements(&block.dt_proj_bias, &keep_indices);

        // Prune a_log: [d_inner, d_state] -> keep rows
        let new_a_log = select_rows(&block.a_log, old_d_inner, d_state, &keep_indices);

        // Prune d_param: [d_inner] -> keep elements
        let new_d_param = select_elements(&block.d_param, &keep_indices);

        // Prune out_proj: [d_model, d_inner] -> keep columns for kept channels
        let new_out_proj = select_cols(&block.out_proj_weight, d_model, old_d_inner, &keep_indices);

        let new_block = MambaBlock {
            d_model,
            d_inner: new_d_inner,
            d_state,
            d_conv,
            dt_rank,
            norm_weight: block.norm_weight.clone(),
            norm_eps: block.norm_eps,
            in_proj_weight: new_in_proj,
            in_proj_bias: if block.in_proj_bias.is_empty() {
                vec![]
            } else {
                let mut bias = select_elements(&block.in_proj_bias[..old_d_inner], &keep_indices);
                bias.extend(select_elements(&block.in_proj_bias[old_d_inner..], &keep_indices));
                bias
            },
            conv1d_weight: new_conv1d_weight,
            conv1d_bias: new_conv1d_bias,
            x_proj_weight: new_x_proj,
            dt_proj_weight: new_dt_proj,
            dt_proj_bias: new_dt_proj_bias,
            a_log: new_a_log,
            d_param: new_d_param,
            out_proj_weight: new_out_proj,
            out_proj_bias: if block.out_proj_bias.is_empty() {
                vec![]
            } else {
                block.out_proj_bias.clone() // [d_model], unchanged
            },
        };

        let result = ChannelPruneResult {
            original_d_inner: old_d_inner,
            pruned_d_inner: new_d_inner,
            channels_removed: old_d_inner - new_d_inner,
        };

        (new_block, result)
    }
}

/// RWKV head pruner: reduces num_heads by removing least important attention heads.
///
/// Head importance is measured by the L2 norm of the corresponding output projection
/// columns for that head's channels.
pub struct RwkvHeadPruner {
    /// Target number of heads after pruning.
    pub target_num_heads: usize,
}

impl RwkvHeadPruner {
    pub fn new(target_num_heads: usize) -> Self {
        Self { target_num_heads }
    }

    /// Compute per-head importance scores.
    ///
    /// For each head, sums the L2 norms of the o_proj columns corresponding
    /// to that head's channels.
    pub fn head_importance(block: &RwkvBlock) -> Vec<f32> {
        let h = block.hidden_size;
        let num_heads = block.num_heads;
        let head_size = block.head_size;

        // o_proj: [h, h] row-major
        let mut scores = vec![0.0f32; num_heads];
        for head in 0..num_heads {
            let col_start = head * head_size;
            let mut sum = 0.0f32;
            for row in 0..h {
                for c in 0..head_size {
                    let w = block.o_proj[row * h + col_start + c];
                    sum += w * w;
                }
            }
            scores[head] = sum.sqrt();
        }
        scores
    }

    /// Prune heads from an RWKV block, returning a new block with fewer heads.
    pub fn prune_block(&self, block: &RwkvBlock) -> (RwkvBlock, HeadPruneResult) {
        let old_num_heads = block.num_heads;
        let new_num_heads = self.target_num_heads.min(old_num_heads);

        if new_num_heads >= old_num_heads {
            return (
                super::layer_removal::clone_rwkv_block(block),
                HeadPruneResult {
                    original_num_heads: old_num_heads,
                    pruned_num_heads: old_num_heads,
                    heads_removed: 0,
                },
            );
        }

        let scores = Self::head_importance(block);
        let head_size = block.head_size;
        let old_h = block.hidden_size;
        let new_h = new_num_heads * head_size;

        // Get indices of top-k heads
        let mut indexed: Vec<(usize, f32)> = scores.into_iter().enumerate().collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        let mut keep_heads: Vec<usize> = indexed.iter().take(new_num_heads).map(|&(i, _)| i).collect();
        keep_heads.sort();

        // Convert head indices to channel indices
        let keep_channels: Vec<usize> = keep_heads
            .iter()
            .flat_map(|&h| h * head_size..(h + 1) * head_size)
            .collect();

        // Prune all [h, h] projection matrices
        let new_r_proj = select_rows_and_cols(&block.r_proj, old_h, old_h, &keep_channels, &keep_channels);
        let new_k_proj = select_rows_and_cols(&block.k_proj, old_h, old_h, &keep_channels, &keep_channels);
        let new_v_proj = select_rows_and_cols(&block.v_proj, old_h, old_h, &keep_channels, &keep_channels);
        let new_o_proj = select_rows_and_cols(&block.o_proj, old_h, old_h, &keep_channels, &keep_channels);

        // Prune per-channel vectors [h]
        let new_x_r = select_elements(&block.x_r, &keep_channels);
        let new_x_k = select_elements(&block.x_k, &keep_channels);
        let new_x_v = select_elements(&block.x_v, &keep_channels);
        let new_x_w = select_elements(&block.x_w, &keep_channels);
        let new_x_a = select_elements(&block.x_a, &keep_channels);
        let new_x_g = select_elements(&block.x_g, &keep_channels);
        let new_w0 = select_elements(&block.w0, &keep_channels);
        let new_a0 = select_elements(&block.a0, &keep_channels);
        let new_k_k = select_elements(&block.k_k, &keep_channels);
        let new_k_a = select_elements(&block.k_a, &keep_channels);

        // r_k: [num_heads, head_size] -> select kept heads
        let new_r_k = select_rows(&block.r_k, old_num_heads, head_size, &keep_heads);

        // GroupNorm weights/biases: [h] -> select kept channels
        let new_g_norm_weight = select_elements(&block.g_norm_weight, &keep_channels);
        let new_g_norm_bias = select_elements(&block.g_norm_bias, &keep_channels);

        // Low-rank matrices: w1 [h, decay_rank], w2 [decay_rank, h]
        let new_w1 = select_rows(&block.w1, old_h, block.decay_rank, &keep_channels);
        let new_w2 = select_cols(&block.w2, block.decay_rank, old_h, &keep_channels);
        let new_a1 = select_rows(&block.a1, old_h, block.alpha_rank, &keep_channels);
        let new_a2 = select_cols(&block.a2, block.alpha_rank, old_h, &keep_channels);
        let new_g1 = select_rows(&block.g1, old_h, block.gate_rank, &keep_channels);
        let new_g2 = select_cols(&block.g2, block.gate_rank, old_h, &keep_channels);

        // LayerNorm weights: [h]
        let new_ln1_weight = select_elements(&block.ln1_weight, &keep_channels);
        let new_ln1_bias = select_elements(&block.ln1_bias, &keep_channels);
        let new_ln2_weight = select_elements(&block.ln2_weight, &keep_channels);
        let new_ln2_bias = select_elements(&block.ln2_bias, &keep_channels);

        // FFN: keep ffn_x_k [h], resize key/value weights
        let new_ffn_x_k = select_elements(&block.ffn_x_k, &keep_channels);
        // ffn_key_weight: [intermediate, h]
        let new_ffn_key = select_cols(&block.ffn_key_weight, block.intermediate_size, old_h, &keep_channels);
        // ffn_value_weight: [h, intermediate]
        let new_ffn_value = select_rows(&block.ffn_value_weight, old_h, block.intermediate_size, &keep_channels);

        // Value residual: v0 [h], v1 [v_rank, h], v2 [h, v_rank]
        let new_v0 = if block.v0.is_empty() { vec![] } else { select_elements(&block.v0, &keep_channels) };
        let new_v1 = if block.v1.is_empty() { vec![] } else { select_cols(&block.v1, block.v_rank, old_h, &keep_channels) };
        let new_v2 = if block.v2.is_empty() { vec![] } else { select_rows(&block.v2, old_h, block.v_rank, &keep_channels) };

        let new_block = RwkvBlock {
            hidden_size: new_h,
            num_heads: new_num_heads,
            head_size,
            intermediate_size: block.intermediate_size,
            decay_rank: block.decay_rank,
            alpha_rank: block.alpha_rank,
            gate_rank: block.gate_rank,
            norm_eps: block.norm_eps,
            ln1_weight: new_ln1_weight,
            ln1_bias: new_ln1_bias,
            x_r: new_x_r,
            x_k: new_x_k,
            x_v: new_x_v,
            x_w: new_x_w,
            x_a: new_x_a,
            x_g: new_x_g,
            r_proj: new_r_proj,
            k_proj: new_k_proj,
            v_proj: new_v_proj,
            o_proj: new_o_proj,
            w0: new_w0,
            w1: new_w1,
            w2: new_w2,
            a0: new_a0,
            a1: new_a1,
            a2: new_a2,
            g1: new_g1,
            g2: new_g2,
            k_k: new_k_k,
            k_a: new_k_a,
            r_k: new_r_k,
            g_norm_weight: new_g_norm_weight,
            g_norm_bias: new_g_norm_bias,
            ln2_weight: new_ln2_weight,
            ln2_bias: new_ln2_bias,
            ffn_x_k: new_ffn_x_k,
            v_rank: block.v_rank,
            v0: new_v0,
            v1: new_v1,
            v2: new_v2,
            ffn_key_weight: new_ffn_key,
            ffn_value_weight: new_ffn_value,
        };

        let result = HeadPruneResult {
            original_num_heads: old_num_heads,
            pruned_num_heads: new_num_heads,
            heads_removed: old_num_heads - new_num_heads,
        };

        (new_block, result)
    }
}

// ── Matrix selection helpers ──────────────────────────────────────

/// Select specific rows from a row-major matrix.
fn select_rows(mat: &[f32], _rows: usize, cols: usize, keep: &[usize]) -> Vec<f32> {
    let mut out = Vec::with_capacity(keep.len() * cols);
    for &r in keep {
        let start = r * cols;
        out.extend_from_slice(&mat[start..start + cols]);
    }
    out
}

/// Select specific columns from a row-major matrix.
fn select_cols(mat: &[f32], rows: usize, cols: usize, keep: &[usize]) -> Vec<f32> {
    let mut out = Vec::with_capacity(rows * keep.len());
    for r in 0..rows {
        for &c in keep {
            out.push(mat[r * cols + c]);
        }
    }
    out
}

/// Select specific rows and columns from a square matrix.
fn select_rows_and_cols(
    mat: &[f32],
    _rows: usize,
    cols: usize,
    keep_rows: &[usize],
    keep_cols: &[usize],
) -> Vec<f32> {
    let mut out = Vec::with_capacity(keep_rows.len() * keep_cols.len());
    for &r in keep_rows {
        for &c in keep_cols {
            out.push(mat[r * cols + c]);
        }
    }
    out
}

/// Select specific elements from a 1D vector.
fn select_elements(vec: &[f32], keep: &[usize]) -> Vec<f32> {
    keep.iter().map(|&i| vec[i]).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_select_rows() {
        let mat = vec![
            1.0, 2.0, 3.0, // row 0
            4.0, 5.0, 6.0, // row 1
            7.0, 8.0, 9.0, // row 2
        ];
        let selected = select_rows(&mat, 3, 3, &[0, 2]);
        assert_eq!(selected, vec![1.0, 2.0, 3.0, 7.0, 8.0, 9.0]);
    }

    #[test]
    fn test_select_cols() {
        let mat = vec![
            1.0, 2.0, 3.0, // row 0
            4.0, 5.0, 6.0, // row 1
        ];
        let selected = select_cols(&mat, 2, 3, &[0, 2]);
        assert_eq!(selected, vec![1.0, 3.0, 4.0, 6.0]);
    }

    #[test]
    fn test_select_elements() {
        let vec = vec![10.0, 20.0, 30.0, 40.0, 50.0];
        let selected = select_elements(&vec, &[1, 3, 4]);
        assert_eq!(selected, vec![20.0, 40.0, 50.0]);
    }

    #[test]
    fn test_mamba_channel_importance() {
        let block = MambaBlock {
            d_model: 4,
            d_inner: 3,
            d_state: 2,
            d_conv: 2,
            dt_rank: 2,
            norm_weight: vec![1.0; 4],
            norm_eps: 1e-5,
            in_proj_weight: vec![0.1; 6 * 4],
            in_proj_bias: vec![],
            conv1d_weight: vec![0.1; 3 * 2],
            conv1d_bias: vec![0.0; 3],
            x_proj_weight: vec![0.01; 6 * 3],
            dt_proj_weight: vec![0.01; 3 * 2],
            dt_proj_bias: vec![0.5; 3],
            a_log: vec![-1.0; 3 * 2],
            d_param: vec![1.0; 3],
            // out_proj: [4, 3] -> channel 1 has highest norm
            out_proj_weight: vec![
                0.1, 0.9, 0.1,
                0.1, 0.8, 0.1,
                0.1, 0.7, 0.1,
                0.1, 0.6, 0.1,
            ],
            out_proj_bias: vec![],
        };

        let scores = MambaChannelPruner::channel_importance(&block);
        assert_eq!(scores.len(), 3);
        // Channel 1 should have highest importance
        assert!(scores[1] > scores[0]);
        assert!(scores[1] > scores[2]);
    }

    #[test]
    fn test_mamba_channel_prune() {
        let block = MambaBlock {
            d_model: 4,
            d_inner: 4,
            d_state: 2,
            d_conv: 2,
            dt_rank: 2,
            norm_weight: vec![1.0; 4],
            norm_eps: 1e-5,
            in_proj_weight: (0..8 * 4).map(|i| i as f32 * 0.01).collect(),
            in_proj_bias: vec![],
            conv1d_weight: vec![0.1; 4 * 2],
            conv1d_bias: vec![0.1; 4],
            x_proj_weight: vec![0.01; 6 * 4],
            dt_proj_weight: vec![0.01; 4 * 2],
            dt_proj_bias: vec![0.5; 4],
            a_log: vec![-1.0; 4 * 2],
            d_param: vec![1.0; 4],
            out_proj_weight: (0..4 * 4).map(|i| i as f32 * 0.1).collect(),
            out_proj_bias: vec![],
        };

        let pruner = MambaChannelPruner::new(2);
        let (new_block, result) = pruner.prune_block(&block);

        assert_eq!(result.original_d_inner, 4);
        assert_eq!(result.pruned_d_inner, 2);
        assert_eq!(result.channels_removed, 2);

        assert_eq!(new_block.d_inner, 2);
        assert_eq!(new_block.in_proj_weight.len(), 2 * 2 * 4); // [2*new_d_inner, d_model]
        assert_eq!(new_block.out_proj_weight.len(), 4 * 2); // [d_model, new_d_inner]
        assert_eq!(new_block.conv1d_weight.len(), 2 * 2);
        assert_eq!(new_block.conv1d_bias.len(), 2);
        assert_eq!(new_block.dt_proj_bias.len(), 2);
        assert_eq!(new_block.a_log.len(), 2 * 2);
        assert_eq!(new_block.d_param.len(), 2);
    }

    #[test]
    fn test_no_prune_when_target_exceeds() {
        let block = MambaBlock {
            d_model: 4,
            d_inner: 2,
            d_state: 2,
            d_conv: 2,
            dt_rank: 2,
            norm_weight: vec![1.0; 4],
            norm_eps: 1e-5,
            in_proj_weight: vec![0.1; 4 * 4],
            in_proj_bias: vec![],
            conv1d_weight: vec![0.1; 2 * 2],
            conv1d_bias: vec![0.0; 2],
            x_proj_weight: vec![0.01; 6 * 2],
            dt_proj_weight: vec![0.01; 2 * 2],
            dt_proj_bias: vec![0.5; 2],
            a_log: vec![-1.0; 2 * 2],
            d_param: vec![1.0; 2],
            out_proj_weight: vec![0.1; 4 * 2],
            out_proj_bias: vec![],
        };

        let pruner = MambaChannelPruner::new(10); // target > actual
        let (_, result) = pruner.prune_block(&block);
        assert_eq!(result.channels_removed, 0);
    }
}
