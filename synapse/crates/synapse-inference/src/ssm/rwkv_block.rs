//! RwkvBlock: a single RWKV-7 layer.
//!
//! Implements the full RWKV block forward pass:
//! LayerNorm → Time Mixing (token shift → R/K/V/G → WKV → gate → output) → residual
//! → LayerNorm → Channel Mixing (token shift → R/K/V → squared ReLU → receptance gate) → residual.

use crate::ops::matmul::matmul_t;
use crate::ssm::rwkv_state::RwkvLayerState;
use crate::ssm::wkv::wkv_step;

/// A single RWKV-7 block (one layer).
pub struct RwkvBlock {
    pub hidden_size: usize,
    pub num_heads: usize,
    pub head_size: usize,
    pub intermediate_size: usize,
    pub norm_eps: f32,

    // Pre-attention LayerNorm
    pub ln1_weight: Vec<f32>,    // [hidden_size]
    pub ln1_bias: Vec<f32>,      // [hidden_size]

    // Time mixing token shift parameter
    pub time_mix_x: Vec<f32>,    // [hidden_size] lerp weight for mixing

    // Time mixing projections
    pub receptance_weight: Vec<f32>,  // [hidden_size, hidden_size]
    pub key_weight: Vec<f32>,         // [hidden_size, hidden_size]
    pub value_weight: Vec<f32>,       // [hidden_size, hidden_size]
    pub gate_weight: Vec<f32>,        // [hidden_size, hidden_size]
    pub output_weight: Vec<f32>,      // [hidden_size, hidden_size]

    // WKV decay (per head)
    pub time_decay: Vec<f32>,    // [num_heads * head_size] in log space

    // Attention output LayerNorm (GroupNorm per head)
    pub att_ln_weight: Vec<f32>, // [hidden_size]
    pub att_ln_bias: Vec<f32>,   // [hidden_size]

    // Pre-FFN LayerNorm
    pub ln2_weight: Vec<f32>,    // [hidden_size]
    pub ln2_bias: Vec<f32>,      // [hidden_size]

    // Channel mixing token shift
    pub channel_mix_x: Vec<f32>, // [hidden_size]

    // Channel mixing projections
    pub ffn_receptance_weight: Vec<f32>, // [hidden_size, hidden_size]
    pub ffn_key_weight: Vec<f32>,        // [intermediate_size, hidden_size]
    pub ffn_value_weight: Vec<f32>,      // [hidden_size, intermediate_size]
}

#[inline]
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// LayerNorm: y = (x - mean) / sqrt(var + eps) * weight + bias
fn layernorm(x: &[f32], weight: &[f32], bias: &[f32], eps: f32, hidden_size: usize) -> Vec<f32> {
    let n = x.len() / hidden_size;
    let mut out = vec![0.0f32; x.len()];

    for i in 0..n {
        let off = i * hidden_size;
        let row = &x[off..off + hidden_size];

        let mean: f32 = row.iter().sum::<f32>() / hidden_size as f32;
        let var: f32 = row.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / hidden_size as f32;
        let inv_std = 1.0 / (var + eps).sqrt();

        for j in 0..hidden_size {
            out[off + j] = (row[j] - mean) * inv_std * weight[j] + bias[j];
        }
    }
    out
}

impl RwkvBlock {
    /// Process a single token `[hidden_size]` -> `[hidden_size]`.
    pub fn forward_one(&self, hidden: &[f32], state: &mut RwkvLayerState) -> Vec<f32> {
        let h = self.hidden_size;

        // ── Time Mixing ──────────────────────────────────────────────

        // 1. LayerNorm with ln1
        let normed = layernorm(hidden, &self.ln1_weight, &self.ln1_bias, self.norm_eps, h);

        // 2. Token shift: x_mixed = x * (1 - time_mix_x) + prev * time_mix_x
        let x_mixed: Vec<f32> = (0..h)
            .map(|i| normed[i] * (1.0 - self.time_mix_x[i]) + state.time_mix_prev[i] * self.time_mix_x[i])
            .collect();

        // Save current normed as previous for next step
        state.time_mix_prev.copy_from_slice(&normed);

        // 3. R/K/V/G projections: [1, h] x [h, h]^T -> [1, h]
        let r_proj = matmul_t(&x_mixed, &self.receptance_weight, 1, h, h);
        let k_proj = matmul_t(&x_mixed, &self.key_weight, 1, h, h);
        let v_proj = matmul_t(&x_mixed, &self.value_weight, 1, h, h);
        let g_proj = matmul_t(&x_mixed, &self.gate_weight, 1, h, h);

        // 4. Reshape to [num_heads, head_size] and run WKV per head
        let hs = self.head_size;
        let nh = self.num_heads;
        let mut wkv_out = vec![0.0f32; h];

        for head in 0..nh {
            let off = head * hs;
            let r_head = &r_proj[off..off + hs];
            let k_head = &k_proj[off..off + hs];
            let v_head = &v_proj[off..off + hs];
            let w_head = &self.time_decay[off..off + hs];
            let state_head = &mut state.wkv_state[head * hs * hs..(head + 1) * hs * hs];

            let o_head = wkv_step(r_head, k_head, v_head, w_head, state_head, hs);
            wkv_out[off..off + hs].copy_from_slice(&o_head);
        }

        // 5. Apply attention LayerNorm (per-head group norm)
        let normed_wkv = layernorm(&wkv_out, &self.att_ln_weight, &self.att_ln_bias, self.norm_eps, h);

        // 6. Gate: output = sigmoid(G) * normed_wkv
        let gated: Vec<f32> = (0..h)
            .map(|i| sigmoid(g_proj[i]) * normed_wkv[i])
            .collect();

        // 7. Output projection: [1, h] x [h, h]^T -> [1, h]
        let time_out = matmul_t(&gated, &self.output_weight, 1, h, h);

        // 8. Residual add
        let hidden_after_time: Vec<f32> = (0..h)
            .map(|i| hidden[i] + time_out[i])
            .collect();

        // ── Channel Mixing ───────────────────────────────────────────

        // 1. LayerNorm with ln2
        let normed2 = layernorm(&hidden_after_time, &self.ln2_weight, &self.ln2_bias, self.norm_eps, h);

        // 2. Token shift: x_mixed = x * (1 - channel_mix_x) + prev * channel_mix_x
        let x_mixed2: Vec<f32> = (0..h)
            .map(|i| {
                normed2[i] * (1.0 - self.channel_mix_x[i])
                    + state.channel_mix_prev[i] * self.channel_mix_x[i]
            })
            .collect();

        // Save current normed as previous for next step
        state.channel_mix_prev.copy_from_slice(&normed2);

        // 3. k = ffn_key_weight @ x_mixed  [intermediate_size]
        let k_ffn = matmul_t(&x_mixed2, &self.ffn_key_weight, 1, h, self.intermediate_size);

        // 4. k = ReLU(k)^2  (squared ReLU)
        let k_sq: Vec<f32> = k_ffn
            .iter()
            .map(|&x| {
                let relu = if x > 0.0 { x } else { 0.0 };
                relu * relu
            })
            .collect();

        // 5. v = ffn_value_weight @ k  [hidden_size]
        let v_ffn = matmul_t(&k_sq, &self.ffn_value_weight, 1, self.intermediate_size, h);

        // 6. r = sigmoid(ffn_receptance_weight @ x_mixed)
        let r_ffn = matmul_t(&x_mixed2, &self.ffn_receptance_weight, 1, h, h);
        let r_sigmoid: Vec<f32> = r_ffn.iter().map(|&x| sigmoid(x)).collect();

        // 7. output = r * v
        let channel_out: Vec<f32> = (0..h)
            .map(|i| r_sigmoid[i] * v_ffn[i])
            .collect();

        // 8. Residual add
        let output: Vec<f32> = (0..h)
            .map(|i| hidden_after_time[i] + channel_out[i])
            .collect();

        output
    }

    /// Process a full sequence `[seq_len * hidden_size]` -> `[seq_len * hidden_size]`.
    pub fn forward_seq(
        &self,
        hidden: &[f32],
        seq_len: usize,
        state: &mut RwkvLayerState,
    ) -> Vec<f32> {
        let h = self.hidden_size;
        let mut output = Vec::with_capacity(seq_len * h);

        // Process token by token (sequential due to recurrent state dependency)
        for t in 0..seq_len {
            let token = &hidden[t * h..(t + 1) * h];
            let out = self.forward_one(token, state);
            output.extend_from_slice(&out);
        }

        output
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ssm::rwkv_state::RwkvLayerState;

    fn pseudo_random_vec(seed: u64, len: usize) -> Vec<f32> {
        let mut state = seed;
        (0..len)
            .map(|_| {
                state = state
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                let bits = 0x3F800000u32 | ((state >> 41) as u32 & 0x7FFFFF);
                (f32::from_bits(bits) - 1.5) * 0.2
            })
            .collect()
    }

    fn make_test_block() -> RwkvBlock {
        let hidden_size = 64usize;
        let num_heads = 2usize;
        let head_size = 32usize;
        let intermediate_size = 128usize;

        RwkvBlock {
            hidden_size,
            num_heads,
            head_size,
            intermediate_size,
            norm_eps: 1e-5,
            ln1_weight: vec![1.0f32; hidden_size],
            ln1_bias: vec![0.0f32; hidden_size],
            time_mix_x: vec![0.5f32; hidden_size],
            receptance_weight: pseudo_random_vec(1, hidden_size * hidden_size),
            key_weight: pseudo_random_vec(2, hidden_size * hidden_size),
            value_weight: pseudo_random_vec(3, hidden_size * hidden_size),
            gate_weight: pseudo_random_vec(4, hidden_size * hidden_size),
            output_weight: pseudo_random_vec(5, hidden_size * hidden_size),
            time_decay: pseudo_random_vec(6, num_heads * head_size)
                .into_iter()
                .map(|v| -v.abs() - 0.1)
                .collect(),
            att_ln_weight: vec![1.0f32; hidden_size],
            att_ln_bias: vec![0.0f32; hidden_size],
            ln2_weight: vec![1.0f32; hidden_size],
            ln2_bias: vec![0.0f32; hidden_size],
            channel_mix_x: vec![0.5f32; hidden_size],
            ffn_receptance_weight: pseudo_random_vec(7, hidden_size * hidden_size),
            ffn_key_weight: pseudo_random_vec(8, intermediate_size * hidden_size),
            ffn_value_weight: pseudo_random_vec(9, hidden_size * intermediate_size),
        }
    }

    #[test]
    fn test_rwkv_block_forward_one() {
        let block = make_test_block();
        let mut state = RwkvLayerState::new(block.hidden_size, block.num_heads, block.head_size);
        let input = pseudo_random_vec(42, block.hidden_size);

        let output = block.forward_one(&input, &mut state);

        assert_eq!(output.len(), block.hidden_size, "output length should be hidden_size");
        for (i, &v) in output.iter().enumerate() {
            assert!(v.is_finite(), "output[{i}] = {v} is not finite");
        }
    }

    #[test]
    fn test_rwkv_block_forward_seq() {
        let block = make_test_block();
        let seq_len = 4usize;
        let mut state = RwkvLayerState::new(block.hidden_size, block.num_heads, block.head_size);
        let input = pseudo_random_vec(7, seq_len * block.hidden_size);

        let output = block.forward_seq(&input, seq_len, &mut state);

        assert_eq!(
            output.len(),
            seq_len * block.hidden_size,
            "output length should be seq_len * hidden_size"
        );
        for (i, &v) in output.iter().enumerate() {
            assert!(v.is_finite(), "output[{i}] = {v} is not finite");
        }
    }

    #[test]
    fn test_rwkv_block_seq_matches_steps() {
        let block = make_test_block();
        let seq_len = 4usize;
        let h = block.hidden_size;

        let input = pseudo_random_vec(99, seq_len * h);

        // Run forward_seq
        let mut state_seq = RwkvLayerState::new(h, block.num_heads, block.head_size);
        let seq_output = block.forward_seq(&input, seq_len, &mut state_seq);

        // Run forward_one step-by-step
        let mut state_steps = RwkvLayerState::new(h, block.num_heads, block.head_size);
        let mut step_outputs = Vec::with_capacity(seq_len * h);
        for t in 0..seq_len {
            let token = &input[t * h..(t + 1) * h];
            let out = block.forward_one(token, &mut state_steps);
            step_outputs.extend_from_slice(&out);
        }

        assert_eq!(seq_output.len(), step_outputs.len());

        let max_diff = seq_output
            .iter()
            .zip(step_outputs.iter())
            .map(|(&a, &b)| (a - b).abs())
            .fold(0.0f32, f32::max);

        assert!(
            max_diff < 1e-4,
            "seq and step-by-step outputs differ by {max_diff} (threshold 1e-4)"
        );
    }

    #[test]
    fn test_layernorm_basic() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let weight = vec![1.0; 4];
        let bias = vec![0.0; 4];
        let out = layernorm(&x, &weight, &bias, 1e-5, 4);

        // Mean = 2.5, should be centered
        let mean: f32 = out.iter().sum::<f32>() / 4.0;
        assert!(mean.abs() < 1e-5, "layernorm output should be zero-mean, got {mean}");

        // Variance should be ~1
        let var: f32 = out.iter().map(|v| v * v).sum::<f32>() / 4.0;
        assert!(
            (var - 1.0).abs() < 0.1,
            "layernorm output variance should be ~1, got {var}"
        );
    }
}
