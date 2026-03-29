//! MambaBlock: a single Mamba layer.
//!
//! Implements the full Mamba block forward pass:
//! RMSNorm → in_proj (split x, z) → Conv1d → SiLU → SSM → Gate (silu(z)*y) → out_proj → residual.

use crate::ops::activation::{silu, softplus};
use crate::ops::matmul::matmul_t;
use crate::ops::pure_rust_ops::rmsnorm;
use super::selective_scan::selective_scan_step;
use super::state::MambaLayerState;

/// A single Mamba block (one layer).
pub struct MambaBlock {
    pub d_model: usize,
    pub d_inner: usize,
    pub d_state: usize,
    pub d_conv: usize,
    pub dt_rank: usize,
    pub norm_weight: Vec<f32>,    // [d_model]
    pub norm_eps: f32,
    pub in_proj_weight: Vec<f32>, // [2*d_inner, d_model]
    pub in_proj_bias: Vec<f32>,   // may be empty
    pub conv1d_weight: Vec<f32>,  // [d_inner, d_conv]
    pub conv1d_bias: Vec<f32>,    // [d_inner]
    pub x_proj_weight: Vec<f32>,  // [dt_rank + 2*d_state, d_inner]
    pub dt_proj_weight: Vec<f32>, // [d_inner, dt_rank]
    pub dt_proj_bias: Vec<f32>,   // [d_inner]
    pub a_log: Vec<f32>,          // [d_inner, d_state]
    pub d_param: Vec<f32>,        // [d_inner]
    pub out_proj_weight: Vec<f32>,// [d_model, d_inner]
    pub out_proj_bias: Vec<f32>,  // may be empty
}

impl MambaBlock {
    /// Apply conv1d to a single timestep's input vector.
    ///
    /// Shifts the conv state left (dropping oldest), inserts new `x`, then
    /// computes dot-product with learned kernel + bias for each channel.
    ///
    /// `state.conv_state` layout: `[d_inner, d_conv]` (row-major).
    /// For channel `i`, `conv_state[i*d_conv .. (i+1)*d_conv]` is the ring buffer,
    /// with `[d_conv-1]` being the most recent slot after the shift.
    pub fn conv1d_step(&self, x: &[f32], state: &mut MambaLayerState) -> Vec<f32> {
        let d_inner = self.d_inner;
        let d_conv = self.d_conv;
        let mut out = vec![0.0f32; d_inner];

        for i in 0..d_inner {
            let buf = &mut state.conv_state[i * d_conv..(i + 1) * d_conv];
            // Shift left: [a, b, c, d] → [b, c, d, _]
            buf.copy_within(1.., 0);
            // Insert new value at end
            buf[d_conv - 1] = x[i];
            // Dot product with kernel
            let w = &self.conv1d_weight[i * d_conv..(i + 1) * d_conv];
            let sum: f32 = buf.iter().zip(w.iter()).map(|(&b, &k)| b * k).sum();
            out[i] = sum + self.conv1d_bias[i];
        }

        out
    }

    /// Project `x` through x_proj, then dt_proj linear layer, then run one SSM step.
    ///
    /// x_proj outputs `[dt_rank + 2*d_state]`: the first `dt_rank` values are the
    /// low-rank dt input, followed by B and C vectors. The dt input is then projected
    /// through dt_proj `[d_inner, dt_rank]` to produce the per-channel delta.
    ///
    /// Returns `y` of shape `[d_inner]`.
    pub fn ssm_forward_step(&self, x: &[f32], state: &mut MambaLayerState) -> Vec<f32> {
        let d_inner = self.d_inner;
        let d_state = self.d_state;
        let dt_rank = self.dt_rank;

        // x_proj: [1, d_inner] × [dt_rank+2*d_state, d_inner]^T → [1, dt_rank+2*d_state]
        let x_proj_out = matmul_t(x, &self.x_proj_weight, 1, d_inner, dt_rank + 2 * d_state);

        let dt_input = &x_proj_out[0..dt_rank];
        let b_slice = &x_proj_out[dt_rank..dt_rank + d_state];
        let c_slice = &x_proj_out[dt_rank + d_state..dt_rank + 2 * d_state];

        // dt_proj: [1, dt_rank] × [d_inner, dt_rank]^T → [1, d_inner]
        let dt_projected = matmul_t(dt_input, &self.dt_proj_weight, 1, dt_rank, d_inner);

        // Apply softplus to get positive delta values
        let delta: Vec<f32> = (0..d_inner)
            .map(|i| softplus(dt_projected[i] + self.dt_proj_bias[i]))
            .collect();

        selective_scan_step(
            x,
            &delta,
            &self.a_log,
            b_slice,
            c_slice,
            &self.d_param,
            &mut state.ssm_state,
        )
    }

    /// Process a single token `[d_model]` → `[d_model]`.
    pub fn forward_one(&self, hidden: &[f32], state: &mut MambaLayerState) -> Vec<f32> {
        let d_model = self.d_model;
        let d_inner = self.d_inner;

        // 1. RMSNorm
        let normed = rmsnorm(hidden, &self.norm_weight, self.norm_eps, d_model);

        // 2. in_proj: [1, d_model] × [2*d_inner, d_model]^T → [1, 2*d_inner]
        let proj = matmul_t(&normed, &self.in_proj_weight, 1, d_model, 2 * d_inner);

        // Add bias if present
        let proj = if self.in_proj_bias.is_empty() {
            proj
        } else {
            proj.iter().zip(self.in_proj_bias.iter()).map(|(&p, &b)| p + b).collect()
        };

        // 3. Split into x and z
        let x_proj = &proj[0..d_inner];
        let z_proj = &proj[d_inner..2 * d_inner];

        // 4. Conv1d step
        let x_conv = self.conv1d_step(x_proj, state);

        // 5. SiLU activation
        let x_act: Vec<f32> = x_conv.iter().map(|&v| silu(v)).collect();

        // 6. SSM step
        let y = self.ssm_forward_step(&x_act, state);

        // 7. Gate: silu(z) * y
        let gated: Vec<f32> = z_proj
            .iter()
            .zip(y.iter())
            .map(|(&z, &yi)| silu(z) * yi)
            .collect();

        // 8. out_proj: [1, d_inner] × [d_model, d_inner]^T → [1, d_model]
        let out = matmul_t(&gated, &self.out_proj_weight, 1, d_inner, d_model);

        let out = if self.out_proj_bias.is_empty() {
            out
        } else {
            out.iter().zip(self.out_proj_bias.iter()).map(|(&o, &b)| o + b).collect()
        };

        // 9. Residual
        out.iter().zip(hidden.iter()).map(|(&o, &h)| o + h).collect()
    }

    /// Process a full sequence `[seq_len * d_model]` → `[seq_len * d_model]`.
    pub fn forward_seq(&self, hidden: &[f32], seq_len: usize, state: &mut MambaLayerState) -> Vec<f32> {
        let d_model = self.d_model;
        let d_inner = self.d_inner;

        // 1. RMSNorm over the full sequence (handles multiple tokens automatically)
        let normed = rmsnorm(hidden, &self.norm_weight, self.norm_eps, d_model);

        // 2. in_proj: [seq_len, d_model] × [2*d_inner, d_model]^T → [seq_len, 2*d_inner]
        let proj = matmul_t(&normed, &self.in_proj_weight, seq_len, d_model, 2 * d_inner);

        // Add bias if present
        let proj: Vec<f32> = if self.in_proj_bias.is_empty() {
            proj
        } else {
            proj.chunks(2 * d_inner)
                .flat_map(|row| row.iter().zip(self.in_proj_bias.iter()).map(|(&p, &b)| p + b))
                .collect()
        };

        // 3. Process each timestep sequentially through conv + SSM
        let mut ssm_outs = Vec::with_capacity(seq_len * d_inner);

        for t in 0..seq_len {
            let x_t = &proj[t * 2 * d_inner..t * 2 * d_inner + d_inner];

            // 4. Conv1d step
            let x_conv = self.conv1d_step(x_t, state);

            // 5. SiLU
            let x_act: Vec<f32> = x_conv.iter().map(|&v| silu(v)).collect();

            // 6. SSM step
            let y_t = self.ssm_forward_step(&x_act, state);
            ssm_outs.extend_from_slice(&y_t);
        }

        // 7. Gate: silu(z[t]) * y[t] for each timestep
        let mut gated = Vec::with_capacity(seq_len * d_inner);
        for t in 0..seq_len {
            let z_t = &proj[t * 2 * d_inner + d_inner..(t + 1) * 2 * d_inner];
            let y_t = &ssm_outs[t * d_inner..(t + 1) * d_inner];
            for (&z, &yi) in z_t.iter().zip(y_t.iter()) {
                gated.push(silu(z) * yi);
            }
        }

        // 8. out_proj: [seq_len, d_inner] × [d_model, d_inner]^T → [seq_len, d_model]
        let out = matmul_t(&gated, &self.out_proj_weight, seq_len, d_inner, d_model);

        let out: Vec<f32> = if self.out_proj_bias.is_empty() {
            out
        } else {
            out.chunks(d_model)
                .flat_map(|row| row.iter().zip(self.out_proj_bias.iter()).map(|(&o, &b)| o + b))
                .collect()
        };

        // 9. Residual: output + input
        out.iter().zip(hidden.iter()).map(|(&o, &h)| o + h).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::ssm::mamba::state::MambaLayerState;

    fn pseudo_random_vec(seed: u64, len: usize) -> Vec<f32> {
        let mut state = seed;
        (0..len)
            .map(|_| {
                state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                let bits = 0x3F800000u32 | ((state >> 41) as u32 & 0x7FFFFF);
                (f32::from_bits(bits) - 1.5) * 0.2
            })
            .collect()
    }

    fn make_test_block() -> MambaBlock {
        let d_model = 16usize;
        let d_inner = 32usize;
        let d_state = 4usize;
        let d_conv = 4usize;
        let dt_rank = 2usize; // ceil(16 / 16) = 1, but use 2 for test variety

        MambaBlock {
            d_model,
            d_inner,
            d_state,
            d_conv,
            dt_rank,
            norm_weight: vec![1.0f32; d_model],
            norm_eps: 1e-5,
            in_proj_weight: pseudo_random_vec(1, 2 * d_inner * d_model),
            in_proj_bias: vec![],
            conv1d_weight: pseudo_random_vec(2, d_inner * d_conv),
            conv1d_bias: vec![0.0f32; d_inner],
            x_proj_weight: pseudo_random_vec(3, (dt_rank + 2 * d_state) * d_inner),
            dt_proj_weight: pseudo_random_vec(4, d_inner * dt_rank),
            dt_proj_bias: vec![0.0f32; d_inner],
            a_log: pseudo_random_vec(5, d_inner * d_state)
                .into_iter()
                .map(|v| -v.abs() - 0.1)
                .collect(),
            d_param: vec![1.0f32; d_inner],
            out_proj_weight: pseudo_random_vec(6, d_model * d_inner),
            out_proj_bias: vec![],
        }
    }

    #[test]
    fn test_mamba_block_forward_one() {
        let block = make_test_block();
        let mut state = MambaLayerState::new(block.d_inner, block.d_state, block.d_conv);
        let input = pseudo_random_vec(42, block.d_model);

        let output = block.forward_one(&input, &mut state);

        assert_eq!(output.len(), block.d_model, "output length should be d_model");
        for (i, &v) in output.iter().enumerate() {
            assert!(v.is_finite(), "output[{i}] = {v} is not finite");
        }
    }

    #[test]
    fn test_mamba_block_forward_seq() {
        let block = make_test_block();
        let seq_len = 4usize;
        let mut state = MambaLayerState::new(block.d_inner, block.d_state, block.d_conv);
        let input = pseudo_random_vec(7, seq_len * block.d_model);

        let output = block.forward_seq(&input, seq_len, &mut state);

        assert_eq!(output.len(), seq_len * block.d_model, "output length should be seq_len * d_model");
        for (i, &v) in output.iter().enumerate() {
            assert!(v.is_finite(), "output[{i}] = {v} is not finite");
        }
    }

    #[test]
    fn test_mamba_block_seq_matches_steps() {
        let block = make_test_block();
        let seq_len = 4usize;
        let d_model = block.d_model;

        let input = pseudo_random_vec(99, seq_len * d_model);

        // Run forward_seq
        let mut state_seq = MambaLayerState::new(block.d_inner, block.d_state, block.d_conv);
        let seq_output = block.forward_seq(&input, seq_len, &mut state_seq);

        // Run forward_one step-by-step
        let mut state_steps = MambaLayerState::new(block.d_inner, block.d_state, block.d_conv);
        let mut step_outputs = Vec::with_capacity(seq_len * d_model);
        for t in 0..seq_len {
            let token = &input[t * d_model..(t + 1) * d_model];
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
}
