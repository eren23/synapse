//! RwkvBlock: a single RWKV-7 "Goose" layer.
//!
//! Implements the full RWKV-7 block with:
//! - 6 per-component token shift lerps (x_r, x_k, x_v, x_w, x_a, x_g)
//! - Low-rank decay (w0 + tanh(xw @ w1) @ w2), alpha (a0 + (xa @ a1) @ a2), gate (g1, g2)
//! - Key modulation (k_k, k_a) and R-K coupling (r_k)
//! - WKV7 recurrence with feedback term
//! - Squared ReLU FFN with single token shift

use crate::ops::matmul::matmul_t;
use crate::ssm::rwkv_state::RwkvLayerState;

/// A single RWKV-7 "Goose" block.
pub struct RwkvBlock {
    pub hidden_size: usize,
    pub num_heads: usize,
    pub head_size: usize,
    pub intermediate_size: usize,
    pub decay_rank: usize,
    pub alpha_rank: usize,
    pub gate_rank: usize,
    pub norm_eps: f32,

    // Pre-attention LayerNorm
    pub ln1_weight: Vec<f32>,    // [h]
    pub ln1_bias: Vec<f32>,      // [h]

    // Per-component token shift lerps
    pub x_r: Vec<f32>,  // [h]
    pub x_k: Vec<f32>,  // [h]
    pub x_v: Vec<f32>,  // [h]
    pub x_w: Vec<f32>,  // [h]
    pub x_a: Vec<f32>,  // [h]
    pub x_g: Vec<f32>,  // [h]

    // Linear projections (bias-free)
    pub r_proj: Vec<f32>,  // [h, h]
    pub k_proj: Vec<f32>,  // [h, h]
    pub v_proj: Vec<f32>,  // [h, h]
    pub o_proj: Vec<f32>,  // [h, h]

    // Decay low-rank: w = exp(-0.606531 * sigmoid(w0 + tanh(xw @ w1) @ w2))
    pub w0: Vec<f32>,     // [h]
    pub w1: Vec<f32>,     // [h, decay_rank]
    pub w2: Vec<f32>,     // [decay_rank, h]

    // Alpha low-rank: a = sigmoid(a0 + (xa @ a1) @ a2)
    pub a0: Vec<f32>,     // [h]
    pub a1: Vec<f32>,     // [h, alpha_rank]
    pub a2: Vec<f32>,     // [alpha_rank, h]

    // Gate low-rank: g = sigmoid(xg @ g1) @ g2
    pub g1: Vec<f32>,     // [h, gate_rank]
    pub g2: Vec<f32>,     // [gate_rank, h]

    // Key modulation
    pub k_k: Vec<f32>,    // [h]
    pub k_a: Vec<f32>,    // [h]

    // R-K coupling per head
    pub r_k: Vec<f32>,    // [num_heads, head_size]

    // Attention output GroupNorm
    pub g_norm_weight: Vec<f32>,  // [h]
    pub g_norm_bias: Vec<f32>,    // [h]

    // Pre-FFN LayerNorm
    pub ln2_weight: Vec<f32>,    // [h]
    pub ln2_bias: Vec<f32>,      // [h]

    // FFN
    pub ffn_x_k: Vec<f32>,           // [h]
    // Value residual mixing (layers 1+, empty for layer 0)
    pub v_rank: usize,  // typically 32, 0 if no value residual
    pub v0: Vec<f32>,   // [h] or empty
    pub v1: Vec<f32>,   // [v_rank, h] (transposed for matmul_t) or empty
    pub v2: Vec<f32>,   // [h, v_rank] (transposed for matmul_t) or empty

    pub ffn_key_weight: Vec<f32>,     // [intermediate, h]
    pub ffn_value_weight: Vec<f32>,   // [h, intermediate]
}

#[inline]
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

#[inline]
fn tanh_f32(x: f32) -> f32 {
    x.tanh()
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

/// GroupNorm per head: normalize each head_size-chunk independently.
fn group_norm_heads(x: &[f32], weight: &[f32], bias: &[f32], num_heads: usize, head_size: usize, eps: f32) -> Vec<f32> {
    let h = num_heads * head_size;
    let mut out = vec![0.0f32; h];
    for head in 0..num_heads {
        let off = head * head_size;
        let chunk = &x[off..off + head_size];
        let mean: f32 = chunk.iter().sum::<f32>() / head_size as f32;
        let var: f32 = chunk.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / head_size as f32;
        let inv_std = 1.0 / (var + eps).sqrt();
        for j in 0..head_size {
            out[off + j] = (chunk[j] - mean) * inv_std * weight[off + j] + bias[off + j];
        }
    }
    out
}

/// L2-normalize a vector per head, scaling by k_k.
fn l2_norm_per_head(x: &[f32], scale: &[f32], num_heads: usize, head_size: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; num_heads * head_size];
    for head in 0..num_heads {
        let off = head * head_size;
        let chunk = &x[off..off + head_size];
        let norm: f32 = chunk.iter().map(|v| v * v).sum::<f32>().sqrt().max(1e-12);
        for j in 0..head_size {
            out[off + j] = (chunk[j] / norm) * scale[off + j];
        }
    }
    out
}

impl RwkvBlock {
    /// Process a single token `[hidden_size]` -> `(hidden_out, v_out)`.
    ///
    /// `v_first`: value from layer 0 (None for layer 0, Some for layers 1+).
    /// Returns `(hidden_out, v)` where `v` is the raw value vector (for v_first tracking).
    pub fn forward_one(&self, hidden: &[f32], state: &mut RwkvLayerState, v_first: Option<&[f32]>) -> (Vec<f32>, Vec<f32>) {
        let h = self.hidden_size;
        let nh = self.num_heads;
        let hs = self.head_size;

        // ── Time Mixing ──────────────────────────────────────────────

        // 1. LayerNorm
        let normed = layernorm(hidden, &self.ln1_weight, &self.ln1_bias, self.norm_eps, h);

        // 2. Token shift: xx = prev - normed (shifted - current)
        //    Per-component blend: xr = normed + xx * x_r, etc.
        let mut xr = vec![0.0f32; h];
        let mut xk = vec![0.0f32; h];
        let mut xv = vec![0.0f32; h];
        let mut xw = vec![0.0f32; h];
        let mut xa = vec![0.0f32; h];
        let mut xg = vec![0.0f32; h];
        for i in 0..h {
            let xx = state.time_mix_prev[i] - normed[i];
            xr[i] = normed[i] + xx * self.x_r[i];
            xk[i] = normed[i] + xx * self.x_k[i];
            xv[i] = normed[i] + xx * self.x_v[i];
            xw[i] = normed[i] + xx * self.x_w[i];
            xa[i] = normed[i] + xx * self.x_a[i];
            xg[i] = normed[i] + xx * self.x_g[i];
        }
        state.time_mix_prev.copy_from_slice(&normed);

        // 3. Linear projections
        let r = matmul_t(&xr, &self.r_proj, 1, h, h);
        let k = matmul_t(&xk, &self.k_proj, 1, h, h);
        let mut v = matmul_t(&xv, &self.v_proj, 1, h, h);
        let v_out = v.clone(); // save raw v for v_first tracking

        // 3b. Value residual mixing (layers 1+)
        if let Some(vf) = v_first {
            if !self.v0.is_empty() {
                let vr = self.v_rank;
                let xv_v1 = matmul_t(&xv, &self.v1, 1, h, vr);
                let v_mix = matmul_t(&xv_v1, &self.v2, 1, vr, h);
                for i in 0..h {
                    let lerp = sigmoid(self.v0[i] + v_mix[i]);
                    v[i] = v[i] + (vf[i] - v[i]) * lerp;
                }
            }
        }

        // 4. Decay: w = exp(-0.606531 * sigmoid(w0 + tanh(xw @ w1) @ w2))
        let xw_w1 = matmul_t(&xw, &self.w1, 1, h, self.decay_rank);
        let xw_w1_tanh: Vec<f32> = xw_w1.iter().map(|&v| tanh_f32(v)).collect();
        let w_proj = matmul_t(&xw_w1_tanh, &self.w2, 1, self.decay_rank, h);
        let w: Vec<f32> = (0..h)
            .map(|i| (-0.606531f32 * sigmoid(self.w0[i] + w_proj[i])).exp())
            .collect();

        // 5. Alpha: a = sigmoid(a0 + (xa @ a1) @ a2)
        let xa_a1 = matmul_t(&xa, &self.a1, 1, h, self.alpha_rank);
        let a_proj = matmul_t(&xa_a1, &self.a2, 1, self.alpha_rank, h);
        let alpha: Vec<f32> = (0..h)
            .map(|i| sigmoid(self.a0[i] + a_proj[i]))
            .collect();

        // 6. Gate: g = sigmoid(xg @ g1) @ g2
        let xg_g1 = matmul_t(&xg, &self.g1, 1, h, self.gate_rank);
        let xg_g1_sig: Vec<f32> = xg_g1.iter().map(|&v| sigmoid(v)).collect();
        let g = matmul_t(&xg_g1_sig, &self.g2, 1, self.gate_rank, h);

        // 7. Key modulation:
        //    kk = L2_norm(k * k_k) per head  — used for ab feedback in WKV
        //    k = k * (1 + (a-1) * k_a)       — modifies k used for vk outer product
        let k_scaled: Vec<f32> = (0..h).map(|i| k[i] * self.k_k[i]).collect();
        let kk = l2_norm_per_head(&k_scaled, &vec![1.0f32; h], nh, hs);
        let k_mod: Vec<f32> = (0..h)
            .map(|i| k[i] * (1.0 + (alpha[i] - 1.0) * self.k_a[i]))
            .collect();

        // 8. Per-head WKV7 recurrence
        //    Uses kk (normalized) for ab feedback, k_mod for vk outer product
        //    Python: vk = outer(v, k), ab = outer(-kk, kk*a)
        //    We pass kk to the kernel (used for ab), but need k_mod for vk.
        //    The kernel uses k for both ab and vk, so we need to do this:
        //    Pass kk for ab computation, but the vk term uses k_mod.
        //    Simplest: compute vk and ab separately, or modify the kernel.
        //    For correctness, let's modify in-place: the kernel's k param is used for
        //    ab (should be kk) and vk (should be k_mod). These differ.
        //    → We must compute ab outside the kernel, or split the kernel.
        //    For now, let's do it inline per head.
        let mut wkv_out = vec![0.0f32; h];
        for head in 0..nh {
            let off = head * hs;
            let r_head = &r[off..off + hs];
            let kk_head = &kk[off..off + hs];
            let k_mod_head = &k_mod[off..off + hs];
            let v_head = &v[off..off + hs];
            let w_head = &w[off..off + hs];
            let a_head = &alpha[off..off + hs];
            let wkv_state = &mut state.wkv_state[head * hs * hs..(head + 1) * hs * hs];

            // Inline WKV7 step with separate kk (for ab) and k_mod (for vk)
            // ka[j] = kk[j] * a[j]
            let ka: Vec<f32> = (0..hs).map(|j| kk_head[j] * a_head[j]).collect();

            // sdk[d] = sum_l(state[d,l] * kk[l])
            let mut sdk = vec![0.0f32; hs];
            for d in 0..hs {
                let mut dot = 0.0f32;
                for l in 0..hs {
                    dot += wkv_state[d * hs + l] * kk_head[l];
                }
                sdk[d] = dot;
            }

            // state[d,j] = w[j]*state[d,j] - sdk[d]*ka[j] + v[d]*k_mod[j]
            for d in 0..hs {
                let v_d = v_head[d];
                for j in 0..hs {
                    wkv_state[d * hs + j] = w_head[j] * wkv_state[d * hs + j]
                        - sdk[d] * ka[j]
                        + v_d * k_mod_head[j];
                }
            }

            // output[d] = sum_j(state[d,j] * r[j])
            for d in 0..hs {
                let mut sum = 0.0f32;
                for j in 0..hs {
                    sum += wkv_state[d * hs + j] * r_head[j];
                }
                wkv_out[off + d] = sum;
            }
        }

        // 9. GroupNorm (per-head, eps = head_size * 1e-5)
        let gn_eps = hs as f32 * 1e-5;
        let normed_wkv = group_norm_heads(&wkv_out, &self.g_norm_weight, &self.g_norm_bias, nh, hs, gn_eps);

        // 10. R-K coupling: sum(r*k*r_k) per head → scalar, then * v
        //    Python: ((r*k*self.r_k).sum(dim=-1, keepdim=True) * v)
        let mut rk_contrib = vec![0.0f32; h];
        for head in 0..nh {
            let off = head * hs;
            let mut dot = 0.0f32;
            for j in 0..hs {
                dot += r[off + j] * k[off + j] * self.r_k[head * hs + j];
            }
            for j in 0..hs {
                rk_contrib[off + j] = dot * v[off + j];
            }
        }

        // 11. Combine: output = o_proj((normed_wkv + rk_contrib) * g)
        let gated: Vec<f32> = (0..h)
            .map(|i| (normed_wkv[i] + rk_contrib[i]) * g[i])
            .collect();
        let time_out = matmul_t(&gated, &self.o_proj, 1, h, h);

        // 12. Residual
        let hidden_after_time: Vec<f32> = (0..h)
            .map(|i| hidden[i] + time_out[i])
            .collect();

        // ── Channel Mixing (FFN) ─────────────────────────────────────

        // 1. LayerNorm
        let normed2 = layernorm(&hidden_after_time, &self.ln2_weight, &self.ln2_bias, self.norm_eps, h);

        // 2. Token shift (single lerp for FFN): xx = prev - current
        let xk_ffn: Vec<f32> = (0..h)
            .map(|i| {
                let xx = state.channel_mix_prev[i] - normed2[i];
                normed2[i] + xx * self.ffn_x_k[i]
            })
            .collect();
        state.channel_mix_prev.copy_from_slice(&normed2);

        // 3. k = relu(ffn_key @ xk)², v = ffn_value @ k
        let k_ffn = matmul_t(&xk_ffn, &self.ffn_key_weight, 1, h, self.intermediate_size);
        let k_sq: Vec<f32> = k_ffn.iter().map(|&x| {
            let relu = if x > 0.0 { x } else { 0.0 };
            relu * relu
        }).collect();
        let v_ffn = matmul_t(&k_sq, &self.ffn_value_weight, 1, self.intermediate_size, h);

        // 4. Residual
        let result: Vec<f32> = (0..h).map(|i| hidden_after_time[i] + v_ffn[i]).collect();
        (result, v_out)
    }

    /// Process a full sequence `[seq_len * hidden_size]` -> `[seq_len * hidden_size]`.
    /// Returns `(output, last_v)` where `last_v` is the raw v from the last token.
    pub fn forward_seq(&self, hidden: &[f32], seq_len: usize, state: &mut RwkvLayerState, v_first: Option<&[f32]>) -> (Vec<f32>, Vec<f32>) {
        let h = self.hidden_size;
        let mut output = Vec::with_capacity(seq_len * h);
        let mut last_v = vec![0.0f32; h];
        for t in 0..seq_len {
            let token = &hidden[t * h..(t + 1) * h];
            let (out, v) = self.forward_one(token, state, v_first);
            output.extend_from_slice(&out);
            last_v = v;
        }
        (output, last_v)
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
        let decay_rank = 8usize;
        let alpha_rank = 8usize;
        let gate_rank = 16usize;

        RwkvBlock {
            hidden_size,
            num_heads,
            head_size,
            intermediate_size,
            decay_rank,
            alpha_rank,
            gate_rank,
            norm_eps: 1e-5,
            ln1_weight: vec![1.0f32; hidden_size],
            ln1_bias: vec![0.0f32; hidden_size],
            x_r: pseudo_random_vec(10, hidden_size),
            x_k: pseudo_random_vec(11, hidden_size),
            x_v: pseudo_random_vec(12, hidden_size),
            x_w: pseudo_random_vec(13, hidden_size),
            x_a: pseudo_random_vec(14, hidden_size),
            x_g: pseudo_random_vec(15, hidden_size),
            r_proj: pseudo_random_vec(1, hidden_size * hidden_size),
            k_proj: pseudo_random_vec(2, hidden_size * hidden_size),
            v_proj: pseudo_random_vec(3, hidden_size * hidden_size),
            o_proj: pseudo_random_vec(5, hidden_size * hidden_size),
            w0: pseudo_random_vec(20, hidden_size),
            w1: pseudo_random_vec(21, hidden_size * decay_rank),
            w2: pseudo_random_vec(22, decay_rank * hidden_size),
            a0: pseudo_random_vec(30, hidden_size),
            a1: pseudo_random_vec(31, hidden_size * alpha_rank),
            a2: pseudo_random_vec(32, alpha_rank * hidden_size),
            g1: pseudo_random_vec(40, hidden_size * gate_rank),
            g2: pseudo_random_vec(41, gate_rank * hidden_size),
            k_k: vec![1.0f32; hidden_size],
            k_a: vec![1.0f32; hidden_size],
            r_k: pseudo_random_vec(50, num_heads * head_size),
            g_norm_weight: vec![1.0f32; hidden_size],
            g_norm_bias: vec![0.0f32; hidden_size],
            ln2_weight: vec![1.0f32; hidden_size],
            ln2_bias: vec![0.0f32; hidden_size],
            v_rank: 0,
            v0: vec![], v1: vec![], v2: vec![],
            ffn_x_k: pseudo_random_vec(60, hidden_size),
            ffn_key_weight: pseudo_random_vec(8, intermediate_size * hidden_size),
            ffn_value_weight: pseudo_random_vec(9, hidden_size * intermediate_size),
        }
    }

    #[test]
    fn test_rwkv_block_forward_one() {
        let block = make_test_block();
        let mut state = RwkvLayerState::new(block.hidden_size, block.num_heads, block.head_size);
        let input = pseudo_random_vec(42, block.hidden_size);

        let (output, _v) = block.forward_one(&input, &mut state, None);

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

        let (output, _v) = block.forward_seq(&input, seq_len, &mut state, None);

        assert_eq!(output.len(), seq_len * block.hidden_size);
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

        let mut state_seq = RwkvLayerState::new(h, block.num_heads, block.head_size);
        let (seq_output, _) = block.forward_seq(&input, seq_len, &mut state_seq, None);

        let mut state_steps = RwkvLayerState::new(h, block.num_heads, block.head_size);
        let mut step_outputs = Vec::with_capacity(seq_len * h);
        for t in 0..seq_len {
            let token = &input[t * h..(t + 1) * h];
            let (out, _) = block.forward_one(token, &mut state_steps, None);
            step_outputs.extend_from_slice(&out);
        }

        assert_eq!(seq_output.len(), step_outputs.len());
        let max_diff = seq_output.iter().zip(step_outputs.iter())
            .map(|(&a, &b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(max_diff < 1e-4, "seq and step-by-step differ by {max_diff}");
    }
}
