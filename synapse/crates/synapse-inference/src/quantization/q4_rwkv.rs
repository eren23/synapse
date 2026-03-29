//! Q4-quantized RWKV-7 model.
//!
//! Quantizes the 6 large linear projections (r_proj, k_proj, v_proj, o_proj,
//! ffn_key_weight, ffn_value_weight) to Q4_0 (4-bit) while keeping SSM-specific
//! parameters (token shift lerps, low-rank matrices, norms) in f32.
//! This reduces model size by ~6.4x for ESP32/WASM deployment.

use std::cell::RefCell;

use crate::config::ModelConfig;
use crate::model::causal_lm::ModelOutput;
use crate::model::traits::{Model, ModelState};
use crate::ops::matmul::matmul_t;
use crate::quantization::q4::Q4Linear;
use crate::ssm::rwkv_config::RwkvConfig;
use crate::ssm::rwkv_model::RwkvModel;
use crate::ops::activation::sigmoid;
use crate::ssm::rwkv_state::{RwkvLayerState, RwkvState};

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

/// A single RWKV-7 block with Q4-quantized linear projections.
pub struct Q4RwkvBlock {
    pub hidden_size: usize,
    pub num_heads: usize,
    pub head_size: usize,
    pub intermediate_size: usize,
    pub decay_rank: usize,
    pub alpha_rank: usize,
    pub gate_rank: usize,
    pub norm_eps: f32,

    // Pre-attention LayerNorm (f32)
    pub ln1_weight: Vec<f32>,    // [h]
    pub ln1_bias: Vec<f32>,      // [h]

    // Per-component token shift lerps (f32)
    pub x_r: Vec<f32>,  // [h]
    pub x_k: Vec<f32>,  // [h]
    pub x_v: Vec<f32>,  // [h]
    pub x_w: Vec<f32>,  // [h]
    pub x_a: Vec<f32>,  // [h]
    pub x_g: Vec<f32>,  // [h]

    // Linear projections — Q4 quantized
    pub r_proj: Q4Linear,  // [h, h]
    pub k_proj: Q4Linear,  // [h, h]
    pub v_proj: Q4Linear,  // [h, h]
    pub o_proj: Q4Linear,  // [h, h]

    // Decay low-rank (f32)
    pub w0: Vec<f32>,     // [h]
    pub w1: Vec<f32>,     // [h, decay_rank]
    pub w2: Vec<f32>,     // [decay_rank, h]

    // Alpha low-rank (f32)
    pub a0: Vec<f32>,     // [h]
    pub a1: Vec<f32>,     // [h, alpha_rank]
    pub a2: Vec<f32>,     // [alpha_rank, h]

    // Gate low-rank (f32)
    pub g1: Vec<f32>,     // [h, gate_rank]
    pub g2: Vec<f32>,     // [gate_rank, h]

    // Key modulation (f32)
    pub k_k: Vec<f32>,    // [h]
    pub k_a: Vec<f32>,    // [h]

    // R-K coupling per head (f32)
    pub r_k: Vec<f32>,    // [num_heads, head_size]

    // Attention output GroupNorm (f32)
    pub g_norm_weight: Vec<f32>,  // [h]
    pub g_norm_bias: Vec<f32>,    // [h]

    // Pre-FFN LayerNorm (f32)
    pub ln2_weight: Vec<f32>,    // [h]
    pub ln2_bias: Vec<f32>,      // [h]

    // FFN token shift (f32)
    pub ffn_x_k: Vec<f32>,           // [h]

    // Value residual mixing (f32)
    pub v_rank: usize,
    pub v0: Vec<f32>,   // [h] or empty
    pub v1: Vec<f32>,   // [v_rank, h] or empty
    pub v2: Vec<f32>,   // [h, v_rank] or empty

    // FFN projections — Q4 quantized
    pub ffn_key_weight: Q4Linear,     // [intermediate, h]
    pub ffn_value_weight: Q4Linear,   // [h, intermediate]
}

impl Q4RwkvBlock {
    /// Process a single token `[hidden_size]` -> `(hidden_out, v_out)`.
    ///
    /// `v_first`: value from layer 0 (None for layer 0, Some for layers 1+).
    /// Returns `(hidden_out, v)` where `v` is the raw value vector (for v_first tracking).
    pub fn forward_one(&self, hidden: &[f32], state: &mut RwkvLayerState, v_first: Option<&[f32]>) -> (Vec<f32>, Vec<f32>) {
        let h = self.hidden_size;
        let nh = self.num_heads;
        let hs = self.head_size;

        // -- Time Mixing --

        // 1. LayerNorm
        let normed = layernorm(hidden, &self.ln1_weight, &self.ln1_bias, self.norm_eps, h);

        // 2. Token shift: xx = prev - normed
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

        // 3. Linear projections (Q4)
        let r = self.r_proj.forward(&xr, 1);
        let k = self.k_proj.forward(&xk, 1);
        let mut v = self.v_proj.forward(&xv, 1);
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

        // 7. Key modulation
        let k_scaled: Vec<f32> = (0..h).map(|i| k[i] * self.k_k[i]).collect();
        let kk = l2_norm_per_head(&k_scaled, &vec![1.0f32; h], nh, hs);
        let k_mod: Vec<f32> = (0..h)
            .map(|i| k[i] * (1.0 + (alpha[i] - 1.0) * self.k_a[i]))
            .collect();

        // 8. Per-head WKV7 recurrence
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

            let ka: Vec<f32> = (0..hs).map(|j| kk_head[j] * a_head[j]).collect();

            let mut sdk = vec![0.0f32; hs];
            for d in 0..hs {
                let mut dot = 0.0f32;
                for l in 0..hs {
                    dot += wkv_state[d * hs + l] * kk_head[l];
                }
                sdk[d] = dot;
            }

            for d in 0..hs {
                let v_d = v_head[d];
                for j in 0..hs {
                    wkv_state[d * hs + j] = w_head[j] * wkv_state[d * hs + j]
                        - sdk[d] * ka[j]
                        + v_d * k_mod_head[j];
                }
            }

            for d in 0..hs {
                let mut sum = 0.0f32;
                for j in 0..hs {
                    sum += wkv_state[d * hs + j] * r_head[j];
                }
                wkv_out[off + d] = sum;
            }
        }

        // 9. GroupNorm (per-head)
        let gn_eps = hs as f32 * 1e-5;
        let normed_wkv = group_norm_heads(&wkv_out, &self.g_norm_weight, &self.g_norm_bias, nh, hs, gn_eps);

        // 10. R-K coupling
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
        let time_out = self.o_proj.forward(&gated, 1);

        // 12. Residual
        let hidden_after_time: Vec<f32> = (0..h)
            .map(|i| hidden[i] + time_out[i])
            .collect();

        // -- Channel Mixing (FFN) --

        // 1. LayerNorm
        let normed2 = layernorm(&hidden_after_time, &self.ln2_weight, &self.ln2_bias, self.norm_eps, h);

        // 2. Token shift
        let xk_ffn: Vec<f32> = (0..h)
            .map(|i| {
                let xx = state.channel_mix_prev[i] - normed2[i];
                normed2[i] + xx * self.ffn_x_k[i]
            })
            .collect();
        state.channel_mix_prev.copy_from_slice(&normed2);

        // 3. k = relu(ffn_key @ xk)^2, v = ffn_value @ k (Q4)
        let k_ffn = self.ffn_key_weight.forward(&xk_ffn, 1);
        let k_sq: Vec<f32> = k_ffn.iter().map(|&x| {
            let relu = if x > 0.0 { x } else { 0.0 };
            relu * relu
        }).collect();
        let v_ffn = self.ffn_value_weight.forward(&k_sq, 1);

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

/// Q4-quantized RWKV-7 language model.
pub struct Q4RwkvModel {
    pub config: RwkvConfig,
    pub embed_tokens: Vec<f32>,
    pub pre_ln_weight: Option<Vec<f32>>,
    pub pre_ln_bias: Option<Vec<f32>>,
    pub blocks: Vec<Q4RwkvBlock>,
    pub final_norm_weight: Vec<f32>,
    pub final_norm_bias: Vec<f32>,
    pub lm_head_weight: Vec<f32>,
    state: RefCell<RwkvState>,
}

impl Q4RwkvModel {
    /// Quantize a full-precision RwkvModel to Q4.
    pub fn from_f32(model: &RwkvModel) -> Self {
        let config = model.config.clone();
        let blocks: Vec<Q4RwkvBlock> = model.blocks.iter().map(|block| {
            let h = block.hidden_size;
            let inter = block.intermediate_size;
            Q4RwkvBlock {
                hidden_size: h,
                num_heads: block.num_heads,
                head_size: block.head_size,
                intermediate_size: inter,
                decay_rank: block.decay_rank,
                alpha_rank: block.alpha_rank,
                gate_rank: block.gate_rank,
                norm_eps: block.norm_eps,
                ln1_weight: block.ln1_weight.clone(),
                ln1_bias: block.ln1_bias.clone(),
                x_r: block.x_r.clone(),
                x_k: block.x_k.clone(),
                x_v: block.x_v.clone(),
                x_w: block.x_w.clone(),
                x_a: block.x_a.clone(),
                x_g: block.x_g.clone(),
                r_proj: Q4Linear::from_f32(&block.r_proj, h, h),
                k_proj: Q4Linear::from_f32(&block.k_proj, h, h),
                v_proj: Q4Linear::from_f32(&block.v_proj, h, h),
                o_proj: Q4Linear::from_f32(&block.o_proj, h, h),
                w0: block.w0.clone(),
                w1: block.w1.clone(),
                w2: block.w2.clone(),
                a0: block.a0.clone(),
                a1: block.a1.clone(),
                a2: block.a2.clone(),
                g1: block.g1.clone(),
                g2: block.g2.clone(),
                k_k: block.k_k.clone(),
                k_a: block.k_a.clone(),
                r_k: block.r_k.clone(),
                g_norm_weight: block.g_norm_weight.clone(),
                g_norm_bias: block.g_norm_bias.clone(),
                ln2_weight: block.ln2_weight.clone(),
                ln2_bias: block.ln2_bias.clone(),
                ffn_x_k: block.ffn_x_k.clone(),
                v_rank: block.v_rank,
                v0: block.v0.clone(),
                v1: block.v1.clone(),
                v2: block.v2.clone(),
                ffn_key_weight: Q4Linear::from_f32(&block.ffn_key_weight, inter, h),
                ffn_value_weight: Q4Linear::from_f32(&block.ffn_value_weight, h, inter),
            }
        }).collect();

        let state = RwkvState::new(
            config.num_layers,
            config.hidden_size,
            config.num_heads,
            config.head_size,
        );

        Q4RwkvModel {
            config,
            embed_tokens: model.embed_tokens.clone(),
            pre_ln_weight: model.pre_ln_weight.clone(),
            pre_ln_bias: model.pre_ln_bias.clone(),
            blocks,
            final_norm_weight: model.final_norm_weight.clone(),
            final_norm_bias: model.final_norm_bias.clone(),
            lm_head_weight: model.lm_head_weight.clone(),
            state: RefCell::new(state),
        }
    }

    pub fn reset_state(&self) {
        self.state.borrow_mut().reset();
    }

    pub fn prefill(&self, token_ids: &[u32]) -> ModelOutput {
        let h = self.config.hidden_size;
        let vocab = self.config.vocab_size;
        let seq_len = token_ids.len();

        // 1. Embedding lookup
        let mut hidden = vec![0.0f32; seq_len * h];
        for (t, &id) in token_ids.iter().enumerate() {
            let id = id as usize;
            if id < vocab {
                let src = &self.embed_tokens[id * h..(id + 1) * h];
                hidden[t * h..(t + 1) * h].copy_from_slice(src);
            }
        }

        // 1b. Pre-LayerNorm
        if let (Some(w), Some(b)) = (&self.pre_ln_weight, &self.pre_ln_bias) {
            hidden = layernorm(&hidden, w, b, self.config.norm_eps as f32, h);
        }

        // 2. Process through all blocks
        let mut state = self.state.borrow_mut();
        let mut v_first: Option<Vec<f32>> = None;
        for (i, block) in self.blocks.iter().enumerate() {
            let vf = v_first.as_deref();
            let (h_out, v_out) = block.forward_seq(&hidden, seq_len, &mut state.layers[i], vf);
            hidden = h_out;
            if i == 0 { v_first = Some(v_out); }
        }
        state.advance(seq_len);

        // 3. Final LayerNorm on last token
        let last_hidden = &hidden[(seq_len - 1) * h..seq_len * h];
        let normed = layernorm(
            last_hidden,
            &self.final_norm_weight,
            &self.final_norm_bias,
            self.config.norm_eps as f32,
            h,
        );

        // 4. LM head
        let logits = matmul_t(&normed, &self.lm_head_weight, 1, h, vocab);

        ModelOutput {
            logits,
            shape: [1, 1, vocab],
        }
    }

    pub fn decode_one(&self, token: u32) -> ModelOutput {
        let h = self.config.hidden_size;
        let vocab = self.config.vocab_size;

        // 1. Embedding lookup
        let mut hidden = vec![0.0f32; h];
        let id = token as usize;
        if id < vocab {
            hidden.copy_from_slice(&self.embed_tokens[id * h..(id + 1) * h]);
        }

        // 1b. Pre-LayerNorm
        if let (Some(w), Some(b)) = (&self.pre_ln_weight, &self.pre_ln_bias) {
            hidden = layernorm(&hidden, w, b, self.config.norm_eps as f32, h);
        }

        // 2. Process through all blocks
        let mut state = self.state.borrow_mut();
        let mut v_first: Option<Vec<f32>> = None;
        for (i, block) in self.blocks.iter().enumerate() {
            let vf = v_first.as_deref();
            let (h_out, v_out) = block.forward_one(&hidden, &mut state.layers[i], vf);
            hidden = h_out;
            if i == 0 { v_first = Some(v_out); }
        }
        state.advance(1);

        // 3. Final LayerNorm
        let normed = layernorm(
            &hidden,
            &self.final_norm_weight,
            &self.final_norm_bias,
            self.config.norm_eps as f32,
            h,
        );

        // 4. LM head
        let logits = matmul_t(&normed, &self.lm_head_weight, 1, h, vocab);

        ModelOutput {
            logits,
            shape: [1, 1, vocab],
        }
    }

    /// Memory saved vs f32 (bytes).
    pub fn memory_savings(&self) -> usize {
        self.blocks.iter().map(|b| {
            let h = b.hidden_size;
            let inter = b.intermediate_size;
            // f32 sizes: r_proj[h*h] + k_proj[h*h] + v_proj[h*h] + o_proj[h*h]
            //          + ffn_key[inter*h] + ffn_value[h*inter]
            let f32_size = (4 * h * h + 2 * inter * h) * 4;
            let q4_size = b.r_proj.memory_bytes()
                + b.k_proj.memory_bytes()
                + b.v_proj.memory_bytes()
                + b.o_proj.memory_bytes()
                + b.ffn_key_weight.memory_bytes()
                + b.ffn_value_weight.memory_bytes();
            f32_size - q4_size
        }).sum()
    }

    /// Total model size in bytes (approximate).
    pub fn model_size_bytes(&self) -> usize {
        let embed_size = self.embed_tokens.len() * 4;
        let norm_size = (self.final_norm_weight.len() + self.final_norm_bias.len()) * 4;
        let lm_head_size = self.lm_head_weight.len() * 4;
        let block_size: usize = self.blocks.iter().map(|b| {
            // Q4 projections
            b.r_proj.memory_bytes()
                + b.k_proj.memory_bytes()
                + b.v_proj.memory_bytes()
                + b.o_proj.memory_bytes()
                + b.ffn_key_weight.memory_bytes()
                + b.ffn_value_weight.memory_bytes()
                // f32 parameters
                + (b.ln1_weight.len() + b.ln1_bias.len()
                    + b.x_r.len() + b.x_k.len() + b.x_v.len()
                    + b.x_w.len() + b.x_a.len() + b.x_g.len()
                    + b.w0.len() + b.w1.len() + b.w2.len()
                    + b.a0.len() + b.a1.len() + b.a2.len()
                    + b.g1.len() + b.g2.len()
                    + b.k_k.len() + b.k_a.len() + b.r_k.len()
                    + b.g_norm_weight.len() + b.g_norm_bias.len()
                    + b.ln2_weight.len() + b.ln2_bias.len()
                    + b.ffn_x_k.len()
                    + b.v0.len() + b.v1.len() + b.v2.len()) * 4
        }).sum();
        embed_size + norm_size + lm_head_size + block_size
    }
}

impl Model for Q4RwkvModel {
    fn forward(&self, token_ids: &[u32]) -> ModelOutput {
        self.reset_state();
        self.prefill(token_ids)
    }

    fn forward_prefill(&self, token_ids: &[u32], _state: &mut ModelState) -> ModelOutput {
        self.reset_state();
        self.prefill(token_ids)
    }

    fn forward_one(&self, token: u32, _state: &mut ModelState) -> ModelOutput {
        self.decode_one(token)
    }

    fn num_layers(&self) -> usize {
        self.blocks.len()
    }

    fn config(&self) -> &ModelConfig {
        unimplemented!("Q4RwkvModel uses RwkvConfig, not ModelConfig")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ssm::rwkv_block::RwkvBlock;
    use crate::ssm::rwkv_config::RwkvConfig;

    fn pseudo_random_vec(seed: u64, len: usize) -> Vec<f32> {
        let mut state = seed;
        (0..len).map(|_| {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let bits = 0x3F800000u32 | ((state >> 41) as u32 & 0x7FFFFF);
            (f32::from_bits(bits) - 1.5) * 0.2
        }).collect()
    }

    fn make_test_config() -> RwkvConfig {
        RwkvConfig {
            hidden_size: 64,
            num_layers: 2,
            vocab_size: 128,
            num_heads: 4,
            head_size: 16,
            intermediate_size: 128,
            norm_eps: 1e-5,
            decay_rank: 8,
            alpha_rank: 8,
            gate_rank: 16,
        }
    }

    fn make_f32_model() -> RwkvModel {
        let config = make_test_config();
        let h = config.hidden_size;
        let nh = config.num_heads;
        let hs = config.head_size;
        let inter = config.intermediate_size;
        let dr = config.decay_rank;
        let ar = config.alpha_rank;
        let gr = config.gate_rank;
        let vocab = config.vocab_size;

        let embed_tokens = pseudo_random_vec(100, vocab * h);
        let final_norm_weight = vec![1.0f32; h];
        let final_norm_bias = vec![0.0f32; h];
        let lm_head_weight = pseudo_random_vec(200, vocab * h);

        let mut blocks = Vec::new();
        for layer_idx in 0..config.num_layers {
            let s = (layer_idx as u64 + 1) * 1000;
            blocks.push(RwkvBlock {
                hidden_size: h,
                num_heads: nh,
                head_size: hs,
                intermediate_size: inter,
                decay_rank: dr,
                alpha_rank: ar,
                gate_rank: gr,
                norm_eps: config.norm_eps as f32,
                ln1_weight: vec![1.0f32; h],
                ln1_bias: vec![0.0f32; h],
                x_r: pseudo_random_vec(s + 10, h),
                x_k: pseudo_random_vec(s + 11, h),
                x_v: pseudo_random_vec(s + 12, h),
                x_w: pseudo_random_vec(s + 13, h),
                x_a: pseudo_random_vec(s + 14, h),
                x_g: pseudo_random_vec(s + 15, h),
                r_proj: pseudo_random_vec(s + 1, h * h),
                k_proj: pseudo_random_vec(s + 2, h * h),
                v_proj: pseudo_random_vec(s + 3, h * h),
                o_proj: pseudo_random_vec(s + 5, h * h),
                w0: pseudo_random_vec(s + 20, h),
                w1: pseudo_random_vec(s + 21, h * dr),
                w2: pseudo_random_vec(s + 22, dr * h),
                a0: pseudo_random_vec(s + 30, h),
                a1: pseudo_random_vec(s + 31, h * ar),
                a2: pseudo_random_vec(s + 32, ar * h),
                g1: pseudo_random_vec(s + 40, h * gr),
                g2: pseudo_random_vec(s + 41, gr * h),
                k_k: vec![1.0f32; h],
                k_a: vec![1.0f32; h],
                r_k: pseudo_random_vec(s + 50, nh * hs),
                g_norm_weight: vec![1.0f32; h],
                g_norm_bias: vec![0.0f32; h],
                v_rank: 0,
                v0: vec![],
                v1: vec![],
                v2: vec![],
                ln2_weight: vec![1.0f32; h],
                ln2_bias: vec![0.0f32; h],
                ffn_x_k: pseudo_random_vec(s + 60, h),
                ffn_key_weight: pseudo_random_vec(s + 8, inter * h),
                ffn_value_weight: pseudo_random_vec(s + 9, h * inter),
            });
        }

        RwkvModel::new(
            config,
            embed_tokens,
            blocks,
            final_norm_weight,
            final_norm_bias,
            lm_head_weight,
        )
    }

    #[test]
    fn test_q4_rwkv_produces_finite_output() {
        let f32_model = make_f32_model();
        let q_model = Q4RwkvModel::from_f32(&f32_model);

        let output = q_model.forward(&[1, 2, 3]);
        assert_eq!(output.logits.len(), q_model.config.vocab_size);
        for (i, &v) in output.logits.iter().enumerate() {
            assert!(v.is_finite(), "logit[{i}] = {v} is not finite");
        }
    }

    #[test]
    fn test_q4_rwkv_similar_to_f32() {
        let f32_model = make_f32_model();
        let q_model = Q4RwkvModel::from_f32(&f32_model);

        let f32_out = f32_model.forward(&[1, 2, 3]);
        let q_out = q_model.forward(&[1, 2, 3]);

        let dot: f64 = f32_out.logits.iter().zip(q_out.logits.iter())
            .map(|(&a, &b)| a as f64 * b as f64).sum();
        let norm_a: f64 = f32_out.logits.iter().map(|&x| (x as f64).powi(2)).sum::<f64>().sqrt();
        let norm_b: f64 = q_out.logits.iter().map(|&x| (x as f64).powi(2)).sum::<f64>().sqrt();
        let cos_sim = dot / (norm_a * norm_b);

        // Q4 is lossy, so cosine > 0.7 is acceptable
        assert!(cos_sim > 0.7, "Q4 cosine sim {cos_sim:.4} too low vs f32");
    }

    #[test]
    fn test_q4_rwkv_saves_memory() {
        let f32_model = make_f32_model();
        let q4_model = Q4RwkvModel::from_f32(&f32_model);

        let savings = q4_model.memory_savings();
        assert!(savings > 0, "Q4 should save memory vs f32, got {savings}");

        // Verify savings are substantial: 6 large matrices quantized from f32 to Q4
        // Each matrix element goes from 4 bytes to ~0.625 bytes (20 bytes per 32 elements)
        let h = q4_model.config.hidden_size;
        let inter = q4_model.config.intermediate_size;
        let total_f32_elements_per_block = 4 * h * h + 2 * inter * h;
        let expected_min_savings_per_block = total_f32_elements_per_block; // at least 1 byte per element saved
        let num_blocks = q4_model.config.num_layers;
        assert!(savings > expected_min_savings_per_block * num_blocks,
            "savings {savings} should be > {}", expected_min_savings_per_block * num_blocks);
    }

    #[test]
    fn test_q4_rwkv_model_size() {
        let f32_model = make_f32_model();
        let q4_model = Q4RwkvModel::from_f32(&f32_model);
        let size = q4_model.model_size_bytes();
        assert!(size > 0);

        // Compute approximate f32 model size for the same parameters
        let h = q4_model.config.hidden_size;
        let inter = q4_model.config.intermediate_size;
        let vocab = q4_model.config.vocab_size;
        let num_layers = q4_model.config.num_layers;
        // Just the 6 quantized matrices per layer (f32 baseline)
        let f32_proj_size = num_layers * (4 * h * h + 2 * inter * h) * 4;
        // Shared weights
        let shared_size = (vocab * h * 2 + h * 2) * 4; // embed + lm_head + norms
        let f32_total = f32_proj_size + shared_size;

        assert!(size < f32_total, "Q4 model size {size} should be < f32 size {f32_total}");
    }
}
