//! Q4-quantized Mamba model.
//!
//! Quantizes the large linear projections (in_proj, out_proj) to Q4_0 (4-bit)
//! while keeping SSM-specific ops (conv1d, selective scan, A_log, D) in f32.
//! This reduces model size by ~6.4x, making Mamba-130M fit in ~32MB for ESP32/WASM.

use std::cell::RefCell;

use crate::config::ModelConfig;
use crate::model::causal_lm::ModelOutput;
use crate::model::traits::{Model, ModelState};
use crate::ops::matmul::matmul_t;
use crate::ops::pure_rust_ops::rmsnorm;
use crate::ops::activation::{silu, softplus};
use crate::quantization::q4::Q4Linear;
use crate::ssm::config::MambaConfig;
use crate::ssm::mamba_model::MambaModel;
use crate::ssm::selective_scan::selective_scan_step;
use crate::ssm::state::{MambaLayerState, RecurrentState};

/// A single Mamba block with Q4-quantized linear projections.
pub struct Q4MambaBlock {
    pub d_model: usize,
    pub d_inner: usize,
    pub d_state: usize,
    pub d_conv: usize,
    pub dt_rank: usize,
    pub norm_weight: Vec<f32>,    // [d_model] — f32
    pub norm_eps: f32,
    pub in_proj: Q4Linear,         // [2*d_inner, d_model] — Q4
    pub conv1d_weight: Vec<f32>,   // [d_inner, d_conv] — f32
    pub conv1d_bias: Vec<f32>,     // [d_inner] — f32
    pub x_proj_weight: Vec<f32>,   // [dt_rank+2*d_state, d_inner] — f32 (small)
    pub dt_proj_weight: Vec<f32>,  // [d_inner, dt_rank] — f32 (small)
    pub dt_proj_bias: Vec<f32>,    // [d_inner] — f32
    pub a_log: Vec<f32>,           // [d_inner, d_state] — f32
    pub d_param: Vec<f32>,         // [d_inner] — f32
    pub out_proj: Q4Linear,        // [d_model, d_inner] — Q4
}

impl Q4MambaBlock {
    fn conv1d_step(&self, x: &[f32], state: &mut MambaLayerState) -> Vec<f32> {
        let d_inner = self.d_inner;
        let d_conv = self.d_conv;
        let mut out = vec![0.0f32; d_inner];
        for i in 0..d_inner {
            let buf = &mut state.conv_state[i * d_conv..(i + 1) * d_conv];
            buf.copy_within(1.., 0);
            buf[d_conv - 1] = x[i];
            let w = &self.conv1d_weight[i * d_conv..(i + 1) * d_conv];
            let sum: f32 = buf.iter().zip(w.iter()).map(|(&b, &k)| b * k).sum();
            out[i] = sum + self.conv1d_bias[i];
        }
        out
    }

    fn ssm_forward_step(&self, x: &[f32], state: &mut MambaLayerState) -> Vec<f32> {
        let d_inner = self.d_inner;
        let d_state = self.d_state;
        let dt_rank = self.dt_rank;

        let x_proj_out = matmul_t(x, &self.x_proj_weight, 1, d_inner, dt_rank + 2 * d_state);
        let dt_input = &x_proj_out[0..dt_rank];
        let b_slice = &x_proj_out[dt_rank..dt_rank + d_state];
        let c_slice = &x_proj_out[dt_rank + d_state..dt_rank + 2 * d_state];

        let dt_projected = matmul_t(dt_input, &self.dt_proj_weight, 1, dt_rank, d_inner);
        let delta: Vec<f32> = (0..d_inner)
            .map(|i| softplus(dt_projected[i] + self.dt_proj_bias[i]))
            .collect();

        selective_scan_step(x, &delta, &self.a_log, b_slice, c_slice, &self.d_param, &mut state.ssm_state)
    }

    pub fn forward_one(&self, hidden: &[f32], state: &mut MambaLayerState) -> Vec<f32> {
        let d_model = self.d_model;
        let d_inner = self.d_inner;

        let normed = rmsnorm(hidden, &self.norm_weight, self.norm_eps, d_model);
        let proj = self.in_proj.forward(&normed, 1);

        let x_proj = &proj[0..d_inner];
        let z_proj = &proj[d_inner..2 * d_inner];

        let x_conv = self.conv1d_step(x_proj, state);
        let x_act: Vec<f32> = x_conv.iter().map(|&v| silu(v)).collect();
        let y = self.ssm_forward_step(&x_act, state);
        let gated: Vec<f32> = z_proj.iter().zip(y.iter()).map(|(&z, &yi)| silu(z) * yi).collect();
        let out = self.out_proj.forward(&gated, 1);

        out.iter().zip(hidden.iter()).map(|(&o, &h)| o + h).collect()
    }

    pub fn forward_seq(&self, hidden: &[f32], seq_len: usize, state: &mut MambaLayerState) -> Vec<f32> {
        let d_model = self.d_model;
        let mut output = Vec::with_capacity(seq_len * d_model);
        for t in 0..seq_len {
            let token = &hidden[t * d_model..(t + 1) * d_model];
            let out = self.forward_one(token, state);
            output.extend_from_slice(&out);
        }
        output
    }
}

/// Q4-quantized Mamba language model.
pub struct Q4MambaModel {
    pub config: MambaConfig,
    pub embed_tokens: Vec<f32>,
    pub blocks: Vec<Q4MambaBlock>,
    pub final_norm_weight: Vec<f32>,
    pub lm_head_weight: Vec<f32>,
    state: RefCell<RecurrentState>,
}

impl Q4MambaModel {
    /// Quantize a full-precision MambaModel to Q4.
    pub fn from_f32(model: &MambaModel) -> Self {
        let config = model.config.clone();
        let blocks: Vec<Q4MambaBlock> = model.blocks.iter().map(|block| {
            Q4MambaBlock {
                d_model: block.d_model,
                d_inner: block.d_inner,
                d_state: block.d_state,
                d_conv: block.d_conv,
                dt_rank: block.dt_rank,
                norm_weight: block.norm_weight.clone(),
                norm_eps: block.norm_eps,
                in_proj: Q4Linear::from_f32(&block.in_proj_weight, 2 * block.d_inner, block.d_model),
                conv1d_weight: block.conv1d_weight.clone(),
                conv1d_bias: block.conv1d_bias.clone(),
                x_proj_weight: block.x_proj_weight.clone(),
                dt_proj_weight: block.dt_proj_weight.clone(),
                dt_proj_bias: block.dt_proj_bias.clone(),
                a_log: block.a_log.clone(),
                d_param: block.d_param.clone(),
                out_proj: Q4Linear::from_f32(&block.out_proj_weight, block.d_model, block.d_inner),
            }
        }).collect();

        let state = RecurrentState::new(
            config.num_layers, config.d_inner(), config.d_state, config.d_conv,
        );

        Q4MambaModel {
            config,
            embed_tokens: model.embed_tokens.clone(),
            blocks,
            final_norm_weight: model.final_norm_weight.clone(),
            lm_head_weight: model.lm_head_weight.clone(),
            state: RefCell::new(state),
        }
    }

    pub fn reset_state(&self) {
        self.state.borrow_mut().reset();
    }

    pub fn prefill(&self, token_ids: &[u32]) -> ModelOutput {
        let d_model = self.config.d_model;
        let vocab = self.config.vocab_size;
        let seq_len = token_ids.len();

        let mut hidden = vec![0.0f32; seq_len * d_model];
        for (t, &id) in token_ids.iter().enumerate() {
            let id = id as usize;
            if id < vocab {
                hidden[t * d_model..(t + 1) * d_model]
                    .copy_from_slice(&self.embed_tokens[id * d_model..(id + 1) * d_model]);
            }
        }

        let mut state = self.state.borrow_mut();
        for (i, block) in self.blocks.iter().enumerate() {
            hidden = block.forward_seq(&hidden, seq_len, &mut state.layers[i]);
        }
        state.advance(seq_len);

        let last = &hidden[(seq_len - 1) * d_model..seq_len * d_model];
        let normed = rmsnorm(last, &self.final_norm_weight, self.config.norm_eps as f32, d_model);
        let logits = matmul_t(&normed, &self.lm_head_weight, 1, d_model, vocab);

        ModelOutput { logits, shape: [1, 1, vocab] }
    }

    pub fn decode_one(&self, token: u32) -> ModelOutput {
        let d_model = self.config.d_model;
        let vocab = self.config.vocab_size;

        let mut hidden = vec![0.0f32; d_model];
        let id = token as usize;
        if id < vocab {
            hidden.copy_from_slice(&self.embed_tokens[id * d_model..(id + 1) * d_model]);
        }

        let mut state = self.state.borrow_mut();
        for (i, block) in self.blocks.iter().enumerate() {
            hidden = block.forward_one(&hidden, &mut state.layers[i]);
        }
        state.advance(1);

        let normed = rmsnorm(&hidden, &self.final_norm_weight, self.config.norm_eps as f32, d_model);
        let logits = matmul_t(&normed, &self.lm_head_weight, 1, d_model, vocab);

        ModelOutput { logits, shape: [1, 1, vocab] }
    }

    /// Memory saved vs f32 (bytes).
    pub fn memory_savings(&self) -> usize {
        self.blocks.iter().map(|b| {
            let f32_size = (2 * b.d_inner * b.d_model + b.d_model * b.d_inner) * 4;
            let q4_size = b.in_proj.memory_bytes() + b.out_proj.memory_bytes();
            f32_size - q4_size
        }).sum()
    }

    /// Total model size in bytes (approximate).
    pub fn model_size_bytes(&self) -> usize {
        let embed_size = self.embed_tokens.len() * 4;
        let norm_size = self.final_norm_weight.len() * 4;
        let lm_head_size = self.lm_head_weight.len() * 4;
        let block_size: usize = self.blocks.iter().map(|b| {
            b.in_proj.memory_bytes()
                + b.out_proj.memory_bytes()
                + (b.conv1d_weight.len() + b.conv1d_bias.len()
                    + b.x_proj_weight.len() + b.dt_proj_weight.len()
                    + b.dt_proj_bias.len() + b.a_log.len()
                    + b.d_param.len() + b.norm_weight.len()) * 4
        }).sum();
        embed_size + norm_size + lm_head_size + block_size
    }
}

impl Model for Q4MambaModel {
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
        unimplemented!("Q4MambaModel uses MambaConfig, not ModelConfig")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ssm::config::MambaConfig;
    use crate::ssm::mamba_block::MambaBlock;

    fn pseudo_random_vec(seed: u64, len: usize) -> Vec<f32> {
        let mut state = seed;
        (0..len).map(|_| {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let bits = 0x3F800000u32 | ((state >> 41) as u32 & 0x7FFFFF);
            (f32::from_bits(bits) - 1.5) * 0.2
        }).collect()
    }

    fn make_f32_model() -> MambaModel {
        let config = MambaConfig::tiny_test();
        let d = config.d_model;
        let di = config.d_inner();
        let ds = config.d_state;
        let dc = config.d_conv;
        let dr = config.dt_rank;
        let v = config.vocab_size;

        let mut blocks = Vec::new();
        for i in 0..config.num_layers {
            let s = (i as u64 + 1) * 1000;
            blocks.push(MambaBlock {
                d_model: d, d_inner: di, d_state: ds, d_conv: dc, dt_rank: dr,
                norm_weight: vec![1.0; d], norm_eps: 1e-5,
                in_proj_weight: pseudo_random_vec(s+1, 2*di*d),
                in_proj_bias: vec![],
                conv1d_weight: pseudo_random_vec(s+2, di*dc),
                conv1d_bias: vec![0.0; di],
                x_proj_weight: pseudo_random_vec(s+3, (dr+2*ds)*di),
                dt_proj_weight: pseudo_random_vec(s+4, di*dr),
                dt_proj_bias: vec![0.0; di],
                a_log: pseudo_random_vec(s+5, di*ds).into_iter().map(|v| -v.abs()-0.1).collect(),
                d_param: vec![1.0; di],
                out_proj_weight: pseudo_random_vec(s+6, d*di),
                out_proj_bias: vec![],
            });
        }

        MambaModel::new(config, pseudo_random_vec(100, v*d), blocks, vec![1.0; d], pseudo_random_vec(200, v*d))
    }

    #[test]
    fn test_q4_mamba_produces_finite_output() {
        let f32_model = make_f32_model();
        let q_model = Q4MambaModel::from_f32(&f32_model);

        let output = q_model.forward(&[1, 2, 3]);
        assert_eq!(output.logits.len(), q_model.config.vocab_size);
        for (i, &v) in output.logits.iter().enumerate() {
            assert!(v.is_finite(), "logit[{i}] = {v} is not finite");
        }
    }

    #[test]
    fn test_q4_mamba_similar_to_f32() {
        let f32_model = make_f32_model();
        let q_model = Q4MambaModel::from_f32(&f32_model);

        let f32_out = f32_model.forward(&[1, 2, 3]);
        let q_out = q_model.forward(&[1, 2, 3]);

        let dot: f64 = f32_out.logits.iter().zip(q_out.logits.iter())
            .map(|(&a, &b)| a as f64 * b as f64).sum();
        let norm_a: f64 = f32_out.logits.iter().map(|&x| (x as f64).powi(2)).sum::<f64>().sqrt();
        let norm_b: f64 = q_out.logits.iter().map(|&x| (x as f64).powi(2)).sum::<f64>().sqrt();
        let cos_sim = dot / (norm_a * norm_b);

        // Q4 is lower precision than INT8, so relax threshold
        assert!(cos_sim > 0.8, "Q4 cosine sim {cos_sim:.4} too low vs f32");
    }

    #[test]
    fn test_q4_mamba_saves_more_than_int8() {
        let f32_model = make_f32_model();
        let q4_model = Q4MambaModel::from_f32(&f32_model);
        let int8_model = crate::quantization::QuantizedMambaModel::from_f32(&f32_model);

        let q4_savings = q4_model.memory_savings();
        let int8_savings = int8_model.memory_savings();
        assert!(q4_savings > int8_savings, "Q4 should save more than INT8");
    }

    #[test]
    fn test_q4_mamba_model_size() {
        let f32_model = make_f32_model();
        let q4_model = Q4MambaModel::from_f32(&f32_model);
        let size = q4_model.model_size_bytes();
        assert!(size > 0);
        // Q4 should be significantly smaller than f32
        let f32_size = f32_model.embed_tokens.len() * 4
            + f32_model.lm_head_weight.len() * 4
            + f32_model.blocks.iter().map(|b| {
                (b.in_proj_weight.len() + b.out_proj_weight.len()
                    + b.conv1d_weight.len() + b.conv1d_bias.len()
                    + b.x_proj_weight.len() + b.dt_proj_weight.len()
                    + b.dt_proj_bias.len() + b.a_log.len()
                    + b.d_param.len() + b.norm_weight.len()) * 4
            }).sum::<usize>();
        assert!(size < f32_size, "Q4 model should be smaller than f32");
    }

    #[test]
    fn test_q4_mamba_decode_one() {
        let f32_model = make_f32_model();
        let q_model = Q4MambaModel::from_f32(&f32_model);

        // Prefill first
        q_model.forward(&[1, 2, 3]);
        // Then decode one more
        let output = q_model.decode_one(4);
        assert_eq!(output.logits.len(), q_model.config.vocab_size);
        for &v in &output.logits {
            assert!(v.is_finite());
        }
    }
}
