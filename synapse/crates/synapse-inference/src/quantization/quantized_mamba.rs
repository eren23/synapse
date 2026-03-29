//! INT8-quantized Mamba model.
//!
//! Quantizes the large linear projections (in_proj, out_proj) to INT8 while
//! keeping SSM-specific ops (conv1d, selective scan, A_log, D) in f32.
//! This reduces model size by ~4x with minimal quality loss.

use std::cell::RefCell;

use crate::config::ModelConfig;
use crate::model::causal_lm::ModelOutput;
use crate::model::traits::{Model, ModelState};
use crate::ops::matmul::matmul_t;
use crate::ops::pure_rust_ops::rmsnorm;
use crate::ops::activation::{silu, softplus};
use crate::quantization::QuantizedLinear;
use crate::ssm::config::MambaConfig;
use crate::ssm::mamba_model::MambaModel;
use crate::ssm::selective_scan::selective_scan_step;
use crate::ssm::state::{MambaLayerState, RecurrentState};

/// A single Mamba block with INT8-quantized linear projections.
pub struct QuantizedMambaBlock {
    pub d_model: usize,
    pub d_inner: usize,
    pub d_state: usize,
    pub d_conv: usize,
    pub dt_rank: usize,
    pub norm_weight: Vec<f32>,    // [d_model] — stays f32
    pub norm_eps: f32,
    pub in_proj: QuantizedLinear,  // [2*d_inner, d_model] — INT8
    pub conv1d_weight: Vec<f32>,   // [d_inner, d_conv] — f32 (small)
    pub conv1d_bias: Vec<f32>,     // [d_inner] — f32
    pub x_proj_weight: Vec<f32>,   // [dt_rank+2*d_state, d_inner] — f32 (small)
    pub dt_proj_weight: Vec<f32>,  // [d_inner, dt_rank] — f32 (small)
    pub dt_proj_bias: Vec<f32>,    // [d_inner] — f32
    pub a_log: Vec<f32>,           // [d_inner, d_state] — f32
    pub d_param: Vec<f32>,         // [d_inner] — f32
    pub out_proj: QuantizedLinear,  // [d_model, d_inner] — INT8
}

impl QuantizedMambaBlock {
    /// Conv1d step (identical to f32 version).
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

    /// SSM forward step (identical to f32 version).
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

    /// Forward one token.
    pub fn forward_one(&self, hidden: &[f32], state: &mut MambaLayerState) -> Vec<f32> {
        let d_model = self.d_model;
        let d_inner = self.d_inner;

        // 1. RMSNorm
        let normed = rmsnorm(hidden, &self.norm_weight, self.norm_eps, d_model);

        // 2. in_proj via INT8: [1, d_model] → [1, 2*d_inner]
        let proj = self.in_proj.forward(&normed, 1);

        // 3. Split x, z
        let x_proj = &proj[0..d_inner];
        let z_proj = &proj[d_inner..2 * d_inner];

        // 4. Conv1d
        let x_conv = self.conv1d_step(x_proj, state);

        // 5. SiLU
        let x_act: Vec<f32> = x_conv.iter().map(|&v| silu(v)).collect();

        // 6. SSM step (all f32)
        let y = self.ssm_forward_step(&x_act, state);

        // 7. Gate
        let gated: Vec<f32> = z_proj.iter().zip(y.iter()).map(|(&z, &yi)| silu(z) * yi).collect();

        // 8. out_proj via INT8: [1, d_inner] → [1, d_model]
        let out = self.out_proj.forward(&gated, 1);

        // 9. Residual
        out.iter().zip(hidden.iter()).map(|(&o, &h)| o + h).collect()
    }

    /// Forward a sequence.
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

/// INT8-quantized Mamba language model.
pub struct QuantizedMambaModel {
    pub config: MambaConfig,
    pub embed_tokens: Vec<f32>,
    pub blocks: Vec<QuantizedMambaBlock>,
    pub final_norm_weight: Vec<f32>,
    pub lm_head_weight: Vec<f32>,
    state: RefCell<RecurrentState>,
}

impl QuantizedMambaModel {
    /// Quantize a full-precision MambaModel to INT8.
    pub fn from_f32(model: &MambaModel) -> Self {
        let config = model.config.clone();
        let blocks: Vec<QuantizedMambaBlock> = model.blocks.iter().map(|block| {
            QuantizedMambaBlock {
                d_model: block.d_model,
                d_inner: block.d_inner,
                d_state: block.d_state,
                d_conv: block.d_conv,
                dt_rank: block.dt_rank,
                norm_weight: block.norm_weight.clone(),
                norm_eps: block.norm_eps,
                in_proj: QuantizedLinear::from_f32(&block.in_proj_weight, 2 * block.d_inner, block.d_model),
                conv1d_weight: block.conv1d_weight.clone(),
                conv1d_bias: block.conv1d_bias.clone(),
                x_proj_weight: block.x_proj_weight.clone(),
                dt_proj_weight: block.dt_proj_weight.clone(),
                dt_proj_bias: block.dt_proj_bias.clone(),
                a_log: block.a_log.clone(),
                d_param: block.d_param.clone(),
                out_proj: QuantizedLinear::from_f32(&block.out_proj_weight, block.d_model, block.d_inner),
            }
        }).collect();

        let state = RecurrentState::new(
            config.num_layers, config.d_inner(), config.d_state, config.d_conv,
        );

        QuantizedMambaModel {
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
            let int8_size = b.in_proj.memory_bytes() + b.out_proj.memory_bytes();
            f32_size - int8_size
        }).sum()
    }
}

impl Model for QuantizedMambaModel {
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
        unimplemented!("QuantizedMambaModel uses MambaConfig, not ModelConfig")
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
    fn test_quantized_mamba_produces_finite_output() {
        let f32_model = make_f32_model();
        let q_model = QuantizedMambaModel::from_f32(&f32_model);

        let output = q_model.forward(&[1, 2, 3]);
        assert_eq!(output.logits.len(), q_model.config.vocab_size);
        for (i, &v) in output.logits.iter().enumerate() {
            assert!(v.is_finite(), "logit[{i}] = {v} is not finite");
        }
    }

    #[test]
    fn test_quantized_mamba_similar_to_f32() {
        let f32_model = make_f32_model();
        let q_model = QuantizedMambaModel::from_f32(&f32_model);

        let f32_out = f32_model.forward(&[1, 2, 3]);
        let q_out = q_model.forward(&[1, 2, 3]);

        // Cosine similarity should be high
        let dot: f64 = f32_out.logits.iter().zip(q_out.logits.iter())
            .map(|(&a, &b)| a as f64 * b as f64).sum();
        let norm_a: f64 = f32_out.logits.iter().map(|&x| (x as f64).powi(2)).sum::<f64>().sqrt();
        let norm_b: f64 = q_out.logits.iter().map(|&x| (x as f64).powi(2)).sum::<f64>().sqrt();
        let cos_sim = dot / (norm_a * norm_b);

        assert!(cos_sim > 0.9, "INT8 cosine sim {cos_sim:.4} too low vs f32");
    }

    #[test]
    fn test_quantized_mamba_saves_memory() {
        let f32_model = make_f32_model();
        let q_model = QuantizedMambaModel::from_f32(&f32_model);
        assert!(q_model.memory_savings() > 0, "should save memory");
    }
}
