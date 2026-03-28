//! RwkvModel: a full RWKV-7 language model implementing the `Model` trait.
//!
//! Uses internal `RwkvState` (not KV cache). The `Model` trait takes
//! `&mut ModelState` params but RwkvModel ignores them (passes `Recurrent`
//! variant) and uses its own state via `RefCell<RwkvState>`.

use std::cell::RefCell;
use std::collections::HashMap;

use crate::model::causal_lm::ModelOutput;
use crate::model::traits::{Model, ModelState};
use crate::config::ModelConfig;
use crate::ops::matmul::matmul_t;
use crate::ssm::rwkv_block::RwkvBlock;
use crate::ssm::rwkv_config::RwkvConfig;
use crate::ssm::rwkv_state::RwkvState;
use crate::weight_loading::RawTensor;
#[cfg(test)]
use crate::weight_loading::AlignedBuffer;

/// LayerNorm for the final output (same impl as in rwkv_block but needed here).
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

/// A full RWKV-7 language model.
///
/// Wraps N `RwkvBlock` layers with embedding, final LayerNorm, and LM head.
/// Maintains internal recurrent state via `RefCell` so that the `Model` trait
/// (which takes `&self`) can still mutate state.
pub struct RwkvModel {
    pub config: RwkvConfig,
    /// Embedding table: `[vocab_size, hidden_size]` (row-major).
    pub embed_tokens: Vec<f32>,
    pub blocks: Vec<RwkvBlock>,
    /// Final LayerNorm weight: `[hidden_size]`.
    pub final_norm_weight: Vec<f32>,
    /// Final LayerNorm bias: `[hidden_size]`.
    pub final_norm_bias: Vec<f32>,
    /// LM head projection: `[vocab_size, hidden_size]` (row-major).
    pub lm_head_weight: Vec<f32>,
    state: RefCell<RwkvState>,
}

impl RwkvModel {
    /// Create a new RwkvModel. The `RwkvState` is created internally
    /// from the config dimensions.
    pub fn new(
        config: RwkvConfig,
        embed_tokens: Vec<f32>,
        blocks: Vec<RwkvBlock>,
        final_norm_weight: Vec<f32>,
        final_norm_bias: Vec<f32>,
        lm_head_weight: Vec<f32>,
    ) -> Self {
        let state = RwkvState::new(
            config.num_layers,
            config.hidden_size,
            config.num_heads,
            config.head_size,
        );
        RwkvModel {
            config,
            embed_tokens,
            blocks,
            final_norm_weight,
            final_norm_bias,
            lm_head_weight,
            state: RefCell::new(state),
        }
    }

    /// Reset the internal recurrent state to zeros.
    pub fn reset_state(&self) {
        self.state.borrow_mut().reset();
    }

    /// Process a sequence of tokens through the full model (prefill).
    ///
    /// Embeds all tokens, runs them through every block sequentially,
    /// then applies final norm + LM head on the **last** token only.
    /// Returns logits with shape `[1, 1, vocab_size]`.
    pub fn prefill(&self, token_ids: &[u32]) -> ModelOutput {
        let h = self.config.hidden_size;
        let vocab = self.config.vocab_size;
        let seq_len = token_ids.len();

        // 1. Embedding lookup -> [seq_len, hidden_size]
        let mut hidden = vec![0.0f32; seq_len * h];
        for (t, &id) in token_ids.iter().enumerate() {
            let id = id as usize;
            if id < vocab {
                let src = &self.embed_tokens[id * h..(id + 1) * h];
                hidden[t * h..(t + 1) * h].copy_from_slice(src);
            }
        }

        // 2. Process through all blocks sequentially
        let mut state = self.state.borrow_mut();
        for (i, block) in self.blocks.iter().enumerate() {
            hidden = block.forward_seq(&hidden, seq_len, &mut state.layers[i]);
        }
        state.advance(seq_len);

        // 3. Final LayerNorm on last token only
        let last_hidden = &hidden[(seq_len - 1) * h..seq_len * h];
        let normed = layernorm(
            last_hidden,
            &self.final_norm_weight,
            &self.final_norm_bias,
            self.config.norm_eps as f32,
            h,
        );

        // 4. LM head: [1, h] x [vocab, h]^T -> [1, vocab]
        let logits = matmul_t(&normed, &self.lm_head_weight, 1, h, vocab);

        ModelOutput {
            logits,
            shape: [1, 1, vocab],
        }
    }

    /// Process a single token through the full model (decode step).
    ///
    /// Returns logits with shape `[1, 1, vocab_size]`.
    pub fn decode_one(&self, token: u32) -> ModelOutput {
        let h = self.config.hidden_size;
        let vocab = self.config.vocab_size;

        // 1. Embedding lookup -> [hidden_size]
        let mut hidden = vec![0.0f32; h];
        let id = token as usize;
        if id < vocab {
            hidden.copy_from_slice(&self.embed_tokens[id * h..(id + 1) * h]);
        }

        // 2. Process through all blocks
        let mut state = self.state.borrow_mut();
        for (i, block) in self.blocks.iter().enumerate() {
            hidden = block.forward_one(&hidden, &mut state.layers[i]);
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

    /// Total heap memory used by the internal recurrent state (in bytes).
    pub fn state_memory_bytes(&self) -> usize {
        self.state.borrow().memory_bytes()
    }
}

impl Model for RwkvModel {
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
        unimplemented!("RwkvModel uses RwkvConfig, not ModelConfig")
    }
}

// ── Weight loading ──────────────────────────────────────────────────

impl RwkvModel {
    /// Build an RwkvModel from a HuggingFace-style weight dictionary.
    ///
    /// Expects RWKV-7 HF weight names like `blocks.{i}.att.receptance.weight`,
    /// `emb.weight`, `ln_out.weight`, `head.weight`, etc.
    pub fn from_weights(
        config: RwkvConfig,
        weights: &HashMap<String, RawTensor>,
    ) -> Result<Self, String> {
        let h = config.hidden_size;
        let nh = config.num_heads;
        let hs = config.head_size;
        let inter = config.intermediate_size;
        let vocab = config.vocab_size;

        // Helper: get a weight tensor as Vec<f32>.
        let get = |name: &str| -> Result<Vec<f32>, String> {
            weights
                .get(name)
                .map(|t| t.data.to_vec())
                .ok_or_else(|| format!("missing weight: {name}"))
        };

        // Embedding
        let embed_tokens = get("emb.weight")?;
        if embed_tokens.len() != vocab * h {
            return Err(format!(
                "emb.weight shape mismatch: expected {} got {}",
                vocab * h,
                embed_tokens.len()
            ));
        }

        // Final LayerNorm
        let final_norm_weight = get("ln_out.weight")?;
        if final_norm_weight.len() != h {
            return Err(format!(
                "ln_out.weight shape mismatch: expected {h} got {}",
                final_norm_weight.len()
            ));
        }
        let final_norm_bias = get("ln_out.bias")?;

        // LM head
        let lm_head_weight = get("head.weight")?;
        if lm_head_weight.len() != vocab * h {
            return Err(format!(
                "head.weight shape mismatch: expected {} got {}",
                vocab * h,
                lm_head_weight.len()
            ));
        }

        // Blocks
        let mut blocks = Vec::with_capacity(config.num_layers);
        for i in 0..config.num_layers {
            let p = format!("blocks.{i}");

            let ln1_weight = get(&format!("{p}.ln1.weight"))?;
            let ln1_bias = get(&format!("{p}.ln1.bias"))?;

            let time_mix_x = get(&format!("{p}.att.time_mix_x"))?;

            let receptance_weight = get(&format!("{p}.att.receptance.weight"))?;
            if receptance_weight.len() != h * h {
                return Err(format!(
                    "layer {i} att.receptance.weight shape mismatch: expected {} got {}",
                    h * h,
                    receptance_weight.len()
                ));
            }

            let key_weight = get(&format!("{p}.att.key.weight"))?;
            let value_weight = get(&format!("{p}.att.value.weight"))?;
            let gate_weight = get(&format!("{p}.att.gate.weight"))?;
            let output_weight = get(&format!("{p}.att.output.weight"))?;

            let time_decay = get(&format!("{p}.att.time_decay"))?;
            if time_decay.len() != nh * hs {
                return Err(format!(
                    "layer {i} att.time_decay shape mismatch: expected {} got {}",
                    nh * hs,
                    time_decay.len()
                ));
            }

            let att_ln_weight = get(&format!("{p}.att.ln_x.weight"))?;
            let att_ln_bias = get(&format!("{p}.att.ln_x.bias"))?;

            let ln2_weight = get(&format!("{p}.ln2.weight"))?;
            let ln2_bias = get(&format!("{p}.ln2.bias"))?;

            let channel_mix_x = get(&format!("{p}.ffn.time_mix_x"))?;

            let ffn_receptance_weight = get(&format!("{p}.ffn.receptance.weight"))?;
            let ffn_key_weight = get(&format!("{p}.ffn.key.weight"))?;
            if ffn_key_weight.len() != inter * h {
                return Err(format!(
                    "layer {i} ffn.key.weight shape mismatch: expected {} got {}",
                    inter * h,
                    ffn_key_weight.len()
                ));
            }
            let ffn_value_weight = get(&format!("{p}.ffn.value.weight"))?;
            if ffn_value_weight.len() != h * inter {
                return Err(format!(
                    "layer {i} ffn.value.weight shape mismatch: expected {} got {}",
                    h * inter,
                    ffn_value_weight.len()
                ));
            }

            blocks.push(RwkvBlock {
                hidden_size: h,
                num_heads: nh,
                head_size: hs,
                intermediate_size: inter,
                norm_eps: config.norm_eps as f32,
                ln1_weight,
                ln1_bias,
                time_mix_x,
                receptance_weight,
                key_weight,
                value_weight,
                gate_weight,
                output_weight,
                time_decay,
                att_ln_weight,
                att_ln_bias,
                ln2_weight,
                ln2_bias,
                channel_mix_x,
                ffn_receptance_weight,
                ffn_key_weight,
                ffn_value_weight,
            });
        }

        Ok(RwkvModel::new(
            config,
            embed_tokens,
            blocks,
            final_norm_weight,
            final_norm_bias,
            lm_head_weight,
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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

    fn make_test_model() -> RwkvModel {
        let config = RwkvConfig::tiny_test();
        let h = config.hidden_size;
        let nh = config.num_heads;
        let hs = config.head_size;
        let inter = config.intermediate_size;
        let vocab = config.vocab_size;

        let embed_tokens = pseudo_random_vec(100, vocab * h);
        let final_norm_weight = vec![1.0f32; h];
        let final_norm_bias = vec![0.0f32; h];
        let lm_head_weight = pseudo_random_vec(200, vocab * h);

        let mut blocks = Vec::new();
        for layer_idx in 0..config.num_layers {
            let seed_base = (layer_idx as u64 + 1) * 1000;
            blocks.push(RwkvBlock {
                hidden_size: h,
                num_heads: nh,
                head_size: hs,
                intermediate_size: inter,
                norm_eps: config.norm_eps as f32,
                ln1_weight: vec![1.0f32; h],
                ln1_bias: vec![0.0f32; h],
                time_mix_x: vec![0.5f32; h],
                receptance_weight: pseudo_random_vec(seed_base + 1, h * h),
                key_weight: pseudo_random_vec(seed_base + 2, h * h),
                value_weight: pseudo_random_vec(seed_base + 3, h * h),
                gate_weight: pseudo_random_vec(seed_base + 4, h * h),
                output_weight: pseudo_random_vec(seed_base + 5, h * h),
                time_decay: pseudo_random_vec(seed_base + 6, nh * hs)
                    .into_iter()
                    .map(|v| -v.abs() - 0.1)
                    .collect(),
                att_ln_weight: vec![1.0f32; h],
                att_ln_bias: vec![0.0f32; h],
                ln2_weight: vec![1.0f32; h],
                ln2_bias: vec![0.0f32; h],
                channel_mix_x: vec![0.5f32; h],
                ffn_receptance_weight: pseudo_random_vec(seed_base + 7, h * h),
                ffn_key_weight: pseudo_random_vec(seed_base + 8, inter * h),
                ffn_value_weight: pseudo_random_vec(seed_base + 9, h * inter),
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
    fn test_rwkv_model_forward() {
        let model = make_test_model();
        let vocab = model.config.vocab_size;

        let output = model.forward(&[1, 2, 3]);
        assert_eq!(output.shape, [1, 1, vocab]);
        assert_eq!(output.logits.len(), vocab);
        for (i, &v) in output.logits.iter().enumerate() {
            assert!(v.is_finite(), "logit[{i}] = {v} is not finite");
        }
    }

    #[test]
    fn test_rwkv_model_prefill_then_decode() {
        let model = make_test_model();
        let vocab = model.config.vocab_size;

        // Reset and prefill 3 tokens
        model.reset_state();
        let out1 = model.prefill(&[1, 2, 3]);
        assert_eq!(out1.shape, [1, 1, vocab]);
        assert_eq!(out1.logits.len(), vocab);
        for (i, &v) in out1.logits.iter().enumerate() {
            assert!(v.is_finite(), "prefill logit[{i}] = {v} is not finite");
        }

        // Decode 2 more tokens
        let out2 = model.decode_one(4);
        assert_eq!(out2.shape, [1, 1, vocab]);
        for (i, &v) in out2.logits.iter().enumerate() {
            assert!(v.is_finite(), "decode1 logit[{i}] = {v} is not finite");
        }

        let out3 = model.decode_one(5);
        assert_eq!(out3.shape, [1, 1, vocab]);
        for (i, &v) in out3.logits.iter().enumerate() {
            assert!(v.is_finite(), "decode2 logit[{i}] = {v} is not finite");
        }
    }

    #[test]
    fn test_rwkv_model_state_memory_constant() {
        let model = make_test_model();
        let mem_before = model.state_memory_bytes();
        assert!(mem_before > 0, "state memory should be nonzero");

        // Process some tokens
        model.reset_state();
        let _ = model.prefill(&[1, 2, 3]);
        let _ = model.decode_one(4);

        let mem_after = model.state_memory_bytes();
        assert_eq!(
            mem_before, mem_after,
            "state memory should be constant: before={mem_before}, after={mem_after}"
        );
    }

    #[test]
    fn test_rwkv_model_from_weights() {
        let config = RwkvConfig::tiny_test();
        let h = config.hidden_size;
        let nh = config.num_heads;
        let hs = config.head_size;
        let inter = config.intermediate_size;
        let vocab = config.vocab_size;

        let mut weights: HashMap<String, RawTensor> = HashMap::new();

        let make_tensor = |shape: Vec<usize>, seed: u64| -> RawTensor {
            let len: usize = shape.iter().product();
            RawTensor {
                data: AlignedBuffer::from_slice(&pseudo_random_vec(seed, len)),
                shape,
            }
        };

        // Global weights
        weights.insert("emb.weight".to_string(), make_tensor(vec![vocab, h], 10));
        weights.insert("ln_out.weight".to_string(), make_tensor(vec![h], 11));
        weights.insert("ln_out.bias".to_string(), make_tensor(vec![h], 12));
        weights.insert("head.weight".to_string(), make_tensor(vec![vocab, h], 13));

        for i in 0..config.num_layers {
            let seed_base = (i as u64 + 1) * 100;
            let p = format!("blocks.{i}");

            weights.insert(format!("{p}.ln1.weight"), make_tensor(vec![h], seed_base));
            weights.insert(format!("{p}.ln1.bias"), make_tensor(vec![h], seed_base + 1));
            weights.insert(format!("{p}.att.time_mix_x"), make_tensor(vec![h], seed_base + 2));
            weights.insert(
                format!("{p}.att.receptance.weight"),
                make_tensor(vec![h, h], seed_base + 3),
            );
            weights.insert(
                format!("{p}.att.key.weight"),
                make_tensor(vec![h, h], seed_base + 4),
            );
            weights.insert(
                format!("{p}.att.value.weight"),
                make_tensor(vec![h, h], seed_base + 5),
            );
            weights.insert(
                format!("{p}.att.gate.weight"),
                make_tensor(vec![h, h], seed_base + 6),
            );
            weights.insert(
                format!("{p}.att.output.weight"),
                make_tensor(vec![h, h], seed_base + 7),
            );
            // time_decay: negative values for stability
            let td_len = nh * hs;
            let td_data: Vec<f32> = pseudo_random_vec(seed_base + 8, td_len)
                .into_iter()
                .map(|v| -v.abs() - 0.1)
                .collect();
            weights.insert(
                format!("{p}.att.time_decay"),
                RawTensor {
                    data: AlignedBuffer::from_slice(&td_data),
                    shape: vec![nh, hs],
                },
            );
            weights.insert(
                format!("{p}.att.ln_x.weight"),
                make_tensor(vec![h], seed_base + 9),
            );
            weights.insert(
                format!("{p}.att.ln_x.bias"),
                make_tensor(vec![h], seed_base + 10),
            );
            weights.insert(format!("{p}.ln2.weight"), make_tensor(vec![h], seed_base + 11));
            weights.insert(format!("{p}.ln2.bias"), make_tensor(vec![h], seed_base + 12));
            weights.insert(
                format!("{p}.ffn.time_mix_x"),
                make_tensor(vec![h], seed_base + 13),
            );
            weights.insert(
                format!("{p}.ffn.receptance.weight"),
                make_tensor(vec![h, h], seed_base + 14),
            );
            weights.insert(
                format!("{p}.ffn.key.weight"),
                make_tensor(vec![inter, h], seed_base + 15),
            );
            weights.insert(
                format!("{p}.ffn.value.weight"),
                make_tensor(vec![h, inter], seed_base + 16),
            );
        }

        let model =
            RwkvModel::from_weights(config, &weights).expect("from_weights should succeed");
        let output = model.forward(&[1, 2, 3]);
        assert_eq!(output.shape, [1, 1, model.config.vocab_size]);
        for (i, &v) in output.logits.iter().enumerate() {
            assert!(
                v.is_finite(),
                "from_weights logit[{i}] = {v} is not finite"
            );
        }
    }
}
