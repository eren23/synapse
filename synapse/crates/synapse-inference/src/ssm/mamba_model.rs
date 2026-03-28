//! MambaModel: a full Mamba language model implementing the `Model` trait.
//!
//! Uses internal `RecurrentState` (not KV cache). The `Model` trait requires
//! `&mut KVCache` params but MambaModel ignores them and uses its own state
//! via `RefCell<RecurrentState>`.

use std::cell::RefCell;
use std::collections::HashMap;

use crate::kv_cache::KVCache;
use crate::model::causal_lm::ModelOutput;
use crate::model::traits::Model;
use crate::config::ModelConfig;
use crate::ops::matmul::matmul_t;
use crate::ops::pure_rust_ops::rmsnorm;
use crate::ssm::config::MambaConfig;
use crate::ssm::mamba_block::MambaBlock;
use crate::ssm::state::RecurrentState;
use crate::weight_loading::RawTensor;
#[cfg(test)]
use crate::weight_loading::AlignedBuffer;

/// A full Mamba language model.
///
/// Wraps N `MambaBlock` layers with embedding, final RMSNorm, and LM head.
/// Maintains internal recurrent state via `RefCell` so that the `Model` trait
/// (which takes `&self`) can still mutate state.
pub struct MambaModel {
    pub config: MambaConfig,
    /// Embedding table: `[vocab_size, d_model]` (row-major).
    pub embed_tokens: Vec<f32>,
    pub blocks: Vec<MambaBlock>,
    /// Final RMSNorm weight: `[d_model]`.
    pub final_norm_weight: Vec<f32>,
    /// LM head projection: `[vocab_size, d_model]` (row-major).
    pub lm_head_weight: Vec<f32>,
    state: RefCell<RecurrentState>,
}

impl MambaModel {
    /// Create a new MambaModel. The `RecurrentState` is created internally
    /// from the config dimensions.
    pub fn new(
        config: MambaConfig,
        embed_tokens: Vec<f32>,
        blocks: Vec<MambaBlock>,
        final_norm_weight: Vec<f32>,
        lm_head_weight: Vec<f32>,
    ) -> Self {
        let state = RecurrentState::new(
            config.num_layers,
            config.d_inner(),
            config.d_state,
            config.d_conv,
        );
        MambaModel {
            config,
            embed_tokens,
            blocks,
            final_norm_weight,
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
        let d_model = self.config.d_model;
        let vocab = self.config.vocab_size;
        let seq_len = token_ids.len();

        // 1. Embedding lookup -> [seq_len, d_model]
        let mut hidden = vec![0.0f32; seq_len * d_model];
        for (t, &id) in token_ids.iter().enumerate() {
            let id = id as usize;
            if id < vocab {
                let src = &self.embed_tokens[id * d_model..(id + 1) * d_model];
                hidden[t * d_model..(t + 1) * d_model].copy_from_slice(src);
            }
        }

        // 2. Process through all blocks sequentially
        let mut state = self.state.borrow_mut();
        for (i, block) in self.blocks.iter().enumerate() {
            hidden = block.forward_seq(&hidden, seq_len, &mut state.layers[i]);
        }
        state.advance(seq_len);

        // 3. Final RMSNorm on last token only
        let last_hidden = &hidden[(seq_len - 1) * d_model..seq_len * d_model];
        let normed = rmsnorm(last_hidden, &self.final_norm_weight, self.config.norm_eps as f32, d_model);

        // 4. LM head: [1, d_model] x [vocab, d_model]^T -> [1, vocab]
        let logits = matmul_t(&normed, &self.lm_head_weight, 1, d_model, vocab);

        ModelOutput {
            logits,
            shape: [1, 1, vocab],
        }
    }

    /// Process a single token through the full model (decode step).
    ///
    /// Returns logits with shape `[1, 1, vocab_size]`.
    pub fn decode_one(&self, token: u32) -> ModelOutput {
        let d_model = self.config.d_model;
        let vocab = self.config.vocab_size;

        // 1. Embedding lookup -> [d_model]
        let mut hidden = vec![0.0f32; d_model];
        let id = token as usize;
        if id < vocab {
            hidden.copy_from_slice(&self.embed_tokens[id * d_model..(id + 1) * d_model]);
        }

        // 2. Process through all blocks
        let mut state = self.state.borrow_mut();
        for (i, block) in self.blocks.iter().enumerate() {
            hidden = block.forward_one(&hidden, &mut state.layers[i]);
        }
        state.advance(1);

        // 3. Final RMSNorm
        let normed = rmsnorm(&hidden, &self.final_norm_weight, self.config.norm_eps as f32, d_model);

        // 4. LM head
        let logits = matmul_t(&normed, &self.lm_head_weight, 1, d_model, vocab);

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

impl Model for MambaModel {
    fn forward(&self, token_ids: &[u32]) -> ModelOutput {
        self.reset_state();
        self.prefill(token_ids)
    }

    fn forward_prefill(&self, token_ids: &[u32], _cache: &mut KVCache) -> ModelOutput {
        self.reset_state();
        self.prefill(token_ids)
    }

    fn forward_one(&self, token: u32, _cache: &mut KVCache) -> ModelOutput {
        self.decode_one(token)
    }

    fn num_layers(&self) -> usize {
        self.blocks.len()
    }

    fn config(&self) -> &ModelConfig {
        unimplemented!("MambaModel uses MambaConfig, not ModelConfig")
    }
}

// ── Weight loading ──────────────────────────────────────────────────

impl MambaModel {
    /// Build a MambaModel from a HuggingFace-style weight dictionary.
    ///
    /// Tries `"backbone."` prefix first (original Mamba checkpoints),
    /// then falls back to `"model."` prefix (HuggingFace transformers style).
    pub fn from_weights(
        config: MambaConfig,
        weights: &HashMap<String, RawTensor>,
    ) -> Result<Self, String> {
        // Determine prefix
        let prefix = if weights.keys().any(|k| k.starts_with("backbone.")) {
            "backbone"
        } else {
            "model"
        };

        let d_model = config.d_model;
        let d_inner = config.d_inner();
        let d_state = config.d_state;
        let d_conv = config.d_conv;
        let vocab = config.vocab_size;

        // Helper: get a weight tensor as Vec<f32>, with optional reshape tolerance.
        let get = |name: &str| -> Result<Vec<f32>, String> {
            weights
                .get(name)
                .map(|t| t.data.to_vec())
                .ok_or_else(|| format!("missing weight: {name}"))
        };

        // Helper: get optional weight, return empty vec if missing.
        let get_opt = |name: &str| -> Vec<f32> {
            weights.get(name).map(|t| t.data.to_vec()).unwrap_or_default()
        };

        // Embedding
        let embed_key = format!("{prefix}.embedding.weight");
        let embed_tokens = get(&embed_key).map_err(|_| {
            // Also try layers.embedding for some variants
            format!("missing embedding weight at {embed_key}")
        })?;
        if embed_tokens.len() != vocab * d_model {
            return Err(format!(
                "embed_tokens shape mismatch: expected {} got {}",
                vocab * d_model,
                embed_tokens.len()
            ));
        }

        // Final norm
        let norm_key = format!("{prefix}.norm_f.weight");
        let final_norm_weight = get(&norm_key)?;
        if final_norm_weight.len() != d_model {
            return Err(format!(
                "final_norm shape mismatch: expected {d_model} got {}",
                final_norm_weight.len()
            ));
        }

        // LM head
        let lm_head_weight = get("lm_head.weight")?;
        if lm_head_weight.len() != vocab * d_model {
            return Err(format!(
                "lm_head shape mismatch: expected {} got {}",
                vocab * d_model,
                lm_head_weight.len()
            ));
        }

        // Blocks
        let mut blocks = Vec::with_capacity(config.num_layers);
        for i in 0..config.num_layers {
            let layer_prefix = format!("{prefix}.layers.{i}");

            let norm_weight = get(&format!("{layer_prefix}.norm.weight"))?;

            let in_proj_weight = get(&format!("{layer_prefix}.mixer.in_proj.weight"))?;
            if in_proj_weight.len() != 2 * d_inner * d_model {
                return Err(format!(
                    "layer {i} in_proj shape mismatch: expected {} got {}",
                    2 * d_inner * d_model,
                    in_proj_weight.len()
                ));
            }
            let in_proj_bias = get_opt(&format!("{layer_prefix}.mixer.in_proj.bias"));

            // conv1d weight may be [d_inner, d_conv] or [d_inner, 1, d_conv]
            let conv1d_weight_raw = get(&format!("{layer_prefix}.mixer.conv1d.weight"))?;
            let conv1d_weight = if conv1d_weight_raw.len() == d_inner * d_conv {
                conv1d_weight_raw
            } else if conv1d_weight_raw.len() == d_inner * 1 * d_conv {
                // [d_inner, 1, d_conv] -> [d_inner, d_conv] (squeeze dim 1)
                conv1d_weight_raw
            } else {
                return Err(format!(
                    "layer {i} conv1d weight unexpected size: {}",
                    conv1d_weight_raw.len()
                ));
            };

            let conv1d_bias = get(&format!("{layer_prefix}.mixer.conv1d.bias"))?;

            let x_proj_weight = get(&format!("{layer_prefix}.mixer.x_proj.weight"))?;
            if x_proj_weight.len() != (2 * d_state + 1) * d_inner {
                return Err(format!(
                    "layer {i} x_proj shape mismatch: expected {} got {}",
                    (2 * d_state + 1) * d_inner,
                    x_proj_weight.len()
                ));
            }

            // dt_proj weight may be [d_inner], [d_inner, 1], or [1, d_inner]
            let dt_proj_weight_raw = get(&format!("{layer_prefix}.mixer.dt_proj.weight"))?;
            let dt_proj_weight = if dt_proj_weight_raw.len() == d_inner {
                dt_proj_weight_raw
            } else {
                return Err(format!(
                    "layer {i} dt_proj weight unexpected size: {}",
                    dt_proj_weight_raw.len()
                ));
            };

            let dt_proj_bias = get(&format!("{layer_prefix}.mixer.dt_proj.bias"))?;

            let a_log = get(&format!("{layer_prefix}.mixer.A_log"))?;
            if a_log.len() != d_inner * d_state {
                return Err(format!(
                    "layer {i} A_log shape mismatch: expected {} got {}",
                    d_inner * d_state,
                    a_log.len()
                ));
            }

            let d_param = get(&format!("{layer_prefix}.mixer.D"))?;
            if d_param.len() != d_inner {
                return Err(format!(
                    "layer {i} D param shape mismatch: expected {d_inner} got {}",
                    d_param.len()
                ));
            }

            let out_proj_weight = get(&format!("{layer_prefix}.mixer.out_proj.weight"))?;
            if out_proj_weight.len() != d_model * d_inner {
                return Err(format!(
                    "layer {i} out_proj shape mismatch: expected {} got {}",
                    d_model * d_inner,
                    out_proj_weight.len()
                ));
            }
            let out_proj_bias = get_opt(&format!("{layer_prefix}.mixer.out_proj.bias"));

            blocks.push(MambaBlock {
                d_model,
                d_inner,
                d_state,
                d_conv,
                norm_weight,
                norm_eps: config.norm_eps as f32,
                in_proj_weight,
                in_proj_bias,
                conv1d_weight,
                conv1d_bias,
                x_proj_weight,
                dt_proj_weight,
                dt_proj_bias,
                a_log,
                d_param,
                out_proj_weight,
                out_proj_bias,
            });
        }

        Ok(MambaModel::new(
            config,
            embed_tokens,
            blocks,
            final_norm_weight,
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

    fn make_test_model() -> MambaModel {
        let config = MambaConfig::tiny_test();
        let d_model = config.d_model;
        let d_inner = config.d_inner();
        let d_state = config.d_state;
        let d_conv = config.d_conv;
        let vocab = config.vocab_size;

        let embed_tokens = pseudo_random_vec(100, vocab * d_model);
        let final_norm_weight = vec![1.0f32; d_model];
        let lm_head_weight = pseudo_random_vec(200, vocab * d_model);

        let mut blocks = Vec::new();
        for layer_idx in 0..config.num_layers {
            let seed_base = (layer_idx as u64 + 1) * 1000;
            blocks.push(MambaBlock {
                d_model,
                d_inner,
                d_state,
                d_conv,
                norm_weight: vec![1.0f32; d_model],
                norm_eps: config.norm_eps as f32,
                in_proj_weight: pseudo_random_vec(seed_base + 1, 2 * d_inner * d_model),
                in_proj_bias: vec![],
                conv1d_weight: pseudo_random_vec(seed_base + 2, d_inner * d_conv),
                conv1d_bias: vec![0.0f32; d_inner],
                x_proj_weight: pseudo_random_vec(seed_base + 3, (2 * d_state + 1) * d_inner),
                dt_proj_weight: pseudo_random_vec(seed_base + 4, d_inner),
                dt_proj_bias: vec![0.0f32; d_inner],
                a_log: pseudo_random_vec(seed_base + 5, d_inner * d_state)
                    .into_iter()
                    .map(|v| -v.abs() - 0.1)
                    .collect(),
                d_param: vec![1.0f32; d_inner],
                out_proj_weight: pseudo_random_vec(seed_base + 6, d_model * d_inner),
                out_proj_bias: vec![],
            });
        }

        MambaModel::new(config, embed_tokens, blocks, final_norm_weight, lm_head_weight)
    }

    #[test]
    fn test_mamba_model_forward() {
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
    fn test_mamba_model_prefill_then_decode() {
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
    fn test_mamba_model_state_memory_constant() {
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
    fn test_mamba_model_from_weights() {
        let config = MambaConfig::tiny_test();
        let d_model = config.d_model;
        let d_inner = config.d_inner();
        let d_state = config.d_state;
        let d_conv = config.d_conv;
        let vocab = config.vocab_size;

        let mut weights: HashMap<String, RawTensor> = HashMap::new();

        let make_tensor = |shape: Vec<usize>, seed: u64| -> RawTensor {
            let len: usize = shape.iter().product();
            RawTensor {
                data: AlignedBuffer::from_slice(&pseudo_random_vec(seed, len)),
                shape,
            }
        };

        // Use "backbone." prefix
        weights.insert(
            "backbone.embedding.weight".to_string(),
            make_tensor(vec![vocab, d_model], 10),
        );
        weights.insert(
            "backbone.norm_f.weight".to_string(),
            make_tensor(vec![d_model], 11),
        );
        weights.insert(
            "lm_head.weight".to_string(),
            make_tensor(vec![vocab, d_model], 12),
        );

        for i in 0..config.num_layers {
            let seed_base = (i as u64 + 1) * 100;
            let p = format!("backbone.layers.{i}");

            weights.insert(format!("{p}.norm.weight"), make_tensor(vec![d_model], seed_base));
            weights.insert(
                format!("{p}.mixer.in_proj.weight"),
                make_tensor(vec![2 * d_inner, d_model], seed_base + 1),
            );
            weights.insert(
                format!("{p}.mixer.conv1d.weight"),
                make_tensor(vec![d_inner, d_conv], seed_base + 2),
            );
            weights.insert(
                format!("{p}.mixer.conv1d.bias"),
                make_tensor(vec![d_inner], seed_base + 3),
            );
            weights.insert(
                format!("{p}.mixer.x_proj.weight"),
                make_tensor(vec![2 * d_state + 1, d_inner], seed_base + 4),
            );
            weights.insert(
                format!("{p}.mixer.dt_proj.weight"),
                make_tensor(vec![d_inner], seed_base + 5),
            );
            weights.insert(
                format!("{p}.mixer.dt_proj.bias"),
                make_tensor(vec![d_inner], seed_base + 6),
            );
            // A_log values should be negative for stability
            let a_log_len = d_inner * d_state;
            let a_log_data: Vec<f32> = pseudo_random_vec(seed_base + 7, a_log_len)
                .into_iter()
                .map(|v| -v.abs() - 0.1)
                .collect();
            weights.insert(
                format!("{p}.mixer.A_log"),
                RawTensor {
                    data: AlignedBuffer::from_slice(&a_log_data),
                    shape: vec![d_inner, d_state],
                },
            );
            weights.insert(
                format!("{p}.mixer.D"),
                make_tensor(vec![d_inner], seed_base + 8),
            );
            weights.insert(
                format!("{p}.mixer.out_proj.weight"),
                make_tensor(vec![d_model, d_inner], seed_base + 9),
            );
        }

        let model = MambaModel::from_weights(config, &weights).expect("from_weights should succeed");
        let output = model.forward(&[1, 2, 3]);
        assert_eq!(output.shape, [1, 1, model.config.vocab_size]);
        for (i, &v) in output.logits.iter().enumerate() {
            assert!(v.is_finite(), "from_weights logit[{i}] = {v} is not finite");
        }
    }
}
