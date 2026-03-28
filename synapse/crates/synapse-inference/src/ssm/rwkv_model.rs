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
    /// Supports two naming conventions:
    /// - SmerkyG: `model.blocks.{i}.attention.*`, `model.blocks.{i}.feed_forward.*`
    /// - Official: `model.layers.{i}.attn.*`, `model.layers.{i}.ffn.*`
    pub fn from_weights(
        config: RwkvConfig,
        weights: &HashMap<String, RawTensor>,
    ) -> Result<Self, String> {
        let h = config.hidden_size;
        let inter = config.intermediate_size;
        let vocab = config.vocab_size;
        let dr = config.decay_rank;
        let ar = config.alpha_rank;
        let gr = config.gate_rank;

        // Detect naming convention
        let (embed_key, head_key, norm_w_key, norm_b_key, layer_fmt) =
            if weights.keys().any(|k| k.starts_with("model.blocks.")) {
                // SmerkyG style
                ("model.embeddings.weight", "head.weight", "model.ln_out.weight", "model.ln_out.bias", "smerkyg")
            } else if weights.keys().any(|k| k.starts_with("model.layers.")) {
                // Official RWKV HF style
                ("model.embeddings.weight", "lm_head.weight", "model.norm.weight", "model.norm.bias", "official")
            } else {
                // Legacy style
                ("emb.weight", "head.weight", "ln_out.weight", "ln_out.bias", "legacy")
            };

        let get = |name: &str| -> Result<Vec<f32>, String> {
            weights.get(name).map(|t| t.data.to_vec())
                .ok_or_else(|| format!("missing weight: {name}"))
        };

        // Helper: try primary name, fallback to alternate
        let get_or = |a: &str, b: &str| -> Result<Vec<f32>, String> {
            get(a).or_else(|_| get(b))
        };

        let embed_tokens = get(embed_key)?;
        if embed_tokens.len() != vocab * h {
            return Err(format!("embed shape mismatch: expected {} got {}", vocab * h, embed_tokens.len()));
        }
        let final_norm_weight = get_or(norm_w_key, "model.norm.weight")?;
        let final_norm_bias = get_or(norm_b_key, "model.norm.bias")?;
        let lm_head_weight = get_or(head_key, "lm_head.weight")?;

        let mut blocks = Vec::with_capacity(config.num_layers);
        for i in 0..config.num_layers {
            // Build per-layer weight prefix based on naming convention
            let (att_p, ffn_p, ln1_p, ln2_p) = match layer_fmt {
                "smerkyg" => (
                    format!("model.blocks.{i}.attention"),
                    format!("model.blocks.{i}.feed_forward"),
                    format!("model.blocks.{i}.ln1"),
                    format!("model.blocks.{i}.ln2"),
                ),
                "official" => (
                    format!("model.layers.{i}.attn"),
                    format!("model.layers.{i}.ffn"),
                    format!("model.layers.{i}.attn_norm"),
                    format!("model.layers.{i}.ffn_norm"),
                ),
                _ => (
                    format!("blocks.{i}.att"),
                    format!("blocks.{i}.ffn"),
                    format!("blocks.{i}.ln1"),
                    format!("blocks.{i}.ln2"),
                ),
            };

            // Helper to squeeze leading 1-dims from lerp weights like [1,1,h] -> [h]
            let get_squeeze = |name: &str| -> Result<Vec<f32>, String> {
                let raw = get(name)?;
                if raw.len() == h { return Ok(raw); }
                // Squeeze: [1,1,h] or [1,h] → [h]
                if raw.len() == h {
                    Ok(raw)
                } else {
                    Ok(raw) // data is the same regardless of shape metadata
                }
            };

            blocks.push(RwkvBlock {
                hidden_size: h,
                num_heads: config.num_heads,
                head_size: config.head_size,
                intermediate_size: inter,
                decay_rank: dr,
                alpha_rank: ar,
                gate_rank: gr,
                norm_eps: config.norm_eps as f32,
                ln1_weight: get(&format!("{ln1_p}.weight"))?,
                ln1_bias: get(&format!("{ln1_p}.bias"))?,
                x_r: get_squeeze(&format!("{att_p}.x_r"))?,
                x_k: get_squeeze(&format!("{att_p}.x_k"))?,
                x_v: get_squeeze(&format!("{att_p}.x_v"))?,
                x_w: get_squeeze(&format!("{att_p}.x_w"))?,
                x_a: get_squeeze(&format!("{att_p}.x_a"))?,
                x_g: get_squeeze(&format!("{att_p}.x_g"))?,
                r_proj: get(&format!("{att_p}.receptance.weight"))
                    .or_else(|_| get(&format!("{att_p}.r_proj.weight")))?,
                k_proj: get(&format!("{att_p}.key.weight"))
                    .or_else(|_| get(&format!("{att_p}.k_proj.weight")))?,
                v_proj: get(&format!("{att_p}.value.weight"))
                    .or_else(|_| get(&format!("{att_p}.v_proj.weight")))?,
                o_proj: get(&format!("{att_p}.output.weight"))
                    .or_else(|_| get(&format!("{att_p}.o_proj.weight")))?,
                w0: get_squeeze(&format!("{att_p}.w0"))?,
                w1: get(&format!("{att_p}.w1"))?,
                w2: get(&format!("{att_p}.w2"))?,
                a0: get_squeeze(&format!("{att_p}.a0"))?,
                a1: get(&format!("{att_p}.a1"))?,
                a2: get(&format!("{att_p}.a2"))?,
                g1: get(&format!("{att_p}.g1"))?,
                g2: get(&format!("{att_p}.g2"))?,
                k_k: get_squeeze(&format!("{att_p}.k_k"))?,
                k_a: get_squeeze(&format!("{att_p}.k_a"))?,
                r_k: get(&format!("{att_p}.r_k"))?,
                g_norm_weight: get(&format!("{att_p}.ln_x.weight"))
                    .or_else(|_| get(&format!("{att_p}.g_norm.weight")))?,
                g_norm_bias: get(&format!("{att_p}.ln_x.bias"))
                    .or_else(|_| get(&format!("{att_p}.g_norm.bias")))?,
                ln2_weight: get(&format!("{ln2_p}.weight"))?,
                ln2_bias: get(&format!("{ln2_p}.bias"))?,
                ffn_x_k: get_squeeze(&format!("{ffn_p}.x_k"))?,
                ffn_key_weight: get(&format!("{ffn_p}.key.weight"))?,
                ffn_value_weight: get(&format!("{ffn_p}.value.weight"))?,
            });
        }

        Ok(RwkvModel::new(config, embed_tokens, blocks, final_norm_weight, final_norm_bias, lm_head_weight))
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
                hidden_size: h, num_heads: nh, head_size: hs,
                intermediate_size: inter, decay_rank: dr, alpha_rank: ar, gate_rank: gr,
                norm_eps: config.norm_eps as f32,
                ln1_weight: vec![1.0f32; h], ln1_bias: vec![0.0f32; h],
                x_r: pseudo_random_vec(s+10, h), x_k: pseudo_random_vec(s+11, h),
                x_v: pseudo_random_vec(s+12, h), x_w: pseudo_random_vec(s+13, h),
                x_a: pseudo_random_vec(s+14, h), x_g: pseudo_random_vec(s+15, h),
                r_proj: pseudo_random_vec(s+1, h*h), k_proj: pseudo_random_vec(s+2, h*h),
                v_proj: pseudo_random_vec(s+3, h*h), o_proj: pseudo_random_vec(s+5, h*h),
                w0: pseudo_random_vec(s+20, h),
                w1: pseudo_random_vec(s+21, h*dr), w2: pseudo_random_vec(s+22, dr*h),
                a0: pseudo_random_vec(s+30, h),
                a1: pseudo_random_vec(s+31, h*ar), a2: pseudo_random_vec(s+32, ar*h),
                g1: pseudo_random_vec(s+40, h*gr), g2: pseudo_random_vec(s+41, gr*h),
                k_k: vec![1.0f32; h], k_a: vec![1.0f32; h],
                r_k: pseudo_random_vec(s+50, nh*hs),
                g_norm_weight: vec![1.0f32; h], g_norm_bias: vec![0.0f32; h],
                ln2_weight: vec![1.0f32; h], ln2_bias: vec![0.0f32; h],
                ffn_x_k: pseudo_random_vec(s+60, h),
                ffn_key_weight: pseudo_random_vec(s+8, inter*h),
                ffn_value_weight: pseudo_random_vec(s+9, h*inter),
            });
        }

        RwkvModel::new(config, embed_tokens, blocks, final_norm_weight, final_norm_bias, lm_head_weight)
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
        let dr = config.decay_rank;
        let ar = config.alpha_rank;
        let gr = config.gate_rank;
        let vocab = config.vocab_size;

        let mut weights: HashMap<String, RawTensor> = HashMap::new();

        let make = |shape: Vec<usize>, seed: u64| -> RawTensor {
            let len: usize = shape.iter().product();
            RawTensor {
                data: AlignedBuffer::from_slice(&pseudo_random_vec(seed, len)),
                shape,
            }
        };

        // Legacy naming (no prefix)
        weights.insert("emb.weight".into(), make(vec![vocab, h], 10));
        weights.insert("ln_out.weight".into(), make(vec![h], 11));
        weights.insert("ln_out.bias".into(), make(vec![h], 12));
        weights.insert("head.weight".into(), make(vec![vocab, h], 13));

        for i in 0..config.num_layers {
            let s = (i as u64 + 1) * 100;
            let p = format!("blocks.{i}");
            let a = format!("{p}.att");
            let f = format!("{p}.ffn");

            weights.insert(format!("{p}.ln1.weight"), make(vec![h], s));
            weights.insert(format!("{p}.ln1.bias"), make(vec![h], s+1));
            // 6 token shift lerps
            weights.insert(format!("{a}.x_r"), make(vec![h], s+10));
            weights.insert(format!("{a}.x_k"), make(vec![h], s+11));
            weights.insert(format!("{a}.x_v"), make(vec![h], s+12));
            weights.insert(format!("{a}.x_w"), make(vec![h], s+13));
            weights.insert(format!("{a}.x_a"), make(vec![h], s+14));
            weights.insert(format!("{a}.x_g"), make(vec![h], s+15));
            // Projections
            weights.insert(format!("{a}.receptance.weight"), make(vec![h, h], s+20));
            weights.insert(format!("{a}.key.weight"), make(vec![h, h], s+21));
            weights.insert(format!("{a}.value.weight"), make(vec![h, h], s+22));
            weights.insert(format!("{a}.output.weight"), make(vec![h, h], s+23));
            // Low-rank decay
            weights.insert(format!("{a}.w0"), make(vec![h], s+30));
            weights.insert(format!("{a}.w1"), make(vec![h, dr], s+31));
            weights.insert(format!("{a}.w2"), make(vec![dr, h], s+32));
            // Low-rank alpha
            weights.insert(format!("{a}.a0"), make(vec![h], s+40));
            weights.insert(format!("{a}.a1"), make(vec![h, ar], s+41));
            weights.insert(format!("{a}.a2"), make(vec![ar, h], s+42));
            // Low-rank gate
            weights.insert(format!("{a}.g1"), make(vec![h, gr], s+50));
            weights.insert(format!("{a}.g2"), make(vec![gr, h], s+51));
            // Key modulation
            weights.insert(format!("{a}.k_k"), make(vec![h], s+60));
            weights.insert(format!("{a}.k_a"), make(vec![h], s+61));
            weights.insert(format!("{a}.r_k"), make(vec![nh, hs], s+62));
            // GroupNorm
            weights.insert(format!("{a}.ln_x.weight"), make(vec![h], s+70));
            weights.insert(format!("{a}.ln_x.bias"), make(vec![h], s+71));
            // LN2 + FFN
            weights.insert(format!("{p}.ln2.weight"), make(vec![h], s+80));
            weights.insert(format!("{p}.ln2.bias"), make(vec![h], s+81));
            weights.insert(format!("{f}.x_k"), make(vec![h], s+82));
            weights.insert(format!("{f}.key.weight"), make(vec![inter, h], s+83));
            weights.insert(format!("{f}.value.weight"), make(vec![h, inter], s+84));
        }

        let model = RwkvModel::from_weights(config, &weights).expect("from_weights should succeed");
        let output = model.forward(&[1, 2, 3]);
        assert_eq!(output.shape, [1, 1, model.config.vocab_size]);
        for (i, &v) in output.logits.iter().enumerate() {
            assert!(v.is_finite(), "from_weights logit[{i}] = {v} is not finite");
        }
    }
}
