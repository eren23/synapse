//! HybridModel: a full Qwen3.5-style hybrid model implementing the `Model` trait.
//!
//! Combines DeltaNet (linear attention) and GQA (full attention) layers in a
//! repeating pattern. DeltaNet layers use constant-size recurrent state, while
//! GQA layers use a traditional KV cache. The model manages both state types
//! internally via `RefCell`.

use std::cell::RefCell;

use crate::config::ModelConfig;
use crate::model::causal_lm::ModelOutput;
use crate::model::traits::{Model, ModelState};
use crate::ops::matmul::matmul_t;
use crate::ops::pure_rust_ops::rmsnorm;
use crate::ssm::deltanet_state::DeltaNetLayerState;
use crate::ssm::hybrid_config::HybridConfig;
use crate::ssm::hybrid_layer::{DeltaNetDecoderLayer, GqaDecoderLayer, KvLayerState};

/// A layer in the hybrid model: either DeltaNet or GQA.
pub enum HybridLayer {
    DeltaNet(DeltaNetDecoderLayer),
    Gqa(GqaDecoderLayer),
}

/// Combined state for all layers in the hybrid model.
pub struct HybridState {
    /// DeltaNet recurrent states, indexed by DeltaNet-layer ordinal.
    pub deltanet_states: Vec<DeltaNetLayerState>,
    /// KV cache entries, indexed by GQA-layer ordinal.
    pub kv_states: Vec<KvLayerState>,
    /// Current position (number of tokens processed so far).
    pub position: usize,
}

impl HybridState {
    pub fn new(config: &HybridConfig, max_kv_seq: usize) -> Self {
        let num_dn = config.num_deltanet_layers();
        let num_gqa = config.num_gqa_layers();

        let deltanet_states = (0..num_dn)
            .map(|_| {
                DeltaNetLayerState::new(
                    config.deltanet_num_heads,
                    config.deltanet_head_dim,
                    config.deltanet_conv_kernel,
                )
            })
            .collect();

        let kv_states = (0..num_gqa)
            .map(|_| KvLayerState::new(max_kv_seq, config.num_kv_heads, config.gqa_head_dim))
            .collect();

        HybridState {
            deltanet_states,
            kv_states,
            position: 0,
        }
    }

    pub fn reset(&mut self) {
        for s in &mut self.deltanet_states {
            s.reset();
        }
        for s in &mut self.kv_states {
            s.reset();
        }
        self.position = 0;
    }

    /// Total heap memory used by all state buffers (bytes).
    pub fn memory_bytes(&self) -> usize {
        let dn: usize = self.deltanet_states.iter().map(|s| s.memory_bytes()).sum();
        let kv: usize = self.kv_states.iter().map(|s| s.memory_bytes()).sum();
        dn + kv
    }

    /// Memory used by DeltaNet states only (bytes). This is constant.
    pub fn deltanet_memory_bytes(&self) -> usize {
        self.deltanet_states.iter().map(|s| s.memory_bytes()).sum()
    }

    /// Memory used by KV caches only (bytes). The allocation is constant
    /// (pre-allocated to max_kv_seq), but logical occupancy grows.
    pub fn kv_memory_bytes(&self) -> usize {
        self.kv_states.iter().map(|s| s.memory_bytes()).sum()
    }
}

/// A hybrid DeltaNet + GQA language model (e.g. Qwen3.5).
pub struct HybridModel {
    pub config: HybridConfig,
    /// Embedding table: `[vocab_size, hidden_size]`.
    pub embed_tokens: Vec<f32>,
    /// Decoder layers in order (mix of DeltaNet and GQA).
    pub layers: Vec<HybridLayer>,
    /// Final RMSNorm weight: `[hidden_size]`.
    pub final_norm_weight: Vec<f32>,
    /// LM head weight: `[vocab_size, hidden_size]`. None if tied to embed_tokens.
    pub lm_head_weight: Option<Vec<f32>>,
    /// Precomputed RoPE cos table: `[max_pos, head_dim/2]`.
    pub rope_cos: Vec<f32>,
    /// Precomputed RoPE sin table: `[max_pos, head_dim/2]`.
    pub rope_sin: Vec<f32>,
    /// Internal hybrid state, managed via RefCell for interior mutability.
    state: RefCell<HybridState>,
}

impl HybridModel {
    /// Create a new HybridModel. State is initialised internally.
    ///
    /// `max_kv_seq` controls the pre-allocated KV cache length for GQA layers.
    pub fn new(
        config: HybridConfig,
        embed_tokens: Vec<f32>,
        layers: Vec<HybridLayer>,
        final_norm_weight: Vec<f32>,
        lm_head_weight: Option<Vec<f32>>,
        rope_cos: Vec<f32>,
        rope_sin: Vec<f32>,
        max_kv_seq: usize,
    ) -> Self {
        let state = HybridState::new(&config, max_kv_seq);
        HybridModel {
            config,
            embed_tokens,
            layers,
            final_norm_weight,
            lm_head_weight,
            rope_cos,
            rope_sin,
            state: RefCell::new(state),
        }
    }

    /// Reset all internal state (both DeltaNet recurrent state and KV caches).
    pub fn reset_state(&self) {
        self.state.borrow_mut().reset();
    }

    /// Get the effective LM head weight (own or tied to embeddings).
    fn lm_head(&self) -> &[f32] {
        self.lm_head_weight.as_deref().unwrap_or(&self.embed_tokens)
    }

    /// Process a sequence of tokens (prefill). Returns logits for the last token.
    pub fn prefill(&self, token_ids: &[u32]) -> ModelOutput {
        let d = self.config.hidden_size;
        let vocab = self.config.vocab_size;
        let seq_len = token_ids.len();

        // 1. Embedding lookup -> [seq_len, hidden_size]
        let mut hidden = vec![0.0f32; seq_len * d];
        for (t, &id) in token_ids.iter().enumerate() {
            let id = id as usize;
            if id < vocab {
                hidden[t * d..(t + 1) * d]
                    .copy_from_slice(&self.embed_tokens[id * d..(id + 1) * d]);
            }
        }

        // 2. Process through all layers
        let mut state = self.state.borrow_mut();
        let pos_offset = state.position;
        let mut dn_idx = 0usize;
        let mut gqa_idx = 0usize;

        for layer in self.layers.iter() {
            match layer {
                HybridLayer::DeltaNet(dn_layer) => {
                    hidden = dn_layer.forward_seq(
                        &hidden,
                        seq_len,
                        &mut state.deltanet_states[dn_idx],
                    );
                    dn_idx += 1;
                }
                HybridLayer::Gqa(gqa_layer) => {
                    hidden = gqa_layer.forward_seq(
                        &hidden,
                        seq_len,
                        &mut state.kv_states[gqa_idx],
                        &self.rope_cos,
                        &self.rope_sin,
                        pos_offset,
                    );
                    gqa_idx += 1;
                }
            }
        }

        state.position += seq_len;

        // 3. Final norm on last token
        let last = &hidden[(seq_len - 1) * d..seq_len * d];
        let normed = rmsnorm(last, &self.final_norm_weight, self.config.norm_eps as f32, d);

        // 4. LM head
        let logits = matmul_t(&normed, self.lm_head(), 1, d, vocab);

        ModelOutput {
            logits,
            shape: [1, 1, vocab],
        }
    }

    /// Process a single token (decode step). Returns logits.
    pub fn decode_one(&self, token: u32) -> ModelOutput {
        let d = self.config.hidden_size;
        let vocab = self.config.vocab_size;

        // 1. Embedding
        let mut hidden = vec![0.0f32; d];
        let id = token as usize;
        if id < vocab {
            hidden.copy_from_slice(&self.embed_tokens[id * d..(id + 1) * d]);
        }

        // 2. Process through all layers
        let mut state = self.state.borrow_mut();
        let position = state.position;
        let mut dn_idx = 0usize;
        let mut gqa_idx = 0usize;

        for layer in self.layers.iter() {
            match layer {
                HybridLayer::DeltaNet(dn_layer) => {
                    hidden = dn_layer.forward_one(
                        &hidden,
                        &mut state.deltanet_states[dn_idx],
                    );
                    dn_idx += 1;
                }
                HybridLayer::Gqa(gqa_layer) => {
                    hidden = gqa_layer.forward_one(
                        &hidden,
                        &mut state.kv_states[gqa_idx],
                        &self.rope_cos,
                        &self.rope_sin,
                        position,
                    );
                    gqa_idx += 1;
                }
            }
        }

        state.position += 1;

        // 3. Final norm
        let normed = rmsnorm(&hidden, &self.final_norm_weight, self.config.norm_eps as f32, d);

        // 4. LM head
        let logits = matmul_t(&normed, self.lm_head(), 1, d, vocab);

        ModelOutput {
            logits,
            shape: [1, 1, vocab],
        }
    }

    /// Total heap memory used by the internal state (bytes).
    pub fn state_memory_bytes(&self) -> usize {
        self.state.borrow().memory_bytes()
    }

    /// Memory used by DeltaNet recurrent states only (bytes). Constant.
    pub fn deltanet_state_memory_bytes(&self) -> usize {
        self.state.borrow().deltanet_memory_bytes()
    }

    /// Current number of tokens in the KV cache (grows with each decode step).
    pub fn kv_cache_len(&self) -> usize {
        let st = self.state.borrow();
        st.kv_states.first().map_or(0, |s| s.len)
    }
}

impl Model for HybridModel {
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
        self.layers.len()
    }

    fn config(&self) -> &ModelConfig {
        unimplemented!("HybridModel uses HybridConfig, not ModelConfig")
    }
}

// ── Weight loading ──────────────────────────────────────────────────

use std::collections::HashMap;
use crate::weight_loading::RawTensor;

impl HybridModel {
    /// Build a HybridModel from a HuggingFace-style weight dictionary.
    ///
    /// Expects Qwen3.5-style weight names:
    /// - `model.embed_tokens.weight`
    /// - `model.layers.{i}.input_layernorm.weight`
    /// - GQA layers: `model.layers.{i}.self_attn.{q,k,v,o}_proj.weight`,
    ///   `q_norm.weight`, `k_norm.weight`
    /// - DeltaNet layers: `model.layers.{i}.self_attn.qkv_proj.weight`,
    ///   `gate_proj.weight`, `alpha_proj.weight`, `beta_proj.weight`,
    ///   `q_conv1d.{weight,bias}`, `k_conv1d.{weight,bias}`, `v_conv1d.{weight,bias}`,
    ///   `o_norm.weight`, `o_proj.weight`
    /// - Both: `model.layers.{i}.mlp.{gate,up,down}_proj.weight`
    /// - `model.norm.weight`, `lm_head.weight`
    pub fn from_weights(
        config: HybridConfig,
        weights: &HashMap<String, RawTensor>,
        max_kv_seq: usize,
    ) -> Result<Self, String> {
        let d = config.hidden_size;
        let vocab = config.vocab_size;
        let im = config.intermediate_size;
        let nh_dn = config.deltanet_num_heads;
        let hd_dn = config.deltanet_head_dim;
        let nh_hd_dn = nh_dn * hd_dn;
        let ck = config.deltanet_conv_kernel;
        let nq = config.num_attention_heads;
        let nkv = config.num_kv_heads;
        let hd_gqa = config.gqa_head_dim;

        let prefix = if weights.keys().any(|k| k.starts_with("model.")) {
            "model"
        } else {
            ""
        };

        let get = |name: &str| -> Result<Vec<f32>, String> {
            weights.get(name)
                .map(|t| t.data.to_vec())
                .ok_or_else(|| format!("missing weight: {name}"))
        };

        // Embedding
        let embed_key = if prefix.is_empty() {
            "embed_tokens.weight".to_string()
        } else {
            format!("{prefix}.embed_tokens.weight")
        };
        let embed_tokens = get(&embed_key)?;
        if embed_tokens.len() != vocab * d {
            return Err(format!("embed_tokens shape mismatch: expected {} got {}", vocab * d, embed_tokens.len()));
        }

        // Final norm
        let norm_key = if prefix.is_empty() { "norm.weight".to_string() } else { format!("{prefix}.norm.weight") };
        let final_norm_weight = get(&norm_key)?;

        // LM head
        let lm_head_weight = weights.get("lm_head.weight").map(|t| t.data.to_vec());

        // RoPE tables
        let (rope_cos, rope_sin) = compute_rope_tables(max_kv_seq, hd_gqa);

        // Layers
        let mut layers = Vec::with_capacity(config.num_layers);
        for i in 0..config.num_layers {
            let lp = if prefix.is_empty() {
                format!("layers.{i}")
            } else {
                format!("{prefix}.layers.{i}")
            };

            if config.is_full_attention(i) {
                // GQA layer
                let attn_norm = get(&format!("{lp}.input_layernorm.weight"))?;
                let w_q = get(&format!("{lp}.self_attn.q_proj.weight"))?;
                let w_k = get(&format!("{lp}.self_attn.k_proj.weight"))?;
                let w_v = get(&format!("{lp}.self_attn.v_proj.weight"))?;
                let w_o = get(&format!("{lp}.self_attn.o_proj.weight"))?;
                let q_norm = get(&format!("{lp}.self_attn.q_norm.weight"))?;
                let k_norm = get(&format!("{lp}.self_attn.k_norm.weight"))?;
                let ffn_norm = get(&format!("{lp}.post_attention_layernorm.weight"))?;
                let ffn_gate = get(&format!("{lp}.mlp.gate_proj.weight"))?;
                let ffn_up = get(&format!("{lp}.mlp.up_proj.weight"))?;
                let ffn_down = get(&format!("{lp}.mlp.down_proj.weight"))?;

                layers.push(HybridLayer::Gqa(GqaDecoderLayer {
                    hidden_size: d,
                    num_q_heads: nq,
                    num_kv_heads: nkv,
                    head_dim: hd_gqa,
                    intermediate_size: im,
                    norm_eps: config.norm_eps as f32,
                    attn_norm_weight: attn_norm,
                    w_q, w_k, w_v, w_o,
                    q_norm_weight: q_norm,
                    k_norm_weight: k_norm,
                    ffn_norm_weight: ffn_norm,
                    ffn_gate_weight: ffn_gate,
                    ffn_up_weight: ffn_up,
                    ffn_down_weight: ffn_down,
                }));
            } else {
                // DeltaNet layer
                let attn_norm = get(&format!("{lp}.input_layernorm.weight"))?;
                let qkv = get(&format!("{lp}.self_attn.qkv_proj.weight"))?;
                let gate = get(&format!("{lp}.self_attn.gate_proj.weight"))?;
                let beta = get(&format!("{lp}.self_attn.beta_proj.weight"))?;
                let alpha = get(&format!("{lp}.self_attn.alpha_proj.weight"))?;
                let q_conv_w = get(&format!("{lp}.self_attn.q_conv1d.weight"))?;
                let q_conv_b = get(&format!("{lp}.self_attn.q_conv1d.bias"))?;
                let k_conv_w = get(&format!("{lp}.self_attn.k_conv1d.weight"))?;
                let k_conv_b = get(&format!("{lp}.self_attn.k_conv1d.bias"))?;
                let v_conv_w = get(&format!("{lp}.self_attn.v_conv1d.weight"))?;
                let v_conv_b = get(&format!("{lp}.self_attn.v_conv1d.bias"))?;
                let o_norm = get(&format!("{lp}.self_attn.o_norm.weight"))?;
                let o_proj = get(&format!("{lp}.self_attn.o_proj.weight"))?;
                let ffn_norm = get(&format!("{lp}.post_attention_layernorm.weight"))?;
                let ffn_gate = get(&format!("{lp}.mlp.gate_proj.weight"))?;
                let ffn_up = get(&format!("{lp}.mlp.up_proj.weight"))?;
                let ffn_down = get(&format!("{lp}.mlp.down_proj.weight"))?;

                layers.push(HybridLayer::DeltaNet(DeltaNetDecoderLayer {
                    hidden_size: d,
                    num_heads: nh_dn,
                    head_dim: hd_dn,
                    intermediate_size: im,
                    conv_kernel: ck,
                    norm_eps: config.norm_eps as f32,
                    attn_norm_weight: attn_norm,
                    qkv_weight: qkv,
                    gate_proj_weight: gate,
                    beta_proj_weight: beta,
                    alpha_proj_weight: alpha,
                    q_conv_weight: q_conv_w,
                    q_conv_bias: q_conv_b,
                    k_conv_weight: k_conv_w,
                    k_conv_bias: k_conv_b,
                    v_conv_weight: v_conv_w,
                    v_conv_bias: v_conv_b,
                    o_norm_weight: o_norm,
                    o_proj_weight: o_proj,
                    ffn_norm_weight: ffn_norm,
                    ffn_gate_weight: ffn_gate,
                    ffn_up_weight: ffn_up,
                    ffn_down_weight: ffn_down,
                }));
            }
        }

        Ok(HybridModel::new(
            config,
            embed_tokens,
            layers,
            final_norm_weight,
            lm_head_weight,
            rope_cos,
            rope_sin,
            max_kv_seq,
        ))
    }
}

/// Compute RoPE cos/sin tables for the given max position and head dimension.
fn compute_rope_tables(max_pos: usize, head_dim: usize) -> (Vec<f32>, Vec<f32>) {
    let half_d = head_dim / 2;
    let mut cos = vec![0.0f32; max_pos * half_d];
    let mut sin = vec![0.0f32; max_pos * half_d];
    for pos in 0..max_pos {
        for i in 0..half_d {
            let freq = 1.0 / (10000.0f32).powf(2.0 * i as f32 / head_dim as f32);
            let angle = pos as f32 * freq;
            cos[pos * half_d + i] = angle.cos();
            sin[pos * half_d + i] = angle.sin();
        }
    }
    (cos, sin)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ssm::hybrid_config::HybridConfig;

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

    fn make_rope_tables(max_pos: usize, head_dim: usize) -> (Vec<f32>, Vec<f32>) {
        let half_d = head_dim / 2;
        let mut cos = vec![0.0f32; max_pos * half_d];
        let mut sin = vec![0.0f32; max_pos * half_d];
        for pos in 0..max_pos {
            for i in 0..half_d {
                let freq = 1.0 / (10000.0f32).powf(2.0 * i as f32 / head_dim as f32);
                let angle = pos as f32 * freq;
                cos[pos * half_d + i] = angle.cos();
                sin[pos * half_d + i] = angle.sin();
            }
        }
        (cos, sin)
    }

    fn make_test_model() -> HybridModel {
        let config = HybridConfig::tiny_test();
        let d = config.hidden_size;       // 64
        let vocab = config.vocab_size;    // 128
        let nh_dn = config.deltanet_num_heads; // 4
        let hd_dn = config.deltanet_head_dim;  // 16
        let ck = config.deltanet_conv_kernel;   // 4
        let nq = config.num_attention_heads;    // 4
        let nkv = config.num_kv_heads;          // 2
        let hd_gqa = config.gqa_head_dim;       // 16
        let im = config.intermediate_size;      // 128
        let nh_hd_dn = nh_dn * hd_dn;          // 64

        let embed_tokens = pseudo_random_vec(100, vocab * d);
        let final_norm_weight = vec![1.0f32; d];
        let lm_head_weight = pseudo_random_vec(200, vocab * d);

        let max_kv_seq = 64;
        let (rope_cos, rope_sin) = make_rope_tables(max_kv_seq, hd_gqa);

        let mut layers = Vec::new();
        for layer_idx in 0..config.num_layers {
            let seed_base = (layer_idx as u64 + 1) * 1000;
            if config.is_full_attention(layer_idx) {
                // GQA layer
                layers.push(HybridLayer::Gqa(GqaDecoderLayer {
                    hidden_size: d,
                    num_q_heads: nq,
                    num_kv_heads: nkv,
                    head_dim: hd_gqa,
                    intermediate_size: im,
                    norm_eps: config.norm_eps as f32,
                    attn_norm_weight: vec![1.0; d],
                    w_q: pseudo_random_vec(seed_base + 1, nq * hd_gqa * d),
                    w_k: pseudo_random_vec(seed_base + 2, nkv * hd_gqa * d),
                    w_v: pseudo_random_vec(seed_base + 3, nkv * hd_gqa * d),
                    w_o: pseudo_random_vec(seed_base + 4, d * nq * hd_gqa),
                    q_norm_weight: vec![1.0; hd_gqa],
                    k_norm_weight: vec![1.0; hd_gqa],
                    ffn_norm_weight: vec![1.0; d],
                    ffn_gate_weight: pseudo_random_vec(seed_base + 5, im * d),
                    ffn_up_weight: pseudo_random_vec(seed_base + 6, im * d),
                    ffn_down_weight: pseudo_random_vec(seed_base + 7, d * im),
                }));
            } else {
                // DeltaNet layer
                layers.push(HybridLayer::DeltaNet(DeltaNetDecoderLayer {
                    hidden_size: d,
                    num_heads: nh_dn,
                    head_dim: hd_dn,
                    intermediate_size: im,
                    conv_kernel: ck,
                    norm_eps: config.norm_eps as f32,
                    attn_norm_weight: vec![1.0; d],
                    qkv_weight: pseudo_random_vec(seed_base + 1, 3 * nh_hd_dn * d),
                    gate_proj_weight: pseudo_random_vec(seed_base + 2, nh_hd_dn * d),
                    beta_proj_weight: pseudo_random_vec(seed_base + 3, nh_dn * d),
                    alpha_proj_weight: pseudo_random_vec(seed_base + 4, nh_dn * d),
                    q_conv_weight: pseudo_random_vec(seed_base + 5, nh_hd_dn * ck),
                    q_conv_bias: vec![0.0; nh_hd_dn],
                    k_conv_weight: pseudo_random_vec(seed_base + 6, nh_hd_dn * ck),
                    k_conv_bias: vec![0.0; nh_hd_dn],
                    v_conv_weight: pseudo_random_vec(seed_base + 7, nh_hd_dn * ck),
                    v_conv_bias: vec![0.0; nh_hd_dn],
                    o_norm_weight: vec![1.0; nh_hd_dn],
                    o_proj_weight: pseudo_random_vec(seed_base + 8, d * nh_hd_dn),
                    ffn_norm_weight: vec![1.0; d],
                    ffn_gate_weight: pseudo_random_vec(seed_base + 9, im * d),
                    ffn_up_weight: pseudo_random_vec(seed_base + 10, im * d),
                    ffn_down_weight: pseudo_random_vec(seed_base + 11, d * im),
                }));
            }
        }

        HybridModel::new(
            config,
            embed_tokens,
            layers,
            final_norm_weight,
            Some(lm_head_weight),
            rope_cos,
            rope_sin,
            max_kv_seq,
        )
    }

    #[test]
    fn test_hybrid_forward_produces_finite_logits() {
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
    fn test_hybrid_prefill_then_decode() {
        let model = make_test_model();
        let vocab = model.config.vocab_size;

        model.reset_state();
        let out1 = model.prefill(&[1, 2, 3]);
        assert_eq!(out1.shape, [1, 1, vocab]);
        for (i, &v) in out1.logits.iter().enumerate() {
            assert!(v.is_finite(), "prefill logit[{i}] = {v} is not finite");
        }

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
    fn test_hybrid_deltanet_state_constant() {
        let model = make_test_model();

        // DeltaNet state memory should be constant regardless of tokens processed
        let dn_mem_before = model.deltanet_state_memory_bytes();
        assert!(dn_mem_before > 0, "DeltaNet state memory should be nonzero");

        model.reset_state();
        let _ = model.prefill(&[1, 2, 3]);
        let dn_mem_after_prefill = model.deltanet_state_memory_bytes();
        assert_eq!(
            dn_mem_before, dn_mem_after_prefill,
            "DeltaNet state memory should not change after prefill"
        );

        let _ = model.decode_one(4);
        let _ = model.decode_one(5);
        let dn_mem_after_decode = model.deltanet_state_memory_bytes();
        assert_eq!(
            dn_mem_before, dn_mem_after_decode,
            "DeltaNet state memory should not change after decode steps"
        );

        // KV cache logical length should grow
        assert_eq!(
            model.kv_cache_len(),
            5,
            "KV cache should hold 5 tokens (3 prefill + 2 decode)"
        );
    }

    #[test]
    fn test_hybrid_reset_state() {
        let model = make_test_model();

        // Process some tokens
        model.reset_state();
        let out1 = model.prefill(&[1, 2, 3]);

        // Reset and process the same tokens again
        model.reset_state();
        let out2 = model.prefill(&[1, 2, 3]);

        // Should produce identical logits
        assert_eq!(out1.logits.len(), out2.logits.len());
        for (i, (&a, &b)) in out1.logits.iter().zip(out2.logits.iter()).enumerate() {
            assert!(
                (a - b).abs() < 1e-5,
                "logit[{i}] differs after reset: {a} vs {b}"
            );
        }

        // KV cache should be back to fresh state
        assert_eq!(model.kv_cache_len(), 3);
    }

    #[test]
    fn test_hybrid_layer_pattern() {
        let model = make_test_model();
        assert_eq!(model.layers.len(), 4);

        // Layers 0, 1, 2 should be DeltaNet; layer 3 should be GQA
        for (i, layer) in model.layers.iter().enumerate() {
            match layer {
                HybridLayer::DeltaNet(_) => assert!(
                    !model.config.is_full_attention(i),
                    "layer {i} is DeltaNet but config says full attention"
                ),
                HybridLayer::Gqa(_) => assert!(
                    model.config.is_full_attention(i),
                    "layer {i} is GQA but config says DeltaNet"
                ),
            }
        }
    }

    #[test]
    fn test_hybrid_model_from_weights() {
        use crate::weight_loading::AlignedBuffer;

        let config = HybridConfig::tiny_test();
        let d = config.hidden_size;
        let vocab = config.vocab_size;
        let im = config.intermediate_size;
        let nh_dn = config.deltanet_num_heads;
        let hd_dn = config.deltanet_head_dim;
        let nh_hd_dn = nh_dn * hd_dn;
        let ck = config.deltanet_conv_kernel;
        let nq = config.num_attention_heads;
        let nkv = config.num_kv_heads;
        let hd_gqa = config.gqa_head_dim;

        let mut weights: HashMap<String, RawTensor> = HashMap::new();
        let make = |shape: Vec<usize>, seed: u64| -> RawTensor {
            let len: usize = shape.iter().product();
            RawTensor {
                data: AlignedBuffer::from_slice(&pseudo_random_vec(seed, len)),
                shape,
            }
        };

        weights.insert("model.embed_tokens.weight".into(), make(vec![vocab, d], 10));
        weights.insert("model.norm.weight".into(), make(vec![d], 11));
        weights.insert("lm_head.weight".into(), make(vec![vocab, d], 12));

        for i in 0..config.num_layers {
            let s = (i as u64 + 1) * 100;
            let p = format!("model.layers.{i}");

            if config.is_full_attention(i) {
                weights.insert(format!("{p}.input_layernorm.weight"), make(vec![d], s));
                weights.insert(format!("{p}.self_attn.q_proj.weight"), make(vec![nq * hd_gqa, d], s+1));
                weights.insert(format!("{p}.self_attn.k_proj.weight"), make(vec![nkv * hd_gqa, d], s+2));
                weights.insert(format!("{p}.self_attn.v_proj.weight"), make(vec![nkv * hd_gqa, d], s+3));
                weights.insert(format!("{p}.self_attn.o_proj.weight"), make(vec![d, nq * hd_gqa], s+4));
                weights.insert(format!("{p}.self_attn.q_norm.weight"), make(vec![hd_gqa], s+5));
                weights.insert(format!("{p}.self_attn.k_norm.weight"), make(vec![hd_gqa], s+6));
                weights.insert(format!("{p}.post_attention_layernorm.weight"), make(vec![d], s+7));
                weights.insert(format!("{p}.mlp.gate_proj.weight"), make(vec![im, d], s+8));
                weights.insert(format!("{p}.mlp.up_proj.weight"), make(vec![im, d], s+9));
                weights.insert(format!("{p}.mlp.down_proj.weight"), make(vec![d, im], s+10));
            } else {
                weights.insert(format!("{p}.input_layernorm.weight"), make(vec![d], s));
                weights.insert(format!("{p}.self_attn.qkv_proj.weight"), make(vec![3 * nh_hd_dn, d], s+1));
                weights.insert(format!("{p}.self_attn.gate_proj.weight"), make(vec![nh_hd_dn, d], s+2));
                weights.insert(format!("{p}.self_attn.beta_proj.weight"), make(vec![nh_dn, d], s+3));
                weights.insert(format!("{p}.self_attn.alpha_proj.weight"), make(vec![nh_dn, d], s+4));
                weights.insert(format!("{p}.self_attn.q_conv1d.weight"), make(vec![nh_hd_dn, ck], s+5));
                weights.insert(format!("{p}.self_attn.q_conv1d.bias"), make(vec![nh_hd_dn], s+6));
                weights.insert(format!("{p}.self_attn.k_conv1d.weight"), make(vec![nh_hd_dn, ck], s+7));
                weights.insert(format!("{p}.self_attn.k_conv1d.bias"), make(vec![nh_hd_dn], s+8));
                weights.insert(format!("{p}.self_attn.v_conv1d.weight"), make(vec![nh_hd_dn, ck], s+9));
                weights.insert(format!("{p}.self_attn.v_conv1d.bias"), make(vec![nh_hd_dn], s+10));
                weights.insert(format!("{p}.self_attn.o_norm.weight"), make(vec![nh_hd_dn], s+11));
                weights.insert(format!("{p}.self_attn.o_proj.weight"), make(vec![d, nh_hd_dn], s+12));
                weights.insert(format!("{p}.post_attention_layernorm.weight"), make(vec![d], s+13));
                weights.insert(format!("{p}.mlp.gate_proj.weight"), make(vec![im, d], s+14));
                weights.insert(format!("{p}.mlp.up_proj.weight"), make(vec![im, d], s+15));
                weights.insert(format!("{p}.mlp.down_proj.weight"), make(vec![d, im], s+16));
            }
        }

        let model = HybridModel::from_weights(config, &weights, 64)
            .expect("from_weights should succeed");

        let output = model.forward(&[1, 2, 3]);
        assert_eq!(output.shape, [1, 1, model.config.vocab_size]);
        for (i, &v) in output.logits.iter().enumerate() {
            assert!(v.is_finite(), "from_weights logit[{i}] = {v} is not finite");
        }
    }
}
