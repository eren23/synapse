//! Post-norm bidirectional encoder matching HuggingFace RoBERTa numerics.
//!
//! A [`RoBERTaEncoder`] is a stack of [`RoBERTaLayer`] blocks wrapped by
//! [`RoBERTaEmbeddings`]. The embeddings reproduce RoBERTa's padding-index
//! quirk (position ids start at `pad_token_id + 1` for the first real token);
//! the layers use post-norm ordering plus padding-aware bidirectional
//! attention so that `last_hidden_state[:, 0, :]` — i.e. `[CLS]` — matches
//! `AutoModel("microsoft/unixcoder-base")` within ≤ 1e-4 max-abs.

use std::collections::{HashMap, HashSet};

use crate::ops::attention::bidirectional_attention;
use crate::ops::matmul::matmul_t;
use crate::ops::pure_rust_ops::layernorm_with_bias;
use crate::weight_loading::{AlignedBuffer, RawTensor, WeightError, WeightMapper};

/// Configuration matching HuggingFace's `RobertaConfig` fields.
#[derive(Debug, Clone)]
pub struct RoBERTaConfig {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub intermediate_size: usize,
    pub max_position_embeddings: usize,
    pub type_vocab_size: usize,
    pub layer_norm_eps: f32,
    pub pad_token_id: u32,
}

impl RoBERTaConfig {
    pub fn head_dim(&self) -> usize {
        self.hidden_size / self.num_attention_heads
    }
}

/// Parse an HF-style `config.json` (as shipped by `microsoft/unixcoder-base`)
/// into a [`RoBERTaConfig`]. Missing fields fall back to RoBERTa-base defaults.
pub fn parse_roberta_config(json: &str) -> Result<RoBERTaConfig, Box<dyn std::error::Error>> {
    let v: serde_json::Value = serde_json::from_str(json)?;
    Ok(RoBERTaConfig {
        vocab_size: v["vocab_size"].as_u64().unwrap_or(50265) as usize,
        hidden_size: v["hidden_size"].as_u64().unwrap_or(768) as usize,
        num_hidden_layers: v["num_hidden_layers"].as_u64().unwrap_or(12) as usize,
        num_attention_heads: v["num_attention_heads"].as_u64().unwrap_or(12) as usize,
        intermediate_size: v["intermediate_size"].as_u64().unwrap_or(3072) as usize,
        max_position_embeddings: v["max_position_embeddings"].as_u64().unwrap_or(514) as usize,
        type_vocab_size: v["type_vocab_size"].as_u64().unwrap_or(1) as usize,
        layer_norm_eps: v["layer_norm_eps"].as_f64().unwrap_or(1e-5) as f32,
        pad_token_id: v["pad_token_id"].as_u64().unwrap_or(1) as u32,
    })
}

// ── Activation ──────────────────────────────────────────────────────────────
//
// HF RoBERTa/UniXcoder uses PyTorch's default `F.gelu` (exact erf form), not
// the tanh approximation in [`crate::ops::pure_rust_ops::gelu`]. The two
// differ by up to ~4e-4 per element — enough to blow a 1e-4 parity test.
//
// Abramowitz & Stegun 7.1.26 gives erf accurate to ~1.5e-7, which is below
// single-precision rounding, so the result is bit-equivalent to HF on the
// ranges that occur in practice.

fn erf(x: f32) -> f32 {
    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let x = x.abs();
    let a1 = 0.254829592f32;
    let a2 = -0.284496736f32;
    let a3 = 1.421413741f32;
    let a4 = -1.453152027f32;
    let a5 = 1.061405429f32;
    let p = 0.3275911f32;
    let t = 1.0 / (1.0 + p * x);
    let y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * (-x * x).exp();
    sign * y
}

fn gelu_erf(x: f32) -> f32 {
    0.5 * x * (1.0 + erf(x / std::f32::consts::SQRT_2))
}

// ── Embeddings ──────────────────────────────────────────────────────────────

/// Word + position + token_type embeddings, summed and layer-normalized.
#[derive(Debug)]
pub struct RoBERTaEmbeddings {
    pub hidden_size: usize,
    pub max_position_embeddings: usize,
    pub type_vocab_size: usize,
    pub pad_token_id: u32,
    pub layer_norm_eps: f32,

    pub word: AlignedBuffer,      // [vocab, hidden]
    pub position: AlignedBuffer,  // [max_pos, hidden]
    pub token_type: AlignedBuffer,// [type_vocab, hidden]
    pub ln_weight: AlignedBuffer, // [hidden]
    pub ln_bias: AlignedBuffer,   // [hidden]
}

impl RoBERTaEmbeddings {
    pub fn new(cfg: &RoBERTaConfig) -> Self {
        Self {
            hidden_size: cfg.hidden_size,
            max_position_embeddings: cfg.max_position_embeddings,
            type_vocab_size: cfg.type_vocab_size,
            pad_token_id: cfg.pad_token_id,
            layer_norm_eps: cfg.layer_norm_eps,
            word: AlignedBuffer::new_zeroed(0),
            position: AlignedBuffer::new_zeroed(0),
            token_type: AlignedBuffer::new_zeroed(0),
            ln_weight: AlignedBuffer::new_zeroed(0),
            ln_bias: AlignedBuffer::new_zeroed(0),
        }
    }

    /// HuggingFace RoBERTa's position-id construction, mirroring
    /// `transformers.models.roberta.modeling_roberta.create_position_ids_from_input_ids`
    /// verbatim:
    ///
    /// ```text
    /// mask = attention_mask.long()                    # 1 for real, 0 for pad
    /// incremental = cumsum(mask, dim=1) * mask
    /// position_ids = incremental + pad_token_id
    /// ```
    ///
    /// For `pad_token_id = 1` and `attention_mask = [1, 1, 1, 0, 0]` this
    /// yields `[2, 3, 4, 1, 1]` — pad positions collapse back to `pad_id`.
    /// This matches the way UniXcoder was pretrained (position `0` is
    /// reserved).
    pub fn position_ids(&self, attention_mask: &[i64], seq_len: usize) -> Vec<usize> {
        let mut out = vec![0usize; seq_len];
        let mut running: i64 = 0;
        for (i, &m) in attention_mask.iter().enumerate().take(seq_len) {
            running += m;
            let inc = running * m;
            out[i] = inc as usize + self.pad_token_id as usize;
        }
        out
    }

    /// Build the embedding output for a single sequence (batch size 1).
    /// `input_ids` and `attention_mask` have length `seq_len`. Returns a
    /// flat `[seq_len, hidden_size]` buffer in row-major order.
    pub fn forward(&self, input_ids: &[i64], attention_mask: &[i64]) -> Vec<f32> {
        let seq_len = input_ids.len();
        let h = self.hidden_size;
        let mut out = vec![0.0f32; seq_len * h];

        // 1. Word embeddings. Clamp to vocab_size - 1 so a mis-tokenized
        //    id from a JS-side BPE can't trap the WASM module — an
        //    out-of-range id is a caller bug, but we'd rather return a
        //    degenerate feature than `RuntimeError: unreachable`.
        let vocab_rows = if h > 0 { self.word.len() / h } else { 0 };
        for (i, &tok) in input_ids.iter().enumerate() {
            let tok = (tok as usize).min(vocab_rows.saturating_sub(1));
            let src = &self.word[tok * h..(tok + 1) * h];
            out[i * h..(i + 1) * h].copy_from_slice(src);
        }

        // 2. Position embeddings with the pad-aware cumsum trick.
        let pos_rows = if h > 0 { self.position.len() / h } else { 0 };
        let pos_ids = self.position_ids(attention_mask, seq_len);
        for (i, &p) in pos_ids.iter().enumerate() {
            let p = p.min(pos_rows.saturating_sub(1));
            let src = &self.position[p * h..(p + 1) * h];
            for j in 0..h {
                out[i * h + j] += src[j];
            }
        }

        // 3. Token-type embeddings. We always pass `token_type_ids = 0`
        //    (same as HF `AutoModel` default when the user doesn't supply
        //    them), so only the row 0 slice is ever used.
        let type_row = &self.token_type[..h];
        for i in 0..seq_len {
            for j in 0..h {
                out[i * h + j] += type_row[j];
            }
        }

        // 4. LayerNorm with bias, eps matches the HF config.
        layernorm_with_bias(&out, &self.ln_weight, &self.ln_bias,
                            self.layer_norm_eps, h)
    }
}

// ── Encoder layer ───────────────────────────────────────────────────────────

/// Padding-aware bidirectional attention. Tokens where
/// `attention_mask[i] == 0` are excluded from the softmax (scores become
/// `-inf`) so `[CLS]` does not attend to padding slots.
fn masked_bidirectional_attention(
    q: &[f32], k: &[f32], v: &[f32],
    seq_len: usize, num_heads: usize, head_dim: usize,
    attention_mask: &[i64],
) -> Vec<f32> {
    let qk_dim = num_heads * head_dim;
    let scale = 1.0 / (head_dim as f32).sqrt();
    let mut output = vec![0.0f32; seq_len * qk_dim];
    for head in 0..num_heads {
        for qi in 0..seq_len {
            let mut scores = vec![f32::NEG_INFINITY; seq_len];
            let mut max = f32::NEG_INFINITY;
            for ki in 0..seq_len {
                if attention_mask[ki] == 0 {
                    continue;
                }
                let mut dot = 0.0f32;
                for d in 0..head_dim {
                    dot += q[qi * qk_dim + head * head_dim + d]
                         * k[ki * qk_dim + head * head_dim + d];
                }
                let s = dot * scale;
                scores[ki] = s;
                if s > max {
                    max = s;
                }
            }
            let mut sum = 0.0f32;
            for s in scores.iter_mut() {
                if s.is_finite() {
                    *s = (*s - max).exp();
                    sum += *s;
                } else {
                    *s = 0.0;
                }
            }
            if sum > 0.0 {
                for s in scores.iter_mut() {
                    *s /= sum;
                }
            }
            for d in 0..head_dim {
                let mut val = 0.0f32;
                for ki in 0..seq_len {
                    val += scores[ki] * v[ki * qk_dim + head * head_dim + d];
                }
                output[qi * qk_dim + head * head_dim + d] = val;
            }
        }
    }
    output
}

/// One post-norm RoBERTa encoder block.
///
/// HF structure (`RobertaLayer`):
/// ```text
///     attn_out = self_attn_out.dense(attn_probs @ V)             # + bias
///     x = LN_attn(x + attn_out)
///     mlp_hidden = gelu(intermediate.dense(x))                   # + bias
///     mlp_out = output.dense(mlp_hidden)                         # + bias
///     x = LN_ffn(x + mlp_out)
/// ```
#[derive(Debug)]
pub struct RoBERTaLayer {
    pub hidden_size: usize,
    pub num_heads: usize,
    pub intermediate_size: usize,
    pub layer_norm_eps: f32,

    // Q/K/V projections. HF stores .weight as [hidden, hidden] so
    // `matmul_t(x, w, S, H, H)` computes `x @ w^T`.
    pub w_q: AlignedBuffer, pub q_bias: AlignedBuffer,
    pub w_k: AlignedBuffer, pub k_bias: AlignedBuffer,
    pub w_v: AlignedBuffer, pub v_bias: AlignedBuffer,
    pub w_o: AlignedBuffer, pub o_bias: AlignedBuffer,

    pub attn_ln_weight: AlignedBuffer, pub attn_ln_bias: AlignedBuffer,

    // FFN: intermediate.dense [inter, hidden], output.dense [hidden, inter].
    pub w_up: AlignedBuffer,   pub up_bias: AlignedBuffer,
    pub w_down: AlignedBuffer, pub down_bias: AlignedBuffer,

    pub ffn_ln_weight: AlignedBuffer, pub ffn_ln_bias: AlignedBuffer,
}

impl RoBERTaLayer {
    pub fn new(cfg: &RoBERTaConfig) -> Self {
        Self {
            hidden_size: cfg.hidden_size,
            num_heads: cfg.num_attention_heads,
            intermediate_size: cfg.intermediate_size,
            layer_norm_eps: cfg.layer_norm_eps,
            w_q: AlignedBuffer::new_zeroed(0), q_bias: AlignedBuffer::new_zeroed(0),
            w_k: AlignedBuffer::new_zeroed(0), k_bias: AlignedBuffer::new_zeroed(0),
            w_v: AlignedBuffer::new_zeroed(0), v_bias: AlignedBuffer::new_zeroed(0),
            w_o: AlignedBuffer::new_zeroed(0), o_bias: AlignedBuffer::new_zeroed(0),
            attn_ln_weight: AlignedBuffer::new_zeroed(0), attn_ln_bias: AlignedBuffer::new_zeroed(0),
            w_up: AlignedBuffer::new_zeroed(0),   up_bias: AlignedBuffer::new_zeroed(0),
            w_down: AlignedBuffer::new_zeroed(0), down_bias: AlignedBuffer::new_zeroed(0),
            ffn_ln_weight: AlignedBuffer::new_zeroed(0), ffn_ln_bias: AlignedBuffer::new_zeroed(0),
        }
    }

    fn add_col_bias(x: &mut [f32], bias: &[f32], rows: usize, cols: usize) {
        if bias.is_empty() { return; }
        for r in 0..rows {
            for c in 0..cols {
                x[r * cols + c] += bias[c];
            }
        }
    }

    /// `x` is `[seq_len, hidden]`. Returns the same shape.
    pub fn forward(&self, x: &[f32], seq_len: usize, attention_mask: Option<&[i64]>) -> Vec<f32> {
        let h = self.hidden_size;
        let head_dim = h / self.num_heads;

        let mut q = matmul_t(x, &self.w_q, seq_len, h, h);
        Self::add_col_bias(&mut q, &self.q_bias, seq_len, h);
        let mut k = matmul_t(x, &self.w_k, seq_len, h, h);
        Self::add_col_bias(&mut k, &self.k_bias, seq_len, h);
        let mut v = matmul_t(x, &self.w_v, seq_len, h, h);
        Self::add_col_bias(&mut v, &self.v_bias, seq_len, h);

        let attn = match attention_mask {
            Some(mask) => masked_bidirectional_attention(
                &q, &k, &v, seq_len, self.num_heads, head_dim, mask),
            None => bidirectional_attention(
                &q, &k, &v, seq_len, self.num_heads, head_dim),
        };

        let mut out = matmul_t(&attn, &self.w_o, seq_len, h, h);
        Self::add_col_bias(&mut out, &self.o_bias, seq_len, h);

        // Residual + post-norm (attention sub-layer).
        let mut resid = vec![0.0f32; seq_len * h];
        for i in 0..resid.len() { resid[i] = x[i] + out[i]; }
        let normed = layernorm_with_bias(&resid, &self.attn_ln_weight, &self.attn_ln_bias,
                                         self.layer_norm_eps, h);

        // Feed-forward sub-layer.
        let mut up = matmul_t(&normed, &self.w_up, seq_len, h, self.intermediate_size);
        Self::add_col_bias(&mut up, &self.up_bias, seq_len, self.intermediate_size);
        for v in up.iter_mut() { *v = gelu_erf(*v); }
        let mut down = matmul_t(&up, &self.w_down, seq_len, self.intermediate_size, h);
        Self::add_col_bias(&mut down, &self.down_bias, seq_len, h);

        for i in 0..resid.len() { resid[i] = normed[i] + down[i]; }
        layernorm_with_bias(&resid, &self.ffn_ln_weight, &self.ffn_ln_bias,
                            self.layer_norm_eps, h)
    }

    fn set_weight(&mut self, field: &str, t: &RawTensor) {
        match field {
            "attention.w_q"        => self.w_q = t.data.clone(),
            "attention.q_bias"     => self.q_bias = t.data.clone(),
            "attention.w_k"        => self.w_k = t.data.clone(),
            "attention.k_bias"     => self.k_bias = t.data.clone(),
            "attention.w_v"        => self.w_v = t.data.clone(),
            "attention.v_bias"     => self.v_bias = t.data.clone(),
            "attention.w_o"        => self.w_o = t.data.clone(),
            "attention.o_bias"     => self.o_bias = t.data.clone(),
            "attn_norm.weight"     => self.attn_ln_weight = t.data.clone(),
            "attn_norm.bias"       => self.attn_ln_bias = t.data.clone(),
            "ffn.w_up"             => self.w_up = t.data.clone(),
            "ffn.up_bias"          => self.up_bias = t.data.clone(),
            "ffn.w_down"           => self.w_down = t.data.clone(),
            "ffn.down_bias"        => self.down_bias = t.data.clone(),
            "ffn_norm.weight"      => self.ffn_ln_weight = t.data.clone(),
            "ffn_norm.bias"        => self.ffn_ln_bias = t.data.clone(),
            _ => {}
        }
    }

    fn expected_keys(&self, i: usize) -> Vec<String> {
        vec![
            format!("layers[{i}].attention.w_q"),
            format!("layers[{i}].attention.q_bias"),
            format!("layers[{i}].attention.w_k"),
            format!("layers[{i}].attention.k_bias"),
            format!("layers[{i}].attention.w_v"),
            format!("layers[{i}].attention.v_bias"),
            format!("layers[{i}].attention.w_o"),
            format!("layers[{i}].attention.o_bias"),
            format!("layers[{i}].attn_norm.weight"),
            format!("layers[{i}].attn_norm.bias"),
            format!("layers[{i}].ffn.w_up"),
            format!("layers[{i}].ffn.up_bias"),
            format!("layers[{i}].ffn.w_down"),
            format!("layers[{i}].ffn.down_bias"),
            format!("layers[{i}].ffn_norm.weight"),
            format!("layers[{i}].ffn_norm.bias"),
        ]
    }
}

// ── Encoder ─────────────────────────────────────────────────────────────────

#[derive(Debug)]
pub struct RoBERTaEncoder {
    pub config: RoBERTaConfig,
    pub embeddings: RoBERTaEmbeddings,
    pub layers: Vec<RoBERTaLayer>,
}

impl RoBERTaEncoder {
    pub fn from_config(config: RoBERTaConfig) -> Self {
        let embeddings = RoBERTaEmbeddings::new(&config);
        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for _ in 0..config.num_hidden_layers {
            layers.push(RoBERTaLayer::new(&config));
        }
        Self { config, embeddings, layers }
    }

    /// Run the full encoder on a single sequence.
    ///
    /// Returns the final `last_hidden_state` of shape `[seq_len, hidden]`
    /// (row-major, flat). `attention_mask` has length `seq_len` and is `1`
    /// for real tokens, `0` for pads.
    pub fn forward(&self, input_ids: &[i64], attention_mask: &[i64]) -> Vec<f32> {
        let seq_len = input_ids.len();
        let mut h = self.embeddings.forward(input_ids, attention_mask);
        for layer in &self.layers {
            h = layer.forward(&h, seq_len, Some(attention_mask));
        }
        h
    }

    /// The CLS feature (`last_hidden_state[:, 0, :]`) used by the CDT paper.
    pub fn cls_feature(&self, input_ids: &[i64], attention_mask: &[i64]) -> Vec<f32> {
        let h = self.forward(input_ids, attention_mask);
        h[..self.config.hidden_size].to_vec()
    }

    // ── weight loading ─────────────────────────────────────────────────────

    fn expected_weight_keys(&self) -> Vec<String> {
        let mut keys = vec![
            "embeddings.word".to_string(),
            "embeddings.position".to_string(),
            "embeddings.token_type".to_string(),
            "embeddings.ln.weight".to_string(),
            "embeddings.ln.bias".to_string(),
        ];
        for i in 0..self.layers.len() {
            keys.extend(self.layers[i].expected_keys(i));
        }
        keys
    }

    fn set_weight(&mut self, key: &str, t: &RawTensor) -> Result<(), WeightError> {
        match key {
            "embeddings.word"        => self.embeddings.word = t.data.clone(),
            "embeddings.position"    => self.embeddings.position = t.data.clone(),
            "embeddings.token_type"  => self.embeddings.token_type = t.data.clone(),
            "embeddings.ln.weight"   => self.embeddings.ln_weight = t.data.clone(),
            "embeddings.ln.bias"     => self.embeddings.ln_bias = t.data.clone(),
            _ if key.starts_with("layers[") => {
                if let Some((idx, field)) = parse_layer_key(key) {
                    if let Some(layer) = self.layers.get_mut(idx) {
                        layer.set_weight(field, t);
                    }
                }
            }
            _ => {}
        }
        Ok(())
    }

    /// Load weights from an HF-style tensor dict. The mapper should rename
    /// `roberta.*` → the canonical names above; see
    /// [`WeightMapper::unixcoder`](crate::weight_loading::WeightMapper::unixcoder).
    pub fn load_weights(
        &mut self,
        weights: HashMap<String, RawTensor>,
        mapper: &WeightMapper,
    ) -> Result<crate::models::lm::LoadResult, WeightError> {
        let source_keys: Vec<String> = weights.keys().cloned().collect();
        let mapping = mapper.map_keys(&source_keys);

        let expected: HashSet<String> = self.expected_weight_keys().into_iter().collect();
        let mut loaded: HashSet<String> = HashSet::new();

        for (source, target) in &mapping.mapping {
            if let Some(raw) = weights.get(source) {
                self.set_weight(target, raw)?;
                if expected.contains(target) {
                    loaded.insert(target.clone());
                }
            }
        }

        let missing: Vec<String> = expected.difference(&loaded).cloned().collect();
        let unexpected = mapping.unmapped;
        Ok(crate::models::lm::LoadResult { missing, unexpected })
    }
}

fn parse_layer_key(key: &str) -> Option<(usize, &str)> {
    let rest = key.strip_prefix("layers[")?;
    let bracket = rest.find(']')?;
    let idx: usize = rest[..bracket].parse().ok()?;
    let field = rest[bracket + 1..].strip_prefix('.')?;
    Some((idx, field))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn position_ids_match_hf_cumsum_trick() {
        // Example from the HF source comment:
        // attention_mask = [1, 1, 1, 0, 0], pad_token_id = 1.
        // Expected position_ids = [2, 3, 4, 1, 1].
        let cfg = crate::models::text_encoder::unixcoder_base();
        let emb = RoBERTaEmbeddings::new(&cfg);
        let mask = vec![1i64, 1, 1, 0, 0];
        let ids = emb.position_ids(&mask, 5);
        assert_eq!(ids, vec![2, 3, 4, 1, 1]);
    }

    #[test]
    fn gelu_erf_matches_pytorch_reference_values() {
        // Values from PyTorch: `torch.nn.functional.gelu(torch.tensor([...]))`
        // exact mode (approximate='none').
        let cases = [
            (-3.0_f32, -0.0040_f32),
            (-1.0, -0.1587),
            ( 0.0,  0.0),
            ( 0.5,  0.3457),
            ( 1.0,  0.8413),
            ( 2.0,  1.9545),
            ( 3.0,  2.9960),
        ];
        for (x, expected) in cases {
            let got = gelu_erf(x);
            assert!((got - expected).abs() < 2e-4,
                    "gelu_erf({x}) = {got}, expected ~{expected}");
        }
    }
}
