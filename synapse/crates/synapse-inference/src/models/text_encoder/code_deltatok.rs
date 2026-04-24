//! CodeDeltaTok head — rides on top of a frozen UniXcoder backbone to
//! compress each (before, after) code pair into a single dense 768-dim
//! delta token plus a reconstruction of the after-state.
//!
//! Port of `architectures/code_deltatok/code_deltatok.py` in the
//! crucible-community-tap. The reference Python uses
//! `torch.nn.MultiheadAttention` (fused `in_proj_weight`) and pre-norm
//! transformer blocks with LayerScale + SwiGLU MLP. We mirror the forward
//! pass exactly so bit-level parity with a PyTorch state_dict is
//! achievable.

use std::collections::{HashMap, HashSet};

use crate::ops::attention::bidirectional_attention;
use crate::ops::matmul::matmul_t;
use crate::ops::pure_rust_ops::{layernorm_with_bias, silu};
use crate::weight_loading::{AlignedBuffer, RawTensor, WeightError, WeightMapper};

/// Configuration for the CDT head. Defaults mirror the paper's headline
/// run (`cdt-K1-4blk-contrast0.1-s42-5k`).
#[derive(Debug, Clone)]
pub struct CodeDeltaTokConfig {
    pub feature_dim: usize,
    pub num_blocks: usize,
    pub num_heads: usize,
    pub num_delta_tokens: usize,
    pub mlp_ratio: f32,
    pub layer_norm_eps: f32,
}

impl CodeDeltaTokConfig {
    pub fn paper_default() -> Self {
        Self {
            feature_dim: 768,
            num_blocks: 4,
            num_heads: 12,
            num_delta_tokens: 1,
            mlp_ratio: 4.0,
            layer_norm_eps: 1e-5,
        }
    }

    pub fn head_dim(&self) -> usize {
        self.feature_dim / self.num_heads
    }

    pub fn intermediate(&self) -> usize {
        (self.feature_dim as f32 * self.mlp_ratio) as usize
    }
}

// ── Block ──────────────────────────────────────────────────────────────────

/// Pre-norm transformer block with LayerScale + SwiGLU MLP, matching
/// `DeltaTokBlock` in the reference Python.
#[derive(Debug)]
pub struct DeltaTokBlock {
    pub dim: usize,
    pub num_heads: usize,
    pub intermediate: usize,
    pub layer_norm_eps: f32,

    pub norm1_w: AlignedBuffer, pub norm1_b: AlignedBuffer,
    pub norm2_w: AlignedBuffer, pub norm2_b: AlignedBuffer,

    // nn.MultiheadAttention fuses Q/K/V into in_proj_weight[3D, D]. We split
    // on load so the Rust forward can run ordinary q/k/v matmuls.
    pub w_q: AlignedBuffer, pub q_bias: AlignedBuffer,
    pub w_k: AlignedBuffer, pub k_bias: AlignedBuffer,
    pub w_v: AlignedBuffer, pub v_bias: AlignedBuffer,
    pub w_o: AlignedBuffer, pub o_bias: AlignedBuffer,

    pub mlp_gate_w: AlignedBuffer, pub mlp_gate_b: AlignedBuffer,
    pub mlp_up_w:   AlignedBuffer, pub mlp_up_b:   AlignedBuffer,
    pub mlp_down_w: AlignedBuffer, pub mlp_down_b: AlignedBuffer,

    pub scale1: AlignedBuffer,
    pub scale2: AlignedBuffer,
}

impl DeltaTokBlock {
    pub fn new(cfg: &CodeDeltaTokConfig) -> Self {
        Self {
            dim: cfg.feature_dim,
            num_heads: cfg.num_heads,
            intermediate: cfg.intermediate(),
            layer_norm_eps: cfg.layer_norm_eps,
            norm1_w: AlignedBuffer::new_zeroed(0), norm1_b: AlignedBuffer::new_zeroed(0),
            norm2_w: AlignedBuffer::new_zeroed(0), norm2_b: AlignedBuffer::new_zeroed(0),
            w_q: AlignedBuffer::new_zeroed(0), q_bias: AlignedBuffer::new_zeroed(0),
            w_k: AlignedBuffer::new_zeroed(0), k_bias: AlignedBuffer::new_zeroed(0),
            w_v: AlignedBuffer::new_zeroed(0), v_bias: AlignedBuffer::new_zeroed(0),
            w_o: AlignedBuffer::new_zeroed(0), o_bias: AlignedBuffer::new_zeroed(0),
            mlp_gate_w: AlignedBuffer::new_zeroed(0), mlp_gate_b: AlignedBuffer::new_zeroed(0),
            mlp_up_w:   AlignedBuffer::new_zeroed(0), mlp_up_b:   AlignedBuffer::new_zeroed(0),
            mlp_down_w: AlignedBuffer::new_zeroed(0), mlp_down_b: AlignedBuffer::new_zeroed(0),
            scale1: AlignedBuffer::new_zeroed(0),
            scale2: AlignedBuffer::new_zeroed(0),
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

    /// `x` has shape `[seq_len, dim]` (flat, row-major). Returns same shape.
    pub fn forward(&self, x: &[f32], seq_len: usize) -> Vec<f32> {
        let d = self.dim;
        let head_dim = d / self.num_heads;

        // ── Self-attention sub-layer (pre-norm + LayerScale residual). ──
        let h1 = layernorm_with_bias(x, &self.norm1_w, &self.norm1_b, self.layer_norm_eps, d);
        let mut q = matmul_t(&h1, &self.w_q, seq_len, d, d);
        Self::add_col_bias(&mut q, &self.q_bias, seq_len, d);
        let mut k = matmul_t(&h1, &self.w_k, seq_len, d, d);
        Self::add_col_bias(&mut k, &self.k_bias, seq_len, d);
        let mut v = matmul_t(&h1, &self.w_v, seq_len, d, d);
        Self::add_col_bias(&mut v, &self.v_bias, seq_len, d);

        // CDT never sees padding inside the fixed-length [z, prev, next]
        // sequence, so plain bidirectional attention is exact.
        let attn = bidirectional_attention(&q, &k, &v, seq_len, self.num_heads, head_dim);
        let mut attn_out = matmul_t(&attn, &self.w_o, seq_len, d, d);
        Self::add_col_bias(&mut attn_out, &self.o_bias, seq_len, d);

        // Residual with LayerScale: x = x + scale1 * attn_out.
        let mut y = vec![0.0f32; seq_len * d];
        for i in 0..seq_len {
            for j in 0..d {
                y[i * d + j] = x[i * d + j] + self.scale1[j] * attn_out[i * d + j];
            }
        }

        // ── Feed-forward sub-layer. ────────────────────────────────────
        let h2 = layernorm_with_bias(&y, &self.norm2_w, &self.norm2_b, self.layer_norm_eps, d);
        let inter = self.intermediate;
        let mut gate = matmul_t(&h2, &self.mlp_gate_w, seq_len, d, inter);
        Self::add_col_bias(&mut gate, &self.mlp_gate_b, seq_len, inter);
        for v in gate.iter_mut() { *v = silu(*v); }

        let mut up = matmul_t(&h2, &self.mlp_up_w, seq_len, d, inter);
        Self::add_col_bias(&mut up, &self.mlp_up_b, seq_len, inter);

        // gate * up (elementwise).
        for i in 0..gate.len() {
            gate[i] *= up[i];
        }

        let mut down = matmul_t(&gate, &self.mlp_down_w, seq_len, inter, d);
        Self::add_col_bias(&mut down, &self.mlp_down_b, seq_len, d);

        // Residual with LayerScale: y = y + scale2 * down.
        for i in 0..seq_len {
            for j in 0..d {
                y[i * d + j] += self.scale2[j] * down[i * d + j];
            }
        }

        y
    }

    fn set_weight(&mut self, field: &str, t: &RawTensor) {
        match field {
            "norm1.weight" => self.norm1_w = t.data.clone(),
            "norm1.bias"   => self.norm1_b = t.data.clone(),
            "norm2.weight" => self.norm2_w = t.data.clone(),
            "norm2.bias"   => self.norm2_b = t.data.clone(),

            // Split fused in_proj_weight [3D, D] into Q/K/V row slices.
            "attn.in_proj_weight" => {
                let d = self.dim;
                assert!(t.data.len() == 3 * d * d,
                        "in_proj_weight len {}, expected {}", t.data.len(), 3 * d * d);
                self.w_q = AlignedBuffer::from_slice(&t.data[0..d * d]);
                self.w_k = AlignedBuffer::from_slice(&t.data[d * d..2 * d * d]);
                self.w_v = AlignedBuffer::from_slice(&t.data[2 * d * d..3 * d * d]);
            }
            "attn.in_proj_bias" => {
                let d = self.dim;
                assert!(t.data.len() == 3 * d);
                self.q_bias = AlignedBuffer::from_slice(&t.data[0..d]);
                self.k_bias = AlignedBuffer::from_slice(&t.data[d..2 * d]);
                self.v_bias = AlignedBuffer::from_slice(&t.data[2 * d..3 * d]);
            }
            "attn.out_proj.weight" => self.w_o = t.data.clone(),
            "attn.out_proj.bias"   => self.o_bias = t.data.clone(),

            "mlp_gate.weight" => self.mlp_gate_w = t.data.clone(),
            "mlp_gate.bias"   => self.mlp_gate_b = t.data.clone(),
            "mlp_up.weight"   => self.mlp_up_w = t.data.clone(),
            "mlp_up.bias"     => self.mlp_up_b = t.data.clone(),
            "mlp_down.weight" => self.mlp_down_w = t.data.clone(),
            "mlp_down.bias"   => self.mlp_down_b = t.data.clone(),
            "scale1" => self.scale1 = t.data.clone(),
            "scale2" => self.scale2 = t.data.clone(),
            _ => {}
        }
    }
}

// ── Head ───────────────────────────────────────────────────────────────────

/// CodeDeltaTok encoder/decoder head. Takes pre-computed UniXcoder CLS
/// features and produces a `K × D` delta token and the decoder's
/// `D`-dim reconstruction of the after-state.
#[derive(Debug)]
pub struct CodeDeltaTokHead {
    pub config: CodeDeltaTokConfig,

    pub z_embed:   AlignedBuffer,  // [K*D]
    pub pos_prev:  AlignedBuffer,  // [D]
    pub pos_next:  AlignedBuffer,  // [D]
    pub pos_z:     AlignedBuffer,  // [K*D]

    pub encoder: Vec<DeltaTokBlock>,
    pub enc_norm_w: AlignedBuffer,
    pub enc_norm_b: AlignedBuffer,

    pub decoder: Vec<DeltaTokBlock>,
    pub dec_norm_w: AlignedBuffer,
    pub dec_norm_b: AlignedBuffer,

    pub out_proj_w: AlignedBuffer, // [D, D]
    pub out_proj_b: AlignedBuffer, // [D]
}

impl CodeDeltaTokHead {
    pub fn from_config(config: CodeDeltaTokConfig) -> Self {
        let enc = (0..config.num_blocks).map(|_| DeltaTokBlock::new(&config)).collect();
        let dec = (0..config.num_blocks).map(|_| DeltaTokBlock::new(&config)).collect();
        Self {
            z_embed:  AlignedBuffer::new_zeroed(0),
            pos_prev: AlignedBuffer::new_zeroed(0),
            pos_next: AlignedBuffer::new_zeroed(0),
            pos_z:    AlignedBuffer::new_zeroed(0),
            encoder: enc,
            enc_norm_w: AlignedBuffer::new_zeroed(0), enc_norm_b: AlignedBuffer::new_zeroed(0),
            decoder: dec,
            dec_norm_w: AlignedBuffer::new_zeroed(0), dec_norm_b: AlignedBuffer::new_zeroed(0),
            out_proj_w: AlignedBuffer::new_zeroed(0),
            out_proj_b: AlignedBuffer::new_zeroed(0),
            config,
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

    /// Encode `(h_b, h_a)` pair (each flat `[D]`) into a flat `[K*D]`
    /// delta-token buffer. Matches `CodeDeltaTok.encode`.
    pub fn encode(&self, h_b: &[f32], h_a: &[f32]) -> Vec<f32> {
        let cfg = &self.config;
        let d = cfg.feature_dim;
        let k = cfg.num_delta_tokens;

        // Build [z, prev, next] sequence of length K + 2.
        let mut seq = vec![0.0f32; (k + 2) * d];
        // z + pos_z
        for i in 0..k {
            for j in 0..d {
                seq[i * d + j] = self.z_embed[i * d + j] + self.pos_z[i * d + j];
            }
        }
        // prev + pos_prev
        for j in 0..d {
            seq[k * d + j] = h_b[j] + self.pos_prev[j];
        }
        // next + pos_next
        for j in 0..d {
            seq[(k + 1) * d + j] = h_a[j] + self.pos_next[j];
        }

        let mut h = seq;
        for blk in &self.encoder {
            h = blk.forward(&h, k + 2);
        }
        let h = layernorm_with_bias(&h, &self.enc_norm_w, &self.enc_norm_b,
                                    cfg.layer_norm_eps, d);

        // Return first K positions flattened.
        let mut delta = vec![0.0f32; k * d];
        delta.copy_from_slice(&h[..k * d]);
        delta
    }

    /// Decode `(delta, h_b)` back into a flat `[D]` reconstruction of the
    /// after-state. Matches `CodeDeltaTok.decode`.
    pub fn decode(&self, delta: &[f32], h_b: &[f32]) -> Vec<f32> {
        let cfg = &self.config;
        let d = cfg.feature_dim;
        let k = cfg.num_delta_tokens;

        let mut seq = vec![0.0f32; (k + 1) * d];
        seq[..k * d].copy_from_slice(delta);
        for j in 0..d {
            seq[k * d + j] = h_b[j] + self.pos_prev[j];
        }

        let mut h = seq;
        for blk in &self.decoder {
            h = blk.forward(&h, k + 1);
        }
        let h = layernorm_with_bias(&h, &self.dec_norm_w, &self.dec_norm_b,
                                    cfg.layer_norm_eps, d);

        // Take the prev position (index K, after the K delta slots).
        let slice = &h[k * d..(k + 1) * d];
        let mut recon = matmul_t(slice, &self.out_proj_w, 1, d, d);
        Self::add_col_bias(&mut recon, &self.out_proj_b, 1, d);
        recon
    }

    // ── weight loading ─────────────────────────────────────────────────────

    fn expected_weight_keys(&self) -> Vec<String> {
        let mut keys = vec![
            "z_embed".to_string(),
            "pos_prev".to_string(),
            "pos_next".to_string(),
            "pos_z".to_string(),
            "encoder_norm.weight".to_string(),
            "encoder_norm.bias".to_string(),
            "decoder_norm.weight".to_string(),
            "decoder_norm.bias".to_string(),
            "out_proj.weight".to_string(),
            "out_proj.bias".to_string(),
        ];
        for (prefix, count) in [("encoder", self.encoder.len()), ("decoder", self.decoder.len())] {
            for i in 0..count {
                for suffix in [
                    "norm1.weight", "norm1.bias", "norm2.weight", "norm2.bias",
                    "attn.in_proj_weight", "attn.in_proj_bias",
                    "attn.out_proj.weight", "attn.out_proj.bias",
                    "mlp_gate.weight", "mlp_gate.bias",
                    "mlp_up.weight",   "mlp_up.bias",
                    "mlp_down.weight", "mlp_down.bias",
                    "scale1", "scale2",
                ] {
                    keys.push(format!("{prefix}[{i}].{suffix}"));
                }
            }
        }
        keys
    }

    fn set_weight(&mut self, key: &str, t: &RawTensor) {
        match key {
            "z_embed"  => self.z_embed  = t.data.clone(),
            "pos_prev" => self.pos_prev = t.data.clone(),
            "pos_next" => self.pos_next = t.data.clone(),
            "pos_z"    => self.pos_z    = t.data.clone(),
            "encoder_norm.weight" => self.enc_norm_w = t.data.clone(),
            "encoder_norm.bias"   => self.enc_norm_b = t.data.clone(),
            "decoder_norm.weight" => self.dec_norm_w = t.data.clone(),
            "decoder_norm.bias"   => self.dec_norm_b = t.data.clone(),
            "out_proj.weight"     => self.out_proj_w = t.data.clone(),
            "out_proj.bias"       => self.out_proj_b = t.data.clone(),

            k if k.starts_with("encoder[") || k.starts_with("decoder[") => {
                if let Some((tag, idx, field)) = parse_block_key(k) {
                    let blocks = if tag == "encoder" { &mut self.encoder } else { &mut self.decoder };
                    if let Some(blk) = blocks.get_mut(idx) {
                        blk.set_weight(field, t);
                    }
                }
            }
            _ => {}
        }
    }

    /// Load a converted CDT checkpoint (see
    /// `scripts/export_unixcoder_reference.py convert-cdt`). The mapper
    /// should rewrite the raw `encoder.0.attn.in_proj_weight` style keys
    /// into the canonical `encoder[0].attn.in_proj_weight` form this head
    /// expects; see [`WeightMapper::code_deltatok`].
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
                self.set_weight(target, raw);
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

/// Parse `"encoder[3].attn.in_proj_weight"` → `("encoder", 3, "attn.in_proj_weight")`.
fn parse_block_key(key: &str) -> Option<(&str, usize, &str)> {
    let (tag, rest) = if let Some(r) = key.strip_prefix("encoder[") {
        ("encoder", r)
    } else if let Some(r) = key.strip_prefix("decoder[") {
        ("decoder", r)
    } else {
        return None;
    };
    let bracket = rest.find(']')?;
    let idx: usize = rest[..bracket].parse().ok()?;
    let field = rest[bracket + 1..].strip_prefix('.')?;
    Some((tag, idx, field))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_block_key_roundtrip() {
        assert_eq!(
            parse_block_key("encoder[7].attn.in_proj_weight"),
            Some(("encoder", 7, "attn.in_proj_weight")),
        );
        assert_eq!(
            parse_block_key("decoder[0].mlp_down.weight"),
            Some(("decoder", 0, "mlp_down.weight")),
        );
        assert_eq!(parse_block_key("z_embed"), None);
    }

    #[test]
    fn config_paper_default_matches_cdt_k1_4blk() {
        let c = CodeDeltaTokConfig::paper_default();
        assert_eq!(c.feature_dim, 768);
        assert_eq!(c.num_blocks, 4);
        assert_eq!(c.num_heads, 12);
        assert_eq!(c.num_delta_tokens, 1);
        assert_eq!(c.intermediate(), 3072);
    }
}
