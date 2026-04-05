//! CodeWorldModel (CWM) — text-token world model for code edits.
//!
//! Architecture: weight-shared looped encoder + tiny action MLP + 2-block
//! looped predictor. Unlike LEWM (which uses DiT-style adaLN), CWM uses
//! vanilla pre-norm transformer blocks with standard PyTorch MultiheadAttention.
//!
//! Pipeline (per forward):
//!   tokens[i64]  → Embedding[662,128] → CLS prepend → + PE[513,128]
//!                → LoopedTransformerBlock ×6 (weight-shared)
//!                → extract CLS[0] → LayerNorm → [128]
//!   action[f32;7] → Linear(7→128) → GELU → Linear(128→128) → [128]
//!   (z_state, z_action) → stack [2,128]
//!                       → Block_0 ×6 → Block_1 ×6
//!                       → extract token[0] → LayerNorm → [128]
//!
//! Zero-drift: load per-stage reference activations from
//! tests/fixtures/code_wm_reference_{g8,g1b}.safetensors and assert
//! cosine ≥ 0.99999, max_abs < 1e-5 at every intermediate.

use std::collections::HashMap;

use crate::ops::attention::bidirectional_attention;
use crate::ops::matmul::matmul_t;
use crate::ops::pure_rust_ops::layernorm_with_bias;
use crate::weight_loading::{AlignedBuffer, RawTensor, WeightError};

/// GELU activation variant: PyTorch's `nn.GELU()` defaults to exact erf (`approximate='none'`).
/// The tanh approximation is `nn.GELU(approximate='tanh')`. These differ by ~4e-4 max.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GeluKind {
    Erf,
    Tanh,
}

impl GeluKind {
    fn from_str(s: &str) -> Self {
        match s {
            "tanh" => GeluKind::Tanh,
            _ => GeluKind::Erf,
        }
    }
}

/// Configuration for a CodeWorldModel. Loaded from configs/code_wm_{g8,g1b}.json.
#[derive(Debug, Clone)]
pub struct CodeWorldModelConfig {
    pub vocab_size: usize,        // 662
    pub max_seq_len: usize,       // 512
    pub model_dim: usize,         // 128
    pub num_heads: usize,         // 4
    pub head_dim: usize,          // 32
    pub mlp_hidden: usize,        // 512
    pub encoder_loops: usize,     // 6
    pub predictor_depth: usize,   // 2
    pub predictor_loops: usize,   // 6
    pub action_dim: usize,        // 7
    pub layernorm_eps: f32,       // 1e-5
    pub gelu_kind: GeluKind,      // Erf (PyTorch default)
}

impl CodeWorldModelConfig {
    /// G8 / G1b default (they share the same architecture, different weights).
    pub fn g8() -> Self {
        Self {
            vocab_size: 662,
            max_seq_len: 512,
            model_dim: 128,
            num_heads: 4,
            head_dim: 32,
            mlp_hidden: 512,
            encoder_loops: 6,
            predictor_depth: 2,
            predictor_loops: 6,
            action_dim: 7,
            layernorm_eps: 1e-5,
            gelu_kind: GeluKind::Erf,
        }
    }

    pub fn from_json(path: &std::path::Path) -> Result<Self, String> {
        let data = std::fs::read_to_string(path)
            .map_err(|e| format!("Failed to read config: {e}"))?;
        Self::from_json_str(&data)
    }

    /// Parse config directly from a JSON string (WASM-friendly, no filesystem).
    pub fn from_json_str(data: &str) -> Result<Self, String> {
        let v: serde_json::Value = serde_json::from_str(data)
            .map_err(|e| format!("Failed to parse config JSON: {e}"))?;
        let model_dim = v["model_dim"].as_u64().unwrap_or(128) as usize;
        let num_heads = v["num_heads"].as_u64().unwrap_or(4) as usize;
        Ok(Self {
            vocab_size: v["vocab_size"].as_u64().unwrap_or(662) as usize,
            max_seq_len: v["max_seq_len"].as_u64().unwrap_or(512) as usize,
            model_dim,
            num_heads,
            head_dim: v["head_dim"].as_u64().map(|x| x as usize).unwrap_or(model_dim / num_heads),
            mlp_hidden: v["mlp_hidden"].as_u64().map(|x| x as usize).unwrap_or(model_dim * 4),
            encoder_loops: v["encoder_loops"].as_u64().unwrap_or(6) as usize,
            predictor_depth: v["predictor_depth"].as_u64().unwrap_or(2) as usize,
            predictor_loops: v["predictor_loops"].as_u64().unwrap_or(6) as usize,
            action_dim: v["action_dim"].as_u64().unwrap_or(7) as usize,
            layernorm_eps: v["layernorm_eps"].as_f64().unwrap_or(1e-5) as f32,
            gelu_kind: v["gelu_kind"].as_str().map(GeluKind::from_str).unwrap_or(GeluKind::Erf),
        })
    }
}

// ── GELU (exact erf variant) ────────────────────────────────────────
// Abramowitz & Stegun 7.1.26, max error ~1.5e-7 on the unit of precision.
// This matches PyTorch's erff within f32 precision.
#[inline]
fn erf_f32(x: f32) -> f32 {
    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let x = x.abs();
    let t = 1.0 / (1.0 + 0.3275911 * x);
    let y = 1.0
        - (((((1.061405429_f32 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592)
            * t
            * (-x * x).exp();
    sign * y
}

#[inline]
fn gelu_erf(x: f32) -> f32 {
    // y = 0.5 * x * (1 + erf(x / sqrt(2)))
    const INV_SQRT2: f32 = 0.70710678118654752440_f32;
    0.5 * x * (1.0 + erf_f32(x * INV_SQRT2))
}

#[inline]
fn gelu_tanh(x: f32) -> f32 {
    // y = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    const K0: f32 = 0.7978845608028654;
    0.5 * x * (1.0 + (K0 * (x + 0.044715 * x * x * x)).tanh())
}

fn apply_gelu(buf: &mut [f32], kind: GeluKind) {
    match kind {
        GeluKind::Erf => {
            for v in buf.iter_mut() {
                *v = gelu_erf(*v);
            }
        }
        GeluKind::Tanh => {
            for v in buf.iter_mut() {
                *v = gelu_tanh(*v);
            }
        }
    }
}

// ── Module weights ──────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct LinearWeights {
    pub weight: AlignedBuffer, // [out_dim, in_dim] row-major
    pub bias: AlignedBuffer,   // [out_dim]
    pub out_dim: usize,
    pub in_dim: usize,
}

impl LinearWeights {
    fn zeroed(out_dim: usize, in_dim: usize) -> Self {
        Self {
            weight: AlignedBuffer::new_zeroed(out_dim * in_dim),
            bias: AlignedBuffer::new_zeroed(out_dim),
            out_dim,
            in_dim,
        }
    }

    /// y = x @ W.T + b, where x: [m, in_dim], returns [m, out_dim].
    fn forward(&self, x: &[f32], m: usize) -> Vec<f32> {
        let mut out = matmul_t(x, &self.weight, m, self.in_dim, self.out_dim);
        if !self.bias.is_empty() {
            for r in 0..m {
                for j in 0..self.out_dim {
                    out[r * self.out_dim + j] += self.bias[j];
                }
            }
        }
        out
    }
}

#[derive(Debug, Clone)]
pub struct LayerNormWeights {
    pub weight: AlignedBuffer, // [dim]
    pub bias: AlignedBuffer,   // [dim]
    pub dim: usize,
    pub eps: f32,
}

impl LayerNormWeights {
    fn zeroed(dim: usize, eps: f32) -> Self {
        Self {
            weight: AlignedBuffer::new_zeroed(dim),
            bias: AlignedBuffer::new_zeroed(dim),
            dim,
            eps,
        }
    }

    fn forward(&self, x: &[f32]) -> Vec<f32> {
        layernorm_with_bias(x, &self.weight, &self.bias, self.eps, self.dim)
    }
}

// ── Transformer block (vanilla pre-norm) ────────────────────────────

#[derive(Debug, Clone)]
pub struct TransformerBlock {
    pub norm1: LayerNormWeights,
    pub attn_in_proj: LinearWeights,  // [3*D, D] fused QKV
    pub attn_out_proj: LinearWeights, // [D, D]
    pub norm2: LayerNormWeights,
    pub mlp_up: LinearWeights,   // [mlp_hidden, D]
    pub mlp_down: LinearWeights, // [D, mlp_hidden]
}

impl TransformerBlock {
    fn zeroed(cfg: &CodeWorldModelConfig) -> Self {
        let d = cfg.model_dim;
        Self {
            norm1: LayerNormWeights::zeroed(d, cfg.layernorm_eps),
            attn_in_proj: LinearWeights::zeroed(3 * d, d),
            attn_out_proj: LinearWeights::zeroed(d, d),
            norm2: LayerNormWeights::zeroed(d, cfg.layernorm_eps),
            mlp_up: LinearWeights::zeroed(cfg.mlp_hidden, d),
            mlp_down: LinearWeights::zeroed(d, cfg.mlp_hidden),
        }
    }

    /// Forward pass that also returns all intermediate activations (for drift debugging).
    #[cfg(feature = "debug_activations")]
    pub(crate) fn forward_debug(
        &self,
        x: &[f32],
        seq_len: usize,
        cfg: &CodeWorldModelConfig,
    ) -> (Vec<f32>, LoopTrace) {
        let d = cfg.model_dim;
        let normed1 = self.norm1.forward(x);
        let qkv = self.attn_in_proj.forward(&normed1, seq_len);
        let (q, k, v) = split_qkv(&qkv, seq_len, d);
        let attn_out = bidirectional_attention(&q, &k, &v, seq_len, cfg.num_heads, cfg.head_dim);
        let attn_proj = self.attn_out_proj.forward(&attn_out, seq_len);
        let mut res1 = vec![0.0_f32; seq_len * d];
        for i in 0..res1.len() {
            res1[i] = x[i] + attn_proj[i];
        }
        let normed2 = self.norm2.forward(&res1);
        let mut up = self.mlp_up.forward(&normed2, seq_len);
        apply_gelu(&mut up, cfg.gelu_kind);
        let down = self.mlp_down.forward(&up, seq_len);
        let mut res2 = vec![0.0_f32; seq_len * d];
        for i in 0..res2.len() {
            res2[i] = res1[i] + down[i];
        }
        let trace = LoopTrace {
            norm1: normed1,
            attn: attn_proj.clone(),
            res1: res1.clone(),
            norm2: normed2,
            mlp: down,
            res2: res2.clone(),
        };
        (res2, trace)
    }

    /// Forward pass: PreNorm → MHA → +res → PreNorm → MLP → +res.
    ///
    /// Input x: [seq_len, model_dim] row-major. Returns [seq_len, model_dim].
    fn forward(&self, x: &[f32], seq_len: usize, cfg: &CodeWorldModelConfig) -> Vec<f32> {
        let d = cfg.model_dim;
        debug_assert_eq!(x.len(), seq_len * d);

        // ── Attention branch ──
        let normed1 = self.norm1.forward(x); // [S, D]
        let qkv = self.attn_in_proj.forward(&normed1, seq_len); // [S, 3D]

        // Split fused QKV along last dim: q rows [0..D], k rows [D..2D], v rows [2D..3D].
        let (q, k, v) = split_qkv(&qkv, seq_len, d);

        // Scaled-dot-product attention across all heads.
        let attn_out = bidirectional_attention(&q, &k, &v, seq_len, cfg.num_heads, cfg.head_dim);
        let attn_proj = self.attn_out_proj.forward(&attn_out, seq_len); // [S, D]

        let mut after_res1 = vec![0.0_f32; seq_len * d];
        for i in 0..after_res1.len() {
            after_res1[i] = x[i] + attn_proj[i];
        }

        // ── MLP branch ──
        let normed2 = self.norm2.forward(&after_res1); // [S, D]
        let mut up = self.mlp_up.forward(&normed2, seq_len); // [S, mlp_hidden]
        apply_gelu(&mut up, cfg.gelu_kind);
        let down = self.mlp_down.forward(&up, seq_len); // [S, D]

        let mut out = vec![0.0_f32; seq_len * d];
        for i in 0..out.len() {
            out[i] = after_res1[i] + down[i];
        }
        out
    }
}

// ── CodeWorldModel ──────────────────────────────────────────────────

/// Stepwise activation trace for zero-drift debugging. Contains every
/// intermediate produced during `encode_debug`. Only compiled with the
/// `debug_activations` feature flag.
#[cfg(feature = "debug_activations")]
#[derive(Debug, Default)]
pub struct EncoderTrace {
    pub after_embed: Vec<f32>,
    pub after_cls_prepend: Vec<f32>,
    pub after_pe: Vec<f32>,
    pub loops: Vec<LoopTrace>,
    pub cls_extracted: Vec<f32>,
    pub encoder_final: Vec<f32>,
    pub seq_len_with_cls: usize,
    pub model_dim: usize,
}

#[cfg(feature = "debug_activations")]
#[derive(Debug, Default)]
pub struct LoopTrace {
    pub norm1: Vec<f32>,
    pub attn: Vec<f32>,
    pub res1: Vec<f32>,
    pub norm2: Vec<f32>,
    pub mlp: Vec<f32>,
    pub res2: Vec<f32>,
}

/// Stepwise predictor trace for zero-drift debugging.
#[cfg(feature = "debug_activations")]
#[derive(Debug, Default)]
pub struct PredictorTrace {
    pub stacked: Vec<f32>,
    // blocks[block_idx][loop_idx]
    pub blocks: Vec<Vec<LoopTrace>>,
    pub token0_extracted: Vec<f32>,
    pub pred_final: Vec<f32>,
    pub model_dim: usize,
}

pub struct CodeWorldModel {
    pub config: CodeWorldModelConfig,
    // Encoder
    pub token_embedding: AlignedBuffer, // [vocab_size, model_dim]
    pub cls_token: AlignedBuffer,       // [model_dim]
    pub pos_enc: AlignedBuffer,         // [max_seq_len+1, model_dim] baked from PyTorch buffer
    pub encoder_block: TransformerBlock,
    pub encoder_final_norm: LayerNormWeights,
    // Action encoder (Linear → GELU → Linear)
    pub action_fc1: LinearWeights, // [model_dim, action_dim]
    pub action_fc2: LinearWeights, // [model_dim, model_dim]
    // Predictor: 2 distinct blocks, each looped predictor_loops times
    pub predictor_blocks: Vec<TransformerBlock>,
    pub predictor_final_norm: LayerNormWeights,
}

/// Statistics from weight loading.
#[derive(Debug, Clone)]
pub struct LoadStats {
    pub loaded: usize,
    pub skipped: Vec<String>,
}

impl CodeWorldModel {
    pub fn from_config(cfg: &CodeWorldModelConfig) -> Self {
        let d = cfg.model_dim;
        let pe_rows = cfg.max_seq_len + 1;
        Self {
            token_embedding: AlignedBuffer::new_zeroed(cfg.vocab_size * d),
            cls_token: AlignedBuffer::new_zeroed(d),
            pos_enc: AlignedBuffer::new_zeroed(pe_rows * d),
            encoder_block: TransformerBlock::zeroed(cfg),
            encoder_final_norm: LayerNormWeights::zeroed(d, cfg.layernorm_eps),
            action_fc1: LinearWeights::zeroed(d, cfg.action_dim),
            action_fc2: LinearWeights::zeroed(d, d),
            predictor_blocks: (0..cfg.predictor_depth).map(|_| TransformerBlock::zeroed(cfg)).collect(),
            predictor_final_norm: LayerNormWeights::zeroed(d, cfg.layernorm_eps),
            config: cfg.clone(),
        }
    }

    /// Route safetensors keys to their buffers. Returns LoadStats.
    pub fn load_weights(
        &mut self,
        weights: HashMap<String, RawTensor>,
    ) -> Result<LoadStats, WeightError> {
        let mut loaded = 0usize;
        let mut skipped = Vec::new();
        for (name, tensor) in &weights {
            if self.set_weight(name, tensor) {
                loaded += 1;
            } else {
                skipped.push(name.clone());
            }
        }
        Ok(LoadStats { loaded, skipped })
    }

    fn set_weight(&mut self, key: &str, tensor: &RawTensor) -> bool {
        // Encoder buffers
        match key {
            "state_encoder.embedding.weight" => {
                self.token_embedding = tensor.data.clone();
                return true;
            }
            "state_encoder.cls_token" => {
                // Checkpoint shape is [1,1,128]; store flat.
                self.cls_token = tensor.data.clone();
                return true;
            }
            "state_encoder.pos_enc.pe" => {
                // Checkpoint shape is [1, 513, 128]; store flat [513*128].
                self.pos_enc = tensor.data.clone();
                return true;
            }
            "state_encoder.norm.weight" => {
                self.encoder_final_norm.weight = tensor.data.clone();
                return true;
            }
            "state_encoder.norm.bias" => {
                self.encoder_final_norm.bias = tensor.data.clone();
                return true;
            }
            "action_encoder.net.0.weight" => {
                self.action_fc1.weight = tensor.data.clone();
                return true;
            }
            "action_encoder.net.0.bias" => {
                self.action_fc1.bias = tensor.data.clone();
                return true;
            }
            "action_encoder.net.2.weight" => {
                self.action_fc2.weight = tensor.data.clone();
                return true;
            }
            "action_encoder.net.2.bias" => {
                self.action_fc2.bias = tensor.data.clone();
                return true;
            }
            "predictor.norm.weight" => {
                self.predictor_final_norm.weight = tensor.data.clone();
                return true;
            }
            "predictor.norm.bias" => {
                self.predictor_final_norm.bias = tensor.data.clone();
                return true;
            }
            _ => {}
        }

        if let Some(rest) = key.strip_prefix("state_encoder.block.") {
            return set_block_weight(&mut self.encoder_block, rest, tensor);
        }
        if let Some(rest) = key.strip_prefix("predictor.blocks.") {
            // "{idx}.{...}"
            let (idx_str, rest) = match rest.split_once('.') {
                Some(p) => p,
                None => return false,
            };
            let idx: usize = match idx_str.parse() {
                Ok(i) => i,
                Err(_) => return false,
            };
            if idx >= self.predictor_blocks.len() {
                return false;
            }
            return set_block_weight(&mut self.predictor_blocks[idx], rest, tensor);
        }

        false
    }

    /// Encode token IDs to a 128-d latent. Pads/truncates to max_seq_len internally.
    /// Returns `[model_dim]`.
    pub fn encode(&self, tokens: &[i64]) -> Vec<f32> {
        let d = self.config.model_dim;
        let s = tokens.len();
        let seq_with_cls = s + 1;

        // 1. Token embedding + CLS prepend: h[0] = cls_token, h[1..s+1] = embedding[tokens]
        let mut h = vec![0.0_f32; seq_with_cls * d];
        h[..d].copy_from_slice(&self.cls_token);
        for (i, &tok) in tokens.iter().enumerate() {
            let t = tok as usize;
            debug_assert!(t < self.config.vocab_size, "token id {t} out of vocab range");
            let src = &self.token_embedding[t * d..(t + 1) * d];
            h[(i + 1) * d..(i + 2) * d].copy_from_slice(src);
        }

        // 2. Add positional encoding (first seq_with_cls rows).
        for i in 0..seq_with_cls {
            let row_off = i * d;
            for j in 0..d {
                h[row_off + j] += self.pos_enc[row_off + j];
            }
        }

        // 3. 6 loops of the shared transformer block.
        for _ in 0..self.config.encoder_loops {
            h = self.encoder_block.forward(&h, seq_with_cls, &self.config);
        }

        // 4. Extract CLS token (position 0) and apply final LayerNorm.
        let cls_out: Vec<f32> = h[..d].to_vec();
        self.encoder_final_norm.forward(&cls_out)
    }

    /// Encode tokens via the fused Zig kernel (experimental scaffolding).
    ///
    /// **Performance**: currently SLOWER than `encode()` because the inner
    /// attention call uses a scalar implementation while `encode()` dispatches
    /// to the SIMD-tiled `syn_fused_attention_bidi`. The fused kernel's value
    /// is as infrastructure for future specializations (quantized weights,
    /// Metal GPU forward, etc.), not as a current-day speedup.
    ///
    /// Correctness: verified byte-for-byte vs sequential path (cos ≈ 1.0).
    #[cfg(feature = "zig-ffi")]
    pub fn encode_fused(&self, tokens: &[i64]) -> Vec<f32> {
        let d = self.config.model_dim;
        let s = tokens.len();
        let seq_with_cls = s + 1;
        let mlp_h = self.config.mlp_hidden;

        // 1. Build initial seq: CLS at [0..d], embedding rows, + PE.
        let mut seq = vec![0.0_f32; seq_with_cls * d];
        seq[..d].copy_from_slice(&self.cls_token);
        for (i, &tok) in tokens.iter().enumerate() {
            let t = tok as usize;
            debug_assert!(t < self.config.vocab_size);
            let src = &self.token_embedding[t * d..(t + 1) * d];
            seq[(i + 1) * d..(i + 2) * d].copy_from_slice(src);
        }
        for i in 0..seq_with_cls {
            let off = i * d;
            for j in 0..d { seq[off + j] += self.pos_enc[off + j]; }
        }

        // 2. Pre-allocate scratch buffers large enough for max(seq_len*d, seq_len*mlp_h).
        let big = seq_with_cls * std::cmp::max(d, mlp_h);
        let mut normed_buf = vec![0.0_f32; big];
        let mut qkv_buf = vec![0.0_f32; seq_with_cls * 3 * d];
        let mut attn_buf = vec![0.0_f32; seq_with_cls * d];
        let mut proj_buf = vec![0.0_f32; big];
        let mut scores_buf = vec![0.0_f32; seq_with_cls * seq_with_cls];
        // GEMM packing buffers (match Synapse's MC/KC/NC constants via matmul_ops)
        let max_n = std::cmp::max(3 * d, mlp_h);
        let max_k = std::cmp::max(d, mlp_h);
        // Rough upper bound (MR=NR=8, MC=64, KC=256, NC=256 in matmul.zig)
        let pa_size = ((64 + 7) / 8) * 8 * 256.min(max_k);
        let pb_size = ((256 + 7) / 8) * 8 * 256.min(max_k);
        let _ = max_n;
        let mut packed_a = vec![0.0_f32; pa_size];
        let mut packed_b = vec![0.0_f32; pb_size];

        // 3. Dispatch to Zig. On macOS we can set mode=0x08 (BLAS_ACCELERATE).
        #[cfg(target_os = "macos")]
        let mode: u32 = 0x08;
        #[cfg(not(target_os = "macos"))]
        let mode: u32 = 0;

        let block = &self.encoder_block;
        synapse_core::code_wm_encoder_fused(
            &mut seq, seq_with_cls, d, self.config.num_heads, mlp_h, self.config.encoder_loops,
            &block.norm1.weight, &block.norm1.bias,
            &block.attn_in_proj.weight, &block.attn_in_proj.bias,
            &block.attn_out_proj.weight, &block.attn_out_proj.bias,
            &block.norm2.weight, &block.norm2.bias,
            &block.mlp_up.weight, &block.mlp_up.bias,
            &block.mlp_down.weight, &block.mlp_down.bias,
            &mut normed_buf, &mut qkv_buf, &mut attn_buf,
            &mut proj_buf, &mut scores_buf,
            &mut packed_a, &mut packed_b,
            mode,
        ).expect("code_wm_encoder_fused failed");

        // 4. Extract CLS token and apply final LayerNorm.
        let cls_out: Vec<f32> = seq[..d].to_vec();
        self.encoder_final_norm.forward(&cls_out)
    }

    /// Encode action vector (length action_dim = 7) to 128-d latent.
    pub fn encode_action(&self, action: &[f32]) -> Vec<f32> {
        debug_assert_eq!(action.len(), self.config.action_dim);
        let mut h = self.action_fc1.forward(action, 1);
        apply_gelu(&mut h, self.config.gelu_kind);
        self.action_fc2.forward(&h, 1)
    }

    /// Encode tokens, returning every intermediate activation for drift debugging.
    #[cfg(feature = "debug_activations")]
    pub fn encode_debug(&self, tokens: &[i64]) -> EncoderTrace {
        let d = self.config.model_dim;
        let s = tokens.len();
        let seq_with_cls = s + 1;

        let mut h = vec![0.0_f32; seq_with_cls * d];
        // CLS position 0 is zero since we haven't prepended yet — first fill embed into rows 1..s+1
        let mut after_embed = vec![0.0_f32; s * d];
        for (i, &tok) in tokens.iter().enumerate() {
            let t = tok as usize;
            let src = &self.token_embedding[t * d..(t + 1) * d];
            after_embed[i * d..(i + 1) * d].copy_from_slice(src);
        }

        // CLS prepend: h[0] = cls_token, h[1..] = after_embed
        h[..d].copy_from_slice(&self.cls_token);
        h[d..].copy_from_slice(&after_embed);
        let after_cls_prepend = h.clone();

        // Add PE (first seq_with_cls rows).
        for i in 0..seq_with_cls {
            let row_off = i * d;
            for j in 0..d {
                h[row_off + j] += self.pos_enc[row_off + j];
            }
        }
        let after_pe = h.clone();

        let mut loops = Vec::with_capacity(self.config.encoder_loops);
        for _ in 0..self.config.encoder_loops {
            let (next, trace) = self.encoder_block.forward_debug(&h, seq_with_cls, &self.config);
            h = next;
            loops.push(trace);
        }

        let cls_extracted: Vec<f32> = h[..d].to_vec();
        let encoder_final = self.encoder_final_norm.forward(&cls_extracted);

        EncoderTrace {
            after_embed,
            after_cls_prepend,
            after_pe,
            loops,
            cls_extracted,
            encoder_final,
            seq_len_with_cls: seq_with_cls,
            model_dim: d,
        }
    }

    /// Predictor trace for drift debugging.
    #[cfg(feature = "debug_activations")]
    pub fn predict_debug(&self, z_state: &[f32], z_action: &[f32]) -> PredictorTrace {
        let d = self.config.model_dim;
        let mut x = vec![0.0_f32; 2 * d];
        x[..d].copy_from_slice(z_state);
        x[d..].copy_from_slice(z_action);
        let stacked = x.clone();

        let mut blocks: Vec<Vec<LoopTrace>> = Vec::with_capacity(self.predictor_blocks.len());
        for block in &self.predictor_blocks {
            let mut loops = Vec::with_capacity(self.config.predictor_loops);
            for _ in 0..self.config.predictor_loops {
                let (next, trace) = block.forward_debug(&x, 2, &self.config);
                x = next;
                loops.push(trace);
            }
            blocks.push(loops);
        }
        let token0_extracted: Vec<f32> = x[..d].to_vec();
        let pred_final = self.predictor_final_norm.forward(&token0_extracted);

        PredictorTrace {
            stacked,
            blocks,
            token0_extracted,
            pred_final,
            model_dim: d,
        }
    }

    /// Predict next latent from (z_state, z_action). Both must be length model_dim.
    pub fn predict(&self, z_state: &[f32], z_action: &[f32]) -> Vec<f32> {
        let d = self.config.model_dim;
        debug_assert_eq!(z_state.len(), d);
        debug_assert_eq!(z_action.len(), d);

        // Stack [z_state, z_action] into [2, D].
        let mut x = vec![0.0_f32; 2 * d];
        x[..d].copy_from_slice(z_state);
        x[d..].copy_from_slice(z_action);

        // depth distinct blocks × predictor_loops each.
        for block in &self.predictor_blocks {
            for _ in 0..self.config.predictor_loops {
                x = block.forward(&x, 2, &self.config);
            }
        }

        // Extract token 0 and apply final LayerNorm.
        let tok0: Vec<f32> = x[..d].to_vec();
        self.predictor_final_norm.forward(&tok0)
    }
}

/// Split the fused QKV output `[seq_len, 3*D]` into `(q, k, v)`, each `[seq_len, D]`.
/// PyTorch nn.MultiheadAttention stores in_proj_weight as [3D, D] with rows concat
/// as [Wq; Wk; Wv], so after `x @ W.T` the last dim is Q|K|V in that order.
fn split_qkv(qkv: &[f32], seq_len: usize, d: usize) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let mut q = vec![0.0_f32; seq_len * d];
    let mut k = vec![0.0_f32; seq_len * d];
    let mut v = vec![0.0_f32; seq_len * d];
    for t in 0..seq_len {
        let off_src = t * 3 * d;
        let off_dst = t * d;
        q[off_dst..off_dst + d].copy_from_slice(&qkv[off_src..off_src + d]);
        k[off_dst..off_dst + d].copy_from_slice(&qkv[off_src + d..off_src + 2 * d]);
        v[off_dst..off_dst + d].copy_from_slice(&qkv[off_src + 2 * d..off_src + 3 * d]);
    }
    (q, k, v)
}

fn set_block_weight(block: &mut TransformerBlock, rest: &str, tensor: &RawTensor) -> bool {
    match rest {
        "norm1.weight" => block.norm1.weight = tensor.data.clone(),
        "norm1.bias" => block.norm1.bias = tensor.data.clone(),
        "attn.in_proj_weight" => block.attn_in_proj.weight = tensor.data.clone(),
        "attn.in_proj_bias" => block.attn_in_proj.bias = tensor.data.clone(),
        "attn.out_proj.weight" => block.attn_out_proj.weight = tensor.data.clone(),
        "attn.out_proj.bias" => block.attn_out_proj.bias = tensor.data.clone(),
        "norm2.weight" => block.norm2.weight = tensor.data.clone(),
        "norm2.bias" => block.norm2.bias = tensor.data.clone(),
        "mlp.0.weight" => block.mlp_up.weight = tensor.data.clone(),
        "mlp.0.bias" => block.mlp_up.bias = tensor.data.clone(),
        "mlp.3.weight" => block.mlp_down.weight = tensor.data.clone(),
        "mlp.3.bias" => block.mlp_down.bias = tensor.data.clone(),
        _ => return false,
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn config_defaults() {
        let c = CodeWorldModelConfig::g8();
        assert_eq!(c.vocab_size, 662);
        assert_eq!(c.model_dim, 128);
        assert_eq!(c.num_heads, 4);
        assert_eq!(c.head_dim, 32);
        assert_eq!(c.mlp_hidden, 512);
        assert_eq!(c.encoder_loops, 6);
        assert_eq!(c.predictor_depth, 2);
        assert_eq!(c.predictor_loops, 6);
        assert_eq!(c.action_dim, 7);
        assert_eq!(c.gelu_kind, GeluKind::Erf);
    }

    #[test]
    fn gelu_erf_matches_pytorch_reference_values() {
        // Reference values from torch.nn.GELU(approximate='none')
        // on inputs [-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 3.0]
        let reference: [(f32, f32); 8] = [
            (-2.0, -0.04550010),
            (-1.0, -0.15865529),
            (-0.5, -0.15426880),
            (0.0, 0.0),
            (0.5, 0.3457312),
            (1.0, 0.8413447),
            (2.0, 1.9545000),
            (3.0, 2.9959502),
        ];
        for (x, expected) in reference {
            let got = gelu_erf(x);
            let diff = (got - expected).abs();
            assert!(
                diff < 1e-5,
                "gelu_erf({x}) = {got}, expected {expected}, diff = {diff:.3e}"
            );
        }
    }

    #[test]
    fn zero_model_has_expected_shapes() {
        let cfg = CodeWorldModelConfig::g8();
        let m = CodeWorldModel::from_config(&cfg);
        assert_eq!(m.token_embedding.len(), 662 * 128);
        assert_eq!(m.cls_token.len(), 128);
        assert_eq!(m.pos_enc.len(), 513 * 128);
        assert_eq!(m.encoder_block.attn_in_proj.weight.len(), 3 * 128 * 128);
        assert_eq!(m.encoder_block.mlp_up.weight.len(), 512 * 128);
        assert_eq!(m.encoder_block.mlp_down.weight.len(), 128 * 512);
        assert_eq!(m.predictor_blocks.len(), 2);
        assert_eq!(m.action_fc1.weight.len(), 128 * 7);
        assert_eq!(m.action_fc2.weight.len(), 128 * 128);
    }
}
