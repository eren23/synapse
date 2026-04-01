//! Model loading and inference for ESP32.
//!
//! Supports multiple model types:
//! - LEWM world model (encode/predict/rollout) — f32 or Q4 quantized
//! - Mamba Q4 language model (text generation)
//! - RWKV-7 Q4 language model (text generation)

use std::time::Instant;
use synapse_inference::models::ssm::mamba::block::MambaBlock;
use synapse_inference::models::ssm::mamba::config::MambaConfig;
use synapse_inference::models::ssm::mamba::model::MambaModel;
use synapse_inference::models::ssm::rwkv::block::RwkvBlock;
use synapse_inference::models::ssm::rwkv::config::RwkvConfig;
use synapse_inference::models::ssm::rwkv::model::RwkvModel;
use synapse_inference::models::traits::{Model, ModelState};
use synapse_inference::weight_loading::AlignedBuffer;
use synapse_inference::models::vision::lewm::{LeWMConfig, LeWorldModel};
use synapse_inference::quantization::{
    FullyQuantizedLeWM, Q4MambaModel, Q4RwkvModel, QuantizedQ4LeWM,
};

// ---------------------------------------------------------------------------
// Result and info types
// ---------------------------------------------------------------------------

/// Result from a text generation call.
#[derive(Debug, serde::Serialize)]
pub struct GenerateResult {
    pub tokens: Vec<u32>,
    pub latency_ms: f64,
    pub tokens_per_sec: f64,
}

/// Metadata about a loaded model.
#[derive(Debug, serde::Serialize)]
pub struct ModelInfo {
    pub name: String,
    pub model_type: String,
    pub num_layers: usize,
    pub quantization: String,
}

// ---------------------------------------------------------------------------
// Esp32LeWM  (unchanged from original)
// ---------------------------------------------------------------------------

/// Loaded LEWM model ready for inference on ESP32.
///
/// Internally uses either f32 (for testing) or Q4 quantized weights
/// (for deployment). Q4 models are loaded from LQ40 binary format.
pub struct Esp32LeWM {
    config: LeWMConfig,
    inner: Esp32LeWMInner,
}

enum Esp32LeWMInner {
    F32(LeWorldModel),
    Q4Pred(QuantizedQ4LeWM),
    Full(FullyQuantizedLeWM),
}

/// Timing information for a single inference call.
#[derive(Debug, serde::Serialize)]
pub struct InferenceMetrics {
    pub operation: String,
    pub latency_ms: f64,
    pub output_dim: usize,
}

impl Esp32LeWM {
    /// Create a model with default PushT config and zeroed weights (f32).
    /// For testing without real weights.
    pub fn new_zeroed() -> Self {
        let config = LeWMConfig::pusht();
        let model = LeWorldModel::from_config(&config);
        Esp32LeWM {
            config,
            inner: Esp32LeWMInner::F32(model),
        }
    }

    /// Create a model with seeded predictor weights for benchmarking.
    /// Uses pushT config with deterministically seeded f32 weights.
    pub fn new_benchmark() -> Self {
        let config = LeWMConfig::pusht();
        let mut model = LeWorldModel::from_config(&config);
        let hidden = config.predictor_hidden;
        let inner = config.predictor_inner_dim;
        let inter = config.predictor_inter;

        // Seed predictor weights deterministically (non-zero so matmul asserts pass).
        let seed_w = |len: usize, s: u32| -> AlignedBuffer {
            AlignedBuffer::from_slice(
                &(0..len)
                    .map(|i| {
                        let x = (((i as u32).wrapping_mul(2_654_435_761).wrapping_add(s)) as f32);
                        x / u32::MAX as f32 * 0.36 - 0.18
                    })
                    .collect::<Vec<f32>>(),
            )
        };
        let seed_bias = |len: usize, s: u32| AlignedBuffer::from_slice(
            &(0..len)
                .map(|i| {
                    let x = (((i as u32).wrapping_mul(3_333_333).wrapping_add(s)) as f32);
                    x / u32::MAX as f32 * 0.1
                })
                .collect::<Vec<f32>>(),
        );

        // Action encoder weights (mirrors build_test_lewm_pushT seeds 600-605)
        model.action_conv_weight = seed_w(config.action_dim * config.action_dim, 600);
        model.action_conv_bias = seed_bias(config.action_dim, 601);
        model.action_mlp1_weight = seed_w(config.encoder_inter * config.action_dim, 602);
        model.action_mlp1_bias = seed_bias(config.encoder_inter, 603);
        model.action_mlp2_weight = seed_w(hidden * config.encoder_inter, 604);
        model.action_mlp2_bias = seed_bias(hidden, 605);

        for (i, layer) in model.predictor_layers.iter_mut().enumerate() {
            let s = (i as u32 + 1) * 1000;
            layer.adaln_weight = seed_w(6 * hidden * hidden, s + 1);
            layer.adaln_bias = seed_bias(6 * hidden, s + 2);
            layer.to_qkv = seed_w(3 * inner * hidden, s + 3);
            layer.attn_out_weight = seed_w(hidden * inner, s + 4);
            layer.attn_out_bias = seed_bias(hidden, s + 5);
            layer.attn_norm_weight = AlignedBuffer::from_slice(&vec![1.0; hidden]);
            layer.attn_norm_bias = AlignedBuffer::from_slice(&vec![0.0; hidden]);
            layer.mlp_norm_weight = AlignedBuffer::from_slice(&vec![1.0; hidden]);
            layer.mlp_norm_bias = AlignedBuffer::from_slice(&vec![0.0; hidden]);
            layer.mlp_up_weight = seed_w(inter * hidden, s + 10);
            layer.mlp_up_bias = seed_bias(inter, s + 11);
            layer.mlp_down_weight = seed_w(hidden * inter, s + 12);
            layer.mlp_down_bias = seed_bias(hidden, s + 13);
        }
        model.predictor_norm_weight = AlignedBuffer::from_slice(&vec![1.0; hidden]);
        model.predictor_norm_bias = AlignedBuffer::from_slice(&vec![0.0; hidden]);
        Esp32LeWM {
            config,
            inner: Esp32LeWMInner::F32(model),
        }
    }

    /// Create a slim model with zeroed weights (f32).
    /// For testing the slim architecture path without real weights.
    pub fn new_slim_zeroed() -> Self {
        let config = LeWMConfig::slim();
        let model = LeWorldModel::from_config(&config);
        Esp32LeWM {
            config,
            inner: Esp32LeWMInner::F32(model),
        }
    }

    /// Load model from LQ40 binary data (Q4 quantized).
    ///
    /// Format: `[4B "LQ40"][4B config_len][JSON config][weight data]`
    ///
    /// Automatically detects slim models from config. Keeps weights in Q4
    /// format (~12-17MB) to fit in ESP32-P4's 32MB PSRAM.
    pub fn from_binary(data: &[u8]) -> Result<Self, String> {
        let mode = lq40_mode(data)?;
        match mode.as_str() {
            "full" => {
                let model = FullyQuantizedLeWM::from_lq40_bytes(data)?;
                let config = model.config.clone();
                Ok(Esp32LeWM {
                    config,
                    inner: Esp32LeWMInner::Full(model),
                })
            }
            "q4-pred" | "wanda20-q4" | "wanda40-q4" => {
                let model = QuantizedQ4LeWM::from_lq40_bytes(data)?;
                let config = model.config.clone();
                Ok(Esp32LeWM {
                    config,
                    inner: Esp32LeWMInner::Q4Pred(model),
                })
            }
            _ => Err(format!("Unsupported LQ40 mode '{mode}'")),
        }
    }

    /// Encode an image to a latent state.
    /// image: flat [H*W*3] f32 pixel data, normalized to [0,1].
    pub fn encode(
        &self,
        image: &[f32],
        height: usize,
        width: usize,
    ) -> (Vec<f32>, InferenceMetrics) {
        let start = Instant::now();
        let latent = match &self.inner {
            Esp32LeWMInner::F32(m) => m.encode(image, height, width),
            Esp32LeWMInner::Q4Pred(m) => m.encode(image, height, width),
            Esp32LeWMInner::Full(m) => m.encode(image, height, width),
        };
        let ms = start.elapsed().as_secs_f64() * 1000.0;
        let metrics = InferenceMetrics {
            operation: "encode".into(),
            latency_ms: ms,
            output_dim: latent.len(),
        };
        (latent, metrics)
    }

    /// Predict next latent state given current state and action.
    pub fn predict_next(&self, state: &[f32], action: &[f32]) -> (Vec<f32>, InferenceMetrics) {
        let start = Instant::now();
        let next = match &self.inner {
            Esp32LeWMInner::F32(m) => m.predict_next(state, action),
            Esp32LeWMInner::Q4Pred(m) => m.predict_next(state, action),
            Esp32LeWMInner::Full(m) => m.predict_next(state, action),
        };
        let ms = start.elapsed().as_secs_f64() * 1000.0;
        let metrics = InferenceMetrics {
            operation: "predict".into(),
            latency_ms: ms,
            output_dim: next.len(),
        };
        (next, metrics)
    }

    /// Multi-step rollout.
    pub fn rollout(
        &self,
        state: &[f32],
        actions: &[Vec<f32>],
    ) -> (Vec<Vec<f32>>, InferenceMetrics) {
        let start = Instant::now();
        let trajectory = match &self.inner {
            Esp32LeWMInner::F32(m) => m.rollout(state, actions),
            Esp32LeWMInner::Q4Pred(m) => m.rollout(state, actions),
            Esp32LeWMInner::Full(m) => m.rollout(state, actions),
        };
        let ms = start.elapsed().as_secs_f64() * 1000.0;
        let steps = trajectory.len();
        let metrics = InferenceMetrics {
            operation: format!("rollout_{steps}_steps"),
            latency_ms: ms,
            output_dim: self.config.latent_dim,
        };
        (trajectory, metrics)
    }

    /// Fused multi-step rollout: encodes all actions, builds one N×3-token sequence,
    /// runs predictor layers once. Same z_start for all positions — parallel futures.
    ///
    /// Currently only implemented for the F32 variant. Q4Pred/Full fall back
    /// to sequential rollout (slower but correct).
    pub fn rollout_fused(
        &self,
        state: &[f32],
        actions: &[Vec<f32>],
    ) -> (Vec<Vec<f32>>, InferenceMetrics) {
        let start = Instant::now();
        let trajectory = match &self.inner {
            Esp32LeWMInner::F32(m) => {
                // F32: use the truly-fused path
                let mut bufs =
                    synapse_inference::models::vision::lewm::LeWMBuffers::new(&m.config);
                m.predict_rollout_fused(state, actions, &mut bufs)
            }
            Esp32LeWMInner::Q4Pred(m) => {
                // Q4Pred: fall back to sequential rollout (predict_rollout_fused not yet impl)
                m.rollout(state, actions)
            }
            Esp32LeWMInner::Full(m) => {
                // Full: fall back to sequential rollout
                m.rollout(state, actions)
            }
        };
        let ms = start.elapsed().as_secs_f64() * 1000.0;
        let steps = trajectory.len();
        let metrics = InferenceMetrics {
            operation: format!("rollout_fused_{steps}_steps"),
            latency_ms: ms,
            output_dim: self.config.latent_dim,
        };
        (trajectory, metrics)
    }

    pub fn config(&self) -> &LeWMConfig {
        &self.config
    }
    pub fn latent_dim(&self) -> usize {
        self.config.latent_dim
    }
    pub fn action_dim(&self) -> usize {
        self.config.action_dim
    }

    /// Whether this model is Q4-quantized (vs f32 testing mode).
    pub fn is_quantized(&self) -> bool {
        !matches!(self.inner, Esp32LeWMInner::F32(_))
    }

    pub fn model_info(&self) -> ModelInfo {
        let quant = match &self.inner {
            Esp32LeWMInner::F32(_) => "f32",
            Esp32LeWMInner::Q4Pred(_) => "Q4_0",
            Esp32LeWMInner::Full(_) => "INT8+Q4",
        };
        let slim = if self.config.has_projection() {
            " slim"
        } else {
            ""
        };
        ModelInfo {
            name: format!(
                "LeWorldModel{} ({}d/{}e/{}p)",
                slim,
                self.config.latent_dim,
                self.config.encoder_layers,
                self.config.predictor_layers
            ),
            model_type: "lewm".into(),
            num_layers: self.config.predictor_layers,
            quantization: quant.into(),
        }
    }
}

fn lq40_mode(data: &[u8]) -> Result<String, String> {
    if data.len() < 8 {
        return Err("LQ40 data too short".into());
    }
    if &data[0..4] != b"LQ40" {
        return Err("Not LQ40 format".into());
    }

    let config_len = u32::from_le_bytes([data[4], data[5], data[6], data[7]]) as usize;
    if data.len() < 8 + config_len {
        return Err("LQ40 config truncated".into());
    }

    let config_json: serde_json::Value = serde_json::from_slice(&data[8..8 + config_len])
        .map_err(|e| format!("LQ40 config parse error: {e}"))?;
    config_json
        .get("mode")
        .and_then(|v| v.as_str())
        .map(str::to_owned)
        .ok_or_else(|| "LQ40 config missing mode".into())
}

// ---------------------------------------------------------------------------
// Esp32Mamba  — Q4-quantized Mamba for text generation
// ---------------------------------------------------------------------------

/// Q4-quantized Mamba model for ESP32 deployment.
pub struct Esp32Mamba {
    config: MambaConfig,
    model: Q4MambaModel,
}

impl Esp32Mamba {
    /// Create a tiny Mamba model with zeroed weights for testing.
    pub fn new_zeroed() -> Self {
        let config = MambaConfig {
            d_model: 64,
            d_state: 4,
            d_conv: 4,
            expand: 2,
            dt_rank: 4,
            num_layers: 2,
            vocab_size: 128,
            norm_eps: 1e-5,
        };
        let d = config.d_model;
        let di = config.d_inner();
        let ds = config.d_state;
        let dc = config.d_conv;
        let dr = config.dt_rank;
        let v = config.vocab_size;

        let blocks: Vec<MambaBlock> = (0..config.num_layers)
            .map(|_| MambaBlock {
                d_model: d,
                d_inner: di,
                d_state: ds,
                d_conv: dc,
                dt_rank: dr,
                norm_weight: vec![1.0; d],
                norm_eps: 1e-5,
                in_proj_weight: vec![0.0; 2 * di * d],
                in_proj_bias: vec![],
                conv1d_weight: vec![0.0; di * dc],
                conv1d_bias: vec![0.0; di],
                x_proj_weight: vec![0.0; (dr + 2 * ds) * di],
                dt_proj_weight: vec![0.0; di * dr],
                dt_proj_bias: vec![0.0; di],
                a_log: vec![-1.0; di * ds],
                d_param: vec![1.0; di],
                out_proj_weight: vec![0.0; d * di],
                out_proj_bias: vec![],
            })
            .collect();

        let f32_model = MambaModel::new(
            config.clone(),
            vec![0.0; v * d],
            blocks,
            vec![1.0; d],
            vec![0.0; v * d],
        );
        let model = Q4MambaModel::from_f32(&f32_model);

        Esp32Mamba { config, model }
    }

    /// Generate tokens using greedy argmax decoding.
    pub fn generate(
        &self,
        prompt_tokens: &[u32],
        max_tokens: usize,
        _temperature: f32,
    ) -> GenerateResult {
        let start = Instant::now();
        let mut state = ModelState::Recurrent;

        // Prefill
        let mut output = self.model.forward_prefill(prompt_tokens, &mut state);
        let mut tokens = Vec::with_capacity(max_tokens);

        // Decode
        for _ in 0..max_tokens {
            let next = argmax(&output.logits);
            tokens.push(next);
            output = self.model.forward_one(next, &mut state);
        }

        let elapsed = start.elapsed().as_secs_f64() * 1000.0;
        let tok_per_sec = if elapsed > 0.0 {
            tokens.len() as f64 / (elapsed / 1000.0)
        } else {
            0.0
        };

        GenerateResult {
            tokens,
            latency_ms: elapsed,
            tokens_per_sec: tok_per_sec,
        }
    }

    pub fn model_info(&self) -> ModelInfo {
        ModelInfo {
            name: format!(
                "Mamba (d={}, L={})",
                self.config.d_model, self.config.num_layers
            ),
            model_type: "mamba".into(),
            num_layers: self.config.num_layers,
            quantization: "Q4_0".into(),
        }
    }
}

// ---------------------------------------------------------------------------
// Esp32Rwkv  — Q4-quantized RWKV-7 for text generation
// ---------------------------------------------------------------------------

/// Q4-quantized RWKV-7 model for ESP32 deployment.
pub struct Esp32Rwkv {
    config: RwkvConfig,
    model: Q4RwkvModel,
}

impl Esp32Rwkv {
    /// Create a tiny RWKV model with zeroed weights for testing.
    pub fn new_zeroed() -> Self {
        let config = RwkvConfig {
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
        };
        let h = config.hidden_size;
        let nh = config.num_heads;
        let hs = config.head_size;
        let inter = config.intermediate_size;
        let dr = config.decay_rank;
        let ar = config.alpha_rank;
        let gr = config.gate_rank;
        let vocab = config.vocab_size;

        let blocks: Vec<RwkvBlock> = (0..config.num_layers)
            .map(|_| RwkvBlock {
                hidden_size: h,
                num_heads: nh,
                head_size: hs,
                intermediate_size: inter,
                decay_rank: dr,
                alpha_rank: ar,
                gate_rank: gr,
                norm_eps: config.norm_eps as f32,
                ln1_weight: vec![1.0; h],
                ln1_bias: vec![0.0; h],
                x_r: vec![0.0; h],
                x_k: vec![0.0; h],
                x_v: vec![0.0; h],
                x_w: vec![0.0; h],
                x_a: vec![0.0; h],
                x_g: vec![0.0; h],
                r_proj: vec![0.0; h * h],
                k_proj: vec![0.0; h * h],
                v_proj: vec![0.0; h * h],
                o_proj: vec![0.0; h * h],
                w0: vec![0.0; h],
                w1: vec![0.0; h * dr],
                w2: vec![0.0; dr * h],
                a0: vec![0.0; h],
                a1: vec![0.0; h * ar],
                a2: vec![0.0; ar * h],
                g1: vec![0.0; h * gr],
                g2: vec![0.0; gr * h],
                k_k: vec![1.0; h],
                k_a: vec![1.0; h],
                r_k: vec![0.0; nh * hs],
                g_norm_weight: vec![1.0; h],
                g_norm_bias: vec![0.0; h],
                ln2_weight: vec![1.0; h],
                ln2_bias: vec![0.0; h],
                ffn_x_k: vec![0.0; h],
                v_rank: 0,
                v0: vec![],
                v1: vec![],
                v2: vec![],
                ffn_key_weight: vec![0.0; inter * h],
                ffn_value_weight: vec![0.0; h * inter],
            })
            .collect();

        let f32_model = RwkvModel::new(
            config.clone(),
            vec![0.0; vocab * h],
            blocks,
            vec![1.0; h],
            vec![0.0; h],
            vec![0.0; vocab * h],
        );
        let model = Q4RwkvModel::from_f32(&f32_model);

        Esp32Rwkv { config, model }
    }

    /// Generate tokens using greedy argmax decoding.
    pub fn generate(
        &self,
        prompt_tokens: &[u32],
        max_tokens: usize,
        _temperature: f32,
    ) -> GenerateResult {
        let start = Instant::now();
        let mut state = ModelState::Recurrent;

        // Prefill
        let mut output = self.model.forward_prefill(prompt_tokens, &mut state);
        let mut tokens = Vec::with_capacity(max_tokens);

        // Decode
        for _ in 0..max_tokens {
            let next = argmax(&output.logits);
            tokens.push(next);
            output = self.model.forward_one(next, &mut state);
        }

        let elapsed = start.elapsed().as_secs_f64() * 1000.0;
        let tok_per_sec = if elapsed > 0.0 {
            tokens.len() as f64 / (elapsed / 1000.0)
        } else {
            0.0
        };

        GenerateResult {
            tokens,
            latency_ms: elapsed,
            tokens_per_sec: tok_per_sec,
        }
    }

    pub fn model_info(&self) -> ModelInfo {
        ModelInfo {
            name: format!(
                "RWKV-7 (h={}, L={})",
                self.config.hidden_size, self.config.num_layers
            ),
            model_type: "rwkv".into(),
            num_layers: self.config.num_layers,
            quantization: "Q4_0".into(),
        }
    }
}

// ---------------------------------------------------------------------------
// Esp32Model  — enum wrapping all supported models
// ---------------------------------------------------------------------------

/// Supported models for ESP32 deployment.
pub enum Esp32Model {
    /// LEWM world model: encode/predict/rollout.
    LeWM(Esp32LeWM),
    /// Mamba text generation (Q4 quantized).
    Mamba(Esp32Mamba),
    /// RWKV-7 text generation (Q4 quantized).
    Rwkv(Esp32Rwkv),
}

impl Esp32Model {
    /// Return metadata about the loaded model.
    pub fn model_info(&self) -> ModelInfo {
        match self {
            Esp32Model::LeWM(m) => m.model_info(),
            Esp32Model::Mamba(m) => m.model_info(),
            Esp32Model::Rwkv(m) => m.model_info(),
        }
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Greedy argmax over a logit vector, returning the token id.
fn argmax(logits: &[f32]) -> u32 {
    let mut best_idx = 0u32;
    let mut best_val = f32::NEG_INFINITY;
    for (i, &v) in logits.iter().enumerate() {
        if v > best_val {
            best_val = v;
            best_idx = i as u32;
        }
    }
    best_idx
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -- Existing LeWM tests (preserved) ------------------------------------

    #[test]
    fn esp32_model_creates_with_zeroed_weights() {
        let model = Esp32LeWM::new_zeroed();
        assert_eq!(model.latent_dim(), 192);
        assert_eq!(model.action_dim(), 10);
    }

    #[test]
    fn esp32_model_config_matches_pusht() {
        let model = Esp32LeWM::new_zeroed();
        let cfg = model.config();
        assert_eq!(cfg.image_size, 224);
        assert_eq!(cfg.patch_size, 14);
        assert_eq!(cfg.predictor_layers, 6);
        assert_eq!(cfg.predictor_heads, 16);
        assert!(!model.is_quantized());
        assert_eq!(model.model_info().quantization, "f32");
    }

    #[test]
    fn esp32_model_binary_loading_rejects_invalid() {
        let result = Esp32LeWM::from_binary(&[0u8; 64]);
        assert!(result.is_err());
    }

    #[test]
    fn esp32_model_binary_loading_rejects_short() {
        let result = Esp32LeWM::from_binary(b"LQ40");
        assert!(result.is_err());
    }

    #[test]
    fn esp32_model_binary_loading_rejects_unknown_mode() {
        let config = br#"{"mode":"mystery","image_size":224,"patch_size":14,"encoder_hidden":192,"encoder_layers":4,"encoder_heads":3,"encoder_inter":768,"predictor_hidden":192,"predictor_layers":4,"predictor_heads":16,"predictor_inner_dim":1024,"predictor_inter":2048,"action_dim":10,"latent_dim":96,"channels":3}"#;
        let mut blob = b"LQ40".to_vec();
        blob.extend_from_slice(&(config.len() as u32).to_le_bytes());
        blob.extend_from_slice(config);
        let result = Esp32LeWM::from_binary(&blob);
        assert!(matches!(result, Err(e) if e.contains("Unsupported LQ40 mode")));
    }

    #[test]
    fn esp32_slim_model_creates_zeroed() {
        let model = Esp32LeWM::new_slim_zeroed();
        assert_eq!(model.latent_dim(), 96);
        assert_eq!(model.action_dim(), 10);
        assert!(!model.is_quantized());
        let info = model.model_info();
        assert!(info.name.contains("slim"));
        assert_eq!(info.quantization, "f32");
    }

    #[test]
    fn esp32_slim_model_config_correct() {
        let model = Esp32LeWM::new_slim_zeroed();
        let cfg = model.config();
        // Slim: 96d latent, 4 encoder layers, 4 predictor layers
        assert_eq!(cfg.latent_dim, 96);
        assert_eq!(cfg.predictor_hidden, 192);
        assert_eq!(cfg.encoder_layers, 4);
        assert_eq!(cfg.predictor_layers, 4);
        // Slim needs projection weights to run predict —
        // predict_next requires loaded weights, not tested with zeroed model.
    }

    #[test]
    fn esp32_model_loads_local_slim_q4_fixture() {
        let fixture = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../web/lewm-compress-demo/lewm-slim-96d-q4.bin");
        let data = std::fs::read(&fixture).expect("q4 fixture should exist");
        let model = Esp32LeWM::from_binary(&data).expect("q4 fixture should load");
        assert_eq!(model.latent_dim(), 96);
        assert!(model.is_quantized());
        assert_eq!(model.model_info().quantization, "Q4_0");
        let state = vec![0.0; model.latent_dim()];
        let action = vec![0.0; model.action_dim()];
        let (next, metrics) = model.predict_next(&state, &action);
        assert_eq!(next.len(), model.latent_dim());
        assert!(next.iter().all(|v| v.is_finite()));
        assert_eq!(metrics.output_dim, model.latent_dim());
    }

    #[test]
    fn esp32_model_loads_local_slim_full_fixture() {
        let fixture = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../web/lewm-compress-demo/lewm-slim-96d-full.bin");
        let data = std::fs::read(&fixture).expect("full fixture should exist");
        let model = Esp32LeWM::from_binary(&data).expect("full fixture should load");
        assert_eq!(model.latent_dim(), 96);
        assert!(model.is_quantized());
        assert_eq!(model.model_info().quantization, "INT8+Q4");
        let state = vec![0.0; model.latent_dim()];
        let action = vec![0.0; model.action_dim()];
        let (next, metrics) = model.predict_next(&state, &action);
        assert_eq!(next.len(), model.latent_dim());
        assert!(next.iter().all(|v| v.is_finite()));
        assert_eq!(metrics.output_dim, model.latent_dim());
    }

    #[test]
    fn esp32_inference_metrics_serializes() {
        let m = InferenceMetrics {
            operation: "predict".into(),
            latency_ms: 42.5,
            output_dim: 192,
        };
        let json = serde_json::to_string(&m).unwrap();
        assert!(json.contains("predict"));
        assert!(json.contains("42.5"));
    }

    // -- Mamba Q4 tests -----------------------------------------------------

    #[test]
    fn esp32_mamba_creates_zeroed() {
        let mamba = Esp32Mamba::new_zeroed();
        let info = mamba.model_info();
        assert_eq!(info.model_type, "mamba");
        assert_eq!(info.num_layers, 2);
        assert_eq!(info.quantization, "Q4_0");
    }

    #[test]
    fn esp32_mamba_generate_returns_tokens() {
        let mamba = Esp32Mamba::new_zeroed();
        let result = mamba.generate(&[1, 2, 3], 5, 1.0);
        assert_eq!(result.tokens.len(), 5);
        assert!(result.latency_ms >= 0.0);
        assert!(result.tokens_per_sec >= 0.0);
    }

    #[test]
    fn esp32_mamba_generate_result_serializes() {
        let result = GenerateResult {
            tokens: vec![10, 20, 30],
            latency_ms: 12.5,
            tokens_per_sec: 240.0,
        };
        let json = serde_json::to_string(&result).unwrap();
        assert!(json.contains("tokens"));
        assert!(json.contains("12.5"));
    }

    // -- RWKV Q4 tests ------------------------------------------------------

    #[test]
    fn esp32_rwkv_creates_zeroed() {
        let rwkv = Esp32Rwkv::new_zeroed();
        let info = rwkv.model_info();
        assert_eq!(info.model_type, "rwkv");
        assert_eq!(info.num_layers, 2);
        assert_eq!(info.quantization, "Q4_0");
    }

    #[test]
    fn esp32_rwkv_generate_returns_tokens() {
        let rwkv = Esp32Rwkv::new_zeroed();
        let result = rwkv.generate(&[1, 2, 3], 5, 1.0);
        assert_eq!(result.tokens.len(), 5);
        assert!(result.latency_ms >= 0.0);
        assert!(result.tokens_per_sec >= 0.0);
    }

    // -- Esp32Model enum tests ----------------------------------------------

    #[test]
    fn esp32_model_enum_mamba_info() {
        let model = Esp32Model::Mamba(Esp32Mamba::new_zeroed());
        let info = model.model_info();
        assert_eq!(info.model_type, "mamba");
    }

    #[test]
    fn esp32_model_enum_rwkv_info() {
        let model = Esp32Model::Rwkv(Esp32Rwkv::new_zeroed());
        let info = model.model_info();
        assert_eq!(info.model_type, "rwkv");
    }

    #[test]
    fn esp32_model_enum_lewm_info() {
        let model = Esp32Model::LeWM(Esp32LeWM::new_zeroed());
        let info = model.model_info();
        assert_eq!(info.model_type, "lewm");
    }

    #[test]
    fn model_info_serializes() {
        let info = ModelInfo {
            name: "Test Model".into(),
            model_type: "mamba".into(),
            num_layers: 4,
            quantization: "Q4_0".into(),
        };
        let json = serde_json::to_string(&info).unwrap();
        assert!(json.contains("Test Model"));
        assert!(json.contains("mamba"));
    }
}
