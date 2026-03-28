use std::path::{Path, PathBuf};

use crate::capabilities::CapabilityReport;
use crate::chat_template::{ChatMessage, ChatTemplate};
use crate::config::ModelConfig;
use crate::generation::{GenerationConfig, GenerationOutput, GenerationPipeline};
use crate::kv_cache::KVCache;
use crate::model::traits::Model;
use crate::model::{CausalLM, ModelBuilder, ModelOutput, ModelState};
use crate::model_adapter::{
    adapter_for_kind, ModelAdapter, ModelAdapterKind, ReasoningMarkers, ThinkingMode,
};
use crate::quantization::{quantize_model, quantize_model_ternary, QuantizedCausalLM, TernaryCausalLM};
use crate::ssm::config::MambaConfig;
use crate::ssm::mamba_model::MambaModel;
use crate::tokenizer::{Tokenizer, TokenizerError};
use crate::weight_loading::{
    load_gguf, load_safetensors, load_safetensors_sharded, WeightError, WeightMapper,
};

#[cfg(feature = "metal")]
use std::cell::RefCell;

/// Which compute backend to use for dispatch.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BackendSelection {
    /// Always use CPU SIMD (Zig FFI).
    CpuSimd,
    /// Use Metal GPU when available, CPU SIMD fallback.
    #[cfg(feature = "metal")]
    Auto,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RuntimeBackend {
    CpuSimd,
    #[cfg(feature = "metal")]
    Metal,
}

impl RuntimeBackend {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::CpuSimd => "cpu_simd",
            #[cfg(feature = "metal")]
            Self::Metal => "metal",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RuntimePath {
    CpuCached,
    QuantizedCpu,
    BackendDispatch,
    MetalPerLayer,
}

impl RuntimePath {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::CpuCached => "cpu_cached",
            Self::QuantizedCpu => "quantized_cpu",
            Self::BackendDispatch => "backend_dispatch",
            Self::MetalPerLayer => "metal_per_layer",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RuntimePlan {
    pub family: &'static str,
    pub backend: RuntimeBackend,
    pub quantized: bool,
    pub prefill_path: RuntimePath,
    pub decode_path: RuntimePath,
    pub prefill_strategy: &'static str,
    pub decode_strategy: &'static str,
}

impl RuntimePlan {
    pub fn log_line(&self) -> String {
        format!(
            "Runtime: family={} backend={} quantized={} prefill={} prefill_strategy={} decode={} decode_strategy={}",
            self.family,
            self.backend.as_str(),
            self.quantized,
            self.prefill_path.as_str(),
            self.prefill_strategy,
            self.decode_path.as_str(),
            self.decode_strategy,
        )
    }
}

pub type RuntimeSummary = RuntimePlan;

/// High-level inference orchestrator.
///
/// Assembles config → model → weights → KV-cache → tokenizer into a single
/// entry point for text generation.
pub struct InferenceEngine {
    pub model: CausalLM,
    pub quantized_model: Option<QuantizedCausalLM>,
    pub ternary_model: Option<TernaryCausalLM>,
    /// Optional SSM model (Mamba, RWKV, etc.) — takes priority over transformer models.
    pub ssm_model: Option<Box<dyn Model>>,
    pub config: ModelConfig,
    pub model_adapter_kind: ModelAdapterKind,
    pub tokenizer: Option<Tokenizer>,
    pub chat_template: Option<ChatTemplate>,
    #[cfg(feature = "metal")]
    pub backend: crate::metal::ComputeBackend,
    /// GPU-resident model buffers for Phase 3 all-layers-in-one-command-buffer.
    /// Wrapped in RefCell because generate_text borrows &self but the decode loop
    /// needs mutable access to update the GPU KV cache position.
    /// Allocated when Metal backend is active; `None` for CPU-only or quantized.
    #[cfg(feature = "metal")]
    pub metal_model_bufs_cell: Option<RefCell<crate::metal::MetalModelBuffers>>,
}

impl InferenceEngine {
    /// Build an engine from a Hugging Face-style pretrained model directory.
    ///
    /// Automatically detects the model type from `config.json`. For Mamba/SSM
    /// models, delegates to [`from_pretrained_mamba`].
    pub fn from_pretrained(model_path: &Path) -> Result<Self, Box<dyn std::error::Error>> {
        // Detect model type from config.json to route SSMs differently
        let model_type = detect_model_type(&model_path.join("config.json")).unwrap_or_default();
        match model_type.as_str() {
            "mamba" | "mamba2" => return Self::from_pretrained_mamba(model_path),
            _ => {}
        }

        let config = ModelConfig::from_hf_file(&model_path.join("config.json"))?;
        let tokenizer = Tokenizer::from_model_dir(model_path)?;
        let mapper = WeightMapper::from_model_type(&config.name)?;
        let model_adapter_kind = ModelAdapterKind::from_model_name(&config.name);

        let mut model = ModelBuilder::from_config(&config);
        let mut weights = if model_path.join("model.safetensors.index.json").exists() {
            load_safetensors_sharded(model_path)?
        } else {
            let checkpoint = find_checkpoint_file(model_path)?;
            match checkpoint.extension().and_then(|ext| ext.to_str()) {
                Some("gguf") => load_gguf(&checkpoint)?,
                _ => load_safetensors(&checkpoint)?,
            }
        };

        // Split fused projections (Phi-3 uses qkv_proj and gate_up_proj)
        crate::weight_loading::weight_map::split_fused_projections(
            &mut weights,
            config.architecture.hidden_size,
            config.ffn.intermediate_size(),
            config.architecture.num_layers,
        );

        // CODEx: exact-mode pretrained loading should fail on checkpoint drift
        // instead of silently dropping required tensors.
        let result = model.load_weights(weights, &mapper)?;
        if !result.missing.is_empty() {
            return Err(Box::new(WeightError::MissingKeys(result.missing)));
        }
        if !result.unexpected.is_empty() {
            eprintln!(
                "Warning: {} unexpected weight keys (ignored)",
                result.unexpected.len()
            );
        }

        // Try to load chat template from tokenizer_config.json (best-effort).
        let chat_template = {
            let tokenizer_config_path = model_path.join("tokenizer_config.json");
            if tokenizer_config_path.exists() {
                ChatTemplate::from_tokenizer_config(&tokenizer_config_path).ok()
            } else {
                None
            }
        };

        #[cfg(feature = "metal")]
        let backend = crate::metal::ComputeBackend::auto();
        #[cfg(feature = "metal")]
        let metal_model_bufs_cell = match &backend {
            crate::metal::ComputeBackend::Metal {
                backend: ref mb, ..
            } => {
                let max_seq = config.position.max_position_embeddings();
                Some(RefCell::new(
                    crate::metal::MetalModelBuffers::from_causal_lm(&model, max_seq, &mb.device),
                ))
            }
            _ => None,
        };

        Ok(Self {
            model,
            quantized_model: None,
            ternary_model: None,
            ssm_model: None,
            config,
            model_adapter_kind,
            tokenizer: Some(tokenizer),
            chat_template,
            #[cfg(feature = "metal")]
            backend,
            #[cfg(feature = "metal")]
            metal_model_bufs_cell,
        })
    }

    /// Build an engine from just a config (no weights loaded).
    pub fn from_config(config: ModelConfig) -> Self {
        let model_adapter_kind = ModelAdapterKind::from_model_name(&config.name);
        let model = ModelBuilder::from_config(&config);
        Self {
            model,
            quantized_model: None,
            ternary_model: None,
            ssm_model: None,
            config,
            model_adapter_kind,
            tokenizer: None,
            chat_template: None,
            #[cfg(feature = "metal")]
            backend: crate::metal::ComputeBackend::auto(),
            #[cfg(feature = "metal")]
            metal_model_bufs_cell: None,
        }
    }

    /// Build an engine from a config with explicit backend selection.
    #[cfg(feature = "metal")]
    pub fn from_config_with_backend(config: ModelConfig, selection: BackendSelection) -> Self {
        let model_adapter_kind = ModelAdapterKind::from_model_name(&config.name);
        let model = ModelBuilder::from_config(&config);
        let backend = match selection {
            BackendSelection::CpuSimd => crate::metal::ComputeBackend::CpuSimd,
            BackendSelection::Auto => crate::metal::ComputeBackend::auto(),
        };
        Self {
            model,
            quantized_model: None,
            ternary_model: None,
            ssm_model: None,
            config,
            model_adapter_kind,
            tokenizer: None,
            chat_template: None,
            backend,
            metal_model_bufs_cell: None,
        }
    }

    /// Run a forward pass on token ids, returning logits.
    ///
    /// When the `metal` feature is enabled, dispatches through the
    /// `ComputeBackend` (GPU for large ops, CPU for small ones).
    pub fn forward(&self, token_ids: &[u32]) -> ModelOutput {
        #[cfg(feature = "metal")]
        {
            self.model.forward_with_backend(token_ids, &self.backend)
        }
        #[cfg(not(feature = "metal"))]
        {
            self.model.forward(token_ids)
        }
    }

    /// Total parameter count of the underlying model.
    pub fn param_count(&self) -> usize {
        self.model.param_count()
    }

    pub fn encode(&self, text: &str) -> Result<Vec<u32>, TokenizerError> {
        let tokenizer = self
            .tokenizer
            .as_ref()
            .ok_or_else(|| TokenizerError::Invalid("No tokenizer loaded".into()))?;
        tokenizer.encode(text)
    }

    pub fn decode(&self, token_ids: &[u32]) -> Result<String, TokenizerError> {
        let tokenizer = self
            .tokenizer
            .as_ref()
            .ok_or_else(|| TokenizerError::Invalid("No tokenizer loaded".into()))?;
        tokenizer.decode(token_ids)
    }

    pub fn tokenizer(&self) -> Option<&Tokenizer> {
        self.tokenizer.as_ref()
    }

    /// Return the current build/runtime capability report, annotated with the loaded model name.
    pub fn capability_report(&self) -> CapabilityReport {
        CapabilityReport::for_model_name(Some(&self.config.name))
    }

    pub fn model_adapter(&self) -> &'static dyn ModelAdapter {
        adapter_for_kind(self.model_adapter_kind)
    }

    pub fn model_family(&self) -> &'static str {
        self.model_adapter().family()
    }

    pub fn default_cli_thinking_mode(&self) -> ThinkingMode {
        self.model_adapter().default_cli_thinking_mode()
    }

    pub fn reasoning_markers(&self) -> Option<ReasoningMarkers> {
        self.model_adapter().reasoning_markers()
    }

    /// Format a list of chat messages into a prompt string using the loaded
    /// chat template.  Falls back to the default Qwen3 / ChatML format if
    /// no template was loaded from `tokenizer_config.json`.
    pub fn format_chat(
        &self,
        messages: &[ChatMessage],
    ) -> Result<String, Box<dyn std::error::Error>> {
        self.format_chat_with_mode(messages, ThinkingMode::Auto)
    }

    pub fn format_chat_with_mode(
        &self,
        messages: &[ChatMessage],
        thinking_mode: ThinkingMode,
    ) -> Result<String, Box<dyn std::error::Error>> {
        self.model_adapter().format_chat_prompt(
            self.chat_template.as_ref(),
            messages,
            thinking_mode,
        )
    }

    /// Create a KV cache sized for this model's architecture.
    pub fn create_kv_cache(
        &self,
        max_seq_len: usize,
    ) -> Result<KVCache, Box<dyn std::error::Error>> {
        let cache = KVCache::new(
            self.config.architecture.num_layers,
            max_seq_len,
            self.config.attention.num_kv_heads(),
            self.config.attention.head_dim(),
        )?;
        Ok(cache)
    }

    /// Create a model state sized for this model's architecture.
    ///
    /// For transformer models this wraps a KV cache.
    /// For SSM models (Mamba, RWKV) this returns `ModelState::Recurrent`
    /// since SSMs manage their own internal state.
    pub fn create_state(
        &self,
        max_seq_len: usize,
    ) -> Result<ModelState, Box<dyn std::error::Error>> {
        if self.ssm_model.is_some() {
            // SSMs manage their own state internally via RefCell
            Ok(ModelState::Recurrent)
        } else {
            let cache = KVCache::new(
                self.config.architecture.num_layers,
                max_seq_len,
                self.config.attention.num_kv_heads(),
                self.config.attention.head_dim(),
            )?;
            Ok(ModelState::KvCache(cache))
        }
    }

    /// Quantize the model to INT8. Call after loading weights.
    pub fn quantize(&mut self) {
        self.quantized_model = Some(quantize_model(&self.model));
    }

    /// Whether the engine has a quantized model available.
    pub fn is_quantized(&self) -> bool {
        self.quantized_model.is_some()
    }

    /// Quantize the model to ternary 2-bit weights. Call after loading weights.
    pub fn quantize_ternary(&mut self) {
        self.ternary_model = Some(quantize_model_ternary(&self.model));
    }

    /// Whether the engine has a ternary model available.
    pub fn is_ternary(&self) -> bool {
        self.ternary_model.is_some()
    }

    pub fn runtime_plan(&self) -> RuntimePlan {
        let quantized = self.is_quantized();
        #[cfg(feature = "metal")]
        let backend = if self.backend.is_gpu() {
            RuntimeBackend::Metal
        } else {
            RuntimeBackend::CpuSimd
        };
        #[cfg(not(feature = "metal"))]
        let backend = RuntimeBackend::CpuSimd;
        #[cfg(feature = "metal")]
        let is_metal = matches!(backend, RuntimeBackend::Metal);
        #[cfg(not(feature = "metal"))]
        let is_metal = false;

        let prefill_path = if quantized {
            RuntimePath::QuantizedCpu
        } else if is_metal {
            RuntimePath::BackendDispatch
        } else {
            RuntimePath::CpuCached
        };
        let decode_path = if quantized {
            RuntimePath::QuantizedCpu
        } else if is_metal {
            RuntimePath::MetalPerLayer
        } else {
            RuntimePath::CpuCached
        };
        let prefill_strategy = if quantized {
            "quantized_prefill_cpu"
        } else if is_metal {
            "metal_prefill_dispatch"
        } else {
            "cpu_cached_prefill"
        };
        let decode_strategy = if quantized {
            "quantized_cached_decode_cpu"
        } else if is_metal {
            "metal_decode_per_layer"
        } else {
            "cpu_cached_decode"
        };

        RuntimePlan {
            family: self.model_family(),
            backend,
            quantized,
            prefill_path,
            decode_path,
            prefill_strategy,
            decode_strategy,
        }
    }

    pub fn runtime_summary(&self) -> RuntimeSummary {
        self.runtime_plan()
    }

    pub fn generation_pipeline(&self) -> GenerationPipeline<'_> {
        if let Some(ref ssm) = self.ssm_model {
            GenerationPipeline::new(ssm.as_ref())
        } else if let Some(ref tmodel) = self.ternary_model {
            GenerationPipeline::new(tmodel)
        } else if let Some(ref qmodel) = self.quantized_model {
            GenerationPipeline::new(qmodel)
        } else {
            #[cfg(feature = "metal")]
            {
                GenerationPipeline::with_backend(&self.model, &self.backend)
            }
            #[cfg(not(feature = "metal"))]
            {
                GenerationPipeline::new(&self.model)
            }
        }
    }

    pub fn generate_text(
        &self,
        prompt: &str,
        config: GenerationConfig,
    ) -> Result<GenerationOutput, Box<dyn std::error::Error>> {
        let prompt_tokens = self.encode(prompt)?;
        let max_seq = prompt_tokens.len() + config.max_new_tokens;
        let mut state = self.create_state(max_seq)?;
        let pipeline = self.generation_pipeline();
        let mut output = pipeline.generate(&prompt_tokens, config, Some(&mut state));
        let generated = &output.token_ids[output.num_prompt_tokens..];
        output.text = self.decode(generated)?;
        Ok(output)
    }

    /// Load a Mamba model from a HuggingFace-style directory.
    ///
    /// Reads `config.json` for `MambaConfig`, loads safetensors weights,
    /// and creates a `MambaModel` stored as `ssm_model`.
    pub fn from_pretrained_mamba(model_path: &Path) -> Result<Self, Box<dyn std::error::Error>> {
        let mamba_config = parse_mamba_config(&model_path.join("config.json"))?;

        // Load weights from safetensors
        let weights = if model_path.join("model.safetensors.index.json").exists() {
            load_safetensors_sharded(model_path)?
        } else {
            let checkpoint = find_checkpoint_file(model_path)?;
            load_safetensors(&checkpoint)?
        };

        let mamba_model = MambaModel::from_weights(mamba_config.clone(), &weights)
            .map_err(|e| -> Box<dyn std::error::Error> { e.into() })?;

        // Try to load tokenizer (best-effort for Mamba models)
        let tokenizer = Tokenizer::from_model_dir(model_path).ok();

        // Try to load chat template (best-effort)
        let chat_template = {
            let tokenizer_config_path = model_path.join("tokenizer_config.json");
            if tokenizer_config_path.exists() {
                ChatTemplate::from_tokenizer_config(&tokenizer_config_path).ok()
            } else {
                None
            }
        };

        // Build a minimal ModelConfig for engine metadata.
        // SSM models don't use this for inference (they use MambaConfig internally),
        // but it is needed for adapter selection and runtime metadata.
        let config = minimal_config_for_ssm("mamba", &mamba_config);

        let model_adapter_kind = ModelAdapterKind::from_model_name("mamba");
        let model = ModelBuilder::from_config(&config);

        #[cfg(feature = "metal")]
        let backend = crate::metal::ComputeBackend::auto();

        Ok(Self {
            model,
            quantized_model: None,
            ternary_model: None,
            ssm_model: Some(Box::new(mamba_model)),
            config,
            model_adapter_kind,
            tokenizer,
            chat_template,
            #[cfg(feature = "metal")]
            backend,
            #[cfg(feature = "metal")]
            metal_model_bufs_cell: None,
        })
    }

    /// Whether the engine has an SSM model loaded.
    pub fn is_ssm(&self) -> bool {
        self.ssm_model.is_some()
    }
}

/// Detect the `model_type` field from a HuggingFace `config.json`.
fn detect_model_type(config_path: &Path) -> Result<String, Box<dyn std::error::Error>> {
    let file = std::fs::File::open(config_path)?;
    let json: serde_json::Value = serde_json::from_reader(file)?;
    let model_type = json
        .get("model_type")
        .and_then(|v| v.as_str())
        .unwrap_or("unknown");
    Ok(model_type.to_string())
}

/// Parse a `MambaConfig` from a HuggingFace-style `config.json`.
///
/// Supports both original Mamba naming (`d_state`, `d_conv`, `expand`) and
/// HuggingFace transformers naming (`state_size`, `conv_kernel`).
fn parse_mamba_config(config_path: &Path) -> Result<MambaConfig, Box<dyn std::error::Error>> {
    let file = std::fs::File::open(config_path)?;
    let json: serde_json::Value = serde_json::from_reader(file)?;
    Ok(MambaConfig {
        d_model: json["hidden_size"]
            .as_u64()
            .or(json["d_model"].as_u64())
            .unwrap_or(768) as usize,
        d_state: json["state_size"]
            .as_u64()
            .or(json["d_state"].as_u64())
            .unwrap_or(16) as usize,
        d_conv: json["conv_kernel"]
            .as_u64()
            .or(json["d_conv"].as_u64())
            .unwrap_or(4) as usize,
        expand: json["expand"].as_u64().unwrap_or(2) as usize,
        num_layers: json["num_hidden_layers"]
            .as_u64()
            .or(json["n_layer"].as_u64())
            .unwrap_or(24) as usize,
        vocab_size: json["vocab_size"].as_u64().unwrap_or(50280) as usize,
        norm_eps: json["layer_norm_epsilon"]
            .as_f64()
            .or(json["norm_epsilon"].as_f64())
            .unwrap_or(1e-5),
    })
}

/// Build a minimal `ModelConfig` for SSM models that need engine metadata.
///
/// The SSM model itself uses its own config for inference, but
/// `InferenceEngine` needs a `ModelConfig` for things like adapter selection.
fn minimal_config_for_ssm(name: &str, mamba: &MambaConfig) -> ModelConfig {
    use crate::config::*;
    ModelConfig {
        name: name.to_string(),
        architecture: ArchitectureConfig {
            hidden_size: mamba.d_model,
            num_layers: mamba.num_layers,
            vocab_size: mamba.vocab_size,
            max_sequence_length: 2048, // SSMs have no fixed max
            tie_word_embeddings: false,
            embed_scale: None,
        },
        // SSMs don't use attention, but we need valid placeholder values.
        attention: AttentionConfig::GQA {
            num_heads: 1,
            num_kv_heads: 1,
            head_dim: mamba.d_model,
        },
        norm: NormConfig::RMSNorm {
            eps: mamba.norm_eps,
        },
        ffn: FFNConfig::SwiGLU {
            intermediate_size: mamba.d_inner(),
        },
        position: PositionConfig::RoPE {
            base: 10000.0,
            max_position_embeddings: 2048,
            style: Default::default(),
            scaling: Default::default(),
        },
        quantization: QuantConfig::F32,
    }
}

fn find_checkpoint_file(model_path: &Path) -> Result<PathBuf, WeightError> {
    let canonical = model_path.join("model.safetensors");
    if canonical.exists() {
        return Ok(canonical);
    }

    let mut safetensors = Vec::new();
    let mut gguf = Vec::new();
    for entry in std::fs::read_dir(model_path).map_err(WeightError::Io)? {
        let entry = entry.map_err(WeightError::Io)?;
        let path = entry.path();
        if !path.is_file() {
            continue;
        }
        match path.extension().and_then(|ext| ext.to_str()) {
            Some("safetensors") => safetensors.push(path),
            Some("gguf") => gguf.push(path),
            _ => {}
        }
    }

    safetensors.sort();
    gguf.sort();

    if let Some(path) = safetensors.into_iter().next() {
        return Ok(path);
    }
    if let Some(path) = gguf.into_iter().next() {
        return Ok(path);
    }

    Err(WeightError::InvalidFormat(
        "No checkpoint file found in model directory".into(),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::*;

    fn test_config() -> ModelConfig {
        ModelConfig {
            name: "InferenceEngineRuntimeSummaryTest".to_string(),
            architecture: ArchitectureConfig {
                hidden_size: 64,
                num_layers: 2,
                vocab_size: 128,
                max_sequence_length: 64,
                tie_word_embeddings: true,
                embed_scale: None,
            },
            attention: AttentionConfig::GQA {
                num_heads: 4,
                num_kv_heads: 2,
                head_dim: 16,
            },
            norm: NormConfig::RMSNorm { eps: 1e-6 },
            ffn: FFNConfig::SwiGLU {
                intermediate_size: 128,
            },
            position: PositionConfig::RoPE {
                base: 10000.0,
                max_position_embeddings: 64,
                style: Default::default(),
                scaling: Default::default(),
            },
            quantization: QuantConfig::F32,
        }
    }

    #[test]
    fn runtime_summary_reports_cpu_cached_path() {
        let engine = InferenceEngine::from_config(test_config());
        let summary = engine.runtime_summary();

        assert_eq!(summary.quantized, false);
        assert_eq!(summary.family, "generic");
        // Backend depends on feature flags
        #[cfg(feature = "metal")]
        assert_eq!(summary.backend, RuntimeBackend::Metal);
        #[cfg(not(feature = "metal"))]
        assert_eq!(summary.backend, RuntimeBackend::CpuSimd);
    }

    #[cfg(feature = "metal")]
    #[test]
    fn runtime_summary_reports_cpu_backend_selection() {
        let engine =
            InferenceEngine::from_config_with_backend(test_config(), BackendSelection::CpuSimd);
        let summary = engine.runtime_summary();

        assert_eq!(summary.backend, RuntimeBackend::CpuSimd);
        assert_eq!(summary.prefill_path, RuntimePath::CpuCached);
        assert_eq!(summary.decode_path, RuntimePath::CpuCached);
        assert_eq!(summary.prefill_strategy, "cpu_cached_prefill");
        assert_eq!(summary.decode_strategy, "cpu_cached_decode");
    }

    #[test]
    fn test_engine_quantize_ternary() {
        use std::collections::HashMap;
        use crate::weight_loading::{AlignedBuffer, RawTensor, WeightMapper};

        let config = test_config();

        // Generate minimal fake weights so quantize_ternary has non-empty matrices.
        let h = config.architecture.hidden_size;
        let vocab = config.architecture.vocab_size;
        let q_dim = config.attention.num_heads() * config.attention.head_dim();
        let kv_dim = config.attention.num_kv_heads() * config.attention.head_dim();
        let inter = config.ffn.intermediate_size();
        let nl = config.architecture.num_layers;

        let fake = |shape: Vec<usize>| -> RawTensor {
            let n: usize = shape.iter().product();
            RawTensor {
                data: AlignedBuffer::from_slice(&vec![0.01f32; n]),
                shape,
            }
        };

        let mut weights: HashMap<String, RawTensor> = HashMap::new();
        weights.insert("model.embed_tokens.weight".into(), fake(vec![vocab, h]));
        weights.insert("model.norm.weight".into(), fake(vec![h]));
        weights.insert("lm_head.weight".into(), fake(vec![vocab, h]));
        for i in 0..nl {
            weights.insert(format!("model.layers.{i}.input_layernorm.weight"), fake(vec![h]));
            weights.insert(format!("model.layers.{i}.self_attn.q_proj.weight"), fake(vec![q_dim, h]));
            weights.insert(format!("model.layers.{i}.self_attn.k_proj.weight"), fake(vec![kv_dim, h]));
            weights.insert(format!("model.layers.{i}.self_attn.v_proj.weight"), fake(vec![kv_dim, h]));
            weights.insert(format!("model.layers.{i}.self_attn.o_proj.weight"), fake(vec![h, q_dim]));
            weights.insert(format!("model.layers.{i}.post_attention_layernorm.weight"), fake(vec![h]));
            weights.insert(format!("model.layers.{i}.mlp.gate_proj.weight"), fake(vec![inter, h]));
            weights.insert(format!("model.layers.{i}.mlp.up_proj.weight"), fake(vec![inter, h]));
            weights.insert(format!("model.layers.{i}.mlp.down_proj.weight"), fake(vec![h, inter]));
        }

        let mut engine = InferenceEngine::from_config(config);
        let mapper = WeightMapper::from_model_type("InferenceEngineRuntimeSummaryTest")
            .unwrap_or_else(|_| WeightMapper::llama());
        let _ = engine.model.load_weights(weights, &mapper);

        assert!(!engine.is_ternary());
        engine.quantize_ternary();
        assert!(engine.is_ternary());
        let _pipeline = engine.generation_pipeline();
    }

    #[test]
    fn qwen3_config_selects_qwen3_adapter() {
        let mut config = test_config();
        config.name = "qwen3".into();
        let engine = InferenceEngine::from_config(config);
        assert_eq!(engine.model_adapter_kind, ModelAdapterKind::Qwen3);
        assert_eq!(engine.model_family(), "qwen3");
        assert_eq!(
            engine.reasoning_markers(),
            Some(ReasoningMarkers {
                start: "<think>",
                end: "</think>",
            })
        );
    }

    // ── SSM / Mamba engine tests ─────────────────────────────────────

    fn make_tiny_mamba_model() -> MambaModel {
        use crate::ssm::mamba_block::MambaBlock;

        let config = MambaConfig::tiny_test();
        let d_model = config.d_model;
        let d_inner = config.d_inner();
        let d_state = config.d_state;
        let d_conv = config.d_conv;
        let vocab = config.vocab_size;

        let pseudo = |seed: u64, len: usize| -> Vec<f32> {
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
        };

        let embed_tokens = pseudo(100, vocab * d_model);
        let final_norm_weight = vec![1.0f32; d_model];
        let lm_head_weight = pseudo(200, vocab * d_model);

        let mut blocks = Vec::new();
        for layer_idx in 0..config.num_layers {
            let s = (layer_idx as u64 + 1) * 1000;
            blocks.push(MambaBlock {
                d_model,
                d_inner,
                d_state,
                d_conv,
                norm_weight: vec![1.0f32; d_model],
                norm_eps: config.norm_eps as f32,
                in_proj_weight: pseudo(s + 1, 2 * d_inner * d_model),
                in_proj_bias: vec![],
                conv1d_weight: pseudo(s + 2, d_inner * d_conv),
                conv1d_bias: vec![0.0f32; d_inner],
                x_proj_weight: pseudo(s + 3, (2 * d_state + 1) * d_inner),
                dt_proj_weight: pseudo(s + 4, d_inner),
                dt_proj_bias: vec![0.0f32; d_inner],
                a_log: pseudo(s + 5, d_inner * d_state)
                    .into_iter()
                    .map(|v| -v.abs() - 0.1)
                    .collect(),
                d_param: vec![1.0f32; d_inner],
                out_proj_weight: pseudo(s + 6, d_model * d_inner),
                out_proj_bias: vec![],
            });
        }

        MambaModel::new(config, embed_tokens, blocks, final_norm_weight, lm_head_weight)
    }

    #[test]
    fn test_engine_with_ssm_model() {
        let mamba = make_tiny_mamba_model();
        let mamba_config = mamba.config.clone();

        let config = minimal_config_for_ssm("mamba", &mamba_config);
        let mut engine = InferenceEngine::from_config(config);
        engine.ssm_model = Some(Box::new(mamba));

        assert!(engine.is_ssm());

        // generation_pipeline should use the SSM model
        let pipeline = engine.generation_pipeline();
        let output = pipeline.generate(
            &[1, 2, 3],
            crate::generation::GenerationConfig {
                max_new_tokens: 4,
                ..Default::default()
            },
            Some(&mut ModelState::Recurrent),
        );
        assert!(
            output.token_ids.len() > 3,
            "SSM engine should generate tokens beyond the prompt"
        );
    }

    #[test]
    fn test_engine_ssm_create_state_returns_recurrent() {
        let mamba = make_tiny_mamba_model();
        let mamba_config = mamba.config.clone();

        let config = minimal_config_for_ssm("mamba", &mamba_config);
        let mut engine = InferenceEngine::from_config(config);
        engine.ssm_model = Some(Box::new(mamba));

        let state = engine.create_state(128).expect("create_state should succeed");
        assert!(
            matches!(state, ModelState::Recurrent),
            "SSM engine state should be Recurrent, got {:?}",
            std::mem::discriminant(&state),
        );
    }

    #[test]
    fn test_engine_transformer_create_state_returns_kv_cache() {
        let engine = InferenceEngine::from_config(test_config());
        assert!(!engine.is_ssm());
        let state = engine.create_state(64).expect("create_state should succeed");
        assert!(
            matches!(state, ModelState::KvCache(_)),
            "Transformer engine state should be KvCache"
        );
    }

    #[test]
    fn test_detect_model_type_mamba() {
        let dir = tempfile::tempdir().unwrap();
        let config_path = dir.path().join("config.json");
        std::fs::write(
            &config_path,
            r#"{"model_type": "mamba", "hidden_size": 768}"#,
        )
        .unwrap();
        let model_type = detect_model_type(&config_path).unwrap();
        assert_eq!(model_type, "mamba");
    }

    #[test]
    fn test_detect_model_type_unknown_fallback() {
        let dir = tempfile::tempdir().unwrap();
        let config_path = dir.path().join("config.json");
        std::fs::write(&config_path, r#"{"hidden_size": 768}"#).unwrap();
        let model_type = detect_model_type(&config_path).unwrap();
        assert_eq!(model_type, "unknown");
    }

    #[test]
    fn test_parse_mamba_config_hf_style() {
        let dir = tempfile::tempdir().unwrap();
        let config_path = dir.path().join("config.json");
        std::fs::write(
            &config_path,
            r#"{
                "model_type": "mamba",
                "hidden_size": 768,
                "state_size": 16,
                "conv_kernel": 4,
                "expand": 2,
                "num_hidden_layers": 24,
                "vocab_size": 50280,
                "layer_norm_epsilon": 1e-5
            }"#,
        )
        .unwrap();

        let cfg = parse_mamba_config(&config_path).unwrap();
        assert_eq!(cfg.d_model, 768);
        assert_eq!(cfg.d_state, 16);
        assert_eq!(cfg.d_conv, 4);
        assert_eq!(cfg.expand, 2);
        assert_eq!(cfg.num_layers, 24);
        assert_eq!(cfg.vocab_size, 50280);
        assert!((cfg.norm_eps - 1e-5).abs() < 1e-10);
    }

    #[test]
    fn test_parse_mamba_config_original_style() {
        let dir = tempfile::tempdir().unwrap();
        let config_path = dir.path().join("config.json");
        std::fs::write(
            &config_path,
            r#"{
                "model_type": "mamba",
                "d_model": 1024,
                "d_state": 16,
                "d_conv": 4,
                "expand": 2,
                "n_layer": 48,
                "vocab_size": 50280
            }"#,
        )
        .unwrap();

        let cfg = parse_mamba_config(&config_path).unwrap();
        assert_eq!(cfg.d_model, 1024);
        assert_eq!(cfg.d_state, 16);
        assert_eq!(cfg.d_conv, 4);
        assert_eq!(cfg.num_layers, 48);
    }
}
