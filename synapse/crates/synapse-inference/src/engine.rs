use std::path::{Path, PathBuf};

use crate::capabilities::CapabilityReport;
use crate::chat_template::{ChatMessage, ChatTemplate};
use crate::config::ModelConfig;
use crate::generation::{GenerationConfig, GenerationOutput, GenerationPipeline};
use crate::kv_cache::KVCache;
use crate::model::{CausalLM, ModelBuilder, ModelOutput};
use crate::model_adapter::{
    adapter_for_kind, ModelAdapter, ModelAdapterKind, ReasoningMarkers, ThinkingMode,
};
use crate::quantization::{quantize_model, QuantizedCausalLM};
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
    pub fn from_pretrained(model_path: &Path) -> Result<Self, Box<dyn std::error::Error>> {
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

    /// Quantize the model to INT8. Call after loading weights.
    pub fn quantize(&mut self) {
        self.quantized_model = Some(quantize_model(&self.model));
    }

    /// Whether the engine has a quantized model available.
    pub fn is_quantized(&self) -> bool {
        self.quantized_model.is_some()
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
        if let Some(ref qmodel) = self.quantized_model {
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
        let mut cache = self.create_kv_cache(max_seq)?;
        let pipeline = self.generation_pipeline();
        let mut output = pipeline.generate(&prompt_tokens, config, Some(&mut cache));
        let generated = &output.token_ids[output.num_prompt_tokens..];
        output.text = self.decode(generated)?;
        Ok(output)
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
        assert_eq!(summary.backend, RuntimeBackend::CpuSimd);
        assert_eq!(summary.prefill_path, RuntimePath::CpuCached);
        assert_eq!(summary.decode_path, RuntimePath::CpuCached);
        assert_eq!(summary.prefill_strategy, "cpu_cached_prefill");
        assert_eq!(summary.decode_strategy, "cpu_cached_decode");
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
}
