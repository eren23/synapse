use std::path::{Path, PathBuf};

use crate::config::ModelConfig;
use crate::generation::{GenerationConfig, GenerationOutput, GenerationPipeline};
use crate::kv_cache::KVCache;
use crate::model::{CausalLM, ModelBuilder, ModelOutput};
use crate::tokenizer::{Tokenizer, TokenizerError};
use crate::weight_loading::{load_gguf, load_safetensors, WeightError, WeightMapper};

/// Which compute backend to use for dispatch.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BackendSelection {
    /// Always use CPU SIMD (Zig FFI).
    CpuSimd,
    /// Use Metal GPU when available, CPU SIMD fallback.
    #[cfg(feature = "metal")]
    Auto,
}

/// High-level inference orchestrator.
///
/// Assembles config → model → weights → KV-cache → tokenizer into a single
/// entry point for text generation.
pub struct InferenceEngine {
    pub model: CausalLM,
    pub config: ModelConfig,
    pub tokenizer: Option<Tokenizer>,
    #[cfg(feature = "metal")]
    pub backend: crate::metal::ComputeBackend,
}

impl InferenceEngine {
    /// Build an engine from a Hugging Face-style pretrained model directory.
    pub fn from_pretrained(model_path: &Path) -> Result<Self, Box<dyn std::error::Error>> {
        let config = ModelConfig::from_hf_file(&model_path.join("config.json"))?;
        let checkpoint = find_checkpoint_file(model_path)?;
        let tokenizer = Tokenizer::from_model_dir(model_path)?;
        let mapper = WeightMapper::qwen3();

        let mut model = ModelBuilder::from_config(&config);
        let weights = match checkpoint.extension().and_then(|ext| ext.to_str()) {
            Some("gguf") => load_gguf(&checkpoint)?,
            _ => load_safetensors(&checkpoint)?,
        };

        // CODEx: exact-mode pretrained loading should fail on checkpoint drift
        // instead of silently dropping required tensors.
        let result = model.load_weights(weights, &mapper)?;
        if !result.missing.is_empty() {
            return Err(Box::new(WeightError::MissingKeys(result.missing)));
        }
        if !result.unexpected.is_empty() {
            return Err(Box::new(WeightError::UnexpectedKeys(result.unexpected)));
        }

        Ok(Self {
            model,
            config,
            tokenizer: Some(tokenizer),
            #[cfg(feature = "metal")]
            backend: crate::metal::ComputeBackend::auto(),
        })
    }

    /// Build an engine from just a config (no weights loaded).
    pub fn from_config(config: ModelConfig) -> Self {
        let model = ModelBuilder::from_config(&config);
        Self {
            model,
            config,
            tokenizer: None,
            #[cfg(feature = "metal")]
            backend: crate::metal::ComputeBackend::auto(),
        }
    }

    /// Build an engine from a config with explicit backend selection.
    #[cfg(feature = "metal")]
    pub fn from_config_with_backend(config: ModelConfig, selection: BackendSelection) -> Self {
        let model = ModelBuilder::from_config(&config);
        let backend = match selection {
            BackendSelection::CpuSimd => crate::metal::ComputeBackend::CpuSimd,
            BackendSelection::Auto => crate::metal::ComputeBackend::auto(),
        };
        Self {
            model,
            config,
            tokenizer: None,
            backend,
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

    /// Create a KV cache sized for this model's architecture.
    pub fn create_kv_cache(&self, max_seq_len: usize) -> Result<KVCache, Box<dyn std::error::Error>> {
        let cache = KVCache::new(
            self.config.architecture.num_layers,
            max_seq_len,
            self.config.attention.num_kv_heads(),
            self.config.attention.head_dim(),
        )?;
        Ok(cache)
    }

    pub fn generate_text(
        &self,
        prompt: &str,
        config: GenerationConfig,
    ) -> Result<GenerationOutput, Box<dyn std::error::Error>> {
        let prompt_tokens = self.encode(prompt)?;
        let max_seq = prompt_tokens.len() + config.max_new_tokens;
        let mut cache = self.create_kv_cache(max_seq)?;
        let pipeline = GenerationPipeline::new(&self.model);
        let mut output = pipeline.generate(&prompt_tokens, config, Some(&mut cache));
        let generated = &output.token_ids[output.num_prompt_tokens..];
        output.text = self.decode(generated)?;
        Ok(output)
    }
}

fn find_checkpoint_file(model_path: &Path) -> Result<PathBuf, WeightError> {
    let index = model_path.join("model.safetensors.index.json");
    if index.exists() {
        return Err(WeightError::InvalidFormat(
            "Sharded safetensors checkpoints are not supported yet".into(),
        ));
    }

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
