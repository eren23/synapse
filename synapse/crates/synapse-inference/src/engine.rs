use std::path::Path;

use crate::config::ModelConfig;
use crate::model::{CausalLM, ModelBuilder, ModelOutput};

/// High-level inference orchestrator.
///
/// Assembles config → model → weights → KV-cache → tokenizer into a single
/// entry point for text generation.
pub struct InferenceEngine {
    pub model: CausalLM,
    pub config: ModelConfig,
}

impl InferenceEngine {
    /// Build an engine from a pretrained model directory.
    ///
    /// Steps:
    /// 1. Read model config (passed in or from `config.json`)
    /// 2. Build model skeleton from config
    /// 3. Load weights from checkpoint files
    /// 4. Init KV-cache (deferred to first forward)
    /// 5. Init tokenizer (not yet implemented)
    pub fn from_pretrained(
        model_path: &Path,
        config: ModelConfig,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        // Build model from config
        let model = ModelBuilder::from_config(&config);

        // Weight loading from files would go here:
        //   - Scan for *.safetensors / *.gguf in model_path
        //   - Use appropriate loader + weight mapper
        //   - Call model.load_weights(...)
        let _ = model_path; // acknowledge path for future use

        Ok(Self { model, config })
    }

    /// Build an engine from just a config (no weights loaded).
    pub fn from_config(config: ModelConfig) -> Self {
        let model = ModelBuilder::from_config(&config);
        Self {
            model,
            config,
        }
    }

    /// Run a forward pass on token ids, returning logits.
    pub fn forward(&self, token_ids: &[u32]) -> ModelOutput {
        self.model.forward(token_ids)
    }

    /// Total parameter count of the underlying model.
    pub fn param_count(&self) -> usize {
        self.model.param_count()
    }
}
