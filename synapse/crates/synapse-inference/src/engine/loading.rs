use std::path::Path;

use super::InferenceEngine;
use super::config_parsers::{
    detect_model_type, find_checkpoint_file, minimal_config_for_hybrid, minimal_config_for_rwkv,
    minimal_config_for_ssm, parse_hybrid_config, parse_mamba_config, parse_rwkv_config,
};

use crate::chat_template::ChatTemplate;
use crate::config::ModelConfig;
use crate::models::ssm::hybrid::model::HybridModel;
use crate::models::ssm::mamba::model::MambaModel;
use crate::models::ssm::rwkv::model::RwkvModel;
use crate::models::{ModelBuilder};
use crate::model_adapter::ModelAdapterKind;
use crate::tokenizer::Tokenizer;
use crate::weight_loading::{
    load_gguf, load_safetensors, load_safetensors_sharded, WeightError, WeightMapper,
};

#[cfg(feature = "metal")]
use std::cell::RefCell;

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
            "rwkv" | "rwkv7" => return Self::from_pretrained_rwkv(model_path),
            "qwen3_5" | "qwen3.5" => return Self::from_pretrained_hybrid(model_path),
            _ => {
                // Also detect hybrid from config fields
                let config_path = model_path.join("config.json");
                if config_path.exists() {
                    if let Ok(file) = std::fs::File::open(&config_path) {
                        if let Ok(json) = serde_json::from_reader::<_, serde_json::Value>(file) {
                            if json.get("full_attention_interval").is_some()
                                || json.get("layer_types").is_some()
                            {
                                return Self::from_pretrained_hybrid(model_path);
                            }
                        }
                    }
                }
            }
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
    pub fn from_config_with_backend(config: ModelConfig, selection: super::BackendSelection) -> Self {
        let model_adapter_kind = ModelAdapterKind::from_model_name(&config.name);
        let model = ModelBuilder::from_config(&config);
        let backend = match selection {
            super::BackendSelection::CpuSimd => crate::metal::ComputeBackend::CpuSimd,
            super::BackendSelection::Auto => crate::metal::ComputeBackend::auto(),
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

    /// Load an RWKV-7 model from a HuggingFace-style directory.
    ///
    /// Reads `config.json` for `RwkvConfig`, loads safetensors weights,
    /// and creates an `RwkvModel` stored as `ssm_model`.
    pub fn from_pretrained_rwkv(model_path: &Path) -> Result<Self, Box<dyn std::error::Error>> {
        let rwkv_config = parse_rwkv_config(&model_path.join("config.json"))?;

        // Load weights from safetensors
        let weights = if model_path.join("model.safetensors.index.json").exists() {
            load_safetensors_sharded(model_path)?
        } else {
            let checkpoint = find_checkpoint_file(model_path)?;
            load_safetensors(&checkpoint)?
        };

        let rwkv_model = RwkvModel::from_weights(rwkv_config.clone(), &weights)
            .map_err(|e| -> Box<dyn std::error::Error> { e.into() })?;

        // Try to load tokenizer (best-effort)
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

        let config = minimal_config_for_rwkv("rwkv", &rwkv_config);
        let model_adapter_kind = ModelAdapterKind::from_model_name("rwkv");
        let model = ModelBuilder::from_config(&config);

        #[cfg(feature = "metal")]
        let backend = crate::metal::ComputeBackend::auto();

        Ok(Self {
            model,
            quantized_model: None,
            ternary_model: None,
            ssm_model: Some(Box::new(rwkv_model)),
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

    /// Load a Qwen3.5-style hybrid model from a HuggingFace-style directory.
    ///
    /// Reads `config.json` for `HybridConfig`, loads safetensors weights,
    /// and creates a `HybridModel` stored as `ssm_model`.
    pub fn from_pretrained_hybrid(model_path: &Path) -> Result<Self, Box<dyn std::error::Error>> {
        let hybrid_config = parse_hybrid_config(&model_path.join("config.json"))?;

        let weights = if model_path.join("model.safetensors.index.json").exists() {
            load_safetensors_sharded(model_path)?
        } else {
            let checkpoint = find_checkpoint_file(model_path)?;
            load_safetensors(&checkpoint)?
        };

        let max_kv_seq = 2048;
        let hybrid_model = HybridModel::from_weights(hybrid_config.clone(), &weights, max_kv_seq)
            .map_err(|e| -> Box<dyn std::error::Error> { e.into() })?;

        let tokenizer = Tokenizer::from_model_dir(model_path).ok();

        let chat_template = {
            let tokenizer_config_path = model_path.join("tokenizer_config.json");
            if tokenizer_config_path.exists() {
                ChatTemplate::from_tokenizer_config(&tokenizer_config_path).ok()
            } else {
                None
            }
        };

        let config = minimal_config_for_hybrid("qwen3.5", &hybrid_config);
        let model_adapter_kind = ModelAdapterKind::from_model_name("qwen3.5");
        let model = ModelBuilder::from_config(&config);

        #[cfg(feature = "metal")]
        let backend = crate::metal::ComputeBackend::auto();

        Ok(Self {
            model,
            quantized_model: None,
            ternary_model: None,
            ssm_model: Some(Box::new(hybrid_model)),
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
}
