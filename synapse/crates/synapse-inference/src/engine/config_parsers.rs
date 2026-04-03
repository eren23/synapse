use std::path::{Path, PathBuf};

use crate::config::ModelConfig;
use crate::models::ssm::mamba::config::MambaConfig;
use crate::models::ssm::rwkv::config::RwkvConfig;
use crate::models::ssm::hybrid::config::{HybridConfig, LayerKind};
use crate::weight_loading::WeightError;

/// Detect the `model_type` field from a HuggingFace `config.json`.
pub(crate) fn detect_model_type(config_path: &Path) -> Result<String, Box<dyn std::error::Error>> {
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
pub(crate) fn parse_mamba_config(config_path: &Path) -> Result<MambaConfig, Box<dyn std::error::Error>> {
    let file = std::fs::File::open(config_path)?;
    let json: serde_json::Value = serde_json::from_reader(file)?;
    let d_model = json["hidden_size"]
        .as_u64()
        .or(json["d_model"].as_u64())
        .unwrap_or(768) as usize;

    // dt_rank defaults to ceil(d_model / 16) per Mamba paper
    let dt_rank = json["dt_rank"]
        .as_u64()
        .or(json["time_step_rank"].as_u64())
        .map(|v| v as usize)
        .unwrap_or_else(|| (d_model + 15) / 16);

    Ok(MambaConfig {
        d_model,
        d_state: json["state_size"]
            .as_u64()
            .or(json["d_state"].as_u64())
            .unwrap_or(16) as usize,
        d_conv: json["conv_kernel"]
            .as_u64()
            .or(json["d_conv"].as_u64())
            .unwrap_or(4) as usize,
        expand: json["expand"].as_u64().unwrap_or(2) as usize,
        dt_rank,
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
pub(crate) fn minimal_config_for_ssm(name: &str, mamba: &MambaConfig) -> ModelConfig {
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

/// Build a minimal `ModelConfig` for RWKV models.
pub(crate) fn minimal_config_for_rwkv(name: &str, rwkv: &RwkvConfig) -> ModelConfig {
    use crate::config::*;
    ModelConfig {
        name: name.to_string(),
        architecture: ArchitectureConfig {
            hidden_size: rwkv.hidden_size,
            num_layers: rwkv.num_layers,
            vocab_size: rwkv.vocab_size,
            max_sequence_length: 4096,
            tie_word_embeddings: false,
            embed_scale: None,
        },
        attention: AttentionConfig::GQA {
            num_heads: rwkv.num_heads,
            num_kv_heads: rwkv.num_heads,
            head_dim: rwkv.head_size,
        },
        norm: NormConfig::LayerNorm {
            eps: rwkv.norm_eps,
        },
        ffn: FFNConfig::SwiGLU {
            intermediate_size: rwkv.intermediate_size,
        },
        position: PositionConfig::RoPE {
            base: 10000.0,
            max_position_embeddings: 4096,
            style: Default::default(),
            scaling: Default::default(),
        },
        quantization: QuantConfig::F32,
    }
}

/// Parse an `RwkvConfig` from a HuggingFace-style `config.json`.
pub(crate) fn parse_rwkv_config(config_path: &Path) -> Result<RwkvConfig, Box<dyn std::error::Error>> {
    let file = std::fs::File::open(config_path)?;
    let json: serde_json::Value = serde_json::from_reader(file)?;

    let hidden_size = json["hidden_size"].as_u64().unwrap_or(768) as usize;
    let num_heads = json["num_attention_heads"]
        .as_u64()
        .or(json["num_heads"].as_u64())
        .unwrap_or(12) as usize;
    let head_size = json["head_size"]
        .as_u64()
        .unwrap_or_else(|| (hidden_size / num_heads) as u64) as usize;

    Ok(RwkvConfig {
        hidden_size,
        num_heads,
        head_size,
        num_layers: json["num_hidden_layers"]
            .as_u64()
            .or(json["num_layers"].as_u64())
            .unwrap_or(12) as usize,
        vocab_size: json["vocab_size"].as_u64().unwrap_or(50304) as usize,
        intermediate_size: json["intermediate_size"]
            .as_u64()
            .map(|v| v as usize)
            .unwrap_or_else(|| {
                let ratio = json["hidden_ratio"].as_f64().unwrap_or(4.0);
                (hidden_size as f64 * ratio) as usize
            }),
        norm_eps: json["layer_norm_epsilon"]
            .as_f64()
            .or(json["norm_eps"].as_f64())
            .or(json["norm_epsilon"].as_f64())
            .unwrap_or(1e-5),
        decay_rank: json["decay_low_rank_dim"]
            .as_u64()
            .or(json["lora_rank_decay"].as_u64())
            .unwrap_or(64) as usize,
        alpha_rank: json["a_low_rank_dim"]
            .as_u64()
            .or(json["lora_rank_iclr"].as_u64())
            .unwrap_or(64) as usize,
        gate_rank: json["gate_low_rank_dim"]
            .as_u64()
            .or(json["lora_rank_gate"].as_u64())
            .unwrap_or(128) as usize,
    })
}

/// Build a minimal `ModelConfig` for hybrid models.
pub(crate) fn minimal_config_for_hybrid(name: &str, hybrid: &HybridConfig) -> ModelConfig {
    use crate::config::*;
    ModelConfig {
        name: name.to_string(),
        architecture: ArchitectureConfig {
            hidden_size: hybrid.hidden_size,
            num_layers: hybrid.num_layers,
            vocab_size: hybrid.vocab_size,
            max_sequence_length: 4096,
            tie_word_embeddings: false,
            embed_scale: None,
        },
        attention: AttentionConfig::GQA {
            num_heads: hybrid.num_attention_heads,
            num_kv_heads: hybrid.num_kv_heads,
            head_dim: hybrid.gqa_head_dim,
        },
        norm: NormConfig::RMSNorm {
            eps: hybrid.norm_eps,
        },
        ffn: FFNConfig::SwiGLU {
            intermediate_size: hybrid.intermediate_size,
        },
        position: PositionConfig::RoPE {
            base: 10000.0,
            max_position_embeddings: 4096,
            style: Default::default(),
            scaling: Default::default(),
        },
        quantization: QuantConfig::F32,
    }
}

/// Parse the `layer_types` array from LFM2.5-style config.json.
///
/// Returns `None` if the field is absent (falls back to interval-based pattern).
fn parse_layer_types(json: &serde_json::Value) -> Option<Vec<LayerKind>> {
    let arr = json["layer_types"].as_array()?;
    let types: Vec<LayerKind> = arr
        .iter()
        .map(|v| match v.as_str().unwrap_or("") {
            "conv" => LayerKind::LivConv,
            "full_attention" | "attention" | "gqa" => LayerKind::Gqa,
            "deltanet" | "delta_net" => LayerKind::DeltaNet,
            other => panic!("unknown layer_type in config.json: {other:?}"),
        })
        .collect();
    Some(types)
}

/// Parse a `HybridConfig` from a HuggingFace-style `config.json`.
pub(crate) fn parse_hybrid_config(config_path: &Path) -> Result<HybridConfig, Box<dyn std::error::Error>> {
    let file = std::fs::File::open(config_path)?;
    let json: serde_json::Value = serde_json::from_reader(file)?;

    let hidden_size = json["hidden_size"].as_u64().unwrap_or(1024) as usize;
    let num_attention_heads = json["num_attention_heads"].as_u64().unwrap_or(16) as usize;
    let num_kv_heads = json["num_key_value_heads"]
        .as_u64()
        .or(json["num_kv_heads"].as_u64())
        .unwrap_or(8) as usize;

    let gqa_head_dim = json["head_dim"]
        .as_u64()
        .unwrap_or_else(|| (hidden_size / num_attention_heads) as u64) as usize;

    let full_attention_interval = json["full_attention_interval"]
        .as_u64()
        .unwrap_or(4) as usize;

    Ok(HybridConfig {
        hidden_size,
        num_layers: json["num_hidden_layers"].as_u64().unwrap_or(24) as usize,
        vocab_size: json["vocab_size"].as_u64().unwrap_or(151936) as usize,
        norm_eps: json["rms_norm_eps"].as_f64()
            .or(json["layer_norm_epsilon"].as_f64())
            .unwrap_or(1e-6),
        num_attention_heads,
        num_kv_heads,
        gqa_head_dim,
        deltanet_num_heads: json["deltanet_num_heads"]
            .as_u64()
            .unwrap_or(num_attention_heads as u64) as usize,
        deltanet_head_dim: json["deltanet_head_dim"]
            .as_u64()
            .unwrap_or(gqa_head_dim as u64) as usize,
        deltanet_conv_kernel: json["deltanet_conv_kernel"]
            .as_u64()
            .or(json["conv_kernel_size"].as_u64())
            .unwrap_or(4) as usize,
        livconv_inner_size: json["conv_dim"]
            .as_u64()
            .unwrap_or(hidden_size as u64) as usize,
        livconv_kernel_size: json["conv_L_cache"]
            .as_u64()
            .unwrap_or(3) as usize,
        intermediate_size: json["intermediate_size"].as_u64().unwrap_or(3072) as usize,
        full_attention_interval,
        layer_types: parse_layer_types(&json),
        rope_theta: json["rope_parameters"]["rope_theta"]
            .as_f64()
            .or(json["rope_theta"].as_f64())
            .unwrap_or(10000.0) as f32,
        tie_embedding: json["tie_embedding"].as_bool()
            .or(json["tie_word_embeddings"].as_bool())
            .unwrap_or(false),
    })
}

pub(crate) fn find_checkpoint_file(model_path: &Path) -> Result<PathBuf, WeightError> {
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
