use serde::{Deserialize, Serialize};
use std::fs;
use std::path::Path;

use super::attention::AttentionConfig;
use super::ffn::FFNConfig;
use super::norm::NormConfig;
use super::position::{self, PositionConfig};
use super::quantization::QuantConfig;

/// Top-level model configuration, deserializable from JSON.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ModelConfig {
    pub name: String,
    pub architecture: ArchitectureConfig,
    pub attention: AttentionConfig,
    pub norm: NormConfig,
    pub ffn: FFNConfig,
    pub position: PositionConfig,
    pub quantization: QuantConfig,
}

/// Core architecture hyperparameters.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ArchitectureConfig {
    pub hidden_size: usize,
    pub num_layers: usize,
    pub vocab_size: usize,
    pub max_sequence_length: usize,
    pub tie_word_embeddings: bool,
}

impl ModelConfig {
    /// Parse a `ModelConfig` from a JSON string.
    pub fn from_json(json: &str) -> Result<Self, serde_json::Error> {
        serde_json::from_str(json)
    }

    /// Parse a Hugging Face config JSON into Synapse's internal model config.
    pub fn from_hf_json(json: &str) -> Result<Self, serde_json::Error> {
        let cfg: HuggingFaceConfig = serde_json::from_str(json)?;
        Ok(cfg.into_model_config())
    }

    pub fn from_hf_file(path: &Path) -> Result<Self, Box<dyn std::error::Error>> {
        let json = fs::read_to_string(path)?;
        Ok(Self::from_hf_json(&json)?)
    }

    /// Serialize this config to a JSON string.
    pub fn to_json(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string_pretty(self)
    }
}

#[derive(Debug, Deserialize)]
struct HuggingFaceConfig {
    #[serde(default)]
    model_type: String,
    #[serde(default)]
    hidden_act: String,
    hidden_size: usize,
    intermediate_size: usize,
    max_position_embeddings: usize,
    num_attention_heads: usize,
    num_hidden_layers: usize,
    #[serde(default)]
    num_key_value_heads: Option<usize>,
    #[serde(default)]
    head_dim: Option<usize>,
    #[serde(default)]
    rms_norm_eps: Option<f64>,
    #[serde(default)]
    rope_theta: Option<f64>,
    #[serde(default)]
    tie_word_embeddings: bool,
    vocab_size: usize,
    #[serde(default)]
    torch_dtype: Option<String>,
    #[serde(default)]
    sliding_window: Option<usize>,
    #[serde(default)]
    use_sliding_window: bool,
    #[serde(default)]
    rope_scaling: Option<HfRoPEScaling>,
}

#[derive(Debug, Deserialize)]
struct HfRoPEScaling {
    #[serde(rename = "type", alias = "rope_type")]
    scaling_type: String,
    #[serde(default = "default_scaling_factor")]
    factor: f64,
}

fn default_scaling_factor() -> f64 {
    1.0
}

impl HuggingFaceConfig {
    fn into_model_config(self) -> ModelConfig {
        let num_kv_heads = self.num_key_value_heads.unwrap_or(self.num_attention_heads);
        let head_dim = self
            .head_dim
            .unwrap_or(self.hidden_size / self.num_attention_heads);
        let name = if self.model_type.is_empty() {
            "HuggingFaceModel".to_string()
        } else {
            self.model_type.clone()
        };

        let attention = if self.use_sliding_window {
            AttentionConfig::SlidingWindow {
                num_heads: self.num_attention_heads,
                num_kv_heads,
                head_dim,
                window_size: self.sliding_window.unwrap_or(self.max_position_embeddings),
            }
        } else if num_kv_heads == self.num_attention_heads {
            AttentionConfig::MHA {
                num_heads: self.num_attention_heads,
                head_dim,
            }
        } else if num_kv_heads == 1 {
            AttentionConfig::MQA {
                num_heads: self.num_attention_heads,
                head_dim,
            }
        } else {
            AttentionConfig::GQA {
                num_heads: self.num_attention_heads,
                num_kv_heads,
                head_dim,
            }
        };

        let ffn = match self.hidden_act.as_str() {
            // CODEx: Qwen checkpoints report `hidden_act="silu"` but use a gated
            // MLP layout (`gate_proj`, `up_proj`, `down_proj`), so we map to SwiGLU.
            "gelu" => FFNConfig::GELU {
                intermediate_size: self.intermediate_size,
            },
            "geglu" => FFNConfig::GeGLU {
                intermediate_size: self.intermediate_size,
            },
            _ => FFNConfig::SwiGLU {
                intermediate_size: self.intermediate_size,
            },
        };

        let quantization = match self.torch_dtype.as_deref() {
            Some("float16") | Some("half") => QuantConfig::F16,
            _ => QuantConfig::F32,
        };

        ModelConfig {
            name,
            architecture: ArchitectureConfig {
                hidden_size: self.hidden_size,
                num_layers: self.num_hidden_layers,
                vocab_size: self.vocab_size,
                max_sequence_length: self.max_position_embeddings,
                tie_word_embeddings: self.tie_word_embeddings,
            },
            attention,
            norm: NormConfig::RMSNorm {
                eps: self.rms_norm_eps.unwrap_or(1e-6),
            },
            ffn,
            position: PositionConfig::RoPE {
                base: self.rope_theta.unwrap_or(10_000.0),
                max_position_embeddings: self.max_position_embeddings,
                style: Default::default(),
                scaling: match &self.rope_scaling {
                    Some(rs) => match rs.scaling_type.as_str() {
                        "linear" => position::RoPEScaling::Linear { factor: rs.factor },
                        "dynamic" => position::RoPEScaling::Dynamic { factor: rs.factor },
                        _ => position::RoPEScaling::None,
                    },
                    None => position::RoPEScaling::None,
                },
            },
            quantization,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_hf_qwen_config() {
        let json = r#"{
            "model_type": "qwen3",
            "hidden_act": "silu",
            "hidden_size": 1024,
            "intermediate_size": 3072,
            "max_position_embeddings": 40960,
            "num_attention_heads": 16,
            "num_hidden_layers": 28,
            "num_key_value_heads": 8,
            "head_dim": 128,
            "rms_norm_eps": 1e-6,
            "rope_theta": 1000000.0,
            "tie_word_embeddings": true,
            "torch_dtype": "bfloat16",
            "vocab_size": 151936
        }"#;

        let cfg = ModelConfig::from_hf_json(json).unwrap();
        assert_eq!(cfg.name, "qwen3");
        assert_eq!(cfg.architecture.hidden_size, 1024);
        assert_eq!(cfg.architecture.max_sequence_length, 40960);
        assert!(cfg.architecture.tie_word_embeddings);
        assert_eq!(
            cfg.attention,
            AttentionConfig::GQA {
                num_heads: 16,
                num_kv_heads: 8,
                head_dim: 128,
            }
        );
        assert_eq!(
            cfg.ffn,
            FFNConfig::SwiGLU {
                intermediate_size: 3072,
            }
        );
        assert_eq!(
            cfg.position,
            PositionConfig::RoPE {
                base: 1_000_000.0,
                max_position_embeddings: 40960,
                style: Default::default(),
                scaling: Default::default(),
            }
        );
        assert_eq!(cfg.quantization, QuantConfig::F32);
    }

    #[test]
    fn parse_hf_llama_with_rope_scaling() {
        let json = r#"{
            "model_type": "llama",
            "hidden_act": "silu",
            "hidden_size": 2048,
            "intermediate_size": 8192,
            "max_position_embeddings": 131072,
            "num_attention_heads": 32,
            "num_hidden_layers": 16,
            "num_key_value_heads": 8,
            "rms_norm_eps": 1e-5,
            "rope_theta": 500000.0,
            "rope_scaling": {
                "type": "linear",
                "factor": 8.0
            },
            "tie_word_embeddings": true,
            "vocab_size": 128256
        }"#;

        let cfg = ModelConfig::from_hf_json(json).unwrap();
        assert_eq!(cfg.name, "llama");
        match cfg.position {
            PositionConfig::RoPE { base, scaling, .. } => {
                assert_eq!(base, 500000.0);
                match scaling {
                    position::RoPEScaling::Linear { factor } => assert_eq!(factor, 8.0),
                    other => panic!("Expected Linear scaling, got {other:?}"),
                }
            }
            _ => panic!("Expected RoPE position config"),
        }
    }
}
