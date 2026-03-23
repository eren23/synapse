use serde::{Deserialize, Serialize};

use super::attention::AttentionConfig;
use super::ffn::FFNConfig;
use super::norm::NormConfig;
use super::position::PositionConfig;
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

    /// Serialize this config to a JSON string.
    pub fn to_json(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string_pretty(self)
    }
}
