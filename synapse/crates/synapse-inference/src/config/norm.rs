use serde::{Deserialize, Serialize};

/// Configuration for the normalization layer variant.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "type")]
pub enum NormConfig {
    RMSNorm { eps: f64 },
    LayerNorm { eps: f64 },
}

impl NormConfig {
    pub fn eps(&self) -> f64 {
        match self {
            Self::RMSNorm { eps } | Self::LayerNorm { eps } => *eps,
        }
    }
}
