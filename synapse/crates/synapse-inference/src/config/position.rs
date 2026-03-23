use serde::{Deserialize, Serialize};

/// Configuration for the positional encoding variant.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "type")]
pub enum PositionConfig {
    RoPE {
        base: f64,
        max_position_embeddings: usize,
    },
    Learned {
        max_position_embeddings: usize,
    },
    Sinusoidal {
        max_position_embeddings: usize,
    },
}

impl PositionConfig {
    pub fn max_position_embeddings(&self) -> usize {
        match self {
            Self::RoPE {
                max_position_embeddings,
                ..
            }
            | Self::Learned {
                max_position_embeddings,
            }
            | Self::Sinusoidal {
                max_position_embeddings,
            } => *max_position_embeddings,
        }
    }
}
