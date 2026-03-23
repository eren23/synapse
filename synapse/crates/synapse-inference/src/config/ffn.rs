use serde::{Deserialize, Serialize};

/// Configuration for the feed-forward network variant.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "type")]
pub enum FFNConfig {
    SwiGLU { intermediate_size: usize },
    GELU { intermediate_size: usize },
    GeGLU { intermediate_size: usize },
}

impl FFNConfig {
    pub fn intermediate_size(&self) -> usize {
        match self {
            Self::SwiGLU { intermediate_size }
            | Self::GELU { intermediate_size }
            | Self::GeGLU { intermediate_size } => *intermediate_size,
        }
    }
}
