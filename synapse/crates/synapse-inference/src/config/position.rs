use serde::{Deserialize, Serialize};

/// How RoPE pairs dimensions for rotation.
///
/// - **RotateHalf**: pairs `(i, i + d/2)` — used by Qwen3, LLaMA 3, Mistral.
/// - **Interleaved**: pairs `(2i, 2i + 1)` — used by GPT-NeoX, some older models.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum RoPEStyle {
    RotateHalf,
    Interleaved,
}

impl Default for RoPEStyle {
    fn default() -> Self {
        Self::RotateHalf
    }
}

/// Configuration for the positional encoding variant.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "type")]
pub enum PositionConfig {
    RoPE {
        base: f64,
        max_position_embeddings: usize,
        #[serde(default)]
        style: RoPEStyle,
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
