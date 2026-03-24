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

/// RoPE frequency scaling for extended context models.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub enum RoPEScaling {
    /// No scaling (default for base context models).
    None,
    /// Linear scaling: divide frequencies by factor. Used by LLaMA 3.1/3.2.
    Linear { factor: f64 },
    /// Dynamic NTK scaling: adjust base by factor. Used by some CodeLlama variants.
    Dynamic { factor: f64 },
}

impl Default for RoPEScaling {
    fn default() -> Self {
        Self::None
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
        #[serde(default)]
        scaling: RoPEScaling,
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
