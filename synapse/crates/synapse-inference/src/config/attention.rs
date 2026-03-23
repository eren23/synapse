use serde::{Deserialize, Serialize};

/// Configuration for the attention mechanism variant.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "type")]
pub enum AttentionConfig {
    /// Grouped-Query Attention: fewer KV heads than query heads.
    GQA {
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
    },
    /// Multi-Head Attention: equal query and KV heads.
    MHA {
        num_heads: usize,
        head_dim: usize,
    },
    /// Multi-Query Attention: single KV head shared across all query heads.
    MQA {
        num_heads: usize,
        head_dim: usize,
    },
    /// Sliding-window attention with a fixed context window.
    SlidingWindow {
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        window_size: usize,
    },
}

impl AttentionConfig {
    pub fn num_heads(&self) -> usize {
        match self {
            Self::GQA { num_heads, .. }
            | Self::MHA { num_heads, .. }
            | Self::MQA { num_heads, .. }
            | Self::SlidingWindow { num_heads, .. } => *num_heads,
        }
    }

    pub fn head_dim(&self) -> usize {
        match self {
            Self::GQA { head_dim, .. }
            | Self::MHA { head_dim, .. }
            | Self::MQA { head_dim, .. }
            | Self::SlidingWindow { head_dim, .. } => *head_dim,
        }
    }

    pub fn num_kv_heads(&self) -> usize {
        match self {
            Self::GQA { num_kv_heads, .. } | Self::SlidingWindow { num_kv_heads, .. } => {
                *num_kv_heads
            }
            Self::MHA { num_heads, .. } => *num_heads,
            Self::MQA { .. } => 1,
        }
    }
}
