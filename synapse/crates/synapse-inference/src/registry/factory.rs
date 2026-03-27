use super::{AttentionVariant, FFNVariant, NormVariant, PositionVariant};
use crate::config::{AttentionConfig, FFNConfig, NormConfig, PositionConfig};

// ── Attention concrete types ────────────────────────────────────────

#[derive(Debug)]
struct GQAAttention {
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
}

#[derive(Debug)]
struct MHAAttention {
    num_heads: usize,
    head_dim: usize,
}

#[derive(Debug)]
struct MQAAttention {
    num_heads: usize,
    head_dim: usize,
}

#[derive(Debug)]
struct SlidingWindowAttention {
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    window_size: usize,
}

#[derive(Debug)]
struct BidirectionalAttention {
    num_heads: usize,
    head_dim: usize,
}

impl AttentionVariant for GQAAttention {
    fn num_heads(&self) -> usize {
        self.num_heads
    }
    fn head_dim(&self) -> usize {
        self.head_dim
    }
    fn num_kv_heads(&self) -> usize {
        self.num_kv_heads
    }
    fn name(&self) -> &str {
        "GQA"
    }
}

impl AttentionVariant for MHAAttention {
    fn num_heads(&self) -> usize {
        self.num_heads
    }
    fn head_dim(&self) -> usize {
        self.head_dim
    }
    fn num_kv_heads(&self) -> usize {
        self.num_heads
    }
    fn name(&self) -> &str {
        "MHA"
    }
}

impl AttentionVariant for MQAAttention {
    fn num_heads(&self) -> usize {
        self.num_heads
    }
    fn head_dim(&self) -> usize {
        self.head_dim
    }
    fn num_kv_heads(&self) -> usize {
        1
    }
    fn name(&self) -> &str {
        "MQA"
    }
}

impl AttentionVariant for SlidingWindowAttention {
    fn num_heads(&self) -> usize {
        self.num_heads
    }
    fn head_dim(&self) -> usize {
        self.head_dim
    }
    fn num_kv_heads(&self) -> usize {
        self.num_kv_heads
    }
    fn window_size(&self) -> Option<usize> {
        Some(self.window_size)
    }
    fn name(&self) -> &str {
        "SlidingWindow"
    }
}

impl AttentionVariant for BidirectionalAttention {
    fn num_heads(&self) -> usize {
        self.num_heads
    }
    fn head_dim(&self) -> usize {
        self.head_dim
    }
    fn num_kv_heads(&self) -> usize {
        self.num_heads
    }
    fn name(&self) -> &str {
        "Bidirectional"
    }
}

/// Create an attention variant trait object from config.
pub fn create_attention(config: &AttentionConfig) -> Box<dyn AttentionVariant> {
    match config {
        AttentionConfig::GQA {
            num_heads,
            num_kv_heads,
            head_dim,
        } => Box::new(GQAAttention {
            num_heads: *num_heads,
            num_kv_heads: *num_kv_heads,
            head_dim: *head_dim,
        }),
        AttentionConfig::MHA {
            num_heads,
            head_dim,
        } => Box::new(MHAAttention {
            num_heads: *num_heads,
            head_dim: *head_dim,
        }),
        AttentionConfig::MQA {
            num_heads,
            head_dim,
        } => Box::new(MQAAttention {
            num_heads: *num_heads,
            head_dim: *head_dim,
        }),
        AttentionConfig::SlidingWindow {
            num_heads,
            num_kv_heads,
            head_dim,
            window_size,
        } => Box::new(SlidingWindowAttention {
            num_heads: *num_heads,
            num_kv_heads: *num_kv_heads,
            head_dim: *head_dim,
            window_size: *window_size,
        }),
        AttentionConfig::Bidirectional {
            num_heads,
            head_dim,
        } => Box::new(BidirectionalAttention {
            num_heads: *num_heads,
            head_dim: *head_dim,
        }),
    }
}

// ── Norm concrete types ─────────────────────────────────────────────

#[derive(Debug)]
struct RMSNorm {
    eps: f64,
}

#[derive(Debug)]
struct LayerNorm {
    eps: f64,
}

impl NormVariant for RMSNorm {
    fn eps(&self) -> f64 {
        self.eps
    }
    fn name(&self) -> &str {
        "RMSNorm"
    }
}

impl NormVariant for LayerNorm {
    fn eps(&self) -> f64 {
        self.eps
    }
    fn name(&self) -> &str {
        "LayerNorm"
    }
}

/// Create a norm variant trait object from config.
pub fn create_norm(config: &NormConfig) -> Box<dyn NormVariant> {
    match config {
        NormConfig::RMSNorm { eps } => Box::new(RMSNorm { eps: *eps }),
        NormConfig::LayerNorm { eps } => Box::new(LayerNorm { eps: *eps }),
    }
}

// ── FFN concrete types ──────────────────────────────────────────────

#[derive(Debug)]
struct SwiGLUFFN {
    intermediate_size: usize,
}

#[derive(Debug)]
struct GELUFFN {
    intermediate_size: usize,
}

#[derive(Debug)]
struct GeGLUFFN {
    intermediate_size: usize,
}

impl FFNVariant for SwiGLUFFN {
    fn intermediate_size(&self) -> usize {
        self.intermediate_size
    }
    fn name(&self) -> &str {
        "SwiGLU"
    }
}

impl FFNVariant for GELUFFN {
    fn intermediate_size(&self) -> usize {
        self.intermediate_size
    }
    fn name(&self) -> &str {
        "GELU"
    }
}

impl FFNVariant for GeGLUFFN {
    fn intermediate_size(&self) -> usize {
        self.intermediate_size
    }
    fn name(&self) -> &str {
        "GeGLU"
    }
}

/// Create an FFN variant trait object from config.
pub fn create_ffn(config: &FFNConfig) -> Box<dyn FFNVariant> {
    match config {
        FFNConfig::SwiGLU { intermediate_size } => Box::new(SwiGLUFFN {
            intermediate_size: *intermediate_size,
        }),
        FFNConfig::GELU { intermediate_size } => Box::new(GELUFFN {
            intermediate_size: *intermediate_size,
        }),
        FFNConfig::GeGLU { intermediate_size } => Box::new(GeGLUFFN {
            intermediate_size: *intermediate_size,
        }),
    }
}

// ── Position concrete types ─────────────────────────────────────────

#[derive(Debug)]
struct RoPEPosition {
    base: f64,
    max_position_embeddings: usize,
}

#[derive(Debug)]
struct LearnedPosition {
    max_position_embeddings: usize,
}

#[derive(Debug)]
struct SinusoidalPosition {
    max_position_embeddings: usize,
}

impl PositionVariant for RoPEPosition {
    fn max_position_embeddings(&self) -> usize {
        self.max_position_embeddings
    }
    fn base(&self) -> Option<f64> {
        Some(self.base)
    }
    fn name(&self) -> &str {
        "RoPE"
    }
}

impl PositionVariant for LearnedPosition {
    fn max_position_embeddings(&self) -> usize {
        self.max_position_embeddings
    }
    fn name(&self) -> &str {
        "Learned"
    }
}

impl PositionVariant for SinusoidalPosition {
    fn max_position_embeddings(&self) -> usize {
        self.max_position_embeddings
    }
    fn name(&self) -> &str {
        "Sinusoidal"
    }
}

/// Create a position variant trait object from config.
pub fn create_position(config: &PositionConfig) -> Box<dyn PositionVariant> {
    match config {
        PositionConfig::RoPE {
            base,
            max_position_embeddings,
            ..
        } => Box::new(RoPEPosition {
            base: *base,
            max_position_embeddings: *max_position_embeddings,
        }),
        PositionConfig::Learned {
            max_position_embeddings,
        } => Box::new(LearnedPosition {
            max_position_embeddings: *max_position_embeddings,
        }),
        PositionConfig::Sinusoidal {
            max_position_embeddings,
        } => Box::new(SinusoidalPosition {
            max_position_embeddings: *max_position_embeddings,
        }),
    }
}
