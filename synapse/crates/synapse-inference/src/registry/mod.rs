pub mod attention;
pub mod factory;
pub mod ffn;
pub mod norm;
pub mod position;

use std::fmt::Debug;

/// Trait for attention mechanism variants instantiated from config.
pub trait AttentionVariant: Send + Sync + Debug {
    fn num_heads(&self) -> usize;
    fn head_dim(&self) -> usize;
    fn num_kv_heads(&self) -> usize;
    fn window_size(&self) -> Option<usize> { None }
    fn name(&self) -> &str;
}

/// Trait for normalization layer variants instantiated from config.
pub trait NormVariant: Send + Sync + Debug {
    fn eps(&self) -> f64;
    fn name(&self) -> &str;
}

/// Trait for feed-forward network variants instantiated from config.
pub trait FFNVariant: Send + Sync + Debug {
    fn intermediate_size(&self) -> usize;
    fn name(&self) -> &str;
}

/// Trait for positional encoding variants instantiated from config.
pub trait PositionVariant: Send + Sync + Debug {
    fn max_position_embeddings(&self) -> usize;
    fn base(&self) -> Option<f64> { None }
    fn name(&self) -> &str;
}

pub use factory::{
    create_attention, create_ffn, create_norm, create_position,
};
