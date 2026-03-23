pub mod converter;
pub mod gguf;
pub mod safetensors;
pub mod weight_map;

pub use converter::{bf16_to_f32, f16_to_f32, transpose};
pub use gguf::{load_gguf, parse_gguf};
pub use safetensors::{load_safetensors, parse_safetensors};
pub use weight_map::WeightMapper;

use std::fmt;

/// Raw tensor data before wrapping in a synapse-core Tensor.
#[derive(Debug, Clone)]
pub struct RawTensor {
    pub data: Vec<f32>,
    pub shape: Vec<usize>,
}

/// Errors from weight loading operations.
#[derive(Debug)]
pub enum WeightError {
    Io(std::io::Error),
    InvalidFormat(String),
    UnsupportedDtype(String),
    ShapeMismatch(String),
    MissingKeys(Vec<String>),
    UnexpectedKeys(Vec<String>),
    TensorError(synapse_core::SynapseError),
}

impl fmt::Display for WeightError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            WeightError::Io(e) => write!(f, "IO error: {e}"),
            WeightError::InvalidFormat(msg) => write!(f, "Invalid format: {msg}"),
            WeightError::UnsupportedDtype(dtype) => write!(f, "Unsupported dtype: {dtype}"),
            WeightError::ShapeMismatch(msg) => write!(f, "Shape mismatch: {msg}"),
            WeightError::MissingKeys(keys) => write!(f, "Missing keys: {keys:?}"),
            WeightError::UnexpectedKeys(keys) => write!(f, "Unexpected keys: {keys:?}"),
            WeightError::TensorError(e) => write!(f, "Tensor error: {e}"),
        }
    }
}

impl std::error::Error for WeightError {}
