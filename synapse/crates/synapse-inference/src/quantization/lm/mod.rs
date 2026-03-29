pub mod int8;
pub mod ternary;

pub use int8::{f32_model_memory_bytes, quantize_model, QuantizedCausalLM, QuantizedDecoderLayer};
pub use ternary::{quantize_model_ternary, TernaryCausalLM, TernaryDecoderLayer};
