pub mod capabilities;
#[cfg(not(target_os = "espidf"))]
pub mod chat_template;
pub mod config;
#[cfg(feature = "diffusion")]
pub mod diffusion;
#[cfg(not(target_os = "espidf"))]
pub mod engine;
pub mod generation;
pub mod kv_cache;
#[cfg(feature = "metal")]
pub mod metal;
pub mod models;
#[cfg(not(target_os = "espidf"))]
pub mod model_adapter;
pub mod ops;
pub mod pruning;
pub mod quantization;
pub mod registry;
pub mod tokenizer;
pub mod weight_loading;

pub mod prelude {
    pub use crate::capabilities::{
        ArtifactBudget, CapabilityReport, FeatureStatus, ModelProfile, ModelSupportLevel,
        NativeKernelInfo, RuntimeProfile, SupportLevel,
    };
    #[cfg(not(target_os = "espidf"))]
    pub use crate::chat_template::{ChatMessage, ChatTemplate, ChatTemplateOptions};
    pub use crate::config::{
        ArchitectureConfig, AttentionConfig, FFNConfig, ModelConfig, NormConfig, PositionConfig,
        QuantConfig,
    };
    #[cfg(not(target_os = "espidf"))]
    pub use crate::engine::{InferenceEngine, RuntimePlan, RuntimeSummary};
    pub use crate::generation::{
        CombinedSampler, GenerationConfig, GenerationOutput, GenerationPipeline, GreedySampler,
        RepetitionPenalty, Sampler, StopChecker, StopCondition, TemperatureSampler, TopKSampler,
        TopPSampler,
    };
    pub use crate::models::{CausalLM, DecoderLayer, LoadResult, Model, ModelBuilder, ModelOutput, ModelState};
    #[cfg(not(target_os = "espidf"))]
    pub use crate::model_adapter::{
        ModelAdapter, ModelAdapterKind, ReasoningMarkers, ThinkingMode,
    };
    pub use crate::quantization::{
        f32_model_memory_bytes, quantize_model, MinMaxCalibration, PercentileCalibration,
        QuantizedCausalLM, QuantizedDecoderLayer, QuantizedLinear,
    };
    pub use crate::registry::{
        create_attention, create_ffn, create_norm, create_position, AttentionVariant, FFNVariant,
        NormVariant, PositionVariant,
    };
    pub use crate::tokenizer::{Tokenizer, TokenizerError};
}

#[cfg(test)]
#[path = "lib_tests.rs"]
mod lib_tests;
