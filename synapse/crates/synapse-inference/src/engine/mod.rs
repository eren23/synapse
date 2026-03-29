mod loading;
pub(crate) mod config_parsers;

use crate::capabilities::CapabilityReport;
use crate::chat_template::{ChatMessage, ChatTemplate};
use crate::config::ModelConfig;
use crate::generation::{GenerationConfig, GenerationOutput, GenerationPipeline};
use crate::kv_cache::KVCache;
use crate::models::traits::Model;
use crate::models::{CausalLM, ModelOutput, ModelState};
use crate::model_adapter::{
    adapter_for_kind, ModelAdapter, ModelAdapterKind, ReasoningMarkers, ThinkingMode,
};
use crate::quantization::{quantize_model, quantize_model_ternary, QuantizedCausalLM, TernaryCausalLM};
use crate::models::ssm::mamba::model::MambaModel;
use crate::tokenizer::{Tokenizer, TokenizerError};

#[cfg(feature = "metal")]
use std::cell::RefCell;

/// Which compute backend to use for dispatch.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BackendSelection {
    /// Always use CPU SIMD (Zig FFI).
    CpuSimd,
    /// Use Metal GPU when available, CPU SIMD fallback.
    #[cfg(feature = "metal")]
    Auto,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RuntimeBackend {
    CpuSimd,
    #[cfg(feature = "metal")]
    Metal,
}

impl RuntimeBackend {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::CpuSimd => "cpu_simd",
            #[cfg(feature = "metal")]
            Self::Metal => "metal",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RuntimePath {
    CpuCached,
    QuantizedCpu,
    BackendDispatch,
    MetalPerLayer,
}

impl RuntimePath {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::CpuCached => "cpu_cached",
            Self::QuantizedCpu => "quantized_cpu",
            Self::BackendDispatch => "backend_dispatch",
            Self::MetalPerLayer => "metal_per_layer",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RuntimePlan {
    pub family: &'static str,
    pub backend: RuntimeBackend,
    pub quantized: bool,
    pub prefill_path: RuntimePath,
    pub decode_path: RuntimePath,
    pub prefill_strategy: &'static str,
    pub decode_strategy: &'static str,
}

impl RuntimePlan {
    pub fn log_line(&self) -> String {
        format!(
            "Runtime: family={} backend={} quantized={} prefill={} prefill_strategy={} decode={} decode_strategy={}",
            self.family,
            self.backend.as_str(),
            self.quantized,
            self.prefill_path.as_str(),
            self.prefill_strategy,
            self.decode_path.as_str(),
            self.decode_strategy,
        )
    }
}

pub type RuntimeSummary = RuntimePlan;

/// High-level inference orchestrator.
///
/// Assembles config -> model -> weights -> KV-cache -> tokenizer into a single
/// entry point for text generation.
pub struct InferenceEngine {
    pub model: CausalLM,
    pub quantized_model: Option<QuantizedCausalLM>,
    pub ternary_model: Option<TernaryCausalLM>,
    /// Optional SSM model (Mamba, RWKV, etc.) -- takes priority over transformer models.
    pub ssm_model: Option<Box<dyn Model>>,
    pub config: ModelConfig,
    pub model_adapter_kind: ModelAdapterKind,
    pub tokenizer: Option<Tokenizer>,
    pub chat_template: Option<ChatTemplate>,
    #[cfg(feature = "metal")]
    pub backend: crate::metal::ComputeBackend,
    /// GPU-resident model buffers for Phase 3 all-layers-in-one-command-buffer.
    /// Wrapped in RefCell because generate_text borrows &self but the decode loop
    /// needs mutable access to update the GPU KV cache position.
    /// Allocated when Metal backend is active; `None` for CPU-only or quantized.
    #[cfg(feature = "metal")]
    pub metal_model_bufs_cell: Option<RefCell<crate::metal::MetalModelBuffers>>,
}

impl InferenceEngine {
    /// Run a forward pass on token ids, returning logits.
    ///
    /// When the `metal` feature is enabled, dispatches through the
    /// `ComputeBackend` (GPU for large ops, CPU for small ones).
    pub fn forward(&self, token_ids: &[u32]) -> ModelOutput {
        #[cfg(feature = "metal")]
        {
            self.model.forward_with_backend(token_ids, &self.backend)
        }
        #[cfg(not(feature = "metal"))]
        {
            self.model.forward(token_ids)
        }
    }

    /// Total parameter count of the underlying model.
    pub fn param_count(&self) -> usize {
        self.model.param_count()
    }

    pub fn encode(&self, text: &str) -> Result<Vec<u32>, TokenizerError> {
        let tokenizer = self
            .tokenizer
            .as_ref()
            .ok_or_else(|| TokenizerError::Invalid("No tokenizer loaded".into()))?;
        tokenizer.encode(text)
    }

    pub fn decode(&self, token_ids: &[u32]) -> Result<String, TokenizerError> {
        let tokenizer = self
            .tokenizer
            .as_ref()
            .ok_or_else(|| TokenizerError::Invalid("No tokenizer loaded".into()))?;
        tokenizer.decode(token_ids)
    }

    pub fn tokenizer(&self) -> Option<&Tokenizer> {
        self.tokenizer.as_ref()
    }

    /// Return the current build/runtime capability report, annotated with the loaded model name.
    pub fn capability_report(&self) -> CapabilityReport {
        CapabilityReport::for_model_name(Some(&self.config.name))
    }

    pub fn model_adapter(&self) -> &'static dyn ModelAdapter {
        adapter_for_kind(self.model_adapter_kind)
    }

    pub fn model_family(&self) -> &'static str {
        self.model_adapter().family()
    }

    pub fn default_cli_thinking_mode(&self) -> ThinkingMode {
        self.model_adapter().default_cli_thinking_mode()
    }

    pub fn reasoning_markers(&self) -> Option<ReasoningMarkers> {
        self.model_adapter().reasoning_markers()
    }

    /// Format a list of chat messages into a prompt string using the loaded
    /// chat template.  Falls back to the default Qwen3 / ChatML format if
    /// no template was loaded from `tokenizer_config.json`.
    pub fn format_chat(
        &self,
        messages: &[ChatMessage],
    ) -> Result<String, Box<dyn std::error::Error>> {
        self.format_chat_with_mode(messages, ThinkingMode::Auto)
    }

    pub fn format_chat_with_mode(
        &self,
        messages: &[ChatMessage],
        thinking_mode: ThinkingMode,
    ) -> Result<String, Box<dyn std::error::Error>> {
        self.model_adapter().format_chat_prompt(
            self.chat_template.as_ref(),
            messages,
            thinking_mode,
        )
    }

    /// Create a KV cache sized for this model's architecture.
    pub fn create_kv_cache(
        &self,
        max_seq_len: usize,
    ) -> Result<KVCache, Box<dyn std::error::Error>> {
        let cache = KVCache::new(
            self.config.architecture.num_layers,
            max_seq_len,
            self.config.attention.num_kv_heads(),
            self.config.attention.head_dim(),
        )?;
        Ok(cache)
    }

    /// Create a model state sized for this model's architecture.
    ///
    /// For transformer models this wraps a KV cache.
    /// For SSM models (Mamba, RWKV) this returns `ModelState::Recurrent`
    /// since SSMs manage their own internal state.
    pub fn create_state(
        &self,
        max_seq_len: usize,
    ) -> Result<ModelState, Box<dyn std::error::Error>> {
        if self.ssm_model.is_some() {
            // SSMs manage their own state internally via RefCell
            Ok(ModelState::Recurrent)
        } else {
            let cache = KVCache::new(
                self.config.architecture.num_layers,
                max_seq_len,
                self.config.attention.num_kv_heads(),
                self.config.attention.head_dim(),
            )?;
            Ok(ModelState::KvCache(cache))
        }
    }

    /// Quantize the model to INT8. Call after loading weights.
    ///
    /// For transformer models, creates an INT8 quantized copy.
    /// For SSM models (Mamba), replaces the SSM model with INT8 version.
    pub fn quantize(&mut self) {
        if let Some(ref ssm) = self.ssm_model {
            // Try to downcast to MambaModel for quantization
            let ssm_ptr = ssm.as_ref() as *const dyn Model;
            // Safety: we know SSM models from from_pretrained_mamba are MambaModel
            let mamba: &MambaModel = unsafe { &*(ssm_ptr as *const MambaModel) };
            let quantized = crate::quantization::QuantizedMambaModel::from_f32(mamba);
            self.ssm_model = Some(Box::new(quantized));
            return;
        }
        self.quantized_model = Some(quantize_model(&self.model));
    }

    /// Whether the engine has a quantized model available.
    pub fn is_quantized(&self) -> bool {
        self.quantized_model.is_some()
    }

    /// Quantize the model to ternary 2-bit weights. Call after loading weights.
    pub fn quantize_ternary(&mut self) {
        self.ternary_model = Some(quantize_model_ternary(&self.model));
    }

    /// Whether the engine has a ternary model available.
    pub fn is_ternary(&self) -> bool {
        self.ternary_model.is_some()
    }

    pub fn runtime_plan(&self) -> RuntimePlan {
        let quantized = self.is_quantized();
        #[cfg(feature = "metal")]
        let backend = if self.backend.is_gpu() {
            RuntimeBackend::Metal
        } else {
            RuntimeBackend::CpuSimd
        };
        #[cfg(not(feature = "metal"))]
        let backend = RuntimeBackend::CpuSimd;
        #[cfg(feature = "metal")]
        let is_metal = matches!(backend, RuntimeBackend::Metal);
        #[cfg(not(feature = "metal"))]
        let is_metal = false;

        let prefill_path = if quantized {
            RuntimePath::QuantizedCpu
        } else if is_metal {
            RuntimePath::BackendDispatch
        } else {
            RuntimePath::CpuCached
        };
        let decode_path = if quantized {
            RuntimePath::QuantizedCpu
        } else if is_metal {
            RuntimePath::MetalPerLayer
        } else {
            RuntimePath::CpuCached
        };
        let prefill_strategy = if quantized {
            "quantized_prefill_cpu"
        } else if is_metal {
            "metal_prefill_dispatch"
        } else {
            "cpu_cached_prefill"
        };
        let decode_strategy = if quantized {
            "quantized_cached_decode_cpu"
        } else if is_metal {
            "metal_decode_per_layer"
        } else {
            "cpu_cached_decode"
        };

        RuntimePlan {
            family: self.model_family(),
            backend,
            quantized,
            prefill_path,
            decode_path,
            prefill_strategy,
            decode_strategy,
        }
    }

    pub fn runtime_summary(&self) -> RuntimeSummary {
        self.runtime_plan()
    }

    /// Whether the engine has an SSM model loaded.
    pub fn is_ssm(&self) -> bool {
        self.ssm_model.is_some()
    }

    pub fn generation_pipeline(&self) -> GenerationPipeline<'_> {
        if let Some(ref ssm) = self.ssm_model {
            GenerationPipeline::new(ssm.as_ref())
        } else if let Some(ref tmodel) = self.ternary_model {
            GenerationPipeline::new(tmodel)
        } else if let Some(ref qmodel) = self.quantized_model {
            GenerationPipeline::new(qmodel)
        } else {
            #[cfg(feature = "metal")]
            {
                GenerationPipeline::with_backend(&self.model, &self.backend)
            }
            #[cfg(not(feature = "metal"))]
            {
                GenerationPipeline::new(&self.model)
            }
        }
    }

    pub fn generate_text(
        &self,
        prompt: &str,
        config: GenerationConfig,
    ) -> Result<GenerationOutput, Box<dyn std::error::Error>> {
        let prompt_tokens = self.encode(prompt)?;
        let max_seq = prompt_tokens.len() + config.max_new_tokens;
        let mut state = self.create_state(max_seq)?;
        let pipeline = self.generation_pipeline();
        let mut output = pipeline.generate(&prompt_tokens, config, Some(&mut state));
        let generated = &output.token_ids[output.num_prompt_tokens..];
        output.text = self.decode(generated)?;
        Ok(output)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::config_parsers::minimal_config_for_ssm;
    use crate::config::*;
    use crate::models::ssm::mamba::config::MambaConfig;

    fn test_config() -> ModelConfig {
        ModelConfig {
            name: "InferenceEngineRuntimeSummaryTest".to_string(),
            architecture: ArchitectureConfig {
                hidden_size: 64,
                num_layers: 2,
                vocab_size: 128,
                max_sequence_length: 64,
                tie_word_embeddings: true,
                embed_scale: None,
            },
            attention: AttentionConfig::GQA {
                num_heads: 4,
                num_kv_heads: 2,
                head_dim: 16,
            },
            norm: NormConfig::RMSNorm { eps: 1e-6 },
            ffn: FFNConfig::SwiGLU {
                intermediate_size: 128,
            },
            position: PositionConfig::RoPE {
                base: 10000.0,
                max_position_embeddings: 64,
                style: Default::default(),
                scaling: Default::default(),
            },
            quantization: QuantConfig::F32,
        }
    }

    #[test]
    fn runtime_summary_reports_cpu_cached_path() {
        let engine = InferenceEngine::from_config(test_config());
        let summary = engine.runtime_summary();

        assert_eq!(summary.quantized, false);
        assert_eq!(summary.family, "generic");
        // Backend depends on feature flags
        #[cfg(feature = "metal")]
        assert_eq!(summary.backend, RuntimeBackend::Metal);
        #[cfg(not(feature = "metal"))]
        assert_eq!(summary.backend, RuntimeBackend::CpuSimd);
    }

    #[cfg(feature = "metal")]
    #[test]
    fn runtime_summary_reports_cpu_backend_selection() {
        let engine =
            InferenceEngine::from_config_with_backend(test_config(), BackendSelection::CpuSimd);
        let summary = engine.runtime_summary();

        assert_eq!(summary.backend, RuntimeBackend::CpuSimd);
        assert_eq!(summary.prefill_path, RuntimePath::CpuCached);
        assert_eq!(summary.decode_path, RuntimePath::CpuCached);
        assert_eq!(summary.prefill_strategy, "cpu_cached_prefill");
        assert_eq!(summary.decode_strategy, "cpu_cached_decode");
    }

    #[test]
    fn test_engine_quantize_ternary() {
        use std::collections::HashMap;
        use crate::weight_loading::{AlignedBuffer, RawTensor, WeightMapper};

        let config = test_config();

        // Generate minimal fake weights so quantize_ternary has non-empty matrices.
        let h = config.architecture.hidden_size;
        let vocab = config.architecture.vocab_size;
        let q_dim = config.attention.num_heads() * config.attention.head_dim();
        let kv_dim = config.attention.num_kv_heads() * config.attention.head_dim();
        let inter = config.ffn.intermediate_size();
        let nl = config.architecture.num_layers;

        let fake = |shape: Vec<usize>| -> RawTensor {
            let n: usize = shape.iter().product();
            RawTensor {
                data: AlignedBuffer::from_slice(&vec![0.01f32; n]),
                shape,
            }
        };

        let mut weights: HashMap<String, RawTensor> = HashMap::new();
        weights.insert("model.embed_tokens.weight".into(), fake(vec![vocab, h]));
        weights.insert("model.norm.weight".into(), fake(vec![h]));
        weights.insert("lm_head.weight".into(), fake(vec![vocab, h]));
        for i in 0..nl {
            weights.insert(format!("model.layers.{i}.input_layernorm.weight"), fake(vec![h]));
            weights.insert(format!("model.layers.{i}.self_attn.q_proj.weight"), fake(vec![q_dim, h]));
            weights.insert(format!("model.layers.{i}.self_attn.k_proj.weight"), fake(vec![kv_dim, h]));
            weights.insert(format!("model.layers.{i}.self_attn.v_proj.weight"), fake(vec![kv_dim, h]));
            weights.insert(format!("model.layers.{i}.self_attn.o_proj.weight"), fake(vec![h, q_dim]));
            weights.insert(format!("model.layers.{i}.post_attention_layernorm.weight"), fake(vec![h]));
            weights.insert(format!("model.layers.{i}.mlp.gate_proj.weight"), fake(vec![inter, h]));
            weights.insert(format!("model.layers.{i}.mlp.up_proj.weight"), fake(vec![inter, h]));
            weights.insert(format!("model.layers.{i}.mlp.down_proj.weight"), fake(vec![h, inter]));
        }

        let mut engine = InferenceEngine::from_config(config);
        let mapper = WeightMapper::from_model_type("InferenceEngineRuntimeSummaryTest")
            .unwrap_or_else(|_| WeightMapper::llama());
        let _ = engine.model.load_weights(weights, &mapper);

        assert!(!engine.is_ternary());
        engine.quantize_ternary();
        assert!(engine.is_ternary());
        let _pipeline = engine.generation_pipeline();
    }

    #[test]
    fn qwen3_config_selects_qwen3_adapter() {
        let mut config = test_config();
        config.name = "qwen3".into();
        let engine = InferenceEngine::from_config(config);
        assert_eq!(engine.model_adapter_kind, ModelAdapterKind::Qwen3);
        assert_eq!(engine.model_family(), "qwen3");
        assert_eq!(
            engine.reasoning_markers(),
            Some(ReasoningMarkers {
                start: "<think>",
                end: "</think>",
            })
        );
    }

    // -- SSM / Mamba engine tests --

    fn make_tiny_mamba_model() -> MambaModel {
        use crate::models::ssm::mamba::block::MambaBlock;

        let config = MambaConfig::tiny_test();
        let d_model = config.d_model;
        let d_inner = config.d_inner();
        let d_state = config.d_state;
        let d_conv = config.d_conv;
        let dt_rank = config.dt_rank;
        let vocab = config.vocab_size;

        let pseudo = |seed: u64, len: usize| -> Vec<f32> {
            let mut state = seed;
            (0..len)
                .map(|_| {
                    state = state
                        .wrapping_mul(6364136223846793005)
                        .wrapping_add(1442695040888963407);
                    let bits = 0x3F800000u32 | ((state >> 41) as u32 & 0x7FFFFF);
                    (f32::from_bits(bits) - 1.5) * 0.2
                })
                .collect()
        };

        let embed_tokens = pseudo(100, vocab * d_model);
        let final_norm_weight = vec![1.0f32; d_model];
        let lm_head_weight = pseudo(200, vocab * d_model);

        let mut blocks = Vec::new();
        for layer_idx in 0..config.num_layers {
            let s = (layer_idx as u64 + 1) * 1000;
            blocks.push(MambaBlock {
                d_model,
                d_inner,
                d_state,
                d_conv,
                dt_rank,
                norm_weight: vec![1.0f32; d_model],
                norm_eps: config.norm_eps as f32,
                in_proj_weight: pseudo(s + 1, 2 * d_inner * d_model),
                in_proj_bias: vec![],
                conv1d_weight: pseudo(s + 2, d_inner * d_conv),
                conv1d_bias: vec![0.0f32; d_inner],
                x_proj_weight: pseudo(s + 3, (dt_rank + 2 * d_state) * d_inner),
                dt_proj_weight: pseudo(s + 4, d_inner * dt_rank),
                dt_proj_bias: vec![0.0f32; d_inner],
                a_log: pseudo(s + 5, d_inner * d_state)
                    .into_iter()
                    .map(|v| -v.abs() - 0.1)
                    .collect(),
                d_param: vec![1.0f32; d_inner],
                out_proj_weight: pseudo(s + 6, d_model * d_inner),
                out_proj_bias: vec![],
            });
        }

        MambaModel::new(config, embed_tokens, blocks, final_norm_weight, lm_head_weight)
    }

    #[test]
    fn test_engine_with_ssm_model() {
        let mamba = make_tiny_mamba_model();
        let mamba_config = mamba.config.clone();

        let config = minimal_config_for_ssm("mamba", &mamba_config);
        let mut engine = InferenceEngine::from_config(config);
        engine.ssm_model = Some(Box::new(mamba));

        assert!(engine.is_ssm());

        // generation_pipeline should use the SSM model
        let pipeline = engine.generation_pipeline();
        let output = pipeline.generate(
            &[1, 2, 3],
            crate::generation::GenerationConfig {
                max_new_tokens: 4,
                ..Default::default()
            },
            Some(&mut ModelState::Recurrent),
        );
        assert!(
            output.token_ids.len() > 3,
            "SSM engine should generate tokens beyond the prompt"
        );
    }

    #[test]
    fn test_engine_ssm_create_state_returns_recurrent() {
        let mamba = make_tiny_mamba_model();
        let mamba_config = mamba.config.clone();

        let config = minimal_config_for_ssm("mamba", &mamba_config);
        let mut engine = InferenceEngine::from_config(config);
        engine.ssm_model = Some(Box::new(mamba));

        let state = engine.create_state(128).expect("create_state should succeed");
        assert!(
            matches!(state, ModelState::Recurrent),
            "SSM engine state should be Recurrent, got {:?}",
            std::mem::discriminant(&state),
        );
    }

    #[test]
    fn test_engine_transformer_create_state_returns_kv_cache() {
        let engine = InferenceEngine::from_config(test_config());
        assert!(!engine.is_ssm());
        let state = engine.create_state(64).expect("create_state should succeed");
        assert!(
            matches!(state, ModelState::KvCache(_)),
            "Transformer engine state should be KvCache"
        );
    }
}
