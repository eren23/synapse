pub mod config;
pub mod engine;
pub mod generation;
pub mod kv_cache;
#[cfg(feature = "metal")]
pub mod metal;
pub mod model;
pub mod quantization;
pub mod registry;
pub mod tokenizer;
pub mod weight_loading;

pub mod prelude {
    pub use crate::config::{
        ArchitectureConfig, AttentionConfig, FFNConfig, ModelConfig, NormConfig, PositionConfig,
        QuantConfig,
    };
    pub use crate::engine::InferenceEngine;
    pub use crate::model::{CausalLM, DecoderLayer, LoadResult, ModelBuilder, ModelOutput};
    pub use crate::generation::{
        CombinedSampler, GenerationConfig, GenerationOutput, GenerationPipeline, GreedySampler,
        RepetitionPenalty, Sampler, StopChecker, StopCondition, TemperatureSampler, TopKSampler,
        TopPSampler,
    };
    pub use crate::registry::{
        create_attention, create_ffn, create_norm, create_position, AttentionVariant, FFNVariant,
        NormVariant, PositionVariant,
    };
    pub use crate::quantization::{
        f32_model_memory_bytes, quantize_model, MinMaxCalibration, PercentileCalibration,
        QuantizedCausalLM, QuantizedDecoderLayer, QuantizedLinear,
    };
    pub use crate::tokenizer::{Tokenizer, TokenizerError};
}

#[cfg(test)]
mod tests {
    use crate::config::*;
    use crate::registry;

    const QWEN3_JSON: &str = include_str!("../../../configs/qwen3_0.6b.json");
    const LLAMA_JSON: &str = include_str!("../../../configs/llama3.2_1b.json");
    const MISTRAL_JSON: &str = include_str!("../../../configs/mistral_7b.json");

    // ── Qwen3 deserialization ───────────────────────────────────────

    #[test]
    fn deserialize_qwen3_config() {
        let cfg = ModelConfig::from_json(QWEN3_JSON).expect("Qwen3 config should parse");
        assert_eq!(cfg.name, "Qwen3-0.6B");
        assert_eq!(cfg.architecture.hidden_size, 1024);
        assert_eq!(cfg.architecture.num_layers, 28);
        assert_eq!(cfg.architecture.vocab_size, 151936);
        assert_eq!(cfg.architecture.max_sequence_length, 32768);
        assert!(cfg.architecture.tie_word_embeddings);

        assert_eq!(
            cfg.attention,
            AttentionConfig::GQA {
                num_heads: 16,
                num_kv_heads: 8,
                head_dim: 64,
            }
        );
        assert_eq!(cfg.norm, NormConfig::RMSNorm { eps: 1e-6 });
        assert_eq!(
            cfg.ffn,
            FFNConfig::SwiGLU {
                intermediate_size: 3072,
            }
        );
        assert_eq!(
            cfg.position,
            PositionConfig::RoPE {
                base: 1_000_000.0,
                max_position_embeddings: 32768,
                style: Default::default(),
                scaling: Default::default(),
            }
        );
        assert_eq!(cfg.quantization, QuantConfig::F32);
    }

    // ── LLaMA deserialization ───────────────────────────────────────

    #[test]
    fn deserialize_llama_config() {
        let cfg = ModelConfig::from_json(LLAMA_JSON).expect("LLaMA config should parse");
        assert_eq!(cfg.name, "LLaMA-3.2-1B");
        assert_eq!(cfg.architecture.hidden_size, 2048);
        assert_eq!(cfg.architecture.num_layers, 16);
        assert_eq!(cfg.architecture.vocab_size, 128256);
        assert_eq!(cfg.architecture.max_sequence_length, 131072);
        assert!(cfg.architecture.tie_word_embeddings);

        assert_eq!(
            cfg.attention,
            AttentionConfig::GQA {
                num_heads: 32,
                num_kv_heads: 8,
                head_dim: 64,
            }
        );
        assert_eq!(cfg.norm, NormConfig::RMSNorm { eps: 1e-5 });
        assert_eq!(
            cfg.ffn,
            FFNConfig::SwiGLU {
                intermediate_size: 8192,
            }
        );
        assert_eq!(
            cfg.position,
            PositionConfig::RoPE {
                base: 500_000.0,
                max_position_embeddings: 131072,
                style: Default::default(),
                scaling: Default::default(),
            }
        );
        assert_eq!(cfg.quantization, QuantConfig::F16);
    }

    // ── Mistral deserialization (SlidingWindow) ─────────────────────

    #[test]
    fn deserialize_mistral_config() {
        let cfg = ModelConfig::from_json(MISTRAL_JSON).expect("Mistral config should parse");
        assert_eq!(cfg.name, "Mistral-7B-v0.3");
        assert_eq!(
            cfg.attention,
            AttentionConfig::SlidingWindow {
                num_heads: 32,
                num_kv_heads: 8,
                head_dim: 128,
                window_size: 4096,
            }
        );
        assert!(!cfg.architecture.tie_word_embeddings);
    }

    // ── Factory creates correct variants ────────────────────────────

    #[test]
    fn factory_attention_gqa() {
        let config = AttentionConfig::GQA {
            num_heads: 16,
            num_kv_heads: 8,
            head_dim: 64,
        };
        let variant = registry::create_attention(&config);
        assert_eq!(variant.name(), "GQA");
        assert_eq!(variant.num_heads(), 16);
        assert_eq!(variant.num_kv_heads(), 8);
        assert_eq!(variant.head_dim(), 64);
    }

    #[test]
    fn factory_attention_mha() {
        let config = AttentionConfig::MHA {
            num_heads: 12,
            head_dim: 64,
        };
        let variant = registry::create_attention(&config);
        assert_eq!(variant.name(), "MHA");
        assert_eq!(variant.num_kv_heads(), 12);
    }

    #[test]
    fn factory_attention_mqa() {
        let config = AttentionConfig::MQA {
            num_heads: 32,
            head_dim: 128,
        };
        let variant = registry::create_attention(&config);
        assert_eq!(variant.name(), "MQA");
        assert_eq!(variant.num_kv_heads(), 1);
    }

    #[test]
    fn factory_attention_sliding_window() {
        let config = AttentionConfig::SlidingWindow {
            num_heads: 32,
            num_kv_heads: 8,
            head_dim: 128,
            window_size: 4096,
        };
        let variant = registry::create_attention(&config);
        assert_eq!(variant.name(), "SlidingWindow");
    }

    #[test]
    fn factory_norm_rmsnorm() {
        let config = NormConfig::RMSNorm { eps: 1e-6 };
        let variant = registry::create_norm(&config);
        assert_eq!(variant.name(), "RMSNorm");
        assert!((variant.eps() - 1e-6).abs() < 1e-12);
    }

    #[test]
    fn factory_norm_layernorm() {
        let config = NormConfig::LayerNorm { eps: 1e-5 };
        let variant = registry::create_norm(&config);
        assert_eq!(variant.name(), "LayerNorm");
    }

    #[test]
    fn factory_ffn_variants() {
        let swiglu = registry::create_ffn(&FFNConfig::SwiGLU { intermediate_size: 3072 });
        assert_eq!(swiglu.name(), "SwiGLU");
        assert_eq!(swiglu.intermediate_size(), 3072);

        let gelu = registry::create_ffn(&FFNConfig::GELU { intermediate_size: 4096 });
        assert_eq!(gelu.name(), "GELU");

        let geglu = registry::create_ffn(&FFNConfig::GeGLU { intermediate_size: 2048 });
        assert_eq!(geglu.name(), "GeGLU");
    }

    #[test]
    fn factory_position_variants() {
        let rope = registry::create_position(&PositionConfig::RoPE {
            base: 10000.0,
            max_position_embeddings: 4096,
            style: Default::default(),
            scaling: Default::default(),
        });
        assert_eq!(rope.name(), "RoPE");
        assert_eq!(rope.max_position_embeddings(), 4096);

        let learned = registry::create_position(&PositionConfig::Learned {
            max_position_embeddings: 2048,
        });
        assert_eq!(learned.name(), "Learned");

        let sinusoidal = registry::create_position(&PositionConfig::Sinusoidal {
            max_position_embeddings: 512,
        });
        assert_eq!(sinusoidal.name(), "Sinusoidal");
    }

    // ── Unknown / invalid config values produce clear errors ────────

    #[test]
    fn unknown_attention_type_produces_error() {
        let json = r#"{"type": "FlashAttention", "num_heads": 8}"#;
        let result: Result<AttentionConfig, _> = serde_json::from_str(json);
        assert!(result.is_err());
        let err_msg = result.unwrap_err().to_string();
        assert!(
            err_msg.contains("FlashAttention") || err_msg.contains("unknown variant"),
            "Error should mention the unknown variant, got: {err_msg}"
        );
    }

    #[test]
    fn missing_required_field_produces_error() {
        let json = r#"{"type": "GQA", "num_heads": 16}"#;
        let result: Result<AttentionConfig, _> = serde_json::from_str(json);
        assert!(result.is_err());
        let err_msg = result.unwrap_err().to_string();
        assert!(
            err_msg.contains("missing field"),
            "Error should mention missing field, got: {err_msg}"
        );
    }

    #[test]
    fn invalid_model_config_missing_section() {
        let json = r#"{"name": "Test", "architecture": {"hidden_size": 1024, "num_layers": 12, "vocab_size": 32000, "max_sequence_length": 2048, "tie_word_embeddings": false}}"#;
        let result = ModelConfig::from_json(json);
        assert!(result.is_err());
        let err_msg = result.unwrap_err().to_string();
        assert!(
            err_msg.contains("missing field"),
            "Error should mention missing field, got: {err_msg}"
        );
    }

    // ── Round-trip: serialize → deserialize → compare ───────────────

    #[test]
    fn round_trip_qwen3() {
        let original = ModelConfig::from_json(QWEN3_JSON).unwrap();
        let serialized = original.to_json().unwrap();
        let deserialized = ModelConfig::from_json(&serialized).unwrap();
        assert_eq!(original, deserialized);
    }

    #[test]
    fn round_trip_llama() {
        let original = ModelConfig::from_json(LLAMA_JSON).unwrap();
        let serialized = original.to_json().unwrap();
        let deserialized = ModelConfig::from_json(&serialized).unwrap();
        assert_eq!(original, deserialized);
    }

    #[test]
    fn round_trip_mistral() {
        let original = ModelConfig::from_json(MISTRAL_JSON).unwrap();
        let serialized = original.to_json().unwrap();
        let deserialized = ModelConfig::from_json(&serialized).unwrap();
        assert_eq!(original, deserialized);
    }

    // ── Full pipeline: parse config → create all variants ───────────

    #[test]
    fn full_pipeline_qwen3() {
        let cfg = ModelConfig::from_json(QWEN3_JSON).unwrap();
        let attn = registry::create_attention(&cfg.attention);
        let norm = registry::create_norm(&cfg.norm);
        let ffn = registry::create_ffn(&cfg.ffn);
        let pos = registry::create_position(&cfg.position);

        assert_eq!(attn.name(), "GQA");
        assert_eq!(norm.name(), "RMSNorm");
        assert_eq!(ffn.name(), "SwiGLU");
        assert_eq!(pos.name(), "RoPE");
    }
}

// ══════════════════════════════════════════════════════════════════════
// Model builder + CausalLM + Engine tests
// ══════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod model_tests {
    use std::collections::HashMap;

    use crate::config::*;
    use crate::engine::InferenceEngine;
    use crate::model::ModelBuilder;
    use crate::weight_loading::{RawTensor, WeightMapper};

    const QWEN3_JSON: &str = include_str!("../../../configs/qwen3_0.6b.json");
    const LLAMA_JSON: &str = include_str!("../../../configs/llama3.2_1b.json");

    // ── Helper: tiny config for forward / weight tests ───────────────

    fn tiny_config() -> ModelConfig {
        ModelConfig {
            name: "TinyTest".to_string(),
            architecture: ArchitectureConfig {
                hidden_size: 32,
                num_layers: 2,
                vocab_size: 64,
                max_sequence_length: 16,
                tie_word_embeddings: true,
            },
            attention: AttentionConfig::GQA {
                num_heads: 4,
                num_kv_heads: 2,
                head_dim: 8,
            },
            norm: NormConfig::RMSNorm { eps: 1e-6 },
            ffn: FFNConfig::SwiGLU {
                intermediate_size: 64,
            },
            position: PositionConfig::RoPE {
                base: 10000.0,
                max_position_embeddings: 16,
                style: Default::default(),
                scaling: Default::default(),
            },
            quantization: QuantConfig::F32,
        }
    }

    /// Compute expected param count from config dimensions.
    fn expected_param_count(cfg: &ModelConfig) -> usize {
        let h = cfg.architecture.hidden_size;
        let vocab = cfg.architecture.vocab_size;
        let q_dim = cfg.attention.num_heads() * cfg.attention.head_dim();
        let kv_dim = cfg.attention.num_kv_heads() * cfg.attention.head_dim();
        let inter = cfg.ffn.intermediate_size();
        let nl = cfg.architecture.num_layers;

        let embed = vocab * h;
        let per_layer = 2 * h                 // attn_norm + ffn_norm
            + 2 * cfg.attention.head_dim()   // q_norm + k_norm
            + q_dim * h       // w_q
            + kv_dim * h      // w_k
            + kv_dim * h      // w_v
            + h * q_dim       // w_o
            + 3 * inter * h;  // SwiGLU: gate + up + down
        let norm = h;
        let lm_head = if cfg.architecture.tie_word_embeddings { 0 } else { vocab * h };

        embed + nl * per_layer + norm + lm_head
    }

    /// Generate fake HuggingFace-format weights for a config.
    fn generate_fake_hf_weights(cfg: &ModelConfig) -> HashMap<String, RawTensor> {
        let h = cfg.architecture.hidden_size;
        let vocab = cfg.architecture.vocab_size;
        let q_dim = cfg.attention.num_heads() * cfg.attention.head_dim();
        let kv_dim = cfg.attention.num_kv_heads() * cfg.attention.head_dim();
        let inter = cfg.ffn.intermediate_size();
        let nl = cfg.architecture.num_layers;

        let fake = |shape: Vec<usize>| -> RawTensor {
            let n: usize = shape.iter().product();
            RawTensor {
                data: (0..n).map(|i| (i as f32 * 0.001) % 0.1 + 0.01).collect(),
                shape,
            }
        };

        let mut w = HashMap::new();
        w.insert(
            "model.embed_tokens.weight".into(),
            fake(vec![vocab, h]),
        );
        for i in 0..nl {
            w.insert(format!("model.layers.{i}.input_layernorm.weight"), fake(vec![h]));
            w.insert(format!("model.layers.{i}.self_attn.q_proj.weight"), fake(vec![q_dim, h]));
            w.insert(format!("model.layers.{i}.self_attn.k_proj.weight"), fake(vec![kv_dim, h]));
            w.insert(format!("model.layers.{i}.self_attn.v_proj.weight"), fake(vec![kv_dim, h]));
            w.insert(format!("model.layers.{i}.self_attn.o_proj.weight"), fake(vec![h, q_dim]));
            w.insert(
                format!("model.layers.{i}.self_attn.q_norm.weight"),
                fake(vec![cfg.attention.head_dim()]),
            );
            w.insert(
                format!("model.layers.{i}.self_attn.k_norm.weight"),
                fake(vec![cfg.attention.head_dim()]),
            );
            w.insert(format!("model.layers.{i}.post_attention_layernorm.weight"), fake(vec![h]));
            w.insert(format!("model.layers.{i}.mlp.gate_proj.weight"), fake(vec![inter, h]));
            w.insert(format!("model.layers.{i}.mlp.up_proj.weight"), fake(vec![inter, h]));
            w.insert(format!("model.layers.{i}.mlp.down_proj.weight"), fake(vec![h, inter]));
        }
        w.insert("model.norm.weight".into(), fake(vec![h]));
        w.insert("lm_head.weight".into(), fake(vec![vocab, h]));
        w
    }

    // ── Test 1: Build Qwen3-0.6B from config ────────────────────────

    #[test]
    fn build_qwen3_from_config() {
        let cfg = ModelConfig::from_json(QWEN3_JSON).unwrap();
        let model = ModelBuilder::from_config(&cfg);

        // 28 layers
        assert_eq!(model.layers.len(), 28);

        // RMSNorm everywhere
        assert_eq!(model.final_norm.name(), "RMSNorm");
        assert_eq!(model.layers[0].attn_norm.name(), "RMSNorm");
        assert_eq!(model.layers[0].ffn_norm.name(), "RMSNorm");

        // GQA(16Q/8KV)
        assert_eq!(model.layers[0].attention.name(), "GQA");
        assert_eq!(model.layers[0].attention.num_heads(), 16);
        assert_eq!(model.layers[0].attention.num_kv_heads(), 8);
        assert_eq!(model.layers[0].attention.head_dim(), 64);

        // SwiGLU
        assert_eq!(model.layers[0].ffn.name(), "SwiGLU");
        assert_eq!(model.layers[0].ffn.intermediate_size(), 3072);

        // lm_head=None (tied embeddings)
        assert!(model.lm_head_weight.is_none());

        // Param count within 1% of expected
        let expected = expected_param_count(&cfg);
        let actual = model.param_count();
        assert_eq!(actual, expected);
        let ratio = actual as f64 / expected as f64;
        assert!(
            (ratio - 1.0).abs() < 0.01,
            "Param count {actual} not within 1% of expected {expected}"
        );
    }

    // ── Test 2: Build LLaMA from config ─────────────────────────────

    #[test]
    fn build_llama_from_config() {
        let cfg = ModelConfig::from_json(LLAMA_JSON).unwrap();
        let model = ModelBuilder::from_config(&cfg);

        // Different layer count
        assert_eq!(model.layers.len(), 16);

        // Different head geometry
        assert_eq!(model.layers[0].attention.num_heads(), 32);
        assert_eq!(model.layers[0].attention.num_kv_heads(), 8);
        assert_eq!(model.layers[0].attention.head_dim(), 64);

        // Param count matches
        let expected = expected_param_count(&cfg);
        let actual = model.param_count();
        assert_eq!(actual, expected);
    }

    // ── Test 3: Forward with random weights → correct output shape ──

    #[test]
    fn forward_output_shape() {
        let cfg = tiny_config();
        let mut model = ModelBuilder::from_config(&cfg);

        // Load fake weights so forward can run
        let weights = generate_fake_hf_weights(&cfg);
        let mapper = WeightMapper::qwen3();
        let result = model.load_weights(weights, &mapper).unwrap();
        assert!(result.missing.is_empty(), "Missing: {:?}", result.missing);

        // Forward pass
        let seq_len = 4;
        let token_ids: Vec<u32> = (0..seq_len as u32).collect();
        let output = model.forward(&token_ids);

        // Output shape [1, seq_len, vocab_size]
        assert_eq!(output.shape, [1, seq_len, cfg.architecture.vocab_size]);
        assert_eq!(
            output.logits.len(),
            seq_len * cfg.architecture.vocab_size
        );

        // Logits should contain finite values
        assert!(
            output.logits.iter().all(|v| v.is_finite()),
            "Logits contain non-finite values"
        );
    }

    // ── Test 4: Weight loading with fake weights → zero missing/unexpected

    #[test]
    fn weight_loading_zero_missing_unexpected() {
        let cfg = tiny_config();
        let mut model = ModelBuilder::from_config(&cfg);

        let weights = generate_fake_hf_weights(&cfg);
        let mapper = WeightMapper::qwen3();
        let result = model.load_weights(weights, &mapper).unwrap();

        assert!(
            result.missing.is_empty(),
            "Missing keys: {:?}",
            result.missing
        );
        assert!(
            result.unexpected.is_empty(),
            "Unexpected keys: {:?}",
            result.unexpected
        );
    }

    // ── Test 5: Config assembly <= 2 seconds ────────────────────────

    #[test]
    fn config_assembly_under_2_seconds() {
        let cfg = ModelConfig::from_json(QWEN3_JSON).unwrap();

        let start = std::time::Instant::now();
        let _model = ModelBuilder::from_config(&cfg);
        let elapsed = start.elapsed();

        assert!(
            elapsed.as_secs_f64() < 2.0,
            "Assembly took {:.3}s, expected < 2s",
            elapsed.as_secs_f64()
        );
    }

    // ── InferenceEngine basic smoke test ─────────────────────────────

    #[test]
    fn engine_from_config() {
        let cfg = ModelConfig::from_json(QWEN3_JSON).unwrap();
        let engine = InferenceEngine::from_config(cfg);
        assert_eq!(engine.model.layers.len(), 28);
        assert!(engine.param_count() > 0);
    }
}

// ══════════════════════════════════════════════════════════════════════
// INT8 quantization tests
// ══════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod quantization_tests {
    use crate::config::*;
    use crate::model::decoder_layer::matmul_t;
    use crate::model::ModelBuilder;
    use crate::quantization::*;

    fn tiny_config() -> ModelConfig {
        ModelConfig {
            name: "TinyTest".to_string(),
            architecture: ArchitectureConfig {
                hidden_size: 32,
                num_layers: 2,
                vocab_size: 64,
                max_sequence_length: 16,
                tie_word_embeddings: true,
            },
            attention: AttentionConfig::GQA {
                num_heads: 4,
                num_kv_heads: 2,
                head_dim: 8,
            },
            norm: NormConfig::RMSNorm { eps: 1e-6 },
            ffn: FFNConfig::SwiGLU {
                intermediate_size: 64,
            },
            position: PositionConfig::RoPE {
                base: 10000.0,
                max_position_embeddings: 16,
                style: Default::default(),
                scaling: Default::default(),
            },
            quantization: QuantConfig::F32,
        }
    }

    /// Deterministic pseudo-random weight generator with realistic magnitude.
    ///
    /// Produces values in approximately [-0.18, 0.18] (Xavier-scale for d=32),
    /// which keeps INT8 quantization error well below 1%.
    fn gen_weights(len: usize, seed: u32) -> crate::weight_loading::AlignedBuffer {
        let v: Vec<f32> = (0..len)
            .map(|i| {
                let x = ((i as u32).wrapping_mul(2654435761).wrapping_add(seed)) as f32;
                (x / u32::MAX as f32) * 0.36 - 0.18 // range ~[-0.18, 0.18]
            })
            .collect();
        crate::weight_loading::AlignedBuffer::from_slice(&v)
    }

    /// Normalized RMSE: ||f32 - int8||_2 / ||f32||_2.
    /// Standard metric for quantization quality.
    fn nrmse(reference: &[f32], quantized: &[f32]) -> f32 {
        let l2_err: f32 = reference
            .iter()
            .zip(quantized.iter())
            .map(|(f, q)| (f - q).powi(2))
            .sum();
        let l2_ref: f32 = reference.iter().map(|f| f.powi(2)).sum();
        if l2_ref == 0.0 {
            return 0.0;
        }
        (l2_err / l2_ref).sqrt()
    }

    /// Fill a CausalLM with deterministic weights.
    fn fill_model_weights(model: &mut crate::model::CausalLM) {
        let cfg = &model.config;
        let h = cfg.architecture.hidden_size;
        let vocab = cfg.architecture.vocab_size;
        let q_dim = cfg.attention.num_heads() * cfg.attention.head_dim();
        let kv_dim = cfg.attention.num_kv_heads() * cfg.attention.head_dim();
        let inter = cfg.ffn.intermediate_size();

        model.embed_tokens = gen_weights(vocab * h, 1);
        model.final_norm_weight = crate::weight_loading::AlignedBuffer::from_slice(&vec![1.0f32; h]);

        for (i, layer) in model.layers.iter_mut().enumerate() {
            let s = (i as u32 + 1) * 100;
            layer.attn_norm_weight = crate::weight_loading::AlignedBuffer::from_slice(&vec![1.0f32; h]);
            layer.w_q = gen_weights(q_dim * h, s + 1);
            layer.w_k = gen_weights(kv_dim * h, s + 2);
            layer.w_v = gen_weights(kv_dim * h, s + 3);
            layer.w_o = gen_weights(h * q_dim, s + 4);
            layer.ffn_norm_weight = crate::weight_loading::AlignedBuffer::from_slice(&vec![1.0f32; h]);
            layer.ffn_gate = gen_weights(inter * h, s + 5);
            layer.ffn_up = gen_weights(inter * h, s + 6);
            layer.ffn_down = gen_weights(h * inter, s + 7);
        }
    }

    // ── Test 1: MinMax calibration computes correct scales ───────────

    #[test]
    fn minmax_calibration_correct_scales() {
        let weights = vec![
            1.0, -2.0, 3.0, -0.5, // Channel 0: max_abs = 3.0
            0.5, -1.0, 0.0, 2.0,  // Channel 1: max_abs = 2.0
        ];
        let scales = MinMaxCalibration::compute_scales(&weights, 2, 4);
        assert_eq!(scales.len(), 2);
        assert!(
            (scales[0] - 3.0 / 127.0).abs() < 1e-7,
            "Channel 0 scale: expected {}, got {}",
            3.0 / 127.0,
            scales[0]
        );
        assert!(
            (scales[1] - 2.0 / 127.0).abs() < 1e-7,
            "Channel 1 scale: expected {}, got {}",
            2.0 / 127.0,
            scales[1]
        );
    }

    #[test]
    fn minmax_calibration_zero_channel() {
        let weights = vec![0.0, 0.0, 0.0, 0.0];
        let scales = MinMaxCalibration::compute_scales(&weights, 1, 4);
        assert_eq!(scales[0], 1.0, "Zero channel should produce scale=1.0");
    }

    #[test]
    fn percentile_calibration_clips_outliers() {
        let mut weights = vec![0.1f32; 10];
        weights[9] = 10.0;
        let cal = PercentileCalibration::new(90.0);
        let scales = cal.compute_scales(&weights, 1, 10);
        assert!(
            scales[0] < 10.0 / 127.0,
            "Percentile scale should be less than min/max scale"
        );
    }

    // ── Test 2: QuantizedLinear forward within 1% of f32 ────────────

    #[test]
    fn quantized_linear_forward_within_1_percent() {
        let out_features = 32;
        let in_features = 64;
        let m = 4;

        let weights = gen_weights(out_features * in_features, 42);
        let x = gen_weights(m * in_features, 99);

        let f32_output = matmul_t(&x, &weights, m, in_features, out_features);
        let ql = QuantizedLinear::from_f32(&weights, out_features, in_features);
        let q_output = ql.forward(&x, m);

        assert_eq!(f32_output.len(), q_output.len());
        let err = nrmse(&f32_output, &q_output);
        assert!(
            err < 0.01,
            "Normalized RMSE {err:.6} exceeds 1%"
        );
    }

    #[test]
    fn quantized_linear_forward_larger_matrix() {
        let out_features = 128;
        let in_features = 256;
        let m = 8;

        let weights = gen_weights(out_features * in_features, 7);
        let x = gen_weights(m * in_features, 13);

        let f32_output = matmul_t(&x, &weights, m, in_features, out_features);
        let ql = QuantizedLinear::from_f32(&weights, out_features, in_features);
        let q_output = ql.forward(&x, m);

        let err = nrmse(&f32_output, &q_output);
        assert!(
            err < 0.01,
            "Normalized RMSE {err:.6} exceeds 1%"
        );
    }

    // ── Test 3: Quantize model, verify INT8 weights and scales ──────

    #[test]
    fn quantize_model_int8_weights_and_scales() {
        let cfg = tiny_config();
        let mut model = ModelBuilder::from_config(&cfg);
        fill_model_weights(&mut model);

        let quantized = quantize_model(&model);

        assert_eq!(quantized.layers.len(), model.layers.len());

        for (i, layer) in quantized.layers.iter().enumerate() {
            let q_dim = cfg.attention.num_heads() * cfg.attention.head_dim();
            let kv_dim = cfg.attention.num_kv_heads() * cfg.attention.head_dim();
            let h = cfg.architecture.hidden_size;
            let inter = cfg.ffn.intermediate_size();

            assert_eq!(layer.w_q.weights_int8.len(), q_dim * h, "Layer {i} w_q size");
            assert_eq!(layer.w_q.scales.len(), q_dim, "Layer {i} w_q scales");
            assert_eq!(layer.w_k.weights_int8.len(), kv_dim * h);
            assert_eq!(layer.w_k.scales.len(), kv_dim);
            assert_eq!(layer.w_v.weights_int8.len(), kv_dim * h);
            assert_eq!(layer.w_o.weights_int8.len(), h * q_dim);
            assert_eq!(layer.ffn_up.weights_int8.len(), inter * h);
            assert_eq!(layer.ffn_down.weights_int8.len(), h * inter);
            assert_eq!(layer.ffn_gate.weights_int8.len(), inter * h);

            // Scales are positive
            assert!(layer.w_q.scales.iter().all(|&s| s > 0.0));
            assert!(layer.w_k.scales.iter().all(|&s| s > 0.0));

            // INT8 values are populated (not all zero)
            assert!(layer.w_q.weights_int8.iter().any(|&w| w != 0));
        }

        // Embedding stays f32
        assert_eq!(quantized.embed_tokens.len(), model.embed_tokens.len());
    }

    // ── Test 4: Quantized model has ~50% memory of f32 model ────────

    #[test]
    fn quantized_model_memory_reduction() {
        let cfg = tiny_config();
        let mut model = ModelBuilder::from_config(&cfg);
        fill_model_weights(&mut model);

        let f32_mem = f32_model_memory_bytes(&model);
        let quantized = quantize_model(&model);
        let q_mem = quantized.memory_bytes();

        let ratio = q_mem as f64 / f32_mem as f64;
        assert!(
            ratio < 0.55,
            "Memory ratio {ratio:.3} should be < 0.55 (~50% reduction). \
             f32={f32_mem} bytes, int8={q_mem} bytes"
        );
        assert!(
            ratio > 0.15,
            "Memory ratio {ratio:.3} suspiciously low, possible bug"
        );
    }

    // ── Test 5: Quantized model forward produces finite output ──────

    #[test]
    fn quantized_model_forward_finite() {
        let cfg = tiny_config();
        let mut model = ModelBuilder::from_config(&cfg);
        fill_model_weights(&mut model);

        let quantized = quantize_model(&model);

        let token_ids: Vec<u32> = vec![1, 2, 3, 4];
        let output = quantized.forward(&token_ids);

        assert_eq!(output.shape, [1, 4, cfg.architecture.vocab_size]);
        assert_eq!(output.logits.len(), 4 * cfg.architecture.vocab_size);
        assert!(
            output.logits.iter().all(|v| v.is_finite()),
            "Quantized model output contains non-finite values"
        );
    }

    // ── Test 6: Quantized forward within 1% of f32 forward ─────────

    #[test]
    fn quantized_model_forward_within_1_percent() {
        let cfg = tiny_config();
        let mut model = ModelBuilder::from_config(&cfg);
        fill_model_weights(&mut model);

        let quantized = quantize_model(&model);

        let token_ids: Vec<u32> = vec![1, 2, 3, 4];
        let f32_output = model.forward(&token_ids);
        let q_output = quantized.forward(&token_ids);

        assert_eq!(f32_output.logits.len(), q_output.logits.len());

        let err = nrmse(&f32_output.logits, &q_output.logits);
        assert!(
            err < 0.02,
            "Quantized model NRMSE {err:.6} exceeds 2%"
        );
    }
}
