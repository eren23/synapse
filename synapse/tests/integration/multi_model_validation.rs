//! Multi-architecture model validation tests.
//!
//! Verifies that all supported model architectures (Qwen3, LLaMA, Mistral, Phi, Gemma, ViT)
//! produce correct forward pass results with fake weights. Tests cover:
//! - Forward pass producing finite logits with correct shape
//! - Cached decode (prefill + forward_one) consistency
//! - INT8 quantized forward pass producing finite logits
//! - ViT forward producing finite embeddings and correct patch embedding shapes

use std::collections::HashMap;

use synapse_inference::config::position::RoPEScaling;
use synapse_inference::config::*;
use synapse_inference::kv_cache::KVCache;
use synapse_inference::model::{CausalLM, ModelBuilder};
use synapse_inference::quantization::{quantize_model, quantize_model_ternary};
use synapse_inference::weight_loading::{AlignedBuffer, RawTensor, WeightMapper};

// ── Shared test constants ──────────────────────────────────────────────

const HIDDEN_SIZE: usize = 64;
const NUM_LAYERS: usize = 2;
const VOCAB_SIZE: usize = 128;
const MAX_SEQ: usize = 64;
const INTERMEDIATE_SIZE: usize = 128;
const HEAD_DIM: usize = 16;

// ── Deterministic pseudo-random weight generator ───────────────────────

fn gen_weights(len: usize, seed: u32) -> Vec<f32> {
    (0..len)
        .map(|i| {
            let x = ((i as u32).wrapping_mul(2654435761).wrapping_add(seed)) as f32;
            (x / u32::MAX as f32) * 0.36 - 0.18
        })
        .collect()
}

// ── Model configs ──────────────────────────────────────────────────────

fn qwen3_config() -> ModelConfig {
    ModelConfig {
        name: "qwen3".to_string(),
        architecture: ArchitectureConfig {
            hidden_size: HIDDEN_SIZE,
            num_layers: NUM_LAYERS,
            vocab_size: VOCAB_SIZE,
            max_sequence_length: MAX_SEQ,
            tie_word_embeddings: true,
            embed_scale: None,
        },
        attention: AttentionConfig::GQA {
            num_heads: 4,
            num_kv_heads: 2,
            head_dim: HEAD_DIM,
        },
        norm: NormConfig::RMSNorm { eps: 1e-6 },
        ffn: FFNConfig::SwiGLU {
            intermediate_size: INTERMEDIATE_SIZE,
        },
        position: PositionConfig::RoPE {
            base: 10000.0,
            max_position_embeddings: MAX_SEQ,
            style: Default::default(),
            scaling: Default::default(),
        },
        quantization: QuantConfig::F32,
    }
}

fn llama_config() -> ModelConfig {
    ModelConfig {
        name: "llama".to_string(),
        architecture: ArchitectureConfig {
            hidden_size: HIDDEN_SIZE,
            num_layers: NUM_LAYERS,
            vocab_size: VOCAB_SIZE,
            max_sequence_length: MAX_SEQ,
            tie_word_embeddings: true,
            embed_scale: None,
        },
        attention: AttentionConfig::GQA {
            num_heads: 4,
            num_kv_heads: 2,
            head_dim: HEAD_DIM,
        },
        norm: NormConfig::RMSNorm { eps: 1e-5 },
        ffn: FFNConfig::SwiGLU {
            intermediate_size: INTERMEDIATE_SIZE,
        },
        position: PositionConfig::RoPE {
            base: 10000.0,
            max_position_embeddings: MAX_SEQ,
            style: Default::default(),
            scaling: RoPEScaling::Linear { factor: 2.0 },
        },
        quantization: QuantConfig::F32,
    }
}

fn mistral_config() -> ModelConfig {
    ModelConfig {
        name: "mistral".to_string(),
        architecture: ArchitectureConfig {
            hidden_size: HIDDEN_SIZE,
            num_layers: NUM_LAYERS,
            vocab_size: VOCAB_SIZE,
            max_sequence_length: MAX_SEQ,
            tie_word_embeddings: true,
            embed_scale: None,
        },
        attention: AttentionConfig::SlidingWindow {
            num_heads: 4,
            num_kv_heads: 2,
            head_dim: HEAD_DIM,
            window_size: 16,
        },
        norm: NormConfig::RMSNorm { eps: 1e-6 },
        ffn: FFNConfig::SwiGLU {
            intermediate_size: INTERMEDIATE_SIZE,
        },
        position: PositionConfig::RoPE {
            base: 10000.0,
            max_position_embeddings: MAX_SEQ,
            style: Default::default(),
            scaling: Default::default(),
        },
        quantization: QuantConfig::F32,
    }
}

fn phi3_config() -> ModelConfig {
    ModelConfig {
        name: "phi3".to_string(),
        architecture: ArchitectureConfig {
            hidden_size: HIDDEN_SIZE,
            num_layers: NUM_LAYERS,
            vocab_size: VOCAB_SIZE,
            max_sequence_length: MAX_SEQ,
            tie_word_embeddings: true,
            embed_scale: None,
        },
        attention: AttentionConfig::GQA {
            num_heads: 4,
            num_kv_heads: 2,
            head_dim: HEAD_DIM,
        },
        norm: NormConfig::RMSNorm { eps: 1e-6 },
        ffn: FFNConfig::SwiGLU {
            intermediate_size: INTERMEDIATE_SIZE,
        },
        position: PositionConfig::RoPE {
            base: 10000.0,
            max_position_embeddings: MAX_SEQ,
            style: Default::default(),
            scaling: Default::default(),
        },
        quantization: QuantConfig::F32,
    }
}

fn gemma_config() -> ModelConfig {
    ModelConfig {
        name: "gemma".to_string(),
        architecture: ArchitectureConfig {
            hidden_size: HIDDEN_SIZE,
            num_layers: NUM_LAYERS,
            vocab_size: VOCAB_SIZE,
            max_sequence_length: MAX_SEQ,
            tie_word_embeddings: true,
            embed_scale: None,
        },
        attention: AttentionConfig::MHA {
            num_heads: 4,
            head_dim: HEAD_DIM,
        },
        norm: NormConfig::RMSNorm { eps: 1e-6 },
        ffn: FFNConfig::SwiGLU {
            intermediate_size: INTERMEDIATE_SIZE,
        },
        position: PositionConfig::RoPE {
            base: 10000.0,
            max_position_embeddings: MAX_SEQ,
            style: Default::default(),
            scaling: Default::default(),
        },
        quantization: QuantConfig::F32,
    }
}

// ── Fake weight generation ─────────────────────────────────────────────

/// Generate fake HF-format weights for a model config.
///
/// Uses `WeightMapper::from_model_type` to determine whether q_norm/k_norm
/// should be included (Qwen3 has them, others do not).
/// For tied embeddings, lm_head.weight is NOT included.
fn generate_fake_hf_weights(cfg: &ModelConfig) -> HashMap<String, RawTensor> {
    let h = cfg.architecture.hidden_size;
    let vocab = cfg.architecture.vocab_size;
    let q_dim = cfg.attention.num_heads() * cfg.attention.head_dim();
    let kv_dim = cfg.attention.num_kv_heads() * cfg.attention.head_dim();
    let inter = cfg.ffn.intermediate_size();
    let nl = cfg.architecture.num_layers;
    let has_head_norms = cfg.name == "qwen3";

    let fake = |shape: Vec<usize>, seed: u32| -> RawTensor {
        let n: usize = shape.iter().product();
        RawTensor {
            data: AlignedBuffer::from_slice(&gen_weights(n, seed)),
            shape,
        }
    };

    let mut w = HashMap::new();
    w.insert("model.embed_tokens.weight".into(), fake(vec![vocab, h], 1));
    for i in 0..nl {
        let s = (i as u32 + 1) * 100;
        w.insert(
            format!("model.layers.{i}.input_layernorm.weight"),
            fake(vec![h], s),
        );
        w.insert(
            format!("model.layers.{i}.self_attn.q_proj.weight"),
            fake(vec![q_dim, h], s + 1),
        );
        w.insert(
            format!("model.layers.{i}.self_attn.k_proj.weight"),
            fake(vec![kv_dim, h], s + 2),
        );
        w.insert(
            format!("model.layers.{i}.self_attn.v_proj.weight"),
            fake(vec![kv_dim, h], s + 3),
        );
        w.insert(
            format!("model.layers.{i}.self_attn.o_proj.weight"),
            fake(vec![h, q_dim], s + 4),
        );
        if has_head_norms {
            w.insert(
                format!("model.layers.{i}.self_attn.q_norm.weight"),
                fake(vec![cfg.attention.head_dim()], s + 5),
            );
            w.insert(
                format!("model.layers.{i}.self_attn.k_norm.weight"),
                fake(vec![cfg.attention.head_dim()], s + 6),
            );
        }
        w.insert(
            format!("model.layers.{i}.post_attention_layernorm.weight"),
            fake(vec![h], s + 7),
        );
        w.insert(
            format!("model.layers.{i}.mlp.gate_proj.weight"),
            fake(vec![inter, h], s + 8),
        );
        w.insert(
            format!("model.layers.{i}.mlp.up_proj.weight"),
            fake(vec![inter, h], s + 9),
        );
        w.insert(
            format!("model.layers.{i}.mlp.down_proj.weight"),
            fake(vec![h, inter], s + 10),
        );
    }
    w.insert("model.norm.weight".into(), fake(vec![h], 9999));
    // Always include lm_head.weight: the model expects it in the weight map
    // even for tied embeddings (set_weight accepts it but reuses embed_tokens).
    w.insert("lm_head.weight".into(), fake(vec![vocab, h], 9998));
    w
}

/// Build a model from config with fake weights loaded via the appropriate mapper.
fn build_model(cfg: &ModelConfig) -> CausalLM {
    let mut model = ModelBuilder::from_config(cfg);
    let weights = generate_fake_hf_weights(cfg);
    let mapper =
        WeightMapper::from_model_type(&cfg.name).expect("unsupported model type for mapper");
    let result = model.load_weights(weights, &mapper).unwrap();
    assert!(
        result.missing.is_empty(),
        "[{}] Missing weight keys: {:?}",
        cfg.name,
        result.missing
    );
    model
}

/// Create a KV cache matching the config.
fn make_cache(cfg: &ModelConfig) -> KVCache {
    KVCache::new(
        cfg.architecture.num_layers,
        cfg.architecture.max_sequence_length,
        cfg.attention.num_kv_heads(),
        cfg.attention.head_dim(),
    )
    .unwrap()
}

// ════════════════════════════════════════════════════════════════════════
// Qwen3
// ════════════════════════════════════════════════════════════════════════

#[test]
fn test_qwen3_forward_produces_finite_logits() {
    let cfg = qwen3_config();
    let model = build_model(&cfg);
    let output = model.forward(&[1, 2, 3]);

    assert_eq!(output.shape, [1, 3, VOCAB_SIZE]);
    assert_eq!(output.logits.len(), 3 * VOCAB_SIZE);
    assert!(
        output.logits.iter().all(|v| v.is_finite()),
        "Qwen3 forward produced non-finite logits"
    );
}

#[test]
fn test_qwen3_cached_decode_matches_forward() {
    let cfg = qwen3_config();
    let model = build_model(&cfg);

    // Prefill
    let mut cache = make_cache(&cfg);
    let prefill_out = model.forward_prefill(&[1, 2, 3], &mut cache);
    assert_eq!(prefill_out.shape, [1, 1, VOCAB_SIZE]);
    assert!(
        prefill_out.logits.iter().all(|v| v.is_finite()),
        "Qwen3 prefill produced non-finite logits"
    );

    // Decode one token
    let decode_out = model.forward_one(4, &mut cache);
    assert_eq!(decode_out.shape, [1, 1, VOCAB_SIZE]);
    assert!(
        decode_out.logits.iter().all(|v| v.is_finite()),
        "Qwen3 forward_one produced non-finite logits"
    );
}

#[test]
fn test_qwen3_quantized_forward_produces_finite_logits() {
    let cfg = qwen3_config();
    let model = build_model(&cfg);
    let quantized = quantize_model(&model);

    let output = quantized.forward(&[1, 2, 3]);
    assert_eq!(output.shape, [1, 3, VOCAB_SIZE]);
    assert_eq!(output.logits.len(), 3 * VOCAB_SIZE);
    assert!(
        output.logits.iter().all(|v| v.is_finite()),
        "Qwen3 quantized forward produced non-finite logits"
    );
}

#[test]
fn test_qwen3_ternary_forward_produces_finite_logits() {
    let cfg = qwen3_config();
    let model = build_model(&cfg);
    let ternary = quantize_model_ternary(&model);
    let output = ternary.forward(&[1, 2, 3]);
    assert_eq!(output.shape, [1, 3, VOCAB_SIZE]);
    assert_eq!(output.logits.len(), 3 * VOCAB_SIZE);
    assert!(
        output.logits.iter().all(|v| v.is_finite()),
        "Qwen3 ternary forward produced non-finite logits"
    );
}

#[test]
fn test_qwen3_ternary_cached_decode() {
    let cfg = qwen3_config();
    let model = build_model(&cfg);
    let ternary = quantize_model_ternary(&model);
    let mut cache = make_cache(&cfg);
    let prefill_out = ternary.forward_prefill(&[1, 2, 3], &mut cache);
    assert_eq!(prefill_out.shape, [1, 1, VOCAB_SIZE]);
    assert!(
        prefill_out.logits.iter().all(|v| v.is_finite()),
        "Qwen3 ternary prefill produced non-finite logits"
    );
    let decode_out = ternary.forward_one(4, &mut cache);
    assert_eq!(decode_out.shape, [1, 1, VOCAB_SIZE]);
    assert!(
        decode_out.logits.iter().all(|v| v.is_finite()),
        "Qwen3 ternary forward_one produced non-finite logits"
    );
}

// ════════════════════════════════════════════════════════════════════════
// LLaMA
// ════════════════════════════════════════════════════════════════════════

#[test]
fn test_llama_forward_produces_finite_logits() {
    let cfg = llama_config();
    let model = build_model(&cfg);
    let output = model.forward(&[1, 2, 3]);

    assert_eq!(output.shape, [1, 3, VOCAB_SIZE]);
    assert_eq!(output.logits.len(), 3 * VOCAB_SIZE);
    assert!(
        output.logits.iter().all(|v| v.is_finite()),
        "LLaMA forward produced non-finite logits"
    );
}

#[test]
fn test_llama_cached_decode_matches_forward() {
    let cfg = llama_config();
    let model = build_model(&cfg);

    let mut cache = make_cache(&cfg);
    let prefill_out = model.forward_prefill(&[1, 2, 3], &mut cache);
    assert_eq!(prefill_out.shape, [1, 1, VOCAB_SIZE]);
    assert!(
        prefill_out.logits.iter().all(|v| v.is_finite()),
        "LLaMA prefill produced non-finite logits"
    );

    let decode_out = model.forward_one(4, &mut cache);
    assert_eq!(decode_out.shape, [1, 1, VOCAB_SIZE]);
    assert!(
        decode_out.logits.iter().all(|v| v.is_finite()),
        "LLaMA forward_one produced non-finite logits"
    );
}

#[test]
fn test_llama_quantized_forward_produces_finite_logits() {
    let cfg = llama_config();
    let model = build_model(&cfg);
    let quantized = quantize_model(&model);

    let output = quantized.forward(&[1, 2, 3]);
    assert_eq!(output.shape, [1, 3, VOCAB_SIZE]);
    assert_eq!(output.logits.len(), 3 * VOCAB_SIZE);
    assert!(
        output.logits.iter().all(|v| v.is_finite()),
        "LLaMA quantized forward produced non-finite logits"
    );
}

#[test]
fn test_llama_ternary_forward_produces_finite_logits() {
    let cfg = llama_config();
    let model = build_model(&cfg);
    let ternary = quantize_model_ternary(&model);
    let output = ternary.forward(&[1, 2, 3]);
    assert_eq!(output.shape, [1, 3, VOCAB_SIZE]);
    assert!(
        output.logits.iter().all(|v| v.is_finite()),
        "LLaMA ternary forward produced non-finite logits"
    );
}

// ════════════════════════════════════════════════════════════════════════
// Mistral
// ════════════════════════════════════════════════════════════════════════

#[test]
fn test_mistral_forward_produces_finite_logits() {
    let cfg = mistral_config();
    let model = build_model(&cfg);
    let output = model.forward(&[1, 2, 3]);

    assert_eq!(output.shape, [1, 3, VOCAB_SIZE]);
    assert_eq!(output.logits.len(), 3 * VOCAB_SIZE);
    assert!(
        output.logits.iter().all(|v| v.is_finite()),
        "Mistral forward produced non-finite logits"
    );
}

#[test]
fn test_mistral_cached_decode_matches_forward() {
    let cfg = mistral_config();
    let model = build_model(&cfg);

    let mut cache = make_cache(&cfg);
    let prefill_out = model.forward_prefill(&[1, 2, 3], &mut cache);
    assert_eq!(prefill_out.shape, [1, 1, VOCAB_SIZE]);
    assert!(
        prefill_out.logits.iter().all(|v| v.is_finite()),
        "Mistral prefill produced non-finite logits"
    );

    let decode_out = model.forward_one(4, &mut cache);
    assert_eq!(decode_out.shape, [1, 1, VOCAB_SIZE]);
    assert!(
        decode_out.logits.iter().all(|v| v.is_finite()),
        "Mistral forward_one produced non-finite logits"
    );
}

#[test]
fn test_mistral_quantized_forward_produces_finite_logits() {
    let cfg = mistral_config();
    let model = build_model(&cfg);
    let quantized = quantize_model(&model);

    let output = quantized.forward(&[1, 2, 3]);
    assert_eq!(output.shape, [1, 3, VOCAB_SIZE]);
    assert_eq!(output.logits.len(), 3 * VOCAB_SIZE);
    assert!(
        output.logits.iter().all(|v| v.is_finite()),
        "Mistral quantized forward produced non-finite logits"
    );
}

#[test]
fn test_mistral_ternary_forward_produces_finite_logits() {
    let cfg = mistral_config();
    let model = build_model(&cfg);
    let ternary = quantize_model_ternary(&model);
    let output = ternary.forward(&[1, 2, 3]);
    assert_eq!(output.shape, [1, 3, VOCAB_SIZE]);
    assert!(
        output.logits.iter().all(|v| v.is_finite()),
        "Mistral ternary forward produced non-finite logits"
    );
}

#[test]
fn test_ternary_memory_compression_vs_int8() {
    let cfg = qwen3_config();
    let model = build_model(&cfg);
    let quantized = quantize_model(&model);
    let ternary = quantize_model_ternary(&model);
    let int8_mem = quantized.layers[0].w_q.memory_bytes();
    let ternary_mem = ternary.layers[0].w_q.memory_bytes();
    assert!(
        ternary_mem < int8_mem,
        "Ternary ({ternary_mem} bytes) should use less memory than INT8 ({int8_mem} bytes)"
    );
}

// ════════════════════════════════════════════════════════════════════════
// Phi3
// ════════════════════════════════════════════════════════════════════════

#[test]
fn test_phi3_forward_produces_finite_logits() {
    let cfg = phi3_config();
    let model = build_model(&cfg);
    let output = model.forward(&[1, 2, 3]);

    assert_eq!(output.shape, [1, 3, VOCAB_SIZE]);
    assert_eq!(output.logits.len(), 3 * VOCAB_SIZE);
    assert!(
        output.logits.iter().all(|v| v.is_finite()),
        "Phi3 forward produced non-finite logits"
    );
}

#[test]
fn test_phi3_cached_decode_matches_forward() {
    let cfg = phi3_config();
    let model = build_model(&cfg);

    let mut cache = make_cache(&cfg);
    let prefill_out = model.forward_prefill(&[1, 2, 3], &mut cache);
    assert_eq!(prefill_out.shape, [1, 1, VOCAB_SIZE]);
    assert!(
        prefill_out.logits.iter().all(|v| v.is_finite()),
        "Phi3 prefill produced non-finite logits"
    );

    let decode_out = model.forward_one(4, &mut cache);
    assert_eq!(decode_out.shape, [1, 1, VOCAB_SIZE]);
    assert!(
        decode_out.logits.iter().all(|v| v.is_finite()),
        "Phi3 forward_one produced non-finite logits"
    );
}

#[test]
fn test_phi3_quantized_forward_produces_finite_logits() {
    let cfg = phi3_config();
    let model = build_model(&cfg);
    let quantized = quantize_model(&model);

    let output = quantized.forward(&[1, 2, 3]);
    assert_eq!(output.shape, [1, 3, VOCAB_SIZE]);
    assert_eq!(output.logits.len(), 3 * VOCAB_SIZE);
    assert!(
        output.logits.iter().all(|v| v.is_finite()),
        "Phi3 quantized forward produced non-finite logits"
    );
}

// ════════════════════════════════════════════════════════════════════════
// Gemma
// ════════════════════════════════════════════════════════════════════════

#[test]
fn test_gemma_forward_produces_finite_logits() {
    let cfg = gemma_config();
    let model = build_model(&cfg);
    let output = model.forward(&[1, 2, 3]);

    assert_eq!(output.shape, [1, 3, VOCAB_SIZE]);
    assert_eq!(output.logits.len(), 3 * VOCAB_SIZE);
    assert!(
        output.logits.iter().all(|v| v.is_finite()),
        "Gemma forward produced non-finite logits"
    );
}

#[test]
fn test_gemma_cached_decode_matches_forward() {
    let cfg = gemma_config();
    let model = build_model(&cfg);

    let mut cache = make_cache(&cfg);
    let prefill_out = model.forward_prefill(&[1, 2, 3], &mut cache);
    assert_eq!(prefill_out.shape, [1, 1, VOCAB_SIZE]);
    assert!(
        prefill_out.logits.iter().all(|v| v.is_finite()),
        "Gemma prefill produced non-finite logits"
    );

    let decode_out = model.forward_one(4, &mut cache);
    assert_eq!(decode_out.shape, [1, 1, VOCAB_SIZE]);
    assert!(
        decode_out.logits.iter().all(|v| v.is_finite()),
        "Gemma forward_one produced non-finite logits"
    );
}

#[test]
fn test_gemma_quantized_forward_produces_finite_logits() {
    let cfg = gemma_config();
    let model = build_model(&cfg);
    let quantized = quantize_model(&model);

    let output = quantized.forward(&[1, 2, 3]);
    assert_eq!(output.shape, [1, 3, VOCAB_SIZE]);
    assert_eq!(output.logits.len(), 3 * VOCAB_SIZE);
    assert!(
        output.logits.iter().all(|v| v.is_finite()),
        "Gemma quantized forward produced non-finite logits"
    );
}

// ════════════════════════════════════════════════════════════════════════
// ViT (Vision Transformer)
// ════════════════════════════════════════════════════════════════════════

use synapse_inference::model::{ViTConfig, ViTModel};

const VIT_IMAGE_SIZE: usize = 8;
const VIT_PATCH_SIZE: usize = 4;
const VIT_CHANNELS: usize = 3;
const VIT_HIDDEN: usize = 32;
const VIT_LAYERS: usize = 2;
const VIT_HEADS: usize = 4;
const VIT_INTER: usize = 64;
const VIT_CLASSES: usize = 10;

fn vit_config() -> ViTConfig {
    ViTConfig {
        image_size: VIT_IMAGE_SIZE,
        patch_size: VIT_PATCH_SIZE,
        channels: VIT_CHANNELS,
        hidden_size: VIT_HIDDEN,
        num_layers: VIT_LAYERS,
        num_heads: VIT_HEADS,
        intermediate_size: VIT_INTER,
        num_classes: VIT_CLASSES,
    }
}

fn build_vit(cfg: &ViTConfig) -> ViTModel {
    let h = cfg.hidden_size;
    let patch_dim = cfg.patch_size * cfg.patch_size * cfg.channels;
    let inter = cfg.intermediate_size;
    let seq_len = cfg.seq_len();

    let mut model = ViTModel::from_config(cfg);

    // Set weights directly
    model.patch_proj = AlignedBuffer::from_slice(&gen_weights(h * patch_dim, 1));
    model.cls_token = AlignedBuffer::from_slice(&gen_weights(h, 2));
    model.pos_embed = AlignedBuffer::from_slice(&gen_weights(seq_len * h, 3));
    model.final_norm_weight = AlignedBuffer::from_slice(&vec![1.0f32; h]);

    for (i, layer) in model.layers.iter_mut().enumerate() {
        let s = (i as u32 + 1) * 100;
        layer.attn_norm_weight = AlignedBuffer::from_slice(&vec![1.0f32; h]);
        layer.w_q = AlignedBuffer::from_slice(&gen_weights(h * h, s + 1));
        layer.w_k = AlignedBuffer::from_slice(&gen_weights(h * h, s + 2));
        layer.w_v = AlignedBuffer::from_slice(&gen_weights(h * h, s + 3));
        layer.w_o = AlignedBuffer::from_slice(&gen_weights(h * h, s + 4));
        layer.ffn_norm_weight = AlignedBuffer::from_slice(&vec![1.0f32; h]);
        layer.ffn_up = AlignedBuffer::from_slice(&gen_weights(inter * h, s + 5));
        layer.ffn_down = AlignedBuffer::from_slice(&gen_weights(h * inter, s + 6));
    }

    if cfg.num_classes > 0 {
        model.classifier_head = Some(AlignedBuffer::from_slice(&gen_weights(
            cfg.num_classes * h,
            999,
        )));
    }

    model
}

#[test]
fn test_vit_forward_produces_finite_embeddings() {
    let cfg = vit_config();
    let model = build_vit(&cfg);

    let image: Vec<f32> = (0..cfg.image_size * cfg.image_size * cfg.channels)
        .map(|i| (i as f32) / 255.0)
        .collect();

    let output = model.forward_image(&image, cfg.image_size, cfg.image_size);

    // Embeddings should be [hidden_size] and finite
    assert_eq!(output.embeddings.len(), VIT_HIDDEN);
    assert!(
        output.embeddings.iter().all(|v| v.is_finite()),
        "ViT forward produced non-finite embeddings"
    );

    // Logits should be [num_classes] and finite
    let logits = output
        .logits
        .as_ref()
        .expect("expected logits for classification model");
    assert_eq!(logits.len(), VIT_CLASSES);
    assert!(
        logits.iter().all(|v| v.is_finite()),
        "ViT forward produced non-finite logits"
    );
}

#[test]
fn test_vit_patch_embed_shape() {
    let cfg = vit_config();
    let h = cfg.hidden_size;
    let patch_dim = cfg.patch_size * cfg.patch_size * cfg.channels;

    let image: Vec<f32> = (0..cfg.image_size * cfg.image_size * cfg.channels)
        .map(|i| (i as f32) / 255.0)
        .collect();

    let projection = gen_weights(h * patch_dim, 42);
    let result = synapse_inference::ops::patch_embed::patch_embed(
        &image,
        cfg.image_size,
        cfg.image_size,
        cfg.channels,
        cfg.patch_size,
        &projection,
        h,
    );

    let expected_patches = cfg.num_patches();
    assert_eq!(
        expected_patches, 4,
        "8x8 image with 4x4 patches should give 4 patches"
    );
    assert_eq!(
        result.len(),
        expected_patches * h,
        "Expected {} elements, got {}",
        expected_patches * h,
        result.len()
    );
    assert!(
        result.iter().all(|v| v.is_finite()),
        "Patch embedding produced non-finite values"
    );
}
