//! Config-driven assembly test: Qwen3 + LLaMA configs build different architectures
//! from the same engine. Assembly must complete in <= 2 seconds.

use std::time::Instant;

use synapse_inference::config::*;
use synapse_inference::models::ModelBuilder;

const QWEN3_JSON: &str = include_str!("../../configs/qwen3_0.6b.json");
const LLAMA_JSON: &str = include_str!("../../configs/llama3.2_1b.json");
const MISTRAL_JSON: &str = include_str!("../../configs/mistral_7b.json");

/// Qwen3 and LLaMA produce structurally different models from the same builder.
#[test]
fn config_assembly_different_architectures() {
    let qwen_cfg = ModelConfig::from_json(QWEN3_JSON).unwrap();
    let llama_cfg = ModelConfig::from_json(LLAMA_JSON).unwrap();

    let qwen_model = ModelBuilder::from_config(&qwen_cfg);
    let llama_model = ModelBuilder::from_config(&llama_cfg);

    // Different layer counts
    assert_eq!(qwen_model.layers.len(), 28);
    assert_eq!(llama_model.layers.len(), 16);
    assert_ne!(qwen_model.layers.len(), llama_model.layers.len());

    // Different attention geometry
    assert_eq!(qwen_model.layers[0].attention.num_heads(), 16);
    assert_eq!(llama_model.layers[0].attention.num_heads(), 32);

    // Same attention type (both GQA) but different head counts
    assert_eq!(qwen_model.layers[0].attention.name(), "GQA");
    assert_eq!(llama_model.layers[0].attention.name(), "GQA");
    assert_eq!(qwen_model.layers[0].attention.num_kv_heads(), 8);
    assert_eq!(llama_model.layers[0].attention.num_kv_heads(), 8);

    // Different FFN sizes
    assert_eq!(qwen_model.layers[0].ffn.intermediate_size(), 3072);
    assert_eq!(llama_model.layers[0].ffn.intermediate_size(), 8192);

    // Both use RMSNorm but with different epsilon
    assert_eq!(qwen_model.layers[0].attn_norm.name(), "RMSNorm");
    assert_eq!(llama_model.layers[0].attn_norm.name(), "RMSNorm");

    // Different param counts
    let qwen_params = qwen_model.param_count();
    let llama_params = llama_model.param_count();
    assert_ne!(qwen_params, llama_params);
    eprintln!("Qwen3 params: {qwen_params}, LLaMA params: {llama_params}");
}

/// Mistral uses SlidingWindow attention — different from GQA.
#[test]
fn config_assembly_mistral_sliding_window() {
    let cfg = ModelConfig::from_json(MISTRAL_JSON).unwrap();
    let model = ModelBuilder::from_config(&cfg);

    assert_eq!(model.layers.len(), 32);
    assert_eq!(model.layers[0].attention.name(), "SlidingWindow");
    assert_eq!(model.layers[0].attention.num_heads(), 32);
    assert_eq!(model.layers[0].attention.num_kv_heads(), 8);
    assert_eq!(model.layers[0].attention.head_dim(), 128);

    // Mistral does NOT tie embeddings
    assert!(model.lm_head_weight.is_some());
}

/// Assembly of Qwen3-0.6B from config must complete in <= 2 seconds.
#[test]
fn config_assembly_qwen3_under_2_seconds() {
    let cfg = ModelConfig::from_json(QWEN3_JSON).unwrap();

    let start = Instant::now();
    let _model = ModelBuilder::from_config(&cfg);
    let elapsed = start.elapsed();

    eprintln!("Qwen3 assembly: {:.3}s", elapsed.as_secs_f64());
    assert!(
        elapsed.as_secs_f64() < 2.0,
        "Qwen3 assembly took {:.3}s, expected < 2s",
        elapsed.as_secs_f64()
    );
}

/// Assembly of LLaMA-3.2-1B from config must complete in <= 2 seconds.
#[test]
fn config_assembly_llama_under_2_seconds() {
    let cfg = ModelConfig::from_json(LLAMA_JSON).unwrap();

    let start = Instant::now();
    let _model = ModelBuilder::from_config(&cfg);
    let elapsed = start.elapsed();

    eprintln!("LLaMA assembly: {:.3}s", elapsed.as_secs_f64());
    assert!(
        elapsed.as_secs_f64() < 2.0,
        "LLaMA assembly took {:.3}s, expected < 2s",
        elapsed.as_secs_f64()
    );
}

/// Both Qwen3 and LLaMA produce correct param counts from their configs.
#[test]
fn config_assembly_param_count_matches_config() {
    for (name, json) in [("Qwen3", QWEN3_JSON), ("LLaMA", LLAMA_JSON)] {
        let cfg = ModelConfig::from_json(json).unwrap();
        let model = ModelBuilder::from_config(&cfg);

        let h = cfg.architecture.hidden_size;
        let vocab = cfg.architecture.vocab_size;
        let q_dim = cfg.attention.num_heads() * cfg.attention.head_dim();
        let kv_dim = cfg.attention.num_kv_heads() * cfg.attention.head_dim();
        let inter = cfg.ffn.intermediate_size();
        let nl = cfg.architecture.num_layers;

        let embed = vocab * h;
        let per_layer = 2 * h + q_dim * h + kv_dim * h + kv_dim * h + h * q_dim + 3 * inter * h;
        let norm = h;
        let lm_head = if cfg.architecture.tie_word_embeddings {
            0
        } else {
            vocab * h
        };
        let expected = embed + nl * per_layer + norm + lm_head;

        assert_eq!(
            model.param_count(),
            expected,
            "{name} param count mismatch: got {}, expected {expected}",
            model.param_count()
        );
    }
}

/// All three configs round-trip through JSON serialization.
#[test]
fn config_assembly_json_round_trip() {
    for (name, json) in [
        ("Qwen3", QWEN3_JSON),
        ("LLaMA", LLAMA_JSON),
        ("Mistral", MISTRAL_JSON),
    ] {
        let original = ModelConfig::from_json(json).unwrap();
        let serialized = original.to_json().unwrap();
        let deserialized = ModelConfig::from_json(&serialized).unwrap();
        assert_eq!(original, deserialized, "{name} config round-trip failed");
    }
}
