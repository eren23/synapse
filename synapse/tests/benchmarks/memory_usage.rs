//! Memory usage benchmark: verify Qwen3-0.6B model weight footprint.
//!
//! Thresholds:
//! - f32: <= 3 GB
//! - INT8: <= 1.5 GB

use synapse_inference::config::*;

const QWEN3_JSON: &str = include_str!("../../configs/qwen3_0.6b.json");

/// Compute f32 memory footprint analytically from config dimensions.
fn compute_f32_memory_bytes(cfg: &ModelConfig) -> usize {
    let h = cfg.architecture.hidden_size;
    let vocab = cfg.architecture.vocab_size;
    let q_dim = cfg.attention.num_heads() * cfg.attention.head_dim();
    let kv_dim = cfg.attention.num_kv_heads() * cfg.attention.head_dim();
    let inter = cfg.ffn.intermediate_size();
    let nl = cfg.architecture.num_layers;

    let embed = vocab * h;
    let per_layer = 2 * h          // attn_norm + ffn_norm weights
        + q_dim * h                // w_q
        + kv_dim * h               // w_k
        + kv_dim * h               // w_v
        + h * q_dim                // w_o
        + 3 * inter * h;           // SwiGLU: gate + up + down
    let norm = h;
    let lm_head = if cfg.architecture.tie_word_embeddings { 0 } else { vocab * h };

    let total_params = embed + nl * per_layer + norm + lm_head;
    total_params * std::mem::size_of::<f32>()
}

/// Compute INT8 memory footprint analytically:
/// - Embedding stays f32
/// - Linear weights become INT8 (1 byte each) + per-channel f32 scales
/// - Norm weights stay f32
/// - LM head stays f32
fn compute_int8_memory_bytes(cfg: &ModelConfig) -> usize {
    let h = cfg.architecture.hidden_size;
    let vocab = cfg.architecture.vocab_size;
    let q_dim = cfg.attention.num_heads() * cfg.attention.head_dim();
    let kv_dim = cfg.attention.num_kv_heads() * cfg.attention.head_dim();
    let inter = cfg.ffn.intermediate_size();
    let nl = cfg.architecture.num_layers;
    let sz_f32 = std::mem::size_of::<f32>();
    let sz_i8 = std::mem::size_of::<i8>();

    // Embedding: f32
    let embed = vocab * h * sz_f32;

    // Per layer:
    // - Norm weights: f32 (2 * h)
    // - Linear weights: INT8 (weights) + f32 (scales per output channel)
    let per_layer_norms = 2 * h * sz_f32;
    let per_layer_linear =
        // w_q: [q_dim, h] INT8 + [q_dim] f32 scales
        q_dim * h * sz_i8 + q_dim * sz_f32
        // w_k: [kv_dim, h] INT8 + [kv_dim] f32 scales
        + kv_dim * h * sz_i8 + kv_dim * sz_f32
        // w_v: [kv_dim, h] INT8 + [kv_dim] f32 scales
        + kv_dim * h * sz_i8 + kv_dim * sz_f32
        // w_o: [h, q_dim] INT8 + [h] f32 scales
        + h * q_dim * sz_i8 + h * sz_f32
        // ffn_gate: [inter, h] INT8 + [inter] f32 scales
        + inter * h * sz_i8 + inter * sz_f32
        // ffn_up: [inter, h] INT8 + [inter] f32 scales
        + inter * h * sz_i8 + inter * sz_f32
        // ffn_down: [h, inter] INT8 + [h] f32 scales
        + h * inter * sz_i8 + h * sz_f32;

    // Final norm: f32
    let norm = h * sz_f32;

    // LM head: f32 (or tied = 0)
    let lm_head = if cfg.architecture.tie_word_embeddings { 0 } else { vocab * h * sz_f32 };

    embed + nl * (per_layer_norms + per_layer_linear) + norm + lm_head
}

#[test]
fn memory_usage_f32_under_3gb() {
    let cfg = ModelConfig::from_json(QWEN3_JSON).unwrap();
    let f32_bytes = compute_f32_memory_bytes(&cfg);
    let f32_gb = f32_bytes as f64 / (1024.0 * 1024.0 * 1024.0);

    eprintln!("Qwen3-0.6B f32 memory: {f32_bytes} bytes = {f32_gb:.2} GB");
    eprintln!("  Params: {}", f32_bytes / 4);

    assert!(
        f32_gb <= 3.0,
        "f32 memory {f32_gb:.2} GB exceeds 3 GB threshold"
    );
}

#[test]
fn memory_usage_int8_under_1_5gb() {
    let cfg = ModelConfig::from_json(QWEN3_JSON).unwrap();
    let int8_bytes = compute_int8_memory_bytes(&cfg);
    let int8_gb = int8_bytes as f64 / (1024.0 * 1024.0 * 1024.0);

    eprintln!("Qwen3-0.6B INT8 memory: {int8_bytes} bytes = {int8_gb:.2} GB");

    assert!(
        int8_gb <= 1.5,
        "INT8 memory {int8_gb:.2} GB exceeds 1.5 GB threshold"
    );
}

#[test]
fn memory_usage_int8_smaller_than_f32() {
    let cfg = ModelConfig::from_json(QWEN3_JSON).unwrap();
    let f32_bytes = compute_f32_memory_bytes(&cfg);
    let int8_bytes = compute_int8_memory_bytes(&cfg);

    let ratio = int8_bytes as f64 / f32_bytes as f64;
    eprintln!(
        "INT8/f32 memory ratio: {ratio:.3} ({int8_bytes} / {f32_bytes})"
    );

    assert!(
        int8_bytes < f32_bytes,
        "INT8 memory ({int8_bytes}) should be less than f32 ({f32_bytes})"
    );

    // INT8 should be roughly 25-50% of f32 (weight-only quantization)
    assert!(
        ratio < 0.60,
        "INT8/f32 ratio {ratio:.3} too high (expected < 0.60)"
    );
}

/// Verify memory computation matches the actual model allocation.
#[test]
fn memory_usage_analytical_matches_actual() {
    // Use a small config to avoid allocating GBs in the test
    let cfg = ModelConfig {
        name: "TinyMemTest".to_string(),
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
        },
        quantization: QuantConfig::F32,
    };

    let analytical = compute_f32_memory_bytes(&cfg);

    // Build model with actual weights and measure
    let mut model = synapse_inference::model::ModelBuilder::from_config(&cfg);

    // Fill weights
    let h = cfg.architecture.hidden_size;
    let vocab = cfg.architecture.vocab_size;
    let q_dim = cfg.attention.num_heads() * cfg.attention.head_dim();
    let kv_dim = cfg.attention.num_kv_heads() * cfg.attention.head_dim();
    let inter = cfg.ffn.intermediate_size();

    model.embed_tokens = vec![0.0f32; vocab * h];
    model.final_norm_weight = vec![1.0f32; h];
    for layer in model.layers.iter_mut() {
        layer.attn_norm_weight = vec![1.0f32; h];
        layer.w_q = vec![0.0f32; q_dim * h];
        layer.w_k = vec![0.0f32; kv_dim * h];
        layer.w_v = vec![0.0f32; kv_dim * h];
        layer.w_o = vec![0.0f32; h * q_dim];
        layer.ffn_norm_weight = vec![1.0f32; h];
        layer.ffn_gate = vec![0.0f32; inter * h];
        layer.ffn_up = vec![0.0f32; inter * h];
        layer.ffn_down = vec![0.0f32; h * inter];
    }

    let actual = synapse_inference::quantization::f32_model_memory_bytes(&model);

    assert_eq!(
        analytical, actual,
        "Analytical f32 memory ({analytical}) != actual ({actual})"
    );

    // Also verify INT8 analytical matches actual
    let int8_analytical = compute_int8_memory_bytes(&cfg);
    let quantized = synapse_inference::quantization::quantize_model(&model);
    let int8_actual = quantized.memory_bytes();

    assert_eq!(
        int8_analytical, int8_actual,
        "Analytical INT8 memory ({int8_analytical}) != actual ({int8_actual})"
    );
}
