//! Quantization speedup benchmark: compare INT8 vs f32 throughput.
//! Reports the speedup ratio and verifies both produce valid output.

use std::time::Instant;

use synapse_inference::config::*;
use synapse_inference::model::ModelBuilder;
use synapse_inference::quantization::quantize_model;

fn bench_config() -> ModelConfig {
    ModelConfig {
        name: "Qwen3-QuantSpeedupBench".to_string(),
        architecture: ArchitectureConfig {
            hidden_size: 128,
            num_layers: 4,
            vocab_size: 512,
            max_sequence_length: 128,
            tie_word_embeddings: true,
        },
        attention: AttentionConfig::GQA {
            num_heads: 4,
            num_kv_heads: 2,
            head_dim: 32,
        },
        norm: NormConfig::RMSNorm { eps: 1e-6 },
        ffn: FFNConfig::SwiGLU {
            intermediate_size: 256,
        },
        position: PositionConfig::RoPE {
            base: 10000.0,
            max_position_embeddings: 128,
        },
        quantization: QuantConfig::F32,
    }
}

fn gen_weights(len: usize, seed: u32) -> Vec<f32> {
    (0..len)
        .map(|i| {
            let x = ((i as u32).wrapping_mul(2654435761).wrapping_add(seed)) as f32;
            (x / u32::MAX as f32) * 0.36 - 0.18
        })
        .collect()
}

fn fill_model_weights(model: &mut synapse_inference::model::CausalLM) {
    let cfg = &model.config;
    let h = cfg.architecture.hidden_size;
    let vocab = cfg.architecture.vocab_size;
    let q_dim = cfg.attention.num_heads() * cfg.attention.head_dim();
    let kv_dim = cfg.attention.num_kv_heads() * cfg.attention.head_dim();
    let inter = cfg.ffn.intermediate_size();

    model.embed_tokens = gen_weights(vocab * h, 1);
    model.final_norm_weight = vec![1.0f32; h];

    for (i, layer) in model.layers.iter_mut().enumerate() {
        let s = (i as u32 + 1) * 100;
        layer.attn_norm_weight = vec![1.0f32; h];
        layer.w_q = gen_weights(q_dim * h, s + 1);
        layer.w_k = gen_weights(kv_dim * h, s + 2);
        layer.w_v = gen_weights(kv_dim * h, s + 3);
        layer.w_o = gen_weights(h * q_dim, s + 4);
        layer.ffn_norm_weight = vec![1.0f32; h];
        layer.ffn_gate = gen_weights(inter * h, s + 5);
        layer.ffn_up = gen_weights(inter * h, s + 6);
        layer.ffn_down = gen_weights(h * inter, s + 7);
    }
}

#[test]
fn quantization_speedup_int8_vs_f32() {
    let cfg = bench_config();
    let mut model = ModelBuilder::from_config(&cfg);
    fill_model_weights(&mut model);

    let quantized = quantize_model(&model);
    let vocab = cfg.architecture.vocab_size;

    let prompt = vec![1u32, 2, 3, 4, 5];
    let num_decode_steps = 15;
    let warmup = 3;

    // ── f32 benchmark ────────────────────────────────────────────────
    // Warmup
    for _ in 0..warmup {
        let _ = model.forward(&prompt);
    }

    let mut f32_tokens = prompt.clone();
    let start = Instant::now();
    for _ in 0..num_decode_steps {
        let output = model.forward(&f32_tokens);
        let seq_len = output.shape[1];
        let last_logits = &output.logits[(seq_len - 1) * vocab..seq_len * vocab];
        let token = last_logits
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(i, _)| i as u32)
            .unwrap();
        f32_tokens.push(token);
    }
    let f32_elapsed = start.elapsed();
    let f32_tps = num_decode_steps as f64 / f32_elapsed.as_secs_f64();

    // ── INT8 benchmark ───────────────────────────────────────────────
    // Warmup
    for _ in 0..warmup {
        let _ = quantized.forward(&prompt);
    }

    let mut int8_tokens = prompt.clone();
    let start = Instant::now();
    for _ in 0..num_decode_steps {
        let output = quantized.forward(&int8_tokens);
        let seq_len = output.shape[1];
        let last_logits = &output.logits[(seq_len - 1) * vocab..seq_len * vocab];
        let token = last_logits
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(i, _)| i as u32)
            .unwrap();
        int8_tokens.push(token);
    }
    let int8_elapsed = start.elapsed();
    let int8_tps = num_decode_steps as f64 / int8_elapsed.as_secs_f64();

    // ── Report ───────────────────────────────────────────────────────
    let speedup = int8_tps / f32_tps;
    eprintln!("f32  decode: {:.1} tok/s ({:.3}s)", f32_tps, f32_elapsed.as_secs_f64());
    eprintln!("INT8 decode: {:.1} tok/s ({:.3}s)", int8_tps, int8_elapsed.as_secs_f64());
    eprintln!("Speedup ratio: {:.2}x", speedup);

    // Both must produce valid output
    assert_eq!(f32_tokens.len(), prompt.len() + num_decode_steps);
    assert_eq!(int8_tokens.len(), prompt.len() + num_decode_steps);

    // Both must produce finite logits
    let f32_output = model.forward(&f32_tokens);
    assert!(f32_output.logits.iter().all(|v| v.is_finite()));

    let int8_output = quantized.forward(&int8_tokens);
    assert!(int8_output.logits.iter().all(|v| v.is_finite()));

    // Report speedup (INT8 may or may not be faster depending on
    // SIMD optimization; with scalar code they are approximately equal)
    eprintln!(
        "INT8 is {:.2}x {} than f32",
        if speedup >= 1.0 { speedup } else { 1.0 / speedup },
        if speedup >= 1.0 { "faster" } else { "slower" }
    );
}
