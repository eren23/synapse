//! Quantization speedup benchmark: compare INT8 vs f32 throughput.
//! Reports the speedup ratio and verifies both produce valid output.

use std::time::Instant;

use synapse_inference::config::*;
use synapse_inference::model::ModelBuilder;
use synapse_inference::quantization::quantize_model;
use synapse_inference::weight_loading::AlignedBuffer;

extern crate synapse_sys;

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
            style: Default::default(),
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

    model.embed_tokens = AlignedBuffer::from_vec(gen_weights(vocab * h, 1));
    model.final_norm_weight = AlignedBuffer::from_vec(vec![1.0f32; h]);

    for (i, layer) in model.layers.iter_mut().enumerate() {
        let s = (i as u32 + 1) * 100;
        layer.attn_norm_weight = AlignedBuffer::from_vec(vec![1.0f32; h]);
        layer.w_q = AlignedBuffer::from_vec(gen_weights(q_dim * h, s + 1));
        layer.w_k = AlignedBuffer::from_vec(gen_weights(kv_dim * h, s + 2));
        layer.w_v = AlignedBuffer::from_vec(gen_weights(kv_dim * h, s + 3));
        layer.w_o = AlignedBuffer::from_vec(gen_weights(h * q_dim, s + 4));
        layer.ffn_norm_weight = AlignedBuffer::from_vec(vec![1.0f32; h]);
        layer.ffn_gate = AlignedBuffer::from_vec(gen_weights(inter * h, s + 5));
        layer.ffn_up = AlignedBuffer::from_vec(gen_weights(inter * h, s + 6));
        layer.ffn_down = AlignedBuffer::from_vec(gen_weights(h * inter, s + 7));
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

/// Isolated matmul benchmark: INT8 GEMM vs f32 SGEMM at real model dimensions.
///
/// Tests M=1 (single-token decode) at realistic K/N for SwiGLU FFN layers.
/// This isolates the kernel performance from overhead in the full model forward.
#[test]
fn isolated_matmul_int8_vs_f32() {
    let warmup = 10;
    let iterations = 200;

    // Real model dimensions: FFN gate projection in Qwen3-0.6B
    let configs = [
        ("FFN gate  (1×1024→3072)", 1, 1024, 3072),
        ("FFN down  (1×3072→1024)", 1, 3072, 1024),
        ("Attn Q    (1×1024→1024)", 1, 1024, 1024),
        ("Prefill   (128×1024→3072)", 128, 1024, 3072),
    ];

    for (label, m, k, n) in &configs {
        // Generate f32 data. B is [N, K] for f32 SGEMM (transposed internally).
        let a_f32: Vec<f32> = gen_weights(m * k, 42);
        let b_f32: Vec<f32> = gen_weights(n * k, 43);

        // For INT8: generate random i8 data and scales directly.
        // Layout: A_int8 [M, K], B_int8 [K, N], scales_a [M], scales_b [N].
        let a_int8: Vec<i8> = (0..(m * k)).map(|i| ((i * 7 + 13) % 256) as i8).collect();
        let b_int8: Vec<i8> = (0..(k * n)).map(|i| ((i * 11 + 17) % 256) as i8).collect();
        let scales_a: Vec<f32> = (0..*m).map(|i| 0.01 + 0.001 * i as f32).collect();
        let scales_b: Vec<f32> = (0..*n).map(|i| 0.01 + 0.001 * i as f32).collect();

        // f32 SGEMM via FFI (same path as inference)
        let sgemm = |a: &[f32], b: &[f32], m: usize, k: usize, n: usize| -> Vec<f32> {
            let mut c = vec![0.0f32; m * n];
            unsafe {
                synapse_sys::syn_sgemm(
                    m, n, k,
                    a.as_ptr(), k, 0,
                    b.as_ptr(), k, 1, // B transposed (same as matmul_t)
                    c.as_mut_ptr(), n,
                );
            }
            c
        };

        // Warmup f32
        for _ in 0..warmup {
            sgemm(&a_f32, &b_f32, *m, *k, *n);
        }
        // Bench f32 (SIMD tiled SGEMM)
        let start = Instant::now();
        for _ in 0..iterations {
            let _ = sgemm(&a_f32, &b_f32, *m, *k, *n);
        }
        let f32_us = start.elapsed().as_micros() as f64 / iterations as f64;

        // Warmup INT8
        for _ in 0..warmup {
            synapse_core::qgemm_int8(*m, *n, *k, &a_int8, &b_int8, &scales_a, &scales_b).unwrap();
        }
        // Bench INT8 (SIMD tiled INT8 GEMM)
        let start = Instant::now();
        for _ in 0..iterations {
            let _ = synapse_core::qgemm_int8(*m, *n, *k, &a_int8, &b_int8, &scales_a, &scales_b).unwrap();
        }
        let int8_us = start.elapsed().as_micros() as f64 / iterations as f64;

        let speedup = f32_us / int8_us;
        let gflops_f32 = (2.0 * *m as f64 * *n as f64 * *k as f64) / (f32_us * 1e3);
        let gflops_int8 = (2.0 * *m as f64 * *n as f64 * *k as f64) / (int8_us * 1e3);
        eprintln!(
            "{label}: f32={f32_us:.0}μs ({gflops_f32:.1} GFLOPS), INT8={int8_us:.0}μs ({gflops_int8:.1} GFLOPS), speedup={speedup:.2}x"
        );
    }
}
