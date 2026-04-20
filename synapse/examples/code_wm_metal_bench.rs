//! Benchmark: Metal GPU vs CPU (sequential + fused Zig) for CodeWM encoder.
//!
//! Usage:
//!   cargo run --release --features metal --example code_wm_metal_bench -- \
//!       models/code_wm/vicreg_promotion.safetensors \
//!       configs/code_wm_vicreg_promotion.json \
//!       tests/fixtures/code_wm_reference_vicreg_promotion.safetensors

use std::env;
use std::path::Path;
use std::time::Instant;

use synapse_inference::models::vision::{CodeWorldModel, CodeWorldModelConfig};
use synapse_inference::weight_loading::load_safetensors;

#[cfg(feature = "metal")]
use synapse_inference::metal::{MetalBackend, MetalCodeWMState};
#[cfg(feature = "metal")]
use synapse_inference::metal::code_wm_forward::code_wm_encode_metal;

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let mut dot = 0.0f32; let mut na = 0.0f32; let mut nb = 0.0f32;
    for i in 0..a.len() { dot += a[i]*b[i]; na += a[i]*a[i]; nb += b[i]*b[i]; }
    dot / (na.sqrt() * nb.sqrt() + 1e-30)
}

fn max_abs(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x,y)| (x-y).abs()).fold(0f32, f32::max)
}

fn bench<F: FnMut() -> Vec<f32>>(label: &str, iters: usize, warmup: usize, mut f: F) -> (f64, Vec<f32>) {
    let mut last = vec![];
    for _ in 0..warmup { last = f(); }
    let mut samples = Vec::with_capacity(iters);
    for _ in 0..iters {
        let t0 = Instant::now();
        last = f();
        samples.push(t0.elapsed().as_secs_f64() * 1000.0);
    }
    samples.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let p50 = samples[samples.len() / 2];
    let mean: f64 = samples.iter().sum::<f64>() / samples.len() as f64;
    println!("  {:<28} p50={:>7.2}ms  mean={:>7.2}ms", label, p50, mean);
    (p50, last)
}

fn main() {
    let mut args = env::args().skip(1);
    let weights = args.next().expect("arg 1: weights .safetensors");
    let config = args.next().expect("arg 2: config .json");
    let goldens = args.next().expect("arg 3: reference .safetensors");

    let cfg = CodeWorldModelConfig::from_json(Path::new(&config)).unwrap();
    let tensors = load_safetensors(Path::new(&weights)).unwrap();
    let mut model = CodeWorldModel::from_config(&cfg);
    model.load_weights(tensors).unwrap();

    let g = load_safetensors(Path::new(&goldens)).unwrap();
    let tokens_golden: Vec<i64> = g["seed0_input_tokens"].data.iter().map(|&v| v as i64).collect();
    let z_ref = &g["seed0_encoder_final"].data;

    // Build the initial sequence (embedding + CLS + PE) via the model's encode path
    // We need the pre-attention sequence for the Metal kernel.
    // For now, use the model's encode() as reference and benchmark the full encode.

    println!("=== Correctness vs PyTorch golden (seed 0) ===");
    let z_seq = model.encode(&tokens_golden);
    println!("  sequential:  cos={:.10}  max_abs={:.3e}", cosine(&z_seq, z_ref), max_abs(&z_seq, z_ref));

    #[cfg(feature = "zig-ffi")]
    {
        let z_fused = model.encode_fused(&tokens_golden);
        println!("  fused Zig:   cos={:.10}  max_abs={:.3e}", cosine(&z_fused, z_ref), max_abs(&z_fused, z_ref));
    }

    #[cfg(feature = "metal")]
    {
        let backend = MetalBackend::new().expect("Metal GPU init failed");
        println!("  Metal GPU:   (testing at multiple seq lengths below)");
        println!("\n=== Latency: CPU sequential vs fused Zig vs Metal GPU ===");

        for &s in &[64_usize, 128, 256, 512] {
            println!("S={s}:");
            let toks: Vec<i64> = (0..s).map(|i| (i * 7) as i64 % cfg.vocab_size as i64).collect();

            let (seq_time, z_seq) = bench("sequential", 30, 5, || model.encode(&toks));

            #[cfg(feature = "zig-ffi")]
            let (fused_time, z_fused) = bench("fused Zig", 30, 5, || model.encode_fused(&toks));

            // Metal: build state for this seq_len, then benchmark
            let seq_with_cls = s + 1;
            let metal_state = MetalCodeWMState::from_model(&model, seq_with_cls, &backend);

            // Build the input sequence manually (same as model.encode but stop before loops)
            let d = cfg.model_dim;
            let mut input_seq = vec![0.0f32; seq_with_cls * d];
            input_seq[..d].copy_from_slice(&model.cls_token);
            for (i, &tok) in toks.iter().enumerate() {
                let t = tok as usize;
                let src = &model.token_embedding[t * d..(t + 1) * d];
                input_seq[(i + 1) * d..(i + 2) * d].copy_from_slice(src);
            }
            for i in 0..seq_with_cls {
                let off = i * d;
                for j in 0..d { input_seq[off + j] += model.pos_enc[off + j]; }
            }

            let (metal_time, z_metal_raw) = bench("Metal GPU", 30, 5, || {
                code_wm_encode_metal(&metal_state, &input_seq, &backend)
            });

            // Metal returns raw encoder output (before readout + final LN).
            // Apply readout + final LN to compare with sequential.
            let pooled: Vec<f32> = match cfg.pool_mode {
                synapse_inference::models::vision::PoolMode::Cls => z_metal_raw[..d].to_vec(),
                synapse_inference::models::vision::PoolMode::Attn => model
                    .attn_pool.as_ref().unwrap().forward(&z_metal_raw, seq_with_cls),
            };
            let z_metal = synapse_inference::ops::pure_rust_ops::layernorm_with_bias(
                &pooled, &model.encoder_final_norm.weight, &model.encoder_final_norm.bias,
                model.config.layernorm_eps, d,
            );

            println!("  Metal vs seq: cos={:.10}  max_abs={:.3e}",
                     cosine(&z_metal, &z_seq), max_abs(&z_metal, &z_seq));
            #[cfg(feature = "zig-ffi")]
            println!("  speedup vs fused: {:.2}x", fused_time / metal_time);
            println!("  speedup vs seq:   {:.2}x", seq_time / metal_time);
        }
    }

    #[cfg(not(feature = "metal"))]
    {
        println!("\nMetal not enabled. Build with --features metal.");
    }
}
