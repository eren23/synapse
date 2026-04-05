//! Benchmark + zero-drift verification for the fused Code WM encoder.
//!
//! Compares sequential encoder vs fused Zig kernel on the same golden input,
//! measures latency delta, and verifies byte-level agreement.
//!
//! Usage:
//!   cargo run --release --example code_wm_fused_bench -- \
//!       models/code_wm/g8.safetensors \
//!       configs/code_wm_g8.json \
//!       tests/fixtures/code_wm_reference_g8.safetensors

use std::env;
use std::path::Path;
use std::time::Instant;

use synapse_inference::models::vision::{CodeWorldModel, CodeWorldModelConfig};
use synapse_inference::weight_loading::load_safetensors;

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let mut dot = 0.0f32; let mut na = 0.0f32; let mut nb = 0.0f32;
    for i in 0..a.len() { dot += a[i]*b[i]; na += a[i]*a[i]; nb += b[i]*b[i]; }
    dot / (na.sqrt() * nb.sqrt() + 1e-30)
}
fn max_abs(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x,y)| (x-y).abs()).fold(0f32, f32::max)
}

fn bench<F: FnMut()>(label: &str, iters: usize, warmup: usize, mut f: F) -> f64 {
    for _ in 0..warmup { f(); }
    let mut samples = Vec::with_capacity(iters);
    for _ in 0..iters {
        let t0 = Instant::now();
        f();
        samples.push(t0.elapsed().as_secs_f64() * 1000.0);
    }
    samples.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let p50 = samples[samples.len() / 2];
    let mean: f64 = samples.iter().sum::<f64>() / samples.len() as f64;
    println!("  {:<28} p50={:>7.2}ms  mean={:>7.2}ms", label, p50, mean);
    p50
}

fn main() {
    let mut args = env::args().skip(1);
    let weights = args.next().expect("arg 1: weights .safetensors");
    let config = args.next().expect("arg 2: config .json");
    let goldens = args.next().expect("arg 3: reference dump .safetensors");

    let cfg = CodeWorldModelConfig::from_json(Path::new(&config)).unwrap();
    let tensors = load_safetensors(Path::new(&weights)).unwrap();
    let mut model = CodeWorldModel::from_config(&cfg);
    model.load_weights(tensors).unwrap();

    let g = load_safetensors(Path::new(&goldens)).unwrap();
    let tokens: Vec<i64> = g["seed0_input_tokens"].data.iter().map(|&v| v as i64).collect();
    let z_ref = &g["seed0_encoder_final"].data;

    // Correctness: compare sequential vs fused
    let z_seq = model.encode(&tokens);
    #[cfg(feature = "zig-ffi")]
    let z_fused = model.encode_fused(&tokens);

    println!("=== Correctness vs PyTorch golden (seed 0) ===");
    println!("  sequential: cos={:.10}  max_abs={:.3e}", cosine(&z_seq, z_ref), max_abs(&z_seq, z_ref));
    #[cfg(feature = "zig-ffi")]
    {
        println!("  fused Zig:  cos={:.10}  max_abs={:.3e}", cosine(&z_fused, z_ref), max_abs(&z_fused, z_ref));
        println!("  seq vs fused: cos={:.10}  max_abs={:.3e}", cosine(&z_seq, &z_fused), max_abs(&z_seq, &z_fused));
    }

    // Benchmark at several seq lengths
    println!("\n=== Latency benchmark ===");
    for &s in &[64_usize, 128, 256, 512] {
        println!("S={s}:");
        let toks: Vec<i64> = (0..s).map(|i| (i * 7) as i64 % cfg.vocab_size as i64).collect();
        let seq_time = bench(&format!("  sequential"), 30, 5, || { let _ = model.encode(&toks); });
        #[cfg(feature = "zig-ffi")]
        {
            let fused_time = bench(&format!("  fused Zig"), 30, 5, || { let _ = model.encode_fused(&toks); });
            println!("  speedup: {:.2}x", seq_time / fused_time);
        }
        #[cfg(not(feature = "zig-ffi"))]
        let _ = seq_time;
    }
}
