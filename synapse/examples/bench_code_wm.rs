//! Code WM latency + throughput benchmark.
//!
//! Measures encoder / action / predictor latencies at multiple sequence lengths.
//! Reports p50 / p95 / mean per stage.
//!
//! Usage:
//!   cargo run --release --example bench_code_wm -- \
//!       models/code_wm/g8.safetensors \
//!       configs/code_wm_g8.json \
//!       [iters=100] [warmup=10]
//!
//! Target on M-series laptop (f32, CPU via Zig SIMD + Accelerate):
//!   encoder S=64  : < 2ms p50
//!   encoder S=256 : < 8ms p50
//!   encoder S=512 : < 15ms p50
//!   predictor     : < 2ms p50

use std::env;
use std::path::Path;
use std::time::Instant;

use synapse_inference::models::vision::{CodeWorldModel, CodeWorldModelConfig};
use synapse_inference::weight_loading::load_safetensors;

fn percentile(xs: &mut [f64], p: f64) -> f64 {
    xs.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let idx = ((xs.len() as f64 - 1.0) * p).round() as usize;
    xs[idx.min(xs.len() - 1)]
}

fn bench<F: FnMut()>(label: &str, iters: usize, warmup: usize, mut f: F) {
    for _ in 0..warmup {
        f();
    }
    let mut samples_ms = Vec::with_capacity(iters);
    for _ in 0..iters {
        let t0 = Instant::now();
        f();
        samples_ms.push(t0.elapsed().as_secs_f64() * 1000.0);
    }
    let mean: f64 = samples_ms.iter().sum::<f64>() / samples_ms.len() as f64;
    let p50 = percentile(&mut samples_ms.clone(), 0.5);
    let p95 = percentile(&mut samples_ms, 0.95);
    let throughput = 1000.0 / mean;
    println!(
        "  {:<28} p50={:>7.3}ms  p95={:>7.3}ms  mean={:>7.3}ms  thr={:>7.1}/s",
        label, p50, p95, mean, throughput
    );
}

fn main() {
    let mut args = env::args().skip(1);
    let weights_path = args.next().expect("arg 1: weights .safetensors");
    let config_path = args.next().expect("arg 2: config .json");
    let iters: usize = args.next().and_then(|s| s.parse().ok()).unwrap_or(100);
    let warmup: usize = args.next().and_then(|s| s.parse().ok()).unwrap_or(10);

    let cfg = CodeWorldModelConfig::from_json(Path::new(&config_path)).unwrap();
    let tensors = load_safetensors(Path::new(&weights_path)).unwrap();
    let mut model = CodeWorldModel::from_config(&cfg);
    let _ = model.load_weights(tensors).unwrap();

    println!("Code WM benchmark — f32, CPU (zig-ffi + Accelerate on macOS)");
    println!("  model_dim={}, num_heads={}, encoder_loops={}, predictor_depth={}, predictor_loops={}",
             cfg.model_dim, cfg.num_heads, cfg.encoder_loops, cfg.predictor_depth, cfg.predictor_loops);
    println!("  iters={iters}, warmup={warmup}\n");

    // Encoder at multiple sequence lengths.
    for &s in &[16_usize, 64, 128, 256, 512] {
        let tokens: Vec<i64> = (0..s).map(|i| (i * 7) as i64 % cfg.vocab_size as i64).collect();
        bench(&format!("encoder S={s}"), iters, warmup, || {
            let _ = model.encode(&tokens);
        });
    }

    // Action encoder.
    let action: Vec<f32> = vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.5];
    bench("action encoder", iters, warmup, || {
        let _ = model.encode_action(&action);
    });

    // Predictor (single step).
    let z_state: Vec<f32> = vec![0.1; cfg.model_dim];
    let z_action: Vec<f32> = vec![0.2; cfg.model_dim];
    bench("predictor (1 step)", iters, warmup, || {
        let _ = model.predict(&z_state, &z_action);
    });

    // Full pipeline: encode tokens (S=64) + encode action + predict.
    let tokens_64: Vec<i64> = (0..64).map(|i| (i * 7) as i64 % cfg.vocab_size as i64).collect();
    bench("full pipeline (S=64)", iters, warmup, || {
        let z = model.encode(&tokens_64);
        let a = model.encode_action(&action);
        let _ = model.predict(&z, &a);
    });
}
