//! Load LFM2.5-350M from GGUF and benchmark CPU vs Metal GPU inference.
//!
//! Usage: cargo run --release --features metal -p synapse-inference --example lfm25_inference -- <path-to-gguf>

use std::path::Path;

use synapse_inference::models::ssm::hybrid::config::HybridConfig;
use synapse_inference::models::ssm::hybrid::model::HybridModel;
use synapse_inference::models::traits::Model;
use synapse_inference::models::traits::ModelState;
use synapse_inference::weight_loading::{load_gguf, load_gguf_with_raw_q4};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let gguf_path = args.get(1).expect("Usage: lfm25_inference <path-to-gguf>");

    println!("Loading LFM2.5-350M from {gguf_path}...");
    let start = std::time::Instant::now();

    let (weights, q4_raw) = load_gguf_with_raw_q4(Path::new(gguf_path)).expect("Failed to load GGUF");
    println!("  Loaded {} tensors ({} Q4 raw) in {:.2}s",
        weights.len(), q4_raw.len(), start.elapsed().as_secs_f32());

    let config = HybridConfig::lfm25_350m();
    let max_kv_seq = 2048;
    let model = HybridModel::from_weights_lfm25(config, &weights, max_kv_seq)
        .expect("Failed to build model");
    println!("  Built model in {:.2}s  ({} conv + {} GQA layers)",
        start.elapsed().as_secs_f32(),
        model.config.num_livconv_layers(), model.config.num_gqa_layers());
    println!("  State memory: {} KB", model.state_memory_bytes() / 1024);

    // === CPU Benchmark ===
    println!("\n=== CPU (Accelerate BLAS) ===");
    let mut state = ModelState::Recurrent;

    // Warmup
    model.reset_state();
    let _ = model.forward_prefill(&[1, 2, 3], &mut state);

    // Prefill
    let prompt: Vec<u32> = (1..=128).collect();
    model.reset_state();
    let t0 = std::time::Instant::now();
    let out = model.forward_prefill(&prompt, &mut state);
    let prefill_ms = t0.elapsed().as_secs_f64() * 1000.0;
    println!("  Prefill pp128: {:.1}ms ({:.1} tok/s)", prefill_ms, 128.0 / (prefill_ms / 1000.0));

    // Decode
    let mut next = out.logits.iter().enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .map(|(i, _)| i as u32).unwrap();
    let n_decode = 64;
    let t0 = std::time::Instant::now();
    for _ in 0..n_decode {
        let out = model.forward_one(next, &mut state);
        next = out.logits.iter().enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(i, _)| i as u32).unwrap();
    }
    let decode_ms = t0.elapsed().as_secs_f64() * 1000.0;
    let cpu_tps = n_decode as f64 / (decode_ms / 1000.0);
    println!("  Decode  tg{n_decode}: {:.1}ms ({:.1} tok/s)", decode_ms, cpu_tps);

    // === Metal GPU Benchmark ===
    #[cfg(feature = "metal")]
    {
        use synapse_inference::metal::MetalBackend;
        use synapse_inference::metal::hybrid_gpu_buffers::MetalHybridBuffers;

        println!("\n=== Metal GPU ===");
        let backend = MetalBackend::new().expect("Failed to init Metal");
        println!("  GPU: {}", backend.device.name());

        // Build GPU buffers (upload f32 + raw Q4 weights)
        let t0 = std::time::Instant::now();
        // Try Q4 first, fall back to f32 if Q4 produces NaN
        let use_q4 = std::env::var("NO_Q4_GPU").is_err();
        let mut gpu_bufs = if use_q4 {
            MetalHybridBuffers::from_hybrid_model_with_q4(&model, max_kv_seq, &backend, &q4_raw)
        } else {
            MetalHybridBuffers::from_hybrid_model(&model, max_kv_seq, &backend)
        };
        println!("  Uploaded weights in {:.2}s (Q4 GPU: {})", t0.elapsed().as_secs_f32(), use_q4);

        // CPU prefill first (populates state), then copy to GPU
        model.reset_state();
        let prompt: Vec<u32> = (1..=128).collect();
        let out = model.forward_prefill(&prompt, &mut state);
        gpu_bufs.populate_from_cpu_state(&model, &backend.device);
        println!("  Prefill on CPU + state copied to GPU");

        // GPU decode
        let mut next = out.logits.iter().enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(i, _)| i as u32).unwrap();

        // Warmup — check first token for correctness
        let out = model.forward_one_gpu_resident(next, &mut gpu_bufs, &backend);
        let finite = out.logits.iter().filter(|v| v.is_finite()).count();
        println!("  First GPU token: {} finite / {} total logits, logits[0..5]: [{:.4}, {:.4}, {:.4}, {:.4}, {:.4}]",
            finite, out.logits.len(), out.logits[0], out.logits[1], out.logits[2], out.logits[3], out.logits[4]);
        if finite == 0 {
            println!("  ERROR: All logits are NaN/Inf — GPU forward has a bug. Skipping GPU benchmark.");
            println!("\n=== llama-bench baseline ===");
            println!("  llama.cpp (Metal GPU):  pp128 = 11,367 tok/s  |  tg32 = 357 tok/s");
            return;
        }
        next = out.logits.iter().enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i as u32).unwrap_or(0);
        // 2 more warmup
        for _ in 0..2 {
            let out = model.forward_one_gpu_resident(next, &mut gpu_bufs, &backend);
            next = out.logits.iter().enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(i, _)| i as u32).unwrap_or(0);
        }

        let t0 = std::time::Instant::now();
        for _ in 0..n_decode {
            let out = model.forward_one_gpu_resident(next, &mut gpu_bufs, &backend);
            next = out.logits.iter().enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(i, _)| i as u32).unwrap_or(0);
        }
        let gpu_ms = t0.elapsed().as_secs_f64() * 1000.0;
        let gpu_tps = n_decode as f64 / (gpu_ms / 1000.0);
        println!("  Decode  tg{n_decode}: {:.1}ms ({:.1} tok/s)", gpu_ms, gpu_tps);
        println!("  Speedup vs CPU: {:.1}x", gpu_tps / cpu_tps);
    }

    println!("\n=== llama-bench baseline ===");
    println!("  llama.cpp (Metal GPU):  pp128 = 11,367 tok/s  |  tg32 = 357 tok/s");
}
