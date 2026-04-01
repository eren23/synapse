//! Benchmark: f32 vs FullyQuantized LEWM inference on hybrid ALAL.
//!
//! Usage: cargo run -p synapse --release --example bench_lewm_quantized -- /tmp/lewm-64d-variants/hybrid_alal

use std::path::Path;
use std::time::Instant;

use synapse_inference::models::{LeWMConfig, LeWorldModel};
use synapse_inference::models::vision::lewm::LeWMBuffers;
use synapse_inference::quantization::quantize_lewm_full;
use synapse_inference::weight_loading::load_safetensors;

fn main() {
    let dir = std::env::args().nth(1).unwrap_or("/tmp/lewm-64d-variants/hybrid_alal".into());
    let config_path = Path::new(&dir).join("config.json");
    let weights_path = Path::new(&dir).join("lejepa_weights.safetensors");

    let config = LeWMConfig::from_json(&config_path).expect("config");
    let mut model = LeWorldModel::from_config(&config);
    let weights = load_safetensors(&weights_path).expect("weights");
    model.load_weights(weights).expect("load");

    println!("Model: {}d latent, {}d encoder, {}e/{}p",
        config.latent_dim, config.encoder_hidden, config.encoder_layers, config.predictor_layers);
    println!();

    // Quantize
    let qmodel = quantize_lewm_full(&model);

    // Test data
    let image = create_test_image(config.image_size, config.image_size, config.channels);
    let action: Vec<f32> = (0..config.action_dim).map(|i| (i as f32 * 0.1).sin()).collect();

    // Warmup
    let _ = model.encode(&image, config.image_size, config.image_size);
    let _ = qmodel.encode(&image, config.image_size, config.image_size);

    // Benchmark encode
    let iters = 50;
    let start = Instant::now();
    for _ in 0..iters {
        let _ = model.encode(&image, config.image_size, config.image_size);
    }
    let f32_encode_us = start.elapsed().as_micros() as f64 / iters as f64;

    let start = Instant::now();
    for _ in 0..iters {
        let _ = qmodel.encode(&image, config.image_size, config.image_size);
    }
    let int8_encode_us = start.elapsed().as_micros() as f64 / iters as f64;

    println!("Encode:");
    println!("  f32:  {:.1} us ({:.2} ms)", f32_encode_us, f32_encode_us / 1000.0);
    println!("  INT8: {:.1} us ({:.2} ms)", int8_encode_us, int8_encode_us / 1000.0);
    println!("  Speedup: {:.2}x", f32_encode_us / int8_encode_us);

    // Benchmark predict_next
    let z = model.encode(&image, config.image_size, config.image_size);
    let z_q = qmodel.encode(&image, config.image_size, config.image_size);

    let iters = 200;
    let start = Instant::now();
    for _ in 0..iters {
        let _ = model.predict_next(&z, &action);
    }
    let f32_pred_us = start.elapsed().as_micros() as f64 / iters as f64;

    let start = Instant::now();
    for _ in 0..iters {
        let _ = qmodel.predict_next(&z_q, &action);
    }
    let q4_pred_us = start.elapsed().as_micros() as f64 / iters as f64;

    println!("\nPredict_next:");
    println!("  f32:  {:.1} us ({:.2} ms)", f32_pred_us, f32_pred_us / 1000.0);
    println!("  Q4:   {:.1} us ({:.2} ms)", q4_pred_us, q4_pred_us / 1000.0);
    println!("  Speedup: {:.2}x", f32_pred_us / q4_pred_us);

    // Benchmark predict_next_fused (arena buffers, zero-alloc)
    let mut bufs = LeWMBuffers::new(&config);
    // Warmup
    let _ = model.predict_next_fused(&z, &action, &mut bufs);
    let start = Instant::now();
    for _ in 0..iters {
        let _ = model.predict_next_fused(&z, &action, &mut bufs);
    }
    let fused_pred_us = start.elapsed().as_micros() as f64 / iters as f64;

    println!("\nPredict_next_fused (arena):");
    println!("  f32:   {:.1} us ({:.2} ms)", fused_pred_us, fused_pred_us / 1000.0);
    println!("  vs naive f32: {:.2}x", f32_pred_us / fused_pred_us);

    // Cosine similarity check
    let pred_f32 = model.predict_next(&z, &action);
    let pred_fused = model.predict_next_fused(&z, &action, &mut bufs);
    let pred_q = qmodel.predict_next(&z_q, &action);
    let cos_q = cosine_sim(&pred_f32, &pred_q);
    let cos_fused = cosine_sim(&pred_f32, &pred_fused);
    println!("\nQuality:");
    println!("  cos(f32, INT8+Q4) = {:.6}", cos_q);
    println!("  cos(f32, fused)   = {:.6}", cos_fused);
}

fn cosine_sim(a: &[f32], b: &[f32]) -> f32 {
    let mut dot = 0.0f64;
    let mut na = 0.0f64;
    let mut nb = 0.0f64;
    for i in 0..a.len() {
        dot += a[i] as f64 * b[i] as f64;
        na += (a[i] as f64).powi(2);
        nb += (b[i] as f64).powi(2);
    }
    let denom = na.sqrt() * nb.sqrt();
    if denom < 1e-12 { 0.0 } else { (dot / denom) as f32 }
}

fn create_test_image(height: usize, width: usize, channels: usize) -> Vec<f32> {
    let mean = [0.485f32, 0.456, 0.406];
    let std = [0.229f32, 0.224, 0.225];
    let mut image = vec![0.0f32; height * width * channels];
    for y in 0..height {
        for x in 0..width {
            for c in 0..channels {
                let raw = match c {
                    0 => y as f32 / height as f32,
                    1 => x as f32 / width as f32,
                    _ => 0.5 + 0.5 * ((x + y) as f32 / (width + height) as f32).sin(),
                };
                image[(y * width + x) * channels + c] = (raw - mean[c]) / std[c];
            }
        }
    }
    image
}
