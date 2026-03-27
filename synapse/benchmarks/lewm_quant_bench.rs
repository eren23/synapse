use std::time::Instant;
use synapse_inference::model::lewm::{LeWMBuffers, LeWMConfig, LeWorldModel};
use synapse_inference::quantization::quantized_lewm::quantize_lewm;
use synapse_inference::quantization::q4::{quantize_lewm_q4, cached_q4_lewm};

fn main() {
    let config = LeWMConfig::pusht();
    
    // Load real model
    let weights_path = "/tmp/lewm-pusht/pusht/lejepa_weights.safetensors";
    let weights = synapse_inference::weight_loading::safetensors::load_safetensors(
        std::path::Path::new(weights_path)
    ).expect("Failed to load weights");
    
    let mut model = LeWorldModel::from_config(&config);
    let _ = model.load_weights(weights);
    
    // Quantize
    let int8_model = quantize_lewm(&model);
    let q4_model = quantize_lewm_q4(&model);
    let cached_q4 = cached_q4_lewm(&model);
    
    let image = vec![0.5f32; 224 * 224 * 3];
    let action = vec![0.1f32; config.action_dim];
    
    // Encode (same for all — encoder stays f32)
    let z = model.encode(&image, 224, 224);
    
    // Metal GPU setup (if available)
    #[cfg(feature = "metal")]
    let metal_state = {
        use synapse_inference::metal::{MetalBackend, MetalLeWMState};
        match MetalBackend::new() {
            Ok(backend) => {
                let state = MetalLeWMState::from_model(&model, &backend);
                println!("Metal GPU: {} (pipelines: {})", backend.device_name(), backend.pipeline_count());
                Some((backend, state))
            }
            Err(_) => {
                println!("Metal GPU: not available");
                None
            }
        }
    };

    println!("═══════════════════════════════════════════════════");
    println!("  LEWM Benchmark: f32 vs INT8 vs Q4 vs Metal");
    println!("═══════════════════════════════════════════════════\n");

    // Pre-allocate fused buffers
    let mut bufs = LeWMBuffers::new(&config);

    // Warmup
    let _ = model.predict_next(&z, &action);
    let _ = model.predict_next_fused(&z, &action, &mut bufs);
    let _ = int8_model.predict_next(&z, &action);
    let _ = q4_model.predict_next(&z, &action);
    let _ = cached_q4.predict_next(&z, &action);
    #[cfg(feature = "metal")]
    if let Some((ref backend, ref state)) = metal_state {
        let _ = model.predict_next_metal(&z, &action, state, backend);
        let _ = model.predict_next_metal_fused(&z, &action, state, backend);
        let _ = model.predict_next_metal_v3(&z, &action, state, backend);
    }

    // Benchmark predict_next
    let runs = 20;

    let t = Instant::now();
    for _ in 0..runs { let _ = model.predict_next(&z, &action); }
    let f32_ms = t.elapsed().as_secs_f64() * 1000.0 / runs as f64;

    let t = Instant::now();
    for _ in 0..runs { let _ = model.predict_next_fused(&z, &action, &mut bufs); }
    let fused_ms = t.elapsed().as_secs_f64() * 1000.0 / runs as f64;

    let t = Instant::now();
    for _ in 0..runs { let _ = int8_model.predict_next(&z, &action); }
    let int8_ms = t.elapsed().as_secs_f64() * 1000.0 / runs as f64;

    let t = Instant::now();
    for _ in 0..runs { let _ = q4_model.predict_next(&z, &action); }
    let q4_ms = t.elapsed().as_secs_f64() * 1000.0 / runs as f64;

    let t = Instant::now();
    for _ in 0..runs { let _ = cached_q4.predict_next(&z, &action); }
    let cq4_ms = t.elapsed().as_secs_f64() * 1000.0 / runs as f64;

    #[cfg(feature = "metal")]
    let (metal_ms, metal_fused_ms) = if let Some((ref backend, ref state)) = metal_state {
        let t = Instant::now();
        for _ in 0..runs { let _ = model.predict_next_metal(&z, &action, state, backend); }
        let ms = t.elapsed().as_secs_f64() * 1000.0 / runs as f64;
        let t = Instant::now();
        for _ in 0..runs { let _ = model.predict_next_metal_fused(&z, &action, state, backend); }
        let fms = t.elapsed().as_secs_f64() * 1000.0 / runs as f64;
        (Some(ms), Some(fms))
    } else {
        (None, None)
    };

    println!("predict_next ({runs} runs avg):");
    println!("  f32:          {f32_ms:.2}ms");
    println!("  f32 FUSED:    {fused_ms:.2}ms  ({:.2}x vs f32)", f32_ms / fused_ms);
    println!("  INT8:         {int8_ms:.2}ms  ({:.2}x vs f32)", f32_ms / int8_ms);
    println!("  Q4 (scalar):  {q4_ms:.1}ms  ({:.1}x vs f32)", f32_ms / q4_ms);
    println!("  Q4 (cached):  {cq4_ms:.2}ms  ({:.2}x vs f32)", f32_ms / cq4_ms);
    #[cfg(feature = "metal")]
    if let Some((ref backend, ref state)) = metal_state {
        if let Some(ms) = metal_ms {
            println!("  Metal (v1):     {ms:.2}ms  ({:.2}x vs f32)", f32_ms / ms);
        }
        if let Some(fms) = metal_fused_ms {
            println!("  Metal FUSED:    {fms:.2}ms  ({:.2}x vs f32)", f32_ms / fms);
        }
        // V3: vectorized
        let t = Instant::now();
        for _ in 0..runs { let _ = model.predict_next_metal_v3(&z, &action, state, backend); }
        let v3 = t.elapsed().as_secs_f64() * 1000.0 / runs as f64;
        println!("  Metal V3 (vec): {v3:.2}ms  ({:.2}x vs f32)", f32_ms / v3);
    }

    // Benchmark 50-step rollout
    let actions: Vec<Vec<f32>> = (0..50).map(|_| vec![0.1f32; config.action_dim]).collect();

    let t = Instant::now();
    let _ = model.rollout(&z, &actions);
    let f32_roll = t.elapsed().as_secs_f64() * 1000.0;

    let t = Instant::now();
    let _ = model.rollout_fused(&z, &actions);
    let fused_roll = t.elapsed().as_secs_f64() * 1000.0;

    let t = Instant::now();
    let _ = int8_model.rollout(&z, &actions);
    let int8_roll = t.elapsed().as_secs_f64() * 1000.0;

    let t = Instant::now();
    let _ = q4_model.rollout(&z, &actions);
    let q4_roll = t.elapsed().as_secs_f64() * 1000.0;

    let t = Instant::now();
    let _ = cached_q4.rollout(&z, &actions);
    let cq4_roll = t.elapsed().as_secs_f64() * 1000.0;

    #[cfg(feature = "metal")]
    let metal_roll = if let Some((ref backend, ref state)) = metal_state {
        let t = Instant::now();
        let _ = model.rollout_metal(&z, &actions, state, backend);
        Some(t.elapsed().as_secs_f64() * 1000.0)
    } else {
        None
    };

    println!("\n50-step rollout:");
    println!("  f32:          {f32_roll:.0}ms ({:.2}ms/step)", f32_roll / 50.0);
    println!("  f32 FUSED:    {fused_roll:.0}ms ({:.1}ms/step)  ({:.2}x)", fused_roll / 50.0, f32_roll / fused_roll);
    println!("  INT8:         {int8_roll:.0}ms ({:.2}ms/step)  ({:.1}x)", int8_roll / 50.0, f32_roll / int8_roll);
    println!("  Q4 (scalar):  {q4_roll:.0}ms ({:.1}ms/step)  ({:.1}x)", q4_roll / 50.0, f32_roll / q4_roll);
    println!("  Q4 (cached):  {cq4_roll:.0}ms ({:.2}ms/step)  ({:.2}x)", cq4_roll / 50.0, f32_roll / cq4_roll);
    #[cfg(feature = "metal")]
    if let Some(metal_roll) = metal_roll {
        println!("  Metal fused:  {metal_roll:.0}ms ({:.2}ms/step)  ({:.2}x)", metal_roll / 50.0, f32_roll / metal_roll);
    }

    // Cosine similarity check
    let f32_next = model.predict_next(&z, &action);
    let fused_next = model.predict_next_fused(&z, &action, &mut bufs);
    let int8_next = int8_model.predict_next(&z, &action);
    let q4_next = q4_model.predict_next(&z, &action);
    let cq4_next = cached_q4.predict_next(&z, &action);

    let cos = |a: &[f32], b: &[f32]| -> f32 {
        let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
        let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
        if na > 0.0 && nb > 0.0 { dot / (na * nb) } else { 0.0 }
    };

    println!("\nAccuracy (cosine similarity vs f32):");
    println!("  FUSED:      {:.6}", cos(&f32_next, &fused_next));
    println!("  INT8:       {:.6}", cos(&f32_next, &int8_next));
    println!("  Q4 scalar:  {:.6}", cos(&f32_next, &q4_next));
    println!("  Q4 cached:  {:.6}", cos(&f32_next, &cq4_next));
    #[cfg(feature = "metal")]
    if let Some((ref backend, ref state)) = metal_state {
        let metal_next = model.predict_next_metal(&z, &action, state, backend);
        println!("  Metal:      {:.6}", cos(&f32_next, &metal_next));
    }
}
