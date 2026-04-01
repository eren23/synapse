//! Compare LEWM slim variant checkpoints side-by-side.
//!
//! Loads multiple safetensors+config pairs, runs f32 encode+rollout on the same
//! test image and actions, then prints a cosine-similarity matrix.
//!
//! Usage:
//!   cargo run -p synapse --release --example lewm_compare_variants -- \
//!     /tmp/lewm-64d-variants/baseline /tmp/lewm-64d-variants/elastic_fixed100

use std::path::Path;
use std::time::Instant;

use synapse_inference::models::{LeWMConfig, LeWorldModel};
use synapse_inference::weight_loading::load_safetensors;

fn main() {
    let dirs: Vec<String> = std::env::args().skip(1).collect();
    if dirs.is_empty() {
        eprintln!("Usage: lewm_compare_variants <dir1> <dir2> [dir3...]");
        eprintln!("Each dir must contain lejepa_weights.safetensors + config.json");
        std::process::exit(1);
    }

    println!("Loading {} variants...\n", dirs.len());

    let mut models: Vec<(String, LeWorldModel, LeWMConfig)> = Vec::new();
    for dir in &dirs {
        let config_path = Path::new(dir).join("config.json");
        let weights_path = Path::new(dir).join("lejepa_weights.safetensors");
        if !weights_path.exists() {
            eprintln!("  SKIP {}: no lejepa_weights.safetensors", dir);
            continue;
        }
        let config = LeWMConfig::from_json(&config_path)
            .unwrap_or_else(|e| { eprintln!("Config error for {dir}: {e}"); std::process::exit(1); });
        let mut model = LeWorldModel::from_config(&config);
        let weights = load_safetensors(&weights_path).expect("Failed to load safetensors");
        let stats = model.load_weights(weights).expect("Failed to load weights");

        let name = Path::new(dir).file_name().unwrap().to_str().unwrap().to_string();
        println!("  [{}] {}d latent, {}e/{}p, {} tensors loaded",
            name, config.latent_dim, config.encoder_layers, config.predictor_layers, stats.loaded);
        models.push((name, model, config));
    }

    if models.len() < 2 {
        eprintln!("\nNeed at least 2 variants to compare.");
        std::process::exit(1);
    }

    // Use first model's config for test data (all should share same latent_dim)
    let ref_config = &models[0].2;
    let image = create_test_image(ref_config.image_size, ref_config.image_size, ref_config.channels);
    let num_steps = 20;
    let actions: Vec<Vec<f32>> = (0..num_steps)
        .map(|i| {
            let t = i as f32 / num_steps as f32;
            let mut a = vec![0.0f32; ref_config.action_dim];
            a[0] = (t * std::f32::consts::PI).sin() * 0.5;
            a[1] = (t * std::f32::consts::PI).cos() * 0.3;
            a
        })
        .collect();

    // Run inference for each variant
    println!("\nRunning encode + {}-step rollout...\n", num_steps);
    let mut trajectories: Vec<(String, Vec<f32>, Vec<Vec<f32>>, f64, f64)> = Vec::new();

    for (name, model, config) in &models {
        let start = Instant::now();
        let z = model.encode(&image, config.image_size, config.image_size);
        let encode_ms = start.elapsed().as_secs_f64() * 1000.0;

        let start = Instant::now();
        let traj = model.rollout(&z, &actions);
        let rollout_ms = start.elapsed().as_secs_f64() * 1000.0;

        println!("  [{}] encode: {:.1}ms, rollout: {:.1}ms, z L2: {:.4}",
            name, encode_ms, rollout_ms, l2_norm(&z));
        trajectories.push((name.clone(), z, traj, encode_ms, rollout_ms));
    }

    // Print cosine similarity matrix at key steps
    let steps_to_check = [0, 4, 9, 14, 19]; // step 1, 5, 10, 15, 20
    let n = trajectories.len();

    println!("\n{}", "=".repeat(70));
    println!("Cosine Similarity: Encode (z)");
    println!("{}", "=".repeat(70));
    print!("{:>20}", "");
    for (name, _, _, _, _) in &trajectories {
        print!(" {:>12}", name);
    }
    println!();
    for i in 0..n {
        print!("{:>20}", trajectories[i].0);
        for j in 0..n {
            let cos = cosine_sim(&trajectories[i].1, &trajectories[j].1);
            print!(" {:>12.6}", cos);
        }
        println!();
    }

    for &step in &steps_to_check {
        if step >= num_steps { continue; }
        println!("\n{}", "=".repeat(70));
        println!("Cosine Similarity: Rollout step {} (of {})", step + 1, num_steps);
        println!("{}", "=".repeat(70));
        print!("{:>20}", "");
        for (name, _, _, _, _) in &trajectories {
            print!(" {:>12}", name);
        }
        println!();
        for i in 0..n {
            print!("{:>20}", trajectories[i].0);
            for j in 0..n {
                let cos = cosine_sim(&trajectories[i].2[step], &trajectories[j].2[step]);
                print!(" {:>12.6}", cos);
            }
            println!();
        }
    }

    // Summary: L2 drift per variant
    println!("\n{}", "=".repeat(70));
    println!("Trajectory L2 norms (latent drift)");
    println!("{}", "=".repeat(70));
    for (name, z, traj, _, _) in &trajectories {
        let z_l2 = l2_norm(z);
        let last_l2 = l2_norm(&traj[num_steps - 1]);
        let drift = (last_l2 - z_l2).abs() / z_l2;
        println!("  [{}] z_L2={:.4}, step20_L2={:.4}, drift={:.2}%",
            name, z_l2, last_l2, drift * 100.0);
    }

    // Per-step self-cosine: cos(z, step_i) and cos(step_i, step_{i-1})
    println!("\n{}", "=".repeat(70));
    println!("Per-step trajectory analysis");
    println!("{}", "=".repeat(70));
    print!("{:>6}", "step");
    for (name, _, _, _, _) in &trajectories {
        print!(" {:>16} {:>10} {:>8}", format!("{} cos_z", name), "cos_prev", "L2");
    }
    println!();
    for step in 0..num_steps {
        print!("{:>6}", step + 1);
        for (_name, z, traj, _, _) in &trajectories {
            let cos_z = cosine_sim(z, &traj[step]);
            let cos_prev = if step > 0 {
                cosine_sim(&traj[step - 1], &traj[step])
            } else {
                0.0
            };
            let l2 = l2_norm(&traj[step]);
            print!(" {:>16.6} {:>10.6} {:>8.4}", cos_z, cos_prev, l2);
        }
        println!();
    }
}

fn cosine_sim(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
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

fn l2_norm(v: &[f32]) -> f32 {
    v.iter().map(|x| x * x).sum::<f32>().sqrt()
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
