//! LeWorldModel (LeWM) Demo
//!
//! Loads the PushT checkpoint and runs encode + rollout.
//!
//! Usage:
//!   cargo run --release --example lewm_demo
//!
//! Requires the checkpoint at /tmp/lewm-pusht/pusht/lejepa_weights.safetensors

use std::path::Path;
use std::time::Instant;

use synapse_inference::model::{LeWMConfig, LeWorldModel};
use synapse_inference::weight_loading::load_safetensors;

fn main() {
    let model_path = "/tmp/lewm-pusht/pusht/lejepa_weights.safetensors";

    println!("=== LeWorldModel (LeWM) Demo ===");
    println!();

    // 1. Config (hardcoded from checkpoint inspection)
    let config = LeWMConfig::pusht();
    println!("Config:");
    println!("  Encoder: {}x{} patches, {} hidden, {} layers, {} heads",
        config.image_size / config.patch_size,
        config.image_size / config.patch_size,
        config.encoder_hidden,
        config.encoder_layers,
        config.encoder_heads,
    );
    println!("  Predictor: {} layers, {} heads, inner_dim={}, inter={}",
        config.predictor_layers,
        config.predictor_heads,
        config.predictor_inner_dim,
        config.predictor_inter,
    );
    println!("  Action dim: {}, Latent dim: {}", config.action_dim, config.latent_dim);
    println!();

    // 2. Build model
    let mut model = LeWorldModel::from_config(&config);

    // 3. Load weights
    let path = Path::new(model_path);
    if !path.exists() {
        eprintln!("Checkpoint not found at {model_path}");
        eprintln!("Please download the PushT LeJEPA weights first.");
        std::process::exit(1);
    }

    println!("Loading weights from {}...", model_path);
    let weights = load_safetensors(path).expect("Failed to load safetensors");
    println!("  Loaded {} tensors from safetensors", weights.len());

    let stats = model.load_weights(weights).expect("Failed to load weights");
    println!("  Mapped {} tensors", stats.loaded);
    if !stats.skipped.is_empty() {
        println!("  Skipped {} unrecognized keys", stats.skipped.len());
        for key in stats.skipped.iter().take(5) {
            println!("    - {key}");
        }
        if stats.skipped.len() > 5 {
            println!("    ... and {} more", stats.skipped.len() - 5);
        }
    }
    println!();

    // 4. Create test observation
    let image = create_test_image(config.image_size, config.image_size, config.channels);

    // 5. Encode
    println!("Encoding {}x{}x{} test image...", config.image_size, config.image_size, config.channels);
    let start = Instant::now();
    let z = model.encode(&image, config.image_size, config.image_size);
    let encode_time = start.elapsed();
    println!("  Encoded in {:.1}ms", encode_time.as_secs_f64() * 1000.0);
    println!("  Latent L2 norm: {:.4}", l2_norm(&z));
    println!("  Latent dim: {}", z.len());
    assert!(z.iter().all(|v| v.is_finite()), "Encode produced non-finite values!");
    println!();

    // 6. Single predict_next
    let action = vec![0.0f32; config.action_dim];
    let start = Instant::now();
    let z_next = model.predict_next(&z, &action);
    let predict_time = start.elapsed();
    println!("Single predict_next (zero action):");
    println!("  {:.2}ms", predict_time.as_secs_f64() * 1000.0);
    println!("  Next latent L2 norm: {:.4}", l2_norm(&z_next));
    assert!(z_next.iter().all(|v| v.is_finite()), "predict_next produced non-finite values!");
    println!();

    // 7. Rollout
    let num_steps = 50;
    let actions: Vec<Vec<f32>> = (0..num_steps)
        .map(|_| vec![0.0f32; config.action_dim])
        .collect();
    let start = Instant::now();
    let trajectory = model.rollout(&z, &actions);
    let rollout_time = start.elapsed();
    println!("{}-step rollout:", num_steps);
    println!("  {:.1}ms total ({:.2}ms/step)",
        rollout_time.as_secs_f64() * 1000.0,
        rollout_time.as_secs_f64() * 1000.0 / num_steps as f64,
    );
    println!("  All finite: {}", trajectory.iter().all(|s| s.iter().all(|v| v.is_finite())));
    println!("  Final state L2 norm: {:.4}", l2_norm(trajectory.last().unwrap()));

    // Show norm evolution
    println!();
    println!("Latent norm evolution (every 10 steps):");
    for (i, state) in trajectory.iter().enumerate() {
        if i % 10 == 0 || i == trajectory.len() - 1 {
            println!("  Step {:3}: L2 = {:.4}", i + 1, l2_norm(state));
        }
    }
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
                let normalized = (raw - mean[c]) / std[c];
                image[(y * width + x) * channels + c] = normalized;
            }
        }
    }
    image
}

fn l2_norm(v: &[f32]) -> f32 {
    v.iter().map(|x| x * x).sum::<f32>().sqrt()
}
