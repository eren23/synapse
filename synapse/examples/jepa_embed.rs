//! Load DINOv2 encoder weights into a JEPA model and run a forward pass.
//!
//! Usage:
//!   cargo run --release --example jepa_embed -- --model-dir /tmp/dinov2-base
//!
//! The model directory must contain `config.json` and `model.safetensors`
//! from `facebook/dinov2-base`.

use std::path::PathBuf;

use synapse_inference::model::{parse_vit_config, JEPAConfig, JEPAModel};
use synapse_inference::weight_loading::{load_safetensors, WeightMapper};

fn parse_args() -> PathBuf {
    let args: Vec<String> = std::env::args().collect();
    let mut model_dir = None;
    let mut i = 1;
    while i < args.len() {
        if args[i] == "--model-dir" && i + 1 < args.len() {
            model_dir = Some(PathBuf::from(&args[i + 1]));
            i += 2;
        } else {
            i += 1;
        }
    }
    model_dir.unwrap_or_else(|| {
        eprintln!("Usage: jepa_embed --model-dir /tmp/dinov2-base");
        std::process::exit(1);
    })
}

/// Create a test image with a gradient pattern.
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

fn main() {
    let model_dir = parse_args();

    // 1. Parse ViT config (DINOv2 uses standard ViT config format)
    let config_path = model_dir.join("config.json");
    println!("Loading DINOv2 config from {}...", config_path.display());
    let vit_config = parse_vit_config(&config_path).expect("Failed to parse config");
    println!(
        "  Encoder: hidden={}, layers={}, heads={}, image={}x{}, patch={}",
        vit_config.hidden_size,
        vit_config.num_layers,
        vit_config.num_heads,
        vit_config.image_size,
        vit_config.image_size,
        vit_config.patch_size,
    );

    // 2. Build JEPA model with this encoder config
    let jepa_config = JEPAConfig {
        encoder: vit_config.clone(),
        predictor_hidden_size: vit_config.hidden_size / 2,
        predictor_num_layers: 6,
        predictor_num_heads: vit_config.num_heads,
    };
    let mut model = JEPAModel::from_config(&jepa_config);

    // 3. Load DINOv2 weights into the encoder
    let safetensors_path = model_dir.join("model.safetensors");
    println!("Loading weights from {}...", safetensors_path.display());
    let weights = load_safetensors(&safetensors_path).expect("Failed to load safetensors");
    println!("  Loaded {} tensors from safetensors", weights.len());

    let mapper = WeightMapper::dinov2();
    let result = model
        .load_encoder_weights(weights, &mapper)
        .expect("Failed to load encoder weights");

    if !result.missing.is_empty() {
        println!(
            "  Warning: {} missing weights (expected for predictor): {:?}",
            result.missing.len(),
            &result.missing[..result.missing.len().min(10)]
        );
    }
    if !result.unexpected.is_empty() {
        println!(
            "  Note: {} unmapped source tensors",
            result.unexpected.len()
        );
    }
    println!("  Encoder weights loaded successfully.");

    // 4. Create test image and run forward pass
    let image = create_test_image(
        vit_config.image_size,
        vit_config.image_size,
        vit_config.channels,
    );
    let num_patches = vit_config.num_patches();
    println!(
        "\nRunning JEPA forward on {}x{}x{} test image ({} patches)...",
        vit_config.image_size, vit_config.image_size, vit_config.channels, num_patches
    );

    // Context mask: first 75% of patches; target mask: last 25%
    let context_count = (num_patches * 3) / 4;
    let mut context_mask = vec![false; num_patches];
    let mut target_mask = vec![false; num_patches];
    for i in 0..context_count {
        context_mask[i] = true;
    }
    for i in context_count..num_patches {
        target_mask[i] = true;
    }
    let target_count = num_patches - context_count;

    let start = std::time::Instant::now();
    let output = model.forward(
        &image,
        vit_config.image_size,
        vit_config.image_size,
        &context_mask,
        &target_mask,
    );
    let elapsed = start.elapsed();

    // 5. Print results
    println!("\nResults:");
    println!(
        "  Context embeddings: {} patches x {} dim",
        context_count, vit_config.hidden_size
    );
    println!("    L2 norm (mean): {:.4}", {
        let mut sum = 0.0;
        for i in 0..context_count {
            let start = i * vit_config.hidden_size;
            let end = start + vit_config.hidden_size;
            sum += l2_norm(&output.context_embeddings[start..end]);
        }
        sum / context_count as f32
    });

    println!(
        "  Predicted target embeddings: {} patches x {} dim",
        target_count, vit_config.hidden_size
    );
    println!("    L2 norm (mean): {:.4}", {
        let mut sum = 0.0;
        for i in 0..target_count {
            let start = i * vit_config.hidden_size;
            let end = start + vit_config.hidden_size;
            sum += l2_norm(&output.predicted_embeddings[start..end]);
        }
        sum / target_count as f32
    });

    println!(
        "  All finite: context={}, predicted={}",
        output.context_embeddings.iter().all(|v| v.is_finite()),
        output.predicted_embeddings.iter().all(|v| v.is_finite()),
    );
    println!("  Inference time: {:.1}ms", elapsed.as_secs_f64() * 1000.0);
}
