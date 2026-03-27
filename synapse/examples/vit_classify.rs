//! Classify an image using HuggingFace ViT-base weights.
//!
//! Usage:
//!   cargo run --release --example vit_classify -- --model-dir /tmp/vit-base
//!
//! The model directory must contain `config.json` and `model.safetensors`
//! from `google/vit-base-patch16-224`.

use std::path::PathBuf;

use synapse_inference::model::{parse_vit_config, parse_vit_labels, ViTModel};
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
        eprintln!("Usage: vit_classify --model-dir /tmp/vit-base");
        std::process::exit(1);
    })
}

/// Create a test image (224x224x3) with a gradient pattern.
fn create_test_image(height: usize, width: usize, channels: usize) -> Vec<f32> {
    // ImageNet normalization: pixel values are typically normalized to
    // mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225].
    // We create a gradient pattern that produces non-trivial activations.
    let mean = [0.485f32, 0.456, 0.406];
    let std = [0.229f32, 0.224, 0.225];
    let mut image = vec![0.0f32; height * width * channels];
    for y in 0..height {
        for x in 0..width {
            for c in 0..channels {
                // Raw pixel in [0, 1]
                let raw = match c {
                    0 => y as f32 / height as f32, // R: vertical gradient
                    1 => x as f32 / width as f32,  // G: horizontal gradient
                    _ => 0.5 + 0.5 * ((x + y) as f32 / (width + height) as f32).sin(), // B
                };
                // Apply ImageNet normalization
                let normalized = (raw - mean[c]) / std[c];
                image[(y * width + x) * channels + c] = normalized;
            }
        }
    }
    image
}

/// Return indices of the top-k largest values.
fn top_k_indices(values: &[f32], k: usize) -> Vec<usize> {
    let mut indexed: Vec<(usize, f32)> = values.iter().copied().enumerate().collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    indexed.into_iter().take(k).map(|(i, _)| i).collect()
}

fn main() {
    let model_dir = parse_args();

    // 1. Load config
    let config_path = model_dir.join("config.json");
    println!("Loading config from {}...", config_path.display());
    let config = parse_vit_config(&config_path).expect("Failed to parse ViT config");
    println!(
        "  Model: ViT-base (hidden={}, layers={}, heads={}, patches={}x{}, classes={})",
        config.hidden_size,
        config.num_layers,
        config.num_heads,
        config.image_size / config.patch_size,
        config.image_size / config.patch_size,
        config.num_classes,
    );

    // 2. Load class labels
    let labels = parse_vit_labels(&config_path).unwrap_or_default();
    println!("  Labels: {} classes loaded", labels.len());

    // 3. Build model
    let mut model = ViTModel::from_config(&config);
    model.class_labels = labels;

    // 4. Load weights
    let safetensors_path = model_dir.join("model.safetensors");
    println!("Loading weights from {}...", safetensors_path.display());
    let weights = load_safetensors(&safetensors_path).expect("Failed to load safetensors");
    println!("  Loaded {} tensors from safetensors", weights.len());

    let mapper = WeightMapper::vit();
    let result = model
        .load_weights(weights, &mapper)
        .expect("Failed to load weights");

    if !result.missing.is_empty() {
        println!(
            "  Warning: {} missing weights: {:?}",
            result.missing.len(),
            result.missing
        );
    }
    if !result.unexpected.is_empty() {
        println!(
            "  Note: {} unmapped source tensors",
            result.unexpected.len()
        );
    }
    println!("  Weights loaded successfully.");

    // 5. Create test image
    let image = create_test_image(config.image_size, config.image_size, config.channels);
    println!(
        "\nRunning inference on {}x{}x{} test image (gradient pattern)...",
        config.image_size, config.image_size, config.channels
    );

    // 6. Forward pass
    let start = std::time::Instant::now();
    let output = model.forward_image(&image, config.image_size, config.image_size);
    let elapsed = start.elapsed();

    // 7. Print results
    println!("\nResults:");
    println!("  Embedding dim: {}", output.embeddings.len());
    println!(
        "  Embedding finite: {}",
        output.embeddings.iter().all(|v| v.is_finite())
    );
    println!(
        "  Embedding L2 norm: {:.4}",
        output.embeddings.iter().map(|v| v * v).sum::<f32>().sqrt()
    );

    if let Some(ref logits) = output.logits {
        println!("  Logits: {} classes", logits.len());
        println!("  Logits finite: {}", logits.iter().all(|v| v.is_finite()));

        let top5 = top_k_indices(logits, 5);
        println!("\n  Top-5 predictions:");
        for (rank, &idx) in top5.iter().enumerate() {
            let label = if idx < model.class_labels.len() {
                model.class_labels[idx].as_str()
            } else {
                "unknown"
            };
            println!(
                "    {}: class {:>4} (logit {:>8.3}) - {}",
                rank + 1,
                idx,
                logits[idx],
                label
            );
        }
    }

    println!(
        "\n  Inference time: {:.1}ms",
        elapsed.as_secs_f64() * 1000.0
    );
}
