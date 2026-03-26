//! Compute image-text similarity using HuggingFace CLIP weights.
//!
//! Usage:
//!   cargo run --release --example clip_similarity -- --model-dir /tmp/clip-vit-base
//!
//! The model directory must contain `config.json` and `model.safetensors`
//! from `openai/clip-vit-base-patch32`.

use std::path::PathBuf;

use synapse_inference::model::{parse_clip_config, CLIPModel};
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
        eprintln!("Usage: clip_similarity --model-dir /tmp/clip-vit-base");
        std::process::exit(1);
    })
}

/// Create a test image with a gradient pattern.
fn create_test_image(height: usize, width: usize, channels: usize) -> Vec<f32> {
    let mean = [0.48145466f32, 0.4578275, 0.40821073];
    let std = [0.26862954f32, 0.26130258, 0.27577711];
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

/// Basic whitespace tokenizer for testing (maps words to sequential IDs).
/// Real CLIP uses a BPE tokenizer -- this is just to prove the forward pass works.
fn simple_tokenize(text: &str, max_len: usize) -> Vec<u32> {
    let mut tokens = vec![49406u32]; // <BOS> token
    for (i, _word) in text.split_whitespace().enumerate() {
        // Map each word to a pseudo-ID (offset to avoid special tokens)
        tokens.push(1000 + i as u32);
    }
    tokens.push(49407); // <EOS> token
    // Pad to max_len
    tokens.resize(max_len, 0);
    tokens
}

fn l2_norm(v: &[f32]) -> f32 {
    v.iter().map(|x| x * x).sum::<f32>().sqrt()
}

fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

fn main() {
    let model_dir = parse_args();

    // 1. Load config
    let config_path = model_dir.join("config.json");
    println!("Loading CLIP config from {}...", config_path.display());
    let config = parse_clip_config(&config_path).expect("Failed to parse CLIP config");
    println!(
        "  Vision: hidden={}, layers={}, heads={}, image={}x{}, patch={}",
        config.vision.hidden_size,
        config.vision.num_layers,
        config.vision.num_heads,
        config.vision.image_size,
        config.vision.image_size,
        config.vision.patch_size,
    );
    println!(
        "  Text: hidden={}, layers={}, heads={}, vocab={}, max_pos={}",
        config.text_hidden_size,
        config.text_num_layers,
        config.text_num_heads,
        config.vocab_size,
        config.text_max_position,
    );
    println!("  Embed dim: {}", config.embed_dim);

    // 2. Build model
    let mut model = CLIPModel::from_config(&config);

    // 3. Load weights
    let safetensors_path = model_dir.join("model.safetensors");
    println!("Loading weights from {}...", safetensors_path.display());
    let weights = load_safetensors(&safetensors_path).expect("Failed to load safetensors");
    println!("  Loaded {} tensors from safetensors", weights.len());

    let mapper = WeightMapper::clip();
    let result = model
        .load_weights(weights, &mapper)
        .expect("Failed to load weights");

    if !result.missing.is_empty() {
        println!(
            "  Warning: {} missing weights: {:?}",
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
    println!("  Weights loaded successfully.");

    // 4. Encode test image
    let image = create_test_image(
        config.vision.image_size,
        config.vision.image_size,
        config.vision.channels,
    );
    println!(
        "\nEncoding {}x{}x{} test image...",
        config.vision.image_size, config.vision.image_size, config.vision.channels
    );

    let start = std::time::Instant::now();
    let img_embed = model.encode_image(
        &image,
        config.vision.image_size,
        config.vision.image_size,
    );
    let img_time = start.elapsed();

    println!(
        "  Image embedding L2 norm: {:.4} ({}ms)",
        l2_norm(&img_embed),
        img_time.as_millis()
    );

    // 5. Encode text prompts and compute similarities
    let prompts = ["a photo of a cat", "a photo of a dog", "a diagram"];
    println!("\nImage-text similarities:");
    for prompt in &prompts {
        let tokens = simple_tokenize(prompt, config.text_max_position);
        let txt_embed = model.encode_text(&tokens);
        let sim = cosine_similarity(&img_embed, &txt_embed);
        println!("  '{prompt}': similarity = {sim:.4}");
    }
}
