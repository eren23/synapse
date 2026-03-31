//! LEWM Slim vs Baseline Compression Benchmark
//!
//! Benchmarks multiple LEWM architecture variants (different latent dims,
//! encoder/predictor layer counts) against the baseline, measuring quality
//! (cosine similarity) and size at f32 and Q4.
//!
//! Usage:
//!   cargo run --release --example lewm_slim_vs_baseline -- \
//!     --models-dir /tmp/lewm-variants/ \
//!     --baseline /tmp/lewm-pusht/pusht/lejepa_weights.safetensors

use std::path::{Path, PathBuf};
use std::time::Instant;

use synapse_inference::models::{LeWMConfig, LeWorldModel};
use synapse_inference::quantization::{quantize_lewm_q4, quantize_lewm_q4_full};
use synapse_inference::weight_loading::load_safetensors;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let mut models_dir: Option<PathBuf> = None;
    let mut baseline_path: Option<PathBuf> = None;
    let mut slim_path: Option<PathBuf> = None;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--models-dir" | "-d" => {
                i += 1;
                models_dir = Some(PathBuf::from(&args[i]));
            }
            "--baseline" | "-b" => {
                i += 1;
                baseline_path = Some(PathBuf::from(&args[i]));
            }
            "--slim" | "-s" => {
                i += 1;
                slim_path = Some(PathBuf::from(&args[i]));
            }
            "--help" | "-h" => {
                eprintln!("Usage: lewm_slim_vs_baseline [OPTIONS]");
                eprintln!();
                eprintln!("Options:");
                eprintln!("  --models-dir <dir>     Directory with model variants (each subdir has config.json + safetensors)");
                eprintln!("  --baseline <path>      Path to baseline safetensors (default: /tmp/lewm-pusht/pusht/lejepa_weights.safetensors)");
                eprintln!("  --slim <path>          Path to a single slim model directory (with config.json)");
                std::process::exit(0);
            }
            _ => {}
        }
        i += 1;
    }

    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║       LEWM Slim vs Baseline Compression Benchmark          ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!();

    // Collect model configs to benchmark
    let mut models: Vec<ModelEntry> = Vec::new();

    // Add baseline
    let baseline = baseline_path.unwrap_or_else(|| {
        PathBuf::from("/tmp/lewm-pusht/pusht/lejepa_weights.safetensors")
    });
    if baseline.exists() {
        models.push(ModelEntry {
            name: "baseline 192d/6e/6p".into(),
            config: LeWMConfig::pusht(),
            weights_path: baseline.clone(),
            is_baseline: true,
        });
    } else {
        eprintln!("Warning: Baseline not found at {}", baseline.display());
        eprintln!("  Download: https://huggingface.co/le-wm/lejepa-pusht");
    }

    // Add single slim model if specified
    if let Some(slim_dir) = slim_path {
        if let Some(entry) = load_model_entry(&slim_dir) {
            models.push(entry);
        }
    }

    // Discover models from directory
    if let Some(dir) = models_dir {
        if let Ok(entries) = std::fs::read_dir(&dir) {
            let mut subdirs: Vec<_> = entries
                .filter_map(|e| e.ok())
                .filter(|e| e.path().is_dir())
                .collect();
            subdirs.sort_by_key(|e| e.file_name());
            for entry in subdirs {
                if let Some(model_entry) = load_model_entry(&entry.path()) {
                    models.push(model_entry);
                }
            }
        }
    }

    if models.is_empty() {
        eprintln!("No models found. Provide --baseline, --slim, or --models-dir.");
        std::process::exit(1);
    }

    println!("Found {} model(s) to benchmark:", models.len());
    for m in &models {
        println!(
            "  {} — {}d latent, {}e/{}p",
            m.name, m.config.latent_dim, m.config.encoder_layers, m.config.predictor_layers
        );
    }
    println!();

    // Generate test data using baseline config for image
    let image_config = &models[0].config;
    let image = create_test_image(image_config.image_size, image_config.image_size, image_config.channels);
    let num_steps = 20;

    // Collect results
    let mut all_results: Vec<ModelResults> = Vec::new();
    let mut baseline_traj: Option<Vec<Vec<f32>>> = None;

    for entry in &models {
        println!("━━━ {} ━━━", entry.name);
        let config = &entry.config;

        // Load f32 model
        println!("  Loading f32 weights from {}...", entry.weights_path.display());
        let mut model = LeWorldModel::from_config(config);
        let weights = match load_safetensors(&entry.weights_path) {
            Ok(w) => w,
            Err(e) => {
                eprintln!("  SKIP: Failed to load: {e}");
                continue;
            }
        };
        let stats = model.load_weights(weights).expect("Failed to load weights");
        println!("  Loaded {} tensors (skipped {})", stats.loaded, stats.skipped.len());
        if !stats.skipped.is_empty() && stats.skipped.len() <= 10 {
            for s in &stats.skipped {
                println!("    skip: {s}");
            }
        }

        // Generate actions appropriate for this model's action_dim
        let actions: Vec<Vec<f32>> = (0..num_steps)
            .map(|i| {
                let t = i as f32 / num_steps as f32;
                let mut a = vec![0.0f32; config.action_dim];
                a[0] = (t * std::f32::consts::PI).sin() * 0.5;
                if config.action_dim > 1 {
                    a[1] = (t * std::f32::consts::PI).cos() * 0.3;
                }
                a
            })
            .collect();

        // f32 baseline rollout
        let z = model.encode(&image, config.image_size, config.image_size);
        let start = Instant::now();
        let traj_f32 = model.rollout(&z, &actions);
        let f32_ms = start.elapsed().as_secs_f64() * 1000.0;
        let f32_size = estimate_f32_size(config);

        // Cosine vs baseline (if we have baseline trajectory)
        let cos_vs_baseline = if entry.is_baseline {
            baseline_traj = Some(traj_f32.clone());
            1.0
        } else if let Some(ref bt) = baseline_traj {
            // Can only compare if latent dims match
            if bt[num_steps - 1].len() == traj_f32[num_steps - 1].len() {
                cosine_sim(&bt[num_steps - 1], &traj_f32[num_steps - 1])
            } else {
                f32::NAN // different latent dims, can't directly compare
            }
        } else {
            f32::NAN
        };

        // Q4 quantization
        println!("  Quantizing to Q4...");
        let q4_model = quantize_lewm_q4(&model);
        let z_q4 = q4_model.encode(&image, config.image_size, config.image_size);
        let start = Instant::now();
        let traj_q4 = q4_model.rollout(&z_q4, &actions);
        let q4_ms = start.elapsed().as_secs_f64() * 1000.0;
        let q4_size = estimate_q4_size(config);
        let q4_cos = cosine_sim(&traj_f32[num_steps - 1], &traj_q4[num_steps - 1]);

        // Q4 full (encoder + predictor)
        println!("  Quantizing to Q4-full...");
        let q4f_model = quantize_lewm_q4_full(&model);
        let z_q4f = q4f_model.encode(&image, config.image_size, config.image_size);
        let start = Instant::now();
        let traj_q4f = q4f_model.rollout(&z_q4f, &actions);
        let q4f_ms = start.elapsed().as_secs_f64() * 1000.0;
        let q4f_size = q4f_model.model_size_bytes();
        let q4f_cos = cosine_sim(&traj_f32[num_steps - 1], &traj_q4f[num_steps - 1]);

        println!(
            "  f32: {:.1}MB {:.1}ms | Q4: {:.1}MB {:.1}ms cos={:.4} | Q4f: {:.1}MB {:.1}ms cos={:.4}",
            f32_size as f64 / 1e6, f32_ms,
            q4_size as f64 / 1e6, q4_ms, q4_cos,
            q4f_size as f64 / 1e6, q4f_ms, q4f_cos,
        );
        println!();

        all_results.push(ModelResults {
            name: entry.name.clone(),
            latent_dim: config.latent_dim,
            encoder_layers: config.encoder_layers,
            predictor_layers: config.predictor_layers,
            total_params: stats.loaded,
            f32_size,
            f32_ms,
            cos_vs_baseline,
            q4_size,
            q4_ms,
            q4_cos,
            q4f_size,
            q4f_ms,
            q4f_cos,
        });
    }

    // Print comparison table
    println!();
    println!("╔════════════════════════╦══════╦════════╦════════╦════════╦═════════╦════════╦════════╦═════════╗");
    println!("║ Model                  ║ d/e/p║ f32 MB ║ f32 ms ║ vs base║ Q4  MB  ║ Q4  ms ║ Q4 cos ║ Q4f MB  ║");
    println!("╠════════════════════════╬══════╬════════╬════════╬════════╬═════════╬════════╬════════╬═════════╣");
    for r in &all_results {
        let vs_base = if r.cos_vs_baseline.is_nan() {
            "  N/A ".to_string()
        } else {
            format!("{:.4}", r.cos_vs_baseline)
        };
        println!(
            "║ {:<22} ║{:>3}/{}/{} ║ {:>5.1}  ║ {:>5.1}  ║ {} ║ {:>6.1} ║ {:>5.1}  ║ {:.4} ║ {:>6.1} ║",
            r.name,
            r.latent_dim,
            r.encoder_layers,
            r.predictor_layers,
            r.f32_size as f64 / 1e6,
            r.f32_ms,
            vs_base,
            r.q4_size as f64 / 1e6,
            r.q4_ms,
            r.q4_cos,
            r.q4f_size as f64 / 1e6,
        );
    }
    println!("╚════════════════════════╩══════╩════════╩════════╩════════╩═════════╩════════╩════════╩═════════╝");

    // Summary: sort by Q4 size
    println!();
    println!("Ranked by Q4 size (smallest first):");
    let mut sorted = all_results.clone();
    sorted.sort_by_key(|r| r.q4_size);
    for (i, r) in sorted.iter().enumerate() {
        let usable = r.q4_cos > 0.90;
        let marker = if usable { "OK " } else { "BAD" };
        println!(
            "  {}. [{marker}] {} — Q4: {:.1}MB, cos={:.4}, {:.1}ms/rollout",
            i + 1,
            r.name,
            r.q4_size as f64 / 1e6,
            r.q4_cos,
            r.q4_ms,
        );
    }
}

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

struct ModelEntry {
    name: String,
    config: LeWMConfig,
    weights_path: PathBuf,
    is_baseline: bool,
}

#[derive(Clone)]
struct ModelResults {
    name: String,
    latent_dim: usize,
    encoder_layers: usize,
    predictor_layers: usize,
    total_params: usize,
    f32_size: usize,
    f32_ms: f64,
    cos_vs_baseline: f32,
    q4_size: usize,
    q4_ms: f64,
    q4_cos: f32,
    q4f_size: usize,
    q4f_ms: f64,
    q4f_cos: f32,
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn load_model_entry(dir: &Path) -> Option<ModelEntry> {
    let config_path = dir.join("config.json");
    let weights_path = dir.join("lejepa_weights.safetensors");

    if !config_path.exists() || !weights_path.exists() {
        return None;
    }

    let config = LeWMConfig::from_json(&config_path).ok()?;
    let dir_name = dir.file_name()?.to_str()?;
    let name = format!(
        "{}d/{}e/{}p",
        config.latent_dim, config.encoder_layers, config.predictor_layers
    );

    Some(ModelEntry {
        name: format!("{dir_name} ({name})"),
        config,
        weights_path,
        is_baseline: false,
    })
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
    if denom < 1e-12 {
        0.0
    } else {
        (dot / denom) as f32
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
                image[(y * width + x) * channels + c] = (raw - mean[c]) / std[c];
            }
        }
    }
    image
}

fn estimate_f32_size(config: &LeWMConfig) -> usize {
    let h = config.predictor_hidden;
    let inner = config.predictor_inner_dim;
    let inter = config.predictor_inter;
    let layers = config.predictor_layers;

    let per_layer = (h * 6 * h) + (h * 3 * inner) + (inner * h) + (h * inter) + (inter * h);
    let predictor = per_layer * layers * 4;

    let enc_h = config.encoder_hidden;
    let enc_inter = config.encoder_inter;
    let per_enc_layer =
        (enc_h * 3 * enc_h) + (enc_h * enc_h) + (enc_h * enc_inter) + (enc_inter * enc_h);
    let encoder = per_enc_layer * config.encoder_layers * 4;

    let patches = (config.image_size / config.patch_size).pow(2);
    let misc = (config.patch_size * config.patch_size * config.channels * enc_h
        + patches * enc_h
        + enc_h * h
        + config.action_dim * h
        + h * config.latent_dim)
        * 4;

    // Add input_proj/cond_proj if bottleneck
    let proj = if config.latent_dim != config.predictor_hidden {
        2 * config.predictor_hidden * config.latent_dim * 4
    } else {
        0
    };

    predictor + encoder + misc + proj
}

fn estimate_q4_size(config: &LeWMConfig) -> usize {
    let f32_total = estimate_f32_size(config);
    let h = config.predictor_hidden;
    let inner = config.predictor_inner_dim;
    let inter = config.predictor_inter;
    let layers = config.predictor_layers;

    let per_layer_elems = (h * 6 * h) + (h * 3 * inner) + (inner * h) + (h * inter) + (inter * h);
    let predictor_f32_bytes = per_layer_elems * 4 * layers;

    let per_layer_blocks = (per_layer_elems + 31) / 32;
    let predictor_q4_bytes = per_layer_blocks * 20 * layers;

    f32_total - predictor_f32_bytes + predictor_q4_bytes
}
