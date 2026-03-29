//! LEWM Compression Benchmark
//!
//! Systematically tests f32/INT8/Q4 + pruning combinations on the LEWM model,
//! measuring quality (cosine similarity of rollout trajectories) vs model size.
//!
//! Usage:
//!   cargo run --release --example lewm_compress
//!
//! Requires the checkpoint at /tmp/lewm-pusht/pusht/lejepa_weights.safetensors
//! (same as lewm_demo.rs)

use std::path::Path;
use std::time::Instant;

use synapse_inference::model::{LeWMConfig, LeWorldModel};
use synapse_inference::quantization::{quantize_lewm, quantize_lewm_q4, cached_q4_lewm, quantize_lewm_ternary, quantize_lewm_full, quantize_lewm_q4_full};
use synapse_inference::weight_loading::load_safetensors;

fn main() {
    let model_path = std::env::args().nth(1).unwrap_or_else(||
        "/tmp/lewm-pusht/pusht/lejepa_weights.safetensors".into()
    );

    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║           LEWM Compression Benchmark                       ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!();

    // Load f32 baseline model
    let config = LeWMConfig::pusht();
    let mut model = LeWorldModel::from_config(&config);

    let path = Path::new(&model_path);
    if !path.exists() {
        eprintln!("Checkpoint not found at {model_path}");
        eprintln!("Download: https://huggingface.co/le-wm/lejepa-pusht");
        std::process::exit(1);
    }

    println!("Loading f32 weights from {model_path}...");
    let weights = load_safetensors(path).expect("Failed to load safetensors");
    let stats = model.load_weights(weights).expect("Failed to load weights");
    println!("  Loaded {} tensors\n", stats.loaded);

    // Generate test data
    let image = create_test_image(config.image_size, config.image_size, config.channels);
    let num_steps = 20;
    let actions: Vec<Vec<f32>> = (0..num_steps)
        .map(|i| {
            // Varied actions for a meaningful trajectory
            let t = i as f32 / num_steps as f32;
            let mut a = vec![0.0f32; config.action_dim];
            a[0] = (t * std::f32::consts::PI).sin() * 0.5;
            a[1] = (t * std::f32::consts::PI).cos() * 0.3;
            a
        })
        .collect();

    // ── Baseline: f32 ─────────────────────────────────────────────
    println!("Running f32 baseline...");
    let z_f32 = model.encode(&image, config.image_size, config.image_size);
    let start = Instant::now();
    let traj_f32 = model.rollout(&z_f32, &actions);
    let f32_ms = start.elapsed().as_secs_f64() * 1000.0;
    let f32_size = estimate_f32_size(&config);

    println!("  Encode L2: {:.4}, Rollout: {:.1}ms ({} steps)", l2_norm(&z_f32), f32_ms, num_steps);
    println!();

    // ── Results table ─────────────────────────────────────────────
    let mut results: Vec<CompressionResult> = Vec::new();

    results.push(CompressionResult {
        name: "f32 baseline".into(),
        size_bytes: f32_size,
        cos_encode: 1.0,
        cos_step1: 1.0,
        cos_step10: 1.0,
        cos_step20: 1.0,
        rollout_ms: f32_ms,
        notes: "reference".into(),
    });

    // ── INT8 quantized ────────────────────────────────────────────
    println!("Quantizing to INT8...");
    let int8_model = quantize_lewm(&model);
    let z_int8 = int8_model.encode(&image, config.image_size, config.image_size);
    let start = Instant::now();
    let traj_int8 = int8_model.rollout(&z_int8, &actions);
    let int8_ms = start.elapsed().as_secs_f64() * 1000.0;

    results.push(CompressionResult {
        name: "INT8 predictor".into(),
        size_bytes: estimate_int8_size(&config),
        cos_encode: cosine_sim(&z_f32, &z_int8),
        cos_step1: cosine_sim(&traj_f32[0], &traj_int8[0]),
        cos_step10: cosine_sim(&traj_f32[9], &traj_int8[9]),
        cos_step20: cosine_sim(&traj_f32[num_steps-1], &traj_int8[num_steps-1]),
        rollout_ms: int8_ms,
        notes: "predictor quantized, encoder f32".into(),
    });

    // ── Q4 quantized (basic) ──────────────────────────────────────
    println!("Quantizing to Q4...");
    let q4_model = quantize_lewm_q4(&model);
    let z_q4 = q4_model.encode(&image, config.image_size, config.image_size);
    let start = Instant::now();
    let traj_q4 = q4_model.rollout(&z_q4, &actions);
    let q4_ms = start.elapsed().as_secs_f64() * 1000.0;

    results.push(CompressionResult {
        name: "Q4 predictor".into(),
        size_bytes: estimate_q4_size(&config),
        cos_encode: cosine_sim(&z_f32, &z_q4),
        cos_step1: cosine_sim(&traj_f32[0], &traj_q4[0]),
        cos_step10: cosine_sim(&traj_f32[9], &traj_q4[9]),
        cos_step20: cosine_sim(&traj_f32[num_steps-1], &traj_q4[num_steps-1]),
        rollout_ms: q4_ms,
        notes: "predictor quantized, encoder f32".into(),
    });

    // ── Q4 with dequant caching ───────────────────────────────────
    println!("Quantizing to Q4 (cached dequant)...");
    let cq4_model = cached_q4_lewm(&model);
    let z_cq4 = cq4_model.encode(&image, config.image_size, config.image_size);
    let start = Instant::now();
    let traj_cq4 = cq4_model.rollout(&z_cq4, &actions);
    let cq4_ms = start.elapsed().as_secs_f64() * 1000.0;

    results.push(CompressionResult {
        name: "Q4 cached".into(),
        size_bytes: estimate_q4_size(&config),  // same storage, faster forward
        cos_encode: cosine_sim(&z_f32, &z_cq4),
        cos_step1: cosine_sim(&traj_f32[0], &traj_cq4[0]),
        cos_step10: cosine_sim(&traj_f32[9], &traj_cq4[9]),
        cos_step20: cosine_sim(&traj_f32[num_steps-1], &traj_cq4[num_steps-1]),
        rollout_ms: cq4_ms,
        notes: "Q4 with f32 dequant cache (faster, same size on disk)".into(),
    });

    // ── Full quantization: INT8 encoder + Q4 predictor ──────────────
    println!("Quantizing to INT8 encoder + Q4 predictor (fully quantized)...");
    let full_q_model = quantize_lewm_full(&model);
    let z_fq = full_q_model.encode(&image, config.image_size, config.image_size);
    let start = Instant::now();
    let traj_fq = full_q_model.rollout(&z_fq, &actions);
    let fq_ms = start.elapsed().as_secs_f64() * 1000.0;

    results.push(CompressionResult {
        name: "INT8enc + Q4pred".into(),
        size_bytes: full_q_model.model_size_bytes(),
        cos_encode: cosine_sim(&z_f32, &z_fq),
        cos_step1: cosine_sim(&traj_f32[0], &traj_fq[0]),
        cos_step10: cosine_sim(&traj_f32[9], &traj_fq[9]),
        cos_step20: cosine_sim(&traj_f32[num_steps-1], &traj_fq[num_steps-1]),
        rollout_ms: fq_ms,
        notes: "INT8 ViT encoder + Q4 predictor".into(),
    });

    // ── Full Q4: Q4 encoder + Q4 predictor (~8MB) ───────────────────
    println!("Quantizing to Q4 encoder + Q4 predictor (full Q4)...");
    let q4_full_model = quantize_lewm_q4_full(&model);
    let z_q4f = q4_full_model.encode(&image, config.image_size, config.image_size);
    let start = Instant::now();
    let traj_q4f = q4_full_model.rollout(&z_q4f, &actions);
    let q4f_ms = start.elapsed().as_secs_f64() * 1000.0;

    results.push(CompressionResult {
        name: "Q4enc + Q4pred".into(),
        size_bytes: q4_full_model.model_size_bytes(),
        cos_encode: cosine_sim(&z_f32, &z_q4f),
        cos_step1: cosine_sim(&traj_f32[0], &traj_q4f[0]),
        cos_step10: cosine_sim(&traj_f32[9], &traj_q4f[9]),
        cos_step20: cosine_sim(&traj_f32[num_steps-1], &traj_q4f[num_steps-1]),
        rollout_ms: q4f_ms,
        notes: "Q4 encoder + Q4 predictor (~8MB)".into(),
    });

    // ── Q4 full + Wanda 20% predictor ───────────────────────────────
    println!("Quantizing to Q4 encoder + Wanda 20% + Q4 predictor...");
    let mut wanda20_model = reload_model(path, &config);
    let _ = wanda_prune_lewm_predictor(&mut wanda20_model, 0.2);
    let q4fw20 = quantize_lewm_q4_full(&wanda20_model);
    let z_q4fw20 = q4fw20.encode(&image, config.image_size, config.image_size);
    let start = Instant::now();
    let traj_q4fw20 = q4fw20.rollout(&z_q4fw20, &actions);
    let q4fw20_ms = start.elapsed().as_secs_f64() * 1000.0;

    results.push(CompressionResult {
        name: "Q4full+W20%".into(),
        size_bytes: q4fw20.model_size_bytes(),
        cos_encode: cosine_sim(&z_f32, &z_q4fw20),
        cos_step1: cosine_sim(&traj_f32[0], &traj_q4fw20[0]),
        cos_step10: cosine_sim(&traj_f32[9], &traj_q4fw20[9]),
        cos_step20: cosine_sim(&traj_f32[num_steps-1], &traj_q4fw20[num_steps-1]),
        rollout_ms: q4fw20_ms,
        notes: "Q4 enc + Wanda 20% Q4 pred".into(),
    });

    // ── Q4 full + Wanda 40% predictor ───────────────────────────────
    println!("Quantizing to Q4 encoder + Wanda 40% + Q4 predictor...");
    let mut wanda40_model = reload_model(path, &config);
    let _ = wanda_prune_lewm_predictor(&mut wanda40_model, 0.4);
    let q4fw40 = quantize_lewm_q4_full(&wanda40_model);
    let z_q4fw40 = q4fw40.encode(&image, config.image_size, config.image_size);
    let start = Instant::now();
    let traj_q4fw40 = q4fw40.rollout(&z_q4fw40, &actions);
    let q4fw40_ms = start.elapsed().as_secs_f64() * 1000.0;

    results.push(CompressionResult {
        name: "Q4full+W40%".into(),
        size_bytes: q4fw40.model_size_bytes(),
        cos_encode: cosine_sim(&z_f32, &z_q4fw40),
        cos_step1: cosine_sim(&traj_f32[0], &traj_q4fw40[0]),
        cos_step10: cosine_sim(&traj_f32[9], &traj_q4fw40[9]),
        cos_step20: cosine_sim(&traj_f32[num_steps-1], &traj_q4fw40[num_steps-1]),
        rollout_ms: q4fw40_ms,
        notes: "Q4 enc + Wanda 40% Q4 pred".into(),
    });

    // ── Ternary (2-bit) with RMSNorm fix ────────────────────────────
    println!("Quantizing to Ternary (2-bit) with RMSNorm fix...");
    let ternary_model = quantize_lewm_ternary(&model);
    let z_tern = ternary_model.encode(&image, config.image_size, config.image_size);
    let start = Instant::now();
    let traj_tern = ternary_model.rollout(&z_tern, &actions);
    let tern_ms = start.elapsed().as_secs_f64() * 1000.0;

    results.push(CompressionResult {
        name: "Ternary+RMSNorm".into(),
        size_bytes: ternary_model.model_size_bytes(),
        cos_encode: cosine_sim(&z_f32, &z_tern),
        cos_step1: cosine_sim(&traj_f32[0], &traj_tern[0]),
        cos_step10: cosine_sim(&traj_f32[9], &traj_tern[9]),
        cos_step20: cosine_sim(&traj_f32[num_steps-1], &traj_tern[num_steps-1]),
        rollout_ms: tern_ms,
        notes: "2-bit predictor + INT8 adaLN + RMSNorm stabilization".into(),
    });

    // ── Wanda 40% + Q4 cached (the interesting one for deployment) ──
    println!("Applying Wanda 40% pruning + Q4 cached...");
    let mut pruned_40c = reload_model(path, &config);
    let _ = wanda_prune_lewm_predictor(&mut pruned_40c, 0.4);
    let pq4c_40_model = cached_q4_lewm(&pruned_40c);
    let z_pq4c = pq4c_40_model.encode(&image, config.image_size, config.image_size);
    let start = Instant::now();
    let traj_pq4c = pq4c_40_model.rollout(&z_pq4c, &actions);
    let pq4c_ms = start.elapsed().as_secs_f64() * 1000.0;

    results.push(CompressionResult {
        name: "Wanda40+Q4 cached".into(),
        size_bytes: estimate_q4_size(&config),
        cos_encode: cosine_sim(&z_f32, &z_pq4c),
        cos_step1: cosine_sim(&traj_f32[0], &traj_pq4c[0]),
        cos_step10: cosine_sim(&traj_f32[9], &traj_pq4c[9]),
        cos_step20: cosine_sim(&traj_f32[num_steps-1], &traj_pq4c[num_steps-1]),
        rollout_ms: pq4c_ms,
        notes: "Wanda 40% prune → Q4 with dequant cache (fast)".into(),
    });

    // ── Wanda-pruned + Q4 experiments (uncached, for comparison) ──
    for (sparsity_pct, sparsity) in &[(20, 0.2f32), (40, 0.4), (60, 0.6)] {
        println!("Applying Wanda {}% pruning + Q4...", sparsity_pct);
        let mut pruned_model = reload_model(path, &config);
        let pruned_count = wanda_prune_lewm_predictor(&mut pruned_model, *sparsity);
        let pq4_model = quantize_lewm_q4(&pruned_model);
        let z_pq4 = pq4_model.encode(&image, config.image_size, config.image_size);
        let start = Instant::now();
        let traj_pq4 = pq4_model.rollout(&z_pq4, &actions);
        let pq4_ms = start.elapsed().as_secs_f64() * 1000.0;

        results.push(CompressionResult {
            name: format!("Wanda {}% + Q4", sparsity_pct),
            size_bytes: estimate_q4_size(&config),
            cos_encode: cosine_sim(&z_f32, &z_pq4),
            cos_step1: cosine_sim(&traj_f32[0], &traj_pq4[0]),
            cos_step10: cosine_sim(&traj_f32[9], &traj_pq4[9]),
            cos_step20: cosine_sim(&traj_f32[num_steps-1], &traj_pq4[num_steps-1]),
            rollout_ms: pq4_ms,
            notes: format!("{} weights zeroed then Q4", pruned_count),
        });
    }

    // ── Block-aware pruning + Q4 full (actually reduces size) ──────
    println!("Block-aware pruning 30% + Q4 full...");
    let mut blk_model = reload_model(path, &config);
    let blk_pruned = block_prune_lewm_predictor(&mut blk_model, 0.3);
    let q4fblk = quantize_lewm_q4_full(&blk_model);
    let z_blk = q4fblk.encode(&image, config.image_size, config.image_size);
    let start = Instant::now();
    let traj_blk = q4fblk.rollout(&z_blk, &actions);
    let blk_ms = start.elapsed().as_secs_f64() * 1000.0;
    // Estimate actual size: Q4 blocks with scale==0 can be skipped
    let zero_blocks = count_zero_q4_blocks(&q4fblk);
    let saved_bytes = zero_blocks * 20; // 20 bytes per skipped block
    let actual_size = q4fblk.model_size_bytes() - saved_bytes;

    results.push(CompressionResult {
        name: "BlkPrune30+Q4f".into(),
        size_bytes: actual_size,
        cos_encode: cosine_sim(&z_f32, &z_blk),
        cos_step1: cosine_sim(&traj_f32[0], &traj_blk[0]),
        cos_step10: cosine_sim(&traj_f32[9], &traj_blk[9]),
        cos_step20: cosine_sim(&traj_f32[num_steps-1], &traj_blk[num_steps-1]),
        rollout_ms: blk_ms,
        notes: format!("block prune 30% + Q4 full ({} blocks zeroed, saves {}KB)", blk_pruned, saved_bytes / 1024),
    });

    // ── Print results table ───────────────────────────────────────
    println!();
    println!("╔════════════════════╦═════════╦══════════╦══════════╦══════════╦══════════╦═════════╗");
    println!("║ Config             ║ Size    ║ cos enc  ║ cos@1    ║ cos@10   ║ cos@20   ║ ms/roll ║");
    println!("╠════════════════════╬═════════╬══════════╬══════════╬══════════╬══════════╬═════════╣");
    for r in &results {
        let size_mb = r.size_bytes as f64 / 1_048_576.0;
        println!("║ {:<18} ║ {:>5.1}MB ║ {:<8.6} ║ {:<8.6} ║ {:<8.6} ║ {:<8.6} ║ {:>5.1}ms ║",
            r.name, size_mb,
            r.cos_encode, r.cos_step1, r.cos_step10, r.cos_step20,
            r.rollout_ms);
    }
    println!("╚════════════════════╩═════════╩══════════╩══════════╩══════════╩══════════╩═════════╝");

    // ── Recommendations ───────────────────────────────────────────
    println!();
    println!("Recommendations:");
    for r in &results {
        let usable = r.cos_step20 > 0.90;
        let size_mb = r.size_bytes as f64 / 1_048_576.0;
        if usable {
            println!("  [OK]  {}: {:.1}MB, cos@20={:.4} — usable", r.name, size_mb, r.cos_step20);
        } else {
            println!("  [BAD] {}: {:.1}MB, cos@20={:.4} — quality too low", r.name, size_mb, r.cos_step20);
        }
    }

    // Find best usable config
    let best = results.iter()
        .filter(|r| r.cos_step20 > 0.90)
        .min_by(|a, b| a.size_bytes.cmp(&b.size_bytes));

    if let Some(best) = best {
        println!();
        println!("Best usable config: {} ({:.1}MB, cos@20={:.6})",
            best.name, best.size_bytes as f64 / 1_048_576.0, best.cos_step20);
        println!("  Notes: {}", best.notes);
    }
}

struct CompressionResult {
    name: String,
    size_bytes: usize,
    cos_encode: f32,
    cos_step1: f32,
    cos_step10: f32,
    cos_step20: f32,
    rollout_ms: f64,
    notes: String,
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

/// Estimate f32 model size in bytes.
fn estimate_f32_size(config: &LeWMConfig) -> usize {
    let h = config.predictor_hidden;
    let inner = config.predictor_inner_dim;
    let inter = config.predictor_inter;
    let layers = config.predictor_layers;

    // Predictor layers: adaLN + QKV + attn_out + MLP
    let per_layer = (h * 6 * h) + (h * 3 * inner) + (inner * h) + (h * inter) + (inter * h);
    let predictor = per_layer * layers * 4; // f32

    // Encoder (ViT): approx
    let enc_h = config.encoder_hidden;
    let enc_inter = config.encoder_inter;
    let per_enc_layer = (enc_h * 3 * enc_h) + (enc_h * enc_h) + (enc_h * enc_inter) + (enc_inter * enc_h);
    let encoder = per_enc_layer * config.encoder_layers * 4;

    // Embeddings, projections
    let patches = (config.image_size / config.patch_size).pow(2);
    let misc = (config.patch_size * config.patch_size * config.channels * enc_h  // patch embed
        + patches * enc_h  // pos embed
        + enc_h * h  // projector
        + config.action_dim * h  // action embed
        + h * config.latent_dim  // pred_proj
    ) * 4;

    predictor + encoder + misc
}

/// Estimate INT8 model size: predictor INT8, rest f32.
fn estimate_int8_size(config: &LeWMConfig) -> usize {
    let f32_total = estimate_f32_size(config);
    let h = config.predictor_hidden;
    let inner = config.predictor_inner_dim;
    let inter = config.predictor_inter;
    let layers = config.predictor_layers;

    // Predictor weight bytes at f32
    let per_layer_f32 = ((h * 6 * h) + (h * 3 * inner) + (inner * h) + (h * inter) + (inter * h)) * 4;
    let predictor_f32 = per_layer_f32 * layers;

    // INT8 weight bytes (weight + scale per row)
    let per_layer_int8 = ((h * 6 * h) + (h * 3 * inner) + (inner * h) + (h * inter) + (inter * h))
        + (6 * h + 3 * inner + h + inter + h) * 4; // scales
    let predictor_int8 = per_layer_int8 * layers;

    f32_total - predictor_f32 + predictor_int8
}

/// Estimate Q4 model size: predictor Q4, rest f32.
fn estimate_q4_size(config: &LeWMConfig) -> usize {
    let f32_total = estimate_f32_size(config);
    let h = config.predictor_hidden;
    let inner = config.predictor_inner_dim;
    let inter = config.predictor_inter;
    let layers = config.predictor_layers;

    // Predictor weight elements
    let per_layer_elems = (h * 6 * h) + (h * 3 * inner) + (inner * h) + (h * inter) + (inter * h);
    let predictor_f32_bytes = per_layer_elems * 4 * layers;

    // Q4: 32 elements per block, each block = 4 (scale) + 16 (nibbles) = 20 bytes
    let per_layer_blocks = (per_layer_elems + 31) / 32;
    let predictor_q4_bytes = per_layer_blocks * 20 * layers;

    f32_total - predictor_f32_bytes + predictor_q4_bytes
}

/// Reload the f32 model from disk (needed since AlignedBuffer doesn't impl Clone).
fn reload_model(path: &Path, config: &LeWMConfig) -> LeWorldModel {
    let mut model = LeWorldModel::from_config(config);
    let weights = load_safetensors(path).expect("Failed to reload weights");
    model.load_weights(weights).expect("Failed to load weights");
    model
}

/// Apply Wanda-style magnitude pruning to LEWM predictor layers.
///
/// Prunes the large weight matrices in each adaLN layer by zeroing out
/// the bottom `sparsity` fraction per output row (by magnitude).
/// Returns total number of pruned weights.
fn wanda_prune_lewm_predictor(model: &mut LeWorldModel, sparsity: f32) -> usize {
    use synapse_inference::pruning::wanda::wanda_prune_matrix;

    let hidden = model.config.predictor_hidden;
    let inner_dim = model.config.predictor_inner_dim;
    let inter = model.config.predictor_inter;
    let mut total = 0;

    let norms_h = vec![1.0f32; hidden];
    let norms_inner = vec![1.0f32; inner_dim];
    let norms_inter = vec![1.0f32; inter];

    for layer in &mut model.predictor_layers {
        // adaLN weight: [6*hidden, hidden]
        let six_h = 6 * hidden;
        if layer.adaln_weight.len() == six_h * hidden {
            total += wanda_prune_matrix(&mut layer.adaln_weight, six_h, hidden, &norms_h, sparsity);
        }

        // QKV: [3*inner_dim, hidden]
        let three_inner = 3 * inner_dim;
        if layer.to_qkv.len() == three_inner * hidden {
            total += wanda_prune_matrix(&mut layer.to_qkv, three_inner, hidden, &norms_h, sparsity);
        }

        // attn_out: [hidden, inner_dim]
        if layer.attn_out_weight.len() == hidden * inner_dim {
            total += wanda_prune_matrix(&mut layer.attn_out_weight, hidden, inner_dim, &norms_inner, sparsity);
        }

        // MLP up: [inter, hidden]
        if layer.mlp_up_weight.len() == inter * hidden {
            total += wanda_prune_matrix(&mut layer.mlp_up_weight, inter, hidden, &norms_h, sparsity);
        }

        // MLP down: [hidden, inter]
        if layer.mlp_down_weight.len() == hidden * inter {
            total += wanda_prune_matrix(&mut layer.mlp_down_weight, hidden, inter, &norms_inter, sparsity);
        }
    }

    total
}

/// Block-aware pruning: zero entire 32-element blocks (Q4 block granularity).
///
/// For each weight matrix row, computes L2 norm of each 32-element block,
/// sorts by norm, and zeros the bottom `sparsity` fraction of blocks.
/// This produces full-zero Q4 blocks that can be skipped in a sparse format.
/// Returns total number of blocks zeroed.
fn block_prune_lewm_predictor(model: &mut LeWorldModel, sparsity: f32) -> usize {
    let hidden = model.config.predictor_hidden;
    let inner_dim = model.config.predictor_inner_dim;
    let inter = model.config.predictor_inter;
    let mut total_blocks = 0;

    for layer in &mut model.predictor_layers {
        // Prune each weight matrix at block granularity
        total_blocks += block_prune_matrix(&mut layer.adaln_weight, 6 * hidden, hidden, sparsity);
        total_blocks += block_prune_matrix(&mut layer.to_qkv, 3 * inner_dim, hidden, sparsity);
        total_blocks += block_prune_matrix(&mut layer.attn_out_weight, hidden, inner_dim, sparsity);
        total_blocks += block_prune_matrix(&mut layer.mlp_up_weight, inter, hidden, sparsity);
        total_blocks += block_prune_matrix(&mut layer.mlp_down_weight, hidden, inter, sparsity);
    }
    total_blocks
}

/// Zero entire 32-element blocks with lowest L2 norm in a weight matrix.
fn block_prune_matrix(weights: &mut [f32], rows: usize, cols: usize, sparsity: f32) -> usize {
    let block_size = 32;
    let padded_cols = (cols + block_size - 1) / block_size * block_size;
    let blocks_per_row = padded_cols / block_size;
    let prune_count = (blocks_per_row as f32 * sparsity) as usize;
    if prune_count == 0 { return 0; }

    let mut total_pruned = 0;
    for row in 0..rows {
        // Compute L2 norm of each block in this row
        let mut block_norms: Vec<(usize, f32)> = (0..blocks_per_row)
            .map(|b| {
                let start = b * block_size;
                let mut norm = 0.0f32;
                for i in 0..block_size {
                    let col = start + i;
                    if col < cols {
                        let w = weights[row * cols + col];
                        norm += w * w;
                    }
                }
                (b, norm.sqrt())
            })
            .collect();

        // Sort by norm ascending (weakest blocks first)
        block_norms.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

        // Zero the weakest blocks
        for &(b, _) in block_norms.iter().take(prune_count) {
            let start = b * block_size;
            for i in 0..block_size {
                let col = start + i;
                if col < cols {
                    weights[row * cols + col] = 0.0;
                }
            }
            total_pruned += 1;
        }
    }
    total_pruned
}

/// Count Q4 blocks with scale==0 (all-zero) in a Q4FullLeWM.
fn count_zero_q4_blocks(model: &synapse_inference::quantization::Q4FullLeWM) -> usize {
    let mut count = 0;
    // Encoder blocks
    for layer in &model.encoder_layers {
        for ql in [&layer.w_q, &layer.w_k, &layer.w_v, &layer.w_o, &layer.ffn_up, &layer.ffn_down] {
            count += ql.blocks.iter().filter(|b| b.scale == 0.0).count();
        }
    }
    // Predictor blocks
    for layer in &model.predictor_layers {
        for ql in [&layer.adaln_linear, &layer.to_qkv, &layer.attn_out, &layer.mlp_up, &layer.mlp_down] {
            count += ql.blocks.iter().filter(|b| b.scale == 0.0).count();
        }
    }
    count
}
