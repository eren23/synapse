//! Export LEWM models in compact Q4 binary format for WASM loading.
//!
//! Supports four modes:
//!   - `q4-pred`:     Q4 predictor only, f32 encoder (~17MB)
//!   - `full`:        INT8 encoder + Q4 predictor (~10MB)
//!   - `wanda20-q4`:  Wanda 20% prune then Q4 (~17MB, better compressed)
//!   - `wanda40-q4`:  Wanda 40% prune then Q4 (~17MB, better compressed)
//!
//! Binary format:
//!   [4 bytes] Magic: "LQ40"
//!   [4 bytes] u32 LE: JSON config length
//!   [N bytes] JSON config (model dimensions, mode, quantization info)
//!   [rest]    Weight data
//!
//! Usage:
//!   cargo run -p synapse --release --example export_lewm_q4 -- \
//!     --checkpoint /tmp/lewm-pusht/pusht/lejepa_weights.safetensors \
//!     --mode q4-pred \
//!     --output web/lewm-q4.bin

use std::io::Write;
use std::path::{Path, PathBuf};

use synapse_inference::models::{LeWMConfig, LeWorldModel};
use synapse_inference::quantization::{
    quantize_lewm_q4, quantize_lewm_full,
    Q4Linear, QuantizedLinear,
    QuantizedQ4AdaLNLayer, QuantizedQ4LeWM,
    FullyQuantizedLeWM,
};
use synapse_inference::weight_loading::load_safetensors;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let mut checkpoint: Option<PathBuf> = None;
    let mut mode: String = "q4-pred".into();
    let mut output: Option<PathBuf> = None;
    let mut config_path: Option<PathBuf> = None;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--checkpoint" | "-c" => { i += 1; checkpoint = Some(PathBuf::from(&args[i])); }
            "--mode" | "-m" => { i += 1; mode = args[i].clone(); }
            "--output" | "-o" => { i += 1; output = Some(PathBuf::from(&args[i])); }
            "--config" => { i += 1; config_path = Some(PathBuf::from(&args[i])); }
            "--help" | "-h" => {
                eprintln!("Usage: export_lewm_q4 --checkpoint <path> --mode <mode> --output <path> [--config <json>]");
                eprintln!();
                eprintln!("Modes:");
                eprintln!("  q4-pred     Q4 predictor only, f32 encoder (~17MB)");
                eprintln!("  full        INT8 encoder + Q4 predictor (~10MB)");
                eprintln!("  wanda20-q4  Wanda 20% prune then Q4 (~17MB, better compressed)");
                eprintln!("  wanda40-q4  Wanda 40% prune then Q4 (~17MB, better compressed)");
                eprintln!();
                eprintln!("Options:");
                eprintln!("  --config    config.json from convert_lewm_ckpt.py (for slim variants)");
                std::process::exit(0);
            }
            _ => {}
        }
        i += 1;
    }

    let checkpoint = checkpoint.expect("--checkpoint required");
    let output = output.expect("--output required");

    if !matches!(mode.as_str(), "q4-pred" | "full" | "wanda20-q4" | "wanda40-q4") {
        eprintln!("Unknown mode '{}'. Use: q4-pred, full, wanda20-q4, wanda40-q4", mode);
        std::process::exit(1);
    }

    // 1. Load f32 model from safetensors
    let path = Path::new(&checkpoint);
    if !path.exists() {
        eprintln!("Checkpoint not found at {}", checkpoint.display());
        std::process::exit(1);
    }

    let config = if let Some(cp) = config_path {
        LeWMConfig::from_json(&cp).expect("Failed to load config.json")
    } else {
        LeWMConfig::pusht()
    };
    eprintln!("Loading f32 LEWM from {}...", checkpoint.display());
    eprintln!("  Config: {}d latent, {}e/{}p", config.latent_dim, config.encoder_layers, config.predictor_layers);
    let mut model = LeWorldModel::from_config(&config);
    let weights = load_safetensors(path).expect("Failed to load safetensors");
    let stats = model.load_weights(weights).expect("Failed to load weights");
    eprintln!("  Loaded {} tensors", stats.loaded);

    // 2. Apply mode-specific quantization and export
    match mode.as_str() {
        "q4-pred" => {
            eprintln!("Quantizing predictor to Q4 (f32 encoder)...");
            let q4_model = quantize_lewm_q4(&model);
            export_q4_pred(&q4_model, &config, &output);
        }
        "full" => {
            eprintln!("Quantizing: INT8 encoder + Q4 predictor...");
            let full_model = quantize_lewm_full(&model);
            export_full(&full_model, &config, &output);
        }
        "wanda20-q4" => {
            eprintln!("Applying Wanda 20% pruning to predictor...");
            let pruned = wanda_prune_lewm_predictor(&mut model, 0.2);
            eprintln!("  Pruned {} weights", pruned);
            eprintln!("Quantizing pruned predictor to Q4...");
            let q4_model = quantize_lewm_q4(&model);
            export_q4_pred_with_mode(&q4_model, &config, &output, "wanda20-q4");
        }
        "wanda40-q4" => {
            eprintln!("Applying Wanda 40% pruning to predictor...");
            let pruned = wanda_prune_lewm_predictor(&mut model, 0.4);
            eprintln!("  Pruned {} weights", pruned);
            eprintln!("Quantizing pruned predictor to Q4...");
            let q4_model = quantize_lewm_q4(&model);
            export_q4_pred_with_mode(&q4_model, &config, &output, "wanda40-q4");
        }
        _ => unreachable!(),
    }

    let size_bytes = std::fs::metadata(&output).unwrap().len();
    let size_mb = size_bytes as f64 / 1_048_576.0;
    eprintln!();
    eprintln!("Exported {} to {}", mode, output.display());
    eprintln!("  Size: {:.2} MB ({} bytes)", size_mb, size_bytes);
}

// ---------------------------------------------------------------------------
// Export: q4-pred mode (and wanda40-q4 which uses same layout)
// ---------------------------------------------------------------------------

fn export_q4_pred(model: &QuantizedQ4LeWM, config: &LeWMConfig, output: &Path) {
    export_q4_pred_with_mode(model, config, output, "q4-pred");
}

fn export_q4_pred_with_mode(
    model: &QuantizedQ4LeWM,
    config: &LeWMConfig,
    output: &Path,
    mode_str: &str,
) {
    let mut file = std::fs::File::create(output).expect("Failed to create output file");

    // Magic
    file.write_all(b"LQ40").unwrap();

    // JSON config
    let config_json = serde_json::json!({
        "mode": mode_str,
        "image_size": config.image_size,
        "patch_size": config.patch_size,
        "encoder_hidden": config.encoder_hidden,
        "encoder_layers": config.encoder_layers,
        "encoder_heads": config.encoder_heads,
        "encoder_inter": config.encoder_inter,
        "predictor_hidden": config.predictor_hidden,
        "predictor_layers": config.predictor_layers,
        "predictor_heads": config.predictor_heads,
        "predictor_inner_dim": config.predictor_inner_dim,
        "predictor_inter": config.predictor_inter,
        "action_dim": config.action_dim,
        "latent_dim": config.latent_dim,
        "channels": config.channels,
    });
    let config_bytes = serde_json::to_vec(&config_json).unwrap();
    file.write_all(&(config_bytes.len() as u32).to_le_bytes()).unwrap();
    file.write_all(&config_bytes).unwrap();

    let mut total_written: usize = 8 + config_bytes.len();

    // --- Encoder weights (f32) ---
    eprintln!("  Writing f32 encoder...");

    // Patch projection
    total_written += write_f32(&mut file, &model.encoder.patch_proj);
    total_written += write_f32(&mut file, &model.encoder.patch_proj_bias);
    total_written += write_f32(&mut file, &model.encoder.cls_token);
    total_written += write_f32(&mut file, &model.encoder.pos_embed);

    // Encoder layers
    for layer in &model.encoder.layers {
        total_written += write_f32(&mut file, &layer.attn_norm_weight);
        total_written += write_f32(&mut file, &layer.attn_norm_bias);
        total_written += write_f32(&mut file, &layer.w_q);
        total_written += write_f32(&mut file, &layer.q_bias);
        total_written += write_f32(&mut file, &layer.w_k);
        total_written += write_f32(&mut file, &layer.k_bias);
        total_written += write_f32(&mut file, &layer.w_v);
        total_written += write_f32(&mut file, &layer.v_bias);
        total_written += write_f32(&mut file, &layer.w_o);
        total_written += write_f32(&mut file, &layer.o_bias);
        total_written += write_f32(&mut file, &layer.ffn_norm_weight);
        total_written += write_f32(&mut file, &layer.ffn_norm_bias);
        total_written += write_f32(&mut file, &layer.ffn_up);
        total_written += write_f32(&mut file, &layer.ffn_up_bias);
        total_written += write_f32(&mut file, &layer.ffn_down);
        total_written += write_f32(&mut file, &layer.ffn_down_bias);
    }

    // Encoder final norm
    total_written += write_f32(&mut file, &model.encoder.final_norm_weight);
    total_written += write_f32(&mut file, &model.encoder.final_norm_bias);

    let encoder_bytes = total_written - 8 - config_bytes.len();
    eprintln!("    Encoder: {:.2} MB", encoder_bytes as f64 / 1_048_576.0);

    // --- Predictor Q4 layers ---
    eprintln!("  Writing Q4 predictor...");
    let pred_start = total_written;

    // Predictor positional embeddings (f32)
    total_written += write_f32_vec(&mut file, &model.predictor_pos_embed);

    // Predictor layers (Q4Linear blocks + f32 biases/norms)
    for layer in &model.predictor_layers {
        total_written += write_q4_layer(&mut file, layer);
    }

    // Predictor final norm (f32)
    total_written += write_f32_vec(&mut file, &model.predictor_norm_weight);
    total_written += write_f32_vec(&mut file, &model.predictor_norm_bias);

    let pred_bytes = total_written - pred_start;
    eprintln!("    Predictor: {:.2} MB", pred_bytes as f64 / 1_048_576.0);

    // --- Action encoder (f32) ---
    eprintln!("  Writing f32 action encoder...");
    let action_start = total_written;
    total_written += write_f32_vec(&mut file, &model.action_conv_weight);
    total_written += write_f32_vec(&mut file, &model.action_conv_bias);
    total_written += write_f32_vec(&mut file, &model.action_mlp1_weight);
    total_written += write_f32_vec(&mut file, &model.action_mlp1_bias);
    total_written += write_f32_vec(&mut file, &model.action_mlp2_weight);
    total_written += write_f32_vec(&mut file, &model.action_mlp2_bias);
    let action_bytes = total_written - action_start;
    eprintln!("    Action encoder: {:.2} MB", action_bytes as f64 / 1_048_576.0);

    // --- Projectors (f32) ---
    eprintln!("  Writing f32 projectors...");
    let proj_start = total_written;
    total_written += write_projection_head(&mut file, &model.projector);
    total_written += write_projection_head(&mut file, &model.pred_proj);
    let proj_bytes = total_written - proj_start;
    eprintln!("    Projectors: {:.2} MB", proj_bytes as f64 / 1_048_576.0);

    // --- Input/Cond projections (f32, for slim bottleneck models) ---
    total_written += write_f32_vec(&mut file, &model.input_proj_weight);
    total_written += write_f32_vec(&mut file, &model.input_proj_bias);
    total_written += write_f32_vec(&mut file, &model.cond_proj_weight);
    total_written += write_f32_vec(&mut file, &model.cond_proj_bias);

    eprintln!("  Total: {:.2} MB", total_written as f64 / 1_048_576.0);
}

// ---------------------------------------------------------------------------
// Export: full mode (INT8 encoder + Q4 predictor)
// ---------------------------------------------------------------------------

fn export_full(model: &FullyQuantizedLeWM, config: &LeWMConfig, output: &Path) {
    let mut file = std::fs::File::create(output).expect("Failed to create output file");

    // Magic
    file.write_all(b"LQ40").unwrap();

    // JSON config
    let config_json = serde_json::json!({
        "mode": "full",
        "image_size": config.image_size,
        "patch_size": config.patch_size,
        "encoder_hidden": config.encoder_hidden,
        "encoder_layers": config.encoder_layers,
        "encoder_heads": config.encoder_heads,
        "encoder_inter": config.encoder_inter,
        "predictor_hidden": config.predictor_hidden,
        "predictor_layers": config.predictor_layers,
        "predictor_heads": config.predictor_heads,
        "predictor_inner_dim": config.predictor_inner_dim,
        "predictor_inter": config.predictor_inter,
        "action_dim": config.action_dim,
        "latent_dim": config.latent_dim,
        "channels": config.channels,
    });
    let config_bytes = serde_json::to_vec(&config_json).unwrap();
    file.write_all(&(config_bytes.len() as u32).to_le_bytes()).unwrap();
    file.write_all(&config_bytes).unwrap();

    let mut total_written: usize = 8 + config_bytes.len();

    // --- INT8 Encoder ---
    eprintln!("  Writing INT8 encoder...");
    let enc_start = total_written;

    // Patch projection (f32, small)
    total_written += write_f32_vec(&mut file, &model.patch_proj);
    total_written += write_f32_vec(&mut file, &model.patch_proj_bias);
    total_written += write_f32_vec(&mut file, &model.cls_token);
    total_written += write_f32_vec(&mut file, &model.pos_embed);

    // INT8 encoder layers
    for layer in &model.encoder_layers {
        total_written += write_f32_vec(&mut file, &layer.attn_norm_weight);
        total_written += write_f32_vec(&mut file, &layer.attn_norm_bias);
        total_written += write_quantized_linear(&mut file, &layer.w_q);
        total_written += write_f32_vec(&mut file, &layer.q_bias);
        total_written += write_quantized_linear(&mut file, &layer.w_k);
        total_written += write_f32_vec(&mut file, &layer.k_bias);
        total_written += write_quantized_linear(&mut file, &layer.w_v);
        total_written += write_f32_vec(&mut file, &layer.v_bias);
        total_written += write_quantized_linear(&mut file, &layer.w_o);
        total_written += write_f32_vec(&mut file, &layer.o_bias);
        total_written += write_f32_vec(&mut file, &layer.ffn_norm_weight);
        total_written += write_f32_vec(&mut file, &layer.ffn_norm_bias);
        total_written += write_quantized_linear(&mut file, &layer.ffn_up);
        total_written += write_f32_vec(&mut file, &layer.ffn_up_bias);
        total_written += write_quantized_linear(&mut file, &layer.ffn_down);
        total_written += write_f32_vec(&mut file, &layer.ffn_down_bias);
    }

    // Final norm (f32)
    total_written += write_f32_vec(&mut file, &model.final_norm_weight);
    total_written += write_f32_vec(&mut file, &model.final_norm_bias);

    let enc_bytes = total_written - enc_start;
    eprintln!("    Encoder (INT8): {:.2} MB", enc_bytes as f64 / 1_048_576.0);

    // --- Q4 Predictor ---
    eprintln!("  Writing Q4 predictor...");
    let pred_start = total_written;

    total_written += write_f32_vec(&mut file, &model.predictor_pos_embed);

    for layer in &model.predictor_layers {
        total_written += write_q4_layer(&mut file, layer);
    }

    total_written += write_f32_vec(&mut file, &model.predictor_norm_weight);
    total_written += write_f32_vec(&mut file, &model.predictor_norm_bias);

    let pred_bytes = total_written - pred_start;
    eprintln!("    Predictor (Q4): {:.2} MB", pred_bytes as f64 / 1_048_576.0);

    // --- Action encoder (f32) ---
    eprintln!("  Writing f32 action encoder...");
    let action_start = total_written;
    total_written += write_f32_vec(&mut file, &model.action_conv_weight);
    total_written += write_f32_vec(&mut file, &model.action_conv_bias);
    total_written += write_f32_vec(&mut file, &model.action_mlp1_weight);
    total_written += write_f32_vec(&mut file, &model.action_mlp1_bias);
    total_written += write_f32_vec(&mut file, &model.action_mlp2_weight);
    total_written += write_f32_vec(&mut file, &model.action_mlp2_bias);
    let action_bytes = total_written - action_start;
    eprintln!("    Action encoder: {:.2} MB", action_bytes as f64 / 1_048_576.0);

    // --- Projectors (f32) ---
    eprintln!("  Writing f32 projectors...");
    let proj_start = total_written;
    total_written += write_projection_head_from_lewm(&mut file, &model.projector);
    total_written += write_projection_head_from_lewm(&mut file, &model.pred_proj);
    let proj_bytes = total_written - proj_start;
    eprintln!("    Projectors: {:.2} MB", proj_bytes as f64 / 1_048_576.0);

    // --- Input/Cond projections (f32, for slim bottleneck models) ---
    total_written += write_f32_vec(&mut file, &model.input_proj_weight);
    total_written += write_f32_vec(&mut file, &model.input_proj_bias);
    total_written += write_f32_vec(&mut file, &model.cond_proj_weight);
    total_written += write_f32_vec(&mut file, &model.cond_proj_bias);

    eprintln!("  Total: {:.2} MB", total_written as f64 / 1_048_576.0);
}

// ---------------------------------------------------------------------------
// Serialization helpers
// ---------------------------------------------------------------------------

/// Write a length-prefixed f32 slice (from AlignedBuffer via Deref<[f32]>).
/// Returns bytes written.
fn write_f32(f: &mut std::fs::File, data: &[f32]) -> usize {
    f.write_all(&(data.len() as u32).to_le_bytes()).unwrap();
    for &v in data {
        f.write_all(&v.to_le_bytes()).unwrap();
    }
    4 + data.len() * 4
}

/// Write a length-prefixed f32 Vec.
fn write_f32_vec(f: &mut std::fs::File, data: &[f32]) -> usize {
    write_f32(f, data)
}

/// Write a Q4Linear: [u32 out_features] [u32 in_features] [u32 num_blocks] [blocks...]
/// Each Q4Block is: 4 bytes scale (LE f32) + 16 bytes nibbles = 20 bytes.
fn write_q4_linear(f: &mut std::fs::File, ql: &Q4Linear) -> usize {
    f.write_all(&(ql.out_features as u32).to_le_bytes()).unwrap();
    f.write_all(&(ql.in_features as u32).to_le_bytes()).unwrap();
    let total_blocks = ql.blocks.len();
    f.write_all(&(total_blocks as u32).to_le_bytes()).unwrap();

    // Sparse format: write bitmap (1 bit per block, 1=non-zero, 0=zero)
    // then only non-zero blocks. Zero blocks have scale==0.0.
    let bitmap_bytes = (total_blocks + 7) / 8;
    let mut bitmap = vec![0u8; bitmap_bytes];
    let mut nonzero_count = 0u32;
    for (i, block) in ql.blocks.iter().enumerate() {
        if block.scale != 0.0 {
            bitmap[i / 8] |= 1 << (i % 8);
            nonzero_count += 1;
        }
    }
    f.write_all(&nonzero_count.to_le_bytes()).unwrap();
    f.write_all(&bitmap).unwrap();

    // Write only non-zero blocks
    for block in &ql.blocks {
        if block.scale != 0.0 {
            f.write_all(&block.scale.to_le_bytes()).unwrap();
            f.write_all(&block.nibbles).unwrap();
        }
    }
    12 + 4 + bitmap_bytes + nonzero_count as usize * 20
}

/// Write a QuantizedLinear (INT8): [u32 out] [u32 in] [u32 len_weights] [i8 weights...] [u32 len_scales] [f32 scales...]
fn write_quantized_linear(f: &mut std::fs::File, ql: &QuantizedLinear) -> usize {
    f.write_all(&(ql.out_features as u32).to_le_bytes()).unwrap();
    f.write_all(&(ql.in_features as u32).to_le_bytes()).unwrap();
    // INT8 weights
    f.write_all(&(ql.weights_int8.len() as u32).to_le_bytes()).unwrap();
    let bytes: Vec<u8> = ql.weights_int8.iter().map(|&v| v as u8).collect();
    f.write_all(&bytes).unwrap();
    // f32 scales
    f.write_all(&(ql.scales.len() as u32).to_le_bytes()).unwrap();
    for &s in &ql.scales {
        f.write_all(&s.to_le_bytes()).unwrap();
    }
    8 + 4 + ql.weights_int8.len() + 4 + ql.scales.len() * 4
}

/// Write all fields of a Q4-quantized adaLN predictor layer.
fn write_q4_layer(f: &mut std::fs::File, layer: &QuantizedQ4AdaLNLayer) -> usize {
    let mut n = 0;
    // Q4 weight matrices
    n += write_q4_linear(f, &layer.adaln_linear);
    n += write_f32_vec(f, &layer.adaln_bias);
    n += write_q4_linear(f, &layer.to_qkv);
    n += write_q4_linear(f, &layer.attn_out);
    n += write_f32_vec(f, &layer.attn_out_bias);
    // Norms (f32)
    n += write_f32_vec(f, &layer.attn_norm_weight);
    n += write_f32_vec(f, &layer.attn_norm_bias);
    n += write_f32_vec(f, &layer.mlp_norm_weight);
    n += write_f32_vec(f, &layer.mlp_norm_bias);
    // MLP Q4
    n += write_q4_linear(f, &layer.mlp_up);
    n += write_f32_vec(f, &layer.mlp_up_bias);
    n += write_q4_linear(f, &layer.mlp_down);
    n += write_f32_vec(f, &layer.mlp_down_bias);
    n
}

/// Write a ProjectionHead from QuantizedQ4LeWM (uses AlignedBuffer layers via Deref).
fn write_projection_head(
    f: &mut std::fs::File,
    proj: &synapse_inference::models::vision::lewm::ProjectionHead,
) -> usize {
    let mut n = 0;
    // Number of layers
    f.write_all(&(proj.layers.len() as u32).to_le_bytes()).unwrap();
    n += 4;
    for (weight, bias) in &proj.layers {
        let w_slice: &[f32] = weight;
        let b_slice: &[f32] = bias;
        n += write_f32(f, w_slice);
        n += write_f32(f, b_slice);
    }
    n
}

/// Write a ProjectionHead from FullyQuantizedLeWM (same struct, same layout).
fn write_projection_head_from_lewm(
    f: &mut std::fs::File,
    proj: &synapse_inference::models::vision::lewm::ProjectionHead,
) -> usize {
    write_projection_head(f, proj)
}

// ---------------------------------------------------------------------------
// Wanda pruning (inlined from lewm_compress.rs since it lives in an example)
// ---------------------------------------------------------------------------

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
