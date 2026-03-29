//! Export a Mamba model as a compact INT8 binary for WASM.
//!
//! Usage:
//!   cargo run --example export_mamba_int8 --release -- --model-dir models/mamba-130m --output web/ssm-demo/mamba-130m-int8.bin
//!
//! The output file contains:
//!   - 4 bytes: magic "SMI8"
//!   - JSON config (length-prefixed)
//!   - Binary weight blobs (f32 for small tensors, int8+scales for large projections)

use std::io::Write;
use std::path::PathBuf;

use synapse_inference::engine::InferenceEngine;
use synapse_inference::quantization::{QuantizedMambaModel, QuantizedLinear};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let mut model_dir: Option<PathBuf> = None;
    let mut output: Option<PathBuf> = None;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--model-dir" => { i += 1; model_dir = Some(PathBuf::from(&args[i])); }
            "--output" | "-o" => { i += 1; output = Some(PathBuf::from(&args[i])); }
            _ => {}
        }
        i += 1;
    }

    let model_dir = model_dir.expect("--model-dir required");
    let output = output.expect("--output required");

    eprintln!("Loading f32 model from {}...", model_dir.display());
    let engine = InferenceEngine::from_pretrained(&model_dir).expect("Failed to load model");
    let mamba = engine.ssm_model.as_ref().expect("Not an SSM model");

    // Downcast to MambaModel
    let mamba_model: &synapse_inference::models::MambaModel = unsafe {
        // The ssm_model is Box<dyn Model>, and we know it's MambaModel
        &*(mamba.as_ref() as *const dyn synapse_inference::models::Model as *const synapse_inference::models::MambaModel)
    };

    eprintln!("Quantizing to INT8...");
    let q_model = QuantizedMambaModel::from_f32(mamba_model);

    eprintln!("Exporting to {}...", output.display());
    let mut file = std::fs::File::create(&output).expect("Failed to create output file");

    // Magic
    file.write_all(b"SMI8").unwrap();

    // Config JSON
    let config_json = serde_json::json!({
        "d_model": q_model.config.d_model,
        "d_state": q_model.config.d_state,
        "d_conv": q_model.config.d_conv,
        "expand": q_model.config.expand,
        "dt_rank": q_model.config.dt_rank,
        "num_layers": q_model.config.num_layers,
        "vocab_size": q_model.config.vocab_size,
        "norm_eps": q_model.config.norm_eps,
    });
    let config_bytes = serde_json::to_vec(&config_json).unwrap();
    file.write_all(&(config_bytes.len() as u32).to_le_bytes()).unwrap();
    file.write_all(&config_bytes).unwrap();

    // Helper: write f32 slice
    let write_f32 = |f: &mut std::fs::File, data: &[f32]| {
        f.write_all(&(data.len() as u32).to_le_bytes()).unwrap();
        for &v in data {
            f.write_all(&v.to_le_bytes()).unwrap();
        }
    };

    // Helper: write QuantizedLinear
    let write_ql = |f: &mut std::fs::File, ql: &QuantizedLinear| {
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
    };

    // Embedding
    write_f32(&mut file, &q_model.embed_tokens);
    // Final norm
    write_f32(&mut file, &q_model.final_norm_weight);
    // LM head
    write_f32(&mut file, &q_model.lm_head_weight);

    // Blocks
    for block in &q_model.blocks {
        write_f32(&mut file, &block.norm_weight);
        write_ql(&mut file, &block.in_proj);
        write_f32(&mut file, &block.conv1d_weight);
        write_f32(&mut file, &block.conv1d_bias);
        write_f32(&mut file, &block.x_proj_weight);
        write_f32(&mut file, &block.dt_proj_weight);
        write_f32(&mut file, &block.dt_proj_bias);
        write_f32(&mut file, &block.a_log);
        write_f32(&mut file, &block.d_param);
        write_ql(&mut file, &block.out_proj);
    }

    let size_mb = std::fs::metadata(&output).unwrap().len() as f64 / 1e6;
    eprintln!("Done! {:.1}MB (vs 671MB f32 = {:.1}x compression)", size_mb, 671.0 / size_mb);
}
