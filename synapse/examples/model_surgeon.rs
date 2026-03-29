//! Model Surgeon: analyze, prune, and compress SSM models.
//!
//! Usage:
//!   # Analyze layer sensitivity
//!   cargo run --example model_surgeon --release -- --model-dir models/mamba-130m --analyze
//!
//!   # Prune layers + Wanda + quantize to Q4
//!   cargo run --example model_surgeon --release -- --model-dir models/mamba-130m \
//!     --prune layers:2,wanda:0.3 --quantize q4 --output pruned.bin
//!
//!   # Channel pruning (reduce d_inner)
//!   cargo run --example model_surgeon --release -- --model-dir models/mamba-130m \
//!     --prune channels:1024 --analyze

use std::path::PathBuf;

use synapse_inference::engine::InferenceEngine;
use synapse_inference::model::traits::Model;
use synapse_inference::pruning::{
    SensitivityAnalyzer, LayerRemover, WandaPruner, MambaChannelPruner,
    SurgeonPipeline, PruningStrategy,
};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let mut model_dir: Option<PathBuf> = None;
    let mut do_analyze = false;
    let mut prune_spec: Option<String> = None;
    let mut quantize_format: Option<String> = None;
    let mut output_path: Option<PathBuf> = None;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--model-dir" => { i += 1; model_dir = Some(PathBuf::from(&args[i])); }
            "--analyze" => { do_analyze = true; }
            "--prune" => { i += 1; prune_spec = Some(args[i].clone()); }
            "--quantize" => { i += 1; quantize_format = Some(args[i].clone()); }
            "--output" | "-o" => { i += 1; output_path = Some(PathBuf::from(&args[i])); }
            "--help" | "-h" => {
                print_usage();
                return;
            }
            _ => {}
        }
        i += 1;
    }

    let model_dir = model_dir.unwrap_or_else(|| {
        eprintln!("Error: --model-dir required");
        print_usage();
        std::process::exit(1);
    });

    eprintln!("Loading model from {}...", model_dir.display());
    let engine = InferenceEngine::from_pretrained(&model_dir).expect("Failed to load model");
    let mamba_model = engine.ssm_model.as_ref().expect("Not an SSM model");

    // Downcast to MambaModel
    let mamba: &synapse_inference::ssm::MambaModel = unsafe {
        &*(mamba_model.as_ref() as *const dyn Model as *const synapse_inference::ssm::MambaModel)
    };

    eprintln!("Model: {} layers, d_model={}, d_inner={}, vocab={}",
        mamba.config.num_layers, mamba.config.d_model,
        mamba.config.d_inner(), mamba.config.vocab_size);

    // Calibration tokens (simple prompt for sensitivity analysis)
    let calibration = vec![
        464, 530, 310, 257, 1598, 640, 2005, 262,  // "The quick brown fox jumps over the"
        8564, 3290, 13, 383, 886, 1517, 318,        // "lazy dog. The end result is"
    ];

    if do_analyze {
        eprintln!("\n=== Sensitivity Analysis ===");
        let analyzer = SensitivityAnalyzer::new(0.99);
        let importances = analyzer.analyze_mamba(
            &mamba.config,
            &mamba.embed_tokens,
            &mamba.blocks,
            &mamba.final_norm_weight,
            &mamba.lm_head_weight,
            &calibration,
        );

        println!("{:<6} {:<12} {:<12} {}", "Layer", "Cos Sim", "KL Div", "Removable?");
        println!("{}", "-".repeat(50));
        for imp in &importances {
            println!("{:<6} {:<12.6} {:<12.6} {}",
                imp.layer_idx,
                imp.cosine_similarity,
                imp.kl_divergence,
                if imp.removable { "YES" } else { "no" },
            );
        }

        let removable_count = importances.iter().filter(|i| i.removable).count();
        eprintln!("\n{} of {} layers can be safely removed (cos > 0.99)", removable_count, mamba.config.num_layers);
    }

    if let Some(spec) = prune_spec {
        let strategies = parse_prune_spec(&spec);
        if strategies.is_empty() {
            eprintln!("Error: invalid --prune spec: {}", spec);
            return;
        }

        eprintln!("\n=== Surgery Pipeline ===");
        let pipeline = SurgeonPipeline::new(strategies, 0.1); // max 10% quality loss
        let (pruned_model, report) = pipeline.run_mamba(
            clone_mamba_model(mamba),
            &calibration,
        );

        for step in &report.steps {
            eprintln!("  [{}] {} (cos={:.6})",
                step.strategy, step.details, step.similarity_after);
        }
        eprintln!("\nOriginal params: {}", report.original_params);
        eprintln!("Pruned params:   {}", report.pruned_params);
        eprintln!("Compression:     {:.2}x", report.compression_ratio);
        eprintln!("Final cos sim:   {:.6}", report.final_similarity);

        if let Some(format) = &quantize_format {
            match format.as_str() {
                "q4" => {
                    let q4 = synapse_inference::quantization::Q4MambaModel::from_f32(&pruned_model);
                    eprintln!("Q4 model size:   {} bytes ({:.1} MB)",
                        q4.model_size_bytes(),
                        q4.model_size_bytes() as f64 / 1_048_576.0);
                }
                "int8" => {
                    let int8 = synapse_inference::quantization::QuantizedMambaModel::from_f32(&pruned_model);
                    eprintln!("INT8 savings:    {} bytes", int8.memory_savings());
                }
                _ => eprintln!("Unknown quantize format: {}", format),
            }
        }
    }
}

fn parse_prune_spec(spec: &str) -> Vec<PruningStrategy> {
    let mut strategies = Vec::new();
    for part in spec.split(',') {
        let kv: Vec<&str> = part.split(':').collect();
        match kv.get(0).copied() {
            Some("layers") => {
                let n: usize = kv.get(1).and_then(|s| s.parse().ok()).unwrap_or(2);
                strategies.push(PruningStrategy::LayerRemoval { max_layers: n });
            }
            Some("wanda") => {
                let s: f32 = kv.get(1).and_then(|s| s.parse().ok()).unwrap_or(0.3);
                strategies.push(PruningStrategy::WandaPruning { sparsity: s });
            }
            Some("channels") => {
                let d: usize = kv.get(1).and_then(|s| s.parse().ok()).unwrap_or(1024);
                strategies.push(PruningStrategy::MambaChannelPruning { target_d_inner: d });
            }
            _ => {}
        }
    }
    strategies
}

fn clone_mamba_model(m: &synapse_inference::ssm::MambaModel) -> synapse_inference::ssm::MambaModel {
    use synapse_inference::ssm::mamba_block::MambaBlock;

    let blocks: Vec<MambaBlock> = m.blocks.iter().map(|b| {
        MambaBlock {
            d_model: b.d_model,
            d_inner: b.d_inner,
            d_state: b.d_state,
            d_conv: b.d_conv,
            dt_rank: b.dt_rank,
            norm_weight: b.norm_weight.clone(),
            norm_eps: b.norm_eps,
            in_proj_weight: b.in_proj_weight.clone(),
            in_proj_bias: b.in_proj_bias.clone(),
            conv1d_weight: b.conv1d_weight.clone(),
            conv1d_bias: b.conv1d_bias.clone(),
            x_proj_weight: b.x_proj_weight.clone(),
            dt_proj_weight: b.dt_proj_weight.clone(),
            dt_proj_bias: b.dt_proj_bias.clone(),
            a_log: b.a_log.clone(),
            d_param: b.d_param.clone(),
            out_proj_weight: b.out_proj_weight.clone(),
            out_proj_bias: b.out_proj_bias.clone(),
        }
    }).collect();

    synapse_inference::ssm::MambaModel::new(
        m.config.clone(),
        m.embed_tokens.clone(),
        blocks,
        m.final_norm_weight.clone(),
        m.lm_head_weight.clone(),
    )
}

fn print_usage() {
    eprintln!("Model Surgeon — analyze, prune, and compress SSM models");
    eprintln!();
    eprintln!("Usage:");
    eprintln!("  model_surgeon --model-dir <path> [options]");
    eprintln!();
    eprintln!("Options:");
    eprintln!("  --analyze            Analyze layer sensitivity (no modification)");
    eprintln!("  --prune <spec>       Pruning specification, comma-separated:");
    eprintln!("                         layers:N     Remove up to N redundant layers");
    eprintln!("                         wanda:F      Wanda weight pruning at F sparsity");
    eprintln!("                         channels:D   Reduce d_inner to D channels");
    eprintln!("  --quantize <fmt>     Post-pruning quantization: q4 or int8");
    eprintln!("  --output <path>      Export pruned model (not yet implemented)");
    eprintln!();
    eprintln!("Examples:");
    eprintln!("  model_surgeon --model-dir models/mamba-130m --analyze");
    eprintln!("  model_surgeon --model-dir models/mamba-130m --prune layers:2,wanda:0.3 --quantize q4");
}
