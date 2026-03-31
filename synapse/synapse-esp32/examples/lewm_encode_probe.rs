use std::path::PathBuf;

use synapse_esp32::model::Esp32LeWM;
use synapse_inference::ops::patch_embed::patch_embed;
use synapse_inference::ops::pure_rust_ops::layernorm;
use synapse_inference::quantization::FullyQuantizedLeWM;

fn main() {
    let mut model_path: Option<PathBuf> = None;

    let args: Vec<String> = std::env::args().collect();
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--model" | "-m" => {
                i += 1;
                model_path = Some(PathBuf::from(&args[i]));
            }
            "--help" | "-h" => {
                eprintln!(
                    "Usage: cargo run -p synapse-esp32 --example lewm_encode_probe -- --model <path>"
                );
                std::process::exit(0);
            }
            _ => {}
        }
        i += 1;
    }

    let model_path = model_path.expect("--model is required");
    let data = std::fs::read(&model_path).expect("failed to read model");
    let model = Esp32LeWM::from_binary(&data).expect("failed to load LQ40 model");
    let full_model = FullyQuantizedLeWM::from_lq40_bytes(&data)
        .expect("lewm_encode_probe expects a full INT8+Q4 LQ40 blob");
    let config = model.config();

    let image_len = config.image_size * config.image_size * config.channels;
    let image = deterministic_vec(image_len, 7);
    let action = deterministic_vec(config.action_dim, 101);

    let (latent, encode_metrics) = model.encode(&image, config.image_size, config.image_size);
    let (next, predict_metrics) = model.predict_next(&latent, &action);
    let encoder_probe = run_encoder_probe(&full_model, &image, config.image_size, config.image_size);

    let payload = serde_json::json!({
        "encoder_probe": encoder_probe,
        "encode": {
            "latency_ms": encode_metrics.latency_ms,
            "output": summarize(&latent),
        },
        "encode_predict_next": {
            "latency_ms": predict_metrics.latency_ms,
            "output": summarize(&next),
        }
    });

    println!("{}", serde_json::to_string_pretty(&payload).unwrap());
}

fn run_encoder_probe(
    model: &FullyQuantizedLeWM,
    image: &[f32],
    height: usize,
    width: usize,
) -> serde_json::Value {
    let hidden = model.vit_config.hidden_size;
    let seq_len = model.vit_config.seq_len();
    let num_patches = model.vit_config.num_patches();

    let patches = patch_embed(
        image,
        height,
        width,
        model.vit_config.channels,
        model.vit_config.patch_size,
        &model.patch_proj,
        hidden,
    );

    let mut x = vec![0.0f32; seq_len * hidden];
    x[..hidden].copy_from_slice(&model.cls_token);
    x[hidden..hidden + num_patches * hidden].copy_from_slice(&patches);
    let patch0_embed = patches[..hidden].to_vec();

    for (dst, pos) in x.iter_mut().zip(model.pos_embed.iter()) {
        *dst += pos;
    }

    let patch0_with_pos = x[hidden..hidden + hidden].to_vec();

    let mut layer0_cls = vec![];
    for (idx, layer) in model.encoder_layers.iter().enumerate() {
        x = layer.forward(&x, seq_len);
        if idx == 0 {
            layer0_cls = x[..hidden].to_vec();
        }
    }

    let cls_norm = layernorm(&x[..hidden], &model.final_norm_weight, 1e-6, hidden);

    serde_json::json!({
        "patch0_embed": summarize(&patch0_embed),
        "patch0_with_pos": summarize(&patch0_with_pos),
        "layer0_cls": summarize(&layer0_cls),
        "cls_norm": summarize(&cls_norm),
    })
}

fn summarize(values: &[f32]) -> serde_json::Value {
    let sum: f32 = values.iter().copied().sum();
    let l2 = values.iter().map(|v| v * v).sum::<f32>().sqrt();
    let max_abs = values
        .iter()
        .copied()
        .map(f32::abs)
        .fold(0.0f32, f32::max);

    serde_json::json!({
        "len": values.len(),
        "first": values.iter().take(8).copied().collect::<Vec<_>>(),
        "sum": sum,
        "l2": l2,
        "max_abs": max_abs,
    })
}

fn deterministic_vec(len: usize, seed: u32) -> Vec<f32> {
    (0..len)
        .map(|idx| {
            let mixed = seed
                .wrapping_mul(1_664_525)
                .wrapping_add((idx as u32).wrapping_mul(1_013_904_223));
            let centered = (mixed % 2_001) as i32 - 1_000;
            centered as f32 / 1_000.0
        })
        .collect()
}
