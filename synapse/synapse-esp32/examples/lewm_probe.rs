use std::path::PathBuf;

use synapse_inference::ops::pure_rust_ops::{gelu, layernorm, matmul_t};
use synapse_inference::quantization::QuantizedQ4LeWM;

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
                eprintln!("Usage: cargo run -p synapse-esp32 --example lewm_probe -- --model <path>");
                std::process::exit(0);
            }
            _ => {}
        }
        i += 1;
    }

    let model_path = model_path.expect("--model is required");
    let data = std::fs::read(&model_path).expect("failed to read model");
    let model = QuantizedQ4LeWM::from_lq40_bytes(&data).expect("failed to load LQ40 model");

    let state = deterministic_vec(model.config.latent_dim, 11);
    let action = deterministic_vec(model.config.action_dim, 101);

    let action_embed = encode_action(&model, &action);
    let conditioning = if !model.input_proj_weight.is_empty() {
        apply_cond_proj(&model, &action_embed)
    } else {
        action_embed.clone()
    };

    let mut layer0_adaln = model.predictor_layers[0].adaln_linear.forward(&conditioning, 1);
    add_bias_inplace(
        &mut layer0_adaln,
        &model.predictor_layers[0].adaln_bias,
        1,
        6 * model.config.predictor_hidden,
    );

    let (layer0_seq, final_target) = forward_probe(&model, &state, &action_embed, &conditioning);
    let pred_proj_probe = projection_head_probe(&model, &final_target);
    let predict_next = model.pred_proj.forward(&final_target);

    let payload = serde_json::json!({
        "probe": {
            "action_embed": summarize(&action_embed),
            "conditioning": summarize(&conditioning),
            "layer0_adaln": summarize(&layer0_adaln),
            "layer0_seq": summarize(&layer0_seq),
            "final_target": summarize(&final_target),
            "pred_proj": pred_proj_probe,
            "predict_next": summarize(&predict_next),
        }
    });

    println!("{}", serde_json::to_string_pretty(&payload).unwrap());
}

fn encode_action(model: &QuantizedQ4LeWM, action: &[f32]) -> Vec<f32> {
    let act_dim = model.config.action_dim;
    let hidden = model.config.latent_dim;

    let mut conv_out = if !model.action_conv_weight.is_empty() {
        matmul_t(action, &model.action_conv_weight, 1, act_dim, act_dim)
    } else {
        action.to_vec()
    };
    add_bias_inplace(&mut conv_out, &model.action_conv_bias, 1, act_dim);

    let inter = if !model.action_mlp1_weight.is_empty() {
        model.action_mlp1_weight.len() / act_dim
    } else {
        hidden * 4
    };

    let mut h1 = if !model.action_mlp1_weight.is_empty() {
        matmul_t(&conv_out, &model.action_mlp1_weight, 1, act_dim, inter)
    } else {
        vec![0.0f32; inter]
    };
    add_bias_inplace(&mut h1, &model.action_mlp1_bias, 1, inter);
    for value in &mut h1 {
        *value = gelu(*value);
    }

    let mut out = if !model.action_mlp2_weight.is_empty() {
        matmul_t(&h1, &model.action_mlp2_weight, 1, inter, hidden)
    } else {
        vec![0.0f32; hidden]
    };
    add_bias_inplace(&mut out, &model.action_mlp2_bias, 1, hidden);
    out
}

fn apply_cond_proj(model: &QuantizedQ4LeWM, cond: &[f32]) -> Vec<f32> {
    let latent = model.config.latent_dim;
    let hidden = model.config.predictor_hidden;
    let mut out = matmul_t(cond, &model.cond_proj_weight, 1, latent, hidden);
    add_bias_inplace(&mut out, &model.cond_proj_bias, 1, hidden);
    out
}

fn apply_input_proj(model: &QuantizedQ4LeWM, seq: &[f32]) -> Vec<f32> {
    let latent = model.config.latent_dim;
    let hidden = model.config.predictor_hidden;
    let seq_len = 3usize;
    let mut out = vec![0.0f32; seq_len * hidden];
    for token in 0..seq_len {
        let projected = matmul_t(&seq[token * latent..(token + 1) * latent], &model.input_proj_weight, 1, latent, hidden);
        out[token * hidden..(token + 1) * hidden].copy_from_slice(&projected);
    }
    add_bias_inplace(&mut out, &model.input_proj_bias, seq_len, hidden);
    out
}

fn forward_probe(
    model: &QuantizedQ4LeWM,
    state: &[f32],
    action_embed: &[f32],
    conditioning: &[f32],
) -> (Vec<f32>, Vec<f32>) {
    let hidden = model.config.predictor_hidden;
    let latent = model.config.latent_dim;
    let seq_len = 3usize;
    let has_proj = !model.input_proj_weight.is_empty();
    let seq_dim = if has_proj { latent } else { hidden };

    let mut seq = vec![0.0f32; seq_len * seq_dim];
    seq[..seq_dim].copy_from_slice(state);
    seq[seq_dim..2 * seq_dim].copy_from_slice(action_embed);
    let pos_len = model.predictor_pos_embed.len().min(seq.len());
    for i in 0..pos_len {
        seq[i] += model.predictor_pos_embed[i];
    }

    let mut seq = if has_proj {
        apply_input_proj(model, &seq)
    } else {
        seq
    };

    let mut layer0_seq = Vec::new();
    for (idx, layer) in model.predictor_layers.iter().enumerate() {
        seq = layer.forward(
            &seq,
            conditioning,
            seq_len,
            hidden,
            model.config.predictor_heads,
            model.config.predictor_inner_dim,
            model.config.predictor_inter,
        );
        if idx == 0 {
            layer0_seq = seq.clone();
        }
    }

    let mut normed = layernorm(&seq, &model.predictor_norm_weight, 1e-6, hidden);
    add_bias_inplace(&mut normed, &model.predictor_norm_bias, seq_len, hidden);
    let final_target = normed[2 * hidden..3 * hidden].to_vec();
    (layer0_seq, final_target)
}

fn projection_head_probe(
    model: &QuantizedQ4LeWM,
    input: &[f32],
) -> serde_json::Value {
    let mut current = input.to_vec();
    let mut layers = Vec::new();

    for (idx, (weight, bias)) in model.pred_proj.layers.iter().enumerate() {
        let in_dim = current.len();
        let out_dim = if in_dim == 0 { 0 } else { weight.len() / in_dim };
        let mut pre_gelu = matmul_t(&current, weight, 1, in_dim, out_dim);
        add_bias_inplace(&mut pre_gelu, bias, 1, out_dim);
        let mut entry = serde_json::json!({
            "layer": idx,
            "weight_len": weight.len(),
            "bias_len": bias.len(),
            "in_dim": in_dim,
            "out_dim": out_dim,
            "pre_gelu": summarize(&pre_gelu),
        });

        if idx + 1 < model.pred_proj.layers.len() {
            for value in &mut pre_gelu {
                *value = gelu(*value);
            }
            entry["post_gelu"] = summarize(&pre_gelu);
        }

        current = pre_gelu;
        layers.push(entry);
    }

    serde_json::Value::Array(layers)
}

fn add_bias_inplace(values: &mut [f32], bias: &[f32], rows: usize, cols: usize) {
    if bias.is_empty() {
        return;
    }
    for row in 0..rows {
        for col in 0..cols {
            values[row * cols + col] += bias[col];
        }
    }
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
