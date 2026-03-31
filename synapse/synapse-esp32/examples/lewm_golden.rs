use std::path::PathBuf;

use synapse_esp32::model::Esp32LeWM;

fn main() {
    let mut model_path: Option<PathBuf> = None;
    let mut steps: usize = 5;

    let args: Vec<String> = std::env::args().collect();
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--model" | "-m" => {
                i += 1;
                model_path = Some(PathBuf::from(&args[i]));
            }
            "--steps" | "-s" => {
                i += 1;
                steps = args[i].parse().expect("--steps must be an integer");
            }
            "--help" | "-h" => {
                eprintln!("Usage: cargo run -p synapse-esp32 --example lewm_golden -- --model <path> [--steps <n>]");
                std::process::exit(0);
            }
            _ => {}
        }
        i += 1;
    }

    let model_path = model_path.expect("--model is required");
    let data = std::fs::read(&model_path).expect("failed to read model");
    let model = Esp32LeWM::from_binary(&data).expect("failed to load LQ40 model");

    let state = deterministic_vec(model.latent_dim(), 11);
    let actions: Vec<Vec<f32>> = (0..steps)
        .map(|step| deterministic_vec(model.action_dim(), 101 + step as u32 * 17))
        .collect();

    let (next, next_metrics) = model.predict_next(&state, &actions[0]);
    let (rollout, rollout_metrics) = model.rollout(&state, &actions);

    let payload = serde_json::json!({
        "model": model.model_info(),
        "source": model_path,
        "state": state,
        "actions": actions,
        "predict_next": {
            "output": next,
            "latency_ms": next_metrics.latency_ms,
        },
        "rollout": {
            "states": rollout,
            "latency_ms": rollout_metrics.latency_ms,
        },
    });

    println!("{}", serde_json::to_string_pretty(&payload).unwrap());
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
