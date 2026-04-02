//! Benchmark: sequential vs fused rollout latency.
use synapse_esp32::model::Esp32LeWM;

fn det(len: usize, seed: u32) -> Vec<f32> {
    (0..len)
        .map(|i| {
            let m = seed.wrapping_mul(1_664_525).wrapping_add((i as u32).wrapping_mul(1_013_904_223));
            let centered = (m % 2_001).wrapping_sub(1_000) as i32;
            centered as f32 / 1_000.0
        })
        .collect()
}

fn cosim(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if na > 0.0 && nb > 0.0 {
        dot / (na * nb)
    } else {
        0.0
    }
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut model_path: Option<std::path::PathBuf> = None;
    let mut steps = 3usize;
    let mut esp32_encode_ms = 922.0;

    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "-m" => {
                i += 1;
                model_path = Some(std::path::PathBuf::from(&args[i]));
            }
            "-s" => {
                i += 1;
                steps = args[i].parse().unwrap();
            }
            "--esp32-encode" => {
                i += 1;
                esp32_encode_ms = args[i].parse().unwrap();
            }
            _ => {}
        }
        i += 1;
    }

    let model: Esp32LeWM = match model_path {
        Some(ref p) => {
            println!("Model: {:?}", p);
            let data = std::fs::read(p).expect("read failed");
            Esp32LeWM::from_binary(&data).expect("parse failed")
        }
        None => {
            println!("Using seeded benchmark model (--esp32-encode for absolute estimates)");
            Esp32LeWM::new_benchmark()
        }
    };

    let ldim = model.latent_dim();
    let adim = model.action_dim();
    let state = det(ldim, 11);
    let actions: Vec<Vec<f32>> = (0..steps)
        .map(|s| det(adim, 101u32.wrapping_mul(17).wrapping_add(s as u32)))
        .collect();

    let (seq_traj, sm) = model.rollout(&state, &actions);
    let (fused_traj, fm) = model.rollout_fused(&state, &actions);

    println!(
        "Sequential {:.3} ms  Fused {:.3} ms  Speedup {:.1}x",
        sm.latency_ms,
        fm.latency_ms,
        sm.latency_ms / fm.latency_ms.max(0.001)
    );
    let cos = cosim(&seq_traj[0], &fused_traj[0]);
    let finite = seq_traj.iter().chain(&fused_traj).all(|v| v.iter().all(|x| x.is_finite()));
    println!("cos_sim(step0) {:.6}  finite: {}", cos, finite);
}
