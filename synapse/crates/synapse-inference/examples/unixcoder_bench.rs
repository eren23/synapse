//! Micro-benchmark: UniXcoder CLS latency + (optional) CDT encode/decode.
//!
//! Measures steady-state per-snippet latency on a single CPU thread — the
//! quantity the paper reports as `encoding latency` in
//! `sections/experiments.tex`.
//!
//! Usage:
//! ```bash
//! cargo run --release -p synapse-inference --example unixcoder_bench -- \
//!     --model-dir /tmp/cdt_demo/unixcoder \
//!     [--cdt-ckpt path/to/cdt.safetensors] \
//!     [--seq-len 64] [--repeats 20]
//! ```

use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::Instant;

use synapse_inference::models::text_encoder::{
    unixcoder_base, CodeDeltaTokConfig, CodeDeltaTokHead, Q4CodeDeltaTokHead,
    Q4RoBERTaEncoder, RoBERTaEncoder,
};
use synapse_inference::weight_loading::{load_safetensors, WeightMapper};

struct Args {
    model_dir: PathBuf,
    cdt_ckpt: Option<PathBuf>,
    seq_len: usize,
    repeats: usize,
    q4: bool,
}

fn parse() -> Args {
    let mut model_dir: Option<PathBuf> = None;
    let mut cdt_ckpt: Option<PathBuf> = None;
    let mut seq_len: usize = 64;
    let mut repeats: usize = 20;
    let mut q4 = false;
    let mut it = env::args().skip(1);
    while let Some(a) = it.next() {
        match a.as_str() {
            "--model-dir" => model_dir = it.next().map(PathBuf::from),
            "--cdt-ckpt"  => cdt_ckpt = it.next().map(PathBuf::from),
            "--seq-len"   => seq_len = it.next().and_then(|v| v.parse().ok()).unwrap_or(64),
            "--repeats"   => repeats = it.next().and_then(|v| v.parse().ok()).unwrap_or(20),
            "--q4"        => q4 = true,
            other => {
                eprintln!("Unknown arg {other}"); std::process::exit(2);
            }
        }
    }
    Args {
        model_dir: model_dir.expect("--model-dir required"),
        cdt_ckpt, seq_len, repeats, q4,
    }
}

fn first_safetensors(dir: &Path) -> Option<PathBuf> {
    let p = dir.join("model.safetensors");
    if p.exists() { return Some(p); }
    for entry in fs::read_dir(dir).ok()?.flatten() {
        let path = entry.path();
        if path.extension().and_then(|s| s.to_str()) == Some("safetensors") {
            return Some(path);
        }
    }
    None
}

fn main() {
    let args = parse();

    let weights_path = first_safetensors(&args.model_dir).expect("no safetensors");
    println!("Loading {}", weights_path.display());
    let weights = load_safetensors(&weights_path).expect("load weights");
    let mut model = RoBERTaEncoder::from_config(unixcoder_base());
    model.load_weights(weights, &WeightMapper::unixcoder())
        .expect("load_weights");
    let q4_model = if args.q4 {
        let q = Q4RoBERTaEncoder::from_fp32(&model);
        println!(
            "Q4 UniXcoder linear-storage: {:.1} MB",
            q.q4_memory_bytes() as f32 / 1e6,
        );
        Some(q)
    } else { None };

    // Deterministic fake input: `<s>` + some tokens + `</s>` + pad.
    let mut ids: Vec<i64> = vec![0, 729, 1103, 126, 183, 130, 442, 953, 317, 377, 483, 434, 513, 442, 317, 2];
    while ids.len() < args.seq_len { ids.push(1); }
    ids.truncate(args.seq_len);
    let mut mask: Vec<i64> = vec![1; 16];
    while mask.len() < args.seq_len { mask.push(0); }
    mask.truncate(args.seq_len);

    // Warmup.
    let _ = model.cls_feature(&ids, &mask);
    let _ = model.cls_feature(&ids, &mask);

    let start = Instant::now();
    for _ in 0..args.repeats {
        let _ = model.cls_feature(&ids, &mask);
    }
    let elapsed = start.elapsed();
    let per_run_ms = elapsed.as_secs_f64() * 1e3 / args.repeats as f64;
    println!(
        "UniXcoder CLS [seq_len={}, fp32]: {:.1} ms/snippet  ({:.1} snippets/s)",
        args.seq_len, per_run_ms, 1000.0 / per_run_ms,
    );

    if let Some(q) = q4_model.as_ref() {
        let _ = q.cls_feature(&ids, &mask);
        let _ = q.cls_feature(&ids, &mask);
        let start = Instant::now();
        for _ in 0..args.repeats {
            let _ = q.cls_feature(&ids, &mask);
        }
        let elapsed = start.elapsed();
        let per_run_ms = elapsed.as_secs_f64() * 1e3 / args.repeats as f64;
        println!(
            "UniXcoder CLS [seq_len={},  Q4 ]: {:.1} ms/snippet  ({:.1} snippets/s)",
            args.seq_len, per_run_ms, 1000.0 / per_run_ms,
        );
    }

    if let Some(ckpt_path) = args.cdt_ckpt {
        println!("Loading CDT {}", ckpt_path.display());
        let ck = load_safetensors(&ckpt_path).expect("load cdt");
        let mut head = CodeDeltaTokHead::from_config(CodeDeltaTokConfig::paper_default());
        head.load_weights(ck, &WeightMapper::code_deltatok()).expect("cdt load");
        let q4_head = if args.q4 { Some(Q4CodeDeltaTokHead::from_fp32(&head)) } else { None };

        let h_b = model.cls_feature(&ids, &mask);
        let h_a = model.cls_feature(&ids, &mask);

        // fp32 head.
        let _ = head.encode(&h_b, &h_a);
        let _ = head.decode(&head.encode(&h_b, &h_a), &h_b);
        let start = Instant::now();
        for _ in 0..args.repeats {
            let d = head.encode(&h_b, &h_a);
            let _ = head.decode(&d, &h_b);
        }
        let elapsed = start.elapsed();
        let per_run_ms = elapsed.as_secs_f64() * 1e3 / args.repeats as f64;
        println!(
            "CDT encode+decode [fp32]:        {:.2} ms/pair    ({:.1} pairs/s)",
            per_run_ms, 1000.0 / per_run_ms,
        );

        if let Some(q) = q4_head.as_ref() {
            let _ = q.encode(&h_b, &h_a);
            let _ = q.decode(&q.encode(&h_b, &h_a), &h_b);
            let start = Instant::now();
            for _ in 0..args.repeats {
                let d = q.encode(&h_b, &h_a);
                let _ = q.decode(&d, &h_b);
            }
            let elapsed = start.elapsed();
            let per_run_ms = elapsed.as_secs_f64() * 1e3 / args.repeats as f64;
            println!(
                "CDT encode+decode  [Q4 ]:        {:.2} ms/pair    ({:.1} pairs/s)",
                per_run_ms, 1000.0 / per_run_ms,
            );

            if let Some(qe) = q4_model.as_ref() {
                let h_b_q = qe.cls_feature(&ids, &mask);
                let h_a_q = qe.cls_feature(&ids, &mask);
                let d_q = q.encode(&h_b_q, &h_a_q);
                let r_q = q.decode(&d_q, &h_b_q);
                // Compare end-to-end Q4+Q4 vs fp32+fp32.
                let d_f = head.encode(&h_b, &h_a);
                let r_f = head.decode(&d_f, &h_b);
                let cos = {
                    let dot: f32 = r_q.iter().zip(&r_f).map(|(x, y)| x * y).sum();
                    let na: f32 = r_q.iter().map(|x| x * x).sum::<f32>().sqrt();
                    let nb: f32 = r_f.iter().map(|y| y * y).sum::<f32>().sqrt();
                    dot / (na * nb + 1e-30)
                };
                let dcos = {
                    let dot: f32 = d_q.iter().zip(&d_f).map(|(x, y)| x * y).sum();
                    let na: f32 = d_q.iter().map(|x| x * x).sum::<f32>().sqrt();
                    let nb: f32 = d_f.iter().map(|y| y * y).sum::<f32>().sqrt();
                    dot / (na * nb + 1e-30)
                };
                println!(
                    "End-to-end Q4 vs fp32 drift:    delta cos = {dcos:.4}, recon cos = {cos:.4}",
                );
            }
        }
    }
}
