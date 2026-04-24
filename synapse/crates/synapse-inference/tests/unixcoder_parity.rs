//! HuggingFace-parity test for the UniXcoder (RoBERTa) encoder.
//!
//! Loads `microsoft/unixcoder-base` from the HF cache (or from a directory
//! pointed at by the `UNIXCODER_BASE_DIR` env var) and runs the same
//! 16 reference code snippets that
//! `scripts/export_unixcoder_reference.py export` pushed into
//! `tests/fixtures/unixcoder_ref.safetensors`. Required parity: max-abs
//! difference < 1e-3 and CLS cosine >= 0.9999 on every snippet.
//!
//! Skips gracefully if neither the HF cache nor the override dir contains
//! UniXcoder so CI boxes without HuggingFace credentials still build.

use std::path::PathBuf;

use synapse_inference::models::text_encoder::{unixcoder_base, RoBERTaEncoder};
use synapse_inference::weight_loading::{load_safetensors, WeightMapper};

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .to_path_buf()
}

fn find_unixcoder_safetensors() -> Option<PathBuf> {
    if let Ok(dir) = std::env::var("UNIXCODER_BASE_DIR") {
        let p = PathBuf::from(dir).join("model.safetensors");
        if p.exists() { return Some(p); }
    }
    // HF hub layout: ~/.cache/huggingface/hub/models--microsoft--unixcoder-base/snapshots/*/model.safetensors
    let home = std::env::var("HOME").ok()?;
    let base = PathBuf::from(home)
        .join(".cache/huggingface/hub/models--microsoft--unixcoder-base/snapshots");
    if !base.exists() { return None; }
    for entry in std::fs::read_dir(&base).ok()?.flatten() {
        let p = entry.path().join("model.safetensors");
        if p.exists() { return Some(p); }
    }
    None
}

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let mut dot = 0.0f32;
    let mut na = 0.0f32;
    let mut nb = 0.0f32;
    for (x, y) in a.iter().zip(b) {
        dot += x * y;
        na += x * x;
        nb += y * y;
    }
    dot / (na.sqrt() * nb.sqrt() + 1e-30)
}

fn max_abs(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0f32, f32::max)
}

#[test]
fn unixcoder_cls_matches_huggingface() {
    let fixture_path = repo_root()
        .join("crates/synapse-inference/tests/fixtures/unixcoder_ref.safetensors");
    if !fixture_path.exists() {
        eprintln!("SKIP: fixture {} missing. Run \
                   `python scripts/export_unixcoder_reference.py export \
                   --out crates/synapse-inference/tests/fixtures/unixcoder_ref.safetensors`.",
                  fixture_path.display());
        return;
    }
    let weights_path = match find_unixcoder_safetensors() {
        Some(p) => p,
        None => {
            eprintln!("SKIP: microsoft/unixcoder-base not in HF cache. \
                       Set UNIXCODER_BASE_DIR or run \
                       `huggingface-cli download microsoft/unixcoder-base`.");
            return;
        }
    };

    let fixture = load_safetensors(&fixture_path).expect("load fixture");
    let weights = load_safetensors(&weights_path).expect("load UniXcoder safetensors");

    let mut model = RoBERTaEncoder::from_config(unixcoder_base());
    let result = model.load_weights(weights, &WeightMapper::unixcoder())
        .expect("load_weights");
    assert!(
        result.missing.is_empty(),
        "Missing weight targets: {:?}", result.missing,
    );
    // The pooler tensors and the `embeddings.position_ids` registered buffer
    // are expected to land in `unexpected`; everything else is a bug.
    for key in &result.unexpected {
        let ok = matches!(
            key.as_str(),
            "pooler.dense.weight" | "pooler.dense.bias"
                | "roberta.pooler.dense.weight"
                | "roberta.pooler.dense.bias"
                | "embeddings.position_ids"
                | "roberta.embeddings.position_ids",
        );
        assert!(ok, "Unexpected source key: {key}");
    }

    // Fixture shapes: [B, S] for ids/mask, [B, S, H] for hidden states,
    // [B, H] for cls_feature.
    let input_ids = fixture.get("input_ids").expect("input_ids");
    let attn_mask = fixture.get("attention_mask").expect("attention_mask");
    let cls_ref = fixture.get("cls_feature").expect("cls_feature");
    let last_ref = fixture.get("last_hidden_state").expect("last_hidden_state");

    assert_eq!(input_ids.shape.len(), 2);
    let batch = input_ids.shape[0];
    let seq_len = input_ids.shape[1];
    let hidden = model.config.hidden_size;
    assert_eq!(cls_ref.shape, vec![batch, hidden]);
    assert_eq!(last_ref.shape, vec![batch, seq_len, hidden]);

    // Fixture ids/mask are stored as i64 little-endian. Our safetensors
    // loader converts everything to f32 (see weight_loading::converter),
    // so the 0/1/2/... mask values ended up as f32. Round and cast.
    let ids: Vec<i64> = input_ids.data.iter().map(|&v| v as i64).collect();
    let mask: Vec<i64> = attn_mask.data.iter().map(|&v| v as i64).collect();

    let mut worst_cos = 1.0f32;
    let mut worst_abs = 0.0f32;

    for b in 0..batch {
        let id_row = &ids[b * seq_len..(b + 1) * seq_len];
        let mask_row = &mask[b * seq_len..(b + 1) * seq_len];

        let last = model.forward(id_row, mask_row);
        assert_eq!(last.len(), seq_len * hidden);

        let cls_ours = &last[..hidden];
        let cls_theirs = &cls_ref.data[b * hidden..(b + 1) * hidden];
        let c = cosine(cls_ours, cls_theirs);
        let m = max_abs(cls_ours, cls_theirs);
        worst_cos = worst_cos.min(c);
        worst_abs = worst_abs.max(m);

        // Also check the full hidden state at real (non-pad) positions.
        let real_len: usize = mask_row.iter().map(|&x| x as usize).sum();
        for t in 0..real_len {
            let ours = &last[t * hidden..(t + 1) * hidden];
            let theirs =
                &last_ref.data[((b * seq_len) + t) * hidden..((b * seq_len) + t + 1) * hidden];
            let cc = cosine(ours, theirs);
            let mm = max_abs(ours, theirs);
            assert!(
                cc > 0.999 && mm < 2e-3,
                "snippet {b} position {t} drifted: cos={cc:.5}, max_abs={mm:.4}",
            );
        }
    }

    println!(
        "CLS parity: worst cosine = {worst_cos:.6}, worst max_abs = {worst_abs:.5}",
    );
    assert!(worst_cos >= 0.9999, "worst CLS cosine {worst_cos} < 0.9999");
    assert!(worst_abs < 1e-3,    "worst CLS max_abs {worst_abs} >= 1e-3");
}
