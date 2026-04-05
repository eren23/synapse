//! Export a browser-ready corpus JSON: walk a dir, tokenize + encode each .py
//! file, save as {files: [{path, preview, embedding: [128 f32]}]}.
//!
//! The browser demo loads this JSON once and runs cosine search against it.
//!
//! Usage:
//!   cargo run --release --example export_browser_corpus -- \
//!       models/code_wm/g1b.safetensors \
//!       configs/code_wm_g1b.json \
//!       scripts \
//!       web/code_wm_demo/corpus.json

use std::env;
use std::fs;
use std::path::{Path, PathBuf};

use synapse_code_tokenizer::tokenize;
use synapse_inference::models::vision::{CodeWorldModel, CodeWorldModelConfig};
use synapse_inference::weight_loading::load_safetensors;

fn l2_norm(v: &[f32]) -> Vec<f32> {
    let n = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if n < 1e-30 { v.to_vec() } else { v.iter().map(|x| x / n).collect() }
}

fn collect_py_files(root: &Path, max_files: usize) -> Vec<PathBuf> {
    let mut out = Vec::new();
    let mut stack = vec![root.to_path_buf()];
    while let Some(dir) = stack.pop() {
        if out.len() >= max_files { break; }
        if let Ok(entries) = fs::read_dir(&dir) {
            let mut entries: Vec<_> = entries.filter_map(|e| e.ok()).collect();
            entries.sort_by_key(|e| e.path());
            for entry in entries {
                let p = entry.path();
                if p.is_dir() { stack.push(p); }
                else if p.extension().map(|e| e == "py").unwrap_or(false) {
                    out.push(p);
                    if out.len() >= max_files { break; }
                }
            }
        }
    }
    out.sort();
    out
}

fn main() {
    let mut args = env::args().skip(1);
    let weights = args.next().expect("arg 1: weights .safetensors");
    let config = args.next().expect("arg 2: config .json");
    let dir = args.next().expect("arg 3: corpus directory");
    let out_path = args.next().expect("arg 4: output .json");
    let max_len: usize = 512;
    let max_files = 64;
    let preview_chars = 200;

    let cfg = CodeWorldModelConfig::from_json(Path::new(&config)).unwrap();
    let tensors = load_safetensors(Path::new(&weights)).unwrap();
    let mut model = CodeWorldModel::from_config(&cfg);
    model.load_weights(tensors).unwrap();

    let files = collect_py_files(Path::new(&dir), max_files);
    let root = Path::new(&dir).canonicalize().unwrap();
    println!("Encoding {} files from {dir}", files.len());

    let mut entries: Vec<String> = Vec::new();
    for path in &files {
        let source = match fs::read_to_string(path) { Ok(s) => s, Err(_) => continue };
        let toks_u16 = tokenize(&source, max_len);
        let toks_i64: Vec<i64> = toks_u16.iter().map(|&t| t as i64).collect();
        let z = model.encode(&toks_i64);
        let z = l2_norm(&z);
        let rel = path.strip_prefix(&root).unwrap_or(path).display().to_string();

        // Preview: first N non-blank chars, escape for JSON
        let preview: String = source.chars().take(preview_chars).collect();
        let preview_escaped = preview
            .replace('\\', "\\\\").replace('"', "\\\"")
            .replace('\n', "\\n").replace('\r', "\\r").replace('\t', "\\t");

        let emb_json: Vec<String> = z.iter().map(|v| format!("{:.6}", v)).collect();
        entries.push(format!(
            "    {{\"path\":\"{}\",\"preview\":\"{}\",\"embedding\":[{}]}}",
            rel.replace('\\', "\\\\").replace('"', "\\\""),
            preview_escaped,
            emb_json.join(",")
        ));
    }

    let json = format!(
        "{{\n  \"model_dim\": {},\n  \"count\": {},\n  \"files\": [\n{}\n  ]\n}}",
        cfg.model_dim,
        entries.len(),
        entries.join(",\n")
    );

    if let Some(parent) = Path::new(&out_path).parent() { fs::create_dir_all(parent).ok(); }
    fs::write(&out_path, json).unwrap();
    println!("Wrote {} ({} entries)", out_path, entries.len());
}
