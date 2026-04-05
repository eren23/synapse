# Code WM — Browser Demo

Semantic code search in the browser. Paste Python code, find structurally-similar
files in a pre-indexed corpus. **Runs entirely client-side** — the AST tokenizer,
transformer encoder, and cosine search all execute in WASM.

## Architecture

```
[User's code in textarea]
   ↓
[tokenize_python()  in WASM]  ← native AST tokenizer (rustpython-parser + FNV-1a)
   ↓  (u16 tokens)
[WasmCodeWM.encode()  in WASM]  ← 6-loop weight-shared transformer
   ↓  (128-d f32 latent)
[L2 normalize + cosine vs corpus  in JS]
   ↓
[top-5 results rendered]
```

## Build + serve

```bash
# 1. Build WASM bundle (~2.6 MB after wasm-opt)
cd synapse-wasm
wasm-pack build --target web --release

# 2. Export corpus (runs Rust encoder offline on scripts/*.py)
cd ..
cargo run --release --example export_browser_corpus -- \
    models/code_wm/g1b.safetensors \
    configs/code_wm_g1b.json \
    scripts \
    web/code_wm_demo/corpus.json

# 3. Copy weights + config into the demo dir (gitignored)
cp models/code_wm/g1b.safetensors web/code_wm_demo/g1b.safetensors
cp configs/code_wm_g1b.json web/code_wm_demo/g1b.config.json

# 4. Serve
cd web
python3 -m http.server 8080
# → browse http://localhost:8080/code_wm_demo/
```

## Expected performance (M-series, release build)

| Stage | Latency |
|---|---|
| Tokenize (512 tokens) | ~0.5-2ms |
| Encode (S=512) | ~80-150ms (pure-Rust, no SIMD in WASM) |
| Cosine over 20 files | <1ms |
| **Total** | **~100-200ms per query** |

The WASM encoder is ~3-5x slower than native Zig SIMD — acceptable for
interactive search.

## Files

- `index.html` — demo page, loads WASM + corpus + weights
- `corpus.json` — 20 pre-encoded .py files with 128-d embeddings (29 KB)
- `g1b.safetensors` — model weights (2.9 MB, gitignored)
- `g1b.config.json` — model architecture config

## Why this matters

- **No Python at runtime**: the AST tokenizer runs in WASM via rustpython-parser
- **No server required**: everything is static — hosts on GitHub Pages, Cloudflare, etc.
- **Zero drift**: Rust tokenizer produces byte-identical tokens to the Python reference
  used during training (see `crates/synapse-code-tokenizer/tests/cross_validation.rs`)
- **Small footprint**: 2.6 MB WASM + 3 MB weights = 5.6 MB total, gzipped ~2 MB

For offline corpora, swap `corpus.json` with any precomputed embedding index.
