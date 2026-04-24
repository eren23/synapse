# CodeDeltaTok · WASM demo

Browser-side UniXcoder backbone + CodeDeltaTok head. 768-dim delta token per
(before, after) code pair, computed on-device in pure-Rust WASM.

## Run locally

From the `synapse/` repo root:

```bash
# 1. Build / refresh the WASM bundle
(cd synapse-wasm && wasm-pack build --release --target web -- --offline)
cp synapse-wasm/pkg/* web/synapse-wasm-pkg/

# 2. Serve. Any static server works; this one ships with Python:
python3 -m http.server 8000 --directory web
# → open http://localhost:8000/unixcoder-delta/
```

## Inputs

The demo does not bundle model weights (UniXcoder alone is 480 MB fp32). Point
the two file pickers at:

- `model.safetensors` from
  `huggingface-cli download microsoft/unixcoder-base`
- A converted CDT checkpoint, produced with
  `python scripts/export_unixcoder_reference.py convert-cdt
  --ckpt code_deltatok_final.pt --ref <fixture> --out cdt.safetensors`

## Download-size modes

| Mode | UniXcoder | CDT head | Status | Fidelity |
|---|---|---|---|---|
| fp32 | 480 MB | 305 MB | Works in the demo | Bit-exact vs HuggingFace (parity test cos 1.000000) |
| **fp16** (recommended) | **252 MB** | **152 MB** | Works, 2× smaller, use `scripts/export_unixcoder_reference.py to-fp16 --in ... --out ...` | Identical 4-decimal-place output on the local benchmark |
| Q4 | ~70 MB | ~48 MB linears | Struct + parity tests land (cos ≥ 0.94); no fast kernel yet so latency is 30× worse than fp32 in the browser. Disk-format loader hasn't been wired — roadmap item |

## Caveats

- The inline tokenizer is a placeholder (`mockEncode` in `index.html`). Its
  token ids do **not** match `RobertaTokenizer` byte-for-byte, so the CLS
  feature is slightly off for novel inputs. The encoder itself is bit-exact
  with HuggingFace (see `crates/synapse-inference/tests/unixcoder_parity.rs`).
  Replacing the mock with a real Roberta BPE ported to JS is a follow-up.
- WASM inference is single-threaded on the main UI thread right now —
  encoding 2 snippets @ seq_len=128 takes ~13 s. Move to a WebWorker or add
  `--target web --reference-types --simd` / a fast Q4 GEMM to drop that by
  ~3–5×.
