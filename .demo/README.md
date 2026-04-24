# Demo weights

Shipped via git-lfs so the browser demo at
`synapse/web/unixcoder-delta/` runs without a HuggingFace round-trip.

## Contents

| Path | Bytes | What |
|---|---|---|
| `unixcoder-q4/model.safetensors` | 71 MB | `microsoft/unixcoder-base` packed to Q4_0 on the matmul weights (per-row blocks, F16 scales); biases/norms/embeddings stay F16. |
| `unixcoder-q4/{config,vocab,merges,tokenizer_config,special_tokens_map}.*` | ~1.4 MB | Stock HF tokenizer + model config for `microsoft/unixcoder-base`. |
| `cdt_paper_q4.safetensors` | 43 MB | 1500-step CodeDeltaTok head (`cdt-K1-4blk-contrast0.1-s42`) packed to Q4_0. |

Drop the three files into the demo's file pickers and it runs end-to-end
in the browser — no external downloads.

## Provenance

`unixcoder-q4/model.safetensors` was produced by:

```bash
huggingface-cli download microsoft/unixcoder-base \
    --local-dir /tmp/unixcoder-base
python synapse/scripts/export_unixcoder_reference.py to-q4 \
    --in   /tmp/unixcoder-base/model.safetensors \
    --out  .demo/unixcoder-q4/model.safetensors \
    --only-matmuls
```

`cdt_paper_q4.safetensors` was produced by training a 1500-step CDT on
cached UniXcoder features, converting to safetensors, then Q4-packing:

```bash
# 1. Train (CPU ≈ 25 min / MPS ≈ 5 min on M-series; CUDA much faster):
CDT_HDF5_PATH=~/.cache/huggingface/.../commitpackft_unixcoder_features.h5 \
CDT_FEATURE_DIM=768 CDT_NUM_BLOCKS=4 CDT_NUM_HEADS=12 CDT_NUM_TOKENS=1 \
CDT_LAMBDA_CONTRAST=0.1 CDT_CONTRAST_TEMP=0.07 \
CDT_STEPS=1500 CDT_BATCH_SIZE=256 CDT_SEED=42 \
OUTPUT_DIR=/tmp/cdt_ckpt WANDB_MODE=disabled \
python ~/.crucible-hub/taps/crucible-community-tap/launchers/code_deltatok/train_deltatok.py

# 2. Convert to safetensors with parity tensors (fp32 intermediate):
python synapse/scripts/export_unixcoder_reference.py convert-cdt \
    --ckpt /tmp/cdt_ckpt/code_deltatok_final.pt \
    --ref  synapse/crates/synapse-inference/tests/fixtures/unixcoder_ref.safetensors \
    --out  /tmp/cdt_paper.safetensors \
    --num-blocks 4 --num-heads 12 --num-tokens 1

# 3. Q4-pack:
python synapse/scripts/export_unixcoder_reference.py to-q4 \
    --in   /tmp/cdt_paper.safetensors \
    --out  .demo/cdt_paper_q4.safetensors \
    --only-matmuls
```

## What's not here

- `unixcoder-fp32/` and `unixcoder-fp16/` dirs (regenerable by
  `huggingface-cli download` + optional `to-fp16`).
- `cdt_paper.safetensors` / `cdt_paper_fp16.safetensors` (regenerable via
  the `convert-cdt` / `to-fp16` steps above).

Those are ignored in `.gitignore` so you can keep them locally for
development without paying the LFS storage / bandwidth cost.

## Quality vs download trade-off

| Mode | Download | CLS cos(h_b, h_a) | recon cos |
|---|---|---|---|
| fp32 (desktop) | 785 MB | 0.886 | 0.799 |
| fp16 (browser) | 404 MB | 0.886 | 0.799 |
| **Q4 (shipped)** | **115 MB** | 0.846 | 0.780 |

Numbers from the `def add` → `# sum two numbers` example pair. Q4 drift
is small and the demo remains interactive.
