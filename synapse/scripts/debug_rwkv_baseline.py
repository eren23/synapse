#!/usr/bin/env python3
"""HF baseline for RWKV logit / tokenizer comparison (pair with rwkv_logit_probe).

  python -m venv .venv && source .venv/bin/activate
  pip install torch transformers safetensors

  # From HuggingFace (needs modeling_rwkv7.py + configuration_rwkv7.py — use hub id):
  HF_HOME=$PWD/.hf-cache python scripts/debug_rwkv_baseline.py SmerkyG/RWKV7-Goose-0.1B-Pile-HF hello

  # From a full local snapshot (must include configuration_rwkv7.py, modeling_rwkv7.py):
  python scripts/debug_rwkv_baseline.py models/rwkv7-pile-0.1b hello

  Note: upstream modeling_rwkv7.py imports Triton; Linux+CUDA can pip install triton.
  macOS often cannot — use Linux or a CPU-only fork for strict numeric baseline.
"""
from __future__ import annotations

import sys

import torch


def main() -> None:
    if len(sys.argv) < 2:
        print(
            "Usage: python scripts/debug_rwkv_baseline.py MODEL_OR_HUB_ID [PROMPT]",
            file=sys.stderr,
        )
        sys.exit(1)
    source = sys.argv[1]
    prompt = sys.argv[2] if len(sys.argv) > 2 else "hello"

    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"# loading {source}", file=sys.stderr)
    tok = AutoTokenizer.from_pretrained(source, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        source, trust_remote_code=True, dtype=torch.float32
    )
    model.eval()

    inputs = tok(prompt, return_tensors="pt")
    ids = inputs["input_ids"][0].tolist()
    print("HF_TOKEN_IDS:" + ",".join(str(x) for x in ids))

    with torch.no_grad():
        out = model(inputs["input_ids"])
        logits = out.logits[0, -1, :].float()

    vocab = int(logits.shape[0])
    print(f"HF_VOCAB_SIZE:{vocab}")

    top = torch.topk(logits, k=min(15, vocab))
    for rank in range(top.indices.shape[0]):
        tid = int(top.indices[rank].item())
        val = float(top.values[rank].item())
        piece = tok.decode([tid])
        esc = repr(piece)
        print(f"HF_TOP_{rank + 1}: id={tid} logit={val:.4f} decode={esc}")


if __name__ == "__main__":
    main()
