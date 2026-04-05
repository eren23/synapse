#!/usr/bin/env python3
"""Build a retrieval index from CommitPackFT Python.

Streams the CommitPackFT dataset, extracts N before-state snippets, tokenizes
each with the FNV tokenizer, and saves tokens + metadata for Rust-side retrieval.

Dataset: bigcode/commitpackft (https://huggingface.co/datasets/bigcode/commitpackft)
— provides clean Python code edits with (old_contents, new_contents, message).

Output:
  tests/fixtures/commitpack_index.safetensors  (N x max_len int-as-f32 tokens)
  tests/fixtures/commitpack_meta.json          (list of {repo, subject, message, nbytes})

Usage:
  python3 scripts/build_commitpack_index.py --n 500 --max-len 512
"""

import argparse
import importlib.util
import json
import os
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import save_file


def load_tokenizer(fnv_path: str):
    spec = importlib.util.spec_from_file_location("ast_tokenizer_fnv", fnv_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.tokenize_fnv


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=500, help="Number of snippets to index")
    p.add_argument("--max-len", type=int, default=512)
    p.add_argument("--dataset", default="bigcode/commitpackft", help="HF dataset name")
    p.add_argument("--subset", default="python", help="Dataset subset/config name")
    p.add_argument("--split", default="train")
    p.add_argument("--tokenizer", default="scripts/ast_tokenizer_fnv.py")
    p.add_argument("--out-tokens", default="tests/fixtures/commitpack_index.safetensors")
    p.add_argument("--out-meta", default="tests/fixtures/commitpack_meta.json")
    args = p.parse_args()

    tokenize = load_tokenizer(args.tokenizer)

    print(f"Streaming {args.dataset}:{args.subset}:{args.split} (target: {args.n} snippets)...")
    from datasets import load_dataset
    ds = load_dataset(args.dataset, args.subset, split=args.split, streaming=True)

    tokens_batch = np.zeros((args.n, args.max_len), dtype=np.int64)
    meta = []
    kept = 0
    scanned = 0

    for row in ds:
        scanned += 1
        src = row.get("old_contents") or row.get("old_content") or ""
        if not src or not isinstance(src, str):
            continue
        if len(src) < 60 or len(src) > 20000:  # filter tiny/huge
            continue
        try:
            toks = tokenize(src, max_len=args.max_len)
        except Exception:
            continue

        tokens_batch[kept] = toks
        nonpad = int((toks != 612).sum())
        meta.append({
            "repo": row.get("repos") or row.get("repo") or "",
            "subject": (row.get("subject") or "")[:120],
            "message": (row.get("message") or "")[:200],
            "src_len": len(src),
            "nonpad": nonpad,
        })
        kept += 1
        if kept % 100 == 0:
            print(f"  [{kept}/{args.n}] scanned {scanned}")
        if kept >= args.n:
            break

    if kept < args.n:
        print(f"WARN: only found {kept}/{args.n} usable snippets (scanned {scanned})")
        tokens_batch = tokens_batch[:kept]

    os.makedirs(os.path.dirname(args.out_tokens) or ".", exist_ok=True)
    save_file({"tokens": torch.from_numpy(tokens_batch.astype(np.float32)).contiguous()}, args.out_tokens)
    with open(args.out_meta, "w") as f:
        json.dump({"count": kept, "max_len": args.max_len, "entries": meta}, f, indent=2)

    print(f"\nWrote {args.out_tokens} ({os.path.getsize(args.out_tokens)/1024:.1f} KB) + {args.out_meta}")
    print(f"Total kept: {kept} snippets")


if __name__ == "__main__":
    main()
