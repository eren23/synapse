#!/usr/bin/env python3
"""Build a retrieval index by walking a directory tree of .py files.

For real-world validation: encode 500+ diverse Python files (from site-packages,
stdlib, or any corpus) and measure retrieval quality on actual code.

Usage:
  python3 scripts/build_file_index.py --dir .venv-rwkv-debug/lib/python3.14/site-packages --n 500
"""

import argparse
import importlib.util
import json
import os
import random
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
    p.add_argument("--dir", required=True, help="Root directory to walk")
    p.add_argument("--n", type=int, default=500)
    p.add_argument("--max-len", type=int, default=512)
    p.add_argument("--min-bytes", type=int, default=300, help="Skip files smaller than this")
    p.add_argument("--max-bytes", type=int, default=50000, help="Skip files larger than this")
    p.add_argument("--tokenizer", default="scripts/ast_tokenizer_fnv.py")
    p.add_argument("--out-tokens", default="tests/fixtures/file_index.safetensors")
    p.add_argument("--out-meta", default="tests/fixtures/file_index_meta.json")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    tokenize = load_tokenizer(args.tokenizer)

    print(f"Walking {args.dir} for .py files ({args.min_bytes}-{args.max_bytes} bytes)...")
    root = Path(args.dir).resolve()
    candidates = []
    for path in root.rglob("*.py"):
        if not path.is_file():
            continue
        try:
            size = path.stat().st_size
        except OSError:
            continue
        if size < args.min_bytes or size > args.max_bytes:
            continue
        candidates.append(path)
    print(f"  {len(candidates)} candidates")

    random.seed(args.seed)
    random.shuffle(candidates)
    candidates = candidates[: args.n * 2]  # overshoot to allow rejections

    tokens_batch = np.zeros((args.n, args.max_len), dtype=np.int64)
    meta = []
    kept = 0
    for path in candidates:
        if kept >= args.n:
            break
        try:
            src = path.read_text(encoding="utf-8", errors="replace")
        except (OSError, UnicodeDecodeError):
            continue
        try:
            toks = tokenize(src, max_len=args.max_len)
        except (SyntaxError, ValueError, RecursionError):
            continue
        if (toks == 612).all():  # all PAD
            continue
        tokens_batch[kept] = toks
        rel = str(path.relative_to(root))
        # Get 2-line preview
        preview = "\n".join(src.split("\n")[:2])[:140]
        meta.append({
            "path": rel,
            "preview": preview,
            "src_len": len(src),
            "nonpad": int((toks != 612).sum()),
        })
        kept += 1

    if kept < args.n:
        print(f"WARN: only kept {kept}/{args.n}")
        tokens_batch = tokens_batch[:kept]

    os.makedirs(os.path.dirname(args.out_tokens) or ".", exist_ok=True)
    save_file({"tokens": torch.from_numpy(tokens_batch.astype(np.float32)).contiguous()}, args.out_tokens)
    with open(args.out_meta, "w") as f:
        json.dump({"count": kept, "max_len": args.max_len, "root": str(root), "entries": meta}, f, indent=2)

    print(f"\nWrote {args.out_tokens} ({os.path.getsize(args.out_tokens)/1024:.1f} KB) + {args.out_meta}")
    print(f"Kept {kept} files")


if __name__ == "__main__":
    main()
