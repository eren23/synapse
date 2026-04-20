#!/usr/bin/env python3
"""Tokenize a directory of Python files for Code WM retrieval demo.

Walks a directory, tokenizes each .py file via the AST tokenizer, and saves
the batch as a single .safetensors file (tokens as f32 because Synapse's
loader doesn't handle i64). File paths are stored in a sidecar .json.

Usage:
    python3 scripts/tokenize_code_dir.py \
        --src /path/to/ast_tokenizer.py \
        --dir synapse/scripts \
        --out tests/fixtures/code_corpus.safetensors \
        --max-len 512
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import save_file

from _shared import load_tokenizer_func


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--src", required=True, help="Path to ast_tokenizer.py")
    p.add_argument("--dir", required=True, help="Directory to walk for .py files")
    p.add_argument("--out", required=True, help="Output .safetensors path")
    p.add_argument("--max-len", type=int, default=512)
    p.add_argument("--max-files", type=int, default=256, help="Limit on files (first N)")
    args = p.parse_args()

    tokenize = load_tokenizer_func(args.src, "ast_tokenizer", "ast_tokenize")

    root = Path(args.dir).resolve()
    py_files = sorted(p for p in root.rglob("*.py") if p.is_file())[: args.max_files]
    if not py_files:
        raise SystemExit(f"No .py files found under {root}")

    print(f"Tokenizing {len(py_files)} .py files from {root}")
    tokens_batch = np.zeros((len(py_files), args.max_len), dtype=np.int64)
    filenames = []
    sizes = []
    for i, path in enumerate(py_files):
        try:
            source = path.read_text(encoding="utf-8", errors="replace")
        except Exception as e:
            print(f"  skip {path}: {e}")
            continue
        toks = tokenize(source, max_len=args.max_len)
        tokens_batch[i] = toks
        rel = str(path.relative_to(root))
        filenames.append(rel)
        sizes.append(len(source))
        if i < 5 or i == len(py_files) - 1:
            nonpad = int((toks != 612).sum())  # PAD = 612
            print(f"  [{i:3d}] {rel:<60} src={len(source):>6}B nonpad={nonpad}")

    # Save as f32 (values < 662 are exact in f32).
    tokens_f32 = torch.from_numpy(tokens_batch.astype(np.float32))
    out = {"tokens": tokens_f32.contiguous()}
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    save_file(out, args.out)

    sidecar = args.out.replace(".safetensors", "_filenames.json")
    with open(sidecar, "w") as f:
        json.dump({"filenames": filenames, "sizes_bytes": sizes, "max_len": args.max_len}, f, indent=2)

    print(f"\nWrote {args.out} ({os.path.getsize(args.out) / 1024:.1f} KB) + {sidecar}")


if __name__ == "__main__":
    main()
