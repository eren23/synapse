#!/usr/bin/env python3
"""Refactor-pair corpus: semantically identical, syntactically different.

The pairs preserve behavior exactly but change surface syntax:
- Pure renames (variables, functions)
- Idiom swaps (for loop ↔ comprehension, if/else ↔ ternary)
- Equivalent formulations (a+b+c ↔ sum([a,b,c]), str() ↔ f-string)

For AST-minimal refactors (just renames), expect cos > 0.95.
For AST-structural refactors (loop→comp), expect cos 0.7-0.9.
"""

import argparse
import importlib.util
import json
import os

import numpy as np
import torch
from safetensors.torch import save_file


# (name, refactor_kind, before, after)
PAIRS = [
    # ── Pure renames (no AST structure change) ─────────────
    ("rename_single", "rename", '''
def area(r):
    return 3.14 * r * r
''', '''
def area(radius):
    return 3.14 * radius * radius
'''),
    ("rename_many", "rename", '''
def process(data):
    result = []
    for item in data:
        value = item * 2
        result.append(value)
    return result
''', '''
def process(records):
    output = []
    for entry in records:
        doubled = entry * 2
        output.append(doubled)
    return output
'''),
    ("rename_function", "rename", '''
def compute_total(items):
    return sum(items)

def main():
    print(compute_total([1, 2, 3]))
''', '''
def total(items):
    return sum(items)

def main():
    print(total([1, 2, 3]))
'''),
    # ── Idiom swaps (some AST change) ──────────────────────
    ("if_to_ternary", "idiom", '''
def abs_val(x):
    if x >= 0:
        result = x
    else:
        result = -x
    return result
''', '''
def abs_val(x):
    result = x if x >= 0 else -x
    return result
'''),
    ("str_to_fstring", "idiom", '''
def format_name(first, last):
    return "Hello, " + first + " " + last + "!"
''', '''
def format_name(first, last):
    return f"Hello, {first} {last}!"
'''),
    ("loop_to_sum", "idiom", '''
def total(arr):
    s = 0
    for x in arr:
        s = s + x
    return s
''', '''
def total(arr):
    return sum(arr)
'''),
    ("is_none_check", "idiom", '''
def is_missing(x):
    if x == None:
        return True
    return False
''', '''
def is_missing(x):
    return x is None
'''),
    ("dict_literal", "idiom", '''
def config():
    d = dict()
    d["host"] = "localhost"
    d["port"] = 8080
    return d
''', '''
def config():
    return {"host": "localhost", "port": 8080}
'''),
    # ── Structural refactors (bigger AST change) ───────────
    ("loop_to_comp", "structural", '''
def evens(arr):
    result = []
    for x in arr:
        if x % 2 == 0:
            result.append(x)
    return result
''', '''
def evens(arr):
    return [x for x in arr if x % 2 == 0]
'''),
    ("nested_to_flat", "structural", '''
def valid(user):
    if user is not None:
        if user.active:
            if user.age >= 18:
                return True
    return False
''', '''
def valid(user):
    if user is None:
        return False
    if not user.active:
        return False
    return user.age >= 18
'''),
]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tokenizer", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--max-len", type=int, default=256)
    args = p.parse_args()

    spec = importlib.util.spec_from_file_location("ast_tokenizer", args.tokenizer)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    tokenize = mod.ast_tokenize

    n = len(PAIRS) * 2
    tokens = np.zeros((n, args.max_len), dtype=np.int64)
    meta = []
    for i, (name, kind, before, after) in enumerate(PAIRS):
        tokens[i * 2] = tokenize(before, max_len=args.max_len)
        tokens[i * 2 + 1] = tokenize(after, max_len=args.max_len)
        nb = int((tokens[i * 2] != 612).sum())
        na = int((tokens[i * 2 + 1] != 612).sum())
        meta.append({"pair": name, "category": kind, "nonpad_before": nb, "nonpad_after": na})
        print(f"  [{i:2d}] {name:<18} kind={kind:<10} before={nb:3d} after={na:3d}")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    save_file({"tokens": torch.from_numpy(tokens.astype(np.float32)).contiguous()}, args.out)
    sidecar = args.out.replace(".safetensors", "_meta.json")
    with open(sidecar, "w") as f:
        json.dump({"pairs": meta, "max_len": args.max_len, "num_pairs": len(PAIRS)}, f, indent=2)
    print(f"\nWrote {args.out} ({len(PAIRS)} pairs)")


if __name__ == "__main__":
    main()
