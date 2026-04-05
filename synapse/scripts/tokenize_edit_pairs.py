#!/usr/bin/env python3
"""Edit-pair corpus: before/after Python snippets mimicking real commits.

The Code WM was trained on CommitPackFT edits (before → after + action).
For each pair, we encode both snippets and measure cosine similarity.
Expected: high cosine (>0.9) because the edits are small, structure-preserving.
Delta norm ≈ low for small edits, higher for bigger edits.

Categories:
  bug_fix          — fix a bug (small, correct)
  add_param        — add a function parameter (signature change)
  add_errors       — add try/except
  add_logging      — add print/log calls
  docstring        — add a docstring
  early_return     — restructure nested if to early returns
  comprehension    — convert loop → list/dict comprehension
  extract_helper   — extract code into helper function
  type_hints       — add type annotations
  rename_var       — rename a variable (pure refactor)
"""

import argparse
import importlib.util
import json
import os

import numpy as np
import torch
from safetensors.torch import save_file


# (name, category, before_src, after_src)
PAIRS = [
    # ── Bug fix: off-by-one in range ──
    ("bug_off_by_one", "bug_fix", '''
def sum_first_n(arr, n):
    total = 0
    for i in range(n - 1):
        total += arr[i]
    return total
''', '''
def sum_first_n(arr, n):
    total = 0
    for i in range(n):
        total += arr[i]
    return total
'''),
    # ── Bug fix: wrong comparison operator ──
    ("bug_wrong_op", "bug_fix", '''
def find_max(arr):
    best = arr[0]
    for x in arr:
        if x < best:
            best = x
    return best
''', '''
def find_max(arr):
    best = arr[0]
    for x in arr:
        if x > best:
            best = x
    return best
'''),
    # ── Add parameter ──
    ("add_param_timeout", "add_param", '''
def fetch(url):
    import urllib.request
    with urllib.request.urlopen(url) as r:
        return r.read()
''', '''
def fetch(url, timeout=30):
    import urllib.request
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return r.read()
'''),
    ("add_param_default", "add_param", '''
def greet(name):
    return f"Hello, {name}!"
''', '''
def greet(name, greeting="Hello"):
    return f"{greeting}, {name}!"
'''),
    # ── Add error handling ──
    ("add_try_except", "add_errors", '''
def load_config(path):
    with open(path) as f:
        return json.load(f)
''', '''
def load_config(path):
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        return None
'''),
    # ── Add logging ──
    ("add_logging", "add_logging", '''
def process(items):
    results = []
    for item in items:
        results.append(transform(item))
    return results
''', '''
def process(items):
    results = []
    for item in items:
        print(f"processing {item}")
        results.append(transform(item))
    print(f"done, {len(results)} results")
    return results
'''),
    # ── Add docstring ──
    ("add_docstring", "docstring", '''
def bsearch(arr, target):
    lo, hi = 0, len(arr) - 1
    while lo <= hi:
        mid = (lo + hi) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            lo = mid + 1
        else:
            hi = mid - 1
    return -1
''', '''
def bsearch(arr, target):
    """Binary search for target in sorted arr. Returns index or -1."""
    lo, hi = 0, len(arr) - 1
    while lo <= hi:
        mid = (lo + hi) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            lo = mid + 1
        else:
            hi = mid - 1
    return -1
'''),
    # ── Nested if → early return ──
    ("early_return", "early_return", '''
def classify(x):
    result = None
    if x is not None:
        if x > 0:
            if x < 10:
                result = "small"
            else:
                result = "big"
        else:
            result = "negative"
    return result
''', '''
def classify(x):
    if x is None:
        return None
    if x <= 0:
        return "negative"
    if x < 10:
        return "small"
    return "big"
'''),
    # ── Loop → comprehension ──
    ("loop_to_comp", "comprehension", '''
def squares(n):
    result = []
    for i in range(n):
        if i % 2 == 0:
            result.append(i * i)
    return result
''', '''
def squares(n):
    return [i * i for i in range(n) if i % 2 == 0]
'''),
    # ── Extract helper ──
    ("extract_helper", "extract_helper", '''
def process_line(line):
    s = line.strip().lower()
    s = s.replace(",", "")
    s = s.replace(".", "")
    return s.split()
''', '''
def normalize(s):
    return s.strip().lower().replace(",", "").replace(".", "")

def process_line(line):
    return normalize(line).split()
'''),
    # ── Add type hints ──
    ("add_type_hints", "type_hints", '''
def add_all(a, b, scale):
    return [(x + y) * scale for x, y in zip(a, b)]
''', '''
def add_all(a: list[float], b: list[float], scale: float) -> list[float]:
    return [(x + y) * scale for x, y in zip(a, b)]
'''),
    # ── Rename variable ──
    ("rename_var", "rename_var", '''
def compute(n):
    result = 0
    for i in range(n):
        result += i * i
    return result
''', '''
def compute(count):
    total = 0
    for idx in range(count):
        total += idx * idx
    return total
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

    n = len(PAIRS) * 2  # (before, after) per pair
    tokens = np.zeros((n, args.max_len), dtype=np.int64)
    meta = []
    for i, (name, cat, before, after) in enumerate(PAIRS):
        tokens[i * 2] = tokenize(before, max_len=args.max_len)
        tokens[i * 2 + 1] = tokenize(after, max_len=args.max_len)
        np_before = int((tokens[i * 2] != 612).sum())
        np_after = int((tokens[i * 2 + 1] != 612).sum())
        meta.append({"pair": name, "category": cat, "nonpad_before": np_before, "nonpad_after": np_after})
        print(f"  [{i:2d}] {name:<20} cat={cat:<14} before={np_before:3d} after={np_after:3d}")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    save_file({"tokens": torch.from_numpy(tokens.astype(np.float32)).contiguous()}, args.out)
    sidecar = args.out.replace(".safetensors", "_meta.json")
    with open(sidecar, "w") as f:
        json.dump({"pairs": meta, "max_len": args.max_len, "num_pairs": len(PAIRS)}, f, indent=2)
    print(f"\nWrote {args.out} ({n} rows = {len(PAIRS)} pairs × 2) and {sidecar}")


if __name__ == "__main__":
    main()
