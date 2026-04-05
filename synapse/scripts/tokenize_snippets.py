#!/usr/bin/env python3
"""Curated semantic similarity test for Code WM.

Hand-picked Python snippets with known semantic relationships. Encode each,
compute pairwise cosine, check whether semantically-similar pairs cluster.

Categories:
  sort   — sorting algorithms (expect high intra-cluster cosine)
  str    — string manipulation (expect high intra-cluster)
  math   — numeric/geometric helpers
  io     — file I/O
  http   — HTTP server/client

Usage:
    python3 scripts/tokenize_snippets.py \
        --tokenizer /tmp/code_wm_test/ast_tokenizer.py \
        --out tests/fixtures/snippets.safetensors
"""

import argparse
import importlib.util
import json
import os

import numpy as np
import torch
from safetensors.torch import save_file


# name, category, source code
SNIPPETS = [
    # ── Sorting ──────────────────────────────────────────
    ("quicksort", "sort", '''
def quicksort(arr):
    if len(arr) <= 1:
        return arr
    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    mid = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    return quicksort(left) + mid + quicksort(right)
'''),
    ("mergesort", "sort", '''
def mergesort(arr):
    if len(arr) <= 1:
        return arr
    mid = len(arr) // 2
    left = mergesort(arr[:mid])
    right = mergesort(arr[mid:])
    return merge(left, right)

def merge(a, b):
    result = []
    i = j = 0
    while i < len(a) and j < len(b):
        if a[i] < b[j]:
            result.append(a[i]); i += 1
        else:
            result.append(b[j]); j += 1
    result.extend(a[i:]); result.extend(b[j:])
    return result
'''),
    ("bubblesort", "sort", '''
def bubblesort(arr):
    n = len(arr)
    for i in range(n):
        for j in range(0, n - i - 1):
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
    return arr
'''),
    # ── Strings ──────────────────────────────────────────
    ("reverse_string", "str", '''
def reverse_string(s):
    return s[::-1]

def reverse_iter(s):
    result = ""
    for ch in s:
        result = ch + result
    return result
'''),
    ("palindrome", "str", '''
def is_palindrome(s):
    s = s.lower().replace(" ", "")
    return s == s[::-1]

def count_palindromes(words):
    return sum(1 for w in words if is_palindrome(w))
'''),
    ("word_count", "str", '''
def word_count(text):
    words = text.lower().split()
    counts = {}
    for w in words:
        counts[w] = counts.get(w, 0) + 1
    return counts
'''),
    # ── Math ─────────────────────────────────────────────
    ("fibonacci", "math", '''
def fibonacci(n):
    if n < 2:
        return n
    a, b = 0, 1
    for _ in range(n - 1):
        a, b = b, a + b
    return b
'''),
    ("gcd", "math", '''
def gcd(a, b):
    while b:
        a, b = b, a % b
    return a

def lcm(a, b):
    return a * b // gcd(a, b)
'''),
    ("prime_sieve", "math", '''
def primes_up_to(n):
    sieve = [True] * (n + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, int(n**0.5) + 1):
        if sieve[i]:
            for j in range(i*i, n + 1, i):
                sieve[j] = False
    return [i for i, p in enumerate(sieve) if p]
'''),
    # ── File I/O ─────────────────────────────────────────
    ("read_csv", "io", '''
import csv

def read_csv(path):
    rows = []
    with open(path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows
'''),
    ("write_json", "io", '''
import json

def write_json(path, data):
    with open(path, "w") as f:
        json.dump(data, f, indent=2)

def read_json(path):
    with open(path, "r") as f:
        return json.load(f)
'''),
    ("file_size", "io", '''
import os

def file_size_kb(path):
    return os.path.getsize(path) / 1024

def list_large_files(dir_path, min_kb):
    results = []
    for name in os.listdir(dir_path):
        full = os.path.join(dir_path, name)
        if os.path.isfile(full) and file_size_kb(full) > min_kb:
            results.append(full)
    return results
'''),
    # ── HTTP ─────────────────────────────────────────────
    ("http_get", "http", '''
import urllib.request

def http_get(url):
    with urllib.request.urlopen(url) as resp:
        return resp.read().decode("utf-8")

def http_json(url):
    import json
    return json.loads(http_get(url))
'''),
    ("simple_server", "http", '''
from http.server import HTTPServer, BaseHTTPRequestHandler

class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/plain")
        self.end_headers()
        self.wfile.write(b"Hello, world!")

def run_server(port=8080):
    server = HTTPServer(("127.0.0.1", port), Handler)
    server.serve_forever()
'''),
    ("webhook_post", "http", '''
import urllib.request
import json

def post_webhook(url, payload):
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req) as resp:
        return resp.status
'''),
]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tokenizer", required=True, help="Path to ast_tokenizer.py")
    p.add_argument("--out", required=True, help="Output .safetensors")
    p.add_argument("--max-len", type=int, default=512)
    args = p.parse_args()

    spec = importlib.util.spec_from_file_location("ast_tokenizer", args.tokenizer)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    tokenize = mod.ast_tokenize

    n = len(SNIPPETS)
    tokens = np.zeros((n, args.max_len), dtype=np.int64)
    meta = []
    for i, (name, cat, src) in enumerate(SNIPPETS):
        toks = tokenize(src, max_len=args.max_len)
        tokens[i] = toks
        nonpad = int((toks != 612).sum())
        meta.append({"name": name, "category": cat, "nonpad": nonpad, "src_chars": len(src)})
        print(f"  [{i:2d}] {name:<16} cat={cat:<5} nonpad={nonpad:3d}")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    save_file({"tokens": torch.from_numpy(tokens.astype(np.float32)).contiguous()}, args.out)

    sidecar = args.out.replace(".safetensors", "_meta.json")
    with open(sidecar, "w") as f:
        json.dump({"snippets": meta, "max_len": args.max_len}, f, indent=2)

    print(f"\nWrote {args.out} and {sidecar}")
    print(f"Categories: {sorted(set(m['category'] for m in meta))}")


if __name__ == "__main__":
    main()
