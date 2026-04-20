#!/usr/bin/env python3
"""Compare LFM2.5-350M logits: Synapse (GGUF) vs HuggingFace (reference).

Usage:
    python3 scripts/lfm25_baseline_comparison.py

Requires: transformers, torch, accelerate
"""
import subprocess, sys, time
import numpy as np

def get_hf_logits(token_ids: list[int]) -> np.ndarray:
    """Get reference logits from HuggingFace transformers."""
    import torch
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        "LiquidAI/LFM2.5-350M", device_map="cpu", torch_dtype=torch.float32
    )
    with torch.no_grad():
        inputs = torch.tensor([token_ids], dtype=torch.long)
        out = model(inputs)
        # Last token logits
        logits = out.logits[0, -1, :].numpy()
    return logits

def get_synapse_logits(token_ids: list[int], gguf_path: str) -> np.ndarray:
    """Get logits from Synapse GGUF inference."""
    # Use the example binary
    result = subprocess.run(
        ["cargo", "run", "--release", "-p", "synapse-inference",
         "--example", "lfm25_inference", "--", gguf_path],
        capture_output=True, text=True, cwd="."
    )
    # Parse logits from output
    for line in result.stdout.split('\n'):
        if 'logits[0..5]' in line and 'Decode' not in line:
            # Extract the first 5 logits
            start = line.index('[') + 1
            end = line.index(']')
            vals = [float(x.strip()) for x in line[start:end].split(',')]
            return np.array(vals)
    print("STDERR:", result.stderr[-500:] if result.stderr else "none")
    raise RuntimeError(f"Could not parse Synapse output:\n{result.stdout}")

def main():
    import glob
    gguf_files = glob.glob("models/lfm25-350m/**/LFM2.5-350M-Q4_0.gguf", recursive=True)
    if not gguf_files:
        print("ERROR: GGUF not found. Download first.")
        sys.exit(1)
    gguf_path = gguf_files[0]

    token_ids = [1, 42, 100, 200, 300]
    print(f"Test tokens: {token_ids}")
    print(f"GGUF: {gguf_path}\n")

    # HuggingFace baseline
    print("=== HuggingFace (f32 reference) ===")
    t0 = time.time()
    hf_logits = get_hf_logits(token_ids)
    hf_time = time.time() - t0
    hf_top5 = np.argsort(hf_logits)[-5:][::-1]
    print(f"  Time: {hf_time:.1f}s")
    print(f"  Top-5 tokens: {hf_top5.tolist()}")
    print(f"  Top-5 logits: {[f'{hf_logits[i]:.4f}' for i in hf_top5]}")
    print(f"  logits[0..5]: {[f'{v:.4f}' for v in hf_logits[:5]]}")

    # Synapse (Q4 GGUF)
    print("\n=== Synapse (Q4_0 GGUF) ===")
    t0 = time.time()
    syn_logits = get_synapse_logits(token_ids, gguf_path)
    syn_time = time.time() - t0
    print(f"  Time: {syn_time:.1f}s")
    print(f"  logits[0..5]: {[f'{v:.4f}' for v in syn_logits]}")

    # Compare first 5 logits
    print("\n=== Comparison (first 5 logits) ===")
    hf5 = hf_logits[:5]
    diff = np.abs(hf5 - syn_logits[:len(hf5)])
    print(f"  HF:      {[f'{v:.4f}' for v in hf5]}")
    print(f"  Synapse: {[f'{v:.4f}' for v in syn_logits[:len(hf5)]]}")
    print(f"  Abs diff:{[f'{v:.4f}' for v in diff]}")
    print(f"  Max diff: {diff.max():.4f}")

    # Cosine similarity of full HF logits vs what we can compare
    if len(syn_logits) >= 5 and len(hf5) >= 5:
        cos = np.dot(hf5, syn_logits[:5]) / (np.linalg.norm(hf5) * np.linalg.norm(syn_logits[:5]) + 1e-8)
        print(f"  Cosine (5 dims): {cos:.6f}")

    print("\n=== Notes ===")
    print("  Q4_0 quantization introduces some divergence from f32.")
    print("  Key metric: top-k token agreement, not exact logit match.")
    print("  For exact match, use F16 or F32 GGUF variant.")

if __name__ == "__main__":
    main()
