#!/usr/bin/env python3
"""Compare Synapse logits against HuggingFace transformers reference.

Usage:
  pip install transformers torch
  python scripts/verify_logits.py /tmp/qwen3-0.6b

Then compare the output against:
  cargo run --example qwen3_chat --release -- --model-dir /tmp/qwen3-0.6b --verify
"""
import sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_dir = sys.argv[1] if len(sys.argv) > 1 else "/tmp/qwen3-0.6b"

print(f"Loading model from {model_dir}...")
tokenizer = AutoTokenizer.from_pretrained(model_dir)
model = AutoModelForCausalLM.from_pretrained(model_dir, torch_dtype=torch.float32)

prompt = "<|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\n"
inputs = tokenizer(prompt, return_tensors="pt")
token_ids = inputs["input_ids"][0].tolist()
print(f"Prompt: {prompt!r}")
print(f"Token IDs ({len(token_ids)} tokens): {token_ids}")

with torch.no_grad():
    outputs = model(**inputs)
    logits = outputs.logits[0, -1]  # Last position
    top10 = torch.topk(logits, 10)

print(f"\nTop-10 next-token logits (HF reference, float32):")
print(f"{'Token ID':<10} {'Logit':<15} Decoded")
print("-" * 45)
for i in range(10):
    tid = top10.indices[i].item()
    val = top10.values[i].item()
    decoded = tokenizer.decode([tid])
    print(f"{tid:<10} {val:<15.6f} {decoded!r}")
