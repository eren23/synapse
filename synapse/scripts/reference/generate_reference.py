#!/usr/bin/env python3
"""Generate reference logits from a HuggingFace model for Synapse validation.

Usage:
    python generate_reference.py --model state-spaces/mamba-130m \
        --prompt "The capital of France is" \
        --output ../../tests/fixtures/mamba_130m_reference.json

The output JSON contains:
    - model: HF model ID
    - prompt: text prompt used
    - token_ids: tokenized input IDs
    - logits: full logit vector from last position (float32)
    - top_k_ids: top-10 predicted token IDs
    - top_k_logits: corresponding logit values
    - vocab_size: model vocabulary size
"""
import argparse
import json
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def generate_reference(model_id: str, prompt: str, output_path: str, trust_remote: bool = False):
    print(f"Loading model: {model_id}")
    model = AutoModelForCausalLM.from_pretrained(
        model_id, trust_remote_code=trust_remote, torch_dtype=torch.float32
    )
    model.train(False)

    print(f"Loading tokenizer: {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=trust_remote)

    inputs = tokenizer(prompt, return_tensors="pt")
    token_ids = inputs.input_ids[0].tolist()
    print(f"Token IDs: {token_ids}")

    with torch.no_grad():
        outputs = model(inputs.input_ids)
        # Last position logits: [vocab_size]
        logits = outputs.logits[0, -1, :].float()

    top_k = torch.topk(logits, 10)
    vocab_size = logits.shape[0]

    reference = {
        "model": model_id,
        "prompt": prompt,
        "token_ids": token_ids,
        "logits": logits.tolist(),
        "top_k_ids": top_k.indices.tolist(),
        "top_k_logits": top_k.values.tolist(),
        "vocab_size": vocab_size,
    }

    with open(output_path, "w") as f:
        json.dump(reference, f, indent=2)

    print(f"Saved reference to {output_path}")
    print(f"Vocab size: {vocab_size}")
    print(f"Top-5 predictions:")
    for i in range(5):
        tid = top_k.indices[i].item()
        val = top_k.values[i].item()
        token_str = tokenizer.decode([tid])
        print(f"  {i+1}. token={tid} ({token_str!r}) logit={val:.4f}")


def main():
    parser = argparse.ArgumentParser(description="Generate reference logits from HuggingFace models")
    parser.add_argument("--model", required=True, help="HuggingFace model ID")
    parser.add_argument("--prompt", default="The capital of France is", help="Input prompt")
    parser.add_argument("--output", required=True, help="Output JSON path")
    parser.add_argument("--trust-remote-code", action="store_true", help="Trust remote code (for RWKV etc)")
    args = parser.parse_args()

    generate_reference(args.model, args.prompt, args.output, args.trust_remote_code)


if __name__ == "__main__":
    main()
