#!/usr/bin/env python3
"""Generate reference logits for RWKV-7 validation.

Downloads the RWKV-7 Goose 0.1B model from HuggingFace, runs a forward pass,
and saves the logits as JSON for Rust integration tests.

Usage:
    pip install -r requirements.txt
    python generate_rwkv_reference.py

Output: ../../tests/fixtures/rwkv7_01b_reference.json

Note: RWKV models on HuggingFace often require trust_remote_code=True.
"""
import json
import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "RWKV/RWKV7-Goose-0.1B-HF"
PROMPT = "The capital of France is"
OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "../../tests/fixtures/rwkv7_01b_reference.json")


def main():
    print(f"Loading model: {MODEL_ID}")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, trust_remote_code=True, torch_dtype=torch.float32
    )
    model.train(False)

    print(f"Loading tokenizer: {MODEL_ID}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)

    inputs = tokenizer(PROMPT, return_tensors="pt")
    token_ids = inputs.input_ids[0].tolist()
    print(f"Prompt: {PROMPT!r}")
    print(f"Token IDs ({len(token_ids)}): {token_ids}")

    with torch.no_grad():
        outputs = model(inputs.input_ids)
        logits = outputs.logits[0, -1, :].float()

    top_k = torch.topk(logits, 10)
    vocab_size = logits.shape[0]

    # Dump config for cross-checking
    hf_config = model.config
    config_info = {
        "hidden_size": getattr(hf_config, "hidden_size", None),
        "num_attention_heads": getattr(hf_config, "num_attention_heads", None),
        "num_heads": getattr(hf_config, "num_heads", None),
        "head_size": getattr(hf_config, "head_size", None),
        "num_hidden_layers": getattr(hf_config, "num_hidden_layers", None),
        "intermediate_size": getattr(hf_config, "intermediate_size", None),
        "vocab_size": getattr(hf_config, "vocab_size", None),
        "model_type": getattr(hf_config, "model_type", None),
    }

    # Also inspect weight names for weight mapping verification
    weight_names = []
    for name, param in model.named_parameters():
        weight_names.append({"name": name, "shape": list(param.shape)})

    reference = {
        "model": MODEL_ID,
        "prompt": PROMPT,
        "token_ids": token_ids,
        "logits": logits.tolist(),
        "top_k_ids": top_k.indices.tolist(),
        "top_k_logits": top_k.values.tolist(),
        "vocab_size": vocab_size,
        "config": config_info,
        "weight_names": weight_names[:20],  # First 20 for reference
    }

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(reference, f, indent=2)

    print(f"\nSaved reference to {OUTPUT_PATH}")
    print(f"Vocab size: {vocab_size}")
    print(f"Config: {json.dumps(config_info, indent=2)}")
    print(f"\nWeight name examples (first 20):")
    for wn in weight_names[:20]:
        print(f"  {wn['name']:60s} {wn['shape']}")
    print(f"  ... ({len(weight_names)} total)")
    print(f"\nTop-5 predictions:")
    for i in range(5):
        tid = top_k.indices[i].item()
        val = top_k.values[i].item()
        token_str = tokenizer.decode([tid])
        print(f"  {i+1}. token={tid} ({token_str!r}) logit={val:.4f}")


if __name__ == "__main__":
    main()
