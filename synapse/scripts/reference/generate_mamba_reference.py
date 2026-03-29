#!/usr/bin/env python3
"""Generate reference logits for Mamba-130M validation.

Usage:
    pip install -r requirements.txt
    python generate_mamba_reference.py

Output: ../../tests/fixtures/mamba_130m_reference.json
"""
import json
import os

import torch
from transformers import MambaConfig, MambaForCausalLM, AutoTokenizer

MODEL_ID = "state-spaces/mamba-130m"
PROMPT = "The capital of France is"
OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "../../tests/fixtures/mamba_130m_reference.json")


def main():
    print(f"Loading model: {MODEL_ID}")

    # Build config manually from the original checkpoint's config.json
    # (the original uses d_model/n_layer, not hidden_size/num_hidden_layers)
    snapshot_dir = os.path.expanduser("~/.cache/huggingface/hub/models--state-spaces--mamba-130m/snapshots")
    orig_cfg = None
    for snap in os.listdir(snapshot_dir):
        cfg_path = os.path.join(snapshot_dir, snap, "config.json")
        if os.path.exists(cfg_path):
            with open(cfg_path) as f:
                orig_cfg = json.load(f)
            break

    if orig_cfg is None:
        raise RuntimeError("Could not find downloaded config.json")

    d_model = orig_cfg.get("d_model", 768)
    n_layer = orig_cfg.get("n_layer", 24)
    # Vocab in weights is padded to multiple of 8: 50277 -> 50280
    pad_mult = orig_cfg.get("pad_vocab_size_multiple", 8)
    raw_vocab = orig_cfg.get("vocab_size", 50277)
    vocab_size = ((raw_vocab + pad_mult - 1) // pad_mult) * pad_mult
    print(f"Config: d_model={d_model}, n_layer={n_layer}, vocab_size={raw_vocab}->{vocab_size}")

    config = MambaConfig(
        hidden_size=d_model,
        num_hidden_layers=n_layer,
        vocab_size=vocab_size,
    )

    # Instantiate model on CPU with real tensors (not meta)
    with torch.device("cpu"):
        model = MambaForCausalLM(config)

    # Load the pytorch checkpoint weights
    ckpt_path = os.path.join(snapshot_dir, snap, "pytorch_model.bin")
    state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict, strict=False)
    model = model.float()
    model.train(False)

    print(f"Loaded {len(state_dict)} weight tensors")

    # Print dt_rank info from loaded weights
    for k, v in state_dict.items():
        if "x_proj" in k and "weight" in k:
            print(f"  {k}: {list(v.shape)}")
            break
    for k, v in state_dict.items():
        if "dt_proj" in k and "weight" in k:
            print(f"  {k}: {list(v.shape)}")
            break

    # Tokenizer: original Mamba uses GPT-NeoX
    print("Loading tokenizer (GPT-NeoX)...")
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")

    inputs = tokenizer(PROMPT, return_tensors="pt")
    token_ids = inputs.input_ids[0].tolist()
    print(f"Prompt: {PROMPT!r}")
    print(f"Token IDs ({len(token_ids)}): {token_ids}")

    with torch.no_grad():
        outputs = model(inputs.input_ids)
        logits = outputs.logits[0, -1, :].float()

    top_k = torch.topk(logits, 10)
    vocab_size_out = logits.shape[0]

    config_info = {
        "d_model": config.hidden_size,
        "d_state": config.state_size,
        "d_conv": config.conv_kernel,
        "expand": config.expand,
        "dt_rank": getattr(config, "time_step_rank", "auto"),
        "num_layers": config.num_hidden_layers,
        "vocab_size": config.vocab_size,
    }

    reference = {
        "model": MODEL_ID,
        "prompt": PROMPT,
        "token_ids": token_ids,
        "logits": logits.tolist(),
        "top_k_ids": top_k.indices.tolist(),
        "top_k_logits": top_k.values.tolist(),
        "vocab_size": vocab_size_out,
        "config": config_info,
    }

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(reference, f, indent=2)

    print(f"\nSaved reference to {OUTPUT_PATH}")
    print(f"Vocab size: {vocab_size_out}")
    print(f"Config: {json.dumps(config_info, indent=2)}")
    print(f"\nTop-5 predictions:")
    for i in range(5):
        tid = top_k.indices[i].item()
        val = top_k.values[i].item()
        token_str = tokenizer.decode([tid])
        print(f"  {i+1}. token={tid} ({token_str!r}) logit={val:.4f}")


if __name__ == "__main__":
    main()
