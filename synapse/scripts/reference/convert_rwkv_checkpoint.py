#!/usr/bin/env python3
"""Convert official RWKV-7 HF checkpoint to Synapse-compatible format.

Handles:
1. bf16 → f32 conversion
2. LoRA-style names → flat names (w_lora.lora.{0,2} → w0/w1/w2)
3. Squeeze [1,1,h] lerp shapes to [h]
4. Add num_heads to config if missing

Usage:
    python convert_rwkv_checkpoint.py models/rwkv7-0.1b models/rwkv7-0.1b-converted
"""
import json
import os
import shutil
import sys

import torch
from safetensors import safe_open
from safetensors.torch import save_file


def convert(src_dir, dst_dir):
    os.makedirs(dst_dir, exist_ok=True)

    # Load config
    with open(os.path.join(src_dir, "config.json")) as f:
        config = json.load(f)

    hidden_size = config["hidden_size"]
    head_dim = config.get("head_dim", 64)
    num_heads = config.get("num_heads") or (hidden_size // head_dim)
    num_layers = config["num_hidden_layers"]
    print(f"Model: hidden={hidden_size}, heads={num_heads}, layers={num_layers}")

    # Load weights
    src_path = os.path.join(src_dir, "model.safetensors")
    with safe_open(src_path, framework="pt") as f:
        keys = list(f.keys())
        src_tensors = {k: f.get_tensor(k).float().contiguous() for k in keys}

    print(f"Loaded {len(src_tensors)} tensors (converted to f32)")

    dst_tensors = {}

    def copy(src_name, dst_name=None, squeeze=False):
        if dst_name is None:
            dst_name = src_name
        t = src_tensors[src_name]
        if squeeze:
            t = t.squeeze()
        dst_tensors[dst_name] = t.clone()

    # Global weights — handle naming variants
    copy("model.embeddings.weight")
    if "lm_head.weight" in src_tensors:
        copy("lm_head.weight")
    elif "head.weight" in src_tensors:
        copy("head.weight", "lm_head.weight")

    # Final norm — detect naming
    if "model.norm.weight" in src_tensors:
        copy("model.norm.weight")
        copy("model.norm.bias")
    elif "model.ln_out.weight" in src_tensors:
        copy("model.ln_out.weight", "model.norm.weight")
        copy("model.ln_out.bias", "model.norm.bias")

    for i in range(num_layers):
        # Detect naming style
        official = f"model.layers.{i}.attn" in ".".join(k for k in src_tensors if f"layers.{i}" in k or f"blocks.{i}" in k)

        if any(k.startswith(f"model.layers.{i}.") for k in src_tensors):
            att = f"model.layers.{i}.attn"
            ffn = f"model.layers.{i}.ffn"
            # Norms
            copy(f"model.layers.{i}.attn_norm.weight")
            copy(f"model.layers.{i}.attn_norm.bias")
            copy(f"model.layers.{i}.ffn_norm.weight")
            copy(f"model.layers.{i}.ffn_norm.bias")
            pn = f"model.layers.{i}.pre_norm.weight"
            if pn in src_tensors:
                copy(pn)
                copy(f"model.layers.{i}.pre_norm.bias")
        else:
            att = f"model.blocks.{i}.attention"
            ffn = f"model.blocks.{i}.feed_forward"
            # SmerkyG norms: ln1, ln2
            copy(f"model.blocks.{i}.ln1.weight")
            copy(f"model.blocks.{i}.ln1.bias")
            copy(f"model.blocks.{i}.ln2.weight")
            copy(f"model.blocks.{i}.ln2.bias")

        # Token shift lerps — squeeze [1,1,h] to [h]
        for name in ["x_r", "x_k", "x_v", "x_w", "x_a", "x_g"]:
            src_key = f"{att}.{name}"
            if src_key in src_tensors:
                copy(src_key, squeeze=True)

        # Linear projections — try both naming conventions
        for proj in ["r_proj", "k_proj", "v_proj", "o_proj"]:
            src_key = f"{att}.{proj}.weight"
            if src_key in src_tensors:
                copy(src_key)
            else:
                # Try SmerkyG naming
                alt_names = {"r_proj": "receptance", "k_proj": "key", "v_proj": "value", "o_proj": "output"}
                alt_key = f"{att}.{alt_names[proj]}.weight"
                if alt_key in src_tensors:
                    copy(alt_key, src_key)

        # Decay (w) — LoRA-style → flat
        w_lora_0 = f"{att}.w_lora.lora.0.weight"
        w_lora_2w = f"{att}.w_lora.lora.2.weight"
        w_lora_2b = f"{att}.w_lora.lora.2.bias"
        if w_lora_0 in src_tensors:
            # lora.0.weight is [decay_rank, h] — our w1 is [h, decay_rank]
            dst_tensors[f"{att}.w1"] = src_tensors[w_lora_0].t().contiguous().clone()
            # lora.2.weight is [h, decay_rank] — our w2 is [decay_rank, h]
            dst_tensors[f"{att}.w2"] = src_tensors[w_lora_2w].t().contiguous().clone()
            # lora.2.bias is [h] — our w0
            dst_tensors[f"{att}.w0"] = src_tensors[w_lora_2b].clone()
        else:
            # SmerkyG flat naming
            for name in ["w0", "w1", "w2"]:
                src_key = f"{att}.{name}"
                if src_key in src_tensors:
                    copy(src_key, squeeze=(name == "w0"))

        # Alpha (a) — LoRA-style → flat
        a_lora_0 = f"{att}.a_lora.lora.0.weight"
        a_lora_2w = f"{att}.a_lora.lora.2.weight"
        a_lora_2b = f"{att}.a_lora.lora.2.bias"
        if a_lora_0 in src_tensors:
            dst_tensors[f"{att}.a1"] = src_tensors[a_lora_0].t().contiguous().clone()
            dst_tensors[f"{att}.a2"] = src_tensors[a_lora_2w].t().contiguous().clone()
            dst_tensors[f"{att}.a0"] = src_tensors[a_lora_2b].clone()
        else:
            for name in ["a0", "a1", "a2"]:
                src_key = f"{att}.{name}"
                if src_key in src_tensors:
                    copy(src_key, squeeze=(name == "a0"))

        # Gate (g) — LoRA-style → flat
        g_lora_0 = f"{att}.g_lora.lora.0.weight"
        g_lora_2w = f"{att}.g_lora.lora.2.weight"
        if g_lora_0 in src_tensors:
            dst_tensors[f"{att}.g1"] = src_tensors[g_lora_0].t().contiguous().clone()
            dst_tensors[f"{att}.g2"] = src_tensors[g_lora_2w].t().contiguous().clone()
        else:
            for name in ["g1", "g2"]:
                src_key = f"{att}.{name}"
                if src_key in src_tensors:
                    copy(src_key)

        # Key modulation — squeeze [1,1,h] or [h]
        for name in ["k_k", "k_a"]:
            src_key = f"{att}.{name}"
            if src_key in src_tensors:
                copy(src_key, squeeze=True)

        # R-K coupling
        rk = f"{att}.r_k"
        if rk in src_tensors:
            copy(rk)

        # GroupNorm — try both naming conventions
        for gn_src, gn_dst in [("g_norm", "g_norm"), ("ln_x", "g_norm")]:
            w_key = f"{att}.{gn_src}.weight"
            b_key = f"{att}.{gn_src}.bias"
            if w_key in src_tensors:
                copy(w_key, f"{att}.g_norm.weight")
                copy(b_key, f"{att}.g_norm.bias")
                break

        # FFN
        ffn_xk = f"{ffn}.x_k"
        if ffn_xk in src_tensors:
            copy(ffn_xk, squeeze=True)
        copy(f"{ffn}.key.weight")
        copy(f"{ffn}.value.weight")

    # Save
    out_path = os.path.join(dst_dir, "model.safetensors")
    save_file(dst_tensors, out_path)
    print(f"Saved {len(dst_tensors)} tensors to {out_path}")
    print(f"Size: {os.path.getsize(out_path) / 1e6:.1f} MB")

    # Save updated config
    config["num_heads"] = num_heads
    config["model_type"] = "rwkv7"
    if config.get("num_heads") is None:
        config["num_heads"] = num_heads
    with open(os.path.join(dst_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    # Copy tokenizer files
    for fname in ["tokenizer_config.json", "special_tokens_map.json",
                   "added_tokens.json", "rwkv_vocab_v20230424.txt",
                   "hf_rwkv_tokenizer.py"]:
        src = os.path.join(src_dir, fname)
        if os.path.exists(src):
            shutil.copy2(src, dst_dir)

    print("Done!")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} <src_dir> <dst_dir>")
        sys.exit(1)
    convert(sys.argv[1], sys.argv[2])
