#!/usr/bin/env python3
"""Convert Code WM .pt checkpoint to safetensors + config.json for Synapse.

Filters out training-only tensors (target_encoder, pred_projector, target_projector)
and keeps: state_encoder.*, action_encoder.*, predictor.*. The sinusoidal PE
buffer (state_encoder.pos_enc.pe) is included by default since it's a registered
buffer — avoids recomputing sin/cos in Rust and prevents log(10000)/exp drift.

Usage:
    python scripts/convert_code_wm_ckpt.py \
        /tmp/synapse_codewm_package/g8_sigreg_dir.pt \
        --out-weights synapse/models/code_wm/g8.safetensors \
        --out-config synapse/configs/code_wm_g8.json

    # Batch mode: convert both g8 and g1b
    python scripts/convert_code_wm_ckpt.py --batch /tmp/synapse_codewm_package \
        --out-dir synapse/models/code_wm
"""

import argparse
import json
import os
from pathlib import Path

import torch
from safetensors.torch import save_file


# Prefixes that are needed at inference time.
KEEP_PREFIXES = (
    "state_encoder.",
    "action_encoder.",
    "predictor.",
)

# Training-only: EMA target, projectors with batchnorm state.
DROP_PREFIXES = (
    "target_encoder.",
    "target_predictor.",
    "target_projector.",
    "pred_projector.",
)


def convert(input_path: str, out_weights: str, out_config: str) -> dict:
    print(f"Loading {input_path} ...")
    ckpt = torch.load(input_path, map_location="cpu", weights_only=False)
    if "model_state_dict" in ckpt:
        sd = ckpt["model_state_dict"]
    elif "state_dict" in ckpt:
        sd = ckpt["state_dict"]
    else:
        raise ValueError(f"No model_state_dict or state_dict in {input_path}")

    ckpt_config = ckpt.get("config", {})
    print(f"  Checkpoint config: {ckpt_config}")
    print(f"  state_dict has {len(sd)} tensors total")

    tensors = {}
    dropped = []
    for key, tensor in sd.items():
        if any(key.startswith(p) for p in DROP_PREFIXES):
            dropped.append(key)
            continue
        if not any(key.startswith(p) for p in KEEP_PREFIXES):
            dropped.append(key)
            continue
        # Coerce to float32 (safetensors wants contiguous tensors).
        if tensor.dtype != torch.float32:
            tensor = tensor.float()
        tensors[key] = tensor.contiguous()

    print(f"  Kept {len(tensors)} tensors, dropped {len(dropped)}")
    if len(dropped) < 20:
        for k in dropped:
            print(f"    drop: {k}")

    total_params = sum(t.numel() for t in tensors.values())
    print(f"  Total inference params: {total_params:,} ({total_params * 4 / 1024:.1f} KB f32)")

    # Verify expected tensors are present.
    required = [
        "state_encoder.embedding.weight",
        "state_encoder.cls_token",
        "state_encoder.pos_enc.pe",
        "state_encoder.block.norm1.weight",
        "state_encoder.block.norm1.bias",
        "state_encoder.block.attn.in_proj_weight",
        "state_encoder.block.attn.in_proj_bias",
        "state_encoder.block.attn.out_proj.weight",
        "state_encoder.block.attn.out_proj.bias",
        "state_encoder.block.norm2.weight",
        "state_encoder.block.norm2.bias",
        "state_encoder.block.mlp.0.weight",
        "state_encoder.block.mlp.0.bias",
        "state_encoder.block.mlp.3.weight",
        "state_encoder.block.mlp.3.bias",
        "state_encoder.norm.weight",
        "state_encoder.norm.bias",
        "action_encoder.net.0.weight",
        "action_encoder.net.0.bias",
        "action_encoder.net.2.weight",
        "action_encoder.net.2.bias",
        "predictor.norm.weight",
        "predictor.norm.bias",
    ]
    for pred_block in (0, 1):
        for part in (
            "norm1.weight", "norm1.bias",
            "attn.in_proj_weight", "attn.in_proj_bias",
            "attn.out_proj.weight", "attn.out_proj.bias",
            "norm2.weight", "norm2.bias",
            "mlp.0.weight", "mlp.0.bias",
            "mlp.3.weight", "mlp.3.bias",
        ):
            required.append(f"predictor.blocks.{pred_block}.{part}")

    missing = [k for k in required if k not in tensors]
    if missing:
        raise ValueError(f"Missing required tensors: {missing}")
    # Auto-detect pool mode from the kept state_dict. Checkpoints trained with
    # WM_POOL_MODE=attn have `state_encoder.attn_pool.*` keys. The 5 tensors
    # are the learnable query + a fused nn.MultiheadAttention(num_heads=1).
    has_attn_pool = any(k.startswith("state_encoder.attn_pool.") for k in tensors)
    pool_mode = "attn" if has_attn_pool else "cls"
    if has_attn_pool:
        required.extend([
            "state_encoder.attn_pool.query",
            "state_encoder.attn_pool.attn.in_proj_weight",
            "state_encoder.attn_pool.attn.in_proj_bias",
            "state_encoder.attn_pool.attn.out_proj.weight",
            "state_encoder.attn_pool.attn.out_proj.bias",
        ])
        missing = [k for k in required if k not in tensors]
        if missing:
            raise ValueError(f"attn_pool required but missing: {missing}")

    print(f"  All {len(required)} required tensors present.")
    print(f"  Pool mode: {pool_mode}")

    # Derive Synapse config JSON. Checkpoint uses 'num_loops' for predictor loops.
    # Auto-derive dimensions from tensor shapes as fallback when checkpoint
    # config is incomplete (robust for v1 vocab=662 AND v2 vocab=700).
    emb_tensor = tensors.get("state_encoder.embedding.weight")
    act_tensor = tensors.get("action_encoder.net.0.weight")
    derived_vocab = emb_tensor.shape[0] if emb_tensor is not None else None
    derived_action = act_tensor.shape[1] if act_tensor is not None else None

    model_dim = ckpt_config.get("model_dim", 128)
    num_heads = ckpt_config.get("num_heads", 4)

    vocab_size = ckpt_config.get("vocab_size") or derived_vocab or 662
    action_dim = ckpt_config.get("action_dim") or derived_action or 7

    if derived_vocab and vocab_size != derived_vocab:
        print(f"  WARNING: config vocab_size={vocab_size} != embedding shape[0]={derived_vocab}")
    if derived_action and action_dim != derived_action:
        print(f"  WARNING: config action_dim={action_dim} != action_fc1 shape[1]={derived_action}")

    synapse_config = {
        "vocab_size": vocab_size,
        "max_seq_len": ckpt_config.get("max_seq_len", 512),
        "model_dim": model_dim,
        "num_heads": num_heads,
        "head_dim": model_dim // num_heads,
        "mlp_hidden": int(model_dim * 4),  # mlp_ratio=4.0 is hard-coded in the model
        "encoder_loops": ckpt_config.get("encoder_loops", 6),
        "predictor_depth": 2,  # 2 distinct blocks, hard-coded in LoopedPredictor
        "predictor_loops": ckpt_config.get("num_loops", 6),
        "action_dim": action_dim,
        "layernorm_eps": 1.0e-5,
        "gelu_kind": "erf",  # nn.GELU() default is approximate='none' (exact erf)
        "pool_mode": pool_mode,
    }

    # ── Export target encoder (for transition prediction demos) ──
    # The predictor outputs in TARGET encoder space. To evaluate prediction
    # quality, we need the target encoder to encode the "after" reference.
    # We remap target_encoder.* → state_encoder.* so a second CodeWorldModel
    # instance can load it with the standard weight-loading code.
    target_tensors = {}
    for key, tensor in sd.items():
        if key.startswith("target_encoder."):
            remapped = key.replace("target_encoder.", "state_encoder.", 1)
            if tensor.dtype != torch.float32:
                tensor = tensor.float()
            target_tensors[remapped] = tensor.contiguous()

    # Write outputs.
    os.makedirs(os.path.dirname(out_weights) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(out_config) or ".", exist_ok=True)
    save_file(tensors, out_weights)
    with open(out_config, "w") as f:
        json.dump(synapse_config, f, indent=2)

    print(f"  Wrote {out_weights} ({os.path.getsize(out_weights) / 1024:.1f} KB)")
    print(f"  Wrote {out_config}")

    # Write target encoder weights if present (for prediction demos).
    if target_tensors:
        target_path = out_weights.replace(".safetensors", "_target.safetensors")
        save_file(target_tensors, target_path)
        target_params = sum(t.numel() for t in target_tensors.values())
        print(f"  Wrote {target_path} ({os.path.getsize(target_path) / 1024:.1f} KB) — {len(target_tensors)} target encoder tensors")
    else:
        print("  No target_encoder.* in checkpoint — skipping target export")

    return synapse_config


def main():
    parser = argparse.ArgumentParser(description="Convert Code WM .pt to safetensors + config.json")
    parser.add_argument("input", nargs="?", help="Input .pt checkpoint")
    parser.add_argument("--out-weights", help="Output .safetensors path")
    parser.add_argument("--out-config", help="Output config.json path")
    parser.add_argument("--batch", help="Directory containing .pt files for batch conversion")
    parser.add_argument("--out-dir", help="Output directory for batch mode")
    args = parser.parse_args()

    if args.batch:
        # Batch mode writes relative to the synapse/ repo root.
        # Weights land in models/code_wm/, configs in configs/.
        out_dir = args.out_dir or "models/code_wm"
        config_dir = "configs"
        os.makedirs(out_dir, exist_ok=True)
        os.makedirs(config_dir, exist_ok=True)
        pt_files = sorted(Path(args.batch).glob("*.pt"))
        if not pt_files:
            raise SystemExit(f"No .pt files in {args.batch}")
        for pt in pt_files:
            name = pt.stem.replace("_sigreg_dir", "").replace("_vicreg", "")
            # e.g. g8_sigreg_dir.pt -> g8; g1b_vicreg.pt -> g1b
            out_w = os.path.join(out_dir, f"{name}.safetensors")
            out_c = os.path.join(config_dir, f"code_wm_{name}.json")
            print(f"\n=== {pt.name} -> {name} ===")
            convert(str(pt), out_w, out_c)
    else:
        if not args.input or not args.out_weights or not args.out_config:
            parser.print_help()
            raise SystemExit(2)
        convert(args.input, args.out_weights, args.out_config)


if __name__ == "__main__":
    main()
