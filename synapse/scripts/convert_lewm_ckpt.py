#!/usr/bin/env python3
"""Convert crucible LEWM .ckpt object files to safetensors + config.json.

Handles PyTorch pickled model objects without requiring the `jepa` or `module`
packages by using a stub unpickler. Auto-detects model config from weight shapes.

NOTE: This script uses pickle to load PyTorch checkpoint files. Only use with
trusted checkpoint files from known sources (e.g., your own W&B artifacts).

Usage:
    # Single checkpoint
    python3 convert_lewm_ckpt.py --input model.ckpt --output /tmp/lewm-slim/

    # Batch: all .ckpt files in a directory
    python3 convert_lewm_ckpt.py --input-dir ~/Downloads/ --output-dir /tmp/lewm-variants/
"""

import argparse
import json
import os
import pickle
import sys
import types
from pathlib import Path

import torch
from safetensors.torch import save_file


# ---------------------------------------------------------------------------
# Stub unpickler: loads PyTorch model objects without source packages
# ---------------------------------------------------------------------------

class FlexClass:
    """Stub replacement for missing nn.Module subclasses."""

    def __init__(self, *args, **kwargs):
        pass

    def __setstate__(self, state):
        if isinstance(state, dict):
            self.__dict__.update(state)

    def __reduce_ex__(self, protocol):
        return (FlexClass, ())


class StubUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        try:
            return super().find_class(module, name)
        except (ModuleNotFoundError, AttributeError):
            if module not in sys.modules:
                sys.modules[module] = types.ModuleType(module)
            return FlexClass


def load_ckpt(path: str):
    """Load a .ckpt file with stub unpickler."""
    return torch.load(
        path,
        map_location="cpu",
        weights_only=False,
        pickle_module=type(
            "M",
            (),
            {
                "Unpickler": StubUnpickler,
                "load": pickle.load,
                "dump": pickle.dump,
                "dumps": pickle.dumps,
                "loads": pickle.loads,
                "PicklingError": pickle.PicklingError,
                "UnpicklingError": pickle.UnpicklingError,
            },
        ),
    )


# ---------------------------------------------------------------------------
# Tensor extraction: walk nn.Module hierarchy
# ---------------------------------------------------------------------------

def extract_tensors(obj, prefix="", depth=0, visited=None):
    """Recursively extract all tensors from a reconstructed nn.Module tree."""
    if visited is None:
        visited = set()
    if id(obj) in visited or depth > 15:
        return {}
    visited.add(id(obj))
    tensors = {}

    if isinstance(obj, torch.Tensor):
        tensors[prefix or "tensor"] = obj
    elif isinstance(obj, torch.nn.Parameter):
        tensors[prefix or "param"] = obj.data
    elif isinstance(obj, dict):
        for k, v in obj.items():
            p = f"{prefix}.{k}" if prefix else str(k)
            tensors.update(extract_tensors(v, p, depth + 1, visited))
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            tensors.update(extract_tensors(v, f"{prefix}[{i}]", depth + 1, visited))
    elif hasattr(obj, "__dict__"):
        for k, v in obj.__dict__.items():
            p = f"{prefix}.{k}" if prefix else k
            tensors.update(extract_tensors(v, p, depth + 1, visited))

    for attr in ("_modules", "_parameters", "_buffers"):
        d = getattr(obj, attr, None)
        if isinstance(d, dict):
            for k, v in d.items():
                p = f"{prefix}.{k}" if prefix else k
                tensors.update(extract_tensors(v, p, depth + 1, visited))

    return tensors


# ---------------------------------------------------------------------------
# Key remapping: nn.Module hierarchy -> synapse flat keys
# ---------------------------------------------------------------------------

def flatten_key(raw_key: str) -> str:
    """Convert nn.Module hierarchy key to flat checkpoint key."""
    parts = raw_key.split(".")
    flat = []
    for part in parts:
        if part in ("_modules", "_parameters", "_buffers"):
            continue
        flat.append(part)
    return ".".join(flat)


def remap_key(flat_key: str):
    """Remap a flat key to synapse naming. Returns None to skip."""
    if "num_batches_tracked" in flat_key:
        return None
    return flat_key


def remap_all(raw_tensors):
    """Flatten and remap all tensor keys."""
    result = {}
    skipped = []
    for raw_key, tensor in sorted(raw_tensors.items()):
        flat = flatten_key(raw_key)
        mapped = remap_key(flat)
        if mapped is None:
            skipped.append(flat)
            continue
        if tensor.dtype != torch.float32:
            tensor = tensor.float()
        result[mapped] = tensor
    return result, skipped


# ---------------------------------------------------------------------------
# Config inference from weight shapes
# ---------------------------------------------------------------------------

def infer_config(tensors):
    """Infer LeWMConfig from weight tensor shapes."""
    config = {
        "image_size": 224,
        "patch_size": 14,
        "channels": 3,
    }

    cls = tensors.get("encoder.embeddings.cls_token")
    if cls is not None:
        config["encoder_hidden"] = cls.shape[-1]

    enc_layers = set()
    for k in tensors:
        if k.startswith("encoder.encoder.layer."):
            idx = k.split(".")[3]
            if idx.isdigit():
                enc_layers.add(int(idx))
    config["encoder_layers"] = len(enc_layers) if enc_layers else 6

    enc_hidden = config.get("encoder_hidden", 192)
    config["encoder_heads"] = max(1, enc_hidden // 64)

    ffn_key = "encoder.encoder.layer.0.intermediate.dense.weight"
    if ffn_key in tensors:
        config["encoder_inter"] = tensors[ffn_key].shape[0]
    else:
        config["encoder_inter"] = enc_hidden * 4

    pred_norm = tensors.get("predictor.transformer.norm.weight")
    if pred_norm is not None:
        config["predictor_hidden"] = pred_norm.shape[0]

    pred_layers = set()
    for k in tensors:
        if k.startswith("predictor.transformer.layers."):
            idx = k.split(".")[3]
            if idx.isdigit():
                pred_layers.add(int(idx))
    config["predictor_layers"] = len(pred_layers) if pred_layers else 6

    qkv_key = "predictor.transformer.layers.0.attn.to_qkv.weight"
    if qkv_key in tensors:
        config["predictor_inner_dim"] = tensors[qkv_key].shape[0] // 3

    inner = config.get("predictor_inner_dim", 1024)
    config["predictor_heads"] = inner // 64

    mlp_key = "predictor.transformer.layers.0.mlp.net.1.weight"
    if mlp_key in tensors:
        config["predictor_inter"] = tensors[mlp_key].shape[0]

    pos = tensors.get("predictor.pos_embedding")
    if pos is not None:
        config["latent_dim"] = pos.shape[-1]
    else:
        config["latent_dim"] = config.get("predictor_hidden", 192)

    act_w = tensors.get("action_encoder.patch_embed.weight")
    if act_w is not None:
        config["action_dim"] = act_w.shape[0]
    else:
        config["action_dim"] = 10

    config["has_input_proj"] = "predictor.transformer.input_proj.weight" in tensors
    config["has_cond_proj"] = "predictor.transformer.cond_proj.weight" in tensors

    return config


# ---------------------------------------------------------------------------
# Main conversion
# ---------------------------------------------------------------------------

def convert_one(input_path, output_dir):
    """Convert a single .ckpt to safetensors + config.json."""
    print(f"\n{'='*60}")
    print(f"Converting: {input_path}")
    print(f"Output:     {output_dir}/")
    print(f"{'='*60}")

    print("Loading checkpoint...")
    ckpt = load_ckpt(input_path)

    print("Extracting tensors...")
    raw_tensors = extract_tensors(ckpt)
    print(f"  Found {len(raw_tensors)} raw tensors")

    tensors, skipped = remap_all(raw_tensors)
    print(f"  Mapped {len(tensors)} tensors, skipped {len(skipped)}")
    if skipped:
        for s in skipped:
            print(f"    skip: {s}")

    config = infer_config(tensors)
    print(f"\nInferred config:")
    for k, v in sorted(config.items()):
        print(f"  {k}: {v}")

    print(f"\nTensor keys ({len(tensors)}):")
    total_params = 0
    for k in sorted(tensors.keys()):
        t = tensors[k]
        params = t.numel()
        total_params += params
        print(f"  {k}: {list(t.shape)} ({params:,} params)")
    print(f"\nTotal parameters: {total_params:,}")
    print(f"f32 size: {total_params * 4 / 1024 / 1024:.1f} MB")

    os.makedirs(output_dir, exist_ok=True)

    safetensors_path = os.path.join(output_dir, "lejepa_weights.safetensors")
    print(f"\nSaving safetensors to {safetensors_path}...")
    save_file(tensors, safetensors_path)

    config_path = os.path.join(output_dir, "config.json")
    print(f"Saving config to {config_path}...")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    print("Done!")
    return config


def main():
    parser = argparse.ArgumentParser(description="Convert crucible LEWM .ckpt to safetensors")
    parser.add_argument("--input", "-i", help="Single .ckpt file to convert")
    parser.add_argument("--output", "-o", help="Output directory for single conversion")
    parser.add_argument("--input-dir", help="Directory of .ckpt files for batch conversion")
    parser.add_argument("--output-dir", help="Output base directory for batch conversion")
    args = parser.parse_args()

    if args.input:
        output = args.output or "/tmp/lewm-converted/"
        convert_one(args.input, output)
    elif args.input_dir:
        output_base = args.output_dir or "/tmp/lewm-variants/"
        ckpt_files = sorted(Path(args.input_dir).glob("*.ckpt"))
        if not ckpt_files:
            print(f"No .ckpt files found in {args.input_dir}")
            sys.exit(1)
        print(f"Found {len(ckpt_files)} checkpoint files")
        for ckpt in ckpt_files:
            name = ckpt.stem.replace("_object", "").replace("_epoch_1", "")
            out_dir = os.path.join(output_base, name)
            convert_one(str(ckpt), out_dir)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
