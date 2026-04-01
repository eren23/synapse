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
    """Flatten and remap all tensor keys.

    For hybrid encoder checkpoints (encoder.blocks.N.*), remaps to
    standard ViT naming (encoder.encoder.layer.N.*) and splits fused QKV.
    """
    is_hybrid = any(k.startswith("encoder.blocks.") for k in raw_tensors)

    if is_hybrid:
        return _remap_hybrid(raw_tensors)

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


def _remap_hybrid(raw_tensors):
    """Remap hybrid ALAL encoder keys to standard ViT naming.

    Hybrid encoder structure:
      encoder.cls_token           → encoder.embeddings.cls_token
      encoder.meta_token          → (stored as separate key)
      encoder.pos_embed           → encoder.embeddings.position_embeddings
      encoder.patch_embed.proj.*  → encoder.embeddings.patch_embeddings.projection.*
      encoder.norm.*              → encoder.layernorm.*
      encoder.blocks.N.norm1.*    → encoder.encoder.layer.N.layernorm_before.*
      encoder.blocks.N.norm2.*    → encoder.encoder.layer.N.layernorm_after.*
      encoder.blocks.N.mlp.0.*   → encoder.encoder.layer.N.intermediate.dense.*
      encoder.blocks.N.mlp.2.*   → encoder.encoder.layer.N.output.dense.*

    A blocks (fused QKV): in_proj_weight [3H, H] → split into Q/K/V [H, H]
    L blocks (separate Q/K/V): q_proj, k_proj, v_proj → query, key, value

    Both: out_proj → attention.output.dense
    """
    result = {}
    skipped = []

    # Embedding remapping
    EMBED_MAP = {
        "encoder.cls_token": "encoder.embeddings.cls_token",
        "encoder.pos_embed": "encoder.embeddings.position_embeddings",
        "encoder.patch_embed.proj.weight": "encoder.embeddings.patch_embeddings.projection.weight",
        "encoder.patch_embed.proj.bias": "encoder.embeddings.patch_embeddings.projection.bias",
        "encoder.norm.weight": "encoder.layernorm.weight",
        "encoder.norm.bias": "encoder.layernorm.bias",
    }

    for key, tensor in sorted(raw_tensors.items()):
        if "num_batches_tracked" in key:
            skipped.append(key)
            continue
        # Keep running_mean/running_var in safetensors for reference
        # (Rust code will skip them during loading)

        if tensor.dtype != torch.float32:
            tensor = tensor.float()

        # 1. Direct embedding remaps
        if key in EMBED_MAP:
            result[EMBED_MAP[key]] = tensor
            continue

        # 2. Meta token (new for hybrid — store as-is)
        if key == "encoder.meta_token":
            result["encoder.meta_token"] = tensor
            continue

        # 3. Encoder output projection (encoder.proj.*) — collect for BN folding
        if key.startswith("encoder.proj."):
            result[key] = tensor
            continue

        # 4. Encoder blocks
        if key.startswith("encoder.blocks."):
            parts = key.split(".")
            # encoder.blocks.{idx}.{rest...}
            idx = parts[2]
            rest = ".".join(parts[3:])
            prefix = f"encoder.encoder.layer.{idx}"

            # -- Attention --
            if rest == "attn.in_proj_weight":
                # Fused QKV: [3*H, H] → split into Q [H,H], K [H,H], V [H,H]
                h = tensor.shape[0] // 3
                q_w, k_w, v_w = tensor[:h], tensor[h:2*h], tensor[2*h:]
                result[f"{prefix}.attention.attention.query.weight"] = q_w
                result[f"{prefix}.attention.attention.key.weight"] = k_w
                result[f"{prefix}.attention.attention.value.weight"] = v_w
                continue
            if rest == "attn.in_proj_bias":
                h = tensor.shape[0] // 3
                q_b, k_b, v_b = tensor[:h], tensor[h:2*h], tensor[2*h:]
                result[f"{prefix}.attention.attention.query.bias"] = q_b
                result[f"{prefix}.attention.attention.key.bias"] = k_b
                result[f"{prefix}.attention.attention.value.bias"] = v_b
                continue
            if rest == "attn.q_proj.weight":
                result[f"{prefix}.attention.attention.query.weight"] = tensor
                continue
            if rest == "attn.k_proj.weight":
                result[f"{prefix}.attention.attention.key.weight"] = tensor
                continue
            if rest == "attn.v_proj.weight":
                result[f"{prefix}.attention.attention.value.weight"] = tensor
                continue
            if rest == "attn.out_proj.weight":
                result[f"{prefix}.attention.output.dense.weight"] = tensor
                continue
            if rest == "attn.out_proj.bias":
                result[f"{prefix}.attention.output.dense.bias"] = tensor
                continue

            # -- LayerNorm --
            if rest == "norm1.weight":
                result[f"{prefix}.layernorm_before.weight"] = tensor
                continue
            if rest == "norm1.bias":
                result[f"{prefix}.layernorm_before.bias"] = tensor
                continue
            if rest == "norm2.weight":
                result[f"{prefix}.layernorm_after.weight"] = tensor
                continue
            if rest == "norm2.bias":
                result[f"{prefix}.layernorm_after.bias"] = tensor
                continue

            # -- MLP: mlp.0 = up, mlp.2 = down --
            if rest == "mlp.0.weight":
                result[f"{prefix}.intermediate.dense.weight"] = tensor
                continue
            if rest == "mlp.0.bias":
                result[f"{prefix}.intermediate.dense.bias"] = tensor
                continue
            if rest == "mlp.2.weight":
                result[f"{prefix}.output.dense.weight"] = tensor
                continue
            if rest == "mlp.2.bias":
                result[f"{prefix}.output.dense.bias"] = tensor
                continue

        # 5. Non-encoder keys (predictor, action_encoder, projector, pred_proj)
        mapped = remap_key(key)
        if mapped is None:
            skipped.append(key)
        else:
            result[mapped] = tensor

    # Fold encoder.proj BatchNorm into Linear layer
    # BN: y = gamma * (x - mean) / sqrt(var + eps) + beta
    # Folded: new_W = gamma * W / sqrt(var + eps)  (per output channel)
    #         new_b = gamma * (b - mean) / sqrt(var + eps) + beta
    proj_w = result.get("encoder.proj.0.weight")
    proj_b = result.get("encoder.proj.0.bias")
    bn_gamma = result.get("encoder.proj.1.weight")
    bn_beta = result.get("encoder.proj.1.bias")
    bn_mean = result.get("encoder.proj.1.running_mean")
    bn_var = result.get("encoder.proj.1.running_var")

    if proj_w is not None and bn_mean is not None and bn_var is not None:
        eps = 1e-5
        if bn_gamma is None:
            bn_gamma = torch.ones_like(bn_mean)
        if bn_beta is None:
            bn_beta = torch.zeros_like(bn_mean)
        if proj_b is None:
            proj_b = torch.zeros(proj_w.shape[0])

        std = (bn_var + eps).sqrt()
        scale = bn_gamma / std  # [out_features]

        # Fold into weight: each output row scaled by scale[j]
        new_w = proj_w * scale.unsqueeze(1)
        new_b = scale * (proj_b - bn_mean) + bn_beta

        result["encoder.proj.0.weight"] = new_w
        result["encoder.proj.0.bias"] = new_b
        print(f"  Folded encoder.proj BN into Linear (scale L2={scale.norm():.4f})")

        # Remove BN keys from output
        for k in ["encoder.proj.1.weight", "encoder.proj.1.bias",
                   "encoder.proj.1.running_mean", "encoder.proj.1.running_var"]:
            result.pop(k, None)
    else:
        # Clean up BN keys if no folding needed
        for k in list(result.keys()):
            if k.startswith("encoder.proj.1."):
                result.pop(k)

    return result, skipped


# ---------------------------------------------------------------------------
# Config inference from weight shapes
# ---------------------------------------------------------------------------

def infer_config(tensors, encoder_type="auto"):
    """Infer LeWMConfig from weight tensor shapes.

    Handles two encoder styles:
      - "vit": Standard ViT (encoder.embeddings.*, encoder.encoder.layer.*)
      - "hybrid": Custom blocks — keys are already remapped to ViT naming
                  but encoder_hidden differs (64d vs 192d)

    After remapping, both styles use the same key patterns.
    Pass encoder_type="hybrid" to mark it; otherwise auto-detected.
    """
    config = {
        "image_size": 224,
        "patch_size": 14,
        "channels": 3,
    }

    # --- Detect encoder style ---
    is_hybrid = (encoder_type == "hybrid") or any(
        k.startswith("encoder.blocks.") for k in tensors
    )

    if is_hybrid:
        config["encoder_type"] = "hybrid"
        # After remapping, hybrid keys use ViT naming
        cls = tensors.get("encoder.embeddings.cls_token")
        if cls is not None:
            config["encoder_hidden"] = cls.shape[-1]

        enc_layers = set()
        for k in tensors:
            if k.startswith("encoder.encoder.layer."):
                idx = k.split(".")[3]
                if idx.isdigit():
                    enc_layers.add(int(idx))
        config["encoder_layers"] = len(enc_layers) if enc_layers else 4

        enc_hidden = config.get("encoder_hidden", 64)
        config["encoder_heads"] = max(1, enc_hidden // 64)

        ffn_key = "encoder.encoder.layer.0.intermediate.dense.weight"
        if ffn_key in tensors:
            config["encoder_inter"] = tensors[ffn_key].shape[0]
        else:
            config["encoder_inter"] = enc_hidden * 4

        # Check for meta_token (hybrid-specific)
        meta = tensors.get("encoder.meta_token")
        if meta is not None:
            config["meta_tokens"] = meta.shape[1]  # e.g., 4
    else:
        config["encoder_type"] = "vit"
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

    # --- Predictor (same for both encoder styles) ---
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

def extract_state_dict(ckpt):
    """Extract model state_dict from a checkpoint.

    Handles Lightning checkpoints (with 'state_dict' key containing
    'model.*' prefixed keys) and plain state_dict checkpoints.
    Returns a flat dict of {key: tensor}.
    """
    # Lightning checkpoint: has top-level 'state_dict' key
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        sd = ckpt["state_dict"]
        if isinstance(sd, dict):
            result = {}
            for k, v in sd.items():
                if not isinstance(k, str):
                    continue
                if not isinstance(v, torch.Tensor):
                    continue
                # Strip 'model.' prefix if present
                key = k[len("model."):] if k.startswith("model.") else k
                result[key] = v
            if result:
                return result

    # Fallback: walk the whole object tree (original behavior)
    return None


def convert_one(input_path, output_dir):
    """Convert a single .ckpt to safetensors + config.json."""
    print(f"\n{'='*60}")
    print(f"Converting: {input_path}")
    print(f"Output:     {output_dir}/")
    print(f"{'='*60}")

    print("Loading checkpoint...")
    ckpt = load_ckpt(input_path)

    # Try Lightning state_dict extraction first
    sd_tensors = extract_state_dict(ckpt)
    if sd_tensors is not None:
        print(f"  Lightning checkpoint detected, extracted {len(sd_tensors)} model tensors")
        raw_tensors = sd_tensors
    else:
        print("Extracting tensors (full tree walk)...")
        raw_tensors = extract_tensors(ckpt)
    print(f"  Found {len(raw_tensors)} raw tensors")

    # Detect hybrid before remapping (remapping changes the keys)
    is_hybrid = any(k.startswith("encoder.blocks.") for k in raw_tensors)

    tensors, skipped = remap_all(raw_tensors)
    print(f"  Mapped {len(tensors)} tensors, skipped {len(skipped)}")
    if skipped:
        for s in skipped:
            print(f"    skip: {s}")

    config = infer_config(tensors, encoder_type="hybrid" if is_hybrid else "vit")
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
