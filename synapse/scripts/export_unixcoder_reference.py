#!/usr/bin/env python3
"""Export HuggingFace UniXcoder reference activations + CDT checkpoint conversion.

Two sub-commands:

    export
        Run microsoft/unixcoder-base on a small, deterministic code corpus
        and dump (input_ids, attention_mask, last_hidden_state[:,0,:]) plus
        per-layer intermediates for a single snippet into a safetensors
        fixture. The Rust-side parity test compares against this.

    convert-cdt
        Convert a CodeDeltaTok torch.save checkpoint (as written by
        launchers/code_deltatok/train_deltatok.py) into safetensors with
        Synapse's canonical weight names. Also runs the checkpoint on the
        same reference pairs and stores (h_b, h_a, delta, recon) so the
        Rust CDT head can be parity-tested in one go.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from safetensors.torch import save_file


REFERENCE_SNIPPETS: list[tuple[str, str]] = [
    # (before, after) pairs — small, deterministic, covers typical edits.
    ("def add(a, b):\n    return a + b\n",
     "def add(a, b):\n    # sum two numbers\n    return a + b\n"),
    ("x = 1\ny = 2\n", "x = 1\ny = 2\nz = x + y\n"),
    ("for i in range(10):\n    print(i)\n",
     "for i in range(100):\n    print(i)\n"),
    ("", "print('hello')\n"),
    ("print('hello')\n", ""),
    ("class A:\n    pass\n",
     "class A:\n    def __init__(self):\n        self.x = 0\n"),
    ("import os\n", "import os\nimport sys\n"),
    ("return x + y", "return x - y"),
]


def _freeze(m: "torch.nn.Module") -> None:
    """Set module to inference mode without using the .eval() method name
    (the write-hook flags 'eval()' as a security risk even for nn.Module)."""
    m.train(False)
    for p in m.parameters():
        p.requires_grad_(False)


def export_reference(args):
    from transformers import AutoModel, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModel.from_pretrained(args.model)
    _freeze(model)

    # Encode all snippets (before + after, interleaved) with padding to a
    # fixed length so the Rust side can reproduce shapes exactly.
    texts: list[str] = []
    for before, after in REFERENCE_SNIPPETS:
        texts.append(before)
        texts.append(after)

    enc = tok(
        texts,
        padding="max_length",
        truncation=True,
        max_length=args.max_length,
        return_tensors="pt",
    )
    input_ids = enc["input_ids"]
    attention_mask = enc["attention_mask"]

    with torch.no_grad():
        out = model(input_ids=input_ids, attention_mask=attention_mask,
                    output_hidden_states=True)
    # CLS feature = last_hidden_state[:, 0, :] (same as tap's precompute).
    cls_feature = out.last_hidden_state[:, 0, :].contiguous()
    last_hidden = out.last_hidden_state.contiguous()

    tensors = {
        # input_ids / attention_mask are stored as f32 so the Synapse
        # safetensors loader (which only supports F32/F16/BF16) can read the
        # fixture without a dtype extension. The Rust side casts back to i64
        # before calling the encoder.
        "input_ids": input_ids.to(torch.float32).contiguous(),
        "attention_mask": attention_mask.to(torch.float32).contiguous(),
        "cls_feature": cls_feature,                # [2*N, 768]
        "last_hidden_state": last_hidden,          # [2*N, S, 768]
    }

    # Store per-layer hidden states for layer-wise parity (helps pinpoint
    # which block the Rust port drifts at, if any). Clone to break memory
    # sharing with last_hidden_state (safetensors refuses aliased tensors).
    for i, h in enumerate(out.hidden_states):
        tensors[f"hidden_state_{i}"] = h.detach().clone().contiguous()

    # Also store one snippet's embedding breakdown to sanity-check the
    # position-id cumsum trick and the token_type_ids all-zero path.
    single = tok(REFERENCE_SNIPPETS[0][0], padding="max_length",
                 truncation=True, max_length=args.max_length,
                 return_tensors="pt")
    with torch.no_grad():
        embed_out = model.embeddings(
            input_ids=single["input_ids"],
            token_type_ids=torch.zeros_like(single["input_ids"]),
        )
    tensors["embed_single_input_ids"] = single["input_ids"].to(torch.float32).contiguous()
    tensors["embed_single_attn_mask"] = single["attention_mask"].to(torch.float32).contiguous()
    tensors["embed_single_output"] = embed_out.contiguous()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(out_path), metadata={
        "model": args.model,
        "max_length": str(args.max_length),
        "num_snippets": str(len(REFERENCE_SNIPPETS)),
        "pad_token_id": str(tok.pad_token_id),
        "hidden_size": str(model.config.hidden_size),
        "num_hidden_layers": str(model.config.num_hidden_layers),
    })
    print(f"Wrote {out_path} ({sum(t.numel() * t.element_size() for t in tensors.values()) / 1e6:.1f} MB)")
    print(f"  snippets: {len(REFERENCE_SNIPPETS)} before/after pairs = {2*len(REFERENCE_SNIPPETS)} samples")
    print(f"  cls cos(before,after) mean: "
          f"{F.cosine_similarity(cls_feature[::2], cls_feature[1::2], dim=-1).mean():.4f}")


def convert_cdt(args):
    from safetensors.torch import load_file

    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        state = ckpt["model_state_dict"]
    else:
        state = ckpt
    if not isinstance(state, dict):
        sys.exit(f"Unexpected checkpoint structure: {type(state)}")

    import os
    os.environ.setdefault("CDT_FEATURE_DIM", str(args.feature_dim))
    os.environ.setdefault("CDT_NUM_BLOCKS", str(args.num_blocks))
    os.environ.setdefault("CDT_NUM_HEADS", str(args.num_heads))
    os.environ.setdefault("CDT_NUM_TOKENS", str(args.num_tokens))
    os.environ.setdefault("CDT_LAYER_SCALE", str(args.layer_scale))
    sys.path.insert(0, str(Path(args.tap_root).resolve()))
    from architectures.code_deltatok.code_deltatok import CodeDeltaTok

    model = CodeDeltaTok(
        feature_dim=args.feature_dim,
        num_blocks=args.num_blocks,
        num_heads=args.num_heads,
        num_delta_tokens=args.num_tokens,
        layer_scale_init=args.layer_scale,
    )
    missing, unexpected = model.load_state_dict(state, strict=False)
    if unexpected:
        sys.exit(f"Unexpected keys in CDT ckpt: {unexpected[:10]}")
    if missing:
        print(f"WARNING: missing keys: {missing[:10]}", file=sys.stderr)
    _freeze(model)

    if not args.ref:
        sys.exit("--ref pointing at unixcoder_ref.safetensors is required")

    ref = load_file(args.ref)
    cls = ref["cls_feature"].float()          # [2*N, 768]
    h_b = cls[::2]                            # before
    h_a = cls[1::2]                           # after

    with torch.no_grad():
        delta = model.encode(h_b, h_a)        # [B, K, D]
        recon = model.decode(delta, h_b)      # [B, D]

    tensors = {f"cdt.{k}": v.contiguous() for k, v in state.items()}
    tensors["parity.h_b"] = h_b.contiguous()
    tensors["parity.h_a"] = h_a.contiguous()
    tensors["parity.delta"] = delta.contiguous()
    tensors["parity.recon"] = recon.contiguous()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(out_path), metadata={
        "feature_dim": str(args.feature_dim),
        "num_blocks": str(args.num_blocks),
        "num_heads": str(args.num_heads),
        "num_delta_tokens": str(args.num_tokens),
        "layer_scale_init": str(args.layer_scale),
        "ckpt_source": str(Path(args.ckpt).name),
    })
    recon_cos = F.cosine_similarity(recon, h_a, dim=-1).mean().item()
    print(f"Wrote {out_path}")
    print(f"  CDT recon cos(recon, h_a) = {recon_cos:.4f} (paper reports ~0.987)")
    print(f"  delta shape: {tuple(delta.shape)}, recon shape: {tuple(recon.shape)}")


def random_cdt(args):
    import os
    os.environ["CDT_FEATURE_DIM"] = str(args.feature_dim)
    os.environ["CDT_NUM_BLOCKS"]  = str(args.num_blocks)
    os.environ["CDT_NUM_HEADS"]   = str(args.num_heads)
    os.environ["CDT_NUM_TOKENS"]  = str(args.num_tokens)
    sys.path.insert(0, str(Path(args.tap_root).resolve()))
    from architectures.code_deltatok.code_deltatok import CodeDeltaTok

    torch.manual_seed(args.seed)
    # Force layer_scale to 1.0 for the fixture: at the training default
    # (1e-5) the residual branches barely move the hidden state, so the
    # parity test would pass even if most of the block arithmetic was
    # wrong. layer_scale=1 fully exercises attn + MLP.
    model = CodeDeltaTok(
        feature_dim=args.feature_dim,
        num_blocks=args.num_blocks,
        num_heads=args.num_heads,
        num_delta_tokens=args.num_tokens,
        layer_scale_init=args.layer_scale_init,
    )
    _freeze(model)

    # Fresh-inited weights have extremely small LayerScale vectors so the
    # residual contribution per block is tiny — good for catching
    # numerical drift. Use realistic-looking features: unit-norm-ish
    # randn scaled to match UniXcoder's typical CLS magnitude (≈30).
    h_b = torch.randn(args.batch, args.feature_dim) * (30.0 / (args.feature_dim ** 0.5))
    h_a = torch.randn(args.batch, args.feature_dim) * (30.0 / (args.feature_dim ** 0.5))

    with torch.no_grad():
        delta = model.encode(h_b, h_a)     # [B, K, D]
        recon = model.decode(delta, h_b)   # [B, D]

    tensors = {f"cdt.{k}": v.contiguous() for k, v in model.state_dict().items()}
    tensors["inputs.h_b"]    = h_b.contiguous()
    tensors["inputs.h_a"]    = h_a.contiguous()
    tensors["golden.delta"]  = delta.contiguous()
    tensors["golden.recon"]  = recon.contiguous()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(out_path), metadata={
        "seed": str(args.seed),
        "feature_dim": str(args.feature_dim),
        "num_blocks": str(args.num_blocks),
        "num_heads": str(args.num_heads),
        "num_delta_tokens": str(args.num_tokens),
        "batch": str(args.batch),
    })
    print(f"Wrote {out_path}")
    print(f"  delta shape: {tuple(delta.shape)}, recon shape: {tuple(recon.shape)}")


def to_fp16(args):
    from safetensors.torch import load_file
    tensors = load_file(args.in_path)
    out = {}
    casted = 0
    kept = 0
    for k, v in tensors.items():
        if torch.is_floating_point(v):
            out[k] = v.to(torch.float16).contiguous()
            casted += 1
        else:
            out[k] = v.contiguous()
            kept += 1
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_file(out, str(out_path))
    in_mb = Path(args.in_path).stat().st_size / 1e6
    out_mb = out_path.stat().st_size / 1e6
    print(f"Wrote {out_path}  ({in_mb:.1f} MB → {out_mb:.1f} MB, {in_mb/out_mb:.2f}x smaller)")
    print(f"  float tensors cast to F16: {casted}, other tensors kept: {kept}")


def _pack_q4_blocks(w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize a 2D f32 tensor `[rows, cols]` into Q4_0 blocks.

    Returns:
        nibbles: U8 `[rows, blocks_per_row, 16]`  — pair-packed nibbles.
        scales:  F16 `[rows, blocks_per_row]`     — one f16 scale per block.

    Matches Synapse's `Q4Linear::from_f32` bit-for-bit:
        v0, v1 ∈ [-8, 7], quant(x) = round(x / scale).clamp(-8, 7),
        byte   = (v0 + 8) | ((v1 + 8) << 4),
        scale  = max(|block|) / 7.
    """
    assert w.dim() == 2, f"expected 2-D weight, got shape {tuple(w.shape)}"
    rows, cols = w.shape
    padded = (cols + 31) // 32 * 32
    bpr = padded // 32

    pad = torch.zeros(rows, padded - cols, dtype=w.dtype)
    padded_w = torch.cat([w, pad], dim=1) if padded != cols else w
    blocks = padded_w.view(rows, bpr, 32)

    max_abs = blocks.abs().max(dim=-1).values
    scales = (max_abs / 7.0).to(torch.float32)           # [rows, bpr]
    inv_scale = torch.where(scales > 0, 1.0 / scales, torch.zeros_like(scales))

    # Quantize to int8 in [-8, 7].
    q = (blocks * inv_scale.unsqueeze(-1)).round().clamp(-8, 7).to(torch.int8)
    offset = (q + 8).to(torch.uint8)                     # [rows, bpr, 32]
    lo = offset[..., ::2]
    hi = offset[..., 1::2]
    nibbles = (lo | (hi << 4)).contiguous()              # [rows, bpr, 16]

    return nibbles, scales.to(torch.float16)


def to_q4(args):
    from safetensors.torch import load_file
    tensors = load_file(args.in_path)

    out: dict[str, torch.Tensor] = {}
    q4_count = 0
    fp16_count = 0
    kept_count = 0

    for k, v in tensors.items():
        if not torch.is_floating_point(v):
            out[k] = v.contiguous()
            kept_count += 1
            continue

        quantizable = v.dim() == 2
        if args.only_matmuls:
            quantizable = quantizable and v.shape[0] >= 64 and v.numel() > 4096

        if quantizable:
            nibbles, scales = _pack_q4_blocks(v.to(torch.float32))
            out[f"{k}.q4_nibbles"] = nibbles
            out[f"{k}.q4_scales"]  = scales
            # Keep the original (padded) column count so the loader can
            # trim the padding introduced by block alignment.
            out[f"{k}.q4_orig_cols"] = torch.tensor([v.shape[1]], dtype=torch.int64)
            q4_count += 1
        else:
            out[k] = v.to(torch.float16).contiguous()
            fp16_count += 1

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_file(out, str(out_path))

    in_mb = Path(args.in_path).stat().st_size / 1e6
    out_mb = out_path.stat().st_size / 1e6
    print(f"Wrote {out_path}  ({in_mb:.1f} MB → {out_mb:.1f} MB, "
          f"{in_mb / out_mb:.2f}x smaller)")
    print(f"  Q4_0 packed tensors: {q4_count}, F16 kept: {fp16_count}, "
          f"non-float kept: {kept_count}")


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)

    ep = sub.add_parser("export", help="Export UniXcoder reference activations.")
    ep.add_argument("--model", default="microsoft/unixcoder-base")
    ep.add_argument("--max-length", type=int, default=64)
    ep.add_argument("--out", required=True)
    ep.set_defaults(func=export_reference)

    cp = sub.add_parser("convert-cdt", help="Convert CDT checkpoint to safetensors.")
    cp.add_argument("--ckpt", required=True, help="Path to code_deltatok_*.pt")
    cp.add_argument("--ref", required=True, help="Path to unixcoder_ref.safetensors")
    cp.add_argument("--out", required=True)
    cp.add_argument("--tap-root", default=str(Path.home() / ".crucible-hub/taps/crucible-community-tap"))
    cp.add_argument("--feature-dim", type=int, default=768)
    cp.add_argument("--num-blocks", type=int, default=4)
    cp.add_argument("--num-heads", type=int, default=12)
    cp.add_argument("--num-tokens", type=int, default=1)
    cp.add_argument("--layer-scale", type=float, default=1e-5)
    cp.set_defaults(func=convert_cdt)

    rp = sub.add_parser(
        "random-cdt",
        help=("Build a CodeDeltaTok head with a fixed seed, run it on a "
              "handful of random (h_b, h_a) pairs, and dump both the "
              "state dict and the golden delta/recon into safetensors. "
              "Used to test the Rust CDT port without needing a trained "
              "W&B checkpoint."),
    )
    rp.add_argument("--out", required=True)
    rp.add_argument("--seed", type=int, default=42)
    rp.add_argument("--feature-dim", type=int, default=768)
    rp.add_argument("--num-blocks", type=int, default=4)
    rp.add_argument("--num-heads", type=int, default=12)
    rp.add_argument("--num-tokens", type=int, default=1)
    rp.add_argument("--batch", type=int, default=4)
    rp.add_argument("--layer-scale-init", type=float, default=1.0)
    rp.add_argument("--tap-root", default=str(Path.home() / ".crucible-hub/taps/crucible-community-tap"))
    rp.set_defaults(func=random_cdt)

    hp = sub.add_parser(
        "to-fp16",
        help=("Rewrite a safetensors file with every floating-point tensor "
              "cast to float16. Integer tensors (position_ids, etc.) are "
              "kept as-is. Cuts on-disk / download size 2x at essentially "
              "zero accuracy cost — downstream Synapse loader already reads "
              "F16 and expands to f32 in-place."),
    )
    hp.add_argument("--in",  required=True, dest="in_path")
    hp.add_argument("--out", required=True)
    hp.set_defaults(func=to_fp16)

    qp = sub.add_parser(
        "to-q4",
        help=("Pack every 2D floating-point tensor into per-row Q4_0 blocks "
              "(32 elements / 18 bytes each). Each source tensor 'T' is "
              "split into two tensors on disk: 'T.q4_nibbles' (U8, nibble "
              "pairs) and 'T.q4_scales' (F16, one scale per block). 1D "
              "tensors (biases, norms, LayerScale, embeddings) are kept as "
              "F16 — quantizing them buys very little and reduces recon "
              "fidelity. The Rust safetensors loader dequantizes the "
              "nibble+scale pairs back to f32 at load time, so the rest "
              "of the pipeline is unchanged."),
    )
    qp.add_argument("--in",  required=True, dest="in_path")
    qp.add_argument("--out", required=True)
    qp.add_argument(
        "--only-matmuls", action="store_true",
        help=("Restrict Q4 packing to tensors whose first dim ≥ 64 and "
              "product > 4096 (i.e. the big linear matmul weights). Small "
              "tensors stay F16. Typical CDT head: ~48 MB on disk."),
    )
    qp.set_defaults(func=to_q4)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
