#!/usr/bin/env python3
"""PyTorch reference dump for Code WM zero-drift validation.

Runs the PyTorch CodeWorldModel forward pass with a shadow implementation
that captures every intermediate activation (per encoder loop, per predictor
block/loop). Saves all activations + inputs as a single safetensors file so
the Rust implementation can assert per-stage equivalence.

Three fixed seeds (0, 1, 2) produce three input variants. Each stage is
prefixed with seedN_ in the output keys.

Usage:
    WM_POOL_MODE=cls python3 scripts/reference/code_wm_pytorch_baseline.py \
        --ckpt /tmp/synapse_codewm_package/g8_sigreg_dir.pt \
        --code-wm-src /tmp/code_wm_test \
        --out tests/fixtures/code_wm_reference_g8.safetensors

Requires wm_base.py laid out as <src>/wm_base/wm_base.py and
code_wm.py as <src>/code_wm/code_wm.py (matching the directory layout the
model expects for its relative imports).
"""

from __future__ import annotations

import argparse
import importlib.util
import os

import torch
from safetensors.torch import save_file


def load_code_wm_module(code_wm_src: str):
    """Dynamically import code_wm.py (which imports wm_base via relative path)."""
    src = os.path.abspath(code_wm_src)
    code_wm_path = os.path.join(src, "code_wm", "code_wm.py")
    if not os.path.exists(code_wm_path):
        raise FileNotFoundError(
            f"Expected {code_wm_path}. Use --code-wm-src pointing at a dir with "
            "code_wm/code_wm.py + wm_base/wm_base.py."
        )
    spec = importlib.util.spec_from_file_location("code_wm", code_wm_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def build_model(code_wm_mod, ckpt_config: dict[str, int | float]):
    """Instantiate CodeWorldModel with params from the checkpoint's config."""
    m = code_wm_mod.CodeWorldModel(
        vocab_size=ckpt_config["vocab_size"],
        max_seq_len=ckpt_config["max_seq_len"],
        encoder_loops=ckpt_config["encoder_loops"],
        model_dim=ckpt_config["model_dim"],
        num_loops=ckpt_config["num_loops"],  # predictor loops
        num_heads=ckpt_config["num_heads"],
        predictor_depth=2,  # hard-coded in LoopedPredictor
        ema_decay=ckpt_config.get("ema_decay", 0.996),
        action_dim=ckpt_config["action_dim"],
        mlp_ratio=4.0,
        dropout=0.1,
    )
    return m


def set_inference_mode(model: "torch.nn.Module") -> None:
    model.train(False)
    for mod in model.modules():
        if isinstance(mod, torch.nn.Dropout):
            mod.p = 0.0


def shadow_encoder(m, tokens: torch.Tensor, prefix: str, out: dict[str, torch.Tensor]) -> torch.Tensor:
    """Mirror CodeStateEncoder.forward, recording every intermediate.

    Pool mode (`enc.pool_mode`) is read from the constructed encoder:
    - `cls`: extract token 0 (the standard g8/g1b/g10/expa readout).
    - `attn`: run the learned AttentionPooling (query × MHA), which is what
      the new ema-frozen-15k and phase4-contrast-* checkpoints were trained
      with. In both cases we emit `{prefix}cls_extracted` as the pooled-
      before-norm vector so the Rust golden test compares against a single
      canonical key.
    """
    enc = m.state_encoder
    # Synapse safetensors parser only handles F32/F16/BF16. Save tokens as f32;
    # the Rust test casts back to i64 (values are < 662 so f32 is exact).
    out[f"{prefix}input_tokens"] = tokens.clone().to(torch.float32)

    h = enc.embedding(tokens)  # [B, S, D]
    out[f"{prefix}after_embed"] = h.detach().clone()

    cls = enc.cls_token.expand(tokens.shape[0], -1, -1)  # [B, 1, D]
    h = torch.cat([cls, h], dim=1)  # [B, S+1, D]
    out[f"{prefix}after_cls_prepend"] = h.detach().clone()

    h = h + enc.pos_enc.pe[:, : h.shape[1]]  # dropout disabled in inference mode
    out[f"{prefix}after_pe"] = h.detach().clone()

    block = enc.block
    for i in range(enc.encoder_loops):
        h2 = block.norm1(h)
        out[f"{prefix}loop_{i}_norm1"] = h2.detach().clone()
        h_attn, _ = block.attn(h2, h2, h2, need_weights=False)
        out[f"{prefix}loop_{i}_attn"] = h_attn.detach().clone()
        h = h + h_attn
        out[f"{prefix}loop_{i}_res1"] = h.detach().clone()
        h2 = block.norm2(h)
        out[f"{prefix}loop_{i}_norm2"] = h2.detach().clone()
        h2 = block.mlp(h2)
        out[f"{prefix}loop_{i}_mlp"] = h2.detach().clone()
        h = h + h2
        out[f"{prefix}loop_{i}_res2"] = h.detach().clone()

    # Readout branch — must match `CodeStateEncoder.pool_mode`, which is
    # set at construction time from the `WM_POOL_MODE` env var.
    if enc.pool_mode == "cls":
        pooled = h[:, 0]  # [B, D]
    elif enc.pool_mode == "attn":
        # AttentionPooling: expand query to [B, 1, D], cross-attend against h,
        # squeeze the single query dim to get [B, D]. Mirrors the tap impl
        # exactly so Rust parity is byte-comparable.
        B = h.shape[0]
        q = enc.attn_pool.query.expand(B, -1, -1)  # [B, 1, D]
        out[f"{prefix}attn_pool_q_expanded"] = q.detach().clone()
        attn_out, _ = enc.attn_pool.attn(q, h, h, need_weights=False)  # [B, 1, D]
        out[f"{prefix}attn_pool_raw_out"] = attn_out.detach().clone()
        pooled = attn_out.squeeze(1)  # [B, D]
    else:
        # Legacy mean pool (should never hit for g8/expa/new variants).
        pooled = h.mean(dim=1)
    out[f"{prefix}cls_extracted"] = pooled.detach().clone()
    z = enc.norm(pooled)
    out[f"{prefix}encoder_final"] = z.detach().clone()
    return z


def shadow_action_encoder(m, action: torch.Tensor, prefix: str, out: dict[str, torch.Tensor]) -> torch.Tensor:
    """Mirror CodeActionEncoder.forward, recording each sub-step."""
    act = m.action_encoder
    out[f"{prefix}action_input"] = action.detach().clone()
    h = act.net[0](action)  # Linear(7 -> 128)
    out[f"{prefix}action_after_fc1"] = h.detach().clone()
    h = act.net[1](h)  # GELU
    out[f"{prefix}action_after_gelu"] = h.detach().clone()
    h = act.net[2](h)  # Linear(128 -> 128)
    out[f"{prefix}action_final"] = h.detach().clone()
    return h


def shadow_predictor(m, z_state: torch.Tensor, z_action: torch.Tensor, prefix: str, out: dict[str, torch.Tensor]) -> torch.Tensor:
    """Mirror LoopedPredictor.forward, recording each block+loop intermediate."""
    pred = m.predictor
    out[f"{prefix}pred_z_state"] = z_state.detach().clone()
    out[f"{prefix}pred_z_action"] = z_action.detach().clone()
    x = torch.stack([z_state, z_action], dim=1)  # [B, 2, D]
    out[f"{prefix}pred_stacked"] = x.detach().clone()

    for bi, block in enumerate(pred.blocks):
        for li in range(pred.num_loops):
            h2 = block.norm1(x)
            out[f"{prefix}pred_b{bi}_l{li}_norm1"] = h2.detach().clone()
            h_attn, _ = block.attn(h2, h2, h2, need_weights=False)
            out[f"{prefix}pred_b{bi}_l{li}_attn"] = h_attn.detach().clone()
            x = x + h_attn
            out[f"{prefix}pred_b{bi}_l{li}_res1"] = x.detach().clone()
            h2 = block.norm2(x)
            out[f"{prefix}pred_b{bi}_l{li}_norm2"] = h2.detach().clone()
            h2 = block.mlp(h2)
            out[f"{prefix}pred_b{bi}_l{li}_mlp"] = h2.detach().clone()
            x = x + h2
            out[f"{prefix}pred_b{bi}_l{li}_res2"] = x.detach().clone()

    tok0 = x[:, 0]  # [B, D]
    out[f"{prefix}pred_token0_extracted"] = tok0.detach().clone()
    z_next = pred.norm(tok0)
    out[f"{prefix}pred_final"] = z_next.detach().clone()
    return z_next


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True, help="Path to .pt checkpoint")
    p.add_argument("--code-wm-src", required=True,
                   help="Directory containing code_wm/code_wm.py and wm_base/wm_base.py")
    p.add_argument("--out", required=True, help="Output safetensors path")
    p.add_argument("--seq-len", type=int, default=64,
                   help="Sequence length of the synthetic test inputs (<= max_seq_len)")
    p.add_argument("--num-seeds", type=int, default=3)
    args = p.parse_args()

    print(f"Loading checkpoint: {args.ckpt}")
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    sd = ckpt["model_state_dict"]
    print(f"  Config: {cfg}")

    # Auto-detect pool mode from the state_dict and set WM_POOL_MODE BEFORE
    # constructing the encoder — `CodeStateEncoder.__init__` reads the env
    # var at construction time (architectures/code_wm/code_wm.py:167).
    has_attn_pool = any(k.startswith("state_encoder.attn_pool.") for k in sd)
    detected_pool_mode = "attn" if has_attn_pool else "cls"
    override = os.environ.get("WM_POOL_MODE")
    if override and override != detected_pool_mode:
        print(f"  WARNING: WM_POOL_MODE={override} overrides detected={detected_pool_mode}")
    else:
        os.environ["WM_POOL_MODE"] = detected_pool_mode
        print(f"  Pool mode (auto-detected from state_dict): {detected_pool_mode}")

    # Import code_wm AFTER the env var is set — the encoder reads WM_POOL_MODE
    # at __init__ time, which runs when we call `code_wm_mod.CodeWorldModel(...)`.
    code_wm_mod = load_code_wm_module(args.code_wm_src)

    m = build_model(code_wm_mod, cfg)
    miss = m.load_state_dict(sd, strict=False)
    print(f"  Loaded state_dict: missing={len(miss.missing_keys)}, unexpected={len(miss.unexpected_keys)}")
    set_inference_mode(m)

    if args.seq_len > cfg["max_seq_len"]:
        raise ValueError(f"seq_len {args.seq_len} > max_seq_len {cfg['max_seq_len']}")

    out: dict[str, torch.Tensor] = {}

    for seed in range(args.num_seeds):
        prefix = f"seed{seed}_"
        print(f"\nSeed {seed} (prefix: {prefix}):")
        g = torch.Generator().manual_seed(seed)
        tokens = torch.randint(0, cfg["vocab_size"], (1, args.seq_len), generator=g, dtype=torch.int64)
        action = torch.randn(1, cfg["action_dim"], generator=g)

        with torch.no_grad():
            z_state = shadow_encoder(m, tokens, prefix, out)
            print(f"  encoder output: shape={tuple(z_state.shape)}, norm={z_state.norm().item():.6f}")
            z_act = shadow_action_encoder(m, action, prefix, out)
            print(f"  action output: shape={tuple(z_act.shape)}, norm={z_act.norm().item():.6f}")
            z_next = shadow_predictor(m, z_state, z_act, prefix, out)
            print(f"  predictor output: shape={tuple(z_next.shape)}, norm={z_next.norm().item():.6f}")

    # Contiguify and save.
    out = {k: v.contiguous() for k, v in out.items()}
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    save_file(out, args.out)
    print(f"\nWrote {len(out)} tensors to {args.out}")
    total_bytes = sum(v.numel() * v.element_size() for v in out.values())
    print(f"Total: {total_bytes / 1024:.1f} KB")


if __name__ == "__main__":
    main()
