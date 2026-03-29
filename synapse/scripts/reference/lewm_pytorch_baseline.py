#!/usr/bin/env python3
"""LeWM PyTorch baseline benchmark.

Measures f32/bf16 inference speed and model size for comparison against Synapse.
Uses the same test image + action sequence as lewm_compress.rs.

Usage:
    pip install torch safetensors
    python scripts/reference/lewm_pytorch_baseline.py [checkpoint_path]
"""

import sys
import time
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file

CHECKPOINT = sys.argv[1] if len(sys.argv) > 1 else "/tmp/lewm-pusht/pusht/lejepa_weights.safetensors"


# ── Minimal LeWM reimplementation (enough for inference benchmark) ──

class PatchEmbed(nn.Module):
    def __init__(self, image_size=224, patch_size=14, channels=3, embed_dim=192):
        super().__init__()
        self.proj = nn.Conv2d(channels, embed_dim, kernel_size=patch_size, stride=patch_size, bias=True)
        self.num_patches = (image_size // patch_size) ** 2

    def forward(self, x):
        return self.proj(x).flatten(2).transpose(1, 2)


class ViTEncoderLayer(nn.Module):
    def __init__(self, dim=192, heads=3, mlp_dim=768):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True, bias=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, dim),
        )

    def forward(self, x):
        normed = self.norm1(x)
        x = x + self.attn(normed, normed, normed)[0]
        x = x + self.mlp(self.norm2(x))
        return x


class AdaLNLayer(nn.Module):
    def __init__(self, dim=192, heads=16, dim_head=64, mlp_dim=2048):
        super().__init__()
        inner_dim = heads * dim_head
        self.adaln = nn.Linear(dim, 6 * dim)
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False)
        self.to_qkv = nn.Linear(dim, 3 * inner_dim, bias=False)
        self.to_out = nn.Linear(inner_dim, dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, dim),
        )
        self.heads = heads
        self.dim_head = dim_head

    def forward(self, x, cond):
        mod = self.adaln(cond)
        s1, sh1, g1, s2, sh2, g2 = mod.chunk(6, dim=-1)

        h = self.norm1(x) * (1 + s1.unsqueeze(1)) + sh1.unsqueeze(1)
        qkv = self.to_qkv(h).chunk(3, dim=-1)
        q, k, v = [t.reshape(t.shape[0], t.shape[1], self.heads, self.dim_head).transpose(1, 2) for t in qkv]
        attn = F.scaled_dot_product_attention(q, k, v)
        attn = attn.transpose(1, 2).reshape(x.shape[0], x.shape[1], -1)
        x = x + g1.unsqueeze(1) * self.to_out(attn)

        h = self.norm2(x) * (1 + s2.unsqueeze(1)) + sh2.unsqueeze(1)
        x = x + g2.unsqueeze(1) * self.mlp(h)
        return x


class LeWMModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.patch_embed = PatchEmbed(224, 14, 3, 192)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, 192))
        self.pos_embed = nn.Parameter(torch.zeros(1, 257, 192))
        self.encoder_layers = nn.ModuleList([ViTEncoderLayer(192, 3, 768) for _ in range(6)])
        self.encoder_norm = nn.LayerNorm(192)

        self.projector = nn.Sequential(
            nn.Linear(192, 192), nn.GELU(),
            nn.Linear(192, 192), nn.GELU(),
            nn.Linear(192, 192),
        )

        self.action_conv = nn.Conv1d(10, 10, 1)
        self.action_mlp = nn.Sequential(
            nn.Linear(10, 192), nn.GELU(),
            nn.Linear(192, 192),
        )

        self.pred_pos_embed = nn.Parameter(torch.zeros(1, 1, 192))
        self.predictor_layers = nn.ModuleList([AdaLNLayer(192, 16, 64, 2048) for _ in range(6)])
        self.predictor_norm = nn.LayerNorm(192)
        self.pred_proj = nn.Sequential(
            nn.Linear(192, 192), nn.GELU(),
            nn.Linear(192, 192), nn.GELU(),
            nn.Linear(192, 192),
        )

    def encode(self, image):
        patches = self.patch_embed(image)
        B = patches.shape[0]
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, patches], dim=1)
        x = x + self.pos_embed
        for layer in self.encoder_layers:
            x = layer(x)
        x = self.encoder_norm(x)
        return self.projector(x[:, 0])

    def predict_next(self, z, action):
        act = self.action_conv(action.unsqueeze(-1)).squeeze(-1)
        act_embed = self.action_mlp(act)
        x = z.unsqueeze(1) + self.pred_pos_embed
        for layer in self.predictor_layers:
            x = layer(x, act_embed)
        x = self.predictor_norm(x)
        return self.pred_proj(x.squeeze(1))

    def rollout(self, z, actions_list):
        states = []
        for action in actions_list:
            z = self.predict_next(z, action)
            states.append(z)
        return states


def create_test_image(h=224, w=224, c=3):
    """Same gradient test image as lewm_compress.rs."""
    mean = torch.tensor([0.485, 0.456, 0.406]).view(c, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(c, 1, 1)
    raw = torch.zeros(1, c, h, w)
    for y in range(h):
        for x in range(w):
            raw[0, 0, y, x] = y / h
            raw[0, 1, y, x] = x / w
            raw[0, 2, y, x] = 0.5 + 0.5 * math.sin((x + y) / (w + h))
    return (raw - mean) / std


def main():
    print("=" * 60)
    print("LeWM PyTorch Baseline Benchmark")
    print("=" * 60)

    model = LeWMModel()
    print(f"\nUsing randomly initialized weights for speed benchmark")
    print(f"(Architecture matches LeWM PushT checkpoint)")
    model.set_default_dtype = torch.float32

    total_params = sum(p.numel() for p in model.parameters())
    model_size_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / 1_048_576
    print(f"\nModel: {total_params:,} params, {model_size_mb:.1f} MB (f32)")

    image = create_test_image()
    num_steps = 20
    actions = [torch.zeros(1, 10) for _ in range(num_steps)]
    for i, a in enumerate(actions):
        t = i / num_steps
        a[0, 0] = math.sin(t * math.pi) * 0.5
        a[0, 1] = math.cos(t * math.pi) * 0.3

    # ── f32 CPU ──
    print("\n--- f32 CPU ---")
    model.float()
    with torch.no_grad():
        z = model.encode(image.float())
        _ = model.predict_next(z, actions[0].float())

        t0 = time.perf_counter()
        z = model.encode(image.float())
        enc_ms = (time.perf_counter() - t0) * 1000
        print(f"  Encode: {enc_ms:.1f}ms")

        t0 = time.perf_counter()
        traj = model.rollout(z, [a.float() for a in actions])
        roll_ms = (time.perf_counter() - t0) * 1000
        print(f"  Rollout ({num_steps} steps): {roll_ms:.1f}ms ({roll_ms/num_steps:.2f}ms/step)")

    # ── bf16 CPU ──
    print("\n--- bf16 CPU ---")
    model_bf16 = model.to(torch.bfloat16)
    with torch.no_grad():
        z = model_bf16.encode(image.to(torch.bfloat16))
        t0 = time.perf_counter()
        traj = model_bf16.rollout(z, [a.to(torch.bfloat16) for a in actions])
        roll_ms_bf16 = (time.perf_counter() - t0) * 1000
        print(f"  Rollout ({num_steps} steps): {roll_ms_bf16:.1f}ms ({roll_ms_bf16/num_steps:.2f}ms/step)")

    # ── MPS GPU ──
    roll_ms_mps = None
    if torch.backends.mps.is_available():
        print("\n--- f32 MPS (Apple GPU) ---")
        model_mps = model.float().to("mps")
        image_mps = image.float().to("mps")
        actions_mps = [a.float().to("mps") for a in actions]
        with torch.no_grad():
            z = model_mps.encode(image_mps)
            _ = model_mps.predict_next(z, actions_mps[0])
            torch.mps.synchronize()

            t0 = time.perf_counter()
            z = model_mps.encode(image_mps)
            torch.mps.synchronize()
            enc_ms_mps = (time.perf_counter() - t0) * 1000
            print(f"  Encode: {enc_ms_mps:.1f}ms")

            t0 = time.perf_counter()
            traj = model_mps.rollout(z, actions_mps)
            torch.mps.synchronize()
            roll_ms_mps = (time.perf_counter() - t0) * 1000
            print(f"  Rollout ({num_steps} steps): {roll_ms_mps:.1f}ms ({roll_ms_mps/num_steps:.2f}ms/step)")

    # ── Summary ──
    print("\n" + "=" * 60)
    print("COMPARISON: PyTorch vs Synapse (20-step rollout)")
    print("=" * 60)
    print()
    print(f"{'Engine':<25} {'Precision':<12} {'Size':>8} {'Rollout':>10} {'ms/step':>8}")
    print("-" * 63)
    print(f"{'PyTorch CPU':<25} {'f32':<12} {f'{model_size_mb:.1f}MB':>8} {f'{roll_ms:.1f}ms':>10} {f'{roll_ms/num_steps:.2f}':>8}")
    print(f"{'PyTorch CPU':<25} {'bf16':<12} {f'{model_size_mb/2:.1f}MB':>8} {f'{roll_ms_bf16:.1f}ms':>10} {f'{roll_ms_bf16/num_steps:.2f}':>8}")
    if roll_ms_mps:
        print(f"{'PyTorch MPS':<25} {'f32':<12} {f'{model_size_mb:.1f}MB':>8} {f'{roll_ms_mps:.1f}ms':>10} {f'{roll_ms_mps/num_steps:.2f}':>8}")
    print(f"{'Synapse (Rust+Zig)':<25} {'f32':<12} {'52.1MB':>8} {'51ms':>10} {'2.5':>8}")
    print(f"{'Synapse':<25} {'Q4 cached':<12} {'17.4MB':>8} {'34ms':>10} {'1.7':>8}")
    print(f"{'Synapse':<25} {'INT8e+Q4p':<12} {'10.4MB':>8} {'116ms':>10} {'5.8':>8}")
    print()
    print("Notes:")
    print("  - Same test image (224x224 gradient) and actions for all runs")
    print("  - PyTorch uses nn.MultiheadAttention / F.scaled_dot_product_attention")
    print("  - Synapse uses hand-written Zig SIMD kernels + pure Rust fallback")
    print("  - Synapse Q4: 6.4x compression, cos@20=0.998 vs f32 baseline")
    print("  - No quantization exists in the reference LeWM implementation")


if __name__ == "__main__":
    main()
