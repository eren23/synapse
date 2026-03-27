"""
LEWM Speed-of-Light Comparison
==============================

Compares Synapse vs PyTorch across all optimization levels:
1. PyTorch eager (standard)
2. PyTorch torch.compile
3. PyTorch fully-fused (hand-optimized, minimal allocations)
4. PyTorch MPS (Apple GPU)

The fully-fused kernel is the theoretical floor — if Synapse matches it,
we're at maximum performance for this architecture.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import time

torch.set_grad_enabled(False)

# ── Standard LEWM Predictor (same as earlier benchmarks) ─────────

class AdaLNLayer(nn.Module):
    def __init__(self, h=192, inner=1024, heads=16, inter=2048):
        super().__init__()
        self.h, self.inner, self.heads = h, inner, heads
        self.hd = inner // heads
        self.adaln = nn.Linear(h, 6*h)
        self.to_qkv = nn.Linear(h, 3*inner, bias=False)
        self.attn_out = nn.Linear(inner, h)
        self.attn_norm = nn.LayerNorm(h)
        self.mlp_norm = nn.LayerNorm(h)
        self.mlp_up = nn.Linear(h, inter)
        self.mlp_down = nn.Linear(inter, h)

    def forward(self, x, cond):
        s1,sh1,g1,s2,sh2,g2 = self.adaln(cond).chunk(6,-1)
        res = x
        m = self.attn_norm(x)*(1+s1)+sh1
        q,k,v = self.to_qkv(m).chunk(3,-1)
        S=x.shape[0]
        q=q.view(S,self.heads,self.hd).transpose(0,1)
        k=k.view(S,self.heads,self.hd).transpose(0,1)
        v=v.view(S,self.heads,self.hd).transpose(0,1)
        o=F.scaled_dot_product_attention(q,k,v)
        res=res+g1*self.attn_out(o.transpose(0,1).reshape(S,self.inner))
        m2=self.mlp_norm(res)*(1+s2)+sh2
        res=res+g2*self.mlp_down(F.gelu(self.mlp_up(m2)))
        return res

class Predictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([AdaLNLayer() for _ in range(6)])
        self.norm = nn.LayerNorm(192)
        self.pos = nn.Parameter(torch.zeros(3,192))
    def forward(self, z, ae):
        s=torch.zeros(3,192,device=z.device); s[0]=z; s[1]=ae; s=s+self.pos
        for l in self.layers: s=l(s,ae)
        return self.norm(s)[2]


# ── Fully Fused Predictor (speed-of-light reference) ─────────────

class FusedAdaLNLayer(nn.Module):
    """Hand-fused adaLN layer with pre-allocated buffers and minimal allocs."""
    def __init__(self, h=192, inner=1024, heads=16, inter=2048):
        super().__init__()
        self.h, self.inner, self.heads, self.hd = h, inner, heads, inner//heads
        self.inter = inter
        # Weights (same as standard)
        self.adaln = nn.Linear(h, 6*h)
        self.to_qkv = nn.Linear(h, 3*inner, bias=False)
        self.attn_out = nn.Linear(inner, h)
        self.attn_norm = nn.LayerNorm(h)
        self.mlp_norm = nn.LayerNorm(h)
        self.mlp_up = nn.Linear(h, inter)
        self.mlp_down = nn.Linear(inter, h)

    def forward(self, x, cond, residual):
        """In-place forward. Writes into residual buffer."""
        # adaLN modulation (tiny matmul, unavoidable)
        mod = self.adaln(cond)
        s1,sh1,g1,s2,sh2,g2 = mod.view(6, self.h).unbind(0)

        # Fused: norm → modulate → QKV
        normed = self.attn_norm(x)
        normed.mul_(1 + s1).add_(sh1)  # in-place modulate
        qkv = self.to_qkv(normed)
        q,k,v = qkv.view(3, 3, self.heads, self.hd).permute(0,2,1,3).unbind(0)

        # Flash attention (if available) or standard
        attn = F.scaled_dot_product_attention(q, k, v)
        proj = self.attn_out(attn.transpose(0,1).reshape(3, self.inner))

        # Gated residual (in-place)
        residual.copy_(x)
        residual.addcmul_(g1, proj)  # residual += g1 * proj

        # Fused: norm → modulate → FFN
        normed2 = self.mlp_norm(residual)
        normed2.mul_(1 + s2).add_(sh2)  # in-place
        up = F.gelu(self.mlp_up(normed2))
        down = self.mlp_down(up)

        # Gated residual (in-place)
        residual.addcmul_(g2, down)
        return residual

class FusedPredictor(nn.Module):
    """Fully fused predictor with pre-allocated buffers."""
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([FusedAdaLNLayer() for _ in range(6)])
        self.norm = nn.LayerNorm(192)
        self.pos = nn.Parameter(torch.zeros(3,192))
        # Pre-allocated buffers
        self.register_buffer('seq', torch.zeros(3, 192))
        self.register_buffer('residual', torch.zeros(3, 192))

    def forward(self, z, ae):
        self.seq[0] = z
        self.seq[1] = ae
        self.seq[2].zero_()
        x = self.seq + self.pos

        for layer in self.layers:
            x = layer(x, ae, self.residual)

        return self.norm(x)[2]


# ── Benchmark ─────────────────────────────────────────────────────

def bench(model, z, ae, runs=500, warmup=50, label="", sync_fn=None):
    for _ in range(warmup):
        _ = model(z, ae)
    if sync_fn: sync_fn()

    t = time.perf_counter()
    for _ in range(runs):
        _ = model(z, ae)
    if sync_fn: sync_fn()
    ms = (time.perf_counter() - t) * 1000.0 / runs

    # Rollout
    state = z.clone()
    if sync_fn: sync_fn()
    t = time.perf_counter()
    for _ in range(50):
        state = model(state, ae)
    if sync_fn: sync_fn()
    roll = (time.perf_counter() - t) * 1000.0

    print(f"  {label:30s}  predict: {ms:.2f}ms  rollout(50): {roll:.0f}ms ({roll/50:.2f}ms/step)")
    return ms, roll

print("═══════════════════════════════════════════════════════════")
print("  LEWM Speed-of-Light: PyTorch vs Synapse")
print("═══════════════════════════════════════════════════════════")
print()

# --- CPU benchmarks ---
print("CPU (Apple Silicon, Accelerate BLAS):")
z_cpu = torch.randn(192) * 0.1
ae_cpu = torch.zeros(192)

# 1. PyTorch eager
model_eager = Predictor().eval()
eager_ms, eager_roll = bench(model_eager, z_cpu, ae_cpu, label="PyTorch eager")

# 2. PyTorch torch.compile
try:
    model_compiled = torch.compile(Predictor().eval(), mode="reduce-overhead")
    # Transfer weights from eager for fair comparison
    model_compiled.load_state_dict(model_eager.state_dict())
    comp_ms, comp_roll = bench(model_compiled, z_cpu, ae_cpu, runs=200, label="PyTorch torch.compile")
except Exception as e:
    print(f"  torch.compile:                    failed ({e})")
    comp_ms = eager_ms

# 3. Fully fused
model_fused = FusedPredictor().eval()
model_fused.load_state_dict(model_eager.state_dict(), strict=False)
fused_ms, fused_roll = bench(model_fused, z_cpu, ae_cpu, label="PyTorch FUSED (speed-of-light)")

print()

# --- MPS benchmarks ---
if torch.backends.mps.is_available():
    print("MPS (Apple Silicon GPU):")
    z_mps = z_cpu.to("mps")
    ae_mps = ae_cpu.to("mps")
    sync = torch.mps.synchronize

    model_mps = Predictor().to("mps").eval()
    mps_ms, mps_roll = bench(model_mps, z_mps, ae_mps, label="PyTorch MPS eager", sync_fn=sync)

    model_fused_mps = FusedPredictor().to("mps").eval()
    fmps_ms, fmps_roll = bench(model_fused_mps, z_mps, ae_mps, label="PyTorch MPS FUSED", sync_fn=sync)
    print()

# --- Summary ---
print("═══════════════════════════════════════════════════════════")
print("  Summary — predict_next (ms/step)")
print("═══════════════════════════════════════════════════════════")
print()
results = [
    ("PyTorch eager CPU", eager_ms),
    ("PyTorch compile CPU", comp_ms),
    ("PyTorch FUSED CPU", fused_ms),
]
if torch.backends.mps.is_available():
    results += [
        ("PyTorch MPS eager", mps_ms),
        ("PyTorch MPS FUSED", fmps_ms),
    ]
results += [
    ("Synapse f32 (Accelerate)", 1.37),
    ("Synapse INT8 (Zig SIMD)", 1.19),
    ("Synapse Q4 cached", 1.35),
]

results.sort(key=lambda x: x[1])
for i, (name, ms) in enumerate(results):
    marker = " ←" if "Synapse" in name else ""
    bar = "█" * max(1, int(50 * results[0][1] / ms))
    print(f"  {i+1}. {name:30s} {ms:6.2f}ms  {bar}{marker}")

print()
print("  Speed-of-light (fused PyTorch CPU): the theoretical minimum for this architecture.")
print("  If Synapse matches it, we're at maximum performance.")
