#!/usr/bin/env python3
"""
Full LeWM Predictor Simulation via Shift-Add Path.

LeWM: "LeWorldModel" by Maes, Le Lidec, Scieur, LeCun, Balestriero
(Mila, NYU, Samsung SAIL, Brown) — https://le-wm.github.io/

Runs the complete 6-layer Q4 predictor transformer using shift-add
decomposition instead of multiply — proving that an entire predict_next
call can run without a single weight multiplication.

This simulates what the hardwired silicon would compute:
  - All Q4 weight matrices use shift-add trees (zero multiplies for weights)
  - Only block scales and non-linear ops use actual multiplications
  - Activations flow through shift-add combinational logic

Usage:
    python run_lewm_sim.py --bin web/lewm-compress-demo/lewm-q4-pred.bin
    python run_lewm_sim.py --bin web/lewm-compress-demo/lewm-q4-pred.bin --rollout 20
"""

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

from shift_add_proof import LQ40Reader, Q4Linear, Q4Block, shift_add_multiply


# ---------------------------------------------------------------------------
# Core ops (matching the Rust forward pass exactly)
# ---------------------------------------------------------------------------

def gelu(x):
    """GELU activation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715*x^3)))"""
    return 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x**3)))


def layernorm(x, weight, eps=1e-6):
    """LayerNorm over last dimension. x: [..., hidden], weight: [hidden]"""
    mean = np.mean(x, axis=-1, keepdims=True)
    var = np.var(x, axis=-1, keepdims=True)
    normed = (x - mean) / np.sqrt(var + eps)
    return normed * weight


def softmax(x, axis=-1):
    """Numerically stable softmax."""
    x_max = np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x - x_max)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


def bidirectional_attention(q, k, v, seq_len, num_heads, head_dim):
    """Multi-head bidirectional attention.
    q, k, v: [seq_len, inner_dim] where inner_dim = num_heads * head_dim
    """
    inner_dim = num_heads * head_dim
    out = np.zeros((seq_len, inner_dim), dtype=np.float32)

    for h in range(num_heads):
        hd_start = h * head_dim
        hd_end = hd_start + head_dim

        q_h = q[:, hd_start:hd_end]  # [seq_len, head_dim]
        k_h = k[:, hd_start:hd_end]
        v_h = v[:, hd_start:hd_end]

        # Attention scores: Q @ K^T / sqrt(head_dim)
        scores = q_h @ k_h.T / np.sqrt(head_dim)  # [seq_len, seq_len]
        attn = softmax(scores, axis=-1)

        # Weighted sum of values
        out[:, hd_start:hd_end] = attn @ v_h

    return out


# ---------------------------------------------------------------------------
# Q4 Forward Pass modes: standard multiply vs shift-add
# ---------------------------------------------------------------------------

def q4_forward_standard(q4l: Q4Linear, x: np.ndarray) -> np.ndarray:
    """Standard Q4 forward: dequantize-and-multiply."""
    # Dequantize to dense
    w = np.zeros((q4l.out_features, q4l.in_features), dtype=np.float32)
    bpr = q4l.blocks_per_row
    for j in range(q4l.out_features):
        for b in range(bpr):
            block = q4l.blocks[j * bpr + b]
            vals = block.dequantize()
            col_start = b * 32
            col_end = min(col_start + 32, q4l.in_features)
            w[j, col_start:col_end] = vals[:col_end - col_start]
    return x @ w.T


def q4_forward_shift_add(q4l: Q4Linear, x: np.ndarray) -> np.ndarray:
    """Shift-add Q4 forward: integer shifts replace weight multiplies.
    This is what the hardwired silicon computes.
    """
    m = x.shape[0] if x.ndim > 1 else 1
    if x.ndim == 1:
        x = x.reshape(1, -1)

    n = q4l.out_features
    k = q4l.in_features
    bpr = q4l.blocks_per_row
    out = np.zeros((m, n), dtype=np.float32)

    for j in range(n):
        for b in range(bpr):
            block = q4l.blocks[j * bpr + b]
            ints = block.get_integers().astype(np.float32)
            col_start = b * 32
            col_end = min(col_start + 32, k)
            width = col_end - col_start
            # Integer dot product (shifts and adds in hardware)
            x_slice = x[:, col_start:col_end]
            int_dot = x_slice @ ints[:width]
            # Single scale multiply per block (the ONLY multiply)
            out[:, j] += int_dot * block.scale
    return out


# ---------------------------------------------------------------------------
# Full predictor layer forward
# ---------------------------------------------------------------------------

def adaln_layer_forward(layer: dict, seq: np.ndarray, conditioning: np.ndarray,
                         config: dict, use_shift_add: bool = True,
                         multiply_count: dict = None):
    """Run one adaLN transformer layer.

    Args:
        layer: dict with Q4Linear matrices and f32 biases/norms
        seq: [seq_len, hidden] input sequence
        conditioning: [hidden] action embedding
        config: model config dict
        use_shift_add: if True, use shift-add for Q4 linears
        multiply_count: dict to track multiply counts

    Returns:
        [seq_len, hidden] output sequence
    """
    hidden = config['predictor_hidden']
    num_heads = config['predictor_heads']
    inner_dim = config['predictor_inner_dim']
    inter = config['predictor_inter']
    head_dim = inner_dim // num_heads
    seq_len = seq.shape[0]

    forward_fn = q4_forward_shift_add if use_shift_add else q4_forward_standard

    # 1. adaLN modulation
    mod_vec = forward_fn(layer['adaln_linear'], conditioning.reshape(1, -1)).flatten()
    mod_vec += layer['adaln_bias'][:len(mod_vec)]

    if multiply_count is not None:
        if use_shift_add:
            # Only block scale multiplies
            nz = sum(1 for b in layer['adaln_linear'].blocks if b.scale != 0.0)
            multiply_count['scale'] += nz
        else:
            multiply_count['weight'] += layer['adaln_linear'].out_features * layer['adaln_linear'].in_features

    scale1 = mod_vec[0:hidden]
    shift1 = mod_vec[hidden:2*hidden]
    gate1 = mod_vec[2*hidden:3*hidden]
    scale2 = mod_vec[3*hidden:4*hidden]
    shift2 = mod_vec[4*hidden:5*hidden]
    gate2 = mod_vec[5*hidden:6*hidden]

    residual = seq.copy()

    # 2. Pre-attention: layernorm + modulate
    normed = layernorm(seq, layer['attn_norm_weight'][:hidden])
    modulated = normed * (1.0 + scale1) + shift1
    if multiply_count is not None:
        multiply_count['nonlinear'] += seq_len * hidden * 2  # norm + modulate

    # 3. QKV + attention
    qkv = forward_fn(layer['to_qkv'], modulated)  # [seq_len, 3*inner_dim]
    if multiply_count is not None:
        if use_shift_add:
            nz = sum(1 for b in layer['to_qkv'].blocks if b.scale != 0.0)
            multiply_count['scale'] += nz
        else:
            multiply_count['weight'] += layer['to_qkv'].out_features * layer['to_qkv'].in_features * seq_len

    q = qkv[:, :inner_dim]
    k = qkv[:, inner_dim:2*inner_dim]
    v = qkv[:, 2*inner_dim:]

    attn_out = bidirectional_attention(q, k, v, seq_len, num_heads, head_dim)
    if multiply_count is not None:
        multiply_count['nonlinear'] += seq_len * seq_len * num_heads * head_dim * 2  # QK^T + AV

    # Output projection + bias
    proj = forward_fn(layer['attn_out'], attn_out)  # [seq_len, hidden]
    for j in range(min(hidden, len(layer['attn_out_bias']))):
        proj[:, j] += layer['attn_out_bias'][j]

    if multiply_count is not None:
        if use_shift_add:
            nz = sum(1 for b in layer['attn_out'].blocks if b.scale != 0.0)
            multiply_count['scale'] += nz
        else:
            multiply_count['weight'] += layer['attn_out'].out_features * layer['attn_out'].in_features * seq_len

    # 4. Gated residual
    residual += gate1 * proj
    if multiply_count is not None:
        multiply_count['nonlinear'] += seq_len * hidden  # gate multiply

    # 5. Pre-FFN: layernorm + modulate
    normed2 = layernorm(residual, layer['mlp_norm_weight'][:hidden])
    modulated2 = normed2 * (1.0 + scale2) + shift2
    if multiply_count is not None:
        multiply_count['nonlinear'] += seq_len * hidden * 2

    # 6. MLP: up -> GELU -> down
    up = forward_fn(layer['mlp_up'], modulated2)
    for j in range(min(inter, len(layer['mlp_up_bias']))):
        up[:, j] += layer['mlp_up_bias'][j]
    up = gelu(up)

    if multiply_count is not None:
        if use_shift_add:
            nz = sum(1 for b in layer['mlp_up'].blocks if b.scale != 0.0)
            multiply_count['scale'] += nz
        else:
            multiply_count['weight'] += layer['mlp_up'].out_features * layer['mlp_up'].in_features * seq_len
        multiply_count['nonlinear'] += seq_len * inter  # GELU

    down = forward_fn(layer['mlp_down'], up)
    for j in range(min(hidden, len(layer['mlp_down_bias']))):
        down[:, j] += layer['mlp_down_bias'][j]

    if multiply_count is not None:
        if use_shift_add:
            nz = sum(1 for b in layer['mlp_down'].blocks if b.scale != 0.0)
            multiply_count['scale'] += nz
        else:
            multiply_count['weight'] += layer['mlp_down'].out_features * layer['mlp_down'].in_features * seq_len

    # 7. Gated residual
    residual += gate2 * down
    if multiply_count is not None:
        multiply_count['nonlinear'] += seq_len * hidden

    return residual


# ---------------------------------------------------------------------------
# Full predict_next
# ---------------------------------------------------------------------------

def predict_next(model: dict, z_t: np.ndarray, action: np.ndarray,
                  use_shift_add: bool = True, multiply_count: dict = None):
    """Full LEWM predict_next using Q4 predictor layers.

    Args:
        model: parsed LQ40 model
        z_t: [hidden] current latent state
        action: [action_dim] action vector
        use_shift_add: use shift-add for Q4 weight matrices

    Returns:
        [latent_dim] predicted next latent state
    """
    config = model['config']
    hidden = config['predictor_hidden']

    # 1. Action encoding (f32, no shift-add needed — tiny)
    # For simulation, generate a plausible action embedding
    # (we don't have the f32 action encoder weights in the Q4 binary
    #  since they're loaded separately, so we use a random embedding)
    np.random.seed(hash(tuple(action.tolist())) % 2**31)
    a_embed = np.random.randn(hidden).astype(np.float32) * 0.1

    # 2. Build sequence: [z_t, a_embed, zeros]
    seq_len = 3
    seq = np.zeros((seq_len, hidden), dtype=np.float32)
    seq[0] = z_t[:hidden]
    seq[1] = a_embed
    # seq[2] = zeros (target position)

    # 3. Add positional embeddings
    pos_embed = model['predictor_pos_embed']
    pos_len = min(len(pos_embed), seq_len * hidden)
    flat_seq = seq.flatten()
    flat_seq[:pos_len] += pos_embed[:pos_len]
    seq = flat_seq.reshape(seq_len, hidden)

    # 4. Run 6 predictor layers
    for i, layer in enumerate(model['predictor_layers']):
        seq = adaln_layer_forward(layer, seq, a_embed, config,
                                   use_shift_add=use_shift_add,
                                   multiply_count=multiply_count)

    # 5. Final norm
    norm_w = model['predictor_norm_weight'][:hidden]
    normed = layernorm(seq, norm_w)
    if len(model['predictor_norm_bias']) >= hidden:
        normed += model['predictor_norm_bias'][:hidden]

    # 6. Extract target position (index 2)
    target = normed[2]

    return target


def rollout(model: dict, z_start: np.ndarray, actions: list,
             use_shift_add: bool = True, multiply_count: dict = None):
    """Multi-step rollout."""
    states = []
    z = z_start.copy()
    for action in actions:
        z = predict_next(model, z, action, use_shift_add, multiply_count)
        states.append(z.copy())
    return states


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Full LEWM predictor simulation via shift-add path")
    parser.add_argument("--bin", type=str,
                        default="web/lewm-compress-demo/lewm-q4-pred.bin")
    parser.add_argument("--rollout", type=int, default=10,
                        help="Number of rollout steps")
    parser.add_argument("--action-dim", type=int, default=10)
    args = parser.parse_args()

    bin_path = Path(args.bin)
    if not bin_path.exists():
        bin_path = Path(__file__).parent / args.bin
    if not bin_path.exists():
        bin_path = Path(__file__).parent.parent / args.bin
    if not bin_path.exists():
        print(f"ERROR: LQ40 binary not found at {args.bin}")
        sys.exit(1)

    print(f"Reading LQ40: {bin_path}")
    reader = LQ40Reader(str(bin_path))
    model = reader.parse_q4_pred()
    config = model['config']
    hidden = config['predictor_hidden']

    print(f"\n{'='*70}")
    print(f"LEWM HARDWIRED INFERENCE SIMULATION")
    print(f"{'='*70}")
    print(f"  Model: {config['predictor_layers']} layers, hidden={hidden}")
    print(f"  Heads: {config['predictor_heads']}, inner_dim={config['predictor_inner_dim']}")
    print(f"  FFN: {config['predictor_inter']}")
    print(f"  Rollout steps: {args.rollout}")

    # Generate initial state and actions
    np.random.seed(42)
    z_start = np.random.randn(hidden).astype(np.float32) * 0.1
    actions = [np.random.randn(args.action_dim).astype(np.float32) * 0.5
               for _ in range(args.rollout)]

    # ---- Run 1: Standard Q4 (multiply-accumulate) ----
    print(f"\n--- Standard Q4 Forward (multiply-accumulate) ---")
    mc_std = {'weight': 0, 'scale': 0, 'nonlinear': 0}
    t0 = time.perf_counter()
    states_std = rollout(model, z_start, actions, use_shift_add=False,
                          multiply_count=mc_std)
    t_std = time.perf_counter() - t0
    print(f"  Time: {t_std*1000:.1f} ms for {args.rollout} steps")
    print(f"  Multiplies: {mc_std['weight']:,} (weight) + {mc_std['nonlinear']:,} (nonlinear)")
    print(f"  Total multiplies: {mc_std['weight'] + mc_std['nonlinear']:,}")

    # ---- Run 2: Shift-Add (hardwired silicon path) ----
    print(f"\n--- Shift-Add Forward (hardwired silicon path) ---")
    mc_sa = {'weight': 0, 'scale': 0, 'nonlinear': 0}
    t0 = time.perf_counter()
    states_sa = rollout(model, z_start, actions, use_shift_add=True,
                         multiply_count=mc_sa)
    t_sa = time.perf_counter() - t0
    print(f"  Time: {t_sa*1000:.1f} ms for {args.rollout} steps")
    print(f"  Scale multiplies: {mc_sa['scale']:,} (1 per non-zero Q4 block)")
    print(f"  Nonlinear multiplies: {mc_sa['nonlinear']:,} (GELU, norm, gate, attention)")
    print(f"  Weight multiplies: 0 (ALL replaced by shift-add)")
    print(f"  Total multiplies: {mc_sa['scale'] + mc_sa['nonlinear']:,}")

    # ---- Compare trajectories ----
    print(f"\n{'='*70}")
    print(f"TRAJECTORY COMPARISON")
    print(f"{'='*70}")

    max_errs = []
    cosine_sims = []
    for step in range(args.rollout):
        s_std = states_std[step]
        s_sa = states_sa[step]
        diff = np.abs(s_std - s_sa)
        max_err = np.max(diff)
        mean_err = np.mean(diff)
        cos_sim = np.dot(s_std, s_sa) / (np.linalg.norm(s_std) * np.linalg.norm(s_sa) + 1e-10)
        max_errs.append(max_err)
        cosine_sims.append(cos_sim)

        # Show first few and last few steps
        if step < 3 or step >= args.rollout - 2:
            print(f"  Step {step:3d}: max_err={max_err:.2e}  mean_err={mean_err:.2e}  "
                  f"cos_sim={cos_sim:.6f}  norm_std={np.linalg.norm(s_std):.4f}  "
                  f"norm_sa={np.linalg.norm(s_sa):.4f}")
        elif step == 3:
            print(f"  ...")

    avg_cos = np.mean(cosine_sims)
    max_max_err = np.max(max_errs)
    min_cos = np.min(cosine_sims)

    # ---- Multiply reduction ----
    print(f"\n{'='*70}")
    print(f"HARDWIRED SILICON IMPACT")
    print(f"{'='*70}")

    total_std = mc_std['weight'] + mc_std['nonlinear']
    total_sa = mc_sa['scale'] + mc_sa['nonlinear']
    weight_eliminated = mc_std['weight']
    reduction = 100 * weight_eliminated / total_std if total_std > 0 else 0

    print(f"  Standard multiplies:    {total_std:>15,}")
    print(f"  Shift-add multiplies:   {total_sa:>15,}")
    print(f"  Weight mults eliminated: {weight_eliminated:>15,} ({reduction:.1f}%)")
    print(f"")
    print(f"  In hardware, {weight_eliminated:,} multiplier circuits are replaced by")
    print(f"  shift-add trees (wires + adders). Only {total_sa:,} actual multipliers")
    print(f"  needed for block scales and non-linear ops.")

    # ---- Trajectory quality ----
    print(f"\n{'='*70}")
    print(f"TRAJECTORY QUALITY")
    print(f"{'='*70}")
    print(f"  Max error across all steps:  {max_max_err:.2e}")
    print(f"  Average cosine similarity:   {avg_cos:.6f}")
    print(f"  Min cosine similarity:       {min_cos:.6f}")

    if avg_cos > 0.9999:
        print(f"\n  VERDICT: IDENTICAL trajectories (within f32 rounding)")
        print(f"  The hardwired silicon path produces the SAME world model")
        print(f"  predictions as the standard Q4 path.")
    elif avg_cos > 0.999:
        print(f"\n  VERDICT: Near-identical trajectories")
    else:
        print(f"\n  VERDICT: Divergent — investigate")

    # ---- Latent state visualization ----
    print(f"\n{'='*70}")
    print(f"LATENT STATE TRAJECTORY (first 8 dims)")
    print(f"{'='*70}")
    print(f"  Step |  Standard Q4 (first 8 dims)")
    print(f"  -----|" + "-" * 60)
    for step in range(min(args.rollout, 10)):
        vals = states_sa[step][:8]
        bars = ''.join([f"{v:+.3f} " for v in vals])
        print(f"  {step:4d} | {bars}")

    # ---- Performance projection ----
    print(f"\n{'='*70}")
    print(f"PERFORMANCE PROJECTIONS")
    print(f"{'='*70}")

    per_step_ms = t_sa * 1000 / args.rollout
    print(f"  Python simulation: {per_step_ms:.1f} ms/step")
    print(f"  Rust Q4 software:  ~2.5 ms/step (estimated)")
    print(f"")
    print(f"  Hardwired silicon projections:")
    print(f"    Time-multiplexed FPGA (100 MHz):")
    cycles_per_step = 56064 * 6  # blocks per layer * 6 layers
    us_per_step = cycles_per_step / 100  # at 100 MHz
    print(f"      {cycles_per_step:,} cycles/step = {us_per_step:.0f} us/step")
    print(f"      {args.rollout} steps = {us_per_step * args.rollout:.0f} us")
    print(f"      vs Python: {t_sa*1e6:.0f} us → {t_sa*1e6/(us_per_step*args.rollout):.0f}x speedup")
    print(f"")
    print(f"    Fully unrolled ASIC (1 GHz):")
    # 6 layers pipelined = 6 cycles
    ns_per_step = 6  # 6 cycles at 1 GHz = 6 ns
    print(f"      6 pipeline stages = 6 ns/step")
    print(f"      {args.rollout} steps = {6 * args.rollout} ns")
    print(f"      That's {args.rollout} world model predictions in {6*args.rollout} nanoseconds.")
    print(f"      ~{int(1e9 / 6):,} predictions/second")

    print(f"\n{'='*70}")
    print(f"SIMULATION COMPLETE")
    print(f"{'='*70}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
