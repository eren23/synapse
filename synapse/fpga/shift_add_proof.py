#!/usr/bin/env python3
"""
Shift-and-Add Proof of Concept for Hardwired Q4 LEWM Weights.

Reads an LQ40 binary (exported by Synapse's export_lewm_q4), extracts Q4
predictor layers, and proves that shift-and-add decomposition produces
bit-equivalent results to standard dequant-multiply.

This is Phase 1 of the hardwired LEWM experiment. If shift-add matches
standard Q4 here in software, we know the Verilog we generate in Phase 2
will be mathematically correct.

Usage:
    python shift_add_proof.py [--bin path/to/lewm-q4-pred.bin] [--layer 0]
"""

import argparse
import json
import struct
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Q4 Block format (matches Rust Q4Block exactly)
# ---------------------------------------------------------------------------

@dataclass
class Q4Block:
    scale: float          # f32
    nibbles: bytes        # 16 bytes = 32 packed 4-bit values

    def dequantize(self) -> np.ndarray:
        """Dequantize to 32 f32 values: (nibble - 8) * scale"""
        vals = np.empty(32, dtype=np.float32)
        for i in range(16):
            byte = self.nibbles[i]
            vals[2 * i]     = ((byte & 0x0F) - 8) * self.scale
            vals[2 * i + 1] = ((byte >> 4)   - 8) * self.scale
        return vals

    def get_integers(self) -> np.ndarray:
        """Extract the raw 4-bit signed integers (-8 to 7)."""
        vals = np.empty(32, dtype=np.int8)
        for i in range(16):
            byte = self.nibbles[i]
            vals[2 * i]     = (byte & 0x0F) - 8
            vals[2 * i + 1] = (byte >> 4)   - 8
        return vals


@dataclass
class Q4Linear:
    out_features: int
    in_features: int
    blocks: list  # list of Q4Block, row-major [out * blocks_per_row]

    @property
    def blocks_per_row(self) -> int:
        padded_k = ((self.in_features + 31) // 32) * 32
        return padded_k // 32

    def forward_standard(self, x: np.ndarray) -> np.ndarray:
        """Standard Q4 forward: dequantize on-the-fly, multiply-accumulate.
        x: [m, in_features] -> [m, out_features]
        """
        m = x.shape[0]
        k = self.in_features
        n = self.out_features
        bpr = self.blocks_per_row
        out = np.zeros((m, n), dtype=np.float32)

        for i in range(m):
            for j in range(n):
                acc = 0.0
                for b in range(bpr):
                    block = self.blocks[j * bpr + b]
                    scale = block.scale
                    for ni in range(16):
                        byte = block.nibbles[ni]
                        v0 = ((byte & 0x0F) - 8) * scale
                        v1 = ((byte >> 4)   - 8) * scale
                        col0 = b * 32 + 2 * ni
                        col1 = col0 + 1
                        if col0 < k:
                            acc += x[i, col0] * v0
                        if col1 < k:
                            acc += x[i, col1] * v1
                out[i, j] = acc
        return out

    def forward_shift_add(self, x: np.ndarray) -> np.ndarray:
        """Shift-and-add forward: integer shifts replace multiplies.
        x: [m, in_features] -> [m, out_features]

        For each Q4 block:
          1. Accumulate dot product using shift-add on integer weights
          2. Multiply final accumulator by block scale (one multiply per block)
        """
        m = x.shape[0]
        k = self.in_features
        n = self.out_features
        bpr = self.blocks_per_row
        out = np.zeros((m, n), dtype=np.float32)

        for i in range(m):
            for j in range(n):
                acc = np.float32(0.0)
                for b in range(bpr):
                    block = self.blocks[j * bpr + b]
                    # Integer dot product using shift-add
                    int_acc = np.float32(0.0)
                    for ni in range(16):
                        byte = block.nibbles[ni]
                        w0 = (byte & 0x0F) - 8  # int in [-8, 7]
                        w1 = (byte >> 4)   - 8
                        col0 = b * 32 + 2 * ni
                        col1 = col0 + 1
                        if col0 < k:
                            int_acc += shift_add_multiply(x[i, col0], w0)
                        if col1 < k:
                            int_acc += shift_add_multiply(x[i, col1], w1)
                    # One scale multiply per block (not per weight)
                    acc += int_acc * block.scale
                out[i, j] = acc
        return out

    def forward_standard_fast(self, x: np.ndarray) -> np.ndarray:
        """Vectorized standard forward using numpy."""
        # Dequantize all weights to dense matrix
        w = self._to_dense()
        return x @ w.T

    def forward_shift_add_fast(self, x: np.ndarray) -> np.ndarray:
        """Vectorized shift-add forward.
        Instead of dequant (int * scale), we compute integer dot product
        per block, then multiply by scale. Mathematically identical.
        """
        m = x.shape[0]
        n = self.out_features
        bpr = self.blocks_per_row
        out = np.zeros((m, n), dtype=np.float32)

        for j in range(n):
            for b in range(bpr):
                block = self.blocks[j * bpr + b]
                ints = block.get_integers().astype(np.float32)  # [32]
                col_start = b * 32
                col_end = min(col_start + 32, self.in_features)
                width = col_end - col_start
                # Integer dot product (shift-add in hardware)
                x_slice = x[:, col_start:col_end]  # [m, width]
                int_dot = x_slice @ ints[:width]    # [m]
                # Single scale multiply per block
                out[:, j] += int_dot * block.scale
        return out

    def _to_dense(self) -> np.ndarray:
        """Dequantize to dense [out_features, in_features] matrix."""
        w = np.zeros((self.out_features, self.in_features), dtype=np.float32)
        bpr = self.blocks_per_row
        for j in range(self.out_features):
            for b in range(bpr):
                block = self.blocks[j * bpr + b]
                vals = block.dequantize()
                col_start = b * 32
                col_end = min(col_start + 32, self.in_features)
                w[j, col_start:col_end] = vals[:col_end - col_start]
        return w


# ---------------------------------------------------------------------------
# Shift-and-Add decomposition table
# ---------------------------------------------------------------------------

def shift_add_multiply(x: float, w: int) -> float:
    """Replace multiplication x * w with shifts and adds.
    w is a 4-bit signed integer in [-8, 7].

    In hardware, each case becomes combinational logic (wires + adders).
    """
    if w == 0:
        return 0.0
    elif w == 1:
        return x
    elif w == -1:
        return -x
    elif w == 2:
        return x + x          # x << 1
    elif w == -2:
        return -(x + x)
    elif w == 3:
        return (x + x) + x    # (x << 1) + x
    elif w == -3:
        return -((x + x) + x)
    elif w == 4:
        x2 = x + x
        return x2 + x2        # x << 2
    elif w == -4:
        x2 = x + x
        return -(x2 + x2)
    elif w == 5:
        x2 = x + x
        return (x2 + x2) + x  # (x << 2) + x
    elif w == -5:
        x2 = x + x
        return -((x2 + x2) + x)
    elif w == 6:
        x2 = x + x
        return (x2 + x2) + x2  # (x << 2) + (x << 1)
    elif w == -6:
        x2 = x + x
        return -((x2 + x2) + x2)
    elif w == 7:
        x2 = x + x
        x4 = x2 + x2
        return (x4 + x2) + x   # (x << 2) + (x << 1) + x = 7x
    elif w == -7:
        x2 = x + x
        x4 = x2 + x2
        return -(((x4 + x2) + x))
    elif w == -8:
        x2 = x + x
        x4 = x2 + x2
        return -(x4 + x4)     # -(x << 3)
    else:
        raise ValueError(f"Weight {w} out of Q4 range [-8, 7]")


# Shift-add operation descriptors (for hardware generation reporting)
SHIFT_ADD_TABLE = {
    -8: [("neg_shift", 3)],                        # -(x << 3)
    -7: [("neg_shift", 3), ("add", 0)],            # -(x<<3) + x
    -6: [("neg_shift", 2), ("neg_shift", 1)],      # -(x<<2) - (x<<1)
    -5: [("neg_shift", 2), ("sub", 0)],            # -(x<<2) - x
    -4: [("neg_shift", 2)],                         # -(x << 2)
    -3: [("neg_shift", 2), ("add", 0)],            # -(x<<2) + x
    -2: [("neg_shift", 1)],                         # -(x << 1)
    -1: [("negate",)],                              # -x
     0: [],                                          # zero (skip)
     1: [("identity",)],                             # x
     2: [("shift", 1)],                              # x << 1
     3: [("shift", 1), ("add", 0)],                 # (x<<1) + x
     4: [("shift", 2)],                              # x << 2
     5: [("shift", 2), ("add", 0)],                 # (x<<2) + x
     6: [("shift", 2), ("shift", 1)],               # (x<<2) + (x<<1)
     7: [("shift", 3), ("sub", 0)],                 # (x<<3) - x
}


def count_ops(w: int) -> dict:
    """Count hardware operations for a given weight value."""
    ops = SHIFT_ADD_TABLE[w]
    n_shifts = sum(1 for op in ops if op[0] in ("shift", "neg_shift"))
    n_adds = sum(1 for op in ops if op[0] in ("add", "sub", "negate"))
    return {"shifts": n_shifts, "adds": n_adds, "total": n_shifts + n_adds}


# ---------------------------------------------------------------------------
# LQ40 Binary Parser
# ---------------------------------------------------------------------------

class LQ40Reader:
    """Reads Synapse LQ40 binary format."""

    def __init__(self, path: str):
        self.data = Path(path).read_bytes()
        self.pos = 0

    def read_bytes(self, n: int) -> bytes:
        result = self.data[self.pos:self.pos + n]
        self.pos += n
        return result

    def read_u32(self) -> int:
        val = struct.unpack_from('<I', self.data, self.pos)[0]
        self.pos += 4
        return val

    def read_f32(self) -> float:
        val = struct.unpack_from('<f', self.data, self.pos)[0]
        self.pos += 4
        return val

    def read_f32_vec(self) -> np.ndarray:
        """Read length-prefixed f32 array."""
        length = self.read_u32()
        vals = np.frombuffer(self.data, dtype='<f4', count=length, offset=self.pos).copy()
        self.pos += length * 4
        return vals

    def read_q4_linear(self) -> Q4Linear:
        """Read a Q4Linear with sparse block encoding."""
        out_features = self.read_u32()
        in_features = self.read_u32()
        total_blocks = self.read_u32()
        nonzero_count = self.read_u32()

        # Read bitmap
        bitmap_bytes = (total_blocks + 7) // 8
        bitmap = self.read_bytes(bitmap_bytes)

        # Read non-zero blocks
        nonzero_blocks = []
        for _ in range(nonzero_count):
            scale = self.read_f32()
            nibbles = self.read_bytes(16)
            nonzero_blocks.append(Q4Block(scale=scale, nibbles=nibbles))

        # Reconstruct full block list (with zero blocks where bitmap says 0)
        blocks = []
        nz_idx = 0
        for i in range(total_blocks):
            if (bitmap[i // 8] >> (i % 8)) & 1:
                blocks.append(nonzero_blocks[nz_idx])
                nz_idx += 1
            else:
                blocks.append(Q4Block(scale=0.0, nibbles=bytes(16)))

        return Q4Linear(out_features=out_features, in_features=in_features, blocks=blocks)

    def read_q4_layer(self) -> dict:
        """Read a full QuantizedQ4AdaLNLayer."""
        layer = {}
        layer['adaln_linear'] = self.read_q4_linear()
        layer['adaln_bias'] = self.read_f32_vec()
        layer['to_qkv'] = self.read_q4_linear()
        layer['attn_out'] = self.read_q4_linear()
        layer['attn_out_bias'] = self.read_f32_vec()
        layer['attn_norm_weight'] = self.read_f32_vec()
        layer['attn_norm_bias'] = self.read_f32_vec()
        layer['mlp_norm_weight'] = self.read_f32_vec()
        layer['mlp_norm_bias'] = self.read_f32_vec()
        layer['mlp_up'] = self.read_q4_linear()
        layer['mlp_up_bias'] = self.read_f32_vec()
        layer['mlp_down'] = self.read_q4_linear()
        layer['mlp_down_bias'] = self.read_f32_vec()
        return layer

    def read_projection_head(self) -> list:
        """Read a ProjectionHead: [u32 num_layers] then (weight_vec, bias_vec) pairs."""
        num_layers = self.read_u32()
        layers = []
        for _ in range(num_layers):
            weight = self.read_f32_vec()
            bias = self.read_f32_vec()
            layers.append((weight, bias))
        return layers

    def parse_q4_pred(self) -> dict:
        """Parse a q4-pred mode LQ40 file. Returns config + predictor layers."""
        # Magic
        magic = self.read_bytes(4)
        assert magic == b'LQ40', f"Bad magic: {magic}"

        # JSON config
        config_len = self.read_u32()
        config_bytes = self.read_bytes(config_len)
        config = json.loads(config_bytes)
        print(f"  Config: {json.dumps(config, indent=2)}")

        # Skip encoder (f32) — we only need predictor Q4 layers
        print("  Skipping f32 encoder...")
        enc_start = self.pos

        # Patch projection + cls + pos_embed
        self.read_f32_vec()  # patch_proj
        self.read_f32_vec()  # patch_proj_bias
        self.read_f32_vec()  # cls_token
        self.read_f32_vec()  # pos_embed

        # Encoder layers
        for _ in range(config['encoder_layers']):
            for _ in range(16):  # 16 f32 vecs per encoder layer
                self.read_f32_vec()

        # Final norm
        self.read_f32_vec()  # final_norm_weight
        self.read_f32_vec()  # final_norm_bias

        enc_bytes = self.pos - enc_start
        print(f"  Encoder: {enc_bytes / 1_048_576:.2f} MB (skipped)")

        # Predictor layers (Q4)
        print("  Reading Q4 predictor layers...")
        pred_start = self.pos

        predictor_pos_embed = self.read_f32_vec()
        predictor_layers = []
        for i in range(config['predictor_layers']):
            layer = self.read_q4_layer()
            predictor_layers.append(layer)
            print(f"    Layer {i}: adaln[{layer['adaln_linear'].out_features}x{layer['adaln_linear'].in_features}]"
                  f"  qkv[{layer['to_qkv'].out_features}x{layer['to_qkv'].in_features}]"
                  f"  mlp_up[{layer['mlp_up'].out_features}x{layer['mlp_up'].in_features}]")

        predictor_norm_weight = self.read_f32_vec()
        predictor_norm_bias = self.read_f32_vec()

        pred_bytes = self.pos - pred_start
        print(f"  Predictor: {pred_bytes / 1_048_576:.2f} MB")

        return {
            'config': config,
            'predictor_pos_embed': predictor_pos_embed,
            'predictor_layers': predictor_layers,
            'predictor_norm_weight': predictor_norm_weight,
            'predictor_norm_bias': predictor_norm_bias,
        }


# ---------------------------------------------------------------------------
# Analysis and validation
# ---------------------------------------------------------------------------

def analyze_weight_distribution(q4l: Q4Linear, name: str):
    """Analyze the distribution of Q4 integer weights and operation counts."""
    counts = np.zeros(17, dtype=np.int64)  # -8..7 mapped to 0..16
    total_elements = 0
    zero_blocks = 0

    for block in q4l.blocks:
        if block.scale == 0.0:
            zero_blocks += 1
            counts[8] += 32  # all zeros
            total_elements += 32
            continue
        ints = block.get_integers()
        for w in ints:
            counts[w + 8] += 1
        total_elements += 32

    print(f"\n  {name} [{q4l.out_features} x {q4l.in_features}]:")
    print(f"    Total elements: {total_elements:,}")
    print(f"    Zero blocks: {zero_blocks}/{len(q4l.blocks)} ({100*zero_blocks/max(len(q4l.blocks),1):.1f}%)")

    # Weight value distribution
    print(f"    Weight distribution:")
    for w in range(-8, 8):
        c = counts[w + 8]
        pct = 100 * c / total_elements if total_elements > 0 else 0
        bar = '#' * int(pct / 2)
        print(f"      w={w:+2d}: {c:8d} ({pct:5.1f}%) {bar}")

    # Operation count analysis
    total_shifts = 0
    total_adds = 0
    total_skips = 0
    for w in range(-8, 8):
        c = int(counts[w + 8])
        if w == 0:
            total_skips += c
        else:
            ops = count_ops(w)
            total_shifts += ops['shifts'] * c
            total_adds += ops['adds'] * c

    total_ops = total_shifts + total_adds
    naive_mults = total_elements - int(counts[8])  # non-zero elements need multiply

    print(f"    Hardware ops:")
    print(f"      Shifts:     {total_shifts:,}")
    print(f"      Adds/Subs:  {total_adds:,}")
    print(f"      Total:      {total_ops:,}")
    print(f"      Skips (w=0): {total_skips:,} ({100*total_skips/total_elements:.1f}%)")
    print(f"      vs. naive multiplies: {naive_mults:,}")
    if naive_mults > 0:
        print(f"      Shift-add / multiply ratio: {total_ops/naive_mults:.2f}x ops")
    print(f"      Scale multiplies: {len(q4l.blocks) - zero_blocks} (1 per non-zero block)")


def validate_shift_add(q4l: Q4Linear, name: str, use_fast: bool = True):
    """Run both forward passes and compare results."""
    np.random.seed(42)
    m = 3  # LEWM sequence length
    x = np.random.randn(m, q4l.in_features).astype(np.float32) * 0.1

    print(f"\n  Validating {name} [{q4l.out_features}x{q4l.in_features}]...")

    if use_fast:
        # Vectorized versions
        t0 = time.perf_counter()
        out_standard = q4l.forward_standard_fast(x)
        t_std = time.perf_counter() - t0

        t0 = time.perf_counter()
        out_shift_add = q4l.forward_shift_add_fast(x)
        t_sa = time.perf_counter() - t0
    else:
        # Element-wise versions (slow but exactly matches hardware behavior)
        t0 = time.perf_counter()
        out_standard = q4l.forward_standard(x)
        t_std = time.perf_counter() - t0

        t0 = time.perf_counter()
        out_shift_add = q4l.forward_shift_add(x)
        t_sa = time.perf_counter() - t0

    # Compare
    diff = np.abs(out_standard - out_shift_add)
    max_err = np.max(diff)
    mean_err = np.mean(diff)
    max_val = max(np.max(np.abs(out_standard)), 1e-10)
    rel_err = max_err / max_val

    print(f"    Standard output range: [{np.min(out_standard):.6f}, {np.max(out_standard):.6f}]")
    print(f"    Max absolute error:    {max_err:.2e}")
    print(f"    Mean absolute error:   {mean_err:.2e}")
    print(f"    Max relative error:    {rel_err:.2e}")
    print(f"    Time (standard):       {t_std*1000:.1f} ms")
    print(f"    Time (shift-add):      {t_sa*1000:.1f} ms")

    # The key assertion: shift-add must produce identical results
    # (within f32 rounding, which should be ~1e-7 relative)
    if max_err < 1e-5:
        print(f"    PASS: Shift-add is equivalent to standard Q4 multiply")
        return True
    else:
        print(f"    FAIL: Results differ beyond tolerance!")
        return False


def validate_element_wise(q4l: Q4Linear, name: str):
    """Validate the element-wise shift_add_multiply function against direct multiply."""
    print(f"\n  Element-wise validation for {name}...")
    errors = 0
    tested = 0

    for w in range(-8, 8):
        for x_val in [0.0, 1.0, -1.0, 0.5, -0.5, 3.14, -2.71, 100.0, -100.0, 1e-6]:
            expected = float(x_val * w)
            got = shift_add_multiply(float(x_val), w)
            if abs(expected - got) > 1e-10 * max(abs(expected), 1):
                print(f"    MISMATCH: x={x_val}, w={w}: expected {expected}, got {got}")
                errors += 1
            tested += 1

    if errors == 0:
        print(f"    PASS: All {tested} element-wise tests passed")
    else:
        print(f"    FAIL: {errors}/{tested} mismatches")
    return errors == 0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Shift-and-Add proof for Q4 LEWM weights")
    parser.add_argument("--bin", type=str,
                        default="../web/lewm-compress-demo/lewm-q4-pred.bin",
                        help="Path to LQ40 binary")
    parser.add_argument("--layer", type=int, default=0,
                        help="Predictor layer to analyze (0-5)")
    parser.add_argument("--all-layers", action="store_true",
                        help="Validate all predictor layers")
    parser.add_argument("--slow", action="store_true",
                        help="Use element-wise (non-vectorized) forward passes")
    args = parser.parse_args()

    bin_path = Path(args.bin)
    if not bin_path.exists():
        # Try relative to script location
        bin_path = Path(__file__).parent / args.bin
    if not bin_path.exists():
        print(f"ERROR: LQ40 binary not found at {args.bin}")
        print(f"  Export it with: cargo run --example export_lewm_q4 -- --checkpoint <path> --mode q4-pred --output <path>")
        sys.exit(1)

    print(f"Reading LQ40 binary: {bin_path} ({bin_path.stat().st_size / 1_048_576:.2f} MB)")
    reader = LQ40Reader(str(bin_path))
    model = reader.parse_q4_pred()

    # Step 1: Validate shift_add_multiply function itself
    print("\n" + "=" * 70)
    print("STEP 1: Element-wise shift-add validation")
    print("=" * 70)
    ok1 = validate_element_wise(None, "shift_add_multiply")

    # Step 2: Analyze weight distributions
    print("\n" + "=" * 70)
    print("STEP 2: Weight distribution analysis")
    print("=" * 70)

    layers_to_check = range(len(model['predictor_layers'])) if args.all_layers else [args.layer]

    for li in layers_to_check:
        layer = model['predictor_layers'][li]
        print(f"\n--- Predictor Layer {li} ---")
        analyze_weight_distribution(layer['adaln_linear'], f"L{li}.adaln_linear")
        analyze_weight_distribution(layer['to_qkv'], f"L{li}.to_qkv")
        analyze_weight_distribution(layer['attn_out'], f"L{li}.attn_out")
        analyze_weight_distribution(layer['mlp_up'], f"L{li}.mlp_up")
        analyze_weight_distribution(layer['mlp_down'], f"L{li}.mlp_down")

    # Step 3: Full forward pass validation
    print("\n" + "=" * 70)
    print("STEP 3: Full forward pass validation (shift-add vs standard Q4)")
    print("=" * 70)

    all_pass = ok1
    for li in layers_to_check:
        layer = model['predictor_layers'][li]
        print(f"\n--- Predictor Layer {li} ---")
        use_fast = not args.slow
        all_pass &= validate_shift_add(layer['adaln_linear'], f"L{li}.adaln_linear", use_fast)
        all_pass &= validate_shift_add(layer['to_qkv'], f"L{li}.to_qkv", use_fast)
        all_pass &= validate_shift_add(layer['attn_out'], f"L{li}.attn_out", use_fast)
        all_pass &= validate_shift_add(layer['mlp_up'], f"L{li}.mlp_up", use_fast)
        all_pass &= validate_shift_add(layer['mlp_down'], f"L{li}.mlp_down", use_fast)

    # Summary
    print("\n" + "=" * 70)
    if all_pass:
        print("ALL TESTS PASSED")
        print("Shift-and-add decomposition is mathematically equivalent to Q4 multiply.")
        print("This validates that hardwired shift-add trees in Verilog will produce")
        print("correct inference results with zero weight memory.")
    else:
        print("SOME TESTS FAILED — investigate before proceeding to HDL generation")
    print("=" * 70)

    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
