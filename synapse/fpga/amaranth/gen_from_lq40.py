#!/usr/bin/env python3
"""
Generate synthesizable Verilog from LQ40 Q4 weights.

Reads a real LEWM Q4 binary, extracts a specific weight matrix, and generates
Amaranth HDL modules with the weights hardwired as shift-add trees.

For the proof-of-concept, we target a single Q4 linear layer (e.g. the
adaln_linear [192->1152] from predictor layer 0). This demonstrates:
  - Zero weight memory (all in combinational logic)
  - Correct inference via shift-add trees
  - Synthesizable Verilog that Yosys can analyze

Usage:
    python gen_from_lq40.py --bin ../../web/lewm-compress-demo/lewm-q4-pred.bin \
        --layer 0 --matrix adaln_linear --output ../gen/
"""

import argparse
import json
import sys
from pathlib import Path

# Add parent to path so we can import the LQ40 reader
sys.path.insert(0, str(Path(__file__).parent.parent))
from shift_add_proof import LQ40Reader, Q4Linear

from amaranth.hdl import *
from amaranth.back.rtlil import convert as rtlil_convert


class Q4ShiftAddBlock(Elaboratable):
    """A single Q4 block (32 elements) with hardwired shift-add weights.

    Computes the integer dot product of 32 inputs with 32 fixed weights.
    Scale multiplication is handled externally.
    """

    def __init__(self, weights: list, input_width: int = 16):
        assert len(weights) == 32
        self.weights = weights
        self.input_width = input_width
        self.inputs = [Signal(signed(input_width), name=f"in_{i}") for i in range(32)]
        self.output = Signal(signed(32), name="dot")

    def elaborate(self, platform):
        m = Module()

        products = []
        for i, w in enumerate(self.weights):
            if w == 0:
                continue
            prod = Signal(signed(20), name=f"p{i}")
            x = self.inputs[i]
            if w == 1:
                m.d.comb += prod.eq(x)
            elif w == -1:
                m.d.comb += prod.eq(-x)
            elif w == 2:
                m.d.comb += prod.eq(x << 1)
            elif w == -2:
                m.d.comb += prod.eq(-(x << 1))
            elif w == 3:
                m.d.comb += prod.eq((x << 1) + x)
            elif w == -3:
                m.d.comb += prod.eq(-((x << 1) + x))
            elif w == 4:
                m.d.comb += prod.eq(x << 2)
            elif w == -4:
                m.d.comb += prod.eq(-(x << 2))
            elif w == 5:
                m.d.comb += prod.eq((x << 2) + x)
            elif w == -5:
                m.d.comb += prod.eq(-((x << 2) + x))
            elif w == 6:
                m.d.comb += prod.eq((x << 2) + (x << 1))
            elif w == -6:
                m.d.comb += prod.eq(-((x << 2) + (x << 1)))
            elif w == 7:
                m.d.comb += prod.eq((x << 3) - x)
            elif w == -7:
                m.d.comb += prod.eq(-((x << 3) - x))
            elif w == -8:
                m.d.comb += prod.eq(-(x << 3))
            products.append(prod)

        if not products:
            m.d.comb += self.output.eq(0)
            return m

        # Binary adder tree
        level = products
        stage = 0
        while len(level) > 1:
            nxt = []
            for j in range(0, len(level), 2):
                if j + 1 < len(level):
                    w = max(level[j].shape().width, level[j+1].shape().width) + 1
                    s = Signal(signed(w), name=f"s{stage}_{j//2}")
                    m.d.comb += s.eq(level[j] + level[j+1])
                    nxt.append(s)
                else:
                    nxt.append(level[j])
            level = nxt
            stage += 1

        m.d.comb += self.output.eq(level[0])
        return m


class HardwiredQ4Linear(Elaboratable):
    """A small Q4 linear layer with all weights hardwired.

    For the proof-of-concept, we limit to small dimensions to keep
    synthesis tractable. The adaln_linear [192->1152] would generate
    ~7K blocks — we support generating a subset (e.g., first N output rows)
    to prove the concept, or the full thing for Yosys analysis.

    Parameters
    ----------
    q4_linear : Q4Linear
        The parsed Q4 linear layer with real trained weights.
    max_outputs : int or None
        Limit number of output rows (for faster generation/synthesis).
    """

    def __init__(self, q4_linear: Q4Linear, max_outputs=None, input_width=16,
                 scale_frac=10):
        self.q4l = q4_linear
        self.n = min(q4_linear.out_features, max_outputs or q4_linear.out_features)
        self.k = q4_linear.in_features
        self.bpr = q4_linear.blocks_per_row
        self.input_width = input_width
        self.scale_frac = scale_frac

        self.x = [Signal(signed(input_width), name=f"x{i}") for i in range(self.k)]
        self.outputs = [Signal(signed(32), name=f"y{j}") for j in range(self.n)]

    def elaborate(self, platform):
        m = Module()

        for row in range(self.n):
            # For this output row, compute dot product across all blocks
            block_scaled = []

            for b in range(self.bpr):
                block_idx = row * self.bpr + b
                block = self.q4l.blocks[block_idx]
                ints = list(block.get_integers())

                # Instantiate hardwired MAC for this block
                mac = Q4ShiftAddBlock(ints, self.input_width)
                m.submodules[f"r{row}_b{b}"] = mac

                # Wire inputs
                for i in range(32):
                    col = b * 32 + i
                    if col < self.k:
                        m.d.comb += mac.inputs[i].eq(self.x[col])
                    else:
                        m.d.comb += mac.inputs[i].eq(0)

                # Scale: convert float to fixed-point constant
                scale_int = int(round(block.scale * (1 << self.scale_frac)))
                if scale_int == 0:
                    continue  # Zero scale = zero block, skip

                scaled = Signal(signed(48), name=f"r{row}_b{b}_sc")
                m.d.comb += scaled.eq(mac.output * Const(scale_int, signed(16)))
                block_scaled.append(scaled)

            if not block_scaled:
                m.d.comb += self.outputs[row].eq(0)
                continue

            # Sum scaled blocks for this row
            total = block_scaled[0]
            for s in block_scaled[1:]:
                new_total = Signal(signed(48), name=f"r{row}_acc")
                m.d.comb += new_total.eq(total + s)
                total = new_total

            # Output: right-shift to remove scale fraction bits
            m.d.comb += self.outputs[row].eq(total >> self.scale_frac)

        return m


def generate_verilog(q4_linear: Q4Linear, name: str, output_dir: Path,
                     max_outputs=None):
    """Generate Verilog for a hardwired Q4 linear layer."""
    mod = HardwiredQ4Linear(q4_linear, max_outputs=max_outputs)

    # Build port list for conversion
    ports = list(mod.x) + list(mod.outputs)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate RTLIL (Amaranth's native format, convertible to Verilog via Yosys)
    rtlil_path = output_dir / f"{name}.il"
    rtlil_text = rtlil_convert(mod, ports=ports)
    rtlil_path.write_text(rtlil_text)

    n_out = mod.n
    k_in = mod.k
    n_blocks = n_out * mod.bpr
    nonzero = sum(1 for blk in q4_linear.blocks[:n_blocks] if blk.scale != 0.0)

    print(f"  Generated: {rtlil_path}")
    print(f"  Dimensions: [{n_out} x {k_in}]")
    print(f"  Blocks: {n_blocks} total, {nonzero} non-zero")
    print(f"  All weights are hardwired combinational logic (zero BRAM)")
    print()
    print(f"  To convert to Verilog and get synthesis stats:")
    print(f"    yosys -p 'read_rtlil {rtlil_path}; write_verilog {output_dir / name}.v; synth; stat'")

    return rtlil_path


def main():
    parser = argparse.ArgumentParser(description="Generate Verilog from LQ40 Q4 weights")
    parser.add_argument("--bin", type=str,
                        default="../../web/lewm-compress-demo/lewm-q4-pred.bin",
                        help="Path to LQ40 binary")
    parser.add_argument("--layer", type=int, default=0,
                        help="Predictor layer index (0-5)")
    parser.add_argument("--matrix", type=str, default="adaln_linear",
                        choices=["adaln_linear", "to_qkv", "attn_out", "mlp_up", "mlp_down"],
                        help="Which weight matrix to generate")
    parser.add_argument("--max-outputs", type=int, default=None,
                        help="Limit output rows for faster generation (e.g. 8 for quick test)")
    parser.add_argument("--output", type=str, default="../gen",
                        help="Output directory for generated files")
    args = parser.parse_args()

    bin_path = Path(args.bin)
    if not bin_path.exists():
        bin_path = Path(__file__).parent / args.bin
    if not bin_path.exists():
        print(f"ERROR: LQ40 binary not found at {args.bin}")
        sys.exit(1)

    print(f"Reading LQ40: {bin_path}")
    reader = LQ40Reader(str(bin_path))
    model = reader.parse_q4_pred()

    layer = model['predictor_layers'][args.layer]
    q4l = layer[args.matrix]

    name = f"hardwired_L{args.layer}_{args.matrix}"
    if args.max_outputs:
        name += f"_top{args.max_outputs}"

    print(f"\nGenerating RTL for {name}...")
    print(f"  Matrix: [{q4l.out_features} x {q4l.in_features}]")
    print(f"  Blocks: {len(q4l.blocks)}")

    output_dir = Path(args.output)
    if not output_dir.is_absolute():
        output_dir = Path(__file__).parent / output_dir

    generate_verilog(q4l, name, output_dir, max_outputs=args.max_outputs)


if __name__ == "__main__":
    main()
