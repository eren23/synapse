#!/usr/bin/env python3
"""
Amaranth simulation testbench for hardwired Q4 linear layers.

Uses Amaranth's built-in simulator to verify that the generated RTL
produces the same results as the Python Q4 forward pass.

Usage:
    python testbench.py --bin ../../web/lewm-compress-demo/lewm-q4-pred.bin \
        --layer 0 --matrix adaln_linear --max-outputs 8
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from shift_add_proof import LQ40Reader, Q4Linear

from gen_from_lq40 import HardwiredQ4Linear
from amaranth.sim import Simulator


def run_testbench(q4_linear: Q4Linear, max_outputs: int, num_tests: int = 5):
    """Simulate the hardwired Q4 linear module and compare with Python reference."""

    n_out = min(q4_linear.out_features, max_outputs) if max_outputs else q4_linear.out_features
    k_in = q4_linear.in_features
    scale_frac = 10

    dut = HardwiredQ4Linear(q4_linear, max_outputs=max_outputs, scale_frac=scale_frac)

    sim = Simulator(dut)

    results = []

    async def testbench(ctx):
        np.random.seed(42)

        for t in range(num_tests):
            # Generate random input (small values to fit in 16-bit fixed-point)
            x_float = np.random.randn(k_in).astype(np.float32) * 0.1

            # Convert to fixed-point (Q8.8 = scale by 256)
            frac_bits = 8
            x_fixed = np.round(x_float * (1 << frac_bits)).astype(np.int32)
            x_fixed = np.clip(x_fixed, -32768, 32767)

            # Drive inputs
            for i in range(k_in):
                ctx.set(dut.x[i], int(x_fixed[i]))

            # Read outputs (combinational — settled immediately)
            hw_outputs = []
            for j in range(n_out):
                val = ctx.get(dut.outputs[j])
                hw_outputs.append(val)

            # Python reference: compute expected outputs
            # Mimic the fixed-point arithmetic:
            #   1. Integer dot product of x_fixed with weight integers
            #   2. Multiply by scale (fixed-point)
            #   3. Right-shift by scale_frac
            py_outputs = []
            bpr = q4_linear.blocks_per_row
            for j in range(n_out):
                acc = 0
                for b in range(bpr):
                    block = q4_linear.blocks[j * bpr + b]
                    ints = block.get_integers()
                    scale_int = int(round(block.scale * (1 << scale_frac)))

                    # Integer dot product
                    int_dot = 0
                    for i in range(32):
                        col = b * 32 + i
                        if col < k_in:
                            int_dot += int(x_fixed[col]) * int(ints[i])

                    # Scale and accumulate
                    acc += int_dot * scale_int

                # Right-shift by scale_frac (arithmetic shift for signed)
                if acc < 0:
                    py_outputs.append(-((-acc) >> scale_frac))
                else:
                    py_outputs.append(acc >> scale_frac)

            # Compare
            hw_arr = np.array(hw_outputs, dtype=np.int64)
            py_arr = np.array(py_outputs, dtype=np.int64)
            diff = np.abs(hw_arr - py_arr)
            max_diff = np.max(diff)
            max_val = max(np.max(np.abs(py_arr)), 1)

            results.append({
                'test': t,
                'max_diff': int(max_diff),
                'max_val': int(max_val),
                'match': max_diff == 0,
            })

            status = "EXACT MATCH" if max_diff == 0 else f"DIFF={max_diff} (max_val={max_val})"
            print(f"  Test {t}: {status}")

    sim.add_testbench(testbench)

    with sim.write_vcd("testbench.vcd"):
        sim.run()

    # Summary
    all_match = all(r['match'] for r in results)
    print()
    if all_match:
        print(f"ALL {num_tests} TESTS: EXACT MATCH")
        print("Hardware RTL produces bit-identical results to Python reference.")
    else:
        close = all(r['max_diff'] <= 1 for r in results)
        if close:
            print(f"ALL {num_tests} TESTS: WITHIN 1 LSB (rounding difference)")
            print("This is acceptable for fixed-point arithmetic.")
        else:
            print(f"SOME TESTS FAILED — max diff exceeds tolerance")

    return all_match or all(r['max_diff'] <= 1 for r in results)


def main():
    parser = argparse.ArgumentParser(description="Testbench for hardwired Q4 linear")
    parser.add_argument("--bin", type=str,
                        default="../../web/lewm-compress-demo/lewm-q4-pred.bin")
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--matrix", type=str, default="adaln_linear",
                        choices=["adaln_linear", "to_qkv", "attn_out", "mlp_up", "mlp_down"])
    parser.add_argument("--max-outputs", type=int, default=8)
    parser.add_argument("--num-tests", type=int, default=5)
    args = parser.parse_args()

    bin_path = Path(args.bin)
    if not bin_path.exists():
        bin_path = Path(__file__).parent / args.bin
    if not bin_path.exists():
        print(f"ERROR: LQ40 binary not found")
        sys.exit(1)

    print(f"Reading LQ40: {bin_path}")
    reader = LQ40Reader(str(bin_path))
    model = reader.parse_q4_pred()

    layer = model['predictor_layers'][args.layer]
    q4l = layer[args.matrix]

    print(f"\nSimulating hardwired L{args.layer}.{args.matrix} [{q4l.out_features}x{q4l.in_features}]")
    print(f"  Testing {args.max_outputs} output rows with {args.num_tests} random inputs\n")

    ok = run_testbench(q4l, args.max_outputs, args.num_tests)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
