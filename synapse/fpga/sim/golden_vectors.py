#!/usr/bin/env python3
"""
Generate golden test vectors for Verilator simulation.

Produces input/output numpy arrays by running the Python Q4 forward pass.
These become the validation dataset for cycle-accurate RTL simulation.

Usage:
    python golden_vectors.py --bin ../../web/lewm-compress-demo/lewm-q4-pred.bin \
        --layer 0 --matrix adaln_linear --max-outputs 8 --num-vectors 10
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from shift_add_proof import LQ40Reader, Q4Linear


def generate_vectors(q4_linear: Q4Linear, max_outputs: int, num_vectors: int,
                     frac_bits: int = 8, scale_frac: int = 10):
    """Generate golden input/output vector pairs.

    Returns dict with:
        inputs_float: [num_vectors, in_features] f32
        inputs_fixed: [num_vectors, in_features] int16
        outputs_fixed: [num_vectors, n_out] int32  (hardware-equivalent)
        outputs_float: [num_vectors, n_out] f32     (float reference)
        config: dict with dimensions and fixed-point params
    """
    k = q4_linear.in_features
    n = min(q4_linear.out_features, max_outputs) if max_outputs else q4_linear.out_features
    bpr = q4_linear.blocks_per_row

    np.random.seed(42)

    inputs_float = np.random.randn(num_vectors, k).astype(np.float32) * 0.1
    inputs_fixed = np.round(inputs_float * (1 << frac_bits)).astype(np.int16)
    inputs_fixed = np.clip(inputs_fixed, -32768, 32767)

    outputs_fixed = np.zeros((num_vectors, n), dtype=np.int64)
    outputs_float = np.zeros((num_vectors, n), dtype=np.float32)

    # Float reference (standard Q4 forward)
    for v in range(num_vectors):
        x = inputs_float[v:v+1]
        w = np.zeros((q4_linear.out_features, k), dtype=np.float32)
        for j in range(q4_linear.out_features):
            for b in range(bpr):
                block = q4_linear.blocks[j * bpr + b]
                vals = block.dequantize()
                col_start = b * 32
                col_end = min(col_start + 32, k)
                w[j, col_start:col_end] = vals[:col_end - col_start]
        out = x @ w[:n].T
        outputs_float[v] = out[0]

    # Fixed-point reference (matches hardware exactly)
    for v in range(num_vectors):
        x_fix = inputs_fixed[v].astype(np.int64)
        for j in range(n):
            acc = 0
            for b in range(bpr):
                block = q4_linear.blocks[j * bpr + b]
                ints = block.get_integers().astype(np.int64)
                scale_int = int(round(block.scale * (1 << scale_frac)))

                int_dot = 0
                for i in range(32):
                    col = b * 32 + i
                    if col < k:
                        int_dot += x_fix[col] * ints[i]
                acc += int_dot * scale_int

            # Arithmetic right shift
            if acc < 0:
                outputs_fixed[v, j] = -((-acc) >> scale_frac)
            else:
                outputs_fixed[v, j] = acc >> scale_frac

    config = {
        'in_features': k,
        'out_features': n,
        'num_vectors': num_vectors,
        'frac_bits': frac_bits,
        'scale_frac': scale_frac,
        'input_width': 16,
        'output_width': 32,
    }

    return {
        'inputs_float': inputs_float,
        'inputs_fixed': inputs_fixed,
        'outputs_fixed': outputs_fixed.astype(np.int32),
        'outputs_float': outputs_float,
        'config': config,
    }


def save_vectors(vectors: dict, output_dir: Path, name: str):
    """Save vectors as numpy files for Verilator consumption."""
    output_dir.mkdir(parents=True, exist_ok=True)

    np.save(output_dir / f"{name}_inputs_fixed.npy", vectors['inputs_fixed'])
    np.save(output_dir / f"{name}_outputs_fixed.npy", vectors['outputs_fixed'])
    np.save(output_dir / f"{name}_inputs_float.npy", vectors['inputs_float'])
    np.save(output_dir / f"{name}_outputs_float.npy", vectors['outputs_float'])

    config_path = output_dir / f"{name}_config.json"
    with open(config_path, 'w') as f:
        json.dump(vectors['config'], f, indent=2)

    print(f"  Saved to {output_dir}/")
    print(f"    {name}_inputs_fixed.npy:  {vectors['inputs_fixed'].shape}")
    print(f"    {name}_outputs_fixed.npy: {vectors['outputs_fixed'].shape}")
    print(f"    {name}_config.json")


def main():
    parser = argparse.ArgumentParser(description="Generate golden test vectors")
    parser.add_argument("--bin", type=str,
                        default="../../web/lewm-compress-demo/lewm-q4-pred.bin")
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--matrix", type=str, default="adaln_linear")
    parser.add_argument("--max-outputs", type=int, default=8)
    parser.add_argument("--num-vectors", type=int, default=10)
    parser.add_argument("--output", type=str, default=".")
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

    name = f"golden_L{args.layer}_{args.matrix}_top{args.max_outputs}"
    print(f"\nGenerating {args.num_vectors} golden vectors for {name}...")

    vectors = generate_vectors(q4l, args.max_outputs, args.num_vectors)
    save_vectors(vectors, Path(args.output), name)

    # Sanity check: compare float vs fixed
    max_err = np.max(np.abs(vectors['outputs_float'] -
                            vectors['outputs_fixed'] / (1 << vectors['config']['frac_bits'])))
    print(f"\n  Float vs fixed-point max error: {max_err:.4f}")
    print(f"  (This reflects quantization + fixed-point truncation)")


if __name__ == "__main__":
    main()
