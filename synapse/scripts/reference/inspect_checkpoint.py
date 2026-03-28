#!/usr/bin/env python3
"""Inspect a HuggingFace checkpoint's weight names and shapes.

Usage:
    python inspect_checkpoint.py /path/to/model_dir

Prints all tensor names, shapes, dtypes, and total parameter count.
Useful for verifying weight naming conventions match our from_weights() code.
"""
import argparse
import json
import os
import sys

from safetensors import safe_open


def inspect_safetensors(path: str):
    """Inspect a single safetensors file."""
    with safe_open(path, framework="pt") as f:
        keys = f.keys()
        total_params = 0
        for key in sorted(keys):
            tensor = f.get_tensor(key)
            shape = list(tensor.shape)
            dtype = str(tensor.dtype)
            numel = tensor.numel()
            total_params += numel
            print(f"  {key:60s}  {str(shape):30s}  {dtype:10s}  ({numel:,} params)")
        return total_params, len(keys)


def main():
    parser = argparse.ArgumentParser(description="Inspect HuggingFace checkpoint")
    parser.add_argument("model_dir", help="Path to model directory")
    args = parser.parse_args()

    model_dir = args.model_dir

    # Print config.json
    config_path = os.path.join(model_dir, "config.json")
    if os.path.exists(config_path):
        print("=== config.json ===")
        with open(config_path) as f:
            config = json.load(f)
        print(json.dumps(config, indent=2))
        print()

    # Find safetensors files
    index_path = os.path.join(model_dir, "model.safetensors.index.json")
    if os.path.exists(index_path):
        print("=== Sharded checkpoint ===")
        with open(index_path) as f:
            index = json.load(f)
        shard_files = sorted(set(index["weight_map"].values()))
        print(f"Shards: {shard_files}")
        print()

        total_params = 0
        total_keys = 0
        for shard in shard_files:
            shard_path = os.path.join(model_dir, shard)
            print(f"--- {shard} ---")
            params, keys = inspect_safetensors(shard_path)
            total_params += params
            total_keys += keys
            print()
    else:
        single_path = os.path.join(model_dir, "model.safetensors")
        if os.path.exists(single_path):
            print("=== Single checkpoint ===")
            total_params, total_keys = inspect_safetensors(single_path)
        else:
            print(f"No safetensors found in {model_dir}")
            sys.exit(1)

    print(f"\nTotal: {total_keys} tensors, {total_params:,} parameters ({total_params * 4 / 1e6:.1f} MB f32)")


if __name__ == "__main__":
    main()
