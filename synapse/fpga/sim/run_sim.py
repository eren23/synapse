#!/usr/bin/env python3
"""
Verilator cycle-accurate simulation of hardwired Q4 linear layers.

Compiles generated Verilog to C++ via Verilator, drives test vectors,
and validates against golden references. Reports cycle counts and
theoretical latency at target clock frequencies.

Usage:
    python run_sim.py --rtlil ../gen/hardwired_L0_adaln_linear_top8.il \
        --golden . --name golden_L0_adaln_linear_top8
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np


def rtlil_to_verilog(rtlil_path: Path, verilog_path: Path):
    """Convert Amaranth RTLIL to Verilog via Yosys."""
    cmd = ["yosys", "-p", f"read_rtlil {rtlil_path}; write_verilog {verilog_path}"]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Yosys failed:\n{result.stderr}")
        sys.exit(1)
    print(f"  Converted {rtlil_path.name} -> {verilog_path.name}")


def flatten_verilog(verilog_path: Path, flat_path: Path):
    """Use Yosys to flatten hierarchy for Verilator compatibility."""
    cmd = ["yosys", "-p",
           f"read_verilog {verilog_path}; flatten; clean; write_verilog {flat_path}"]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Yosys flatten failed:\n{result.stderr}")
        sys.exit(1)
    print(f"  Flattened -> {flat_path.name}")


def generate_verilator_testbench(config: dict, inputs_path: Path,
                                  outputs_path: Path, tb_path: Path):
    """Generate a C++ testbench for Verilator."""
    n_in = config['in_features']
    n_out = config['out_features']
    n_vec = config['num_vectors']

    # Build input assignment lines
    input_lines = ""
    for i in range(n_in):
        input_lines += f"        dut->x{i} = inputs[v][{i}];\n"

    # Build output comparison lines
    output_lines = ""
    for j in range(n_out):
        output_lines += (
            f"        {{\n"
            f"            int32_t got = dut->y{j};\n"
            f"            int32_t exp = expected_outputs[v][{j}];\n"
            f"            int64_t d = abs((int64_t)got - (int64_t)exp);\n"
            f"            if (d > max_diff) max_diff = d;\n"
            f"        }}\n"
        )

    cpp = f"""// Auto-generated Verilator testbench for hardwired Q4 linear layer
#include "Vtop.h"
#include "verilated.h"
#include <cstdio>
#include <cstdlib>
#include <cmath>

static const int N_IN = {n_in};
static const int N_OUT = {n_out};
static const int N_VEC = {n_vec};

static int16_t inputs[N_VEC][N_IN];
static int32_t expected_outputs[N_VEC][N_OUT];

void load_npy(const char* path, void* dst, size_t bytes) {{
    FILE* f = fopen(path, "rb");
    if (!f) {{ fprintf(stderr, "Cannot open %s\\n", path); exit(1); }}
    // Parse numpy .npy header: magic(6) + version(2) + header_len(2) + header
    unsigned char hdr[10];
    fread(hdr, 1, 10, f);
    uint16_t hdr_len = hdr[8] | (hdr[9] << 8);
    fseek(f, 10 + hdr_len, SEEK_SET);
    size_t rd = fread(dst, 1, bytes, f);
    if (rd != bytes) {{
        fprintf(stderr, "Short read from %s: got %zu, expected %zu\\n", path, rd, bytes);
        exit(1);
    }}
    fclose(f);
}}

int main(int argc, char** argv) {{
    Verilated::commandArgs(argc, argv);

    load_npy("{inputs_path}", inputs, sizeof(inputs));
    load_npy("{outputs_path}", expected_outputs, sizeof(expected_outputs));

    Vtop* dut = new Vtop;

    int pass_count = 0, fail_count = 0;
    int64_t max_diff_all = 0;

    for (int v = 0; v < N_VEC; v++) {{
        // Drive inputs
{input_lines}
        // Combinational logic settles immediately
        dut->eval();

        // Compare outputs
        int64_t max_diff = 0;
{output_lines}
        if (max_diff <= 1) {{
            pass_count++;
        }} else {{
            fail_count++;
            printf("  Vector %d: FAIL max_diff=%lld\\n", v, (long long)max_diff);
        }}
        if (max_diff > max_diff_all) max_diff_all = max_diff;
    }}

    printf("\\nVerilator Simulation Results:\\n");
    printf("  Vectors: %d pass, %d fail (of %d)\\n", pass_count, fail_count, N_VEC);
    printf("  Max diff across all vectors: %lld\\n", (long long)max_diff_all);
    printf("  Status: %s\\n", fail_count == 0 ? "ALL PASS" : "FAILURES DETECTED");
    printf("\\n");

    // Performance estimates (purely combinational = 1 cycle)
    printf("Performance Estimates (combinational, all outputs parallel):\\n");
    int freqs[] = {{100, 200, 500}};
    for (int fi = 0; fi < 3; fi++) {{
        double period_ns = 1000.0 / freqs[fi];
        printf("  @ %d MHz: %.1f ns per inference (1 cycle)\\n", freqs[fi], period_ns);
    }}
    printf("  Compare: Rust Q4 software ~15 ms per predict_next\\n");
    printf("  Potential speedup: ~%.0fx at 100 MHz\\n", 15e6 / (1000.0 / 100));

    delete dut;
    return fail_count > 0 ? 1 : 0;
}}
"""

    tb_path.write_text(cpp)
    print(f"  Generated testbench: {tb_path.name}")


def run_verilator(verilog_path: Path, tb_path: Path, work_dir: Path):
    """Compile and run via Verilator."""
    print(f"\n  Compiling with Verilator...")

    cmd = [
        "verilator", "--cc", "--exe", "--build",
        "-Wno-WIDTHTRUNC", "-Wno-WIDTHEXPAND", "-Wno-UNUSEDSIGNAL",
        "-Wno-UNOPTFLAT",
        str(verilog_path), str(tb_path),
        "--Mdir", str(work_dir / "obj_dir"),
        "--top-module", "top",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(work_dir))
    if result.returncode != 0:
        print(f"  Verilator compilation failed:")
        # Show last 2000 chars of stderr
        err = result.stderr
        print(err[-2000:] if len(err) > 2000 else err)
        return False

    print(f"  Compiled successfully")

    exe = work_dir / "obj_dir" / "Vtop"
    if not exe.exists():
        print(f"  Executable not found at {exe}")
        return False

    print(f"  Running simulation...\n")
    result = subprocess.run([str(exe)], capture_output=True, text=True)
    print(result.stdout)
    if result.stderr:
        print(result.stderr)

    return result.returncode == 0


def main():
    parser = argparse.ArgumentParser(description="Verilator simulation of hardwired Q4")
    parser.add_argument("--verilog", type=str, help="Path to Verilog file")
    parser.add_argument("--rtlil", type=str, help="Path to RTLIL file (converted via Yosys)")
    parser.add_argument("--golden", type=str, default=".",
                        help="Directory containing golden vector .npy files")
    parser.add_argument("--name", type=str, default="golden_L0_adaln_linear_top8",
                        help="Base name of golden vector files")
    args = parser.parse_args()

    golden_dir = Path(args.golden)
    if not golden_dir.is_absolute():
        golden_dir = Path(__file__).parent / golden_dir

    config_path = golden_dir / f"{args.name}_config.json"
    if not config_path.exists():
        print(f"Golden vectors not found. Generate them first:")
        print(f"  python golden_vectors.py")
        sys.exit(1)

    with open(config_path) as f:
        config = json.load(f)

    inputs_path = golden_dir / f"{args.name}_inputs_fixed.npy"
    outputs_path = golden_dir / f"{args.name}_outputs_fixed.npy"

    work_dir = Path(tempfile.mkdtemp(prefix="verilator_"))
    print(f"  Work dir: {work_dir}")

    if args.rtlil:
        rtlil_path = Path(args.rtlil)
        if not rtlil_path.is_absolute():
            rtlil_path = Path(__file__).parent / rtlil_path
        verilog_path = work_dir / "top.v"
        flat_path = work_dir / "top_flat.v"
        rtlil_to_verilog(rtlil_path, verilog_path)
        flatten_verilog(verilog_path, flat_path)
        verilog_path = flat_path
    elif args.verilog:
        verilog_path = Path(args.verilog)
        if not verilog_path.is_absolute():
            verilog_path = Path(__file__).parent / verilog_path
        flat_path = work_dir / "top_flat.v"
        flatten_verilog(verilog_path, flat_path)
        verilog_path = flat_path
    else:
        print("Provide --verilog or --rtlil")
        sys.exit(1)

    print(f"\n  Config: {json.dumps(config, indent=2)}")

    tb_path = work_dir / "testbench.cpp"
    generate_verilator_testbench(config, inputs_path, outputs_path, tb_path)

    ok = run_verilator(verilog_path, tb_path, work_dir)

    if ok:
        shutil.rmtree(work_dir, ignore_errors=True)
    else:
        print(f"  Work dir preserved at: {work_dir}")

    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
