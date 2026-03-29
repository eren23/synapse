# LeWM Hardwired Weights — FPGA Experiment

Proof-of-concept: bake [LeWM](https://le-wm.github.io/) (Maes, Le Lidec, Scieur, LeCun, Balestriero)
Q4 weights directly into silicon logic (shift-add trees), eliminating all weight memory.
Inspired by Taalas HC1, ITA paper, HNLPU.

## Key Results

| Metric | Value |
|--------|-------|
| Trajectory match (20-step rollout) | **cos_sim = 1.000000** |
| Weight multiplies eliminated | **99.4%** (592M → 0) |
| Weight memory (BRAM) | **0** — all weights in logic gates |
| Verilator simulation | 10/10 vectors pass |
| GELU error / Softmax3 error | < 2.4% / < 0.7% |
| FPGA fit (time-multiplexed) | ~10K LUTs (12% of ECP5-85K) |
| ASIC: predictions/second | **166 million** (6 ns/step at 1 GHz) |

## Quick Start

```bash
# The big demo — full 6-layer predictor, 20-step rollout, shift-add vs standard Q4
python run_lewm_sim.py --bin ../web/lewm-compress-demo/lewm-q4-pred.bin --rollout 20

# Phase 1: Prove shift-add == Q4 multiply (all layers)
python shift_add_proof.py --bin ../web/lewm-compress-demo/lewm-q4-pred.bin --all-layers

# Phase 2: Generate Verilog with hardwired weights
cd amaranth
python gen_from_lq40.py --bin ../../web/lewm-compress-demo/lewm-q4-pred.bin \
    --layer 0 --matrix adaln_linear --max-outputs 8

# Phase 2b: Amaranth RTL simulation
python testbench.py --max-outputs 8

# Phase 3: Verilator cycle-accurate simulation
cd ../sim
python golden_vectors.py --bin ../../web/lewm-compress-demo/lewm-q4-pred.bin
python run_sim.py --rtlil ../gen/hardwired_L0_adaln_linear_top8.il

# Phase 4: Test non-linear ops
cd ../amaranth && python nonlinear.py

# Phase 5: Full layer analysis + Yosys synthesis
python adaln_layer.py --bin ../../web/lewm-compress-demo/lewm-q4-pred.bin
```

## Dependencies

```bash
pip install numpy amaranth
brew install yosys verilator
```

## How It Works

Q4 weights (values -8 to 7) decompose into shift-add trees:
- w=0: skip (no gates — 10.6% of all weights)
- w=±1: identity/negate (wire only)
- w=±2,±4,±8: single shift
- w=±3,±5,±6,±7: shift + add/sub

Block scale: 1 multiply per 32 elements (not per weight). This is the ONLY
multiplication needed for weight application.

Two deployment paths:
- **ASIC (Taalas-style)**: Fully unrolled, 1-cycle latency, ~67M gates/layer, 166M pred/s
- **FPGA (time-multiplexed)**: 32 MACs cycling rows, ~10K LUTs, fits $129 Arty board

## Documentation

See [docs/hardwired_lewm.md](docs/hardwired_lewm.md) for the full writeup including
architecture details, future work roadmap, and references.

## File Structure

```
fpga/
  run_lewm_sim.py              — Full predictor simulation (the main demo)
  shift_add_proof.py           — Phase 1: software proof of equivalence
  requirements.txt
  README.md
  docs/
    hardwired_lewm.md          — Full documentation + future work
  amaranth/
    q4_shift_add_mac.py        — Q4 block MAC unit (32 shift-add trees)
    gen_from_lq40.py           — LQ40 binary → Amaranth → Verilog
    testbench.py               — Amaranth simulation testbench
    nonlinear.py               — GELU, LayerNorm, Softmax3, adaLN
    adaln_layer.py             — Full layer analysis + synthesis
  gen/                         — Generated RTL (gitignored)
  sim/
    golden_vectors.py          — Test vector generator
    run_sim.py                 — Verilator simulation runner
```
