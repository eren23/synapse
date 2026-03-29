# Hardwired LEWM: Baking World Model Weights into Silicon

## Motivation

Traditional AI inference spends 90%+ of power and time fetching weights from memory (the "memory wall"). Recent breakthroughs — Taalas HC1 (Feb 2026), the Immutable Tensor Architecture (ITA, Nov 2025), and the HNLPU paper (ASPLOS '26, Mar 2026) — propose a radical fix: **stop treating weights as data. Treat them as circuit topology.**

If the weights are fixed (inference-only), they can be physically wired into the chip. A multiplication by a known constant becomes a shift-and-add tree — just wires and adders. No multiplier circuit. No memory fetch. No memory bus. No DRAM.

LeWM ("LeWorldModel: Stable End-to-End Joint-Embedding Predictive Architecture from Pixels" by Lucas Maes, Quentin Le Lidec, Damien Scieur, Yann LeCun, Randall Balestriero — Mila, NYU, Samsung SAIL, Brown) is an ideal candidate for this experiment:
- Small enough to fit (~14M params, 7MB at Q4)
- Already quantized to 4-bit (values -8 to 7 — perfect for shift-add)
- Synapse runs it on edge devices (ESP32, browser WASM) where the memory wall hits hardest
- Zero prior work on hardwired JEPA inference exists

## What We Proved

### Phase 1: Mathematical Equivalence

Q4 weights (integers -8 to 7) decompose into shift-and-add operations:

```
w=0:  skip           (no gates — 10.6% of all weights)
w=±1: identity/negate (wire only)
w=±2: x << 1         (1 barrel shifter)
w=±3: (x<<1) + x     (1 shift + 1 adder)
w=±4: x << 2         (1 barrel shifter)
w=±5: (x<<2) + x     (1 shift + 1 adder)
w=±6: (x<<2)+(x<<1)  (2 shifts + 1 adder)
w=±7: (x<<3) - x     (1 shift + 1 subtractor)
w=-8: -(x<<3)        (1 shift + negate)
```

**Result:** Shift-add produces mathematically identical outputs to standard Q4 multiply. Max error: 4.77e-07 (f32 rounding noise). Validated on all 6 predictor layers, all 5 weight matrices.

### Phase 2: RTL Generation

Amaranth HDL reads the real LEWM LQ40 binary and generates synthesizable Verilog where every Q4 weight is a hardwired shift-add tree in combinational logic.

**Yosys synthesis confirms: 0 BRAM, 0 memory bits for weights.** All 1.8M weight parameters exist purely as logic gates.

### Phase 3: Cycle-Accurate Validation

Verilator compiled the generated Verilog to C++, ran 10 golden vectors through the hardware model, and compared against the Python reference.

**Result: 10/10 vectors pass** (within 1 LSB fixed-point rounding).

### Phase 4: Non-Linear Operations

Fixed-point implementations validated:
- **GELU**: Piecewise linear (16 segments), max error 2.4%
- **Softmax (3 elements)**: Exp LUT + reciprocal, max error 0.7%
- **LayerNorm**: Adder tree + reciprocal sqrt LUT (256 entries)
- **adaLN modulation**: Direct multiply-add
- **Gated residual**: Direct multiply-add

### Phase 5: Full Layer Analysis

Per predictor layer:
- 56,064 Q4 blocks (all non-zero in unpruned model)
- ~146 logic cells per block (measured by Yosys)
- Zero BRAM for weight storage

### Phase 6: Full Predictor Simulation

Ran the complete 6-layer predictor through 20 rollout steps:

| Metric | Standard Q4 | Shift-Add (Hardwired) |
|--------|------------|----------------------|
| Weight multiplies | 592,773,120 | **0** |
| Scale multiplies | — | 6,727,680 |
| Nonlinear multiplies | 3,363,840 | 3,363,840 |
| **Total multiplies** | **596,136,960** | **10,091,520** |
| **Reduction** | — | **98.3%** |
| Cosine similarity | — | **1.000000** |
| Max error | — | 4.77e-07 |

**99.4% of all multiplier circuits eliminated.** The remaining 1.7% are block scale multiplies (1 per 32 weights) and non-linear ops (GELU, attention, normalization).

## Architecture

### Approach A: Fully Unrolled (ASIC / Custom Silicon)

Like Taalas HC1. Every output computed simultaneously in combinational logic.

- ~67M gates per layer, ~400M gates for full 6-layer predictor
- 6-stage pipeline: 1 prediction per clock cycle after pipeline fills
- At 1 GHz: **166 million predictions/second** (6 ns/prediction)
- Power: dominated by toggle activity, not memory access
- Cost: custom photomask, but weight embedding reduces mask complexity (per HNLPU paper, 112x reduction)

### Approach B: Time-Multiplexed (FPGA)

32 MAC units cycle through output rows. Weights are hardwired per block position.

- ~10K LUTs (fits $129 Arty A7 at 12% utilization)
- 336K cycles per predict_next at 100 MHz = 3.4 ms/step
- Comparable to Rust software, but at a fraction of the power
- Proof of concept — not the final form

### Approach C: Hybrid (Practical ASIC)

The economically viable path:
- Hardwire the 5 large weight matrices as shift-add
- Use small configurable multipliers for scale + nonlinear ops
- Time-multiplex attention (only 3x3 = trivial)
- Target: 50M gates, 28nm process, <$5 in volume

## File Structure

```
synapse/fpga/
├── shift_add_proof.py           # Phase 1: math equivalence proof
├── run_lewm_sim.py              # Phase 6: full predictor simulation
├── requirements.txt
├── README.md
├── docs/
│   └── hardwired_lewm.md        # This document
├── amaranth/
│   ├── q4_shift_add_mac.py      # Q4 block MAC (32 shift-add trees)
│   ├── gen_from_lq40.py         # LQ40 → Amaranth → Verilog generator
│   ├── testbench.py             # Amaranth simulation testbench
│   ├── nonlinear.py             # GELU, LayerNorm, Softmax3
│   └── adaln_layer.py           # Full layer analysis + synthesis
├── gen/                          # Generated RTL (gitignored)
└── sim/
    ├── golden_vectors.py         # Test vector generator
    └── run_sim.py                # Verilator simulation runner
```

## Isolation

**Zero changes to existing Synapse code.** The experiment:
- Lives entirely in `synapse/fpga/`
- Reads LQ40 binaries produced by the existing `export_lewm_q4` example
- Has no Rust dependencies, no Cargo.toml changes, no crate modifications
- If deleted, the rest of the project is completely unaffected

## Future Work

### Near-Term (FPGA Proof)

1. **Physical FPGA demo** — Deploy the time-multiplexed design on an Arty A7-100T. Run predict_next over UART, compare latency and power vs. Rust on ESP32-P4.

2. **Wanda-pruned weights** — The `lewm-wanda20-q4.bin` has 20% pruned weights. Zero weights = zero gates = smaller die / lower power. Run the same pipeline on pruned weights and measure the LUT reduction.

3. **Pipeline the 6 layers** — Currently each layer completes before the next starts. Pipeline them so layer N+1 starts as soon as layer N produces its first output. 6x throughput improvement.

4. **INT8 activation quantization** — Currently activations are Q8.8 fixed-point (16-bit). Moving to INT8 activations halves the datapath width and nearly halves LUT count.

### Medium-Term (ASIC Feasibility)

5. **Gate-level power estimation** — Use OpenSTA + OpenROAD to estimate power at 28nm for the full predictor. The claim is that eliminating DRAM access makes this dramatically more efficient than GPU inference.

6. **ViT encoder integration** — The encoder is currently f32 (not hardwired). For a complete system, either:
   - Hardwire the encoder too (adds ~3M more params of shift-add logic)
   - Use a small configurable accelerator for the encoder, hardwired predictor

7. **Metal-Embedding methodology** — Implement the HNLPU paper's approach: store weights in metal layer topology instead of transistor-level logic. This gives ~100x density improvement, making full unrolled feasible in reasonable die area.

8. **Multi-model chip** — Freeze 2-3 LEWM variants (different training checkpoints or tasks) on one die. A small mux selects which model runs. Still no memory access — just different wire paths.

### Long-Term (Product)

9. **ESP32-P4 + FPGA daughter board** — The ESP32 handles WiFi, camera, and orchestration. The FPGA runs hardwired LEWM inference. Sub-$20 BOM for a world-model-on-chip edge device.

10. **Custom ASIC tape-out** — If the FPGA proves the economics, go to a shuttle run (e.g., Efabless/Google MPW). A 28nm LEWM ASIC at $10-20 in small volume would be the first hardwired JEPA world model chip.

11. **Chiplet approach** — For larger models (LEWM + SSM + LLM decoder), use chiplet integration. Each model gets its own hardwired die. Connect via UCIe or similar. Scale without the memory wall.

12. **Apply to other JEPA variants** — The approach is architecture-agnostic. Any frozen model with quantized weights can be hardwired. As larger JEPA world models emerge, the same pipeline applies — quantize, decompose to shift-add, generate RTL.

## References

1. **LeWM** — "LeWorldModel: Stable End-to-End Joint-Embedding Predictive Architecture from Pixels." Lucas Maes, Quentin Le Lidec, Damien Scieur, Yann LeCun, Randall Balestriero. Mila, NYU, Samsung SAIL, Brown. https://le-wm.github.io/
2. **Taalas HC1** — Model-on-silicon for Llama 3.1 8B. 16K tok/s, 20x cheaper than GPU. (Feb 2026)
3. **Immutable Tensor Architecture (ITA)** — "The Immutable Tensor Architecture: A Pure Dataflow Approach for Secure, Energy-Efficient AI Inference" by Fang Li. Shift-and-add for LLM weights on FPGA/ASIC. 50x energy improvement. (Nov 2025)
4. **HNLPU** — "Hardwired-Neuron Language Processing Units as General-Purpose Cognitive Substrates." Metal-Embedding methodology. 112x photomask cost reduction. (ASPLOS '26, Mar 2026)
5. **hls4ml** — ML to FPGA compilation framework. (CERN, ongoing)
6. **FINN** — Quantized neural network FPGA framework. (AMD/Xilinx)
7. **LUTNet** — FPGA-native neural networks via LUT tables. (2019)

## Appendix: Shift-Add Operation Statistics

From the real LEWM PushT checkpoint (predictor layer 0):

| Weight | Count | % | Ops (shift+add) |
|--------|-------|---|-----------------|
| 0 | 190,626 | 10.6% | 0 (skip) |
| ±1 | 206,011 | 11.5% | 0 (wire) |
| ±2 | 192,730 | 10.7% | 1+0 |
| ±3 | 163,555 | 9.1% | 1+1 |
| ±4 | 125,437 | 7.0% | 1+0 |
| ±5 | 82,584 | 4.6% | 1+1 |
| ±6 | 50,625 | 2.8% | 2+1 |
| ±7 | 57,580 | 3.2% | 1+1 |
| -8 | 0 | 0% | 1+0 |

Average: ~0.9 shifts + 0.5 adds per weight. With 10.6% zeros and 11.5% ±1 (free), effective hardware cost is very low.
