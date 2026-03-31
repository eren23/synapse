# ESP32-P4 PIE Hardware Accelerator — Research & Alignment Plan

## Current Status

### What is already working on real ESP32-P4 hardware

- The ESP-IDF C app in `synapse-esp32/esp-idf-app/` boots, loads model, connects WiFi, and serves HTTP.
- Slim `q4-pred` predictor parity matches the Rust host reference.
- Full `INT8+Q4` predictor parity also matches the Rust host reference.
- Full `INT8+Q4` encoder + predictor runs end-to-end on device from a deterministic image tensor.
- Patch embedding is bit-for-bit aligned with the Rust host probe at the printed precision.
- Short rollout smoke tests complete on-device for both paths.
- **WiFi HTTP inference server** live on port 80 via ESP32-C6 companion (esp_hosted over SDIO).
- **Companion web dashboard** served from embedded flash at `GET /`.
- **PSRAM at 200 MHz** (requires `CONFIG_IDF_EXPERIMENTAL_FEATURES=y`).
- Endpoints: `POST /predict`, `POST /rollout`, `POST /encode`, `GET /status`.

### Baseline benchmarks (PSRAM @ 200 MHz, scalar C, no PIE)

| Operation | Latency |
|-----------|---------|
| predict_next | 3,009 ms |
| rollout (3 steps) | 9,028 ms |
| encode(image) | 70,913 ms |
| encode + predict | 73,921 ms |

### What is not working yet

- The full encoder path is close to, but not exactly, numerically identical to the Rust host reference.
  - Drift starts in encoder layer 0 after exact patch embedding.
  - This currently looks like scalar C vs host pure-Rust math drift inside encoder compute, not a loader bug.
- There are no PIE kernels yet; all verified timings so far are scalar C reference timings.
- Camera/real image input not yet connected (test image is deterministic).

### Corrected priority order

The repo should not jump to PIE first. The right order is:

1. Get end-to-end functional parity on device:
   - `encode(image)`
   - `encode(image) + predict_next(action)`
2. Add a usable board I/O path:
   - serial command protocol first
   - camera or WiFi only after the compute path is correct
3. Then accelerate the hot kernels with PIE.

## Hardware Overview

### ESP32-P4 Core Specs
| Feature | Specification |
|---------|--------------|
| CPU | Dual-core RISC-V RV32IMAFCZc @ 400 MHz |
| LP Core | Single RISC-V @ 40 MHz (ultra-low-power) |
| ISA Extensions | RV32I + M (hardware multiply) + A (atomics) + F (single-precision FPU) + C (compressed) + Zc + Xhwlp (hardware loop) + **Xai (PIE)** |
| Internal SRAM | 768 KB L2MEM (configurable as cache or scratchpad) |
| TCM | 8 KB zero-wait Tightly Coupled Memory |
| External PSRAM | Up to 32 MB, HEX mode (16-line) @ 200 MHz |
| External Flash | Up to 128 MB |
| DMA | GDMA-AHB (SRAM, 3+3 ch) + GDMA-AXI (SRAM+PSRAM, 3+3 ch) |
| Power (active) | ~78 mW (23.88 mA) |

### PPA vs PIE — Key Distinction

**PPA (Pixel Processing Accelerator)**: Image processing only (scale, rotate, blend, fill). Operates on pixel formats (ARGB8888, RGB565, YUV420). **Not useful for inference.**

**PIE (Processor Instruction Extensions)**: Custom RISC-V SIMD instructions baked into the CPU cores. This is the accelerator we want.

## PIE Architecture

### Registers
- **QR0-QR7**: Eight 128-bit vector registers (16x int8, 8x int16, 4x int32/f32)
- **QACC_H, QACC_L**: Two 256-bit accumulators for multiply-accumulate chains
- **ACCX**: 40-bit scalar accumulator for single-value reduction
- Configurable rounding and saturation modes

### Key Instructions for Inference
```
esp.vld.128.ip    QRn, Rs, imm    # Load 128-bit vector (16x int8)
esp.vst.128.ip    QRn, Rs, imm    # Store 128-bit vector
esp.vadd.s8       QR0, QR1, QR2   # 16-wide INT8 vector add
esp.vmac.s8       QR0, QR1        # 16-wide INT8 multiply-accumulate into QACC
esp.srcmb.s8.qacc QR0, QACC, Rn   # Extract accumulator to vector register
```

### Performance
- 16 INT8 multiply-accumulate operations per cycle at 400 MHz
- Benchmarked at 93.8% faster than ANSI C for vector addition
- 180 MB/s for PSRAM-to-IRAM transfers with SIMD copy
- 128-bit alignment required for optimal vector loads/stores

## Alignment with Synapse

### Hot Paths to Accelerate

For LEWM Q4 inference (primary target):

| Operation | Current (pure Rust) | With PIE | Speedup |
|-----------|-------------------|----------|---------|
| Q4Linear::forward() (per row, K=192) | ~192 scalar muls | 12 PIE MAC + dequant | 5-10x |
| LayerNorm | ~2*N scalar ops | SIMD accumulate + rsqrt LUT | 3-5x |
| GELU activation | ~N scalar evals | 256-entry INT8 LUT in TCM | 4-8x |
| Bidirectional attention | N^2 dot products | PIE MAC for QK^T rows | 3-5x |
| adaLN modulation | ~6*N scalar mul/add | 16-wide vector mul/add | 4-8x |

### Q4-to-PIE Strategy

PIE natively accelerates **INT8** operations (16-wide MAC). Q4 requires a dequant step.

**Recommended: Hybrid Q4 storage + INT8 compute (Option C)**
1. Weights stored as Q4 blocks on PSRAM (~12-17 MB)
2. Per row: dequant 32 Q4 nibbles to 32 INT8 values in SRAM scratch buffer
3. Quantize f32 input vector to INT8 (one pass, per-vector scale)
4. PIE `esp.vmac.s8` on INT8 weight row x INT8 input (16-wide)
5. Accumulate in QACC, extract f32 result with scale correction

**Alternative: Pure INT8 storage (Option B)**
- Re-quantize entire model to INT8 (~14 MB vs 7 MB Q4)
- No dequant overhead, direct PIE MAC
- Still fits in 32 MB PSRAM
- Simpler, slightly lower quality

### Memory Layout

| Data | Location | Size | Access Pattern |
|------|----------|------|----------------|
| Model weights (Q4 blocks) | PSRAM | 7-17 MB | Sequential per-layer |
| Current layer weights | L2 cache (auto) | up to 768 KB | Hardware cache |
| Activation vectors | SRAM heap | ~2-8 KB | Random per-token |
| INT8 dequant scratch | SRAM heap | ~4 KB | Per-row lifecycle |
| GELU/SiLU LUT | TCM (8 KB) | 512 B | Hot loop constant |
| PIE accumulators | Registers | 256-bit | Hardware managed |
| Attention scores | SRAM heap | ~264 KB (257x257x4) | Per-layer |
| DMA double-buffer | SRAM heap | 2x32 KB | Ping-pong |

### DMA Pipeline

Overlap weight transfers from PSRAM with PIE compute:
```
Buffer A: [DMA: load layer L]  [PIE: compute layer L  ] [DMA: load layer L+2]
Buffer B: [PIE: compute L-1 ]  [DMA: load layer L+1   ] [PIE: compute L+1  ]
```

Uses GDMA-AXI channels for PSRAM-to-SRAM concurrent with CPU.

### Dual-Core Strategy

- **Core 0**: Inference compute (PIE MAC, attention, normalization)
- **Core 1**: DMA prefetch, weight decompression, Q4-to-INT8 dequant
- Barrier sync between layers (FreeRTOS task pinning)

## Implementation Plan

### Phase 0: Finish the non-PIE functional path

Before any PIE work, the board still needs:

- Tighten encoder parity from "very close" to the Rust host reference, or explicitly define the acceptable tolerance.
- Keep the deterministic host-vs-board parity fixtures for:
  - patch embedding
  - layer-0 encoder output
  - final encoder output
  - encoder + predictor end-to-end
- Replace the deterministic smoke image with a real board input path.

Only after those are green should the project move into PIE kernel work.

### Phase 1: C Kernels with PIE Assembly
```
synapse-esp32/pie_kernels/
  CMakeLists.txt
  include/pie_kernels.h
  src/
    pie_gemv_int8.c      # 16-wide INT8 MAC for linear layers
    pie_q4_gemv.c         # Q4 dequant + PIE INT8 GEMV
    pie_rmsnorm.c         # Vector accumulation for norm
    pie_layernorm.c       # Same pattern
    pie_activation.c      # LUT-based GELU/SiLU (256-entry in TCM)
    pie_vecops.c           # Vector add, mul, scale
```

C API:
```c
void pie_gemv_int8(int n, int k, const int8_t* weights, const int8_t* input,
                   const float* w_scales, float input_scale, float* output);
void pie_gemv_q4(int n, int k, const uint8_t* q4_blocks, const float* input,
                 float* output);
void pie_layernorm(int n, const float* input, const float* weight,
                   const float* bias, float eps, float* output);
void pie_gelu_lut(int n, const float* input, float* output);
```

### Phase 2: Rust FFI Integration
```rust
// synapse-esp32/src/pie_ffi.rs
extern "C" {
    fn pie_gemv_q4(n: i32, k: i32, q4_blocks: *const u8,
                   input: *const f32, output: *mut f32);
}
```

### Phase 3: Feature Flag in synapse-inference
```toml
[features]
esp32-pie = []  # ESP32-P4 PIE SIMD via C FFI
```

Conditional dispatch in Q4Linear::forward():
```rust
#[cfg(feature = "esp32-pie")]
pub fn forward(&self, x: &[f32], m: usize) -> Vec<f32> {
    unsafe { pie_q4_gemv_batch(self, x, m) }
}
```

## Performance Projections

| Model | Operation | Pure Rust @ 400MHz | With PIE | Notes |
|-------|-----------|-------------------|----------|-------|
| LEWM 192d Q4 | predict step | ~100-150 ms | ~15-30 ms | 6 predictor layers |
| LEWM 96d Q4 slim | predict step | ~40-80 ms | ~8-15 ms | 4 layers, smaller |
| LEWM 96d Q4 slim | rollout (50 steps) | ~2-4 s | ~0.4-0.8 s | Interactive! |
| Mamba-130M Q4 | decode/token | ~300-500 ms | ~50-100 ms | |

## Reference Implementations

- **esp-nn** (github.com/espressif/esp-nn): 57% PIE assembly, INT8 conv/fc/pool kernels
- **esp-dl** (github.com/espressif/esp-dl): Higher-level NN framework using `.espdl` model format
- **ESP32-P4 PIE blog**: developer.espressif.com/blog/2024/12/pie-introduction/
- **ESP-IDF v5.3**: Required for ESP32-P4 support

## Key Risks

1. **PIE docs are sparse** — Use esp-nn as reference (working PIE assembly)
2. **Q4-to-INT8 dequant overhead** — Benchmark Option B (pure INT8, 14 MB) as fallback
3. **Rust inline asm for custom ISA** — Use C kernels + FFI (mature toolchain)
4. **PSRAM bandwidth** — DMA double-buffering + sequential access patterns
5. **768 KB SRAM budget** — Attention (264 KB) + DMA (64 KB) + activations (50 KB) = 378 KB, fits

## Concrete Remaining Work

### Done (2026-03-31)

- WiFi HTTP server live with all inference endpoints
- PSRAM at 200 MHz (was stuck at 20 MHz due to missing `CONFIG_IDF_EXPERIMENTAL_FEATURES`)
- Companion web dashboard embedded and served from flash
- Baseline benchmarks captured (predict: 3009ms, encode: 70913ms at 200MHz PSRAM)

### Next: PIE SIMD kernels

The scalar end-to-end path is stable and profiled. The bottleneck is compute (GEMV inner loops), not memory bandwidth. PIE is the next step:

1. Write `pie_gemv.c` with INT8 GEMV using `esp.vmac.s8` (16-wide MAC)
2. Replace scalar `int8_gemv_row()` in encoder path
3. Replace scalar `q4_gemv_row()` in predictor path (Q4→INT8 dequant + PIE MAC)
4. Benchmark each change against 200MHz PSRAM baseline

### Later

- Camera or real image input (currently deterministic test image)
- Slim model (48d/2e2p) testing for smallest possible inference
- Encoder parity tolerance codification
