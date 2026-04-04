# Modular Zig Rollout Optimization System

## Problem

A 50-step LEWM rollout takes ~304ms on Apple Silicon. The bottleneck is **1,200 tiny GEMM calls** (M=3, 24 calls/step x 50 steps) operating at ~2% of peak FLOPS. Each call repacks weights, has function overhead, and can't tile efficiently.

## Solution

A new `lewmRolloutFused` Zig function that processes all rollout steps in one pass, with a **bitfield flag system** to independently toggle 7 optimizations. Each optimization is additive and can be combined freely.

## Flag System

```
u32 bitfield — passed through FFI as a single integer

Bit 0 (0x01): FUSED_ROLLOUT    — all N steps as seq_len=N*3, one pass per layer
Bit 1 (0x02): ESP_FUSED        — single-pass bias+GELU/residual loops
Bit 2 (0x04): PREPACK_WEIGHTS  — pack B matrices once before rollout, reuse per layer call
Bit 3 (0x08): BLAS_ACCELERATE  — dispatch GEMM to cblas_sgemm (macOS Accelerate)
Bit 4 (0x10): SHARED_ADALN     — compute adaLN modulation once, reuse for all steps
Bit 5 (0x20): QUANT_INT8       — INT8 quantized GEMM for weight matrices
Bit 6 (0x40): QUANT_Q4         — Q4_0 quantized GEMV for weight matrices

Convenience:
  MODE_DEFAULT     = 0x00
  MODE_PORTABLE    = 0x17  (FUSED_ROLLOUT + ESP + PREPACK + SHARED_ADALN)
  MODE_FAST_MAC    = 0x1F  (above + BLAS_ACCELERATE)
  MODE_QUANT_INT8  = 0x37  (PORTABLE + INT8)
  MODE_QUANT_Q4    = 0x57  (PORTABLE + Q4)
```

Q4 and INT8 are mutually exclusive. If both set, Q4 takes precedence.

## Architecture

### New Files

| File | Purpose |
|------|---------|
| `zig/src/ops/fused_lewm_rollout.zig` | Core: multi-step rollout with all optimizations |
| `zig/tests/bench_fused_lewm.zig` | Updated benchmark covering all flag combos |

### Modified Files

| File | Change |
|------|--------|
| `zig/src/ops/matmul.zig` | Add `prepackB()` and `sgemmTiledPrepacked()` |
| `zig/src/ops/fused_lewm_layer.zig` | Extract shared helpers, add GEMM dispatch enum |
| `zig/src/ffi/exports.zig` | Add `syn_lewm_rollout_fused()` export |
| `zig/src/root.zig` | Register `fused_lewm_rollout` module |
| `zig/build.zig` | Already has bench registered |
| `crates/synapse-sys/src/lib.rs` | Add `syn_lewm_rollout_fused()` extern |
| `crates/synapse-core/src/lib.rs` | Add `lewm_rollout_fused()` wrapper |
| `crates/synapse-inference/src/models/vision/lewm.rs` | Use new FFI in `predict_rollout_fused` |

### Component Design

#### 1. GEMM Dispatch (`GemmBackend` enum)

```zig
pub const GemmBackend = enum {
    zig_tiled,       // default: sgemmTiled
    zig_prepacked,   // sgemmTiledPrepacked (B already packed)
    accelerate,      // extern cblas_sgemm (macOS only)
    int8_tiled,      // int8GemmTiled (quantized weights)
    q4_gemv,         // q4_0GemvRow (quantized weights)
};
```

A `gemm_dispatch(backend, m, n, k, a, b, c, ...)` function selects the right kernel. The rollout function resolves the backend once from the mode flags and passes it down.

#### 2. Dynamic Attention (`bidirectional_attention_dynamic`)

The current `small_bidirectional_attention` uses `scores: [16]f32` (stack-allocated, seq_len <= 16).

New function: `bidirectional_attention_dynamic(q, k, v, out, scores_buf, seq_len, num_heads, head_dim)`

- `scores_buf`: pre-allocated scratch of size `seq_len * seq_len` (one head at a time)
- For seq_len=150, that's 150*150 = 22,500 floats = 90KB — fits comfortably in the scratch allocation
- Processes heads sequentially, reusing the same scores buffer
- Falls back to inline `small_bidirectional_attention` when seq_len <= 16 (avoids scratch overhead)

#### 3. Weight Prepacking

Add to `matmul.zig`:

```zig
/// Pack B[n,k] into NR-wide column panels for reuse across GEMM calls.
/// Returns packed buffer of size ceil(n/NR)*NR * k.
pub fn prepackB(b: [*]const f32, ldb: usize, trans_b: bool, n: usize, k: usize, packed: [*]f32) void

/// SGEMM with pre-packed B. Skips packB internally, only packs A.
pub fn sgemmTiledPrepacked(m, n, k, a, lda, trans_a, packed_b, c, ldc, packed_a) void
```

The rollout function prepacks all layer weights once before the layer loop.

#### 4. Shared adaLN

When `SHARED_ADALN` is set and all steps share the same conditioning vector:
- Compute `mod_buf = conditioning @ adaln_weight + adaln_bias` once per layer
- Reuse the same `mod_buf` for all tokens in the fused sequence
- This is valid because in `predict_rollout_fused`, conditioning is the same for all steps

#### 5. Accelerate BLAS

```zig
const builtin = @import("builtin");
const is_macos = builtin.os.tag == .macos;

// Conditionally declare extern (only resolves on macOS link)
extern "c" fn cblas_sgemm(
    order: c_int, transA: c_int, transB: c_int,
    m: c_int, n: c_int, k: c_int,
    alpha: f32, a: [*]const f32, lda: c_int,
    b: [*]const f32, ldb: c_int,
    beta: f32, c: [*]f32, ldc: c_int,
) void;
```

At runtime: if `BLAS_ACCELERATE` flag is set AND `is_macos`, dispatch to `cblas_sgemm`. Otherwise silently fall back to Zig SGEMM.

Build: link `-framework Accelerate` conditionally via `build.zig` when targeting macOS.

#### 6. Quantized Paths

When `QUANT_INT8` or `QUANT_Q4` is set, the rollout function expects **pre-quantized weight buffers** passed alongside the f32 weights. The f32 weights are still needed for operations that don't have quantized variants (adaLN modulation, layernorm).

The FFI signature includes optional quantized weight pointers (nullable). If the quant flag is set but the pointer is null, it falls back to f32.

### Entry Point

```zig
pub fn lewmRolloutFused(
    seq: [*]f32,                    // [num_steps * 3 * hidden], in-place
    conditioning: [*]const f32,     // [hidden]
    num_steps: usize,
    hidden: usize,
    num_heads: usize,
    inner_dim: usize,
    inter: usize,
    num_layers: usize,
    // Per-layer weight arrays (indexed by layer)
    layer_weights: [*]const LayerWeights,
    // Scratch buffers
    scratch: *RolloutScratch,
    // Mode flags
    mode: u32,
) void
```

`LayerWeights` is a packed struct of weight pointers per layer.
`RolloutScratch` holds all pre-allocated buffers (mod_buf, normed_buf, qkv_buf, attn_buf, proj_buf, scores_buf, prepacked buffers).

### Data Flow

```
Input: seq[N*3, hidden], conditioning[hidden], mode flags

For each layer L:
  1. adaLN: if SHARED_ADALN → compute once, broadcast
           else → compute per step (standard)
  2. LayerNorm + modulate (same as standard)
  3. QKV projection:
     - BLAS_ACCELERATE? → cblas_sgemm(N*3, 3*inner, hidden, ...)
     - QUANT_INT8?      → int8GemmTiled(N*3, 3*inner, hidden, ...)
     - QUANT_Q4?        → loop q4_0GemvRow per row
     - PREPACK?         → sgemmTiledPrepacked(N*3, 3*inner, hidden, ...)
     - default          → sgemmTiled(N*3, 3*inner, hidden, ...)
  4. Attention:
     - seq_len <= 16?   → small_bidirectional_attention (stack scores)
     - seq_len > 16?    → bidirectional_attention_dynamic (scratch scores)
  5. Output projection: same GEMM dispatch as (3)
  6. Gated residual:
     - ESP_FUSED? → bias_gated_residual (1 loop)
     - default    → add_bias + gated_residual (2 loops)
  7. FFN norm + modulate
  8. FFN up: GEMM dispatch + bias+GELU (ESP_FUSED? → fused : separate)
  9. FFN down: GEMM dispatch + bias+gated_residual (same)

Output: seq modified in-place with predicted latents at positions 2, 5, 8, ...
```

## Expected Performance

| Mode | Speedup vs baseline (50-step, M-series) | Why |
|------|------------------------------------------|-----|
| Baseline (mode=0) | 1.0x (304ms) | 1,200 tiny M=3 GEMMs |
| FUSED_ROLLOUT (0x01) | ~8-15x | M=150 GEMMs, weight reuse, ~24 calls |
| + ESP_FUSED (0x03) | ~8-16x | Marginal on Mac, matters on ESP32 |
| + PREPACK (0x07) | ~10-18x | Eliminates redundant packB across layers |
| + BLAS (0x0F) | ~15-30x | Apple Accelerate is highly tuned for M-series |
| + SHARED_ADALN (0x1F) | ~15-30x | Saves 50 small GEMV calls per layer |
| + QUANT_Q4 (0x5F) | ~20-40x | 4x less memory bandwidth per GEMM |

Conservative estimates. The FUSED_ROLLOUT alone is the dominant win.

## Verification

1. **Correctness**: For every flag combination, output must match baseline (mode=0) within tolerance:
   - f32 paths: max_abs_diff < 1e-4
   - INT8: cosine_sim > 0.99
   - Q4: cosine_sim > 0.95

2. **Benchmark**: `zig build bench-fused-lewm` tests all flag combos on 50-step rollout.

3. **Rust integration**: `cargo test -p synapse-inference --lib -- rollout` passes all existing rollout tests.

4. **macOS-specific**: BLAS flag works on macOS, silently ignored elsewhere.

## Sequencing

Build incrementally, each step independently testable:

1. **Dynamic attention** — lift seq_len<=16 limit (enables everything else)
2. **FUSED_ROLLOUT** — the big win, new `lewmRolloutFused` function
3. **ESP_FUSED** — port existing fusions into the rollout function
4. **PREPACK_WEIGHTS** — add prepackB + prepacked GEMM variant
5. **SHARED_ADALN** — conditional shared computation
6. **BLAS_ACCELERATE** — extern cblas_sgemm with macOS detection
7. **QUANT_INT8 + QUANT_Q4** — quantized GEMM dispatch

Each step adds one flag, includes correctness test, updates benchmark.
