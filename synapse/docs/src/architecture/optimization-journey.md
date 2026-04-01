# Optimization Journey

This page tracks the end-to-end optimization of LEWM world model inference on ESP32-P4, from initial deployment through architecture changes and kernel-level tuning. Total speedup: **5.2x** on the full encode + predict cycle.

## The Problem

Running a vision world model (ViT encoder + DiT predictor) on a RISC-V microcontroller at 400 MHz with 32 MB PSRAM. The initial 96d model took **7.2 seconds** for encode + 3-step predict -- too slow for real-time robotics control.

## The Journey

### Step 1: Architecture (96d -> 64d hybrid ALAL)

The biggest single improvement came from switching to a smaller, smarter architecture:

| | 96d ViT | 64d Hybrid ALAL | Speedup |
|---|---|---|---|
| Encoder hidden | 192d | 64d | 9x fewer attention MACs |
| Params | ~8M | 3.0M | 2.7x smaller |
| Binary | 9.8 MB | 3.9 MB | 2.5x smaller |
| predict_next | 583 ms | 152 ms | 3.8x |
| encode | 6,416 ms | 1,392 ms | 4.6x |

The ALAL encoder alternates full-attention and linear-attention blocks, and uses a 64d hidden dimension (vs 192d). See the [Hybrid ALAL guide](../guide/hybrid-alal.md) for details.

### Step 2: Linear Attention (skip softmax -> kernel-trick)

The L blocks in the ALAL encoder don't need softmax. Two levels of optimization:

**Level 1 -- Skip softmax, L1 normalize**: Replaced `softmax(QK^T)` with `L1_normalize(QK^T)` on L blocks. Saved 16ms per L block (92 -> 76 ms).

**Level 2 -- Kernel-trick O(nd^2)**: Reformulated attention using the ELU+1 feature map to avoid building the full n x n score matrix entirely:

```
// Before: O(n^2 * d) -- builds 261x261 score matrix
for each query q_i:
    scores = Q[i] @ K^T        // n dot products of dim d
    normalize(scores)
    out[i] = scores @ V         // weighted sum

// After: O(n * d^2) -- no score matrix
KV = phi(K)^T @ V              // [d,d] matrix, computed once
for each query q_i:
    out[i] = phi(Q[i]) @ KV    // [d] vector, O(d^2) per query
```

With n=261 and d=64: 8.7M -> 2.1M MACs (4x reduction). Saved 18ms more per L block (76 -> 58 ms).

### Step 3: PIE SIMD Batch Patch Embedding

The patch embedding was the hidden bottleneck -- **39% of encode time** (470ms of 1364ms).

**Root cause**: The `matmul_t_into` function was a scalar triple-nested loop, called 256 times (once per 14x14 patch). No PIE SIMD, no batching.

**Fix**:
1. Pre-extract all 256 patches into a contiguous `[256, 588]` f32 buffer
2. Quantize `patch_proj` weight to INT8 at model load time
3. Run one batched INT8 GEMM: `[256, 588] @ [64, 588]^T -> [256, 64]`
4. Uses PIE SIMD dot products + dual-core parallelism automatically

Result: **470ms -> 50ms** (9.4x faster).

### Step 4: Q4 GEMV Zig Dispatch (host-side)

The Q4 predictor was using a pure-Rust scalar dequant loop -- **8-11x slower than f32**.

**Fix**: Wired `Q4Linear::forward()` to the existing Zig `q4_0GemvRow` kernel via `synapse_core::q4_0_gemv`. Added cached f32-to-f16 scale conversion to avoid repacking every call.

Result: Q4 predict **3310us -> 983us** (3.4x faster) on Apple Silicon. Still 2.7x slower than f32 Accelerate, but the right path for ESP32/WASM where Q4 saves memory.

## Results

### ESP32-P4 Timeline

| Step | predict | encode | enc + 3 predict | Notes |
|------|---------|--------|-----------------|-------|
| 96d slim | 583 ms | 6,416 ms | **7,165 ms** | Initial deployment |
| 64d baseline | 443 ms | 4,198 ms | 5,527 ms | Same architecture, smaller latent |
| 64d hybrid ALAL | 152 ms | 1,392 ms | 1,852 ms | New 64d encoder |
| + skip softmax | 152 ms | 1,364 ms | 1,824 ms | L1 normalization |
| + kernel-trick attn | 152 ms | ~1,340 ms | ~1,800 ms | O(nd^2) |
| + PIE batch patch | **152 ms** | **922 ms** | **1,382 ms** | INT8 batch GEMM |

**Total: 7,165 ms -> 1,382 ms (5.2x faster)**

### Host (Apple Silicon, Zig SIMD)

| Model | encode | 20-step rollout |
|-------|--------|----------------|
| 64d hybrid ALAL | 7.9 ms | 9.6 ms |
| 64d baseline | 10.5 ms | 20.9 ms |
| 64d elastic | 10.4 ms | 20.5 ms |

### Quantization Quality

| Format | cos vs f32 | Notes |
|--------|-----------|-------|
| INT8 encoder + Q4 predictor | 0.999 | Production format |
| INT8 GEMV (Zig) | 5-9x faster than f32 | For M=1 decode |
| Q4 GEMV (Zig) | 3.4x faster than pure Rust | Still 2.7x slower than f32 Accelerate |

## What's Left

### Parked (Phase 2)
- **WASM SIMD128**: Browser inference uses scalar Rust. `std::arch::wasm32` SIMD intrinsics would give 3-5x speedup.
- **Compile-time dimension specialization**: Zig `comptime K` variants for 64d/192d enable full loop unrolling.

### Future
- **Fused Zig layer kernel**: One function call per predictor layer instead of 12+ FFI calls. Built and working (`fused_lewm_layer.zig`) but on macOS loses to Apple Accelerate. Win on ESP32/WASM.
- **Q4 integer accumulation**: Zig Q4 inner loop still dequants nibbles to f32. Native i8 path would close the 2.7x gap.
- **Metal SSM shaders**: Mamba/RWKV on GPU (currently CPU-only).
- **2-layer encoder**: Train a `64d_2e_4p` model to push encode under 600ms on ESP32.

## Key Insight

For 2-50M parameter models on edge devices, the optimization playbook is inverted vs large LLMs:

| Large LLMs (7B+) | Small edge models (2-50M) |
|---|---|
| Memory bandwidth bottleneck | Kernel overhead bottleneck |
| Weights don't fit in cache | Entire model fits in L2 |
| One architecture (transformer) | Many architectures (SSM, vision, hybrid) |
| Batch size matters | Always M=1 |
| KV cache is critical | No KV pressure (seq_len=3) |

The winning strategy: architecture-level changes (64d ALAL > 192d ViT), then eliminate overhead (batch ops, skip allocations), then optimize kernels last.
