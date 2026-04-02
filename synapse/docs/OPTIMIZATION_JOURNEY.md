# Synapse Optimization Journey — 2-50M Edge Models

Last updated: 2026-04-01

## Summary

This document tracks the optimization work for running LEWM world models on ESP32-P4, from the initial 96d deployment through the hybrid ALAL architecture and kernel-level optimizations. Total speedup: **5.2x on encode + 3-step predict** (7.2s → 1.4s).

## Architecture Evolution

### Phase 1: 96d Slim → 64d Baseline (architecture change)

Deployed three new 64d_4e_4p checkpoints (baseline, hybrid ALAL, elastic) to ESP32-P4.

**Converter improvements** (`scripts/convert_lewm_ckpt.py`):
- Added Lightning checkpoint state_dict extraction (skips optimizer states)
- Added hybrid ALAL encoder key remapping (blocks.N → ViT naming, fused QKV split)
- Added BatchNorm folding for encoder output projection
- Added `encoder_type` and `meta_tokens` to config inference

| Metric | 96d slim | 64d baseline | 64d hybrid ALAL |
|--------|----------|-------------|-----------------|
| Params | ~8M | 10.2M | 3.0M |
| Binary | 9.8 MB | 10.9 MB | 3.9 MB |
| predict_next | 583 ms | 443 ms | 152 ms |
| encode | 6,416 ms | 4,198 ms | 1,392 ms |

### Phase 2: Linear Attention for L Blocks

The hybrid ALAL encoder alternates full-attention (A) and linear-attention (L) blocks. L blocks (1, 3) have separate Q/K/V without bias.

**Step 1 — Skip softmax**: Replaced softmax with L1 normalization on L blocks. Saved 16ms per L block (92→76ms).

**Step 2 — Kernel-trick O(nd²)**: Replaced the full O(n²d) score matrix computation with the kernel-trick formulation using ELU+1 feature map:
```
KV = φ(K)^T @ V          [d,d] — computed once per head
out[q] = φ(Q[q]) @ KV    [d] — O(d²) per query
normalize by φ(Q) · Σφ(K)
```
Saved 18ms more per L block (76→58ms). Total L block: 178ms (was 211ms).

### Phase 3: PIE SIMD Batch Patch Embedding

The patch embedding was calling a **scalar triple-loop** `matmul_t_into` 256 times (once per 14×14 patch). This was 39% of total encode time.

**Fix**: Pre-extract all 256 patches into a contiguous `[256, 588]` buffer, quantize `patch_proj` to INT8 at load time, run one batched INT8 GEMM with PIE SIMD + dual-core parallelism.

Result: **470ms → 50ms** (9.4x faster).

### Phase 4: Host-side Optimizations (Zig + Rust)

**Q4 GEMV dispatch** (`q4_linear.rs`):
- Wired `Q4Linear::forward()` to Zig `q4_0GemvRow` via `synapse_core::q4_0_gemv`
- Added cached f32→f16 scale conversion (`pack_for_zig()`)
- Q4 predict: 3310us → 983us (3.4x faster)

**INT8 benchmark** (was claimed "zero speedup"):
- Actually already 5-9x faster than f32 for M=1 GEMV — claim was outdated
- At 192d M=257, f32 (Apple Accelerate) beats INT8 (Zig) on macOS — expected

**Arena allocator** (`LeWMBuffers`):
- Added `q_split`, `k_split`, `v_split`, `mod_copy`, `latent_seq`, `cond` buffers
- Eliminates ~40 Vec allocations per predict_next_fused call
- Correct (cos=1.0) but fused path needs Zig SIMD internally to beat Accelerate on macOS
- Win on ESP32/WASM where malloc is expensive

**Fused Zig layer kernel** (`zig/src/ops/fused_lewm_layer.zig`):
- Single function per predictor layer — all matmul/attention/norm/GELU in one call
- Uses Zig `sgemmTiled` internally (no FFI overhead between ops)
- On macOS: 1680us (slower than 341us non-fused Accelerate path)
- On ESP32/WASM: will be the fast path (no Accelerate available)

## ESP32-P4 Performance Timeline

| Step | predict | encode | enc + 3 predict | Binary |
|------|---------|--------|-----------------|--------|
| 96d slim (start) | 583 ms | 6,416 ms | 7,165 ms | 9.8 MB |
| 64d baseline | 443 ms | 4,198 ms | 5,527 ms | 10.9 MB |
| 64d hybrid ALAL | 152 ms | 1,392 ms | 1,852 ms | 3.9 MB |
| + skip softmax L blocks | 152 ms | 1,364 ms | 1,824 ms | 3.9 MB |
| **+ PIE batch patch + kernel-trick** | **152 ms** | **922 ms** | **1,382 ms** | **3.9 MB** |

**Total improvement: 5.2x** (7,165ms → 1,382ms)

## Encoder Layer Breakdown (Final, Hybrid ALAL)

| Layer | Type | norm | qkv | attn | oproj | ffn | total |
|-------|------|------|-----|------|-------|-----|-------|
| 0 | A (softmax) | 2ms | 12ms | 94ms | 7ms | 100ms | 215ms |
| 1 | L (kernel-trick) | 2ms | 11ms | 58ms | 7ms | 100ms | 178ms |
| 2 | A (softmax) | 2ms | 12ms | 94ms | 7ms | 100ms | 215ms |
| 3 | L (kernel-trick) | 2ms | 11ms | 58ms | 7ms | 100ms | 178ms |

Plus: patch embed 50ms, overhead 86ms.

## Quality Metrics

| Model | Rollout cos(step_i, step_{i-1}) | z L2 | 20-step drift |
|-------|--------------------------------|------|---------------|
| 64d baseline | 0.9999 | 0.902 | 8.2% |
| 64d elastic | 0.9999 | 1.198 | 14.5% |
| 64d hybrid ALAL | 0.9997 | 0.282 | 192% (low initial, converges) |

All models: INT8+Q4 quantization preserves cos > 0.996 vs f32.

The hybrid's "192% drift" is misleading — the endpoint L2 (0.82) matches baseline (0.83). The initial encoding starts at lower scale (0.28 vs 0.90), but the predictor converges to the same magnitude within a few steps. Step-to-step smoothness is equivalent across all variants.

## Files Changed (branch: `imp/slimmer-esp`)

| File | What changed |
|------|-------------|
| `scripts/convert_lewm_ckpt.py` | Lightning state_dict, hybrid key remapping, BN folding |
| `examples/export_lewm_q4.rs` | meta_tokens in LQ40 config, hybrid weight export |
| `examples/lewm_compare_variants.rs` | New: multi-variant host comparison with per-step drift |
| `examples/bench_int8_vs_f32.rs` | New: INT8 vs f32 GEMV benchmark |
| `examples/bench_lewm_quantized.rs` | New: f32 vs INT8+Q4 vs fused LEWM benchmark |
| `crates/synapse-inference/src/models/vision/vit.rs` | meta_token, enc_proj, linear attention dispatch |
| `crates/synapse-inference/src/models/vision/lewm.rs` | Arena buffers, hybrid weight loading, fused Zig dispatch |
| `crates/synapse-inference/src/ops/attention.rs` | `bidirectional_linear_attention` (L1-normalized) |
| `crates/synapse-inference/src/quantization/primitives/q4_linear.rs` | Zig Q4 GEMV dispatch + cached packing |
| `crates/synapse-inference/src/quantization/vision/full_q_lewm.rs` | Hybrid fields (meta_token, enc_proj) |
| `crates/synapse-core/src/lib.rs` | `lewm_predict_layer` FFI wrapper |
| `crates/synapse-sys/src/lib.rs` | `syn_lewm_predict_layer` FFI binding |
| `zig/src/ops/fused_lewm_layer.zig` | New: fused Zig predictor layer kernel |
| `zig/src/ffi/exports.zig` | `syn_lewm_predict_layer` export |
| `zig/src/root.zig` | Register fused_lewm_layer module |
| `synapse-esp32/esp-idf-app/main/app_main.c` | Hybrid encoder, kernel-trick attention, PIE batch patch embed |

## Phase 5: Fused Multi-Step Rollout

Fused rollout reduces latency for N-step predictions by running all predictor layers **once** over an N×3-token fused sequence `[z, a₀, 0, z, a₁, 0, ...]` instead of N sequential predictor passes. Bidirectional attention across all steps enables parallel future hypotheses.

**ESP32-P4 results (slim 96d):**

| Steps | Sequential | Fused | Speedup |
|-------|-----------|-------|---------|
| 3 | 462 ms | 279 ms | **1.66x** |

**Host results (slim 96d Q4):**

| Steps | Sequential | Fused | Speedup |
|-------|-----------|-------|---------|
| 3 | 66 ms | 53 ms | **1.25x** |
| 10 | 170 ms | 156 ms | **1.09x** |

**Accuracy:** Step-0 of fused matches sequential exactly (cos_sim = 1.000). Steps 1+ differ by design — fused uses bidirectional attention across all steps while sequential is strictly autoregressive.

**Limitation:** ESP32 firmware limited to **3 steps max** due to `MAX_PREDICTOR_SEQ_LEN=9` hard cap.

## Remaining Optimization Targets

### Parked (Phase 2 — do next)
- **WASM SIMD128**: Port GEMV to `std::arch::wasm32` intrinsics (3-5x for browser)
- **Compile-time dimension specialization**: Zig `comptime K` variants for 64d/192d

### Future
- **INT8 encoder on macOS**: Currently slower than f32 Accelerate at 192d; needs NEON `sdot`
- **Q4 integer accumulation**: Zig Q4 kernel still dequants nibbles to f32; native i8 path would close the 2.7x gap vs f32
- **Metal SSM shaders**: Mamba/RWKV stuck on CPU; Metal selective_scan + wkv7 would unlock GPU
- **2-layer encoder variant**: Train 64d_2e_4p for encode under 600ms on ESP32
