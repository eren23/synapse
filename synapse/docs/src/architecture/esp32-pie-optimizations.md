# ESP32-P4 PIE Optimizations Deep-Dive

This document details every optimization technique implemented for LEWM inference on the ESP32-P4, including PIE SIMD assembly, dual-core parallelism, memory hierarchy exploitation, and quantization-aware compute.

## Optimization Summary

| # | Optimization | Where Applied | Speedup | Status |
|---|-------------|---------------|---------|--------|
| 1 | PIE INT8 GEMV | All linear layers | 5-10x per layer | Done |
| 2 | PIE Q4 Block Dot | Predictor layers | 5-8x per layer | Done |
| 3 | PIE Attention QK^T | Encoder attention | 3-5x | Done |
| 4 | GELU Lookup Table | Encoder + predictor FFN | 4-8x | Done |
| 5 | Tiled GEMV | Encoder GEMV | ~8% cache improvement | Done |
| 6 | PSRAM 200 MHz | All weight access | 10x bandwidth | Done |
| 7 | Dual-Core Attention | Encoder attention | ~20% layer speedup | Done |
| 8 | Dual-Core FFN | Encoder FFN | -150 ms/layer (projected) | Roadmap |
| 9 | Shared QKV Quantization | Encoder QKV | -10 ms/layer | Done |
| 10 | Compiler -O2 | Everything | -50 ms/layer (projected) | Roadmap |
| 11 | Batch INT8 Patch Embedding | Encoder patch embed | 9.4x (470->50ms) | Done |
| 12 | Kernel-Trick Linear Attention | Hybrid L blocks | 1.3x (76->58ms) | Done |
| 13 | Meta Token Support | Hybrid encoder | N/A (architectural) | Done |
| 14 | Encoder Output Projection | Hybrid encoder | N/A (architectural) | Done |

Combined result on 96d slim: **81,818 ms -> 6,416 ms encode** (12.8x speedup).

Combined result on 64d hybrid ALAL: **6,416 ms -> 922 ms encode** (7.0x additional speedup).

Total from baseline to optimized hybrid: **81,818 ms -> 922 ms** (88.7x speedup).

## PIE SIMD Architecture

### Registers

| Register | Width | Purpose |
|----------|-------|---------|
| QR0-QR7 | 128-bit | Vector registers (16x int8, 8x int16, 4x int32/f32) |
| QACC_H, QACC_L | 256-bit | Multiply-accumulate accumulators |
| ACCX (XACC) | 40-bit | Scalar accumulator for dot-product reduction |

### Key Instructions

```asm
esp.zero.xacc                      # Clear scalar accumulator
esp.vld.128.ip  QRn, Rs, imm      # Load 128-bit vector (16x int8), post-increment
esp.vst.128.ip  QRn, Rs, imm      # Store 128-bit vector
esp.vmulas.s8.xacc  QR0, QR1      # 16-wide signed INT8 MAC, sum into XACC
esp.vmulas.s8.xacc.ld.ip QR0, Rs, imm, QR0, QR1
                                    # Fused: MAC + load next vector (pipelined)
esp.movx.r.xacc.l  Rd             # Read XACC low 32 bits to general register
```

The critical instruction is `esp.vmulas.s8.xacc.ld.ip` -- it performs a 16-wide INT8 multiply-accumulate AND loads the next 16 elements in a single instruction, keeping the pipeline full.

### Performance Characteristics

- 16 INT8 multiply-accumulate operations per cycle at 400 MHz
- 128-bit alignment required for optimal vector loads/stores
- 40-bit XACC accumulator can hold: 256 elements x 127 x 127 = 4.1M (safely within 2^39 = 549B)
- Must chunk >256-element dot products to avoid overflow

## Optimization 1: PIE INT8 GEMV

**File**: `pie_gemv.c` / `pie_gemv.h`

The core PIE dot product processes 16 INT8 elements per cycle:

```c
static int32_t pie_dot_chunk(const int8_t *a, const int8_t *b, size_t len16) {
    int32_t partial;
    const int8_t *a_ptr = a;
    const int8_t *b_ptr = b;
    size_t count = len16;

    asm volatile(
        "esp.zero.xacc\n"                           // Clear accumulator
        "esp.vld.128.ip q0, %[ap], 16\n"            // Load first 16 bytes of a
        "esp.vld.128.ip q1, %[bp], 16\n"            // Load first 16 bytes of b
        "addi %[n], %[n], -16\n"
        "beqz %[n], 2f\n"                           // Skip loop if only 16 elements
        "1:\n"
        "esp.vmulas.s8.xacc.ld.ip q0, %[ap], 16, q0, q1\n"
                                                     // MAC(q0,q1) + load next a
        "esp.vld.128.ip q1, %[bp], 16\n"            // Load next b
        "addi %[n], %[n], -16\n"
        "bnez %[n], 1b\n"                           // Loop
        "2:\n"
        "esp.vmulas.s8.xacc q0, q1\n"              // Final MAC (no load)
        "esp.movx.r.xacc.l %[res]\n"               // Extract result
        : [ap] "+r" (a_ptr),
          [bp] "+r" (b_ptr),
          [n] "+r" (count),
          [res] "=r" (partial)
        :
        : "memory"
    );
    return partial;
}
```

**Chunking for overflow safety**: Dot products >256 elements are split into chunks of 256, each with its own `esp.zero.xacc` / accumulate / extract cycle:

```c
int32_t pie_dot_int8(const int8_t *a, const int8_t *b, size_t len) {
    int32_t result = 0;
    size_t done = 0;
    while (done + 16 <= len) {
        size_t chunk = min(len - done, PIE_CHUNK_ELEMS);  // max 256
        chunk &= ~15U;  // round down to 16
        result += pie_dot_chunk(a + done, b + done, chunk);
        done += chunk;
    }
    // Scalar remainder for < 16 elements
    for (size_t i = done; i < len; i++)
        result += (int32_t)a[i] * (int32_t)b[i];
    return result;
}
```

**GEMV** builds on the dot product -- one dot per output feature:

```c
void pie_int8_gemv(
    const int8_t *row_quant,      // [in_features] quantized input
    const int8_t *weights_t,      // [out_features][in_features] transposed weights
    size_t out_features,
    size_t in_features,
    int32_t *out_i32              // [out_features] accumulator output
) {
    for (size_t j = 0; j < out_features; j++) {
        out_i32[j] = pie_dot_int8(
            row_quant,
            weights_t + j * in_features,
            in_features
        );
    }
}
```

**Weight transpose**: Weights are stored as `[in][out]` in the model file but transposed to `[out][in_padded]` at load time for cache-friendly sequential access during GEMV:

```c
void transpose_int8_weights(
    const int8_t *src,           // [in_features][out_features]
    size_t in_features,
    size_t out_features,
    size_t in_features_padded,   // rounded up to multiple of 16
    int8_t *out_t                // [out_features][in_features_padded]
);
```

## Optimization 2: PIE Q4 Block Dot Product

**File**: `pie_gemv.c`

Q4 weights are stored as nibble-packed bytes. Each block of 32 weights shares one scale factor. The dot product:

1. **Unpack** 16 nibble bytes into 32 INT8 values `[-8, 7]`
2. **Quantize** 32 input floats to INT8 (per-block dynamic quantization)
3. **PIE dot** on 32 INT8 elements (2 iterations of 16-wide MAC)
4. **Scale** result by input_scale * weight_scale

```c
float pie_q4_block_dot(
    const uint8_t *nibbles,     // 16 packed bytes = 32 Q4 weights
    float weight_scale,
    const float *input,         // 32 input floats
    size_t valid_count
) {
    // Step 1: Unpack nibbles to INT8
    int8_t w_i8[32] __attribute__((aligned(16)));
    for (size_t i = 0; i < 16; i++) {
        w_i8[i * 2]     = (nibbles[i] & 0x0F) - 8;
        w_i8[i * 2 + 1] = (nibbles[i] >> 4) - 8;
    }

    // Step 2: Dynamic quantization of input
    int8_t x_i8[32] __attribute__((aligned(16)));
    float max_abs = 0.0f;
    for (size_t i = 0; i < count; i++) {
        float a = fabsf(input[i]);
        if (a > max_abs) max_abs = a;
    }
    float x_scale = max_abs / 127.0f;
    for (size_t i = 0; i < count; i++)
        x_i8[i] = (int8_t)roundf(input[i] / x_scale);

    // Step 3: PIE INT8 dot product (32 = 2x16)
    int32_t dot = pie_dot_int8(w_i8, x_i8, 32);

    // Step 4: Rescale
    return (float)dot * x_scale * weight_scale;
}
```

## Optimization 3: PIE Attention QK^T

The encoder's bidirectional attention computes a 257x257 score matrix. Each element is a dot product between a query vector and key vector, computed via PIE:

```c
for (size_t head = 0; head < num_heads; ++head) {
    for (size_t q_token = q_start; q_token < q_end; ++q_token) {
        const int8_t *qi = q_i8 + q_token * row_bytes + head * head_dim_padded;
        float q_sc = q_scales[q_token * num_heads + head];

        for (size_t k_token = 0; k_token < seq_len; ++k_token) {
            const int8_t *ki = k_i8 + k_token * row_bytes + head * head_dim_padded;
            float k_sc = k_scales[k_token * num_heads + head];

            // PIE 16-wide INT8 dot product
            int32_t dot_i32 = pie_dot_int8(qi, ki, head_dim_padded);
            scores[k_token] = (float)dot_i32 * q_sc * k_sc * inv_sqrt_hd;
        }

        softmax_inplace(scores, seq_len);

        // Score-weighted V summation (still f32 -- V quantization is roadmap)
        for (size_t k_token = 0; k_token < seq_len; ++k_token) {
            float s = scores[k_token];
            for (size_t d = 0; d < head_dim; ++d)
                out_head[d] += s * v_row[d];
        }
    }
}
```

Q and K vectors are pre-quantized to INT8 (per-head scale), enabling PIE for the dominant O(n^2) computation.

## Optimization 4: GELU Lookup Table

GELU activation uses `tanhf()` which is expensive on RISC-V without a hardware transcendental unit. Replaced with a 1024-entry precomputed lookup table:

```c
// Precomputed at init time, stored in 8KB TCM for zero-wait access
static float gelu_lut[1024];

void init_gelu_lut(void) {
    for (int i = 0; i < 1024; i++) {
        float x = (float)(i - 512) / 128.0f;  // Range: [-4.0, 3.99]
        gelu_lut[i] = 0.5f * x * (1.0f + tanhf(0.7978846f * (x + 0.044715f * x * x * x)));
    }
}

float gelu_scalar(float x) {
    int idx = (int)(x * 128.0f + 512.0f);
    if (idx < 0) return 0.0f;
    if (idx >= 1024) return x;  // GELU(x) ~ x for large x
    return gelu_lut[idx];
}
```

**Speedup**: 4-8x per FFN layer. The LUT fits in the 8KB TCM (Tightly Coupled Memory) which has zero-wait-state access.

## Optimization 5: Tiled GEMV (Weights-Outer Loop)

Standard GEMV iterates: for each output j, dot(input, weights[j]). This means the input vector is re-read from cache for every output.

**Tiled GEMV** (weights-outer loop) reverses the loop order to maximize PSRAM cache reuse:

```
Standard: for j in outputs: dot(input, weights[j])
  -> input re-read from cache N times
  -> weights streamed once each (good)

Tiled: for tile in weight_tiles: accumulate(input_tile, weight_tile, output)
  -> weight rows stay in L2 cache (768 KB) while processing all tokens
  -> sequential PSRAM access pattern enables hardware prefetch
```

This is especially important for the encoder where seq_len=257 tokens all share the same weight matrix. Keeping weights in L2 cache across all 257 tokens avoids repeated PSRAM fetches.

**Impact**: ~8% latency reduction on encoder layers (measured: 13,538 ms -> 12,482 ms for full model encode).

## Optimization 6: PSRAM 200 MHz

The Waveshare ESP32-P4 board has 32 MB PSRAM that can run at 200 MHz in HEX (16-line) mode:

```ini
CONFIG_SPIRAM=y
CONFIG_SPIRAM_MODE_OCT=y
CONFIG_SPIRAM_SPEED_200M=y
CONFIG_IDF_EXPERIMENTAL_FEATURES=y  # REQUIRED!
```

**Without this**: PSRAM defaults to 20 MHz, making all weight accesses 10x slower.

**Gotcha**: `SPIRAM_SPEED_200M` depends on `CONFIG_IDF_EXPERIMENTAL_FEATURES=y` in ESP-IDF v5.4's Kconfig. Without the experimental flag, the speed setting is **silently ignored** and falls back to 20 MHz. There is no warning in the build log.

**Impact**: 81,818 ms -> 70,913 ms encode at 200 MHz (before any other optimizations).

## Optimization 7: Dual-Core Attention

The ESP32-P4 has two RISC-V cores. The encoder attention computation (257 query tokens x 257 key tokens) is split across both cores:

```c
// Core 0 handles first half of query tokens
AttnWorkItem work0 = { .q_start = 0, .q_end = mid, ... };
// Core 1 handles second half
AttnWorkItem work1 = { .q_start = mid, .q_end = seq_len, ... };

// Dispatch to Core 1 (runs in parallel)
core1_dispatch(attn_compute_range_wrapper, &work1);
// Core 0 computes its range
attn_compute_range(&work0);
// Wait for Core 1
core1_wait();
```

The worker task is pinned to Core 1 via FreeRTOS:

```c
static void core1_worker(void *arg) {
    for (;;) {
        xSemaphoreTake(s_core1_start, portMAX_DELAY);
        if (s_core1_fn)
            s_core1_fn((void *)s_core1_arg);
        xSemaphoreGive(s_core1_done);
    }
}

void dual_core_init(void) {
    s_core1_start = xSemaphoreCreateBinary();
    s_core1_done = xSemaphoreCreateBinary();
    xTaskCreatePinnedToCore(core1_worker, "core1", 8192, NULL, 5, NULL, 1);
}
```

**Synchronization**: Binary semaphores for start/done signaling. Core 0 gives `start`, does its own work, then takes `done`.

**Impact**: Slim encoder 7,950 ms -> 6,416 ms (19% improvement). The attention block is 45.8% of layer time, so splitting it across 2 cores saves ~20% of that.

## Optimization 8: Dual-Core FFN (Roadmap)

Same fork-join pattern as attention, applied to the FFN up-projection. Split output features across cores:

```c
// Core 0: features [0, out/2)
// Core 1: features [out/2, out)
GemvWorkItem work1 = { .j_start = out_f/2, .j_end = out_f, ... };
core1_dispatch(gemv_compute_range, &work1);
gemv_compute_range(&work0);  // Core 0's half
core1_wait();
```

The infrastructure is already implemented (`GemvWorkItem`, `gemv_compute_range`). Target: -150 ms per encoder layer.

## Optimization 9: Shared QKV Quantization

See the [LEWM ESP32 guide](../guide/lewm-esp32.md#shared-qkv-quantization-optimization) for details. Quantizes the encoder input once for Q, K, and V projections instead of three times. Saves ~10 ms per layer.

## Self-Tests

Every boot runs 4 PIE validation tests comparing PIE results against scalar reference:

| Test | Elements | Purpose |
|------|----------|---------|
| test_32 | 32 | 2 PIE iterations, basic correctness |
| test_192 | 192 | Typical hidden dim (12 iterations) |
| test_768 | 768 | Encoder inter dim, heap-allocated + aligned |
| test_q4 | 32 | Q4 unpack + quantize + PIE dot, 5% tolerance |

```
I (pie-test) OK test_32: result=-2720
I (pie-test) OK test_192: result=-90566
I (pie-test) OK test_768: result=119888
I (pie-test) OK test_q4: ref=-0.4625 pie=-0.4750 err=0.0125
I (pie-test) All PIE self-tests passed
```

## Benchmark Results

### Full Model (192d, 6 Encoder + 6 Predictor Layers)

| Stage | predict_next | encode(image) |
|-------|-------------|---------------|
| Scalar C, PSRAM 20 MHz | 3,037 ms | 81,818 ms |
| Scalar C, PSRAM 200 MHz | 3,009 ms | 70,913 ms |
| +PIE INT8/Q4 GEMV | 774 ms | 20,524 ms |
| +PIE attention + GELU LUT | 828 ms | 13,538 ms |
| +Tiled GEMV (weights-outer) | 828 ms | 12,482 ms |

### Slim Model (96d, 4 Encoder + 4 Predictor Layers)

| Stage | predict_next | encode(image) |
|-------|-------------|---------------|
| All PIE optimizations | 583 ms | 7,950 ms |
| **+Dual-core attention** | **583 ms** | **6,416 ms** |

### Encoder Layer Breakdown (Slim, Dual-Core)

| Component | Time | % of Layer |
|-----------|------|------------|
| LayerNorm | 12 ms | 1.2% |
| QKV projections (PIE INT8) | 105 ms | 10.5% |
| Attention (dual-core PIE QK^T) | 457 ms | 45.8% |
| O projection (PIE INT8) | 40 ms | 4.0% |
| FFN (PIE INT8 tiled + GELU LUT) | 384 ms | 38.5% |
| **Layer total** | **998 ms** | |

### Speedup Summary

| Model | Operation | Baseline | Optimized | Speedup |
|-------|-----------|----------|-----------|---------|
| Full 192d | predict_next | 3,037 ms | 828 ms | **3.7x** |
| Full 192d | encode | 81,818 ms | ~10,000 ms | **8.2x** |
| Slim 96d | predict_next | ~2,332 ms | 583 ms | **4.0x** |
| Slim 96d | encode | ~12,832 ms | 6,416 ms | **2.0x** (from PIE baseline) |

## Performance Roadmap

### Kernel-Level Optimizations

| Optimization | Target Savings/Layer | Complexity | Description |
|-------------|---------------------|------------|-------------|
| FFN dual-core | -150 ms | Easy | Same fork-join as attention. Split 768 output features. |
| V weighting PIE | -100 ms | Medium | Quantize V per head to INT8, PIE dot for scores x V weighted sum. |
| Compiler -O2 | -50 ms | Easy | `CONFIG_COMPILER_OPTIMIZATION_PERF=y`. Currently `-Og`. |
| DMA weight prefetch | -50 ms | Hard | GDMA-AXI prefetch next layer while computing current. |
| Shared QKV quantization | -10 ms | Easy | Already done. |
| **Combined** | **-360 ms/layer** | | **~5,000 ms encode (projected)** |

### Model-Level Optimizations

| Optimization | Target | Description |
|-------------|--------|-------------|
| 48d/2e/2p slim | ~3s encode, ~300ms predict | Half the layers. Needs W&B export. |
| Patch pruning | 4x fewer attention ops | Keep 128 of 256 patches. No retraining needed. |
| Linear attention | Eliminate O(n^2) bottleneck | Replace full attention with kernel approximation. Needs retraining. |

### The Attention Bottleneck

Attention consumes 45.8% of encoder layer time because it's O(n^2) in sequence length (257x257 = 66,049 dot products per head).

**Patch Pruning** (reduce n, no retraining):
- Score each of the 256 patches by L2 norm or attention weight after the first layer
- Keep top-K patches (K=128 or K=64) + CLS token
- Attention drops from 257^2 = 66K to 129^2 = 16.6K (4x fewer) or 65^2 = 4.2K (16x fewer)
- Risk: quality degradation if important spatial info is in "boring" patches

**Linear Attention** (reduce O(n^2) to O(n), requires retraining):
- Replace softmax(QK^T)V with kernel approximation: phi(Q) * (phi(K)^T * V)
- Changes computation from O(n^2 * d) to O(n * d^2), which is better when n > d (257 > 64)
- The codebase already has RWKV-7 and DeltaNet implementations that could inform the design

## Reference Implementations

- **esp-nn** ([github.com/espressif/esp-nn](https://github.com/espressif/esp-nn)): 57% PIE assembly, INT8 conv/fc/pool kernels
- **esp-dl** ([github.com/espressif/esp-dl](https://github.com/espressif/esp-dl)): Higher-level NN framework using `.espdl` model format
- **ESP32-P4 PIE blog**: [developer.espressif.com/blog/2024/12/pie-introduction/](https://developer.espressif.com/blog/2024/12/pie-introduction/)
