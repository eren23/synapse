# Zig SIMD Kernels

The `zig/` directory contains hand-optimized SIMD kernels that provide the core compute for Synapse's inference engine.

## f32 GEMV

The f32 general matrix-vector multiply uses:

- **8 x F32x4 accumulators** -- 8 independent vector accumulators to saturate the execution units and hide latency
- **4x K-unrolled inner loop** -- processes 4 elements of the K dimension per iteration, reducing loop overhead
- **`@mulAdd` intrinsic** -- fused multiply-add maps directly to NEON `fmla` or AVX2 `vfmadd` instructions
- Tail handling for K dimensions not divisible by the unroll factor

```
for each row of A (output dimension):
    acc[0..7] = zero
    for k in 0..K step 4:
        load 4 x F32x4 from A[row, k..k+16]
        load 4 x F32x4 from B[k..k+16]
        acc[i] = @mulAdd(a[i], b[i], acc[i])
    result[row] = horizontal_sum(acc[0..7])
```

## INT8 GEMV

The INT8 kernel operates on quantized weights with f32 activations:

- **I8x4 -> I32x4 widening** -- loads 4 int8 values, widens to int32 for accumulation without overflow
- **32 columns per iteration** -- processes 32 output columns simultaneously
- **Per-channel scale factors** -- each output channel has an f32 scale applied after the integer dot product
- Final conversion: `i32_accumulator * scale_per_channel = f32_result`

This achieves 2.2x throughput over f32 GEMV on the same hardware.

## Q4_0 GEMV

The 4-bit kernel handles GGUF Q4_0 packed weights:

- **Nibble unpacking** -- each byte contains two 4-bit weights; the kernel extracts high/low nibbles with shift and mask
- **On-the-fly dequantization** -- `(nibble - 8) * block_scale` converts to approximate f32
- **Block structure** -- Q4_0 uses 32-element blocks, each with a shared f16 scale factor
- 1.8x faster than f32 GEMV for equivalent matrix dimensions

## Fused Attention

The fused attention kernel computes `softmax(Q * K^T / sqrt(d)) * V` in a single pass:

- **Tiled Q processing** -- TILE_Q = 32 query rows processed at a time
- **Online softmax** -- computes softmax incrementally across K chunks using the log-sum-exp trick, avoiding a separate normalization pass
- **Internal `sgemmTiled`** -- reuses the tiled GEMM kernel for both Q*K^T and attn*V multiplications
- KV cache integration: reads K and V directly from the cache buffer

This avoids materializing the full attention matrix for long sequences.

## Platform Dispatch

The Zig compiler resolves SIMD targets at compile time:

| Platform | ISA | Vector Width |
|----------|-----|-------------|
| Apple Silicon (M1-M5) | NEON | 128-bit (F32x4) |
| x86_64 | AVX2 | 256-bit (F32x8) |

There is no runtime feature detection. The Zig build system compiles for the target architecture, and `build.rs` passes the appropriate flags.

## Source Files

Key files in `zig/src/`:

- `ops/matmul.zig` -- f32 GEMM/GEMV kernels
- `ops/qmatmul.zig` -- INT8 and Q4_0 quantized kernels
- `ops/attention.zig` -- fused attention
- `ops/rope.zig` -- rotary position embeddings
- `ops/rms_norm.zig` -- RMS normalization
- `ffi/exports.zig` -- C FFI export declarations
