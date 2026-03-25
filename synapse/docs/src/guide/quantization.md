# Quantization

Synapse supports multiple quantization strategies for reducing memory usage and increasing inference speed.

## INT8 Quantization

Convert a loaded f32 model to INT8 with per-channel symmetric quantization:

```rust
let engine = InferenceEngine::from_pretrained("/path/to/model")?;
engine.quantize(); // f32 → INT8
```

**Performance impact** on Qwen3-0.6B (Apple M5):

| Metric | f32 | INT8 | Speedup |
|--------|-----|------|---------|
| Prefill | 18 tok/s | 31 tok/s | 1.7x |
| Decode | 6.6 tok/s | 14.6 tok/s | 2.2x |
| vs baseline | 2.9x | 6.3x | -- |
| Memory | 2.3 GB | 0.6 GB | 4x reduction |

INT8 quantization uses Zig SIMD kernels that perform widening multiply-accumulate: `i8x4 -> i32x4` with 32 columns per iteration.

## Q4_0 GEMV Kernel

For 4-bit models, Synapse includes a native Q4_0 GEMV kernel that operates directly on packed nibbles:

- On-the-fly dequantization: nibble unpack, subtract zero-point, scale
- 1.8x faster than f32 GEMV on equivalent operations
- Used automatically when loading Q4_0 GGUF files

## GGUF Format Support

Synapse reads GGUF files with the following quantization types:

| Type | Bits | Description |
|------|------|-------------|
| F32 | 32 | Full precision |
| F16 | 16 | Half precision |
| Q8_0 | 8 | 8-bit block quantization |
| Q4_0 | 4 | 4-bit block quantization |
| Q4_1 | 4 | 4-bit with non-zero offset |
| Q4_K | 4 | K-quant 4-bit (super-blocks) |
| Q6_K | 6 | K-quant 6-bit (super-blocks) |

### Loading GGUF Models

```bash
cargo run --example qwen3_chat --release -- --model-dir /path/to/gguf/
```

The engine detects GGUF files automatically and uses the appropriate dequantization kernel for each layer.

## Matmul Benchmark Results

Isolated matrix multiply benchmarks (M=1 decode, N=2048, K=2048):

| Kernel | Time | Throughput | vs f32 |
|--------|------|------------|--------|
| f32 GEMV | 48 us | 175 GFLOP/s | 1.0x |
| INT8 GEMV | 22 us | 381 GFLOP/s | 2.2x |
| Q4_0 GEMV | 27 us | 311 GFLOP/s | 1.8x |

These numbers reflect the Zig SIMD kernels on Apple M5 NEON.
