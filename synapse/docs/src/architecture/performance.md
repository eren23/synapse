# Performance

All benchmarks measured on Qwen3-0.6B (596M parameters) running on Apple M5.

## End-to-End Throughput

| Configuration | Prefill (tok/s) | Decode (tok/s) | vs Baseline |
|---------------|-----------------|----------------|-------------|
| f32 CPU | 18 | 6.6 | 2.9x |
| INT8 CPU | 31 | 14.6 | 6.3x |
| Metal f32 | 19 | 8.0 | -- |
| Metal INT8 GPU | 30 | 14.5 | 6.3x |
| llama.cpp Q4_K_M | 5,518 | 173 | reference |

**Baseline**: 2.3 tok/s (initial unoptimized implementation).

The gap with llama.cpp is expected -- llama.cpp uses highly optimized Q4_K kernels with years of tuning, while Synapse prioritizes a clean modular architecture.

## Roofline Analysis

For Qwen3-0.6B decode (M=1, single-token GEMV):

- **Model size**: 596M params x 4 bytes = 2.3 GB (f32), 0.6 GB (INT8)
- **Memory bandwidth**: ~100 GB/s (Apple M5)
- **Theoretical max**: 100 GB/s / 2.3 GB = **43 tok/s** (f32)
- **Achieved**: 6.6 tok/s f32 = **15%** of roofline
- **INT8 theoretical**: 100 GB/s / 0.6 GB = **167 tok/s**
- **INT8 achieved**: 14.6 tok/s = **8.7%** of roofline

The gap to roofline comes from: attention computation, KV cache reads, non-matmul operations (norms, activations), and memory access patterns.

## GPU-Resident Decode (Metal INT8)

The Metal INT8 path keeps all state on GPU. Weight bandwidth per token:

| Precision | Weight bytes per token | Theoretical latency (100 GB/s) |
|-----------|----------------------|-------------------------------|
| f32 | 280 MB | 2.8 ms |
| INT8 | 70 MB | 0.7 ms |

**Actual decode latency**: ~71 ms per token (14.5 tok/s).

This is **10% of the bandwidth peak**. The remaining 90% is overhead from:
- Attention score computation and softmax over the KV cache (scales with sequence length)
- RMSNorm reductions (threadgroup barriers, not purely streaming)
- RoPE rotations (read-modify-write pattern)
- Metal command buffer encoding and commit overhead
- KV cache scatter writes (small but serialized)

The single command buffer design (all 28 layers in one `commit + waitUntilCompleted`) minimizes dispatch overhead but cannot hide the latency of non-matmul kernels.

## Isolated Kernel Benchmarks

Matrix multiply benchmarks (M=1, N=2048, K=2048):

| Kernel | Latency | Throughput | vs f32 |
|--------|---------|------------|--------|
| f32 GEMV | 48 us | 175 GFLOP/s | 1.0x |
| INT8 GEMV | 22 us | 381 GFLOP/s | 2.2x |
| Q4_0 GEMV | 27 us | 311 GFLOP/s | 1.8x |

## Optimization History

Key milestones in the optimization journey:

| Phase | Decode (tok/s) | Change |
|-------|---------------|--------|
| Initial | 2.3 | Naive Rust matmul |
| Zig GEMV | 4.1 | SIMD vectorization |
| Tiled attention | 5.8 | Fused Q*K*V |
| INT8 quantization | 14.6 | Quantized kernels |
| Metal INT8 GPU | 14.5 | GPU-resident decode |

## Running Benchmarks

Full benchmark suite:

```bash
bash scripts/bench_suite.sh --model-dir /path/to/qwen3-0.6b
```

Isolated matmul benchmarks:

```bash
cargo bench -p synapse-core
```

Compare against llama.cpp:

```bash
bash bench_vs_llamacpp.sh /path/to/model.gguf
```
