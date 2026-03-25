# Metal GPU

Synapse includes Metal GPU acceleration for macOS, targeting Apple Silicon's unified memory architecture.

## Enabling Metal

Build with the `metal` feature:

```bash
cargo build --release --features metal
```

At runtime, Synapse detects available Metal devices and automatically dispatches operations to the GPU.

## 3-Tier Architecture

The Metal backend evolved through three tiers, each building on the previous:

### Tier 1: Batched FFN (Phase 1)

The first Metal integration batched Q/K/V projections and FFN matmuls into two command buffers per layer, reducing commit+wait from 4 to 2 per layer. Attention stayed on CPU.

- **Decode throughput**: 8.8 tok/s
- **Approach**: GPU matmul for prefill (M > 1), CPU for decode (M = 1)
- **Limitation**: Still 2 CPU-GPU round-trips per layer (56 total for 28 layers)

### Tier 2: All-Layers Single Command Buffer (Phase 3)

Phase 3 encodes all 28 decoder layers into a single Metal command buffer. Pre-uploaded weight buffers, persistent scratch memory, and a GPU-resident KV cache eliminate all CPU-GPU round-trips during decode.

- **Decode throughput**: 8.0 tok/s (f32)
- **Approach**: One `commit + waitUntilCompleted` per token, all layers GPU-native
- **Limitation**: f32 weights (4 bytes per param) are bandwidth-bound on unified memory

### Tier 3: INT8 GPU GEMV (Current)

The current tier adds INT8 quantized GEMV kernels. Weights are automatically quantized to INT8 with per-column f32 scales at load time, reading 1 byte per weight instead of 4.

- **Decode throughput**: 14.5 tok/s
- **Approach**: `gemv_int8` kernel with 4x bandwidth reduction
- **11 Metal shaders**: matmul, gemv, gemv_int8, rmsnorm, headwise_rmsnorm, silu/swiglu, attention, attention_decode, rope_rotate_half, kv_cache_scatter

## Why GPU Matches CPU on Unified Memory

On Apple Silicon, CPU and GPU share the same physical memory (100 GB/s on M5). Single-token decode is purely memory-bandwidth-bound (every weight is read once per token), so neither CPU nor GPU has a bandwidth advantage.

The GPU path's benefit comes from **INT8**: 4x fewer bytes per weight means 4x less bandwidth consumed. Combined with GPU dispatch (no per-kernel CPU overhead), this yields 14.5 tok/s -- matching the CPU INT8 path while keeping all state GPU-resident for future batched-decode extensions.

## Performance

On Qwen3-0.6B, Apple M5:

| Configuration | Prefill | Decode |
|---------------|---------|--------|
| CPU f32 | 18 tok/s | 6.6 tok/s |
| CPU INT8 | 31 tok/s | 14.6 tok/s |
| Metal f32 (Tier 2) | 19 tok/s | 8.0 tok/s |
| Metal INT8 (Tier 3) | 30 tok/s | 14.5 tok/s |

## Weight Caching and Quantization

When Metal is enabled, Synapse:

1. Transposes weight matrices to GPU-optimal layout `[K, N]` on first use
2. Quantizes each column to INT8: `scale[j] = max(|w[:,j]|) / 127`, `int8[k,j] = round(w[k,j] / scale[j])`
3. Uploads INT8 weights + f32 scales to persistent Metal buffers
4. Allocates GPU-resident KV cache (`[max_seq, kv_dim]` per layer)
5. Pre-allocates scratch buffers and constant buffers (eliminates ~1500 buffer allocations per token)

## Limitations

- **macOS only**: Metal is not available on Linux or Windows.
- **Single-token decode**: Batched decode (multiple sequences) is not yet implemented.
- **Memory**: GPU buffers add to total memory usage. For large models, monitor Metal memory with `sudo powermetrics --samplers gpu_power`.
- **Sequence length**: `attention_decode` kernel supports up to 4096 tokens (limited by threadgroup memory).
