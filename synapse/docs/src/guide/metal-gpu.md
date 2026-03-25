# Metal GPU

Synapse includes Metal GPU acceleration for macOS, targeting the prefill phase of inference.

## Enabling Metal

Build with the `metal` feature:

```bash
cargo build --release --features metal
```

At runtime, Synapse detects available Metal devices and automatically dispatches compatible operations to the GPU.

## Dispatch Strategy

Synapse uses a hybrid CPU/GPU approach:

- **Prefill (M > 1)**: GPU dispatch -- matrix multiplications benefit from GPU parallelism when processing multiple tokens at once
- **Decode (M = 1)**: CPU dispatch -- single-token GEMV is memory-bandwidth-bound, and the CPU's SIMD kernels avoid GPU launch overhead

This is controlled automatically. No user configuration is needed.

## Weight Caching

When Metal is enabled, Synapse:

1. Pre-transposes weight matrices to GPU-optimal layout on first use
2. Allocates persistent Metal buffers that stay resident across forward passes
3. Reuses these buffers for all subsequent computations

This avoids repeated CPU-to-GPU transfers and transposition overhead.

## Performance

On Qwen3-0.6B, Apple M5:

| Configuration | Prefill | Decode |
|---------------|---------|--------|
| CPU f32 | 18 tok/s | 6.6 tok/s |
| Metal f32 | 19 tok/s | 6.5 tok/s |
| CPU INT8 | 31 tok/s | 14.6 tok/s |
| Metal + INT8 | 30 tok/s | 14.6 tok/s |

Metal provides a modest prefill improvement for f32. Decode performance is equivalent since it falls back to CPU SIMD kernels.

## Limitations

- **No INT8 GPU path**: Quantized inference uses CPU SIMD kernels even with Metal enabled. The GPU path only handles f32 operations.
- **No decode acceleration**: Single-token decode is CPU-only by design (GEMV is memory-bound, not compute-bound).
- **macOS only**: Metal is not available on Linux or Windows.
- **Memory**: GPU buffers add to total memory usage. For large models, monitor Metal memory with `sudo powermetrics --samplers gpu_power`.
