# Synapse vs llama.cpp Benchmark Results
**Date**: 2026-03-23
**Hardware**: Apple M4, 19GB unified memory
**Model**: Qwen3-0.6B (596M params)

## Results

| Metric | Synapse (f32, CPU) | llama.cpp (BF16, Metal) | llama.cpp (Q4_K_M, Metal) |
|--------|-------------------|------------------------|--------------------------|
| Prefill tok/s (pp128) | ~5 | 5,368 | 5,518 |
| Decode tok/s (tg64) | ~0.3 | 82 | 173 |
| Model size | 1,938 MB | 1,138 MB (BF16) | 373 MB (Q4_K_M) |
| INT8 decode tok/s | ~0.3 (no speedup) | N/A | N/A |

## Gap Analysis

| Factor | Estimated impact |
|--------|-----------------|
| Metal GPU offload | ~100-500x |
| SIMD matmul (NEON/AMX) | ~10-20x |
| Memory-mapped weights | ~2-5x |
| True quantized compute kernels | ~2-4x |
| KV-cache + attention fusion | ~2-3x |

## Notes
- Synapse INT8 shows NO speedup over f32 — current quantization is "fake" (dequant back to f32)
- llama.cpp uses BLAS + Metal backend on M4
- Q4_K_M decode is 2.1x faster than BF16 decode in llama.cpp
- Full-scale benchmark used `--full-scale` flag (added during this session)
- GGUF models from unsloth/Qwen3-0.6B-GGUF
