# Architecture Overview

Synapse is organized as a Rust workspace with 9 crates, plus a Zig library for SIMD kernels.

## Crate Structure

```
synapse-inference    <- main inference engine
├── synapse-core     <- FFI wrappers for Zig
│   └── synapse-sys  <- raw C bindings + auto-rebuild
synapse-nn           <- neural network modules
synapse-autograd     <- automatic differentiation
synapse-optim        <- optimizers + schedulers
synapse-data         <- data loading
synapse-graph        <- graph IR + optimization
synapse-train        <- training loop
```

**synapse-inference** is the primary crate for running models. It depends on `synapse-core` for SIMD operations and `synapse-nn` for layer definitions.

**synapse-sys** contains the raw C FFI bindings to the Zig library. Its `build.rs` script automatically invokes `zig build -Doptimize=ReleaseFast` and links the resulting static library.

**synapse-core** wraps the raw FFI in safe Rust APIs: `matmul()`, `attention()`, `rope()`, `rms_norm()`, `silu()`.

## Data Flow

The inference pipeline follows this path:

```
HuggingFace checkpoint
    │
    ├── config.json          -> ModelConfig (architecture params)
    ├── tokenizer.json       -> Tokenizer (BPE/SentencePiece)
    ├── tokenizer_config.json -> ChatTemplate (Jinja2)
    └── model.safetensors    -> Weight Mapper -> CausalLM layers
                                                    │
                                            KV Cache (per-layer)
                                                    │
                                          GenerationPipeline
                                                    │
                                              Token output
```

For GGUF files, the weight loading path reads quantized tensors directly and uses the appropriate dequantization kernel.

## Zig SIMD Layer

The Zig library (`zig/`) provides performance-critical operations:

| Function | Purpose |
|----------|---------|
| `sgemm` / `sgemmTiled` | f32 matrix multiply (tiled) |
| `int8Gemv` | INT8 GEMV with widening accumulation |
| `q4Gemv` | Q4_0 GEMV with on-the-fly dequant |
| `fusedAttention` | Tiled Q*K^T -> softmax -> *V |
| `ropeRotary` | Rotary position embeddings |
| `rmsNorm` | RMS normalization |
| `siluMul` | SwiGLU activation (SiLU * gate) |

All functions dispatch to NEON (ARM) or AVX2 (x86) intrinsics at compile time. The Zig compiler handles cross-platform SIMD without runtime detection.

## Layer Execution Order

A single transformer block executes:

1. `rms_norm` on input
2. Q/K/V projections (matmul)
3. `rope` on Q and K
4. `fused_attention` (Q*K^T, softmax, *V) with KV cache update
5. Output projection (matmul)
6. Residual add
7. `rms_norm` on intermediate
8. Gate + up projections (matmul)
9. `silu_mul` activation
10. Down projection (matmul)
11. Residual add
