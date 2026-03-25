# Synapse

A modular LLM inference engine built in Rust with Zig SIMD kernels and Metal GPU acceleration.

## Performance

Qwen3-0.6B on Apple M5:

| Config | Prefill | Decode | vs Baseline |
|--------|---------|--------|-------------|
| f32 CPU | 18 tok/s | 6.6 tok/s | 2.9x |
| INT8 CPU | 31 tok/s | 14.6 tok/s | **6.3x** |
| Metal + INT8 | 30 tok/s | 14.6 tok/s | 6.3x |
| llama.cpp Q4_K_M (ref) | 5518 tok/s | 173 tok/s | — |

## Features

- **SIMD-optimized kernels** — f32, INT8, and Q4 GEMV kernels with NEON/AVX2 dispatch
- **Metal GPU** — Accelerated prefill with weight caching (no per-call transpose)
- **5 model families** — Qwen3, LLaMA 3.2, Mistral, Phi-3, Gemma
- **GGUF + safetensors** — Loads Q4_0, Q4_1, Q4_K, Q6_K, Q8_0, F16, F32
- **Sharded checkpoints** — Models >5GB via `model.safetensors.index.json`
- **Chat templates** — minijinja-based, auto-loaded from `tokenizer_config.json`
- **KV cache** — Batched prefill + cached decode with sliding window support
- **Speculative decoding** — Self-speculative framework with KV cache rollback
- **Fused attention** — Tiled SIMD attention with online softmax
- **RoPE variants** — RotateHalf, Interleaved, Linear/Dynamic scaling
- **225+ tests** — Correctness, performance benchmarks, multi-architecture validation

## Quick Start

```bash
# Build (auto-rebuilds Zig kernels)
cargo build --release

# Download a model
huggingface-cli download Qwen/Qwen3-0.6B --local-dir /tmp/qwen3-0.6b

# Chat (f32)
cargo run --example qwen3_chat --release -- --model-dir /tmp/qwen3-0.6b

# Chat (INT8 quantized — 2x faster)
cargo run --example qwen3_chat --release -- --model-dir /tmp/qwen3-0.6b --quantize

# With Metal GPU (macOS)
cargo run --example qwen3_chat --release --features metal -- --model-dir /tmp/qwen3-0.6b --quantize
```

## Architecture

```
synapse/
├── crates/
│   ├── synapse-inference   # LLM inference: models, generation, quantization
│   ├── synapse-core        # FFI wrappers for Zig tensor ops
│   ├── synapse-sys         # Raw Zig C bindings (auto-rebuild)
│   ├── synapse-nn          # Neural network modules
│   ├── synapse-autograd    # Automatic differentiation
│   ├── synapse-optim       # Optimizers (SGD, Adam, RMSProp)
│   ├── synapse-data        # Data loading pipeline
│   ├── synapse-graph       # Graph IR + optimization
│   └── synapse-train       # Training loop + callbacks
├── zig/                    # SIMD kernels (matmul, attention, RoPE, RMSNorm)
├── configs/                # Model configs (Qwen3, LLaMA, Mistral)
├── examples/               # Chat, benchmarking, training examples
├── scripts/                # Benchmark suite, logit verification
└── docs/                   # mdBook documentation
```

## Supported Models

| Model | Status | Attention | Notes |
|-------|--------|-----------|-------|
| Qwen3 | Validated | GQA | Per-head Q/K norms |
| LLaMA 3.2 | Config ready | GQA | rope_scaling (Linear/Dynamic) |
| Mistral 7B | Config ready | Sliding Window | 4096-token window |
| Phi-3 | Config ready | GQA | Separate projections |
| Gemma | Config ready | MHA | Same as LLaMA naming |

## Quantization Formats

| Format | Source | Compute | Notes |
|--------|--------|---------|-------|
| f32 | safetensors | f32 GEMV | Baseline |
| INT8 | Runtime quantize | INT8 GEMV | `--quantize` flag, 6.3x speedup |
| Q4_0 | GGUF | Q4 GEMV | Native 4-bit compute |
| Q4_K / Q6_K | GGUF | Dequant→f32 | Q4_K_M compatible |
| Q8_0 | GGUF | Dequant→f32 | — |

## Benchmarks

```bash
# Run full benchmark suite
bash scripts/bench_suite.sh --model-dir /tmp/qwen3-0.6b

# Isolated matmul comparison
cargo test --test quantization_speedup --release -- --nocapture isolated_matmul
```

## Testing

```bash
# Library tests (210)
cargo test -p synapse-inference --lib

# Multi-architecture validation (15 tests, 5 model families)
cargo test --test multi_model_validation

# All tests
cargo test --release
```

## License

MIT
