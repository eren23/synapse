# Synapse

A modular local inference engine built in Rust with Zig SIMD kernels, optional Metal acceleration, and a separate pure-Rust WASM runtime for browser delivery. Runs LLMs, Vision Transformers, World Models, and small generative systems across native and browser targets.

## Performance

<!-- status:synapse-benchmark:start -->
| Configuration | Prefill (tok/s) | Decode (tok/s) | Support | Notes |
|---------------|-----------------|----------------|---------|-------|
| f32 CPU | 18 | 6.6 | Stable | CPU SIMD path |
| INT8 CPU | 31 | 14.6 | Stable | Quantized CPU decode |
| Metal f32 | 19 | 8 | Beta | Metal-enabled native build |
| Metal INT8 GPU | 30 | 14.5 | Beta | GPU-resident decode on Apple Silicon |
| llama.cpp Q4_K_M | 5518 | 173 | Reference | Reference only, not a parity claim |
<!-- status:synapse-benchmark:end -->

## GPU Acceleration (Metal)

Synapse supports GPU-resident decode on Apple Silicon via Metal:

| Path | Decode | When to use |
|------|--------|-------------|
| CPU f32 | 6.7 tok/s | Default, no flags |
| CPU INT8 | 14.0 tok/s | `--quantize` |
| Metal INT8 GPU | 14.5 tok/s | `--features metal` |

The Metal path keeps all 28 decoder layers on GPU in a single command buffer -- zero CPU-GPU round-trips during decode. Weights are auto-quantized to INT8 at load time.

## Features

<!-- status:synapse-features:start -->
- **Zig SIMD kernels** (Stable) — Native kernels target NEON and AVX2 through a C ABI layer.
- **Metal GPU** (Beta) — Apple Silicon acceleration is available behind the metal feature.
- **Pure Rust WASM runtime** (Stable) — The browser path avoids Zig FFI and runs entirely client-side.
- **GGUF + safetensors loading** (Stable) — Native runtime loads common checkpoint formats.
- **Speculative decoding** (Experimental) — Self-speculative decode path with KV rollback is available but not a headline stability claim.
- **Training workspace** (Beta) — Autograd, NN, data, graph, and training crates remain available in the workspace.
<!-- status:synapse-features:end -->

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

<!-- status:synapse-models:start -->
| Model Family | Status | Notes |
|--------------|--------|-------|
| Qwen3 | Validated | Logits verified |
| LLaMA 3.2 | Config Ready | Config and weight mapper path present |
| Mistral 7B | Config Ready | Sliding-window config path present |
| Phi-3 | Config Ready | Weight-mapper support in progress |
| Gemma | Config Ready | Same core transformer path |
<!-- status:synapse-models:end -->

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
