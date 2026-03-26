# Synapse

Synapse is an edge-native local inference stack built in Rust, Zig, and Metal, with a separate pure-Rust WASM runtime for browser delivery. It loads common model formats on native targets and runs browser demos entirely client-side.

## Key Features

<!-- status:docs-index-features:start -->
- **Zig SIMD kernels** (Stable) — Native kernels target NEON and AVX2 through a C ABI layer.
- **Metal GPU** (Beta) — Apple Silicon acceleration is available behind the metal feature.
- **Pure Rust WASM runtime** (Stable) — The browser path avoids Zig FFI and runs entirely client-side.
- **GGUF + safetensors loading** (Stable) — Native runtime loads common checkpoint formats.
- **Speculative decoding** (Experimental) — Self-speculative decode path with KV rollback is available but not a headline stability claim.
- **Training workspace** (Beta) — Autograd, NN, data, graph, and training crates remain available in the workspace.
<!-- status:docs-index-features:end -->

## Performance

<!-- status:docs-index-benchmark:start -->
| Configuration | Prefill (tok/s) | Decode (tok/s) | Support | Notes |
|---------------|-----------------|----------------|---------|-------|
| f32 CPU | 18 | 6.6 | Stable | CPU SIMD path |
| INT8 CPU | 31 | 14.6 | Stable | Quantized CPU decode |
| Metal f32 | 19 | 8 | Beta | Metal-enabled native build |
| Metal INT8 GPU | 30 | 14.5 | Beta | GPU-resident decode on Apple Silicon |
| llama.cpp Q4_K_M | 5518 | 173 | Reference | Reference only, not a parity claim |
<!-- status:docs-index-benchmark:end -->

## Quick Start

```bash
# Build
cargo build --release

# Download a model
huggingface-cli download Qwen/Qwen3-0.6B --local-dir /tmp/qwen3-0.6b

# Run chat
cargo run --example qwen3_chat --release -- --model-dir /tmp/qwen3-0.6b
```

See [Installation](getting-started/installation.md) for prerequisites and detailed setup.
