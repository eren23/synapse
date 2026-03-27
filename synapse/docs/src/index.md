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
| Family | Configuration | Prompt | Prefill (tok/s) | Decode (tok/s) | Notes |
|--------|---------------|--------|-----------------|----------------|-------|
| Qwen3 | f32 CPU | hello | 11 | 7.3 | Runtime backend=cpu_simd; prompt=hello |
| Qwen3 | INT8 CPU | hello | 23 | 27.3 | Runtime backend=cpu_simd; prompt=hello |
| LLaMA 3.2 | f32 CPU | hello | 1 | 2.1 | Runtime backend=cpu_simd; prompt=hello |
| LLaMA 3.2 | INT8 CPU | hello | 8 | 9.7 | Runtime backend=cpu_simd; prompt=hello |
| Reference | llama.cpp Q4_K_M | reference_only | 5518 | 173 | Reference only, not a parity claim |
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
