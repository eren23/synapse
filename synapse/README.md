# Synapse

**Modular LLM inference engine in Rust + Zig SIMD kernels.**
Runs on desktop (Metal GPU), browser (WASM), and ESP32 — from one codebase.

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Rust](https://img.shields.io/badge/rust-1.75+-orange.svg)](https://www.rust-lang.org)

---

## Highlights

- **Zig SIMD kernels** — NEON + AVX2 matmul, attention, RoPE, RMSNorm via C ABI
- **Metal GPU** — Apple Silicon acceleration with zero CPU-GPU round-trip forward pass (13 compute shaders)
- **WASM runtime** — Pure Rust browser path, no FFI, runs entirely client-side
- **ESP32-P4** — Edge inference on RISC-V microcontrollers over WiFi HTTP
- **Multi-format weights** — safetensors + GGUF (Q4_0, Q4_K, Q6_K, Q8_0)
- **Speculative decoding** — Self-speculative decode with KV rollback
- **World models** — Built-in LEWM (ViT encoder + DiT predictor) with browser + ESP32 targets

## Benchmarks

<!-- status:synapse-benchmark:start -->
| Model | Quantization | Prefill (tok/s) | Decode (tok/s) |
|-------|-------------|-----------------|----------------|
| Qwen3-0.6B | f32 | 11 | 7.3 |
| Qwen3-0.6B | INT8 | 23 | **27.3** |
| LLaMA 3.2-1B | f32 | 1 | 2.1 |
| LLaMA 3.2-1B | INT8 | 8 | 9.7 |
<!-- status:synapse-benchmark:end -->

> Measured end-to-end on Apple Silicon. Full matrix with synthetic and exploratory rows in [`status/benchmark_matrix.md`](status/benchmark_matrix.md).

## Quick Start

```bash
# Build (Zig kernels auto-rebuild)
cargo build --release

# Download a model
huggingface-cli download Qwen/Qwen3-0.6B --local-dir /tmp/qwen3-0.6b

# Chat
cargo run --example qwen3_chat --release -- --model-dir /tmp/qwen3-0.6b

# Chat with INT8 quantization
cargo run --example qwen3_chat --release -- --model-dir /tmp/qwen3-0.6b --quantize

# With Metal GPU (macOS)
cargo run --example qwen3_chat --release --features metal -- --model-dir /tmp/qwen3-0.6b --quantize

# Profile a single prompt
cargo run --example qwen3_chat --release -- --model-dir /tmp/qwen3-0.6b --quantize --prompt "hello" --profile-stages
```

## Multi-Target Architecture

| Target | Backend | Quantization | Use Case |
|--------|---------|-------------|----------|
| **Desktop** | Zig SIMD (NEON/AVX2) + Metal GPU | f32, INT8, Q4 | Development, serving |
| **Browser** | Pure Rust (WASM) | f32, INT8 | Embeddable widget, client-side demos |
| **ESP32-P4** | Pure Rust + PIE accelerator | INT8, Q4 | Edge inference via WiFi HTTP |

```bash
cargo build --release                      # Desktop
wasm-pack build -p synapse-wasm --release  # Browser (160KB core + 32KB JS)
cargo build -p synapse-esp32               # ESP32 (host test)
```

See [BUILD.md](BUILD.md) for cross-compilation details.

## Supported Models

<!-- status:synapse-models:start -->
| Model Family | Type | Status | Notes |
|--------------|------|--------|-------|
| **Qwen3** | LLM (GQA) | Validated | Benchmarked, logits verified |
| **LLaMA 3.2** | LLM (GQA) | Validated | Benchmarked locally |
| **Mistral 7B** | LLM (Sliding Window) | Config Ready | Synthetic tests passing |
| **Phi-3** | LLM (GQA) | In Progress | Weight mapper underway |
| **Gemma** | LLM (MHA, GeGLU) | Config Ready | Synthetic tests passing |
| **ViT** | Vision | Validated | Patch embedding, classification head |
| **CLIP** | Vision+Text | Supported | Dual-encoder with projection |
| **DINOv2** | Vision | Supported | Self-supervised ViT variant |
| **LEWM** | World Model | Validated | ViT encoder + DiT predictor |
<!-- status:synapse-models:end -->

332 unit tests + 17 multi-architecture integration tests cover all model families.

## Quantization

| Format | Source | Compute | Flag |
|--------|--------|---------|------|
| f32 | safetensors | f32 GEMV | (default) |
| f16 | safetensors | f16 | config |
| INT8 | Runtime | INT8 GEMV | `--quantize` |
| Q4_0 | GGUF | Q4 GEMV | GGUF file |
| Q4_K / Q6_K | GGUF | Dequant -> f32 | GGUF file |
| Q8_0 | GGUF | Dequant -> f32 | GGUF file |

## World Models (LEWM)

Latent Emergent World Model — ViT encoder + DiT predictor for latent state prediction and trajectory rollouts. Runs on all three targets.

| Operation | Latency (Apple Silicon) |
|-----------|------------------------|
| Encode (224x224 -> 192d) | 26.9ms |
| Predict (single step) | 12.8ms |
| Rollout (50 steps) | 609ms |

- **Browser**: Loads 69MB checkpoint, interactive trajectory rollouts (`web/index.html`)
- **ESP32-P4**: Phone camera -> WiFi HTTP -> ESP32 LEWM -> JSON response
- **Quantization**: INT8 (~4x smaller) and Q4 (~6.4x compression, ~7MB weights)

## Examples

| Example | Description |
|---------|-------------|
| `qwen3_chat` | Interactive LLM chat with quantization + Metal support |
| `lewm_demo` | LEWM world model rollout |
| `world_model_rollout` | Multi-step trajectory prediction |
| `vit_classify` | Vision Transformer classification |
| `clip_similarity` | CLIP image-text similarity |
| `jepa_embed` | JEPA embedding extraction |
| `geometric_attention_demo` | Geometric attention visualization |
| `mnist` / `cifar10` | Training examples (autograd workspace) |
| `xor` / `text_classification` / `vision_transformer` | More training examples |

## Workspace

```
synapse/
├── crates/
│   ├── synapse-inference   # Models, generation, quantization, chat templates
│   ├── synapse-core        # FFI wrappers for Zig tensor ops
│   ├── synapse-sys         # Raw C bindings (auto-rebuild via build.rs)
│   ├── synapse-nn          # Neural network modules
│   ├── synapse-autograd    # Tape-based autodiff
│   ├── synapse-optim       # SGD, Adam, RMSProp + schedulers
│   ├── synapse-data        # DataLoader, Dataset, Sampler
│   ├── synapse-graph       # Graph IR + optimization passes
│   └── synapse-train       # Training loop + callbacks
├── synapse-wasm/           # Browser WASM runtime (pure Rust, zero FFI)
├── synapse-esp32/          # ESP32-P4 edge target (WiFi HTTP server)
├── zig/                    # SIMD kernels (matmul, qmatmul, attention, RoPE, RMSNorm)
├── configs/                # Model config JSONs (Qwen3, LLaMA, Mistral, Phi-3, Gemma)
├── scripts/                # Benchmark suite + logit verification
└── web/                    # Browser LEWM demo
```

## Testing

```bash
cargo test -p synapse-inference --lib      # 332 unit tests
cargo test --test multi_model_validation   # 17 multi-architecture tests
cargo test --release                       # Full suite including benchmarks
```

## License

MIT
