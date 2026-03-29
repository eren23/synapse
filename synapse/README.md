# Synapse

**Modular LLM inference engine in Rust + Zig SIMD kernels.**
Runs on desktop (Metal GPU), browser (WASM), and ESP32 — from one codebase.

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Rust](https://img.shields.io/badge/rust-1.75+-orange.svg)](https://www.rust-lang.org)

---

## Highlights

<!-- status:synapse-features:start -->
- **Zig SIMD kernels** (Stable) — Native kernels target NEON and AVX2 through a C ABI layer.
- **Metal GPU** (Beta) — Apple Silicon acceleration is available behind the metal feature.
- **Pure Rust WASM runtime** (Stable) — The browser path avoids Zig FFI and runs entirely client-side.
- **GGUF + safetensors loading** (Stable) — Native runtime loads common checkpoint formats.
- **Speculative decoding** (Experimental) — Self-speculative decode path with KV rollback is available but not a headline stability claim.
- **Training workspace** (Beta) — Autograd, NN, data, graph, and training crates remain available in the workspace.
<!-- status:synapse-features:end -->

## Benchmarks

<!-- status:synapse-benchmark:start -->
| Family | Configuration | Prompt | Prefill (tok/s) | Decode (tok/s) | Notes |
|--------|---------------|--------|-----------------|----------------|-------|
| Qwen3 | f32 CPU | hello | 11 | 7.3 | Runtime backend=cpu_simd; prompt=hello |
| Qwen3 | INT8 CPU | hello | 23 | 27.3 | Runtime backend=cpu_simd; prompt=hello |
| LLaMA 3.2 | f32 CPU | hello | 1 | 2.1 | Runtime backend=cpu_simd; prompt=hello |
| LLaMA 3.2 | INT8 CPU | hello | 8 | 9.7 | Runtime backend=cpu_simd; prompt=hello |
| Reference | llama.cpp Q4_K_M | reference_only | 5518 | 173 | Reference only, not a parity claim |
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
| Model Family | Status | Evidence | Notes |
|--------------|--------|----------|-------|
| Qwen3 | Validated | Benchmarked Local | Real checkpoint benchmarked locally; logits verified |
| LLaMA 3.2 | Benchmarked Local | Benchmarked Local | Real checkpoint benchmarked locally on this machine |
| Mistral 7B | Config Ready | Synthetic Validated | Sliding-window config path present; synthetic correctness tests pass, but the scaled synthetic throughput benchmark is currently failing |
| Phi-3 | In Progress | Synthetic Validated | Weight-mapper support in progress; synthetic validation passing |
| Gemma | Config Ready | Synthetic Validated | Same core transformer path; synthetic validation passing |
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

## Supported Architectures

Synapse supports multiple model architecture families beyond standard transformers:

| Architecture | Models | Memory | Key Feature |
|---|---|---|---|
| **Transformer (GQA/MHA)** | Qwen3, LLaMA, Mistral, Phi-3, Gemma | O(n) KV cache | Standard attention, INT8/Q4/Ternary quantization |
| **Mamba (SSM)** | Mamba-130M, Mamba-370M | O(1) constant | Selective state spaces, no KV cache |
| **RWKV-7** | RWKV-7 0.1B, 0.4B | O(1) constant | WKV recurrence, infinite context |
| **Hybrid DeltaNet** | Qwen3.5-0.8B | Mixed O(1)+O(n) | 75% linear attention + 25% GQA |
| **Diffusion LLM** | Tiny-A2D 0.6B | O(1) per step | Non-autoregressive parallel decode |

### SSM Models (Mamba, RWKV)

State Space Models use constant O(1) memory regardless of sequence length — no KV cache. This makes them ideal for edge devices (ESP32-P4), browser WASM inference, and long-context applications.

```rust
let engine = InferenceEngine::from_pretrained(Path::new("./models/mamba-130m"))?;
let output = engine.generate_text("Hello", GenerationConfig::default())?;
```

### Ternary Quantization

2-bit weight quantization where weights are {-1, 0, +1}. Multiplication becomes addition/subtraction — ideal for WASM and microcontrollers.

```rust
let mut engine = InferenceEngine::from_pretrained(path)?;
engine.quantize_ternary(); // 2-bit weights, ~16x compression
```

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
