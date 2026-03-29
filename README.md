# Synapse

<!-- status:root-positioning:start -->
Edge-native inference stack for local ML across native and browser targets.

- Native builds use Rust orchestration with Zig SIMD kernels and optional Metal acceleration.
- Browser builds use a pure-Rust WASM runtime for portability and client-side demos.
- Public benchmark rows are measured on Apple Silicon and synced from status/benchmark_matrix.json.
<!-- status:root-positioning:end -->

## Benchmarks

<!-- status:root-benchmark:start -->
| Family | Configuration | Prompt | Prefill (tok/s) | Decode (tok/s) | Notes |
|--------|---------------|--------|-----------------|----------------|-------|
| Qwen3 | f32 CPU | hello | 11 | 7.3 | Runtime backend=cpu_simd; prompt=hello |
| Qwen3 | INT8 CPU | hello | 23 | 27.3 | Runtime backend=cpu_simd; prompt=hello |
| LLaMA 3.2 | f32 CPU | hello | 1 | 2.1 | Runtime backend=cpu_simd; prompt=hello |
| LLaMA 3.2 | INT8 CPU | hello | 8 | 9.7 | Runtime backend=cpu_simd; prompt=hello |
| Reference | llama.cpp Q4_K_M | reference_only | 5518 | 173 | Reference only, not a parity claim |
<!-- status:root-benchmark:end -->

> Measured end-to-end on Apple Silicon. Full matrix in [`synapse/status/benchmark_matrix.md`](synapse/status/benchmark_matrix.md).

## Deployment Targets

<!-- status:root-profiles:start -->
| Runtime Profile | Support | Targets | Backends | Quantization |
|-----------------|---------|---------|----------|--------------|
| Native Performance | Stable | aarch64-apple-darwin, x86_64-unknown-linux-gnu | cpu_simd, metal | f32, f16, int8, q4_0, q4_k, q6_k, q8_0 |
| ARM Compact | Beta | aarch64-unknown-linux-musl, aarch64-unknown-linux-gnu | cpu_simd | f32, int8, q4_0, q4_k |
| WASM Portable | Stable | wasm32-unknown-unknown | pure_rust_wasm | f32 |
<!-- status:root-profiles:end -->

<!-- status:root-artifacts:start -->
| Artifact | Current | Budget | Status |
|----------|---------|--------|--------|
| WASM core | ~519 KB | ~160 KB | over |
| WASM JS wrapper | ~43 KB | ~32 KB | over |
<!-- status:root-artifacts:end -->

## Supported Models

| Model Family | Type | Status |
|--------------|------|--------|
| **Qwen3** | LLM (GQA) | Validated — benchmarked, logits verified |
| **LLaMA 3.2** | LLM (GQA) | Validated — benchmarked locally |
| **Mistral 7B** | LLM (Sliding Window) | Config ready — synthetic tests passing |
| **Phi-3** | LLM (GQA) | In progress |
| **Gemma** | LLM (MHA, GeGLU) | Config ready — synthetic tests passing |
| **ViT** | Vision | Validated |
| **CLIP** | Vision+Text | Supported |
| **DINOv2** | Vision | Supported |
| **LEWM** | World Model | Validated — runs on all 3 targets |
| **Mamba** | SSM | Validated — 130M/370M, INT8+Q4, browser WASM |
| **RWKV-7** | SSM | Validated — 0.1B/0.4B, value residuals, pre-LayerNorm |

Adding a new model = write a config JSON + weight mapper. No engine changes.

## Component Registry

Every architectural element is a pluggable trait with config-driven instantiation:

| Component | Variants |
|-----------|----------|
| Attention | GQA, MHA, MQA, SlidingWindow |
| Normalization | RMSNorm, LayerNorm |
| FFN | SwiGLU, GELU, GeGLU |
| Position | RoPE, Learned, Sinusoidal |
| Quantization | f32, f16, INT8, Q4_0, Q4_K, Q6_K, Q8_0 |
| Weights | safetensors, GGUF |

## Quick Start

```bash
# Build (Zig kernels auto-rebuild)
cd synapse && cargo build --release

# Download a model
huggingface-cli download Qwen/Qwen3-0.6B --local-dir /tmp/qwen3-0.6b

# Chat
cargo run --example qwen3_chat --release -- --model-dir /tmp/qwen3-0.6b

# Chat with INT8 quantization
cargo run --example qwen3_chat --release -- --model-dir /tmp/qwen3-0.6b --quantize

# With Metal GPU (macOS)
cargo run --example qwen3_chat --release --features metal -- --model-dir /tmp/qwen3-0.6b --quantize

# Demo mode (random weights, no downloads)
cargo run --example qwen3_chat --release -- --demo

# Build for browser
wasm-pack build -p synapse-wasm --release

# Build for ESP32
cargo build -p synapse-esp32
```

## World Models (LEWM)

Latent Emergent World Model — ViT encoder + DiT predictor for latent state prediction.

| Operation | Latency (Apple Silicon) |
|-----------|------------------------|
| Encode (224x224 -> 192d) | 26.9ms |
| Predict (single step) | 12.8ms |
| Rollout (50 steps) | 609ms |

- **Browser**: 69MB checkpoint, interactive trajectory rollouts (`synapse/web/index.html`)
- **ESP32-P4**: Phone camera -> WiFi HTTP -> LEWM inference -> JSON response
- **Quantization**: INT8 (~4x smaller), Q4 (~6.4x compression, ~7MB weights)

### Compression Results (First-Ever JEPA Quantization)

| Config | Size | Quality (cos@20) |
|--------|------|-------------------|
| f32 baseline | 52.1 MB | 1.000 |
| INT8 predictor | 21.4 MB | 0.9998 |
| Q4 predictor | 17.4 MB | 0.998 |
| Full Q4 (enc+pred) | 9.4 MB | 0.93 |

No published work on JEPA quantization exists — these are first-of-kind results.

**Browser demos**: [Main hub](synapse/web/) · [Compression benchmark](synapse/web/lewm-compress-demo/) · [SSM chat](synapse/web/ssm-demo/)

## Roadmap

| Goal | Status |
|------|--------|
| Sub-8MB LEWM at cos >0.95 | Current best: 9.4 MB, cos 0.93. Next: structured pruning, mixed Q4/Q8, Hadamard rotation |
| ESP32-P4 hardware deployment | Code ready (25 tests passing), awaiting hardware for video demo |
| WASM pre-quantized binaries | Skip the 69 MB f32 download — load ~10 MB Q4 directly |
| npm package for WASM widget | Package synapse-wasm as embeddable `<script>` module |

### Why Synapse?

| Capability | Synapse | Alternatives |
|-----------|---------|-------------|
| JEPA/LEWM quantization | Q4: 9.4 MB, cos 0.93 (first published) | None exist |
| WASM binary | 491 KB (133 KB brotli) | Candle: 2-5 MB |
| SSM inference | Mamba + RWKV-7 via Zig SIMD | Candle: Mamba v1 only |
| Edge deployment | ESP32-P4 ready | TFLite Micro (no world models) |
| Model surgery | Wanda + channel + layer pruning | None in compiled languages |

## Architecture

```
synapse/
├── crates/
│   ├── synapse-inference/    # Models, generation, quantization, chat templates
│   │   ├── model/            # CausalLM, DecoderLayer, ModelBuilder
│   │   ├── generation/       # Pipeline, sampler, speculative decoding
│   │   ├── weight_loading/   # safetensors + GGUF, per-model weight mappers
│   │   ├── tokenizer/        # BPE tokenizer (HuggingFace format)
│   │   ├── kv_cache/         # Pre-allocated KV cache
│   │   ├── quantization/     # INT8 per-channel quantization
│   │   ├── metal/            # Metal GPU backend (13 shaders, zero-roundtrip forward)
│   │   ├── lewm/             # World model (ViT encoder + DiT predictor)
│   │   └── diffusion/        # Diffusion pipeline (scaffolding)
│   ├── synapse-core/         # FFI wrappers for Zig tensor ops
│   ├── synapse-sys/          # Raw C bindings (auto-rebuild via build.rs)
│   ├── synapse-nn/           # Neural network modules
│   ├── synapse-autograd/     # Tape-based autodiff
│   ├── synapse-optim/        # SGD, Adam, RMSProp + schedulers
│   ├── synapse-data/         # DataLoader, Dataset, Sampler
│   ├── synapse-graph/        # Graph IR + optimization passes
│   └── synapse-train/        # Training loop + callbacks
├── synapse-wasm/             # Browser WASM runtime (pure Rust, zero FFI)
├── synapse-esp32/            # ESP32-P4 edge target (WiFi HTTP server)
├── zig/src/ops/              # SIMD kernels: matmul, qmatmul, attention, RoPE, RMSNorm
├── configs/                  # Model configs (Qwen3, LLaMA, Mistral, Phi-3, Gemma)
├── scripts/                  # Benchmark suite + logit verification
└── web/                      # Browser LEWM demo
```

## Testing

```bash
cargo test -p synapse-inference --lib      # 332 unit tests
cargo test --test multi_model_validation   # 17 multi-architecture tests
cargo test --release                       # Full suite including benchmarks
```

## Development History

| Phase | What |
|-------|------|
| 1 | Zig SIMD tensor engine, Rust autograd, training framework |
| 2 | Transformer stack, attention kernels, RoPE |
| 3 | Inference engine, component registry, INT8 quantization, Qwen3 |
| 4 | SIMD kernel wiring, KV cache, Metal GPU shaders |
| 5 | Multi-model support (LLaMA, Mistral, Phi-3, Gemma), GGUF loading, Q4 quantization |
| 6 | LEWM world models, WASM runtime, ESP32 target, speculative decoding |
| 7 | SSM inference (Mamba, RWKV-7), model surgery/pruning, LEWM Q4 compression, WASM demos |

## Built With

- **Rust** — inference engine, autograd, training framework
- **Zig** — SIMD kernels (ARM NEON + AVX2), C ABI FFI
- **Metal Shading Language** — GPU compute shaders for Apple Silicon
- **Swarm development** — built using [attoswarm](https://github.com/attocode) parallel agent orchestration

## License

MIT
