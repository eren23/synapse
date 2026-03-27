# Synapse

A modular local inference engine built in Rust with Zig SIMD kernels, optional Metal acceleration, and a separate pure-Rust WASM runtime for browser delivery. Runs LLMs, Vision Transformers, World Models, and small generative systems across native and browser targets.

## Performance

<!-- status:synapse-benchmark:start -->
| Family | Configuration | Prompt | Prefill (tok/s) | Decode (tok/s) | Notes |
|--------|---------------|--------|-----------------|----------------|-------|
| Qwen3 | f32 CPU | hello | 11 | 7.3 | Runtime backend=cpu_simd; prompt=hello |
| Qwen3 | INT8 CPU | hello | 23 | 27.3 | Runtime backend=cpu_simd; prompt=hello |
| LLaMA 3.2 | f32 CPU | hello | 1 | 2.1 | Runtime backend=cpu_simd; prompt=hello |
| LLaMA 3.2 | INT8 CPU | hello | 8 | 9.7 | Runtime backend=cpu_simd; prompt=hello |
| Reference | llama.cpp Q4_K_M | reference_only | 5518 | 173 | Reference only, not a parity claim |
<!-- status:synapse-benchmark:end -->

The public table above is intentionally narrow: it only shows measured local end-to-end rows that were produced by the benchmark matrix on this machine and then synced into the docs. Synthetic benchmark rows, exploratory checkpoints, and failed/fallback runs are kept out of the headline table and written to `status/benchmark_matrix.md` instead.

The important architectural point is that the recent decode recovery came from shared causal-LM infrastructure, not a one-off prompt hack. The `M=1` INT8 GEMV path, quantized LM head, and cached decode flow are shared kernels. Qwen3 benefits first because it is the most exercised real-checkpoint path here; LLaMA, Mistral, and similar decoder families should inherit those wins once their end-to-end checkpoint path is benchmarked locally, but Synapse does not publish future throughput numbers as promises.

## GPU Acceleration (Metal)

Synapse can be built with Metal support on Apple Silicon, but the active runtime path should be treated as a measured value, not an assumption:

| Path | Decode | When to use |
|------|--------|-------------|
| Default build | See `Runtime:` line | `cargo run --example qwen3_chat --release ...` |
| Metal-feature build | See `Runtime:` line | `cargo run --example qwen3_chat --release --features metal ...` |
| Benchmark matrix | Exact local measurement | `bash scripts/bench_suite.sh --model-dir ...` |

The chat example now prints a `Runtime:` line before generation so you can see the active model family, compute path, and selected prefill/decode strategies. For Qwen3, the interactive CLI defaults to `thinking=disabled`; pass `--thinking auto` to restore the model's default think-first behavior. Use `--inspect-prompt --prompt "..."` to dump the rendered prompt and token IDs, and `--prompt "..." --profile-stages` to print render/encode/prefill/decode timings for a single prompt.

Metal rows are only allowed into the public benchmark table when the runtime line actually reports `backend=metal`. A metal-feature build that falls back to `cpu_simd` is recorded in the raw matrix artifact, but it is not presented as a GPU throughput claim.

## Features

<!-- status:synapse-features:start -->
- **Zig SIMD kernels** (Stable) — Native kernels target NEON and AVX2 through a C ABI layer.
- **Metal GPU** (Beta) — Apple Silicon acceleration is available behind the metal feature.
- **Pure Rust WASM runtime** (Stable) — The browser path avoids Zig FFI and runs entirely client-side.
- **GGUF + safetensors loading** (Stable) — Native runtime loads common checkpoint formats.
- **Speculative decoding** (Experimental) — Self-speculative decode path with KV rollback is available but not a headline stability claim.
- **Training workspace** (Beta) — Autograd, NN, data, graph, and training crates remain available in the workspace.
<!-- status:synapse-features:end -->

## Multi-Target Architecture

Synapse runs the same models across three deployment targets from one codebase:

| Target | Backend | Quantization | Use Case |
|--------|---------|-------------|----------|
| **Desktop** | Zig SIMD (NEON/AVX2) + Metal GPU | f32, INT8, Q4 | Development, benchmarking |
| **Browser** | Pure Rust (WASM) | f32, INT8 | Client-side demos, embeddable widget |
| **ESP32-P4** | Pure Rust + PIE accelerator | INT8, Q4 | Edge inference, IoT |

Build for any target:
```bash
cargo build --release                    # Desktop (default)
wasm-pack build -p synapse-wasm --release  # Browser
cargo build -p synapse-esp32               # ESP32 (host test)
```

See [BUILD.md](BUILD.md) for detailed instructions.

## Quick Start

```bash
# Build (auto-rebuilds Zig kernels)
cargo build --release

# Download a model
huggingface-cli download Qwen/Qwen3-0.6B --local-dir /tmp/qwen3-0.6b

# Chat (f32)
cargo run --example qwen3_chat --release -- --model-dir /tmp/qwen3-0.6b

# Chat (INT8 quantized)
cargo run --example qwen3_chat --release -- --model-dir /tmp/qwen3-0.6b --quantize

# Qwen3 with explicit think-first behavior
cargo run --example qwen3_chat --release -- --model-dir /tmp/qwen3-0.6b --thinking auto

# Inspect the rendered Qwen3 prompt and token IDs
cargo run --example qwen3_chat --release -- --model-dir /tmp/qwen3-0.6b --inspect-prompt --prompt "hello"

# Run one prompt and print render/encode/prefill/decode timings
cargo run --example qwen3_chat --release -- --model-dir /tmp/qwen3-0.6b --quantize --prompt "hello" --profile-stages

# With Metal feature enabled (macOS)
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
│   ├── synapse-train       # Training loop + callbacks
│   ├── synapse-wasm/           # Browser WASM runtime (pure Rust)
│   └── synapse-esp32/          # ESP32-P4 edge inference
├── zig/                    # SIMD kernels (matmul, attention, RoPE, RMSNorm)
├── configs/                # Model configs (Qwen3, LLaMA, Mistral)
├── examples/               # Chat, benchmarking, training examples
├── scripts/                # Benchmark suite, logit verification
└── docs/                   # mdBook documentation
```

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

The model matrix carries two dimensions:

- `Status` is support maturity for that family.
- `Evidence` is the strongest proof currently available, such as a real local benchmark, logit verification, or synthetic validation.

That split keeps the docs honest. A family can be structurally supported and exercised through fake-weight tests without being presented as a measured end-to-end checkpoint path yet.

## World Models (LEWM)

Synapse includes a Latent Emergent World Model (LEWM) — a ViT encoder + DiT predictor that maps observations to latent states and predicts future states conditioned on actions.

| Operation | Latency (Mac) | Notes |
|-----------|-------------|-------|
| Encode (224×224 → 192d) | 26.9ms | ViT encoder, 6 layers |
| Predict (single step) | 12.8ms | DiT predictor, 6 adaLN layers |
| Rollout (50 steps) | 609ms | Sequential predict_next |

**Quantization:**
- INT8: predictor layers quantized (~4x smaller)
- Q4: predictor layers in 4-bit (~6.4x compression, ~7MB weights)

**Browser demo:** Load the 69MB checkpoint in-browser and run interactive trajectory rollouts. See `web/index.html`.

```bash
# Run LEWM demo (requires PushT checkpoint)
cargo run --example lewm_demo --release

# Browser demo
cd web && python3 -m http.server 8000
# Open http://localhost:8000
```

## Quantization Formats

| Format | Source | Compute | Notes |
|--------|--------|---------|-------|
| f32 | safetensors | f32 GEMV | Baseline |
| INT8 | Runtime quantize | INT8 GEMV | `--quantize` uses a fully quantized cached decode LM head |
| Q4_0 | GGUF | Q4 GEMV | Native 4-bit compute |
| Q4_K / Q6_K | GGUF | Dequant→f32 | Q4_K_M compatible |
| Q8_0 | GGUF | Dequant→f32 | — |

## Benchmarks

```bash
# Run the canonical validation + benchmark matrix
python3 scripts/benchmark_matrix.py --include-exploratory

# Human-friendly wrapper for the same pipeline
bash scripts/bench_suite.sh --include-exploratory

# Isolated matmul comparison
cargo test --test quantization_speedup --release -- --nocapture isolated_matmul
```

Artifacts written by the matrix runner:

- `status/benchmark_matrix.json`: raw structured results
- `status/benchmark_matrix.md`: human-readable report with measured, synthetic, and exploratory rows
- `status/public_status.json`: public snapshot consumed by README and mdBook status blocks

## Testing

```bash
# Core inference tests
cargo test -p synapse-inference --lib

# Multi-architecture validation
cargo test --test multi_model_validation

# Regenerate and validate public benchmark/docs status
python3 scripts/benchmark_matrix.py
python3 scripts/sync_public_status.py --check
```

## Workspace

| Crate | Purpose |
|-------|---------|
| **synapse-inference** | Core: models, generation, quantization, chat templates |
| **synapse-core** | FFI wrappers for Zig SIMD ops |
| **synapse-sys** | Raw C bindings to Zig kernels |
| **synapse-nn** | Neural network modules (Linear, Transformer) |
| **synapse-autograd** | Tape-based automatic differentiation |
| **synapse-optim** | Optimizers (SGD, Adam, RMSProp) |
| **synapse-data** | Data loading pipeline |
| **synapse-graph** | Graph IR + optimization passes |
| **synapse-train** | Training loop with callbacks |
| **synapse-wasm** | Browser WASM runtime (pure Rust, no FFI) |
| **synapse-esp32** | ESP32-P4 edge inference target |

## License

MIT
