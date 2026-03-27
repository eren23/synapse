# Synapse — Multi-Platform Build Guide

Synapse is a Rust workspace with 11 crates targeting three platforms: **Desktop** (Zig SIMD + optional Metal GPU), **Browser** (WebAssembly), and **ESP32-P4** (RISC-V embedded).

---

## 1. Prerequisites

### All platforms
```bash
# Rust (stable + nightly for WASM/ESP32)
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
rustup update stable
```

### Desktop (Zig SIMD)
```bash
# Zig 0.13+ — required for native SIMD kernels
# macOS
brew install zig

# Linux
snap install zig --classic --beta
# or download from https://ziglang.org/download/
```

> Zig is optional. Skip it by using `--no-default-features --features pure-rust` (see [Desktop Build](#2-desktop-build)).

### Browser (WASM)
```bash
rustup target add wasm32-unknown-unknown
cargo install wasm-pack
```

### ESP32-P4 (RISC-V)
```bash
cargo install espup
espup install          # installs RISC-V toolchain + target
source ~/export-esp.sh # add to your shell profile
```

---

## 2. Desktop Build

### Standard build (Zig SIMD auto-compiled)
```bash
cd synapse
cargo build --release
```

`build.rs` in `synapse-sys` automatically invokes `zig build -Doptimize=ReleaseFast` whenever any file under `zig/src/**/*.zig` changes. No manual Zig step is needed.

### With Metal GPU acceleration (macOS only)
```bash
cargo build --release --features metal
```

### Pure-Rust build (no Zig dependency)
```bash
cargo build --release --no-default-features --features pure-rust
```

Use this on CI, Windows, or any environment where Zig is not installed.

### Manual Zig rebuild (if needed)
```bash
cd zig
zig build -Doptimize=ReleaseFast
cd ..
```

### Running examples

**Chat with Qwen3** (requires model weights — see [Model Downloads](#6-model-downloads)):
```bash
cargo run --release --example qwen3_chat -- --config configs/qwen3_0.6b.json --weights /path/to/qwen3
```

**LeWM world model demo** (requires `web/lewm_weights.bin`):
```bash
cargo run --release --example lewm_demo
```

**Benchmark all models**:
```bash
cargo run --release --example model_benchmark
# or use the full benchmark suite:
./scripts/bench_suite.sh
./scripts/bench_suite.sh --model-dir /path/to/qwen3
```

### Available configs
| File | Model |
|------|-------|
| `configs/qwen3_0.6b.json` | Qwen3 0.6B |
| `configs/llama3.2_1b.json` | LLaMA 3.2 1B |
| `configs/mistral_7b.json` | Mistral 7B |
| `configs/gemma_2b.json` | Gemma 2B |
| `configs/phi3_mini.json` | Phi-3 Mini |

---

## 3. Browser (WASM)

The `synapse-wasm` crate compiles to a `cdylib` using the `pure-rust` feature (no Zig, no Metal). The resulting WASM core is ~180 KB; model weights are loaded separately at runtime.

### Build
```bash
cd synapse-wasm
wasm-pack build --target web --release
# Output lands in synapse-wasm/pkg/
```

### Serve locally
```bash
cd ../web
python3 -m http.server 8080
# Open http://localhost:8080
```

The `web/` directory already contains `index.html` and pre-built `pkg/` assets. The JS glue imports from `../synapse-wasm/pkg/synapse_wasm.js`.

### Runtime weights (loaded by the browser page)
| File | Size | Purpose |
|------|------|---------|
| `web/lewm_weights.bin` | ~69 MB | LeWM world model |
| `web/neo_unify_weights.bin` | ~9.3 MB | NeoUnify model |

---

## 4. ESP32-P4 (RISC-V)

The `synapse-esp32` crate defaults to `host-test` mode, which builds and runs on your Mac/Linux machine using the `pure-rust` feature. Real hardware support (ESP-IDF HAL bindings) is stubbed out and will be enabled in a future phase.

### Host-test mode (no hardware required)
```bash
cargo run -p synapse-esp32
```

### Real hardware (ESP32-P4)
> Not yet enabled. When ready:
> 1. Uncomment `esp-idf-hal`, `esp-idf-svc`, `esp-idf-sys` in `synapse-esp32/Cargo.toml`.
> 2. Source the ESP toolchain: `source ~/export-esp.sh`
> 3. Build and flash:
>    ```bash
>    cargo build -p synapse-esp32 --target riscv32imc-esp-espidf --release --no-default-features --features esp32
>    espflash flash target/riscv32imc-esp-espidf/release/synapse-esp32 --monitor
>    ```

---

## 5. Testing

### Unit + integration tests (292+ tests)
```bash
# All tests
cargo test --release

# Inference crate only (fastest feedback loop — ~260 tests)
cargo test -p synapse-inference --lib

# Multi-architecture model validation (17 tests)
cargo test --test multi_model_validation
```

### Integration test suites
```bash
cargo test --test inference_e2e
cargo test --test attention_correctness
cargo test --test kvcache_correctness
cargo test --test config_driven_assembly
cargo test --test quantization_accuracy
```

### Benchmark matrix
```bash
# Full benchmark suite (text output)
./scripts/bench_suite.sh

# With a specific model directory
./scripts/bench_suite.sh --model-dir /path/to/qwen3

# Including exploratory benchmarks
./scripts/bench_suite.sh --include-exploratory

# JSON output (updates status/benchmark_matrix.json)
python3 scripts/benchmark_matrix.py --format json
```

### Known flaky benchmarks (pre-existing timing sensitivity)
- `attention_bench_fused_2x_vs_naive` — threshold 2.0x, may measure 1.99x
- `bench_layernorm_residual_fusion_speedup` — threshold 1.20x, may measure 1.15x

Run with `--release` and on an idle machine to minimise noise.

---

## 6. Model Downloads

Models are not included in the repository. Use `huggingface-cli` to download them.

### Install
```bash
pip install huggingface_hub
huggingface-cli login   # optional for gated models
```

### Qwen3 0.6B (recommended starting point)
```bash
huggingface-cli download Qwen/Qwen3-0.6B --local-dir ~/models/qwen3_0.6b
```

### LLaMA 3.2 1B
```bash
huggingface-cli download meta-llama/Llama-3.2-1B --local-dir ~/models/llama3.2_1b
```

### Gemma 2B
```bash
huggingface-cli download google/gemma-2b --local-dir ~/models/gemma_2b
```

### PushT weights for LeWM demo
The `web/lewm_weights.bin` file (69 MB) is included in the repo and is used directly by `examples/lewm_demo.rs` and the browser demo. No separate download is needed.

### Point an example at a model
```bash
# qwen3_chat
cargo run --release --example qwen3_chat -- \
  --config configs/qwen3_0.6b.json \
  --weights ~/models/qwen3_0.6b

# model_benchmark
cargo run --release --example model_benchmark -- \
  --official-model qwen3=~/models/qwen3_0.6b
```
