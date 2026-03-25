# Synapse — Development Guide

## Build

```bash
cargo build --release          # Auto-rebuilds Zig via build.rs
cargo build --release --features metal   # With Metal GPU
```

Zig is rebuilt automatically when `zig/src/**/*.zig` files change. Manual rebuild: `cd zig && zig build -Doptimize=ReleaseFast`.

## Test

```bash
cargo test -p synapse-inference --lib      # 210 library tests
cargo test --test multi_model_validation   # 15 multi-architecture tests
cargo test --release                       # All tests (some benchmarks are flaky on timing)
```

Known flaky benchmarks (timing-sensitive, pre-existing):
- `attention_bench_fused_2x_vs_naive` — 1.99x vs 2.0x threshold
- `bench_layernorm_residual_fusion_speedup` — 1.15x vs 1.20x threshold

## Architecture

9 crates under `crates/`:
- **synapse-inference** — Main inference engine (models, generation, quantization, chat templates)
- **synapse-core** — FFI wrappers for Zig ops (Tensor, KvCache)
- **synapse-sys** — Raw C bindings to `libsynapse_zig.a` (auto-rebuild in build.rs)
- **synapse-nn** — Neural network modules (Linear, Embedding, Transformer)
- **synapse-autograd** — Tape-based automatic differentiation
- **synapse-optim** — Optimizers (SGD, Adam) + schedulers
- **synapse-data** — DataLoader, Dataset, Sampler
- **synapse-graph** — Graph IR + optimization passes
- **synapse-train** — Training loop with callbacks

## Key Files

| File | Role |
|------|------|
| `zig/src/ops/matmul.zig` | f32 SGEMM + GEMV kernel |
| `zig/src/ops/qmatmul.zig` | INT8 + Q4 GEMV kernels |
| `zig/src/ops/attention.zig` | Fused tiled attention |
| `crates/synapse-inference/src/model/decoder_layer.rs` | Core decoder: attention, FFN, RoPE |
| `crates/synapse-inference/src/model/causal_lm.rs` | Full model: forward, prefill, decode |
| `crates/synapse-inference/src/generation/pipeline.rs` | Generation loop, speculative decoding |
| `crates/synapse-inference/src/engine.rs` | High-level InferenceEngine |
| `crates/synapse-inference/src/quantization/int8.rs` | INT8 quantized model |
| `crates/synapse-inference/src/weight_loading/` | safetensors + GGUF loading |
| `crates/synapse-inference/src/metal/dispatch.rs` | Metal GPU dispatch |

## Adding a New Model

1. Add `WeightMapper::new_model()` in `weight_loading/weight_map.rs`
2. Update `from_model_type()` to recognize the model type string
3. Update `has_head_norms` in `model/builder.rs` if needed
4. Add config JSON in `configs/`
5. Add test in `tests/integration/multi_model_validation.rs`

## Performance

Best: 14.6 tok/s INT8 decode on Qwen3-0.6B (Apple M5). Run `scripts/bench_suite.sh` for full comparison.
