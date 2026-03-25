# Building

## Standard Build

Build the full workspace in release mode:

```bash
cargo build --release
```

This automatically rebuilds the Zig SIMD library via `build.rs` in the `synapse-sys` crate. The Zig build is incremental -- only recompiles when source files change.

## Manual Zig Build

To rebuild the Zig library independently:

```bash
cd zig && zig build -Doptimize=ReleaseFast
```

This produces a static library that `synapse-sys` links against. Useful for iterating on kernel code without a full Cargo rebuild.

## Feature Flags

| Feature | Description | Default |
|---------|-------------|---------|
| `metal` | Metal GPU acceleration (macOS only) | Off |

Build with Metal:

```bash
cargo build --release --features metal
```

## Build Requirements

- **Rust 1.75+** with the stable toolchain
- **Zig 0.13+** on PATH
- **macOS**: Xcode Command Line Tools (for Metal SDK, if using `--features metal`)
- **Linux**: standard build tools (`gcc`, `make`)

## Build Times

Approximate build times (Apple M5):

| Build | Time |
|-------|------|
| Full release (first) | 2-3 min |
| Incremental Rust | 5-15 sec |
| Zig rebuild | 3-5 sec |

## Testing

Run the inference test suite:

```bash
cargo test -p synapse-inference --lib
```

This runs 210+ tests. For the full workspace:

```bash
cargo test --workspace
```

## Benchmarks

Run the full benchmark suite against a model:

```bash
bash scripts/bench_suite.sh --model-dir /path/to/qwen3-0.6b
```

Run isolated kernel benchmarks:

```bash
cargo bench -p synapse-core
```

## Troubleshooting

- **`zig` not found**: Install Zig 0.13+ and ensure it is on your PATH.
- **Linker errors**: Clean and rebuild: `cargo clean && cargo build --release`.
- **Metal compilation errors on Linux**: Remove `--features metal`. Metal is macOS-only.
- **Out of memory**: Large models may need 16+ GB RAM. INT8 quantization reduces memory 4x.
