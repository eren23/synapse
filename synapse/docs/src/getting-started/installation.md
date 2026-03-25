# Installation

## Prerequisites

- **Rust 1.75+** -- install via [rustup](https://rustup.rs/)
- **Zig 0.13+** -- install from [ziglang.org](https://ziglang.org/download/) or via `brew install zig`
- **macOS** (for Metal GPU support) or Linux/x86 for AVX2 kernels
- **huggingface-cli** (optional) -- for downloading models: `pip install huggingface_hub`

## Build

Clone and build with optimizations:

```bash
git clone https://github.com/user/synapse.git
cd synapse
cargo build --release
```

The build system automatically compiles the Zig SIMD kernels via `build.rs` in the `synapse-sys` crate. No manual Zig build step is needed.

## Metal GPU Support

To enable Metal GPU acceleration (macOS only):

```bash
cargo build --release --features metal
```

This compiles the Metal shaders and enables GPU dispatch for matrix operations during prefill.

## Verify Installation

Run the library test suite to confirm everything works:

```bash
cargo test -p synapse-inference --lib
```

This runs 210+ tests covering model loading, quantization, attention, KV cache, and generation.

## Troubleshooting

- **Zig not found**: Ensure `zig` is on your PATH. The build script invokes `zig build` directly.
- **Metal errors on Linux**: Metal is macOS-only. Build without `--features metal` on Linux.
- **Slow first build**: The Zig compilation and Rust release build take 2-3 minutes on first run. Subsequent builds are incremental.
