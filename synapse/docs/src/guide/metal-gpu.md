# Metal GPU

Synapse includes a Metal backend for macOS on Apple Silicon, but the public docs now treat Metal as a runtime-measured path instead of a feature-flag assumption.

## Enabling Metal

Build with the `metal` feature:

```bash
cargo build --release --features metal
```

Then run a real checkpoint and inspect the runtime line:

```bash
cargo run --example qwen3_chat --release --features metal -- --model-dir /tmp/qwen3-0.6b --prompt "hello"
```

The only trustworthy statement about the active backend is the printed `Runtime:` line.

## Reporting Rules

- A metal-feature build is **not** automatically a GPU benchmark.
- A row counts as Metal only when the runtime line reports `backend=metal`.
- If the run falls back to `cpu_simd`, that fallback is preserved in the raw matrix artifact but excluded from the public headline performance table.

This avoids stale “Metal is fast” claims surviving after backend routing or fallback behavior changes.

## What Metal Should Accelerate

Metal is most compelling when the runtime actually stays on the GPU for the expensive path:

- large prompt prefill matmuls,
- GPU-resident decode state,
- reduced CPU dispatch overhead when the command flow is truly backend-driven.

If decode still reports a CPU runtime path, the docs should treat the run as CPU-backed even if the binary was compiled with Metal support.

## Relationship to Shared Kernel Work

The recent decode recovery on CPU came from shared causal-LM infrastructure: the `M=1` INT8 GEMV path, quantized LM head, and cached decode flow. Metal should inherit the same architectural improvements when the end-to-end backend dispatch path is active, but Synapse should not publish Metal throughput claims until the runtime report and the measured artifact agree.

## Current Limits

- **macOS only**: Metal is not available on Linux or Windows.
- **Runtime-dependent**: active backend may change depending on the build, device detection, and dispatch routing.
- **Single-sequence focus**: the current public matrix is centered on single-sequence local inference rather than large batched serving.
- **Documentation gate**: public Metal rows must be regenerated from the benchmark matrix, not copied forward manually.
