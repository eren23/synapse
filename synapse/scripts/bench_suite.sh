#!/bin/bash
# Synapse Benchmark Suite
# Runs f32, INT8 decode and prefill benchmarks, plus isolated matmul comparisons.
#
# Usage:
#   ./scripts/bench_suite.sh [--model-dir /path/to/model]
#
# If --model-dir is provided, runs real model benchmarks.
# Otherwise runs synthetic benchmarks only.

set -e
cd "$(dirname "$0")/.."

MODEL_DIR=""
for arg in "$@"; do
    case "$arg" in
        --model-dir) shift; MODEL_DIR="$1"; shift ;;
    esac
done

echo "═══════════════════════════════════════════════════════════════"
echo " Synapse Benchmark Suite — $(date '+%Y-%m-%d %H:%M')"
echo "═══════════════════════════════════════════════════════════════"
echo ""

# ── Isolated matmul benchmarks ────────────────────────────────────
echo "▶ Isolated matmul: f32 vs INT8 vs Q4 (M=1 decode dimensions)"
echo "─────────────────────────────────────────────────────────────"
cargo test --test quantization_speedup --release -- --nocapture isolated_matmul 2>&1 | grep -E "^(FFN|Attn|Prefill)"
echo ""

# ── INT8 vs f32 full-model benchmark ──────────────────────────────
echo "▶ Full-model quantization speedup (tiny model, synthetic)"
echo "─────────────────────────────────────────────────────────────"
cargo test --test quantization_speedup --release -- --nocapture quantization_speedup_int8_vs_f32 2>&1 | grep -E "^(f32|INT8|Speedup)"
echo ""

# ── Real model benchmarks (if model dir provided) ────────────────
if [ -n "$MODEL_DIR" ]; then
    echo "▶ Real model benchmarks: $MODEL_DIR"
    echo "─────────────────────────────────────────────────────────────"

    echo ""
    echo "  [f32 CPU]"
    echo "hello" | cargo run --example qwen3_chat --release -- --model-dir "$MODEL_DIR" 2>&1 | grep "Prefill:"

    echo ""
    echo "  [INT8 CPU]"
    echo "hello" | cargo run --example qwen3_chat --release -- --model-dir "$MODEL_DIR" --quantize 2>&1 | grep "Prefill:"

    echo ""
    echo "  [f32 + Metal]"
    echo "hello" | cargo run --example qwen3_chat --release --features metal -- --model-dir "$MODEL_DIR" 2>&1 | grep "Prefill:" || echo "  (Metal not available)"

    echo ""
    echo "  [INT8 + Metal]"
    echo "hello" | cargo run --example qwen3_chat --release --features metal -- --model-dir "$MODEL_DIR" --quantize 2>&1 | grep "Prefill:" || echo "  (Metal not available)"
fi

echo ""
echo "═══════════════════════════════════════════════════════════════"
echo " Tests: $(cargo test -p synapse-inference --lib 2>&1 | grep 'test result:' | head -1)"
echo "═══════════════════════════════════════════════════════════════"
