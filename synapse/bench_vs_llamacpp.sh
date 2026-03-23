#!/usr/bin/env bash
set -euo pipefail

# Synapse vs llama.cpp Benchmark Script (Phase 4)
# Model: Qwen3-0.6B
# Compares: Synapse (f32, SIMD + KV-cache) vs llama.cpp (F16) vs llama.cpp (Q4_K_M)
#
# Phase 4 targets:
#   CPU-SIMD decode >= 5 tok/s on Qwen3-0.6B (from 0.3)
#   CPU-SIMD prefill >= 50 tok/s on Qwen3-0.6B pp128 (from 5)
#   Metal decode >= 30 tok/s (if Metal feature enabled)
#   llama.cpp gap <= 5x (from ~270x)

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
MODEL_DIR="/tmp/qwen3-0.6b"
GGUF_DIR="/tmp/qwen3-0.6b-gguf"
RESULTS_FILE="$SCRIPT_DIR/benchmark_results_$(date +%Y%m%d-%H%M%S).txt"

PP=128   # prompt tokens (prefill)
TG=64    # generated tokens (decode)
THREADS=1

echo "============================================" | tee "$RESULTS_FILE"
echo " Synapse vs llama.cpp Benchmark (Phase 4)" | tee -a "$RESULTS_FILE"
echo " Model: Qwen3-0.6B" | tee -a "$RESULTS_FILE"
echo " Date: $(date)" | tee -a "$RESULTS_FILE"
echo " PP=$PP TG=$TG Threads=$THREADS" | tee -a "$RESULTS_FILE"
echo "" | tee -a "$RESULTS_FILE"
echo " Phase 4 targets:" | tee -a "$RESULTS_FILE"
echo "   SIMD decode >= 5 tok/s" | tee -a "$RESULTS_FILE"
echo "   SIMD prefill >= 50 tok/s (pp128)" | tee -a "$RESULTS_FILE"
echo "   llama.cpp gap <= 5x" | tee -a "$RESULTS_FILE"
echo "============================================" | tee -a "$RESULTS_FILE"
echo "" | tee -a "$RESULTS_FILE"

# --- Prerequisites ---
check_prereqs() {
    if ! command -v llama-bench &>/dev/null; then
        echo "ERROR: llama-bench not found. Install with: brew install llama.cpp"
        exit 1
    fi
    if [ ! -d "$MODEL_DIR" ]; then
        echo "ERROR: Model dir $MODEL_DIR not found."
        echo "Download with: huggingface-cli download Qwen/Qwen3-0.6B --local-dir $MODEL_DIR"
        exit 1
    fi
}

# --- Download GGUF models if needed ---
download_gguf() {
    mkdir -p "$GGUF_DIR"

    if [ ! -f "$GGUF_DIR/qwen3-0.6b-f16.gguf" ]; then
        echo "Downloading F16 GGUF..."
        huggingface-cli download Qwen/Qwen3-0.6B-GGUF qwen3-0.6b-f16.gguf --local-dir "$GGUF_DIR"
    fi

    if [ ! -f "$GGUF_DIR/qwen3-0.6b-q4_k_m.gguf" ]; then
        echo "Downloading Q4_K_M GGUF..."
        huggingface-cli download Qwen/Qwen3-0.6B-GGUF qwen3-0.6b-q4_k_m.gguf --local-dir "$GGUF_DIR"
    fi
}

# --- llama.cpp benchmarks ---
bench_llamacpp() {
    local label="$1"
    local model_path="$2"

    echo "--- llama.cpp ($label) ---" | tee -a "$RESULTS_FILE"

    # Run llama-bench (outputs a markdown table)
    llama-bench \
        -m "$model_path" \
        -p "$PP" \
        -n "$TG" \
        -t "$THREADS" \
        2>&1 | tee -a "$RESULTS_FILE"

    # Measure peak RSS
    echo "" | tee -a "$RESULTS_FILE"
    echo "Peak memory:" | tee -a "$RESULTS_FILE"
    /usr/bin/time -l llama-bench \
        -m "$model_path" \
        -p "$PP" \
        -n "$TG" \
        -t "$THREADS" \
        2>&1 | grep "maximum resident set size" | tee -a "$RESULTS_FILE"

    echo "" | tee -a "$RESULTS_FILE"
}

# --- Synapse benchmark ---
bench_synapse() {
    echo "--- Synapse Phase 4 (f32, SIMD + KV-cache) ---" | tee -a "$RESULTS_FILE"

    cd "$SCRIPT_DIR"

    # Run model_benchmark example (reports SIMD prefill, KV-cache decode, memory)
    echo "Running model_benchmark (Phase 4 metrics)..." | tee -a "$RESULTS_FILE"
    /usr/bin/time -l cargo run --example model_benchmark --release 2>&1 | tee -a "$RESULTS_FILE"

    echo "" | tee -a "$RESULTS_FILE"

    # Run with full-scale Qwen3-0.6B if model is available
    if [ -d "$MODEL_DIR" ]; then
        echo "Running model_benchmark --full-scale with real model..." | tee -a "$RESULTS_FILE"
        /usr/bin/time -l cargo run --example model_benchmark --release -- --full-scale 2>&1 | tee -a "$RESULTS_FILE" || true

        echo "" | tee -a "$RESULTS_FILE"

        echo "Running qwen3_chat with real model..." | tee -a "$RESULTS_FILE"
        echo "Hello" | timeout 30 cargo run --example qwen3_chat --release -- --model-dir "$MODEL_DIR" 2>&1 | tee -a "$RESULTS_FILE" || true
    fi

    echo "" | tee -a "$RESULTS_FILE"
}

# --- Main ---
check_prereqs
download_gguf

echo "=== LLAMA.CPP F16 ===" | tee -a "$RESULTS_FILE"
bench_llamacpp "F16" "$GGUF_DIR/qwen3-0.6b-f16.gguf"

echo "=== LLAMA.CPP Q4_K_M ===" | tee -a "$RESULTS_FILE"
bench_llamacpp "Q4_K_M" "$GGUF_DIR/qwen3-0.6b-q4_k_m.gguf"

echo "=== SYNAPSE ===" | tee -a "$RESULTS_FILE"
bench_synapse

# --- Phase 4 Summary ---
echo "" | tee -a "$RESULTS_FILE"
echo "============================================" | tee -a "$RESULTS_FILE"
echo " Phase 4 Target Verification" | tee -a "$RESULTS_FILE"
echo "============================================" | tee -a "$RESULTS_FILE"
echo "" | tee -a "$RESULTS_FILE"
echo " Target                      Status" | tee -a "$RESULTS_FILE"
echo " ─────────────────────────── ──────" | tee -a "$RESULTS_FILE"
echo " SIMD decode >= 5 tok/s      [check model_benchmark output above]" | tee -a "$RESULTS_FILE"
echo " SIMD prefill >= 50 tok/s    [check model_benchmark output above]" | tee -a "$RESULTS_FILE"
echo " Metal decode >= 30 tok/s    [requires --features metal]" | tee -a "$RESULTS_FILE"
echo " llama.cpp gap <= 5x         [compare numbers above]" | tee -a "$RESULTS_FILE"
echo "" | tee -a "$RESULTS_FILE"
echo " Phase 4 optimizations included:" | tee -a "$RESULTS_FILE"
echo "   - SIMD-accelerated matmul, RMSNorm, SwiGLU (Zig FFI)" | tee -a "$RESULTS_FILE"
echo "   - KV-cache with zero-alloc append/slice" | tee -a "$RESULTS_FILE"
echo "   - INT8 weight quantization (~25% f32 size)" | tee -a "$RESULTS_FILE"
echo "   - Metal GPU backend (optional, --features metal)" | tee -a "$RESULTS_FILE"
echo "" | tee -a "$RESULTS_FILE"

# Run regression tests as part of benchmark verification
echo "--- Regression Test Suite ---" | tee -a "$RESULTS_FILE"
cd "$SCRIPT_DIR"
echo "Running: cargo test -p synapse-inference" | tee -a "$RESULTS_FILE"
if cargo test -p synapse-inference 2>&1 | tail -5 | tee -a "$RESULTS_FILE"; then
    echo "  PASS: synapse-inference unit tests" | tee -a "$RESULTS_FILE"
else
    echo "  FAIL: synapse-inference unit tests" | tee -a "$RESULTS_FILE"
fi

echo "Running: cargo test --test inference_e2e" | tee -a "$RESULTS_FILE"
if cargo test --test inference_e2e 2>&1 | tail -3 | tee -a "$RESULTS_FILE"; then
    echo "  PASS: inference_e2e" | tee -a "$RESULTS_FILE"
else
    echo "  FAIL: inference_e2e" | tee -a "$RESULTS_FILE"
fi

echo "Running: cargo test --test kvcache_correctness" | tee -a "$RESULTS_FILE"
if cargo test --test kvcache_correctness 2>&1 | tail -3 | tee -a "$RESULTS_FILE"; then
    echo "  PASS: kvcache_correctness" | tee -a "$RESULTS_FILE"
else
    echo "  FAIL: kvcache_correctness" | tee -a "$RESULTS_FILE"
fi

echo "Running: cargo test --test prefill_throughput" | tee -a "$RESULTS_FILE"
if cargo test --test prefill_throughput 2>&1 | tail -3 | tee -a "$RESULTS_FILE"; then
    echo "  PASS: prefill_throughput" | tee -a "$RESULTS_FILE"
else
    echo "  FAIL: prefill_throughput" | tee -a "$RESULTS_FILE"
fi

echo "" | tee -a "$RESULTS_FILE"
echo "============================================" | tee -a "$RESULTS_FILE"
echo "Results saved to: $RESULTS_FILE" | tee -a "$RESULTS_FILE"
echo "============================================" | tee -a "$RESULTS_FILE"
