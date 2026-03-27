#!/bin/bash
# Synapse benchmark wrapper.
#
# Usage:
#   ./scripts/bench_suite.sh
#   ./scripts/bench_suite.sh --model-dir /path/to/qwen3
#   ./scripts/bench_suite.sh --include-exploratory

set -euo pipefail
cd "$(dirname "$0")/.."

ARGS=()
MODEL_DIR=""
INCLUDE_EXPLORATORY=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model-dir)
            MODEL_DIR="${2:?--model-dir requires a path}"
            shift 2
            ;;
        --include-exploratory)
            INCLUDE_EXPLORATORY=1
            shift
            ;;
        *)
            ARGS+=("$1")
            shift
            ;;
    esac
done

if [[ -n "$MODEL_DIR" ]]; then
    ARGS+=("--official-model" "qwen3=$MODEL_DIR")
fi

if [[ "$INCLUDE_EXPLORATORY" -eq 1 ]]; then
    ARGS+=("--include-exploratory")
fi

python3 scripts/benchmark_matrix.py --format text "${ARGS[@]}"
