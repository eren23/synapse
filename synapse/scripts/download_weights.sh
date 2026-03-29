#!/usr/bin/env bash
set -e

# ============================================================================
# Synapse — Model Weight Downloader
# ============================================================================
#
# Downloads public HuggingFace models needed for Synapse inference and tests.
# Idempotent: skips models that already exist locally.
#
# NOTE: The following weights are NOT covered by this script:
#
#   1. LEWM / neo_unify weights — These are custom-trained and must be
#      downloaded from the project-specific HuggingFace repo:
#        huggingface-cli download eren23/synapse-weights
#      (See project docs for placement instructions.)
#
#   2. ssm-demo INT8 binary (web/ssm-demo/mamba-130m-int8.bin) — This file
#      is generated locally from the Mamba-130M weights:
#        cargo run --example export_mamba_int8 --release -- \
#          --model-dir models/mamba-130m \
#          --output web/ssm-demo/mamba-130m-int8.bin
#
# ============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
MODELS_DIR="$PROJECT_DIR/models"
WEB_DEMO_DIR="$PROJECT_DIR/web/ssm-demo"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

status() { printf "\n\033[1;34m==> %s\033[0m\n" "$1"; }
ok()     { printf "    \033[1;32m[done]\033[0m %s\n" "$1"; }
skip()   { printf "    \033[1;33m[skip]\033[0m %s (already exists)\n" "$1"; }
info()   { printf "    \033[0;37m%s\033[0m\n" "$1"; }

HAS_HF_CLI=0
if command -v huggingface-cli &>/dev/null; then
    HAS_HF_CLI=1
fi

# hf_download REPO_ID [EXTRA_ARGS...]
# Downloads a repo via huggingface-cli, returns the snapshot path on stdout.
hf_download() {
    local repo="$1"; shift
    if [ "$HAS_HF_CLI" -eq 1 ]; then
        huggingface-cli download "$repo" "$@"
    else
        echo "ERROR: huggingface-cli is not installed." >&2
        echo "Install it with:  pip install huggingface_hub[cli]" >&2
        return 1
    fi
}

# hf_snapshot_path REPO_ID
# Resolves the local snapshot path for an already-downloaded repo.
hf_snapshot_path() {
    local repo="$1"
    local org name cache_dir
    org="$(echo "$repo" | cut -d/ -f1)"
    name="$(echo "$repo" | cut -d/ -f2)"
    cache_dir="${HF_HOME:-$HOME/.cache/huggingface}/hub/models--${org}--${name}/snapshots"
    if [ -d "$cache_dir" ]; then
        # Return the most recent snapshot
        ls -1t "$cache_dir" | head -1 | xargs -I{} echo "$cache_dir/{}"
    fi
}

# ---------------------------------------------------------------------------
# Create directories
# ---------------------------------------------------------------------------

status "Setting up directories"
mkdir -p "$MODELS_DIR"
mkdir -p "$WEB_DEMO_DIR"
ok "models/ and web/ssm-demo/"

# ---------------------------------------------------------------------------
# 1. Mamba-130M
# ---------------------------------------------------------------------------

status "Mamba-130M (state-spaces/mamba-130m)"

if [ -e "$MODELS_DIR/mamba-130m" ]; then
    skip "models/mamba-130m"
else
    info "Downloading state-spaces/mamba-130m ..."
    hf_download "state-spaces/mamba-130m" >/dev/null
    SNAP="$(hf_snapshot_path "state-spaces/mamba-130m")"
    if [ -n "$SNAP" ] && [ -d "$SNAP" ]; then
        ln -sfn "$SNAP" "$MODELS_DIR/mamba-130m"
        ok "Symlinked models/mamba-130m -> $SNAP"
    else
        echo "ERROR: Could not resolve snapshot path after download." >&2
        exit 1
    fi
fi

# ---------------------------------------------------------------------------
# 2. RWKV-7 0.1B (Pile, HF format)
# ---------------------------------------------------------------------------

status "RWKV-7 0.1B Pile (SmerkyG/RWKV7-Goose-0.1B-Pile-HF)"

if [ -e "$MODELS_DIR/rwkv7-pile-0.1b/model.safetensors" ]; then
    skip "models/rwkv7-pile-0.1b"
else
    info "Downloading SmerkyG/RWKV7-Goose-0.1B-Pile-HF ..."
    mkdir -p "$MODELS_DIR/rwkv7-pile-0.1b"
    hf_download "SmerkyG/RWKV7-Goose-0.1B-Pile-HF" \
        --local-dir "$MODELS_DIR/rwkv7-pile-0.1b" >/dev/null
    ok "Downloaded to models/rwkv7-pile-0.1b/"
    info "NOTE: You may need to run the converter script to adjust weight keys."
    info "See: scripts/reference/ or project docs for conversion steps."
fi

# ---------------------------------------------------------------------------
# 3. GPT-NeoX tokenizer (for Mamba / RWKV)
# ---------------------------------------------------------------------------

status "GPT-NeoX tokenizer (EleutherAI/gpt-neox-20b)"

TOKENIZER_DST="$WEB_DEMO_DIR/tokenizer.json"

if [ -f "$TOKENIZER_DST" ]; then
    skip "web/ssm-demo/tokenizer.json"
else
    info "Downloading tokenizer only from EleutherAI/gpt-neox-20b ..."
    # Download just the tokenizer file
    if [ "$HAS_HF_CLI" -eq 1 ]; then
        hf_download "EleutherAI/gpt-neox-20b" \
            --include "tokenizer.json" >/dev/null
    fi
    # Resolve and copy
    SNAP="$(hf_snapshot_path "EleutherAI/gpt-neox-20b")"
    if [ -n "$SNAP" ] && [ -f "$SNAP/tokenizer.json" ]; then
        cp "$SNAP/tokenizer.json" "$TOKENIZER_DST"
        ok "Copied tokenizer.json to web/ssm-demo/"
    else
        # Fallback: direct curl download
        info "Falling back to direct download ..."
        curl -sL "https://huggingface.co/EleutherAI/gpt-neox-20b/resolve/main/tokenizer.json" \
            -o "$TOKENIZER_DST"
        ok "Downloaded tokenizer.json to web/ssm-demo/"
    fi
fi

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

status "Done"
info "Models directory:  $MODELS_DIR"
info "Web demo assets:   $WEB_DEMO_DIR"
info ""
info "Next steps:"
info "  - Generate INT8 binary for WASM demo:"
info "      cargo run --example export_mamba_int8 --release -- \\"
info "        --model-dir models/mamba-130m --output web/ssm-demo/mamba-130m-int8.bin"
info "  - For LEWM/neo_unify weights, download from:"
info "      huggingface-cli download eren23/synapse-weights"
