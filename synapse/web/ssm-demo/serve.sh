#!/bin/bash
# Serve the SSM demo locally.
#
# Prerequisites:
#   1. Build WASM: cd synapse-wasm && wasm-pack build --target web --release
#   2. Copy model files to this directory:
#      cp models/mamba-130m/config.json web/ssm-demo/
#      cp models/mamba-130m/model.safetensors web/ssm-demo/
#      # For tokenizer (from EleutherAI cache):
#      cp ~/.cache/huggingface/hub/models--EleutherAI--gpt-neox-20b/snapshots/*/tokenizer.json web/ssm-demo/
#
# Then run: cd web/ssm-demo && bash serve.sh
#
# Opens http://localhost:8080

echo "Serving SSM demo at http://localhost:8080"
echo "Press Ctrl-C to stop."
echo ""
echo "Note: model.safetensors is 671MB — first load will take a moment."
echo ""

python3 -m http.server 8080 --directory "$(dirname "$0")/../.."
