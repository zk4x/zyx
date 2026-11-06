#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

MODELS_DIR="$(pwd)/../models/phi-1.5"
mkdir -p "$MODELS_DIR"

# Download phi-1.5 weights and tokenizer (skipped when they already exist).
for file in model.safetensors tokenizer.json; do
    if [ ! -f "$MODELS_DIR/$file" ]; then
        python3.12 -c "
from huggingface_hub import hf_hub_download
hf_hub_download('microsoft/phi-1_5', '$file', local_dir='$MODELS_DIR', local_dir_use_symlinks=False)
"
    fi
done

exec cargo run --release
