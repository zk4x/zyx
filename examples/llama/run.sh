#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

MODELS_DIR="$(pwd)/../models"
mkdir -p "$MODELS_DIR"

GGUF=$(python3 -c "
from huggingface_hub import hf_hub_download
p = hf_hub_download('bartowski/Llama-3.2-3B-Instruct-GGUF', 'Llama-3.2-3B-Instruct-f16.gguf', local_dir='$MODELS_DIR/llama-3.2-3b', local_dir_use_symlinks=False)
print(p)
")

CONFIG=$(python3 -c "
from huggingface_hub import hf_hub_download
import os
p = hf_hub_download('unsloth/Llama-3.2-3B-Instruct', 'config.json', local_dir='$MODELS_DIR/llama-3.2-3b', local_dir_use_symlinks=False)
print(os.path.realpath(p))
")

TOKENIZER=$(python3 -c "
from huggingface_hub import hf_hub_download
import os
p = hf_hub_download('unsloth/Llama-3.2-3B-Instruct', 'tokenizer.json', local_dir='$MODELS_DIR/llama-3.2-3b', local_dir_use_symlinks=False)
print(os.path.realpath(p))
")

if [ $# -eq 0 ]; then
    exec cargo run --release -- \
      --weight-file "$GGUF" \
      --config-file "$CONFIG" \
      --tokenizer-file "$TOKENIZER" \
      --prompt "Hello"
else
    exec cargo run --release -- \
      --weight-file "$GGUF" \
      --config-file "$CONFIG" \
      --tokenizer-file "$TOKENIZER" \
      "$@"
fi
