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

# Local llama.cpp build; override with LLAMA_COMPLETION=/path/to/llama-completion.
# `llama-cli` is chat-only in this build and rejects -no-cnv, so the raw
# completion binary is the one that takes the same prompt as the zyx CLI.
LLAMA_COMPLETION="${LLAMA_COMPLETION:-$HOME/Dev/cpp/llama_cpp/llama.cpp/build/bin/llama-completion}"

# Same settings as run.sh / the zyx CLI defaults:
#   -no-cnv        raw completion: no chat template, the tokenizer sees the
#                  same bytes the zyx CLI encodes for the prompt
#   --temp 0       greedy (zyx samples argmax when --temperature is unset)
#   -co off        no ANSI colours around prompt and generation
#   -ngl 99        whole model on the GPU, like the zyx device path
#   repeat penalty defaults to 1.0 in both.
if [ $# -eq 0 ]; then
    exec "$LLAMA_COMPLETION" \
      -m "$GGUF" \
      -p "Hello" \
      -n 500 \
      -no-cnv \
      --temp 0 \
      -co off \
      -ngl 99
else
    exec "$LLAMA_COMPLETION" \
      -m "$GGUF" \
      -no-cnv \
      --temp 0 \
      -co off \
      -ngl 99 \
      "$@"
fi
