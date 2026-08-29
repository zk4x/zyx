#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

# Download phi-1.5 weights (skipped when the safetensors file already exists).
if [ ! -f phi1_5-model.safetensors ]; then
    python3.12 -c "
from huggingface_hub import hf_hub_download
p = hf_hub_download('microsoft/phi-1_5', 'phi-1_5-model.safetensors')
import shutil
shutil.move(p, 'phi1_5-model.safetensors')
"
fi

exec cargo run --release
