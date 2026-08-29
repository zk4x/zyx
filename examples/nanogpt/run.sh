#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

# Download the tiny shakespeare corpus as training input
# (skipped when data/input.txt already exists).
mkdir -p data
if [ ! -f data/input.txt ]; then
    curl -L -o data/input.txt \
        https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
fi

exec cargo run --release
