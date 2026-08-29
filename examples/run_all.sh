#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

# Run every example, simplest to hardest. Each example's run.sh downloads

run_example() {
    echo "=== $1 ==="
    (cd "$1" && ./run.sh) || exit 1
}

# Simplest to hardest.
for ex in probe tiny-net tiny-net2 readme-test rnn mnist mnist-snn nanogpt phi llama; do
    run_example "$ex"
done
