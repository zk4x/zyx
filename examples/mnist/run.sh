#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

# Download + convert MNIST to examples/data/mnist_dataset.safetensors
# (skipped when the dataset already exists).
if [ ! -f ../data/mnist_dataset.safetensors ]; then
    python3.12 download_mnist.py
fi

exec cargo run --release
