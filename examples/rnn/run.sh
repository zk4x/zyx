#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

# Trains on random data - no download needed.
exec cargo run --release
