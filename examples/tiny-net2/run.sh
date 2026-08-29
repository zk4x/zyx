#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

# No data or model needed.
exec cargo run --release
