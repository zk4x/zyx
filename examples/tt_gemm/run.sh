#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

# No data or model needed. Needs TT_METAL_ROOT for the Tenstorrent runtime build.
exec cargo run --release
