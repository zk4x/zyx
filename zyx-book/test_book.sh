#!/bin/bash
# Build and test book — compiles code examples from markdown
set -e

cd "$(dirname "$0")"

echo "=== Building dependencies ==="
cargo build 2>&1

echo ""
echo "=== Building book ==="
mdbook build 2>&1

echo ""
echo "=== Testing code examples from markdown ==="
# Find rlibs and build --extern flags
RUSTDOCFLAGS=""
for rlib in target/debug/deps/libzyx-*.rlib target/debug/deps/libzyx_nn-*.rlib target/debug/deps/libzyx_optim-*.rlib; do
    if [ -f "$rlib" ]; then
        basename=$(basename "$rlib")
        crate_name=$(echo "$basename" | sed 's/^lib//; s/-[0-9a-f]*\.rlib$//')
        RUSTDOCFLAGS="$RUSTDOCFLAGS --extern $crate_name=$(pwd)/$rlib"
    fi
done

RUSTDOCFLAGS="$RUSTDOCFLAGS" mdbook test --library-path "$(pwd)/target/debug/deps" 2>&1

echo ""
echo "=== All book tests passed ==="
