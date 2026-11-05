#!/bin/bash
# Test the book's Rust examples. With --docs, also rebuild the committed HTML
# output in ../docs (publishing).
set -e

cd "$(dirname "$0")"

docs=0
if [ "${1:-}" = "--docs" ]; then
    docs=1
fi

export CARGO_TARGET_DIR=target

echo "=== Building dependencies ==="
cargo build 2>&1

if [ "$docs" = "1" ]; then
    echo ""
    echo "=== Building book ==="
    mdbook build 2>&1
fi

echo ""
echo "=== Testing code examples from markdown ==="
# Book-local target dir: the shared cache holds many stale duplicate rlibs per
# crate and rustdoc fails on multiple candidates.
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
