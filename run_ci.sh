#!/bin/sh
# Local check gate. Run from anywhere; it cd's into each crate.
#
#   ./run_ci.sh
#
# Covers unit and integration tests, doctests, private-item rustdoc, formatting,
# the examples, the Python bindings (when a torch-enabled interpreter is
# available), and the book's code examples. It needs the prerequisites in
# INSTALL.md (a C compiler is enough; no GPU required).
set -e

repo=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)

echo "ci: zyx tests"
(cd "$repo/zyx" && AGENT=1 ZYX_DEV=cpu cargo test --lib --tests -- --skip mnist)

for crate in zyx-nn zyx-optim zyx-derive; do
    echo "ci: $crate tests"
    (cd "$repo/$crate" && AGENT=1 ZYX_DEV=cpu cargo test --lib --tests)
done

for crate in zyx zyx-nn zyx-optim zyx-derive; do
    echo "ci: $crate doctests"
    (cd "$repo/$crate" && AGENT=1 cargo test --doc)
done

for crate in zyx zyx-nn zyx-optim zyx-derive; do
    echo "ci: $crate docs (private items)"
    (cd "$repo/$crate" && AGENT=1 cargo doc --no-deps --document-private-items)
done

echo "ci: formatting"
(cd "$repo/zyx" && cargo fmt --check)

echo "ci: examples (build)"
# `tt_gemm` and `qwen3-8-27b` still target the pre-rewrite Tenstorrent kernel
# API; exclude them until those two examples are ported.
(cd "$repo/examples" && cargo check --workspace \
    --exclude tt_gemm --exclude qwen3-8-27b --all-targets)

for example in tiny-net tiny-net2 readme-test rnn; do
    echo "ci: example $example"
    (cd "$repo/examples" && cargo run --release --quiet -p "$example")
done

if python3 -c 'import torch, pytest' >/dev/null 2>&1; then
    if [ -n "${VIRTUAL_ENV:-}" ] && command -v maturin >/dev/null 2>&1; then
        echo "ci: zyx-py build"
        (cd "$repo/zyx-py" && maturin develop --release)
    fi
    echo "ci: zyx-py python tests"
    (cd "$repo/zyx-py" && python3 -m pytest tests)
else
    echo "ci: zyx-py python tests skipped (python3 has no torch/pytest)"
fi

echo "ci: book examples"
(cd "$repo/zyx-book" && ./test_book.sh)

echo "ci: all checks passed"
