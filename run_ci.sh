#!/bin/sh
# Local check gate. Run from anywhere; it cd's into each crate.
#
#   ./run_ci.sh
#   ZYX_DEV=cuda ./run_ci.sh
#
# Covers unit and integration tests, doctests, private-item rustdoc, formatting,
# the examples, the Python bindings (when a torch-enabled interpreter is
# available), and the book's code examples. It needs the prerequisites in
# INSTALL.md (a C compiler is enough; no GPU required).
#
# The test runs use `ZYX_DEV` when it is set and `cpu` otherwise.
#
# Every test binary — the lib and each integration test file — runs in its own
# process under a wall-clock cap (`TEST_CAP`, 30 s): a file that runs longer is
# a failed gate, not a slow one. Compilation is not capped; the targets are
# built once up front.
set -e

repo=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
dev=${ZYX_DEV:-cpu}
cap=${TEST_CAP:-30}

# Run one crate's lib tests and every integration test file, each on its own.
run_tests() {
    crate=$1
    dir=$repo/$crate
    (cd "$dir" && cargo test --lib --tests --no-run --quiet)
    names="lib"
    if [ -d "$dir/tests" ]; then
        names="$names $(cd "$dir" && ls tests/*.rs 2>/dev/null | sed 's#tests/##; s#\.rs$##')"
    fi
    for name in $names; do
        if [ "$name" = lib ]; then
            sel="--lib"
        else
            sel="--test $name"
        fi
        start=$(date +%s)
        if (cd "$dir" && ZYX_DEV="$dev" timeout -k 5 "$cap" cargo test $sel --quiet -- --skip mnist); then
            echo "ci: $crate $name: ok ($(( $(date +%s) - start ))s)"
        else
            rc=$?
            echo "ci: $crate $name: FAILED rc=$rc ($(( $(date +%s) - start ))s)"
            return 1
        fi
    done
}

run_tests zyx
for crate in zyx-nn zyx-optim zyx-derive; do
    run_tests "$crate"
done

for crate in zyx zyx-nn zyx-optim zyx-derive; do
    echo "ci: $crate doctests"
    (cd "$repo/$crate" && cargo test --doc)
done

for crate in zyx zyx-nn zyx-optim zyx-derive; do
    echo "ci: $crate docs (private items)"
    (cd "$repo/$crate" && cargo doc --no-deps --document-private-items)
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
    # The tests exercise the installed extension, so it must be the one built
    # from this tree: a stale wheel turns a Rust change into a Python failure
    # (or worse, a pass). `maturin build` + install works with or without a
    # virtualenv; `maturin develop` does not, so it is never used.
    if command -v maturin >/dev/null 2>&1; then
        echo "ci: zyx-py build"
        (cd "$repo/zyx-py" && maturin build --release --quiet)
        wheel=$(ls -t "$repo"/zyx-py/target/wheels/zyx_py-*.whl | head -1)
        python3 -m pip install --force-reinstall --no-deps --quiet "$wheel"
    else
        echo "ci: zyx-py build skipped (no maturin): the installed extension is not this tree"
    fi
    echo "ci: zyx-py python tests"
    (cd "$repo/zyx-py" && python3 -m pytest tests)
else
    echo "ci: zyx-py python tests skipped (python3 has no torch/pytest)"
fi

echo "ci: book examples"
(cd "$repo/zyx-book" && ./test_book.sh)

echo "ci: all checks passed"
