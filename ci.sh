#!/bin/sh
# Local check gate. Run from anywhere; it cd's into each crate.
#
#   ./ci.sh
#
# Covers unit tests, doctests, private-item rustdoc, formatting, and the book's
# code examples. It needs the prerequisites in INSTALL.md (a C compiler is
# enough; no GPU required).
set -e

repo=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)

echo "ci: zyx unit tests"
(cd "$repo/zyx" && AGENT=1 cargo test --lib)

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

echo "ci: book examples"
(cd "$repo/zyx-book" && ./test_book.sh)

echo "ci: all checks passed"
