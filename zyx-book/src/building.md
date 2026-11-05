# Building from Source

Users installing zyx as a dependency do not need this chapter — `cargo add zyx`
or `pip install zyx-py` is enough. Prerequisites for every backend are listed in
[INSTALL.md](https://github.com/zk4x/zyx/blob/main/INSTALL.md).

## Prerequisites

- Rust **1.88** or newer
- a C compiler (`clang` or `gcc`) for the C backend
- optionally, a GPU runtime (CUDA, OpenCL, Vulkan, WGPU) or the Tenstorrent
  simulator; a missing runtime only means that backend contributes no devices

The `zyx` crate has path dependencies on two crates from
[metalium-rs](https://github.com/Knight-Ops/metalium-rs): the Tenstorrent ISA
tables and firmware images. A **sibling checkout** is therefore required:

```bash
git clone https://github.com/zk4x/zyx
git clone https://github.com/Knight-Ops/metalium-rs
cd zyx/zyx
cargo build
```

The layout must be:

```text
your-workspace/
├── zyx/
└── metalium-rs/
```

## Tests

No GPU is needed for the CPU backend:

```bash
cd zyx
AGENT=1 ZYX_DEV=cpu cargo test
AGENT=1 cargo test --doc
```

`AGENT=1` strips ANSI colors; `ZYX_DEV=cpu` pins the C backend. A single test
binary is `cargo test --test 01_unary`, and a single test is
`cargo test --test 01_unary abs_1 -- --exact`.

## The Local Check Gate

The same checks the pre-push hook runs live in `run_ci.sh` at the repository
root:

```bash
./run_ci.sh
```

It runs the unit and integration tests, the doctests for every README-bearing
crate, rustdoc with private items (so stale doc links fail), the examples,
the Python bindings when a torch-enabled interpreter is available,
`cargo fmt --check`, and the book's code examples.

## Examples

```bash
cd examples
./tiny-net/run.sh          # simplest
./readme-test/run.sh       # transformer block
```

Each example has a `run.sh` that downloads its data if needed and runs in
release mode.

## The Book Itself

```bash
cd zyx-book
./test_book.sh             # dependencies + mdbook test
./test_book.sh --docs      # also rebuild the committed docs/ output
```
