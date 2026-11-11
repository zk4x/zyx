# Building from Source

Users installing zyx as a dependency do not need this chapter — `cargo add zyx`
or `pip install zyx-py` is enough. Prerequisites for every backend are listed in
[INSTALL.md](https://github.com/zk4x/zyx/blob/main/INSTALL.md).

## Prerequisites

- Rust **1.88** or newer
- a C compiler (`clang` or `gcc`) for the C backend
- optionally, a GPU runtime (CUDA, OpenCL, Vulkan, WGPU); a missing runtime only
  means that backend contributes no devices
- for the optional Tenstorrent backend, the `libttsim_bh.so` simulator, the
  `riscv32im-unknown-none-elf` target, and `llvm-tools`

The Tenstorrent firmware lives in `zyx/firmware/tt-firmware` and is built by
`zyx/build.rs` when the `tenstorrent` feature is enabled; the ISA tables come
from the `tt-isa` crate. No sibling checkout is required:

```bash
git clone https://github.com/zk4x/zyx
cd zyx/zyx
cargo build
```

The Tenstorrent backend additionally needs the RISC-V target and LLVM tools that
`build.rs` uses for the firmware images:

```bash
rustup target add riscv32im-unknown-none-elf
rustup component add llvm-tools
cargo build --features tenstorrent
```

## Tests

No GPU is needed for the CPU backend:

```bash
cd zyx
AGENT=1 ZYX_DEV=cpu cargo test --lib --tests -- --skip mnist
AGENT=1 cargo test --doc
```

`AGENT=1` strips ANSI colors; `ZYX_DEV=cpu` pins the C backend. `--skip mnist`
leaves out the integration test that needs the MNIST dataset in
`examples/data/`. A single test binary is `cargo test --test 01_unary`, and a
single test is `cargo test --test 01_unary abs_1 -- --exact`.

## The Local Check Gate

The full local check gate lives in `run_ci.sh` at the repository root:

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
