# Contributing

Check out [good first issues](https://github.com/zk4x/zyx/labels/good%20first%20issue)
for a place to start.

## Ways to help

- **Write integration tests.** The simplest and most appreciated contribution.
- **Improve documentation.** The architecture book (`zyx-book/`) and the API
  docs both welcome fixes; examples in the book and READMEs are compiled by the
  check gate.
- **Add `Tensor` functions.** New stateful modules go into `zyx-nn`; pure
  stateless functions go directly on `Tensor`.
- **Add a backend.** Read [ADDING_BACKENDS.md](ADDING_BACKENDS.md). A backend is
  roughly one file in `zyx/src/backend/` plus `DeviceInfo`; no compile-time
  linking is allowed — runtimes are loaded at runtime.
- **Work on optimizations.** The hardest work by far; it is easy to introduce
  bugs. Expect review to focus on provable semantics. Large, measured
  improvements (>10% on some device) are the most valuable.
- **Find bugs.** Correctness is goal number one; a good reproducer is a great
  contribution. **Fixing** bugs is welcome too.

## Development setup

Prerequisites are in [INSTALL.md](INSTALL.md). The short version:

```bash
git clone https://github.com/zk4x/zyx
cd zyx/zyx
cargo build
```

There is no root cargo workspace; the crate lives in `zyx/`, so prefix crate
commands with `cd zyx` (the book is `zyx-book/`, examples are `examples/`).

## Tests

```bash
cd zyx
AGENT=1 ZYX_DEV=cpu cargo test --lib --tests -- --skip mnist
AGENT=1 ZYX_DEV=cpu cargo test --test 01_unary  # one test binary
AGENT=1 cargo test --doc                        # doctests
```

`--skip mnist` leaves out the integration test that needs the MNIST dataset in
`examples/data/`; it is the same filter `run_ci.sh` uses.

`AGENT=1` only strips ANSI colors. `ZYX_DEV=cpu` avoids GPU-driver flakiness
when you do not need a GPU; omit it to let zyx pick the fastest available
device. Backend selection is described in [CONFIG.md](CONFIG.md) and
[ENV_VARS.md](ENV_VARS.md).

Tenstorrent hardware needs the `tenstorrent` cargo feature, the
`riscv32im-unknown-none-elf` target, `llvm-tools`, and the simulator library
(`TT_TTSIM_LIB`); see [INSTALL.md](INSTALL.md). A failed simulator init is an
error, never a retry — do not loop on it.

## Checks before a PR

Run the same gate locally:

```bash
./run_ci.sh
```

It runs the unit and integration tests (CPU), doctests for every
README-bearing crate, private-item rustdoc (so broken doc links fail), the
examples, the Python bindings when a torch-enabled interpreter is available,
`cargo fmt --check`, and the book's code examples.

Formatting: only `zyx/` has a `rustfmt.toml`, so run `cargo fmt` inside `zyx/`.
Do not reformat the other crates with the default config — it would rewrap them
at a different width.

Documentation changes: the book is built and tested with

```bash
cd zyx-book
./test_book.sh          # build dependencies + mdbook test
./test_book.sh --docs   # also rebuild the committed docs/ output
```

Only pass `--docs` when you intend to publish the regenerated site.

Python changes: see [zyx-py/README.md](zyx-py/README.md) for the `maturin`
workflow.

## Code style

Read [STYLE.md](STYLE.md). In short: debuggable over “clean”, duplicate before
abstracting, no silent fallbacks, and a new op needs a correct arm in every
match (or a loud `todo!()`).

The repository also contains [AGENTS.md](AGENTS.md), which is guidance for AI
coding agents rather than humans.

## License and sign-off

zyx is licensed under `LGPL-3.0-only WITH Classpath-exception-2.0` (see
[LICENSE.md](LICENSE.md)).

By contributing code you agree your contribution is licensed under the same
terms, and you retain copyright. Every commit needs a `Signed-off-by` line
(Developer Certificate of Origin): use `git commit -s`. No CLA is required.
