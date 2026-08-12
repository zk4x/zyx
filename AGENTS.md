# ⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️
# ⚠️  "pre-existing" IS FORBIDDEN. NEVER USE IT.  ⚠️
# ⚠️  "Builds clean" / similar                    ⚠️
# ⚠️  IS FORBIDDEN. BUILD SILENTLY.               ⚠️
# ⚠️  "git checkout" IS FORBIDDEN. NEVER USE.     ⚠️
# ⚠️  Use "git restore" instead.                   ⚠️
# ⚠️  VIOLATIONS = BROKEN. REPEATEDLY. FOREVER.   ⚠️
# ⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️

# Agent Guidelines for zyx

## Know Your Limits

- If a task is hardware-specific, deeply subtle, or keeps failing, SAY SO plainly up front. "I can't solve this" beats two days of guesses.
- When the user says they'll do it themselves, stop. Only act when asked for a specific edit.

## Key Commands

The repo root has **no `Cargo.toml`** — crates (`zyx`, `zyx-nn`, ...) are independent packages. **Always run cargo from the `zyx/` subdirectory.**

```bash
cd zyx && cargo build                      # build
cd zyx && AGENT=1 cargo test               # test (AGENT=1 strips ANSI colors)
cd zyx && AGENT=1 cargo test relu_1        # single test
cd zyx && AGENT=1 cargo test --test 1_unary  # one test file
cd zyx && AGENT=1 cargo test -- --nocapture   # with output
cd zyx && AGENT=1 cargo test --doc         # doc tests
cd zyx && cargo fmt                        # format
```

- Tenstorrent: `TT_METAL_ROOT=~/Dev/cpp/tt-metal cargo build --features tenstorrent`
- Clippy is NOT a gate (lint `#![deny(...)]` block in `src/lib.rs` is commented out). Don't run it.
- Tests live in `zyx/tests/`, named `{number}_{category}.rs` (e.g. `1_unary.rs`) plus `mnist.rs` (`mnist.py` + `.safetensors` are fixtures). Tests return `Result<(), ZyxError>` and use `assert!` / `is_equal()` for floats.
- Python: use `python3.12` for everything.

## Layout

```
zyx/           core tensor library (all the real work)
  src/backend/   one file per backend (c, cuda, hip, opencl, vulkan, disk, host, dummy, tenstorrent, wgpu)
  src/graph/     the egraph + autograd + kernelizer + plan
  src/kernel/    kernel IR, codegen, and ALL optimization passes
  tests/         integration tests
zyx-nn, zyx-optim, zyx-onnx, zyx-fuzzy, zyx-py, zyx-bench, zyx-examples, zyx-book
```

Root docs worth reading: `ENV_VARS.md` (ZYX_DEBUG), `CONFIG.md` (backend config), `STYLE.md`, `ADDING_BACKENDS.md`.

## The Graph

Zyx is all about the graph. Lazy (no compute until `Tensor::realize`), dynamic, and **one graph serves both laziness and autograd**. TensorId is a `u32`.

`Node` (`src/graph/mod.rs`) has 14 variants: Const, Leaf, Expand, Permute, Reshape, PadZeros, Flip, Reduce, Cast, Unary, Binary, Assign, ToDevice, Kernel.

## The Egraph

Lives in `zyx/src/graph/`:
- `mod.rs` — the equivalence graph. Each `EClass` holds all equivalent node forms; rewrites produce equivalent forms (CSE via hashconsing, algebraic rewrites, layout rewrites, shape rewrites). A **cost model selects the cheapest extraction** (`Graph::extract`, `mod.rs:687`) for kernel compilation.
- `kernelizer.rs` — pattern-matches subgraphs and replaces them with custom JIT kernels
- `autograd.rs` — gradients on the same graph
- `plan.rs` — execution plans from compiled graphs

How to work with it:
- `ZYX_DEBUG=2` prints the egraph (class structure) after realize — the starting point for graph debugging.
- To replace a graph pattern with a custom kernel, add the match in `kernelizer.rs`.
- Rewrites must preserve semantic equivalence; the egraph picks the cheapest form, so a wrong-cost model surfaces as slow (not wrong) kernels.

## Code Style

- **Simplicity/debuggability > "clean"**: duplicate code first, abstract later. Use enums, not `dyn Trait`. Explicit returns. Minimize Arc/Rc.
- Keep ~1000 LOC per module; add new files only when necessary.
- License header on every new file: `// Copyright (C) 2025 zk4x` / `// SPDX-License-Identifier: LGPL-3.0-only`
- Public items need docs (`#![warn(missing_docs)]` is active). Error constructors use `#[track_caller]` and embed file:line:column. Use `From` for conversions.
- Import order: `crate::` → `super::` → external crates.

## Backends

- Most backends are **always compiled in**, selected at runtime via the zyx config file.
- **NEVER touch `~/.config/zyx/config.json`** (never read/write/create/modify/delete it). Backends are chosen by the user. If a test needs a specific backend, ask the user to configure it.
- Only two backends need cargo features: `--features wgpu` and `--features tenstorrent`.
- Defaults with no config: C on, Dummy off, CUDA/HIP/Vulkan/OpenCL try to init and silently skip if the driver is missing, HIP always tries (ignores config; skipped only if `libamdhip64.so` is missing). If all backends fail, tests print nothing.

## Optimization Passes

All kernel optimizations live in `zyx/src/kernel/` (`autotune.rs` driver + one file per pass: `split_loops.rs`, `coarsen.rs`, `vectorize.rs`, `fold_loops.rs`, `algebraic.rs`, ...).

**Correctness is critical**: no optimization is required for tests to pass; ALL tests must pass with ANY optimization sequence (including empty). If a sequence breaks correctness, the pass that produced invalid IR from valid code is BUGGY — fix or disable it. Run the full `cargo test` after touching any pass; a single failing integration test means the pass produces wrong results.

### Two pipelines

- `run_always_on_optimizations` (`autotune.rs:336`) — always runs before compilation, fixed order, **`dead_code_elimination` must stay last** (backends fail on unused ops, e.g. missing ref-count entries).
- `AVAILABLE_OPTIMIZATIONS` (`autotune.rs:59`) — 9 passes the autotuner tries in combination (split_global_to_local, reassociate_commutative, coarsen, register_blocking, local_reduce, split_loop, pad_index, vectorize, merge_nested_loops).

### Adding an optimization

1. Config fn returns `(Optimization, usize)` (optimization + #variants to try; `1` if no tunable params): `pub fn my_opt_config(&self, _dev: &DeviceInfo) -> (Optimization, usize)`.
2. Apply fn on `Kernel`: `pub fn my_optimization(&mut self, config: usize)`.
3. Register in `AVAILABLE_OPTIMIZATIONS`.

### Key gotchas

- Pipeline order matters: loop-splitting/coarsening run only during autotune, AFTER `run_always_on_optimizations`. Always-on IR has flat loops.
- After unfolding, loop indices flow through `Mad`→Cast→Binary chains — `check_loop` must trace the chain to the loop var.
- Pattern matchers must handle interleaved op ordering (accumulate computation can sit between `load(acc)` and `Add`).
- **Buffer allocation in autotune**: allocate buffers ONCE and reuse across all variant launches (`compile_and_launch`), then deallocate. Never alloc inside the variant loop.
- Hashing kernels avoids re-launching duplicates.

## Debugging

### ZYX_DEBUG (bitmask, `ENV_VARS.md`)

| Value | Output |
|-------|--------|
| 1 | hardware devices + kernel launches |
| 2 | egraph print (after realize) |
| 4 | kernels created by scheduler — IR BEFORE linearization (tensor graph, no loops/indices/loads/stores) |
| 8 | kernel IR AFTER linearization + optimization (loops, indices, loads, stores) |
| 16 | generated assembly/code |
| 32 | kernel launch + memory movement |
| 64 | memory alloc/dealloc |
| 128 | kernel compilation |
| 256 | autotune exploration |

Combine by summing (`24` = ir + asm). The two IR views show different stages: `ZYX_DEBUG=4` prints the kernel as it comes out of the scheduler (a DAG of buffer ops, scalar consts, expands — no loops yet), `ZYX_DEBUG=8` prints the same kernel after linearization and the optimization passes (flat loops over element indices, per-op loads/stores). `ZYX_DEBUG=8` prints IR during compilation, BEFORE any GPU kernel runs — use `timeout 10 bash -c 'ZYX_DEBUG=8 cargo run 2>&1' > /tmp/ir.txt` to capture it without a GPU hang.

In `cargo test`, library `eprintln!` output only shows with `-- --nocapture` (`AGENT=1 cargo test -- --nocapture`), otherwise it's hidden by the test harness.

### Inspecting kernel IR

Add `kernel.debug()` temporarily in code to print the IR (op IDs, args, loop scopes) at any pipeline point. `AGENT=1` strips ANSI colors; `AGENT=0` keeps them.

### Fixing a broken optimization pass

1. Identify the bad kernel: `ZYX_DEBUG=16` → look for O(N²) loops that should have folded.
2. Capture its IR (`ZYX_DEBUG=8` to `/tmp/ir.txt`). The last kernel printed is usually the one compiled before a hang.
3. Write a unit test that reconstructs that IR **exactly** (copy the debug output op by op — `Kernel::new()`, `k.define()`, `k.loop_()`, ...); assert the pass does NOT optimize it, confirming you've reproduced the failure.
4. Fix the pass's pattern matching (small, targeted), assert the test now DOES optimize, then run all tests.

### Crashes (segfault etc.)

Write a minimal reproducer test, then add `panic!("A")`, `panic!("B")`, ... along the suspect path (panics flush; `eprintln!` may not before a SIGSEGV). Narrow down from the last printed marker.

## Tape Design

`src/tape.rs` — the training-loop API around the graph.

- `Tape::realize` takes **persistent state only** (params + optimizer internals). Intermediates don't need realization.
- Eager tensors used inside a tape scope auto-promote to graph (`promote_to_graph`) — optimizer momentum buffers carry across steps without manual `tape.add()`.
- Training step: `Tape::new(&net)` → forward/loss → `tape.gradient(&loss, &net)` → `optim.update(&mut net, grads)` → `tape.realize(net.iter().chain(optim.iter()))`. After drop, realized tensors become eager again.
- Fixed control flow: `tape.freeze(&outputs)` once, then `frozen.replay(&inputs)` per step.

### Two execution paths: eager vs graph

Every tensor op has **two** possible paths, chosen by whether the caller is inside a tape scope:

- **Eager path** (no tape): `Runtime::{pad_zeros, unary, binary, ...}` directly pushes the
  `Op` (a `MoveOp`, or a compute op like `Unary`/`Binary`/`Reduce`) into the tensor's
  current kernel (`eager_ids`, e.g. `runtime.rs:1221` for pad). No graph nodes are
  created. The kernel is built with the custom-kernel API (`Kernel::new`/`define`/
  `pad`/`add`/... in `kernel/custom.rs`).
- **Graph path** (inside `Tape`): the same op pushes a `Node` into the egraph
  (`push_node`, e.g. `Node::PadZeros` at `graph/mod.rs:1238`). Only at `compile_graph`
  does the kernelizer (`kernelizer.rs`) turn those nodes into `Op`s.

This dual path applies to **every op** — unary, binary, reduce, pad, cast, etc. all either push an `Op` into an eager kernel (`runtime.rs`) or push a
`Node` into the egraph (`graph/mod.rs`) depending purely on whether the caller is in a
tape scope. When reasoning about kernel identity / recompilation (e.g. the kv-cache
`narrow(0, start, len)` case), consider which path the consumer actually runs, and
note that changing an `Op` means changing it in **both** places unless the two share a
construction point.

## Interaction Rules

- Every user message: if it contains a `?`, **answer the question and stop** — do not edit/write files. Otherwise proceed.
- When in doubt, ask. Don't guess specs/values — the user has them. Ask before hunting through source.
- **Follow the literal ask exactly — quantity included.** "A test" means exactly ONE test; "add a test for X" means just that test, not a family of tests. Deliver what was asked and stop. Extras (more tests, renames, refactors, extra fixes) are unrequested work.
- **Never start implementing anything beyond the literal ask without asking first.** If a requested change turns out to require fixing/modifying other parts of the code (e.g. a test exposes a library bug), STOP and ask the user how to proceed before writing any fix. Do not debug-and-fix your way down a rabbit hole unprompted. A single simple question ("want me to fix that too?") beats an hour of unrequested surgery.
- A test's pass/fail status is not the deliverable unless the user says so. Adding a test that currently fails (because it documents a bug) is a valid outcome — do not "fix" the code underneath it unprompted.
- **Never leave temporary debug code behind** (eprintln!/println! debug blocks, commented-out scaffolding). If you add debug output while investigating, remove it before finishing. When reverting, revert completely — no stray debug prints, no leftover comments.
- **Never commit unless explicitly asked.** When asked, `git add` + `git commit` a concise message matching repo style and produce zero extra commentary.
- Never use `git stash` or `git checkout --`; use `git restore`. Never discard or hide changes.
- No silent fixes. No commentary after doing something ("Done.", "X lines changed."). Don't dump git diffs verbatim.
- Test failures are test failures. You don't need to fix them unless I tell you to fix them.
- Edit precisely, don't cascade: change only what was referenced; propose (don't do) anything else.
