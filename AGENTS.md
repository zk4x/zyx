# ⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️
# ⚠️  "pre-existing" IS FORBIDDEN. NEVER USE IT.  ⚠️
# ⚠️  "Builds clean" / similar                    ⚠️
# ⚠️  IS FORBIDDEN. BUILD SILENTLY.               ⚠️
# ⚠️  "git checkout" IS FORBIDDEN. NEVER USE.     ⚠️
# ⚠️  Use "git restore" instead.                   ⚠️
# ⚠️  VIOLATIONS = BROKEN. REPEATEDLY. FOREVER.   ⚠️
# ⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️

# Agent Guidelines for zyx

Quick reference for coding agents working in the zyx repository.

## Essential Commands

```bash
# Build (always from the zyx/ subdirectory)
cd zyx && cargo build
cd zyx && cargo build --release
# Tenstorrent backend
TT_METAL_ROOT=~/Dev/cpp/tt-metal cargo build --features tenstorrent

# Lint (do NOT run - shows many unrelated issues)
# cargo clippy --all-features -- -D warnings

# Format
cargo fmt

# Test (always run from the zyx/ subdirectory!)
cd zyx && AGENT=1 cargo test
cd zyx && AGENT=1 cargo test relu_1          # single test
cd zyx && AGENT=1 cargo test --test 1_unary  # test file
cd zyx && AGENT=1 cargo test -- --nocapture  # with output

# Doc tests
cd zyx && AGENT=1 cargo test --doc
```

## Python

- Always use `python3.12` (not system python)
- Install packages with: `python3.12 -m pip install <package>`
- Run scripts with: `python3.12 <script>.py`

**Note**: The repo root has no `Cargo.toml` — the crates (zyx, zyx-nn, zyx-optim, etc.) are independent packages, not a cargo workspace. **Always run commands for the core library from the `zyx/` subdirectory** (not the repo root).

## Project Structure

```
/home/x/Dev/rust/zyx/
├── zyx/           # Core tensor library
├── zyx-derive/    # Procedural macros
├── zyx-nn/        # Neural network modules
├── zyx-optim/     # Optimization algorithms
├── zyx-onnx/      # ONNX support
├── zyx-fuzzy/     # Fuzzy logic
├── zyx-book/      # Book documentation
├── zyx-bench/     # Benchmarks
├── zyx-examples/  # Examples
├── zyx-py/        # Python bindings
├── docs/          # Generated docs
├── generated/     # Generated tools (inspector, watcher)
```

Root files: `ENV_VARS.md` (debug env vars), `CONFIG.md` (backend config), `ADDING_BACKENDS.md`, `STYLE.md`.

## The Graph

Zyx is ALL about the graph. The graph is the core.

- **Lazy**: Puts ops into graph, no calculations until `Tensor::realize`
- **Dynamic**: Graph dynamically grows and shrinks at runtime
- **One graph for everything**: Autograd uses the same graph
- Other libraries use 2 graphs (one for laziness, one for autograd), zyx uses ONE
- **Super lean**: TensorId is `u32` (4 bytes). 10k tensor handles = ~40kB.
- **Few ops**: Graph has 12 high-level categories (Const, Leaf, Expand, Permute, Reshape, PadZeros, Reduce, Cast, Unary, Binary, ToDevice, Kernel), with UOp (13 variants), BOp (18 variants), and MoveOp (4 variants) for detailed operations.

## Core Principles

- **Simplicity first**: Debuggable/understandable > "clean"
- **Duplication > bad abstractions**: Duplicate until patterns emerge
- **Explicit > implicit**: Use explicit returns, make code obvious
- **No virtual tables**: Use enums instead of `dyn Trait`
- **Minimize Arc/Rc**: Only when truly necessary

## Code Style

### Naming
| Type | Convention | Example |
|------|------------|---------|
| Types | PascalCase | `Tensor`, `ZyxError` |
| Variables/functions | snake_case | `tensor`, `relu()` |
| Constants | SCREAMING_SNAKE_CASE | `MAX_DIM` |
| Modules | snake_case | `mod backend` |

### Imports Order
1. `crate::` modules
2. `super::` modules
3. External crates (`std`, `core`)
4. `pub use` exports

### Debugging

- Use `kernel.debug()` to inspect the kernel IR. `AGENT=1` strips ANSI colors, `AGENT=0` preserves them.
- Set `ZYX_DEBUG` environment variable to enable debug output. See [`ENV_VARS.md`](./ENV_VARS.md) for all available options.

### IR Debugging

When debugging kernel transformations (especially in autotune passes), use `kernel.debug()` to inspect the IR:

```rust
// Add temporarily in your code to see the kernel state
kernel.debug();
```

This prints the kernel operations in a human-readable format showing:
- Operation IDs (e.g., `OpId(3)`)
- Each operation with its arguments and type
- Loop scopes and indices

Example output:
```
r18: i32 = def global, len=4
r31: i32 = def global, len=4
r43: i32 = def mut global, len=16
r44: u32 = gidx0    // 0..=0
r3: u32 = gidx1    // 0..=3
r1: u32 = gidx2    // 0..=3
r19: i32 = r18[r1]    // 0..=3 load
```

### File Organization
- Keep ~1000 LOC per module
- Add new files only when necessary
- Include license header:

```rust
// Copyright (C) 2025 zk4x
// SPDX-License-Identifier: LGPL-3.0-only
```

## Documentation & Errors

- All public items need docs
- Use `#[track_caller]` on error constructors
- Include file:line:column in error messages
- Implement `From` traits for conversions

```rust
#[track_caller]
pub fn shape_error(e: Box<str>) -> Self {
    let location = std::panic::Location::caller();
    Self::ShapeError(e, location.into())
}
```

## Testing

- Tests in `zyx/tests/`
- Naming: `{number}_{category}.rs` (e.g., `1_unary.rs`), with a few exceptions (`eager_load.rs`, `mnist.rs`)
- Return `Result<(), ZyxError>`
- Use `assert!` and `is_equal()` for floats

```rust
#[test]
fn relu_1() -> Result<(), ZyxError> {
    let x = Tensor::from(data);
    let z = x.relu();
    assert_eq!(z, expected);
    Ok(())
}
```

### Testing with a Specific Backend

**NEVER touch `~/.config/zyx/config.json`** — if a test needs a specific backend, ask the user to configure it.

zyx runs tests using whatever backends are available. The user controls which backends run via the config file. Ask them which backend to use.

**Do NOT add cargo feature flags** for most backends (C, CUDA, HIP, OpenCL are always compiled). Only `--features wgpu` is needed for WGPU. See "Backend Architecture" section below for config details.

## Clippy Strictness

The project configures lints in `lib.rs`:
- `#![deny(clippy::all)]` and `#![deny(clippy::pedantic)]` (all warnings)
- `#![forbid(clippy::perf)]`
- `#![deny(clippy::style)]`, `#![deny(clippy::nursery)]`
- `#![deny(clippy::fn_to_numeric_cast_any)]`, `#![deny(clippy::as_ptr_cast_mut)]`, `#![deny(clippy::missing_const_for_fn)]`, `#![deny(clippy::separated_literal_suffix)]`
- `#![warn(missing_docs)]` (a warning, not denied)

The cast lints are **allowed**, not denied: `clippy::cast_possible_truncation`, `clippy::cast_lossless`, `clippy::cast_precision_loss`, `clippy::cast_sign_loss`, `clippy::cast_ptr_alignment`, `clippy::cast_possible_wrap`.

Mark allowed exceptions with `#[allow(...)]`.

## Backend Architecture

- Each backend in a single file under `zyx/src/backend/` (`c`, `cuda`, `hip`, `opencl`, `vulkan`, `disk`, `host`, `dummy`, `tenstorrent`, `wgpu`)
- Most backends (C, CUDA, HIP, OpenCL, Vulkan, disk, host, dummy) are **always compiled in** — controlled at **runtime** via config file, not cargo features
- Only WGPU and Tenstorrent require `--features wgpu` / `--features tenstorrent`

### Switching Backends for Testing

Backends are selected at runtime via `$HOME/.config/zyx/config.json` (JSON):

```json
{ "c": { "enabled": true }, "dummy": { "enabled": false } }
```

Key rules:
- **C backend**: on by default → disable with `"c": { "enabled": false }`
- **Dummy backend**: off by default → enable with `"dummy": { "enabled": true }` (fake device, no computation)
- **CUDA**: on by default → override with `"cuda": { "device_ids": [] }` to disable
- **HIP**: always tries to initialize (ignores config); only skipped if `libamdhip64.so` is missing
- **OpenCL**: on by default → disable with `"opencl": { "platform_ids": [] }`
- **WGPU**: on by default (if compiled with `--features wgpu`) → disable with `"wgpu": { "enabled": false }`

Most backends try to initialize and silently skip if hardware/driver is unavailable.
If **all** backends fail, tests produce no output.

To test with the **C backend only** (no GPU needed):
```bash
# Create ~/.config/zyx/config.json:
echo '{"c": {"enabled": true}, "cuda": {"device_ids": []}, "opencl": {"platform_ids": []}}' > ~/.config/zyx/config.json
cd zyx && cargo test
```

To reset to defaults, delete the config file:
```bash
rm ~/.config/zyx/config.json
```

Full config reference: [`CONFIG.md`](./CONFIG.md)

## API Design

- Default API: shortest to write
- Add explicit options later for performance users
- High-level first, then low-level detail

## Autotune System

The autotune system in `zyx/src/kernel/autotune.rs` searches for optimal kernel configurations.

### How Autotune Works

The autotune system explores optimization sequences by:
1. Start with initial kernel and run always-on optimizations
2. Apply ONE optimization variant and run always-on optimizations
3. Hash the kernel and check if visited (duplicate detection)
4. If not visited, launch kernel and record timing
5. Repeat by combining with existing optimization sequences
6. Select the best configuration based on actual timing

### Available Optimizations

The autotune system uses 9 optimizations (defined in `zyx/src/kernel/autotune.rs`):

```rust
const AVAILABLE_OPTIMIZATIONS: [OptConfigFn; 9] = [
    |k, _| Kernel::opt_reassociate_commutative(k),
    Kernel::opt_split_global_to_local,
    |k, _| Kernel::opt_thread_coarse(k),
    |k, _| Kernel::opt_register_blocking(k),
    Kernel::opt_local_reduce,
    |k, _| Kernel::opt_split_loop(k),
    |k, _| Kernel::opt_pad_index(k),
    Kernel::opt_vectorize,
    |k, _| Kernel::opt_merge_nested_loops(k),
];
```

### Optimization Correctness (CRITICAL)

Every optimization must produce correct IR that calculates the same result as the input. **No optimization is needed for tests to pass.** ALL tests must pass with ALL optimizations disabled, and ALL tests must pass no matter which sequence of optimizations (including empty) is applied. If any sequence breaks correctness, the optimization that produced invalid IR from valid code is BUGGY and must be fixed or disabled.

## Debugging Optimization Passes

### Workflow for Fixing a Kernel Optimization Pass

1. **Identify the problematic kernel** — Run with `ZYX_DEBUG=16` to see generated assembly, look for O(N²) loops that should have been folded.

2. **Capture the IR** — Run with `ZYX_DEBUG=8` (IR dump, no GPU execution) to see the kernel IR. Use `timeout 10` to capture output quickly before a GPU hang:
   ```bash
   timeout 10 bash -c 'ZYX_DEBUG=8 cargo run 2>&1' > /tmp/ir.txt
   ```
   The IR is printed during compilation, before GPU execution begins.

3. **Find the target kernel's IR** — Search for the kernel with the problematic pattern. The last kernel printed is usually the one being compiled when the hang occurs.

4. **Write a unit test that replicates the IR exactly** — Use the `Kernel` builder API (`Kernel::new()`, `k.define()`, `k.const_val()`, `k.loop_()`, etc.) to construct the kernel IR op by op. Do NOT simplify or guess the pattern — copy the actual IR from the debug output.

5. **Verify the test reproduces the failure** — Run `simplify_accumulating_loop()` (or whatever pass you're debugging) and assert that it does NOT optimize the pattern. This confirms the test matches the real failure.

6. **Fix the optimization pass** — Modify the pattern matching to handle the real IR structure. Small, targeted changes only.

7. **Verify the fix with the unit test** — After the fix, the test should assert the pattern IS optimized.

8. **Run ALL tests** — Optimization passes affect ALL kernels. Always run `cargo test` after any change:
   ```bash
   cd zyx && cargo test
   ```
   A single failing integration test (e.g., `gather_test`) means the optimization is producing incorrect results.

### ZYX_DEBUG Values

| Level | Output | When |
|-------|--------|------|
| `1`   | Hardware devices and configuration + kernel launches (device and program id) | Startup / During realize |
| `2`   | Egraph print | After realize (graph/extraction) |
| `4`   | Kernels created by scheduler | Kernel selection |
| `8`   | Kernel IR | Kernel compilation |
| `16`  | Generated assembly/code (OpenCL, WGSL, etc.) | Kernel compilation |
| `32`  | Kernel launch and memory movement | Kernel launch |
| `64`  | Memory allocation/deallocation | During realize |
| `128` | Kernel compilation | Kernel compilation |
| `256` | Autotune exploration | During autotune |

Combine flags by summing values (e.g., `ZYX_DEBUG=24` = ir + asm). See [`ENV_VARS.md`](./ENV_VARS.md).

### Key Techniques

- **IR before GPU**: `ZYX_DEBUG=8` prints IR during compilation, before any GPU kernel executes. Use this to inspect IR without GPU hangs.
- **Pipeline order matters**: `simplify_accumulating_loop` runs in `run_always_on_optimizations` at line 344, before `split_loop` and other autotune passes. Check the pipeline order in `autotune.rs` before assuming loop structure.
- **Nested loops appear after splitting**: The `opt_split_loop` pass runs during autotuning, AFTER `run_always_on_optimizations`. The IR at `simplify_accumulating_loop` time has flat loops, not nested ones.
- **Interleaved op ordering**: The real kernel IR may have accumulated value computation interleaved BETWEEN `load(acc)` and `Add`, not before the load. Pattern matchers must account for this.
- **Mad chains**: After unfold, loop index references go through `Mad` instructions that simplify to `loop_id` via constant folding. `check_loop` must trace through Cast, Mad, and Binary chains to find the loop variable.
- **Unit test isolation**: Write unit tests that construct Kernel IR directly. This isolates the optimization pass from the rest of the pipeline and makes debugging fast.

### Debugging Tips

- The exploration can apply the same optimization multiple times to the same kernel
- Use `kernel.debug()` to inspect IR state

### Adding an Optimization

1. Define config function (how many variants). It returns `(Optimization, usize)` — the `Optimization` to apply and the number of config variants:

```rust
pub fn my_opt_config(&self, _dev_info: &DeviceInfo) -> (Optimization, usize) {
    (Optimization::MyOpt { factors }, 4) // try 4 variants
}
```

2. Define the apply function on `Kernel`:

```rust
pub fn my_optimization(&mut self, config: usize) {
    let tile_size = [16, 32, 64, 128][config];
    // apply optimization...
}
```

3. Register in `AVAILABLE_OPTIMIZATIONS` array:

```rust
const AVAILABLE_OPTIMIZATIONS: [OptConfigFn; 10] = [
    |k, _| Kernel::opt_reassociate_commutative(k),
    Kernel::opt_split_global_to_local,
    |k, _| Kernel::opt_thread_coarse(k),
    |k, _| Kernel::opt_register_blocking(k),
    Kernel::opt_local_reduce,
    |k, _| Kernel::opt_split_loop(k),
    |k, _| Kernel::opt_pad_index(k),
    Kernel::opt_vectorize,
    |k, _| Kernel::opt_merge_nested_loops(k),
    Self::my_opt_config, // <-- add here
];
```

### Always-On Optimizations

The `run_always_on_optimizations` method applies optimizations that should always run before kernel compilation. These are defined in `zyx/src/kernel/autotune.rs`:

```rust
pub fn run_always_on_optimizations(&mut self) {
    self.unroll_len1_loops();
    self.constant_folding();
    self.move_constants_to_beginning();
    self.loop_invariant_code_motion();
    self.fold_accs();
    self.delete_zero_len_indices();
    self.delete_zero_len_loops();
    self.unfold_pows();
    self.algebraic_simplification();
    self.simplify_accumulating_loop();
    self.swap_commutative();
    self.common_subexpression_elimination();
    self.instruction_schedule();
    self.dead_code_elimination();
}
```

**Important**: Always run `dead_code_elimination` as the last step. This ensures backends never receive ops that are no longer used, which could cause compilation failures (e.g., missing entries in reference count maps).

### Key Patterns
- Return a config count of `1` if no tunable parameters
- Cost model uses heuristic initially, then actual execution time
- Use kernel hashing to avoid duplicate exploration

### Buffer Allocation in Autotune

The autotune allocates its own temp buffers — callers never pass `PoolBufferId`s.

CRITICAL: Allocate ONCE, reuse across all variant launches, then deallocate.

- `launch_with_timings` is self-contained: allocates, compiles, launches, times, deallocates. For single-use callers like `apply_selected_optimizations`.
- `autotune_` iterates many variants: allocate once up front, use a private `compile_and_launch` helper (no alloc/dealloc) in the loop, then deallocate after.

```rust
// WRONG — allocates every iteration:
for variant in variants {
    kernel.launch_with_timings(device, memory_pool, ...); // allocs inside
}

// RIGHT — allocate once, reuse:
let args = allocate_global_buffers(&kernel, memory_pool)?;
for variant in variants {
    kernel.compile_and_launch(&args, device, memory_pool, ...);
}
deallocate_global_buffers(&args, memory_pool);
```

### Optimization Correctness

Every optimization must produce correct IR that calculates the same result as the input. If one optimization breaks another, that's a bug in the optimization that produced invalid IR from valid code - not a problem with the ordering. When combining optimizations (e.g., upcast + tiled_reduce), each must work correctly on the other's output.

## Debugging Crashes

When investigating a crash (segfault, signal, etc.):

1. **Write a minimal reproducer** — a single test function that triggers the bug. Keep it small.
2. **Isolate the crash line** — add `panic!("A")`, `panic!("B")`, etc. at key points in the suspected code path. Run and see which is the last printed. Do NOT add `eprintln!` — output may not flush before SIGSEGV. Only panics are guaranteed to flush.
3. **Narrow down** — once you know what line crashes, look at what that line does and work forward to figure out what's wrong.

## Answering Questions

**Every user message: before writing ANY tool call, check if the message contains a `?`.**

**If YES → answer the question concisely and stop. Do NOT edit/write files. You may still use any tools (bash, grep, read, etc.) to gather information needed to answer well. Just don't modify files.**

**If NO → proceed normally.**

There are no exceptions. Rhetorical questions are questions. "What do you mean" is a question. "Did you" is a question.

**Edit precisely, don't cascade.** When the user gives feedback on a specific change, only modify exactly what they referenced. Do not revert, restructure, or delete unrelated code. If you think other changes are needed, ask first. Never make multiple reverts in a chain without being asked — each revert is a new change requiring permission.

## Questions

**I love when you ask questions.** When in doubt, ask. Always ask. Never assume.

Ask follow-up questions. Ask clarifying questions. Ask before doing anything you're unsure about. Ask before making assumptions. Ask about specs, values, test results, anything.

The user has all the answers. Just ask.

## What to Avoid

- **Never commit unless the user explicitly asks** — but when they say "commit", just do it. Derive a concise commit message from the diff matching the repo style. Do NOT ask for a message.
- **"commit" means commit, not "proceed"** — If the user says "commit", do exactly: `git add`, `git commit`, then produce zero additional text. No status updates. No summaries. No "what next". No commentary. Zero output after the commit output.
- **When in doubt, ask me immediately** - Don't try to figure things out on your own if uncertain. Just ask.
- **Ask before hunting for specs/values** - If I might have a spec, a mapping, or any information that could save time, ask me first. I always have it, so don't dig through source code or run experiments to derive it.
- **Do as asked, nothing more.** If the user says "rerun", rerun the test. Don't edit files, don't remove debug artifacts, don't "fix" anything unless explicitly told to. Running and editing are different verbs.
- **Ask follow-up questions.** The user loves them. Always ask if uncertain.
- **Never use `git stash` or `git checkout --`. Never discard or hide changes.**
- **Never run tests to check whether something worked before your changes** (e.g. to check if a failure existed before your edits or to "make sure nothing broke" pre-change). Tests run after a change are for validating your change itself.
**THE PHRASE "PRE-EXISTING" IS FORBIDDEN. NEVER USE IT. EVER. IN ANY CONTEXT. IF YOU USE IT, YOU WILL BE CORRECTED. REPEATEDLY. FOREVER.**
- **Never say "Builds clean" or anything similar.** If asked to build, build silently. No commentary on the result.
- **Never dump git stats/diffs verbatim in conversation.** Show the user what matters, not the raw output.
- **Never add commentary about what you just did.** Do the thing, then stop. No "Done.", "Applied.", "X lines changed.", etc.
- **Never blame test failures on anything other than yourself.** If a test fails, it's your fault — find and fix it.
- **NEVER fix anything silently.** No silent fixes, no silent cleanups, no silent anything. Every fix must be explicitly requested or explicitly discussed first.
- **Never touch `~/.config/zyx/config.json`** — never read, write, create, modify, or delete it. If a test needs a specific backend, ask the user to configure it. Do not even look at this file.
- Inheritance (use composition/enums)
- `Rc<RefCell<T>>` unless absolutely necessary
- Too many small files
- Complex lifetime annotations
- Abstractions without proven need

## Tape Design

- **`Tape::realize` should take persistent state only**: model parameters + optimizer internal buffers (momentum, etc.). These are the tensors whose values must carry across iterations.
- **Intermediates (activations, gradients) don't need explicit realization** — they're used within the step and can be dropped.
- **`realize_all`** is not needed — it would wastefully realize intermediates. Currently commented out.
- **Auto-promotion of eager tensors to graph** is necessary and correct. When an eager tensor (e.g., optimizer momentum buffer from a previous step) is used in an operation with a graph tensor inside a tape scope, it gets auto-promoted to graph via `promote_to_graph`. This avoids requiring manual `tape.add()` calls for optimizer internals.
- However, auto-promotion creates a dynamic where the tape has more leaf tensors than were passed to `Tape::new`. This is fine — `realize` handles all leaves as graph inputs.
- **Training loop pattern**:
  1. `let tape = Tape::new(&net)?` — promotes model params
  2. Forward + loss builds graph
  3. `tape.gradient(&loss, &net)` — computes grads as graph tensors
  4. `optim.update(&mut net, grads)` — replaces params with new graph tensors, may auto-promote optim buffers
  5. `tape.realize(net.iter().chain(optim.iter()))?` — realizes persistent state
  6. Tape drops, realized tensors become eager for next iteration
- **`Tape::freeze` / `FrozenTape::replay`** — for fixed control flow, call `let frozen = tape.freeze(&outputs)?` to compile the plan once, then `frozen.replay(&inputs)?` each step with new inputs. Lower overhead than building a fresh tape per step.

## TT Metalium API Reference

- Compute API docs: <https://docs.tenstorrent.com/tt-metal/latest/tt-metalium/tt_metal/apis/index.html>
- Key compute APIs: `cb_wait_front`, `cb_reserve_back`, `cb_push_back`, `cb_pop_front`, `ld_tile`, `st_tile`, `mul_tiles`, `add_tiles`, `typecast_tile`, `binary_op_tile`, `unary_op_tile`, `reduce_tile`

## Performance Tips

- Use arenas for high-performance allocation
- Use `Vec` over `Box<[]>` for flexibility
- Use `Mutex` over `RefCell` for potential multithreading
- Profile before optimizing
- **Build profile does NOT affect GPU kernel performance**: CUDA kernels are compiled by NVRTC at runtime, identically in debug and release builds. The Rust build profile only affects host-side code. Don't assume release mode will make kernels faster.
