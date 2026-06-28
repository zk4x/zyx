# Agent Guidelines for zyx

Quick reference for coding agents working in the zyx repository.

## Essential Commands

```bash
# Build
cargo build -p zyx
cargo build -p zyx --release

# Lint (do NOT run - shows many unrelated issues)
# cargo clippy -p zyx --all-features -- -D warnings

# Format
cargo fmt

# Test (always run from zyx/zyx subdirectory!)
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

**Note**: This is a workspace with multiple crates (zyx, zyx-nn, zyx-optim, etc.). **Always run commands from the `zyx/zyx` subdirectory** (not the workspace root).

## Project Structure

```
/home/x/Dev/rust/zyx/
├── zyx/           # Core tensor library
├── zyx-derive/    # Procedural macros
├── zyx-nn/        # Neural network modules
├── zyx-optim/     # Optimization algorithms
├── zyx-onnx/      # ONNX support
├── zyx-fuzzy/     # Fuzzy logic
└── zyx-examples/  # Example code
```

## The Graph

Zyx is ALL about the graph. The graph is the core.

- **Lazy**: Puts ops into graph, no calculations until `Tensor::realize`
- **Dynamic**: Graph dynamically grows and shrinks at runtime
- **One graph for everything**: Autograd uses the same graph
- Other libraries use 2 graphs (one for laziness, one for autograd), zyx uses ONE
- **Super lean**: Only 16 bytes per tensor. 10k virtual tensors = ~160kB + shape metadata
- **Few ops**: Graph has only ~10 ops (Const, Leaf, Expand, Permute, Reshape, Pad, Reduce, Cast, Unary, Binary)

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
- Set `ZYX_DEBUG` environment variable to enable debug output. See [`zyx/ENV_VARS.md`](./zyx/ENV_VARS.md) for all available options.

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
- Naming: `{number}_{category}.rs` (e.g., `1_unary.rs`)
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

The project denies these in `lib.rs`:
- `clippy::all` (all warnings)
- `clippy::pedantic`
- `clippy::cast_possible_truncation`
- `clippy::cast_lossless`
- `clippy::cast_precision_loss`
- `clippy::cast_sign_loss`
- `clippy::perf` (forbidden)
- `missing_docs`

Mark allowed exceptions with `#[allow(...)]`.

## Backend Architecture

- Each backend in a single file under `zyx/src/backend/`
- Most backends (C, CUDA, HIP, OpenCL) are **always compiled in** — controlled at **runtime** via config file, not cargo features
- Only WGPU and Tenstorrent require `--features wgpu` / `--features tenstorrent`

### Switching Backends for Testing

Backends are selected at runtime via `$HOME/.config/zyx/config.json` (JSON):

```json
{ "c": { "enabled": true }, "dummy": { "enabled": false } }
```

Key rules:
- **C backend**: off by default → enable with `"c": { "enabled": true }`
- **Dummy backend**: off by default → enable with `"dummy": { "enabled": true }` (fake device, no computation)
- **CUDA**: on by default → override with `"cuda": { "device_ids": [] }` to disable
- **OpenCL**: on by default → disable with `"opencl": { "platform_ids": [] }`
- **WGPU**: on by default (if compiled with `--features wgpu`) → disable with `"wgpu": { "enabled": false }`

Most backends try to initialize and silently skip if hardware/driver is unavailable.
If **all** backends fail, tests produce no output.

To test with the **C backend only** (no GPU needed):
```bash
# Create ~/.config/zyx/config.json:
echo '{"c": {"enabled": true}, "cuda": {"device_ids": []}, "opencl": {"platform_ids": []}, "hip": {"device_ids": []}}' > ~/.config/zyx/config.json
cargo test
```

To reset to defaults, delete the config file:
```bash
rm ~/.config/zyx/config.json
```

Full config reference: [`zyx/CONFIG.md`](./zyx/CONFIG.md)

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

### Debugging with apply_selected_optimizations

To debug optimization issues, use the `apply_selected_optimizations` function which launches a kernel:

```rust
// In autotune.rs, find this line and change to true:
if true {  // was: if false
    return self.apply_selected_optimizations(...)
}
```

Then customize the optimizations applied in that function:

```rust
kernel.fuse_mad();
kernel.unfuse_mad();

// Apply in specific order to test
let (split_opt, n_split) = kernel.opt_split_loop();
if n_split > 0 {
    split_opt.apply(&mut kernel, 0);
}
// ... more optimizations
kernel.run_always_on_optimizations();
```

### Available Optimizations

The autotune system uses 7 optimizations (defined in `zyx/src/kernel/autotune.rs`):

```rust
const AVAILABLE_OPTIMIZATIONS: [OptConfigFn; 7] = [
    Kernel::opt_reassociate_commutative,  // reassociate + group operations
    Kernel::opt_split_global_to_local,   // parallelize reduce loops
    Kernel::opt_upcast,                   // upcast for vectorization
    Kernel::opt_register_tiling,         // tile for registers
    Kernel::opt_tiled_reduce,             // tiled parallel reduction
    Kernel::opt_split_loop,              // split large loops
    Kernel::opt_licm,                     // loop-invariant code motion
];
```

### Optimization Correctness (CRITICAL)

Every optimization must produce correct IR that calculates the same result as the input. **No optimization is needed for tests to pass.** ALL tests must pass with ALL optimizations disabled, and ALL tests must pass no matter which sequence of optimizations (including empty) is applied. If any sequence breaks correctness, the optimization that produced invalid IR from valid code is BUGGY and must be fixed or disabled.

## Debugging Optimization Passes

### Workflow for Fixing a Kernel Optimization Pass

1. **Identify the problematic kernel** — Run with `ZYX_DEBUG=16` to see generated CUDA, look for O(N²) loops that should have been folded.

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

8. **Run ALL tests** — Optimization passes affect ALL kernels. Always run `cargo test -p zyx` after any change:
   ```bash
   cd zyx/zyx && cargo test
   ```
   A single failing integration test (e.g., `gather_test`) means the optimization is producing incorrect results.

### ZYX_DEBUG Values

| Level | Output | When |
|-------|--------|------|
| `1`   | Backend selection | Startup |
| `2`   | Graph operations | During realize |
| `4`   | Scheduler decisions | Kernel selection |
| `8`   | Kernel IR (before GPU) | Kernel compilation |
| `16`  | Generated CUDA C++ source | Kernel compilation |
| `32`  | Autotune exploration | During autotune |

### Key Techniques

- **IR before GPU**: `ZYX_DEBUG=8` prints IR during compilation, before any GPU kernel executes. Use this to inspect IR without GPU hangs.
- **Pipeline order matters**: `simplify_accumulating_loop` runs in `run_always_on_optimizations` at line 225, before `split_loops` and other autotune passes. Check the pipeline order in `autotune.rs` before assuming loop structure.
- **Nested loops appear after splitting**: The `split_loops` pass runs during autotuning, AFTER `run_always_on_optimizations`. The IR at `simplify_accumulating_loop` time has flat loops, not nested ones.
- **Interleaved op ordering**: The real kernel IR may have accumulated value computation interleaved BETWEEN `load(acc)` and `Add`, not before the load. Pattern matchers must account for this.
- **Mad chains**: After unfold, loop index references go through `Mad` instructions that simplify to `loop_id` via constant folding. `check_loop` must trace through Cast, Mad, and Binary chains to find the loop variable.
- **Unit test isolation**: Write unit tests that construct Kernel IR directly. This isolates the optimization pass from the rest of the pipeline and makes debugging fast.

### Debugging Tips

- The exploration can apply the same optimization multiple times to the same kernel
- Use `kernel.debug()` to inspect IR state

### Adding an Optimization

1. Define config function (how many variants):

```rust
pub fn my_opt_config(&self) -> u16 {
    4 // try 4 variants
}
```

2. Define optimization function:

```rust
pub fn my_optimization(&mut self, config: u16) {
    let tile_size = [16, 32, 64, 128][config as usize];
    // apply optimization...
}
```

3. Register in `available_opts` array:

```rust
let available_opts: [(fn(&Kernel) -> u16, fn(&mut Kernel, u16)); _] = [
    (Self::opt_no_config, Self::reassociate_commutative),
    (Self::my_opt_config, Self::my_optimization), // <-- add here
];
```

### Always-On Optimizations

The `run_always_on_optimizations` method applies optimizations that should always run before kernel compilation. These are defined in `zyx/src/kernel/autotune.rs`:

```rust
pub fn run_always_on_optimizations(&mut self) {
    self.constant_folding();
    self.move_constants_to_beginning();
    self.loop_invariant_code_motion();
    self.common_subexpression_elimination();
    self.fold_accs();
    self.delete_empty_loops();
    self.dead_code_elimination();
}
```

**Important**: Always run `dead_code_elimination` as the last step. This ensures backends never receive ops that are no longer used, which could cause compilation failures (e.g., missing entries in reference count maps).

### Key Patterns
- Return `1` from config function if no tunable parameters
- Cost model uses heuristic initially, then actual execution time
- Use kernel hashing to avoid duplicate exploration

### Optimization Correctness

Every optimization must produce correct IR that calculates the same result as the input. If one optimization breaks another, that's a bug in the optimization that produced invalid IR from valid code - not a problem with the ordering. When combining optimizations (e.g., upcast + tiled_reduce), each must work correctly on the other's output.

## Debugging Crashes

When investigating a crash (segfault, signal, etc.):

1. **Write a minimal reproducer** — a single test function that triggers the bug. Keep it small.
2. **Isolate the crash line** — add `panic!("A")`, `panic!("B")`, etc. at key points in the suspected code path. Run and see which is the last printed. Do NOT add `eprintln!` — output may not flush before SIGSEGV. Only panics are guaranteed to flush.
3. **Narrow down** — once you know what line crashes, look at what that line does and work forward to figure out what's wrong.

## Answering Questions

**When the user asks you a direct question, answer immediately and stop. Do NOT do anything else until you've answered. Do NOT use tools. Do NOT search. Do NOT explain. Do NOT run commands. Do NOT fix things. Just answer directly in plain text, then stop.**
**If you fail to follow this rule, the user will stop you from doing ANYTHING until you answer.**

**Never jump to fixing.** If the user asks a question about something you did wrong, answer the question first. Do not start editing files or undoing changes in the same message. Answer, then wait for instruction. This means zero tool calls — no Read, no Edit, no Write, no Bash — until you've answered in plain text and the user has told you what to do next.

**Edit precisely, don't cascade.** When the user gives feedback on a specific change, only modify exactly what they referenced. Do not revert, restructure, or delete unrelated code. If you think other changes are needed, ask first. Never make multiple reverts in a chain without being asked — each revert is a new change requiring permission.

## What to Avoid

- **Never commit changes unless the user explicitly asks for it** - Always ask before committing
- **When in doubt, ask me immediately** - Don't try to figure things out on your own if uncertain. Just ask.
- **Ask before hunting for specs/values** - If I might have a spec, a mapping, or any information that could save time, ask me first. I always have it, so don't dig through source code or run experiments to derive it.
- **Stay on task** — Don't run investigations the user didn't ask for. No git archaeology, no random test runs, no looking up history. Only do exactly what the user tells you.
- **Don't run the full test suite unless asked.** Running it to "make sure nothing broke" after a change is going off track. If the user wants it run, they'll say so.
- **Never blame pre-existing test failures.** If a test fails but you didn't touch the code it exercises, the failure is yours to investigate and fix. The phrase "pre-existing test failure" is FORBIDDEN.
- **Never touch `~/.config/zyx/config.json`** — never read, write, create, modify, or delete it. If a test needs a specific backend, ask the user to configure it. Do not even look at this file.
- Inheritance (use composition/enums)
- `Rc<RefCell<T>>` unless absolutely necessary
- Too many small files
- Complex lifetime annotations
- Abstractions without proven need

## Performance Tips

- Use arenas for high-performance allocation
- Use `Vec` over `Box<[]>` for flexibility
- Use `Mutex` over `RefCell` for potential multithreading
- Profile before optimizing
- **Build profile does NOT affect GPU kernel performance**: CUDA kernels are compiled by NVRTC at runtime, identically in debug and release builds. The Rust build profile only affects host-side code. Don't assume release mode will make kernels faster.
