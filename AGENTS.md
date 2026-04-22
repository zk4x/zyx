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
cd zyx && cargo test
cd zyx && cargo test relu_1          # single test
cd zyx && cargo test --test 1_unary  # test file
cd zyx && cargo test -- --nocapture  # with output
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

- Use `kernel.debug_colorless()` instead of `kernel.debug()` for readable output without ANSI color codes
- Set `ZYX_DEBUG` environment variable to enable debug output (see Debug Options table)

### IR Debugging

When debugging kernel transformations (especially in autotune passes), always use `kernel.debug_colorless()` to inspect the IR:

```rust
// Add temporarily in your code to see the kernel state
kernel.debug_colorless();
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

**Always use `debug_colorless()` (not `debug()`)** - the latter includes ANSI color codes that make logs hard to read.

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

- Each backend (CUDA, OpenCL, etc.) in a single file under `runtime/backend/`
- Load backends at runtime via `.so` files (no compile-time linking)
- FFI limited to one file per backend

## API Design

- Default API: shortest to write
- Add explicit options later for performance users
- High-level first, then low-level detail

## Autotune System

The autotune system in `zyx/src/kernel/autotune.rs` searches for optimal kernel configurations.

### How Autotune Works

The autotune system is simple:
1. Start with initial kernel and run always-on optimizations
2. Apply ONE optimization variant and run always-on optimizations
3. Hash the kernel and check if visited (duplicate detection)
4. If not visited, launch kernel and record timing
5. Repeat by combining with existing optimization sequences
6. At the end, launch ONE final kernel configuration

The key insight: **only one kernel is launched at the end**. The exploration phase just builds up optimization sequences without actually running them.

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

### Known Optimization Issues

| Optimization | Status | Notes |
|--------------|--------|-------|
| `opt_fuse_mad` | ✅ Working | Baseline |
| `opt_unfuse_mad` | ✅ Working | Works with fuse_mad |
| `opt_tiled_reduce` | ✅ Working | Skip when local index exists or multiple loops |
| `opt_split_global_to_local` | ✅ Working | Must run before tiled_reduce creates local index |
| `opt_split_loop` | ⚠️ Flaky | Fails in real autotune exploration, works in apply_selected |
| `opt_reassociate_commutative` | ❌ Disabled | Buggy - breaks with licm |
| `opt_upcast` | ❌ Disabled | Fails matmul_disk |
| `opt_register_tiling` | ❌ Disabled | Fails gather |
| `opt_unfuse_mad` (standalone) | ❌ Disabled | Fails reduce when alone |
| `opt_unroll` | ❌ Disabled | Timeout/slow |
| `opt_unroll_constant_loops` | ❌ Disabled | Timeout/slow |
| `opt_licm` | ❌ Disabled | Conflicts with reassociate |

### Current Working Set

```rust
const AVAILABLE_OPTIMIZATIONS: [fn(&Kernel) -> (Optimization, usize); 5] = [
    Kernel::opt_split_global_to_local,
    Kernel::opt_fuse_mad,
    Kernel::opt_unfuse_mad,
    Kernel::opt_split_loop,
    Kernel::opt_tiled_reduce,
];
```

All of these work together and pass all tests.

### Debugging Tips

- The exploration can apply the same optimization multiple times to the same kernel
- Use `kernel.debug_colorless()` to inspect IR state

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

## What to Avoid

- **Never commit changes unless the user explicitly asks for it** - Always ask before committing
- **Ask for help when unsure, uncertain, or struggling** - Don't spend more than 15-30 minutes stuck before asking
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
