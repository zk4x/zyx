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

# Test
cargo test -p zyx
cargo test -p zyx relu_1          # single test
cargo test -p zyx --test 1_unary  # test file
cargo test -p zyx -- --nocapture  # with output
```

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

### File Organization
- Keep ~1000 LOC per module
- Add new files only when necessary
- Include license header:

```rust
// Copyright (C) 2025 zk4x
// SPDX-License-Identifier: AGPL-3.0-only
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

### Key Patterns
- Return `1` from config function if no tunable parameters
- Cost model uses heuristic initially, then actual execution time
- Use kernel hashing to avoid duplicate exploration

## What to Avoid

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
