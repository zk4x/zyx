# AGENTS.md - Agent Guidelines for zyx

This document provides guidelines for agentic coding agents operating in the zyx repository.

## Project Structure

```
/home/x/Dev/rust/zyx/        # Main workspace root
├── zyx/                     # Core tensor library (main crate)
├── zyx-derive/              # Procedural macros
├── zyx-nn/                  # Neural network modules
├── zyx-optim/               # Optimization algorithms
├── zyx-onnx/                # ONNX support
├── zyx-fuzzy/               # Fuzzy logic
├── zyx-examples/            # Example code
├── STYLE.md                 # Detailed style guidelines
└── CONTRIBUTING.md         # Contribution guidelines
```

## Build, Lint, and Test Commands

### Building

```bash
# Build the core library
cargo build -p zyx

# Build with specific features
cargo build -p zyx --features wgpu,py

# Build release
cargo build -p zyx --release
```

### Linting and Formatting

```bash
# Run clippy (strict rules are enforced - see lib.rs)
cargo clippy -p zyx --all-features -- -D warnings

# Format code
cargo fmt -- --check  # Check without modifying
cargo fmt             # Auto-format
```

### Testing

```bash
# Run all tests
cargo test -p zyx

# Run a single test by name
cargo test -p zyx relu_1

# Run tests in a specific file
cargo test -p zyx --test 1_unary

# Run tests matching a pattern
cargo test -p zyx -- unary

# Run with output
cargo test -p zyx -- --nocapture

# Run doc tests
cargo test -p zyx --doc
```

Note: Tests require a backend (CUDA, OpenCL, or CPU) to be available.

## Code Style Guidelines

### General Principles

- **Simplicity is paramount**: Code should be debuggable/understandable over "clean" code
- **Duplication over bad abstractions**: Duplicate code first, abstract later when patterns emerge
- **Explicit over implicit**: Use explicit return keywords, make code obvious
- **Avoid virtual tables**: Use enums instead of `dyn Trait`
- **Minimize Arc/Rc usage**: Only when truly necessary
- **Single mutable state**: Keep state in few places, ideally one struct

### Naming Conventions

- **Types**: PascalCase (e.g., `Tensor`, `ZyxError`, `Runtime`)
- **Variables/functions**: snake_case (e.g., `tensor`, `relu()`, `shape_error()`)
- **Constants**: SCREAMING_SNAKE_CASE
- **Modules**: snake_case (e.g., `mod backend`)

### Imports

Order imports as:
1. `crate::` modules
2. `super::` modules
3. External crates (std, core)
4. `pub use` exports

```rust
use crate::runtime::Runtime;

mod backend;
mod error;

use std::fmt::Display;

pub use tensor::Tensor;
pub use dtype::DType;
```

### Documentation

- All public items must have documentation
- Use `#[track_caller]` on error constructors for better error messages
- Comment complex algorithms (~20% comments, 50% for complex code)
- Document obvious things - what's obvious to writer isn't obvious to reader

### Error Handling

- Use `ZyxError` enum with specific variants
- Use `#[track_caller]` on error constructor methods
- Include file:line:column in error messages
- Implement `From` traits for conversions

```rust
#[track_caller]
pub fn shape_error(e: Box<str>) -> Self {
    let location = std::panic::Location::caller();
    // ... include location in error
}

impl From<std::io::Error> for ZyxError {
    #[track_caller]
    fn from(value: std::io::Error) -> Self {
        Self::IOError(value)
    }
}
```

### Assertions

- Use debug asserts for invariants
- Check limits, sizes, maximum values

### Clippy Strictness

The project enforces strict clippy rules in `lib.rs`:
- `deny(clippy::all)` - all warnings
- `deny(clippy::pedantic)` - pedantic warnings
- `deny(clippy::cast_possible_truncation)`
- `deny(clippy::cast_lossless)`
- `deny(clippy::cast_precision_loss)`
- `deny(clippy::cast_sign_loss)`
- `forbid(clippy::perf)` - performance anti-patterns
- `deny(missing_docs)` - documentation required

Allowed exceptions are explicitly marked with `#[allow(...)]`.

### File Organization

- Keep related code together in few large files (~1000 LOC per module)
- Add new files only when truly necessary
- Each file should have a header comment with license:

```rust
// Copyright (C) 2025 zk4x
// SPDX-License-Identifier: AGPL-3.0-only
```

### API Design

- Default API should be shortest to write
- Add explicit options later for performance users
- Write high-level API first, then add detailed low-level API

### Backend Interface

- Each backend (CUDA, OpenCL, etc.) in a single file under `runtime/backend/`
- No compile-time linking - load backends at runtime via `.so` files
- FFI limited to single file per backend for easy refactoring

### Functions

- Functions should only exist if called in 2+ places
- Long functions are okay - use code blocks to organize
- Duplicate code freely until patterns emerge

### Testing

- Integration tests in `zyx/tests/`
- Test naming: `{number}_{category}.rs` (e.g., `1_unary.rs`, `2_binary.rs`)
- Return `Result<(), ZyxError>` from tests
- Use `assert!` and `is_equal()` for float comparisons

```rust
#[test]
fn relu_1() -> Result<(), ZyxError> {
    let x = Tensor::from(data);
    let z = x.relu();
    assert_eq!(z, expected);
    Ok(())
}
```

### What to Avoid

- Inheritance (use composition/enums instead)
- `Rc<RefCell<T>>` unless absolutely necessary
- Too many small files (creates navigation overhead)
- Complex lifetime annotations (prefer global state or Rc/Arc)
- Abstractions without proven need

### Performance Tips

- Use arenas for high-performance allocation
- Use Vec over Box<[]> for flexibility
- Use Mutex over RefCell for potential multithreading
- Profile before optimizing

### Writing Autotune Optimization Functions

The autotune system in `zyx/src/kernel/autotune.rs` performs beam search over optimization sequences to find the fastest kernel configuration. To add a new optimization:

#### 1. Define the optimization function

```rust
pub fn my_optimization(&mut self, config: u16) {
    // config: varies from 0 to N-1 to try different variants
    // Use config to control behavior (e.g., which pass to apply)
}
```

#### 2. Define the config count function

```rust
pub fn my_opt_config(&self) -> u16 {
    N // Number of configurations to try (1 if no variants)
}
```

#### 3. Register in the autotune array

Add to the `available_opts` array in `autotune()`:

```rust
let available_opts: [(fn(&Kernel) -> u16, fn(&mut Kernel, u16)); _] =
    [
        (Self::opt_no_config, Self::reassociate_commutative),
        (Self::my_opt_config, Self::my_optimization), // <-- add here
    ];
```

#### Key patterns:

- **Config function**: Returns how many variants to try. Return `1` if the optimization has no tunable parameters.
- **Optimization function**: Takes `config: u16` (unused if `opt_config` returns `1`). Apply the optimization transformation to the kernel IR.
- **Cost model**: The autotune uses a heuristic cost model (`get_cost`) for initial beam search, then measures actual execution time for final selection.
- **Hashing**: The system uses kernel hashing to avoid duplicate exploration. Ensure your optimization produces deterministic IR.

#### Example: optimization with variants

```rust
impl Kernel {
    pub fn tile_sizes(&self) -> u16 {
        4 // Try 4 different tile sizes
    }

    pub fn tile_loops(&mut self, config: u16) {
        let tile_size = [16, 32, 64, 128][config as usize];
        // Apply loop tiling with tile_size...
    }
}
```
