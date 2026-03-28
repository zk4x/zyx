# Agent Guidelines for Zyx

Zyx is a machine learning library with JIT kernel generation, fusion, and compilation. It provides dynamic graph construction with lazy execution.

## Project Structure

```
zyx/                      # Main crate
├── src/
│   ├── lib.rs           # Crate root with module declarations
│   ├── autograd.rs      # Gradient tracking and backpropagation
│   ├── backend/         # Backend implementations (CUDA, OpenCL, etc.)
│   ├── cache.rs         # Kernel caching
│   ├── dtype.rs         # Data types (F32, F16, I32, etc.)
│   ├── error.rs         # Error types (ZyxError, BackendError)
│   ├── graph.rs         # Computation graph
│   ├── kernel/          # Kernel IR and operations
│   ├── kernelize.rs     # Kernel generation from graph
│   ├── runtime.rs       # Global runtime management
│   ├── scalar.rs       # Scalar trait for dtype operations
│   ├── schedule.rs      # Kernel scheduling
│   ├── shape.rs        # Shape handling
│   ├── slab.rs         # Memory slab allocator
│   ├── tensor/         # Tensor implementation
│   │   ├── binary_ops.rs
│   │   ├── index_ops.rs
│   │   ├── mod.rs
│   │   └── reduce_ops.rs
│   └── view.rs         # Tensor view/stride handling
├── tests/              # Integration tests (numbered: 1_unary.rs, 2_binary.rs, etc.)
├── rustfmt.toml        # Formatting configuration
├── deny.toml           # Cargo-deny configuration
└── Cargo.toml          # Manifest (edition = "2024")
```

## Build/Lint/Test Commands

### Build
```bash
cargo build              # Debug build
cargo build --release    # Release build (LTO, strip, opt-level 3)
cargo build --features wgpu  # With WGPU backend
```

### Lint
```bash
cargo clippy             # Run clippy (pedantic lints enabled)
cargo fmt --check        # Check formatting (max_width = 120)
```

### Test
```bash
cargo test                    # Run all tests
cargo test -- --test-threads=1  # Run tests single-threaded (recommended)
cargo test relu_1             # Run single test by name
cargo test --test 1_unary     # Run single test file
cargo test --test 1_unary relu_1  # Run specific test in specific file
```

### Security
```bash
cargo deny check             # Check licenses and advisories
```

## Code Style

### Formatting
- Max line width: **120 characters**
- Single-line functions enabled (`fn_single_line = true`)
- Use `cargo fmt` before committing
- See `rustfmt.toml` for full configuration

### Rust Edition
- **Edition 2024** (latest Rust edition)

### Copyright Header
Every source file must include:
```rust
// Copyright (C) 2025 zk4x
// SPDX-License-Identifier: AGPL-3.0-only
```

### Imports
- Use absolute paths from crate root (`crate::module`)
- Group imports by: std → core → external → local
- Sort alphabetically within groups

### Naming Conventions
- Types/Structs: `PascalCase` (e.g., `Tensor`, `GradientTape`)
- Functions/Methods: `snake_case` (e.g., `realize`, `gradient_persistent`)
- Constants: `SCREAMING_SNAKE_CASE` for true constants, `PascalCase` for type-level constants
- Private fields: `snake_case`
- Type aliases: `PascalCase`
- Enum variants: `PascalCase`

### Documentation
- Required on all public items (`#![deny(missing_docs)]`)
- Use `///` for doc comments
- Cross-reference with `[`item`]` syntax
- No broken intra-doc links (`#![forbid(rustdoc::broken_intra_doc_links)]`)

### Error Handling
- **User input errors**: Return `ZyxError` (e.g., shape mismatches, dtype issues)
- **Bugs/internal errors**: Panics with clear messages
- **Hardware failures**: Panics (OOM, driver issues)
- Pattern: `ZyxError::shape_error("message".into())` with `#[track_caller]`
- Use `Result<T, ZyxError>` for fallible operations
- Use `Ok::<(), ZyxError>(())` in examples

### Clippy Configuration
The project enables extensive pedantic lints. Key allowances in `lib.rs`:
- `use_self` - allowed
- `single_call_fn` - allowed
- `similar_names` - allowed
- `explicit_iter_loop` - allowed
- `module_name_repetitions` - allowed
- `too_many_lines` - allowed
- `multiple_inherent_impl` - allowed
- `self_named_module_files` - allowed

### Unsafe Code
- Required for FFI and hardware access
- Must be clearly documented why unsafe is necessary
- Keep unsafe blocks minimal and isolated

### Type Conventions
- Custom type aliases: `type Set<T> = HashSet<...>` and `type Map<K, V> = HashMap<...>`
- Use `#[must_use]` on functions returning values that shouldn't be ignored
- Prefer const methods where possible
- Use newtypes for ID types (e.g., `struct TensorId(u32)`)

### Module Organization
- One module per file or file per major component
- Private implementation details in `mod.rs`
- Use `pub(super)`, `pub(crate)`, and `pub` visibility appropriately
- Feature-gated modules with `#[cfg(feature = "...")]`

### Testing Patterns
- Integration tests in `tests/` directory
- Numbered naming: `1_unary.rs`, `2_binary.rs`, etc.
- Test structure:
  ```rust
  #[test]
  fn test_name() -> Result<(), ZyxError> {
      let x = Tensor::from([1.0f32, 2.0]);
      let y = x.relu();
      assert_eq!(y, [0.0f32, 2.0]);
      Ok(())
  }
  ```
- Use `Tensor::from()` for initialization
- Use `is_equal()` for floating point comparisons
- Use `try_into()` to extract results

### Tensor API Patterns
- Methods take `&self` and return `Result<Tensor, ZyxError>`
- Destructive operations return new tensors (immutable design)
- Use method chaining: `x.sum(axes)?.relu()?.exp2()`
- Broadcasting follows NumPy semantics
- All tensors are differentiable by default

### Performance Considerations
- Lazy evaluation: tensors don't compute until `realize()` or accessed
- Kernel fusion: combine operations automatically
- Custom hasher (`chasher`) for deterministic hashing
- Release builds use LTO, strip symbols, opt-level 3

### Key Abstractions
- **Runtime**: Global singleton managing all tensors and devices
- **Graph**: Lazy computation graph with automatic differentiation
- **Kernel**: Fused kernel representation for GPU execution
- **Slab**: Memory allocator for tensor storage
- **GradientTape**: Records operations for backpropagation

### Feature Flags
- `default`: Basic tensor operations only
- `wgpu`: Enable WGSL/WGPU backend
- `py`: Enable Python bindings via PyO3

### Dependencies Philosophy
- Minimal dependencies (only `nanoserde`, `libloading`, `half`, `paste`)
- All dependencies must be carefully justified
- Optional heavy dependencies (WGPU) kept behind feature flags
