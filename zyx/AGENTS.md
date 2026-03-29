# Agent Guidelines for zyx (Core Crate)

Quick reference for coding agents working in the core `zyx` crate - the tensor library.

## Essential Commands

```bash
# Build
cargo build -p zyx
cargo build -p zyx --release

# Lint (strict rules enforced)
cargo clippy -p zyx --all-features -- -D warnings

# Format
cargo fmt

# Test
cargo test -p zyx
cargo test -p zyx relu_1          # single test
cargo test -p zyx --test 1_unary  # test file
cargo test -p zyx -- --nocapture  # with output
```

## Module Structure

```
zyx/src/
├── lib.rs           # Crate root, public API exports
├── tensor/          # Tensor operations
│   ├── mod.rs       # Tensor struct and methods
│   ├── binary_ops.rs
│   ├── index_ops.rs
│   └── reduce_ops.rs
├── runtime.rs       # Runtime execution engine
├── kernelize.rs     # Kernel compilation/JIT
├── scalar.rs        # Scalar value handling
├── dtype.rs         # Data types
├── autograd.rs      # Automatic differentiation
├── graph.rs         # Computation graph
├── view.rs          # View/strides handling
├── cache.rs         # Caching mechanisms
├── module.rs        # Neural network modules
├── backend/         # Backend implementations
├── kernel/          # Kernel definitions
├── error.rs         # Error types
├── shape.rs         # Shape handling
├── rng.rs           # Random number generation
└── schedule.rs      # Scheduling
```

## Key Concepts

### Tensor Structure
- Main struct: `Tensor` in `tensor/mod.rs`
- All tensor operations defined there
- Uses views (`view.rs`) for memory efficiency

### Runtime System
- `runtime.rs`: Execution engine for tensor operations
- Supports synchronous and asynchronous execution

### Kernel System
- `kernelize.rs`: JIT kernel compilation
- `kernel/`: Kernel definitions and autotuning
- Backends loaded dynamically (see `backend/`)

### Autograd
- `autograd.rs`: Automatic differentiation
- `graph.rs`: Computation graph building

## Code Style

Follow the same conventions as the root AGENTS.md:
- PascalCase for types, snake_case for functions/variables
- Imports: `crate::`, `super::`, external crates, then `pub use`
- License header required
- All public items need docs
- Strict clippy enforcement

## Testing

- Tests in `zyx/tests/`
- Naming: `{number}_{category}.rs`
- Return `Result<(), ZyxError>`
- Use `assert!` and `is_equal()` for float comparison

## Backend Architecture

- Backends in `src/backend/`
- Loaded at runtime via `.so` files
- FFI limited to one file per backend
