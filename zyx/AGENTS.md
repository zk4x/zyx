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

**Note**: This is a workspace with multiple crates. Always run commands from inside this crate subdirectory or use `-p zyx` flag from the workspace root.

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

## Debug Options

Set `ZYX_DEBUG` environment variable (bitmask):

| Value | Flag | Description |
|-------|------|-------------|
| 1     | dev  | Print hardware devices and configuration |
| 2     | perf | Print graph execution characteristics |
| 4     | sched| Print kernels created by scheduler |
| 8     | ir   | Print kernels in intermediate representation |
| 16    | asm  | Print native assembly/code (OpenCL, WGSL, etc.) |

Example: `ZYX_DEBUG=16 cargo test -p zyx --features wgpu relu_1`

## Autotune System

The autotune system in `src/kernel/autotune.rs` searches for optimal kernel configurations.

### Writing Optimization Passes

Optimization passes are methods on `Kernel` that transform the IR. They must accept `config: u16` as a parameter (used for tunable variants).

**1. Define a config function** - returns number of variants to try:

```rust
pub fn my_opt_config(&self) -> u16 {
    4 // try 4 variants (indices 0-3)
}
```

**2. Implement the optimization pass** - transforms kernel IR:

```rust
pub fn my_optimization(&mut self, config: u16) {
    let tile_size = [16, 32, 64, 128][config as usize];
    // Apply transformation to self.ops
}
```

**3. Register in `available_opts` array** in `autotune.rs`:

```rust
let available_opts: [(fn(&Kernel) -> u16, fn(&mut Kernel, u16)); _] = [
    (Self::opt_no_config, Self::reassociate_commutative),
    (Self::my_opt_config, Self::my_optimization), // <-- add here
];
```

#### Always-On Optimizations

The `run_always_on_optimizations` method applies optimizations that should always run before kernel compilation. These are defined in `src/kernel/autotune.rs`:

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

#### Kernel IR Operations

The kernel uses an IR defined in `kernel/mod.rs` with variants like:
- `Op::Binary`, `Op::Unary`, `Op::Mad` - arithmetic
- `Op::Load`, `Op::Store` - memory operations
- `Op::Loop`, `Op::EndLoop` - loop control
- `Op::Vectorize`, `Op::Devectorize` - vectorization
- `Op::WMMA` - warp matrix multiply-accumulate

#### Common Patterns

**Traverse operations:**
```rust
let mut op_id = self.head;
while !op_id.is_null() {
    let op = self.at(op_id);
    // process op
    op_id = self.next_op(op_id);
}
```

**Track loop depth:**
```rust
let mut loop_depth = 0;
while !op_id.is_null() {
    match self.at(op_id) {
        Op::Loop { .. } => loop_depth += 1,
        Op::EndLoop => loop_depth -= 1,
        _ => {}
    }
    op_id = self.next_op(op_id);
}
```

**Insert new operations:**
```rust
let new_op = Op::Binary { x, y, bop };
let new_id = self.insert_before(op_id, new_op);
```

**Verify after changes:**
```rust
#[cfg(debug_assertions)]
self.verify();
```

#### Cost Model

`get_cost()` provides a heuristic estimate. It's used during search to prune the space before actual benchmarking. The cost factors in:
- Instruction count
- Global/local/register scoped loads/stores
- Global work size and local work size

### Key Patterns
- Return `1` from config function if no tunable parameters
- Cost model uses heuristic initially, then actual execution time
- Use kernel hashing to avoid duplicate exploration

## How zyx Works & What NOT to Do

### Architecture Overview

**zyx is a kernel fusion system** - it compiles tensor operations into highly optimized GPU kernels:

✅ **DO:** Follow existing patterns (especially `gather()`, `index_select()`)
✅ **DO:** Use the kernel IR system in `src/kernel/`
✅ **DO:** Leverage one-hot encoding for indexing operations
✅ **DO:** Use tensor operations that fuse into single kernels
✅ **DO:** Follow the autotune system for optimization passes

❌ **DO NOT:** Implement CPU loops for tensor operations
❌ **DO NOT:** Use individual tensor operations that create multiple kernel launches
❌ **DO NOT:** Convert tensors to vectors for manual manipulation
❌ **DO NOT:** Ignore the kernel IR system and try to reimplement at a higher level
❌ **DO NOT:** Fight the architecture - study existing working code and copy patterns

### Performance Expectations

zyx achieves 100-1000x speedups over naive implementations:

- **Optimized zyx scatter**: ~1ms for 1000×1000 operations on GPU
- **Naive CPU loop scatter**: ~100-1000ms (100-1000x slower)
- **zyx approach**: Single fused kernel with massive parallelism
- **Wrong approach**: Sequential CPU loops with individual tensor ops

### Correct Implementation Pattern

Study `gather()` in `tensor/mod.rs` - it shows the right way:

1. **Handle negative indices** using `cmplt()` and `where_()`
2. **Create one-hot encoding** using `unsqueeze()` and `one_hot_along_dim()`
3. **Use `where_()` for conditional operations** - this fuses into GPU kernels
4. **Leverage existing tensor operations** that the runtime can optimize
5. **Let the kernelizer handle the fusion** - don't try to manually optimize

### What NOT to Do (Examples from Real Mistakes)

❌ **WRONG:** CPU loops with individual tensor operations
```rust
for i in 0..n {
    let val = source.slice(i..=i)?.item::<f32>(); // Creates kernel launch
    let idx = indices.slice(i..=i)?.item::<i32>(); // Creates kernel launch  
    target.slice(idx..=idx)?.set_item(val);      // Creates kernel launch
}
```

✅ **RIGHT:** Follow gather pattern with one-hot encoding
```rust
let one_hot = indices.unsqueeze(-1)?.one_hot_along_dim(dim_size, -1)?;
let result = one_hot.where_(&source_expanded, &target_expanded);
```

### Kernel Fusion is Everything

Every tensor operation should fuse into a single optimized kernel:

- **Memory access patterns**: Coalesced global memory access
- **Vectorization**: SIMD instructions on GPU
- **Parallelism**: Thousands of threads working simultaneously
- **Register optimization**: Minimize global memory access

The entire point of zyx is to avoid the "naive" approach of individual operations. Study existing implementations and copy their patterns exactly.
