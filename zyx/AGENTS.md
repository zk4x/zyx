# Agent Guidelines for zyx (Core Crate)

## BEFORE ANY ACTION: ASK THE USER FIRST

**Do NOT run any commands, write any code, or search for answers without first telling the user what you think the problem is and asking what they want you to do.**

This is the single most important rule. Every session you default to acting instead of talking. Stop. Tell the user your understanding. Ask what they want. Only then act.

Examples of what this means in practice:
- User: "wtf is test failing?" → Do NOT run cargo test. Say: "I don't know, should I investigate?"
- User: "why is X done this way?" → Do NOT search the codebase. Say: "I'm not sure, want me to look?"
- User reports a bug → Do NOT open files or write fixes. Say what you think the issue is and ask.

**Even when the user says "investigate" or "look into it", do NOT start running tools.** First state your hypothesis and ask for confirmation. Examples:
- User: "investigate opencl" → Do NOT run cargo test. Say: "The recent MemLayout refactor likely has the same pattern as the C backend bug. Want me to check if the condition is inverted there too?"
- User: "look into the test failure" → Do NOT run the test. Say: "The error mentions undefined symbols, probably from a recent commit. Want me to check?"

**When you don't know how to do something (e.g., how to enable/run a backend), ask the user.** Do NOT search the codebase to figure it out yourself. Examples:
- User: "investigate opencl" → Do NOT search Cargo.toml for features or backend/mod.rs for how OpenCL is loaded. Say: "How do I run the OpenCL tests? Is there a feature flag I should use?"
- User: "test the wgpu backend" → Do NOT look for wgpu features in Cargo.toml. Say: "How do I enable wgpu for testing?"

This document is your single source of truth. If it doesn't contain the answer, ask. Don't search.

**When the user tells you something about the system (e.g., "OpenCL crashes, CUDA doesn't"), trust them.** Do NOT run tests or commands to verify what they said. Use their information to narrow down the problem, not to double-check it.

**Do NOT "fix" things you weren't asked to fix.** If you notice something wrong (missing catch_unwind, formatting issues, whatever), shut up. Do not mention it. Do not fix it. The user knows. If they want it fixed, they'll ask.

**When the user says "commit current state", commit the EXACT current state.** Do not modify anything first. Not even if it looks like a debugging artifact. Not even if it looks obviously wrong. Commit exactly what's on disk.

**Never edit a file without explicit instruction from the user.** "Restore the original handler" is explicit. "Investigate the crash" is NOT explicit. If you're unsure whether the user wants a file edited, you don't edit it.

If you catch yourself typing a tool call before replying to the user, stop. Reply first.

---

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
- **No helpers**: Extract-platform-specific code inline with `#[cfg]`. Never create helper functions that wrap platform differences.

## Testing

- Tests in `zyx/tests/`
- Naming: `{number}_{category}.rs`
- Return `Result<(), ZyxError>`
- Use `assert!` and `is_equal()` for float comparison

## Backend Architecture

- Backends in `src/backend/`
- Loaded at runtime via `.so` files
- FFI limited to one file per backend
- **All backends are compiled in by default** (no feature flags for C, CUDA, OpenCL, HIP, etc. — wgpu and tenstorrent are behind optional features)
- The runtime auto-detects which backends are available and picks the **fastest** one with available memory
- Tests use whatever backend is available — no special flags needed
- See detected backends with `ZYX_DEBUG=1 cargo test -p zyx`
- Backends can be enabled/disabled via config file — see [`CONFIG.md`](./CONFIG.md)

## Debugging Crashes

When investigating a crash (segfault, signal, etc.):

1. **Write a minimal reproducer** — a single test function that triggers the bug. Keep it small.
2. **Isolate the crash line** — add `eprintln!("reached A")`, `eprintln!("reached B")`, etc. at key points. Run and see which is the last printed. If output is missing (not flushed), use `panic!("reached X")` instead — panics always flush.
3. **Narrow down** — once you know what line crashes, look at what that line does and work forward to figure out what's wrong.

### Debugging Hangs

When a test hangs (no crash, no output):

1. **Add `eprintln!("reached A")`, `eprintln!("reached B")`, etc.** at key points in the suspected code path
2. Run the test with `--nocapture` to see where output stops
3. The last printed line is where the hang occurs — no need to hypothesize or ask, just add prints

## Debug Options

Set `ZYX_DEBUG` environment variable (bitmask):

| Value | Flag | Description |
|-------|------|-------------|
| 1     | dev  | Print hardware devices and configuration |
| 2     | perf | Print graph execution characteristics |
| 4     | sched| Print kernels created by scheduler |
| 8     | ir   | Print kernels in intermediate representation |
| 16    | asm  | Print native assembly/code (OpenCL, WGSL, etc.) |

Example: `ZYX_DEBUG=1 cargo test -p zyx` (see detected backends)
Example: `ZYX_DEBUG=16 cargo test -p zyx --features wgpu relu_1`

### Colorless Debug Output

Set `AGENT=1` to print kernel IR debug output without ANSI color codes (agent-friendly):

```bash
AGENT=1 cargo test -p zyx 2>&1
AGENT=1 cargo test -p zyx -- --nocapture 2>&1 | less -R
```

This works with `ZYX_DEBUG=4` (scheduler output) and any `kernel.debug()` calls.

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

#### Optimization Correctness (CRITICAL)

Every optimization must produce correct IR that calculates the same result as the input. **No optimization is needed for tests to pass.** ALL tests must pass with ALL optimizations disabled, and ALL tests must pass no matter which sequence of optimizations (including empty) is applied. If any sequence breaks correctness, the optimization that produced invalid IR from valid code is BUGGY and must be fixed or disabled.

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
