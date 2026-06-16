# Optimization Passes

Optimization passes operate on the linear kernel IR. They are divided into **always-on passes** (run before every compilation) and **autotuned passes** (searched at runtime for the best variant).

The design goal of the small opset is that new passes are easy to write. There is no fixed number zyx aims to have — passes will be added over time as performance opportunities are identified.

## Always-On Optimizations

These run in a fixed pipeline before every kernel compilation:

```rust,ignore
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

### Constant Folding

Evaluates expressions where all inputs are constants. For example, `2 + 3` becomes `5` and `sin(0.0)` becomes `0.0`.

### Motion of Constants to Beginning

Moves all constant definitions to the start of the kernel. This creates better fusion opportunities and simplifies register allocation.

### Loop-Invariant Code Motion (LICM)

Hoists operations that produce the same value on every iteration to before the loop.

### Common Subexpression Elimination (CSE)

Reuses results of identical subexpressions by detecting them through hashing.

### Accumulator Folding

Simplifies accumulator update patterns for more efficient reduction code generation.

### Delete Empty Loops

Removes loops with zero iterations.

### Dead Code Elimination

Removes ops whose results are never used. This is always the final pass — it ensures backends never receive unreferenced ops.

## Autotuned Passes

The autotune system clones the kernel, applies optimization variants, and evaluates each separately. No egraphs — just clone, transform, hash, evaluate. The cost function can evaluate **thousands of variants per second**.

```rust,ignore
const AVAILABLE_OPTIMIZATIONS: [OptConfigFn; 6] = [
    Kernel::opt_reassociate_commutative,
    Kernel::opt_split_global_to_local,
    Kernel::opt_upcast,
    Kernel::opt_register_blocking,
    Kernel::opt_tiled_reduce,
    Kernel::opt_split_loop,
];
```

### Reassociation

Reorders commutative operations to create more fusion opportunities.

### Split Global to Local

Adjusts block and thread dimensions for better memory access patterns. For example, a kernel with `block_dim = 1024, thread_dim = 1` becomes `block_dim = 32, thread_dim = 32`. This enables coalesced memory access on GPU.

### Upcast Vectorization

Expands scalar operations to vector operations (e.g., 4-wide SIMD on CPU, wider on GPU).

### Register Blocking

Unrolls tree reductions and coarsens global threads so each thread processes multiple elements, increasing computational intensity and register reuse.

### Tiled Reduction

Implements multi-stage reduction: threads reduce into registers, workgroups reduce into local memory, then global atomics.

### Loop Splitting

Splits large loops into chunks for better register pressure and instruction-level parallelism.


## How Autotuning Searches

1. Start with the initial kernel and run always-on optimizations
2. Apply ONE optimization variant and run always-on optimizations
3. Hash the kernel — skip if already visited
4. Evaluate with cost function (or launch and time)
5. Repeat by combining with existing optimization sequences
6. Select the best variant based on actual timing or cost estimate

## Correctness Guarantee

> No optimization is needed for tests to pass. All tests must pass no matter which sequence of optimizations (including empty) is applied.

If an optimization breaks a test, the optimization is buggy — not the sequence ordering. The verify pass (`verify.rs`) checks internal IR consistency in debug mode.
