# Optimization Passes

Optimization passes operate on the linear kernel IR. They are divided into **always-on passes** (run before every compilation) and **autotuned passes** (searched at runtime for the best variant).

The design goal of the small opset is that new passes are easy to write. There is no fixed number zyx aims to have — passes will be added over time as performance opportunities are identified.

## Always-On Optimizations

These run in a fixed pipeline (`default_epilogue`) on every kernel state during autotuning and before compilation:

```rust,ignore
pub fn default_epilogue(&mut self) {
    self.unroll_len1_loops();
    self.constant_folding();
    self.move_constants_to_beginning();
    self.loop_invariant_code_motion();
    self.fold_accs();
    self.delete_zero_len_indices();
    self.delete_zero_len_loops();
    self.unfold_pows();
    self.algebraic_simplifications();
    self.simplify_accumulating_loop();
    self.swap_commutative();
    self.common_subexpression_elimination();
    self.instruction_schedule();
    self.dead_code_elimination();          // must stay last
    // + exp ↔ exp2 conversion (per device capability)
    // + tenstorrent tile lowering when targeting TT
}
```

### Constant Folding

Evaluates expressions where all inputs are constants. For example, `2 + 3` becomes `5` and `sin(0.0)` becomes `0.0`.

### Motion of Constants to Beginning

Moves all constant definitions to the start of the kernel. This creates better fusion opportunities and simplifies register allocation.

### Loop-Invariant Code Motion (LICM)

Hoists operations that produce the same value on every iteration to before the loop.

### Algebraic Simplifications / Unfold Pows / Swap Commutative

Rewrites like `x*1 → x`, `x+0 → x`, exponentiation unfolding, and normalizing commutative operand order so CSE's hashing sees identical forms.

### Accumulator Folding / Simplify Accumulating Loop

Simplifies accumulator update patterns and loop-carried reductions for more efficient code generation.

### Instruction Scheduling

Orders ops within basic blocks to improve ILP before codegen.

### Dead Code Elimination

Removes ops whose results are never used. This is always the final pass — backends fail on unreferenced ops.

## Autotuned Passes

The autotune system (`BeamSearch`) clones the kernel, applies optimization variants, and evaluates each separately. No egraphs — just clone, transform, hash, evaluate. The cost function can evaluate **thousands of variants per second**.

```rust,ignore
pub const fn default_optimizations() -> [MakeOpt; 8] {
    [
        Kernel::opt_split_global_to_local,
        Kernel::opt_reassociate_commutative,
        Kernel::opt_coarsen,
        Kernel::opt_register_blocking,
        Kernel::opt_local_reduce,
        Kernel::opt_split_loop,
        Kernel::opt_merge_nested_loops,
        Kernel::opt_fuse_mad,
    ]
}
```

### Split Global to Local

Adjusts block and thread dimensions for better memory access patterns. For example, a kernel with `block_dim = 1024, thread_dim = 1` becomes `block_dim = 32, thread_dim = 32`. This enables coalesced memory access on GPU.

### Reassociation

Reorders commutative operations to create more fusion opportunities.

### Coarsen

Merges parallel work per thread so each thread handles multiple elements (the inverse of splitting).

### Register Blocking

Unrolls tree reductions and coarsens global threads so each thread processes multiple elements, increasing computational intensity and register reuse.

### Local Reduce

Reduces within a thread's local data before hitting global or shared memory.

### Loop Splitting

Splits large loops into chunks for better register pressure and instruction-level parallelism.

### Merge Nested Loops

Collapses nested loop nests into a single loop where semantics allow.

### Fuse Multiply-Add

Fuses `mul` + `add` chains into `Op::Mad`.

## Performance Budget

Every pass must stay fast — tens of microseconds is the ceiling, single-digit microseconds the norm. Passes run once per autotune variant and per compiled kernel, so slow passes compound into seconds of autotuning.

## How Autotuning Searches

1. Start with the initial kernel and run the epilogue
2. Apply ONE optimization variant and run the epilogue
3. Hash the kernel — skip if already visited
4. Launch and measure, or evaluate with the cost function
5. Repeat by combining with existing optimization sequences
6. Select the best variant; only the top variants are compiled and launched

## Correctness Guarantee

> No optimization is needed for tests to pass. All tests must pass no matter which sequence of optimizations (including empty) is applied.

If an optimization breaks a test, the optimization is buggy — not the sequence ordering. The verify pass (`verify.rs`) checks internal IR consistency in debug mode.
