# Optimization Passes

Optimization passes operate on the linear kernel IR. There are two kinds:

- **always-on passes** — the epilogue, run on every candidate and before
  compilation;
- **autotuned passes** — made into a config space by a `MakeOpt` and selected by
  `BeamSearch` using measured launches.

The design goal of the small opset is that new passes are easy to write. There is
no fixed number zyx aims to have; passes are added when a performance
opportunity is identified.

## The Epilogue

`default_epilogue` is the always-on pipeline. It runs after each optimization
step during the search and three times before the search starts:

```rust,ignore
pub fn default_epilogue(&mut self) {
    self.delete_useless_loops();
    self.constant_folding();
    self.move_constants_to_beginning();
    self.loop_invariant_code_motion();
    self.fold_accs();
    self.algebraic_simplifications();
    self.simplify_accumulating_loop();
    self.simplify_accumulating_loop();
    // Second sweep: the first sweep's rewrites expose more work.
    self.constant_folding();
    self.move_constants_to_beginning();
    self.loop_invariant_code_motion();
    self.fold_accs();
    self.algebraic_simplifications();
    self.simplify_accumulating_loop();
    self.swap_commutative();
    self.common_subexpression_elimination();
    self.instruction_schedule();
    self.dead_code_elimination();
}
```

The core runs twice because a first pass of constant folding/LICM/algebraic
simplification often exposes a second round of the same rewrites.
`simplify_accumulating_loop` appears twice in the first sweep for the same
reason. The epilogue is device-independent: it never converts a transcendental
and leaves `Pow` alone.

### What each always-on pass does

- **Delete useless loops** — collapses length-1 counted loops and ranges to the
  constant index `0` (the body still executes once), then removes loops whose
  body has no effect. Boolean-length loops are conditionals, not trip counts,
  and are never collapsed by the length-1 rule.
- **Constant folding** — evaluates any op whose inputs are all constants
  (`2 + 3` → `5`, `sin(0.0)` → `0.0`).
- **Move constants to beginning** — hoists constant definitions to the head,
  which exposes fusion opportunities and simplifies register allocation.
- **Loop-invariant code motion (LICM)** — hoists ops that produce the same value
  on every iteration.
- **Fold accumulators** — removes accumulator updates whose value is never
  stored back inside the loop (dead accumulation).
- **Algebraic simplifications** — `x * 1 → x`, `x + 0 → x`, `x - x → 0`, and
  similar identities over the concrete dtype.
- **Simplify accumulating loop** — replaces a closed-form accumulate loop (for
  example a sum of successive indices) with its arithmetic formula; it processes
  one such loop per call.
- **Swap commutative** — canonicalizes commutative operands so CSE hashes see
  identical forms.
- **Common subexpression elimination (CSE)** — merges structurally identical ops.
  Storage allocation ids and param ids are part of the key, so two physical
  registers or two buffer arguments are never merged after allocation.
- **Instruction scheduling** — orders the head as read-only params, `GlobalMut`
  params, local read-only storages, then local read-write storages; the rest is
  topologically sorted with fewest dependencies first. Memory ops that share a
  base keep their relative order, stores never leave their loop/conditional, and
  nothing moves before a barrier.
- **Dead code elimination (DCE)** — removes ops with no users. `Store`, `Copy`,
  `Barrier`, `Loop`, and program-order roots are always kept. DCE is last:
  backends expect no unreferenced values.

## The Device Epilogue

`device_epilogue` runs exactly once per kernel, after the search and before the
timed launches and codegen. These decisions are lowering, not search: a kernel
optimized ten times during the search must not be re-expanded each time, and a
user-supplied epilogue must not be able to skip them.

```rust,ignore
pub fn device_epilogue(&mut self) {
    self.unfold_pows();               // Pow -> exp2(y * log2 x), every backend
    if device_is_tenstorrent {
        self.tt_tile();               // vector IR -> tiles
        self.tt_unfold();             // Exp2/Log2 become native ops
        self.tt_reindex();            // logical indices -> tile traffic
        self.default_epilogue();      // clean the tile IR
        self.tt_dataflow_select();    // phase boundary: native instructions
        self.tt_sfpu_fusion();        // value -> lane register
        self.tt_lreg_alloc();         // virtual -> physical LREGs
        self.tt_dst_reuse();          // coalesce DST tiles
        self.tt_reg_sync();           // Semset/Wait/Post hand-offs
        self.tt_verify();
    } else {
        self.devectorize();           // lane vectors -> scalar statements
    }
}
```

The Tenstorrent path is the physical IR described in
[Codegen and the Physical IR](./codegen.md); generic passes may not run after
`tt_dataflow_select`, and `tt_verify` replaces the generic `verify`.

## Autotuned Passes

`default_optimizations` returns the make functions the search explores:

```rust,ignore
pub const fn default_optimizations() -> [MakeOpt; 10] {
    [
        Kernel::opt_split,
        Kernel::opt_reassociate_commutative,
        Kernel::opt_local_reduce,
        Kernel::opt_merge,
        Kernel::opt_pad,
        Kernel::opt_tensorize,
        Kernel::opt_local_staging,
        Kernel::opt_recipe,
        Kernel::opt_register_tiling,
        Kernel::opt_tile_and_stage,
    ]
}
```

A make function scans the kernel at a stable state and returns a `Box<dyn
Optimization>`: an object with `nconfigs()` and `apply(kernel, config)`. No pass
runs between make and apply, so `OpId`s embedded by the make function stay valid.
An optimization that cannot fire returns zero configs or bails inside `apply`;
it never aliases a wrong transformation.

### Split

`opt_split` splits one site per config: a group range into a smaller group plus
local threads (`SplitMode::Local`), a group range into a group plus unroll lanes
(`SplitMode::Unroll`), or a loop into two nested loops (`SplitMode::Loop`). The
split index is composed as an affine expression inline, so the IR grows only in
op count, not in vector width; lane growth happens later when devectorization
desugars the chosen variant. Validity — divisibility, lane limits, the
boolean-loop gate — is checked at make time.

### Reassociate commutative

`opt_reassociate_commutative` re-associates commutative chains so that constants
and repeated operands group together. It exposes folding and CSE opportunities
that operand order alone hides.

### Local reduce

`opt_local_reduce` targets a single large loop reducing into one register
accumulator (a standalone `sum`/`max` over a long axis, not a matmul). It
parallelizes the reduction across threads, stores partial results in local
memory, and tree-reduces them. One config per (reduce site, local factor,
global factor).

### Merge

`opt_merge` fuses exactly two axes into one flat axis and rewrites the originals
as div/mod decomposition of the flat axis. Each config merges one nested loop
chain (inner loop inside the outer's region, no store/copy in between) or one
range pair (both group or both local; axes ascending; symbolic lengths bail).
Chains fold through pairwise calls, so a refused later step leaves valid IR.

### Pad

`opt_pad` rounds range lengths up to a hardware-friendly multiple (for example
32) using the round-up idiom as IR, with masks so out-of-range loads read zero
and out-of-range stores are skipped. Loops pad only on Tenstorrent, where tile
loops must run full-tile trip counts. Lengths already provably a multiple are
left untouched, and padding is idempotent.

### Tensorize

`opt_tensorize` recognizes a scalar matmul accumulation loop
(`acc = a[m,k] * b[k,n] + acc` over K) and rewrites it into `Op::Wmma`. It emits
one config per device MMA variant the extent and dtypes fit
(`DeviceInfo::mma_variants`). The pass does the tiling and fragment addressing
itself; the result is scalar IR of slot-wise register accesses, so `devectorize`
has nothing to undo, and it removes the original scalar region.

### Local staging

`opt_local_staging` deduplicates loads that threads repeat. When a load's index
does not mention a local axis, every thread re-loads the same value; staging
fills a `MemScope::Local` tile cooperatively once and lets the threads read it
back. The pass strip-mines the innermost loop the load depends on and splits the
body into a fill half and a compute half (ops between the load and its consumers
that depend only on the load and constants move into the fill, so a dequantized
value is staged rather than its packed form). One config per (repeated load,
strip factor); it bails on devices without local memory, non-affine indices, or
tiles that do not fit or divide over the block.

### The recipes

The last three passes are **recipes**: fixed sequences of known-good steps
exposed as one search move, so the beam can reach a good kernel shape without
walking every intermediate state. A recipe returns a config space, never a
winner, and only fires from a plain kernel (no lanes, barriers, reduces, local
memory, or fused matmul — another pass already ran).

- **`opt_recipe`** — the lane recipe: split a group range into local lanes
  (factors tried largest first), then stage the operand the lanes share.
- **`opt_register_tiling`** — the register-blocking recipe (the former coarsen
  era): unroll the reduce loop into accumulators, then each output range into a
  per-thread tile. Reduce sites are mandatory; output ranges may be skipped.
- **`opt_tile_and_stage`** — lanes on one output range, a per-thread tile on
  both, then stage the shared operand.

## How the Search Works

`BeamSearch` is the driver. Its defaults are:

```rust,ignore
BeamSearch {
    n_launches: 1,          // candidates actually launched and timed
    n_seeds: 200,           // initial optimization samples
    n_added_per_step: 200,  // new candidates per expansion
    n_removed_per_step: 0,  // candidates dropped per step
    n_total_opts: 1000,     // expansion iterations
}
```

The flow:

1. The caller supplies **seeds**: linearized kernels, already epilogue-cleaned,
   all on one device and sharing one parameter signature. The search does no
   seed preprocessing.
2. Every make function is applied to the base kernel; configs are subsampled
   proportionally to `n_seeds`, the epilogue runs after each apply, and the
   resulting kernel hash deduplicates visited states.
3. Candidates are ranked by `cost` (by default `Kernel::base_cost`, a weighted
   feature count — no launch). The best-scoring sequence is sampled and expanded
   with new configs; `n_total_opts` expansions are attempted, dropping the worst
   `n_removed_per_step` each round.
4. The top `n_launches` candidates are rebuilt deterministically, `device_epilogue`
   runs once on each, and each is compiled and launched with `launch_with_timings`.
   Candidates that fail to compile/launch are skipped, and the search keeps going
   until it has measured `n_launches` successes or runs out.
5. Every measured program is released; the fastest kernel is returned with its
   time. The cost function only orders the search — **measured time decides**.

Kernels are identified by content hash (`Kernel::get_hash`), so the same state
reached by different sequences is explored once. `ZYX_DEBUG=256` selects
`NoSearch`, which skips the seed search and compiles a cleaned seed once —
useful to bisect whether a wrong result comes from the search space or from seed
preparation/codegen.

## Performance Budget

The repeated passes must stay fast: **30 microseconds per invocation** is the
ceiling for `default_epilogue` and for the whole `default_optimizations` list
per candidate. Those run once per candidate inside the search, so a slow pass
compounds into seconds of autotuning. The device epilogue runs once after the
search and is not bound by that budget.

Passes are measured with the `time_pass!` macro or `std::time::Instant` on
stderr; a regression in algorithmic cost is a performance bug, not a style
issue.

## Correctness Guarantee

> No optimization is needed for tests to pass. All tests must pass no matter
> which sequence of optimizations (including empty) is applied.

If an optimization breaks a test, the optimization is buggy — not the sequence
ordering. Passes must preserve semantics or bail (`todo!()`/zero configs), never
silently alias a wrong transformation. `Kernel::verify` checks IR consistency in
`debug_assertions` builds; the Tenstorrent physical IR is checked by `tt_verify`.
