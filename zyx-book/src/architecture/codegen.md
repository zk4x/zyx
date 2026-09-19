# Codegen and the Physical IR

The kernel IR's two jobs are both value-shaped: lower movement ops into index arithmetic, and be the space the autotuner searches over — tens of thousands of variants, measured time decides. Everything decision-shaped happens before codegen. What remains for the backend is purely machine-shaped: registers, sync, placement, then bytes.

This chapter describes how codegen is organized, and the Tenstorrent physical IR (TTIR) that exists because that hardware needs more than a straight lowering.

## The Two Boundary Tests

Two tests compose into the full boundary between kernel IR and backend code:

### The search test — which IR holds an op

> An op belongs in the kernel IR iff the autotuner should ever enumerate over it — iff there exists a variant of the program where it is placed or structured differently, and the hardware decides between them.

Movement lowering, loop structure, split/coarsen/register blocking, MAD fusion: search dimensions, kernel IR. Locks, syncs, reconfigs: their placement is invariant under restructuring, nothing to search — they cost the search nothing and belong to the physical IR, placed by derivation from loop structure and checked by verify.

Consequences:

- **The search space is closed.** The autotuner never sees a physical IR, and no backend can smuggle a decision out of the search. If a backend-specific dimension is ever worth searching, it enters as a kernel IR config + `MakeOpt`, keeping the cost model measured-time-only and backend-agnostic.
- **The physical IR's no-restructuring rule is a definition, not a discipline.** Restructuring is a search dimension; search lives in the kernel IR; a physical-IR op that wants restructuring failed the search test and is in the wrong IR.
- **Criterion for future disputes:** "would the autotuner enumerate over this?" Yes → kernel IR + `MakeOpt`. No → physical IR, derived placement, structural verify.

### The subject test — where a pass lives

> A pass belongs in an IR only if its subject exists in that IR.

If a pass reasons about values, it belongs in the kernel IR (or above). SSA machinery — rewrites, solvers, def-use analysis — is never duplicated in a physical IR. When a physical-IR pass needs a dataflow fact (e.g. "is this operand's last use?"), it queries the kernel IR through the origin op recorded at lowering time.

The search test decides where an *op* lives; the subject test decides where a *pass* lives.

## Straight Codegen Is the Default

For most backends (C, CUDA, OpenCL, Vulkan, HIP, WGPU), codegen is trivial by design: deSSA, then one linear pass over the kernel IR emitting target code. No further optimizations, no backend-specific lowering. The hard work happened in the optimization passes; the IR was deliberately shaped so backends stay thin.

## TTIR — the Tenstorrent Physical IR

The Tenstorrent (Wormhole) compute model — three RISC-V threads (reader, math, writer), circular buffers with FIFO semantics, DST tile slots with lock/commit protocol, SFPU LREGs, unpacker/packer format state — cannot be expressed as a straight per-op lowering. The legacy codegen fused lowering, scheduling, and text emission into one walk with emergent state machines (CB accounting, init placement, format trackers). TTIR makes that state explicit: a typed, ordered, physical instruction stream that passes rewrite and a verifier checks, so invalid programs never reach the JIT.

### Representation

One program = one `Vec<TTOp>` covering all three RISC-V kernel sources. Section boundaries are ops in the stream:

```text
[reader ops] EndReader [compute ops] EndCompute [writer ops] EndWriter
```

- The barriers that today exist only as scan positions become first-class ops; verify walks one stream and checks cross-section facts (CB push/pop totals settled at each boundary).
- The renderer splits at the markers — one stream in, three kernel files out.
- Ids are typed and distinct — `VId` (virtual, kernel `OpId`s at lowering), `LRegId` (SFPU LREGs), `DstId` (DST slots), `VarId` (scalar C registers), `CBId` (circular buffers) — so misuse is a compile error in the IR's own definitions.
- Not SSA: value ops carry their target (`z: OpId`); SSA-ness holds as an asserted invariant (single definition, in order) only until a pass rewrites it.
- Every variant carries a **static signature** — which CB it moves (produce/consume, n tiles), DST lock effects, arg reads, which thread executes it. `CopyTile` and its `Unpacr` expansion carry the same signature, so swapping between levels is sound by construction.
- No catch-all `Raw(String)` variant. Every line the backend emits is an enum variant; anything opaque is an explicit variant (including inline assembly with a typed operand list).

### Pipeline

Fixed order, one pass per concern, each pass TT-specific:

1. **Convert + materialize** — 1:1 kernel IR → TTOp, then duplicate cross-section values per section (each section is a separate kernel with its own registers and runtime args; arg ordinals stay global; CBs are one hardware object and never duplicated). SSA dedups globally, physical duplicates locally.
2. **Sync insertion** — `ReserveBack`/`WaitFront`/`PushBack`/`PopFront` per traffic op.
3. **Lock insertion** — `Acquire` placed directly at the LICM position (preheader of the outermost loop containing a pack): the lowering derives the same placement kernel LICM would compute, because the TTIR stream mirrors kernel loop structure 1:1. No effect ops enter the kernel IR for this — the placement is derivable, so there is nothing to search over.
4. **Reconfig/init insertion** — naive full reconfig before every copy/pack; correctness first, redundancy is safe.
5. **Hoist + dedup** — adjacent same-config reconfigs collapse; effect ops hoist out of constant-trip loops with trip-count accounting (FIFO depth × trip). Runs on the fully-placed stream: place first, then clean.
6. **Regalloc + coalesce** — late; linear scan over the linear stream, producing in-place forms.
7. **Verify** — structural: budgets per id class, CB balance and traffic correspondence, lock pairing, section termination, and the materialization invariant (every value consumed in a section is defined in that section).
8. **Render** — a table walk, not synthesis.

Each pass rebuilds the stream (drain → push), states which phase it expects (all-virtual → mixed → all-physical), and fails loudly at the exact op it cannot place. Verify guards the composition: whatever the passes did, the final stream must be launchable.

### Why no effect ops in the kernel IR

Hoisting the math lock out of the tile loop looks like a job for kernel LICM. It isn't: the acquire is operand-free, so LICM would trivially hoist it maximally every time — the result is a fixed, derivable position, and the lowering can place the acquire exactly there at insertion time. Adding an effect op to the SSA IR would instead mean auditing CSE (which must never merge two identical acquires straddling a release), DCE, and every restructuring pass — new machinery in the shared IR for zero searchable freedom. Effects whose placement genuinely interacts with state (FIFO trips × loop trip counts, format configs with adjacency) are expressed in TTIR, where the hardware semantics live.

### What the physical IR is not for

- No SSA machinery of any kind (subject test).
- No fused-composite matching: at instruction level the chain *is* the fusion — exp/add/reciprocal sequences share LREGs inside one walk, and the register allocator subsumes the containment analysis. There is no dedicated `elu` op and none is needed: the decomposition is the only spelling, and it is the fast path.
- No scalar optimization: kernel IR folds all scalar SSA work before lowering; scalar variants render dumbly.
- No restructuring: restructuring is a search dimension, and search lives in the kernel IR.

### Migration

The high-level variants are pure abbreviations — macros over lower-level variants with identical signatures. A variant like `BinaryTileAdd` emits today's output; when its decomposition into raw instructions is proven, the variant is deleted and nothing is lost. Rust's exhaustive matches make deletion a compile event that walks through every site, so a half-removed variant cannot exist. The differential test does the proving: the pre- and post-deletion streams must render identically for the whole test corpus. The variant set is the migration ledger — the distance to "no LLK" is countable at any moment by listing the remaining high-level variants.

### Other backends

Each backend gets its own physical IR if and when it needs one. The differences between machines are structural (SIMT warps and shared-memory barriers vs FIFOs and DST locks), so a shared physical IR would degenerate into a lowest-common-denominator enum that expresses neither honestly. What *is* shared: the kernel IR above the boundary, the backend-facing input contract (linear SSA, epilogue-cleaned, arg ordering), and the methodology (typed variants, signature table, phase invariants, rebuild-per-pass, verify, render, differential gate). If two backend physical IRs later converge on the same variants for a machine-independent concept, a shared sub-IR can be extracted then — convergence-discovered unification is sound; designed-upfront unification is a layer nobody asked for.
