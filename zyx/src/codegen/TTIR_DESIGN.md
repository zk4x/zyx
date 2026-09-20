# TTIR — Tenstorrent Physical IR Design

Status: design settled, not implemented. Companion to `tenstorrent.rs` (the
legacy string-emitting codegen). The migration target is a new module that
replaces legacy emission op by op, gated by differential tests.

## Goal

Every line the legacy codegen emits as a string becomes an enum variant.
The legacy compiler fuses lowering, scheduling, and text emission into one
walk with emergent state machines (CB accounting, init placement, format
trackers, DST locks). TTIR makes that state explicit: a typed, ordered,
physical instruction stream that passes rewrite and a verifier checks —
so invalid programs never reach the JIT, and the board-safety checks
become structural instead of conventional.

## The two-IR split

- **Kernel IR** (`src/kernel/`) — linear SSA. Does the heavy lifting and
  stays shared across all backends: CSE, DCE, constant folding, algebraic
  rewrites, `opt_fuse_mad`, general LICM, symbolic dims. Everything
  value-shaped happens here.
- **TTIR** (this document) — physical, Tenstorrent-specific, `Vec<TTOp>`,
  ordered, **not SSA**. Only what does not exist in the kernel IR:
  registers, CBs, FIFO state, DST locks, walks, engine configs, thread
  placement, NOC traffic.

Boundary test for what may enter TTIR: *a pass belongs here only if its
subject doesn't exist in the kernel IR.* If a pass reasons about values,
it is in the wrong IR. SSA machinery (rewrites, solvers, phi, def-use
analysis) must never be duplicated — when a TTIR pass needs a dataflow
fact (e.g. "is this operand's last use?"), it queries the kernel IR
through the 1:1 origin `OpId` recorded at lowering time, or consumes a
flag the lowering computed. No liveness solvers over TTIR, ever.

### The search test — which IR holds an op

The kernel IR exists **for the search**: the autotuner enumerates
variants over it and measured time decides. Therefore the primary
criterion for where an op lives:

> An op belongs in the kernel IR iff the autotuner should ever enumerate
> over it — iff there exists a variant of the program where it is placed
> or structured differently, and the hardware decides between them.

Movement lowering, loop structure, split/coarsen/blocking, MAD fusion:
search dimensions, kernel IR. Locks, syncs, reconfigs: their placement is
invariant under restructuring, nothing to search, so they cost the search
nothing and belong to the physical IR — placed by derivation from loop
structure, checked by verify. Consequences:

- **The search space is closed.** The autotuner never sees a physical IR,
  and no backend can smuggle a decision out of the search — if a
  backend-specific dimension is ever worth searching, it enters as a
  kernel IR config + `MakeOpt`, keeping the cost model measured-time-only
  and backend-agnostic.
- **The physical IR's no-restructuring rule is a definition, not a
  discipline.** Restructuring is a search dimension; search lives in the
  kernel IR; a TTIR op that wants restructuring is an op that failed the
  search test and is in the wrong IR.
- **Criterion for future disputes:** "would the autotuner enumerate over
  this?" Yes → kernel IR + `MakeOpt`. No → physical IR, derived
  placement, structural verify.

(The subject test above then decides where a *pass* lives; the search
test decides where an *op* lives.)

## Representation

One program = one `Vec<TTOp>` covering all three RISC-V kernel sources.
Section boundaries are ops in the stream:

```text
[reader ops] EndReader [compute ops] EndCompute [writer ops] EndWriter
```

- The barriers that today exist only as `get_needed_ops` scan positions
  become first-class ops (the honest 1:1 of `Op::Barrier`). `EndWriter`
  is explicit, not implicit end-of-Vec.
- Verify walks one stream and checks cross-section facts where they
  live: CBs settled at each `End*`, push/pop totals program-wide,
  per-section entry states.
- The renderer splits at the markers — one stream in, three kernel
  files out.
- Lock/reconfig cones cannot silently cross sections: boundary semantics
  are in the signature table, not in per-emitter discipline.

### Ids — typed, never one id enum

| type | meaning | budget |
|---|---|---|
| `OpId` | original kernel-IR SSA id, carried verbatim by the 1:1 lowering; completely gone by the end of all passes | unbounded |
| `LRegId` | SFPU LREG | 64 |
| `DstId` | DST tile slot | 16 BF16 / 8 FP32 mode |
| `VarId` | scalar C register (`r{reg}` slots) | unbounded |
| `CBId` | circular buffer | hardware/config |

Distinct types so misuse (SFPU opcode fed a scalar register) is a compile
error in the IR's own definitions, following the `TileId`/`CBId` pattern.

### SSA-ness

Non-SSA by type, **SSA by invariant until rewritten**: the 1:1-mapped
stream writes each `OpId` exactly once, in order. Passes that need def-use
(fusion, regalloc liveness) rely on that invariant as a
`debug_assert!`-checked property (linear scan), not a type guarantee.
De-SSAing happens only when a pass actually rewrites (in-place chains
`SfpMad(z=v7, x=v7, ...)`; coalescing) — by then the def-use-hungry
passes have run. No two-phase vreg/phys ceremony: one IR, phase invariants
asserted (below). The terminal invariant is total: after regalloc no
`OpId` remains anywhere in the stream — every value has become a
physical `LRegId`/`DstId`/`VarId`, and verify asserts the absence.

## `TTOp`

One enum, both levels mixed — high-level vendor-wrapper equivalents and
raw Tensix/SFPU instructions. Each variant is one emitted line (or, after
decomposition, one instruction). Every variant carries:

- typed operands (`OpId`/`LRegId`/... — never formatted strings; the
  scalar expression `(idx*elem)/4096` is a small operand tree over typed
  ids, rendered by the backend),
- a **static signature**: which CB it moves (produce/consume, n tiles),
  DST lock effects, arg reads, which thread (BRISC/NCRISC/TRISC) executes
  it. The signature is what makes mixed levels safe: `CopyTile` and its
  `Unpacr` expansion carry the same signature, so level swaps are sound
  by construction and the verifier runs over any mix.

No catch-all `Raw(String)` variant. Anything opaque is an explicit
variant (including `Asm` with a typed operand list), or the verifier is
blind exactly where it matters.

Variant families (non-exhaustive, provenance-tagged at definition):

- **Structure**: `EndReader`, `EndCompute`, `EndWriter`, `Loop`/`EndLoop`,
  `If`/`EndIf`.
- **Scalar C**: `SArg`, `SConst`, `SAdd`/`SMul`/`SDiv`/..., `SDeclare`,
  `SNocAddr` (`get_noc_addr` line), `TensorAccessor` declaration/chaining.
  These are *rendering*, not duplication: kernel IR does all scalar SSA
  work before lowering; scalar TTIR variants are dumb statements with no
  optimization passes over them.
- **CB traffic**: `ReserveBack`, `PushBack`, `WaitFront`, `PopFront`,
  CB declaration.
- **NOC**: `AsyncRead`, `AsyncWrite`, `NocReadBarrier`, `NocWriteBarrier`
  — real firmware calls, stay C forever.
- **Tile movement**: `CopyTile`, `PackTile`, and their decompositions
  (unpacker config + `Unpacr`, packer config + `Pacr`).
- **DST state**: `Acquire`, `Commit`, `Wait`, `Release`.
- **Format**: `CopyInitWithDt`, `PackReconfig`, and post-decomposition
  unpacker/packer config instructions.
- **SFPU**: high level (`Binary { z, x, y, bop }`, `Unary`, `Cast`,
  `Mad`) and low level (`SfpLoad`, `SfpStore`, `SfpMad`, `SfpShft`,
  `SfpAnd`, `SfpCast`, imm loads, `SfpNop`).
- **FPU**: matmul/reduce (vendor calls; last to decompose — MVMUL
  encoding and accumulator semantics are the hardest and least urgent).
- **Asm**: typed-operand escape hatch.

### Opcode provenance

Every low-level opcode names its evidence at definition: device-validated
(draft-tested), same-encoding-unverified (flagged, first device use must
re-check), or `todo!()` (never guessed). The dequant draft
(`dequant_q4k_tt_asm`) established: `TT_SFPLOAD`/`TT_SFPSTORE`,
`TT_SFPMAD`, `TT_SFPSHFT`, `TT_SFPAND`, `TT_SFPCAST(..INT32_TO_FP32_RNE)`,
`TT_SFPNOP`, `_sfpu_load_imm32_` (raw 32 bits — floats as `to_bits()`),
and the walk helpers. Derived only from those: Mul/Add/Sub/Neg via
`SFPMAD` + imm loads. Everything else enters the table only with
evidence.

## Pipeline

Fixed order, one pass per concern, each pass TT-specific:

1. **Convert + replicate_ops_per_section** — 1:1 kernel-IR → TTOp (`Op::Binary { x, y,
   bop }` → `TTOp::Binary { z: OpId, x, y, bop }`), then
   replicate_ops_per_section: values consumed in multiple sections are duplicated
   per section (each section is a separate kernel with its own registers
   and runtime args; arg ordinals stay global so the launch contract
   survives). CBs are the exception: one hardware object, one `CBId`,
   never duplicated. This replaces the three-phase `get_needed_ops`
   closure machinery with one explicit pass.
   Division of labor: **SSA dedups globally, physical duplicates
   locally.**
2. **Sync insertion** — `ReserveBack`/`WaitFront`/`PushBack`/`PopFront`
   per traffic op (FIFO state machine, the `CBEmitter::CBState` logic as
   a pass).
3. **Lock insertion** — `Acquire` placed directly at the LICM position:
   the preheader of the outermost loop containing a pack. TTIR's
   `Loop`/`EndLoop` mirror the kernel IR's loops 1:1 at conversion, so
   the lowering derives the same placement LICM would compute — no hoist
   pass, no effect ops in the kernel IR. `Commit` before pack, deferred
   `Release` across pack cones. (Why not a kernel IR op: the acquire is
   operand-free, so LICM would trivially hoist it maximally every time —
   the result is a fixed derivable position; and adding an effect op to
   the SSA IR means auditing CSE (must never merge two identical
   acquires straddling a release), DCE, and every restructuring pass, in
   the shared IR, for zero searchable freedom. The release is caused by
   pack and lives in TTIR either way.)
4. **Reconfig/init insertion** — naive: full reconfig before *every*
   copy/pack. Correctness first, redundancy is safe.
5. **Hoist + dedup** — adjacent same-config reconfigs collapse; effect
   ops hoist out of constant-trip loops with trip-count accounting
   (FIFO depth × trip). Locks are *not* hoisted here — they were placed
   at the LICM position during insertion (step 3); hoisting them would
   duplicate kernel LICM for a derivable placement. This is NOT kernel
   LICM — it hoists effect ops with hardware semantics the kernel IR
   cannot express. Runs on the fully-placed stream: place first, then
   clean.
6. **Regalloc + coalesce** — late. Linear scan over the linear stream;
   `OpId` → `LRegId`/`DstId`/`VarId`; coalescing produces the in-place
   forms (dst merged into a dead operand). Liveness from single-def
   chains (the pre-rewrite invariant), not a solver. After this pass no
   `OpId` remains in the stream.
7. **Verify** — structural, on the fully-physical stream: budgets per id
   class, CB balance and traffic correspondence, lock pairing, walk
   completeness, section termination, and the replicate_ops_per_section invariant
   (every value consumed in a section is defined in that section).
8. **Render** — C++ vehicle (TensorAccessor, NOC calls, `get_arg_val`,
   scalar C stay C; TTI words via the existing volatile-write macros).
   Rendering is a table walk, not synthesis.

Decomposition (high-level variant → instruction sequences) is an in-place
replacement pass that can slot anywhere after step 1; lazily per variant
as the encoding table grows.

### Pass rules

- **Rebuild, don't splice**: each pass drains the input `Vec` and pushes
  into a fresh one — O(n) per pass total, trivially snapshotable for
  `ZYX_DEBUG` dumps between passes. Neighbor pattern-matching on
  consecutive indices. Streams are hundreds–thousands of ops; every pass
  is microseconds (30µs budget applies).
- **Phase invariants, asserted**: all-`OpId` (post-conversion) → mixed
  (mid-regalloc, transient) → all-physical, zero `OpId`s (verify +
  render). Each pass states which phase it expects and panics otherwise.
- **Loud failure at the exact op**: a pass that cannot place an event
  names the op and the reason. Verify guards the composition — whatever
  the passes did, the final stream must be launchable. Pass order and
  verifier independence is what makes each pass safely fixable.

## What deliberately does not enter TTIR

- SSA machinery of any kind (see the boundary test above).
- **Fused-composite matching.** `FusedKind` (sigmoid/silu → one vendor
  call) dies because it was a wrapper-level artifact: at instruction
  level the chain *is* the fusion — exp/add/reciprocal sequences share
  LREGs inside one walk, and the register allocator subsumes the
  containment analysis. Cost moves to the leaves: exp/reciprocal/div
  microcode becomes provenance-tagged table entries with device tests.
  Numerics become ours (same accuracy class, different rounding) —
  tests compare against a reference model with tolerance, never against
  the vendor call's exact bits.
- **A fusion pass over TTIR.** `SfpMul`+`SfpAdd`→`SfpMad` duplicates
  `opt_fuse_mad`, which already ran in the kernel IR; the stream receives
  `Mad` directly. Physical-only peepholes (e.g. two `SfpLoad`s of the
  same slot) are permitted but expected to be rare.
- Scalar optimization passes.

## Migration

- **Differential testing**: render the TTIR stream and diff
  byte-for-byte against legacy output for a corpus of kernels (the
  existing TT test suite is the corpus). Land section by section with
  the diff as the gate; drift is a compile-time bug, not a board event.
- **Gradual per-op replacement**: steps emit high-level variants, which
  are lowered step by step by in-place replacement. `Vec` is fast enough
  (rebuild-per-pass makes it O(n) regardless).
- **Validation ladder** for new opcodes: (a) 1-input/1-output draft
  shape, (b) multi-input single-walk chain (the one deviation from the
  draft: multiple fixed-offset `ADDR_MOD_3` loads per row), (c) derived
  ALU ops, (d) per-opcode growth, each with a device test before it
  becomes default.
- Compile-time traffic-correspondence checks (the launch policy) move
  from legacy codegen invariants into `verify` — same guarantees, now
  structural.

## What gets deleted from tenstorrent.rs

| legacy piece | fate |
|---|---|
| `TileEmitter` + `Cfg`/`PlacedInit` backward placement pass | deleted — naive insert + dedup + hoist replaces optimal-once placement |
| `CBBatch` anchor machinery | deleted — sync insertion pass + trip-count hoist |
| `get_needed_ops` closures (3 phases) | deleted — convert + replicate_ops_per_section pass |
| `FusedKind`/`FusedPat` prepass | deleted — instruction-level fusion is free |
| unpack/pack format trackers | become reconfig-insertion pass state, back-edge modeled properly (fixes the known mixed-format loop-body limitation) |
| string emission | deleted — render is a table walk |

Kernel IR, the section barriers, and the launch contract are untouched.
The egraph and kernel IR stay generic — nothing Tenstorrent leaks into
them, and every other backend keeps the kernel IR exactly as it is.
