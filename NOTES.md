## HANDOFF (2026-11-03) — elementwise parity: sigmoid/exp2/log2 at 1 load + 1 store

**Tree state:** uncommitted changes in `zyx/src/kernel/tenstorrent.rs` (the two
fusion fixes below + `tt_log2`'s specials and its new polynomial),
`zyx/src/kernel/autotune.rs` (`unfold_pows` moved into `device_epilogue`, whole
`default_epilogue` before `tt_dataflow_select`), `zyx/tests/23_tt_lowering.rs`
(the four activation gates) and this file. 21 of 22 ttsim gates green (`elu` is
the red bug doc below), `cargo test --lib` = 82, changed files fmt clean.
Committing partial state is allowed (the user squashes later).

### The result

These are tensor-graph kernels (no hand-written programs): one tile in, one tile
out, everything else in `L0..L7`.

| kernel | replays | loads | stores |
|---|---|---|---|
| `sigmoid` | **1** | **1** | **1** |
| `exp2` | **1** | **1** | **1** |
| `log2` | **1** | **1** | **1** |
| `sin_cos` | 1 per kernel | 3 | 3 |

Handwritten reference (metalium-rs): `sigmoid_program` 2 loads + 2 stores (it
spills `x`), `relu`'s floor is 1 + 1. `exp2` and `log2` now hit that floor, with
no `tt_lreg_alloc` split. The `sin_cos` *test* compiles **two kernels**
(`xs.sin()`, `xs.cos()`), so its 6/6 is 3/3 each, with two register splits per
kernel -- also pressure, not fusion.

### What made the difference, in one line each

1. **Width, not the allocator** (`sigmoid`). The five `Select`s in `tt_exp2` (two
   clamps + the denormal/overflow/NaN specials) kept 9 values live at a point;
   removing them (the ML approximation the user asked for) put the fused body
   inside eight registers, so `tt_lreg_alloc` splits nothing.
2. **The fusion's two rules were the `exp2` blocker**, not pressure: its body
   peaked at 5 registers against a budget of 8, yet the chain came out as two
   row bodies, because the kernel *input* is read both before and after the
   rounding (`t = x + magic`, `r = x - nf`) and its unpack `Copy` was emitted
   between the two bodies -- and a cone's gap may only hold `SfpConst`/`SFPENCC`.
   Fix (a): the segment-CSE carry guard now asks `tt_flag_effect(..) != Keep`
   instead of "is it a const", so a preceding `SFPLOAD` no longer vetoes a carry
   (`exp2` 4 -> 3 loads). Fix (b): a whole-tile `Unpack` in a gap is **hoisted in
   front of the merged body** (it runs once per tile there, not per row) when no
   part of the cone writes the tile it fills or its circular buffer; that closes
   the cone (`exp2` 2 replays 4/2 -> 1 replay 1/1).
3. **`tt_log2` was rebuilt on tt-metal's own algorithm.** First its specials went
   (the `tiny`/`neg`/`inf`/NaN `Select`s, like `tt_exp2`'s): 2 replays 7/6 ->
   1 replay 2/2. The 2/2 was one *register* split from the two `sqrt 2`
   `Select`s that the atanh series needed. Then the series itself was replaced by
   the LLK's `calculate_log_body` (Norbert Juffa's `logf`): `bits(x) -
   bits(0.75)` with the mantissa cleared gives the exponent as `k << 23`,
   `bits(x) - e` gives `m in [0.75, 1.5)`, one degree-9 Horner chain in
   `m - 1` gives `log1p`, and `log2 x = k + log2(m)` folds the `ln2`/`log2 e`
   round trip into one multiply. No `Select` at all, fewer live values:
   **1 replay, 1 load, 1 store**. Accuracy is the LLK's (measured worst
   **1.4e-7** on the gate's metric over `x in [0.01, 100]`).
   Trap for next time: `BOp::And` is the **logical** and and types its result
   `Bool`; the bitwise mask is `BOp::BitAnd` (both lower to `INT_AND`).

### Activations are composed, and they fuse (new gates)

Four new gates in `23_tt_lowering`, on `[-4, 4]` input: `tanh`, `silu`
(`x*sigmoid(x)`), `gelu` (the tanh form) and `elu`. The frontend composes every
activation from primitives, so these test the *fuser*, not a hand-written
program:

| gate | kernels | replays | loads | stores | body |
|---|---|---|---|---|---|
| `tanh` | 1 | 1 | 1 | 1 | 230 words |
| `silu` | 1 | 1 | 1 | 1 | 220 words |
| `gelu` | 1 | 1 | 1 | 1 | 272 words |
| `elu` | -- | -- | -- | -- | **blocked** |

`silu` is the interesting one: `x` is used twice (once directly, once through
the sigmoid), and it is still 1+1 -- the second use is a register carry, which
is the segment CSE doing its job.

`elu` does not lower yet. `Tensor::elu` goes through `Tensor::relu`, which is
`cmpgt(0) * x`, so the graph contains `f32(bool_tile)`. By the time it reaches
`tt_lower_value` the Bool has been materialised in an F32 tile, so the `Cast`
resolves to `F32 -> F32` and hits the `todo!()` in the `Op::Cast` arm
(`tenstorrent.rs:1969`, "cast F32 -> F32 has no native row-op program").

**No optimizer can remove it**: the IR cast is `Bool -> F32`, and `F32 -> F32` is
only what `tt_resolve_dst` reports for the *tile* the Bool already lives in
(comparisons write 0.0/1.0 into an F32 DST tile). Running `default_epilogue`
before `tt_dataflow_select` (which the TT path now does) leaves it untouched --
verified. The fix is in the lowering: a same-tile-dtype cast is a remap, and the
`Bool` case should be named explicitly. The gate is left **red on purpose** as the
bug doc.

### The pipeline as it stands (`device_epilogue`, `autotune.rs`)

```
unfold_pows                     (in device_epilogue, for every backend)
tt_tile -> tt_unfold -> tt_reindex -> default_epilogue -> tt_dataflow_select
  -> tt_sfpu_fusion -> tt_lreg_alloc -> tt_dst_reuse -> tt_reg_sync -> verify
```

- **`unfold_pows` is a lowering decision, so it lives in `device_epilogue`**,
  once, for every backend (TT's `tt_unfold` then turns the resulting
  `Exp2`/`Log2` into primitives; elsewhere `devectorize` follows). It is gone
  from `default_epilogue`, which is now device-independent -- no exp conversion,
  and it can be run many times during the search without re-expanding `Pow`.
- **`tt_dataflow_select` gets the minimal IR**: the TT path runs the whole
  `default_epilogue` (constant folding, LICM, algebraic simplification, CSE,
  DCE) right before it, not just CSE + DCE.

- **`tt_dataflow_select`** is the phase boundary/emitter driver: pattern table ->
  `tt_sfp_*` emitters -> one `Replay` body per op.
- **`tt_sfpu_fusion`** merges adjacent bodies into cones: every register of every
  merged unit becomes a **virtual name (`>= 16`, `u16` space, never `8..15`)**;
  an interior value crosses on one `SFPMOV`; the producer's store becomes
  `SFPMOV carry <- reg` **at the store's position**; segment-aware tile CSE; a
  demotion retry (budget 2) when a cone would not fit. The cone's feasibility
  uses `tt_plan_splits` (same model as the allocator).
- **`tt_lreg_alloc`** = `tt_assign_regs` (interval scan, reuse only) + the
  **split-at-pressure fixpoint**: find the over-budget run (8 registers, 16 for
  16-bit tiles), split one value around it (store before, reload after), rerun.
  `tt_split_at_pressure` is mask-aware via `TtFlags`/`tt_mask_states`.
- **`tt_dst_reuse`** allocates the 8 DST rows and now **asserts** when the file
  is full instead of silently allocating a 9th row.

### Load-bearing invariants and bugs (do not undo)

Allocator/split (each cost a bug this session):

1. The split value must be **live at the peak**; dead values have huge windows and
   loop forever (observed: 33 splits, pressure rising).
2. Not read **or written** inside the window.
3. **Defined under a full mask** (lane registers are lane-masked; stale lanes are
   copied by a full-mask store and handed to consumers that never saw them).
4. The store must be **strictly before the peak** (a store at the peak keeps the
   value live there; the fixpoint loops).
5. Both window ends full-mask (`SFPENCC` reset, `SFPSETCC`/`SFPCOMPC` mask,
   `SFPPUSHC`/`SFPPOPC` stack).
6. The DST file is a second budget; `tt_dst_reuse` over-allocating silently was
   the "every value wrong" corruption.

Fusion (from earlier sessions, still load-bearing): never virtualize `8..15`;
carry copy at the store's position; per-part `SFPENCC` **inside** the body;
`SFPENCC` is a predicate reset, not a scope; hand-over only when formats allow;
coalescing must not drop an `SFPMOV` whose `Mod1` is set (log2 was off by `-2e`).

Fusion rules (this session, both are *semantic*, not tuning):

7. **The carry guard is about the mask, not about op kinds.** Only `SFPENCC`
   (`Reset`), `SFPSETCC`/`SFPCOMPC` (`Mask`) and `SFPPUSHC`/`SFPPOPC`
   (`Push`/`Pop`) change the lane mask -- `tt_flag_effect` is the one place that
   knows this. `SFPLOAD`/`SFPSTORE`/arithmetic are `Keep`; treating them as
   flag-touching is what made `exp2` reload `nf`.
8. **A tile move cannot be inlined into a row body, but it can be hoisted out of
   one.** The gap rule (`SfpConst` + `SFPENCC` only) exists because gap ops are
   inlined and a merged body runs once per row: a `Copy` moved inside would run
   once per row and clobber the tile mid-body. Hoisting it *in front of* the
   merged body keeps it once per tile and is sound exactly when no part of the
   cone writes the tile it fills (or its circular buffer); the hoisted op has no
   lane registers, so the body's remap must skip it. Only `Unpack` is eligible
   (a `Pack` writes a circular buffer the cone reads).

### Dead ends (measured, do not retry)

- **Scheduling the bodies** (per instruction, or per `PUSHC..POPC` scope): the
  peak sits inside a conditional and cannot be moved out; moving a scope is
  *unsafe* because a conditional write only updates enabled lanes (order is
  non-commutative in a way a read/write-set scheduler cannot model). Gate-broke.
- **Cheapest / use-count spill eviction**: cascades (dead value chosen, no drop).
- **Cone shrinking instead of splitting**: `sigmoid` 3 replays / `log2` 4 replays
  both miscompile -- do not use "make the cone smaller" as a pressure fix.
- **Scheduling in `instr_sched`** (high-level topological order) does not reach
  the lowered body's pressure.

### Open work, in order

1. **`elu` is blocked by a no-op `Cast`** (see the activation table above):
   `tt_lower_value`'s `Op::Cast` arm has no `F32 -> F32` case, and the frontend's
   `relu` (`cmpgt(0) * x`) produces exactly that after the Bool is materialised in
   an F32 tile. Tiny fix, but it blocks `elu`, `relu`, `leaky_relu`, `selu` and
   anything else composing `relu`.
2. **`sin_cos`'s cost is pressure, not fusion.** (Corrected -- the old note said
   "2 replays"; it is **two kernels**, `xs.sin()` and `xs.cos()`, one replay
   each.) Each kernel has **two register splits** (3 loads / 3 stores): e.g. the
   sin kernel stores tiles 841 and 656 at body ops ~217/223 and reloads them at
   ~314/316. Its two `Select`s are the quadrant/sign choice (`q & 1` picks the
   core, `q >> 1` flips the sign), so the `log2` trick does not apply. Next
   diagnosis: which values keep the peak over 8 (two-limb pi/2 + quadrant i32 +
   Taylor core), from the planner's peak, not a guess. The `log2` rewrite is the
   template if the polynomial can be re-based to a narrower range.
3. **`tt_sink_const_splats` has a precision bug**: with it, `exp2` is ~5e-5
   relative; disabled, the same kernel is ~1e-9. Fine for ML, but it is a real
   defect (suspect: the word-count/`len` bookkeeping when a 1-word load becomes a
   2-word `SfpConst`, or the constant path). The `exp2` gate was relaxed from
   `1e-5` to `1e-4` with a comment; do not leave it there once the sink is fixed.
4. **The acceptance gate**: for a pure elementwise kernel the compiled T1 stream
   must have one `SFPLOAD` + one `SFPSTORE` per row (`sigmoid`, `exp2` and `log2`
   already are). Needs a test-visible accessor for the compiled stream (the user
   approved adding one).

Done this session: the two fusion fixes, `exp2` 1/1/1, `log2`'s specials *and*
its polynomial (1/1/1), the `sin_cos` correction above, and the four activation
gates (`tanh`/`silu`/`gelu` 1/1/1, `elu` blocked by the no-op cast).

### How to run things

```bash
cd zyx && cargo build
cd zyx && AGENT=1 cargo test --lib
cd zyx && AGENT=1 cargo test --test 23_tt_lowering --no-run
cd zyx && BIN=$(ls -t target/debug/deps/23_tt_lowering-* | grep -v '\.d$' | head -1)
# one gate at a time (a failed ttsim thread poisons later ones):
TT_TTSIM_LIB=/home/x/Dev/rust/metalium-rs/vendor/libttsim_bh.so AGENT=1 $BIN sigmoid --exact
```

DST traffic per row is measured from the `ZYX_DEBUG=264` dump: count `^ sfp_load`
/ `^ sfp_store` / `^ replay`. That dump is **post-epilogue**, so it is the real
program.

## TARGET DESIGN — the register/memory hierarchy and the pass pipeline

**This is the architecture the user wants. It supersedes the "one row loop per op,
then fuse pairs, then reclaim" shape that the code has today.** The rule it comes
from: *lowering does the simplest possible thing and allocates nothing; every
register, tile and buffer decision is a pass.* "Fusion" is not a concept in this
design — a value that fits in a lane register simply never needs the tile that
lowering gave it, and the tile allocator drops it.

**The hierarchy (what is actually shared with what):**

```
LREG   8 general-purpose lane registers      SFPU only (L0..L7; L9 = 0, L10 = 1,
                                             L8 = 0.8373, L15 = 2*lane are fixed
                                             and NOT writable: hardware silently
                                             drops writes outside L0..L7)
DST    8 tiles (32-bit) / 16 (16-bit)        ONE resource with three clients:
                                             SFPU operands+results, the Matrix
                                             Unit's accumulator, the packer's source
SrcA   2 banks x 1 tile                      Matrix Unit input only, unpacker
SrcB   2 banks x 1 tile                      writes it, SFPU cannot address it
L1     circular buffers                      unpacker reads, packer writes
DRAM
```

- **The Matrix Unit does not read DST and does not read L1.** `UNPACR` fills
  `SrcA`/`SrcB` from a CB, `MVMUL` reads them and accumulates into DST, the packer
  takes DST to a CB (`tenstorrent.rs:2880-2935`).
- **`SrcA`/`SrcB` are a hand-off pipe, not storage.** A bank is owned by the
  unpacker or the Matrix Unit; ownership moves only with `UNPACR`'s `FlipSrc` and
  `MVMUL`/`SETRWC`'s `FlipSrcA/B`, and breaking the lockstep makes the hardware
  *wait*, not fault (`tt-isa/src/matrix.rs`). So they are never a spill target;
  the compiler's only obligation is lockstep and, later, double-buffering the two
  banks.
- **DST is the shared file** — that is why DST pressure is the real constraint,
  and why the matmul accumulator pins a tile for the whole matmul.
- The accumulator is *not* materialised even with several uses: every consumer
  reads its DST tile in place (`tenstorrent.rs:2920-2935`).

**Spill direction (the user's rule):** `lreg -> DST`, `DST -> L1`. An lreg value
is **never** pushed down to L1 and reloaded, because coming back costs an
L1→DST unpack on T0 *plus* a DST→lreg load on T1, i.e. two role crossings and two
seams for one value. L1 is in the picture only when a value genuinely has to
survive a role boundary (that is `tt_materialize`'s decision today).

**The pipeline:**

```
zyx IR
 1. lower        one TT op per IR op, FLAT. Virtual lane registers (unbounded),
                 one virtual DST tile per value. No replay marker, no grouping,
                 no reuse, no sync. Allocation = none.
 2. lreg alloc   liveness over the flat stream, assign L0..L7, coalesce
                 (no move when the producer's register is free), spill past 8
                 into a DST tile (SFPSTORE/SFPLOAD inside the row body).
 3. dst alloc    the 8 tiles: disjoint live ranges share; spill past 8.
                 (Today's `tt_dst_reuse`, which already renames logical
                 `Storage` op ids to physical tiles — the tile side is already
                 mostly LLVM-shaped; the round-trip reclamation in it is what
                 "fusion" degenerates to once lregs can stay live.)
 4. cb alloc     circular buffers for the cross-role values.
 5. loop form    post-RA, so minimal: the flat stream is already row-shaped
                 (every op is `at: 0`, the DST row counter is implicit). This
                 pass only picks the cut points -- where one `REPLAY` ends and
                 the next begins -- and the `AddrMod` on each body's last DST
                 access. Nothing is hoisted: row-invariant constants are 1-word
                 `SfpConst` in the body and address math is already outside the
                 loop. If this pass ever needs to hoist or rediscover a loop,
                 the passes before it are wrong.
 6. sync         semaphores (`tt_reg_sync`, exists).
```

**Acceptance test (the reason for all of this):** for a pure elementwise kernel
(`sigmoid`, `elu`, `silu`, `gelu`, …) the emitted T1 program must be, per row,
**one `SFPLOAD` from the input tile and one `SFPSTORE` to the output tile**, with
everything else in `L0..L7` — two DST accesses, one tile in and one out. This is
the user's version of geohot's "`self.elu()` must cost the same as `self.relu()`".
**Measured 2026-11-03: met for `sigmoid` and `exp2` (both 1 load + 1 store per
row); `log2` is 2/2 (one register split from its two sqrt-2 selects) and
`sin_cos` 6/6 (two replays, its selects are the quadrant choice).** It is still
not a *test*: the acceptance gate (an accessor for the compiled T1 stream) is
open.

**Where the code is now (2026-11-03):**

- step 1 (lowering, one `Replay` body per IR op) + step 5 (grouping) exist as
  `tt_dataflow_select` -> `tt_sfpu_fusion`; fusion emits virtual lane registers.
- step 2 (lreg alloc) is `tt_lreg_alloc`: interval scan + split-at-pressure,
  8/16 by tile width.
- step 3 (dst alloc) is `tt_dst_reuse` (plus `tt_reclaim_round_trips`), step 4
  `tt_new_cb`, step 6 `tt_reg_sync`.
- Register numbering: `0..7` physical, `8..15` the hardware's fixed constants
  (never remapped), `>= 16` virtual until allocation; `tt_lreg_alloc` renames
  every virtual before codegen.

## HANDOFF (2026-11-02, seventh pass) — virtual registers are IN, all gates green

**Tree state:** uncommitted work in `zyx/src/kernel/tenstorrent.rs`,
`zyx/src/kernel/autotune.rs`, `NOTES.md`, `AGENTS.md`. 18/18 ttsim gates green,
`cargo test --lib` = 78, fmt clean. The user commits; nothing is committed by the
agent.

### What is in place

1. **`tt_sfpu_fusion` emits virtual registers.** Every register of every merged
   unit becomes a fresh virtual (`>= 16`); `8..15` are the hardware's read-only
   constants (`LREG_ZERO` = 9, `LREG_ONE` = 10, ...) and are never touched.
   An interior value crosses a body on one `SFPMOV`; the producer's store is
   replaced by `SFPMOV carry <- reg` **at the store's position** (not read at
   body end), the consumer's load by `SFPMOV dst <- carry`.
2. **`tt_lreg_alloc`** (registered right after the fusion in
   `device_epilogue`, before `tt_dst_reuse`) linear-scans `(def, last_use)` per
   virtual into `L0..L7` on each straight-line row body, drops the coalesced
   moves, and recomputes each `Replay`'s length.
3. **`tt_alloc_regs`** is one free function used twice: the fusion builds the
   body it is about to emit in virtual space (`shadow`) and forms a shorter cone
   when it would not fit; the allocator re-runs it on the real body. Nine live
   values is still a hard error -- **spilling is the next step** (lreg spills to
   a DST tile, never to L1; the tile allocator already runs after this pass).
4. **The second `tt_sfpu_fusion()` call in `device_epilogue` is gone** (fusion
   now leaves virtuals, which must be allocated exactly once).

### Measured (DST accesses per row -- the metric that matters)

| kernel | replays | loads | stores | committed pass |
|---|---|---|---|---|
| `sigmoid` | 7 | **25** | **10** | 21 / 36 / 21 |
| `exp2` | 6 | 22 | 7 | 17 / 32 / 17 |
| `log2` | 16 | 39 | 19 | 31 / 51 / 31 |
| `sin_cos` | 18 | 59 | 27 | 18 / 34 / 18 |

`sigmoid`'s stores are **halved** and loads down a third: that is the
"one tile in, one tile out" direction. `sin_cos` went *up* -- it now keeps
values in registers instead of fusing units, so check it before claiming a
uniform win.

### Seven bugs found this session (all load-bearing -- do not undo)

1. **`8..15` must never be virtualized** (`LREG_ZERO`/`LREG_ONE` appear in a
   body's read mask; renaming them makes every `1.0` read an unwritten
   register).
2. **Virtual numbers must not restart at 16 per merge** (a body an earlier merge
   already virtualized keeps its numbers).
3. **The carry copy belongs at the store's position**, not at body end: a unit
   may rewrite that register after the store.
4. **A per-part `SFPENCC` reset must be inserted *inside* the body** (before its
   first op, not before the marker): its word is counted in `len`, and a reset
   parked outside leaves the body one word short of its own length, so the
   replay never closes.
5. **`SFPENCC` is a predicate reset, not a scope** (`mod1 2 | 8` sets
   `use_flags` and every `lane_flags = true`, `kern/sfpu/interp.rs:766`), so
   each interleaved part needs one.
6. **A hand-over is only bit-equal when the store and every load use the same
   32-bit format** (`INT32`/`FP32` convert through a tile); and a virtual read
   before its first write is live from the body's start.
7. **Coalescing must not drop a `SFPMOV` whose `Mod1` is set.** `SFPMOV x -> x`
   with `Mod1 = 0` is a no-op, but `Mod1 = 1` is the *negate*. Dropping it made
   `cast(i32 -> f32)` return `|e|`, which is why `log2` was wrong by exactly
   `-2e` for negative exponents (`0.5 -> 1` instead of `-1`, `0.3 -> 2.263`
   instead of `-1.737`), while `e >= 0` and every `big`-mantissa case looked
   fine. This was the last red gate; the fix is in `tt_lreg_alloc`'s drop test
   (`insn.args[2] == 0`).

### Debugging recipe that worked (re-add these; they were removed)

- **Bisect merges, not cones**: a counter in `tt_sfpu_fusion` that returns after
  N successful `tt_fuse_cone` calls. `log2` was fine for N<=5 and wrong from
  N=6 -- that pinned it to one merge, not to "big bodies".
- **Bisect body size**: refuse a cone whose `total` exceeds a cap. 24 passed,
  25 failed. (Both knobs are temporary; delete them again.)
- **Dump the *applied* body**, not a candidate: print inside the apply path
  (after the feasibility check), with `maps` applied, else you inspect a plan
  that was discarded.
- **Shrink the repro to values**: a 32x32 tile with hand-picked constants showed
  the error was `-2e` (value-dependent), not lane 27 (position) -- the
  `[10, 256]` gate's "one lane" was a red herring. `Tensor::from(vec)
  .reshape([10, 256])?` avoids the 10-grid limit.
- Coalescing, allocation and liveness are per-body; if a merged body looks
  right on inspection, suspect the *pass after it* (the allocator's peepholes)
  before the fusion.

### What is left, in order

1. **Spill instead of `todo!()`**: a body needing a ninth live value should
   store it to an ordinary `Storage{Register}` tile and reload before use (the
   user's rule: lreg spills to DST, never to L1; `tt_dst_reuse` runs after this
   pass, so a spill tile is allocated like any other value).
2. **The acceptance gate**: for a pure elementwise kernel the emitted T1 must be
   one `SFPLOAD` + one `SFPSTORE` per row. Needs a test-visible accessor for the
   compiled instruction stream (the user approved adding one).
3. **The virtual name space**: numbering is per `(unit, register)` and the mask
   is `u128` (so `16..127`, 112 names); a large cone can be cut by name
   exhaustion rather than by live count. The user's preference is a `Set`/`Map`
   instead of the mask (`device_epilogue` is NOT under the 30us budget --
   `AGENTS.md` was updated to say so). Compacting a cone's virtuals into a dense
   range after each merge is the cheap half of that fix.
4. `tt_sink_const_splats` + interior carrying are on by default; if a
   regression shows up, the two are separable (`ZYX_NOSINK`/`ZYX_NOINTERIOR`
   existed for exactly that bisection).

### Handoff honesty

The direction is right and the numbers are better, but this pass now owns a real
register allocator, so its failure modes are compiler failure modes (silent
miscompiles, not loud panics). Every fix above came from a *gate*; keep the
practice: after any change to fusion, allocation or `SFPENCC` handling, run all
18 gates one at a time, and re-measure the DST table above.

### Start here — session handoff (2026-11-02, fifth pass — end of session)

**Every transcendental is now IR: `tt_unfold` covers `Exp2`, `Log2`, `Sin` and
`Cos`, and `tt_sfp_sigmoid`/`tt_sfp_pow`/`tt_sfp_sin_cos` are gone.**

**State:** the tree is clean except for this session's work; the user commits.
Run cargo from `zyx/`.

**Gates — 18 green on ttsim (one at a time, `TT_TTSIM_LIB=…`):** `add`,
`unary_math`, `neg_sub_mul`, `shared_input`, `tiling`, `transfer_roundtrip`,
`matmul`, `sum`, `max`, `sum_scalar`, `max_scalar`, `sum_all`, `exp2`,
`sigmoid`, `sum_sigmoid`, `sin_cos`, `log2` (new), **`matmul_sigmoid` (now
green)**. `cargo test --lib` = **78**. No red gates.

**Added after the fifth-pass block (same day, sixth pass): `log2`, the integer
`Sub` arm, the constant-splat sink, and two fusion extensions.**

6. **Integer `Sub` has an arm** (`tt_sfp_int_sub`, `SFPIADD` with mod1 `2 | 4`
   = `VD = VC − VD`, after a `mov b -> L2`), and the `tt_lower_value` arm now
   covers `BOp::Add | BOp::Sub` for `I32` — the float arm's unpack refuses an
   `I32` tile, which is what broke `tt_log2`'s `e − 127`. Gate: `log2`
   (`x ∈ [0.01, 100]`, `1e-5 * max(|want|, 1)`), which also passes under
   `ZYX_DEBUG=264`.
7. **Constant splats are sunk into their readers** (`tt_sink_const_splats`,
   called at the top of `tt_sfpu_fusion`). `tt_fill_const` materialises every
   scalar constant as a whole DST tile: one `SfpConst` + 32 `SfpStore`s of the
   32x32 tile. On `sigmoid` that was 22 tiles and ~700 of the kernel's 734 DST
   stores, and it also *blocked* fusion (the splat sits between a producer's
   store and the consumer's load). The sink replaces each load of such a tile
   with the `SfpConst` it was filled from, adjusts the owning `Replay`'s `len`
   (a `SfpConst` is 2 words unless its low half is zero), and deletes the splat
   plus the tile. Conditions: the tile's only writer is one splat (`SfpConst` +
   exactly the 32 stores, ascending `at`, same register/format); nothing else
   names the slot (no pack/unpack, no `Sfpu` result); every reader is an
   `SfpLoad` of a **32-bit** format (`FP32` or `INT32` — both move the raw lane
   word, which is what a `Bitcast` of a constant asks for; a 16-bit load
   converts and is refused). Result on `sigmoid`: **734 → 21 stores, 67 → 36
   loads, 30 → 21 replay units**; `matmul_sigmoid`'s 9-tile peak is gone (it is
   green).
8. **`tt_sfpu_fusion` is general cone fusion now** (`tt_fuse_cone` replaced
   `tt_fuse_pair`). A *cone* is the reader-closure of a start unit: a value is
   *interior* when exactly one unit in the cone writes it and every read is in
   a later unit of the same cone, and then it never touches a tile -- the
   consumer's `SFPLOAD` becomes the `SFPMOV` that hands the lane register over
   (or a rename, when the value dies in that unit and the body never writes the
   loaded register). The closure means a multi-reader value is fine: the cone
   simply grows to include every reader. Nothing here knows what an activation
   is, which is the point -- `elu`/`silu`/`gelu` are composed at the frontend
   (`elementwise.rs:190/265`) and this is what has to make them cost the same
   as `relu`.

   Units after fusion (`cargo` gate, `--nocapture` count): `sin_cos` **1**,
   `sigmoid` **2**, `log2` **2**, `exp2` **3**, `matmul_sigmoid` **2**. Before
   the cone: 21, 21, 31, 17, 21. All 18 gates green, `cargo test --lib` = 78.

   The two things that were wrong on the way, both worth not re-learning:
   - **`SFPENCC` is a predicate reset, not a scope.** `mod1 2 | 8` sets
     `use_flags` for every lane and sets every `lane_flags[l] = true`
     (`kern/sfpu/interp.rs:766`). So a body's preamble resets the predicate, and
     a merged body that interleaves rows must reset again between its parts.
     The pass now puts one `SFPENCC` at the head of the merged body and keeps
     each gap's own copy in place. Hoisting the reset out of the body gave
     `-inf` in one lane of `log2`, and *dropping* it broke `sigmoid`/`exp2`.
   - **A hand-over is only bit-equal when the store and every load use the same
     32-bit format** (`INT32` and `FP32` convert on the way through a tile, so a
     `mov` across a mismatch is not the same value). The cone requires equality.
     Landing a value in the consumer's own register removes the load entirely
     (`drop`), not just rewrites it.
   - a merge may cross the 32-word replay buffer, but **not past `loops::MAX`
     (4) block repeats**: the count is every long body before the merge, minus
     the ones joined, plus one if the merged body is longer.

   **Register pressure is the limiter, and it is deliberately conservative.**
   A live value keeps its register; if the next unit writes that register the
   cone stops one unit short and is retried, so no value is ever guessed. I did
   implement a "move the live value to a free register first" variant and it
   failed every gate that cones deeply (`sigmoid`, `exp2`, `log2`) even with the
   `SFPENCC` fix in place -- the bug was not found, so it is *not* in the tree.
   That mechanism is what stands between `sigmoid` at 2 units and 1.
**This session, in order**

0. **`sin`/`cos` unfolded (last item of the port list).** `tt_unfold` now
   lowers `UOp::Sin`/`Cos` through `tt_sin_cos` + `tt_horner`:
   - `n = round(x·2/π)` by the magic constant (as `tt_exp2`'),
   - `r = x − n·PIO2_HI − n·PIO2_LO`, two-limb π/2 with the high part keeping
     12 mantissa bits (`to_bits() & !0xfff`), so `n·PIO2_HI` is exact for
     `|n| < 2^12`,
   - five-term Taylor cores for `sin r`/`cos r` (error < 1e-9 over `|r| ≤ π/4`),
   - quadrant from `q = n & 3`: low bit picks the core, `q >> 1` (for `cos`,
     `(q + 1) >> 1 & 1`) the sign. **`Select` needs a `Bool` condition** —
     verify rejects an `I32` one, so the quadrant bits are compared against 0.
   Replaces Payne–Hanek (≈180 instructions, and `SFPMUL24` is not expressible
   in IR): the body went from ≈225 words to ≈34. Exact to about 6400 rad;
   beyond that the two-limb reduction loses precision — the same trade every ML
   library makes (TF32 matmuls are coarser). Gate: `sin_cos` in
   `tests/23_tt_lowering.rs`, host tolerance `1e-5 * max(|want|, 1)`, `x ∈
   [-100, 100]`, and it also passes under `ZYX_DEBUG=264`.
   Deleted with it: `tt_sfp_sin_cos` (272 lines), its dispatch line,
   `UOp::Sin`/`Cos` in `uop_kind` (they now return `None`, i.e. loud), and
   `SIN`/`COS` from `sfp_row_ops`'s three kind lists.
1. **`SIGMOID`/`POW` emitters deleted** (proved dead with a `todo!("PROOF")`
   first): `tt_sfp_sigmoid` + `tt_sfp_pow` (≈470 lines), their dispatch
   branches, both `sigmoid_pat` rows and their `Pat` construction, and the two
   kinds from the `sfp_row_ops` lists. `Tensor::sigmoid` is `1/(exp(-x)+1)` at
   the tensor level, so no sigmoid op ever exists; `unfold_pows` in
   `default_epilogue` removes every `Pow` before lowering.
2. **`tt_dst_reuse` reclaims round trips** (`tt_reclaim_round_trips`), and
   `tt_sfpu_fusion` runs a second time after it. Conditions are exact: one pack
   + one unpack on the CB and no publish/drain; the unpack's row written only
   by the unlock; nothing writing the producer's row between the pack and the
   last consumer; and **the projected peak must still fit** — `tt_live_peak`
   simulates the row count the allocator will need (8/16 tiles by width), so a
   reclamation that would need one tile too many is skipped, never forced.
   Measured on `sigmoid`: `tt_copy` 14 → 4 (only the DRAM publish, input
   unpack, output pack, drain remain), seam hand-offs 23 → 2.
3. **Block repeats** (`mailbox::loops`): `tt_row_loop` always emits
   `TTOp::Replay { len, count }`, codegen picks the mechanism — ≤ 32 words is
   the hardware `REPLAY`, longer is a block repeat (body once + `[start, len,
   count]` in the program header). `TTOp::Math` carries
   `TTRoleProgram { words, loops }`; the backend hands them to `code::stored`.
   **Trap:** a loop `start` indexes the role program, whose preamble is
   `tt_math_role`'s state id **plus** its appended drain — `role.len() −
   body.len()` is 2, not 1 (`tt_math_role` now returns the preamble). Limit:
   `loops::MAX = 4` per role (loud compile error past it). Also mask
   `LOOPED`/`STREAMED` wherever a program length is used as a count (the
   launch check and the `ZYX_TT_DUMP` print both did).
4. Earlier this session: per-edge semaphores in `tt_reg_sync` (the `semget`
   underflow — ttsim prints the counter *value*, not the index), the missing
   packer→unpacker seam, the integer-`Add` arm, `tt_sfpu_fusion`.

**What is left**

1. `sigmoid` is 2 units (goal: 1). What blocks the last step is only register
   pressure: the live value's register is written by the next unit, and the
   fix -- move it to a free lane register before that unit -- was implemented
   and then removed because it failed the deep-cone gates. That is the single
   next task for fusion; everything else (multi-reader, gaps, formats,
   `SFPENCC`, block-repeat budget) is in place. `sin_cos` already reaches 1
   unit, so the mechanism is proven end to end.
2. `recip`/`div`/`sqrt`/`round` arms are program transcriptions that *could*
   become IR like sin/cos. `sqrt`'s seed is already integer magic
   (`0x5f3759df - (bits >> 1)`); `recip`/`div` need `SFPARECIP` replaced by the
   magic-constant seed (`0x7EF311C3 - bits`) plus one more Newton step. The user
   said these are optional ("only if you find it really easy"). The rest of the
   `tt_sfp_*` arms are primitives and stay.
3. The builder module (`kern/sfpu/ops.rs` 4882 lines + `interp.rs` 942) is the
   A/B oracle and the source of `kind_sfpu`'s numbering; deleting it needs
   `kind_sfpu` moved to `kern/kind.rs` first (needs the user's OK).
4. `Cast F32 → F32` (a no-op cast the graph leaves behind, e.g. in `relu`)
   hits `todo!()`; folding it in `tt_lower_value` would be a one-liner.
5. Once the DST picture is settled, the next milestone the user named is
   **running llama on Tenstorrent** — that is the reason this polishing matters.

**Traps from this session (do not re-learn)**

- ttsim's `tensix_semget: sem=N sem_max=15` prints the **counter value**, not
  the semaphore index.
- `SEMWAIT` is a level test, not a reservation: one counter per
  `(producer, consumer)` edge, never per direction.
- A materialised value's CB needs a seam between the pack (T2) and the unpack
  (T0); `tt_reg_sync` models CBs for exactly those two `Copy` kinds.
- `Select` requires a `Bool` condition in `verify`.
- `ZYX_DEBUG=264` (256 `NoSearch` + 8 IR) is the way to dump post-epilogue TT
  IR; `ZYX_DEBUG=8` alone prints nothing on the autotune path.
- Run the gates one at a time: the ttsim owner thread dies with a failed test
  and every later test in the process is a cascade.

### Earlier handoff (fourth pass, superseded by the block above)

**Both bugs fixed; `exp2`, `sigmoid` and `sum_sigmoid` are green,
`tt_sfpu_fusion` exists, the materialised CB round trips are reclaimed, and the
`SIGMOID`/`POW` emitters are deleted.**

0. **`SIGMOID` and `POW` emitters deleted.** Proved dead with the notes' own
   recipe first: both dispatch branches were replaced by
   `todo!("PROOF: ...")` and all 16 ttsim gates ran green — nothing reached
   them. Then removed:
   - `tt_sfp_sigmoid` and `tt_sfp_pow` (≈470 lines);
   - their `tt_sfp_sequence` branches;
   - both `sigmoid_pat` pattern rows *and* their `Pat` construction (they match
     a cone that cannot exist by then: `tt_unfold` has already turned `Exp2`
     into IR, and `Tensor::sigmoid` is `1/(exp(-x)+1)` at the tensor level, so
     there is no sigmoid op at all);
   - `SIGMOID`/`POW` from `sfp_row_ops`'s kind lists (`ported`, `unary`, and the
     A/B lists).
   `bop_kind`'s `BOp::Pow => sf::POW` mapping was left in place: it only names a
   kind, and `unfold_pows` runs in `default_epilogue` before lowering, so no
   `Pow` survives to use it. **`SIN`/`COS` are still live** — `tt_unfold` only
   handles `Exp2 | Log2`, so `tt_sfp_sin_cos` stays until the trigonometry
   program is transcribed into IR ops (deliberately left for later).

1. **`tt_dst_reuse` now reclaims round trips (`tt_reclaim_round_trips`), and
   `tt_sfpu_fusion` runs a second time after it.**
   A value with more than one use is materialised by lowering (pack its row,
   unpack it back for the consumers) only because the producer's row *might* be
   reused first. The pass deletes that pack and unpack when:
   - the CB is touched by exactly one pack and one unpack and nothing else
     (`Publish`/`Drain` excluded — that would be a kernel output);
   - the unpack's destination is written by the unpack alone (consumers only
     read, so moving them onto the producer's row is a pure rename);
   - nothing writes the producer's row between the pack and the last consumer;
   - the projected peak live tiles still fit the budget
     (`tt_live_peak(ops, Some((x, y)))`, which is the allocator's row count: a
     row frees once its last use is past; 8 for 32-bit tiles, 16 for 16-bit).

   A candidate that fails the budget is skipped (`tried`), not forced: the pass
   must never turn a working kernel into a `tt_verify` budget error. The CB and
   the now-unused unpack row are removed with it.

   Then the epilogue runs **`tt_sfpu_fusion` again**: with the round trip gone,
   the consumers read the producer's row directly, so the store→load edges it
   skips (`Pack`/`Unpack` in between) are now adjacent. Both passes only remove
   row ops, so the physical rows stay valid.

   Measured on `sigmoid` (one kernel): `tt_copy` **14 → 4** (only the DRAM
   publish, the input unpack, the output pack and the drain are left — every
   intermediate round trip is gone), seams **23 → 2** posts/takes, values still
   bit-identical, 15 gates green. `matmul_sigmoid` is still the pre-existing
   `9 DST register tiles exceed the 8-tile budget` — the reclamation cannot help
   there because its baseline peak is already over the budget.

2. **Block repeats are wired (`mailbox::loops`), and row loops no longer
   unroll.** `tt_row_loop` now always emits `TTOp::Replay { len, count }`; the
   *codegen* picks the mechanism from the length:
   - `len <= REPLAY_BUFFER (32)` — the hardware `REPLAY` record, unchanged.
   - longer — a **block repeat**: no record word, the body appears once in the
     role program, and a `[start, len, count]` entry goes in the program's loop
     header. `tt_isa::mailbox::loops` already had the format (`MAX = 4`, two
     deep, `count <= 128`), `code::stored(words, loops)` already produced the
     header, and `tt_isa::dataflow::LENGTH_MASK` already existed on the firmware
     side — the missing piece was plumbing.

   Plumbing, in order: `TTOp::Math` carries `TTRoleProgram { words, loops }`
   (one `Box`, so `Op` stays 24 bytes — the `const _` assert in `ops.rs` still
   fires); `TtSection::loops: [Vec<[u32; 3]>; 3]` collects them in
   `finish_replay`; `render` returns them, shifted by `tt_math_role`'s preamble
   (it now returns `(words, preamble)` — the trap below); `TTProgram.loops`
   carries them to `render_tt`; the backend turns them into `code::Loop` and
   hands them to `stored`. The launch's `ProgramTooLong` check masks the
   `LOOPED`/`STREAMED` bits now (a looped program's length word is
   `len | 1 << 31`).

   **Trap found while wiring it:** the loop `start` indexes the *role program
   the runner pushes*, which begins with `tt_math_role`'s preamble — and that
   function both prepends the state id **and appends an SFPU drain**, so
   `role.len() - body.len()` is 2, not 1. The first version shifted by 2, which
   moved `start` one word past the body: the loop then pushed
   `[load b, mad, store, SETRWC]` instead of `[load a, load b, mad, store]`, so
   the first operand was missing *and* the trailing clear reset the row counter
   every iteration. Symptom: the first row group right, the rest stale. Hence
   the explicit `preamble` return.

   Measured: `sigmoid` T1 **4551 → 2172 words** (2 block repeats), `unary_math`
   T1 58 words with its 43-word sqrt body as one block repeat, `exp2` T1 1930
   with one 34-word repeat. All 15 gates green, values unchanged.

   Limits/gotchas: at most 4 block repeats per role (`loops::MAX`, a loud
   compile error past it), `1 <= len < 4096`, `count <= 128`, and two loops may
   be nested at most once (the runner checks the table and fails loudly).

3. **`tt_sfpu_fusion` (new, `device_epilogue` between `tt_dataflow_select`
   and `tt_dst_reuse`).** DST only, exactly as settled: splice adjacent
   `Replay` units along `A`'s trailing `SfpStore slot S` → `B`'s leading
   `SfpLoad slot S`, keep the value in `A`'s result register, rename the
   loaded register onto it (`tt_row_rename` by ISA field name, guarded by the
   read/write sets from `tt_row_regs`). Conditions, all exact: same `slot`,
   `at`, `fmt` and repeat count; the slot has exactly **two** accesses in the
   whole kernel (so no other reader, no second writer, and never a kernel
   output); `B` never writes the loaded register and never reads it after
   first writing the result register; the spliced body stays ≤ 32 words (the
   unrolled fallback shape is not spliced yet). It re-scans and merges one
   pair per round, so chains fold. Measured on `sigmoid`: **35 → 28 replay
   bodies**, T1 4817 → 4551 words, all 15 gates unchanged.
   Deliberately *not* in this pass: anything CB-shaped. A `Pack`/`Unpack`
   means T2/T0, so those sequences are not adjacent and never fuse.

4. **The missing packer → unpacker seam (fixed).** `tt_reg_sync` only put
   `Register` storages in `touches`, so a `tt_materialize` round trip
   (DST → CB on the packer, CB → DST on the unpacker) had **no edge at all**:
   the unpack raced the pack and read whatever the CB held before. Symptom: the
   whole output tile was one constant that moved with the allocation
   (`sigmoid` 0.2, `exp2` 4.0), and a probe with `full(-200)` returned the
   answer for `+200` (the previous kernel's tile). Fix: the `Pack` arm records
   the CB as a write by role 2 and the `Unpack` arm the CB as a read by role 0,
   and the `touches` filter accepts `Register | Circular`; the existing epoch
   logic then emits the T2 → T0 data seam and its reuse seam. It was a real
   race, not a sim artifact — `y = x*2` used twice passed by luck (its unpack
   happened to run after the pack), any longer chain lost.
5. **Integer `Add` lowered through the F32 arm (fixed).**
   `Op::Binary { bop: Add }` was dispatched by `bop` alone, so `tt_exp2`'s
   `bits(d) + (n << 23)` became `SFPMAD`'s float add
   (`0x3f800000 + 0x00800000` = `1.0 + 1.2e-38`, a no-op) and every lane with
   `|x| >= 0.5` came out exactly half of `2^x` (507/1024 lanes). Fix:
   `tt_sfp_int_add` (`SFPMOV` then `SFPIADD Imm12 = 0`, `Mod1 = 4` so the lane
   flags are left alone, `SFPENCC` first like every other sequence), dispatched
   from `tt_lower_value` when `self.dtype(id) == DType::I32`; a scalar-constant
   operand is filled with `tt_fill_const` so it cannot fall back to the float
   path. New unit test `int_add_wraps_like_the_host` (interp A/B, wrap
   included).

**Gates:** 15 green on ttsim, one at a time — `add`, `unary_math`,
`neg_sub_mul`, `shared_input`, `tiling`, `transfer_roundtrip`, `matmul`, `sum`,
`max`, `sum_scalar`, `max_scalar`, `sum_all`, `exp2` (un-ignored, now a real
gate), `sigmoid` (0 bad / 2560, `worst 0` — bit-exact against the host),
`sum_sigmoid`. `cargo test --lib` = **78**. **Red:** `matmul_sigmoid`, still
`9 DST register tiles exceed the 8-tile budget` (allocator peak, unrelated to
the two bugs; it also hits ttsim's `SRCA_SET_SetOvrdWithAddr=1` limitation).

**Next (in order):**

1. **`tt_dst_reuse`'s materialisation decision** — the settled design. It is
   *not* "delete the first unpack": the executable form is "do not
   materialise when the row can stay live to the last consumer", i.e. drop the
   pack and the (single, cached) unpack. Note the tiering changed on
   measurement: the "dead round trip" tier I first proposed is nearly vacuous,
   because `tt_resolve_dst`'s per-CB cache already leaves **one unpack per
   CB** for all consumers — so one pack + one unpack no longer proves one
   consumer. What remains is the real tier: extend the allocator's liveness
   over the pack/unpack span (the same two `Copy` arms `tt_reg_sync` now
   models), alias the unpack onto the producer's row when nothing else claims
   it, and keep it only if a re-run of the linear scan still fits the 8-tile
   budget (greedy + revert; there is no spill yet). `sigmoid` has 7 such round
   trips = 14 of its 23 hand-offs, so that is where the remaining win is.
2. **`matmul_sigmoid` and the allocator peak** (fusion and (1) both reduce live
   tiles, so this may fall out).
3. **The integer `Sub`** in `tt_log2`'s `e - 127` has the same dtype-blind
   problem the `Add` had (follow the `DType::I32` dispatch), and `tt_log2`
   still fails to compile on
   `native unpack ... needs an F32 DST tile, got I32`.
4. **More block repeats than 4**: a role program may carry only
   `mailbox::loops::MAX` of them, so a kernel with five long row loops is a
   loud compile error today. If that shows up, either nest the row loops inside
   one outer `count = 32` loop (they all repeat the same number of times), which
   is the natural fix, or fall back to unrolling for the extra ones.

**How the hunt went (so it is not repeated):** `ZYX_DEBUG=264` gives the
post-epilogue TT IR. What localized it: a temporary print of the role streams
(`SFPLOAD`/`SFPSTORE` `AddrMod`/`Imm10`, `REPLAY` records) and of
`tt_dst_reuse`'s `last_use` / `logical -> physical` map. `tt_exp2` was bisected
by returning early at stages (clamp / clamp+round / Horner / full) under a
temporary env var; that showed "compare+select fine, materialize + *select*
consumer broken, materialize + *add* consumer fine", which pointed at the
CB hand-off rather than the math. All instrumentation is gone; no
`ZYX_TT_*`/`ZYX_EXP2_*` hooks remain in the passes.

### Start here — session handoff (2026-11-02, third pass)

**Landed since the block below:** `tt_reg_sync` now assigns **one semaphore per
ordered `(producer role, consumer role)` edge**, not one per direction
(`edges` collected after the epoch walk, sorted, `assert!(≤ 8)`, index via
`sem(edge)`; a reuse seam is one post *and* one take per `(reader, next_writer)`
pair, so the next writer takes once on each such edge).

Why: `SEMWAIT` is a **level test, not a reservation**. One counter per direction
put T0 and T1 both waiting on `FREE`, so a single T2 post released both, the
first `SEMGET` took it and the second hit zero. ttsim's message
`tensix_semget: sem=0 sem_max=15` prints the **counter value, not the index**
(`tensix.cpp`, `TENSIX_EXECUTE_SEMGET`) — the whole previous hunt assumed the
index, which made a balanced 6-post/6-take stream look impossible. That was the
bug; the "take inside an unrolled fallback" theory is dead (a render-time scan
of every `REPLAY` record's body span showed no `W`/`G`/`P` inside any body).

Verified with the 12 green ttsim gates (still green, one at a time) and
`cargo test --lib` = 77. `sigmoid`/`sum_sigmoid` **no longer abort** on the
semaphore and now report plain wrong values.

**New failure, not diagnosed (next session):** `sigmoid` and `sum_sigmoid` now
return **one constant for every lane** — `0.20000006` (shape `[10, 256]`, so 80
tiles). Disabling the per-CB unpack cache (`tt_resolve_dst`'s `unpacked` map)
gives a *different* constant, `-1.1e-7`, so the value depends on the tiling but
not on the data. `exp2` (`#[ignore]`) returns all zeros. Both are multi-stage
kernels: DRAM→CB→DST, compute, DST→CB, unpack, compute, … The single-stage gates
(`add`, `unary_math`, `neg_sub_mul`, `shared_input`) are exact, so the suspect is
the **materialise→pack→unpack chain** (`tt_materialize`, `tt_copy`, the
`unpacked` cache), not an SFPU arm (the arm A/B tests pass).

Suspect worth checking first: the `unpacked` cache is keyed by the CB storage and
stores the **physical DST row** from the first unpack; it is never invalidated,
and `tt_dst_reuse` recycles rows, so a second use of the same CB (or any row
reuse after the cache entry) can read a different value's tile.

Also note: `ZYX_DEBUG=8` alone prints nothing on the autotune path; use
`ZYX_DEBUG=264` (256 `NoSearch` + 8 IR) to get a post-epilogue TT dump.

### Start here — session handoff (2026-11-02, second session)

**State:** HEAD is the user's last commit (the tree was committed twice during
this session: `tt better reuse of dsts` and after the replay work). `git status`
should be exactly:

```
NOTES.md
zyx/src/kernel/tenstorrent.rs
zyx/src/codegen/tenstorrent.rs
zyx/tests/23_tt_lowering.rs
```

The user writes commit messages; commit only when asked.

**What this session did, in order**

1. **`TTSfpInsn.args` is `[u16; 5]`** and **`TTOp::Sfp` is unboxed** (`Op` stays
   24 bytes; every SFPU field is ≤ 16 bits — the only wider ISA fields are
   `SETC16.NewValue` and two `AddrLo`, both 18, neither SFPU). `tt_sfp_encode`
   converts to `u32` for `field.fits`/`place` and `debug_assert!`s width ≤ 16.
   `LREG_ZERO`/`LREG_ONE` are `u16`. The other `TTSfp*`/`Dm*` boxes were left
   alone; unboxing them is mechanical (`Box::new(TTSfp… { … }))))` shape, 67
   sites) if wanted.
2. **`tt_unfold`** (in `kernel/tenstorrent.rs`, called from the TT branch of
   `device_epilogue` after `tt_tile`): rewrites `Exp2`/`Log2` in normal `Op`
   space as `exp_program`/`log_program` arithmetic. `Pow` reaches it through
   `unfold_pows`' `Exp2(y Log2 x)`. Scalar constants stay `Op::Const` operands;
   the new `TtEmit::SfpuConst` + `tt_fill_const` fill one into a DST tile
   (`Expand` is refused post-linearize, so no shape op). `Bitcast`, `Cast`
   (`I32↔F32`) and `Select` lower directly in `tt_lower_value`; a permanent
   `#[cfg(test)] mod tt_lower_ops` A/Bs them through `interp`.
3. **Pattern matching is done before lowering** (`TtLowering::matched`), with
   the bindings recorded: once a value lowers its `Op` becomes a row op and
   `shape` reads the slot's buffer length, so re-matching a DAG's shared
   operand fails. `tt_lower_value` replays the recorded decision, then falls
   back to the table if the emitter declines.
4. **`TTOp::Replay { len, count }`** (word count + repeats) + codegen
   `begin_replay`/`finish_replay`: address-modifier entries 0/7, `RWC` clears,
   `REPLAY(load=1, exec=1)` record word before the body, `count-1` replay words
   after, and `AddrMod = 7` patched onto the body's last `SFPLOAD`/`SFPSTORE`.
   `tt_verify` checks `len ≤ 32`, `count ≥ 2`, row ops only, no nesting, body
   complete. `tt_row_loop` is the shared emitter: every SFPU kind arm now emits
   one row group and the helper wraps it; bodies over 32 words fall back to 32
   unrolled copies with `at += 2k`. The test harness (`encode_row_ops`) expands
   a `Replay` with the same two-row step. `tt_fill_const` and the two reduce
   loops still write their own `for o` loops.
5. **`tt_reg_sync`**: the FREE reuse seam is implemented (reader posts after
   its last access, next writer takes before its first); seams are kept
   *outside* replay bodies (`replay_of` maps member ops to the body bounds, so
   `Wait`s go before the marker and `Post`s after the body); **`TTOp::WaitBelowMax`**
   (`SEMWAIT` C1) is emitted before every `Post` as back-pressure for the 4-bit
   counter.
6. **Register model**: `tt_sfp_regs(op, args) -> (reads, writes)` with the
   per-mnemonic table (resolved by ISA field name on `tt_sfp_def`'s form, the
   same one the encoder picks). A test walks all 28 ported kinds and pins
   `SFPIADD`/`SFPSETCC`/`SFPGT`. This is the foundation for fusion.
7. **Unpack cache**: `tt_resolve_dst` keeps one DST tile per reader CB
   (`TtLowering::unpacked`), because the unpack always reads slot 0 and one CB
   holds one tile per phase. `sigmoid`'s `UNPACKED` hand-offs fell 21 → 6.

**Gates / tests**

- `cargo test --lib` = **77** (70 + 2 `sfp_row_ops` + 4 `tt_lower_ops` + 1
  register-model test).
- Green on ttsim, one at a time: `unary_math`, `tiling`, `neg_sub_mul`, `add`,
  `transfer_roundtrip`, `matmul`, `sum`, `max`, `sum_scalar`, `max_scalar`,
  `sum_all`, and the new `shared_input` (`x * x`, exact; the unpack-cache
  regression).
- **Red — `sigmoid`, `sum_sigmoid`**: `NonContractualBehavior: tensix_semget:
  sem=0 sem_max=15`. Measured with a temporary seam trace: 6 posts (T0) / 6
  takes (T1) on sem 0, every take is `SEMWAIT sem 0 (nonzero)` immediately
  followed by `SEMGET sem 0`, the `WaitBelowMax` throttle makes no difference,
  and the unpack cache does not change the failure. So a take reaches `get`
  with the counter at zero although the static stream pairs everything. The
  remaining suspects: (a) a take that codegen replays (the `replay_of` mapping
  covers only marked bodies; an unrolled fallback region is invisible), or
  (b) ttsim's model of wait/get across concurrent roles with several
  consumer roles on one semaphore.
- **Red — `matmul_sigmoid`**: `9 DST register tiles exceed the 8-tile budget`
  (the allocator's peak; it also hits ttsim's documented
  `SRCA_SET_SetOvrdWithAddr=1` limitation for mixed CB→DST unpack + matmul).
- **Red, ignored — `exp2`** in `tests/23_tt_lowering.rs`: the smallest repro
  of the seam failure (`tensix_semget`), `#[ignore]`d and documented.

**What to do next (in order)**

1. **Fix the seam pairing.** Cheapest lead: give each `(writer, reader)` role
   pair its own semaphore instead of one `UNPACKED`/`COMPUTED`/`FREE` per
   direction (8 semaphores, 4 bits each; assign lazily and error past 8). A
   kernel where T1 and T2 both consume T0's writes is exactly where one
   semaphore's single consumer assumption breaks. Otherwise reproduce by
   bisecting the `exp2` kernel: drop the comparisons one at a time (they are
   the big bodies), or force `LoopPolicy`-style unrolling of the chain, and
   see whether the `get`-at-zero disappears — that separates (a) from (b).
2. **`tt_sfpu_fusion`** (the register model is ready). After
   `tt_dataflow_select`, before `tt_dst_reuse`. Merge a chain's per-row-group
   bodies (the `Replay` bodies are exactly that unit), delete the intermediate
   store/next load when the storage's only readers are those loads with the
   same `fmt`/`at`, and rename registers across the merged body with
   `tt_sfp_regs`. **Needs block repeats** (`mailbox::loops`, `MAX = 4`): a
   fused body is ~1 word per IR op, over the 32-word `Replay` buffer, so
   `TTOp::Replay` has to learn a block-repeat form (codegen collects
   `Vec<Loop>` per role and the launch uploads through `code::stored(words,
   loops)`; `TTOp::Math`/`Unpack`/`Pack` carry `{ words, loops }`). Without
   that, fusion can only merge short pairs and will *not* fix the epoch churn.
3. **Then delete the arms**: `tt_sfp_sigmoid`, `tt_sfp_sin_cos`, `tt_sfp_pow`,
   the `Sfpu{kind}` fallback, `tt_sfpu_program`/`tt_row_remap`, and the builder
   half — after `tt_unfold` covers `Sin`/`Cos` too (it does not yet) and
   `ops::kind_sfpu` moves to `kern/kind.rs` (unasked design move).
4. Loose ends: the 10 `unused_mut` warnings from the uniform `|mut ops, o|`
   closure binding; `tt_fill_const` and the reduce loops are still unrolled;
   `ZEROACC` scoping for `matmul_add`; `Op::Loop` lowering for `softmax`/`REM`.

**Traps measured this session**

- `Replay`'s `Count` operand is the **body length**, not a repeat count; a
  record is `REPLAY(index, len, exec=1, load=1)` before the body, and each
  further iteration is one `REPLAY(index, len, 0, 0)` word (`frontend::record`).
  The buffer holds 32 words.
- A word count for a body must count `SfpConst` as two words when
  `bits & 0xffff != 0` (`load_f32` is two `SFPLOADI`s); `tt_row_words` is that
  count in both the emitter and `tt_verify`.
- `verify` refuses a NaN **constant**, and constant folding folds
  `bitcast(0x7fc00000)` into one. `tt_exp2` propagates the input NaN (the
  select returns `x`), `tt_log2` makes its NaN with `sqrt` of a negative.
- A `Replay` body must end with a `Dst` access; codegen patches that access
  with `AddrMod = 7` (the two-row step), and the setup (`thread_entry`s +
  `SETRWC` clear) is emitted at the marker, so a seam landing inside the body
  would be replayed too — `tt_reg_sync` maps member ops to the body bounds.
- `tt_dataflow_select`'s `rcs` is pre-lowering; `tt_materialize` packs a
  multi-use *computed* value to a CB, and every consumer then unpacks it (a
  fresh `UNPACKED` epoch). The unpack cache is per CB storage, sound because
  the unpack always reads slot 0.
- `Expand` is a graph-shape op and is rejected post-linearize; scalar
  constants reach `tt_dataflow_select` as `Op::Const` operands, handled by
  `TtEmit::SfpuConst` and by the `Select` arm's const-branch fill.

### Earlier handoff (superseded 2026-11-02, first session)

**State at handoff:** HEAD `5a3b1dd3` "add comparison and bit ops". The tree
adds item 2's structure change, its last reachable kind, two real fixes the new
A/B test forced out, and a permanent regression test. `git status` should be
exactly:

```
NOTES.md
zyx/src/kernel/tenstorrent.rs
zyx/src/codegen/tenstorrent.rs                                  (one visibility)
zyx/src/backend/tenstorrent/kern/sfpu/ops.rs                    (one visibility)
```

**What this tree changes**
- **`tt_sfp_sequence` is a dispatcher now.** Each builder program it
  transcribes is its own method beside it: `tt_sfp_bool_not`, `tt_sfp_add`/
  `sub`/`mul`, `tt_sfp_neg`/`abs`/`int_not`, `tt_sfp_sqrt`, `tt_sfp_recip`,
  `tt_sfp_round`, `tt_sfp_div`, `tt_sfp_sigmoid`, `tt_sfp_sin_cos`,
  `tt_sfp_pow`, `tt_sfp_compare`, `tt_sfp_integer`. The dispatcher only checks
  arity, calls one, and swaps the last op into `id`.
- **`POW` is ported** (`tt_sfp_pow`: `log_program` + the multiply +
  `exp_program` + `pow_fix`). It has no reachable frontend path: the dynamic
  float `Pow` is rewritten to `Exp2(Log2 x * y)` by `fold_constants` and the
  TT backend has no `Exp2`/`Log2` lowering, so `Tensor::pow` on TT stops in
  `tt_verify` ("DRAM read in compute"). The `sfp_row_ops` A/B is its only
  coverage; giving it a gate means lowering `Exp2`/`Log2` first, which is a
  separate task.
- **SFPU operands are one module-level block** at the top of
  `kernel/tenstorrent.rs`, derived from `tt_isa` (`LReg::L0.index()`,
  `LREG_ZERO`/`LREG_ONE`, `mod0_fmt::FP32`/`INT32`). It replaced four copies
  (the arms and the reduce emitter).
- **`SIN`/`COS` are ported** (`tt_sfp_sin_cos`: `trig_reduce` +
  `sin_cos_cores` + the quadrant tail). Its three spill tiles come from the
  same `tt_new_dst` every value uses, so no signature changed and
  `tt_dst_reuse` places/reuses them normally.
- **`ABS` fixed:** `exact_program`'s `ABS` is `x & 0x7fff_ffff` on raw bits,
  not `SFPABS` with `Mod1 = 1`; they differ on a negative NaN (`+NaN` vs
  `-NaN`). The committed arm had the wrong one.
- **`SFPENCC` added to `SQRT`/`RSQRT`, `RECIP`, `FLOOR`/`TRUNC`, `DIV`.** Only
  `SIGMOID`, the comparisons and the integer arm emitted it. Without it
  `SFPSETCC`/`SFPGT`/`SFPLE` predicate nothing: the interpreter starts with
  lane flags disabled (`SFPENCC.md` says the same on silicon after another
  kernel). `DIV` was the visible failure.
- **Permanent `#[cfg(test)] mod sfp_row_ops`** (bottom of
  `kernel/tenstorrent.rs`): every ported kind's arm and its `ops::code2`
  builder program run through `kern/sfpu/interp.rs` on specials, bit-exact.
  `tt_sfp_encode` is `pub(crate)` and `ops::two_over_pi_limb` is `pub` for it.
- **`Bitcast`, `Cast` and `Select` lower directly** (they are `Op`s, not
  `UOp`/`BOp` kinds, so their arms are in `tt_lower_value` beside
  `tt_lower_load`):
  - `tt_sfp_bitcast`: a raw `INT32` copy, no instruction (the coalescing is
    for a later pass);
  - `tt_sfp_cast_i32_f32` (`ops::i32_to_float`: two's complement split, then
    `SFPCAST` RNE) and `tt_sfp_cast_f32_i32` (`round::to_i32`: Rust's
    saturating `as i32`, NaN to zero); other dtype pairs are `todo!()`;
  - `tt_sfp_select`: `cond != 0` predicates a `mov` of the raw bits.
  Tests: permanent `#[cfg(test)] mod tt_lower_ops`, each arm through `interp`
  against the host (`as f32`, `as i32`, `if cond != 0`). They have no builder
  program to A/B, and `Bool` cannot be packed for ttsim, so this is the only
  coverage `Select` can have.

**Green on ttsim, one test at a time** (the file cascades failures; shared RT:
a full-file run reports 7 failures, 5 of them cascade — always isolate):
`add`, `transfer_roundtrip`, `matmul`, `sigmoid`, `sum`, `sum_sigmoid`,
`max`, `sum_scalar`, `max_scalar`, `sum_all`, `matmul_sigmoid`,
`neg_sub_mul`, `tiling`, `unary_math`. `cargo test --lib` = **76** (70 + the
two `sfp_row_ops` tests + the four `tt_lower_ops` tests).

```
cd zyx && TT_TTSIM_LIB=/home/x/Dev/rust/metalium-rs/vendor/libttsim_bh.so \
  AGENT=1 cargo test --test 23_tt_lowering <name> -- --exact --nocapture
```

**Red on purpose:**
- `softmax` — `Op::Loop` bails `tt_dataflow_select` (`missing EndReader`).
  Item 4.
- `matmul_add` — two matmuls in one kernel; the second `ZEROACC` is
  `CLR_ALL`, which would wipe the first accumulator, and codegen refuses it
  loudly (`CLR_ALL after 64 compute op(s) would destroy live DST rows`). It
  is the gate that will exercise `SETRWC`'s base *once the clear is scoped to
  the accumulator's rows*. Item 5.

The committed base also contains: `tt_reg_sync` (the only pass emitting
`Semset`/`Wait`/`Post`, after `tt_dst_reuse`); the matmul arm's `Dst`
row-counter restore; `Kernel::dtype` walking `Setrwc`; the deleted
`kern/sfpu/reduce.rs`; the codegen pack branch's `continue`; the reuse
(`FREE`) seam as a `todo!()` with `TT_SEM_FREE` reserved;
`codegen::tenstorrent::tests::setrwc_names_the_accumulator_in_dst_halves`, and
the `SETRWC.DstVal` = four bits of 32-row halves fact.

Where the reasoning lives: tiling in "Tiling: the search explores, `tt_tile`
completes"; seams and the `Dst` counter in "SFPU emission: the shape of the
port"; pass order in "Pipeline and pass names" and "Implementation order
(agreed)". Read those before touching `tt_tile`, `tt_reg_sync` or matmul.

**Pick up here:**
1. Confirm the tree (`cargo build`, the gates above, `cargo test --lib`). The
   user writes commit messages: commit only when asked.
2. **Next: `tt_unfold`** (the user's order: Bitcast/Cast/Select first, done;
   now the pass). It lives in the TT branch of `device_epilogue` **after
   `tt_tile`**, before `tt_dataflow_select`, and rewrites the leaf math
   (`Sin`/`Cos`/`Pow`/`Exp2`/`Log2`) into ordinary `Op`s — the builder
   instruction sequences expressed with `Bitcast`/shifts/masks/`Select`/
   `mul`/`mad`. Same approach as `unfold_pows`; for `Pow` the dynamic-float
   branch of `unfold_pows` has to skip TT so the pass sees it. It needs the
   sigmoid chain's `Exp2` too. After it lands: delete `sigmoid_pat`,
   `tt_sfp_sigmoid`, `tt_sfp_sin_cos`, `tt_sfp_pow`, eventually
   `Sfpu{kind}`/`tt_sfpu_program`/`tt_row_remap`. Fusion is optimization
   only (it keeps the chain in `L0..L7`), not correctness, and is NOT a
   prerequisite.
3. **Item 2 — finish it: `REM`.**
   - `POW` is **ported** (`tt_sfp_pow`: `log_program` + the multiply +
     `exp_program` + `pow_fix`) and in `sfp_row_ops`'s binary list. It is
     **not gateable on ttsim, and not reachable from the frontend**: `Tensor::
     pow` lowers `BOp::Pow` to host semantics, and `fold_constants` rewrites a
     dynamic float `Pow` into `Exp2(Log2 x * y)`; `uop_kind` deliberately has
     no `Exp2`/`Log2` entry, so the TT kernel keeps `Unary Log2`, the multiply,
     `Unary Exp2` and the store unlowered and `tt_verify` rejects the DRAM
     `Load` inside compute ("tt_verify: DRAM read in compute"). Same class of
     finding as `Tensor::abs` (which never reaches `ABS`). A `pow` gate would
     be testing `Exp2`/`Log2` lowering, not this arm; that lowering is a
     separate, unasked task. The permanent A/B is the arm's coverage.
   - `REM` (`ops/rem.rs`) is 276 restoring steps: **wait for item 4**. The
     builder stores the body once and the runner repeats it; unrolled it is
     ~7000 ops per row group. Its spill (`SPILL_ROW + 256`) is four standard
     `tt_new_dst` tiles, not a parameter.
   - Then `Sfpu { kind }`/`TtEmit::Sfpu`/`tt_sfpu_program`/`tt_row_remap` and
     codegen's `Sfpu` encoder go, and `kern/sfpu` shrinks to `interp` (the
     oracle). `ops::kind_sfpu` is still used by `uop_kind`/`bop_kind` and the
     pattern table, so **move those constants first** (natural home
     `kern/kind.rs`, whose doc already points at them) — that is an unasked
     design choice, confirm it with the user. `two_over_pi_limb` moves with
     them (the trig arm calls it).
   - `EXP`/`LOG`/`TANH`/`ERF`/`GELU`/other trig have no `uop_kind`/`bop_kind`
     entry, so no table reaches them; they stay the builder's only.
3. **Item 4 — loop lowering for `softmax`** (`REPLAY` for bodies that fit its
   32-word buffer, `mailbox::loops` for longer; a `Param` trip count is a TT
   compile error). `REM` depends on this.
4. **Item 5 — `ZEROACC` scoping** (`matmul_add`), then a true `RWC.Dst = 0` for
   a nonzero accumulator.

**Coverage rules for the next agent**
- `sfp_row_ops` is the only coverage for the kinds ttsim cannot gate (a `Bool`
  output has no tile format; an `I32` input stops at `native unpack ... needs
  an F32 DST tile`) and for the special lanes the random gates miss. It is a
  *supplement*, not a gate: it compares against the builder, so it proves the
  arm transcribes the program, not that the program is right.
- Adding a kind: write the method, add it to the dispatcher, extend
  `sfp_row_ops` (unary or binary list; unary kinds get `OpId::NULL` for `b`).
  The harness maps `a`/`b`/`slot` to `A_ROW`/`B_ROW`/`OUT_ROW` and every
  scratch storage after row 512.
- The builder side is authoritative: when the A/B fails, the arm is wrong
  unless the builder's own program is demonstrably broken.

**Traps measured this session — do not re-derive:**
- `SFPENCC` is not optional. The interpreter's `Vector::new` has
  `use_flags = false`, so without it every scope in an arm silently does
  nothing; ttsim also accepts it either way, so a gate cannot catch this.
- `SFPABS Mod1 = 1` is *not* `x & 0x7fff_ffff`: it keeps a negative NaN's
  sign. `exact_program` uses the mask for `ABS`, `SFPMOV Mod1 = 1` for `NEG`.
- `Program::push` inserts `SFPNOP` only after MAD-class ops before
  `SFPAND`/`SFPOR` (with `USE_VB`), `SFPIADD`, `SFPSHFT` (and two `SFPSHFT2`/
  `SFPSWAP` modes). Unrolled arms must reproduce those NOPs; `trig_reduce`'s
  `mul24`/`iadd` pairs are the ones in the tree.
- An `SFPLOAD`/`SFPSTORE` address is `Imm10 + RWC.Dst` (`interp` models it).
  `MVMUL`'s address modifiers advance the counter, so a row op fused after a
  matmul lands a whole tile too far unless it is restored. Symptom: the pack
  read zeros at rows 64..127 and the sigmoid output was at 128..191.
- The runtime's DST dump covers **rows 0..15 only** (`mailbox::DUMP_MAX_ROWS`
  = 16). To see a later tile, read the writer CB out of L1 at the end of
  `execute` (`run.writes[..].1` is its L1 address; `dev.read32` per word), or
  set a pack's `DEST_TARGET_REG_CFG_PACK_SEC0_Offset` by hand. Both were
  temporary and are reverted; nothing in the tree dumps DST.
- `ZYX_DEBUG=8` prints the optimized IR once per compiled winner; the
  no-search path additionally runs `dead_code_elimination` after
  `device_epilogue`, so "the IR I printed" and "the IR codegen sees" can
  differ there.
- The codegen pack branch ends in `continue`, not `return Ok(())`: a pack is
  one op, not the section's end.
- **A bare unary op on a tensor already on the card runs as one kernel**
  (there is no implicit graph: `Tape` is explicit, and a host expression only
  later moved to TT is evaluated on the **host** — a gate built that way
  proves nothing about the lowering). `neg_sub_mul` (`[10, 256]`) is the
  tiling regression test.
- **`tt_tile` was rewritten to the contract** ("Tiling: the search explores,
  `tt_tile` completes"). Two defects it had: the "already a tile" cap was
  re-read per group, so after the first split raised the index rank every
  later group was skipped and iteration order decided the tiling (`[10, 256]`
  unary: columns padded first, rows left a grid extent, grid `[10, 1]`, one
  4 KB page per row); and the one-group pre-split ran unconditionally, so a
  `[32, 32]` unary (already one tile) got a second grid extent.
- **`Tensor::abs` is `relu(x) + relu(-x)`**, not `UOp::Abs`, so no frontend
  reaches the native `ABS` kind; `relu` has no `uop_kind` entry either. An I32
  elementwise kernel dies later at `native unpack ... needs an F32 DST tile,
  got I32`, so `INT_NOT` and the integer kinds have no runnable gate.
- **Verify an SFPU port is actually exercised**: a temporary `eprintln!` at the
  top of `tt_sfp_sequence` printing `(kind, b.is_null())` shows which arms a
  gate hits. Without it a gate can pass on the *host* and say nothing about the
  device lowering.
- The `tt_tile` probe (a temporary sweep over `[10,256] [32,32] [32,256]
  [64,64] [64,256] [128,128] [10,32] [10,64]` × `neg`/`mul`) was a one-off;
  the search picks different variants per run, so the gates are the regression.

**Unimplemented / next blockers, in the order they bite:**
- **`SETRWC`'s restore** after a matmul is exact only while the accumulator is
  tile 0 (it sets the counter to the accumulator's base, which is 0 there). A
  nonzero accumulator needs an IR way to say `RWC.Dst = 0`; `Setrwc`'s operand
  is a storage. Nothing can move the accumulator off tile 0 today, so this is
  latent, not a live bug.
- **Two matmuls in one kernel** need a `ZEROACC` scoped to the accumulator's
  rows (`CLR_ALL` is the only mode emitted). This is the "matmul tile state"
  item that item 5 also lists.
- **Mixed `CB → DST` unpack plus matmul in one kernel**: ttsim refuses with
  `tensix_unpacr: unpack_to_dst: SRCA_SET_SetOvrdWithAddr=1` — the DST
  unpack's config does not clear the Src override the matmul's `Unpacr` set.
  Found by `w + matmul(x, y)`; not on any gate.
- **ttsim's matmul only runs with the accumulator at tile 0** (it refuses
  `MVMUL dst_row` past ~64). Silicon is unmeasured for a nonzero accumulator.
- Item 3 leftovers and `tt_tile`'s multi-tile index math are still open
  (see item 3 and item 5).

### 0. Memory model (the card, not the host)

- **Host** = CPU DDR5.
- **Device DRAM** = the card's VRAM (28 GB on the P100A; `Dram::FULL` is 8
  channels × `CHANNEL_BYTES` ≈ 4.27 GB).
- **L1** = the per-core CBs and scratch (the `Circular`/`Local` storages).
- **DST** = the compute core's register file (8 × 32-bit tiles or 16 ×
  16-bit tiles, single mode).

`Pool::TT` means **device DRAM**, and since 2026-11-01 it is (§1).

### 1. `Pool::TT` IS DEVICE DRAM (restored 2026-11-01)

The C++ shim allocated a real device buffer:
`alloc_buffer` created a tt-metal `BufferType::DRAM` MeshBuffer (2 KB page
rounding), `write_buffer` uploaded to it through the command queue, and
`read_buffer` downloaded from it. `Pool::TT` chunks were therefore in the
card's DRAM, and the reader/writer moved DRAM↔L1.

It happened on the `tt-native` branch:

- **`cca1c452` "tt: drop feature gates and the C++ shim" (2026-10-28 03:32)**
  deleted `build.rs` and `tt_runtime_shim.{cpp,h}` and, in the same diff,
  rewired every `Pool::TT` operation to the host pool:
  `Pool::TT(id) => (lock(self, tenstorrent::pool(id)?).allocate(bytes), …)`
  became `Pool::TT(_) => (lock(self, host::pool()).allocate(bytes), …)`, and
  the same for `release`, `try_reuse_allocations`, `dispose`, `free_bytes`,
  `pool_to_host`, `buffer_ptr_mut`.
- **`88156a9b` "tt: native transport — owner thread over tt-ttsim, host
  DRAM, native compile/launch" (2026-10-28 16:03)** made the consequence
  explicit: the comment `// DRAM is host memory in this transport: a
  Pool::TT chunk is a Vec<u8>`, host-side staging of reader moves into L1,
  and host read-back for writers.
- **`82f441f7`** (squash merge) brought both to main. `git blame` names the
  merge because it squashed the branch; the change originated in
  `cca1c452`.

Consequences while it stood: the card's DRAM was unused, `Pool::TT` chunks
were CPU `Vec<u8>`s, `buffer_ptr_mut` handed out a CPU pointer to "VRAM", the
reader was a host copy into simulated L1, and the writer a host read-back plus
a host narrowing write. The DM cores were never involved.

**Restored (2026-11-01).** All `Pool::TT` arms in `backend/mod.rs` are back on
`tenstorrent::pool(id)`; a TT chunk is a flat range of the card's GDDR6.

- `TTMemoryPool` (`backend/tenstorrent/mod.rs`) is a flat DRAM allocator:
  `DramChunk { addr, len }` in the chip's address space, `Slab` + `free_set`
  + `free_bytes` + a 4 KB-aligned `bump`, best-fit reuse and `dispose` copied
  from `HostMemoryPool`. `total` is `min(detect_dram_bytes(), channels *
  CHANNEL_BYTES)`, so the flat space never needs a channel it does not have.
- `host_to_pool`/`pool_to_host` (plus `read_at`/`write_at` for sub-chunk
  slices) translate the flat range into per-channel `DramRange`s and go
  through the owner thread: `ttsim::dram_write`/`dram_read` →
  `Device::dram_write`/`dram_read` on one held `TwoMib` window. Translation:
  page `p = addr / 4096`, `channel = p % channels`, offset
  `(p / channels) * 4096 + addr % 4096`; accesses are cut at page
  boundaries. One command carries every segment of a transfer.
- `Pool::TT` `buffer_ptr_mut` is `todo!("TT buffers have no staging
  pointer")` — it never existed for TT and must not come back.
- Launch: the reader `read_at(chunk, off, &mut data)` into a host staging
  buffer and stages L1 as before; the writer takes the L1 read-back, narrows
  on the host (`TT_NARROW_NONE/F16/BF16`, no raw-pointer write), and
  `write_at(chunk, off, &narrowed)`. Bounds are checked against `chunk_len`.
  `TTDevice::copy` downloads through the source pool and `host_to_pool`s the
  bytes; `host::copy`'s `Pool::TT` arm is a DRAM read straight into the host
  destination.
- `Dram::FULL` is the pool's grid (the transport opens a simulated chip).
  Silicon must take it from `Device::dram_grid` — the ported layer refuses a
  harvested chip (a p100's seven channels) until the translated DRAM
  numbering lands.

**Interleaving and host transfers (2026-11-01).** The flat space is striped in
4 KB pages (`STRIPE`, round-robin over the chip's usable channels). tt-metal
interleaves a buffer in units of *its* page rounded to `DRAM_ALIGNMENT` (64 on
Blackhole) — 4 KB for an FP32 tile, 2 KB for BF16 — so 4 KB is the default here
and a per-buffer page is a later change (it would need the page at
`allocate` time). Host upload/readback coalesces: a channel's pages are
contiguous in channel offset, so a transfer is gathered per channel and each
run goes out as **one** `dram_write`/`dram_read` — at most `channel_count`
commands and visibility barriers per tensor, instead of one per page. A 1 GB
upload is ~512 host window retargets (2 MiB TLB window) rather than ~512 K
command round trips. The gather costs one extra host pass over the bytes,
which is where tt-metal instead pins host memory and lets the command queue
DMA (`k_pin_write_threshold_bytes` = 32 MB); `dm::op::HOST_READ`/`HOST_WRITE`
is the same idea for us and is not wired yet.

**The boundary is the tile's data movers (2026-11-01).** No tensor byte crosses
the host inside a launch:

- Each tile boots RISCV B (`tt_firmware_images::DM_B`) and RISCV NC (`DM_NC`,
  through the stub at NC's reset PC) once: the two are `Image`s in the device's
  cache like any other program, so this boot is the first residency miss on the B
  and NC slots (`ttsim::boot_image`, bytes read back from the card's DRAM copy).
  The mailbox is written first (`MY_X`/`MY_Y`, `USABLE` = the DRAM grid's mask,
  `IN_FLIGHT_CAP`, and every word the host owns zeroed), then the image starts
  and the host waits for `RUNNING` and `CAPS`.
- A launch hands B its reads as **one list** before the roles run and NC its
  writes as **one list** after they finish
  (`tt_isa::dm::op::LIST` → `dm::LIST` ring, `OP`, `LEN`, `SEQ`, then `DONE` /
  `ERROR`). Entries are `[op, channel, port, offset, l1, len, 0, 0]`, one per
  4 KB page of the flat chunk — the page is the unit that is contiguous on both
  sides. Lists are cut at `dm::LIST_MAX` (512 entries, 1 MB of pages) and each
  is waited for.
- The launch no longer stages L1 or reads DRAM: it only translates flat chunk
  ranges into `(DramRange, l1)` pairs (`TTMemoryPool::ranges`) and hands them
  over. `TTDmMove` is still the *description* of a move; the mover is the
  executor.
- One exception stays on the host: a 16-bit output (BF16/F16) is read back and
  narrowed there, because the packer's format conversion is refused by the
  pinned ttsim (`pack_fmt_conv_mode=0x105`). Those moves are `read_back` and
  `write_at`, not mover entries.
- Constraints this introduced: a `READ` entry needs
  `offset % 64 == l1 % 64`, a `WRITE` `offset % 16 == l1 % 16`
  (`Descriptor::decode`). Chunks are 4 KB aligned and tile moves are 4 KB
  aligned, so the gates fit; a violating move is refused loudly by the mover,
  never silently truncated.
- Still host-issued, in order of what it costs: the three role images are
  re-loaded every launch (`runtime::run`), and a tensor's placement/readback is
  still a per-page host DRAM copy. The first goes away with a resident
  `dm::op::KERNEL` entry naming programs in the program cache (one list per
  op, one host round trip — the shape the shim had); the second with
  `dm::op::HOST_READ`/`HOST_WRITE`.

Verification: the Rust ttsim wrapper's `dram_read`/`dram_write` **do work**.
`23_tt_lowering`, each gate run alone: `transfer_roundtrip` 0/200, `matmul`
0/120, `sigmoid` 0/2560, `sum` 0/10, `sum_sigmoid` 0/10. `matmul_sigmoid`
and `softmax` still fail where they did (`tt_verify` I64 DST; `Op::Loop`
bails `tt_dataflow_select`).

**Pre-existing, not the DRAM path:** running the whole `23_tt_lowering`
binary in one process (any thread count) makes every SFPU kernel *after*
`matmul` read zeros from DST (`sigmoid` → 0.5) while `transfer_roundtrip`
still passes. A host-shadow A/B (pool bytes kept in host memory, identical
launch path) reproduced it exactly: it is device/tile state left by the
Matrix Unit program, not memory. Run the gates one at a time until it is
found.

### 2. COMPILATION ENDS AT CODEGEN — THE DEVICE PROGRAM IS WHAT RUNS

The rule:

> Compilation turns a `Kernel` into the backend's own images and uploads them to the
> device once. What runs is those images, not the `Kernel`: a launch binds arguments
> and issues. Nothing is encoded, expanded or decided at launch.

**The `Kernel` is the compiler's input, not the program.** The passes and codegen are
the last compilation. Codegen turns the Kernel into per-core bytes; the backend
collects them into two tables.

**Table A — the images on the card.** One image is one core's bytes: nothing on the
card occupies two cores at once.

```rust
pub struct Image {
    core: Core,      // B, NC, T0, T1, T2
    address: u32,    // flat device DRAM address of the bytes
    len: u32,        // byte length
}
images: Slab<ImageId, Image>,
```

**Table B — the launchable unit**, which is what `DeviceProgramId` names (unchanged
meaning: the handle a plan holds, a launch takes and `release` frees):

```rust
struct TTProgram {
    /// The legs this kernel runs, in program order.
    sequence: Box<[ImageId]>,
    n_inputs, n_outputs, gws, max_grid,
}
```

**A zyx kernel's boundary is DRAM**, so a launch reads DRAM, computes, writes DRAM,
and `sequence` is the legs the kernel actually has:

```
compute kernel      B: reader, T0: unpack, T1: math, T2: pack, NC: writer
gather / scatter    B: reader, NC: writer
```

The compute leg is not a hole to leave empty: a kernel with nothing to compute emits
two images, one that computes emits five. Adjacent role images (T0/T1/T2) are started
together — they hand off through semaphores — while B's and NC's keep their order.

**The backend owns them afterwards.** An image is uploaded into device DRAM once at
compile (into L1 by NoC when it runs), and after that the backend keeps only the two
tables. No blobs in host RAM after compilation; a copy kept for debugging is a debug
artifact, not an image. That is CUDA's module/handle: the bytes are on the card, the
host holds the address.

**A launch binds arguments and issues.** It writes the operand words an image's holes
name — chunk addresses, variable values, grid size — plus the mailbox and sequence
words that start it. It never adds, removes, reorders, splits or encodes.

**Residency is per (tile, core), over five slots:**

```
resident[tile] = [Option<(ImageId, generation)>; 5]   // B, NC, T0, T1, T2
```

At most five integer compares per tile per launch; a load only when a slot's identity
changes, and a load is a PCIe write the first time or a NoC read from the image's DRAM
copy after that. The two mover firmware images are images as well — created by the
backend at init from `tt_firmware_images::DM_B`/`DM_NC`, not derived from any `Kernel`
— so **boot is not a concept**: it is the first residency miss on B or NC.

- Lifetime: refcounted (a `TTProgram` holds one ref per image in its sequence). Slab
  slots may be recycled, so identity is `(id, generation)` and a stale ledger entry is
  a miss by construction. zyx releases almost nothing; the one release is autotuning
  dropping a candidate when a better one wins.
- Invalidation: a tile that failed clears its ledger entries and reloads on next use —
  the same idea as `kern::runtime::Resident`'s `poisoned` flag, one level down.

**Implemented 2026-11-01:** `Image`/`ImageId` and the `images` slab; `TTProgram` holds
`sequence: Box<[ImageId]>` plus `lens: Box<[u32]>` (each leg's program-length word)
instead of role words; `compile` uploads each role's stored form into DRAM and creates
the two firmware images on first use; the owner thread keeps `resident[tile]` over the
five slots, boots a mover on its first miss from the card's DRAM copy, and loads a
role's program with the B mover (`runtime::run` takes a `load` callback, so the load
happens after the setup run that clobbers slot 0); `release` frees a program's images
and their DRAM. The boundary stream is IR too: `render_tt` emits one
`TTOp::DmRead`/`DmWrite` per 4 KB page and `compile` turns them into the program's
`reader`/`writer` tables, so no side table is re-read at launch. Still launch-side:
the reader/writer *lists* (per-launch ranges written into the mover's L1 ring —
see item 7 for why they cannot be images), and the three role **firmware** images,
which `runtime::run` still loads into T0/T1/T2 on every launch — backend constants
kept running across launches is the `Resident` generation protocol, not this cache.

**PCIe is the budget.** That is *why* compilation exists: CUDA's execution is "bind
the args, launch the kernel the pointer names", and the only reason it can be that is
that the kernel's bytes already went over the bus at load time. On TT the same must
hold:

- a compiled kernel's bytes — role blobs, mover entries, CB/DST config, mailbox
  preloads — are uploaded **once**, and live in DRAM (28 GB, free) and/or a tile's L1
  program cache;
- a launch writes **args only**: chunk addresses, variable values, grid size, and the
  sequence/generation words that start things. Cost proportional to the argument
  count, never to program size or tensor size;
- the scarce resource is L1, not DRAM: 1.5 MB per tile, of which the data arena takes
  1 MB, so residency has a policy and a miss is a NoC load (a `Dm` read from DRAM into
  the program cache), not a PCIe transfer;
- which is why per-page entries are pre-written with operand holes rather than built
  at launch, and why `Dm(Kernel)` naming a resident program is the endpoint of the
  `Dm` work, not an optimization on top of it.

The CPU exists because Tenstorrent has no "launch kernel" command. So the dispatch
is the physical picture, one line per core:

```
reader list -> RISCV B     Unpack blob -> T0     Math blob -> T1
                           Pack blob   -> T2     writer list -> RISCV NC
```

and with `Dm(Kernel)` as an entry of the reader list even that collapses: B runs
the reads, starts the roles, then runs the writes — one list, one round trip.

What this forbids, each of which the launch does today (compilation at runtime):

- expanding a `TTDmMove` into per-page mover entries — the entry count is stream
  shape, so **codegen** emits one op per page (settled below);
- cutting a list at `dm::LIST_MAX` — stream shape; the per-core entry counts must
  already fit, checked at compile time (a `tt_verify` invariant), so the launch's
  list boundary is just "ops changed core", never a split decision;
- computing strides at launch — they are compile-time constants carried by the op;
  substituting a grid point's coordinates into them is dispatch (settled below);
- deciding per writer move whether the host narrows;
- reading a *second program representation* out of the IR: the `Kernel` must not carry
  tables the backend re-reads (`TTNativeProgram`'s `TTDmMove` / `TTLoop` /
  `TTSemaphoreInit` / `TTDstRow` / CB `l1` table, `TTProgramDesc`). The backend's own
  program table is fine — that is the compiled form, not a second IR.

Mover boot is *not* on that list: it changes no op and no entry, it is the first
residency miss on a core (`resident[tile]`), not a decision about the stream.

**Target op order as codegen leaves it:** `Params`, `TensixGrid`,
`Unpack`/`Math`/`Pack`, then the flow — `DmRead`*, `Dm(Kernel)` (which starts the
role blobs), `DmWrite`*, `Dm(Wait)`. **Today** the reader's `DmRead` pages sit in the
reader section (before `EndReader`), the writer's `DmWrite` pages and the `Semset`s
after `EndCompute`, and `compile` turns them into the program's tables. The
`DmKernel`/`DmWait` variants exist but nothing emits them yet, so the launch still
issues the role start and the waits around the lists; `tt_verify` already counts
every `Dm` op as one entry of its section's list and rejects a stream past
`dm::LIST_MAX`. Every address-carrying op holds the operand the dispatch fills.
Nothing in the dispatch is a `match` over policy, only a `match` over how to issue.

**What must therefore move into the passes and codegen:** the per-page expansion of a
move (codegen, one op per page), the section assignment of `Dm` ops (codegen consumes
`EndReader`/`EndCompute`), grid dispatch as an operand substitution the dispatch
performs over a launch-time count, and narrowing as an op.

**Settled (2026-11-01):**

- **One flat `TTOp`.** The mover's ops are plain `TTOp` variants, no sub-enum;
  a payload that would grow `Op` past its budget is boxed, like the blobs are.
- **Grid dispatch is launch-time and legitimate.** How many Tensix cores a launch
  covers is a launch value, the way a CUDA grid size is: the dispatch iterates the
  points and substitutes each point's coordinates into the ops that name them. The
  ops themselves are tile-uniform and their strides are compile-time constants, so
  nothing about the stream's shape changes with the grid.
- **The role blob is the program-slot image.** Codegen emits exactly what the slot
  holds — loop header (when there is one), code, and the length word — and `compile`
  turns it into an `Image`. `REPLAY` stays an *instruction* in the stream for a body
  that fits its 32-word buffer; the `mailbox::loops` header exists only for a body
  too long for that buffer. Today no role program carries loops (SFPU bodies are
  unrolled), so `compile` calls `code::stored(words, &[])` and the length is the word
  count; loop lowering is step 4.
- **No dynamic stride math on TT.** A stride or extent that bottoms out in
  `Param { Variable }` is a loud TT compile error, not a runtime computation: shapes
  must fold to `Const` before codegen. Chunk addresses stay launch values, but they
  are *operands*, not strides, and the pool aligns them to the 4 KB stripe — which
  is what keeps every move's entry count compile-time.
- **Host narrowing is a simulator artifact.** The pinned `libttsim` refuses PACR
  `pack_fmt_conv_mode = 0x105` (`docs/learnings/ttsim-divergence.md`, "BF16 late
  narrowing"); the card packs `F32 → BF16` itself, and origin/main's shim had no
  narrowing path at all because tt-metal's packer did it. So the writer's host narrow
  is ttsim-only, and disappears on silicon.
- **`REPLAY` is an instruction and stays.** What goes is the `Program` / `LoopForm`
  *builders* and the decode-shaped `Instruction` plumbing; `tt_isa::frontend::REPLAY`
  words are part of the stream. `Code` shrinks to `Loop` plus the free `stored()`.
- **Compilation output is two tables, not the `Kernel`.** A `Kernel` is the compiler's
  input; codegen turns it into per-core bytes, the backend uploads each into device
  DRAM and keeps `images` (`Image { core, address, len }`) plus `programs`
  (`TTProgram { sequence: Box<[ImageId]>, … }`). After compilation there is no second
  program structure in host RAM, and no IR-side table the launch consults.
- **Direction is the op kind.** B-bound and NC-bound work are different `Dm` variants,
  so there is no "which core" field to fill at launch. `EndReader`/`EndCompute` are
  consumed by codegen and do not appear in the compiled kernel.
- **`Op::Loop` with a dynamic trip count is IR-legal** (CUDA does it), but on TT the
  count must fold to `Const`; a `Param`-backed count is a loud compile error, not a
  runtime value.
- **Per-page expansion is codegen's.** One `Dm` op per 4 KB page, carrying a symbolic
  chunk-relative offset; the dispatch substitutes `param → chunk address` and derives
  channel/offset. Address translation, no shape decision.

**Consequences:** `TTProgram` (what `DeviceProgramId` names) holds a `sequence` of
`ImageId`s instead of bytes; `TTNativeProgram` and `TTProgramDesc` die as IR-side
structures; `TTDmMove` becomes `Dm` ops; the launch becomes a walk that binds values
and issues; and the `Kernel` does not survive compilation beyond debugging.

### Tiling: the search explores, `tt_tile` completes (contract, 2026-11-02)

- **The search is free to explore tilings; `tt_tile` is the guarantee.** The
  search may vectorize any axis it likes, and `opt_split` is already bounded to
  what the device can hold: on TT the only lane factor is 32
  (`group_unroll`/`loop_factors` are `[32]`, `split.rs`), local factors are
  capped by `max_local_work_dims` (`[1, 1024, 1]`), and a new lane axis is
  refused once an access already has rank > 1 (`tile_access`). So the search
  hands `tt_tile` zero, one or two 32-wide axes — never more.
- **`tt_tile` fills the remaining gaps and re-tiles nothing.** An access that
  already carries `DevInfo::tile.len()` (2) axes *is* a tile: the pass does
  nothing to it. A partially tiled access gets its missing axes; a fully scalar
  one gets all of them. It never replaces or reorders an axis the search chose.
- **The axes present = the access index's rank.** One `Unroll` per axis. This
  is read **once, at entry**, before any `pad_to`/`split` here adds an axis —
  read per group it is wrong, because the first split raises the rank and every
  later group then looks already tiled (that is how iteration order decided the
  tiling before the rewrite).
- **Which group completes an axis.** A group whose scaled form is
  `32 * g + lane` is counted in **tiles**: it is that dimension's grid axis and
  is left alone. The other group counts **elements** and is the one padded to 32
  and split into an inner 32-lane `Unroll` — the axis being added. The test is
  the lane **added** to the scaled group, not a bare `g * 32`: 32 is also a
  legal row stride (`[10, 32]` has `rows = g * 32`, and the row group is exactly
  the one that must be split).
- **The one-group pre-split is for a fully scalar access only.** One group must
  become a 2-D grid (split into two nested groups, each then padded and split).
  If the access already has a lane axis (rank ≥ 1) that pre-split double-counts
  the dimension: `[32, 32]` unary arrives as a 32-lane unroll over rows plus one
  group over columns, and pre-splitting the group made a second grid axis, then
  a rank-3 load `tt_dataflow_select` bails on.
- **Reduces keep splitting every group.** `Op::Reduce` only exists here after
  `split` unrolled a loop into lanes; that rank-3 cube is deliberate and
  `tt_reindex` rewrites it for the movers.
- **Why not let `tt_tile` choose?** Then there would be exactly one fixed
  tiling for every kernel, and the best one differs (which axis is vectorized,
  where the grid goes). Search explores; `tt_tile` normalizes.
- **Order of the loop**: groups sorted by group axis, low first, so the axis
  added is deterministic.

### Pipeline and pass names

```
tt_tile -> tt_reindex -> CSE/DCE -> tt_dataflow_select -> tt_dst_reuse
        -> tt_reg_sync -> verify -> tt_verify
```

Four stages, and the middle one is what the TTOp form exists for:

1. **`tt_dataflow_select`** lowers the tiled SSA into the TT opset and performs
   the *primitive* allocation that representation cannot avoid: every result
   gets a fresh tile, in order, zero reuse. Lowering and allocation are one
   step — a TT op must name a tile, and there are no TT ops before lowering, so
   neither can happen without the other.
2. **`tt_dst_reuse`** is the real allocator: it minimizes that naive tile set
   (coalescing, in-place merging, store/load forwarding, dead-tile removal) and
   is where the simple post-regalloc optimizations live — ordering, peepholes,
   later L1 spills. It only rewrites *which* tile an op names.
3. **`tt_reg_sync`** derives the role hand-off over the final physical tiles
   (unpacker / math / packer) and is the only pass that emits
   `Semset`/`Wait`/`Post`. It runs after the allocator because coalescing
   changes which edge exists at all, and before codegen because those are
   instructions, not an encoding detail.
4. **Codegen** converts that post-regalloc IR into bytecode, the `u32` words;
   the `TTOp` variants do not survive it.

So the TTOp set is not a machine-independent abstraction: it is **whatever the
simple post-regalloc optimizations need**, the way an LLVM machine IR is
whatever regalloc and the machine passes need. Data ops manipulate tiles; SFPU
ops address rows; anything finer (lanes, `LReg` assignment, flag stack,
`REPLAY` vs unroll, address arithmetic) is the encoder's and never appears in
the IR.

**Granularity is a tool, not a risk.** The IR goes as fine as the passes need —
tile-level data ops, row-level SFPU ops, and instruction-level ops where a pass
must see them — and nothing about that "blows up": `Op` is 24 bytes today, the
slab is flat, and thousands of ops per kernel are ordinary (the elementwise
stream that is one `Sfpu { kind }` today is a row or instruction per row then).
Variants are shaped for what the post-regalloc passes must inspect, for
orderings they must preserve, and for the granularity at which they must fire —
never for compactness, and never limited by the 32-byte `Op`, which boxing only
keeps from growing. `REPLAY` stays an encoder/peephole decision on that stream,
not a reason to keep the IR macro-sized.

- **`tt_dataflow_select`** (renamed from `tt_lower_llks` / `tt_reg_alloc`) is
  the phase boundary, instruction selection, and the primitive allocation:
  it converts the tiled SSA (`Unary`/`Binary`/`Reduce`/`Matmul`, UOps
  `Exp2`/`Log2`/`Reciprocal`) into the mutating ISA form and gives every result
  its own tile (zero reuse) as it goes — that assignment is not an optimization
  decision, it is what makes the ops expressible. It also emits the boundary
  traffic (`Copy` publish/unpack/pack) and the section nesting. No generic pass
  may run after it.
- **`tt_dst_reuse`** is the real allocator over the typed form: tile
  liveness, coalescing, in-place merging, store/load forwarding, dead tile
  removal. Peepholes belong here. L1 spills, when they land, are this pass's
  job (today pressure is resolved by coalescing or it fails loudly).
- **`tt_verify`** owns the post-allocation invariants: no
  `Op::Load`/`Store`/`Copy`/`Asm`, no float SSA compute, no `Reduce`, DST
  width (16/32) and single-mode, 8/16-tile budget, row bounds, flag balance.
- CSE/DCE run **before** `tt_dataflow_select`; nothing generic runs after.

### Target IR: `TTOp` is the instruction

One variant per ISA instruction form, operands typed:

```
Sfp      { op: u8, a, b, c, d: u8, mod1: u8, imm: u32 }   // ISA ALU form
SfpLoad  { slot: OpId, at: u16, reg: u8, fmt: u8 }
SfpStore { slot: OpId, at: u16, reg: u8, fmt: u8 }
SfpConst { reg: u8, bits: u32 }
```

plus the existing ISA ops (`Unpacr`, `Setadczw`, `Setadcx`, `Setrwc`,
`Zeroacc`, `Mvmul`, `Setc16`, `Setcfg`, `Post`, `Wait` — `ReduceTile` was
here until 2026-11-02, now deleted) and the non-ISA placement/run/boundary ops
(`Copy`, `Semset`, `Dm*`, `EndReader`, `EndCompute`, `TensixGrid`).

- `a`/`b`/`c` are source LREGs (`L0..L7` writable; `L8..L15` read-only
  constants/config), `d` is the destination for forms that write one,
  `mod1`/`imm` carry the modifier and immediate where the form has one.
  `op` is the ISA opcode — the form decides which operands are read and
  written.
- `slot` is the allocated Register tile and `at` the `Dst` address inside
  it, so the encoder's row is `slot.id * rows + at`. Row-level ops are
  numerous by design; `REPLAY` and the address modifiers are the encoder's
  peephole, not a reason to keep the IR macro-sized.
- `row` is a DST row, not a free index: a tile lives in an
  `Op::Storage { Register }` whose `id` is the slot the allocator bound, and a
  row is `id * 32` (16-bit tiles) or `id * 64` (32-bit) plus the row inside the
  tile. The tile-level identity stays the storage; the encoded instruction
  carries the row.
- **The opset is deliberately not frozen.** It is what lowering emits, what the
  post-allocation passes optimize, and what codegen packs, so variants get
  added, merged, or dropped as the tests demand — one per ISA instruction form
  only for as long as that is the easiest shape to work with. The fixed points
  are the ISA words it must encode and the invariants in `tt_verify`.
- `Sfpu { kind }` is a macro and goes away with this cut. Its expansions live
  in `tt_dataflow_select`: `exp2(x) = exp(x * ln2)`, `log2(x) =
  log(x) * log2e`, `reciprocal` = `SFPARECIP` + Newton. The zyx UOp surface
  has only `Exp2`/`Log2`/`Reciprocal`; `exp`, `log`, `sigmoid` are frontend
  compositions, not backend ops. (`ReduceTile` was the last structured
  compute op; it is deleted as of 2026-11-02 — a reduce is row ops now.)
- `kern/sfpu`'s `Program` / `Instruction` / `Code` / `LoopForm` / `REPLAY`
  bookkeeping is to be **deleted** from the lowering path. The macro bodies
  remain as instruction-selection functions that emit typed ops;
  `tt_isa::sfpu::encode` is the only encoder, and codegen calls it.

### SFPU emission: the shape of the port (settled 2026-11-01)

Working notes for writing the emitter — these are the mechanics, not opinions.

- **Rows are unrolled in the IR.** `for_each_row_group(rows=64)` runs
  `iterations = rows/2 = 32` times with offsets `o = 0, 2, …, 62` (an `SFPLOAD`
  address is a four-row group plus a column half, hence step 2). One IR op per
  instruction, per iteration. No `LoopForm`, no `REPLAY` in the IR.
- **The loop bookkeeping goes away with it.** Today, when the body fits the
  32-entry buffer, the builder emits `thread_entry(addr_mod 0)`,
  `thread_entry(addr_mod 7, step 2)`, `clear_dst_rwc()`, records the body with
  its last `Dst` access patched to the stepping modifier, replays it, and
  clears the counter again (`mod.rs:668-742`). Unrolled offsets make all of
  that unnecessary; the role preamble codegen already emits stays. Words for a
  short body (`BOOL_NOT`, `SQRT`) therefore *differ* from today's replayed
  form — same program, different encoding — so the acceptance check for those
  is `interp`/gates, not word parity.
- **Row ranges map onto the op's allocated tiles.** The bodies address global
  `Dst` rows: `A_ROW 0..64`, `B_ROW 64..128`, `OUT_ROW 128..192`,
  `C_ROW 192..256`, `SPILL_ROW 256..` (`kernel.rs:38-47`). The emitter resolves
  `at` to `(storage, at - base)` by range and carries `slot` + intra-tile
  offset; codegen adds `slot.id * rows`. `SPILL_ROW` is used by the Precise
  `sigmoid` (and by `log1p`/`pow`), so those kinds need scratch. The row-op
  path does **not** allocate a tile for it: `ops::sigmoid_program` spills only
  the input `x` across the exponential, and the output tile's rows are dead
  until the last store of the iteration, so `SfpStore { slot, at: o }` then
  `SfpLoad { slot, at: o }` uses the output as scratch (coalescing the input
  and output onto one row is still safe: the store rewrites the bytes it read,
  before the row's last use, and iterations touch disjoint addresses). A kind
  whose spill must survive the whole body (`log1p`, `pow`) does need a real
  scratch tile — settle that when those are ported.
- **Flag scopes are plain `Sfp` ops in sequence** (`SFPPUSHC`, `SFPSETCC`,
  `SFPCOMPC`, `SFPPOPC`), so a pass can see and cancel them. If a pass can
  unbalance them, `tt_verify` gains a stack-depth-returns-to-zero invariant;
  no new variant.
- **NOP after MAD is codegen's.** `sfpu/mod.rs` inserts `SFPNOP` where
  `SFPMAD`'s automatic stall misses a dependency; with unrolled IR that is a
  peephole over consecutive ops in the encoder, exactly where `SFPLOADI`
  expansion already lives.
- **`Sfp`'s generic encoding: two traps, both hit by `SIGMOID`.**
  `tt_sfp_encode` places `args` into `InstructionDef::fields()` order (high
  bits first), and `args` is a fixed five-word array, so a form with fewer
  fields reads the leading entries and ignores the rest (do **not** ask for
  as many args as fields). The other trap is *which* definition an opcode
  names: `ALL` carries both layouts where Blackhole replaced a Wormhole form,
  the Wormhole one keeps the plain key (`SFPAND`, `SFPPUSHC`, `SFPPOPC`) and
  Blackhole's is suffixed (`SFPAND_BH`), so preferring `key == mnemonic` picks
  the **superseded** entry and silently drops fields — `SFPAND` lost `VB`
  (`0x7e000010` instead of `0x7e000141`) and the mask came out wrong. Select
  by `opcode` with `Provenance::SupersededOnBlackhole` excluded. Verified
  against the builder word-for-word while porting `SIGMOID`; keep doing that
  (the builder's stream is the reference), because a wrong field order is
  silent and only shows as wrong numbers.
- **`SFPENCC` is part of the program, not the preamble.** `SFPSETCC` is a
  no-op unless `UseLaneFlagsForLaneEnable` is set, and `SFPCOMPC` clears
  `LaneFlags` when it is not; the builder emits `sfpencc(3, 0, EI|RI)` at the
  start of every program (`Program::new`). The row-op path must emit the same
  `Sfp { SFPENCC, [3, 0, 10] }` once, before the first row, or every flag
  scope silently executes its body in all lanes. `add`/`BOOL_NOT` do not need
  it (no scopes); anything ported with `if_` does.
- **A suspicious `if_` result is usually the transcript, not the flags.**
  `SIGMOID` first came out as `sigmoid(-|x|)`; the flag scopes were fine. The
  bug was `isub_from(t, magic)` written as `SFPMAD`-style `[c, d]` instead of
  `SFPIADD`'s `[c, d]` = `d = c - d` order — one swapped pair of `LReg`s, a
  factor-of-two result. Compare the emitted words against the builder's before
  believing a structural diagnosis.
- **Port order**: `BOOL_NOT`, `ADD`, `SIGMOID` (2026-11-02; the reduce came
  with them and deleted `ReduceTile`), then `NEG`, `ABS`, `SUB`, `MUL`,
  `INT_NOT` (one-instruction arms), then `SQRT`/`RSQRT`
  (`sqrt_program`), `RECIP` (`Program::recip`), `FLOOR`/`TRUNC`
  (`round::body`), `DIV` (`scale_large_divisor` + `recip` + `divide`) — gate
  `unary_math`; then the comparisons (`EQ..LE` via `compare_body`), the
  integer bit ops (`integer::body` ops 3–7), `SIN`/`COS`
  (`sin_cos_program`, three standard spill tiles), and `POW`
  (`log_program` + the multiply + `exp_program` + `pow_fix`). All the arms are
  now one function per builder program, dispatched from `tt_sfp_sequence`.
  `POW` has no frontend path (see the handoff: `fold_constants` turns `Pow`
  into `Exp2(Log2 x * y)`, which the TT backend does not lower), so the A/B
  is its only coverage. Remaining reachable: `REM` (`ops::rem`), which also
  wants item 4's loop lowering.
  `EXP`/`LOG` are builder-only kinds (no table entry), and `EXP`'s body is
  already inlined in the sigmoid arm.
- **Acceptance** = the gates, plus `interp` (`kern/sfpu/interp.rs`) for the
  kinds no gate touches. Word parity is a debugging aid, not the criterion.
- **Row ops own the lowered value.** The root of a ported kind is an
  `SfpStore`, not an `Sfpu { slot }`, so five places must treat a row op as
  "the tile it touches": `tt_lowered_storage` (phase 3 silently skipped the
  DRAM store → "DRAM write in compute"), `Kernel::dtype`,
  `Kernel::shape` (`compute_dtypes_and_rcs` maps *and* the shape walker), and
  `verify`'s dtype map. Forgetting one shows up as "no entry found for key"
  in a verify, or as the store surviving into the compute section.
- **A vectorized kernel is legal input to `tt_tile`.** See "Tiling: the search
  explores, `tt_tile` completes" for the contract: the access index's rank is
  the number of tile axes already present, the pass adds only the missing ones,
  and a group already counted in tiles is left alone. `sigmoid` arrives scalar
  (rank 0), `add`/`mul` arrive rank 1, a reduce keeps its rank-3 cube for
  `tt_reindex`.
- **`SIGMOID` ported (2026-11-01)**, gate `0/2560`, worst `1.19e-7` — the
  same number the builder path gives. The ported kind is the **Precise**
  program (`ops::sigmoid_program`, kind `0x12b`), which is what the lowering's
  fused-sigmoid pattern emits and what the gate runs; the notes had said the
  Approx twin (`approx::sigmoid_body`, `0x1d3`) — that one is only reachable
  through `MathMode::Approx`, and `MathMode` is never applied anywhere in the
  backend (`MathMode::lower` has no caller), so the Approx bodies are not on
  any gate's path. Port the Precise programs. The sequence needed:
  `SFPENCC` once, `abs`/`neg`, `exp_program`, `1+e`, `recip` (seed + two
  Newton steps + its four special cases, one `if_else`), the three `x < 0`
  scopes (one nested), the `SFPMOV`-negate forms, `SFPAND` (mask), `SFPIADD`
  (`isub_from`/`iadd`), `SFPSHFT`, `SFPARECIP`, `SFPSETSGN`, `SFPGT`/`SFPLE`/
  `SFPSETCC`, and the output tile as spill. One `SFPNOP` is required: after
  the two final Horner `SFPMAD`s, `SFPIADD` is a read the MAD stall misses.
- **`sum` ported (2026-11-01)**, gate `0/10`, worst `9.536743e-7` — the same
  number as the builder path. The gated shape is `TileDim::Row`: `reduce.rs`'s
  `accumulate(Sum, Cols, first)` (a tile copy — the first tile *is* the
  accumulator) then `finish(Sum, Cols)` (four faces lanewise per tile row,
  then the eight lanes' seven `SFPSHFT2` rotations). `tt_dataflow_select`'s
  Reduce arm emits it as row ops.
- **`tt_reindex` is traffic, not algebra (2026-11-02).** It rewrote the
  single-reduce template's load/store indices to the physical pitch
  (`row * 32 + lane`) but gated on `rop == Add`, so every `max` reduce stayed
  at the logical pitch: `[32x32x32] load → reduce max` instead of
  `[32x32] load → reduce max over the lane axis`. `tt_tile`'s output for
  `max_keepdim([1])` is **identical to `sum`'s**; the difference was only
  this gate. Dropping it to `Add | Max` makes the `max` gate green (exact,
  `0/10`) and exercises the ported `Row + Max` arm. The fold is not this
  pass's business — the index rewrite is the same for both.
- **`TileDim` names the result, not the folded axis — read it as "one value
  per ___".** `Row` = sum up over each of the rows (each row collapses to one
  value, so the answer is one value per row, carried in the result tile's
  first **column**); `Col` = the same over each column (one value per column,
  carried in the first **row**). `tt_reduce_program` agrees: `Row` →
  `accumulate(op, Axis::Cols)` (Burn's `dim = 1`), `Col` →
  `accumulate(op, Axis::Rows)`. If a sentence here ever says "`Col` reduces
  over the rows", it is wrong — that is `Row`.
- **A `[32]` rank-1 value is a column of values (one per tile row, first
  column), so folding it is `TileDim::Col` (2026-11-02).** The second kernel
  of `sum([0, 1])` / `max([0, 1])` reduces a `[10]` vector to a scalar, and
  after `tt_tile`'s loop split its IR is `r21: f32[32] = load vec32 x[r8];
  r15: f32 = reduce … over r8; store y[0] = r15` — **no group range at all**
  (the loop *was* the tensor's only axis), so `tt_tile` has nothing to pad or
  split and leaves a rank-1 access. The fix is in the lowering, not in
  `tt_tile`: phase 1 accepts a `[32]` DRAM load (the reader moves a whole
  tile either way, and the untilizer reads element `i` from row `i`, column
  0), and the `[32] → []` pattern maps to `TileDim::Col` = one value per
  column = fold down column 0, the values' column. The wrong choice is quiet
  rather than loud: `Row` compiles and runs, and since the input tile is a
  reduce's output (each row's value **replicated across its 32 columns**) it
  folds 32 copies of *one row's* value — `237.0044 = 32 × 7.406` for the sum,
  and `0.973` (= row 0's max, not the global `0.996`) for the max. That is
  exactly how the first attempt failed, and why a green-looking kernel is not
  evidence here: only the gate's numbers were. Phase 3 accepts a rank-0 store
  for the same kernel (a whole-tile reduce's first lane), and
  `compute_dtypes_and_rcs`'s "store value rank exceeds access rank" assert is
  skipped for a lowered (`Op::TT`) source: a native result is a whole `Dst`
  tile whatever logical rank it stands for.
- **`TileDim::Col` is live now; `TileDim::Scalar` still is not.** The two
  live shapes are `Row` (one value per row of a tile) and `Col` (one value
  per column — here, a whole `[32]` vector). A whole-tensor reduce never
  reaches `Scalar`: `Tensor::reduce_axes` (`tensor/reduce_ops.rs:40`)
  permutes the reduced axes last and folds **one axis at a time**, so
  `sum([0,1])` *and* the no-argument `sum_all()` (`axes = []` → every axis,
  `into_axes`, `shape.rs:47`) both become a `Row` kernel and then a `Col`
  kernel. The `sum_all` gate is bit-identical to `sum_scalar`, which is the
  evidence. `Op::Reduce` carries a single `reduce_axis`, so no IR can fold
  both axes of one tile — `Scalar` is structurally unreachable, not merely
  unused. `TileDim::Scalar`'s arm in the emitter therefore stays transcribed
  and unexercised — do not read a green reduce gate as covering it.
- **`TTOp::ReduceTile` is deleted (2026-11-02).** With every pattern ported,
  nothing produced it, so the variant, `tt_reduce_program`, `tt_dst_copy`,
  the codegen unit test that called them, and all ~20 match arms
  (`ops`/`mod`/`cost`/`verify`/`instr_sched`/`debug`/`tenstorrent`/`codegen`)
  are gone. The reduce arm now ends with `todo!("reduce {rop:?}/{dim:?} has no
  native row-op program")` for an unported combination instead of falling back
  to a structured op. `TileDim` stays (patterns + emitter + the public
  `kernel::TileDim`); `Op::Reduce` stays (the pattern input).
  **`kern/sfpu/reduce.rs` is now entirely orphaned** — its only importer was
  `tt_reduce_program` — and its module doc says so; deleting the file is the
  next cleanup, and until then it is *not* the source of truth for a reduce
  program (the emitter in `kernel/tenstorrent.rs` is).
- **The scalar reduce's scratch is an IR storage now.** It used to be
  `sec.fresh_slot() * 64` inside codegen, a row outside the declared register
  file that no op named. `SfpLoad`/`SfpStore` have to name a `Storage`, so the
  lowering declares it with `tt_new_dst`. The in-place prohibition stays safe
  because `tt_dst_reuse` frees a row only once `last_use` is past, and the
  accumulator's last use is the closing copy — *after* the zeroing stores that
  first mention the scratch, which is exactly what `accumulate_in_order`
  needs. Note that liveness, not the row's position, is what guarantees it
  now.
- **`SFPSHFT2` is a missed-stall reader too.** `rotate_row` is `Mod1 = 3`, one
  of `stall_modes::SHFT2_MISSED`, so every rotation that follows an `SFPMAD`
  needs an `SFPNOP` (the first one follows an `SFPMOV` and does not). The
  builder's `push` does this implicitly; a straight transcription of
  `finish()` drops six NOPs per group without changing the names, so compare
  words or the gate will only show it on other data.
- **`tt_reg_sync` owns the DST hand-off (2026-11-02).** A new pass between
  `tt_dst_reuse` and `verify`, and the only place that emits
  `Semset`/`Wait`/`Post` for tile registers. It walks the final physical tiles
  and derives, per tile, one *epoch* per write, then emits only the seams that
  exist: a **data** seam when a reader in another role follows a writer
  (writer posts after its last write, reader takes before its first read:
  `UNPACKED` for T0's writes, `COMPUTED` for T1's). A **reuse** seam (reader
  posts `FREE`, next writer takes) is a `todo!()` until a kernel writes a tile
  twice — see below. The semaphores are fixed (0/1/2), not per tile: each
  role's stream is in order, so posts and takes pair FIFO. Initial values are
  zero — a producer always posts before any take of its seam. `tt_verify` now
  rejects a wait/post whose semaphore has no `Semset`.
  - The matmul's hand-spelled `ready`/`free` protocol is **gone**: the pass
    derives the accumulator's edge from the `MVMUL`s themselves (the lowerer
    no longer carries `TtLowering.matmul`, and phase 3's pack is a plain
    move). `SETRWC` names the accumulator, every `MVMUL` after it writes that
    tile, and `SFPSTORE`/`SFPU` write their slot.
  - A `Post` from role 1 drains **both** Matrix and SFPU: a role names the
    stream, not the unit, and T1 owns both. (`tt_isa::sync::post_after` takes
    one `Unit`; codegen emits the two `STALLWAIT`s itself for role 1.)
  - Why it matters at all: a kernel with any `Semset` runs
    `Schedule::Concurrent`, i.e. all three roles start together; without a
    seam, T2 packs a tile T1 is still writing. Kernels with no seam stay
    `InOrder`.
- **The Matrix Unit leaves the thread's `Dst` row counter advanced.** An
  `SFPLOAD`/`SFPSTORE` address is `Imm10 + RWC.Dst`
  (`kern/sfpu/interp.rs` models it that way), the `MVMUL` address modifiers
  advance it, and the row-op path writes *absolute* rows (`dst_row(slot) +
  at`). So in a fused kernel every SFPU op after a matmul was shifted by one
  whole tile (measured: the pack read zeros at 64, the sigmoid was at 128).
  The matmul arm now ends with a `Setrwc { dst: slot }` to restore the counter;
  that is exact because the accumulator is slot 0 (`dst_val = row / 32` = 0),
  and a nonzero accumulator would need a true zero, which the IR cannot
  express yet (`Setrwc`'s operand is a storage).
- **`SETRWC`'s `DstVal` is four bits** (bit 14, 0..15), so one unit is a
  32-row half-tile: `row / 32`, not `row` (a row of 64 does not encode at
  all). The suspicion that `Dst` is ten bits of rows confused the *counter*
  (`RWC.Dst` is) with the *field*. Pinned by
  `codegen::tenstorrent::tests::setrwc_names_the_accumulator_in_dst_halves`,
  which renders a one-op kernel with a `Register` storage at index 1 and
  asserts the emitted `SETRWC` word's `DstVal == 2`; it also asserts that
  `dst_val(64)` is an encoding error. The remaining open item is the
  **restore**, not the base: with the accumulator *not* at tile 0 there is no
  way to say `RWC.Dst = 0`.
- **The reuse seam is a `todo!()`, not dead code** (2026-11-02). No gate writes
  a DST tile twice, so emitting `FREE` would be unexercised; `tt_reg_sync`
  panics with `todo!("... FREE reuse seam is not implemented")` when it needs
  one. The reserved index is still 2.
- **Two matmuls in one kernel do not lower yet**: the second `ZEROACC CLR_ALL`
  would wipe the first accumulator, and codegen refuses loudly. `matmul_add`
  (two matmuls, so the second accumulator is off tile 0) documents it; it is
  the `SETRWC`-base gate once the clear is scoped to the accumulator's rows.
- **A pack is not the end of the compute section.** codegen's
  `(None, Some(dcb), false)` pack branch ended in `return Ok(())`, which was
  invisible while a pack was the section's last op. With two packs (an
  accumulator output and a fused SFPU output) it silently dropped the second
  one — the compiled T2 program was half its IR. Now `continue`.
- **A matmul result is not materialized, even with several uses.** The pack
  `tt_materialize` emitted landed right after `SETRWC`, i.e. *before* the
  Matrix Unit ran, and its unpack would have been T0 reading a T2 pack with no
  seam. Every consumer now reads the accumulator's DST tile in place; the
  pass's `COMPUTED` seam orders the packs.
- **`Kernel::dtype` walks `Setrwc` to its `dst`** instead of returning
  `IDX_T`. That was the `matmul_sigmoid` `tt_verify` failure (the SFPU's DST
  slot was declared I64).
- **Next** (matching "Implementation order"): **item 2 — finish the port.**
  The remaining kinds — `FLOOR`/`TRUNC` (`round::body`), the comparisons, the
  integers, then `EXP`/`LOG`/`RECIP`/`TANH`/`ERF`/`GELU`/trig — each
  transcribed from its `Program` body and checked against the builder's words
  first (23 kinds still reach `program_for`); then `Sfpu { kind }`/
  `tt_sfpu_program`/`tt_row_remap` die with the `kern/sfpu` builder half.
  **Item 4 (loops → `softmax`)** is the next real feature. Matmul needs
  nothing: it is already instruction-level ops
  (`Unpacr`/`Setadczw`/`Setrwc`/`Zeroacc`/`Mvmul`/`Setc16`/`Setcfg`), and
  `kern::matmul` only supplies constants.

### Codegen IS the encoder; the backend is transport

- **Codegen owns encoding.** It walks the post-regalloc typed IR and emits
  the blob: the `match SfpOp -> tt_isa::sfpu::encode::*` dispatch lives
  there, and so do `SFPLOADI` expansion, the `REPLAY`/loop tables, L1 address
  resolution and param ordinals. `SFPENCC` and the `SFPNOP` after a MAD are
  **not** codegen's: the emitter writes them as ops (`SFPENCC` once per
  program, the NOP where the stall is missed), so they are visible to passes.
  There is **no separate encoder component in the backend**; there is no
  decode in the backend either. `tt_isa`'s word packers are called from
  codegen only.
- **Blobs are IR ops** — **done (2026-11-01)**: `TTOp::Unpack(Box<[u32]>)`,
  `TTOp::Math(Box<[u32]>)`, `TTOp::Pack(Box<[u32]>)` (boxed to keep `Op` small)
  are pushed by `render_tt` right after `TensixGrid`, in T0/T1/T2 order. The
  rendered op order is `Params`, `TensixGrid`, `Unpack`, `Math`, `Pack`,
  `Source(reader)`, `DmRead`*, `EndReader`, `Source(compute)`, `EndCompute`,
  `Semset`*, `DmWrite`*, `Source(writer)`; `TTDevice::compile` walks it.
  `tt_decode` is gone, and the runtime takes words (`&[u32]`), never an
  `InstructionDef`. Parity: the streams are the same objects codegen built
  (sigmoid T0/T1/T2 = 39/3753/68 words before and after) and the gates are
  bit-identical. `ZYX_TT_DUMP` prints per-role byte counts only.
  Reader/Writer are not blobs yet: they are the tile's movers (RISCV B/NC) and the
  launch hands them the page lists `compile` collected (§1). The launch walks
  whatever legs a kernel has; it never assumes exactly three.
- **Codegen is the last compilation; the backend's images are what run** (§2).
  The `Kernel` is the compiler's input: codegen turns it into per-core images, the
  backend uploads them once into the two tables (`images`, `programs`), and the launch
  only binds operands and issues. The boundary stream is IR (`TTOp::DmRead`/
  `DmWrite`, one per 4 KB page); `TTProgramDesc`, `TTNativeProgram`, `TTLoop` and
  `TTDstRow` are gone. `tt_decode` is gone and the runtime takes words (`&[u32]`).
  The simulator may decode internally; the transport must not.
- ~~Parity gate for the container swap~~ — done: see the "Blobs are IR ops"
  bullet above for the final op order and the parity evidence.

### Loops

- The Tensix coprocessor stream has **no branch**. Loop control is `REPLAY`
  (record a block, repeat it) and the mailbox loop table
  (`Code::Loop { start, len, count }`), nested under its field limits.
- Trip counts must be resolvable at launch (compile-time or launch arg), like
  grid extents. On-device data-dependent counts are not expressible in this
  model without the host re-pushing the body.
- Unrolling is a search/cost decision *before* the device epilogue, never a
  backend choice.

### Status (2026-11-02)

| test | state |
| --- | --- |
| `transfer_roundtrip` | green, 0/200 |
| `matmul` | green, 0/120 (0.0075 = TF32 in `Src`) |
| `sigmoid` | green, 0/2560 (worst 1.2e-7) — on the row-op path as of 2026-11-01 (Precise program, `sigmoid_program`) |
| `sum` | green, 0/10 (worst 9.5e-7) — on the row-op path as of 2026-11-01 (sum over a tile row's columns) |
| `sum_sigmoid` | green, 0/10 (exact) |
| `max` | green, 0/10 (exact) — `Row + Max` on the row-op path, as of 2026-11-02 |
| `add` | green, 0/2560 (exact) — first kind on the row-op path (`SFPMAD` + `LREG_ONE`) |
| `sum_scalar` | green, err 4.6e-5 — `Col + Add` (rank-1 kernel: one tile column), as of 2026-11-02 |
| `max_scalar` | green, exact — `Col + Max`, same kernel |
| `sum_all` | green, err 4.6e-5 (bit-identical to `sum_scalar`) — the no-arg `sum_all()` is the same `Row` then `Col` chain |
| `matmul_sigmoid` | green, 0/120 (worst 1.2e-4) — matmul + SFPU in one stream, as of 2026-11-02 (needed `tt_reg_sync` and the `Dst` counter restore) |
| `neg_sub_mul` | green, bit-exact — `NEG`/`SUB`/`MUL` row arms (`[10,256]`, the shape that exposed the `tt_tile` group-cap bug) |
| `tiling` | green, bit-exact — `neg`/`mul` on `[32,32] [64,64] [10,32] [128,128]`; locks the `tt_tile` contract (one-tile no-op, element vs tile-counted group, bare `g*32` row stride, multi-tile grid) |
| `unary_math` | green — `SQRT`/`RSQRT`/`RECIP`/`DIV` within `4 ulp`, `FLOOR`/`TRUNC` bit-exact; the special lanes (`0`/`±inf`/`NaN`) are not covered by this gate, but the `sfp_row_ops` A/B compares every arm bit-exactly against its builder program |
| `sfp_row_ops` (lib) | green — all 28 ported kinds, arm vs builder through `interp`, bit-exact on specials; covers `SIN`/`COS`, `POW`, the comparisons and the integer ops, which no ttsim gate can reach |
| comparisons, integer bit ops | ungated: a `Bool` output has no tile format and an `I32` input cannot unpack; verified bit-exact against the builder through `interp` on specials and shift amounts `>= 32` (temporary test, removed) |
| `softmax` | red: `Op::Loop` bails `tt_dataflow_select`, so `tt_verify` reports `missing EndReader` |
| `matmul_add` | red: a second matmul's `ZEROACC CLR_ALL` would wipe the first accumulator (`CLR_ALL after 64 compute op(s)`); the `SETRWC`-base gate once the clear is scoped |

CPU `01_unary` (64) and `04_reduce` (15) are green.

Build hygiene: `~/.cargo/config.toml` no longer sets a global `target-dir` (it did,
pointing at `/home/x/.cache/cargo-target`), so cargo uses each checkout's own
`target/`. Keep it that way — with the shared dir, `/home/x/Dev/rust/zyx2` (a second
checkout of this repo) could shadow this tree's test binaries, and a "green" run then
belongs to the wrong source tree.

Since the 2026-10-30 handoff:
- grid dispatch committed (`bae9c617`);
- the unpack header offset fixed (datums start at `(Base_address + 1) * 16`;
  passing the raw CB base shifted every tile by four F32 datums);
- the DST register file is the prologue's Register storages, and each one carries
  its allocation index in `Op::Storage::id`: `tt_dst_reuse` stamps them, codegen
  reads them (`row = id × 64`, or × 32 for 16-bit), instead of numbering slots by
  first use; `tt_verify` enforces single width, 16/32-bit only, 8/16 budget, and
  that the indices are distinct and inside the file;
- packs are `TTOp::Copy` (Register → CB), not `Op::Store`; the `Copy` traffic
  matrix is param→CB, CB→DST, DST→CB, CB→param;
- CSE/DCE moved before instruction selection; `tt_verify` rejects SSA
  leftovers;
- `tt_lower_llks`/`tt_reg_alloc` renamed to `tt_dataflow_select`.

### Implementation order (agreed)

0. ~~**Restore `Pool::TT` as device DRAM**~~ — **done (2026-11-01)**, see §1.
1. ~~**Blob container + parity.**~~ — **done (2026-11-01)**: `Unpack`/`Math`/
   `Pack` are rendered ops, the launch reads them from the kernel,
   `native.compute`/`native_compute`/`compute_native` and `tt_decode` are
   deleted, and the runtime takes words. No env-var path was needed: the
   blobs are the same vectors codegen already built, so parity is by
   construction and the gates are bit-identical.
2. **Typed SFPU port.** The opset is ISA-level and row-addressed; the
   per-kind sequence bodies move into the pass; codegen only encodes.
   Settled shape (2026-11-01):
   - **Ops**: `SfpLoad` / `SfpStore` (Register tile + the intra-tile `Dst`
     address, LREG, format), `SfpConst` (LREG immediate, the `SFPLOADI`
     pair when the low half is non-zero), `Sfp` (the ISA ALU form: opcode +
     LREG operands and mods). One op per instruction. Per-operation
     variants (`SfpAbs`, `SfpRecip`, …) are added only when a pass must
     match on that operation, and the reason is written down.
   - **Where ids come from**: `tt_dataflow_select` lowers and assigns the
     tile ids as it goes (primitive allocation, one fresh tile per result);
     `tt_dst_reuse` is the real allocator and only rewrites which tile an op
     names (see the storage decision).
   - **The sequences** (the ~24 kinds the pass can select, plus the row
     loop, const staging, `if_`/`if_else` flag scopes, NOP-after-MAD) are
     emitted by `tt_dataflow_select` / a pass it drives. `kern/sfpu`'s
     `Program` / `Instruction` / `Code` / `LoopForm` go as the
     encoder-side layer; codegen calls `tt_isa`'s encoders directly.
   - **`Sfpu { kind }` is deleted in this cut**, together with the
     `TtEmit::Sfpu` route — i.e. when the last kind ports. It is still the
     fallback for the ~23 kinds that have not (2026-11-02).
   - ~~**`ReduceTile` stays until item 5.**~~ **Superseded 2026-11-02: it is
     deleted.** The reduce is row ops in this pass, keyed on *shape*
     (`[32,32]→[32]` = `Row`, `[32]→[]` = `Col`, `[32,32]→[]` = `Scalar`, the
     last unreachable), not on `reduce_axis`. Every reduce gate is green
     (`sum`, `sum_sigmoid`, `max`, `sum_scalar`, `max_scalar`, `sum_all`);
     see the SFPU-emission section for the two gotchas (`TileDim` names the
     result; a rank-1 value is a tile *column*).
   - **Acceptance is the tests**, not word parity: identical words where the
     old builder already unrolled, interpreter-equal (`kern/sfpu/interp.rs`,
     which stays as the oracle) otherwise. The interpreter is the only check
     for kinds no gate touches (`INT_AND`, `REM`, `COS`, …).
   - **Size is not a reason to abstract**: row-level ops are allowed to be
     numerous; `REPLAY` is an encoder/peephole decision on that stream.
   - `Op::Source` stays for C/CUDA/SPIR-V/PTX — render just stops emitting it
     on the TT path.
   *Done 2026-11-01:* the storage model (`Op::Storage::id`, see the decision
   below) — the allocator stamps the physical index and codegen reads it, so the
   DST file is IR rather than a codegen side map; the `Sfp*` opset and encoder.
   *Done 2026-11-02:*
   - Kinds on the row-op path: `BOOL_NOT` (raw-bit XOR), `ADD` (`SFPMAD` +
     `LREG_ONE`, bit-exact), `SIGMOID` (`ops::sigmoid_program`, the Precise
     body; gate `0/2560` at `1.2e-7`), `NEG`/`ABS`/`SUB`/`MUL`/`INT_NOT`
     (`ABS` is the builder's `x & 0x7fff_ffff`, not `SFPABS`; gate
     `neg_sub_mul` for `NEG`/`SUB`/`MUL`), `SQRT`/`RSQRT` (`sqrt_program`),
     `RECIP` (`Program::recip`), `FLOOR`/`TRUNC` (`round::body`), `DIV`
     (`scale_large_divisor` + `recip` + `divide`; gate `unary_math`), the
     comparisons (`compare_body`), the integer bit ops (`integer::body` ops
     3–7), `SIN`/`COS` (`sin_cos_program`, three standard spill tiles), and
     `POW` (`log_program` + multiply + `exp_program` + `pow_fix`; no frontend
     path, A/B only). Every arm is its own method and emits `SFPENCC` once
     when it predicates.
     `tt_sfp_sequence` carries the shared emitter pieces: `SfpConst` staging,
     nested `if_`/`if_else`, `abs`/`neg`/`mov`/`mad`/`nmad`/`mul`/`sub`/
     `and`/`or`/`xor`/`not`/`iadd`/`isub_from`/`shl`/`shr_by`/
     `approx_recip`/`setsgn`/`SFPGT`/`SFPLE`/`SFPSETCC`, and the
     NOP-after-MAD rule.
   - *Still to come in this item:* `REM` (see the handoff), then
     `Sfpu { kind }`/`TtEmit::Sfpu`, `tt_sfpu_program` and `tt_row_remap` go,
     and `kern/sfpu`'s `Program`/`Code`/`LoopForm` half with them
     (`ops::kind_sfpu` must move to `kern/kind.rs` first).
   - The **reduce** moved off the builder entirely and `TTOp::ReduceTile`
     is deleted (`tt_reduce_program`, `tt_dst_copy` and its unit test with
     it). `kern/sfpu/reduce.rs` is deleted too (2026-11-02, with its
     `pub mod reduce;` line); the emitter in this file is the truth.
   - **`tt_reg_sync`** exists (see the SFPU-emission section) and the
     matmul's hand-spelled `ready`/`free` is gone; `matmul_sigmoid` is green.
   - *Still to come in this item:* the ~23 kinds still reaching `program_for`
     (`FLOOR`/`TRUNC`, the comparisons, the integers, `EXP`/`LOG`/`RECIP`/
     `TANH`/`ERF`/`GELU`, trig), then `Sfpu { kind }`/`TtEmit::Sfpu`,
     `tt_sfpu_program` and `tt_row_remap` go, and `kern/sfpu`'s
     `Program`/`Code`/`LoopForm` half with them.
3. **The `Dm` instruction stream (§2).** Flat `TTOp` variants, one per mover op —
   `DmRead`/`DmReadTransposed`/`DmReadBroadcastCol`/`DmWrite`/`DmFill`/
   `DmCopyWords`/`DmIndexPick`, `DmKernel` (no program payload) / `DmWait` —
   emitted by codegen into the reader (B) and writer (NC) sections. (The ISA has
   no tilize/untilize or host read/write mover op, so those names are gone: the
   unpacker tilizes, and `Pool::TT` is device DRAM with no host data path.
   `DmBarrier` is multi-core sync and belongs with multi-tile dispatch, not with
   the serial stream. The async family moved to item 7.) Codegen expands each move into
   one `Dm` op per 4 KB page (symbolic chunk-relative operand), so the number and
   order of entries never depend on the launch; `EndReader`/`EndCompute` are
   consumed here. `TTNativeProgram` (`TTDmMove`, `TTLoop`, `TTSemaphoreInit`,
   `TTDstRow`, the CB `l1` table) and `TTProgramDesc` die as the launch stops
   reading side tables, and the launch becomes substitution + issue only.
   **The movers stay serial here** — one list per direction per launch, reads
   then writes, no overlap with compute. That is deliberate: async CBs are step 7.
   Follow-on in this step: move the bytes into the `images` slab (§2) —
   `Unpack`/`Math`/`Pack` holding an `ImageId` instead of inline
   `Box<[u32]>` (the inline boxes of step 1 are the intermediate state). That is
   what makes a launch args-only. The reader/writer *lists*, by contrast, are not
   images and never will be under this design: `mover_list` writes absolute
   channel/offset words into the mover's **L1** list ring, and a mover's list
   cannot be loaded by that mover out of DRAM. That leg moved to item 7, where
   the async pipeline decides the final list shape anyway.
   *Done 2026-11-01:*
   - **Two tables + residency** (§2 "Implemented"): `Image`/`ImageId`, the DRAM
     upload at compile, `resident[tile]` over the five slots, the B-mover load
     path, the firmware boot as a first miss, `release`; `TTProgram` holds
     `sequence`/`lens` instead of role words.
   - **The boundary stream is IR**: `TTOp::DmRead(Box<TTDmRead>)` and
     `TTOp::DmWrite(Box<TTDmWrite>)`, one per 4 KB page, emitted by `render_tt`
     (`tt_pages`), with `param`/`l1`/`len`/`base`/grid strides on the op and
     `narrow` on the write. `TTOp::ProgramDesc`, `TTNativeProgram`, `TTLoop`,
     `TTDstRow` are gone; `compile` walks the rendered kernel and fills its own
     `reader`/`writer`/`sems`/`n_inputs`/`n_outputs`/`gws` table. The launch
     substitutes `param -> chunk` and the grid point and issues — no side table
     is read.
   - **The other mover ops exist** (2026-11-01): `DmReadTransposed`/
     `DmReadBroadcastCol` (`TTDmTileRead`), `DmFill`, `DmCopyWords`,
     `DmIndexPick`, `DmKernel` and `DmWait` are flat `Box`ed `TTOp` variants
     mirroring their ISA entry words. Nothing emits them yet; they are the
     vocabulary the stream will be written in, and every op match treats them as
     operand-free structural entries.
   - **`LIST_MAX` is a compile-time invariant** (2026-11-01): `tt_verify` counts
     the mover entries of the reader section (B) and the writer section (NC)
     separately and rejects either over `dm::LIST_MAX` (512), and rejects a mover
     op inside the compute section. `mover_list` no longer splits a long stream
     into several lists; a stream past the ring **panics** instead, so a bypassed
     check is loud rather than a silent slowdown.
   - **Still to come in this item:** emitting `DmKernel`/`DmWait` from codegen and
     letting `compile`/the launch issue them (the launch still starts the roles
     and waits around the page lists itself), and `Unpack`/`Math`/`Pack` naming an
     `ImageId` instead of inline words.
4. **Loop lowering** (`REPLAY` for bodies that fit its 32-word buffer, the
   `mailbox::loops` header only for longer ones) for `softmax`; multi-tile per
   core. A trip count that does not fold to `Const` is a TT compile error (§2).
5. **Fixes** (the no-gate-needed list):
   - ~~**`Setrwc` dtype**~~ — **done 2026-11-02**: `Kernel::dtype` walks to
     `dst` instead of returning `IDX_T`. That cleared the `matmul_sigmoid`
     verify failure; the kernel then exposed the three bugs recorded under
     "SFPU emission" (the misplaced materialize pack, the pack branch's early
     `return`, and the Matrix Unit's leftover `Dst` row counter), and it is
     green now.
   - ~~the reduce kind keyed on `reduce_axis`~~ — **superseded 2026-11-02**:
     the patterns key on shape, and `Row`/`Col` cover both live cases.
   - **`tt_tile`'s grouped index math** (32 tiles per step where one is
     meant) — the single-tile reduces are green, but no gate runs a
     *multi-tile* grid, so treat this as still open until one does.
   - **the tile state `matmul` leaves behind**, which makes a later SFPU
     kernel in the same process read zeros from DST (`sigmoid` → 0.5) while
     copies still work — proven not to be the memory path (host-shadow A/B),
     so it is device/tile state from the Matrix Unit program. `matmul_sigmoid`
     is one kernel, so this is about *separate* kernels in one process.
6. **L1 spills** in `tt_dst_reuse` (needs the scheduling story: a mid-T1
   spill needs T0/T2 and the roles currently run in order).
7. **Async CB pipeline (deferred; last).** CBs become multi-tile rings with
   credits so the reader streams into a CB while T0/T1/T2 consume it and T2's
   packs drain through NC at the same time — double-buffered, the shape
   origin/main already had. Emitted as `Dm` entries (`op::CB` with stream 0/1,
   `op::PAIR`/`PAIR_CALL` so B starts NC, `op::RELEASED`, `WAIT`/`BARRIER`) plus
   per-tile role programs written against those credits, and the packer/unpacker
   synchronisation between roles that origin/main already had. Requires: more
   than one tile per core per launch, the tile loop inside the stream instead of
   in the grid dispatch, and role programs built for per-tile credit
   synchronisation. Until then the movers stay serial (step 3) — this is the step
   that recovers throughput, and it is explicitly not a prerequisite for anything
   above.
   Moved here from item 3 (2026-11-01), because both are properties of the
   pipeline, not of the serial stream:
   - **The async `Dm` family**: `Pair`, `PairCall`, `Cb` (stream 0/1 credits),
     `Shared` (NC runs B's writer entries), `Released` (B waits on NC/T2 output
     credits) — as flat `TTOp` variants, emitted by codegen like the serial ones.
   - **The reader/writer lists as `images`.** Not the naive "list is a blob the
     B mover loads": `mover_list` writes absolute channel/offset words into the
     mover's **L1** list ring, and a mover cannot load its own list out of DRAM.
     The async pipeline settles where the list lives (ring in L1 vs. a DRAM image
     streamed by `CALL`, which the ISA supports via `TRACE_CHUNK`), so
     `TTProgram::sequence` grows to five legs here or not at all.

### Decisions / non-negotiables

- `TTOp` is the instruction. One representation per phase: typed ops before
  codegen, words after; no third instruction type.
- Codegen is the encoder; the backend never decodes, re-encodes, or names an
  `InstructionDef`.
- No `Program` in the lowering. The macro math stays; the object goes.
- zyx UOps are `Exp2`, `Log2`, `Reciprocal` (plus the generic unary/binary);
  `exp`, `log`, `sigmoid` are frontend compositions, never backend ops.
- No LLK layer, no fused user-op names in the backend.
- The IR may be as low-level as the machine, and no larger than the passes
  need: op width and op count are not design constraints (`Op` is 24 bytes; a
  variant is boxed only to keep it so). An interpreter over it is a viable
  second oracle.
- Placement lives in the IR (rows, L1); no side tables.
- **A storage has two levels: what it is, and where it lives.** `Op::Storage`'s
  `dtype`/`scope`/`len` are the declaration (the object's type and size — CUDA's
  `__shared__` array, a DST tile, a CB ring); `id` is the allocation index, `0`
  until the allocator runs. Before allocation a storage is only "a request for an
  object of this shape", so all such requests are `Eq`/`Hash`-equal, exactly as
  if `id` did not exist; the allocator assigns one id per physical object the
  device can hold and, from then on, `id` is part of the storage's identity, so
  post-allocation CSE cannot merge tile 3 into tile 7. Backends that expose the
  file (Tenstorrent) read `id` as the physical slot; backends that hide it
  (CUDA/CPU) never read it and their compiler colours behind the same
  declaration. One allocator pass, parameterised by device capacity:
  `Register` = 8/16 tiles on TT, unbounded on CUDA/CPU; `Local` =
  `local_mem_size`; `Circular`/`Global` are declared, not allocated, and keep
  `id = 0`. The pass also enforces liveness ≤ capacity and fails loudly.
  CSE is the reason `id` is in `Eq`/`Hash` (like `Op::Param`'s buffer `id`,
  canonicalised before cache lookups): an identity field added later cannot
  repair a CSE that already merged two registers. Status (2026-11-01): wired —
  `tt_dst_reuse` stamps one index per surviving physical Register storage in
  prologue order, codegen reads `id` as the slot (`sec.slots` is left for the
  non-storage intermediates), and `tt_verify` rejects duplicate indices or one
  past the file. `Circular`/`Global` keep `id = 0`.
- Codegen is the last compilation step. A kernel that has been through the
  passes runs on the device as it is; the dispatch fills operands and issues,
  and decides nothing (§2).
- Compilation's output is two tables (§2): `images: Slab<ImageId, Image>` (one
  image per core, its bytes in device DRAM) and `programs: Slab<DeviceProgramId,
  TTProgram>` (a `sequence` of image ids plus the arg-shaped metadata). The `Kernel`
  is the compiler's input and does not run. Per-tile residency (five slots: B, NC,
  T0, T1, T2) is backend state keyed by `(ImageId, generation)`; there is no "boot"
  step and no hidden program structure.
- Loops are `REPLAY`/loop-table entries with launch-time counts; unrolling is
  a search decision before the device epilogue.
- `Pool::TT` is device DRAM. Host staging is only upload/readback.
- The DRAM↔L1 boundary belongs to the tile's data movers (RISCV B reads,
  RISCV NC writes), never to the host. A launch hands the movers lists; no
  tensor byte crosses the host between placement and readback. Only a 16-bit
  output's narrowing is the host's, and only because the pinned ttsim refuses
  the packer's format conversion (silicon does it on the card).
- The movers run **serial** with compute until step 7: reads, then the roles,
  then writes, one list per direction. Async CBs with double buffering are
  origin/main's shape and stay deferred until the rest lands.

## HISTORY (2026-10-30) — superseded by the handoff above

> Everything below this line is history from the 2026-10-30 handoff. The
> `Pool::TT` description here (host-memory DRAM) was the regression that
> CURRENT STATE §1 documents and has since undone.

### Layout and dependencies

`src/backend/tenstorrent/` is the whole TT backend, ported in-tree from
`metalium-rs` (attribution and the Apache-2.0 note are in `mod.rs`):

| path | what |
| --- | --- |
| `mod.rs` | `Dev::TT` backend: pools (device DRAM, see §1), `TTDevice`, program staging (`TTNativeProgram`/`TTDmMove`/`TTDstRow`), launch |
| `ttsim.rs` | owner thread over the simulator + `run_native` |
| `dev/` | ported `tt-device`: device, TLB windows, config writes, core control, DRAM, trace, telemetry, arc, fencing |
| `kern/` | ported `tt-kernels` subset: `runtime` (runner), `datapath`, `sfpu`, `l1`, `loops`, `code`, `kind`, `elem`, trimmed `matmul` |
| `sim/` | ported `tt-ttsim` + `tt-ttsim-sys` (dlopen wrapper) |

Dependencies: **`tt-isa` and `tt-firmware-images` stay external** (generated
encoding tables; RISC-V role images). `tt-kernels`, `tt-device` and
`tt-ttsim` were removed; `libc` (fork helper) and `libm` (SFPU fits) added.
There is deliberately **no `tenstorrent` feature** — TT is always compiled in
(~16 s clean builds). The simulator blob is a vendor binary: set
`TT_TTSIM_LIB`, or put it at `vendor/libttsim_bh.so`.

Known layering wart, deliberately deferred: `kern/` sits under `backend/`
but is imported by `kernel/tenstorrent.rs` and `codegen/tenstorrent.rs`
(inverted edges). The durable split is builders → `codegen/tenstorrent/`,
and the few constants the lowering needs (`MATH_AM_*`, SFPU kind ids) →
`kernel/ops.rs`.

### Tests

`tests/23_tt_lowering.rs` is the only TT gate (the old `24_tt` was folded into
it and deleted). Everything runs in **F32 on purpose**: `ttsim` models only
`(0,0)`, `(0,4)` and `(8,8)` unpack-to-`Dst` (`ttsim-divergence.md` rows 31
and 87), so a 16-bit elementwise kernel cannot run on the simulator at all.
There are **no implicit conversions**: a BF16 compute kernel is a loud
codegen error ("needs an F32 DST tile"), and the user wants it kept that way.

| test | state |
| --- | --- |
| `transfer_roundtrip` | **green**, 0/200 (exact) |
| `matmul` | **green**, 0/120, worst 0.0075 (TF32: the MU truncates F32 in `Src`) |
| `sigmoid` | dispatches 8 cores now, but **2532/2560 wrong**, worst 0.23 |
| `sum` | runs, **10/10 wrong**, worst 3.43 |
| `sum_sigmoid` | 1/10 bad, worst 0.0025 (nearly right — different drain path) |
| `matmul_sigmoid` | `native unpack ... got I64` (pattern bug, below) |
| `softmax` | `Op::Loop` still bails `tt_lower_llks` |

CPU suites `01_unary`…`05_misc` are green. `08_graph`'s three
`custom_kernel_*` tests now autotune on `TT(0)` because TT is always
compiled in and fail on a 44-byte param — the user disables TT in config for
those; **do not change those tests**.

Run one TT gate (simulator only, no board):

```bash
cd zyx
TT_TTSIM_LIB=/home/x/Dev/rust/metalium-rs/vendor/libttsim_bh.so \
  AGENT=1 cargo test --test 23_tt_lowering matmul -- --exact --nocapture
```

`ZYX_DEBUG=8` dumps the winning IR, `ZYX_TT_DUMP=1` dumps the role words.
`libttsim` calls `_Exit` on a BAR contract violation (no unwinding, no panic
hook), so a test that vanishes without a failure line is a decoder violation,
not a panic.

### Open work, in the order it should be attacked

1. **Tile payload mapping** — this is what blocks `sigmoid`. Grid dispatch is
   in and executes 8 points, but the result is wrong *including the first
   point at DRAM offset 0*, so the bug is inside one tile: the unpack (a flat
   1024-datum run into DST rows 0..63), the SFPU row walk, and the pack
   (rows 0..63 back out) disagree on lane/face order. `transfer_roundtrip`
   passes (tilize/untilize agree) and `matmul` passes (the MU handles faces),
   so this pair has never been validated. Next probe: a **copy kernel with no
   SFPU** — unpack → pack — compared against `transfer_roundtrip`.
2. **`tt_tile`'s grouped index math is wrong.** The folded reader index
   carries a 32-tile stride per grid step (131072 B where one tile is meant).
   `tt_slice` in codegen ignores it for strides (base comes from the index's
   const term, strides from the grid map: 1024 datums × higher-axis extents),
   but the index math itself still needs fixing or deleting.
3. **Reduce axis.** The phase-2 pattern table picks
   `Reduce(TileDim::Row)` from the *result shape* and never looks at the IR's
   `reduce_axis`; `TileDim::Row` maps to `Axis::Cols`. That is `sum`'s
   10/10 wrong. Needs a `Col` entry keyed on the reduce axis, not on shape.
4. **`matmul_sigmoid`.** The sigmoid pattern's `Pat::Const` matches an
   *index* constant, so an I64 storage reaches the unpack. Needs a
   dtype-aware const pattern.
5. **`softmax`.** `Op::Loop` bails the pass; loops (multi-tile per core) are
   not lowered yet.
6. **DST rows and L1 placement are still decided in codegen.** The row
   *numbers* are correct now (`dst_row = slot × 64`, or × 32 for 16-bit;
   `Setrwc.DstVal = row/32`; `Mvmul.dst_row` offset by the accumulator's row;
   the packer offset is a row; `native.dst` is rows), but no pass can see
   them. The ordered step was a `TTOp::DstBase { storage, row }` placement op
   emitted by `tt_dst_reuse`, checked by `tt_verify` (unique, aligned,
   within 8 tiles), and read by codegen. `tt_place_cbs` (bump allocator over
   `tt_isa::l1::DATA`) is the same story for L1.

### Grid dispatch (implemented, uncommitted)

The program is **tile-uniform**: identical L1 layout and instruction streams
on every core, so only the DRAM slice differs.

- `TTDmMove { param, l1, bytes, narrow, base, gstride0, gstride1 }`;
  `tt_move_offset(mv, gidx0, gidx1)` gives a point's byte offset.
- `TTDevice::launch` runs one program per grid point over
  `tt_isa::noc::grid::Tensix::FULL.tiles::<Noc0>()` (row-major; `gidx0` = row,
  `gidx1` = column), staging and reading back that point's slices, with bounds
  checks in all three narrowing arms. The 1×1 restriction is gone.
- `NativeRun.tile: (u8, u8)`; `ttsim.rs` runs the requested tile instead of the
  old hardcoded `(3,4)`.

Reference for the concurrent version: metalium's `Session` (`TileChoice`,
`Unit` with resident roles + its own data mover, wave dispatch,
`step20_many_tiles`). That layer was pruned on purpose; it is the model to
copy when sequential runs become waves.

### Why the merge, in hindsight

The tt-metal path had grid dispatch, 16-bit staging and multi-tile *for free*
(tt-metal's launcher and LLK supplied them), and its cost was the per-op
CB/pop/lock discipline. The native path owns the stack and has to build those
capabilities: grid (done), tile payload (open), BF16 staging (decided out —
F32 tests, no implicit conversions), multi-tile and concurrency (open).
Grading on green tests alone, the tt-metal path was ahead; the merge was
bought for ownership.

Commits: `82f441f7` merge, `590d4008` "reduce the tt dependencies, rewrite
the backend", `03246130` "tt tests fp32 only", then the grid work
(uncommitted at the time of writing).

## Goal

`zyx` compiles a whole `Kernel` into native Tenstorrent programs and runs them
on the device layer zyx now owns: `src/backend/tenstorrent/` (`dev/`, `kern/`,
`sim/`, ported from `metalium-rs`), with `tt-isa` and `tt-firmware-images` as
the only external TT crates. Constraints:

- no C++ shim and no tt-metal link;
- no `Session` between zyx and the device;
- no DRAM or host round-trip *inside* a zyx kernel: intermediates stay in L1
  (or `Dst`), and only the kernel's inputs and outputs cross the DRAM/L1
  boundary.

`Dev::TT` / `Pool::TT` are the only TT entry points.

## Hard invariant: NO DRAM round-trip inside a zyx kernel

This is non-negotiable and overrides convenience paths in the lowering:

- One zyx `Kernel` is one program: one reader list, one compute role set, one
  writer list, launched once.
- DRAM is touched only at the kernel boundary:
  - inputs: `Param Global` / `Param GlobalMut` reads by the reader;
  - outputs: `Param GlobalMut` writes by the writer.
- Every intermediate is an SSA value in DST or in an L1 Circular buffer. It is
  never stored to a DRAM param and loaded back.
- A `TTOp::Copy` to/from a DRAM `Param` is legal only in the reader/writer
  sections. A DRAM copy or store in compute is a lowering bug.
- A lowered intermediate must not be packed to a writer CB unless that value
  is a kernel output. Intermediate CBs are L1-only and consumed by the next
  compute op; they are never drained to DRAM.
- No per-op launch, no per-op `Session`, no host staging between compute ops.
- The verifier must reject an intra-kernel `Store` to a `GlobalMut` param
  followed by a `Load` from that param, instead of silently hoisting the load
  into the reader section.
- `23_tt_lowering::sum_sigmoid` and `23_tt_lowering::matmul_sigmoid` are the
  acceptance tests: one kernel, no DRAM traffic between the two compute ops.

## DST registers: SSA lowering, physical reuse in a separate pass

DST is a small physical register file (8-16 tile rows per core), not a
logical SSA namespace. The design separates the two concerns:

- **Lowering is SSA over DST.** Every native op writes a *fresh*
  `Op::Storage { scope: Register }` and the `TTOp` carries that storage
  OpId as its `slot`. Lowering never allocates or frees a physical row.
- **`tt_dst_reuse` is the only physical allocator.** After lowering it
  computes the last use of every logical Register storage over all
  operands (compute ops, packs, unpacks), linear-scans a free list keyed
  by dtype, creates the physical Register storages in the prologue, and
  rewrites every logical operand to its physical row. Dead inputs are
  released before the output is assigned, so an in-place native op can
  coalesce its input and output onto one row.
- Codegen only maps each physical Register storage OpId to a DST row; it
  does not decide liveness.
- Use counts for deciding *when a value needs L1* come from
  `Kernel::compute_dtypes_and_rcs`, snapshotted before lowering.
- Matmul writes a fresh DST row; its operands are CBs, not DST.
- Only a kernel output is packed: phase 3 packs a DST result into a writer
  CB and drains that CB to DRAM. An intermediate is never packed *unless*
  it has multiple uses (next section).

### Double use is a CB roundtrip (hardware fact)

Tensix is a dataflow machine: DST is single-use scratch. A *computed*
value with more than one consumer cannot stay in one DST row, because the
first consumer may overwrite it and the row cannot be read by a second
consumer at an arbitrary later time. Such a value must be materialized in
L1:

```
producer -> DST x
pack x -> CB1                 // Store, Register -> Circular
every consumer unpacks CB1    // Copy, Circular -> Register
```

The first consumer's unpack is redundant with `x` on purpose. Lowering
stays simple (one pack, one unpack per consumer); a later optimization
pass can prove the first unpack is the producer's row and delete it.
A value that is already a CB (a `Load` from a reader CB, or a storage)
needs no extra pack. A computed value with exactly one consumer stays in
DST.

Current implementation status: phase 2 lowers recursively from the DRAM
output stores through one biggest-first pattern list; `tt_dst_reuse` runs
after it and assigns physical rows. Phase 3 drains a materialized CB
directly, or packs a single-use DST result to a writer CB. The old
`TtRegAlloc`/`TtPatKind` forward-walk design is gone.

## Phase-2 pattern list

`tt_lower_llks` keeps one list, biggest first:

| Pattern | Emits | Inputs |
| --- | --- | --- |
| sigmoid cone (rank 2, rank 1) | `Sfpu { kind: SIGMOID, arity: 1 }` | DST |
| matmul (both orders) | `MatmulTiles { cb_a, cb_b, slot }` | CBs (hardware), DST output |
| sum/max reduce (row, scalar) | `ReduceTile { rop, kind, input, slot }` | DST -> DST |
| generic unary `UOp` (rank 2, rank 1) | `Sfpu { kind, arity: 1 }` | DST |
| generic binary `BOp` (rank 2, rank 1) | `Sfpu { kind, arity: 2 }` | DST |

The table is built from `UOPS`/`BOPS` plus `uop_kind`/`bop_kind`; an op
with no native kind is filtered out and stays high-level. Adding an op is
a table/kind entry, not a new variant or match arm.

Lowering is **recursive and root-first**, not a forward walk. It collects
the DRAM output stores, then calls `tt_lower_value(root)`:

1. If the value is already a native op / storage / materialized CB,
   return its storage.
2. If it is a `Load`, return the reader CB (phase 1 already published it).
3. Otherwise try the pattern list in order. For the first pattern that
   matches, recursively resolve its bound inputs (`tt_resolve_dst`), then
   emit the native op and return its fresh DST storage.

Recursion is what lets a fused root (sigmoid, matmul) win over its inner
ops: the root is tried before its inputs are lowered. It also makes the
silu cone fall out naturally as `Mul(x, sigmoid(x))` through the generic
binary `Mul` and fused `Sigmoid` entries; there is no silu pass and no
silu variant.

`tt_resolve_dst` is the only CB -> DST conversion: if the resolved
storage is Circular, it emits `TTOp::Copy` (Circular -> Register) before
the consumer. Matmul is the exception: its operands stay CBs because the
Matrix Unit reads SrcA/SrcB from CBs. Every other compute op consumes
DST.

### Generic SFPU op

tt-metal exposes a huge kernel-op surface, not just silu/sigmoid/sin/cos.
Per-op `TTOp` variants and per-op match arms do not scale and are
forbidden. The native op is:

```rust
TTOp::Sfpu { kind: u32, a: OpId, b: OpId, slot: OpId }
```

`kind` is the metalium `kind` / `sfpu::ops::kind_sfpu` code; `b` is
`OpId::NULL` for unary. Adding an op is a table entry, not a variant.
Silu is `Mul(x, sigmoid(x))`; the generic `Mul` and fused `Sigmoid`
entries lower it as two DST ops (with a CB roundtrip only if the sigmoid
result is used more than once).

Matmul stays the hardware special case: `MatmulTiles { cb_a, cb_b, slot }`
keeps CB inputs. This is not a lowering shortcut and must NOT be
"normalized" to DST operands: the hardware matmul reads SrcA/SrcB from
CBs. Moving its operands to DST would break the Matrix Unit. Reduce stays
a structural tile op (`ReduceTile { input, slot }`).

## Directives (condensed; every directive's substance is kept)

### Scope and shape

- Native-Rust `metalium-rs` backend exposed as `Dev::TT` in
  `tenstorrent.rs` (not `Dev::Metalium`). No C++ shim, no tt-metal link, no
  `Session`.
- Compile a whole zyx `Kernel` into at most three native instruction
  streams: reader, compute, writer. Data movement only = one reader/writer
  stream pair, no compute roles.
- Keep origin/main's compiler model verbatim; only the C++ leaf emission
  and the C++/tt-metal transport are replaced.
- No separate Metalium IR and no `match_graph`/AOT graph-op shortcut. It
  must be a real zyx backend.
- No TT feature flags. The `tenstorrent`/`metalium` features, the old
  tt-metal backend, its codegen/shim, and useless passes are gone.
- No subgraphs and no fusion step: everything is already fused inside the
  zyx kernel. Native codegen emits compute ops in IR order; no `l1::fuse`,
  no run merging, no kernel splitting.
- Reader = DRAM -> L1; compute = L1 -> L1; writer = L1 -> DRAM. No
  intermediate crosses DRAM or the host inside a kernel.
- Preserve zyx's CB handling, CB synchronization, math/pack DST locks, and
  verifier behaviour. The native builders carry the behaviour; the tt-metal
  API ops themselves are deleted.
- Table stakes: a working port within one week, not necessarily fast.
- This is mechanical: follow `metalium-rs`. The old C++ tt-metal path
  passed `23_tt_lowering::matmul` and handled CBs, sync, and locks. That
  test is the only known green TT baseline. Not new semantics.

### Memory and layout

- `Pool::Host` is host RAM. `Pool::TT` is device DRAM, not
  host-addressable.
- Host boundary: tilize Host row-major -> TT DRAM tile layout; untilize TT
  DRAM tile layout -> Host row-major; TT -> TT is a physical byte copy.
- Inside TT DRAM the layout is always tile layout, never row-major. No
  face-order -> row-major -> face round trips; keep hardware face order
  and convert dtype only.
- `Op::Storage { scope: Circular }` gets an L1 address. `Op::Storage
  { scope: Register }` is DST tile registers inside the cores, never L1.
- Not constrained by old tt-metal semantics such as 32x32 tiles for every
  op.

### Instruction set and IR

- If DST locks or CB ops are not in the actual instruction set, delete them
  from `TTOp` and rewrite the passes to emit only the ops that are actually
  needed.
- `TTOp::Copy` is the only thing that copies CB -> DST. It also carries
  DRAM <-> CB traffic and the CB -> DRAM drain.
- Compute results live in DST. Other elementwise ops read DST directly:
  zero CB round-trips.
- Follow each native instruction's actual calling convention; do not force
  uniformity:
  - SFPU sigmoid/silu: in place on DST.
  - Reduce: DST input -> DST accumulator.
  - Matmul: CB inputs -> DST accumulator (the Matrix Unit has no DST
    operand).
- Do not reintroduce `LLKBcast`: old, wrong, and absent from tt-metal.
- In `Op::TT` match arms use the `Op::TT(_)` catch-all whenever detailed
  handling is not needed.

### Process

- Keep `NOTES.md` current: design, directives, architecture, finished
  steps, and next steps. Quoting the user verbatim is not required; the
  substance is.

## `TTOp` after the native-convention refactor

The IR now names each native instruction's real operands. `MatmulTiles` is
gone: a matmul is a *sequence* of these, emitted by lowering, not a
structured op.

```rust
pub enum TTOp {
    EndReader,
    EndCompute,
    ProgramDesc(Box<TTProgramDesc>),
    TensixGrid(Box<TensixGrid>),
    /// DRAM<->CB traffic, CB->DRAM drain, and the CB->DST unpack.
    /// A Register `dst` is the unpack destination; `dst_idx` is unused.
    Copy { src: OpId, src_idx: OpId, dst: OpId, dst_idx: OpId, bytes: u32 },
    /// Generic SFPU: `kind` is the metalium opcode, `b` is NULL for
    /// unary. The preceding `Copy` brings a CB input into DST.
    Sfpu { kind: u32, a: OpId, b: OpId, slot: OpId },
    /// Native SFPU reduce: DST input, DST accumulator.
    ReduceTile { rop: BOp, kind: TileDim, input: OpId, slot: OpId },

    // Instruction level (the Matrix Unit path):
    /// UNPACR: `unit` (0=SrcA, 1=SrcB) moves `count` datums of `src`'s tile
    /// image into its Src bank, at the face the preceding Setadczw chose.
    Unpacr { src: OpId, unit: u8, count: u32, flip: bool },
    /// SETADCZW: face selection for one unpacker.
    Setadczw { unit: u8, face: u32 },
    /// SETADCX: the datum range one unpacker walks.
    Setadcx { unit: u8, first: u32, count: u32 },
    /// SETRWC: the accumulator base (`dst`'s DST row).
    Setrwc { dst: OpId },
    /// ZEROACC: `mode` 3 is CLR_ALL (a fused-kernel hazard).
    Zeroacc { mode: u8, imm10: u32 },
    /// MVMUL: row within the accumulator base, the addr modifier, and the
    /// bank handoff (`flip_a`/`flip_b`) a releasing multiply performs.
    Mvmul { dst_row: u32, addr_mod: u8, flip_a: bool, flip_b: bool, broadcast_b: bool },
    /// SETC16: one ThreadConfig field (`thread::*` address) and its value.
    Setc16 { field: u32, value: u16 },
    /// SETCFG: a whole `Config` word (`ConfigWords::seed` + `WRCFG`), for the
    /// `alu::*`/`thcon::*` fields that are not `ThreadConfigField`s.
    Setcfg { addr: u32, value: u32 },
    /// SEMPOST / SEMWAIT: the role-handoff semaphores.
    Post { sem: u8 },
    Wait { sem: u8, before: u8 },
    /// SEMSET: the run's semaphore init (`TTNativeProgram::sems`), not an
    /// instruction.
    Semset { sem: u8, value: u8 },
}
```

Operand wiring: `Unpacr.src`, `Setrwc.dst` are real operands (an SSA
dependency and a DST row); the rest carry immediates only.

Codegen encodes each 1:1 (`tt_sfpu_program`, `encode::{Setadczw, Setrwc,
zeroacc, Mvmul}`, `sync::{post, wait_nonzero}`, `datapath::{set_adc_x,
unpack_src_instruction, thread_entry, tile_base_units, config_program}`),
and routes by op kind: `Unpacr`/`Setadczw`/`Setadcx` -> T0, `Setrwc`/
`Zeroacc`/`Mvmul`/`Setc16` -> T1, packs -> T2. `Post`/`Wait` join the stream
of the instruction before them (`TtSection::last_role`), since a semaphore
op names no role.

Known gaps in the instruction path:

- `Wait.before` is a *code* (0..7 -> `Before::{UNPACKER, PACKER, MATRIX,
  SFPU, CONFIG, SCALAR, MOVER, EVERYTHING}`) because `Before` has no public
  constructor from bits.
- `Setc16.field` is a `ThreadConfig` address resolved through a table of
  the fields zyx emits; the `alu::ALU_ACC_CTRL_*` fields are `ConfigField`s,
  not `ThreadConfigField`s, so they need a `ConfigWords` write instead and
  are not in the table yet.
- `Unpacr` emits wait + `WRCFG` + stallwait + `UNPACR`. The first unpack on
  each unpacker carries the full Src configuration, later ones only retarget
  the base address (`TtSection::src_unpacker_ready`). See the Src-config
  facts below.

### Src-unpack configuration (verified in metalium-rs)

- The builders are **public**, so nothing has to be replicated:
  `datapath::unpack_src_config(words, unpacker, descriptor, l1, out)`
  (`crates/tt-kernels/src/datapath.rs:203`),
  `matmul::tile_descriptor(L1Format)` (`:310`), `matmul::FACE_DATUMS = 256`
  (`:323`), `datapath::{set_adc_x, unpack_src_instruction,
  tile_base_units, config_program, src_thread_config}`.
- `ConfigWords` is re-exported *privately* by `datapath`; zyx uses
  `tt_isa::backend::ConfigWords` directly.
- `unpack_src_config` sets, for `SrcA` (SEC0): base address, `REG7` offset 0,
  `REG5_Tile_x_dim_cntx0`, `REG2_Out_data_format`, `REG2_Unpack_if_sel_cntx0
  = 0` (`Src`, not `Dst`), `REG2_Disable_zero_compress_cntx0 = 1`,
  `REG5_Dest_cntx0_address = DST_BASE`, `UNP0` Y/Z strides 0, plus the
  non-zero descriptor words. For `SrcB` (SEC1) it is much shorter: base,
  `REG2_Out_data_format`, `Disable_zero_compress`, `UNP1_ADDR_CTRL_ZW`
  Z stride — ttsim refuses most of `THCON_SEC1_REG5`/`REG7`.
- **Operand wiring is swapped**: `A` is in0 and goes to `SrcB` (SEC1),
  `B` is in1 and goes to `SrcA` (SEC0) — `unpack_src_config(SrcA, .., b0)`
  and `(SrcB, .., a0)`. Lowering must swap `cb_a`/`cb_b` accordingly.
- The Src output format is TF32 (`L1Format::Tf32`), not FP32: FP32 *into*
  `Src` is `UndefinedBehavior`; the pcrelude also sets
  `alu::ALU_ACC_CTRL_Fp32_enabled = 1` and `Zero_Flag_disabled_src = 0`.
- Two different thread configs: `src_thread_config`
  (`SRCA_SET_SetOvrdWithAddr = 1`) for Src unpacks, `thread_config`
  (same field `= 0`) for `Dst` unpacks. One T0 role cannot be both — a
  fused kernel with a matmul and an SFPU currently silently uses the Src
  form. Open: reject that mix, or split the role. The hazard is narrower
  than it looks: a value the matmul computes stays in DST, so an SFPU
  consumer reads it without any unpack; the mix only bites when the same
  kernel also needs a CB -> DST unpack (a multi-use value, or an input read
  by two consumers), which is exactly `tt_resolve_dst`'s Copy path.
- Preclude order in `matmul_items`: `src_thread_config` -> one
  `config_program` for both unpackers plus the ALU words -> `set_adc_x` per
  unpacker over `0..FACE_DATUMS-1` -> per pair only the base retarget.
- `Mvmul` carries its bank handoff (`flip_a`/`flip_b`, from
  `Banks::mvmul_release_a`/`_b`/`_both`) and `broadcast_b`
  (`BroadcastSrcBRow`), and `Unpacr` carries `flip` (`Banks::unpack_a`/
  `unpack_b` set it for a whole-operand fill, `*_partial` leaves it clear).
  Both are now encodable, so the sequence below can be emitted.
- The matmul sequence, transcribed from `matmul::matmul_items` for one
  tile pair at HiFi4 (the shape `tt_lower_llks` must emit):
  - T1 prelude: `Setc16` addr mods (`ADDR_MOD_AB_SEC0_SrcAIncr` = 0,
    `ADDR_MOD_DST_SEC0_DestIncr` = 0, `ADDR_MOD_AB_SEC1_SrcAIncr` = 0,
    `ADDR_MOD_DST_SEC1_FidelityIncr` = 1, the `math_addr_mods` group),
    `Setc16(FIDELITY_BASE_Phase, 0)`, `Setrwc { dst: slot }`,
    `Zeroacc { mode: 3 }`.
  - per `fi` in 0..2, per `k` in 0..2:
    `Setadczw(SrcB, 2fi+k)`, `Unpacr(cb_b, SrcB, 1024, flip)`,
    `Setadczw(SrcA, 2k)`, `Unpacr(cb_a, SrcA, 1024, flip)`,
    7 `Mvmul` (rows 0 and 8, `MATH_AM_PHASE`/`MATH_AM_HALF`, no flips),
    `Mvmul(8, MATH_AM_GROUP, flip_a)`,
    `Setadczw(SrcA, 2k+1)`, `Unpacr(cb_a, SrcA, 1024, flip)`,
    7 `Mvmul` (rows 16 and 24, no flips),
    `Mvmul(24, MATH_AM_GROUP | MATH_AM_FI, flip_a, flip_b)` — the last `k`
    of an `fi` moves the accumulator base 32 rows on (`MATH_AM_FI`).
  - `Zeroacc` mode 3 clears *all* DST, so codegen rejects it once any compute
    value op has been emitted in the section (`TtSection::compute_ops`): it is
    only valid as the section's first accumulator clear.
  - `Semset` is the semaphore init, folded into `TTNativeProgram::sems`
    (`TTSemaphoreInit { index, value, max }`) rather than emitted as a word;
    a duplicate index is a compile error. `Post`/`Wait` are real instructions
    and join the role stream of the op before them.
- The lowering is in (`tt_lower_llks`, `TtEmit::Matmul`): the root's op
  becomes `TTOp::Setrwc { dst: slot }` and the rest of the sequence is
  inserted around it — the `Setc16`/`Setcfg` address modifiers *before* it,
  `Zeroacc` + the `fi`/`k` grid after. The root carrying the accumulator is
  not a hack: `dtype`, `shape`, rcs (1 use per `dst`) and the flops counter
  already treated `TTOp::Setrwc` as the matmul value. `tt_lower_value` and
  `tt_lowered_storage` resolve a `Setrwc` to its `dst`, so a single-use
  consumer reads the accumulator's slot directly (a multi-use one gets
  packed into a CB by `tt_materialize`, and that pack is T2, i.e. after the
  math role drained — role order, not stream order).
- One accumulator slot is the whole 32x32 result: `fi` 0 fills rows 0..31,
  `fi` 1 rows 32..63 (a DST row is 16 datums, a tile is 1024), and
  `MATH_AM_FI`'s `DestIncr = 32` moves the base between them.
- The `math_addr_mods` words are *multi-field*: SEC2 / SEC3 / SEC4 each
  carry `SrcBIncr`+`SrcBClear` or `DestIncr`+`DestCR`+`FidelityClear` in one
  word, so they go through `Setcfg` and lowering composes the word with
  `ThreadConfigEntry::zeroed(addr32).set(field, v)...`, passing
  `(addr32, value)`. Only the four single-field SEC0/SEC1 entries use
  `Setc16`. Values: SEC2 = `SrcBIncr 8`, `DestIncr 0`, `DestCR 0`,
  `FidelityClear 1`; SEC3 = all zero except `SrcBClear 1` and
  `FidelityClear 1`; SEC4 = SEC3 plus `DestIncr 32`.
- Operand naming, fixed: A's CB goes to `SrcB` (unit 1), B's CB to `SrcA`
  (unit 0). The transcription above wrote the source CB by *bank* name
  (`cb_b`), not by operand, which reads backwards.
- Unpack drain: with Src unpacks both unpackers run, so `tt_unpack_role`
  now ends with `wait_for_unpacker0` **and** `wait_for_unpacker1`
  (`matmul_items` ends the same two waits). `tt_verify` also knows every
  compute-section instruction op now (`Setcfg`, `Semset` were missing).
- No semaphores in the emitted sequence: the roles run in order
  (`Schedule::InOrder`), so the pack role starts after the math role has
  drained and the `ready`/`free` handshake `matmul_items` needs for its
  two output tiles is not needed for one tile in DST at a time.
- Still open: the T0 mixed-role question (a fused matmul + SFPU kernel), and
  the sequence is compile-checked only — no device run yet.

`FillTile` and `ReduceTile::cb_sc` are gone. They existed only for the
tt-metal `reduce_tile` scaler CB; the metalium reduce program carries its
own identity and never reads a scaler.

The lowering emits, per computed value:

```
reader:  Copy Param -> reader_cb            // DM READ
compute: Copy reader_cb -> dst_a            // unpack_tile_to_dst
         Sfpu { kind, a: dst_a, slot: dst_b }   // generic elementwise
         // or ReduceTile { input: dst_a, slot: dst_b }
         // or MatmulTiles { cb_a, cb_b, slot: dst_b }  // no unpack
         // multi-use result: Store dst_b -> CB, consumers Copy CB -> DST
writer:  Copy writer_cb -> Param            // DM WRITE
```

The `Store` from a Register storage to a Circular writer CB is the output
boundary pack. It is not an intermediate roundtrip: the writer core cannot
read DST, so the final tile must reach L1. An intermediate is packed only
when it has multiple uses (the dataflow CB roundtrip above).

## ISA facts (verified against `metalium-rs`)

These are hardware/ISA facts, not zyx conventions.

- There is **no single `REDUCE` instruction** in `tt-isa`. A tile reduce is a
  sequence of SFPU instructions (`SFPLOAD`/`SFPSTORE` plus arithmetic and
  shuffles). `sfpu::reduce::accumulate` and `sfpu::reduce::finish` are the
  actual math programs; `sfpu::reduce::roles`/`roles_at` wrap them in the
  LLK-style unpack/math/pack roles and add L1 staging.
- The reduce math program reads DST, not a CB. `accumulate` loads the input
  from `A_ROW` (DST row 0) and accumulates into `OUT_ROW` (DST row 128);
  `finish` folds the accumulator in `OUT_ROW`. The unpacker (T0) moves
  L1/CB -> DST and the packer (T2) moves DST -> L1. The CB is only staging.
- `SFPLOAD`/`SFPSTORE` carry the DST row in their `Imm10` field.
  `Program::new` clears the DST row counter (`clear_dst_rwc`) and sets
  address modifier 0 to step nothing, so `Imm10` is an **absolute DST row**
  for the program's instructions.
- The metalium SFPU/reduce builders hardcode `A_ROW = 0`, `B_ROW = 64`,
  `OUT_ROW = 128` (`sfpu/kernel.rs`). To run them directly on arbitrary DST
  rows, rebuild each `SFPLOAD`/`SFPSTORE` word with a remapped `Imm10`:
  input rows `A_ROW..A_ROW+64` -> the reduce input row, output rows
  `OUT_ROW..OUT_ROW+64` -> the reduce output row. `Field::extract`,
  `Field::place`, and `Instruction::new` make this a local word rewrite.
  The elementwise `program_for` uses the same constants (`A_ROW`/`B_ROW`/
  `OUT_ROW`), so the same relocation applies.
- `MatmulTiles` is the exception: the Matrix Unit's `MVMUL` reads `SrcA`/
  `SrcB` from **CBs** and writes a DST row. It has no DST operand.
  `MOVD2A`/`MOVD2B` can move DST -> `SrcA`/`SrcB`, but the native matmul
  path keeps CB inputs.
- A tile is 32x32 datums; DST rows are 16 datums wide, so a tile occupies
  64 DST rows. `A_ROW`/`B_ROW`/`OUT_ROW` are tile slots 0/1/2.
- Data-mover list entries are `[u32; 8]`. `dm::op::READ = 1` and
  `dm::op::WRITE = 2` move DRAM <-> L1; `READ_RUN = 0x15` and
  `WRITE_RUN = 0x16` move whole tile runs. The reader/writer roles run
  these lists; the compute roles run the instruction streams.
- The three Tensix roles are T0 (unpack / reader DM), T1 (math), T2
  (pack / writer DM). They synchronize through semaphores. The native
  program needs the role streams plus the semaphore init list.
- `TileDim` -> metalium reduce axis: `Row` (one value per row, first
  column) -> `Axis::Cols`; `Col` (one value per column, first row) ->
  `Axis::Rows`; `Scalar` -> two passes (`Cols` then `Rows`).
- The current `codegen/tenstorrent.rs` still emits C++ strings. The native
  path must carry raw instruction words (`Instruction::word()`) in
  `TTProgram` and the backend must stage them, because `Instruction` holds
  a `&'static InstructionDef` and is not `SerBin`.

## metalium-rs API facts (verified against the checkout, `b22a245`)

These supersede the older assumptions in the ISA-facts section. Paths below
are inside `~/Dev/rust/metalium-rs`.

### There is no "program" object; a run is roles + schedule

- `tt_isa::isa::Instruction { word, def: &'static InstructionDef }`
  (`crates/tt-isa/src/isa/mod.rs:428`). `Instruction::word()` (`:441`);
  `#[doc(hidden)] Instruction::new(word, def)` (`:436`).
- There is **no public word -> `Instruction` decoder**. A backend that
  stores only words must supply the def itself, either by matching
  `InstructionDef::matches(word)` over `isa::generated::ALL`
  (`generated.rs:2746`) or by keeping the emit-time kind alongside the
  word. The role firmware takes words directly
  (`session`/`harness` stage `program_bytes`), so only the host side ever
  needs a def.
- Runnable unit: `tt_kernels::runtime::Kernel<'a> { roles:
  [&'a [Instruction]; 3], schedule, stage, read_back, dump_rows, clear_dst,
  dst_fmt, trace, restores_semaphores, unwedge, mop,
  loops: [&'a [Loop]; 3] }` (`crates/tt-kernels/src/runtime.rs:66-101`),
  built with `Kernel::new(roles, schedule)` (`:105`) and executed by
  `runtime::run(dev, tile: NocCoord<N>, images: &RoleImages, kernel,
  budget)` (`:421`). `RoleImages = [(Core, &[u8], u64); 3]` (`:48`), with
  `tt_firmware_images::ROLES` for T0/T1/T2.
- Roles are `[Vec<Instruction>; 3]` in T0/T1/T2 order. Builders return
  exactly that: `datapath::dst_round_trip_roles(...) ->
  [Vec<Instruction>; 3]`, `matmul::matmul_roles(...) -> [Vec<Instruction>;3]`,
  `sfpu::kernel::roles(...)`, `sfpu::reduce::roles(...)`.
- Host-side block loops are `tt_kernels::code::Loop { start, len, count }`
  (`crates/tt-kernels/src/code.rs:17-21`) carried in `Kernel::loops`; the
  firmware pushes each range `count` times. `Code::stored()` packs them
  with `mailbox::loops::LOOPED = 1<<31` into the length word
  (`mailbox.rs:136-171`). Max 4 loops per role.

### Semaphores are run state, not program state

- `tt_isa::sync::Semaphore(u8)`, only `0..8` (`crates/tt-isa/src/sync.rs:41`).
  `SemaphoreInit` is **not** in `tt-isa`; it is a type alias
  `pub type SemaphoreInit = (Semaphore, u8, u8)` in
  `tt_kernels::runtime` (`crates/tt-kernels/src/runtime.rs:43`).
- A kernel gets them via `Schedule::Concurrent(&[SemaphoreInit])`
  (`runtime.rs:52`); `runtime::assemble` prepends a setup program of
  `sync::init(sem, value, max)` (`:386-412`). A kernel must leave every
  semaphore as it found it (`Kernel::restores_semaphores`, `:86`).
- Builders hand back their init list in their `Layout`: `Sfpu Layout
  { tiles, a_at, b_at, c_at, out_at, sems, init: Vec<SemaphoreInit> }`
  (`sfpu/kernel.rs:89-101`), `reduce Layout { outputs, per, in_at, out_at,
  sems, init }` (`sfpu/reduce.rs:377-384`), `matmul Layout { a_at, b_at,
  outputs, sems, init }` (`matmul.rs:1151-1161`).

### CBs have no runtime config

- There is no `CbConfig`, no CB descriptor write, and `mailbox::Descriptor`
  (`crates/tt-isa/src/mailbox.rs:442`) has no L1/CB fields. zyx's
  `TTCbConfig` (format/tile_bytes/tiles) is a **zyx-side** table only.
- L1 placement is a build-time plan: `l1::Requirements::cb(name, page,
  pages, align, producer: Endpoint, consumer: Endpoint, live) ->
  Buf` (`crates/tt-kernels/src/l1.rs:218-240`), `plan(arena) -> Plan`
  (`:414`), `Plan::addr(Buf) -> u64` (`:531`),
  `Plan::semaphore_init()` (`:550`), audit `l1::check` (`:559`).
  `Endpoint::{Reader, Writer, Mover, Unpack, Math, Pack}` (`:39-51`).
- Addresses reach the programs as immediates through `datapath::
  tile_unpack_config(&mut ConfigWords, l1)` (`datapath.rs:790`),
  `pack_config(&mut ConfigWords, l1_dest)` (`:288`),
  `config_program(&ConfigWords)` (`:692`).

### Builder signatures zyx will call

- `datapath::unpack_tile_to_dst(l1: u64, row: u32) -> Vec<Instruction>`
  (`datapath.rs:798`) — the only CB -> DST transfer.
- `datapath::pack_tile_from_dst(l1_dest: u64, row: u32) ->
  Vec<Instruction>` (`:839`); staged form `-> Result<_, PackModeError>`
  (`:899`).
- `sfpu::ops::program_for(kind: u32, scalars: [f32; 2], bcast: Broadcast)
  -> Option<(Operands, Vec<Instruction>)>` (`sfpu/ops.rs:2869`) — note it
  lives in `sfpu::ops`, not `sfpu::kernel`, and takes `(scalars, bcast)`.
  `sfpu::kernel::plan_layout(tiles, operands)` (`kernel.rs:119`),
  `roles(layout, operands, math)` (`:179`),
  `roles_code(...) -> ([Vec<Instruction>;3], [Vec<Loop>;3])` (`:194`).
  `Operands::{Unary, Binary, RowBroadcast, ColBroadcast, Ternary}`
  (`kernel.rs:48-66`).
- `sfpu::reduce::{accumulate(op, axis, first, valid),
  accumulate_in_order(first, valid), finish(op, axis)}`
  (`sfpu/reduce.rs:178,248,300`), assembled by `roles(layout, inputs,
  finish)` (`:456`) / `roles_at(...)` (`:468`).
- `matmul::matmul_roles(outputs: &[OutputTile], sems: MatmulSemaphores,
  in_fmt: L1Format, out_fmt: u32, fidelity: Fidelity) ->
  [Vec<Instruction>; 3]` (`matmul.rs:517`); `OutputTile { pairs, out }`
  (`:300`), `MatmulSemaphores { ready, free }` (`:253`).

### Device, DRAM, dispatch

- `Device<T: Transport>`: `open(transport)`, `alloc_window`,
  `write/read(l1)`, `dram_grid(&mut window) -> Dram`,
  `dram_write(range, &[u8])` / `dram_read(range, &mut [u8])`
  (`crates/tt-device/src/dram.rs:26,70,92`). `DramRange` is only minted by
  the chip's `Dram` grid.
- The allocator is `tt_kernels::tensor::DramAlloc::{new(&Dram),
  alloc(tiles) -> Placement, free(&Placement)}` (`tensor.rs:204-292`).
- Launch is per tile: `runtime::run(dev, tile, images, kernel, budget)`.
  **No grid field anywhere.** Multi-tile is a host loop over
  `Tensix::tiles::<Noc0>()` or `Session` with
  `TileChoice::{Exactly, First, Count, All}`.
- `tt_ttsim::Simulator` is `!Send + !Sync` (`tt-ttsim/src/lib.rs:148-157`)
  and a process singleton; fatal simulator errors `_Exit`, so tests run
  inside `fork_scope`. `Transport::tick` is the only way time advances.
  On ttsim the host must set `PUSH_WINDOW = SIM_PUSH_WINDOW` (16) because
  the FIFO-full stall is not modelled (`mailbox.rs:78-103`).
- Program limits: `mailbox::PROGRAM_MAX = 8192` words, `PROGRAM_SLOT =
  0x8000`, four slots (`mailbox.rs:280-288`).
- BF16: `tt_kernels::bf16::{expand, repack*}` and `fp16::expand` are
  `pub(crate)`; reachable downstream only through `Session`, or by
  emitting the same `Work`/`Job` steps.

## Native program (the concrete shape)

```rust
struct TTNativeProgram {
    // One data-mover list: reader entries, then writer entries.
    reader: Vec<[u32; 8]>,          // dm::op::READ
    writer: Vec<[u32; 8]>,          // dm::op::WRITE
    // Three Tensix role streams, raw instruction words so the program can
    // live in the kernel IR (tt_isa::Instruction has a &'static def, not SerBin).
    compute: [Vec<u32>; 3],
    compute_loops: [Vec<code::Loop>; 3],   // tt_kernels::code::Loop
    sems: Vec<runtime::SemaphoreInit>,     // tt_kernels::runtime, = (Semaphore,u8,u8);
                                           // Semaphore is not SerBin -> store (u8,u8,u8)
    // Circular storage -> L1 address; Register storage -> DST row.
    l1: Vec<(OpId, u32)>,
    dst: Vec<(OpId, u32)>,
    // Existing descriptor tables stay.
    reader_params: Vec<u32>,
    compute_params: Vec<u32>,
    writer_params: Vec<u32>,
    n_params: u32,
    input_dtypes: Vec<DType>,
    output_dtypes: Vec<DType>,
    fp32: bool,
}
```

Address rules:

- Allocate Circular storages first-touch, 64-byte aligned, `len` bytes
  (the storage already counts tiles). Reject anything that does not fit the
  data arena.
- `Register` storages get DST rows by liveness (the existing slot
  assignment), and are only consumed by the compute builders; they never
  appear in a DM entry.
- Sigmoid/silu: the `Copy` unpacks its CB into the `slot` row; the SFPU
  program rewrites that row in place. There is no `out_at` CB.
- Reduce: the `Copy` unpacks into `input`; the reduce program reads
  `input` and writes `slot`.
- Matmul: `a_at`/`b_at` are `cb_a`/`cb_b`'s L1 addresses; `MVMUL` writes
  the `slot` DST row. The Matrix Unit has no DST operand; if a later
  change wants DST inputs anyway, `MOVD2A`/`MOVD2B` move DST -> SrcA/SrcB
  (`tt-isa/src/isa/generated.rs:2121,2133`;
  `matrix_eltwise.rs:229` is the reference use).
- Packs to a CB are the output-boundary `Store` that feeds the writer's DM
  `WRITE`, plus the multi-use intermediate pack (Register -> Circular)
  whose CB feeds later consumers' unpacks. No other intermediate is
  packed.
- Semaphores restart per run. Ops are sequential, so the same indices are
  reused; the init list is the union.

Traffic:

- Reader `Copy { src: Param, dst: Circular }` -> one `dm::op::READ` entry
  `[READ, channel, port, dram_offset, l1_addr, len, 0, 0]`.
- Writer `Copy { src: Circular, dst: Param }` -> one `dm::op::WRITE` entry.
- Every reader entry is a kernel-input read; every writer entry is a
  kernel-output write. There is no other DRAM traffic in the list.
- Pure movement: reader + writer list, no compute roles (one stream pair).
- With compute: reader list -> role programs -> writer list. The first cut
  runs them sequentially; the concurrent form is the same three streams
  joined by DM `KERNEL` / `KERNEL_WAIT` entries, which is a later
  optimization, not a semantic change.

## Status / TODO

Legend: `[x]` done, `[~]` in progress, `[ ]` pending.

### Keep the model compiling

- [x] Restore origin/main versions of `kernel/ops.rs`, `kernel/tenstorrent.rs`,
      `kernel/cost.rs`, `kernel/debug.rs`, `kernel/fold_constants.rs`,
      `kernel/instr_sched.rs`, `kernel/licm.rs`, `kernel/verify.rs`,
      `kernel/mod.rs`, `kernel/custom.rs`, `kernel/autotune.rs`,
      `codegen/tenstorrent.rs`, `backend/tenstorrent.rs`.
- [x] Delete `build.rs` and `tt_runtime_shim.cpp/.h`.
- [x] Remove the `tenstorrent` / `metalium` Cargo features and gates.
- [x] Keep native deps (`tt-device`, `tt-isa`, `tt-kernels`, `tt-ttsim`).
      They are path deps in `zyx/Cargo.toml`: `../../metalium-rs/crates/*`,
      i.e. `~/Dev/rust/metalium-rs` (also `tt-firmware-images`). That repo
      is a plain checkout of `Knight-Ops/metalium-rs` and must be pulled
      before a build that depends on a new builder.
- [x] `Config.tenstorrent: TTConfig` (no feature gate).
- [x] `Pool::TT` reads go through `pool_to_host`; `Pool::Host` is host RAM.
- [x] `Dev::tilize` / `Dev::untilize` dispatch to `Dev::tilize_transfer` /
      `Dev::untilize_transfer`.
- [x] Remove the compile-time `ZYX_TT_SHIM` dependency (runtime lookup only
      until the shim is deleted).

### Delete tt-metal-only ops and passes

- [x] Delete `TTOp::{ReserveBack, PushBack, WaitFront, PopFront, MathLock,
      MathUnlock, PackLock, PackUnlock, NocReadBarrier, NocWriteBarrier,
      ReduceUninit, LLKBcast, LLK}`.
- [x] Delete `Kernel::tt_reserve_back` / `tt_push_back` / `tt_wait_front` /
      `tt_pop_front` / `tt_math_lock` / `tt_math_unlock` / `tt_pack_lock` /
      `tt_pack_unlock` / `tt_noc_read_barrier` / `tt_noc_write_barrier`.
- [x] Delete passes `tt_lock_dst`, `tt_dedup_pushes`, `tt_init_math`,
      `tt_sync_cbs`, `tt_place_pops` (and `tt_fifo_check` +
      `fifo_check_copy`, which only accounted for those ops).
- [x] Pipeline becomes `tt_tile -> tt_lower_llks -> tt_dst_reuse -> verify
      -> tt_verify` (then codegen).
- [x] Remove their match arms in `ops.rs`, `cost.rs`, `debug.rs`,
      `fold_constants.rs`, `instr_sched.rs`, `licm.rs`, `mod.rs`,
      `verify.rs`, `custom.rs`, `codegen/tenstorrent.rs`. Generic TT arms
      use `Op::TT(_)` per directive 22.
- [x] Rewrite `tt_verify` for the native op set: section order, no compute
      op in the reader/writer, no `Op::Barrier`.
- [x] Remove the `LLKBcast` builder from `custom.rs`.
- [x] `cargo check` is green again (the old shim backend still compiles).

### Generic SFPU + recursive lowering (finished)

- [x] Delete `TTOp::FillTile` and `TTOp::ReduceTile::cb_sc`. The scaler CB
      was a tt-metal API artifact; the native reduce program has its own
      identity.
- [x] `TTOp::Sfpu { kind, a, b, slot }` replaces `SigmoidTile`/`SiluTile`.
      `kind` is the metalium opcode, `b` is NULL for unary. Dtype/params/
      verify/cost/debug/instr-sched/codegen arms updated.
- [x] `UOPS`/`BOPS` + `uop_kind`/`bop_kind` build the generic elementwise
      table; no per-op lowering arm. Silu is `Mul(x, sigmoid(x))`.
- [x] `ReduceTile { rop, kind, input, slot }` reads DST `input`, writes
      DST `slot`; its identity is in the program, not a scaler CB.
- [x] `MatmulTiles { cb_a, cb_b, slot }`: CB inputs, DST accumulator, as
      the Matrix Unit requires.
- [x] `TTOp::Copy` with a Register `dst` is the only CB -> DST transfer.
      `tt_resolve_dst` emits it on demand; compute never reads a CB
      directly except matmul.
- [x] Phase 2 is recursive/root-first (`tt_lower_value`), not a forward
      walk: fused roots win over their inputs, and the silu cone lowers
      correctly without a silu pass.
- [x] Multi-use computed values are packed to a Circular CB after the
      producer; every consumer unpacks. The redundant first unpack is
      left for a later optimization pass.
- [x] `tt_dst_reuse` assigns physical DST rows by liveness after lowering
      and rewrites operands; lowering is SSA and never frees rows.
- [x] `verify` enforces the new scopes: SFPU/reduce operands are
      Register; matmul inputs are Circular and its slot is Register; a
      Register copy dst must be fed by a Circular src.
- [x] `cargo check` is green.

### Boundary-only DRAM (status)

- [x] Writer drains are collected only from `Op::Store` to `Param
      GlobalMut`. A store to an `Op::Storage` (the writer CB) is not a
      drain.
- [x] Phase 1 publishes only kernel-input `Op::Load`s from `Param
      Global`/`GlobalMut`; these are the reader list.
- [x] Phase 3 packs a single-use DST result into a writer CB and drains
      it; a multi-use value already materialized in a CB is drained
      directly.
- [x] `tt_verify` rejects a DRAM `Copy`/`Store`/`Load` in the compute
      section and a store-then-load of the same `GlobalMut` param.
- [ ] Next optimization: coalesce a materialized value's first unpack
      with the producer's DST row (and delete the redundant pack/unpack
      when a later pass proves it safe).

### Native codegen

- [~] Replace `TtSection.out: String` with native emitters. The reduce
      arm is native; the C++ source is still emitted for the other ops.
- [x] `TTNativeProgram` (declared in `backend/tenstorrent.rs`, carried in the
      IR by `TTProgramDesc.native`): `reader`/`writer: Vec<TTDmMove>`,
      `compute: [Vec<u32>; 3]`, `compute_loops: [Vec<TTLoop>; 3]`,
      `sems: Vec<TTSemaphoreInit>`, `l1: Vec<u32>` (per CB index),
      `dst: Vec<TTDstRow>` (per Register storage). All plain-`SerBin`
      mirrors of the metalium types, since `Semaphore`/`Instruction`/`Loop`
      are not.
- [x] L1 placement: zyx-owned bump allocator (`tt_place_cbs`) over
      `tt_isa::l1::DATA`, first touch = CB index order, 64-byte aligned,
      `Op::Storage.len` bytes; arena overflow is a compile error. The
      addresses are what unpacks/packs will bake in as immediates.
- [x] Boundary moves (`tt_boundary_moves`): one `TTDmMove` per reader
      publish (`Param -> Circular`) and per writer drain (`Circular ->
      Param`), in program order, carrying the param ordinal plus the CB's
      L1 address and the transfer bytes. DRAM channel/port/offset are
      launch-time, so they are *not* in the staged program.
- [x] DST map: `render` snapshots the compute section's slot table into
      `native.dst` (row-sorted), because the emitter's slot table is the
      only place a Register storage is bound to a row.
- [ ] **Open: slots are tile numbers, but two places use them as rows.**
      `fresh_slot` hands out 0,1,2,..., and `slots` is a *tile index* — the
      C++ pack path passes it straight to `pack_tile(slot, dcb)`. Two
      consumers treat it as a DST *row base*:
      `tt_row_remap(code, a_row, ..)` adds `0..63` to it, and `Setrwc`
      writes it as `DstVal`. A 32x32 F32 tile is 64 DST rows, so the base
      row is `64 * slot`; and `SETRWC.DstVal` is a **4-bit** field
      (`generated.rs:800`), i.e. 32-row units, so it wants `2 * slot` for
      F32 (and `slot` for F16, whose tile is 32 rows). With the current
      code only slot 0 is right, and slot 7's `DstVal = 7` would point at
      row 224 instead of 448. Fix: keep `slots` as tile indices and convert
      at the row consumers (row base `slot * 64`, `DstVal = row / 32`).
- [x] Role split, first cut (`Schedule::InOrder`, no semaphores):
      - T0 = `datapath::thread_config` + one
        `datapath::unpack_tile_to_dst(cb_l1, dst_row)` per lowered `Copy`
        (Circular -> Register) + `wait_for_unpacker0(Before::EVERYTHING)`;
      - T1 = `datapath::state_id` + the per-op math words +
        `wait_for_sfpu(Before::EVERYTHING)`;
      - T2 = `datapath::state_id` + one
        `datapath::pack_tile_from_dst(cb_l1, slot_row)` per lowered `Store`
        (Register -> Circular) + `wait_for_packer(Before::EVERYTHING)`.
      Empty roles stay empty (a pure-copy kernel has no compute roles).
- [x] Generic SFPU codegen: `tt_sfpu_program` calls
      `sfpu::ops::program_for(kind, [0.0, 0.0], Broadcast::None)`, checks the
      arity against the op's operand count, and remaps
      `A_ROW`/`B_ROW`/`OUT_ROW` to the op's actual DST rows
      (`tt_row_remap`, shared with reduce). A kind with no metalium program,
      or an arity mismatch, is a loud error.
- [x] Native compute is F32-only for now: a non-F32 unpack/pack errors
      instead of silently unpacking as FP32 (BF16/F16 staging is open).
- [ ] `MatmulTiles` still emits `matmul_tiles(...)` C++ (`MVMUL` roles are
      not native yet); a mixed native/C++ compute section cannot run, so
      matmul is the blocking gap before the first green run.
- [ ] Semaphores: `native.sems` stays empty while roles run in order. The
      concurrent form (DM `KERNEL`/`KERNEL_WAIT`, `Schedule::Concurrent`)
      needs the builder `Layout.init` lists and the ready/free protocol.
- [ ] `compute_loops` unused: every role is a flat instruction list.
- [ ] Reader: `Vec<[u32; 8]>` DM entries (`dm::op::READ`, DRAM -> L1).
- [ ] Writer: `Vec<[u32; 8]>` DM entries (`dm::op::WRITE`, L1 -> DRAM).
- [ ] Compute: `[Vec<tt_isa::isa::Instruction>; 3]` from the metalium
      builders, one stream per role. Currently the raw words are one
      `TTProgramDesc.native_compute` vector; role tags are next.
- [ ] `Copy` with a Circular dst -> DM entry; with a Register dst ->
      `datapath::unpack_tile_to_dst`. The native words still need the CB
      L1 map (the C++ path uses the CB number).
- [ ] `Sfpu { kind, a, b, slot }` -> the metalium SFPU program for `kind`
      on DST inputs; no unpack and no pack inside the op.
- [x] `ReduceTile` -> `sfpu::reduce::accumulate` + `finish` math on
      `input`, writing `slot`; the LLK `reduce_tile` call is gone. The
      program's absolute `A_ROW`/`OUT_ROW` rows are remapped to the
      actual DST rows by rewriting `SFPLOAD`/`SFPSTORE` `Imm10`.
      `TileDim::Row` -> `Axis::Cols`, `Col` -> `Axis::Rows`, `Scalar` ->
      `Cols` then `Rows`. A sum over columns (`accumulate_in_order`)
      zeroes its output before reading, so it stages through a scratch
      DST row; a scalar reduce always uses a scratch row and copies it
      back. Raw words land in `TTProgramDesc.native_compute`.
- [ ] `MatmulTiles` -> `MVMUL` from `cb_a`/`cb_b`, writing `slot`; no
      CB -> DST copy.
- [ ] Packs: the output-boundary `Store` (Register -> Circular) before the
      writer's DM `WRITE`, plus the multi-use intermediate pack (Register
      -> Circular) that feeds later consumers' unpacks.
- [ ] Keep `TTProgramDesc`, per-section params, CB numbering, DST slots,
      `Op::Loop`, `TensixGrid`.

### The `23_tt_lowering` gate (matmul)

The matmul lowering counts as "in" only when
`tests/23_tt_lowering.rs::matmul` passes on tt_sim. Today it cannot even be
built, for four independent reasons, in order:

1. The file is `#![cfg(feature = "tenstorrent")]`, and `Cargo.toml` no
   longer declares that feature (`cca1c452` dropped the gates):
   `cargo test --features tenstorrent --test 23_tt_lowering matmul` fails
   with `the package 'zyx' does not contain this feature: tenstorrent`. So
   the test body has never been compiled since the refactor.
2. `backend/tenstorrent.rs` still `dlopen`s the C++ shim
   (`~/.config/zyx/libzyx_tt_shim.so`); `cca1c452` deleted
   `src/backend/tt_runtime_shim.cpp` and `build.rs`, so nothing builds it
   and `load_shim` panics on the first `Dev::TT(0)`.
3. Even with a shim, the backend never reads `TTProgramDesc.native`: it
   passes the three C++ sources to `compile_program`, so the lowering would
   not be exercised at all.
4. `tt-device`, `tt-ttsim`, `tt-firmware-images` were declared in
   `Cargo.toml` at that point but referenced nowhere in `src/`, i.e. the
   native transport was not written yet.

What the minimum native runtime needs (from `tt_kernels::runtime` and
`tt_device`, all public):
- `Device::open` over `tt_ttsim`, one owner thread (`Simulator` is
  `!Send`), `alloc_window(WindowKind::TwoMib)`, `l1_write`/`l1_read`,
  `write`/`read` for DRAM.
- `runtime::Kernel { roles: [&[Instruction]; 3], schedule, stage,
  read_back, .. }` per launch. `roles` wants `Instruction`, and zyx stores
  `Vec<u32>`, so the boundary needs the instruction defs (decode by word,
  or store `Instruction` in `TTNativeProgram`).
- Boundary moves have no executor: `TTDmMove` is data only. First cut:
  do them host-side — read each move's DRAM bytes into the launch's
  `stage` list (DRAM -> L1 before the roles run) and push the writer
  drains' L1 ranges into `read_back`, then write them to DRAM after the
  run. That keeps the in-kernel rule (no DRAM traffic inside a kernel) and
  is exactly what `runtime::Kernel` models.
- `schedule: InOrder` is what the emitter builds (no semaphores), which
  `runtime::run` supports as the no-synchronisation case.

Until all of that exists, the lowering is compile-checked only.

#### Resolved: the gate is green on tt_sim

`tests/23_tt_lowering.rs::matmul` passes on the native path (metalium's
pinned ttsim, no C++ shim). The four build blockers above are gone: the
transport and the native emitter landed on `tt-native`. What it took, and
the facts behind it:

- **CB L1 placement was sized in datums, not bytes.** `Op::Storage::len` is
  a datum count (`1024` for a tile) and `tt_place_cbs` used it directly, so
  a BF16 CB got 1024 bytes and the next CB started inside it. The reader
  staged A over B, the pack wrote over an operand, and the output tile was
  not where the writer read. Placement is now `len * element size`.
- **The unpacker reads datums 16 bytes past its `Base_address`.** `UNPACR`
  computes `InAddr = (Base_address + 1 + DigestSize) * 16`
  (`tt-kernels/src/datapath.rs::tile_base_units`): the `+1` steps over the
  16-byte tile header that every `tt_layout`/tt-metal tile *image* has. zyx
  CBs hold the datum stream with **no** header (the host tilize writes
  none, and the packer's `pack_config` cancels its own `+1` with
  `l1/16 - 1`, so packed datums start at the CB base), so the unpack base
  is `cb_l1 - 16` (`tt_src_base`). Passing the CB base made the unpacker
  read the first eight BF16 datums as a "header", shifting every operand
  face; the symptom was whole four-column blocks of the result being zero
  (they came from the zero halo).
- **F32 `Dst` -> BF16 pack is not modelled by the pinned ttsim**
  (`tensix_pacr: pack_fmt_conv_mode=0x105`; divergence doc, "BF16 late
  narrowing"). The writer CB is therefore FP32 and the boundary narrows:
  `TTDmMove::narrow` (`TT_NARROW_NONE | F16 | BF16`) is chosen in
  `tt_boundary_moves` by comparing the CB dtype with the param dtype, and
  `launch` converts while writing the bytes back (RNE, the packer's own
  rounding). `render_store` packs the CB's format, not the value's. On
  silicon the packer could narrow directly; for BF16 the host conversion
  is bit-identical (`bf16::from_f32` is RNE).
- For anyone running the gate against the *tt-metal* simulator
  (`~/Dev/cpp/tt-sim/libttsim_bh.so`) instead of metalium's vendor pin:
  that build *does* model `0x105`, but it refuses
  `PACR ReadIntfSel = 0b1111` (`tensix_pacr: read_intf_sel=15`), which is
  what metalium's `pack_rows` emits for a full four-row group. tt-metal
  itself sends `0b0000` (its `ALL_INTF_ACTIVE`), which the ISA defines as
  "all four interfaces". Only relevant if we ever target that library.
- **Slot 0 is the only DST row exercised.** `Setrwc.dst_val` and
  `DEST_TARGET_REG_CFG_PACK_SEC0_Offset` both took 0; the open question
  about slots as tile numbers (64 rows each) is unchanged.
- The emitted role words match `tt_kernels::matmul::matmul_roles` for one
  output tile (`Bf16` in L1 -> `BF16_CODE` Src -> HiFi4) word for word,
  except for where the `free` take sits relative to the two `ZEROACC`s.
  Building that reference program and diffing was the isolation trick.
- `ZYX_TT_DUMP=1` prints every role's `pc word mnemonic` and the staged
  ranges. It is how both bugs above were found.

Still failing at the time this paragraph was written; the live list is in
**CURRENT STATE** at the top. The cause is no longer `tt_verify`: phase 3 now
accepts the flat `[1024]` tile store, and `sigmoid` reaches the device. What is
open is the tile payload mapping (wrong values inside a tile), the reduce axis,
`matmul_sigmoid`'s const-dtype pattern, and `softmax`'s loops.

### Native backend

Landed (the shim and every tt-metal C++ call are gone):

- [x] `src/backend/ttsim.rs`: one owner thread for the whole process. It
      opens `tt_ttsim::Simulator` and `tt_device::Device<LibTtsim>` once and
      answers `Cmd::Run` with one `tt_kernels::runtime::run`. `Simulator` is
      `!Send + !Sync` (the library is process-global and a transport borrows
      it), so the device can never leave that thread; callers block on a
      reply channel, which keeps launches synchronous like the shim was.
- [x] **No `fork_scope`.** It forks and the child `_Exit`s at the end, so it
      cannot host a long-lived device, and its result cannot come back
      across the fork. zyx wants one device for the process, so it opens the
      simulator directly on the owner thread. (tt-tests use `fork_scope` per
      test *because* each test gets a fresh chip; zyx does not.)
- [x] **DRAM is host memory.** `TTBuffer` is `{ data: Vec<u8>, size }`;
      `host_to_pool`/`pool_to_host` are memcpys; `dispose` drops. The device
      only ever sees L1, through `runtime::Kernel::stage` (reader moves,
      before any role starts) and `read_back` (writer moves, after all three
      report `DONE`). The old shim did the same two transfers with NOC
      reads/writes, so the boundary semantics are unchanged and the hard
      invariant (no DRAM traffic *inside* a kernel) still holds.
- [x] `compile` stages the native program: role words decoded to
      `Instruction` (`tt_decode`, `InstructionDef::matches` over
      `isa::generated::ALL`; an unknown word is a compile error), `TTLoop`s
      to `code::Loop`s, `TTSemaphoreInit`s to `SemaphoreInit`s. The old C++
      `Source` ops are still rendered and ignored — the C++ emitter is dead
      weight to delete, not something `compile` reads.
- [x] `launch`: buffer args by param ordinal, reader moves copied into the
      stage list, writer moves read back afterwards. A launch whose grid is
      not 1x1 is a loud error: `runtime::run` programs one tile, and silently
      computing one tile of a grid would be wrong numbers, not a missing
      feature.
- [ ] `Pool::TT` = real DRAM ranges (only needed for silicon; the simulator
      transport is happy with host memory).
- [ ] Grid dispatch (`runtime::run` is one tile).
- [ ] BF16: expose/use metalium's expand/repack, or stage F32 at the
      boundary.

`TTDevice::release` uses `Slab::remove`, not `remove_and_return`: the value
is dropped either way, and `remove` drops it in place (the other pushes an
`MaybeUninit` and swaps the last element in, which is a move-out trick we do
not need).

### Tests / cleanup

- [ ] `23_tt_lowering`: transfer, copy kernel, sigmoid, matmul, sum.
- [ ] Loop/grid kernel.
- [ ] Multi-op chain with DST intermediates: sigmoid -> sigmoid, reduce ->
      sigmoid, matmul -> sigmoid. No CB roundtrip between them; the only
      pack is the final output boundary.
- [ ] Delete obsolete feature-gated tests and `ZYX_TT_DEBUG_MATMUL` /
      `ZYX_TT_TRACE`.

## Next up and session state

Superseded by **CURRENT STATE** at the top of this file — that section is the
single source of truth for what is green, what is broken, and in what order to
fix it. Everything below this point is design detail and history from the
`tt-native` branch.

## What stays (origin/main's compiler model)

The lowering pipeline and its IR are kept:

```
tt_tile
  -> tt_lower_llks        structured tile ops + traffic (SSA over DST)
  -> tt_dst_reuse         physical DST rows by liveness
  -> verify               SSA / tile / shape checks
  -> tt_verify            TT stream checks
```

The whole-kernel codegen also stays:

- scan the kernel once;
- split at `TTOp::EndReader` / `TTOp::EndCompute` into reader, compute,
  writer;
- collect each section's params in head order;
- number circular buffers on first touch and assign `Dst` slots by liveness;
- emit section bodies, including `Op::Loop` and `TensixGrid`;
- `TTProgramDesc`, `Params`, `TensixGrid` remain the launch metadata.

The host boundary also stays:

- `Pool::Host` is host RAM.
- `Pool::TT` is device DRAM.
- `tilize_transfer`: host row-major -> host tile bytes -> `dram_write`.
- `untilize_transfer`: `dram_read` -> host tile bytes -> host row-major.
- TT -> TT copies stay tile bytes.

## What is deleted

These `TTOp` variants are tt-metal API, not Tensix instructions. They are
removed, together with the passes that only inserted them:

| Removed | Why |
| --- | --- |
| `ReserveBack`, `PushBack`, `WaitFront`, `PopFront` | tt-metal circular-buffer API; native programs use the plan's semaphores and slots. |
| `MathLock`, `MathUnlock`, `PackLock`, `PackUnlock` | tt-metal `tile_regs_*`; native math/pack handoff is the `ready` / `free` semaphore pair. |
| `NocReadBarrier`, `NocWriteBarrier` | native data-mover entries complete in order; `dm::op::WAIT` is the only barrier needed. |
| `ReduceUninit` | native reduce programs are complete; there is no engine state to close. |
| `LLKBcast` | old, wrong, and absent from tt-metal; broadcast is an SFPU program. |
| `LLK` | C++ call templates are replaced by structured native ops. |

Removed passes: `tt_lock_dst`, `tt_dedup_pushes`, `tt_sync_cbs`,
`tt_place_pops`, `tt_init_math`. Their work is either structural (already in
`tt_lower_llks`) or handled inside the native builder program.

`Op::Asm` remains for user-supplied custom kernels, but TT lowering never
emits it.

## Native operation set

The compute section is a sequence of the following, each mapping to an
existing `tt-kernels` builder:

| `TTOp` | Native program |
| --- | --- |
| `Copy` (Circular dst) | `tt_isa::dm` entry: `op::READ` (DRAM -> L1), `op::WRITE` (L1 -> DRAM), or a CB -> CB byte copy. |
| `Copy` (Register dst) | `datapath::unpack_tile_to_dst`: the only CB -> DST transfer. |
| `Sfpu { kind, a, b, slot }` | Generic SFPU program; `kind` selects the opcode, `a`/`b` are DST inputs, `slot` is the DST result. |
| `ReduceTile { input, slot }` | `sfpu::reduce` math: DST `input` -> DST `slot`. |
| `MatmulTiles { cb_a, cb_b, slot }` | `MVMUL` reads SrcA/SrcB from the two CBs and writes the DST `slot`. |
| `EndReader` / `EndCompute` | Section markers only. |
| `ProgramDesc` / `TensixGrid` | Launch metadata only. |

A builder may emit unpack/math/pack role programs for its own internal
handoff, but zyx does not wrap that in a run-level op. This is the native
equivalent of the per-tile C++ LLK calls origin/main emitted.

## The mechanical mapping

For each origin/main C++ leaf:

| C++ | Native |
| --- | --- |
| `copy_tile(cb, 0, dst)` | `TTOp::Copy` with Register dst -> `datapath::unpack_tile_to_dst` |
| `pack` store | Output-boundary `Store`: one pack role `datapath::pack_tile_from_dst`, before the writer's DM `WRITE`. Never per intermediate op. |
| `sigmoid_tile` | `Sfpu { kind: SIGMOID, ... }`: SFPU program on DST. |
| `silu_tile` | `Mul(x, sigmoid(x))`: generic `Sfpu` Mul + fused Sigmoid; no silu variant. |
| other unary / binary | `Sfpu { kind, ... }`: one SFPU `Program`, reading/writing DST. |
| `typecast_tile` | SFPU / dtype conversion; no face-order round trip. |
| `matmul_tiles` / `mm_init` | `MatmulTiles { cb_a, cb_b, slot }` -> `matmul::matmul_roles`; CB inputs, DST output. |
| `reduce_tile` / `reduce_init` | `ReduceTile { input, slot }` -> `sfpu::reduce::roles`; DST input, DST output. |
| `cb.*` | Builder plan semaphores and L1 slots. |
| `tile_regs_*` | Builder `ready` / `free` semaphores. |
| `noc_async_read` / `write` | `dm::op::READ` / `WRITE`. |
| `for` + `i * bytes` | `code::Loop` where the body is address-invariant; otherwise unroll into the list/program. |

There is no `fill_tile` mapping: `TTOp::FillTile` was deleted because the
native builders fill DST/identity directly and the tt-metal scaler CB no
longer exists.

## Catches (the real seam)

1. **CB pages and loops.** origin/main's C++ uses a moving CB front pointer
   inside runtime `for` loops. Native builders use fixed tile slots and
   `code::Loop` is a pure block repeat (no counter, no address arithmetic).
   Unrolled kernels map exactly; a loop over a multi-page CB must either be
   unrolled per iteration or carry the index in a GPR and do the pointer math.
2. **Dst locks.** There are no `tile_regs_*` instructions in the native ISA.
   The Dst handoff is the math/pack semaphore pair. The result is identical;
   the literal call is not.
3. **BF16.** The SFPU/reduce builders consume F32 tile images. metalium's
   BF16 expand/repack helpers are `pub(crate)`, so zyx must either expose them
   or stage F32 tiles at the boundary.
4. **`ttsim` thread affinity.** `tt_ttsim::Simulator` is `!Send + !Sync`.
   The native backend needs one owner thread with a command channel; the
   shim used to hide this.
5. **Grid dispatch.** `runtime::run` is one tile. The per-tile grid loop and
   `gidx` binding are ours.

## Backend shape

```
TTDevice
  owner thread: Simulator + Device<LibTtsim> + DramAlloc
  commands: alloc / free / write / read / compile / run

Pool::TT
  ChunkId -> DramRange

compile(kernel):
  native codegen -> TTNativeProgram {
      reader: Vec<dm::Entry>,
      compute: [Vec<Instruction>; 3],
      writer: Vec<dm::Entry>,
      params, grid, cb_config, fp32,
  }

launch:
  for each tile in grid:
      stage roles, run reader list, run compute roles, run writer list
```

The reader and writer are data-mover lists; the compute section is the three
Tensix role streams. A pure data-movement kernel has no compute roles, so it
is one DM list.

## Test ladder

Baseline: the old C++ path passed `23_tt_lowering::matmul` only, and the
native path is at `transfer_roundtrip` + `matmul`. Treat every case below as
unproven until it runs green.

1. `transfer_roundtrip` — host <-> DRAM, no compute.
2. `copy` kernel — reader + writer only.
3. `sigmoid` — SFPU path, CB plan, pack.
4. `matmul` — Matrix Unit path.
5. `sum` — reduce path.
6. `loop`/grid — block repeats and `gidx`.
7. Multi-op DST chain — two compute ops with a DST intermediate, no DRAM
   and no CB round-trip.

## Non-goals

- A separate Metalium IR.
- `match_graph` / AOT graph-op shortcuts.
- TT feature flags.
- LOC or op-count as progress.
- One `Session` per op; the whole zyx kernel is one program.
