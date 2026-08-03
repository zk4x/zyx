# Tenstorrent Reduction Tiling — Implementation Plan

Goal: make the *general case* reduction kernel work on Tenstorrent by converting a
loop-carried accumulator into a tile accumulator, then lowering the terminal collapse
to the hardware `reduce_tile` op.

Reference IR (6-element `bf16_mean`-style reduce):

```
for r19 in 0..6 {
  x   = load r30[r19]                 # global input A
  y   = load r39[r19]                 # global input B
  a   = bf16(x)                       # elementwise fooder
  s   = sin(a)
  b   = bf16(y)
  sum = b + s
  f32 = cast f32(sum)
  acc = acc + v                       # loop-carried accumulator (register r16)
}
result = acc / 6 → bf16 → store       # post-loop scalar ops
```

## Anchor: `reduce_tile` hardware semantics

`reduce_tile<reduce_type, reduce_dim>(icb, icb_scaler, itile, itile_scaler, idst)`
collapses a **whole 32x32 tile** to a scalar (for `REDUCE_SCALAR`). It needs a
`reduce_init` first and a `reduce_uninit` after. `icb_scaler` holds the reduce
identity/scaling factors (all-ones for SUM).

So the loop-carried accumulator chain `acc = acc op v` becomes **one `reduce_tile`**
that collapses a tile accumulator to a single scalar.

## Target IR shape

```
# reader: load the reduce dim as 1024-element tiles into CBs
for tile in 0..chunks {
    reserve_back(1);  noc-read chunk(tile) → CB_a;  barrier;  push_back(1)
    reserve_back(1);  noc-read chunk(tile) → CB_b;  barrier;  push_back(1)
}
barrier

# compute: elementwise fooder + fold into tile accumulator
acc_tile = 0                            # 32x32 f32 register tile
for tile in 0..chunks {
    x = load CB_a tile                  # tile op
    y = load CB_b tile
    sum = bf16(sin(bf16(x))) + bf16(y)  # all elementwise → tile ops
    acc_tile = acc_tile + cast_f32(sum) # fold across chunks, no store/load
}
# writeback: collapse to scalar
reduce_init<SUM, SCALAR>(cb_acc, cb_scaler, cb_out)
reduce_tile<SUM, SCALAR>(cb_acc, cb_scaler, 0, 0, 0)    # → scalar
reduce_uninit
result = scalar / 6 → bf16 → store
```

Every op between the loads and the accumulator combine becomes a **tile op**
(elementwise). Only the accumulator structure and the terminal change.

---

## Phase 0 — Add `Op::ReduceTile` to the IR

New op that carries the tile-collapse. Modeled on `Op::Reduce` but is a *static*
single-tile collapse (no `n_axes`, it's always `REDUCE_SCALAR` — the register tile);
variants for ROW/COL can come later as a perf pass.

```rust
// kernel/mod.rs (Op enum)
ReduceTile {
    x:   OpId,   // the 32x32 accumulator tile to collapse
    rop: BOp,    // Add | Max | Mul  (SUM/MAX for now)
},
```

Thread through every exhaustive match on `Op`:

| File | Location | Action |
|------|----------|--------|
| `kernel/mod.rs` | `ser_bin` (reduce tag `22`) | add tag `23`, ser `x`, `rop` |
| `kernel/mod.rs` | `de_bin` (tag 22) | add `23` → `Op::ReduceTile { x, rop }` |
| `kernel/mod.rs` | `parameters` (~810) | `Op::ReduceTile { x, .. } => vec![*x]` |
| `kernel/mod.rs` | `parameters_mut` (~839) | `vec![x]` |
| `kernel/mod.rs` | `compute_dtypes_and_rcs` (~988) | `unreachable!()` group + ReduceTile (static, unfolded away) |
| `kernel/mod.rs` | `dtype` (~1087) | `Op::ReduceTile { x, .. } => self.dtype(*x)` |
| `kernel/mod.rs` | `name` (~1737) | `parts.push("reduce_tile")` |
| `kernel/mod.rs` | `flop_mem_rw` (~1795) | same shape math as Reduce |
| `kernel/mod.rs` | `is_reduce` (1848) | include ReduceTile |
| `kernel/mod.rs` | `shape_of` (~1891) | `s.truncate(...); s` like Reduce |
| `kernel/mod.rs` | `recursively_move`/`reduce_dims` (unfold.rs) | visit `x` |
| `kernel/licm.rs` | 38, 89 | add to `unreachable!()` group |
| `kernel/fold_constants.rs` | 41 | add to `unreachable!()` group |
| `kernel/cost.rs` | 211, 270 | add `ReduceTile` |
| `kernel/verify.rs` | ~130 | `check(x)`, dtype = dtype(x) |
| `kernel/debug.rs` | ~100 | print `reduce_tile rop r{x}` |

**`Op` must not be added to the toolchain scheduler before the pass exists** — the
existing elementwise path / unfold could encounter it. Guard so ReduceTile only
appears after the tenstorrent pass produces it (it's a native op, not unfolded).

---

## Phase 2 — General-case pass `opt_tenstorrent_tile_loop`

Single function in `kernel/tenstorrent.rs`, step-commented like `tenstor_local`.
Reuse `local_reduce.rs` prologue (lines 92–168) for structure detection:
walk back to find reg accumulator `Define`, scan loop for `reduce_bop` + acc
`load`/`store`.

Steps:
1. **Find structure**: loop, register acc, `reduce_op`, acc we load/store.
2. **Compute `chunks`** = ceil(red_dim / 1024); retarget `pad_loop` to 1024
   (OOB lanes masked to add-identity in the tail chunk; keep the existing
   `r45 * x` scale − that IS the identity mask).
3. **Reader phase**: emit CB defs for the inputs; replace the reduce-strati loop
   with a chunk loop that loads each chunk → CB (reserve/push per iteration),
   then `Op::Barrier`.
4. **Compute phase**: the elementwise foodder becomes tile ops (CB tile loads +
   same elementwise via fooder handled by existing fooder passes — becomes tile).
5. **Accumulator → tile register**: change `r16` to a 32-bit register tile
   (len=1024/256 or a tile layout); replace `acc=acc+v` + acc load/store with
   straight `acc_tile = acc_tile + v_tile`.
6. **Insert `reduce_tile()`** collapsing acc_tile → scalar register after the
   loop; the post-loop `result=acc/6`, `bf16`, `store` stay on the scalar.
7. `self.verify()`.

New codegen obligation: lower `Op::ReduceTile` in `codegen/tenstorrent.rs`
to `reduce_init` + scaler CB(ones) + `reduce_tile` + `reduce_uninit`.

---

## Phase 3 — Codegen for `Op::ReduceTile`

`codegen/tenstorrent.rs`: match `Op::ReduceTile { x, rop }`:
- emit `reduce_init<Sum, REDUCE_SCALAR>(cb_acc, cb_scaler, cb_out);`
- `reduce_tile<Sum, REDUCE_SCALAR>(cb_acc, cb_scaler, 0, 0, 0);`
- `reduce_uninit();`
- allocate/build the `icb_scaler` CB with ones (or 1/N for AVG) at host setup.

---

## 4 — Hook into dispatch

In `opt_tenstorrent_tile` (line 31), the is-a-loop branch: replace
`tenstorrent_reduce_pad + pad_loop` (currently scalar pad-to-32) with
`opt_tenstorrent_tile_loop` when a reduce-accumulator loop is detected.

---

## 5 — Tests

- Keep `bf16_mean` test; it already hits `todo!()` — should now compile/lower to
  `reduce_tile`.
- Unit tests in `tests/` for the reduce loop lowering to ensure `chunks`-based
  loop + accumulator tile + `ReduceTile` appear.
- Confirm the elementwise paths that DON'T have a reduce accumulator still take
  the existing no-loop branch.

---

## Deferred (perf special-pass, later, separate PR)

- If `chunks / gidx > 32` → `REDUCE_ROW`/`REDUCE_COL` partial then final reduce.
- If `chunks / 1024` absorbed by work-per-core → no `reduce_tile` at all (handled
  by a single post-pass dedicating more work per tensix core).
- `reduce_block` to batch consecutive tiles.
- Post-loop scalar elementwise (e.g. relu after reduce) → fold into reduce-avg.
```