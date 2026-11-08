# NOTES — Tenstorrent backend

This file is the current design, state and next steps. History is in git; do not
append session logs here. If something here is wrong, fix or delete it.

## Goal

One forward step of llama-3.2-3B on `Dev::TT(0)` (ttsim), end to end, all 28
layers from embedding to lm_head, `seq_len = 1`, checked against the CPU.

- Correctness only. It may take 10 s per step (0.1 tok/s).
- CB depth 1, no double buffering. Depth > 1 is a later optimization (it is what
  took tt_gemm to 14.444 TFLOPS at `a8fa78b9`).
- The acceptance test is the assembled model against the CPU. Green unit gates
  are a regression net, not progress.

Out of scope: perf, `tt_gemm`, qwen, kmd/silicon transport, GGUF quantization,
the decode loop, kv-cache growth. At `seq_len = 1` the causal mask and the kv
cache are unnecessary by construction.

Non-goals: a separate Metalium IR; `match_graph`/AOT shortcuts; TT feature
flags; LOC or op-count as progress; one `Session` per op (a whole zyx kernel is
one program).

## State (2026-11-08)

- `cef08d59` — `tt_tile` splits an oversized group into a loop.
- `6e886636` — per-iteration traffic at ring slot 0; CB len 1024; compute index
  with loop and group bound out.
- `bef4fa7a` — `large_elementwise` (`[1024,256]`, NEG vs host) in `23_tt_lowering`.
- `tt_add_movers` ends in a gate `todo!()`. Every TT compile stops there on
  purpose. Do not remove it until the IR at the pass exit is right.
- No TT kernel has ever executed end to end; the model has never run.
- `24_tt.rs` holds llama-shaped op groups, not the model's real 16 kernels yet.

## Pipeline

```
unfold_pows
tt_tile          normalize tiling: one tile axis + one 32-lane axis; grid split
tt_unfold        Exp2/Log2/Sin/Cos -> IR
tt_reindex       logical index -> tile-datum units (single-reduce template)
default_epilogue fold/LICM/algebra/CSE/DCE
tt_add_movers    reader/writer CB traffic + sections (device epilogue)
tt_dataflow_select  tiled SSA -> typed TTOp (row programs, matmul, copy), no
                    indexing, no tile layout knowledge
tt_sfpu_fusion   merge row programs into cones (virtual lane registers)
tt_lreg_alloc    virtual -> L0..L7, split at pressure
tt_dst_reuse     physical DST tiles by liveness (one allocator)
tt_reg_sync      role seams (Semset/Wait/Post)
tt_verify        post-allocation invariants
```

`device_epilogue` runs once per candidate. Only the search passes are optional;
everything above is non-optional and must be loud on failure.

Codegen (`codegen/tenstorrent/`) is the encoder and the renderer. The backend
(`backend/tenstorrent/`) is transport plus dispatch; it decodes nothing and
decides nothing.

## Memory model

```
LREG   8 lane registers (L0..L7 writable; L8..L15 fixed constants)
DST    8 tiles (32-bit) / 16 (16-bit). One resource: SFPU results, matmul
       accumulator, packer source
SrcA   2 banks x 1 tile   Matrix Unit input, written by the unpacker
SrcB   2 banks x 1 tile   Matrix Unit input, written by the unpacker
L1     circular buffers   unpacker reads, packer writes, movers fill/drain
DRAM   device memory      only kernel inputs/outputs
```

- The Matrix Unit does not read DST and does not read L1. `UNPACR` fills
  `SrcA`/`SrcB` from a CB; `MVMUL` accumulates into DST; the packer moves DST
  to a CB.
- `SrcA`/`SrcB` are a hand-off pipe, not storage; never a spill target.
- DST is the shared file; pressure is the real constraint.
- Spill direction: `lreg -> DST`, `DST -> L1`. Never `lreg -> L1`.
- Operand wiring: matmul A goes to `SrcB` (SEC1) and B to `SrcA` (SEC0);
  lowering must swap the CBs accordingly. Src format is TF32, not FP32.
- `SETRWC.DstVal` is four bits of 32-row halves (`row / 32`), not rows.
- `ZEROACC CLR_ALL` is only safe with no live DST value; otherwise clear the
  accumulator's rows with eight `CLR_16` blocks at `row / 8`.
- `SFPLOAD`/`SFPSTORE` address = `Imm10 + RWC.Dst`; the Matrix Unit leaves the
  counter advanced, so a matmul arm must restore it.

## Hard invariant: no DRAM round-trip inside a kernel

One zyx `Kernel` is one program: one reader list, one compute role set, one
writer list, launched once.

- DRAM is touched only at the boundary: reads of `Param Global`/`GlobalMut` by
  the reader, writes of `Param GlobalMut` by the writer.
- Intermediates live in DST or an L1 CB and are never stored to DRAM and read
  back.
- A DRAM `Copy`/`Store`/`Load` in compute is a lowering bug; `tt_verify`
  rejects it.
- Only a kernel output is packed to the writer CB. An intermediate is packed
  only when it has multiple uses (DST is single-use scratch).
- No per-op launch, no per-op `Session`, no host staging between compute ops.
- `23_tt_lowering::sum_sigmoid` and `matmul_sigmoid` are the acceptance kernels
  for this rule.

## Tiling: the search explores, `tt_tile` completes

- The search may vectorize any axis; on TT `opt_split` offers only factor 32 and
  refuses a new lane axis once an access has rank > 1. `opt_register_tiling`
  must stay device-gated (it once broke 15 of 30 gates).
- `tt_tile` fills missing axes and re-tiles nothing. An access already carrying
  two axes is a tile; a fully scalar access gets a 1-D grid
  (`Group(L/32) x Unroll(32)`); `gws_from_kernel` pads to two dims.
- Which group completes an axis: a group whose scaled form is `32*g + lane` is
  counted in **tiles** and is left alone; the other group counts **elements**
  and is padded to 32 and split into the lane axis. A bare `g*32` is not the
  test (a row stride can be 32).
- No sqrt pre-split (deleted): a lone group becomes a 1-D grid, never two
  `sqrt` factors.
- Reduces produce rank 2: `Loop(k_tiles) + Unroll(32)`, one tile per iteration.
- **Grid cap.** A group range above `max_global_work_dims[axis]` (TT `[10,14]`)
  is split by `split(g, SplitMode::Loop, factor)` with `factor` the smallest
  divisor of the length whose outer part fits. The loop wraps the whole body, so
  each iteration is one of the group's points. Enabled in `tt_tile`. If no
  divisor leaves an outer part within the max, the split is a loud rejection;
  the later form is `pad_to` the group to a multiple of the grid max, then
  split.
- **`tt_tile` chooses the tiling (user, 2026-11-08): whatever is easiest to
  lower.** The easiest form is: **one loop iteration = one tile**, so every
  access is `grid group(s) + Loop(tiles) + Unroll(32 within the tile)` and a
  group/lane never enumerates elements when a tile is the unit. Concretely:
  - a group/lane whose step is a whole tile (a multiple of 1024 datums) is the
    tile axis and becomes the `Loop`;
  - the grid is over tile ordinals and is capped by the split above;
  - the within-tile row/col lanes stay `Unroll(32)`.
  Why it is easiest: the CB ring is one tile deep, the traffic is one whole
  tile per iteration (a congruent `TILE_SLOT` image), the compute index is the
  tile's own `row*32+col`, and `LIST_MAX` is satisfied by construction. Kernel 1
  (`[4096,8,128]`): 16384 tiles, 140 cores → ~117 tiles per core, 117 write
  entries, versus the current 32-tile-lane form whose 4096 per core rows cannot
  fit one list.

## Boundary indexing: `tt_canonical_tile`

The only place that knows the tile layout. Shape from `self.shape(base)`
(kept on the param for TT since `linearize`):

```
(rows, cols) = last two dims;  rank-1 [c] = (c, 1);  [] = (1, 1)
ntr = ceil(rows/32)   ntc = ceil(cols/32)   el = dtype bits / 8
E     = the load/store's own index, unchanged
col   = E % cols
row   = (E / cols) % rows
b     = E / (rows * cols)
tile  = b*ntr*ntc + (row/32)*ntc + (col/32)
face  = ((row % 32) / 16) * 2 + ((col % 32) / 16)
within= (row % 16) * 16 + (col % 16)
dram  = tile*TILE_SLOT + TILE_DATA + (face*256 + within)*el
l1    = tile*1024 + (row % 32)*32 + (col % 32)
```

- `dram` is a physical byte address with the tilize inverse appended. The index
  is never rewritten into a tile ordinal, never divided by a footprint, and no
  lane is stripped. It may be non-affine (`/`, `%`).
- `l1` is the canonical tile the compute segment and SFPU already use.
- The transform is emitted with plain IR (`Const`, `Binary` Div/Mod/Mul/Add/Sub)
  at the access's anchor; a fresh const per use.
- **Two index conventions.** `tt_reindex` rewrites some kernels to tile-datum
  units (`tile*1024 + pos`); others keep the flat logical index.
  `tt_canonical_tile` accepts both, discriminated by
  `tt_index_max(idx) >= outer*rows*cols` (a logical index cannot exceed its
  buffer) plus a loop whose stride is a multiple of 1024.
- **Compute side**: the index is canonicalized with the loop and group bound out
  (each iteration sees its own tile): `row*32 + col`, max 1023. The reader and
  writer resolve the DRAM address from the loop; the CB side is the ring slot.
- **No metadata, ever**: no layout field, side table, marker, or orientation
  field. `linearize` is shared and stays semantically as it is; TT conforms.
- Per-grid-point offsets: `TTDmRead`/`TTDmWrite`/`TTDmTileRead`/`TTDmMove` carry
  `offsets: Box<[u32]>` in launch order (`axis0*extent1 + axis1`). The launch
  indexes by the **dense** grid point, not the chip's tile index. A boundary
  index naming a `Param { Variable }` cannot be tabled (compile-time need):
  `llama_kv_cache_assign_*` is that case.

## CB ring and credits

tt-metal is the reference (`~/Dev/cpp/tt-metal/tt_metal/hw/inc`).

- **A CB is a fixed-depth ring, independent of the loop.** tt-metal's kernels
  loop over `n_tiles`; the CB is `tiles_per_cb` pages and the trip count never
  sizes it. The operands are per-page. Depth 1 now; the slot is
  `(iter % depth) * 1024`, i.e. 0.
- **Two counters per CB**, `received` (pushes) and `acked` (pops).
  `reserve_back` waits `depth - (received - acked) >= n`; `wait_front` waits
  `received - acked >= n`; both poll and do not advance pointers. `push_back`
  advances the write pointer and `received`; `pop_front` advances the read
  pointer and `acked`. Both wrap only at the fifo limit. Total pushed/popped per
  cycle must equal the depth.
- **One producer and one consumer per CB.** Only one thread pushes a CB, only
  one pops it.
- **Four endpoints** (`tt_isa::dataflow`):
  reader (B) `Reserve`/`Push` on Input; unpack (T0) `Wait`/`Pop` on Input;
  pack (T2) `Reserve`/`Push` on Output; writer (NC) `Wait`/`Pop` on Output.
  Capacity < 0x8000.
- **Where each half runs.** The reader/writer halves are `dm::op::CB` entries
  (`dm::Entry::Buffer`) in their mover lists. The unpack/pack halves run in the
  role runner, which already implements the streamed program:
  `[VERSION, batches, capacity, role]` + `[body_addr, body_len, 0, 0]` per
  batch, with `Wait`/`Pop` (T0) or `Reserve`/`Push` (T2) around each batch body
  and `retire_batch` between batches
  (`firmware/tt-firmware/src/corpus.rs`). Batch bodies must be resident
  programs in `l1::PROGRAM_CACHE`; a `(0,0)` batch is an empty body.
- **The loop lives in the kernel**, one tile per iteration, in all three roles:
  `for i { reserve; read; push }` (reader), `for i { wait; compute; pop;
  reserve; pack; push }` (compute), `for i { wait; write; pop }` (writer).
- **Launch order.** Today `execute()` runs the B list to completion, then the
  roles, then the NC list, and `mover_list` always waits. With per-iteration
  credits at depth 1 the second `Reserve` blocks forever. The launch must
  submit the B and NC lists without waiting, run the roles, then join both.
  Split `mover_list` into submit (`OP`/`LEN`/`SEQ`) and `wait_done`.
- **Caps.** `dm::LIST_MAX = 512` entries per role per core; the entry count is
  stream shape, so it is checked at compile time (`tt_verify`), never split at
  launch. A kernel that does not fit is a loud error.
- A slot index in a transaction must be the loop counter or 0; anything else is
  a compile error.

## 64-byte congruence

The mover validates every entry: a `READ`/`WRITE` needs
`dram_offset % 64 == l1 % 64` for reads (`% 16` for writes), else error 4.

- The canonical transform adds the same datum offset on both sides, so whole
  tiles and rows are congruent.
- CB bases are placed at `≡16 mod 64` (`tt_place_cbs`), where a tile image's
  datum block lives; a tile slot is `TILE_SLOT = 4160` with a 16-byte header.
- A move below the mover's granularity, or whose two sides disagree, is a
  compile error. The fix for a partial move is per-entry chunking: round the
  DRAM address down to 64 bytes and shift the L1 address by the same delta,
  with the CB side in datum units (`tile*1024 + datum`).
- `fifo_check_copy` is the proof: per-variable stride congruence; an unprovable
  index is a loud rejection.

## Compilation, images, dispatch

Compilation ends at codegen. What runs is the images, not the `Kernel`.

- `Image { core, address, len }` per core's bytes in device DRAM; `TTProgram`
  holds a `sequence` of `ImageId`s plus `lens`, `gws`, param counts.
- Residency is per `(tile, core)` over five slots (B, NC, T0, T1, T2), keyed by
  `(ImageId, generation)`; a miss is a NoC load from the image's DRAM copy.
  Mover firmware is an image; boot is its first residency miss.
- A launch binds arguments and issues: chunk addresses, variable values, grid
  size, the sequence/generation words. It adds, removes, reorders or encodes
  nothing, and computes no strides.
- Grid dispatch is a launch loop over the grid points (dense order); the
  program is tile-uniform and only the DRAM slice differs.
- `Pool::TT` is device DRAM; host staging is upload/readback only. The
  DRAM↔L1 boundary belongs to the movers, never the host. A 16-bit output's
  narrowing is the host's only because the pinned ttsim refuses the packer's
  format conversion (`pack_fmt_conv_mode=0x105`); silicon does it on the card.
- `TTDmMove`, role programs (`TTOp::Unpack`/`Math`/`Pack` with
  `TTRoleProgram { words, loops }`), `DmRead`/`DmWrite` pages and the section
  markers are the rendered IR; `compile` collects them, it does not decide.

## Loops

- The Tensix coprocessor has no branch. Loop control is `REPLAY` (a body that
  fits its buffer: 32 words) or the mailbox loop table
  (`code::Loop { start, len, count }`, at most 4 per role, `count <= 128`).
- Trip counts must fold to `Const`; a `Param`-backed count is a loud TT compile
  error.
- Unrolling is a search decision before the device epilogue, never a backend
  choice.
- A role program whose body is longer than the replay buffer is stored once and
  repeated through the loop table; `start` indexes the role program including
  its preamble (mask `LOOPED`/`STREAMED` wherever a length is used as a count).
- `TTRoleProgram` is a flat word list plus `loops` today. The per-batch streamed
  form above is what per-iteration credits need.

## Restore sources (do not reinvent)

- `dc9a1e3d^` (pre-native emitter) is the reference implementation of the CB
  mechanics, and it drove the TT path including tt_gemm:
  - `TTOp::{ReserveBack, PushBack, WaitFront, PopFront}` and
    `{MathLock, MathUnlock, PackLock, PackUnlock}`.
  - `tt_sync_cbs` (577 lines): places the four syncs; one tile per iteration
    when `trip > depth`, a block only when the block fits the CB; rejects a slot
    that is neither the loop counter nor 0.
  - `tt_place_pops`: places pops on the wait-carrying IR; rebases windowed
    slots.
  - `tt_fifo_check`: reserve/wait balance at each section end,
    `pushed == popped`, producer-past-depth (deadlock guard), symbolic-loop
    traffic rejected.
  - `fifo_check_copy`: 64-byte congruence per variable stride.
  - `tt_verify`: DST lock firewall.
  - Pipeline there: `tt_tile -> tt_lower_llks -> tt_lock_dst ->
    tt_dedup_pushes -> tt_init_math -> tt_sync_cbs -> tt_place_pops -> verify
    -> CSE -> DCE -> tt_verify`.
- `588110a9` is the same mechanism one generation earlier
  (`CBState`/`CBBatch`/`CBEvent`/`CBEmitter`, batching pre-pass), where
  `t > config[cb]` stayed streaming and the verifier made an overfill a compile
  error.
- `a8fa78b9`: 14.444 TFLOPS, bad 0, M5120 N11264 K1024 with depth 2.
- `~/Dev/cpp/tt-metal` is the hardware reference for CB semantics and the
  `tiles_per_cb`/`cb_reserve_back`/`cb_push_back`/`cb_wait_front`/`cb_pop_front`
  shape. `examples/qwen3.8-27b/tenstorrent_debugging.md` (git history) is the
  five-core / reserve-push-wait-pop reference.

## Invariants that do not move

1. A group range over the device max is invalid; the grid must fit.
2. Loops must be handled correctly.
3. Streaming CBs and unrolled loops are invalid: the CB is never sized from the
   loop, and a loop's boundary traffic is never hoisted into one block copy.
4. A non-canonical tile access inside compute is invalid.
5. `tt_dataflow_select` only pattern-matches the already-tiled, already-canonical
   kernel and lowers to the native ISA. It must never work out an index or a
   tile; if it has to, fix `tt_add_movers`.
6. No metadata; the shape and the index are the only layout inputs.
7. No silent fallthrough. `todo!()` is a report; a made-up value is a bug.
8. Custom kernels bypass the e-graph and autotune; `linearize` runs first.

## Acceptance of an SFPU kernel

For a pure elementwise kernel (`sigmoid`, `silu`, `gelu`, `elu`, ...) the
emitted T1 program must be, per row, one `SFPLOAD` from the input tile and one
`SFPSTORE` to the output tile, everything else in `L0..L7`. The acceptance test
lives in `#[cfg(test)] mod acceptance` at the end of
`kernel/tenstorrent.rs` (pure IR, ~30 ms). Extend its `cases` array when adding
an activation.

## Known failures and the 16 kernels

`24_tt` (llama-shaped op groups) currently:

| test | failure |
|---|---|
| `llama_attention_softmax`, `llama_promote_realized_eager_load` | `tt_verify: OpId(N) is not fully lowered (SSA compute remains)` |
| `llama_embedding_one_hot` | `cast U32 -> F32 has no native row-op program` |
| `llama_kv_cache_assign_*` | boundary index names `Param { Variable }` (not evaluable at compile time) |
| `llama_rms_norm` | `to_device: host-staged tensor has no buffer` |
| `llama_rope` | `B mover refused list 2: error 4` (64-byte congruence) |
| `23_tt_lowering::sum_drop_col` | scalar-per-column store lands at `row*32`, not the face slot |

`24_tt` must be replaced by the model's real 16 kernels, one test each, in the
order the model compiles them (the order comes from `ZYX_DEBUG=4` on the model,
not from the test file). Each real kernel replaces one approximation as its IR
comes right.

## Debugging

Run cargo from `zyx/` (the root has no `Cargo.toml`).

```bash
cd zyx && cargo build
cd zyx && AGENT=1 cargo test --lib
cd zyx && AGENT=1 cargo test --test 23_tt_lowering --no-run
BIN=$(ls -t target/debug/deps/23_tt_lowering-* | grep -v '\.d$' | head -1)
TT_TTSIM_LIB=/home/x/Dev/rust/metalium-rs/vendor/libttsim_bh.so \
  AGENT=1 timeout 10 $BIN <gate> --exact --nocapture
```

- **One gate per process.** A failed ttsim thread poisons every later test; a
  suite run reports a meaningless cascade.
- 10 s per individual run is the cap; a heavy gate must be run alone.
- `ZYX_TT_IR=1` dumps the IR before/after `tt_tile`, `tt_reindex`,
  `tt_add_movers` and after `tt_dataflow_select`.
- `ZYX_DEBUG=256` NoSearch (seed kernel only); `ZYX_DEBUG=264` (256+8) is the
  post-epilogue TT dump; `ZYX_DEBUG=8` alone prints nothing on the autotune
  path.
- `ZYX_TT_DUMP=1` prints the mover lists and role byte counts; `ZYX_DST_DUMP=1`
  sets `dump_rows = 16` and prints `Outcome.dst`; `ZYX_MOVE_DUMP=1` prints the
  moves.
- The DST dump covers rows 0..15 only (`mailbox::DUMP_MAX_ROWS = 16`).
- A test that vanishes with no failure line is a `libttsim` BAR violation: the
  library calls `_Exit`.
- Run `cargo fmt`; format a single file with
  `rustfmt --config skip_children=true <file>`.
- `TT_DRAM_PAGE_BYTES = 4096`; `TT_TTSIM_LIB` must point at the vendor pin
  (`/home/x/Dev/rust/metalium-rs/vendor/libttsim_bh.so`).

## ISA and ttsim facts

- A tile is 32x32 datums; a DST row is 16 datums, so a tile is 64 DST rows
  (32 rows for 16-bit tiles).
- Data-mover entries are `[u32; 8]`: `[op, channel, port, offset, l1, len, 0, 0]`.
  `dm::op::{READ=1, WRITE=2, LIST=3, READ_TRANSPOSED=4, FILL=5, KERNEL=6,
  WAIT=7, READ_BROADCAST_COL=8, BARRIER=9, CALL=10, CB=0x42}`.
- `dm::LIST = 0x1_4000`, `LIST_MAX = 512`, `ENTRY_BYTES = 32`,
  `TILE_SLOT = 4160`, `TILE_DATA = 16`, `LIST`/`SCRATCH` per mover, NC has its
  own list.
- `tt_isa::dataflow`: `VERSION = 1`, `STEP_WORDS = 4`,
  `STREAMED = 1 << 30`, `LOOPED = 1 << 31`, counters at `INPUT`/`OUTPUT`/
  `TRANSFER`, `Action::{Reserve, Wait, Push, Pop}`,
  `Stream::{Input, Output, Transfer}`, `Endpoint::{Reader, Unpack, Pack,
  Writer}`.
- `mailbox::PROGRAM_MAX = 8192` words, four program slots,
  `loops::MAX = 4`.
- Semaphores: 8, values 0..15, one per `(producer, consumer)` edge. `SEMWAIT`
  is a level test, not a reservation; `SEMGET` must not follow a zero counter.
  `tt_reg_sync` assigns one semaphore per ordered edge.
- `SFPLOAD`/`SFPSTORE` carry an absolute DST row in `Imm10`; `Program::new`
  clears the row counter and sets address modifier 0.
- metalium's SFPU/reduce builders hardcode `A_ROW = 0`, `B_ROW = 64`,
  `OUT_ROW = 128`, `SPILL_ROW = 256`; the emitter remaps them by rewriting
  `Imm10`.
- `SFPENCC` is a predicate reset, not a scope: a program that predicates needs
  one at the start, and interleaved parts need their own.
- An `SFPNOP` is required after MAD-class ops before `SFPAND`/`SFPOR`/
  `SFPIADD`/`SFPSHFT` (and `SFPSHFT2` rotations after a MAD).
- `SFPABS Mod1 = 1` is not `x & 0x7fff_ffff` (it keeps a negative NaN's sign).
- `Bool` is a raw 1/0 word; `f32(bool)` is an int-to-float conversion.
- `Select` requires a `Bool` condition (verify enforces it).
- The three roles are T0 (unpack), T1 (math), T2 (pack); B and NC are the two
  movers. The ISA's role order is the hardware's; the compiler's role tags are
  T0/T1/T2.
- ttsim models only `(0,0)` FP32, `(0,4)` TF32 and `(8,8)` INT32 unpack-to-DST;
  16-bit elementwise kernels cannot run on it. There are no implicit
  conversions: a BF16 compute kernel is a loud codegen error.

## metalium-rs API (verified)

Path prefix `~/Dev/rust/metalium-rs`.

- `runtime::Kernel { roles: [&[Instruction]; 3], schedule, stage, read_back,
  dump_rows, clear_dst, dst_fmt, ... loops }`; `runtime::run(dev, tile,
  images, kernel, budget)`; `RoleImages = [(Core, &[u8], u64); 3]`.
- `SemaphoreInit = (Semaphore, u8, u8)`; `Schedule::Concurrent(&init)` prepends
  a setup program; a kernel must restore semaphores.
- No public word -> `Instruction` decoder: a backend storing words must match
  `InstructionDef::matches` over `isa::generated::ALL` or keep the emit-time
  kind.
- `code::Loop { start, len, count }`; `code::stored(words, &loops)` packs the
  length word with `LOOPED`.
- L1 placement: `l1::Requirements::cb(...)`, `l1::plan(arena)`, `Plan::addr`.
  There is no runtime CB config; addresses are build-time immediates.
- Builders: `datapath::{unpack_tile_to_dst, pack_tile_from_dst, thread_config,
  config_program, tile_base_units}`; `sfpu::ops::program_for(kind, scalars,
  bcast)`; `sfpu::kernel::{plan_layout, roles, roles_code}`;
  `sfpu::reduce::{accumulate, accumulate_in_order, finish, roles, roles_at}`;
  `matmul::matmul_roles(outputs, sems, in_fmt, out_fmt, fidelity)`.
- `Device`: `open`, `alloc_window`, `l1_read`/`l1_write`, `dram_grid`,
  `dram_write`/`dram_read`; `tt_kernels::tensor::DramAlloc`.
- `tt_ttsim::Simulator` is `!Send + !Sync` and a process singleton; fatal
  errors `_Exit`, so tests use `fork_scope` (zyx uses one owner thread
  instead and no fork).
- `PROGRAM_MAX = 8192` words, `PROGRAM_SLOT = 0x8000`.
- `bf16::{expand, repack*}` and `fp16::expand` are `pub(crate)`.

## Reduces, rank-1 layout and the shape cone

- `TileDim` names the **result**, not the folded axis: `Row` = one value per
  row, carried in the result tile's first **column**; `Col` = one value per
  column, carried in the first **row**. `tt_reduce_program`: `Row` ->
  `accumulate(op, Axis::Cols)`, `Col` -> `accumulate(op, Axis::Rows)`.
- A rank-1 `[N]` is a **column** of values (element `i` = row `i`, lane 0).
  `tilize_transfer`/`untilize_transfer` take rank-1 that way, and the reduce
  emitter writes it that way. A `[N,1]` keepdim value uses the same column
  layout.
- `sum([0,1])` and `sum_all()` fold one axis at a time, so the chain is a `Row`
  kernel then a `Col` kernel; `TileDim::Scalar` is structurally unreachable
  (`Op::Reduce` carries one axis).
- `Op::Param.shape` is kept through `linearize` on a tiled device; the shape is
  an operand and its cone must be hoisted ahead of the params by `toposort`,
  `instruction_schedule`, `move_constants_to_beginning`, and exempted in
  `verify`'s declared-before-use phase. Without those the cone lands after the
  params and passes panic.
- `tt_reindex`'s stride classification (when it rewrites): a term that is a
  multiple of the plane is the outer slice; a multiple of `32 * cols` is a tile
  row; a multiple of the row width is a row; a multiple of 32 is a tile column;
  anything else is a lane.
- `tt_tile`'s nested-loop fold must require the outer coefficient to be
  exactly `il` and rebuild the composition as `(acc - flat*il) + flat`; the
  invariant part of the index (a row base) must survive.

## Test ladder

Every case below is unproven until it runs green on ttsim:

1. `transfer_roundtrip` — host <-> DRAM, no compute.
2. `copy` kernel — reader + writer only.
3. `sigmoid` — SFPU path, CB plan, pack.
4. `matmul` — Matrix Unit path.
5. `sum` — reduce path.
6. Loop/grid — block repeats and the grid split.
7. Multi-op DST chain — two compute ops with a DST intermediate, no DRAM and
   no CB round-trip.
8. One llama forward step, compared against CPU.

## Measured dead ends (do not retry)

- `SETRWC`/`DstVal` as a per-loop base: `DstVal` adds in rows and `at` is
  absolute; keep `DstVal = 0` and the base at 0.
- Scheduling the SFPU bodies to move a pressure peak: the peak sits inside a
  conditional and moving a scope is unsafe (lane-masked writes).
- Cheapest / use-count spill eviction: cascades.
- Cone shrinking instead of splitting: miscompiles.
- Exempting shape ops from `verify`'s `has_move_or_reduce`: only moves the
  failure.
- `opt_register_tiling` without the TT gate: breaks reduce/matmul lowering.
- `Op::Expand` before `tt_tile`: breaks `split`'s accumulator pattern.
- Unrolling IR loops inside `render_section`: hangs; use the loop table.
- Pinning a root `rust-toolchain.toml` at 1.88.0: the firmware role images
  overflow their 10 KiB L1 slots.
- A shared global `target-dir`: a second checkout can shadow the test binaries.

## Open questions (unclear, to settle one by one)

### Tiling and loops

1. Kernel 1's tile axis is `Unroll(32)` and one iteration covers 32 tile rows.
   **Answered (2026-11-08): `tt_tile` may choose any tiling that is easiest to
   lower; the chosen rule is one iteration = one tile (see "Tiling").** The
   change is to turn a tile-stepping `Unroll` into the `Loop`.
2. If a group length has no divisor whose outer part fits `max_global_work_dims`
   (a prime above the max), what is the IR: a tail loop, an over-allocated
   outer with masking, or a loud rejection?
   **Answered (2026-11-08): loud rejection for now. Later, `pad_to` the group to
   a multiple of the grid max and split afterwards.**
3. With both a grid-split loop and a tile loop present, which loop is the batch
   loop? (The current answer is the tile loop, one tile per batch.)
4. Does `tt_tile` ever fold an existing lane split back into a loop, and under
   what condition?
5. Are the two group axes split independently, and may their loop trip counts
   differ?

### CB and credits

6. Who places a batch body in `l1::PROGRAM_CACHE`, and who owns that region?
7. Is that placement compile-time (an image) or launch-time (a mover write)?
8. Must T0/T1/T2 have identical batch counts? (T1 does no credit action but is
   synchronized by `retire_batch`.)
9. May `Reserve n` / `Push n` with `n > 1` be the default so a list fits
   `LIST_MAX`, and how does the slot advance per push then?
10. Is one list per mover per launch fixed, or may the launch feed several
    lists as credits free?
11. May the NC list be submitted before the roles run (it starts with a
    `Wait` on Output) and joined after?
12. Is `tt_place_pops` needed at all at depth 1, where every slot is 0?
13. Is an explicit `CbDeclare` needed, or is the depth just
    `storage.len / 1024`?
14. Does the pack's baked L1 destination need the ring slot at depth > 1, and
    does it therefore have to stop carrying the loop at depth 1 only?

### Launch and execution

15. Does ttsim advance the movers and the roles together under `tick` when a
    list is submitted without waiting?
16. Can the B mover load a program while a reader list is outstanding on it?
17. Is the order `load programs -> submit B -> submit NC -> run roles -> join`
    valid, or must program loading happen after the lists are drained?
18. If a mover refuses an entry while the roles are running, how is the error
    surfaced and how are the roles stopped?

### The model and the 16 kernels

19. What is the exact list of the model's 16 kernels (ops, shapes, order)?
20. Are all 16 reached at `seq_len = 1` decode, given that CPU prefill crashes?
21. What counts as one kernel: one compiled `Kernel` per fused op group, with
    repeated invocations sharing it?
22. Does `24_tt` get replaced one-for-one by the 16, or extended?

### Correctness details

23. When a per-iteration DRAM step is not a multiple of 64 bytes, is the answer
    per-entry chunking or a whole-tile move?
24. Is `sum_drop_col`'s orientation needed for llama at `seq_len = 1` (attention
    over one key is `v`), and is its shape the one the model actually uses?

## Next

1. Restore the four CB ops, `tt_sync_cbs`/`tt_place_pops`,
   `tt_fifo_check`/`fifo_check_copy` and `tt_verify` from `dc9a1e3d^`, adapted
   to today's traffic (`Op::Copy`/`Op::Store` on `Circular`).
2. Emit credits in `tt_add_movers` at depth 1, per iteration; keep the gate
   until the IR is right.
3. Codegen: `dm::op::CB` entries in the B/NC lists; streamed role program with
   batch bodies in `l1::PROGRAM_CACHE`; launch submit/join.
4. `tt_fifo_check` + `fifo_check_copy` must pass on the IR.
5. Replace `24_tt` with the model's 16 real kernels, in model order, until they
   lower.
6. One forward step of llama on ttsim, compared against CPU.
