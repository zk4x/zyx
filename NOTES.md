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

## State (2026-11-08, end of session)

Commits this session:

- `cef08d59` — `tt_tile` splits an oversized group into a loop.
- `6e886636` — per-iteration traffic at ring slot 0; CB len 1024; compute index
  with loop and group bound out.
- `bef4fa7a` — `large_elementwise` (`[1024,256]`, NEG vs host) in `23_tt_lowering`.
- `85591f5d` — firmware sources inside `zyx`; `tt-isa` from crates.io; the
  `tt-firmware-images` crate deleted.
- `tt-isa 0.1.0` published on crates.io (published by `zk4x`, repo
  `zk4x/tt-isa`, Apache-2.0, credited to the metalium-rs authors, transfer on
  request). `zyx` itself is **not** published.

Where things stand (2026-11-09):

- **Uncommitted tree.** Everything below is in the working tree, not in a
  commit.
- A debugging gate `todo!()` sits at the **start of `tt_dataflow_select`**. Every
  TT compile stops there on purpose: the IR at that boundary must show the CB
  syncs (reserve/wait/push/pop) before the gate is removed for that kernel. When
  a kernel does not pass, the gate is put back, the IR is checked, and only when
  it is good is the gate removed and the later passes debugged.
- **Step 1 (CB ops) is in the tree:** the four constructors, `tt_sync_cbs`,
  `tt_dedup_pushes`, `tt_place_pops`, `tt_fifo_check`/`fifo_check_copy`, wired in
  `device_epilogue` after `tt_add_movers` in the recorded order.
  `cargo build` clean; `cargo test --lib` 129 pass, 1 fail (the gate);
  `ir_contract` is mid-rewrite: the gather fixture never reaches
  `tt_add_movers`, so the earlier frozen boundary had no CB traffic; the
  tile-copy fixture written after that is dodging (see "No dodging") and the
  test is red until the scalar row gather's ideal IR is in place.
- **The gather is the current task, not out of scope.** The scalar row gather is
  in the tree and its ideal IR is frozen: `weight.gather(0, indices)` shape, rows
  of 16 F32 = 64 B. `tt_add_movers` now has a row-traffic body: every Global
  load moves the 64-byte span containing each lane (one span per lane,
  concatenated in the CB page), a data-dependent index is read from its own CB
  first, and every Global store moves the row span it writes. The contract test
  `ir_contract::gather_pipeline_ir_is_ideal` asserts the entry IR and the
  boundary IR and is green. What is left is the reader/writer programs and the
  launch, i.e. steps 2 and 5 below.
- **The reader and writer programs are emitted and run on ttsim.** New on
  2026-11-10: `codegen/tenstorrent/boundary.rs` lowers the reader section (head
  .. `EndReader`) and the writer section (after `EndCompute`) to RISC-V over the
  boundary IR: index expressions as written (const, grid coordinate, lane loop,
  arithmetic, a CB load), one span per lane for a vector copy, inline NOC and CB
  ops. `riscv.rs` gained `lhu`/`sh`, RV32M, `bltu`/`bgeu`, `cb::{reserve_back,
  wait_front, push_back, pop_front}` over tt-metal's `LocalCBInterface`, and
  `noc::issue_addr` (high address words from registers). Green on ttsim: the four
  CB syncs, RV32M, NOC with register highs, the gather's reader program (index
  span -> token in L1 -> 16 per-lane row spans), and the writer program. What is
  left is wiring them into `render_tt`/the launch (steps 3-5) so the compiled
  kernel uses them instead of the `dm` lists.
- **The `tt_dataflow_select` gate is removed** (both the gather's and the
  elementwise tile kernel's boundary IR are frozen and good). Removing it exposed
  an acceptance failure (two SFPU row bodies per activation): `tt_place_pops`
  was putting a CB's pop after the loaded value's **first** use, which landed
  between two SFPU row programs and blocked `tt_sfpu_fusion`. Fixed: a pop
  follows the whole cone the value feeds (transitive SSA users within the
  section), so the row programs fuse and `activations_are_one_row_body` is green
  again (`135` lib tests pass). The acceptance rule stands; the lowering was
  right.
- **Pure copy kernels are transfers (user, 2026-11-10): `tt_add_movers` must not
  split them into sections.** A kernel whose every Global store writes a Global
  load's value has no compute: following tt-metal's DRAM loopback it is one
  mover program with plain L1 pages, no circular buffers, no four syncs, no
  `EndReader`/`EndCompute`, no writer role. In the tree: `tt_add_transfer_body`
  builds that IR (Local pages, the index read/pick, the row read and write);
  `boundary::reader` lowers it; `render_tt` emits only `TTOp::Reader`; the launch
  builds a `ttsim::Transfer` (words + args: grid point, tile high address,
  per-parameter DRAM address) and `execute` runs it through `exec_b`. Green: the
  transfer program runs on ttsim and writes the gathered row.
  **Blocker found:** the transfer addresses raw row-major DRAM (`token*64`,
  `s*64`), while `Pool::TT` stores tensors as tiled images
  (`TILE_SLOT`/`TILE_DATA` face layout). The end-to-end kernel test
  (`gather_kernel_runs_through_the_backend`) is therefore `#[ignore]`d: the
  transfer must compute canonical tiled addresses (`tt_canonical_tile`) for its
  reads and writes. That is the next step.
- **The gather's boundary IR** (frozen, hash `0xbad0_5a0d_2a40_4353`): two Local
  pages, `len=8` (the 64-byte index span) and `len=16` (the 64-byte row span).
  One section, no CBs: `copy idx_page[0] = idx[0]`, `load idx_page[s]` (the
  token picked in L1), `copy row_page[0] = weight[token*64]`, and
  `copy out[s*64] = row_page[0]`. No compute, no writer, no syncs.
- `tt_tile` no longer pads a group range into lanes when the access already
  carries a lane axis: the group *is* the grid axis (padding it made the gather's
  `idx[s]` a masked lane and guarded the pack, which no writer could match).
- **One TT program has executed end to end:** the gather writeback on RISCV B,
  through the emitted RISC-V and the `exec_b` firmware (see below). No
  tensor-level kernel has run yet; the model has never run.
- `24_tt.rs` holds llama-shaped op groups, not the model's real 16 kernels yet.
- `cargo build`: clean, no warnings.
- `cargo test --lib`: 129 pass, 1 fail —
  `kernel::tenstorrent::acceptance::activations_are_one_row_body` panics at the
  gate. Expected while the gate stands.
- `cargo test --test 23_tt_lowering`: all gates stop at the same gate except the
  new `gather` test, which stops in `tt_sync_cbs` (partial-page publish).
- `cargo test --test 24_tt --no-run`: compiles.
- `cargo package`: was verified earlier (172 files, 1.3 MiB compressed); it will
  not verify now because `tt-isa` is a path dependency.

What exists now:

- `src/codegen/tenstorrent/riscv.rs`: RV32I encoders and `Assembler`
  (labels/branch fixup), inline NOC issue/wait/drain, and `Gather::program` (row
  gather, then sequential writeback, one loop). 7 unit tests.
- `firmware/tt-firmware/src/bin/exec_b.rs`: executes a staged program on RISCV B
  through `tt_isa::mailbox::exec` — poll `SEQ`, validate, invalidate the
  I-cache, call with `a0` = args, publish `RET`/`DONE`. `ttinsn::call_with` is
  the call. The image is built and instruction-gated.
- `ttsim::exec`: the owner-thread command that stages L1 blocks, boots `EXEC_B`,
  runs one generation and reads DRAM back.
- Green test: `backend::tenstorrent::ttsim::tests::gather_writeback_runs_on_riscv_b`
  — four 64-byte rows read in the order 2, 0, 3, 1, checked against the host.
- IR contract test: `kernel::tenstorrent::ir_contract` — **being rewritten to
  the scalar row gather's ideal IR and red until the passes produce it.** Its two
  earlier states are both failures: the gather fixture never reached
  `tt_add_movers` (an `Op::Load` in the index cone made it return early, so the
  frozen boundary had no CB traffic) and the strings were copied from a run; the
  tile-copy fixture written next was dodging (see "No dodging"). See "Debugging"
  for what the
  test must become.
- `TTOp::Reader(Box<TTRoleProgram>)` and `TTOp::Writer(...)` exist in the IR,
  with debug printing; nothing emits them yet.
- `TTOp::{ReserveBack, PushBack, WaitFront, PopFront}` (`{ cb: OpId, n: u8 }`)
  exist as variants, with debug printing and parameter/effect arms; their
  constructors and the passes that place them are in the tree (see "Step 1"
  above and the loop-shape evidence under "Restore sources").
- `tt-isa` is a path dependency (`../../tt-isa`) while it is edited; the
  firmware manifest uses an absolute path to it (`tt-firmware/Cargo.toml.in`),
  which must become the published version before packaging.

## Session start

1. Read this file end to end. Do not grep it for a keyword.
2. `cd zyx && cargo build`; `cargo test --lib` (the gate failure is the only
   expected red); `cargo test --test 24_tt --no-run`.
3. Run the emitter anchor:
   `TT_TTSIM_LIB=/home/x/Dev/rust/metalium-rs/vendor/libttsim_bh.so cargo test
   --lib backend::tenstorrent::ttsim::tests::gather_writeback_runs_on_riscv_b -- --exact`.
   `ir_contract` is red by design while it is rewritten to the scalar row
   gather's ideal IR; do not treat it as an anchor until it is green for that
   reason.
4. Settle the open questions below in order, one at a time. Do not implement
   ahead of an unanswered question.
5. Then work the `Next` list. Do not scope it down: the target is the full
   Tenstorrent backend, one complete step at a time.

## Pipeline

```
unfold_pows
tt_tile          normalize tiling: one tile axis + one 32-lane axis; grid split
tt_unfold        Exp2/Log2/Sin/Cos -> IR
tt_reindex       logical index -> tile-datum units (single-reduce template)
default_epilogue fold/LICM/algebra/CSE/DCE
tt_add_movers    reader/writer CB traffic + EndReader/EndCompute section markers
tt_sync_cbs      place ReserveBack/PushBack/WaitFront/PopFront (restored)
tt_dedup_pushes  merge adjacent PushBack
tt_place_pops    place PopFront after the last read
tt_fifo_check    CB balance and depth check: an overfill is a compile error
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

`tt_dataflow_select` is the **SSA → machine-IR boundary**: its input is tiled,
canonical SSA with the CB plan and the section markers; its output is typed
native ops with no SSA left. That is why the debugging gate sits at its start.

Codegen (`codegen/tenstorrent/`, with `riscv.rs` for the machine-code emitter) is
the encoder and the renderer. The backend (`backend/tenstorrent/`) is transport
plus dispatch; it decodes nothing and decides nothing.

## Role programs are RISC-V code

zyx emits the RISC-V itself. A program is RISC-V code at a fixed address, called
by its firmware and returning with `ret`; the Tensix words are inline `.ttinsn`
constants in that stream. This replaces the staged word list, `run_stream`/
`Pusher`, the loop table, the streamed-batch protocol and MOP/REPLAY loop
lowering.

### The hardware, and what tt-metal does

- Five RISC-V cores per Tensix tile: BRISC (B) and NCRISC (NC) move data,
  TRISC0/1/2 (T0/T1/T2) drive Tensix threads 0/1/2 (unpack/math/pack). All five
  run RV32IM; they differ by memory window, reset PC and what they may issue. The
  Tensix engine is a separate ISA (unpack, math, SFPU, pack) and has no branches.
  Every loop is a RISC-V branch, and every loop-varying address is computed in
  RISC-V registers.
- tt-metal compiles one kernel per core, links it against the resident firmware
  with `-Wl,--just-symbols=<firmware>.elf` (`tt_metal/jit_build/build.cpp:628`)
  so firmware symbols resolve to absolute addresses, and places it at
  `__kernel_text_start = __fw_export_text_end` from the kernel link script — a
  fixed address chosen by the firmware layout, not by the kernel.
- The firmware calls the kernel as a function:
  `reinterpret_cast<uint32_t (*)()>(kernel_lma)()` (`trisc.cc:215`). Arguments
  are an L1 array: the firmware computes the base
  (`rta_l1_base = kernel_config_base + rta_offset`, `firmware_common.h:147`) and
  the kernel reads it with `get_arg_val` → `lw` (`dataflow_api.h:108-152`).
- BRISC invalidates the I-cache for all five cores after the kernels are loaded
  and before `run_triscs` (`brisc.cc:239-241`, `:448-451`). The host cannot do
  it; the register is on the tile.
- At depth 1 the compute loop body is loop-invariant: the CB slot is fixed and
  `cb_reserve_back`/`cb_push_back`/`cb_wait_front`/`cb_pop_front` move the CB
  interface pointers and counters in L1, not the instruction words
  (`dataflow_api.h:207/259/404/474`). That is why inline `.ttinsn` constants are
  correct inside a loop.
- The four CB ops are `FORCE_INLINE` in tt-metal, so no callable symbols exist.
  The compiled reader's `wait_front` poll is five instructions —
  `lbu`/`beq`/`fence`/`lbu`/`bne` (`ncrisc.elf`, 0x609c) — and on Blackhole
  `invalidate_l1_cache()` is `fence` (`risc_common.h:153`).
- The three TRISCs run the same compute source with different
  `UCK_CHLKC_UNPACK/MATH/PACK` (`chlkc_list.h`), so T0/T1/T2 have identical trip
  counts by construction; the CB waits carry the cross-thread pacing.

### What zyx emits

- One RISC-V program per role (T0/T1/T2) and per mover (B/NC), emitted by
  `codegen/tenstorrent/riscv.rs`: encoders, labels and branch fixup, loop
  counters, and the inline Tensix/CB/(for movers) NOC sequences. The result is
  still a flat `Box<[u32]>` (`TTRoleProgram.words`); `loops` is deleted. Mixed
  Tensix and RISC-V words are both 32-bit; a disassembler is the debug tool
  (like the SPIR-V text dump), not a typed stream.
- Registers: caller-saved only, `t0–t6` and `a0–a7`; never `sp`, `gp`, `tp`,
  `ra`, `s0–s11`. A program is a leaf: no calls, no stack, ends with `ret`.
  Arguments: `a0` is the base of this launch's `u32` args array in L1; the
  program copies what it needs into `t*`. Constants are materialized with
  `lui`/`addi`.
- Tensix words are inline (the `.ttinsn` operand form), one word each. The CB ops
  and the NOC ops are inline sequences from the same emitter. No push into
  `INSTRN_BUF`, no push window, no MOP config, no per-run replay.
- Placement: allocated once at first compile in `l1::PROGRAM_CACHE`
  (`0x144000..0x180000`, 240 KiB) and never moved; monotonic allocation, no
  relocation. Addresses are final from first compile, so absolute addressing and
  a direct `jalr` need no fixup. The 16 llama kernels are 16 × 3 = 48 role
  programs and fit (today's programs are tens of words: `sfpu/mod.rs:900`,
  `loops.rs:430/517/538`; the staging slots are 4 × 32 KiB).
- Entry: the firmware reads the program address, length and args base from
  `tt_isa::mailbox::exec`, validates the address, invalidates the I-cache, and
  calls the program with `a0` = args. The executor image is `exec_b`
  (`firmware/tt-firmware/src/bin/exec_b.rs`); `ttinsn::call_with` is the call
  and `ret` the return. The host writes `PROGRAM`/`LEN`/`ARGS`, then `SEQ`; the
  executor publishes `ERROR`, `RET` and `DONE`. The first test of the whole
  path is the gather writeback (`ttsim::exec`).
- `dm` is gone: the reader and writer are programs on B and NC. `KERNEL`,
  `LAUNCH`, `KERNEL_WAIT`, `BARRIER`, `CALL`, `POKE`, `WAIT` are dispatch
  constructs that belong to a command queue (tt-metal keeps them on dispatch
  cores and host traces), not to the worker movers; nothing emits them now.
- The CB ops and sync passes are restored from `dc9a1e3d^`: `TTOp::{ReserveBack,
  PushBack, WaitFront, PopFront}` (each `{ cb: OpId, n: u8 }`), placed by
  `tt_sync_cbs`, deduplicated by `tt_dedup_pushes`, closed by `tt_place_pops`,
  and proven by `tt_fifo_check`/`fifo_check_copy`. `tt_lock_dst` does not come
  back: DST is `tt_dst_reuse` by liveness plus `tt_reg_sync` at role seams.

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

One zyx `Kernel` is one program: one reader program, one compute role set, one
writer program, launched once.

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
- Reduces currently lower as rank 2: `Loop(k_tiles) + Unroll(32)`, one tile per
  iteration. That is the tiled form the search chose, not a precondition for the
  traffic unit.
- **Grid cap.** A group range above `max_global_work_dims[axis]` (TT `[10,14]`)
  is split by `split(g, SplitMode::Loop, factor)` with `factor` the smallest
  divisor of the length whose outer part fits. The loop wraps the whole body, so
  each iteration is one of the group's points. Enabled in `tt_tile`. If no
  divisor leaves an outer part within the max, the split is a loud rejection;
  the later form is `pad_to` the group to a multiple of the grid max, then
  split.
- **`tt_tile` completes the tiling; it assumes nothing about the page.** The
  movement unit is the traffic the index names: a scalar access moves its whole
  row (64-byte congruence is the hardware limit), and each scalar access gets
  its own row load. Row- or tile-granular movement (one iteration = one tile, a
  congruent `TILE_SLOT` image) is an optimization the search may choose, never a
  precondition: `tt_tile`, `tt_add_movers`, `tt_sync_cbs`, the CB page and the
  emitter follow whatever the access actually moves.

## Boundary indexing: `tt_canonical_tile`

The tiled form of boundary addressing: used when the search has tiled the
kernel, never assumed. The default is the traffic the access names (a whole
64-byte-congruent row per scalar gather access). The transform is the only place
that knows the tile layout. Shape from `self.shape(base)` (kept on the param for
TT since `linearize`):

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

## CB ring and the four syncs

tt-metal is the reference (`~/Dev/cpp/tt-metal/tt_metal/hw/inc`).

- **A CB is a fixed-depth ring, independent of the loop.** Depth 1 now, slot 0.
  The page is the amount of traffic the access moves: by default a whole
  congruent row per scalar access, and a tile only when the search has tiled the
  kernel. `n` in the four syncs counts those pages, exactly as tt-metal's
  configured `page_size` does (`num_pages * fifo_page_size`,
  `dataflow_api.h:209`).
- **Two counters per CB**, `received` (pushes) and `acked` (pops).
  `reserve_back` waits `depth - (received - acked) >= n`; `wait_front` waits
  `received - acked >= n`; both poll and do not advance pointers. `push_back`
  advances the write pointer and `received`; `pop_front` advances the read
  pointer and `acked`. Both wrap only at the fifo limit. Total pushed/popped per
  cycle must equal the depth.
- **One producer and one consumer per CB.** Only one thread pushes a CB, only
  one pops it.
- **Four endpoints**: reader (B) `Reserve`/`Push` on Input; unpack (T0)
  `Wait`/`Pop` on Input; pack (T2) `Reserve`/`Push` on Output; writer (NC)
  `Wait`/`Pop` on Output. Capacity < 0x8000.
- **The loop lives in the kernel**, one page per iteration, in every role:
  `for i { reserve; read; push }` (reader), `for i { wait; compute; pop;
  reserve; pack; push }` (compute), `for i { wait; write; pop }` (writer).
- **Where each half runs**: in the RISC-V program of its own core, as inline
  sequences emitted by codegen from `TTOp::{ReserveBack, PushBack, WaitFront,
  PopFront}`. They are not data, not list entries and not pushes.
  `tt_sync_cbs` places them, `tt_dedup_pushes` merges adjacent pushes,
  `tt_place_pops` places the pops, and `tt_fifo_check` proves the balance before
  codegen. A pop follows the **whole cone** the loaded value feeds (transitive
  users within the section), not its first use: a pop between two SFPU row
  programs blocks `tt_sfpu_fusion` and splits one row body into two. Freeing the
  page later costs producer overlap, never correctness.
- **No credits, no batches, no `retire_batch`.** With a real loop the four ops
  are the synchronization. The credit machinery in earlier notes existed only
  because a mover list could not block inside an entry.
- **Launch order**: the reader and writer must run concurrently with compute, or
  a depth-1 `Reserve` blocks forever. The host starts all five cores (a
  generation per mailbox) and joins them; no list is submitted.
- **Caps**: `tt_fifo_check` checks depth at compile time; a producer past depth
  is a compile error.
- A slot index in a transaction must be the loop counter or 0; anything else is
  a compile error.

## 64-byte congruence

The hardware limit: a read needs the DRAM and L1 addresses congruent mod 64, a
write mod 16. `dm` used to enforce it (`dram_offset % 64 == l1 % 64`, else error
4); with `dm` dropped, the mover programs emit the transactions, and
`fifo_check_copy` proves congruence at compile time before codegen.

- **Whole-row reads follow from this limit**: a row that is a multiple of 64
  bytes is congruent, so the default scalar gather moves its whole row.
- The canonical tiled transform adds the same datum offset on both sides, so
  whole tiles and rows stay congruent.
- CB bases are placed at `≡16 mod 64` (`tt_place_cbs`), where a tile image's
  datum block lives; a tile slot is `TILE_SLOT = 4160` with a 16-byte header.
  A **row page** CB (the gather) is placed at `≡0 mod 64`, congruent with the
  raw 64-byte DRAM span it holds; the two kinds are distinguished by the copy's
  address (a rounded span base, i.e. a multiple of 64 or 16, versus the tiled
  `TILE_SLOT + TILE_DATA` address).
- `fifo_check_copy` proves per-variable stride congruence at compile time; an
  unprovable index is a loud rejection.

## Compilation, images, dispatch

Compilation ends at codegen. What runs is the images, not the `Kernel`.

- `Image { core, address, len }` per core's bytes in device DRAM; `TTProgram`
  holds a `sequence` of `ImageId`s plus `lens`, `gws`, param counts.
- Images are uploaded to DRAM once. Each program is written once into its
  permanent `l1::PROGRAM_CACHE` address: monotonic allocation, never moved, no
  relocation. The 16 llama kernels (16 × 3 role programs) fit in the 240 KiB
  region; eviction has no case at this workload.
- Residency is per `(tile, core)` over five slots (B, NC, T0, T1, T2), keyed by
  `(ImageId, generation)`; a miss is a NoC load from the image's DRAM copy.
- A launch binds arguments and issues: program address, args base (`a0`), grid
  size, generation. It adds, removes, reorders or encodes nothing, and computes
  no strides.
- Grid dispatch is a launch loop over the grid points (dense order); the
  program is tile-uniform and only the DRAM slice differs.
- `Pool::TT` is device DRAM; host staging is upload/readback only. The DRAM↔L1
  boundary belongs to the mover programs, never the host. A 16-bit output's
  narrowing is the host's only because the pinned ttsim refuses the packer's
  format conversion (`pack_fmt_conv_mode=0x105`); silicon does it on the card.
- Rendered IR: `TTOp::Unpack`/`Math`/`Pack` with `TTRoleProgram { words }` (a
  flat mixed Tensix/RISC-V word stream), the reader/writer programs, and the
  `EndReader`/`EndCompute` section markers. `compile` collects them; it does not
  decide.

## Loops

- A loop is RISC-V control flow: a counter in a caller-saved register, the body,
  a decrement and a backward branch. There is no loop table, no REPLAY/MOP loop
  lowering and no unrolled loop body.
- `Item::Repeat` becomes that loop. Loop-level replay and `Item::Shared` (the
  matmul face-block replay) are deleted with it. The SFPU builder's own `REPLAY`
  stays: it repeats recorded instructions inside a math program, which is an
  op-level hardware feature, not loop control flow.
- One iteration = one page (`tt_tile`), and a page is whatever the access moves:
  a whole row per scalar gather access by default, a tile when the search tiled
  the kernel. At depth 1 the body's Tensix words are loop-invariant, so they are
  inline constants.
- A trip count may be a launch argument: it is loaded once from the args array
  (`a0` base) before the loop. It does not have to fold to a constant.
- Nesting: the grid split wraps the page loop, two levels now. The emitter is
  built so deeper nesting is a counter stack rather than a rewrite.
- Unrolling is a search decision before the device epilogue, never a backend
  choice.

## Restore sources (do not reinvent)

- `dc9a1e3d^` (pre-native emitter) is the reference implementation of the CB
  mechanics, and it drove the TT path including tt_gemm:
  - `TTOp::{ReserveBack, PushBack, WaitFront, PopFront}`, each
    `{ cb: OpId, n: u8 }`, with the `tt_reserve_back`/`tt_push_back`/
    `tt_wait_front`/`tt_pop_front` constructors. They are emitted by
    `tt_sync_cbs`, not by `tt_add_movers`.
  - `tt_sync_cbs` (577 lines): places the four syncs. Port **only the relevant
    sections**: its window batching applies to an indexed read at a slot other
    than 0, which the new tiling never produces (slot 0), and its arms name
    `TTOp::LLK`/`LLKBcast`/`ReduceTile`/`MatmulTiles`/`SigmoidTile`/`SiluTile`,
    none of which exist today; re-express those against `Copy`/`Unpack`/`Math`/
    `Pack`/`Unpacr`. Section insertion and the publish tally stay.
  - `tt_place_pops` and `tt_dedup_pushes`: shrink to what the per-iteration form
    still needs. `tt_place_pops` places `PopFront` after the last read;
    `tt_dedup_pushes` merges adjacent `PushBack`.
  - `tt_fifo_check`: reserve/wait balance at each section end, `pushed ==
    popped`, producer-past-depth (deadlock guard), symbolic-loop traffic
    rejected. This is the gate that prevents board hangs and must pass.
  - `fifo_check_copy`: 64-byte congruence per variable stride.
  - `tt_verify`: the post-allocation firewall, back in the pipeline.
  - `tt_lock_dst` does **not** come back: DST is `tt_dst_reuse` (liveness) plus
    `tt_reg_sync` (role seams), as today.
  - Pipeline there: `tt_tile -> tt_lower_llks -> tt_lock_dst ->
    tt_dedup_pushes -> tt_init_math -> tt_sync_cbs -> tt_place_pops -> verify
    -> CSE -> DCE -> tt_verify`. The current order is in "Pipeline" above;
    `tt_dedup_pushes` now runs after `tt_sync_cbs`.
- `588110a9` is the same mechanism one generation earlier
  (`CBState`/`CBBatch`/`CBEvent`/`CBEmitter`, batching pre-pass), where
  `t > config[cb]` stayed streaming and the verifier made an overfill a compile
  error.
- `a8fa78b9`: 14.444 TFLOPS, bad 0, M5120 N11264 K1024 with depth 2.
- `~/Dev/cpp/tt-metal` is the hardware reference for CB semantics and for the
  compiled-kernel shape: `dataflow_api.h` (`cb_reserve_back`, `cb_push_back`,
  `cb_wait_front`, `cb_pop_front`, `get_arg_val`), `brisc.cc`/`trisc.cc`/
  `ncrisc.cc` (call, args, I-cache invalidation), `ckernel_ops.h` (`.ttinsn`),
  `matmul_multi_core/kernels/` (loops). `examples/qwen3.8-27b/
  tenstorrent_debugging.md` (git history) is the five-core /
  reserve-push-wait-pop reference.
- **The per-role loop shape is tt-metal's, read from the kernels:**
  `tt_metal/kernels/dataflow/reader_unary.cpp` is `for i { reserve_back(1);
  noc_async_read; barrier; push_back(1); }`;
  `programming_examples/sfpu_eltwise_chain/kernels/dataflow/writer.cpp` is
  `wait_front(1); get_read_ptr; noc_async_write; barrier; pop_front(1)`; its
  `compute/compute.cpp` is `wait_front; copy_tile…; reserve_back(out);
  pack_tile; pop_front(in); push_back(out)`. So the four syncs sit **inside the
  loop body, once per iteration**, `n` is in pages (`cb_push_back` multiplies it
  by `fifo_page_size`), and the consumer addresses the CB through
  `get_read_ptr`/`get_write_ptr` (the current pointer), so no index is rebased
  when a pop advances it.
- **`programming_examples/loopback/kernels/loopback_dram_copy.cpp`** is the
  writeback proof: `for i { noc_async_read_page(i, in0, l1); barrier;
  noc_async_write_page(i, out0, l1); barrier; }` over a plain one-page L1 buffer
  — no CB at all, one page per iteration. That is the shape of the green
  gather-writeback test, and it shows a page is whatever the host configured
  (`fifo_page_size`/L1 page), not necessarily a 32×32 tile.

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
| `llama_rope` | `B mover refused list 2: error 4` (a `dm` mover error; re-check once the movers are programs) |
| `23_tt_lowering::sum_drop_col` | scalar-per-column store lands at `row*32`, not the face slot |

`24_tt` must be replaced by the model's real 16 kernels, one test each, in the
order the model compiles them (the order comes from `ZYX_DEBUG=4` on the model,
not from the test file). Each real kernel replaces one approximation as its IR
comes right.

## Failure modes of this agent (do not repeat)

Every one of these happened in the 2026-11-10 session, with the user saying the
same thing repeatedly. They are process failures, not knowledge gaps.

1. **Invented design.** Wrote "the traffic is one whole tile per iteration" and
   the 1024-datum CB page into NOTES and the passes. The user never said it; the
   rule is arbitrary indexing with the 64-byte-rounded span read and the row as
   the write unit. Inventing is worse than asking.
2. **A frozen snapshot called a contract.** The IR test pasted a run's output as
   the expected string, on a fixture that never reached `tt_add_movers`. It was
   green and tested nothing. Expected IR is written from the design first, red
   until the passes match.
3. **Stand-ins for the real kernel.** The custom gather fixture and a tile copy
   were built and called "the gather", then the user's actual target — the real
   tensor op in `23_tt_lowering::gather` — was left untouched. A hand-built
   kernel is a unit counterpart at most; it is never acceptance.
4. **Progress reported as success.** Test counts ("137 pass") and a transfer
   program proven against staged raw DRAM were reported as progress. NOTES is
   explicit: the acceptance is the real tensor gather against the host.
5. **Ignored the gather being a loopback.** The user said five times that the
   gather kernel has no compute, like tt-metal's DRAM loopback. The time went
   into the index-expand kernel's compute instead.
6. **Assumed affine/contiguous indexing.** Proposed making the store a
   contiguous row and then a tiling rewrite, instead of the recorded rule:
   arbitrary indices, span reads, whole-row writes.
7. **Used `python3`/`sed` for file edits**, which is forbidden; the edit tool
   only.
8. **Asked questions NOTES already answers** (page size, write unit, what the
   acceptance is), instead of reading the file end to end first.
9. **Called a backend launch "working"** when the addresses were raw row-major
   DRAM and the pool stores tiled images; the output was garbage. The test is
   `#[ignore]`d, not green.
10. **Claimed one-hot exists** in the gather path, then defended it after the
    user said the one-hot is optimized away. `fold_loops`' `replace_gather_loop`
    is the intended lowering (`acc = source[index]`), producing the compute-free
    loopback; the index expansion is a separate, earlier kernel.

What the next session must know, concretely:

- `23_tt_lowering::gather` is the acceptance. It compiles several kernels; the
  first is `index_select`'s index expand (`out[i][j] = idx[i] + (idx[i] < 0)*V`),
  which is a compute kernel. The gather kernel after `fold_loops` is
  `acc = source[index]` — a pure transfer/loopback, which the transfer path in
  this tree already lowers and runs on ttsim.
- The index expand being a separate kernel is why the acceptance still fails
  before the gather. The gather must read the compact index directly (fold the
  expand into the gather, or run the fold before kernelization), so that no
  compute kernel exists on the path. That is the next real change, and it is in
  the graph/kernelizer lowering, not in `tt_add_movers`.
- Fixed this session and kept: `merge_ranges` now decomposes the flat index
  **row-major** (last axis low), so vector lanes can be contiguous; the row
  body pages a contiguous vector store as `lanes` datums with the lane offset as
  its page index; `Bool -> I32` cast is an identity (raw 1/0 word).

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
- `ZYX_TT_DUMP=1` prints the rendered reader/writer traffic and role byte
  counts (the program dumps replace this once `dm` is dropped); `ZYX_DST_DUMP=1`
  sets `dump_rows = 16` and prints `Outcome.dst`; `ZYX_MOVE_DUMP=1` prints the
  moves.
- The DST dump covers rows 0..15 only (`mailbox::DUMP_MAX_ROWS = 16`).
- A test that vanishes with no failure line is a `libttsim` BAR violation: the
  library calls `_Exit`.
- **A bug first gets the unit test that isolates its half of the chain**, then
  debugging continues.
- **An IR test asserts the IDEAL IR, never a snapshot of current output.** The
  expected text and hash are written from the design (the recorded rules:
  page = the range the access moves, a whole row per scalar gather access,
  reserve/push around each publisher, wait/pop around each consumer, no SSA
  left, no tile addressing unless the search tiled the kernel). The test may be
  red until the code produces that IR, and it stays red; it is never made green
  by pasting `--nocapture` output back in.
- **Modus operandi for every kernel, from now on (user, 2026-11-09): write
  down the ideal IR first, then fix the passes until they produce it.** The unit
  test asserts the ideal IR written from this design — never a snapshot of
  current output — and stays red until the passes generate exactly that IR. A
  kernel is brought up by writing its ideal IR at each boundary, then making the
  passes match, not by reading what they happen to emit.
- **A test that cannot fail for the bug it guards is worthless, and worse than
  none when it is green.** The `ir_contract` test as first written
  (`gather_pipeline_ir_is_frozen`) is that failure: the fixture's index cone
  contains an `Op::Load`, so `tt_add_movers` returned early and created no CB,
  no reader copy and no `EndReader`; `tt_sync_cbs`/`tt_dedup_pushes`/
  `tt_place_pops`/`tt_fifo_check` were then no-ops, and the "frozen" strings
  were copied from a run of the same test. It asserted that the printer still
  prints the same no-CB IR, so a wrong page, a missing reserve and a whole-tile
  reader all passed. **Fix:** the fixture must reach the pass it claims to
  test (no early return on the path under test), the expected IR must be
  written from the rules above, and the assertions must be able to fail for a
  wrong page, a missing sync, or an SSA op at the boundary. The current test
  does not guard the `tt_dataflow_select` boundary and its frozen strings are
  not the ideal IR. `tt_add_movers`'s silent `return` on a non-affine index cone
  is itself a fail-loud violation.
- Run `cargo fmt`; format a single file with
  `rustfmt --config skip_children=true <file>`.
- `TT_DRAM_PAGE_BYTES = 4096`; `TT_TTSIM_LIB` must point at the vendor pin
  (`/home/x/Dev/rust/metalium-rs/vendor/libttsim_bh.so`).

## ISA and ttsim facts

- A tile is 32x32 datums; a DST row is 16 datums, so a tile is 64 DST rows
  (32 rows for 16-bit tiles).
- `dm` (the mover list, `dm::op::*`, `dm::LIST`, `LIST_MAX`, `ENTRY_BYTES`,
  `KERNEL`/`BARRIER`/`CALL`/`POKE`/`WAIT`) is dropped with the list; the reader
  and writer are RISC-V programs. `TILE_SLOT = 4160` and `TILE_DATA = 16` stay
  (the tile image layout).
- `tt_isa::dataflow`'s streamed program (`VERSION = 1`, `STEP_WORDS = 4`,
  `STREAMED`, `LOOPED`) is dropped with the staged-word runner. The CB counters
  and `Action`/`Stream`/`Endpoint` names stay as the description of the ring.
- `mailbox::PROGRAM_MAX = 8192` words, four program slots (4 × 32 KiB),
  `l1::PROGRAM_CACHE = 0x144000..0x180000` (240 KiB).
- The I-cache invalidate primitive is `ttinsn::invalidate_icache(last_written,
  mask)` (`firmware/tt-firmware/src/ttinsn.rs:298`), mask = one bit per core.
  The corpus `translate` path already uses it after writing code.
- RISC-V words and Tensix words are both 32-bit and share the stream; the
  Tensix word is the `.ttinsn` operand form (`tt_isa::isa::Instruction`).
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

1. ~~**Single mover writeback**~~ — **green on ttsim**
   (`backend::tenstorrent::ttsim::tests::gather_writeback_runs_on_riscv_b`): one
   program on RISCV B, inline NOC read/write, one gather loop, four 64-byte rows
   in the order 2, 0, 3, 1, checked against the host. The emitter, the executor
   image and `ttsim::exec` are what it proves.
2. **Reader program** — DRAM → CB with the inline CB ops; the host reads the CB
   back. The second emitter proof.
3. **`large_elementwise`** — reader, the T0/T1/T2 roles, writer, checked against
   the host.
4. `transfer_roundtrip` — host <-> DRAM, no compute.
5. `copy` kernel — reader and writer programs together.
6. `sigmoid` — SFPU path, CB plan, pack.
7. `matmul` — Matrix Unit path.
8. `sum` — reduce path.
9. Loop/grid — nested loops (the grid split wrapping the tile loop).
10. Multi-op DST chain — two compute ops with a DST intermediate, no DRAM and
    no CB round-trip.
11. One llama forward step, compared against CPU.

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
- Unrolling IR loops inside `render_section`: hangs; use a RISC-V backward
  branch.
- Pinning a root `rust-toolchain.toml` at 1.88.0: the firmware role images
  overflow their 10 KiB L1 slots.
- A shared global `target-dir`: a second checkout can shadow the test binaries.

## Directives (2026-11-08, user)

- **Loops lower to ordinary control flow, as in tt-metal.** tt-metal's kernels
  are compiled code: nested C++ loops become branches, and every address is
  computed from the loop variables at runtime
  (`matmul_multi_core/kernels/compute/mm.cpp`, `reader_mm_output_tiles_partitioned.cpp`).
  Our flat body lists, loop tables and compile-time entry addresses are not
  that. Loops must not be flattened into repeated bodies/entries.
- **No dependency on metalium-rs.** It is not on crates.io and blocked
  publication. `tt-isa` is now published as `tt-isa 0.1.0` (Apache-2.0,
  https://github.com/zk4x/tt-isa, a fork credited to the metalium-rs authors,
  transfer on request) and the firmware sources live in
  `zyx/firmware/tt-firmware`.
- metalium-rs was only ever an assembly reference. Do not preserve its
  structure, its abstractions or its dead code.
- **Follow tt-metal unless there is a good reason not to.** Hardware behaviour,
  kernel shape, launch shape, CB setup and the order work is proven all follow
  tt-metal. A deviation is stated with its reason in the same place the
  deviation is described. When in doubt, follow what tt-metal does.
- **The hardware limit is 64-byte congruence, hence whole-row reads.** A gather
  works at scalar granularity: by default the kernel loads a whole row for each
  scalar it names, and each scalar access gets its own row load. Row or tile
  granularity, and any one transfer covering several scalars, is an
  optimization, never a precondition; the CB page, the four syncs and the
  emitter follow what the access actually moves.
- **The baseline that must work for PyTorch parity: round the address down to
  64-byte congruence, move the aligned span, pick the element in L1.** It is
  ugly and slow, but it is the general fallback. `ttnn.embedding` does exactly
  this: `noc_async_read` the aligned index block, then `input_l1_ptr[index]` to
  pick; the scalar pick is never part of the NOC address.
- **Where the address is resolved:** `tt_tile` handles the tiling, `tt_reindex`
  handles the face orientation, and `tt_add_movers` only splits one kernel into
  the three segments. `tt_add_movers` does not compute tile addresses or page
  sizes itself.
- **Baseline traffic unit: one rounded span per lane (user, 2026-11-09).** A
  vector access whose lanes name N places is N separately rounded spans, one per
  lane: each lane's address rounds down to the congruence boundary, the aligned
  span is moved, and the element is picked in L1 at its offset inside the span.
  One aligned span covering all N lanes is the optimization, only correct when
  the lanes are contiguous; it is never the fallback. The N spans of one traffic
  op are concatenated into that op's CB page (`N * 64 B` for reads), so the
  compute picks lane `i`'s datum at `i * 64 + (address - span_base)`; one
  `reserve`/`push` per page. Writes round to 16 bytes.
- **The write unit is the whole row** (user, 2026-11-09): reads have the
  "round down and pick in L1" fallback, writes do not — a sub-16-byte write would
  clobber its neighbours. Follow tt-metal, whose writer moves
  `output_page_size` per `cb_pop_front` (`embeddings_rm_writer_chunked.cpp`).
  Store indices are affine and row-major by construction: `linearize`'s
  `add_indexing` builds every `Store`'s `dst_idx` from the output view's dims
  with contiguous row-major strides (`write_index = mad(d.idx, s, write_index)`,
  `src/kernel/linearize.rs:546-568`), the worst case being padded; so the whole
  row is addressable without a read-modify-write.
- **Generality limit: llama inference is the demo** (user, 2026-11-09). Nothing
  more general than the kernels llama inference needs is required. That limit
  does **not** exclude the gather: the embedding gather (loads that are not whole
  tiles, an address computed from loaded data) is required and is the current
  task. Express it as `weight.gather(0, indices)` with the index tensor already
  in the output shape (uploaded), so the kernel is the scalar row gather and not
  `index_select`, whose first step materialises an index broadcast with a
  scattered store. Rules:
  - no read-modify-write writer; a store is a row/contiguous run (or a tiled
    output view), never a scattered element set;
  - llama's one-hot embedding (`arange.equal(idx) * weight`, then `sum(vocab)`,
    `zyx-nn/src/embedding.rs:63-110`) means every llama *store* is elementwise,
    matmul, reduce or narrow-assign with an affine row-major output view — the
    store side is easy; the *load* side is where the gather lives.
- **Indexing is arbitrary; affine and tiled addressing are optimizations.** A
  kernel accesses rows and columns by indices computed from data, in any order:
  a gather is the case, not an exception. The reader and writer lower the index
  expression as written (loads, arithmetic, loops, branches); they never assume
  a constant step, never reject a non-affine step, and never recover a loop by
  unrolling. Tile indexing and affine steps are forms the compiler may exploit
  when they happen to hold, never a precondition for lowering. A test that only
  exercises an affine step proves nothing about the gather path: the gather
  test uses arbitrary indices and a non-affine read pattern.
- **No scoping down.** The target is the full Tenstorrent backend, and each step
  is completed in full. No reduced versions, no "minimal" substitutes, no
  deferring part of a step because it is large.
- **No dodging (user, 2026-11-09).** Do the thing that was asked, not an easier
  thing next to it. Concretely: when the gather kernel ("loads that are not whole
  tiles") was asked for, the contract test was rewritten around a **tile copy**
  instead — that is dodging, and the fixture proved nothing about the case that
  matters. Do not:
  - substitute a simpler kernel (tile load, affine load) for the asked one;
  - reclassify an asked kernel as "out of scope" to avoid lowering it;
  - replace a failing case with a passing neighbouring case;
  - report a plan or a test scaffold as progress.
  If the asked kernel is hard, it stays the task and the work continues on it.
- **Flattening (general direction, applied when it bites).** One layer between
  the kernel and the hardware: every type and name corresponds to something the
  spec or ISA has. When an issue lands in an abstraction that only renames,
  groups, schedules or guards what the hardware already says, it is flattened
  then: the abstraction is removed and the two real users (host, firmware) say
  what they need directly. Never design around a bad abstraction. Shared facts
  (addresses, sizes, register values) may stay where both sides need them; what
  must not stay is a contract module that turns them into a protocol. This is a
  direction, not a work item: it is applied only where a concrete problem
  exposes the layer.

## Dependency and firmware layout (2026-11-08)

- `zyx` has no local path dependencies. `tt-isa = "0.1.0"` comes from crates.io
  (also a `[build-dependencies]`, because `build.rs` reads `tt_isa::tensix`).
- The RISC-V firmware sources are inside the crate at
  `zyx/firmware/tt-firmware`; the manifest is `Cargo.toml.in` because cargo does
  not package a nested package. `zyx/build.rs` copies the tree into `OUT_DIR`,
  writes `Cargo.toml`, builds `riscv32im-unknown-none-elf`, gates every image's
  instruction set, and exports `FIRMWARE_*` paths.
  `src/backend/tenstorrent/firmware_images.rs` embeds them.
- `zyx` features forwarding to the firmware build: `fast-reads`,
  `encoded-guard` (implies `fast-reads`), `ttinsn-experiment`.
- `cargo package` verifies: 172 files, 5.7 MiB (1.3 MiB compressed), and the
  packaged crate's `build.rs` rebuilds the firmware from the packaged sources.
  `zyx` itself is not published.

## Decisions (2026-11-08, user)

- Loops are RISC-V branches in the compiled program, as in tt-metal. No flat
  body lists, no loop table, no compile-time entry addresses.
- A program is called by its firmware and returns with `ret`; its address is
  fixed at first compile. Arguments arrive as an L1 `u32` array whose base is in
  `a0`. The program uses caller-saved registers only and is a leaf.
- Tensix words are inline `.ttinsn` constants; CB ops and NOC ops are emitted
  inline by codegen (5–15 words each). No `INSTRN_BUF` push, no push window, no
  MOP config.
- Programs live in `PROGRAM_CACHE` at a fixed address: allocated once, never
  moved, no relocation and no eviction. The 16 llama kernels (48 role programs)
  fit in 240 KiB.
- `dm` is dropped. Reader and writer are programs; `KERNEL`/`LAUNCH`/`BARRIER`/
  `CALL`/`POKE`/`WAIT` are dispatch concepts, not mover work.
- The debugging gate is a `todo!()` at the start of `tt_dataflow_select` (moved
  from the end of `tt_add_movers`). It is added back whenever a kernel does not
  pass and removed only once that kernel's IR at the boundary is good.
- `TTOp::{ReserveBack, PushBack, WaitFront, PopFront}` (`{ cb, n }`) are
  restored and emitted by `tt_sync_cbs`; `tt_add_movers` only builds the
  reader/writer blocks and canonicalizes the compute segment.
- Pass order: `tt_tile, tt_add_movers, tt_sync_cbs, tt_dedup_pushes,
  tt_place_pops, tt_fifo_check, tt_dataflow_select, ...`. `tt_fifo_check` is the
  gate against board deadlocks and is crucial.
- `tt_sync_cbs` is ported in its relevant sections only; `tt_place_pops` and
  `tt_dedup_pushes` are shrunk to what is needed; `tt_lock_dst` stays out (DST is
  `tt_dst_reuse` + `tt_reg_sync`).
- Loop-level replay and `Item::Shared` are deleted; the SFPU builder's op-level
  `REPLAY` stays.
- The word stream stays flat (`Box<[u32]>`). A disassembler is the debug tool,
  like the SPIR-V text dump; no check on the emitted stream for now.
- `EndReader`/`EndCompute` stay and are emitted by `tt_add_movers`; no reason to
  delete them.
- Emitter loop nesting: **two levels now** (grid split wrapping the page loop),
  with the emitter structured so arbitrary nesting is a counter stack later, not
  a rewrite.
- Start/join follows tt-metal: the firmware starts and joins the cores from the
  launch descriptor; the host posts the launch and does not drive generations
  itself.
- CB interface setup follows tt-metal: firmware writes `fifo_addr`,
  `fifo_size`, `fifo_num_pages`, the pointers and the counters before calling
  the program.
- First proof order follows tt-metal's examples: **a single mover kernel that
  writes back first**, then the reader and writer programs, then compute. The
  role-only `large_elementwise` proof is superseded.
- The emitter lives in `codegen/tenstorrent/riscv.rs`.
- The 16-kernel list is not needed for the emitter.
- **The writeback reference is tt-metal's DRAM loopback example**
  (`tt_metal/programming_examples/loopback`, `docs/.../dram_loopback.rst`): one
  data-movement kernel moves a DRAM page into L1 and back out, looped over
  pages. The gather variant reads each page at an arbitrary index; it is not a
  tiled matmul dressed up as a mover.
- **The gather test is a real tensor op** on `Dev::TT(0)`: arbitrary
  out-of-order indexing, scalar loads, no 32×32 tiling. Its unit counterpart is
  a hand-built `Kernel` through the custom kernel API (`fold_loops.rs`'s
  style), not a tensor op.
- **Test shapes and index values are the agent's choice.** Choosing a tensor
  shape in a test is not a design decision and is not to be asked about.
- **IR tests are textual and exact.** `format!("{k}")` must equal the expected
  string and `Kernel::get_hash()` the expected `u64`; both are literals in the
  test. They live under `#[cfg(test)]` in `kernel/tenstorrent.rs`, call the
  passes directly so the debugging gate never fires, and are frozen now and
  updated when `tt_sync_cbs` lands.
- **`tt-isa` is our crate to edit at will.** It is a path dependency while it is
  edited and the published version later. Modules that we would have to change
  anyway move into zyx instead of being designed around.
- **Mailboxes follow tt-metal**: one mailbox block per tile with per-core sync
  words and the launch descriptor, not a per-core trio.
- **`tt_tile` never folds a lane split back into a loop.** Unwanted splits are
  prevented at the splitting passes (`opt_split`, `opt_register_tiling`) by the
  Tenstorrent gate, before `tt_tile` runs.
- **The two group axes are split independently**, and their trip counts may
  differ.
- **DRAM alignment**: a read needs the DRAM and L1 addresses congruent mod 64,
  a write mod 16; a gather is per-row at computed addresses (never a chunked
  whole-tile move), and a row stride that is a multiple of 64 makes every access
  congruent by construction.
- **NoC initialisation is firmware's**, following tt-metal (`noc_init` and
  `noc_local_state_init` in `brisc.cc`); the emitted program only writes the
  initiator registers and triggers.
- **The four CB syncs sit inside the loop body, once per iteration**, as in
  tt-metal's `reader_unary.cpp`/`sfpu_eltwise_chain`: `for i { reserve; read;
  push }` and `for i { wait; compute; pop; reserve; pack; push }`. `n` is in
  pages. A publish run never crosses a `Loop`/`EndLoop` boundary, so nothing is
  hoisted across a depth-1 ring.
- **CB addressing is pointer-relative.** Readers/writers use the CB's current
  read/write pointer (`get_read_ptr`/`get_write_ptr` in
  `dataflow_api.h`/`writer.cpp`), so a `PopFront` advancing the pointer never
  rebases a kernel index; `tt_place_pops` places pops only.

## Open questions

### The model and the 16 kernels

1. What is the exact list of the model's 16 kernels (ops, shapes, order)?
2. Are all 16 reached at `seq_len = 1` decode, given that CPU prefill crashes?
3. What counts as one kernel: one compiled `Kernel` per fused op group, with
   repeated invocations sharing it?
4. Does `24_tt` get replaced one-for-one by the 16, or extended?

### Correctness

5. Is `sum_drop_col`'s orientation needed for llama at `seq_len = 1` (attention
   over one key is `v`), and is its shape the one the model actually uses?
   **Re-ask now that the gather writeback is green.**

Nothing about launch, firmware, mailboxes, NoC init or the `tt-isa` surface is
open: those follow tt-metal and the decisions above. `exec_nc` for the writer is
work, not a question. CB page size is not a question either: it is the traffic
the access moves (a whole row per scalar gather access by default, a tile only
when the search tiled the kernel), and `tt_add_movers`'s hardcoded 1024-datum
page is a bug against that rule.

## Next

The full backend, one complete step at a time. Nothing here is optional and
nothing is to be done in a reduced form.

0. Settle the open questions as each step reaches them.
1. **CB ops: done in the tree.** The four constructors and
   `tt_sync_cbs`/`tt_dedup_pushes`/`tt_place_pops`/`tt_fifo_check`/
   `fifo_check_copy` are wired after `tt_add_movers`. The publish tally is kept
   but scoped to one loop body (a run never crosses `Loop`/`EndLoop`), waits
   precede every CB `Load`/`Copy` source, and pops follow the last same-section
   read. A CB holds exactly one page (depth 1), so a publish run must fill
   exactly `Storage.len` datums and the tag is `n = 1`; `tt_fifo_check` counts
   depth in those pages and a push past one page is the deadlock/overfill
   error; `tt_dataflow_select` takes the packet size from the CB's `len`
   (broadcast keeps `TILE_SLOT`), so no pass hardcodes 1024. The `ir_contract`
   test is mid-rewrite: its tile-copy fixture is dodging and is replaced by the
   scalar row gather's ideal IR (see "No dodging").
1a. **The scalar row gather (current task).** `weight.gather(0, indices)` with
   the index in the output shape. `tt_add_movers` must stop returning on an
   `Op::Load` in the index cone: materialise the index read as its own reader
   copy, rewrite the index load to read from its CB, keep the data copy's
   address as arithmetic over the picked token, and give both the page the
   traffic moves. `ir_contract` asserts the entry IR and the boundary IR (two
   reader copies: index span, then the row the token names) written from this
   design; it is green, with the row gather's hashes frozen.
   **Done:** `tt_tile` keeps the group as the grid when a lane axis exists;
   `tt_add_movers` has the row body (per-lane spans, index CB, row writes);
   `fifo_check_copy` accepts a `multiple-of-64/16` address as congruence by
   construction, and a row CB's base is `≡0 mod 64` (a tile CB's is `≡16`).
   **Next:** wire the reader and writer programs into `render_tt` and the launch
   (steps 3-5): the CB interface table in L1, the per-parameter DRAM addresses
   and the grid point as launch arguments, then run the whole gather kernel
   (reader + writer on their cores, compute still pending).
2. **Reader/writer programs from codegen.** Emit `TTOp::Reader`/`Writer` from
   the kernel IR instead of `DmRead`/`DmWrite` lists: one loop per boundary
   loop, no unrolling; the index expression lowered as written (loads,
   arithmetic, branches) — arbitrary, non-affine indexing is the case, affine
   and tiled are optimizations; inline NOC issue/wait/drain; inline
   `ReserveBack`/`PushBack`/`WaitFront`/`PopFront`; parameters read from the
   `a0` args array; end with `ret`. Prove the reader program next, then the
   tensor `gather` test through the pipeline, then `large_elementwise`.
3. **Role programs as RISC-V.** The T0/T1/T2 programs follow the same emitter:
   loop counters as branches, Tensix words inline as `.ttinsn`, CB ops inline,
   caller-saved registers only, two loop levels (built for a counter stack).
   Delete loop-level replay, `Item::Shared`, the loop table, MOP config and the
   streamed program form.
4. **Firmware, tt-metal style.** Firmware sets up the CB interfaces from the
   launch descriptor, starts and joins the cores, calls each program at its
   fixed address with `a0` = args, and invalidates the I-cache before the run.
   Delete `run_stream`/`Pusher` and the `STREAMED`/`LOOPED` length flags. Add
   the NC executor (`exec_nc`) for the writer.
5. **Backend launch.** Replace the `dm` lists with the programs: stage each into
   `PROGRAM_CACHE` once, boot the executor on its core, run reader and roles
   concurrently (a depth-1 `Reserve` blocks otherwise), then the writer; join
   all five. Delete `dm` from the backend, `ttsim`'s `mover_list`, and
   `tt-isa` (`dm`, the streamed `dataflow`, the list regions), moving the shared
   constants to `mailbox`/`l1`.
6. `tt_tile` completes the tiling without assuming a page: the default traffic
   unit is what the access names (a whole row per scalar gather access), and the
   tiled form (one loop iteration = one tile, a tile-stepping `Unroll` becoming
   the `Loop`, within-tile lanes staying `Unroll(32)`) is an optimization the
   search chooses, never a precondition.
7. Replace `24_tt` with the model's 16 real kernels, in model order, until they
   run; every one stops at the debugging gate and is checked before the gate is
   removed for it.
8. One forward step of llama on ttsim, compared against CPU.
