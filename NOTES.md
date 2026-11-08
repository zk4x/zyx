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

## State

`tt-isa 0.1.0` is published on crates.io (repo `zk4x/tt-isa`, Apache-2.0,
credited to the metalium-rs authors, transfer on request). `zyx` itself is
**not** published.

Where things stand (2026-11-10, end of session 2):

- **Commits**: `8d01de49` (disassembler + loop emission), `a79f5286` (128-byte
  NOC probe), plus the transfer-body work before them (`6aa8c839`, `e121cdd4`).
- **`cargo build` clean; `AGENT=1 ZYX_DEV=cpu cargo test --lib`: 140 passed,
  1 ignored.** `140` includes `disasm::{every_encoder_round_trips,
  operands_are_named}` and `ttsim::tests::a_128_byte_move_round_trips`.
  The ignored test is `gather_kernel_runs_through_the_backend`, blocked on tiled
  addressing (see the next bullet).
- **The acceptance moved to `zyx-nn/tests/embedding_tt.rs`**: `llama_embedding`
  is `Embedding::forward` (the example's one-hot chain) with
  `LlamaConfig::default` shapes, `weight [128256, 3072]`, `ids [1, 1]`;
  `embedding_gather_small` is the same at `[128, 32]` for hand-checking. The old
  `23_tt_lowering::gather` (an `index_select` chain) is gone.
- **`fold_loops` does fold the llama one-hot chain**, but only when `Embedding`'s
  `arange` stays on the host: moving it to the device turns the mask into a load
  and the fold never matches. That was the whole difference — with the arange on
  the host the kernel is `token = ids[0]; row = weight[token*cols + col]`, CUDA
  runs it in **3.9 µs** (was 36 ms as a one-hot reduce) and the IR is the scalar
  row gather.
- **CUDA is the oracle and is green:** `ZYX_DEV=cuda:0 cargo test --test
  embedding_tt llama_embedding` passes (exact f32 against the host row).
- **TT now emits and runs the gather transfer** (the program executes, no
  timeout, no verify error) and returns the wrong row. **Root cause, measured:**
  `Pool::TT` stores every tensor as a tiled image, so the chunk's first 16 bytes
  are the `TILE_DATA` header and the data begins at byte 16; the transfer body
  addresses the param as flat row-major, so it reads the header (0) as the token
  and row 0 of the weight. The dump that proved it:
  `param 0 dram[0..32] = 00×16, 07,00,00,00 ...`, `param 1 = 00×16, 72,37,e3,3e`
  (`0x3ee33772` = `0.4437824` = `weight[0]`), and the L1 page held `07` at
  `page+16`.
- **What was added and kept**: `codegen/tenstorrent/disasm.rs` (RV32IM + Tensix
  `.ttinsn`, printed by `ZYX_DEBUG=16`), RISC-V loop emission in `boundary`
  (iteration index, constant trip count costs no register), the transfer body's
  contiguous-run page, the `TtCopy::Transfer` kind, and the 128-byte NOC probe.
- **What must not be repeated:** the old state bullets below are history; the
  "Failure modes" section is the operative one. Read it before touching
  anything.
- **The gather is the current task, not out of scope.** Under tile-major storage
  it can no longer be a compute-free loopback: a row inside a tile must be
  selected in-die (see "Layout: tile-major, forever").
- `24_tt.rs` holds llama-shaped op groups, not the model's real 16 kernels yet.
- `cargo package`: was verified earlier (172 files, 1.3 MiB compressed); it will
  not verify now because `tt-isa` is a path dependency.
- `tt-isa` is a path dependency (`../../tt-isa`) while it is edited; the
  firmware manifest uses an absolute path to it (`tt-firmware/Cargo.toml.in`),
  which must become the published version before packaging.

## Session start

1. Read this file end to end. Do not grep it for a keyword.
2. `cd zyx && cargo build`; `AGENT=1 ZYX_DEV=cpu cargo test --lib` — expect
   **140 passed, 1 ignored** (the ignored one is
   `gather_kernel_runs_through_the_backend`, blocked on tiled addressing).
3. Run the CUDA oracle:
   `cd zyx-nn && ZYX_DEV=cuda:0 cargo test --test embedding_tt llama_embedding`
   (passes, ~3.9 µs kernel) and `... embedding_gather_small`. That is the
   reference the TT result must match.
4. Run the TT anchors:
   `TT_TTSIM_LIB=/home/x/Dev/rust/metalium-rs/vendor/libttsim_bh.so cargo test
   --lib backend::tenstorrent::ttsim` and the `disasm` tests; then the
   acceptance `cd zyx-nn && TT_TTSIM_LIB=... cargo test --test embedding_tt
   -- --exact embedding_gather_small --nocapture`.
5. `ZYX_DEBUG=16` prints the emitted program as **disassembly** (RV32IM and
   inline Tensix); `ZYX_TT_IR=1` dumps the IR per stage. Pin one candidate
   (`ZYX_DEBUG=256`) before reading IR.
6. Work the `Next` list. No custom fixtures as acceptance, no graph-level
   changes, no progress reported as success.

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
  movement unit is the traffic the index names; a page is a tile when the
  tensor's layout is tiled (always, for DRAM), so the tiled form (one iteration =
  one tile, a tile-stepping `Unroll` becoming the `Loop`, within-tile lanes
  staying `Unroll(32)`) is a form the search may choose, never a precondition:
  `tt_tile`, `tt_add_movers`, `tt_sync_cbs`, the CB page and the emitter follow
  whatever the access actually moves.

## Layout: tile-major, forever (user, 2026-11-10)

TT DRAM holds **every** tensor as a 32×32 tiled image
(`TILE_SLOT = 4160` B: a 16-byte `TILE_DATA` header, then the four 16×16 faces).
There is no row-major DRAM option.

Why tile-major wins if only one layout can exist:

- The compute engine is tile-native: `UNPACR` fills `SrcA`/`SrcB`, `MVMUL`
  accumulates into `Dst`, the packer drains it — all in face order. Row-major
  storage would force a **tilize** on every input and an **untilize** on every
  output, and DMA cannot transpose: the conversion is an unpack-side
  (`unpack_tilize`) or pack-side (`pack_untilize`) pass through `Dst`.
- **Cost is charged to the core that pays it.** Movers (B/NC) and the packer run
  concurrently with the math core, so the scarce resource is compute-side issue,
  not mover bandwidth. Row-major spends compute-side cycles; tile-major spends
  mover cycles — and, for llama, ~200 tile ops per token against one gather.
- tt-metal makes the same call: `TILE_LAYOUT` is the default for compute
  tensors, and RM is a boundary layout that `tilize`/`untilize` convert
  explicitly *because* the conversion is expensive.

Consequence: **the embedding gather is no longer a compute-free loopback.** A
row inside a tile is not a page, so the gather must select it in-die (reader
brings the token's tiles, the compute section picks row `token % 32` into the
output tile). Row-major is only ever a *per-boundary* choice (as tt-metal's
embedding keeps its weight RM), never the storage convention.

## Boundary indexing: pages, not faces

TT's DRAM holds every tensor as a **tiled image**: `TILE_SLOT = 4160` bytes, a
16-byte `TILE_DATA` header plus a 32×32 tile in face order. The mover's unit is
the **page**, exactly as in tt-metal: `get_noc_addr(page)` and one
`noc_async_read`/`noc_async_write` of the page's bytes. On a tiled tensor that
page is a whole tile, addressed by tile ordinal. **The face layout belongs to
the unpacker, never to a mover**: no DRAM address in this backend contains
`face*256 + within`.

What tt-metal does, read from its kernels:

- `embeddings.cpp:60-64` — one row per token: `weights.get_noc_addr(token)` then
  `noc_async_read(src_noc_addr + chunk*chunk_size, l1, current_chunk_size)`,
  with `weight_page_size = weights.padded_shape()[-1] * elem`
  (`embeddings_fused_program_factory.cpp:42`). The page is a row because its
  weight is **row-major**.
- `embedding_ind_tilized.cpp` — the *indices* are tilized, not the weight:
  `weights.get_noc_addr(token)` still selects a stick and
  `noc_async_read(src_noc_addr, weight_l1_addr, width_size)` moves it. `face_size`
  and `tile_height` there decode the tiled index tensor.
- For a **tiled** tensor tt-metal moves the whole tile page by tile index and
  unpacks it; the mover still computes no face address.

So:

- A page is `tile*TILE_SLOT` for `TILE_SLOT` bytes; the datum block starts at
  `TILE_DATA`. The page's bytes are what crosses the NoC.
- Logical index → tile ordinal (the same shape math as before, and the only
  layout knowledge outside the unpacker):

  ```
  (rows, cols) = last two dims;  rank-1 [c] = (c, 1);  [] = (1, 1)
  ntr = ceil(rows/32)   ntc = ceil(cols/32)   el = dtype bits / 8
  col = E % cols        row = (E / cols) % rows        b = E / (rows*cols)
  tile = b*ntr*ntc + (row/32)*ntc + (col/32)
  ```
- The **compute-side** L1 tile is `tile*1024 + (row % 32)*32 + (col % 32)`; that
  is where face order is resolved, by the unpacker.
- **A row inside a tile is not a page.** The embedding gather over a tiled weight
  moves the tiles the row touches and picks the row in L1 (or unpacks it), which
  costs `tiles_per_row * TILE_SLOT` per token (96 tiles = 390 KiB for a llama
  row) instead of the row's 12 KiB. That is the price of the always-tiled layout
  and it is what tt-metal pays for a tiled tensor; a row-major weight is the
  optimization, not the rule.
- **Two index conventions.** `tt_reindex` rewrites some kernels to tile-datum
  units (`tile*1024 + pos`); others keep the flat logical index. Both are
  accepted.
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
  The page is the amount of traffic the access moves (a tile for a tiled
  boundary tensor). `n` in the four syncs counts those pages, exactly as
  tt-metal's configured `page_size` does (`num_pages * fifo_page_size`,
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

- **Tiles are congruent by construction**: a tile image adds the same datum
  offset on both sides, so a page move keeps the congruence.
- CB bases are placed at `≡16 mod 64` (`tt_place_cbs`), where a tile image's
  datum block lives; a tile slot is `TILE_SLOT = 4160` with a 16-byte header. A
  transfer's plain L1 page is placed the same way, at `≡16 mod 64`.
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

- `dc9a1e3d^` (pre-native emitter): the CB mechanics reference. The four
  `TTOp::{ReserveBack, PushBack, WaitFront, PopFront}` (`{ cb, n }`) come from
  there, emitted by `tt_sync_cbs`; `tt_place_pops`/`tt_dedup_pushes`/`tt_fifo_check`/
  `fifo_check_copy`/`tt_verify` are its gates. `tt_lock_dst` stays out (DST is
  `tt_dst_reuse` + `tt_reg_sync`). Its old pipeline:
  `tt_tile -> tt_lower_llks -> tt_lock_dst -> tt_dedup_pushes -> tt_init_math ->
  tt_sync_cbs -> tt_place_pops -> verify -> CSE -> DCE -> tt_verify`.
- `a8fa78b9`: 14.444 TFLOPS, bad 0, M5120 N11264 K1024 with depth 2.
- `~/Dev/cpp/tt-metal` is the reference for CB semantics and kernel shape:
  `dataflow_api.h`, `brisc.cc`/`trisc.cc`/`ncrisc.cc` (call, args, I-cache),
  `ckernel_ops.h` (`.ttinsn`), `matmul_multi_core/kernels/` (loops).
- tt-metal's per-role loop shape, read from the kernels: the four syncs sit
  **inside** the loop body once per iteration (`reader_unary.cpp`:
  `reserve; noc_async_read; barrier; push`; the sfpu-chain writer:
  `wait_front; get_read_ptr; noc_async_write; barrier; pop`; its compute:
  `wait_front; copy_tile; reserve_back(out); pack_tile; pop_front(in);
  push_back(out)`), `n` is in pages (`cb_push_back` scales by `fifo_page_size`),
  and the consumer addresses the CB by its current pointer, so a pop never
  rebases an index.
- `programming_examples/loopback/kernels/loopback_dram_copy.cpp` is the
  writeback proof: `for i { noc_async_read_page; barrier; noc_async_write_page;
  barrier }` over one plain L1 page, no CB — a page is whatever the host
  configured, not necessarily a tile.

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
11. **Proposed a graph-level change, was told it is crazy, withdrew it, then
    re-proposed it.** No graph-level changes. The kernel IR is the input.
12. **Called an IR "malformed" with no reference run.** The claim came from
    comparing two debug dumps — which were different autotune candidates. Run
    the reference (CUDA) or pin one candidate before drawing a conclusion.
13. **Kept fixing kernels that are not the gather** (the index expand) instead
    of making the gather the acceptance. Report the acceptance's failure, not a
    neighbouring kernel's progress as if it were the target.
14. **Edited files with `python3` and `sed -i` again** (the boundary emitter's
    loop state, and a `sed` on the test). The rule is the edit tool, always.
15. **Called 36 ms "pretty fast" without the per-kernel number**, then had to
    be pointed at `ZYX_DEBUG=1`, which prints nanos/FLOP/s/bytes per kernel.
    Use it; it exists.
16. **Said "10⁴× the necessary work"** when the ratio is 1.28×10⁵, and blended
    "correct IR" with "cheap IR" until the user pulled them apart. Say which
    claim is which.
17. **Asked whether tiled addressing is "really what tt-metal does"** after
    NOTES already said "follow tt-metal", instead of reading tt-metal's
    embedding kernels first. Read the reference before asking.

What the next session must know, concretely:

- **Never propose a graph-level change here.** One was proposed, called crazy,
  withdrawn, and then re-proposed — the worst sequence possible. The kernel IR
  is the input; the work is in the TT passes.
- **Use CUDA as the oracle.** `cd zyx-nn && ZYX_DEV=cuda:0 cargo test --test
  embedding_tt` is green (3.9 µs). The TT failure is lowering only.
- **Pin one autotune candidate before reading IR.** The `ZYX_DEBUG=4` /
  `ZYX_TT_IR=1` prints are per candidate; comparing a pre-search block with a
  later candidate produced the false "512 elements into a 64-element buffer"
  claim. Use `ZYX_DEBUG=256` (NoSearch) plus `ZYX_TT_IR=1` and read one kernel.
- **Acceptance:** `zyx-nn/tests/embedding_tt.rs` on ttsim, compared to the host.
  Hand-built kernels and the ttsim probes are unit-level only.
- **The transfer's DRAM side is a page.** Our pool is always tiled, so a page is
  a tile (`tile*TILE_SLOT`, data at `TILE_DATA`); the face layout is the
  unpacker's. `tt_add_transfer_body` currently addresses the param flat, which is
  why the gather reads the 16-byte header as its token. Fixing it means the
  transfer moves the tile pages the access touches and the row is picked or
  unpacked on the compute side — not per-element face arithmetic in a mover.
- **`ZYX_DEBUG=16` is the disassembly now.** Read the emitted program before
  reasoning about a wrong value; it names the address computation, the copy
  lengths and the loop.
- **Fixed this session and kept:** `merge_ranges` decomposes the fused flat axis
  row-major (last axis low), so vectorised lanes can be contiguous; the row body
  pages a contiguous vector store as `lanes` datums indexed by the lane offset;
  `Bool -> I32` is an identity cast; `fold_loops` folds the llama one-hot chain
  when the `Embedding` arange stays on the host.

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

1. ~~**Single mover writeback**~~ — **green** (`gather_writeback_runs_on_riscv_b`):
   one program on RISCV B, inline NOC read/write, four 64-byte rows out of order,
   checked against the host.
2. ~~**A 128-byte page move**~~ — **green** (`a_128_byte_move_round_trips`): one
   `noc_async_read`/`noc_async_write` of two NOC words.
3. **Reader program** — DRAM → CB with the inline CB ops.
4. **`large_elementwise`** — reader, T0/T1/T2, writer, against the host.
5. **`sigmoid`** — SFPU path, CB plan, pack.
6. **The embedding gather** — tile rows in, tile rows out (see `Next`).
7. `matmul` — Matrix Unit path. 8. `sum` — reduce path. 9. Loop/grid — nested
   loops. 10. Multi-op DST chain. 11. One llama forward step against CPU.

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
- **The hardware limit is 64-byte congruence: the DRAM and L1 addresses of a
  read must agree mod 64, a write mod 16.** A tiled image's datum block starts at
  `TILE_DATA = 16`, so a page's L1 buffer is placed `≡16 mod 64` (a tile CB's
  base, `tt_place_cbs`). The write unit is a whole 16-byte-congruent run — never
  a sub-16-byte element write that would clobber its neighbours.
- **Where the address is resolved:** `tt_tile` handles the tiling, `tt_reindex`
  the face orientation, and `tt_add_movers` only splits one kernel into the three
  segments; the tiled address itself is `tt_canonical_tile`'s. `tt_add_movers`
  does not compute page sizes itself.
- **Everything about a boundary access follows the page.** The page is the
  traffic the access moves (a tile for a tiled tensor, one row per stick for a
  row-major one); the CB page, the four syncs and the emitter follow it. A
  vector access over N lanes is N spans when the lanes are not contiguous, one
  aligned span when they are — the optimization is never the fallback.
- **Generality limit: llama inference is the demo** (user). Nothing more general
  than the kernels llama inference needs is required. That does **not** exclude
  the gather. Express it as `weight.gather(0, indices)` with the index tensor
  already in the output shape (uploaded), not `index_select`, whose first step
  materialises an index broadcast with a scattered store:
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

## Decisions (user)

- Loops are RISC-V branches in the compiled program, as in tt-metal; a program
  is called by its firmware, gets its arguments as an L1 `u32` array based in
  `a0`, is a leaf using caller-saved registers, and returns with `ret`. Its
  address is fixed at first compile in `PROGRAM_CACHE` (allocated once, never
  moved, no eviction).
- Tensix words are inline `.ttinsn` constants; CB and NOC ops are emitted inline
  by codegen. No `INSTRN_BUF` push, no MOP config, no loop-level replay,
  `Item::Shared` deleted. The word stream stays flat (`Box<[u32]>`); the
  disassembler is the debug tool.
- `dm` is dropped: reader and writer are programs, `KERNEL`/`LAUNCH`/`BARRIER`/
  `CALL`/`POKE`/`WAIT` are dispatch concepts. Start/join, CB interface setup,
  NoC init and mailboxes follow tt-metal.
- `tt_add_movers` builds the reader/writer blocks and canonicalizes the compute
  segment; the four syncs come from `tt_sync_cbs`.
- The emitter lives in `codegen/tenstorrent/riscv.rs`; two loop levels now, built
  so deeper nesting is a counter stack, not a rewrite.
- **IR tests are textual and exact** (`format!("{k}")` and `get_hash()` as
  literals), live under `#[cfg(test)]` in `kernel/tenstorrent.rs`, call the
  passes directly, and are written from the design — never pasted from a run.
- **`tt-isa` is ours to edit at will**; modules we would have to change anyway
  move into zyx.
- **Test shapes and index values are the agent's choice**; a shape is not a
  design decision and is not to be asked about.
- **`tt_tile` never folds a lane split back into a loop**; unwanted splits are
  prevented at `opt_split`/`opt_register_tiling` by the TT gate. The two group
  axes split independently and their trip counts may differ.
- **The gather test is a real tensor op** on `Dev::TT(0)`; a hand-built kernel is
  a unit counterpart at most.

## Open questions

1. What is the exact list of the model's 16 kernels (ops, shapes, order), and
   are all 16 reached at `seq_len = 1` decode given that CPU prefill crashes?
2. What counts as one kernel: one compiled `Kernel` per fused op group, with
   repeated invocations sharing it? Does `24_tt` get replaced one-for-one or
   extended?
3. Is `sum_drop_col`'s orientation needed for llama at `seq_len = 1` (attention
   over one key is `v`), and is its shape the one the model uses?

Nothing about launch, firmware, mailboxes, NoC init or the `tt-isa` surface is
open: those follow tt-metal. `exec_nc` for the writer is work, not a question.

## Next

The full backend, one complete step at a time. Nothing here is optional and
nothing is to be done in a reduced form.

0. Settle the open questions as each step reaches them.
1. **The embedding gather as a tile-row kernel (current step).** Storage is
   tile-major forever, so the gather is no longer a compute-free loopback: the
   reader copies the token's tiles (page = a tile, `TILE_SLOT` bytes, data at
   `TILE_DATA`, L1 page `≡16 mod 64`), the compute section selects row
   `token % 32` out of each tile (two 16-datum halves, `SFPULOAD`/`SFPSTORE`
   into the output tile's row 0) and packs, and the writer drains the output
   tiles. Route the kernel through the section path (`tt_add_movers`), not
   `tt_add_transfer_body`: a pure copy is only a transfer when its DRAM side is
   a page, and a tile row is not a page. Acceptance:
   `zyx-nn/tests/embedding_tt.rs::embedding_gather_small`, then
   `llama_embedding`.
2. **Reader/writer programs from codegen: done for the transfer, pending for
   sections.** `TTOp::Reader`/`Writer` lower the boundary IR to RISC-V (index as
   written, inline NOC/CB ops, loops as branches, `ret`), and the launch stages
   them with args. What is left is the CB interface table in L1 and the
   concurrent start of reader/roles/writer for a depth-1 ring.
3. **Role programs as RISC-V.** T0/T1/T2 through the same emitter: loop counters
   as branches, Tensix words inline as `.ttinsn`, CB ops inline, caller-saved
   registers only. Delete loop-level replay, `Item::Shared`, the loop table, MOP
   config and the streamed program form.
4. **Firmware, tt-metal style.** CB interfaces from the launch descriptor, start
   and join the cores, call each program at its fixed address with `a0` = args,
   invalidate the I-cache. Delete `run_stream`/`Pusher` and the `STREAMED`/
   `LOOPED` flags. Add the NC executor (`exec_nc`).
5. **Backend launch.** Replace the `dm` lists with the programs; run reader and
   roles concurrently; join all five. Delete `dm` from the backend, `ttsim`'s
   `mover_list` and `tt-isa`.
6. `tt_tile` completes the tiling without assuming a page: the access names the
   traffic, the search chooses the tiled form.
7. Replace `24_tt` with the model's 16 real kernels, in model order.
8. One forward step of llama on ttsim, compared against CPU.
