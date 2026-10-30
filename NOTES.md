# Native Tenstorrent backend — design notes

> Repository root is `/home/x/Dev/rust/zyx2` (crate in `zyx/`), branch
> `tt-native`. All paths below are relative to that root. `/home/x/Dev/rust/zyx`
> is a *different* checkout and is not this work.

## Goal

`zyx` compiles a whole `Kernel` into native Tenstorrent programs and runs them
through `metalium-rs` (`tt-device`, `tt-isa`, `tt-kernels`, `tt-ttsim`), with:

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
- `24_tt::sum_sigmoid` and `24_tt::matmul_sigmoid` are the acceptance tests:
  one kernel, no DRAM traffic between the two compute ops.

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
  test is the only known green TT baseline. `24_tt` has never passed
  anything. Not new semantics.

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
4. `tt-device`, `tt-ttsim`, `tt-firmware-images` are declared in
   `Cargo.toml` but referenced nowhere in `src/`, i.e. the native transport
   is not written yet.

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

`tests/23_tt_lowering.rs::matmul` and `tests/24_tt.rs::matmul` both pass on
the native path (metalium's pinned ttsim, no C++ shim). The four build
blockers above are gone: the transport and the native emitter landed on
`tt-native`. What it took, and the facts behind it:

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

Still failing, and why: `sigmoid`/`matmul_sigmoid` (`24_tt`,
`23_tt_lowering`) trip `tt_verify: DRAM write in compute` because a
`[10, 256]` store is not a `[32, 32]`/`[32]` tile and phase 3 bails;
`sum`/`sum_sigmoid` panic with `tt_verify: missing EndReader`; `softmax`
fails in the runtime layer. All of them need the loop/grid (multi-tile)
path, not another one-tile fix.

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

- [ ] `24_tt`: transfer, copy kernel, sigmoid, matmul, sum.
- [ ] Loop/grid kernel.
- [ ] Multi-op chain with DST intermediates: sigmoid -> sigmoid, reduce ->
      sigmoid, matmul -> sigmoid. No CB roundtrip between them; the only
      pack is the final output boundary.
- [ ] Delete obsolete feature-gated tests and `ZYX_TT_DEBUG_MATMUL` /
      `ZYX_TT_TRACE`.

## Next up (in order)

1. **Loop / grid (multi-tile) lowering.** Every remaining red test needs it:
   `tt_lower_llks` bails on any `Op::Loop`, phase 3 only accepts
   `[32, 32]`/`[32]` stores, and `runtime::run` programs one tile. A
   `[10, 256]` tensor is eight tiles; `[256, 256]` is sixty-four.
2. **Grid dispatch.** Launch a tile grid (`runtime::run` per tile, or one
   kernel per tile with the DM moves indexed by tile) and stop erroring on
   a non-1x1 grid in `TTDevice::launch`.
3. **SFPU path end to end.** `sigmoid` lowers to the native SFPU program
   already; it fails in `tt_verify`/phase 3 before reaching the device.
4. **Multi-op DST chains** (matmul -> sigmoid, reduce -> sigmoid) with no
   CB roundtrip between them.
5. Delete the dead C++ `TtSection.out` emitter and the obsolete
   `ZYX_TT_DEBUG_MATMUL` / `ZYX_TT_TRACE` hooks once nothing reads them.

## Session state (last update)

- `cargo check` is green; tests compile.
- **Green on tt_sim now:** `24_tt::transfer_roundtrip`, `24_tt::matmul`,
  and the gate `23_tt_lowering::matmul`, all on the native path (no C++
  shim, host-memory DRAM, host-side tile layout).
- Not green yet, all for the same reason (one tile only): `24_tt::sigmoid`
  (`tt_verify: DRAM write in compute`), `24_tt::sum` and
  `23_tt_lowering::sum` / `sum_sigmoid` (`tt_verify: missing EndReader`),
  `23_tt_lowering::sigmoid` (`DRAM write in compute`), `softmax` /
  `matmul_sigmoid` (runtime layer).
- The C++ shim files are deleted; the backend runs `TTProgramDesc.native`
  through `tt_kernels::runtime::run` on a `zyx-ttsim` owner thread.
- Phase 2 is recursive/root-first over one biggest-first pattern list;
  every native op writes a fresh logical DST row; `tt_dst_reuse` assigns
  the physical rows afterward. Multi-use computed values get a CB
  roundtrip. Phase 3 packs single-use DST outputs (into an FP32 writer CB)
  and drains materialized CBs directly. `tt_verify` rejects DRAM traffic in
  compute and intra-kernel store->load.
- The generic SFPU IR, pattern table, recursive lowering, and
  `tt_dst_reuse` are in place and `cargo check` is green.
- The reduce codegen is native: `ReduceTile` emits the metalium
  `sfpu::reduce` math program on the actual DST rows (row remap of
  `SFPLOAD`/`SFPSTORE` `Imm10`) and no longer calls the LLK
  `reduce_tile`. The raw words are in `TTProgramDesc.native_compute`.
  A unit test checks the row remap.
- The rest of the compute path still uses the old C++ string source;
  the native role streams and the native backend are the next step.
- Work is committed gradually on branch `tt-native`. Commits so far:
  1. drop TT feature gates, `build.rs`, and the C++ shim;
  2. reduce `TTOp` to native ops and update lowering, passes, verifier;
  3. adapt whole-kernel codegen to the native op set;
  4. add `NOTES.md` and `tests/24_tt.rs`;
  5. one biggest-first pattern list, output-only packs, verifier checks;
  6. generic SFPU, recursive lowering, multi-use CB materialization,
     separate `tt_dst_reuse`;
  7. native reduce math on the actual DST rows (no LLK `reduce_tile`);
  8. ISA facts and the native-reduce status.
- `cargo check --lib` was green at commit `30365dd6`. No simulator/device
  run has been made.

### Current working tree (uncommitted, mid-refactor)

- Codegen still produces `compute_native: Vec<u32>` (raw
  `Instruction::word()`s, the T1 math stream) as a flat mirror of
  `native.compute[1]`; it goes away once the backend reads `native`.
- `MatmulTiles` is the only compute op still on the C++ path, so a kernel
  with a matmul cannot run natively yet.
- `Sfpu` codegen currently returns an explicit error ("requires the native
  emitter"): the generic SFPU path has no C++ fallback and is gated on the
  emitter.

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

Baseline: the old C++ path passed `23_tt_lowering::matmul` only.
`24_tt` has never passed a single test. Treat every `24_tt` case below as
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
