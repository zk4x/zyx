// Copyright (C) 2025 zk4x
// SPDX-License-Identifier: LGPL-3.0-only WITH Classpath-exception-2.0

//! TTIR — Tenstorrent physical IR.
//!
//! One program = one ordered `Vec<TTOp>` covering all three RISC-V kernel
//! sources, split at render time at the `EndReader`/`EndCompute`/`EndWriter`
//! markers.
//!
//! # PIPELINE (fixed shape — every stage is one of exactly two kinds)
//!
//! **Stage 1 — Conversion** ([`Compiler::new`]): the only non-uniform stage.
//! Walks the kernel IR once and builds the initial `Vec<TTOp>`: CB and
//! hardware-object allocation, de-SSA into physical registers
//! (`VarId`/`TileId`), tile lowering, naive init/reconfig emission, and the
//! **fused-composite claim prepass** (sigmoid/silu → `TileFused`/
//! `FusedInit`). The fused claim is the ONLY prepass in the pipeline; it
//! lives here because it decides which tile ops lowering skips. Nothing
//! else may hide inside conversion.
//!
//! **Stage 2 — Lowering passes**: ANY number of simple passes after
//! conversion. Each pass is a method that processes the ops vector and
//! generates a new, transformed vector of ops — `Vec<TTOp> → Vec<TTOp>`,
//! rebuild-don't-splice, O(n), no hidden state, no global rewriting, no
//! prepass-like coupling into conversion. Passes see plain op streams and
//! may read (never mutate) shared tables (CB formats, param ordinals).
//!
//! Single-pass rule (Wirth-style): each pass walks the input vector
//! exactly ONCE, front to back, emitting the replacement vector as it
//! goes. No second scans, no fixpoints, no cross-stream backpatching.
//! A pass that needs non-local context (loop-trip products, CB depths)
//! takes it from the shared tables or a bounded local window — never
//! from re-walking the stream. That's what keeps TTIR simple: every
//! pass is one linear transducer, and the pipeline is their composition.
//! The current fixed order:
//!
//! 1. `lock_dst` — DST lock cones around pack ops.
//! 2. `fill_out_cbs` — output CB packing (`PackTile`/`PackReconfig`).
//! 3. `init_math` — hoists/dedups per-unit init config.
//! 4. `reconfig_pack` — packer format reconfigs.
//! 5. `sync_cbs` — CB reserve/push/wait accounting; compute reads get
//!    waits only, no pops (must see the FINAL traffic shape; batching
//!    changes counts, so any pass that alters traffic must run BEFORE
//!    this).
//! 6. `dedup_waits` — drop same-group repeat waits (shared tiles).
//! 7. `place_pops` — pop after last static use (FIFO order assumed).
//! 8. `hoist_dedup_inits` — hoist init/reconfig effect ops out of
//!    constant-trip loops, dedup adjacent same-config.
//! 9. `noc_movement` — NOC reads/writes for Global params.
//! 10. `hoist_writer_accessors` — writer-section accessor hoist.
//! 11. `batch_cbs` — hoists per-tile sync groups (reader reserve/push,
//!    writer wait/pop) out of innermost constant-trip loops: one
//!    multi-tile `ReserveBack(n)`/`PushBack(n)` (or `WaitFront(n)`/
//!    `PopFront(n)`) around the loop, per-trip transfer writes slot
//!    `counter` via `AsyncRead/Write { off: Some(counter) }`, and a
//!    single barrier covers the whole block. Traffic totals are
//!    unchanged, so the sync accounting that ran before stays valid.
//! 12. `tile_regs` — DST acquire/commit/ release accounting.
//! 13. `verify` — structural checks on the fully-physical stream.
//! 14. `render` — table walk producing the three C++ sources.
//!
//! Adding a transformation = adding a pass method in the order above.
//! NEVER add another prepass; NEVER bury a transformation inside
//! conversion or inside another pass.

use crate::{
    DType, Map, Set,
    dtype::Constant,
    error::{BackendError, ErrorStatus},
    kernel::{BOp, IDX_T, Kernel, MMADType, MemLayout, MemScope, Op, OpId, ParamKind, RangeKind, TileDim, UOp},
    slab::{Slab, SlabId},
    types::TinyString,
};
use std::fmt::{Display, Formatter};

fn is_one_const(kernel: &Kernel, op: OpId) -> bool {
    kernel.resolve_const(op).is_some_and(|c| c.is_one())
}

/// Kernel sections delimited by barriers: reader (head -> 1st barrier),
/// compute (1st -> 2nd), writer (2nd -> end).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum TtSection {
    Reader,
    Compute,
    Writer,
}

impl TtSection {
    /// Step to the next section at a barrier. Errors past the writer:
    /// kernels have exactly 3 sections (2 barriers).
    fn advance(&mut self) -> Result<(), BackendError> {
        *self = match self {
            TtSection::Reader => TtSection::Compute,
            TtSection::Compute => TtSection::Writer,
            TtSection::Writer => {
                return Err(BackendError {
                    status: ErrorStatus::KernelCompilation,
                    context: "tenstorrent2: kernels have exactly 3 sections (2 barriers)".into(),
                });
            }
        };
        Ok(())
    }
}

/// DRAM buffer page size in bytes: every DRAM `TensorAccessor` strides by
/// the buffer page size, never the dtype tile size.
pub(crate) const TT_DRAM_PAGE_BYTES: u32 = 4096;

/// Subset check for inner containment: every consumer is inside
/// the pattern (unlike [`uses_exactly`], the pattern may hold other
/// ops that do not consume this one).
fn uses_within(consumers: &Map<OpId, Vec<OpId>>, inner: OpId, allowed: &[OpId]) -> bool {
    match consumers.get(&inner) {
        None => false,
        Some(cs) => !cs.is_empty() && cs.iter().all(|c| allowed.contains(c)),
    }
}

/// A composite the backend recognizes and emits as one LLK call
/// (`sigmoid_tile` / `silu_tile` from `compute_kernel_api.h`). The
/// kernel IR is unchanged — no new `UOp`, no other backend touched.
/// A missed match only costs speed: the plain composite still emits.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) enum FusedKind {
    Sigmoid,
    Silu,
}

/// Closed op list for one section: the section's ops in IR order with
/// their dtypes and section-local refcounts.
pub(crate) struct SectionData {
    /// Ops in IR order.
    pub(crate) ops: Vec<OpId>,
    /// Dtype and layout per op.
    pub(crate) dtypes: Map<OpId, (DType, MemLayout)>,
    /// Refcounts counting uses inside this section only.
    pub(crate) rcs: Map<OpId, u32>,
}

/// A matched composite: its external (non-const) input plus the ops
/// the single call subsumes (root excluded — the root stays in the
/// section op list, the inners are filtered out of it).
#[derive(Clone, Debug)]
pub(crate) struct FusedPat {
    pub(crate) kind: FusedKind,
    pub(crate) x: OpId,
    pub(crate) inners: Vec<OpId>,
}

/// Set-equality on a consumer list: every consumer is expected and
/// every expected op consumes (order-independent — the two users of
/// a shared `exp` may appear in either IR order).
fn uses_exactly(consumers: &Map<OpId, Vec<OpId>>, inner: OpId, expected: &[OpId]) -> bool {
    match consumers.get(&inner) {
        None => false,
        Some(cs) => cs.len() == expected.len() && cs.iter().all(|c| expected.contains(c)),
    }
}

impl FusedKind {
    pub(crate) fn init_name(self) -> &'static str {
        match self {
            FusedKind::Sigmoid => "sigmoid_tile_init();",
            FusedKind::Silu => "silu_tile_init();",
        }
    }

    pub(crate) fn call_name(self) -> &'static str {
        match self {
            FusedKind::Sigmoid => "sigmoid_tile",
            FusedKind::Silu => "silu_tile",
        }
    }

    /// Either fused shape at `op` (silu first): tile-domain float
    /// roots only (BF16/FP32 DST; F16 SFPU is unproven on this
    /// board). Strict containment — every subsumed op's consumers
    /// are all inside the pattern, and the external input is consumed
    /// only by the pattern (the single call transforms its slot in
    /// place). Anything else falls back to the plain composite —
    /// slower, never wrong.
    pub(crate) fn match_pat(kernel: &Kernel, data: &SectionData, consumers: &Map<OpId, Vec<OpId>>, op: OpId) -> Option<FusedPat> {
        let (dt, layout) = data.dtypes.get(&op).copied()?;
        if !matches!(layout, MemLayout::Tile { .. }) || !matches!(dt, DType::F32 | DType::BF16) {
            return None;
        }
        Self::silu_pat(kernel, data, consumers, op).or_else(|| Self::sigmoid_pat(kernel, data, consumers, op))
    }

    /// Sigmoid shape below `s` (no containment yet): the external
    /// input and the subsumed ops under `s` (`s` excluded — the caller
    /// decides whether `s` stays (standalone root) or goes (silu
    /// inner)). Two spellings: the builder composite
    /// `reciprocal(1 + exp(-x))` and the eager `exp(x) / (exp(x) + 1)`
    /// (shared `exp`, hence the two-consumer shape).
    fn sigmoid_shape(kernel: &Kernel, data: &SectionData, s: OpId) -> Option<(OpId, Vec<OpId>)> {
        if let Op::Unary { x: den, uop: UOp::Reciprocal } = kernel.at(s) {
            let Op::Binary { x: a, y: b, bop: BOp::Add } = kernel.at(*den) else {
                return None;
            };
            let e = if is_one_const(kernel, *a) {
                *b
            } else if is_one_const(kernel, *b) {
                *a
            } else {
                return None;
            };
            let Op::Unary { x: nx, uop: UOp::Exp } = kernel.at(e) else {
                return None;
            };
            let Op::Unary { x, uop: UOp::Neg } = kernel.at(*nx) else {
                return None;
            };
            if !matches!(data.dtypes.get(x).map(|d| d.1), Some(MemLayout::Tile { .. })) {
                return None;
            }
            return Some((*x, vec![*den, e, *nx]));
        }
        let Op::Binary { x: z, y: den, bop: BOp::Div } = kernel.at(s) else {
            return None;
        };
        let Op::Binary { x: a, y: b, bop: BOp::Add } = kernel.at(*den) else {
            return None;
        };
        if !(is_one_const(kernel, *a) && *b == *z || is_one_const(kernel, *b) && *a == *z) {
            return None;
        }
        let Op::Unary { x, uop: UOp::Exp } = kernel.at(*z) else {
            return None;
        };
        if !matches!(data.dtypes.get(x).map(|d| d.1), Some(MemLayout::Tile { .. })) {
            return None;
        }
        Some((*x, vec![*z, *den]))
    }

    /// Silu shape at `op`: `mul(x, s)` (either side) with `s` a
    /// sigmoid shape fed by the mul's other side. `s` itself goes
    /// (its only consumer is the mul); the root stays.
    fn silu_pat(kernel: &Kernel, data: &SectionData, consumers: &Map<OpId, Vec<OpId>>, op: OpId) -> Option<FusedPat> {
        let Op::Binary { x: a, y: b, bop: BOp::Mul } = kernel.at(op) else {
            return None;
        };
        for (s, other) in [(*a, *b), (*b, *a)] {
            let Some((x, below)) = Self::sigmoid_shape(kernel, data, s) else {
                continue;
            };
            if x != other || !uses_exactly(consumers, s, &[op]) {
                continue;
            }
            let mut allowed = below.clone();
            allowed.push(s);
            if !below.iter().all(|&inner| uses_within(consumers, inner, &allowed)) {
                continue;
            }
            let Some(&entry) = below.iter().find(|&&o| matches!(kernel.at(o), Op::Unary { x: ix, .. } if *ix == x)) else {
                continue;
            };
            if !uses_exactly(consumers, x, &[entry, op]) {
                continue;
            }
            let mut inners = below;
            inners.push(s);
            return Some(FusedPat { kind: FusedKind::Silu, x, inners });
        }
        None
    }

    /// Standalone sigmoid shape at `op`: the root stays, only the ops
    /// below it go.
    fn sigmoid_pat(kernel: &Kernel, data: &SectionData, consumers: &Map<OpId, Vec<OpId>>, op: OpId) -> Option<FusedPat> {
        let Some((x, below)) = Self::sigmoid_shape(kernel, data, op) else {
            return None;
        };
        let entry = below.iter().find(|&&o| matches!(kernel.at(o), Op::Unary { x: ix, .. } if *ix == x)).copied().unwrap_or(x);
        let mut allowed = below.clone();
        allowed.push(op);
        if !below.iter().all(|&inner| uses_within(consumers, inner, &allowed)) {
            return None;
        }
        if !uses_exactly(consumers, x, &[entry]) {
            return None;
        }
        Some(FusedPat { kind: FusedKind::Sigmoid, x, inners: below })
    }
}

/// Host-side param data plus dataflow (NOC) traffic emission.
///
/// Holds the global head-order ordinal of every param and the
/// Global/GlobalMut dtypes, and emits the reader/writer NOC sequences:
/// address computation, async read/write, and read/write barriers. The
/// section generators own the source text; every method here checks its
/// inputs and appends exactly one sequence.
pub(crate) struct NocEmitter {
    /// Global head-order ordinal of every param (all kinds).
    pub(crate) param_ordinal_of: Map<OpId, u32>,
}

#[allow(unused_must_use)]
impl NocEmitter {
    /// Build param state from a kernel: the param ordinals and
    /// input/output dtypes. One walk.
    pub(crate) fn new(kernel: &Kernel) -> Result<Self, BackendError> {
        let mut param_ordinal_of: Map<OpId, u32> = Map::default();
        let mut next_param = 0u32;
        let mut input_dtypes: Vec<DType> = Vec::new();
        let mut output_dtypes: Vec<DType> = Vec::new();
        let mut scan = kernel.head;
        for _ in 0..10_000 {
            if scan.is_null() {
                break;
            }
            if let Op::Param { dtype, kind, .. } = &kernel.ops[scan].op {
                param_ordinal_of.insert(scan, next_param);
                next_param += 1;
                match kind {
                    ParamKind::Global => input_dtypes.push(*dtype),
                    ParamKind::GlobalMut => output_dtypes.push(*dtype),
                    ParamKind::Variable => {}
                }
            }
            scan = kernel.next_op(scan);
        }
        if !scan.is_null() {
            return Err(BackendError {
                status: ErrorStatus::KernelCompilation,
                context: "tenstorrent2: compiler scan did not finish in 10000 steps".into(),
            });
        }
        Ok(Self { param_ordinal_of })
    }
}

/// Circular buffer ID for Tenstorrent codegen v2.
///
/// This is a unique identifier for each circular buffer in the compiled
/// Tenstorrent program.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct CBId(pub(crate) u32);

impl CBId {
    /// NULL
    pub const NULL: Self = Self(u32::MAX);

    /// Check if this CBId is null.
    pub const fn is_null(self) -> bool {
        self.0 == u32::MAX
    }
}

impl std::fmt::Display for CBId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::Display::fmt(&self.0, f)
    }
}

impl From<usize> for CBId {
    fn from(value: usize) -> Self {
        CBId(value as u32)
    }
}

impl From<CBId> for usize {
    fn from(value: CBId) -> usize {
        value.0 as usize
    }
}

impl SlabId for CBId {
    const ZERO: Self = Self(0);
    const NULL: Self = Self(u32::MAX);

    fn inc(&mut self) {
        self.0 += 1;
    }
}

/// DST tile slot. Budget: 16 in BF16 mode, 8 in FP32 mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct TileId(pub u8);

impl TileId {
    /// DST budget in BF16 mode (each FP32 tile occupies 2 BF16 slots).
    pub const BUDGET_BF16: usize = 16;
    /// DST budget in FP32 mode.
    pub const BUDGET_FP32: usize = 8;
}

/// Scalar C register slot (`r{reg}`). Unbounded.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct VarId(pub u32);

/// One `Asm` template operand: a CB renders as its index, a tile as
/// its DST slot, a scalar as its register.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AsmOperand {
    /// Circular buffer operand.
    Cb(CBId),
    /// DST slot operand.
    Tile(TileId),
    /// Scalar register operand.
    Var(VarId),
}

/// CB-slot sharing group: which dynamic tile a wait/use refers to.
/// Identity is the source `Op::Load` op (one load op = one logical
/// tile acquisition; compute-side load indices are fresh consts and
/// carry no identity). Same load feeding several matmuls straight-line
/// = same tile (dedup never crosses loop/branch markers, so per-trip
/// tiles stay per-trip). `None` (opaque) never merges.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WaitGroup {
    /// Left input of a `TileMatmul` (its `x_load`).
    MatA(OpId),
    /// Right input of a `TileMatmul` (its `y_load`).
    MatB(OpId),
}

/// One physical instruction. Each variant is one emitted line (or, after
/// decomposition, one instruction) and carries a static signature (CB
/// traffic, DST lock effects, executing thread) used by verify.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TTOp {
    // -- Structure: section boundaries and control flow --
    /// End of the reader (NCRISC/BRISC movement) section.
    EndReader,
    /// End of the compute (TRISC) section.
    EndCompute,
    /// End of the writer section. Explicit, never implicit end-of-Vec.
    EndWriter,
    /// Loop with trip count in a scalar register (defined by a prior
    /// `Const`/`Binary` statement).
    Loop {
        /// Number of iterations (input).
        len: VarId,
        /// Loop counter slot (output, used by body index math).
        counter: VarId,
        /// Counter dtype (always `IDX_T`; carried for render declarations).
        dtype: DType,
        /// Compile-time trip count (`None` = symbolic; traffic under a
        /// symbolic loop is a compilation error in `sync_cbs`).
        trip: Option<u32>,
    },
    /// End of loop body.
    EndLoop,
    /// Branch on a scalar register.
    If {
        /// Condition value.
        cond: VarId,
    },
    /// End of branch body.
    EndIf,
    // -- Scalar C: rendering statements, no passes over these --
    /// Kernel launch argument read: `z = get_arg_val(ordinal)`.
    Arg {
        /// Destination scalar register.
        z: VarId,
        /// Element dtype (render declaration).
        dtype: DType,
        /// Global launch-arg ordinal (backend runtime-arg tables).
        ordinal: u32,
    },
    /// Scalar constant definition (rendered via `c_type`/`c_code`).
    Const {
        /// Destination scalar register.
        z: VarId,
        /// Literal value.
        value: Constant,
    },
    /// Scalar binary: `z = x <bop> y`.
    Binary {
        /// Destination scalar register.
        z: VarId,
        /// Result dtype (render declaration).
        dtype: DType,
        /// Left operand.
        x: VarId,
        /// Right operand.
        y: VarId,
        /// Operation.
        bop: BOp,
    },
    /// Scalar unary: `z = <uop> x`.
    Unary {
        /// Destination scalar register.
        z: VarId,
        /// Result dtype (render declaration).
        dtype: DType,
        /// Source scalar register.
        x: VarId,
        /// Operation.
        uop: UOp,
    },
    /// Scalar cast: `z = (dtype)x`.
    Cast {
        /// Destination scalar register.
        z: VarId,
        /// Result dtype (render declaration).
        dtype: DType,
        /// Source scalar register.
        x: VarId,
    },
    /// Scalar fused multiply-add: `z = x * y + w`.
    Mad {
        /// Destination scalar register.
        z: VarId,
        /// Result dtype (render declaration).
        dtype: DType,
        /// Multiplicand.
        x: VarId,
        /// Multiplier.
        y: VarId,
        /// Addend.
        w: VarId,
    },
    /// Backend assembly escape hatch: `{i}` substitutes the i-th
    /// operand (CBs as indices, tiles as slots, scalars as registers).
    Asm {
        /// Assembly template.
        asm: TinyString,
        /// Operand values.
        ops: Vec<AsmOperand>,
    },
    /// NOC address computation:
    /// `z = accessor.get_noc_addr(page, offset)` (one `uint64_t` line;
    /// page/offset render inline from the index register).
    NocAddr {
        /// Destination scalar register (the address, always `u64`).
        z: VarId,
        /// Source param ordinal (names the accessor `p{ordinal}`).
        ordinal: u32,
        /// Element index (a scalar register holding the tile index).
        index: VarId,
        /// Element size in bytes (scales the index).
        elem_size: u32,
    },
    /// DRAM accessor declaration for one Global/GlobalMut param
    /// (`TensorAccessor` triple; section-implied `p`/`p_out` naming,
    /// chained compile-time offsets). Rendered once per param per
    /// section, ahead of its first traffic op.
    NocAccessor {
        /// Source param ordinal (names the accessor `p{ordinal}`).
        ordinal: u32,
        /// Element dtype (backend input/output dtype tables).
        dtype: DType,
        /// Global (reader) or GlobalMut (writer).
        kind: ParamKind,
    },
    /// Tensix core grid X coordinate (group range axis 0). Render lowers
    /// to the section's group-arg read (`get_arg_val`, one per axis after
    /// the section args, mirroring the legacy `group_arg` layout).
    TensixGridX {
        /// Destination scalar register.
        z: VarId,
        /// Coordinate dtype (render declaration).
        dtype: DType,
        /// Section-local runtime arg index (lowering precomputes the
        /// section param layout, so render stays a single emitter).
        arg: u32,
    },
    /// Tensix core grid Y coordinate (group range axis 1).
    TensixGridY {
        /// Destination scalar register.
        z: VarId,
        /// Coordinate dtype (render declaration).
        dtype: DType,
        /// Section-local runtime arg index.
        arg: u32,
    },
    // -- CB traffic: FIFO sync, one op per event --
    /// Declare one hardware circular buffer (single object, never duplicated
    /// across sections).
    CbDeclare {
        /// Circular buffer.
        cb: CBId,
        /// Depth in tiles.
        n_tiles: u32,
        /// Runtime descriptor format code (CB storage dtype).
        format: u32,
    },
    /// DST register-file mode (kernel-wide header op, first in the
    /// stream): 32-bit iff the kernel touches F32 tiles. Lowering
    /// asserts the alloc budget; `verify` re-checks the max slot;
    /// render forwards the flag to the backend.
    DstMode {
        /// True = 16-tile BF16 file, false = 8-tile FP32 file.
        bf16: bool,
    },
    /// `cb.reserve_back(n)`: open n back slots for writing.
    ReserveBack {
        /// Circular buffer.
        cb: CBId,
        /// Slots to reserve.
        n: u32,
    },
    /// `cb.push_back(n)`: publish n written slots.
    PushBack {
        /// Circular buffer.
        cb: CBId,
        /// Slots to publish.
        n: u32,
    },
    /// `cb.wait_front(m)`: block until m front slots hold data.
    WaitFront {
        /// Circular buffer.
        cb: CBId,
        /// Slots to wait for.
        m: u32,
        /// Sharing group this wait covers (`None` = opaque, never
        /// merged). Set by `sync_cbs` from the consumer op; read by
        /// the wait-dedup pass. Ignored by render/verify/batching.
        grp: Option<WaitGroup>,
    },
    /// `cb.pop_front(n)`: release n consumed slots.
    PopFront {
        /// Circular buffer.
        cb: CBId,
        /// Slots to release.
        n: u32,
    },
    // -- Movement: pre-sync tile transfers, one op per transfer --
    /// Reader transfer: DRAM tile into a CB (`Global` param source).
    /// `sync_cbs` wraps it with reserve/push; `expand_movement` lowers
    /// it to accessor + address + async read + barrier.
    ReadTile {
        /// Source param ordinal (names the accessor `p{ordinal}`).
        ordinal: u32,
        /// Element dtype (accessor + backend input table).
        dtype: DType,
        /// Element index (scalar register holding the tile index).
        index: VarId,
        /// Destination circular buffer.
        cb: CBId,
        /// Transfer size in bytes.
        bytes: u32,
        /// Element size in bytes (scales the index).
        elem_size: u32,
    },
    /// Writer transfer: CB tile into DRAM (`GlobalMut` param sink).
    /// `sync_cbs` wraps it with wait/pop; `expand_movement` lowers it
    /// to accessor + address + async write + barrier.
    WriteTile {
        /// Source circular buffer.
        cb: CBId,
        /// Destination param ordinal (names `p_out{ordinal}`).
        ordinal: u32,
        /// Element dtype (accessor + backend output table).
        dtype: DType,
        /// Element index (scalar register holding the tile index).
        index: VarId,
        /// Transfer size in bytes.
        bytes: u32,
        /// Element size in bytes (scales the index).
        elem_size: u32,
    },
    // -- NOC: real firmware calls, stay C forever --
    /// `noc_async_read(addr, cb.get_write_ptr()[+ off*bytes], bytes)` —
    /// no barrier (one barrier follows each movement sequence, plus a
    /// trailing reader barrier).
    AsyncRead {
        /// Source NOC address (scalar register holding `get_noc_addr` result).
        addr: VarId,
        /// Destination circular buffer.
        dst_cb: CBId,
        /// Transfer size in bytes.
        bytes: u32,
        /// Tile-slot offset within a hoisted block (`None` = slot 0,
        /// plain write pointer).
        off: Option<VarId>,
    },
    /// `noc_async_read_barrier()`.
    NocReadBarrier,
    /// `noc_async_write(cb.get_read_ptr()[+ off*bytes], addr, bytes)` —
    /// no barrier (one barrier follows each movement sequence).
    AsyncWrite {
        /// Source circular buffer.
        src_cb: CBId,
        /// Destination NOC address.
        addr: VarId,
        /// Transfer size in bytes.
        bytes: u32,
        /// Tile-slot offset within a hoisted block (`None` = slot 0,
        /// plain read pointer).
        off: Option<VarId>,
    },
    /// `noc_async_write_barrier()`.
    NocWriteBarrier,
    // -- DST state: operand-free lock effects --
    /// `tile_regs_acquire()`.
    MathLock,
    /// `tile_regs_commit()`.
    MathUnlock,
    /// `tile_regs_wait()`.
    PackLock,
    /// `tile_regs_release()`.
    PackUnlock,
    // -- Engine config: init/reconfig calls, one variant per family --
    /// `copy_tile_init(cb);`.
    CopyInit {
        /// Input circular buffer.
        cb: CBId,
    },
    /// `copy_tile_to_dst_init_short_with_dt(prev, cb);` (reprograms
    /// UNPACK+MATH SRCA to `cb`'s format).
    CopyInitWithDt {
        /// Previously programmed CB (reconfig guard source).
        prev: CBId,
        /// Input circular buffer.
        cb: CBId,
    },
    /// `pack_reconfig_data_format(cb);`.
    PackReconfig {
        /// Output circular buffer.
        cb: CBId,
    },
    /// Tile unary init (`exp_tile_init();`, ...).
    UnaryInit {
        /// Operation (selects the init call).
        uop: UOp,
    },
    /// Tile binary init (`add_binary_tile_init();`, ...).
    BinaryInit {
        /// Operation (selects the init call).
        bop: BOp,
    },
    /// Scalar-binary init: `binop_with_scalar_tile_init();` for
    /// arithmetic, `left/right_shift_tile_init();` for shifts
    /// (tile-scalar binary).
    BinScalarInit {
        /// Operation (selects the init call).
        bop: BOp,
    },
    /// `typecast_tile_init<in, out>();`.
    CastInit {
        /// Source dtype.
        in_dtype: DType,
        /// Target dtype.
        out_dtype: DType,
    },
    /// `transpose_wh_init(cb, out);`.
    TransposeInit {
        /// Input circular buffer.
        cb: CBId,
        /// Output circular buffer (startup triple).
        out: CBId,
    },
    /// `mm_init(a, b, out);` (full init before every matmul; the
    /// hoist/dedup pass later folds these).
    MatmulInit {
        /// Left input circular buffer.
        a: CBId,
        /// Right input circular buffer.
        b: CBId,
        /// Output circular buffer (startup triple).
        out: CBId,
    },
    /// `compute_kernel_hw_startup(in0, in1, out);` (compute front,
    /// ahead of all loops; single-input kernels repeat in0). Emitted
    /// by `verify` when compute both loads and packs (pure movement
    /// needs no startup); matmul kernels carry none (`mm_init` owns
    /// the long init). Render emits it verbatim.
    ComputeStartup {
        /// First-loaded circular buffer (second load, or in0).
        in0: CBId,
        /// Second-loaded circular buffer.
        in1: CBId,
        /// First-packed circular buffer.
        out: CBId,
    },
    /// `reduce_init<op, dim>(ci, cs, acc);` (inline at its op;
    /// `init_math` places it, carrying the op's acc slot).
    ReduceInit {
        /// Input circular buffer.
        ci: CBId,
        /// Scaler circular buffer.
        cs: CBId,
        /// Accumulator DST slot (also the result tile).
        acc: TileId,
        /// Reduction op.
        rop: BOp,
        /// In-tile dimension.
        kind: TileDim,
    },
    /// `reduce_uninit();` — closes the reduce cone opened by
    /// [`TTOp::ReduceInit`]; emitted before the pack that drains the
    /// reduce result (the legacy `reduce_pending` rule).
    ReduceUninit,
    /// Fused broadcast binary init (`add_bcast_cols_init_short`, ...).
    BcastInit {
        /// Operation.
        bop: BOp,
        /// Broadcast dimension.
        kind: TileDim,
        /// Full-tile circular buffer.
        cb_a: CBId,
        /// Broadcast-lane circular buffer.
        cb_b: CBId,
    },
    // -- Tile DST: physical tile registers, no passes over these --
    /// Streaming copy in: `copy_tile(cb, index, slot)` (runs under MATH).
    TileCopy {
        /// Destination DST slot.
        slot: TileId,
        /// Source circular buffer.
        cb: CBId,
        /// Tile slot within the waited block (scalar register).
        index: VarId,
    },
    /// Pack out: `pack_tile(slot, cb)` (runs under PACK).
    TilePack {
        /// Source DST slot.
        slot: TileId,
        /// Destination circular buffer.
        cb: CBId,
    },
    /// Tiled binary ALU: `op(x, y, dst)` (inputs stay live).
    TileBinary {
        /// Destination DST slot.
        dst: TileId,
        /// Left operand slot.
        x: TileId,
        /// Right operand slot.
        y: TileId,
        /// Operation.
        bop: BOp,
        /// Tile dtype (selects the `DataFormat` template arg of the
        /// templated shift LLKs; untemplated ops ignore it).
        dtype: DType,
    },
    /// Fused broadcast binary (`add_tiles_bcast_rows`, ...): the full
    /// tile stays in `cb_a`, the broadcast lane in `cb_b`, the result
    /// lands in a fresh DST slot.
    TileBcastBinary {
        /// Destination DST slot.
        dst: TileId,
        /// Full-tile circular buffer.
        cb_a: CBId,
        /// Broadcast-lane circular buffer.
        cb_b: CBId,
        /// Operation.
        bop: BOp,
        /// Broadcast dimension.
        kind: TileDim,
        /// First-packed compute CB (init pass placeholder: `None` from
        /// lowering, filled when the init pass places `BcastInit`).
        out: Option<CBId>,
    },
    /// Tile-scalar binary (`add_unary_tile`, ... with a scalar
    /// immediate): DST-inplace like unary, no CB traffic for the
    /// scalar side. Arithmetic lowers the const to fp32 bits; shifts
    /// carry the bit count as `Constant::U32`.
    TileBinScalar {
        /// Operand and result slot.
        slot: TileId,
        /// Operation (selects the `*_unary_tile` call).
        bop: BOp,
        /// Scalar side constant (render lowers to fp32 bits).
        value: Constant,
    },
    /// Tiled unary ALU: in-place `op(slot)` (SFPU mutates the slot).
    TileUnary {
        /// Operand and result slot.
        slot: TileId,
        /// Operation.
        uop: UOp,
    },
    /// Fused composite call (`sigmoid_tile` / `silu_tile`): transforms
    /// the input slot in place, like [`TTOp::TileUnary`]. Matched over
    /// the kernel IR before lowering; subsumed ops never reach the
    /// stream.
    TileFused {
        /// Operand and result slot.
        slot: TileId,
        /// Which composite.
        kind: FusedKind,
    },
    /// Fused composite init (`sigmoid_tile_init();`, ...).
    FusedInit {
        /// Which composite.
        kind: FusedKind,
    },
    /// Tiled cast: `typecast_tile<in, out>(slot)` (in-place like unary;
    /// Tenstorrent has no DST->DST copy).
    TileCast {
        /// Operand and result slot.
        slot: TileId,
        /// Source dtype.
        in_dtype: DType,
        /// Target dtype.
        out_dtype: DType,
    },
    /// Streaming transpose: `transpose_wh_tile(cb, 0, dst)` (the input
    /// streams from its CB, fused drain, never copied to DST first).
    TileTranspose {
        /// Destination DST slot.
        dst: TileId,
        /// Source circular buffer.
        cb: CBId,
        /// First-packed compute CB (init pass placeholder: `None` from
        /// lowering, filled when the init pass places `TransposeInit`).
        out: Option<CBId>,
    },
    /// Fused matmul: `matmul_tiles(cb_a, cb_b, acc, acc, acc)`
    /// (inputs stay in CBs, accumulate into the acc slot).
    TileMatmul {
        /// Accumulator DST slot (also the result).
        acc: TileId,
        /// Left input circular buffer.
        cb_a: CBId,
        /// Right input circular buffer.
        cb_b: CBId,
        /// Left source `Op::Load` op: sharing-group identity (same
        /// load feeding several matmuls = one tile, see `WaitGroup`).
        x_load: OpId,
        /// Right source `Op::Load` op: sharing-group identity.
        y_load: OpId,
        /// First-packed compute CB (init pass placeholder: `None` from
        /// lowering, filled when the init pass places `MatmulInit`).
        out: Option<CBId>,
    },
    /// Fused reduce: `reduce_tile<op, dim>(cb_in, cb_sc, 0, 0, acc)`.
    TileReduce {
        /// Accumulator DST slot (also the result).
        acc: TileId,
        /// Input circular buffer.
        cb_in: CBId,
        /// Scaler circular buffer.
        cb_sc: CBId,
        /// Reduction op.
        rop: BOp,
        /// In-tile dimension.
        kind: TileDim,
    },
}

/// Hoisted init call for a tile unary op. Duplicated from the legacy
/// codegen (`tenstorrent.rs`): one table per IR, never shared.
fn unary_init_name(uop: UOp) -> &'static str {
    match uop {
        UOp::Neg => "negative_tile_init();",
        UOp::BitNot => "bitwise_not_tile_init();",
        UOp::Exp => "exp_tile_init();",
        UOp::Exp2 => "exp2_tile_init();",
        UOp::Log2 => "log_with_base_tile_init();",
        UOp::Reciprocal => "recip_tile_init();",
        UOp::Sqrt => "sqrt_tile_init();",
        UOp::Rsqrt => "rsqrt_tile_init();",
        UOp::Sin => "sin_tile_init();",
        UOp::Cos => "cos_tile_init();",
        UOp::Floor | UOp::Trunc => "rounding_op_tile_init();",
        UOp::Abs => "abs_tile_init();",
        UOp::Not => "logical_not_tile_init();",
    }
}

/// Hoisted init call for a tile binary op, if it needs one.
fn binary_init_name(bop: BOp) -> Option<&'static str> {
    match bop {
        BOp::Add => Some("add_binary_tile_init();"),
        BOp::Sub => Some("sub_binary_tile_init();"),
        BOp::Mul => Some("mul_binary_tile_init();"),
        BOp::Div => Some("div_binary_tile_init();"),
        BOp::Max => Some("binary_max_tile_init();"),
        BOp::BitShiftLeft | BOp::BitShiftRight => Some("binary_shift_tile_init();"),
        _ => None,
    }
}

/// Hoisted init call for a fused broadcast binary. `None` means the
/// (op, kind) pair has no LLK (notably every `Div`).
fn bcast_init_name(bop: BOp, kind: TileDim) -> Option<&'static str> {
    match (bop, kind) {
        (BOp::Add, TileDim::Row) => Some("add_bcast_rows_init_short"),
        (BOp::Add, TileDim::Col) => Some("add_bcast_cols_init_short"),
        (BOp::Add, TileDim::Scalar) => Some("add_bcast_scalar_init_short"),
        (BOp::Sub, TileDim::Row) => Some("sub_bcast_rows_init_short"),
        (BOp::Sub, TileDim::Col) => Some("sub_bcast_cols_init_short"),
        (BOp::Sub, TileDim::Scalar) => Some("sub_tiles_bcast_scalar_init_short"),
        (BOp::Mul, TileDim::Row) => Some("mul_bcast_rows_init_short"),
        (BOp::Mul, TileDim::Col) => Some("mul_bcast_cols_init_short"),
        (BOp::Mul, TileDim::Scalar) => Some("mul_tiles_bcast_scalar_init_short"),
        _ => None,
    }
}

/// LLK reduce dimension for a tile reduce kind.
fn reduce_dim_name(kind: TileDim) -> &'static str {
    match kind {
        TileDim::Row => "ReduceDim::REDUCE_ROW",
        TileDim::Col => "ReduceDim::REDUCE_COL",
        TileDim::Scalar => "ReduceDim::REDUCE_SCALAR",
    }
}

/// TT `DataFormat` code for a dtype on the tile path (the
/// `typecast_tile_init<in, out>` template args). This is NOT the CB
/// descriptor code (`cb_fmt` below): the two numberings differ.
fn tt_fmt(dtype: DType) -> Result<u32, BackendError> {
    match dtype {
        DType::F32 => Ok(0),
        DType::F16 | DType::BF16 => Ok(5),
        DType::I32 => Ok(8),
        DType::U16 => Ok(9),
        DType::I8 => Ok(14),
        DType::U32 => Ok(24),
        DType::F8E4M3 => Ok(26),
        DType::U8 => Ok(30),
        dt => Err(BackendError {
            status: ErrorStatus::KernelCompilation,
            context: format!("tenstorrent2: dtype {dt:?} has no tt tile format").into(),
        }),
    }
}

/// Runtime CB descriptor format code for a CB storage dtype
/// (F32=0, F16=1, BF16=2, ...). Follows the CB storage dtype;
/// an unmappable dtype is a compilation error, never a default.
fn cb_fmt(dtype: DType) -> Result<u32, BackendError> {
    match dtype {
        DType::F32 => Ok(0),
        DType::F16 => Ok(1),
        DType::BF16 => Ok(2),
        DType::U16 => Ok(3),
        DType::F8E4M3 => Ok(4),
        DType::U8 => Ok(5),
        DType::I8 => Ok(6),
        DType::U32 => Ok(7),
        DType::I32 => Ok(8),
        dt => Err(BackendError {
            status: ErrorStatus::KernelCompilation,
            context: format!("tenstorrent2: CB dtype {dt:?} has no tt format").into(),
        }),
    }
}

/// Lowered TTIR: one physical op stream with section boundaries.
/// Built once by [`Compiler::new`]; every later pass rewrites only this vector.
struct Compiler {
    ops: Vec<TTOp>,
    /// Startup-triple load order: compute-section `Op::Load` (Tile
    /// layout, CB-mapped) first-touch order over the section's op
    /// list — the same IR-order scan as the legacy generator. The
    /// stream-emission order differs (tile ops skip broadcast-fed
    /// loads), so the triple must NOT be derived from it.
    startup_loads: Vec<CBId>,
    /// Startup-triple out: first compute-section `Op::Store` (Tile
    /// layout) targeting a CB.
    startup_store: Option<CBId>,
    /// Compute-side use counts per `(CB, source-load)` group: how many
    /// `TileMatmul` inputs consume the tile. Bumped at emission; read
    /// by the pop-placement pass for last-use pops. All other compute
    /// reads pop strictly after their op.
    use_counts: Map<(CBId, OpId), u32>,
}

impl Compiler {
    /// Lowering: one walk per section op list from `get_needed_ops`
    /// (already replicated per section: each section is a separate
    /// kernel with its own registers, arg ordinals stay global).
    /// Scalar values bind `VarId`s, tiled values bind `TileId`s,
    /// circular storages bind one global `CBId`; a use count reaching
    /// zero frees its slot for reuse (shared slots stay live while
    /// any bound id is). Single pass, nothing else: fusion, sync,
    /// locks, and inits are later passes over the ops.
    fn new(kernel: &Kernel) -> Result<Self, BackendError> {
        // Section gate, same rule as the legacy `check_sections`:
        // exactly 2 barriers delimiting reader/compute/writer, and no
        // GPU-only Wmma (tenstorrent has `MatmulTile`, no WMMA units).
        let mut barriers = 0u32;
        let mut scan = kernel.head;
        for _ in 0..10_000 {
            if scan.is_null() {
                break;
            }
            match &kernel.ops[scan].op {
                Op::Barrier => barriers += 1,
                Op::Wmma { .. } => {
                    return Err(BackendError {
                        status: ErrorStatus::KernelCompilation,
                        context: "tenstorrent2: Wmma is GPU-only, tenstorrent uses Op::MatmulTile".into(),
                    });
                }
                _ => {}
            }
            scan = kernel.next_op(scan);
        }
        if barriers != 2 {
            return Err(BackendError {
                status: ErrorStatus::KernelCompilation,
                context: format!("tenstorrent2: need exactly 2 barriers (3 sections), found {barriers}").into(),
            });
        }
        // DST mode, same scan as the legacy `generate_tenstorrent`:
        // 32-bit iff the kernel touches F32 tiles (F32 storage, or an
        // F8 circular sharing the core per the Blackhole mandate).
        let mut dst_bf16 = true;
        let mut scan = kernel.head;
        for _ in 0..10_000 {
            if scan.is_null() {
                break;
            }
            if let Op::Storage { dtype, scope, .. } = kernel.ops[scan].op {
                match (dtype, scope) {
                    (DType::F32, _) | (DType::F8E4M3, MemScope::Circular) => {
                        dst_bf16 = false;
                        break;
                    }
                    _ => {}
                }
            }
            scan = kernel.next_op(scan);
        }
        let budget = if dst_bf16 { TileId::BUDGET_BF16 } else { TileId::BUDGET_FP32 };
        let param_ordinal_of = NocEmitter::new(kernel)?.param_ordinal_of;
        // CB allocation: first-touch order over loads/stores of
        // Circular storages, with the legacy validity checks (whole
        // 1024-element tiles, whole pages, single-core L1 budget,
        // hardware CB count fit).
        let mut cbs: Map<OpId, CBId> = Map::default();
        let mut cb_order: Vec<OpId> = Vec::new();
        let mut scan = kernel.head;
        for _ in 0..10_000 {
            if scan.is_null() {
                break;
            }
            let storage = match &kernel.ops[scan].op {
                Op::Load { src, .. } => Some(*src),
                Op::Store { dst, .. } => Some(*dst),
                _ => None,
            };
            if let Some(st) = storage
                && matches!(kernel.ops[st].op, Op::Storage { scope: MemScope::Circular, .. })
                && !cbs.contains_key(&st)
            {
                cbs.insert(st, CBId(cb_order.len() as u32));
                cb_order.push(st);
            }
            scan = kernel.next_op(scan);
        }
        let num_circular_buffers = kernel.device_info().num_circular_buffers;
        if cb_order.len() > num_circular_buffers as usize {
            return Err(BackendError {
                status: ErrorStatus::TooManyCircularBuffers,
                context: format!(
                    "tenstorrent2: kernel needs {} circular buffers, device holds {num_circular_buffers}",
                    cb_order.len()
                )
                .into(),
            });
        }
        let mut ops = vec![TTOp::DstMode { bf16: dst_bf16 }];
        let mut use_counts: Map<(CBId, OpId), u32> = Map::default();
        for (cb, &st) in cb_order.iter().enumerate() {
            let Op::Storage { dtype, len, .. } = kernel.ops[st].op else {
                return Err(BackendError {
                    status: ErrorStatus::KernelCompilation,
                    context: format!("tenstorrent2: CB map entry {st} is not a storage op").into(),
                });
            };
            let elem = dtype.bit_size() as i64 / 8;
            let bytes = len * elem;
            if len % 1024 != 0 {
                return Err(BackendError {
                    status: ErrorStatus::KernelCompilation,
                    context: format!("tenstorrent2: CB{cb} holds {len} elements, not whole 1024-element tiles").into(),
                });
            }
            let page = 1024 * elem;
            if bytes % page != 0 {
                return Err(BackendError {
                    status: ErrorStatus::KernelCompilation,
                    context: format!("tenstorrent2: CB{cb} holds {bytes} bytes, not whole {page}B pages").into(),
                });
            }
            if bytes > 32768 {
                return Err(BackendError {
                    status: ErrorStatus::KernelCompilation,
                    context: format!("tenstorrent2: CB{cb} needs {bytes} bytes, single-core L1 budget is 32768").into(),
                });
            }
            ops.push(TTOp::CbDeclare { cb: CBId(cb as u32), n_tiles: (len / 1024) as u32, format: cb_fmt(dtype)? });
        }
        // A side resolving to a compile-time float constant (follows
        // const expressions): folds into a `*_unary_tile` immediate.
        // Integer constants are NOT converted (a tile op's scalar lane
        // is float; silent int→float would hide dtype bugs).
        let const_scalar = |op: OpId| -> Option<Constant> {
            let c = kernel.resolve_const(op)?;
            match c {
                Constant::F32(_) | Constant::F16(_) | Constant::BF16(_) => Some(c),
                _ => None,
            }
        };
        // A side resolving to a compile-time integer constant (follows
        // const expressions): folds into a shift immediate. Float and
        // bool consts are NOT amounts (`None`, like non-consts).
        let const_shift_amt = |op: OpId| -> Option<u32> {
            let v: i128 = match kernel.resolve_const(op)? {
                Constant::U8(v) => v as i128,
                Constant::U16(v) => v as i128,
                Constant::U32(v) => v as i128,
                Constant::U64(v) => u64::from_le_bytes(v) as i128,
                Constant::I8(v) => v as i128,
                Constant::I16(v) => v as i128,
                Constant::I32(v) => v as i128,
                Constant::I64(v) => i64::from_le_bytes(v) as i128,
                Constant::BF16(_)
                | Constant::F16(_)
                | Constant::F32(_)
                | Constant::F64(_)
                | Constant::F8E4M3(_)
                | Constant::F8E5M2(_)
                | Constant::Bool(_) => return None,
            };
            u32::try_from(v).ok()
        };
        let sections = [TtSection::Reader, TtSection::Compute, TtSection::Writer];
        let mut startup_loads: Vec<CBId> = Vec::new();
        let mut startup_store: Option<CBId> = None;
        for (s, tt_section) in sections.into_iter().enumerate() {
            let data = kernel.get_needed_ops(tt_section)?;
            let total = data.rcs.clone();
            let mut remaining = data.rcs.clone();
            let mut vars: Map<OpId, VarId> = Map::default();
            let mut tiles: Map<OpId, TileId> = Map::default();
            let mut free_vars: Vec<(VarId, DType, u8)> = Vec::new();
            let mut free_tiles: Vec<TileId> = Vec::new();
            let mut next_var = 0u32;
            let mut next_tile = 0u8;
            // Loop nesting level and per-var (dtype, def level), PTX
            // style: refcounts tick only on same-level uses so
            // loop-invariant temps survive loops, and a freed slot is
            // reused only with matching dtype at same-or-outer level.
            let mut loop_level: u8 = 0;
            let mut var_info: Map<VarId, (DType, u8)> = Map::default();
            // Section params in IR order: this section kernel's runtime args.
            let section_params: Vec<OpId> =
                data.ops.iter().copied().filter(|op| matches!(kernel.ops[*op].op, Op::Param { .. })).collect();
            // Consumers per op within this section (parameter edges).
            let mut consumers: Map<OpId, Vec<OpId>> = Map::default();
            for &cid in &data.ops {
                for p in kernel.ops[cid].op.parameters() {
                    consumers.entry(p).or_default().push(cid);
                }
            }
            // Fused-LLK prepass (compute only, kernel immutable): match
            // composites over the section list, then drop the subsumed
            // inners from that list only. Overlapping patterns share ops,
            // so a match whose ops are already claimed loses (its
            // composite still emits — reading the accepted match's slot —
            // slower, never wrong). Mirrors the legacy `Compiler::generate`
            // prepass; the matcher lives in the legacy module.
            let mut fused: Map<OpId, FusedPat> = Map::default();
            let mut fused_gone: Set<OpId> = Set::default();
            if s == 1 {
                let mut claimed: Set<OpId> = Set::default();
                for &op in &data.ops {
                    if let Some(pat) = FusedKind::match_pat(kernel, &data, &consumers, op) {
                        let touched: Vec<OpId> =
                            std::iter::once(op).chain(std::iter::once(pat.x)).chain(pat.inners.iter().copied()).collect();
                        if touched.iter().all(|o| !claimed.contains(o)) {
                            claimed.extend(touched);
                            fused_gone.extend(pat.inners.iter().copied());
                            fused.insert(op, pat);
                        }
                    }
                }
            }
            // True if every consumer of `load` drains it from the CB
            // itself (fused tile ops, or a binary with a
            // broadcast-marked side): the load emits no `TileCopy`
            // and carries no sync — the consuming op waits/ops/pops.
            // Any other consumer needs the tile in DST first. Mirrors
            // the legacy `fused_only_load` rule.
            let fused_only = |consumers: &Map<OpId, Vec<OpId>>, load: OpId| -> bool {
                match consumers.get(&load) {
                    None => false,
                    Some(cs) => cs.iter().all(|&c| match kernel.ops[c].op {
                        Op::ReduceTile { .. } | Op::MatmulTile { .. } | Op::TransposeTile { .. } | Op::BroadcastTile { .. } => {
                            true
                        }
                        Op::Binary { x, y, .. } => {
                            matches!(kernel.ops[x].op, Op::BroadcastTile { .. })
                                || matches!(kernel.ops[y].op, Op::BroadcastTile { .. })
                        }
                        _ => false,
                    }),
                }
            };
            // Bind a scalar register to a value, reusing a freed slot
            // only on dtype match at same-or-outer level (PTX rule:
            // inner-loop temps must not leak outward-allocated names
            // that render still considers live). Records (dtype, def
            // level) for the use-side gate below.
            let def_var = |vars: &mut Map<OpId, VarId>,
                           free_vars: &mut Vec<(VarId, DType, u8)>,
                           next_var: &mut u32,
                           var_info: &mut Map<VarId, (DType, u8)>,
                           kernel: &Kernel,
                           level: u8,
                           id: OpId|
             -> VarId {
                let dtype = kernel.dtype(id);
                let v = free_vars
                    .iter()
                    .position(|(_, dt, lv)| *dt == dtype && level <= *lv)
                    .map(|i| free_vars.swap_remove(i).0)
                    .unwrap_or_else(|| {
                        let v = VarId(*next_var);
                        *next_var += 1;
                        v
                    });
                vars.insert(id, v);
                var_info.insert(v, (dtype, level));
                v
            };
            // Consume one use of a scalar value, freeing its register at
            // zero. Only same-level uses tick the count (PTX rule):
            // deeper uses repeat across trips and must not consume the
            // value out from under later trips.
            let use_var = |vars: &Map<OpId, VarId>,
                           remaining: &mut Map<OpId, u32>,
                           free_vars: &mut Vec<(VarId, DType, u8)>,
                           var_info: &Map<VarId, (DType, u8)>,
                           level: u8,
                           id: OpId|
             -> Result<VarId, BackendError> {
                let &v = vars.get(&id).ok_or_else(|| BackendError {
                    status: ErrorStatus::KernelCompilation,
                    context: format!("tenstorrent2: scalar op {id} has no register").into(),
                })?;
                let &(dtype, def_level) = var_info.get(&v).ok_or_else(|| BackendError {
                    status: ErrorStatus::KernelCompilation,
                    context: format!("tenstorrent2: scalar op {id} has no level info").into(),
                })?;
                if level != def_level {
                    return Ok(v);
                }
                let left = remaining.get_mut(&id).ok_or_else(|| BackendError {
                    status: ErrorStatus::KernelCompilation,
                    context: format!("tenstorrent2: scalar op {id} has no use count").into(),
                })?;
                if !(*left > 0) {
                    return Err(BackendError {
                        status: ErrorStatus::KernelCompilation,
                        context: format!("tenstorrent2: scalar op {id} used past its uses").into(),
                    });
                }
                *left -= 1;
                if *left == 0 {
                    free_vars.push((v, dtype, def_level));
                }
                Ok(v)
            };
            // Bind the lowest dead DST slot (or a fresh one) to a tiled
            // value. Lowest-first matches the legacy slab scan, so slot
            // assignment agrees with legacy text. The use budget comes
            // from `remaining` (the section's consumer counts).
            let def_tile = |tiles: &mut Map<OpId, TileId>,
                            free_tiles: &mut Vec<TileId>,
                            next_tile: &mut u8,
                            id: OpId|
             -> Result<TileId, BackendError> {
                let t = if free_tiles.is_empty() {
                    let t = TileId(*next_tile);
                    *next_tile += 1;
                    t
                } else {
                    let pos =
                        free_tiles.iter().enumerate().min_by_key(|(_, t)| t.0).map(|(i, _)| i).ok_or_else(|| BackendError {
                            status: ErrorStatus::KernelCompilation,
                            context: "tenstorrent2: free tile list went missing".into(),
                        })?;
                    free_tiles.remove(pos)
                };
                if !((t.0 as usize) < budget) {
                    return Err(BackendError {
                        status: ErrorStatus::KernelCompilation,
                        context: "tenstorrent2: DST budget exceeded".into(),
                    });
                }
                tiles.insert(id, t);
                Ok(t)
            };
            // Consume one use of a tiled value, freeing its DST slot at
            // zero. Def-before-uses at each op (like the legacy
            // alloc-then-use order) so a dead operand slot is reused by
            // the result. In-place chains (unary/cast/bitcast/binscalar,
            // acc aliases) transfer ownership to the result id instead:
            // the operand count stays stale-harmless, the slot frees
            // once through the result id.
            let use_tile = |tiles: &Map<OpId, TileId>,
                            remaining: &mut Map<OpId, u32>,
                            free_tiles: &mut Vec<TileId>,
                            id: OpId|
             -> Result<(), BackendError> {
                let &t = tiles.get(&id).ok_or_else(|| BackendError {
                    status: ErrorStatus::KernelCompilation,
                    context: format!("tenstorrent2: tile op {id} has no DST slot").into(),
                })?;
                let left = remaining.get_mut(&id).ok_or_else(|| BackendError {
                    status: ErrorStatus::KernelCompilation,
                    context: format!("tenstorrent2: tile op {id} has no use count").into(),
                })?;
                if !(*left > 0) {
                    return Err(BackendError {
                        status: ErrorStatus::KernelCompilation,
                        context: format!("tenstorrent2: tile op {id} used past its uses").into(),
                    });
                }
                *left -= 1;
                if *left == 0 {
                    free_tiles.push(t);
                }
                Ok(())
            };
            for &id in &data.ops {
                if fused_gone.contains(&id) {
                    continue;
                }
                // Fused composite root: one in-place LLK call that
                // transforms the external input's slot (ownership
                // transfers to the root id, like every in-place chain).
                if let Some(pat) = fused.get(&id) {
                    let slot = tiles.get(&pat.x).copied().ok_or_else(|| BackendError {
                        status: ErrorStatus::KernelCompilation,
                        context: "tenstorrent2: fused op reads a value with no DST slot".into(),
                    })?;
                    tiles.insert(id, slot);
                    ops.push(TTOp::TileFused { slot, kind: pat.kind });
                    continue;
                }
                match &kernel.ops[id].op {
                    Op::Const(c) => {
                        let z = def_var(&mut vars, &mut free_vars, &mut next_var, &mut var_info, kernel, loop_level, id);
                        ops.push(TTOp::Const { z, value: c.clone() });
                    }
                    Op::Param { dtype, kind, .. } => match kind {
                        ParamKind::Variable => {
                            let z = def_var(&mut vars, &mut free_vars, &mut next_var, &mut var_info, kernel, loop_level, id);
                            let ordinal =
                                param_ordinal_of.get(&id).copied().ok_or_else(|| BackendError {
                                    status: ErrorStatus::KernelCompilation,
                                    context: "tenstorrent2: variable param missing ordinal".into(),
                                })?;
                            ops.push(TTOp::Arg { z, dtype: *dtype, ordinal });
                        }
                        ParamKind::Global | ParamKind::GlobalMut => {
                            if s == 1 {
                                return Err(BackendError {
                                    status: ErrorStatus::KernelCompilation,
                                    context: format!("tenstorrent2: compute touches DRAM through param {id}").into(),
                                });
                            }
                            let ordinal =
                                param_ordinal_of.get(&id).copied().ok_or_else(|| BackendError {
                                    status: ErrorStatus::KernelCompilation,
                                    context: "tenstorrent2: DRAM param missing ordinal".into(),
                                })?;
                            ops.push(TTOp::NocAccessor { ordinal, dtype: *dtype, kind: *kind });
                        }
                    },
                    Op::Storage { scope, .. } => match scope {
                        MemScope::Circular => {}
                        MemScope::Register => {
                            def_tile(&mut tiles, &mut free_tiles, &mut next_tile, id)?;
                        }
                        MemScope::Local => {
                            return Err(BackendError {
                                status: ErrorStatus::KernelCompilation,
                                context: "tenstorrent does not have local threads; local indices should have been converted to loops by the opt_tenstorrent_tile optimization pass".into(),
                            })
                        }
                        MemScope::Global => {
                            return Err(BackendError {
                                status: ErrorStatus::KernelCompilation,
                                context: format!("tenstorrent2 storage scope, op {id}").into(),
                            })
                        }
                    },
                    Op::Load { src, index, layout } => {
                        if !matches!(layout, MemLayout::Tile { .. }) {
                            if s == 1 {
                                return Err(BackendError {
                                    status: ErrorStatus::KernelCompilation,
                                    context: "tenstorrent2 compute only supports tile loads".into(),
                                });
                            }
                            continue;
                        }
                        if s != 1 {
                            continue;
                        }
                        if matches!(kernel.ops[*src].op, Op::Storage { scope: MemScope::Register, .. }) {
                            let tile = tiles.get(src).copied().ok_or_else(|| BackendError {
                                status: ErrorStatus::KernelCompilation,
                                context: "tenstorrent2: compute acc load reads an undeclared acc".into(),
                            })?;
                            tiles.insert(id, tile);
                            continue;
                        }
                        let Some(&cb) = cbs.get(src) else {
                            return Err(BackendError {
                                status: ErrorStatus::KernelCompilation,
                                context: format!("tenstorrent2: compute load targets unmapped CB, op {id}").into(),
                            });
                        };
                        if fused_only(&consumers, id) {
                            continue;
                        }
                        let slot = def_tile(&mut tiles, &mut free_tiles, &mut next_tile, id)?;
                        let index = use_var(&vars, &mut remaining, &mut free_vars, &var_info, loop_level, *index)?;
                        ops.push(TTOp::TileCopy { slot, cb, index });
                    }
                    Op::Store { dst, src, index, layout } => {
                        if !matches!(layout, MemLayout::Tile { .. }) {
                            return Err(BackendError {
                                status: ErrorStatus::KernelCompilation,
                                context: format!("tenstorrent2 only supports tile stores, op {id}").into(),
                            });
                        }
                        if s == 0 {
                            let Op::Load { src: ld_src, index: ld_idx, layout: ld_layout } = kernel.ops[*src].op else {
                                return Err(BackendError {
                                    status: ErrorStatus::KernelCompilation,
                                    context: format!(
                                        "tenstorrent2: reader supports only global to local stores, op {id} has ops in between"
                                    )
                                    .into(),
                                })
                            };
                            let Op::Param { kind: ParamKind::Global, .. } = kernel.ops[ld_src].op else {
                                return Err(BackendError {
                                    status: ErrorStatus::KernelCompilation,
                                    context: format!("tenstorrent2: reader load op {id} is not from a Global param").into(),
                                })
                            };
                            let Op::Storage { dtype, scope: MemScope::Circular, .. } = kernel.ops[*dst].op else {
                                return Err(BackendError {
                                    status: ErrorStatus::KernelCompilation,
                                    context: format!("tenstorrent2: reader store op {id} does not target a Circular CB")
                                        .into(),
                                })
                            };
                            let Some(&cb) = cbs.get(dst) else {
                                return Err(BackendError {
                                    status: ErrorStatus::KernelCompilation,
                                    context: format!("tenstorrent2: reader store op {id} targets unmapped CB").into(),
                                })
                            };
                            let MemLayout::Tile { x, y, .. } = ld_layout else {
                                return Err(BackendError {
                                    status: ErrorStatus::KernelCompilation,
                                    context: "tenstorrent2 reader only supports tile stores".into(),
                                })
                            };
                            let elem_size = dtype.bit_size() as u32 / 8;
                            let index = use_var(&vars, &mut remaining, &mut free_vars, &var_info, loop_level, ld_idx)?;
                            ops.push(TTOp::ReadTile {
                                ordinal: param_ordinal_of[&ld_src],
                                dtype,
                                index,
                                cb,
                                bytes: x as u32 * y as u32 * elem_size,
                                elem_size,
                            });
                            continue;
                        }
                        if s == 2 {
                            let Op::Load { src: cb_src, index: _, layout: ld_layout } = kernel.ops[*src].op else {
                                return Err(BackendError {
                                    status: ErrorStatus::KernelCompilation,
                                    context: format!(
                                        "tenstorrent2: writer supports only CB to DRAM stores, op {id} has ops in between"
                                    )
                                    .into(),
                                })
                            };
                            let Some(&cb) = cbs.get(&cb_src) else {
                                return Err(BackendError {
                                    status: ErrorStatus::KernelCompilation,
                                    context: format!("tenstorrent2: writer load op {id} targets unmapped CB").into(),
                                })
                            };
                            let Op::Param { dtype, kind: ParamKind::GlobalMut, .. } = kernel.ops[*dst].op else {
                                return Err(BackendError {
                                    status: ErrorStatus::KernelCompilation,
                                    context: format!("tenstorrent2: writer store dst must be a GlobalMut param, op {id}")
                                        .into(),
                                })
                            };
                            let MemLayout::Tile { x, y, .. } = ld_layout else {
                                return Err(BackendError {
                                    status: ErrorStatus::KernelCompilation,
                                    context: "tenstorrent2 writer only supports tile stores".into(),
                                })
                            };
                            let elem_size = dtype.bit_size() as u32 / 8;
                            let index = use_var(&vars, &mut remaining, &mut free_vars, &var_info, loop_level, *index)?;
                            ops.push(TTOp::WriteTile {
                                cb,
                                ordinal: param_ordinal_of[dst],
                                dtype,
                                index,
                                bytes: x as u32 * y as u32 * elem_size,
                                elem_size,
                            });
                            continue;
                        }
                        if let Op::Storage { scope: MemScope::Register, .. } = kernel.ops[*dst].op {
                            if let Some(&tile) = tiles.get(src) {
                                tiles.insert(*dst, tile);
                            } else if let Op::Load { src: lsrc, .. } = kernel.ops[*src].op
                                && let Some(&tile) = tiles.get(&lsrc)
                            {
                                tiles.insert(*dst, tile);
                            }
                            continue;
                        }
                        let Some(&cb) = cbs.get(dst) else {
                            return Err(BackendError {
                                status: ErrorStatus::KernelCompilation,
                                context: format!("tenstorrent2: compute store op {id} targets unmapped CB").into(),
                            })
                        };
                        let slot = tiles.get(src).copied().ok_or_else(|| BackendError {
                            status: ErrorStatus::KernelCompilation,
                            context: "tenstorrent2: compute acc store reads a tile with no DST slot".into(),
                        })?;
                        ops.push(TTOp::TilePack { slot, cb });
                        use_tile(&tiles, &mut remaining, &mut free_tiles, *src)?;
                    }
                    Op::Cast { x, dtype } => {
                        if matches!(data.dtypes[&id].1, MemLayout::Tile { .. }) {
                            let slot = tiles.get(x).copied().ok_or_else(|| BackendError {
                                status: ErrorStatus::KernelCompilation,
                                context: "tenstorrent2: tiled cast reads a value with no DST slot".into(),
                            })?;
                            if total[&x] != 1 {
                                return Err(BackendError {
                                    status: ErrorStatus::KernelCompilation,
                                    context: format!("tenstorrent2 multi-use tiled cast operand, op {id}").into(),
                                });
                            }
                            let in_dtype = data.dtypes[&x].0;
                            tiles.insert(id, slot);
                            ops.push(TTOp::TileCast { slot, in_dtype, out_dtype: *dtype });
                        } else {
                            let z = def_var(&mut vars, &mut free_vars, &mut next_var, &mut var_info, kernel, loop_level, id);
                            let x = use_var(&vars, &mut remaining, &mut free_vars, &var_info, loop_level, *x)?;
                            ops.push(TTOp::Cast { z, dtype: *dtype, x });
                        }
                    }
                    Op::Bitcast { x, .. } => {
                        if matches!(data.dtypes[&id].1, MemLayout::Tile { .. }) {
                            let tile = tiles.get(x).copied().ok_or_else(|| BackendError {
                                status: ErrorStatus::KernelCompilation,
                                context: "tenstorrent2: tiled bitcast reads a value with no DST slot".into(),
                            })?;
                            tiles.insert(id, tile);
                        } else {
                            return Err(BackendError {
                                status: ErrorStatus::KernelCompilation,
                                context: format!("tenstorrent2 scalar bitcast, op {id}").into(),
                            });
                        }
                    }
                    Op::Unary { x, uop } => {
                        if matches!(data.dtypes[&id].1, MemLayout::Tile { .. }) {
                            let slot = tiles.get(x).copied().ok_or_else(|| BackendError {
                                status: ErrorStatus::KernelCompilation,
                                context: "tenstorrent2: tiled unary reads a value with no DST slot".into(),
                            })?;
                            tiles.insert(id, slot);
                            ops.push(TTOp::TileUnary { slot, uop: *uop });
                        } else {
                            let z = def_var(&mut vars, &mut free_vars, &mut next_var, &mut var_info, kernel, loop_level, id);
                            let x = use_var(&vars, &mut remaining, &mut free_vars, &var_info, loop_level, *x)?;
                            ops.push(TTOp::Unary { z, dtype: data.dtypes[&id].0, x, uop: *uop });
                        }
                    }
                    Op::Binary { x, y, bop } => {
                        if !matches!(data.dtypes[&id].1, MemLayout::Tile { .. }) {
                            let z = def_var(&mut vars, &mut free_vars, &mut next_var, &mut var_info, kernel, loop_level, id);
                            let x = use_var(&vars, &mut remaining, &mut free_vars, &var_info, loop_level, *x)?;
                            let y = use_var(&vars, &mut remaining, &mut free_vars, &var_info, loop_level, *y)?;
                            ops.push(TTOp::Binary { z, dtype: data.dtypes[&id].0, x, y, bop: *bop });
                            continue;
                        }
                        let marker = |side: OpId| match kernel.ops[side].op {
                            Op::BroadcastTile { x: mx, kind } => Some((kind, mx)),
                            _ => None,
                        };
                        let plain_cb = |side: OpId| match kernel.ops[side].op {
                            Op::Load { src: lsrc, .. } => cbs.get(&lsrc).copied(),
                            _ => None,
                        };
                        match (marker(*x), marker(*y)) {
                            (Some(_), Some(_)) => {
                                return Err(BackendError {
                                    status: ErrorStatus::KernelCompilation,
                                    context: format!("tenstorrent2: broadcast op {id} marks both sides").into(),
                                });
                            }
                            (Some((kind, mx)), None) => {
                                let (Some(cb_b), Some(cb_a)) = (plain_cb(mx), plain_cb(*y)) else {
                                    return Err(BackendError {
                                        status: ErrorStatus::KernelCompilation,
                                        context: format!("tenstorrent2: broadcast op {id} side is no CB tile load").into(),
                                    })
                                };
                                let dst = def_tile(&mut tiles, &mut free_tiles, &mut next_tile, id)?;
                                ops.push(TTOp::TileBcastBinary { dst, cb_a, cb_b, bop: *bop, kind, out: None });
                            }
                            (None, Some((kind, my))) => {
                                let (Some(cb_b), Some(cb_a)) = (plain_cb(my), plain_cb(*x)) else {
                                    return Err(BackendError {
                                        status: ErrorStatus::KernelCompilation,
                                        context: format!("tenstorrent2: broadcast op {id} side is no CB tile load").into(),
                                    })
                                };
                                let dst = def_tile(&mut tiles, &mut free_tiles, &mut next_tile, id)?;
                                ops.push(TTOp::TileBcastBinary { dst, cb_a, cb_b, bop: *bop, kind, out: None });
                            }
                            (None, None) => {
                                // Shifts lower to the shift LLKs, which are
                                // int-only (Int32/UInt32/UInt16): anything
                                // else fails here, never on the device.
                                if matches!(bop, BOp::BitShiftLeft | BOp::BitShiftRight) {
                                    let dt = data.dtypes[&id].0;
                                    match dt {
                                        DType::I32 | DType::U32 | DType::U16 => {}
                                        DType::BF16
                                        | DType::F16
                                        | DType::F32
                                        | DType::F64
                                        | DType::F8E4M3
                                        | DType::F8E5M2
                                        | DType::U8
                                        | DType::U64
                                        | DType::I8
                                        | DType::I16
                                        | DType::I64
                                        | DType::Bool => {
                                            return Err(BackendError {
                                                status: ErrorStatus::KernelCompilation,
                                                context: format!(
                                                    "tenstorrent2: tiled shift on {dt:?}, LLK supports Int32/UInt32/UInt16 only, op {id}"
                                                )
                                                .into(),
                                            })
                                        }
                                    }
                                    // A const amount folds into the
                                    // unary-immediate LLK (`tile << amount`);
                                    // const-first has no call. Amounts
                                    // outside 0..=31 are UB in every other
                                    // backend — fail loudly, never emit.
                                    if kernel.resolve_const(*x).is_some() {
                                        return Err(BackendError {
                                            status: ErrorStatus::KernelCompilation,
                                            context: format!("tenstorrent2: const-first {bop:?} has no scalar call, op {id}")
                                                .into(),
                                        });
                                    }
                                    if kernel.resolve_const(*y).is_some() {
                                        let Some(amount) = const_shift_amt(*y) else {
                                            return Err(BackendError {
                                                status: ErrorStatus::KernelCompilation,
                                                context: format!("tenstorrent2: shift amount is no integer const, op {id}")
                                                    .into(),
                                            })
                                        };
                                        if amount > 31 {
                                            return Err(BackendError {
                                                status: ErrorStatus::KernelCompilation,
                                                context: format!("tenstorrent2: shift amount {amount} outside 0..=31, op {id}")
                                                    .into(),
                                            });
                                        }
                                        if *bop == BOp::BitShiftRight && dt == DType::U32 {
                                            return Err(BackendError {
                                                status: ErrorStatus::KernelCompilation,
                                                context: format!(
                                                    "tenstorrent2: U32 right-shift by immediate is arithmetic-only, op {id}"
                                                )
                                                .into(),
                                            });
                                        }
                                        let tile_op = *x;
                                        let t = tiles.get(&tile_op).copied().ok_or_else(|| BackendError {
                                            status: ErrorStatus::KernelCompilation,
                                            context: "tenstorrent2: scalar binary reads a value with no DST slot".into(),
                                        })?;
                                        if total[&tile_op] != 1 {
                                            return Err(BackendError {
                                                status: ErrorStatus::KernelCompilation,
                                                context: format!("tenstorrent2: scalar binary {id} reads a multi-use operand")
                                                    .into(),
                                            });
                                        }
                                        tiles.insert(id, t);
                                        ops.push(TTOp::TileBinScalar { slot: t, bop: *bop, value: Constant::U32(amount) });
                                        continue;
                                    }
                                }
                                let xc = const_scalar(*x);
                                let yc = const_scalar(*y);
                                let scalar = match (xc, yc) {
                                    (None, Some(value)) => Some(("tile", value, *x)),
                                    (Some(value), None) => Some(("const", value, *y)),
                                    _ => None,
                                };
                                if let Some((side, value, tile_op)) = scalar {
                                    match (*bop, side) {
                                        (BOp::Add, _) | (BOp::Mul, _) | (BOp::Sub, _) | (BOp::Div, "tile") => {}
                                        _ => {
                                            return Err(BackendError {
                                                status: ErrorStatus::KernelCompilation,
                                                context: format!("tenstorrent2: const-first {bop:?} has no scalar call, op {id}")
                                                    .into(),
                                            });
                                        }
                                    };
                                    let t = tiles.get(&tile_op).copied().ok_or_else(|| BackendError {
                                        status: ErrorStatus::KernelCompilation,
                                        context: "tenstorrent2: scalar binary reads a value with no DST slot".into(),
                                    })?;
                                    if total[&tile_op] != 1 {
                                        return Err(BackendError {
                                            status: ErrorStatus::KernelCompilation,
                                            context: format!("tenstorrent2: scalar binary {id} reads a multi-use operand")
                                                .into(),
                                        });
                                    }
                                    tiles.insert(id, t);
                                    ops.push(TTOp::TileBinScalar { slot: t, bop: *bop, value });
                                } else {
                                    let ta = tiles.get(x).copied().ok_or_else(|| BackendError {
                                        status: ErrorStatus::KernelCompilation,
                                        context: "tenstorrent2: tiled binary reads a value with no DST slot".into(),
                                    })?;
                                    let tb = tiles.get(y).copied().ok_or_else(|| BackendError {
                                        status: ErrorStatus::KernelCompilation,
                                        context: "tenstorrent2: tiled binary reads a value with no DST slot".into(),
                                    })?;
                                    let dst = def_tile(&mut tiles, &mut free_tiles, &mut next_tile, id)?;
                                    ops.push(TTOp::TileBinary { dst, x: ta, y: tb, bop: *bop, dtype: data.dtypes[&id].0 });
                                    use_tile(&tiles, &mut remaining, &mut free_tiles, *x)?;
                                    use_tile(&tiles, &mut remaining, &mut free_tiles, *y)?;
                                }
                            }
                        }
                    }
                    Op::Mad { x, y, z } => {
                        if matches!(data.dtypes[&id].1, MemLayout::Tile { .. }) {
                            return Err(BackendError {
                                status: ErrorStatus::KernelCompilation,
                                context: format!("tenstorrent2 tiled mad, op {id}").into(),
                            });
                        }
                        let v = def_var(&mut vars, &mut free_vars, &mut next_var, &mut var_info, kernel, loop_level, id);
                        let x = use_var(&vars, &mut remaining, &mut free_vars, &var_info, loop_level, *x)?;
                        let y = use_var(&vars, &mut remaining, &mut free_vars, &var_info, loop_level, *y)?;
                        let z = use_var(&vars, &mut remaining, &mut free_vars, &var_info, loop_level, *z)?;
                        ops.push(TTOp::Mad { z: v, dtype: data.dtypes[&id].0, x, y, w: z });
                    }
                    Op::Stack { .. } => {
                        return Err(BackendError {
                            status: ErrorStatus::KernelCompilation,
                            context: format!("tenstorrent2 scalar stack, op {id}").into(),
                        })
                    }
                    Op::Index { .. } => {
                        return Err(BackendError {
                            status: ErrorStatus::KernelCompilation,
                            context: format!("tenstorrent2 scalar index, op {id}").into(),
                        })
                    }
                    Op::Range { axis, kind } => match kind {
                        RangeKind::Group(_) => {
                            let z = def_var(&mut vars, &mut free_vars, &mut next_var, &mut var_info, kernel, loop_level, id);
                            let arg = section_params.len() as u32 + axis;
                            match axis {
                                0 => ops.push(TTOp::TensixGridX { z, dtype: data.dtypes[&id].0, arg }),
                                1 => ops.push(TTOp::TensixGridY { z, dtype: data.dtypes[&id].0, arg }),
                                _ => {
                                    return Err(BackendError {
                                        status: ErrorStatus::KernelCompilation,
                                        context: format!("tenstorrent2 group range axis {axis}, op {id}").into(),
                                    })
                                }
                            }
                        }
                        RangeKind::Local(_) => {
                            return Err(BackendError {
                                status: ErrorStatus::KernelCompilation,
                                context: "tenstorrent does not have local threads; local indices should have been converted to loops by the opt_tenstorrent_tile optimization pass".into(),
                            })
                        }
                        RangeKind::Warp(_) => {
                            return Err(BackendError {
                                status: ErrorStatus::KernelCompilation,
                                context: "tenstorrent has no warps; warp ranges are gpu-only".into(),
                            })
                        }
                    },
                    Op::Loop { len } => {
                        // Def the counter BEFORE resolving the bound (see
                        // below), and never consume the bound: it stays
                        // live for the whole loop body (the header reads
                        // it every trip), so freeing it here would let a
                        // later def reuse its register while live.
                        let counter = def_var(&mut vars, &mut free_vars, &mut next_var, &mut var_info, kernel, loop_level, id);
                        // The loop header re-reads the counter every trip
                        // (compare + increment in the rendered `for`), but
                        // those uses aren't in `rcs`. Saturate the count so
                        // the counter's register is never freed and reused
                        // by a later def while still live.
                        *remaining
                            .get_mut(&id)
                            .ok_or_else(|| BackendError {
                                status: ErrorStatus::KernelCompilation,
                                context: format!("tenstorrent2: loop counter op {id} has no use count").into(),
                            })? = u32::MAX;
                        let bound = *vars.get(len).ok_or_else(|| BackendError {
                            status: ErrorStatus::KernelCompilation,
                            context: format!("tenstorrent2: loop bound op {len} has no register").into(),
                        })?;
                        let trip = match kernel.resolve_const(*len).and_then(|c| c.as_dim()) {
                            Some(d) if d >= 0 => Some(d as u32),
                            Some(d) => {
                                return Err(BackendError {
                                    status: ErrorStatus::KernelCompilation,
                                    context: format!("tenstorrent2: negative loop trip count {d}, op {id}").into(),
                                })
                            }
                            None => None,
                        };
                        ops.push(TTOp::Loop { len: bound, counter, dtype: IDX_T, trip });
                        // Loop body nests one level deeper (PTX rule):
                        // inner uses must not consume outer temps.
                        loop_level += 1;
                    }
                    Op::EndLoop => {
                        loop_level -= 1;
                        ops.push(TTOp::EndLoop);
                    }
                    Op::If { condition } => {
                        let cond = use_var(&vars, &mut remaining, &mut free_vars, &var_info, loop_level, *condition)?;
                        ops.push(TTOp::If { cond });
                    }
                    Op::EndIf => ops.push(TTOp::EndIf),
                    Op::Barrier => {
                        return Err(BackendError {
                            status: ErrorStatus::KernelCompilation,
                            context: "tenstorrent2: Barrier should've been filtered by kernel sections decomposition"
                                .into(),
                        })
                    }
                    Op::Wmma { .. } => {
                        return Err(BackendError {
                            status: ErrorStatus::KernelCompilation,
                            context: "tenstorrent2: Wmma has no Tenstorrent lowering".into(),
                        })
                    }
                    Op::Move { .. } => {
                        return Err(BackendError {
                            status: ErrorStatus::KernelCompilation,
                            context: "tenstorrent2: Move never survives linearization".into(),
                        })
                    }
                    Op::Reduce { .. } => {
                        return Err(BackendError {
                            status: ErrorStatus::KernelCompilation,
                            context: "tenstorrent2: Reduce never survives linearization".into(),
                        })
                    }
                    Op::ReduceTile { x, scaler, acc, rop, kind } => {
                        let Op::Load { src: lx, layout: MemLayout::Tile { x: wx, y: hx, .. }, .. } =
                            kernel.ops[*x].op
                        else {
                            return Err(BackendError {
                                status: ErrorStatus::KernelCompilation,
                                context: format!("tenstorrent2: reduce side op {x} is no CB tile load").into(),
                            })
                        };
                        if wx as u32 != 32 || hx as u32 != 32 {
                            return Err(BackendError {
                                status: ErrorStatus::KernelCompilation,
                                context: format!("tenstorrent2: reduce is fixed 32x32, op {x} is {wx}x{hx}").into(),
                            });
                        }
                        let Some(&cb_in) = cbs.get(&lx) else {
                            return Err(BackendError {
                                status: ErrorStatus::KernelCompilation,
                                context: format!("tenstorrent2: reduce side op {x} targets unmapped CB").into(),
                            })
                        };
                        let Op::Load { src: la, .. } = kernel.ops[*acc].op else {
                            return Err(BackendError {
                                status: ErrorStatus::KernelCompilation,
                                context: format!("tenstorrent2: reduce acc op {acc} is no acc tile load").into(),
                            })
                        };
                        if !matches!(kernel.ops[la].op, Op::Storage { scope: MemScope::Register, .. }) {
                            return Err(BackendError {
                                status: ErrorStatus::KernelCompilation,
                                context: format!("tenstorrent2: reduce acc op {acc} does not thread a Register acc").into(),
                            });
                        }
                        let Op::Load { src: ls, layout: MemLayout::Tile { .. }, .. } = kernel.ops[*scaler].op
                        else {
                            return Err(BackendError {
                                status: ErrorStatus::KernelCompilation,
                                context: format!("tenstorrent2: reduce scaler op {scaler} is no scaler tile load").into(),
                            })
                        };
                        let Some(&cb_sc) = cbs.get(&ls) else {
                            return Err(BackendError {
                                status: ErrorStatus::KernelCompilation,
                                context: format!("tenstorrent2: reduce scaler op {scaler} targets unmapped CB").into(),
                            })
                        };
                        let Some(&acc_slot) = tiles.get(&la) else {
                            return Err(BackendError {
                                status: ErrorStatus::KernelCompilation,
                                context: format!("tenstorrent2: reduce acc op {acc} reads an undeclared acc").into(),
                            })
                        };
                        tiles.insert(id, acc_slot);
                        ops.push(TTOp::TileReduce { acc: acc_slot, cb_in, cb_sc, rop: *rop, kind: *kind });
                    }
                    Op::MatmulTile { x, y, acc } => {
                        let Op::Load { src: la, layout: MemLayout::Tile { .. }, .. } = kernel.ops[*x].op else {
                            return Err(BackendError {
                                status: ErrorStatus::KernelCompilation,
                                context: format!("tenstorrent2: matmul side op {x} is no CB tile load").into(),
                            })
                        };
                        let Some(&cb_a) = cbs.get(&la) else {
                            return Err(BackendError {
                                status: ErrorStatus::KernelCompilation,
                                context: format!("tenstorrent2: matmul side op {x} targets unmapped CB").into(),
                            })
                        };
                        let Op::Load { src: lb, layout: MemLayout::Tile { .. }, .. } = kernel.ops[*y].op else {
                            return Err(BackendError {
                                status: ErrorStatus::KernelCompilation,
                                context: format!("tenstorrent2: matmul side op {y} is no CB tile load").into(),
                            })
                        };
                        let Some(&cb_b) = cbs.get(&lb) else {
                            return Err(BackendError {
                                status: ErrorStatus::KernelCompilation,
                                context: format!("tenstorrent2: matmul side op {y} targets unmapped CB").into(),
                            })
                        };
                        let Op::Load { src: lacc, .. } = kernel.ops[*acc].op else {
                            return Err(BackendError {
                                status: ErrorStatus::KernelCompilation,
                                context: format!("tenstorrent2: matmul acc op {acc} is no acc tile load").into(),
                            })
                        };
                        let Some(&tile) = tiles.get(&lacc) else {
                            return Err(BackendError {
                                status: ErrorStatus::KernelCompilation,
                                context: format!("tenstorrent2: matmul acc op {acc} reads an undeclared acc").into(),
                            })
                        };
                        tiles.insert(id, tile);
                        *use_counts.entry((cb_a, *x)).or_default() += 1;
                        *use_counts.entry((cb_b, *y)).or_default() += 1;
                        ops.push(TTOp::TileMatmul { acc: tile, cb_a, cb_b, x_load: *x, y_load: *y, out: None });
                    }
                    Op::TransposeTile { x } => {
                        let Op::Load { src: lx, layout: MemLayout::Tile { x: wx, y: hx, .. }, .. } = kernel.ops[*x].op else {
                            return Err(BackendError {
                                status: ErrorStatus::KernelCompilation,
                                context: format!("tenstorrent2: transpose side op {x} is no CB tile load").into(),
                            })
                        };
                        if wx as u32 != 32 || hx as u32 != 32 {
                            return Err(BackendError {
                                status: ErrorStatus::KernelCompilation,
                                context: format!("tenstorrent2: transpose is fixed 32x32, op {x} is {wx}x{hx}").into(),
                            });
                        }
                        let Some(&cb) = cbs.get(&lx) else {
                            return Err(BackendError {
                                status: ErrorStatus::KernelCompilation,
                                context: format!("tenstorrent2: transpose side op {x} targets unmapped CB").into(),
                            })
                        };
                        let dst = def_tile(&mut tiles, &mut free_tiles, &mut next_tile, id)?;
                        ops.push(TTOp::TileTranspose { dst, cb, out: None });
                    }
                    Op::BroadcastTile { .. } => {}
                    Op::Asm { asm, ops: operands } => {
                        let mut resolved = Vec::with_capacity(operands.len());
                        for &operand in operands.iter() {
                            if let Op::Storage { scope: MemScope::Circular, .. } = kernel.ops[operand].op {
                                let Some(&cb) = cbs.get(&operand) else {
                                    return Err(BackendError {
                                        status: ErrorStatus::KernelCompilation,
                                        context: format!("tenstorrent2: asm operand {operand} targets unmapped CB").into(),
                                    })
                                };
                                resolved.push(AsmOperand::Cb(cb));
                            } else if let Some(&slot) = tiles.get(&operand) {
                                resolved.push(AsmOperand::Tile(slot));
                            } else if let Some(&reg) = vars.get(&operand) {
                                resolved.push(AsmOperand::Var(reg));
                            } else {
                                return Err(BackendError {
                                    status: ErrorStatus::KernelCompilation,
                                    context: format!("tenstorrent2: asm operand {operand} is not a CB or live tile").into(),
                                });
                            }
                        }
                        ops.push(TTOp::Asm { asm: asm.clone(), ops: resolved });
                    }
                }
            }
            ops.push(match s {
                0 => TTOp::EndReader,
                1 => TTOp::EndCompute,
                _ => TTOp::EndWriter,
            });
            if s == 1 {
                // Startup-triple scan, legacy rule: IR-order first touch
                // of CB-mapped Tile loads/stores in the compute list.
                for &op in &data.ops {
                    match &kernel.ops[op].op {
                        Op::Load { src, layout: MemLayout::Tile { .. }, .. } => {
                            if let Some(&cb) = cbs.get(src)
                                && !startup_loads.contains(&cb)
                            {
                                startup_loads.push(cb);
                            }
                        }
                        Op::Store { dst, layout: MemLayout::Tile { .. }, .. } => {
                            if startup_store.is_none()
                                && let Some(&cb) = cbs.get(dst)
                            {
                                startup_store = Some(cb);
                            }
                        }
                        _ => {}
                    }
                }
            }
        }
        Ok(Self { ops, startup_loads, startup_store, use_counts })
    }

    /// Replicate_ops_per_section: values consumed in multiple sections are
    /// duplicated per section (each section is a separate kernel with its own
    /// registers and runtime args; arg ordinals stay global so the launch
    /// contract survives). Circular-buffer storages are the exception: one
    /// hardware object, one `CBId`, never duplicated — they are hoisted once
    /// to the stream head. Mirrors the legacy `get_needed_ops` closure per
    /// section (stores + structural starters, transitive data deps), then
    /// re-emits each section's needed ops in stream order.
    /// Sync insertion: `ReserveBack`/`WaitFront`/`PushBack`/`PopFront`
    /// around every CB traffic op. Single-tile transactions wrap the op
    /// inline; a loop body holding exactly one traffic op for a CB, no
    /// branch, and a constant trip > 1 fitting the CB depth upgrades to
    /// a hoisted block (open before the loop, close after its end).
    /// Fused-draining loads carry no syncs; their consumers drain.
    /// `CbDeclare` ops head the stream in `CBId` order.
    /// CB sync insertion (runs after lock+init+reconfig, so sync lands
    /// relative to locks exactly like the legacy event anchors).
    ///
    /// Legacy shapes, straight-line (v1: no hoisted batches):
    /// - reader `ReadTile`: `ReserveBack` before, `PushBack` after;
    /// - writer `WriteTile`: `WaitFront` before, `PopFront` after;
    /// - compute `TileCopy`/`TileTranspose` (event-anchored): `WaitFront`
    ///   BEFORE the cone's `MathLock` (back-scan past inits). No pop:
    ///   pops are placed later by `place_pops` (last use, FIFO order).
    /// - compute `TilePack` (event-anchored): `ReserveBack` BEFORE the
    ///   cone's `MathUnlock` (back-scan past pack lock/reconfig),
    ///   `PushBack` after the op;
    /// - fused `TileMatmul`/`TileBcastBinary`/`TileReduce` (op-internal
    ///   waits): `WaitFront`s after the lock at the current position.
    ///   No pops: placed later by `place_pops`. `WaitFront` carries the
    ///   consumer's sharing group for the wait-dedup pass.
    ///
    /// The back-scan passes inits and the cone's own lock ops; anything
    /// else (a prior traffic op, a loop boundary, a barrier) means this
    /// op opens at the current position and sync goes immediately.
    fn sync_cbs(&mut self) -> Result<(), BackendError> {
        fn is_init(op: &TTOp) -> bool {
            matches!(
                op,
                TTOp::CopyInit { .. }
                    | TTOp::CopyInitWithDt { .. }
                    | TTOp::UnaryInit { .. }
                    | TTOp::BinaryInit { .. }
                    | TTOp::BinScalarInit { .. }
                    | TTOp::FusedInit { .. }
                    | TTOp::CastInit { .. }
                    | TTOp::TransposeInit { .. }
                    | TTOp::MatmulInit { .. }
                    | TTOp::ReduceInit { .. }
                    | TTOp::BcastInit { .. }
                    | TTOp::PackReconfig { .. }
            )
        }
        let old = std::mem::take(&mut self.ops);
        let mut next: Vec<TTOp> = Vec::with_capacity(old.len() * 2);
        let mut section = 0u8;
        // Insert `sync` at the cone lock: scan `next` back past the
        // op's trailing inits and the cone's lock ops to `lock` and
        // insert before it (the legacy open event anchors ahead of the
        // whole op emission, inits included). Anything else stops the
        // scan and sync lands there (past the trailing inits, ahead of
        // the prior traffic op's text). Exhausting the stream without
        // a lock is a lock-pass bug, loud.
        fn before_lock(next: &[TTOp], lock: TTOp) -> Result<usize, BackendError> {
            let mut idx = next.len();
            while idx > 0 {
                let back = &next[idx - 1];
                if *back == lock {
                    return Ok(idx - 1);
                }
                if is_init(back) || matches!(back, TTOp::MathLock | TTOp::MathUnlock | TTOp::PackLock | TTOp::PackUnlock) {
                    idx -= 1;
                    continue;
                }
                return Ok(idx);
            }
            Err(BackendError {
                status: ErrorStatus::KernelCompilation,
                context: "tenstorrent2: sync_cbs: traffic without a cone lock".into(),
            })
        }
        for op in old {
            match &op {
                TTOp::EndReader | TTOp::EndCompute => {
                    section += 1;
                    next.push(op);
                    continue;
                }
                TTOp::EndWriter => {
                    next.push(op);
                    continue;
                }
                _ => {}
            }
            if section != 1 {
                match &op {
                    TTOp::ReadTile { cb, .. } => {
                        let cb = *cb;
                        next.push(TTOp::ReserveBack { cb, n: 1 });
                        next.push(op);
                        next.push(TTOp::PushBack { cb, n: 1 });
                    }
                    TTOp::WriteTile { cb, .. } => {
                        let cb = *cb;
                        next.push(TTOp::WaitFront { cb, m: 1, grp: None });
                        next.push(op);
                        next.push(TTOp::PopFront { cb, n: 1 });
                    }
                    _ => next.push(op),
                }
                continue;
            }
            match &op {
                TTOp::TileCopy { cb, .. } => {
                    let cb = *cb;
                    let at = before_lock(&next, TTOp::MathLock)?;
                    next.insert(at, TTOp::WaitFront { cb, m: 1, grp: None });
                    next.push(op);
                }
                TTOp::TileTranspose { cb, .. } => {
                    let cb = *cb;
                    let at = before_lock(&next, TTOp::MathLock)?;
                    next.insert(at, TTOp::WaitFront { cb, m: 1, grp: None });
                    next.push(op);
                }
                TTOp::TilePack { cb, .. } => {
                    let cb = *cb;
                    let at = before_lock(&next, TTOp::MathUnlock)?;
                    next.insert(at, TTOp::ReserveBack { cb, n: 1 });
                    next.push(op);
                    next.push(TTOp::PushBack { cb, n: 1 });
                }
                TTOp::TileMatmul { cb_a, cb_b, x_load, y_load, .. } => {
                    let (cb_a, cb_b) = (*cb_a, *cb_b);
                    let (x_load, y_load) = (*x_load, *y_load);
                    next.push(TTOp::WaitFront { cb: cb_a, m: 1, grp: Some(WaitGroup::MatA(x_load)) });
                    next.push(TTOp::WaitFront { cb: cb_b, m: 1, grp: Some(WaitGroup::MatB(y_load)) });
                    next.push(op);
                }
                TTOp::TileBcastBinary { cb_a, cb_b, .. } => {
                    let (cb_a, cb_b) = (*cb_a, *cb_b);
                    next.push(TTOp::WaitFront { cb: cb_a, m: 1, grp: None });
                    next.push(TTOp::WaitFront { cb: cb_b, m: 1, grp: None });
                    next.push(op);
                }
                TTOp::TileReduce { cb_in, cb_sc, .. } => {
                    let (cb_in, cb_sc) = (*cb_in, *cb_sc);
                    next.push(TTOp::WaitFront { cb: cb_in, m: 1, grp: None });
                    next.push(TTOp::WaitFront { cb: cb_sc, m: 1, grp: None });
                    next.push(op);
                }
                _ => next.push(op),
            }
        }
        self.ops = next;
        Ok(())
    }

    /// Wait dedup: drop a `WaitFront(cb)` whose tile is already waited.
    /// A wait merges into the nearest preceding wait on the same CB iff
    /// both carry the same sharing group and no CB-affecting event
    /// (`PopFront`/`PushBack`/`ReserveBack`) sits between. Opaque waits
    /// (`grp: None`) never merge. Loop/branch/section markers reset all
    /// chains (per-trip tiles stay per-trip). Single scan, straight-line
    /// only.
    fn dedup_waits(&mut self) {
        let old = std::mem::take(&mut self.ops);
        let mut next = Vec::with_capacity(old.len());
        let mut last: Map<CBId, Option<WaitGroup>> = Map::default();
        for op in old {
            match &op {
                TTOp::EndReader | TTOp::EndCompute | TTOp::EndWriter => {
                    last.clear();
                    next.push(op);
                }
                TTOp::Loop { .. } | TTOp::EndLoop | TTOp::If { .. } | TTOp::EndIf => {
                    last.clear();
                    next.push(op);
                }
                TTOp::PopFront { cb, .. } | TTOp::PushBack { cb, .. } | TTOp::ReserveBack { cb, .. } => {
                    last.insert(*cb, None);
                    next.push(op);
                }
                TTOp::WaitFront { cb, m: 1, grp: Some(g) } => {
                    if last.get(cb) == Some(&Some(*g)) {
                        continue;
                    }
                    last.insert(*cb, Some(*g));
                    next.push(op);
                }
                TTOp::WaitFront { cb, .. } => {
                    last.insert(*cb, None);
                    next.push(op);
                }
                _ => next.push(op),
            }
        }
        self.ops = next;
    }

    /// Pop placement: pop every compute-waited tile exactly once, after
    /// its last static use, in FIFO (push) order. Use totals come from
    /// the conversion table (`(CB, index-op) → uses`); groups absent
    /// there pop strictly after their op (today's shape). A pop whose
    /// tile is not the queue head is a loud compile error (non-FIFO
    /// consumption). Pushes are `ReadTile`s (grouped by index op);
    /// packs and writer pairs pass through untouched. Loops re-execute
    /// placed pops per trip, so trip handling needs no special case.
    /// Single scan.
    fn place_pops(&mut self) -> Result<(), BackendError> {
        let old = std::mem::take(&mut self.ops);
        let mut next = Vec::with_capacity(old.len() + old.len() / 4);
        let mut section = 0u8;
        // Remaining static uses per (CB, source-load) group.
        let mut remaining: Map<(CBId, OpId), u32> = Map::default();
        for op in old {
            match &op {
                TTOp::EndReader | TTOp::EndCompute => {
                    section += 1;
                    next.push(op);
                    continue;
                }
                TTOp::EndWriter => {
                    next.push(op);
                    continue;
                }
                _ => {}
            }
            if section != 1 {
                next.push(op);
                continue;
            }
            // Countdown sides: pop at last use. Strict sides: pop
            // immediately after the op (today's shape).
            let counted: [(CBId, OpId); 2];
            let n_counted: usize;
            let strict: [Option<CBId>; 2];
            match &op {
                // One copy per load: the CB tile is consumed exactly
                // once, here (a fan-out DST value shares the slot, not
                // the CB tile).
                TTOp::TileCopy { cb, .. } => {
                    counted = [(CBId(u32::MAX), OpId::NULL); 2];
                    n_counted = 0;
                    strict = [Some(*cb), None];
                }
                TTOp::TileMatmul { cb_a, cb_b, x_load, y_load, .. } => {
                    counted = [(*cb_a, *x_load), (*cb_b, *y_load)];
                    n_counted = 2;
                    strict = [None, None];
                }
                TTOp::TileReduce { cb_in, cb_sc, .. } => {
                    counted = [(CBId(u32::MAX), OpId::NULL); 2];
                    n_counted = 0;
                    strict = [Some(*cb_in), Some(*cb_sc)];
                }
                TTOp::TileTranspose { cb, .. } => {
                    counted = [(CBId(u32::MAX), OpId::NULL); 2];
                    n_counted = 0;
                    strict = [Some(*cb), None];
                }
                TTOp::TileBcastBinary { cb_a, cb_b, .. } => {
                    counted = [(CBId(u32::MAX), OpId::NULL); 2];
                    n_counted = 0;
                    strict = [Some(*cb_a), Some(*cb_b)];
                }
                _ => {
                    next.push(op);
                    continue;
                }
            }
            next.push(op);
            for (cb, load) in counted.into_iter().take(n_counted) {
                let Some(&total) = self.use_counts.get(&(cb, load)) else {
                    return Err(BackendError {
                        status: ErrorStatus::KernelCompilation,
                        context: format!("tenstorrent2: place_pops: matmul side ({cb}, {load:?}) has no use count")
                            .into(),
                    });
                };
                let left = remaining.entry((cb, load)).or_insert(total);
                if *left == 0 {
                    return Err(BackendError {
                        status: ErrorStatus::KernelCompilation,
                        context: format!("tenstorrent2: place_pops: use beyond counted total on CB{cb}").into(),
                    });
                }
                *left -= 1;
                if *left == 0 {
                    remaining.remove(&(cb, load));
                    next.push(TTOp::PopFront { cb, n: 1 });
                }
            }
            for cb in strict.into_iter().flatten() {
                next.push(TTOp::PopFront { cb, n: 1 });
            }
        }
        if !remaining.is_empty() {
            return Err(BackendError {
                status: ErrorStatus::KernelCompilation,
                context: "tenstorrent2: place_pops: partially consumed groups at end of stream".into(),
            });
        }
        self.ops = next;
        Ok(())
    }

    /// DST lock insertion: `MathLock`/`MathUnlock`/`PackLock`/`PackUnlock`
    /// around compute-section tile traffic. Faithful port of the legacy
    /// `TileEmitter` state machine (`tenstorrent.rs`): lazy MATH acquire
    /// (first take acquires, later takes in the same cone keep), deferred
    /// PACK release (consecutive packs share one cone; the release flushes
    /// at the next MATH take or at section/loop end). MATH ops are tile
    /// compute ops; PACK ops are tile stores draining to a CB. Scalar and
    /// movement sections carry no locks.
    fn lock_dst(&mut self) -> Result<(), BackendError> {
        #[derive(Debug, Clone, Copy, PartialEq, Eq)]
        enum DstState {
            Unlocked,
            MathLock,
            PackLock,
        }
        // MATH ops are tile compute ops (copies stream under MATH
        // like every unpack-side op); PACK ops are `TilePack`s draining
        // to a CB. Scalar and movement sections carry no locks. Acc
        // threading leaves no ops (alias bindings in lowering), so no
        // register scan: only emitted ops lock.
        let is_math = |op: &TTOp| -> bool {
            match op {
                TTOp::TileCopy { .. }
                | TTOp::TileUnary { .. }
                | TTOp::TileBinary { .. }
                | TTOp::TileBcastBinary { .. }
                | TTOp::TileBinScalar { .. }
                | TTOp::TileCast { .. }
                | TTOp::TileMatmul { .. }
                | TTOp::TileReduce { .. }
                | TTOp::TileTranspose { .. } => true,
                _ => false,
            }
        };
        let old = std::mem::take(&mut self.ops);
        let mut next = Vec::with_capacity(old.len());
        let mut state = DstState::Unlocked;
        let mut section = 0u8;
        // Positions of open `Loop` ops in `next` (outermost first) and
        // of the currently open `MathLock` op. A MATH cone that is
        // still open at `EndLoop` would re-execute its acquire on the
        // back edge and wedge the DST — the acquire relocates to the
        // preheader of the outermost loop it crosses (the legacy LICM
        // position).
        let mut loop_starts: Vec<usize> = Vec::new();
        let mut lock_pos: Option<usize> = None;
        let math_lock = |next: &mut Vec<TTOp>, state: &mut DstState, lock_pos: &mut Option<usize>| match *state {
            DstState::MathLock => {}
            DstState::PackLock => {
                next.push(TTOp::PackUnlock);
                next.push(TTOp::MathLock);
                *lock_pos = Some(next.len() - 1);
                *state = DstState::MathLock;
            }
            DstState::Unlocked => {
                next.push(TTOp::MathLock);
                *lock_pos = Some(next.len() - 1);
                *state = DstState::MathLock;
            }
        };
        for op in old {
            match &op {
                TTOp::EndReader | TTOp::EndCompute => {
                    if section == 1 && state == DstState::PackLock {
                        next.push(TTOp::PackUnlock);
                        state = DstState::Unlocked;
                    }
                    section += 1;
                    next.push(op);
                    continue;
                }
                TTOp::EndWriter => {
                    if section == 1 && state == DstState::PackLock {
                        next.push(TTOp::PackUnlock);
                        state = DstState::Unlocked;
                    }
                    next.push(op);
                    continue;
                }
                TTOp::Loop { .. } => {
                    loop_starts.push(next.len());
                    next.push(op.clone());
                    continue;
                }
                TTOp::EndLoop => {
                    // No open cone across the back-edge: the body runs N
                    // times, so a PACK-held file here would deadlock the
                    // next iteration on an acquire past the loop.
                    if section == 1 && state == DstState::PackLock {
                        next.push(TTOp::PackUnlock);
                        state = DstState::Unlocked;
                    }
                    // A MATH-held cone here would re-execute the
                    // acquire every iteration: relocate it to this
                    // loop's preheader (repeat for outer loops).
                    if section == 1 && state == DstState::MathLock {
                        while let Some(&start) = loop_starts.last() {
                            if let Some(pos) = lock_pos
                                && pos > start
                            {
                                let lock_op = next.remove(pos);
                                next.insert(start, lock_op);
                                lock_pos = Some(start);
                                for s in loop_starts.iter_mut() {
                                    if *s > pos {
                                        *s -= 1;
                                    }
                                    if *s >= start {
                                        *s += 1;
                                    }
                                }
                            }
                            break;
                        }
                    }
                    loop_starts.pop();
                    next.push(op);
                    continue;
                }
                _ => {}
            }
            if section != 1 {
                next.push(op);
                continue;
            }
            // Pack path: `TilePack` draining to a CB.
            if let TTOp::TilePack { .. } = &op {
                match state {
                    DstState::MathLock => {
                        next.push(TTOp::MathUnlock);
                        next.push(TTOp::PackLock);
                        state = DstState::PackLock;
                    }
                    DstState::PackLock => {}
                    DstState::Unlocked => {
                        return Err(BackendError {
                            status: ErrorStatus::KernelCompilation,
                            context: "tenstorrent2: pack with DST Unlocked, no live cone (pack of a dead slot)".into(),
                        });
                    }
                }
                next.push(op);
                continue;
            }
            if is_math(&op) {
                math_lock(&mut next, &mut state, &mut lock_pos);
            }
            next.push(op);
        }
        self.ops = next;
        Ok(())
    }

    /// MATH/config init insertion (pass 1 of 2): a full init before
    /// every compute-section tile compute op. Naive: no hoisting, no
    /// dedup (the hoist+dedup pass folds these later); redundancy is
    /// safe. Copy inits track the unpack-A source like the legacy
    /// emitter (`with_dt` form on format change, plain short
    /// otherwise); matmul notes it the same way. Anything the naive
    /// pass cannot resolve fails loudly at the exact op.
    fn init_math(&mut self) -> Result<(), BackendError> {
        // CB runtime-format table (`CbDeclare` is the single source;
        // codes match the legacy `CBEmitter::config` table).
        let mut fmt_of: Map<CBId, u32> = Map::default();
        for op in &self.ops {
            if let TTOp::CbDeclare { cb, format, .. } = op {
                fmt_of.insert(*cb, *format);
            }
        }
        let old = std::mem::take(&mut self.ops);
        let mut next = Vec::with_capacity(old.len());
        let mut section = 0u8;
        // Tracked unpack source (CB + format), mirroring the legacy
        // emitter: `None` until a `with_dt` reconfig or a matmul
        // programs it; plain short inits leave it untouched.
        let mut unpack_src: Option<(CBId, u32)> = None;
        for op in old {
            match &op {
                TTOp::EndReader | TTOp::EndCompute => {
                    section += 1;
                    next.push(op);
                    continue;
                }
                TTOp::EndWriter => {
                    next.push(op);
                    continue;
                }
                _ => {}
            }
            if section != 1 {
                next.push(op);
                continue;
            }
            match &op {
                TTOp::TileCopy { cb, .. } => {
                    let fmt = *fmt_of.get(cb).ok_or_else(|| BackendError {
                        status: ErrorStatus::KernelCompilation,
                        context: "tenstorrent2: init_math: copy on undeclared CB".into(),
                    })?;
                    match unpack_src {
                        Some((prev, f)) if f != fmt => {
                            next.push(TTOp::CopyInitWithDt { prev, cb: *cb });
                            unpack_src = Some((*cb, fmt));
                        }
                        _ => next.push(TTOp::CopyInit { cb: *cb }),
                    }
                }
                TTOp::TileUnary { uop, .. } => next.push(TTOp::UnaryInit { uop: *uop }),
                TTOp::TileBinary { bop, .. } => next.push(TTOp::BinaryInit { bop: *bop }),
                TTOp::TileBinScalar { bop, .. } => next.push(TTOp::BinScalarInit { bop: *bop }),
                TTOp::TileFused { kind, .. } => next.push(TTOp::FusedInit { kind: *kind }),
                TTOp::TileCast { in_dtype, out_dtype, .. } => {
                    next.push(TTOp::CastInit { in_dtype: *in_dtype, out_dtype: *out_dtype })
                }
                TTOp::TileTranspose { cb, out, .. } => next.push(TTOp::TransposeInit {
                    cb: *cb,
                    out: out.ok_or_else(|| BackendError {
                        status: ErrorStatus::KernelCompilation,
                        context: "tenstorrent2: init_math: transpose with unfilled out".into(),
                    })?,
                }),
                TTOp::TileMatmul { cb_a, cb_b, out, .. } => {
                    next.push(TTOp::MatmulInit {
                        a: *cb_a,
                        b: *cb_b,
                        out: out.ok_or_else(|| BackendError {
                            status: ErrorStatus::KernelCompilation,
                            context: "tenstorrent2: init_math: matmul with unfilled out".into(),
                        })?,
                    });
                    let fmt_a = *fmt_of.get(cb_a).ok_or_else(|| BackendError {
                        status: ErrorStatus::KernelCompilation,
                        context: "tenstorrent2: init_math: matmul on undeclared CB".into(),
                    })?;
                    unpack_src = Some((*cb_a, fmt_a));
                }
                TTOp::TileBcastBinary { bop, kind, cb_a, cb_b, .. } => {
                    next.push(TTOp::BcastInit { bop: *bop, kind: *kind, cb_a: *cb_a, cb_b: *cb_b })
                }
                TTOp::TileReduce { acc, cb_in, cb_sc, rop, kind } => {
                    next.push(TTOp::ReduceInit { ci: *cb_in, cs: *cb_sc, acc: *acc, rop: *rop, kind: *kind })
                }
                _ => {}
            }
            next.push(op);
        }
        self.ops = next;
        Ok(())
    }

    /// Pack reconfig insertion (pass 2 of 2): a `PackReconfig` before
    /// every compute-section pack. Naive and unconditional — the first
    /// pack must always reconfigure (silicon starts the packer on
    /// BF16), and redundant reconfigs are safe; the hoist+dedup pass
    /// folds them later.
    fn reconfig_pack(&mut self) {
        let old = std::mem::take(&mut self.ops);
        let mut next = Vec::with_capacity(old.len());
        let mut section = 0u8;
        for op in old {
            match &op {
                TTOp::EndReader | TTOp::EndCompute => {
                    section += 1;
                    next.push(op);
                    continue;
                }
                TTOp::EndWriter => {
                    next.push(op);
                    continue;
                }
                _ => {}
            }
            if section == 1 {
                if let TTOp::TilePack { cb, .. } = &op {
                    next.push(TTOp::PackReconfig { cb: *cb });
                }
            }
            next.push(op);
        }
        self.ops = next;
    }

    /// NOC movement lowering (reader/writer sections only; runs after
    /// `scalar_regs`, before the compute-only passes).
    ///
    /// Reader `Store{dst: Circular, src: Load{src: Global param}}` becomes
    /// `NocAccessor` (once per param) + `NocAddr` + `AsyncRead` + barrier;
    /// writer `Store{dst: GlobalMut param, src: Load{src: CB storage}}`
    /// becomes the `AsyncWrite` form. Loads drop (consumed at the store);
    /// a load with no consuming store is a compilation error, like the
    /// legacy "supports only global to local stores" rule.
    ///
    /// v1: tile layouts only, reader `Global→Circular`, writer
    /// `CB→GlobalMut`. Everything else stays loud.
    /// NOC movement lowering (movement sections only; compute flows
    /// through untouched).
    ///
    /// Reader `ReadTile` becomes `NocAccessor` (once per param per
    /// section) + `NocAddr` + `AsyncRead` + `NocReadBarrier`; writer
    /// `WriteTile` becomes the `AsyncWrite` form. This mirrors the
    /// legacy traffic emission exactly (barrier per transfer plus the
    /// trailing reader barrier pushed at `EndReader` below).
    ///
    /// v1 sync wraps every transaction singly, so the CB slot offset
    /// is always `None` (plain pointer) — the legacy `slot_offset`
    /// per_op == 1 rule. Batch hoisting (offsets inside one open
    /// transaction) is a later pass; it will own the provenance
    /// tracking this shape leaves out.
    fn noc_movement(&mut self) {
        let old = std::mem::take(&mut self.ops);
        // Fresh scalar registers for expanded address temporaries:
        // one past the stream's max VarId.
        let mut fresh = {
            let mut m = 0u32;
            let mut take = |v: VarId| m = m.max(v.0);
            for op in &old {
                match op {
                    TTOp::Arg { z, .. } | TTOp::Const { z, .. } | TTOp::TensixGridX { z, .. } | TTOp::TensixGridY { z, .. } => {
                        take(*z)
                    }
                    TTOp::Binary { z, x, y, .. } => {
                        take(*z);
                        take(*x);
                        take(*y);
                    }
                    TTOp::Unary { z, x, .. } | TTOp::Cast { z, x, .. } => {
                        take(*z);
                        take(*x);
                    }
                    TTOp::Mad { z, x, y, w, .. } => {
                        take(*z);
                        take(*x);
                        take(*y);
                        take(*w);
                    }
                    TTOp::NocAddr { z, index, .. } => {
                        take(*z);
                        take(*index);
                    }
                    TTOp::Loop { len, counter, .. } => {
                        take(*len);
                        take(*counter);
                    }
                    TTOp::If { cond } => take(*cond),
                    TTOp::ReadTile { index, .. } | TTOp::WriteTile { index, .. } => take(*index),
                    TTOp::TileCopy { index, .. } => take(*index),
                    TTOp::AsyncRead { addr, off, .. } | TTOp::AsyncWrite { addr, off, .. } => {
                        take(*addr);
                        if let Some(o) = off {
                            take(*o);
                        }
                    }
                    _ => {}
                }
            }
            m + 1
        };
        let mut next = Vec::with_capacity(old.len());
        let mut section = 0usize;
        for op in old {
            match op {
                TTOp::EndReader | TTOp::EndCompute => {
                    section += 1;
                    if section == 1 {
                        // Trailing reader barrier: every async read lands
                        // before exit (legacy `final_read_barrier`).
                        next.push(TTOp::NocReadBarrier);
                        next.push(TTOp::EndReader);
                    } else {
                        next.push(TTOp::EndCompute);
                    }
                    continue;
                }
                TTOp::EndWriter => {
                    next.push(TTOp::EndWriter);
                    continue;
                }
                _ => {}
            }
            debug_assert!(section < 3, "tenstorrent2: noc_movement: op past EndWriter");
            if section == 1 {
                next.push(op);
                continue;
            }
            // Loop stack pushes need the counter out of the op.
            match op {
                TTOp::ReadTile { ordinal, dtype: _, index, cb, bytes, elem_size, .. } => {
                    debug_assert_eq!(section, 0, "tenstorrent2: noc_movement: reader transfer outside the reader section");
                    let z = VarId(fresh);
                    fresh += 1;
                    next.push(TTOp::NocAddr { z, ordinal, index, elem_size });
                    next.push(TTOp::AsyncRead { addr: z, dst_cb: cb, bytes, off: None });
                    next.push(TTOp::NocReadBarrier);
                }
                TTOp::WriteTile { cb, ordinal, dtype: _, index, bytes, elem_size } => {
                    debug_assert_eq!(section, 2, "tenstorrent2: noc_movement: writer transfer outside the writer section");
                    let z = VarId(fresh);
                    fresh += 1;
                    next.push(TTOp::NocAddr { z, ordinal, index, elem_size });
                    next.push(TTOp::AsyncWrite { src_cb: cb, addr: z, bytes, off: None });
                    next.push(TTOp::NocWriteBarrier);
                }
                other => next.push(other),
            }
        }
        self.ops = next;
    }

    /// Hoist writer DRAM accessors to the writer-section front.
    ///
    /// Legacy v1 declares all writer accessors up front (chained
    /// `TensorAccessor` triples) while the reader declares inline at
    /// first traffic. Render is a single dumb emitter, so the ordering
    /// lives here as a stable partition: writer-section `NocAccessor`
    /// ops move to immediately after `EndCompute`, keeping their
    /// relative (chain) order; every other op keeps its position.
    fn hoist_writer_accessors(&mut self) -> Result<(), BackendError> {
        let old = std::mem::take(&mut self.ops);
        let mut hoisted = Vec::new();
        let mut next = Vec::with_capacity(old.len());
        let mut section = 0u8;
        let mut writer_front = 0usize;
        for op in old {
            match &op {
                TTOp::EndReader | TTOp::EndCompute => {
                    section += 1;
                    next.push(op);
                    if section == 2 {
                        writer_front = next.len();
                    }
                }
                TTOp::NocAccessor { .. } if section == 2 => hoisted.push(op),
                _ => next.push(op),
            }
        }
        if section != 2 {
            return Err(BackendError {
                status: ErrorStatus::KernelCompilation,
                context: "tenstorrent2: hoist_writer_accessors: stream has no writer section".into(),
            });
        }
        next.splice(writer_front..writer_front, hoisted);
        self.ops = next;
        Ok(())
    }

    /// Reduce-cone close: a `TileReduce` opens a reduce cone on its
    /// acc slot; the pack draining that slot closes it — `ReduceUninit`
    /// right before the cone's `MathUnlock` (the legacy `reduce_pending`
    /// rule: `reduce_uninit()` between the reserve and the commit).
    fn close_reduce_cones(&mut self) -> Result<(), BackendError> {
        let old = std::mem::take(&mut self.ops);
        let mut next = Vec::with_capacity(old.len());
        let mut pending: Option<TileId> = None;
        let mut section = 0u8;
        for op in old {
            match &op {
                TTOp::EndReader | TTOp::EndCompute => {
                    section += 1;
                    pending = None;
                    next.push(op);
                    continue;
                }
                TTOp::EndWriter => {
                    next.push(op);
                    continue;
                }
                _ => {}
            }
            match &op {
                TTOp::TileReduce { acc, .. } => {
                    pending = Some(*acc);
                    next.push(op);
                }
                TTOp::TilePack { slot, .. } if section == 1 && pending == Some(*slot) => {
                    let at = next.iter().rposition(|o| matches!(o, TTOp::MathUnlock)).ok_or_else(|| BackendError {
                        status: ErrorStatus::KernelCompilation,
                        context: "tenstorrent2: close_reduce_cones: pack without an open MathUnlock".into(),
                    })?;
                    next.insert(at, TTOp::ReduceUninit);
                    pending = None;
                    next.push(op);
                }
                other => next.push(other.clone()),
            }
        }
        self.ops = next;
        Ok(())
    }

    /// CB batching: hoist per-tile sync groups out of innermost
    /// constant-trip loops. A reader body group is
    /// `ReserveBack(cb,1) … AsyncRead{dst_cb: cb, off: None} …
    /// PushBack(cb,1)`; a writer body group is the `WaitFront`/
    /// `AsyncWrite{src_cb: cb}`/`PopFront` mirror. Batched form: one
    /// `ReserveBack(cb, n)` (or `WaitFront(cb, n)`) before the loop,
    /// the per-trip transfer writes slot `counter` via
    /// `off: Some(counter)`, and after the loop one barrier plus
    /// `PushBack(cb, n)` (or `PopFront(cb, n)`) per group. Any group
    /// that does not match the shape stays per-tile (correct, just
    /// unbatched). Traffic totals are unchanged.
    fn batch_cbs(&mut self) {
        if std::env::var("ZYX_DEBUG").is_ok_and(|v| v == "4") {
            eprintln!("TTIR OPS BEFORE BATCH:\n{:?}", self.ops); // TEMP DEBUG: remove
        }
        let old = std::mem::take(&mut self.ops);
        // Tile capacity per CB (from its CbDeclare): a batched
        // ReserveBack/WaitFront of `n` tiles deadlocks a CB that only
        // holds fewer tiles, so span groups smaller than the trip
        // count stay per-tile.
        let mut cb_tiles: Map<CBId, u32> = Map::default();
        for op in &old {
            if let TTOp::CbDeclare { cb, n_tiles, .. } = op {
                cb_tiles.insert(*cb, *n_tiles);
            }
        }
        let mut next = Vec::with_capacity(old.len());
        let mut section = 0u8;
        let mut i = 0usize;
        while i < old.len() {
            match &old[i] {
                TTOp::EndReader | TTOp::EndCompute => {
                    section += 1;
                    next.push(old[i].clone());
                    i += 1;
                }
                TTOp::EndWriter => {
                    next.push(old[i].clone());
                    i += 1;
                }
                TTOp::Loop { trip: Some(n), counter, .. } if (section == 0 || section == 2) && *n >= 2 => {
                    // Find the matching EndLoop; batch innermost bodies only.
                    let mut depth = 0usize;
                    let mut j = i + 1;
                    let mut nested = false;
                    while j < old.len() {
                        match &old[j] {
                            TTOp::Loop { .. } | TTOp::If { .. } => nested = true,
                            TTOp::EndLoop if depth == 0 => break,
                            TTOp::EndLoop => depth -= 1,
                            _ => {}
                        }
                        j += 1;
                    }
                    if j >= old.len() || nested {
                        next.push(old[i].clone());
                        i += 1;
                        continue;
                    }
                    let reader = section == 0;
                    let body: Vec<TTOp> = old[i + 1..j].to_vec();
                    // Group spans: (open idx, close idx, cb, transfer idx,
                    // barrier idx). Open = ReserveBack/WaitFront(cb,1),
                    // close = PushBack/PopFront(cb,1).
                    let mut spans: Vec<(usize, usize, CBId, usize, Option<usize>)> = Vec::new();
                    for (a, op) in body.iter().enumerate() {
                        let open_cb = match op {
                            TTOp::ReserveBack { cb, n: 1 } if reader => Some(*cb),
                            TTOp::WaitFront { cb, m: 1, .. } if !reader => Some(*cb),
                            _ => None,
                        };
                        let Some(cb) = open_cb else { continue };
                        let Some(b) = body[a + 1..]
                            .iter()
                            .position(|op| match op {
                                TTOp::PushBack { cb: c, n: 1 } if reader => *c == cb,
                                TTOp::PopFront { cb: c, n: 1 } if !reader => *c == cb,
                                _ => false,
                            })
                            .map(|p| a + 1 + p)
                        else {
                            continue;
                        };
                        // Validate the span: exactly one matching
                        // transfer, at most one matching barrier, no
                        // other sync/NOC-traffic ops, no structure.
                        let mut transfer = None;
                        let mut barrier = None;
                        let mut ok = true;
                        for (k, op) in body[a + 1..b].iter().enumerate() {
                            match op {
                                TTOp::AsyncRead { dst_cb, off: None, .. } if reader && *dst_cb == cb => {
                                    if transfer.is_some() {
                                        ok = false;
                                        break;
                                    }
                                    transfer = Some(a + 1 + k);
                                }
                                TTOp::AsyncWrite { src_cb, off: None, .. } if !reader && *src_cb == cb => {
                                    if transfer.is_some() {
                                        ok = false;
                                        break;
                                    }
                                    transfer = Some(a + 1 + k);
                                }
                                TTOp::NocReadBarrier if reader => {
                                    if barrier.is_some() {
                                        ok = false;
                                        break;
                                    }
                                    barrier = Some(a + 1 + k);
                                }
                                TTOp::NocWriteBarrier if !reader => {
                                    if barrier.is_some() {
                                        ok = false;
                                        break;
                                    }
                                    barrier = Some(a + 1 + k);
                                }
                                TTOp::ReserveBack { .. }
                                | TTOp::PushBack { .. }
                                | TTOp::WaitFront { .. }
                                | TTOp::PopFront { .. }
                                | TTOp::AsyncRead { .. }
                                | TTOp::AsyncWrite { .. }
                                | TTOp::NocReadBarrier
                                | TTOp::NocWriteBarrier
                                | TTOp::Loop { .. }
                                | TTOp::If { .. }
                                | TTOp::EndReader
                                | TTOp::EndCompute
                                | TTOp::EndWriter => {
                                    ok = false;
                                    break;
                                }
                                _ => {}
                            }
                        }
                        if ok
                            && let Some(transfer) = transfer
                            && cb_tiles.get(&cb).copied().unwrap_or(0) >= *n
                        {
                            spans.push((a, b, cb, transfer, barrier));
                        }
                    }
                    if spans.is_empty() {
                        next.push(old[i].clone());
                        i += 1;
                        continue;
                    }
                    let mut in_span = vec![false; body.len()];
                    for &(a, b, _, _, _) in &spans {
                        for k in a..=b {
                            in_span[k] = true;
                        }
                    }
                    let reader = section == 0;
                    for &(_, _, cb, _, _) in &spans {
                        if reader {
                            next.push(TTOp::ReserveBack { cb, n: *n });
                        } else {
                            next.push(TTOp::WaitFront { cb, m: *n, grp: None });
                        }
                    }
                    next.push(old[i].clone());
                    for (k, op) in body.iter().enumerate() {
                        if !in_span[k] {
                            next.push(op.clone());
                            continue;
                        }
                        match op {
                            TTOp::ReserveBack { .. } | TTOp::WaitFront { .. } | TTOp::PushBack { .. } | TTOp::PopFront { .. } => {
                            }
                            TTOp::AsyncRead { addr, dst_cb, bytes, .. } => {
                                next.push(TTOp::AsyncRead { addr: *addr, dst_cb: *dst_cb, bytes: *bytes, off: Some(*counter) });
                            }
                            TTOp::AsyncWrite { src_cb, addr, bytes, .. } => {
                                next.push(TTOp::AsyncWrite { src_cb: *src_cb, addr: *addr, bytes: *bytes, off: Some(*counter) });
                            }
                            TTOp::NocReadBarrier | TTOp::NocWriteBarrier => next.push(op.clone()),
                            other => next.push(other.clone()),
                        }
                    }
                    next.push(TTOp::EndLoop);
                    next.push(if reader { TTOp::NocReadBarrier } else { TTOp::NocWriteBarrier });
                    for &(_, _, cb, _, _) in &spans {
                        if reader {
                            next.push(TTOp::PushBack { cb, n: *n });
                        } else {
                            next.push(TTOp::PopFront { cb, n: *n });
                        }
                    }
                    i = j + 1;
                }
                _ => {
                    next.push(old[i].clone());
                    i += 1;
                }
            }
        }
        self.ops = next;
    }

    /// Fill `out: Option<CBId>` placeholders on transpose/matmul/bcast
    /// tile ops. The output CB is the compute section's first-packed
    /// CB — the same source the startup triple's third slot reads
    /// (see `verify`), matching legacy (`transpose_wh_init`/`mm_init`
    /// read it off the startup triple).
    fn fill_out_cbs(&mut self) -> Result<(), BackendError> {
        let mut section = 0u8;
        let mut first_pack: Option<CBId> = None;
        for op in &self.ops {
            match op {
                TTOp::EndReader | TTOp::EndCompute => section += 1,
                TTOp::TilePack { cb, .. } if section == 1 && first_pack.is_none() => {
                    first_pack = Some(*cb);
                }
                _ => {}
            }
        }
        for op in self.ops.iter_mut() {
            match op {
                TTOp::TileTranspose { out, .. } | TTOp::TileMatmul { out, .. } => {
                    if out.is_none() {
                        *out = Some(first_pack.ok_or_else(|| BackendError {
                            status: ErrorStatus::KernelCompilation,
                            context: "tenstorrent2: fill_out_cbs: transpose/matmul with no packed CB".into(),
                        })?);
                    }
                }
                TTOp::TileBcastBinary { out, .. } => {
                    if out.is_none() {
                        *out = Some(first_pack.ok_or_else(|| BackendError {
                            status: ErrorStatus::KernelCompilation,
                            context: "tenstorrent2: fill_out_cbs: bcast with no packed CB".into(),
                        })?);
                    }
                }
                _ => {}
            }
        }
        Ok(())
    }

    fn tile_regs(&mut self) {
        // BIGBANG: subsumed by lowering.
    }

    /// Hoist + dedup of init/reconfig ops (compute section only).
    ///
    /// Hoist: a constant-trip loop body holding exactly one distinct init
    /// config per unit moves that init to the loop preheader. Single-config
    /// is the only safe shape (sticky hardware state: hoisting two configs
    /// would leave the last programmed for the first use). Trip must be a
    /// known `>= 1` (a zero-trip loop would program state for uses that
    /// never run while dedup believes it did); symbolic or zero-trip loops
    /// keep their inits inside, which is always correct. Bodies holding
    /// lock ops are never hoisted across.
    ///
    /// Dedup: per-unit last-programmed state drops redundant inits within
    /// one lock epoch (`MathLock` clears, `PackUnlock` clears). Compute and
    /// sync ops never touch the state. Rebuild, locks/sync/SSA untouched.
    fn hoist_dedup_inits(&mut self) -> Result<(), BackendError> {
        fn is_init(op: &TTOp) -> bool {
            matches!(
                op,
                TTOp::CopyInit { .. }
                    | TTOp::CopyInitWithDt { .. }
                    | TTOp::UnaryInit { .. }
                    | TTOp::BinaryInit { .. }
                    | TTOp::BinScalarInit { .. }
                    | TTOp::FusedInit { .. }
                    | TTOp::CastInit { .. }
                    | TTOp::TransposeInit { .. }
                    | TTOp::MatmulInit { .. }
                    | TTOp::ReduceInit { .. }
                    | TTOp::BcastInit { .. }
                    | TTOp::PackReconfig { .. }
            )
        }
        fn is_lock(op: &TTOp) -> bool {
            matches!(op, TTOp::MathLock | TTOp::MathUnlock | TTOp::PackLock | TTOp::PackUnlock)
        }
        // Unit class: unpack programs the unpacker source, math programs the
        // compute unit, pack programs the packer. Dedup tracks one state
        // per unit; hoisting requires a single distinct config per unit.
        #[derive(Clone, Copy, PartialEq, Eq)]
        enum Unit {
            Unpack,
            Math,
            Pack,
        }
        fn unit(op: &TTOp) -> Unit {
            match op {
                TTOp::CopyInit { .. } | TTOp::CopyInitWithDt { .. } => Unit::Unpack,
                TTOp::PackReconfig { .. } => Unit::Pack,
                _ => Unit::Math,
            }
        }
        // Recursive hoist over one level. `section` threads the caller
        // position; only section 1 (compute) holds inits.
        fn hoist_level(ops: Vec<TTOp>, compiler: &Compiler, section: &mut u8) -> Result<Vec<TTOp>, BackendError> {
            let mut out: Vec<TTOp> = Vec::with_capacity(ops.len());
            let mut idx = 0;
            while idx < ops.len() {
                match &ops[idx] {
                    TTOp::EndReader | TTOp::EndCompute => {
                        *section += 1;
                        out.push(ops[idx].clone());
                        idx += 1;
                    }
                    TTOp::Loop { trip, .. } if *section == 1 => {
                        // Find the matching end (nesting depth counted).
                        let mut depth = 1;
                        let mut end = idx + 1;
                        while end < ops.len() && depth > 0 {
                            match &ops[end] {
                                TTOp::Loop { .. } => depth += 1,
                                TTOp::EndLoop => depth -= 1,
                                _ => {}
                            }
                            end += 1;
                        }
                        if depth != 0 {
                            return Err(BackendError {
                                status: ErrorStatus::KernelCompilation,
                                context: "tenstorrent2: hoist_dedup_inits: unbalanced loop".into(),
                            });
                        }
                        let open = ops[idx].clone();
                        let body = hoist_level(ops[idx + 1..end - 1].to_vec(), compiler, &mut 1u8)?;
                        let close = ops[end - 1].clone();
                        let single = trip.is_some_and(|t| t >= 1) && !body.iter().any(is_lock);
                        if single {
                            // Distinct init configs per unit, first-occurrence
                            // order. Only STICKY inits hoist: SFPU opcode
                            // config (unary/binary/scalar/fused/cast)
                            // survives every per-call use. Per-call unpacker/
                            // packer config (copy/matmul/transpose/reduce/
                            // bcast inits, pack reconfig) is consumed by each
                            // tile op — legacy re-issues a short form per
                            // trip and never relies on a hoisted one.
                            let sticky = |op: &TTOp| match op {
                                TTOp::BinaryInit { bop } => !matches!(bop, BOp::Mul),
                                TTOp::UnaryInit { .. }
                                | TTOp::BinScalarInit { .. }
                                | TTOp::FusedInit { .. }
                                | TTOp::CastInit { .. }
                                | TTOp::MatmulInit { .. } => true,
                                _ => false,
                            };
                            let mut seen: Vec<TTOp> = Vec::new();
                            for op in &body {
                                if is_init(op) && sticky(op) && !seen.contains(op) {
                                    seen.push(op.clone());
                                }
                            }
                            let mut units: Vec<Unit> = Vec::new();
                            let mut multi = false;
                            for op in &seen {
                                let u = unit(op);
                                if units.contains(&u) {
                                    multi = true;
                                    break;
                                }
                                units.push(u);
                            }
                            if !multi && !seen.is_empty() {
                                out.extend(seen.iter().cloned());
                                out.push(open);
                                out.extend(body.into_iter().filter(|op| !seen.contains(op)));
                                out.push(close);
                            } else {
                                out.push(open);
                                out.extend(body);
                                out.push(close);
                            }
                        } else {
                            out.push(open);
                            out.extend(body);
                            out.push(close);
                        }
                        idx = end;
                    }
                    _ => {
                        out.push(ops[idx].clone());
                        idx += 1;
                    }
                }
            }
            Ok(out)
        }
        let old = std::mem::take(&mut self.ops);
        let mut section = 0u8;
        let hoisted = hoist_level(old, self, &mut section)?;
        // CB descriptor formats for unpack-state tracking, off the
        // `CbDeclare` head.
        let mut cb_format: Map<CBId, u32> = Map::default();
        for op in &hoisted {
            if let TTOp::CbDeclare { cb, format, .. } = op {
                cb_format.insert(*cb, *format);
            }
        }
        // Linear dedup over the hoisted stream.
        let mut next: Vec<TTOp> = Vec::with_capacity(hoisted.len());
        let mut section = 0u8;
        let mut unpack: Option<(CBId, u32)> = None;
        let mut math: Option<TTOp> = None;
        let mut pack: Option<CBId> = None;
        for op in hoisted {
            match &op {
                TTOp::EndReader | TTOp::EndCompute => {
                    section += 1;
                    next.push(op);
                    continue;
                }
                TTOp::EndWriter => {
                    next.push(op);
                    continue;
                }
                _ => {}
            }
            if section != 1 {
                next.push(op);
                continue;
            }
            match op {
                TTOp::MathLock => {
                    unpack = None;
                    math = None;
                    pack = None;
                    next.push(TTOp::MathLock);
                }
                TTOp::PackUnlock => {
                    unpack = None;
                    math = None;
                    pack = None;
                    next.push(TTOp::PackUnlock);
                }
                TTOp::MathUnlock | TTOp::PackLock => next.push(op),
                TTOp::CopyInitWithDt { prev, cb } => {
                    let fmt = cb_format[&cb];
                    if unpack != Some((cb, fmt)) {
                        // An unpacker (re)program invalidates the math
                        // unit's config view: never dedup a math init
                        // across an unpacker init (legacy re-emits both).
                        math = None;
                        next.push(TTOp::CopyInitWithDt { prev, cb });
                        unpack = Some((cb, fmt));
                    }
                }
                TTOp::CopyInit { cb } => {
                    let fmt = cb_format[&cb];
                    if unpack != Some((cb, fmt)) {
                        math = None;
                        next.push(TTOp::CopyInit { cb });
                        unpack = Some((cb, fmt));
                    }
                }
                TTOp::PackReconfig { cb } => {
                    if pack != Some(cb) {
                        next.push(TTOp::PackReconfig { cb });
                        pack = Some(cb);
                    }
                }
                TTOp::BinaryInit { bop: BOp::Mul } => {
                    // Per-call init (legacy re-emits before every
                    // `mul_binary_tile`) and it invalidates the
                    // unpacker view (a copy after a mul re-inits).
                    unpack = None;
                    math = None;
                    next.push(op);
                }
                TTOp::UnaryInit { .. }
                | TTOp::BinaryInit { .. }
                | TTOp::BinScalarInit { .. }
                | TTOp::FusedInit { .. }
                | TTOp::CastInit { .. }
                | TTOp::TransposeInit { .. }
                | TTOp::MatmulInit { .. }
                | TTOp::ReduceInit { .. }
                | TTOp::BcastInit { .. } => {
                    if math.as_ref() != Some(&op) {
                        // A math-unit init invalidates the unpacker config
                        // view unless this init itself programs it (legacy
                        // re-emits copy inits after math inits).
                        unpack = None;
                        // Broadcast/matmul/transpose/reduce inits also
                        // program the unpacker side; keep it in sync.
                        match &op {
                            TTOp::BcastInit { cb_a, .. } => {
                                unpack = Some((*cb_a, cb_format[cb_a]));
                            }
                            TTOp::MatmulInit { a, .. } => {
                                unpack = Some((*a, cb_format[a]));
                            }
                            TTOp::TransposeInit { cb, .. } => {
                                unpack = Some((*cb, cb_format[cb]));
                            }
                            TTOp::ReduceInit { ci, .. } => {
                                unpack = Some((*ci, cb_format[ci]));
                            }
                            _ => {}
                        }
                        next.push(op.clone());
                        math = Some(op);
                    }
                }
                other => next.push(other),
            }
        }
        self.ops = next;
        Ok(())
    }

    /// Verify the fully-physical stream (runs after `tile_regs`, before
    /// render). Structural firewall: whatever the passes did, the final
    /// stream must be launchable. Loud error at the exact op.
    ///
    /// Checks: section termination (any marker prefix, so reader-only
    /// kernels pass; in order, nothing past the last marker), zero `SSA*`
    /// ops (phase invariant), DST lock pairing per section (compute-only),
    /// CB open/close balance per section + pushed==popped program-wide,
    /// declare-before-use (CBs, accessors) and def-before-use (`VarId`,
    /// `TileId`) per section, tile-slot budgets, Loop/If balance.
    fn verify(&mut self) -> Result<(), BackendError> {
        // Section markers seen (bitmask: 1 reader, 2 compute, 4 writer).
        let mut seen = 0u8;
        let mut section = 0usize;
        // DST lock state (compute only).
        #[derive(PartialEq)]
        enum Lock {
            Unlocked,
            Math,
            Pack,
        }
        let mut lock = Lock::Unlocked;
        // CB FIFO: declared set, per-CB (reserved, avail, waited),
        // program-wide (pushed, popped).
        let mut declared: Set<CBId> = Set::default();
        let mut fifo: Map<CBId, (u32, u32, u32)> = Map::default();
        let mut totals: Map<CBId, (u32, u32)> = Map::default();
        // Per-section defined values (cleared at each End*).
        let mut scalars: Set<VarId> = Set::default();
        let mut accessors: Set<u32> = Set::default();
        // Startup triple inputs, same scan as the legacy
        // `generate_compute` (`loaded_order` over compute loads,
        // `stored_first` over compute stores, single-input kernels
        // repeat in0). Fused drains count as loads — they are `Load`
        // ops in the legacy list with no `TileCopy` here.
        let mut loaded_order: Vec<CBId> = Vec::new();
        let mut stored_first: Option<CBId> = None;
        let note_load = |loaded_order: &mut Vec<CBId>, cb: CBId| {
            if !loaded_order.contains(&cb) {
                loaded_order.push(cb);
            }
        };
        let mut max_slot = 0u8;
        let mut depth = 0u32;
        // Loop-trip multiplier for FIFO accounting: sync ops inside a
        // constant-trip loop execute `trip` times, so their reserve/
        // push/wait/pop counts multiply by the enclosing trip product.
        // `0` on the stack marks a symbolic-trip loop; FIFO traffic
        // under it is a compile error (see `TTOp::Loop`).
        let mut mult: u32 = 1;
        let mut loop_trips: Vec<u32> = Vec::new();
        let mut sym_loops = 0u32;
        let end_section = |seen: &mut u8,
                           section: &mut usize,
                           lock: &mut Lock,
                           fifo: &mut Map<CBId, (u32, u32, u32)>,
                           scalars: &mut Set<VarId>,
                           accessors: &mut Set<u32>,
                           depth: &mut u32,
                           marker: u8|
         -> Result<(), BackendError> {
            if !(*seen & marker == 0) {
                return Err(BackendError {
                    status: ErrorStatus::KernelCompilation,
                    context: "tenstorrent2: verify: duplicate section marker".into(),
                });
            }
            *seen |= marker;
            *section += 1;
            if !(*lock == Lock::Unlocked) {
                return Err(BackendError {
                    status: ErrorStatus::KernelCompilation,
                    context: "tenstorrent2: verify: section ends with DST locked".into(),
                });
            }
            for (cb, (reserved, _, waited)) in fifo.iter() {
                if !(*reserved == 0) {
                    return Err(BackendError {
                        status: ErrorStatus::KernelCompilation,
                        context: format!("tenstorrent2: verify: section ends with CB{cb} reserve open").into(),
                    });
                }
                if !(*waited == 0) {
                    return Err(BackendError {
                        status: ErrorStatus::KernelCompilation,
                        context: format!("tenstorrent2: verify: section ends with CB{cb} wait open").into(),
                    });
                }
            }
            if !(*depth == 0) {
                return Err(BackendError {
                    status: ErrorStatus::KernelCompilation,
                    context: "tenstorrent2: verify: section ends inside a walk".into(),
                });
            }
            scalars.clear();
            accessors.clear();
            Ok(())
        };
        for op in self.ops.iter() {
            match op {
                TTOp::EndReader => {
                    end_section(&mut seen, &mut section, &mut lock, &mut fifo, &mut scalars, &mut accessors, &mut depth, 1)?
                }
                TTOp::EndCompute => {
                    if !(seen & 1 != 0) {
                        return Err(BackendError {
                            status: ErrorStatus::KernelCompilation,
                            context: "tenstorrent2: verify: EndCompute without EndReader".into(),
                        });
                    }
                    end_section(&mut seen, &mut section, &mut lock, &mut fifo, &mut scalars, &mut accessors, &mut depth, 2)?;
                }
                TTOp::EndWriter => {
                    if !(seen & 3 != 0) {
                        return Err(BackendError {
                            status: ErrorStatus::KernelCompilation,
                            context: "tenstorrent2: verify: EndWriter without a prior section".into(),
                        });
                    }
                    end_section(&mut seen, &mut section, &mut lock, &mut fifo, &mut scalars, &mut accessors, &mut depth, 4)?;
                }
                _ => {}
            }
            if matches!(op, TTOp::EndReader | TTOp::EndCompute | TTOp::EndWriter) {
                continue;
            }
            if seen == 7 {
                return Err(BackendError {
                    status: ErrorStatus::KernelCompilation,
                    context: "tenstorrent2: verify: op past EndWriter".into(),
                });
            }
            match op {
                TTOp::Loop { trip, .. } => {
                    depth += 1;
                    match trip {
                        Some(t) => {
                            mult *= t;
                            loop_trips.push(*t);
                        }
                        None => {
                            sym_loops += 1;
                            loop_trips.push(0);
                        }
                    }
                }
                TTOp::EndLoop => {
                    if !(depth > 0) {
                        return Err(BackendError {
                            status: ErrorStatus::KernelCompilation,
                            context: "tenstorrent2: verify: EndLoop without Loop".into(),
                        });
                    }
                    depth -= 1;
                    let t = loop_trips.pop().ok_or_else(|| BackendError {
                        status: ErrorStatus::KernelCompilation,
                        context: "tenstorrent2: verify: EndLoop without Loop".into(),
                    })?;
                    if t == 0 {
                        sym_loops -= 1;
                    } else {
                        mult /= t;
                    }
                }
                TTOp::If { .. } => {
                    depth += 1;
                }
                TTOp::EndIf => {
                    if !(depth > 0) {
                        return Err(BackendError {
                            status: ErrorStatus::KernelCompilation,
                            context: "tenstorrent2: verify: EndIf without If".into(),
                        });
                    }
                    depth -= 1;
                }
                TTOp::Arg { .. } | TTOp::Const { .. } | TTOp::TensixGridX { .. } | TTOp::TensixGridY { .. } => {}
                TTOp::Binary { .. } => {}
                TTOp::Cast { .. } => {}
                TTOp::Mad { .. } => {}
                TTOp::Asm { ops: operands, .. } => {
                    for operand in operands {
                        match operand {
                            AsmOperand::Cb(cb) => {
                                if !declared.contains(cb) {
                                    return Err(BackendError {
                                        status: ErrorStatus::KernelCompilation,
                                        context: format!("tenstorrent2: verify: asm on undeclared CB{cb}").into(),
                                    });
                                }
                            }
                            AsmOperand::Tile(_) => {}
                            AsmOperand::Var(_) => {}
                        }
                    }
                }
                TTOp::ReadTile { cb, .. } => {
                    if !declared.contains(cb) {
                        return Err(BackendError {
                            status: ErrorStatus::KernelCompilation,
                            context: format!("tenstorrent2: verify: read on undeclared CB{cb}").into(),
                        });
                    }
                }
                TTOp::WriteTile { cb, .. } => {
                    if !declared.contains(cb) {
                        return Err(BackendError {
                            status: ErrorStatus::KernelCompilation,
                            context: format!("tenstorrent2: verify: write on undeclared CB{cb}").into(),
                        });
                    }
                }
                TTOp::DstMode { .. } => {}
                TTOp::ComputeStartup { in0, in1, out } => {
                    if !declared.contains(in0) {
                        return Err(BackendError {
                            status: ErrorStatus::KernelCompilation,
                            context: format!("tenstorrent2: verify: startup on undeclared CB{in0}").into(),
                        });
                    }
                    if !declared.contains(in1) {
                        return Err(BackendError {
                            status: ErrorStatus::KernelCompilation,
                            context: format!("tenstorrent2: verify: startup on undeclared CB{in1}").into(),
                        });
                    }
                    if !declared.contains(out) {
                        return Err(BackendError {
                            status: ErrorStatus::KernelCompilation,
                            context: format!("tenstorrent2: verify: startup on undeclared CB{out}").into(),
                        });
                    }
                }
                TTOp::Unary { .. } => {}
                TTOp::NocAccessor { ordinal, .. } => {
                    if !accessors.insert(*ordinal) {
                        return Err(BackendError {
                            status: ErrorStatus::KernelCompilation,
                            context: format!("tenstorrent2: verify: duplicate accessor p{ordinal}").into(),
                        });
                    }
                }
                TTOp::NocAddr { z: _, ordinal, .. } => {
                    if !accessors.contains(ordinal) {
                        return Err(BackendError {
                            status: ErrorStatus::KernelCompilation,
                            context: format!("tenstorrent2: verify: address uses undeclared accessor p{ordinal}").into(),
                        });
                    }
                }
                TTOp::AsyncRead { dst_cb, .. } => {
                    if !declared.contains(dst_cb) {
                        return Err(BackendError {
                            status: ErrorStatus::KernelCompilation,
                            context: format!("tenstorrent2: verify: read on undeclared CB{dst_cb}").into(),
                        });
                    }
                }
                TTOp::AsyncWrite { src_cb, .. } => {
                    if !declared.contains(src_cb) {
                        return Err(BackendError {
                            status: ErrorStatus::KernelCompilation,
                            context: format!("tenstorrent2: verify: write on undeclared CB{src_cb}").into(),
                        });
                    }
                }
                TTOp::NocReadBarrier | TTOp::NocWriteBarrier => {}
                TTOp::CbDeclare { cb, .. } => {
                    if !declared.insert(*cb) {
                        return Err(BackendError {
                            status: ErrorStatus::KernelCompilation,
                            context: format!("tenstorrent2: verify: duplicate CB{cb} declaration").into(),
                        });
                    }
                    fifo.insert(*cb, (0, 0, 0));
                    totals.insert(*cb, (0, 0));
                }
                TTOp::ReserveBack { cb, n } => {
                    if sym_loops != 0 {
                        return Err(BackendError {
                            status: ErrorStatus::KernelCompilation,
                            context: "tenstorrent2: verify: FIFO traffic under a symbolic loop".into(),
                        });
                    }
                    if !declared.contains(cb) {
                        return Err(BackendError {
                            status: ErrorStatus::KernelCompilation,
                            context: format!("tenstorrent2: verify: reserve on undeclared CB{cb}").into(),
                        });
                    }
                    let e = fifo.get_mut(cb).ok_or_else(|| BackendError {
                        status: ErrorStatus::KernelCompilation,
                        context: format!("tenstorrent2: verify: reserve on undeclared CB").into(),
                    })?;
                    if !(e.0 == 0 && e.2 == 0) {
                        return Err(BackendError {
                            status: ErrorStatus::KernelCompilation,
                            context: format!("tenstorrent2: verify: reserve on CB{cb} with open transaction").into(),
                        });
                    }
                    e.0 = *n * mult;
                }
                TTOp::PushBack { cb, n } => {
                    if sym_loops != 0 {
                        return Err(BackendError {
                            status: ErrorStatus::KernelCompilation,
                            context: "tenstorrent2: verify: FIFO traffic under a symbolic loop".into(),
                        });
                    }
                    let e = fifo.get_mut(cb).ok_or_else(|| BackendError {
                        status: ErrorStatus::KernelCompilation,
                        context: format!("tenstorrent2: verify: push on undeclared CB").into(),
                    })?;
                    if !(e.0 >= *n * mult) {
                        return Err(BackendError {
                            status: ErrorStatus::KernelCompilation,
                            context: format!("tenstorrent2: verify: push of {n} on CB{cb} with {e:?} reserved").into(),
                        });
                    }
                    e.0 -= *n * mult;
                    e.1 += *n * mult;
                    totals
                        .get_mut(cb)
                        .ok_or_else(|| BackendError {
                            status: ErrorStatus::KernelCompilation,
                            context: format!("tenstorrent2: verify: push on undeclared CB").into(),
                        })?
                        .0 += *n * mult;
                }
                TTOp::WaitFront { cb, m, .. } => {
                    if sym_loops != 0 {
                        return Err(BackendError {
                            status: ErrorStatus::KernelCompilation,
                            context: "tenstorrent2: verify: FIFO traffic under a symbolic loop".into(),
                        });
                    }
                    let e = fifo.get_mut(cb).ok_or_else(|| BackendError {
                        status: ErrorStatus::KernelCompilation,
                        context: format!("tenstorrent2: verify: wait on undeclared CB").into(),
                    })?;
                    if !(e.2 == 0) {
                        return Err(BackendError {
                            status: ErrorStatus::KernelCompilation,
                            context: format!("tenstorrent2: verify: wait on CB{cb} with open wait").into(),
                        });
                    }
                    if !(e.1 >= *m * mult) {
                        return Err(BackendError {
                            status: ErrorStatus::KernelCompilation,
                            context: format!("tenstorrent2: verify: wait of {m} on CB{cb} with {e:?} available").into(),
                        });
                    }
                    e.2 = *m * mult;
                    e.1 -= *m * mult;
                }
                TTOp::PopFront { cb, n } => {
                    if sym_loops != 0 {
                        return Err(BackendError {
                            status: ErrorStatus::KernelCompilation,
                            context: "tenstorrent2: verify: FIFO traffic under a symbolic loop".into(),
                        });
                    }
                    let e = fifo.get_mut(cb).ok_or_else(|| BackendError {
                        status: ErrorStatus::KernelCompilation,
                        context: format!("tenstorrent2: verify: pop on undeclared CB").into(),
                    })?;
                    if !(e.2 >= *n * mult) {
                        return Err(BackendError {
                            status: ErrorStatus::KernelCompilation,
                            context: format!("tenstorrent2: verify: pop of {n} on CB{cb} with {e:?} waited").into(),
                        });
                    }
                    e.2 -= *n * mult;
                    totals
                        .get_mut(cb)
                        .ok_or_else(|| BackendError {
                            status: ErrorStatus::KernelCompilation,
                            context: format!("tenstorrent2: verify: pop on undeclared CB").into(),
                        })?
                        .1 += *n * mult;
                }
                TTOp::MathLock => {
                    if section != 1 {
                        return Err(BackendError {
                            status: ErrorStatus::KernelCompilation,
                            context: "tenstorrent2: verify: DST lock outside compute".into(),
                        });
                    }
                    if !(lock == Lock::Unlocked || lock == Lock::Pack) {
                        return Err(BackendError {
                            status: ErrorStatus::KernelCompilation,
                            context: "tenstorrent2: verify: acquire with DST already held".into(),
                        });
                    }
                    lock = Lock::Math;
                }
                TTOp::MathUnlock => {
                    if !(lock == Lock::Math) {
                        return Err(BackendError {
                            status: ErrorStatus::KernelCompilation,
                            context: "tenstorrent2: verify: commit without MATH lock".into(),
                        });
                    }
                    lock = Lock::Unlocked;
                }
                TTOp::PackLock => {
                    if section != 1 {
                        return Err(BackendError {
                            status: ErrorStatus::KernelCompilation,
                            context: "tenstorrent2: verify: DST lock outside compute".into(),
                        });
                    }
                    if !(lock == Lock::Unlocked) {
                        return Err(BackendError {
                            status: ErrorStatus::KernelCompilation,
                            context: "tenstorrent2: verify: pack wait without release".into(),
                        });
                    }
                    lock = Lock::Pack;
                }
                TTOp::PackUnlock => {
                    if !(lock == Lock::Pack) {
                        return Err(BackendError {
                            status: ErrorStatus::KernelCompilation,
                            context: "tenstorrent2: verify: release without PACK lock".into(),
                        });
                    }
                    lock = Lock::Unlocked;
                }
                TTOp::CopyInit { .. }
                | TTOp::CopyInitWithDt { .. }
                | TTOp::PackReconfig { .. }
                | TTOp::UnaryInit { .. }
                | TTOp::BinaryInit { .. }
                | TTOp::BinScalarInit { .. }
                | TTOp::FusedInit { .. }
                | TTOp::CastInit { .. }
                | TTOp::TransposeInit { .. }
                | TTOp::MatmulInit { .. }
                | TTOp::BcastInit { .. } => {}
                TTOp::ReduceInit { acc, .. } => {
                    max_slot = max_slot.max(acc.0);
                }
                TTOp::ReduceUninit => {}
                TTOp::TileCopy { slot, cb, .. } => {
                    if !declared.contains(cb) {
                        return Err(BackendError {
                            status: ErrorStatus::KernelCompilation,
                            context: format!("tenstorrent2: verify: copy on undeclared CB{cb}").into(),
                        });
                    }
                    max_slot = max_slot.max(slot.0);
                    if section == 1 {
                        note_load(&mut loaded_order, *cb);
                    }
                }
                TTOp::TilePack { cb, .. } => {
                    if !declared.contains(cb) {
                        return Err(BackendError {
                            status: ErrorStatus::KernelCompilation,
                            context: format!("tenstorrent2: verify: pack on undeclared CB{cb}").into(),
                        });
                    }
                    if section == 1 && stored_first.is_none() {
                        stored_first = Some(*cb);
                    }
                }
                TTOp::TileBinary { dst, .. } => {
                    max_slot = max_slot.max(dst.0);
                }
                TTOp::TileUnary { .. } => {}
                TTOp::TileFused { slot, .. } => {
                    max_slot = max_slot.max(slot.0);
                }
                TTOp::TileCast { .. } => {}
                TTOp::TileTranspose { dst, cb, .. } => {
                    if !declared.contains(cb) {
                        return Err(BackendError {
                            status: ErrorStatus::KernelCompilation,
                            context: format!("tenstorrent2: verify: transpose on undeclared CB{cb}").into(),
                        });
                    }
                    max_slot = max_slot.max(dst.0);
                    if section == 1 {
                        note_load(&mut loaded_order, *cb);
                    }
                }
                TTOp::TileMatmul { acc, cb_a, cb_b, .. } => {
                    if !declared.contains(cb_a) {
                        return Err(BackendError {
                            status: ErrorStatus::KernelCompilation,
                            context: format!("tenstorrent2: verify: matmul on undeclared CB{cb_a}").into(),
                        });
                    }
                    if !declared.contains(cb_b) {
                        return Err(BackendError {
                            status: ErrorStatus::KernelCompilation,
                            context: format!("tenstorrent2: verify: matmul on undeclared CB{cb_b}").into(),
                        });
                    }
                    max_slot = max_slot.max(acc.0);
                    if section == 1 {
                        note_load(&mut loaded_order, *cb_a);
                        note_load(&mut loaded_order, *cb_b);
                    }
                }
                TTOp::TileBcastBinary { dst, cb_a, cb_b, .. } => {
                    if !declared.contains(cb_a) {
                        return Err(BackendError {
                            status: ErrorStatus::KernelCompilation,
                            context: format!("tenstorrent2: verify: bcast on undeclared CB{cb_a}").into(),
                        });
                    }
                    if !declared.contains(cb_b) {
                        return Err(BackendError {
                            status: ErrorStatus::KernelCompilation,
                            context: format!("tenstorrent2: verify: bcast on undeclared CB{cb_b}").into(),
                        });
                    }
                    max_slot = max_slot.max(dst.0);
                    if section == 1 {
                        note_load(&mut loaded_order, *cb_a);
                        note_load(&mut loaded_order, *cb_b);
                    }
                }
                TTOp::TileBinScalar { .. } => {}
                TTOp::TileReduce { cb_in, cb_sc, .. } => {
                    if !declared.contains(cb_in) {
                        return Err(BackendError {
                            status: ErrorStatus::KernelCompilation,
                            context: format!("tenstorrent2: verify: reduce on undeclared CB{cb_in}").into(),
                        });
                    }
                    if !declared.contains(cb_sc) {
                        return Err(BackendError {
                            status: ErrorStatus::KernelCompilation,
                            context: format!("tenstorrent2: verify: reduce on undeclared CB{cb_sc}").into(),
                        });
                    }
                    if section == 1 {
                        note_load(&mut loaded_order, *cb_in);
                        note_load(&mut loaded_order, *cb_sc);
                    }
                }
                _ => {
                    return Err(BackendError {
                        status: ErrorStatus::KernelCompilation,
                        context: format!("tenstorrent2: verify: op {op:?} is not fully lowered (SSA remains)").into(),
                    });
                }
            }
        }
        if seen == 0 {
            return Err(BackendError {
                status: ErrorStatus::KernelCompilation,
                context: "tenstorrent2: verify: stream holds no section".into(),
            });
        }
        if !(lock == Lock::Unlocked) {
            return Err(BackendError {
                status: ErrorStatus::KernelCompilation,
                context: "tenstorrent2: verify: stream ends with DST locked".into(),
            });
        }
        if !(depth == 0) {
            return Err(BackendError {
                status: ErrorStatus::KernelCompilation,
                context: "tenstorrent2: verify: stream ends inside a walk".into(),
            });
        }
        for (cb, (pushed, popped)) in totals.iter() {
            if !(pushed == popped) {
                return Err(BackendError {
                    status: ErrorStatus::KernelCompilation,
                    context: format!("tenstorrent2: verify: CB{cb} pushed {pushed} but popped {popped} program-wide").into(),
                });
            }
        }
        let bf16 = self
            .ops
            .iter()
            .find_map(|op| match op {
                TTOp::DstMode { bf16 } => Some(*bf16),
                _ => None,
            })
            .ok_or_else(|| BackendError {
                status: ErrorStatus::KernelCompilation,
                context: "tenstorrent2: verify: stream has no DstMode head".into(),
            })?;
        let budget = if bf16 {
            TileId::BUDGET_BF16 as u8
        } else {
            TileId::BUDGET_FP32 as u8
        };
        if max_slot >= budget {
            return Err(BackendError {
                status: ErrorStatus::KernelCompilation,
                context: format!("tenstorrent2: verify: tile t{max_slot} exceeds the DST budget {budget}").into(),
            });
        }
        // Startup triple, legacy rule: needs a load and a store
        // (pure movement needs no startup); single-input kernels
        // repeat in0. Matmul kernels carry none (`mm_init` owns the
        // long init and replaces startup). Goes in the stream as a
        // compute-front op so render emits it with no scan and no state.
        let has_matmul = self.ops.iter().any(|op| matches!(op, TTOp::TileMatmul { .. }));
        if !has_matmul {
            if let (Some(&in0), Some(out)) = (self.startup_loads.first(), self.startup_store) {
                let in1 = self.startup_loads.get(1).copied().unwrap_or(in0);
                let front = self.ops.iter().position(|op| matches!(op, TTOp::EndReader)).ok_or_else(|| BackendError {
                    status: ErrorStatus::KernelCompilation,
                    context: "tenstorrent2: verify: stream has no reader section".into(),
                })? + 1;
                self.ops.insert(front, TTOp::ComputeStartup { in0, in1, out });
            }
        }
        Ok(())
    }

    /// Render the TTIR stream, one op per line (`v{id}` = scalar
    /// registers, `t{id}` = DST slots, `cb{id}` = circular buffers).
    /// Single walk over `ops`, single emission per op, no scans.
    pub fn render(&self, out: &mut impl std::fmt::Write) -> Result<(), BackendError> {
        use crate::scalar::{bf16, f16};
        let mut section = 0u8;
        let mut indent = [String::from("  "), String::from("  "), String::from("  ")];
        let mut started = [false, false, false];
        let mut next_r = [0u32; 3];
        let mut next_noc = [0u32; 3];
        let mut arg_count = [0u32; 3];
        let mut const_vals: Map<(u8, VarId), String> = Map::default();
        let mut reg: Map<(u8, VarId), u32> = Map::default();
        let mut declared: Set<(u8, VarId)> = Set::default();
        let mut noc_names: Map<(u8, VarId), String> = Map::default();
        let mut arg_idx: Map<(u8, u32), u32> = Map::default();
        let mut acc_prev: [Option<String>; 3] = [None, None, None];
        let mut cb_declares: Vec<CBId> = Vec::new();
        // Fresh `r` slot for a def (a reused VarId keeps its slot and
        // emits a typeless assignment instead of a declaration).
        let def_reg = |reg: &mut Map<(u8, VarId), u32>,
                       next_r: &mut [u32; 3],
                       declared: &mut Set<(u8, VarId)>,
                       s: u8,
                       z: VarId|
         -> (u32, bool) {
            let n = *reg.entry((s, z)).or_insert_with(|| {
                let n = next_r[s as usize];
                next_r[s as usize] += 1;
                n
            });
            (n, declared.insert((s, z)))
        };
        // Operand text: consts inline as literals, regs as `r{n}`.
        let operand = |const_vals: &Map<(u8, VarId), String>,
                       reg: &Map<(u8, VarId), u32>,
                       s: u8,
                       v: VarId|
         -> Result<String, BackendError> {
            if let Some(lit) = const_vals.get(&(s, v)) {
                Ok(lit.clone())
            } else {
                let n = reg.get(&(s, v)).ok_or_else(|| BackendError {
                    status: ErrorStatus::KernelCompilation,
                    context: "tenstorrent2: render: use before def".into(),
                })?;
                Ok(format!("r{n}"))
            }
        };
        // Section-local runtime arg index for a global ordinal.
        let arg_index = |arg_idx: &mut Map<(u8, u32), u32>, arg_count: &mut [u32; 3], s: u8, ord: u32| -> u32 {
            *arg_idx.entry((s, ord)).or_insert_with(|| {
                let a = arg_count[s as usize];
                arg_count[s as usize] += 1;
                a
            })
        };
        for op in &self.ops {
            let s = section;
            let si = s as usize;
            // Section preamble: every section gets its header, even an
            // empty one (legacy `generate_compute` always emits
            // `kernel_main`, so a pure-copy kernel still renders three
            // sections).
            if !started[si] {
                match s {
                    0 => {
                        writeln!(out, "#include <cstdint>")?;
                        writeln!(out, "#include \"api/dataflow/dataflow_api.h\"")?;
                        writeln!(out, "#include \"api/dataflow/noc.h\"")?;
                        writeln!(out, "#include \"api/dataflow/circular_buffer.h\"")?;
                        writeln!(out, "#include \"api/tensor/noc_traits.h\"")?;
                        writeln!(out, "#include \"api/debug/device_print.h\"")?;
                        writeln!(out, "void kernel_main() {{")?;
                    }
                    1 => {
                        writeln!(out, "#include <cstdint>")?;
                        writeln!(out, "#include \"api/compute/common.h\"")?;
                        writeln!(out, "#include \"api/compute/compute_kernel_api.h\"")?;
                        writeln!(out, "#include \"api/compute/eltwise_binary_sfpu.h\"")?;
                        writeln!(out, "#include \"api/compute/eltwise_unary/binop_with_scalar.h\"")?;
                        writeln!(out, "#include \"api/compute/eltwise_unary/left_shift.h\"")?;
                        writeln!(out, "#include \"api/compute/eltwise_unary/right_shift.h\"")?;
                        writeln!(out, "#include \"api/compute/tile_move_copy.h\"")?;
                        writeln!(out, "#include \"api/compute/eltwise_unary/eltwise_unary.h\"")?;
                        writeln!(out, "#include \"api/compute/eltwise_unary/trigonometry.h\"")?;
                        writeln!(out, "#include \"api/compute/eltwise_unary/exp.h\"")?;
                        writeln!(out, "#include \"api/compute/eltwise_unary/recip.h\"")?;
                        writeln!(out, "#include \"api/compute/eltwise_unary/rsqrt.h\"")?;
                        writeln!(out, "#include \"api/compute/eltwise_unary/sqrt.h\"")?;
                        writeln!(out, "#include \"api/compute/eltwise_unary/rounding.h\"")?;
                        writeln!(out, "#include \"api/compute/eltwise_unary/negative.h\"")?;
                        writeln!(out, "#include \"api/compute/eltwise_unary/bitwise_not.h\"")?;
                        writeln!(out, "#include \"api/compute/eltwise_unary/typecast.h\"")?;
                        writeln!(out, "#include \"api/compute/eltwise_unary/logical_not.h\"")?;
                        writeln!(out, "#include \"api/compute/binary_max_min.h\"")?;
                        writeln!(out, "#include \"api/compute/binary_shift.h\"")?;
                        writeln!(out, "#include \"api/compute/eltwise_unary/fill.h\"")?;
                        writeln!(out, "#include \"api/compute/matmul.h\"")?;
                        writeln!(out, "#include \"api/compute/bcast.h\"")?;
                        writeln!(out, "#include \"api/compute/reduce.h\"")?;
                        writeln!(out, "#include \"api/compute/transpose_wh.h\"")?;
                        writeln!(out, "#include \"api/compute/reconfig_data_format.h\"")?;
                        writeln!(out, "#include \"api/dataflow/circular_buffer.h\"")?;
                        writeln!(out, "#include \"api/debug/device_print.h\"")?;
                        writeln!(out, "void kernel_main() {{")?;
                    }
                    _ => {
                        writeln!(out, "#include <cstdint>")?;
                        writeln!(out, "#include \"api/dataflow/dataflow_api.h\"")?;
                        writeln!(out, "#include \"api/dataflow/noc.h\"")?;
                        writeln!(out, "#include \"api/dataflow/circular_buffer.h\"")?;
                        writeln!(out, "#include \"api/tensor/noc_traits.h\"")?;
                        writeln!(out, "#include \"api/debug/dprint.h\"")?;
                        writeln!(out, "void kernel_main() {{")?;
                    }
                }
                for cb in &cb_declares {
                    writeln!(out, "  CircularBuffer cb{cb}(tt::CBIndex::c_{cb});")?;
                }
                started[si] = true;
            }
            let ind = indent[si].clone();
            match op {
                TTOp::EndReader => {
                    if started[0] {
                        writeln!(out, "}}")?;
                    }
                    section = 1;
                }
                TTOp::EndCompute => {
                    if started[1] {
                        writeln!(out, "}}")?;
                    }
                    section = 2;
                }
                TTOp::EndWriter => {
                    if started[2] {
                        writeln!(out, "}}")?;
                    }
                    section = 3;
                }
                TTOp::DstMode { .. } => {}
                TTOp::CbDeclare { cb, .. } => {
                    // Declares sit at the stream head: the reader flushes
                    // them from the pending list at its preamble, later
                    // sections replay the list. A declare landing after
                    // its section started (never happens from lowering)
                    // emits inline to stay loud-safe.
                    if started[si] {
                        writeln!(out, "{ind}CircularBuffer cb{cb}(tt::CBIndex::c_{cb});")?;
                    }
                    if !cb_declares.contains(cb) {
                        cb_declares.push(*cb);
                    }
                }
                TTOp::Loop { len, counter, .. } => {
                    let bound = operand(&const_vals, &reg, s, *len)?;
                    const_vals.remove(&(s, *counter));
                    let (n, fresh_decl) = def_reg(&mut reg, &mut next_r, &mut declared, s, *counter);
                    debug_assert!(fresh_decl, "tenstorrent2: render: loop counter reuses a live register");
                    writeln!(out, "{ind}for (uint32_t r{n} = 0; r{n} < {bound}; r{n}++) {{")?;
                    indent[si] += "  ";
                }
                TTOp::EndLoop => {
                    indent[si].pop();
                    indent[si].pop();
                    writeln!(out, "{ind}}}", ind = indent[si].clone())?;
                }
                TTOp::If { cond } => {
                    let c = operand(&const_vals, &reg, s, *cond)?;
                    writeln!(out, "{ind}if ({c}) {{")?;
                    indent[si] += "  ";
                }
                TTOp::EndIf => {
                    indent[si].pop();
                    indent[si].pop();
                    writeln!(out, "{ind}}}", ind = indent[si].clone())?;
                }
                TTOp::Arg { z, dtype, ordinal } => {
                    let ai = arg_index(&mut arg_idx, &mut arg_count, s, *ordinal);
                    const_vals.remove(&(s, *z));
                    let t = dtype.c_type();
                    let (n, fresh_decl) = def_reg(&mut reg, &mut next_r, &mut declared, s, *z);
                    if fresh_decl {
                        writeln!(out, "{ind}{t} r{n} = ({t})get_arg_val<uint32_t>({ai});")?;
                    } else {
                        writeln!(out, "{ind}r{n} = ({t})get_arg_val<uint32_t>({ai});")?;
                    }
                }
                TTOp::Const { z, value } => {
                    const_vals.insert((s, *z), format!("{}", value.c_code()));
                }
                TTOp::Binary { z, x, y, bop, dtype, .. } => {
                    let xo = operand(&const_vals, &reg, s, *x)?;
                    let yo = operand(&const_vals, &reg, s, *y)?;
                    const_vals.remove(&(s, *z));
                    let t = dtype.c_type();
                    let (n, fresh_decl) = def_reg(&mut reg, &mut next_r, &mut declared, s, *z);
                    let decl = if fresh_decl {
                        format!("{t} r{n} = ")
                    } else {
                        format!("r{n} = ")
                    };
                    match bop {
                        BOp::Add => writeln!(out, "{ind}{decl}{xo} + {yo};")?,
                        BOp::Sub => writeln!(out, "{ind}{decl}{xo} - {yo};")?,
                        BOp::Mul => writeln!(out, "{ind}{decl}{xo} * {yo};")?,
                        BOp::Div => writeln!(out, "{ind}{decl}{xo} / {yo};")?,
                        BOp::Mod => writeln!(out, "{ind}{decl}{xo} % {yo};")?,
                        BOp::Max => writeln!(out, "{ind}{decl}{xo} > {yo} ? {xo} : {yo};")?,
                        BOp::Cmplt => writeln!(out, "{ind}{decl}{xo} < {yo};")?,
                        BOp::Cmpgt => writeln!(out, "{ind}{decl}{xo} > {yo};")?,
                        BOp::Cmpge => writeln!(out, "{ind}{decl}{xo} >= {yo};")?,
                        BOp::Eq => writeln!(out, "{ind}{decl}{xo} == {yo};")?,
                        BOp::NotEq => writeln!(out, "{ind}{decl}{xo} != {yo};")?,
                        BOp::And => writeln!(out, "{ind}{decl}{xo} && {yo};")?,
                        BOp::Or => writeln!(out, "{ind}{decl}{xo} || {yo};")?,
                        BOp::BitXor => writeln!(out, "{ind}{decl}{xo} ^ {yo};")?,
                        BOp::BitOr => writeln!(out, "{ind}{decl}{xo} | {yo};")?,
                        BOp::BitAnd => writeln!(out, "{ind}{decl}{xo} & {yo};")?,
                        BOp::BitShiftLeft => writeln!(out, "{ind}{decl}{xo} << {yo};")?,
                        BOp::BitShiftRight => writeln!(out, "{ind}{decl}{xo} >> {yo};")?,
                        BOp::Pow => {
                            return Err(BackendError {
                                status: ErrorStatus::KernelCompilation,
                                context: "tenstorrent2: render scalar pow".into(),
                            });
                        }
                    }
                }
                TTOp::Unary { .. } => {
                    return Err(BackendError {
                        status: ErrorStatus::KernelCompilation,
                        context: "tenstorrent2: render scalar unary".into(),
                    });
                }
                TTOp::Cast { z, x, dtype, .. } => {
                    let xo = operand(&const_vals, &reg, s, *x)?;
                    const_vals.remove(&(s, *z));
                    let t = dtype.c_type();
                    let (n, fresh_decl) = def_reg(&mut reg, &mut next_r, &mut declared, s, *z);
                    if fresh_decl {
                        writeln!(out, "{ind}{t} r{n} = ({t}){xo};")?;
                    } else {
                        writeln!(out, "{ind}r{n} = ({t}){xo};")?;
                    }
                }
                TTOp::Mad { z, x, y, w, dtype, .. } => {
                    let xo = operand(&const_vals, &reg, s, *x)?;
                    let yo = operand(&const_vals, &reg, s, *y)?;
                    let wo = operand(&const_vals, &reg, s, *w)?;
                    const_vals.remove(&(s, *z));
                    let t = dtype.c_type();
                    let (n, fresh_decl) = def_reg(&mut reg, &mut next_r, &mut declared, s, *z);
                    if fresh_decl {
                        writeln!(out, "{ind}{t} r{n} = {xo} * {yo} + {wo};")?;
                    } else {
                        writeln!(out, "{ind}r{n} = {xo} * {yo} + {wo};")?;
                    }
                }
                TTOp::Asm { .. } => {
                    return Err(BackendError {
                        status: ErrorStatus::KernelCompilation,
                        context: "tenstorrent2: render scalar asm".into(),
                    });
                }
                TTOp::TensixGridX { z, arg, .. } | TTOp::TensixGridY { z, arg, .. } => {
                    const_vals.remove(&(s, *z));
                    let (n, fresh_decl) = def_reg(&mut reg, &mut next_r, &mut declared, s, *z);
                    if fresh_decl {
                        writeln!(out, "{ind}uint32_t r{n} = get_arg_val<uint32_t>({arg});")?;
                    } else {
                        writeln!(out, "{ind}r{n} = get_arg_val<uint32_t>({arg});")?;
                    }
                }
                TTOp::NocAccessor { ordinal, kind, .. } => {
                    let ai = arg_index(&mut arg_idx, &mut arg_count, s, *ordinal);
                    let cta = match acc_prev[si].clone() {
                        None => String::from("0"),
                        Some(prev) => format!("{prev}.next_compile_time_args_offset()"),
                    };
                    let page = TT_DRAM_PAGE_BYTES;
                    match (s, kind) {
                        (0, ParamKind::Global) => {
                            writeln!(out, "{ind}uint32_t src{ordinal} = get_arg_val<uint32_t>({ai});")?;
                            writeln!(out, "{ind}auto args{ordinal} = TensorAccessorArgs<{cta}>({ai});")?;
                            writeln!(out, "{ind}auto p{ordinal} = TensorAccessor(args{ordinal}, src{ordinal}, {page});")?;
                            acc_prev[si] = Some(format!("args{ordinal}"));
                        }
                        (0, ParamKind::GlobalMut) => {
                            writeln!(out, "{ind}uint32_t dst{ordinal} = get_arg_val<uint32_t>({ai});")?;
                            writeln!(out, "{ind}auto args{ordinal} = TensorAccessorArgs<{cta}>({ai});")?;
                            writeln!(out, "{ind}auto p{ordinal} = TensorAccessor(args{ordinal}, dst{ordinal}, {page});")?;
                            acc_prev[si] = Some(format!("args{ordinal}"));
                        }
                        (2, ParamKind::GlobalMut) => {
                            writeln!(out, "{ind}uint32_t out{ordinal} = get_arg_val<uint32_t>({ai});")?;
                            writeln!(out, "{ind}auto args_out{ordinal} = TensorAccessorArgs<{cta}>({ai});")?;
                            writeln!(out, "{ind}auto p_out{ordinal} = TensorAccessor(args_out{ordinal}, out{ordinal}, {page});")?;
                            acc_prev[si] = Some(format!("args_out{ordinal}"));
                        }
                        _ => {
                            return Err(BackendError {
                                status: ErrorStatus::KernelCompilation,
                                context: format!("tenstorrent2: render: accessor {kind:?} in section {s}").into(),
                            });
                        }
                    }
                }
                TTOp::NocAddr { z, ordinal, index, elem_size } => {
                    let idx = operand(&const_vals, &reg, s, *index)?;
                    let page = TT_DRAM_PAGE_BYTES;
                    let k = next_noc[si];
                    next_noc[si] += 1;
                    let (name, acc) = if s == 2 {
                        (format!("wnoc{k}"), format!("p_out{ordinal}"))
                    } else {
                        (format!("rnoc{k}"), format!("p{ordinal}"))
                    };
                    writeln!(
                        out,
                        "{ind}uint64_t {name} = {acc}.get_noc_addr((uint32_t)(({idx}*{elem_size})/{page}), (uint32_t)(({idx}*{elem_size})%{page}));"
                    )?;
                    noc_names.insert((s, *z), name);
                }
                TTOp::ReserveBack { cb, n } => writeln!(out, "{ind}cb{cb}.reserve_back({n});")?,
                TTOp::PushBack { cb, n } => writeln!(out, "{ind}cb{cb}.push_back({n});")?,
                TTOp::WaitFront { cb, m, .. } => writeln!(out, "{ind}cb{cb}.wait_front({m});")?,
                TTOp::PopFront { cb, n } => writeln!(out, "{ind}cb{cb}.pop_front({n});")?,
                TTOp::AsyncRead { addr, dst_cb, bytes, off } => {
                    let an = noc_names
                        .get(&(s, *addr))
                        .ok_or_else(|| BackendError {
                            status: ErrorStatus::KernelCompilation,
                            context: "tenstorrent2: render: read on unnamed addr".into(),
                        })?
                        .clone();
                    if let Some(o) = off {
                        let os = operand(&const_vals, &reg, s, *o)?;
                        writeln!(out, "{ind}noc_async_read({an}, cb{dst_cb}.get_write_ptr() + {os}*{bytes}, {bytes});")?;
                    } else {
                        writeln!(out, "{ind}noc_async_read({an}, cb{dst_cb}.get_write_ptr(), {bytes});")?;
                    }
                }
                TTOp::NocReadBarrier => writeln!(out, "{ind}noc_async_read_barrier();")?,
                TTOp::AsyncWrite { src_cb, addr, bytes, off } => {
                    let an = noc_names
                        .get(&(s, *addr))
                        .ok_or_else(|| BackendError {
                            status: ErrorStatus::KernelCompilation,
                            context: "tenstorrent2: render: write on unnamed addr".into(),
                        })?
                        .clone();
                    if let Some(o) = off {
                        let os = operand(&const_vals, &reg, s, *o)?;
                        writeln!(out, "{ind}noc_async_write(cb{src_cb}.get_read_ptr() + {os}*{bytes}, {an}, {bytes});")?;
                    } else {
                        writeln!(out, "{ind}noc_async_write(cb{src_cb}.get_read_ptr(), {an}, {bytes});")?;
                    }
                }
                TTOp::NocWriteBarrier => writeln!(out, "{ind}noc_async_write_barrier();")?,
                TTOp::MathLock => writeln!(out, "{ind}tile_regs_acquire();")?,
                TTOp::MathUnlock => writeln!(out, "{ind}tile_regs_commit();")?,
                TTOp::PackLock => writeln!(out, "{ind}tile_regs_wait();")?,
                TTOp::PackUnlock => writeln!(out, "{ind}tile_regs_release();")?,
                TTOp::CopyInit { cb } => writeln!(out, "{ind}copy_tile_init({cb});")?,
                TTOp::CopyInitWithDt { prev, cb } => {
                    writeln!(out, "{ind}copy_tile_to_dst_init_short_with_dt({prev}, {cb});")?;
                }
                TTOp::PackReconfig { cb } => writeln!(out, "{ind}pack_reconfig_data_format({cb});")?,
                TTOp::UnaryInit { uop } => writeln!(out, "{ind}{}", unary_init_name(*uop))?,
                TTOp::BinaryInit { bop } => writeln!(
                    out,
                    "{ind}{}",
                    binary_init_name(*bop).ok_or_else(|| BackendError {
                        status: ErrorStatus::KernelCompilation,
                        context: "tenstorrent2: placed binary init without an init call".into(),
                    })?
                )?,
                TTOp::BinScalarInit { bop } => match bop {
                    BOp::Add | BOp::Sub | BOp::Mul | BOp::Div => writeln!(out, "{ind}binop_with_scalar_tile_init();")?,
                    BOp::BitShiftLeft => writeln!(out, "{ind}left_shift_tile_init();")?,
                    BOp::BitShiftRight => writeln!(out, "{ind}right_shift_tile_init();")?,
                    BOp::Pow
                    | BOp::Mod
                    | BOp::Cmplt
                    | BOp::Cmpgt
                    | BOp::Cmpge
                    | BOp::Max
                    | BOp::Or
                    | BOp::And
                    | BOp::BitXor
                    | BOp::BitOr
                    | BOp::BitAnd
                    | BOp::NotEq
                    | BOp::Eq => {
                        return Err(BackendError {
                            status: ErrorStatus::KernelCompilation,
                            context: format!("tenstorrent2: scalar init for {bop:?} has no init call").into(),
                        });
                    }
                },
                TTOp::FusedInit { kind } => writeln!(out, "{ind}{}", kind.init_name())?,
                TTOp::CastInit { in_dtype, out_dtype } => {
                    writeln!(out, "{ind}typecast_tile_init<{}, {}>();", tt_fmt(*in_dtype)?, tt_fmt(*out_dtype)?)?;
                }
                TTOp::TransposeInit { cb, out: cb_out } => writeln!(out, "{ind}transpose_wh_init({cb}, {cb_out});")?,
                TTOp::MatmulInit { a, b, out: cb_out } => writeln!(out, "{ind}mm_init({a}, {b}, {cb_out});")?,
                TTOp::ComputeStartup { in0, in1, out: cb_out } => {
                    writeln!(out, "{ind}compute_kernel_hw_startup({in0}, {in1}, {cb_out});")?
                }
                TTOp::ReduceInit { ci, cs, acc, rop, kind } => {
                    let (op_name, dim_name) = match rop {
                        BOp::Max => ("PoolType::MAX", reduce_dim_name(*kind)),
                        BOp::Add => ("PoolType::SUM", reduce_dim_name(*kind)),
                        _ => {
                            return Err(BackendError {
                                status: ErrorStatus::KernelCompilation,
                                context: format!("tenstorrent2: reduce op {rop:?} has no init call").into(),
                            });
                        }
                    };
                    writeln!(out, "{ind}reduce_init<{op_name}, {dim_name}>({ci}, {cs}, {});", acc.0)?;
                }
                TTOp::ReduceUninit => writeln!(out, "{ind}reduce_uninit();")?,
                TTOp::BcastInit { bop, kind, cb_a, cb_b } => {
                    let Some(init) = bcast_init_name(*bop, *kind) else {
                        return Err(BackendError {
                            status: ErrorStatus::KernelCompilation,
                            context: format!("tenstorrent2: broadcast ({bop:?}, {kind:?}) has no init call").into(),
                        });
                    };
                    writeln!(out, "{ind}{init}({cb_a}, {cb_b});")?;
                }
                TTOp::TileCopy { slot, cb, .. } => {
                    // v1 sync wraps every transaction singly (Reserve/Wait
                    // with n == 1), so the CB slot is always 0 — the
                    // legacy `slot_offset` per_op == 1 rule. The stored
                    // index names the DRAM tile (consumed by the reader
                    // address); it never addresses the CB.
                    writeln!(out, "{ind}copy_tile({cb}, 0, {});", slot.0)?;
                }
                TTOp::TilePack { slot, cb } => {
                    writeln!(out, "{ind}pack_tile({}, {cb});", slot.0)?;
                }
                TTOp::TileBinary { dst, x, y, bop, dtype } => {
                    // Shift LLKs are `template <DataFormat>` over
                    // Int32/UInt32/UInt16 (lowering rejects anything
                    // else); U32 right-shift uses the logical form (the
                    // plain one saturates amounts >= 32 to 31). Every
                    // other binary LLK is untemplated.
                    let tmpl = match bop {
                        BOp::BitShiftLeft | BOp::BitShiftRight => match dtype {
                            DType::I32 => "<DataFormat::Int32>",
                            DType::U32 => "<DataFormat::UInt32>",
                            DType::U16 => "<DataFormat::UInt16>",
                            dt => {
                                return Err(BackendError {
                                    status: ErrorStatus::KernelCompilation,
                                    context: format!("tenstorrent2: tiled shift on {dt:?} has no LLK format").into(),
                                });
                            }
                        },
                        _ => "",
                    };
                    let name = match bop {
                        BOp::Add => "add_binary_tile",
                        BOp::Sub => "sub_binary_tile",
                        BOp::Mul => "mul_binary_tile",
                        BOp::Div => "div_binary_tile",
                        BOp::Max => "binary_max_tile",
                        BOp::BitShiftLeft => "binary_left_shift_tile",
                        BOp::BitShiftRight if matches!(dtype, DType::U32) => "binary_logical_right_shift_tile",
                        BOp::BitShiftRight => "binary_right_shift_tile",
                        _ => {
                            return Err(BackendError {
                                status: ErrorStatus::KernelCompilation,
                                context: format!("tenstorrent2: tiled binary {bop:?} has no LLK call").into(),
                            });
                        }
                    };
                    writeln!(out, "{ind}{name}{tmpl}({}, {}, {});", x.0, y.0, dst.0)?;
                }
                TTOp::TileFused { slot, kind } => {
                    writeln!(out, "{ind}{}({});", kind.call_name(), slot.0)?;
                }
                TTOp::TileUnary { slot, uop } => {
                    // Log2 passes its base scale explicitly (legacy form).
                    if *uop == UOp::Log2 {
                        writeln!(out, "{ind}log_with_base_tile({}, 0x3fb8aa3b);", slot.0)?;
                    } else {
                        let name = match uop {
                            UOp::Neg => "negative_tile",
                            UOp::BitNot => "bitwise_not_tile",
                            UOp::Exp => "exp_tile",
                            UOp::Exp2 => "exp2_tile",
                            UOp::Log2 => {
                                return Err(BackendError {
                                    status: ErrorStatus::KernelCompilation,
                                    context: "tenstorrent2: log2 is emitted above".into(),
                                });
                            }
                            UOp::Reciprocal => "recip_tile",
                            UOp::Sqrt => "sqrt_tile",
                            UOp::Rsqrt => "rsqrt_tile",
                            UOp::Sin => "sin_tile",
                            UOp::Cos => "cos_tile",
                            UOp::Floor => "floor_tile",
                            UOp::Trunc => "trunc_tile",
                            UOp::Abs => "abs_tile",
                            UOp::Not => "logical_not_tile",
                        };
                        writeln!(out, "{ind}{name}({});", slot.0)?;
                    }
                }
                TTOp::TileCast { slot, in_dtype, out_dtype } => {
                    writeln!(out, "{ind}typecast_tile<{}, {}>({});", tt_fmt(*in_dtype)?, tt_fmt(*out_dtype)?, slot.0)?;
                }
                TTOp::TileTranspose { dst, cb, .. } => {
                    writeln!(out, "{ind}transpose_wh_tile({cb}, 0, {});", dst.0)?;
                }
                TTOp::TileMatmul { acc, cb_a, cb_b, .. } => {
                    writeln!(out, "{ind}matmul_tiles({cb_a}, {cb_b}, {}, {}, {});", acc.0, acc.0, acc.0)?;
                }
                TTOp::TileReduce { acc, cb_in, cb_sc, rop, kind } => {
                    let (op_name, dim_name) = match rop {
                        BOp::Max => ("PoolType::MAX", reduce_dim_name(*kind)),
                        BOp::Add => ("PoolType::SUM", reduce_dim_name(*kind)),
                        _ => {
                            return Err(BackendError {
                                status: ErrorStatus::KernelCompilation,
                                context: format!("tenstorrent2: reduce op {rop:?} has no LLK call").into(),
                            });
                        }
                    };
                    writeln!(out, "{ind}reduce_tile<{op_name}, {dim_name}>({cb_in}, {cb_sc}, 0, 0, {});", acc.0)?;
                }
                TTOp::TileBcastBinary { dst, cb_a, cb_b, bop, kind, .. } => {
                    let name = match (bop, kind) {
                        (BOp::Add, TileDim::Row) => "add_tiles_bcast_rows",
                        (BOp::Add, TileDim::Col) => "add_tiles_bcast_cols",
                        (BOp::Add, TileDim::Scalar) => "add_tiles_bcast_scalar",
                        (BOp::Sub, TileDim::Row) => "sub_tiles_bcast_rows",
                        (BOp::Sub, TileDim::Col) => "sub_tiles_bcast_cols",
                        (BOp::Sub, TileDim::Scalar) => "sub_tiles_bcast_scalar",
                        (BOp::Mul, TileDim::Row) => "mul_tiles_bcast_rows",
                        (BOp::Mul, TileDim::Col) => "mul_tiles_bcast_cols",
                        (BOp::Mul, TileDim::Scalar) => "mul_tiles_bcast_scalar",
                        _ => {
                            return Err(BackendError {
                                status: ErrorStatus::KernelCompilation,
                                context: format!("tenstorrent2: broadcast ({bop:?}, {kind:?}) has no LLK call").into(),
                            });
                        }
                    };
                    if matches!(kind, TileDim::Row) {
                        writeln!(out, "{ind}{name}({cb_a}, {cb_b}, 0, 0, {}, 0);", dst.0)?;
                    } else {
                        writeln!(out, "{ind}{name}({cb_a}, {cb_b}, 0, 0, {});", dst.0)?;
                    }
                }
                TTOp::TileBinScalar { slot, bop, value } => {
                    // Immediate shifts (`left/right_shift_tile`) take a
                    // U32 bit count, not fp32 bits.
                    if matches!(bop, BOp::BitShiftLeft | BOp::BitShiftRight) {
                        let Constant::U32(amount) = value else {
                            return Err(BackendError {
                                status: ErrorStatus::KernelCompilation,
                                context: format!("tenstorrent2: render: shift-scalar on non-U32 const {value}").into(),
                            });
                        };
                        let name = if matches!(bop, BOp::BitShiftLeft) {
                            "left_shift_tile"
                        } else {
                            "right_shift_tile"
                        };
                        writeln!(out, "{ind}{name}({}, {amount});", slot.0)?
                    } else {
                        let bits = match value {
                            Constant::F32(b) => f32::from_le_bytes(*b).to_bits(),
                            Constant::F16(b) => f16::from_le_bytes(*b).to_f32().to_bits(),
                            Constant::BF16(b) => bf16::from_le_bytes(*b).to_f32().to_bits(),
                            v => {
                                return Err(BackendError {
                                    status: ErrorStatus::KernelCompilation,
                                    context: format!("tenstorrent2: render: binscalar on non-float const {v}").into(),
                                });
                            }
                        };
                        match bop {
                            BOp::Add => writeln!(out, "{ind}add_unary_tile({}, {bits:#x});", slot.0)?,
                            BOp::Mul => writeln!(out, "{ind}mul_unary_tile({}, {bits:#x});", slot.0)?,
                            BOp::Div => writeln!(out, "{ind}div_unary_tile({}, {bits:#x});", slot.0)?,
                            BOp::Sub => {
                                return Err(BackendError {
                                    status: ErrorStatus::KernelCompilation,
                                    context: "tenstorrent2: render TileBinScalar sub needs operand side".into(),
                                });
                            }
                            _ => {
                                return Err(BackendError {
                                    status: ErrorStatus::KernelCompilation,
                                    context: format!("tenstorrent2: tiled scalar {bop:?} has no LLK call").into(),
                                });
                            }
                        }
                    }
                }
                TTOp::ReadTile { .. } | TTOp::WriteTile { .. } => {
                    return Err(BackendError {
                        status: ErrorStatus::KernelCompilation,
                        context: "tenstorrent2: render: unexpanded movement op (noc_movement bug)".into(),
                    });
                }
            }
        }
        writeln!(out)?;
        Ok(())
    }
}

/// Launch tables for the Tenstorrent backend, built from the TTIR
/// pipeline: section sources plus the CB config, param ordinals,
/// dtypes, and DST mode the backend needs.
pub struct TTProgram {
    /// Reader section source.
    pub(crate) reader_src: String,
    /// Compute section source (empty when the kernel is pure copy).
    pub(crate) compute_src: String,
    /// Writer section source.
    pub(crate) writer_src: String,
    /// Global head-order ordinals of the reader section params.
    pub(crate) reader_params: Vec<u32>,
    /// Global head-order ordinals of the compute section params.
    pub(crate) compute_params: Vec<u32>,
    /// Global head-order ordinals of the writer section params.
    pub(crate) writer_params: Vec<u32>,
    /// Total param count (all kinds, global head order).
    pub(crate) n_params: u32,
    /// Global params in head order (kernel inputs).
    pub(crate) input_dtypes: Vec<DType>,
    /// GlobalMut params in head order (kernel outputs).
    pub(crate) output_dtypes: Vec<DType>,
    /// Runtime CB config: (tt format, tile bytes, tile count) per CB.
    pub(crate) cb_config: Slab<CBId, (u32, u32, u32)>,
    /// True iff the kernel touches F32 tiles (32-bit DST mode).
    pub(crate) fp32: bool,
}

impl Kernel {
    /// Full TTIR codegen returning launch tables: pipeline, render split
    /// at the section boundaries, param/CB tables.
    pub fn generate_tenstorrent(&self) -> Result<TTProgram, BackendError> {
        let mut c = Compiler::new(self)?;
        c.lock_dst()?;
        c.fill_out_cbs()?;
        c.init_math()?;
        c.reconfig_pack();
        c.sync_cbs()?;
        c.dedup_waits();
        c.place_pops()?;
        c.close_reduce_cones()?;
        c.hoist_dedup_inits()?;
        c.noc_movement();
        c.hoist_writer_accessors()?;
        c.batch_cbs();
        c.tile_regs();
        c.verify()?;
        let mut full = String::new();
        c.render(&mut full)?;
        // Split the render at the three `void kernel_main() {` blocks:
        // each section source keeps its own includes.
        let marks: Vec<usize> = full.match_indices("void kernel_main() {").map(|(i, _)| i).collect();
        if marks.len() != 3 {
            return Err(BackendError {
                status: ErrorStatus::KernelCompilation,
                context: format!("tenstorrent2: render holds {} sections, want 3", marks.len()).into(),
            });
        }
        // Back up from each mark over the contiguous `#include` block that
        // precedes it: each section source keeps its whole include block.
        let mut starts = Vec::with_capacity(3);
        for &m in &marks {
            let mut start = m;
            loop {
                let head = &full[..start];
                let Some(inc) = head.rfind("#include") else { break };
                let line = head[..inc].rfind('\n').map(|p| p + 1).unwrap_or(0);
                let gap = &full[line..start];
                if !gap.lines().all(|l| l.starts_with("#include")) {
                    break;
                }
                start = line;
            }
            starts.push(start);
        }
        starts.push(full.len());
        let reader_src = full[starts[0]..starts[1]].to_string();
        let compute_src = full[starts[1]..starts[2]].to_string();
        let writer_src = full[starts[2]..starts[3]].to_string();
        // Param ordinals + input/output dtypes, same walk as legacy `NocEmitter::new`.
        let mut param_ordinal_of: Map<OpId, u32> = Map::default();
        let mut next_param = 0u32;
        let mut input_dtypes: Vec<DType> = Vec::new();
        let mut output_dtypes: Vec<DType> = Vec::new();
        let mut scan = self.head;
        for _ in 0..10_000 {
            if scan.is_null() {
                break;
            }
            if let Op::Param { dtype, kind, .. } = &self.ops[scan].op {
                param_ordinal_of.insert(scan, next_param);
                next_param += 1;
                match kind {
                    ParamKind::Global => input_dtypes.push(*dtype),
                    ParamKind::GlobalMut => output_dtypes.push(*dtype),
                    ParamKind::Variable => {}
                }
            }
            scan = self.next_op(scan);
        }
        // Section param lists in the render's first-use order: the render
        // assigns section-local arg indices the first time an ordinal is
        // emitted (see `arg_index`), so the lists the backend sends must
        // follow exactly that order or rt args misalign with the source.
        let mut section_param_lists: [Vec<u32>; 3] = [Vec::new(), Vec::new(), Vec::new()];
        let mut section = 0usize;
        for op in c.ops.iter() {
            match op {
                TTOp::EndReader => section = 1,
                TTOp::EndCompute => section = 2,
                TTOp::Arg { ordinal, .. }
                | TTOp::NocAccessor { ordinal, .. }
                | TTOp::NocAddr { ordinal, .. }
                | TTOp::ReadTile { ordinal, .. }
                | TTOp::WriteTile { ordinal, .. } => {
                    let list = &mut section_param_lists[section];
                    if !list.contains(ordinal) {
                        list.push(*ordinal);
                    }
                }
                _ => {}
            }
        }
        // Sanity: the first-use set must equal the section's needed params
        // (the `TensixGridX/Y` arg precompute uses the list length).
        for (s, list) in section_param_lists.iter().enumerate() {
            let tt_section = [TtSection::Reader, TtSection::Compute, TtSection::Writer][s];
            let mut ir_set: Vec<u32> = self
                .get_needed_ops(tt_section)?
                .ops
                .iter()
                .copied()
                .filter(|op| matches!(self.ops[*op].op, Op::Param { .. }))
                .map(|p| param_ordinal_of[&p])
                .collect();
            ir_set.sort_unstable();
            let mut used = list.clone();
            used.sort_unstable();
            if used != ir_set {
                return Err(BackendError {
                    status: ErrorStatus::KernelCompilation,
                    context: format!("tenstorrent2: section {s} arg first-use set != needed params").into(),
                });
            }
        }
        let reader_params = section_param_lists[0].clone();
        let compute_params = section_param_lists[1].clone();
        let writer_params = section_param_lists[2].clone();
        // CB ids by first touch (load-then-store per op), same as legacy `CBEmitter::new`.
        let mut map: Map<OpId, CBId> = Map::default();
        let mut next_cb = CBId::ZERO;
        let mut section = TtSection::Reader;
        let mut scan = self.head;
        for _ in 0..10_000 {
            if scan.is_null() {
                break;
            }
            match self.ops[scan].op {
                Op::Barrier => {
                    section = match section {
                        TtSection::Reader => TtSection::Compute,
                        TtSection::Compute => TtSection::Writer,
                        TtSection::Writer => {
                            return Err(BackendError {
                                status: ErrorStatus::KernelCompilation,
                                context: "tenstorrent2: kernels have exactly 3 sections (2 barriers)".into(),
                            });
                        }
                    };
                }
                Op::Load { ref src, .. } => {
                    if let Op::Storage { scope: MemScope::Circular, .. } = self.ops[*src].op {
                        if !map.contains_key(src) {
                            map.insert(*src, next_cb);
                            next_cb.inc();
                        }
                    }
                }
                Op::Store { ref dst, .. } => {
                    if let Op::Storage { scope: MemScope::Circular, .. } = self.ops[*dst].op {
                        if !map.contains_key(dst) {
                            map.insert(*dst, next_cb);
                            next_cb.inc();
                        }
                    }
                }
                _ => {}
            }
            scan = self.next_op(scan);
        }
        let num_circular_buffers = self.device_info().num_circular_buffers;
        if map.len() > num_circular_buffers as usize {
            return Err(BackendError {
                status: ErrorStatus::TooManyCircularBuffers,
                context: format!(
                    "tenstorrent2: kernel needs {} circular buffers, device holds {num_circular_buffers}",
                    map.len()
                )
                .into(),
            });
        }
        let mut cb_ops: Vec<(CBId, OpId)> = map.iter().map(|(&op, &cb)| (cb, op)).collect();
        cb_ops.sort_by_key(|&(cb, _)| cb);
        let mut cb_config: Slab<CBId, (u32, u32, u32)> = Slab::new();
        for (cb, op) in cb_ops {
            let Op::Storage { dtype, len, .. } = &self.ops[op].op else {
                return Err(BackendError {
                    status: ErrorStatus::KernelCompilation,
                    context: format!("tenstorrent2: cb entry {op} is not a storage op").into(),
                });
            };
            let (fmt, tb) = match dtype {
                DType::F32 => (0, 4096),
                DType::F16 => (1, 2048),
                DType::BF16 => (2, 2048),
                DType::U16 => (3, 2048),
                DType::F8E4M3 => (4, 1024),
                DType::U8 => (5, 1024),
                DType::I8 => (6, 1024),
                DType::U32 => (7, 4096),
                DType::I32 => (8, 4096),
                dt => {
                    return Err(BackendError {
                        status: ErrorStatus::KernelCompilation,
                        context: format!("tenstorrent2: CB dtype {dt:?} has no tt format").into(),
                    });
                }
            };
            let pushed = cb_config.push((fmt, tb, (len / 1024) as u32));
            debug_assert_eq!(pushed, cb, "tenstorrent2: CB config out of sync with allocation");
        }
        // DST mode, same scan as legacy `generate_tenstorrent`.
        let mut fp32 = false;
        let mut scan = self.head;
        for _ in 0..10_000 {
            if scan.is_null() {
                break;
            }
            if let Op::Storage { dtype, scope, .. } = self.ops[scan].op {
                match (dtype, scope) {
                    (DType::F32, _) | (DType::F8E4M3, MemScope::Circular) => {
                        fp32 = true;
                        break;
                    }
                    _ => {}
                }
            }
            scan = self.next_op(scan);
        }
        Ok(TTProgram {
            reader_src,
            compute_src,
            writer_src,
            reader_params,
            compute_params,
            writer_params,
            n_params: next_param,
            input_dtypes,
            output_dtypes,
            cb_config,
            fp32,
        })
    }
}

impl Display for Compiler {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        self.render(f).map_err(|e| {
            eprintln!("tenstorrent2: render for display failed: {e}");
            std::fmt::Error
        })
    }
}

impl Kernel {
    /// All ops needed by the stores inside the given section, in IR order,
    /// with their dtypes and section-local refcounts.
    ///
    /// The list holds the section's stores, the transitive closure of
    /// their data dependencies, and the structural ops (loops, branches,
    /// ranges, barriers) lexically inside the section. `dtypes`/`rcs`
    /// mirror [`Kernel::compute_dtypes_and_rcs`] restricted to this set:
    /// refcounts only count uses inside the section.
    pub(crate) fn get_needed_ops(&self, tt_section: TtSection) -> Result<SectionData, BackendError> {
        // Phase 1: stores and structural ops lexically inside the section.
        // Loop/range length operands seed the closure: the section walk
        // references them (r{len}) and they would otherwise dangle.
        let mut section = TtSection::Reader;
        let mut stores: Vec<OpId> = Vec::new();
        let mut starters: Vec<OpId> = Vec::new();
        let mut structural: Set<OpId> = Set::default();
        let mut scan = self.head;
        for _ in 0..10_000 {
            if scan.is_null() {
                break;
            }
            // Barriers always track the section; every other arm carries
            // a guard so the op matches only inside the target section.
            match self.ops[scan].op {
                Op::Barrier => {
                    // Delimiters only: barriers advance the section scan
                    // but never join any section's op list.
                    section.advance()?;
                }
                Op::Store { .. } if section == tt_section => {
                    stores.push(scan);
                }
                Op::Loop { len } if section == tt_section => {
                    structural.insert(scan);
                    starters.push(len);
                }
                Op::Range { kind, .. } if section == tt_section => {
                    structural.insert(scan);
                    match kind {
                        RangeKind::Group(len) | RangeKind::Warp(len) => starters.push(len),
                        RangeKind::Local(_) => {}
                    }
                }
                Op::EndLoop | Op::If { .. } | Op::EndIf if section == tt_section => {
                    structural.insert(scan);
                }
                Op::Asm { ref ops, .. } if section == tt_section => {
                    // Opaque side effect (e.g. `init_sfpu` setup): always
                    // belongs to its lexical section, even with no users.
                    structural.insert(scan);
                    starters.extend(ops.iter().copied());
                }
                _ => {}
            }
            scan = self.next_op(scan);
        }
        if !scan.is_null() {
            return Err(BackendError {
                status: ErrorStatus::KernelCompilation,
                context: "tenstorrent2: get_needed_ops did not finish in 10000 steps".into(),
            });
        }
        // Phase 2: transitive data-dependency closure over the stores.
        let mut needed: Set<OpId> = Set::default();
        let mut stack: Vec<OpId> = starters;
        for &store in &stores {
            needed.insert(store);
            if let Op::Store { dst, src, index, .. } = self.ops[store].op {
                stack.push(dst);
                stack.push(src);
                stack.push(index);
            } else {
                return Err(BackendError {
                    status: ErrorStatus::KernelCompilation,
                    context: "tenstorrent2: get_needed_ops collected a non-store".into(),
                });
            }
        }
        for _ in 0..10_000 {
            let Some(id) = stack.pop() else { break };
            if id.is_null() || needed.contains(&id) {
                continue;
            }
            needed.insert(id);
            match self.ops[id].op {
                Op::Const(_) | Op::Storage { .. } | Op::EndLoop | Op::EndIf | Op::Barrier => {}
                Op::Param { shape, .. } => {
                    stack.push(shape);
                }
                Op::Cast { x, .. } | Op::Bitcast { x, .. } | Op::Unary { x, .. } | Op::BroadcastTile { x, .. } => {
                    stack.push(x);
                }
                Op::Binary { x, y, .. } => {
                    stack.push(x);
                    stack.push(y);
                }
                Op::Stack { ref ops } => {
                    stack.extend(ops.iter().copied());
                }
                Op::Store { dst, src, index, .. } => {
                    stack.push(dst);
                    stack.push(src);
                    stack.push(index);
                }
                Op::Load { src, index, .. } => {
                    stack.push(src);
                    stack.push(index);
                }
                Op::Range { kind, .. } => match kind {
                    RangeKind::Group(len) | RangeKind::Warp(len) => {
                        stack.push(len);
                    }
                    RangeKind::Local(_) => {}
                },
                Op::Loop { len } => {
                    stack.push(len);
                }
                Op::If { condition } => {
                    stack.push(condition);
                }
                Op::Mad { x, y, z } => {
                    stack.push(x);
                    stack.push(y);
                    stack.push(z);
                }
                Op::Index { vec, .. } => {
                    stack.push(vec);
                }
                Op::Wmma { a, b, c, .. } => {
                    stack.push(a);
                    stack.push(b);
                    stack.push(c);
                }
                Op::ReduceTile { x, scaler, acc, .. } => {
                    stack.push(x);
                    stack.push(scaler);
                    stack.push(acc);
                }
                Op::MatmulTile { x, y, acc } => {
                    stack.push(x);
                    stack.push(y);
                    stack.push(acc);
                }
                Op::TransposeTile { x } => {
                    stack.push(x);
                }
                Op::Asm { ref ops, .. } => {
                    stack.extend(ops.iter().copied());
                }
                Op::Move { x, .. } => {
                    stack.push(x);
                }
                Op::Reduce { x, reduce_axis, .. } => {
                    stack.push(x);
                    stack.push(reduce_axis);
                }
            }
        }
        if !stack.is_empty() {
            return Err(BackendError {
                status: ErrorStatus::KernelCompilation,
                context: "tenstorrent2: get_needed_ops closure did not finish in 10000 steps".into(),
            });
        }
        // Phase 3: emit in IR order with dtypes and section-local refcounts.
        let mut ops: Vec<OpId> = Vec::new();
        let mut dtypes: Map<OpId, (DType, MemLayout)> = Map::default();
        let mut rcs: Map<OpId, u32> = Map::default();
        let mut op_id = self.head;
        for _ in 0..10_000 {
            if op_id.is_null() {
                break;
            }
            if needed.contains(&op_id) || structural.contains(&op_id) {
                ops.push(op_id);
                // Every listed op carries a section refcount, even when
                // nothing consumes it (zero uses). A missing entry
                // downstream is a phase-3 bug, never a default.
                rcs.entry(op_id).or_insert(0);
                match self.ops[op_id].op {
                    Op::Move { .. } | Op::Reduce { .. } => {
                        return Err(BackendError {
                            status: ErrorStatus::KernelCompilation,
                            context: "tenstorrent2: get_needed_ops collected a Move/Reduce (never lowered)".into(),
                        });
                    }
                    Op::ReduceTile { x, scaler, acc, .. } => {
                        dtypes.insert(op_id, dtypes[&acc]);
                        *rcs.entry(x).or_insert(0) += 1;
                        *rcs.entry(scaler).or_insert(0) += 1;
                        *rcs.entry(acc).or_insert(0) += 1;
                    }
                    Op::Const(x) => {
                        dtypes.insert(op_id, (x.dtype(), MemLayout::Scalar));
                    }
                    Op::Param { dtype, .. } => {
                        dtypes.insert(op_id, (dtype, MemLayout::Scalar));
                    }
                    Op::Storage { dtype, .. } => {
                        dtypes.insert(op_id, (dtype, MemLayout::Scalar));
                    }
                    Op::Load { src, index, layout } => {
                        dtypes.insert(op_id, (dtypes[&src].0, layout));
                        *rcs.entry(index).or_insert(0) += 1;
                    }
                    Op::Store { dst, src: x, index, layout } => {
                        debug_assert_eq!(dtypes[&x].1, layout);
                        dtypes.insert(op_id, dtypes[&x]);
                        *rcs.entry(dst).or_insert(0) += 1;
                        *rcs.entry(x).or_insert(0) += 1;
                        *rcs.entry(index).or_insert(0) += 1;
                    }
                    Op::Cast { x, dtype } => {
                        dtypes.insert(op_id, (dtype, dtypes[&x].1));
                        *rcs.entry(x).or_insert(0) += 1;
                    }
                    Op::Bitcast { x, dtype } => {
                        dtypes.insert(op_id, (dtype, dtypes[&x].1));
                        *rcs.entry(x).or_insert(0) += 1;
                    }
                    Op::Unary { x, .. } => {
                        dtypes.insert(op_id, dtypes[&x]);
                        *rcs.entry(x).or_insert(0) += 1;
                    }
                    Op::Binary { x, y, bop } => {
                        let dtype = if bop.returns_bool() {
                            (DType::Bool, dtypes[&x].1)
                        } else {
                            dtypes[&x]
                        };
                        dtypes.insert(op_id, dtype);
                        *rcs.entry(x).or_insert(0) += 1;
                        *rcs.entry(y).or_insert(0) += 1;
                    }
                    Op::Asm { ref ops, .. } => {
                        let dtype = dtypes[&ops[0]];
                        dtypes.insert(op_id, dtype);
                        for &x in ops.iter() {
                            *rcs.entry(x).or_insert(0) += 1;
                        }
                    }
                    Op::Stack { ref ops } => {
                        let dtype = dtypes[&ops[0]];
                        dtypes.insert(op_id, (dtype.0, MemLayout::Vector(ops.len().try_into().unwrap())));
                        for &x in ops.iter() {
                            *rcs.entry(x).or_insert(0) += 1;
                        }
                    }
                    Op::Index { vec, idx: _ } => {
                        let dtype = dtypes[&vec];
                        dtypes.insert(op_id, (dtype.0, MemLayout::Scalar));
                        *rcs.entry(vec).or_insert(0) += 1;
                    }
                    Op::Wmma { dims: _, layout: _, dtype, a, b, c } => {
                        let out_dtype = match dtype {
                            MMADType::f16_f16_f16_f32 => DType::F32,
                            MMADType::f16_f16_f16_f16 => DType::F16,
                            MMADType::s8_s8_s32_s32
                            | MMADType::s4_s4_s32_s32
                            | MMADType::b1_b1_s32_xor_popc
                            | MMADType::b1_b1_s32_and_popc => DType::I32,
                        };
                        dtypes.insert(op_id, (out_dtype, MemLayout::Vector(4)));
                        *rcs.entry(a).or_insert(0) += 1;
                        *rcs.entry(b).or_insert(0) += 1;
                        *rcs.entry(c).or_insert(0) += 1;
                    }
                    Op::MatmulTile { x, y, acc } => {
                        dtypes.insert(op_id, dtypes[&acc]);
                        *rcs.entry(x).or_insert(0) += 1;
                        *rcs.entry(y).or_insert(0) += 1;
                        *rcs.entry(acc).or_insert(0) += 1;
                    }
                    Op::TransposeTile { x } => {
                        dtypes.insert(op_id, dtypes[&x]);
                        *rcs.entry(x).or_insert(0) += 1;
                    }
                    Op::BroadcastTile { x, .. } => {
                        dtypes.insert(op_id, dtypes[&x]);
                        *rcs.entry(x).or_insert(0) += 1;
                    }
                    Op::Mad { x, y, z } => {
                        dtypes.insert(op_id, dtypes[&x]);
                        *rcs.entry(x).or_insert(0) += 1;
                        *rcs.entry(y).or_insert(0) += 1;
                        *rcs.entry(z).or_insert(0) += 1;
                    }
                    Op::Range { kind, .. } => {
                        if let RangeKind::Group(len) = kind {
                            *rcs.entry(len).or_insert(0) += 1;
                        }
                        if let RangeKind::Warp(local_id) = kind {
                            *rcs.entry(local_id).or_insert(0) += 1;
                        }
                        dtypes.insert(op_id, (IDX_T, MemLayout::Scalar));
                    }
                    Op::Loop { len, .. } => {
                        *rcs.entry(len).or_insert(0) += 1;
                        dtypes.insert(op_id, (IDX_T, MemLayout::Scalar));
                    }
                    Op::If { condition } => {
                        *rcs.entry(condition).or_insert(0) += 1;
                    }
                    Op::Barrier | Op::EndIf | Op::EndLoop => {}
                }
            }
            op_id = self.next_op(op_id);
        }
        if !op_id.is_null() {
            return Err(BackendError {
                status: ErrorStatus::KernelCompilation,
                context: "tenstorrent2: get_needed_ops did not finish in 10000 steps".into(),
            });
        }
        Ok(SectionData { ops, dtypes, rcs })
    }
}
