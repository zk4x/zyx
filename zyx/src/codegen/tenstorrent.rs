use crate::{
    DType, Map, Set,
    dtype::Constant,
    error::{BackendError, ErrorStatus},
    kernel::{BOp, IDX_T, Kernel, MMADType, MemLayout, MemScope, Op, OpId, ParamKind, RangeKind, TileDim, UOp},
};

use nanoserde::{DeBin, SerBin};
use std::fmt::Write;

use crate::slab::{Slab, SlabId};

/// DRAM buffer page size in bytes: every DRAM `TensorAccessor` strides by
/// the buffer page size, never the dtype tile size.
const TT_DRAM_PAGE_BYTES: u32 = 4096;

/// Circular buffer ID for Tenstorrent codegen v2.
///
/// This is a unique identifier for each circular buffer in the compiled
/// Tenstorrent program.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, SerBin, DeBin)]
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

/// DST slot ID for Tenstorrent codegen v2: one tile register in the
/// compute-core register file. The DST mode is a type-level constant:
/// BF16 mode holds 16 tiles of 32x32, FP32 mode holds 8 (each FP32 tile
/// occupies 2 BF16 slots, pack offsets 0-3 instead of 0-7).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct TileId<const DSTBF16: bool>(pub(crate) u32);

impl<const DSTBF16: bool> TileId<DSTBF16> {
    /// NULL
    pub const NULL: Self = Self(u32::MAX);

    /// Check if this TileId is null.
    pub const fn is_null(self) -> bool {
        self.0 == u32::MAX
    }
}

impl<const DSTBF16: bool> std::fmt::Display for TileId<DSTBF16> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::Display::fmt(&self.0, f)
    }
}

impl<const DSTBF16: bool> From<usize> for TileId<DSTBF16> {
    fn from(value: usize) -> Self {
        TileId(value as u32)
    }
}

impl<const DSTBF16: bool> From<TileId<DSTBF16>> for usize {
    fn from(value: TileId<DSTBF16>) -> usize {
        value.0 as usize
    }
}

impl<const DSTBF16: bool> SlabId for TileId<DSTBF16> {
    const ZERO: Self = Self(0);
    const NULL: Self = Self(u32::MAX);

    /// Advance to the next slot, asserting the DST capacity: 16 tiles in
    /// BF16 mode, 8 in FP32 mode. Over-allocation is a compilation error,
    /// never silent slot reuse.
    fn inc(&mut self) {
        if DSTBF16 {
            assert!(self.0 < 16, "tenstorrent2: DST holds 16 BF16 tiles, over-allocated");
        } else {
            assert!(self.0 < 8, "tenstorrent2: DST holds 8 FP32 tiles, over-allocated");
        }
        self.0 += 1;
    }
}

/// Circular-buffer runtime state, one per CBId in the [`CBEmitter`] slab.
///
/// The cycle is free (`Popped`) --reserve(n)--> `Reserved` --push(n)-->
/// `Pushed` --wait(m)--> `Waiting` --pop(m)--> free (`Popped`). The
/// payloads carry the block transaction: `Reserved` tracks the block
/// size `n` and how many tiles `filled` it so far; `Pushed` tracks
/// `avail`, the tiles pushed minus tiles popped (the consumer-facing
/// FIFO depth); `Waiting` tracks the waited count and how many of
/// those tiles were consumed. `avail` persists across transactions
/// within a section pass: `reserve` from `Pushed` carries the old
/// `avail` forward, and a pop that drains the last waited tile wraps
/// back to `Pushed` while tiles remain. `Popped` is exactly `avail
/// == 0`.
///
/// At every section end all CBs must be settled (`Popped` or `Pushed`);
/// a `Reserved`/`Waiting` remainder is a leaked transaction. The
/// machine models ONE pass over the emitted text (loop bodies execute
/// the same text at runtime); cross-section balance is a pre-pass
/// check (see [`CBBatch`]), not a state-machine property.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CBState {
    /// Free: no live transaction, no unread tiles (`avail == 0`).
    Popped,
    /// `reserve_back(n)` emitted; `filled` tiles of the block written.
    /// `avail` = tiles still unread from earlier transactions.
    Reserved { n: u32, filled: u32, avail: u32 },
    /// Live data: `avail` tiles sit unconsumed at the front.
    Pushed { avail: u32 },
    /// `wait_front(m)` emitted; `filled` of the `m` waited tiles
    /// consumed. `avail` = unconsumed tiles including the waited ones.
    Waiting { waited: u32, filled: u32, avail: u32 },
}

/// Batched CB transaction assigned to one traffic op by the pre-pass
/// (see [`CBEmitter::new`]). A traffic op (reader store, compute load
/// or store, writer store) whose runtime tile count exceeds 1 moves
/// its sync ops out of line: the open (`reserve_back`/`wait_front`)
/// anchors before the enclosing hoist loop's `Loop` op, the close
/// (`push_back`/`pop_front`) after its `EndLoop`, and the data
/// movement addresses tile `index + counter` within the block.
/// Transactions never span a barrier (barriers delimit sections) and
/// the hoist loop's trip count is a compile-time constant.
#[derive(Debug, Clone, Copy)]
struct CBBatch {
    /// The CB this transaction moves tiles on.
    cb: CBId,
    /// Tiles moved per pass over this op's movement text (= the hoist
    /// loop trip count; the transaction text runs once per
    /// enclosing-loop iteration). Feeds the state machine's fill
    /// accounting and the produced/consumed totals.
    per_op: u32,
    /// The hoist loop: open before its `Loop` op, close after its
    /// `EndLoop`. `None` never occurs: every batch is loop-hoisted
    /// (straight-line multi-store runs degenerate to n = 1 and stay
    /// on the legacy inline path).
    hoist: OpId,
}

/// Open/close event anchored at an op: producers reserve/push,
/// consumers wait/pop.
#[derive(Debug, Clone, Copy)]
enum CBEvent {
    Produce { cb: CBId, n: u32 },
    Consume { cb: CBId, n: u32 },
}

impl CBEvent {
    /// The CB this event syncs, for deterministic event ordering.
    fn cb(&self) -> CBId {
        match *self {
            CBEvent::Produce { cb, .. } | CBEvent::Consume { cb, .. } => cb,
        }
    }
}

/// True if the operand closure of `root` contains `target` (e.g. a
/// hoist loop's counter variable appearing in a tile index).
fn deps_contain(kernel: &Kernel, root: OpId, target: OpId) -> bool {
    if root.is_null() {
        return false;
    }
    let mut stack = vec![root];
    for _ in 0..10_000 {
        let Some(id) = stack.pop() else { return false };
        if id == target {
            return true;
        }
        stack.extend(kernel.ops[id].op.parameters());
    }
    panic!("deps_contain did not finish in 10000 steps");
}

/// One CB-movement op: the CB it moves tiles on, the tile-slot index
/// operand, and the kind (produce = store-side traffic, consume =
/// load-side traffic).
#[derive(Debug, Clone, Copy)]
enum TrafficKind {
    /// store_circular: reader/compute stores. Reader slots run ahead
    /// of the FIFO; compute slots pack sequentially.
    Produce,
    /// load_circular: compute copies and writer drains.
    Consume,
}

/// True if every consumer of `load` is a fused tile op: the load
/// drains at the consuming op (legacy 1-tile path), never at the
/// load itself.
fn fused_only_load(kernel: &Kernel, consumers: &Map<OpId, Vec<OpId>>, load: OpId) -> bool {
    match consumers.get(&load) {
        None => false,
        Some(cs) => cs.iter().all(|&c| match kernel.ops[c].op {
            Op::ReduceTile { .. } | Op::MatmulTile { .. } | Op::TransposeTile { .. } | Op::BroadcastTile { .. } => true,
            // A binary with a broadcast-marked side consumes both
            // sides from CBs (fused form, no pre-copies).
            Op::Binary { x, y, .. } => {
                matches!(kernel.ops[x].op, Op::BroadcastTile { .. }) || matches!(kernel.ops[y].op, Op::BroadcastTile { .. })
            }
            _ => false,
        }),
    }
}

/// The CB a fused tile op's side (`x`/`scaler`) reads from: the side
/// must be a tile load from a mapped storage.
fn tile_cb(kernel: &Kernel, side: OpId, map: &Map<OpId, CBId>) -> Option<CBId> {
    let Op::Load { src, .. } = kernel.ops[side].op else {
        return None;
    };
    map.get(&src).copied()
}

/// A side that resolves to a compile-time float constant (follows
/// `Cast`/`Unary`/`Binary` const expressions): folds into a
/// `*_unary_tile` immediate instead of CB traffic.
fn is_const_scalar(kernel: &Kernel, op: OpId) -> bool {
    const_f32_bits(kernel, op).is_some()
}

/// The fp32-bits immediate for a compile-time float constant side.
/// Integer constants are NOT converted (a tile op's scalar lane is
/// float; silent int→float would hide dtype bugs).
fn const_f32_bits(kernel: &Kernel, op: OpId) -> Option<u32> {
    use crate::scalar::{bf16, f16};
    match kernel.resolve_const(op)? {
        Constant::F32(b) => Some(f32::from_le_bytes(b).to_bits()),
        Constant::F16(b) => Some(f16::from_le_bytes(b).to_f32().to_bits()),
        Constant::BF16(b) => Some(bf16::from_le_bytes(b).to_f32().to_bits()),
        _ => None,
    }
}

/// The engine config a compute op programs, if any: the same
/// classification the compute emitter's arms use, shared by the init
/// placement pass (and its cleanliness scan).
fn classify_tile_cfg(
    kernel: &Kernel,
    map: &Map<OpId, CBId>,
    data: &SectionData,
    consumers: &Map<OpId, Vec<OpId>>,
    op: OpId,
) -> Option<Cfg> {
    match kernel.ops[op].op {
        Op::Load { src, layout, .. } if matches!(layout, MemLayout::Tile { .. }) && !fused_only_load(kernel, consumers, op) => {
            map.get(&src).copied().map(Cfg::Copy)
        }
        Op::MatmulTile { x, y, .. } => match (tile_cb(kernel, x, map), tile_cb(kernel, y, map)) {
            (Some(a), Some(b)) => Some(Cfg::Matmul(a, b)),
            _ => None,
        },
        Op::ReduceTile { x, scaler, rop, kind, .. } => match (tile_cb(kernel, x, map), tile_cb(kernel, scaler, map)) {
            (Some(ci), Some(cs)) => Some(Cfg::Reduce(rop, kind, ci, cs)),
            _ => None,
        },
        Op::TransposeTile { x } => tile_cb(kernel, x, map).map(Cfg::Transpose),
        Op::Binary { x, y, bop } if matches!(data.dtypes[&op].1, MemLayout::Tile { .. }) => {
            // A BroadcastTile-marked side fuses: the marker names the
            // broadcast (B) operand, the plain side must be a CB load
            // (the full A tile). Marker on both sides is unsupported.
            let marker = |side: OpId| match kernel.ops[side].op {
                Op::BroadcastTile { x: mx, kind } => Some((kind, tile_cb(kernel, mx, map))),
                _ => None,
            };
            match (marker(x), marker(y)) {
                (Some((kind, Some(cb_b))), None) => tile_cb(kernel, y, map).map(|cb_a| Cfg::Bcast(bop, kind, cb_a, cb_b)),
                (None, Some((kind, Some(cb_b)))) => tile_cb(kernel, x, map).map(|cb_a| Cfg::Bcast(bop, kind, cb_a, cb_b)),
                (None, None) => {
                    // A compile-time-const side folds into the
                    // immediate form (no CB traffic for the scalar).
                    if is_const_scalar(kernel, x) || is_const_scalar(kernel, y) {
                        Some(Cfg::BinScalar)
                    } else {
                        Some(Cfg::Binary(bop))
                    }
                }
                _ => None,
            }
        }
        Op::Unary { uop, .. } if matches!(data.dtypes[&op].1, MemLayout::Tile { .. }) => Some(Cfg::Unary(uop)),
        Op::Cast { x, dtype, .. } if matches!(data.dtypes[&op].1, MemLayout::Tile { .. }) => {
            Some(Cfg::Typecast(data.dtypes[&x].0, dtype))
        }
        _ => None,
    }
}

/// Records one classified traffic op in the batching pre-pass: the
/// traffic kind (for event construction), per-pass tile totals
/// (product of all enclosing trips — symbolic trips are a loud
/// error), and innermost-loop ownership for hoisting.
fn record_traffic(
    op: OpId,
    cb: CBId,
    index: OpId,
    kind: TrafficKind,
    stack: &mut [LoopFrame],
    produced: &mut Map<CBId, u32>,
    consumed: &mut Map<CBId, u32>,
    traffic_kind: &mut Map<OpId, (OpId, TrafficKind)>,
) -> Result<(), BackendError> {
    // Per-pass tile count: the product of every enclosing loop's
    // trip. Symbolic trips cannot bound the CB traffic — a loud
    // error, never a guess.
    let mut count: u32 = 1;
    for f in stack.iter() {
        let Some(t) = f.trip else {
            return Err(BackendError {
                status: ErrorStatus::KernelCompilation,
                context: format!(
                    "tenstorrent2: CB{cb} traffic inside a non-constant loop (op {op}); tile counts must be compile-time constants"
                )
                .into(),
            });
        };
        count *= t;
    }
    match kind {
        TrafficKind::Produce => *produced.get_mut(&cb).expect("CB totals pre-recorded") += count,
        TrafficKind::Consume => *consumed.get_mut(&cb).expect("CB totals pre-recorded") += count,
    }
    traffic_kind.insert(op, (index, kind));
    if let Some(f) = stack.last_mut() {
        if f.traffic.insert(cb, op).is_some() {
            f.dirty.insert(cb);
        }
    }
    Ok(())
}

/// Loop body tracking for the batching pre-pass: direct traffic per
/// CB, branch presence, and the constant trip count. A loop hoists a
/// CB's transaction iff the body holds exactly one traffic op for
/// that CB, no branch, and a constant trip > 1.
struct LoopFrame {
    /// The `Loop` op (open anchor; its counter is the block position).
    op: OpId,
    /// Compile-time trip count; `None` = symbolic (never hoisted, and
    /// any CB traffic under it is a compilation error).
    trip: Option<u32>,
    /// A branch sits in the body: hoisting would emit sync ops for a
    /// body that may not run.
    if_seen: bool,
    /// One traffic op per CB directly in the body (deeper loops own
    /// their own traffic).
    traffic: Map<CBId, OpId>,
    /// CBs with more than one traffic op in the body: no hoist.
    dirty: Set<CBId>,
}

/// DST register-file lock state (MATH/PACK engines), one for ALL tiles.
///
/// Transition matrix (rows old, cols new; Y allowed, X forbidden).
/// No combined MATH+PACK state exists (the hardware allows holding
/// both; the emitter stays simpler without it): a lock take from the
/// other lock is a loud error, never an implicit release. Callers
/// route through `Unlocked` themselves. Unlocking an unlocked file
/// is a loud error too: every lock must pair. Re-taking a held lock
/// emits nothing (lazy keep); the matrix covers emitted transitions
/// only.
///
/// ```text
///              math  pack  unlocked
///   math         X     X       Y
///   pack         X     X       Y
///   unlocked     Y     Y       X
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum TileState {
    /// No engine holds the DST file.
    Unlocked,
    /// MATH holds it (`tile_regs_acquire` taken, not yet committed).
    MathLock,
    /// PACK holds it (`tile_regs_wait` taken, not yet released).
    PackLock,
}

impl TileState {
    /// `tile_regs_acquire`. Only valid from `Unlocked`; asserts it.
    fn math_lock(&mut self) {
        assert!(matches!(self, TileState::Unlocked), "tenstorrent2: tile_regs_acquire from {self:?}, must be Unlocked");
        *self = TileState::MathLock;
    }

    /// `tile_regs_commit`. Only valid from `MathLock`; asserts it.
    fn math_unlock(&mut self) {
        assert!(matches!(self, TileState::MathLock), "tenstorrent2: tile_regs_commit from {self:?}, must be MathLock");
        *self = TileState::Unlocked;
    }

    /// `tile_regs_wait`. Only valid from `Unlocked`; asserts it.
    fn pack_lock(&mut self) {
        assert!(matches!(self, TileState::Unlocked), "tenstorrent2: tile_regs_wait from {self:?}, must be Unlocked");
        *self = TileState::PackLock;
    }

    /// `tile_regs_release`. Only valid from `PackLock`; asserts it.
    fn pack_unlock(&mut self) {
        assert!(matches!(self, TileState::PackLock), "tenstorrent2: tile_regs_release from {self:?}, must be PackLock");
        *self = TileState::Unlocked;
    }
}

/// Engine config identity for init placement: one variant per
/// config-programming tile op. Two executions of the same variant
/// with the same operands share one programmed state. This is the
/// init-placement pass's lattice element — NOT emission state (the
/// emitter reads the placement plan; it keeps no cursor).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum Cfg {
    Unary(UOp),
    Binary(BOp),
    Typecast(DType, DType),
    Matmul(CBId, CBId),
    /// `copy_tile_init(cb)`: shares unpack-A with matmul, reduce, and
    /// transpose, so it is a config kind like any other.
    Copy(CBId),
    Reduce(BOp, TileDim, CBId, CBId),
    Transpose(CBId),
    /// Fused broadcast binary (`add/sub/mul_tiles_bcast_*`): the full
    /// tile comes from `CBId` 0, the broadcast lane tile from `CBId` 1.
    /// Like other CB-consuming configs the init programs the unpacker
    /// for both CBs; unlike the DST-register `Binary` it takes no
    /// pre-copies.
    Bcast(BOp, TileDim, CBId, CBId),
    /// Tile-scalar binary (`*_unary_tile` with an fp32-bits immediate):
    /// one side resolved to a compile-time `Const`. No CB traffic for
    /// the scalar, DST-inplace. The init programs no CBs.
    BinScalar,
}

impl Cfg {
    /// Whether this config's init needs the per-op acc slot: reduce
    /// inits take the DST slot as an argument, which only exists once
    /// emission reaches the op — so reduce inits never hoist and
    /// always emit inline.
    fn needs_acc_slot(self) -> bool {
        matches!(self, Cfg::Reduce(..))
    }
}

/// Scope occupant for the backward init-placement pass is just the
/// engine config kind: packs need no init of their own and tear down
/// nothing the pass merges across (reduce, the only teardown
/// victim, never hoists or merges — its init stays inline at every
/// op — so same-config reduces never collapse).

/// One init call placed by the backward init-placement pass
/// (`Compiler::compute_init_plans`): a single backward walk over the
/// compute ops with a scope stack (`Vec<Map<OpId, PlacedInit>>`,
/// index 0 = the global scope). Every config op carries a
/// provisional entry in its own scope; when a scope closes holding
/// exactly one occupant, its members' provisionals retract into one
/// entry keyed by the boundary op in the parent scope — which counts
/// as that occupant outward, so single-config nests collapse to the
/// top with no fixpoint and no forward state.
#[derive(Clone, Copy, Debug, PartialEq)]
enum PlacedInit {
    /// This op's full init, built with the arm's context at emission.
    Full(Cfg),
    /// Matmul re-entry: `mm_init_short_with_dt(cb_a, cb_b, old)`.
    /// `old` is NOT free: the reconfig only programs the unpack-A
    /// format when `old`'s bookkeeping format differs from the new
    /// one's (the `should_reconfigure_cbs` guard), and the short's AB
    /// init does not program formats at all — so after a
    /// different-format clobberer (copy/reduce/transpose from another
    /// format), a same-format `old` skips the only restore. `old` is
    /// therefore the lowest CB whose format differs from `cb_a`'s
    /// (any such CB trips the guard; the programming uses the new
    /// side only), falling back to `cb_a` when every CB shares the
    /// format — then no clobberer can differ and the skip is safe.
    /// The kernel-top `mm_init` covers the virgin entry.
    MmShort(CBId, CBId, CBId),
    /// Kernel-top full `mm_init` (the `out` arg comes from the
    /// startup triple at prepend time).
    MmTop(CBId, CBId),
}

/// Hoisted init call for a tile unary op. Single table shared by the
/// hoist and inline transition emission — add new ops here once.
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

/// Hoisted init call for a fused broadcast binary (all 0.72 forms
/// are `_init_short`: MATH + unpack mode programming, no format
/// reconfig — uniform-format probes only). `None` means the (op,
/// kind) pair has no LLK (notably every `Div`).
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

/// Hoisted init call for a tile binary op, if it needs one. Single
/// table shared by the hoist and inline transition emission.
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

/// LLK reduce dimension for a tile reduce kind.
fn reduce_dim_name(kind: TileDim) -> &'static str {
    match kind {
        TileDim::Row => "ReduceDim::REDUCE_ROW",
        TileDim::Col => "ReduceDim::REDUCE_COL",
        TileDim::Scalar => "ReduceDim::REDUCE_SCALAR",
    }
}

/// One value slot per section kernel: scalar bound arithmetic, loop
/// bounds, and vector values all live in `r{reg}` slots stamped with
/// their refcount. Tiled values are DST slots instead (layout tiled
/// implies DST) — there is no separate slot struct for them.
#[derive(Clone, Copy, Debug)]
struct VarSlot {
    dtype: DType,
    layout: MemLayout,
    rc: u32,
    scope_level: u8,
}

/// Kernel sections delimited by barriers: reader (head -> 1st barrier),
/// compute (1st -> 2nd), writer (2nd -> end).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum TtSection {
    Reader,
    Compute,
    Writer,
}

impl TtSection {
    /// Step to the next section at a barrier. Panics past the writer:
    /// kernels have exactly 3 sections (2 barriers).
    fn advance(&mut self) {
        *self = match self {
            TtSection::Reader => TtSection::Compute,
            TtSection::Compute => TtSection::Writer,
            TtSection::Writer => {
                panic!("tenstorrent kernels have exactly 3 sections (2 barriers)")
            }
        };
    }
}

pub(crate) enum TTKernel {
    Reader {
        src: String,
        /// Global head-order ordinals: this section's runtime args.
        ordinals: Vec<u32>,
    },
    Compute {
        src: String,
        /// Global head-order ordinals: this section's runtime args.
        ordinals: Vec<u32>,
    },
    Writer {
        src: String,
        /// Global head-order ordinals: this section's runtime args.
        ordinals: Vec<u32>,
    },
    None,
}

impl Kernel {
    /// Generate TT Metalium reader, compute, and writer kernels from zyx IR.
    ///
    /// Common phase first (section op lists, param lists, CB allocation),
    /// then one small emitter per section. Returns the [`Compiler`] with
    /// everything the backend needs: the section sources (each
    /// [`TTKernel`] carrying its param ordinals), the runtime CB config,
    /// and the input/output dtypes.
    #[allow(unused_must_use)]
    pub(crate) fn generate_tenstorrent(&self) -> Result<TTCompiler, BackendError> {
        // DST mode is a type-level constant but only known at runtime:
        // 32-bit iff compute unpacks an F32 tile into DST (F32 CB/acc
        // load in the compute section). Matches the typecast header:
        // any F32 input/output needs 32-bit Dest mode.
        let mut fp32 = false;
        let mut section = TtSection::Reader;
        let mut scan = self.head;
        for _ in 0..10_000 {
            if scan.is_null() {
                break;
            }
            match self.ops[scan].op {
                Op::Barrier => section.advance(),
                Op::Load { src, .. } if section == TtSection::Compute => {
                    if matches!(
                        self.ops[src].op,
                        Op::Storage { dtype: DType::F32, scope: MemScope::Circular | MemScope::Register, .. }
                    ) {
                        fp32 = true;
                        break;
                    }
                }
                _ => {}
            }
            scan = self.next_op(scan);
        }
        if fp32 {
            Ok(TTCompiler::Fp32(Compiler::<false>::generate(self)?))
        } else {
            Ok(TTCompiler::Bf16(Compiler::<true>::generate(self)?))
        }
    }

    /// All ops needed by the stores inside the given section, in IR order,
    /// with their dtypes and section-local refcounts.
    ///
    /// The list holds the section's stores, the transitive closure of
    /// their data dependencies, and the structural ops (loops, branches,
    /// ranges, barriers) lexically inside the section. `dtypes`/`rcs`
    /// mirror [`Kernel::compute_dtypes_and_rcs`] restricted to this set:
    /// refcounts only count uses inside the section.
    fn get_needed_ops(&self, tt_section: TtSection) -> SectionData {
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
                    section.advance();
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
            panic!("get_needed_ops did not finish in 10000 steps");
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
                unreachable!("get_needed_ops collected a non-store");
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
            panic!("get_needed_ops closure did not finish in 10000 steps");
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
                        unreachable!()
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
            panic!("get_needed_ops did not finish in 10000 steps");
        }
        SectionData { ops, dtypes, rcs }
    }
}

/// Closed op list for one section: the section's ops in IR order with
/// their dtypes and section-local refcounts.
struct SectionData {
    /// Ops in IR order.
    ops: Vec<OpId>,
    /// Dtype and layout per op.
    dtypes: Map<OpId, (DType, MemLayout)>,
    /// Refcounts counting uses inside this section only.
    rcs: Map<OpId, u32>,
}

/// Scalar register file for one section: the kernel under codegen
/// and the scalar slots. The section generators own the source text,
/// indent, and scope level as locals; every emitting method takes them
/// as arguments.
struct VarEmitter<'a> {
    /// Kernel under codegen, shared-borrowed for the section's life.
    kernel: &'a Kernel,
    /// Scalars map
    var_map: Map<OpId, u32>,
    /// One entry per emitted value, in emission order.
    vars: Vec<VarSlot>,
}

#[allow(unused_must_use)]
impl<'a> VarEmitter<'a> {
    /// Empty emitter over `kernel`: no registers. Each section
    /// generator owns its source text and indent separately.
    fn new(kernel: &'a Kernel) -> Self {
        Self { kernel, var_map: Map::default(), vars: Vec::new() }
    }

    /// Register name of an already-declared value, no refcount
    /// effect (housekeeping lookups such as hoist loop counters).
    fn var_name(&self, op_id: OpId) -> Option<String> {
        self.var_map.get(&op_id).map(|&r| format!("r{r}"))
    }

    /// Shared index resolution: consts inline as literals, every other
    /// value (including Variable params, slot-named at declaration)
    /// must already sit in `var_map`. Same-level uses decrement the
    /// refcount (house use-site rule).
    fn resolve_idx(
        &mut self,
        kernel: &Kernel,
        _data: &SectionData,
        idx_op: OpId,
        scope_level: u8,
        who: &str,
    ) -> Result<String, BackendError> {
        if let Op::Const(c) = &kernel.ops[idx_op].op {
            return Ok(format!("{}", c.c_code()));
        }
        if let Some(&r) = self.var_map.get(&idx_op) {
            if self.vars[r as usize].scope_level == scope_level {
                debug_assert!(self.vars[r as usize].rc > 0);
                self.vars[r as usize].rc -= 1;
            }
            return Ok(format!("r{r}"));
        }
        Err(BackendError {
            status: ErrorStatus::KernelCompilation,
            context: format!("tenstorrent2: {who} index {idx_op} not in registers").into(),
        })
    }

    /// Shared condition resolution: consts inline, registers resolve
    /// through the section map (no refcount change: conditions stay
    /// live for the branch body).
    fn resolve_cond(&self, kernel: &Kernel, condition: OpId) -> Result<String, BackendError> {
        if let Op::Const(c) = &kernel.ops[condition].op {
            return Ok(format!("{}", c.c_code()));
        }
        if let Some(&r) = self.var_map.get(&condition) {
            return Ok(format!("r{r}"));
        }
        Err(BackendError {
            status: ErrorStatus::KernelCompilation,
            context: format!("tenstorrent2: if condition op {condition} not in registers").into(),
        })
    }

    /// Scalar `Variable` param declaration: slot-named register from
    /// the section's runtime arg (looked up in the [`NocEmitter`]).
    fn declare_variable(
        &mut self,
        src: &mut String,
        indent: &str,
        data: &SectionData,
        noc: &NocEmitter,
        dtype: DType,
        op_id: OpId,
        scope_level: u8,
    ) {
        let arg = noc.arg(op_id, "tenstorrent2 variable param missing from section args");
        let slot = self.vars.len() as u32;
        writeln!(src, "{indent}{} r{slot} = ({})get_arg_val<uint32_t>({arg});", dtype.c_type(), dtype.c_type());
        self.vars.push(VarSlot { dtype, layout: MemLayout::Scalar, rc: data.rcs[&op_id], scope_level });
        self.var_map.insert(op_id, slot);
    }

    /// Loop head: bound resolves like any operand, counter is a fresh
    /// scalar slot.
    fn loop_begin(
        &mut self,
        src: &mut String,
        indent: &mut String,
        kernel: &Kernel,
        data: &SectionData,
        op_id: OpId,
        scope: &mut u8,
    ) -> Result<(), BackendError> {
        let Op::Loop { len } = kernel.ops[op_id].op else {
            unreachable!("tenstorrent2 loop_begin on non-loop op {op_id}");
        };
        let bound = self.resolve_idx(kernel, data, len, *scope, "loop")?;
        let dt = data.dtypes[&op_id].0;
        let rlay = data.dtypes[&op_id].1;
        debug_assert!(matches!(rlay, MemLayout::Scalar));
        let slot = self.vars.len() as u32;
        self.vars.push(VarSlot { dtype: dt, layout: rlay, rc: data.rcs[&op_id], scope_level: *scope });
        self.var_map.insert(op_id, slot);
        writeln!(src, "{indent}for (uint32_t r{slot} = 0; r{slot} < {bound}; r{slot}++) {{");
        indent.push_str("  ");
        *scope += 1;
        Ok(())
    }

    /// Branch head.
    fn if_begin(
        &mut self,
        src: &mut String,
        indent: &mut String,
        kernel: &Kernel,
        condition: OpId,
        scope: &mut u8,
    ) -> Result<(), BackendError> {
        let cond = self.resolve_cond(kernel, condition)?;
        writeln!(src, "{indent}if ({cond}) {{");
        indent.push_str("  ");
        *scope += 1;
        Ok(())
    }
}

/// Loop tail: closes the brace scope.
#[allow(unused_must_use)]
fn loop_end(src: &mut String, indent: &mut String, scope: &mut u8) {
    *scope -= 1;
    indent.pop();
    indent.pop();
    writeln!(src, "{indent}}}");
}

/// Branch tail: closes the brace scope.
#[allow(unused_must_use)]
fn if_end(src: &mut String, indent: &mut String, scope: &mut u8) {
    *scope -= 1;
    indent.pop();
    indent.pop();
    writeln!(src, "{indent}}}");
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
    /// Global params in head order (kernel inputs).
    pub(crate) input_dtypes: Vec<DType>,
    /// GlobalMut params in head order (kernel outputs).
    pub(crate) output_dtypes: Vec<DType>,
    /// Section params in list order: this section's runtime args.
    /// Refilled by `begin_section` for every section.
    arg_pos: Map<OpId, u32>,
    /// Chained accessor for the last declared DRAM param: each new
    /// accessor's compile-time args offset chains off the previous
    /// one. Reset by `begin_section`.
    prev_accessor: Option<String>,
}

#[allow(unused_must_use)]
impl NocEmitter {
    /// Build param state from a kernel: the param ordinals and
    /// input/output dtypes. One walk.
    fn new(kernel: &Kernel) -> Self {
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
            panic!("tenstorrent2 compiler scan did not finish in 10000 steps");
        }
        Self { param_ordinal_of, input_dtypes, output_dtypes, arg_pos: Map::default(), prev_accessor: None }
    }

    /// Start a section: the section's params in list order become its
    /// runtime args, and the accessor chain restarts.
    fn begin_section(&mut self, params: &[OpId]) {
        self.arg_pos.clear();
        for (i, &p) in params.iter().enumerate() {
            self.arg_pos.insert(p, i as u32);
        }
        self.prev_accessor = None;
    }

    /// Runtime arg index for a section param.
    fn arg(&self, op_id: OpId, who: &str) -> u32 {
        self.arg_pos.get(&op_id).copied().expect(who)
    }

    /// Group-index runtime arg: section args first, then one per axis.
    fn group_arg(&self, axis: u32) -> u32 {
        self.arg_pos.len() as u32 + axis
    }

    /// `noc_async_read` of one tile plus its barrier: DRAM address
    /// `rnoc{op_id}` from accessor `p{ld_src}`, then read into the CB
    /// write pointer. `off` is the tile-slot offset within the
    /// reserved block, in tile units ("0" = plain write pointer).
    fn async_read_tile(
        &self,
        src: &mut String,
        indent: &str,
        op_id: OpId,
        ld_src: OpId,
        idx: &str,
        elem_size: u32,
        tile_bytes: u32,
        cb: CBId,
        off: &str,
    ) {
        writeln!(
            src,
            "{indent}uint64_t rnoc{op_id} = p{ld_src}.get_noc_addr((uint32_t)(({idx}*{elem_size})/{TT_DRAM_PAGE_BYTES}), (uint32_t)(({idx}*{elem_size})%{TT_DRAM_PAGE_BYTES}));"
        );
        if off == "0" {
            writeln!(src, "{indent}noc_async_read(rnoc{op_id}, cb{cb}.get_write_ptr(), {tile_bytes});");
        } else {
            writeln!(src, "{indent}noc_async_read(rnoc{op_id}, cb{cb}.get_write_ptr() + {off}*{tile_bytes}, {tile_bytes});");
        }
        writeln!(src, "{indent}noc_async_read_barrier();");
    }

    /// `noc_async_write` of one tile plus its barrier: CB read pointer
    /// to DRAM address `wnoc{op_id}` in accessor `p_out{dst}`. `off`
    /// is the tile-slot offset within the waited block, in tile units
    /// ("0" = plain read pointer).
    fn async_write_tile(
        &self,
        src: &mut String,
        indent: &str,
        op_id: OpId,
        dst: OpId,
        idx: &str,
        elem_size: u32,
        tile_bytes: u32,
        cb: CBId,
        off: &str,
    ) {
        writeln!(
            src,
            "{indent}uint64_t wnoc{op_id} = p_out{dst}.get_noc_addr((uint32_t)(({idx}*{elem_size})/{TT_DRAM_PAGE_BYTES}), (uint32_t)(({idx}*{elem_size})%{TT_DRAM_PAGE_BYTES}));"
        );
        if off == "0" {
            writeln!(src, "{indent}noc_async_write(cb{cb}.get_read_ptr(), wnoc{op_id}, {tile_bytes});");
        } else {
            writeln!(src, "{indent}noc_async_write(cb{cb}.get_read_ptr() + {off}*{tile_bytes}, wnoc{op_id}, {tile_bytes});");
        }
        writeln!(src, "{indent}noc_async_write_barrier();");
    }

    /// Reader `Global` param: DRAM address register plus the chained
    /// `TensorAccessor` (each accessor's compile-time args offset chains
    /// off the previous one).
    fn declare_global(&mut self, src: &mut String, indent: &str, op_id: OpId) {
        let arg = self.arg(op_id, "tenstorrent2 reader param missing from section args");
        writeln!(src, "{indent}uint32_t src{op_id} = get_arg_val<uint32_t>({arg});");
        let cta = match &self.prev_accessor {
            None => String::from("0"),
            Some(prev) => format!("{prev}.next_compile_time_args_offset()"),
        };
        writeln!(src, "{indent}auto args{op_id} = TensorAccessorArgs<{cta}>({arg});");
        writeln!(src, "{indent}auto p{op_id} = TensorAccessor(args{op_id}, src{op_id}, {TT_DRAM_PAGE_BYTES});");
        self.prev_accessor = Some(format!("args{op_id}"));
    }

    /// Reader-side `GlobalMut` param: same accessor shape, `dst` naming.
    fn declare_global_mut(&mut self, src: &mut String, indent: &str, op_id: OpId) {
        let arg = self.arg(op_id, "tenstorrent2 reader param missing from section args");
        writeln!(src, "{indent}uint32_t dst{op_id} = get_arg_val<uint32_t>({arg});");
        let cta = match &self.prev_accessor {
            None => String::from("0"),
            Some(prev) => format!("{prev}.next_compile_time_args_offset()"),
        };
        writeln!(src, "{indent}auto args{op_id} = TensorAccessorArgs<{cta}>({arg});");
        writeln!(src, "{indent}auto p{op_id} = TensorAccessor(args{op_id}, dst{op_id}, {TT_DRAM_PAGE_BYTES});");
        self.prev_accessor = Some(format!("args{op_id}"));
    }

    /// Writer-side `GlobalMut` param: `out`/`args_out`/`p_out` naming.
    fn declare_writer_out(&mut self, src: &mut String, indent: &str, op_id: OpId) {
        let arg = self.arg(op_id, "tenstorrent2 writer param missing from section args");
        writeln!(src, "{indent}uint32_t out{op_id} = get_arg_val<uint32_t>({arg});");
        let cta = match &self.prev_accessor {
            None => String::from("0"),
            Some(prev) => format!("{prev}.next_compile_time_args_offset()"),
        };
        writeln!(src, "{indent}auto args_out{op_id} = TensorAccessorArgs<{cta}>({arg});");
        writeln!(src, "{indent}auto p_out{op_id} = TensorAccessor(args_out{op_id}, out{op_id}, {TT_DRAM_PAGE_BYTES});");
        self.prev_accessor = Some(format!("args_out{op_id}"));
    }

    /// Trailing reader barrier: every async read lands before exit.
    fn final_read_barrier(&self, src: &mut String, indent: &str) {
        writeln!(src, "{indent}noc_async_read_barrier();");
    }
}

/// Circular-buffer traffic emission with per-CB runtime state.
///
/// Owns the CB allocation (one [`CBId`] per Circular storage, shared
/// by all three sections), the runtime CB config, and one [`CBState`]
/// per CB. Each traffic method checks the transition, appends the one
/// call, and advances the state. Every section generator resets its
/// entry states up front and asserts settled states at the end.
pub(crate) struct CBEmitter {
    /// One CBId per Circular storage, shared by all three sections.
    map: Map<OpId, CBId>,
    /// Runtime CB config: (tt format, tile bytes, tile count) per CB.
    /// Format and tile bytes follow the CB storage dtype; an
    /// unmappable dtype is a compilation error, never a silent
    /// default.
    pub(crate) config: Slab<CBId, (u32, u32, u32)>,
    /// Runtime state per CB, in id order.
    states: Slab<CBId, CBState>,
    /// Batched transactions per traffic op (only ops with n > 1; the
    /// rest take the legacy inline path). Built by the pre-pass below.
    batches: Map<OpId, CBBatch>,
    /// Open events (reserve/wait) anchored at an op: emitted before
    /// the op's text. Hoisted transactions anchor at the hoist loop's
    /// `Loop` op.
    opens: Map<OpId, Vec<CBEvent>>,
    /// Close events (push/pop) anchored at an op: emitted after the
    /// op's text. Hoisted transactions anchor at the hoist loop's
    /// `EndLoop` op.
    closes: Map<OpId, Vec<CBEvent>>,
    /// Tiles produced per pass per CB (sum of traffic-op runtime
    /// counts). Every mapped CB has an entry.
    produced: Map<CBId, u32>,
    /// Operand → consumers, from the batching pre-pass: classifies
    /// fused-only loads (compute section).
    consumers: Map<OpId, Vec<OpId>>,
}

#[allow(unused_must_use)]
impl CBEmitter {
    /// Full CB allocation from a kernel: first-touch ids, page/L1
    /// validity, hardware count fit, runtime config. The single point
    /// that assigns CB ids; anything unmappable is a compilation
    /// error. Every mapped CB registers in the state slab (free
    /// state), in id order so the slab index matches the [`CBId`].
    fn new(kernel: &Kernel) -> Result<Self, BackendError> {
        // One walk: first-touch CB ids. Registration order is
        // load-then-store per op, same as before.
        let mut map: Map<OpId, CBId> = Map::default();
        let mut states: Slab<CBId, CBState> = Slab::new();
        let mut next_cb = CBId::ZERO;
        let mut section = TtSection::Reader;
        let mut scan = kernel.head;
        for _ in 0..10_000 {
            if scan.is_null() {
                break;
            }
            match kernel.ops[scan].op {
                Op::Barrier => section.advance(),
                Op::Load { ref src, .. } => {
                    if let Op::Storage { scope: MemScope::Circular, .. } = kernel.ops[*src].op {
                        if !map.contains_key(src) {
                            map.insert(*src, next_cb);
                            let id = states.push(CBState::Popped);
                            debug_assert_eq!(id, next_cb, "tenstorrent2: CBs must register in id order");
                            next_cb.inc();
                        }
                    }
                }
                Op::Store { ref dst, .. } => {
                    if let Op::Storage { scope: MemScope::Circular, .. } = kernel.ops[*dst].op {
                        if !map.contains_key(dst) {
                            map.insert(*dst, next_cb);
                            let id = states.push(CBState::Popped);
                            debug_assert_eq!(id, next_cb, "tenstorrent2: CBs must register in id order");
                            next_cb.inc();
                        }
                    }
                }
                _ => {}
            }
            scan = kernel.next_op(scan);
        }
        if !scan.is_null() {
            panic!("tenstorrent2 cb allocation scan did not finish in 10000 steps");
        }
        // Hardware CB count fit.
        let num_circular_buffers = kernel.device_info().num_circular_buffers;
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
        // Every mapped CB holds whole 2048B pages within the
        // single-core L1 budget.
        for (&storage, &cb) in map.iter() {
            let Op::Storage { dtype, len, .. } = kernel.ops[storage].op else {
                return Err(BackendError {
                    status: ErrorStatus::KernelCompilation,
                    context: format!("tenstorrent2: cb_map entry {storage} is not a storage op").into(),
                });
            };
            let elem = dtype.bit_size() as i64 / 8;
            let bytes = len * elem;
            if len % 1024 != 0 {
                return Err(BackendError {
                    status: ErrorStatus::InvalidCircularBuffer,
                    context: format!("tenstorrent2: CB{cb} holds {len} elements, not whole 1024-element tiles").into(),
                });
            }
            if bytes % 2048 != 0 {
                return Err(BackendError {
                    status: ErrorStatus::InvalidCircularBuffer,
                    context: format!("tenstorrent2: CB{cb} holds {bytes} bytes, not whole 2048B pages").into(),
                });
            }
            if bytes > 32768 {
                return Err(BackendError {
                    status: ErrorStatus::InvalidCircularBuffer,
                    context: format!("tenstorrent2: CB{cb} needs {bytes} bytes, single-core L1 budget is 32768").into(),
                });
            }
        }
        // Runtime config in id order. Format and tile bytes follow the
        // CB storage dtype; anything else is a compilation error.
        let mut cb_ops: Vec<(CBId, OpId)> = map.iter().map(|(&op, &cb)| (cb, op)).collect();
        cb_ops.sort_by_key(|&(cb, _)| cb);
        let mut config: Slab<CBId, (u32, u32, u32)> = Slab::new();
        for (cb, op) in cb_ops {
            let Op::Storage { dtype, len, .. } = &kernel.ops[op].op else {
                unreachable!("tenstorrent2: cb_map entry {op} passed validity but is not a storage op")
            };
            let (fmt, tb) = match dtype {
                DType::F32 => (0, 4096),
                DType::F16 => (1, 2048),
                DType::BF16 => (2, 2048),
                DType::U16 => (3, 2048),
                dt => {
                    return Err(BackendError {
                        status: ErrorStatus::KernelCompilation,
                        context: format!("tenstorrent2: CB dtype {dt:?} has no tt format").into(),
                    });
                }
            };
            let pushed = config.push((fmt, tb, (len / 1024) as u32));
            debug_assert_eq!(pushed, cb, "tenstorrent2: CB config out of sync with allocation");
        }
        // Batching pre-pass: one walk over the whole kernel. Groups CB
        // traffic into block transactions (per [`CBBatch`]) hoisted
        // over their innermost enclosing constant-trip loop, totals
        // the per-pass production/consumption per CB, and asserts the
        // cross-section balance. Consumers come first: a compute load
        // fed only by fused tile ops drains at the consuming op
        // (1-tile legacy path), never at the load itself.
        let mut consumers: Map<OpId, Vec<OpId>> = Map::default();
        let mut scan = kernel.head;
        for _ in 0..10_000 {
            if scan.is_null() {
                break;
            }
            for p in kernel.ops[scan].op.parameters() {
                consumers.entry(p).or_default().push(scan);
            }
            scan = kernel.next_op(scan);
        }
        if !scan.is_null() {
            panic!("tenstorrent2 batching consumer scan did not finish in 10000 steps");
        }
        // Traffic op → (tile-slot index operand, kind).
        let mut traffic_kind: Map<OpId, (OpId, TrafficKind)> = Map::default();
        let mut batches: Map<OpId, CBBatch> = Map::default();
        let mut produced: Map<CBId, u32> = Map::default();
        let mut consumed: Map<CBId, u32> = Map::default();
        for &cb in map.values() {
            produced.insert(cb, 0);
            consumed.insert(cb, 0);
        }
        let mut section = TtSection::Reader;
        let mut stack: Vec<LoopFrame> = Vec::new();
        // Loop op → its EndLoop op (close anchors for hoisted blocks).
        let mut loop_end_of: Map<OpId, OpId> = Map::default();
        let mut scan = kernel.head;
        for _ in 0..10_000 {
            if scan.is_null() {
                break;
            }
            match kernel.ops[scan].op {
                Op::Barrier => {
                    debug_assert!(stack.is_empty(), "tenstorrent2: loop crosses a section barrier");
                    section.advance();
                }
                Op::Loop { len } => {
                    let trip = match kernel.resolve_const(len).and_then(|c| c.as_dim()) {
                        Some(d) if d >= 0 => Some(d as u32),
                        Some(d) => {
                            return Err(BackendError {
                                status: ErrorStatus::KernelCompilation,
                                context: format!("tenstorrent2: negative loop trip count {d}, op {scan}").into(),
                            });
                        }
                        None => None,
                    };
                    stack.push(LoopFrame { op: scan, trip, if_seen: false, traffic: Map::default(), dirty: Set::default() });
                }
                Op::If { .. } => {
                    if let Some(f) = stack.last_mut() {
                        f.if_seen = true;
                    }
                }
                Op::EndLoop => {
                    let f = stack.pop().expect("tenstorrent2: EndLoop without Loop");
                    loop_end_of.insert(f.op, scan);
                    for (&cb, &op) in f.traffic.iter() {
                        if f.dirty.contains(&cb) {
                            continue;
                        }
                        let Some(t) = f.trip else { continue };
                        if t <= 1 {
                            continue;
                        }
                        if t > config[cb].2 {
                            // The CB cannot hold the whole block:
                            // streaming. Keep the per-iteration
                            // single-tile transaction (reserve(1) per
                            // loop pass), which is always correct.
                            continue;
                        }
                        let &(index, _) = &traffic_kind[&op];
                        // Fused-only loads drain at the consuming op
                        // (legacy 1-tile path): they appear in the
                        // frame's traffic map for the balance count,
                        // but carry no provisional batch to upgrade.
                        if !batches.contains_key(&op) {
                            continue;
                        }
                        // Block slot = index + counter; the index must
                        // tie to the counter or be plain 0 (producers
                        // pack sequentially, so the same rule covers
                        // both kinds).
                        if !deps_contain(kernel, index, f.op)
                            && !matches!(kernel.ops[index].op, Op::Const(c) if c.as_dim() == Some(0))
                        {
                            return Err(BackendError {
                                status: ErrorStatus::KernelCompilation,
                                context: format!("tenstorrent2: CB{cb} index op {index} is neither the hoist loop counter nor 0")
                                    .into(),
                            });
                        }
                        // Upgrade the provisional single-tile
                        // transaction to the hoisted block.
                        batches.insert(op, CBBatch { cb, per_op: t, hoist: f.op });
                    }
                }
                Op::Store { ref dst, ref src, index, .. } => {
                    // Classify the traffic op, if any. Deeper loops own
                    // their own traffic: only the innermost frame sees
                    // this op.
                    let traffic = match section {
                        TtSection::Reader | TtSection::Compute => {
                            // Reader/compute stores into a Circular
                            // storage; register accs and DRAM stores
                            // carry no CB traffic.
                            map.get(dst).map(|&cb| (TrafficKind::Produce, cb, index))
                        }
                        TtSection::Writer => {
                            // Writer drain: the store's source load
                            // names the CB and its tile slot.
                            match kernel.ops[*src].op {
                                Op::Load { src: cb_src, index, layout: MemLayout::Tile { .. }, .. } => {
                                    map.get(&cb_src).map(|&cb| (TrafficKind::Consume, cb, index))
                                }
                                _ => None,
                            }
                        }
                    };
                    if let Some((kind, cb, index)) = traffic {
                        record_traffic(scan, cb, index, kind, &mut stack, &mut produced, &mut consumed, &mut traffic_kind)?;
                        // Provisional single-tile transaction: open and
                        // close anchor at this op. A hoisting upgrade
                        // may replace it at the loop's EndLoop. The
                        // slot is always 0 (one tile in flight): the
                        // index must be 0 or tie to an enclosing
                        // counter (its DRAM-side meaning does not
                        // affect the CB slot).
                        let counter_tied = stack.iter().any(|f| deps_contain(kernel, index, f.op));
                        if !matches!(kernel.ops[index].op, Op::Const(c) if c.as_dim() == Some(0)) && !counter_tied {
                            return Err(BackendError {
                                status: ErrorStatus::KernelCompilation,
                                context: format!(
                                    "tenstorrent2: single-tile transaction on CB{cb} (op {scan}) needs slot index 0 or the loop counter, got op {index}"
                                )
                                .into(),
                            });
                        }
                        batches.insert(scan, CBBatch { cb, per_op: 1, hoist: scan });
                    }
                }
                Op::Load { ref src, index, ref layout, .. } => {
                    // Compute tile loads from a Circular storage. Deeper
                    // loops own their own traffic: only the innermost
                    // frame sees this op. No `continue` here: every arm
                    // falls through to the walk advance.
                    let mut traffic: Option<(TrafficKind, CBId, OpId)> = None;
                    let mut fused_drain = false;
                    if section == TtSection::Compute && matches!(layout, MemLayout::Tile { .. }) {
                        if let Some(&cb) = map.get(src) {
                            if fused_only_load(kernel, &consumers, scan) {
                                // Fused tile ops drain at a fixed slot
                                // (0): any other index would be
                                // silently dropped. The load still
                                // counts as consumption (recorded
                                // below); it stays on the fused op's
                                // legacy 1-tile path, so no batch.
                                if !matches!(kernel.ops[index].op, Op::Const(c) if c.as_dim() == Some(0)) {
                                    return Err(BackendError {
                                        status: ErrorStatus::KernelCompilation,
                                        context: format!(
                                            "tenstorrent2: fused consumer of CB{cb} drains at slot 0; load index op {index} must be 0"
                                        )
                                        .into(),
                                    });
                                }
                                fused_drain = true;
                                traffic = Some((TrafficKind::Consume, cb, index));
                            } else {
                                traffic = Some((TrafficKind::Consume, cb, index));
                            }
                        }
                    }
                    if let Some((kind, cb, index)) = traffic {
                        record_traffic(scan, cb, index, kind, &mut stack, &mut produced, &mut consumed, &mut traffic_kind)?;
                        if !fused_drain {
                            // Provisional single-tile transaction: open
                            // and close anchor at this op. A hoisting
                            // upgrade may replace it at the loop's
                            // EndLoop. The slot is always 0 (one tile
                            // in flight): the index must be 0 or tie
                            // to the enclosing counter (its DRAM-side
                            // meaning does not affect the CB slot).
                            let counter_tied = stack.iter().any(|f| deps_contain(kernel, index, f.op));
                            if !matches!(kernel.ops[index].op, Op::Const(c) if c.as_dim() == Some(0)) && !counter_tied {
                                return Err(BackendError {
                                    status: ErrorStatus::KernelCompilation,
                                    context: format!(
                                        "tenstorrent2: single-tile transaction on CB{cb} (op {scan}) needs slot index 0 or the loop counter, got op {index}"
                                    )
                                    .into(),
                                });
                            }
                            batches.insert(scan, CBBatch { cb, per_op: 1, hoist: scan });
                        }
                    }
                }
                _ => {}
            }
            scan = kernel.next_op(scan);
        }
        if !scan.is_null() {
            panic!("tenstorrent2 batching pre-pass did not finish in 10000 steps");
        }
        // Anchor events from the final batches: single-tile
        // transactions open and close at the traffic op itself;
        // hoisted blocks open at the hoist loop's `Loop` op and close
        // at its `EndLoop`. Sorted by CB for deterministic text.
        let mut opens: Map<OpId, Vec<CBEvent>> = Map::default();
        let mut closes: Map<OpId, Vec<CBEvent>> = Map::default();
        let mut anchored: Vec<(OpId, OpId, CBEvent)> = Vec::new();
        for (&op, &b) in batches.iter() {
            let event = match traffic_kind[&op].1 {
                TrafficKind::Produce => CBEvent::Produce { cb: b.cb, n: b.per_op },
                TrafficKind::Consume => CBEvent::Consume { cb: b.cb, n: b.per_op },
            };
            if b.hoist == op {
                anchored.push((op, op, event));
            } else {
                anchored.push((b.hoist, loop_end_of[&b.hoist], event));
            }
        }
        anchored.sort_by_key(|&(open, close, ref e)| (open, close, e.cb()));
        for (open, close, event) in anchored {
            opens.entry(open).or_default().push(event);
            closes.entry(close).or_default().push(event);
        }
        // Cross-section balance: every tile produced per pass is
        // consumed per pass, per CB. This is the machine-checked
        // producer/consumer correspondence.
        for (&cb, &p) in produced.iter() {
            let c = consumed[&cb];
            if p != c {
                let Op::Storage { len, .. } =
                    kernel.ops[map.iter().find_map(|(&op, &cid)| (cid == cb).then_some(op)).expect("CB registered")].op
                else {
                    unreachable!("tenstorrent2: CB map entry is a storage op")
                };
                return Err(BackendError {
                    status: ErrorStatus::KernelCompilation,
                    context: format!(
                        "tenstorrent2: CB over {}-tile storage produces {p} but consumes {c} tiles per pass",
                        len / 1024
                    )
                    .into(),
                });
            }
        }
        Ok(Self { map, config, states, batches, opens, closes, produced, consumers })
    }

    /// Reset every CB to `state` (section entry).
    fn reset_all(&mut self, state: CBState) {
        let ids: Vec<CBId> = self.states.ids().collect();
        for id in ids {
            self.states[id] = state;
        }
    }

    /// Reset one CB to `state` (compute/writer input/output partition).
    fn set(&mut self, cb: CBId, state: CBState) {
        self.states[cb] = state;
    }

    /// End-of-section gate: no CB may hold a mid-transaction state.
    /// `Reserved`/`Waiting` remainders are leaked transactions.
    fn assert_settled(&self, section: &str) {
        for id in self.states.ids() {
            assert!(
                matches!(self.states[id], CBState::Popped | CBState::Pushed { .. }),
                "tenstorrent2: {section} ends with CB{id} in {:?}, mid-transaction states must not escape a section",
                self.states[id]
            );
        }
    }

    /// Declare every shared CB (matches v1 output).
    fn declare_all(&self, src: &mut String, indent: &str) {
        let mut cbs: Vec<CBId> = self.states.ids().collect();
        cbs.sort();
        for cb in cbs {
            writeln!(src, "{indent}CircularBuffer cb{cb}(tt::CBIndex::c_{cb});");
        }
    }

    /// `cb.reserve_back(n)`: from the free state or stacked on unread
    /// tiles (`Pushed` — the ring wraps, the reader runs ahead). The
    /// block size was validated against the CB capacity by the
    /// pre-pass.
    fn reserve_back(&mut self, src: &mut String, indent: &str, cb: CBId, n: u32) {
        let avail = match self.states[cb] {
            CBState::Popped => 0,
            CBState::Pushed { avail } => avail,
            s => panic!("tenstorrent2: reserve_back on CB{cb} in {s:?}, must be free or pushed"),
        };
        writeln!(src, "{indent}cb{cb}.reserve_back({n});");
        self.states[cb] = CBState::Reserved { n, filled: 0, avail };
    }

    /// `cb.push_back(n)`: closes the open reservation; the block must
    /// be fully filled (`record_move` accounted every tile).
    fn push_back(&mut self, src: &mut String, indent: &str, cb: CBId, n: u32) {
        match self.states[cb] {
            CBState::Reserved { n: rn, filled, avail } if rn == n => {
                assert_eq!(filled, n, "tenstorrent2: push_back on CB{cb} with {filled}/{n} tiles filled");
                writeln!(src, "{indent}cb{cb}.push_back({n});");
                self.states[cb] = CBState::Pushed { avail: avail + n };
            }
            s => panic!("tenstorrent2: push_back on CB{cb} in {s:?}, must be Reserved with matching block size"),
        }
    }

    /// `cb.wait_front(m)`: waits for at least `m` unread tiles; `m`
    /// must be available (`avail >= m`).
    fn wait_front(&mut self, src: &mut String, indent: &str, cb: CBId, m: u32) {
        match self.states[cb] {
            CBState::Pushed { avail } => {
                assert!(m <= avail, "tenstorrent2: wait_front({m}) on CB{cb} with only {avail} tiles available");
                writeln!(src, "{indent}cb{cb}.wait_front({m});");
                self.states[cb] = CBState::Waiting { waited: m, filled: 0, avail };
            }
            s => panic!("tenstorrent2: wait_front on CB{cb} in {s:?}, must be Pushed"),
        }
    }

    /// `cb.pop_front(m)`: drains the waited block; tiles left over
    /// wrap back to `Pushed` (`avail` minus the popped ones), an
    /// empty front wraps to free.
    fn pop_front(&mut self, src: &mut String, indent: &str, cb: CBId, m: u32) {
        match self.states[cb] {
            CBState::Waiting { waited, filled, avail } if waited == m => {
                assert_eq!(filled, m, "tenstorrent2: pop_front on CB{cb} with {filled}/{m} tiles consumed");
                writeln!(src, "{indent}cb{cb}.pop_front({m});");
                let rest = avail - m;
                self.states[cb] = if rest == 0 {
                    CBState::Popped
                } else {
                    CBState::Pushed { avail: rest }
                };
            }
            s => panic!("tenstorrent2: pop_front on CB{cb} in {s:?}, must be Waiting with matching block size"),
        }
    }

    /// Accounts one data-movement text execution: fills the open
    /// reservation (producer) or drains the waited block (consumer)
    /// by `count` tiles. The pre-pass supplies the runtime count of
    /// the movement text (hoist trip for hoisted ops, 1 otherwise).
    fn record_move(&mut self, cb: CBId, count: u32) {
        match &mut self.states[cb] {
            CBState::Reserved { n, filled, .. } => {
                *filled += count;
                debug_assert!(*filled <= *n, "tenstorrent2: CB{cb} block overfill {}/{}", filled, n);
            }
            CBState::Waiting { waited, filled, .. } => {
                *filled += count;
                debug_assert!(*filled <= *waited, "tenstorrent2: CB{cb} block overdrain {}/{}", filled, waited);
            }
            s => panic!("tenstorrent2: record_move on CB{cb} in {s:?}, must be Reserved or Waiting"),
        }
    }

    /// Emits the open events (reserve/wait) anchored at `op_id`,
    /// before the op's text.
    fn open_events(&mut self, src: &mut String, indent: &str, op_id: OpId) {
        if let Some(events) = self.opens.get(&op_id).cloned() {
            for event in events {
                match event {
                    CBEvent::Produce { cb, n } => self.reserve_back(src, indent, cb, n),
                    CBEvent::Consume { cb, n } => self.wait_front(src, indent, cb, n),
                }
            }
        }
    }

    /// Emits the close events (push/pop) anchored at `op_id`, after
    /// the op's text.
    fn close_events(&mut self, src: &mut String, indent: &str, op_id: OpId) {
        if let Some(events) = self.closes.get(&op_id).cloned() {
            for event in events {
                match event {
                    CBEvent::Produce { cb, n } => self.push_back(src, indent, cb, n),
                    CBEvent::Consume { cb, n } => self.pop_front(src, indent, cb, n),
                }
            }
        }
    }

    /// The batch assigned to a traffic op, if any.
    fn batch(&self, op_id: OpId) -> Option<CBBatch> {
        self.batches.get(&op_id).copied()
    }

    /// Tiles produced per pass for `cb` (entry `avail` for consuming
    /// sections). The pre-pass records every mapped CB.
    fn produced(&self, cb: CBId) -> u32 {
        self.produced.get(&cb).copied().expect("tenstorrent2: pre-pass recorded totals for every CB")
    }

    /// Tile-slot address expression for a batched movement, in tile
    /// units. Single-tile transactions sit at slot 0 (the pre-pass
    /// rejects any other index). Hoisted blocks: the IR index if it is
    /// tied to the hoist loop's counter (it spans the block by
    /// itself), otherwise the counter alone (index is plain 0).
    fn slot_offset(
        &self,
        kernel: &Kernel,
        em: &mut VarEmitter,
        data: &SectionData,
        idx_op: OpId,
        batch: CBBatch,
        scope_level: u8,
        who: &str,
    ) -> Result<String, BackendError> {
        if batch.per_op == 1 {
            return Ok(String::from("0"));
        }
        if deps_contain(kernel, idx_op, batch.hoist) {
            return em.resolve_idx(kernel, data, idx_op, scope_level, who);
        }
        Ok(em.var_name(batch.hoist).expect("tenstorrent2: hoist loop counter not in registers"))
    }
}

/// DST tile emission with register-file state.
///
/// One [`TileState`] for ALL tiles (the DST file locks as a whole),
/// plus a refcount slab and the `tile_map` from IR value ops to DST
/// slots. The tile op methods (`matmul`, `reduce`, `binary`, `unary`,
/// `transpose`, `broadcast`, `copy`, `pack`) each drive the internal
/// `math_lock`/`pack_lock`/`math_unlock`/`pack_unlock` transitions and
/// emit exactly one sequence.
pub(crate) struct TileEmitter<const DSTBF16: bool> {
    /// Refcount per allocated DST slot.
    tiles: Slab<TileId<DSTBF16>, u32>,
    /// IR value op (tile-op result or acc storage) to DST slot.
    tile_map: Map<OpId, TileId<DSTBF16>>,
    /// Next slot to allocate; `inc` asserts the DST capacity.
    next: TileId<DSTBF16>,
    /// Whole-file MATH/PACK lock state.
    state: TileState,
    /// A reduce cone is open: the next `pack` closes it with
    /// `reduce_uninit` before the commit.
    reduce_pending: bool,
    /// Compute-kernel startup triple `[in0, in1, out]` for
    /// `compute_kernel_hw_startup`: recorded at compute entry by the
    /// generator, emitted with the top block.
    startup: Option<[CBId; 3]>,
    /// Init placements from [`Compiler::compute_init_plans`], keyed by
    /// the op whose text they precede: a config op's own entry emits
    /// inline, a `Loop`/`If` boundary's entry emits ahead of the
    /// whole scope (single backward pass: the scope hoisted exactly
    /// one occupant outward). At most one entry per op. Compute-only
    /// state lives here (not on [`CBEmitter`]) because only compute
    /// programs the engines.
    at_op: Map<OpId, PlacedInit>,
    /// Kernel-top inits in emission order: `MmTop` when the kernel
    /// holds a matmul, plus the single top-hoisted `Full` of a
    /// matmul-free single-config kernel. Reduce never lands here
    /// (its init needs the per-op acc slot, so it always stays
    /// inline).
    top: Vec<PlacedInit>,
    /// Matmul presence: matmul kernels skip `compute_kernel_hw_startup`
    /// (0.72 `mm_init` owns the full UNPACK/MATH/PACK programming).
    has_matmul: bool,
    /// Unpack-A format tracking: the `(CB, tt format code)` the
    /// unpacker was last programmed for. `copy_tile_init` is the
    /// SHORT init — it programs copy mechanics but NOT the data
    /// format, faces, or tile size — so a copy after a matmul (whose
    /// short/full init leaves unpack in matmul geometry) unpacks a
    /// different-format CB as zeros unless reconfigured. Only
    /// `MmTop`/`MmShort` (matmul geometry) and `with_dt` copies
    /// (reconfigured geometry) update this; every other init leaves
    /// it (a stale entry only ever causes a redundant — always safe —
    /// reconfig, never a missing one).
    unpack_src: Option<(CBId, u32)>,
}

#[allow(unused_must_use)]
impl<const DSTBF16: bool> TileEmitter<DSTBF16> {
    /// Empty emitter: no tiles, unlocked file, no inits.
    fn new() -> Self {
        Self {
            tiles: Slab::new(),
            tile_map: Map::default(),
            next: TileId::ZERO,
            state: TileState::Unlocked,
            reduce_pending: false,
            startup: None,
            at_op: Map::default(),
            top: Vec::new(),
            has_matmul: false,
            unpack_src: None,
        }
    }

    /// Record the compute-kernel startup triple (generator setup, no
    /// strings): `[in0, in1, out]`.
    fn set_startup(&mut self, triple: [CBId; 3]) {
        assert!(self.startup.is_none(), "tenstorrent2: compute startup triple set twice");
        self.startup = Some(triple);
    }

    /// Record the unpack-A source the engines are now programmed
    /// for (matmul inits and `with_dt` copies program the format;
    /// short copy inits do not and must not call this).
    fn note_unpack(&mut self, cb: CBId, fmt: u32) {
        self.unpack_src = Some((cb, fmt));
    }

    /// Copy-init text for `cb` (tt format code `fmt`): when the
    /// tracked unpack format differs, the `with_dt` form (which
    /// reconfigs UNPACK+MATH SRCA before the short init), otherwise
    /// the plain short init. A reconfig updates the tracking; a
    /// plain init leaves it (short inits program no format).
    fn copy_init(&mut self, cb: CBId, fmt: u32) -> String {
        match self.unpack_src {
            Some((prev, f)) if f != fmt => {
                self.unpack_src = Some((cb, fmt));
                format!("copy_tile_to_dst_init_short_with_dt({prev}, {cb});")
            }
            _ => format!("copy_tile_init({cb});"),
        }
    }

    /// Drain the placement at `op_id`, if the pass put one there
    /// (inline at a config op, or ahead of a `Loop`/`If` boundary).
    /// A missing placement just means no init here — only the pass
    /// decides placement; emission never invents inits.
    fn pop(&mut self, op_id: OpId) -> Option<PlacedInit> {
        self.at_op.remove(&op_id)
    }

    /// Build one placed init's call text. `Full` builds from the
    /// carried config: reduce additionally needs its DST acc slot,
    /// which only exists at its own op — a `Full(Reduce)` anywhere
    /// else is a pass bug, loud, never a default. `MmTop` and
    /// transpose read the output CB off the startup triple (same
    /// source the arms use today).
    fn line(&self, init: PlacedInit, acc: Option<TileId<DSTBF16>>) -> Result<String, BackendError> {
        Ok(match init {
            PlacedInit::Full(Cfg::Unary(uop)) => unary_init_name(uop).to_string(),
            PlacedInit::Full(Cfg::Binary(bop)) => {
                binary_init_name(bop).expect("tenstorrent2: placed binary init without an init call").to_string()
            }
            PlacedInit::Full(Cfg::Typecast(in_d, out_d)) => {
                let (in_fmt, out_fmt) = (tt_fmt(in_d)?, tt_fmt(out_d)?);
                format!("typecast_tile_init<{in_fmt}, {out_fmt}>();")
            }
            PlacedInit::Full(Cfg::Copy(cb)) => format!("copy_tile_init({cb});"),
            PlacedInit::Full(Cfg::Transpose(cb)) => {
                let out = self.startup.map(|[_, _, o]| o).expect("tenstorrent2: transpose init without startup triple");
                format!("transpose_wh_init({cb}, {out});")
            }
            PlacedInit::Full(Cfg::Bcast(bop, kind, cb_a, cb_b)) => {
                let Some(init) = bcast_init_name(bop, kind) else {
                    panic!("tenstorrent2: broadcast ({bop:?}, {kind:?}) has no init call")
                };
                format!("{init}({cb_a}, {cb_b});")
            }
            PlacedInit::Full(Cfg::BinScalar) => "binop_with_scalar_tile_init();".to_string(),
            PlacedInit::Full(Cfg::Reduce(rop, kind, ci, cs)) => {
                let Some(slot) = acc else {
                    panic!("tenstorrent2: reduce init hoisted away from its op (the acc slot lives at the op)");
                };
                let (op_name, dim_name) = match rop {
                    BOp::Max => ("PoolType::MAX", reduce_dim_name(kind)),
                    BOp::Add => ("PoolType::SUM", reduce_dim_name(kind)),
                    _ => panic!("tenstorrent2: reduce op {rop:?} has no init call"),
                };
                format!("reduce_init<{op_name}, {dim_name}>({ci}, {cs}, {slot});")
            }
            PlacedInit::Full(Cfg::Matmul(..)) => {
                panic!("tenstorrent2: matmul full init outside the kernel top (MmTop covers the top, MmShort the rest)")
            }
            PlacedInit::MmShort(cb_a, cb_b, old) => format!("mm_init_short_with_dt({cb_a}, {cb_b}, {old});"),
            PlacedInit::MmTop(cb_a, cb_b) => {
                let out = self.startup.map(|[_, _, o]| o).expect("tenstorrent2: top mm_init without startup triple");
                format!("mm_init({cb_a}, {cb_b}, {out});")
            }
        })
    }

    /// Allocate one DST slot. Capacity is asserted by
    /// [`TileId::inc`]: 16 BF16 tiles, 8 FP32 tiles.
    fn alloc(&mut self, rc: u32) -> TileId<DSTBF16> {
        let id = self.next;
        self.next.inc();
        let pushed = self.tiles.push(rc);
        debug_assert_eq!(pushed, id, "tenstorrent2: tile slab out of sync with allocator");
        id
    }

    /// `tile_regs_acquire`: MATH takes the file. Lazy: the first take
    /// acquires (which zeroes the file, so the acc starts at zero
    /// with no seed traffic); later takes in the same cone keep the
    /// content and emit nothing. Taking from `PackLock` flushes the
    /// deferred release first (see `pack`); taking from `PackLock`
    /// is otherwise a loud error (matrix X): the caller must release
    /// PACK first.
    fn math_lock(&mut self, src: &mut String, indent: &str) {
        if self.state == TileState::MathLock {
            return;
        }
        if self.state == TileState::PackLock {
            // Deferred release: the previous cone's packs are done,
            // MATH takes the file back.
            self.pack_unlock(src, indent);
        }
        assert_eq!(self.state, TileState::Unlocked, "tenstorrent2: math op with DST in {:?}, must be Unlocked", self.state);
        writeln!(src, "{indent}tile_regs_acquire();");
        self.state.math_lock();
    }

    /// `tile_regs_commit`: MATH releases the file. Unlocking an
    /// unlocked file is a loud error: every lock must pair.
    fn math_unlock(&mut self, src: &mut String, indent: &str) {
        assert_eq!(
            self.state,
            TileState::MathLock,
            "tenstorrent2: math_unlock with DST in {:?}, every lock must pair",
            self.state
        );
        writeln!(src, "{indent}tile_regs_commit();");
        self.state.math_unlock();
    }

    /// `tile_regs_wait`: PACK takes the file. Taking from `MathLock`
    /// is a loud error (matrix X): the caller must commit MATH first.
    fn pack_lock(&mut self, src: &mut String, indent: &str) {
        assert_eq!(self.state, TileState::Unlocked, "tenstorrent2: pack op with DST in {:?}, must be Unlocked", self.state);
        writeln!(src, "{indent}tile_regs_wait();");
        self.state.pack_lock();
    }

    /// `tile_regs_release`: PACK releases the file. Releasing without
    /// the PACK lock is a loud error: every lock must pair.
    fn pack_unlock(&mut self, src: &mut String, indent: &str) {
        assert_eq!(
            self.state,
            TileState::PackLock,
            "tenstorrent2: pack_unlock with DST in {:?}, every lock must pair",
            self.state
        );
        writeln!(src, "{indent}tile_regs_release();");
        self.state.pack_unlock();
    }

    /// Section-end flush for the deferred release (see `pack`): if a
    /// pack cone is still open, release it so the section ends with
    /// DST `Unlocked`.
    fn flush_pack(&mut self, src: &mut String, indent: &str) {
        if self.state == TileState::PackLock {
            self.pack_unlock(src, indent);
        }
    }

    /// Fused matmul: waits both input CBs, emits the single
    /// `matmul_tiles` (input tile ids alias the acc slot: the op
    /// sources data from the CBs and accumulates into the one tile),
    /// then pops both CBs. Runs under the MATH lock (lazily acquired).
    /// `mm_init` goes out once at the kernel top; every other placed
    /// init is the short self-old form. Records the result op in
    /// `tile_map`.
    fn matmul(
        &mut self,
        src: &mut String,
        indent: &str,
        cb_em: &mut CBEmitter,
        op_id: OpId,
        cb_a: CBId,
        cb_b: CBId,
        _out: CBId,
        acc: TileId<DSTBF16>,
    ) {
        // `mm_init` owns UNPACK+MATH+PACK engine programming (0.72):
        // full form once at the kernel top (`MmTop`, covering the
        // virgin entry whenever the kernel holds a matmul), short
        // self-old form at every other placed init.
        // `mm_init_short_with_dt` with `old == cb_a` no-ops its
        // reconfig against a matching predecessor (the
        // `should_reconfigure_cbs` guard) and re-enters matmul mode
        // after a foreign one — safe with zero forward information.
        match self.pop(op_id) {
            None => {}
            Some(PlacedInit::MmShort(a, b, old)) => {
                debug_assert_eq!((a, b), (cb_a, cb_b), "tenstorrent2: matmul op {op_id} placed a foreign short init");
                writeln!(src, "{indent}mm_init_short_with_dt({cb_a}, {cb_b}, {old});");
                self.note_unpack(cb_a, cb_em.config[cb_a].0);
            }
            Some(other) => panic!("tenstorrent2: matmul op {op_id} placed a non-short init ({other:?})"),
        }
        self.math_lock(src, indent);
        debug_assert_eq!(self.state, TileState::MathLock, "tenstorrent2: matmul without MATH lock");
        cb_em.wait_front(src, indent, cb_a, 1);
        cb_em.wait_front(src, indent, cb_b, 1);
        writeln!(src, "{indent}matmul_tiles({cb_a}, {cb_b}, {acc}, {acc}, {acc});");
        cb_em.record_move(cb_a, 1);
        cb_em.record_move(cb_b, 1);
        cb_em.pop_front(src, indent, cb_a, 1);
        cb_em.pop_front(src, indent, cb_b, 1);
        self.tile_map.insert(op_id, acc);
    }

    /// Streaming copy in: takes the MATH lock and copies the block
    /// slot into a fresh DST slot. The wait/pop sync anchors at the
    /// transaction's open/close events (see [`CBEmitter`]); `index`
    /// is the tile slot within the waited block. Records the load op
    /// in `tile_map`. The init is the plain short form unless the
    /// tracked unpack format differs from `fmt` (see `copy_init`):
    /// its placement comes from the plan (anchor-hoisted, inline per
    /// the pass, or nothing — a reconfig still goes out when the
    /// tracking demands it).
    fn copy(&mut self, src: &mut String, indent: &str, op_id: OpId, cb: CBId, rc: u32, index: &str, fmt: u32) -> TileId<DSTBF16> {
        self.math_lock(src, indent);
        debug_assert_eq!(self.state, TileState::MathLock, "tenstorrent2: copy without MATH lock");
        let slot = self.alloc(rc);
        match self.pop(op_id) {
            Some(PlacedInit::Full(Cfg::Copy(c))) => {
                debug_assert_eq!(c, cb, "tenstorrent2: copy op {op_id} placed a foreign init");
                let line = self.copy_init(c, fmt);
                writeln!(src, "{indent}{line}");
            }
            None => {
                // No placed init, but a stale unpack format still
                // needs the reconfig before this copy reads.
                if matches!(self.unpack_src, Some((_, f)) if f != fmt) {
                    let line = self.copy_init(cb, fmt);
                    writeln!(src, "{indent}{line}");
                }
            }
            Some(other) => panic!("tenstorrent2: copy op {op_id} placed a non-copy init ({other:?})"),
        }
        writeln!(src, "{indent}copy_tile({cb}, {index}, {slot});");
        self.tile_map.insert(op_id, slot);
        slot
    }

    /// Pack out: closes any open reduce cone, commits MATH, reserves
    /// the CB, takes the PACK lock, packs the slot, pushes the CB.
    /// The release is DEFERRED: consecutive packs share one cone
    /// (commit, wait, pack, pack, release — the only shape where a
    /// second `pack_tile` still finds DST acquired; a second
    /// wait-after-release would stall PACK with no matching commit).
    /// The deferred release flushes at the next `math_lock` or at
    /// section end (`flush_pack`). Slots stay live across packs: the
    /// allocator hands out fresh slots monotonically and never
    /// reuses one, so no MATH op can clobber a packed slot. Packing
    /// from `Unlocked` is a loud error (pack of a dead slot).
    /// The whole drain sequences here, in emission order: uninit,
    /// commit, reserve, wait, [reconfig,] pack, push. The packer
    /// reconfig goes out only for non-native pack targets: the JIT
    /// programs the mode-native triple by construction, so
    /// reprogramming it is redundant there.
    fn pack(&mut self, src: &mut String, indent: &str, cb_em: &CBEmitter, slot: TileId<DSTBF16>, cb: CBId) {
        if self.reduce_pending {
            writeln!(src, "{indent}reduce_uninit();");
            self.reduce_pending = false;
            // Teardown closes the reduce LLK cone: the placement pass
            // models the config as gone from here on (the unpack-A
            // source survives), so the next config op re-inits.
        }
        match self.state {
            TileState::MathLock => {
                self.math_unlock(src, indent);
                self.pack_lock(src, indent);
            }
            TileState::PackLock => {
                debug_assert!(!self.reduce_pending, "tenstorrent2: reduce cone open across packs");
            }
            TileState::Unlocked => {
                panic!("tenstorrent2: pack with DST Unlocked, no live cone (pack of a dead slot)");
            }
        }
        debug_assert_eq!(self.state, TileState::PackLock, "tenstorrent2: pack without PACK lock");
        // Mode-native pack target needs no runtime reconfig: the JIT
        // programs the packer for it by construction. Runtime CB
        // format codes (see CBEmitter::new): F32=0, F16=1, BF16=2.
        // 16-bit DST packs F16/BF16 natively, 32-bit DST packs F32;
        // anything else keeps the override.
        let cb_fmt = cb_em.config[cb].0;
        let native = if DSTBF16 { cb_fmt == 1 || cb_fmt == 2 } else { cb_fmt == 0 };
        if !native {
            writeln!(src, "{indent}pack_reconfig_data_format({cb});");
        }
        writeln!(src, "{indent}pack_tile({slot}, {cb});");
        // No release: the cone stays open for consecutive packs;
        // the next math_lock or the section-end flush releases it.
    }

    /// Acc tile for a Register acc storage: looked up in `tile_map`,
    /// allocated on first use. The slot starts zeroed (the lazy
    /// `math_lock` zeroes the file on first acquire), so seeds emit
    /// nothing.
    fn acc_tile(&mut self, storage: OpId, rc: u32) -> TileId<DSTBF16> {
        if let Some(&tile) = self.tile_map.get(&storage) {
            return tile;
        }
        let tile = self.alloc(rc);
        self.tile_map.insert(storage, tile);
        tile
    }

    /// Fused reduce: waits input + scaler CBs, emits the tile into
    /// the acc slot, pops both CBs. The init goes out hoisted and
    /// inline on kind switches, the uninit at the consuming pack.
    /// Runs under the MATH lock (lazily acquired). Records the
    /// result op in `tile_map`.
    fn reduce(
        &mut self,
        src: &mut String,
        indent: &str,
        cb_em: &mut CBEmitter,
        op_id: OpId,
        cb_in: CBId,
        cb_sc: CBId,
        acc: TileId<DSTBF16>,
        rop: BOp,
        kind: TileDim,
        rc: u32,
    ) -> Result<TileId<DSTBF16>, BackendError> {
        let op_name = match rop {
            BOp::Max => "PoolType::MAX",
            BOp::Add => "PoolType::SUM",
            _ => {
                return Err(BackendError {
                    status: ErrorStatus::KernelCompilation,
                    context: format!("tenstorrent2: reduce op {rop:?} unsupported, op {op_id}").into(),
                });
            }
        };
        let dim_name = reduce_dim_name(kind);
        match self.pop(op_id) {
            None => {}
            Some(init @ PlacedInit::Full(Cfg::Reduce(..))) => {
                writeln!(src, "{indent}{}", self.line(init, Some(acc))?);
            }
            Some(other) => panic!("tenstorrent2: reduce op {op_id} placed a non-reduce init ({other:?})"),
        }
        self.math_lock(src, indent);
        debug_assert_eq!(self.state, TileState::MathLock, "tenstorrent2: reduce without MATH lock");
        cb_em.wait_front(src, indent, cb_in, 1);
        cb_em.wait_front(src, indent, cb_sc, 1);
        writeln!(src, "{indent}reduce_tile<{op_name}, {dim_name}>({cb_in}, {cb_sc}, 0, 0, {acc});");
        self.reduce_pending = true;
        cb_em.record_move(cb_in, 1);
        cb_em.record_move(cb_sc, 1);
        cb_em.pop_front(src, indent, cb_in, 1);
        cb_em.pop_front(src, indent, cb_sc, 1);
        self.tile_map.insert(op_id, acc);
        let _ = rc;
        Ok(acc)
    }

    /// Tiled binary ALU: takes the MATH lock and records the hoisted
    /// init. Three-operand form (`op(x, y, odst)`): inputs stay live,
    /// the result lands in a fresh slot.
    #[allow(dead_code)]
    fn binary(
        &mut self,
        src: &mut String,
        indent: &str,
        op_id: OpId,
        x: TileId<DSTBF16>,
        y: TileId<DSTBF16>,
        bop: BOp,
        rc: u32,
    ) -> TileId<DSTBF16> {
        let name = match bop {
            BOp::Add => "add_binary_tile",
            BOp::Sub => "sub_binary_tile",
            BOp::Mul => "mul_binary_tile",
            BOp::Div => "div_binary_tile",
            BOp::Max => "binary_max_tile",
            BOp::BitShiftLeft => "binary_left_shift_tile",
            BOp::BitShiftRight => "binary_right_shift_tile",
            _ => todo!("tenstorrent2 tiled binary {bop:?} op"),
        };
        self.math_lock(src, indent);
        debug_assert_eq!(self.state, TileState::MathLock, "tenstorrent2: binary without MATH lock");
        if binary_init_name(bop).is_some() {
            match self.pop(op_id) {
                None => {}
                Some(init @ PlacedInit::Full(Cfg::Binary(_))) => {
                    let line = self.line(init, None).expect("tenstorrent2: binary init is infallible");
                    writeln!(src, "{indent}{line}");
                }
                Some(other) => panic!("tenstorrent2: binary op {op_id} placed a non-binary init ({other:?})"),
            }
        }
        let odst = self.alloc(rc);
        writeln!(src, "{indent}{name}({x}, {y}, {odst});");
        self.tile_map.insert(op_id, odst);
        odst
    }

    /// Fused broadcast binary (`add/sub/mul_tiles_bcast_*`): both
    /// operands stay in CBs — the full tile in `cb_a`, the broadcast
    /// lane tile in `cb_b` — and the result lands in a fresh DST slot.
    /// Single-tile traffic (pop both); multi-output reuse of one
    /// broadcast tile is a later refinement.
    #[allow(dead_code)]
    fn bcast(
        &mut self,
        src: &mut String,
        indent: &str,
        cb_em: &mut CBEmitter,
        op_id: OpId,
        cb_a: CBId,
        cb_b: CBId,
        bop: BOp,
        kind: TileDim,
        rc: u32,
    ) -> TileId<DSTBF16> {
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
            _ => todo!("tenstorrent2 broadcast ({bop:?}, {kind:?}) op"),
        };
        match self.pop(op_id) {
            None => {}
            Some(init @ PlacedInit::Full(Cfg::Bcast(..))) => {
                let line = self.line(init, None).expect("tenstorrent2: broadcast init is infallible");
                writeln!(src, "{indent}{line}");
            }
            Some(other) => panic!("tenstorrent2: broadcast op {op_id} placed a non-broadcast init ({other:?})"),
        }
        self.math_lock(src, indent);
        debug_assert_eq!(self.state, TileState::MathLock, "tenstorrent2: broadcast without MATH lock");
        let (fmt_a, fmt_b) = (cb_em.config[cb_a].0, cb_em.config[cb_b].0);
        debug_assert_eq!(fmt_a, fmt_b, "tenstorrent2: broadcast op {op_id} mixes CB formats (short inits reconfigure nothing)");
        cb_em.wait_front(src, indent, cb_a, 1);
        cb_em.wait_front(src, indent, cb_b, 1);
        let odst = self.alloc(rc);
        if matches!(kind, TileDim::Row) {
            writeln!(src, "{indent}{name}({cb_a}, {cb_b}, 0, 0, {odst}, 0);");
        } else {
            writeln!(src, "{indent}{name}({cb_a}, {cb_b}, 0, 0, {odst});");
        }
        cb_em.record_move(cb_a, 1);
        cb_em.record_move(cb_b, 1);
        cb_em.pop_front(src, indent, cb_a, 1);
        cb_em.pop_front(src, indent, cb_b, 1);
        self.note_unpack(cb_a, fmt_a);
        self.tile_map.insert(op_id, odst);
        odst
    }

    /// Tiled unary ALU: SFPU ops are in-place (`op(x)` transforms the
    /// slot), so the result aliases the operand slot. Takes the MATH
    /// lock and records the hoisted init.
    #[allow(dead_code)]
    fn unary(&mut self, src: &mut String, indent: &str, op_id: OpId, x: TileId<DSTBF16>, uop: UOp) -> TileId<DSTBF16> {
        // log2 passes its base scale (bits of 1/ln 2) explicitly, emitted
        // here: the `name` match below evaluates eagerly, so Log2 can never
        // be deferred to a later branch.
        if uop == UOp::Log2 {
            self.math_lock(src, indent);
            debug_assert_eq!(self.state, TileState::MathLock, "tenstorrent2: unary without MATH lock");
            match self.pop(op_id) {
                None => {}
                Some(init @ PlacedInit::Full(Cfg::Unary(_))) => {
                    let line = self.line(init, None).expect("tenstorrent2: unary init is infallible");
                    writeln!(src, "{indent}{line}");
                }
                Some(other) => panic!("tenstorrent2: unary op {op_id} placed a non-unary init ({other:?})"),
            }
            writeln!(src, "{indent}log_with_base_tile({x}, 0x3fb8aa3b);");
            self.tile_map.insert(op_id, x);
            return x;
        }
        let name = match uop {
            UOp::Neg => "negative_tile",
            UOp::BitNot => "bitwise_not_tile",
            UOp::Exp => "exp_tile",
            UOp::Exp2 => "exp2_tile",
            UOp::Log2 => unreachable!("tenstorrent2: log2 is emitted above"),
            UOp::Sin => "sin_tile",
            UOp::Cos => "cos_tile",
            UOp::Reciprocal => "recip_tile",
            UOp::Sqrt => "sqrt_tile",
            UOp::Rsqrt => "rsqrt_tile",
            UOp::Floor => "floor_tile",
            UOp::Trunc => "trunc_tile",
            UOp::Abs => "abs_tile",
            UOp::Not => "logical_not_tile",
        };
        self.math_lock(src, indent);
        debug_assert_eq!(self.state, TileState::MathLock, "tenstorrent2: unary without MATH lock");
        match self.pop(op_id) {
            None => {}
            Some(init @ PlacedInit::Full(Cfg::Unary(_))) => {
                let line = self.line(init, None).expect("tenstorrent2: unary init is infallible");
                writeln!(src, "{indent}{line}");
            }
            Some(other) => panic!("tenstorrent2: unary op {op_id} placed a non-unary init ({other:?})"),
        }
        writeln!(src, "{indent}{name}({x});");
        self.tile_map.insert(op_id, x);
        x
    }

    /// Tile-scalar binary (`*_unary_tile` with an fp32-bits immediate):
    /// DST-inplace like unary, no CB traffic for the scalar side.
    /// `name` is the call (`add/sub/mul/div/rsub_unary_tile`); const-first
    /// Div has no call (recip+mul is IR's job) and never reaches here.
    /// WARNING: the header's doc table numbers the modes
    /// add/mul/sub/div/rsub, but the `ADD_UNARY/SUB_UNARY/...` enum —
    /// which the call names encode — is authoritative. Trust the names.
    fn bin_scalar(
        &mut self,
        src: &mut String,
        indent: &str,
        op_id: OpId,
        x: TileId<DSTBF16>,
        name: &str,
        bits: u32,
    ) -> TileId<DSTBF16> {
        self.math_lock(src, indent);
        debug_assert_eq!(self.state, TileState::MathLock, "tenstorrent2: scalar binary without MATH lock");
        match self.pop(op_id) {
            None => {}
            Some(init @ PlacedInit::Full(Cfg::BinScalar)) => {
                let line = self.line(init, None).expect("tenstorrent2: scalar binary init is infallible");
                writeln!(src, "{indent}{line}");
            }
            Some(other) => panic!("tenstorrent2: scalar binary op {op_id} placed a non-scalar init ({other:?})"),
        }
        writeln!(src, "{indent}{name}({x}, {bits:#x});");
        self.tile_map.insert(op_id, x);
        x
    }

    /// Tiled cast: in-place like unary (`typecast_tile<IN,OUT>(x)`).
    /// DTypes ride through; `tt_fmt` converts at emission. The
    /// transition records the pair and inits inline on kind switches.
    #[allow(dead_code)]
    fn cast(
        &mut self,
        src: &mut String,
        indent: &str,
        op_id: OpId,
        x: TileId<DSTBF16>,
        in_dt: DType,
        out_dt: DType,
    ) -> Result<TileId<DSTBF16>, BackendError> {
        let in_fmt = tt_fmt(in_dt)?;
        let out_fmt = tt_fmt(out_dt)?;
        self.math_lock(src, indent);
        debug_assert_eq!(self.state, TileState::MathLock, "tenstorrent2: cast without MATH lock");
        match self.pop(op_id) {
            None => {}
            Some(init @ PlacedInit::Full(Cfg::Typecast(_, _))) => {
                writeln!(src, "{indent}{}", self.line(init, None)?);
            }
            Some(other) => panic!("tenstorrent2: cast op {op_id} placed a non-cast init ({other:?})"),
        }
        writeln!(src, "{indent}typecast_tile<{in_fmt}, {out_fmt}>({x});");
        self.tile_map.insert(op_id, x);
        Ok(x)
    }

    /// Streaming transpose: waits the CB, takes the MATH lock,
    /// transposes the 32x32 tile into a fresh DST slot, pops the CB.
    /// Records the load op in `tile_map`. The `transpose_wh_init`
    /// emits per the backward pass's placement (inline or ahead of a
    /// scope).
    fn transpose(
        &mut self,
        src: &mut String,
        indent: &str,
        cb_em: &mut CBEmitter,
        op_id: OpId,
        cb: CBId,
        _out: CBId,
        rc: u32,
    ) -> TileId<DSTBF16> {
        cb_em.wait_front(src, indent, cb, 1);
        self.math_lock(src, indent);
        debug_assert_eq!(self.state, TileState::MathLock, "tenstorrent2: transpose without MATH lock");
        let slot = self.alloc(rc);
        match self.pop(op_id) {
            None => {}
            Some(init @ PlacedInit::Full(Cfg::Transpose(_))) => {
                let line = self.line(init, None).expect("tenstorrent2: transpose init is infallible");
                writeln!(src, "{indent}{line}");
            }
            Some(other) => panic!("tenstorrent2: transpose op {op_id} placed a non-transpose init ({other:?})"),
        }
        writeln!(src, "{indent}transpose_wh_tile({cb}, 0, {slot});");
        cb_em.record_move(cb, 1);
        cb_em.pop_front(src, indent, cb, 1);
        self.tile_map.insert(op_id, slot);
        slot
    }

    /// Tiled broadcast: takes the MATH lock. The op text itself is
    /// unproven LLK interaction.
    #[allow(dead_code)]
    fn broadcast(&mut self, src: &mut String, indent: &str, op_id: OpId, _x: TileId<DSTBF16>, rc: u32) -> TileId<DSTBF16> {
        self.math_lock(src, indent);
        debug_assert_eq!(self.state, TileState::MathLock, "tenstorrent2: broadcast without MATH lock");
        let slot = self.alloc(rc);
        self.tile_map.insert(op_id, slot);
        todo!("tenstorrent2 tiled broadcast op text");
    }

    /// Kernel-top block for compute tile ops, inserted at `pos` —
    /// ahead of all loops, at function-scope indent: either the
    /// kernel-top `mm_init` (any kernel holding a matmul — 0.72
    /// `mm_init` owns the full UNPACK/MATH/PACK programming, so it
    /// replaces startup) or `compute_kernel_hw_startup` once, plus a
    /// matmul-free single-config kernel's one top-hoisted `Full`.
    /// Startup comes first (the header requires it exactly once at
    /// the beginning, before any op init; no separate SFPU init
    /// exists); FP32 mode enables 32-bit DST right after. All other
    /// placements emit at their op or ahead of their hoist boundary
    /// (see [`PlacedInit`]).
    fn prepend_compute_inits(&self, src: &mut String, pos: usize) -> Result<(), BackendError> {
        let indent = "  ";
        let mut head = String::new();
        // Matmul kernels skip startup (`mm_init` owns the long init);
        // anything else starts up normally.
        if std::env::var("ZYX_TT_INIT_SFPU").is_ok() {
            // EXPERIMENT ONLY: chain-exact setup — `init_sfpu` instead
            // of startup. Decides whether multi-op episodes need it.
            if let Some([in0, _, out]) = self.startup {
                let _ = std::fmt::Write::write_fmt(&mut head, format_args!("{indent}init_sfpu({in0}, {out});\n"));
            }
        } else if self.has_matmul {
            if !DSTBF16 {
                let _ = std::fmt::Write::write_fmt(&mut head, format_args!("{indent}enable_fp32_dest_acc();\n"));
            }
        } else {
            if let Some([in0, in1, out]) = self.startup {
                let _ = std::fmt::Write::write_fmt(
                    &mut head,
                    format_args!("{indent}compute_kernel_hw_startup({in0}, {in1}, {out});\n"),
                );
                if !DSTBF16 {
                    let _ = std::fmt::Write::write_fmt(&mut head, format_args!("{indent}enable_fp32_dest_acc();\n"));
                }
            }
        }
        let mut inits = head;
        for init in &self.top {
            inits.push_str(&format!("{indent}{}\n", self.line(*init, None)?));
        }
        src.insert_str(pos, &inits);
        Ok(())
    }
}

/// Tenstorrent v2 compiler: per-section code generation over closed op
/// lists (see [`Kernel::get_needed_ops`]) against one shared [`CBId`] map.
///
/// All state lives here (ptx.rs pattern) in four emitters: [`VarEmitter`]
/// (per-section scalar source), [`NocEmitter`] (params + NOC traffic),
/// [`CBEmitter`] (CB traffic + state), [`TileEmitter`] (DST traffic +
/// state). The section generators dispatch to emitter methods and emit
/// no strings themselves.
/// [`Kernel::generate_tenstorrent`] returns this struct to the backend,
/// which reads the section kernels, the CB config, and the dtypes off it.
pub(crate) struct Compiler<const DSTBF16: bool> {
    /// Host params + NOC traffic.
    pub(crate) noc: NocEmitter,
    /// CB allocation, traffic + state.
    pub(crate) cb: CBEmitter,
    /// DST traffic + state.
    pub(crate) tl: TileEmitter<DSTBF16>,
    /// Generated section kernels (filled by generation).
    pub(crate) reader: TTKernel,
    /// Generated section kernels (filled by generation).
    pub(crate) compute: TTKernel,
    /// Generated section kernels (filled by generation).
    pub(crate) writer: TTKernel,
}

/// Codegen output with the DST mode resolved: BF16 kernels carry a
/// 16-tile file, FP32 kernels an 8-tile file. The backend matches once
/// and reads the section kernels, CB config, and dtypes off the inner
/// compiler.
pub(crate) enum TTCompiler {
    /// 16-tile BF16 DST file.
    Bf16(Compiler<true>),
    /// 8-tile FP32 DST file.
    Fp32(Compiler<false>),
}

impl<const DSTBF16: bool> Compiler<DSTBF16> {
    /// Full codegen: section closures, per-section params, CB
    /// allocation, IR checks, then the three section generators.
    fn generate(kernel: &Kernel) -> Result<Self, BackendError> {
        let mut compiler = Compiler::new(kernel)?;
        let reader_data = kernel.get_needed_ops(TtSection::Reader);
        let compute_data = kernel.get_needed_ops(TtSection::Compute);
        let writer_data = kernel.get_needed_ops(TtSection::Writer);
        let [reader_params, compute_params, writer_params] =
            compiler.section_param_lists(kernel, &reader_data, &compute_data, &writer_data);
        // Per-section param ordinals (global head order): each section's
        // params are in IR order, so the ordinals ascend already.
        let [reader_ord, compute_ord, writer_ord] = [&reader_params, &compute_params, &writer_params].map(|params| {
            let ordinals: Vec<u32> = params.iter().map(|p| compiler.noc.param_ordinal_of[p]).collect();
            debug_assert!(ordinals.windows(2).all(|w| w[0] < w[1]), "tenstorrent2 section params not in head order");
            ordinals
        });
        compiler.check_sections(kernel)?;
        compiler.generate_reader(kernel, &reader_data, &reader_params, &reader_ord)?;
        compiler.generate_compute(kernel, &compute_data, &compute_params, &compute_ord)?;
        compiler.generate_writer(kernel, &writer_data, &writer_params, &writer_ord)?;

        Ok(compiler)
    }

    /// Build compiler state from a kernel: the [`NocEmitter`] walks the
    /// params, the [`CBEmitter`] allocates the CBs, the tile emitter
    /// starts empty.
    fn new(kernel: &Kernel) -> Result<Self, BackendError> {
        Ok(Self {
            noc: NocEmitter::new(kernel),
            cb: CBEmitter::new(kernel)?,
            tl: TileEmitter::new(),
            reader: TTKernel::None,
            compute: TTKernel::None,
            writer: TTKernel::None,
        })
    }

    /// Per-section param lists in IR order: each section kernel's
    /// runtime args. The single point that decides what params each
    /// kernel gets.
    fn section_param_lists(
        &self,
        kernel: &Kernel,
        reader: &SectionData,
        compute: &SectionData,
        writer: &SectionData,
    ) -> [Vec<OpId>; 3] {
        [reader, compute, writer].map(|data| {
            data.ops.iter().copied().filter(|op| matches!(kernel.ops[*op].op, Op::Param { .. })).collect::<Vec<OpId>>()
        })
    }

    /// Exactly 2 barriers delimiting reader/compute/writer, else a
    /// compilation error. Also rejects GPU-only ops up front (Wmma:
    /// tenstorrent has `MatmulTile`, no WMMA units).
    fn check_sections(&self, kernel: &Kernel) -> Result<(), BackendError> {
        let mut barriers = 0u32;
        let mut scan = kernel.head;
        for _ in 0..10_000 {
            if scan.is_null() {
                break;
            }
            if matches!(kernel.ops[scan].op, Op::Barrier) {
                barriers += 1;
            }
            if matches!(kernel.ops[scan].op, Op::Wmma { .. }) {
                return Err(BackendError {
                    status: ErrorStatus::KernelCompilation,
                    context: "tenstorrent2: Wmma is GPU-only, tenstorrent uses Op::MatmulTile".into(),
                });
            }
            scan = kernel.next_op(scan);
        }
        if !scan.is_null() {
            panic!("tenstorrent2 section scan did not finish in 10000 steps");
        }
        if barriers != 2 {
            return Err(BackendError {
                status: ErrorStatus::InvalidKernelSections,
                context: format!("tenstorrent2: need exactly 2 barriers (3 sections), found {barriers}").into(),
            });
        }
        Ok(())
    }

    /// Init-placement pass: one backward walk over the compute ops
    /// with a scope stack (`scopes[0]` = the global scope). Every
    /// config op records a provisional entry in its own scope plus
    /// its occupant. When a scope closes holding exactly one
    /// occupant, its members' provisionals retract into one entry
    /// keyed by the boundary op in the parent scope — an init ahead
    /// of the whole scope, running once per enclosing pass — and any
    /// entries hoisted out of child scopes are subsumed by it. A lone
    /// occupant that arrived only via a hoisted child moves that one
    /// entry further outward instead. Reduce never hoists (its init
    /// takes the per-op acc slot) but still occupies; a binary with
    /// no init call programs nothing and occupies nothing.
    /// Multi-occupant scopes keep every provisional inline. At the
    /// global scope a single `Matmul` occupant elides into the
    /// unconditional kernel-top `mm_init`; a single other occupant
    /// tops one `Full`.
    fn compute_init_plans(&mut self, kernel: &Kernel, data: &SectionData) {
        // Scope stack: occupants, direct member config ops, and
        // boundary keys of entries hoisted out of child scopes into
        // this one. scopes[0] is the global scope.
        struct Scope {
            occupants: Set<Cfg>,
            members: Vec<OpId>,
            hoisted: Vec<OpId>,
        }
        let mut scopes: Vec<Scope> = vec![Scope { occupants: Set::default(), members: Vec::new(), hoisted: Vec::new() }];
        let mut at_op: Map<OpId, PlacedInit> = Map::default();
        let mut top: Vec<PlacedInit> = Vec::new();
        let mut has_matmul = false;
        // Provisional init for one classified config op. The matmul
        // short's `old` trips the reconfig guard (see `MmShort`).
        let provisional = |cfg: Cfg, config: &Slab<CBId, (u32, u32, u32)>| match cfg {
            Cfg::Matmul(cb_a, cb_b) => {
                let fmt_a = config[cb_a].0;
                let old = config.ids().find(|cb| config[*cb].0 != fmt_a).unwrap_or(cb_a);
                PlacedInit::MmShort(cb_a, cb_b, old)
            }
            cfg => PlacedInit::Full(cfg),
        };
        // Close the top scope at `boundary` (None = global): hoist a
        // lone occupant outward, keep the rest where they are. Every
        // occupant propagates to the parent (a child scope's content
        // sits mid-body, so the parent may not merge across it).
        let close = |boundary: Option<OpId>,
                     scopes: &mut Vec<Scope>,
                     at_op: &mut Map<OpId, PlacedInit>,
                     top: &mut Vec<PlacedInit>,
                     config: &Slab<CBId, (u32, u32, u32)>| {
            let scope = scopes.pop().expect("tenstorrent2: init pass scope underflow");
            let lone = (scope.occupants.len() == 1)
                .then(|| scope.occupants.iter().next().copied().expect("tenstorrent2: empty lone occupant set"))
                .filter(|cfg| !cfg.needs_acc_slot());
            if let Some(parent) = scopes.last_mut() {
                for &o in &scope.occupants {
                    parent.occupants.insert(o);
                }
                match (boundary, lone) {
                    (Some(b), Some(cfg)) => {
                        for &m in &scope.members {
                            at_op.remove(&m);
                        }
                        for h in scope.hoisted {
                            // Subsumed: the outer init dominates the
                            // same config.
                            at_op.remove(&h);
                        }
                        at_op.insert(b, provisional(cfg, config));
                        parent.hoisted.push(b);
                    }
                    _ => {
                        parent.hoisted.extend(scope.hoisted);
                    }
                }
            } else if let Some(cfg) = lone {
                // Global scope: the one top entry.
                debug_assert!(
                    !scope.members.is_empty() || !scope.hoisted.is_empty(),
                    "tenstorrent2: lone top occupant without any placed init"
                );
                for &m in &scope.members {
                    at_op.remove(&m);
                }
                for h in scope.hoisted {
                    at_op.remove(&h);
                }
                top.push(provisional(cfg, config));
            }
        };
        for &op in data.ops.iter().rev() {
            match kernel.ops[op].op {
                Op::EndLoop | Op::EndIf => {
                    scopes.push(Scope { occupants: Set::default(), members: Vec::new(), hoisted: Vec::new() })
                }
                Op::Loop { .. } | Op::If { .. } => close(Some(op), &mut scopes, &mut at_op, &mut top, &self.cb.config),
                Op::Barrier => panic!("tenstorrent2: control flow crosses a section barrier"),
                _ => {
                    let Some(cfg) = classify_tile_cfg(kernel, &self.cb.map, data, &self.cb.consumers, op) else {
                        continue;
                    };
                    if let Cfg::Binary(bop) = cfg {
                        if binary_init_name(bop).is_none() {
                            continue;
                        }
                    }
                    if matches!(cfg, Cfg::Matmul(..)) {
                        has_matmul = true;
                    }
                    at_op.insert(op, provisional(cfg, &self.cb.config));
                    let scope = scopes.last_mut().expect("tenstorrent2: op without scope");
                    scope.occupants.insert(cfg);
                    scope.members.push(op);
                }
            }
        }
        close(None, &mut scopes, &mut at_op, &mut top, &self.cb.config);
        debug_assert!(scopes.is_empty(), "tenstorrent2: init pass left scopes open");
        // A top-hoisted matmul short is covered by the unconditional
        // kernel-top `mm_init`: elide it.
        if top.iter().any(|t| matches!(t, PlacedInit::MmShort(..))) {
            top.retain(|t| !matches!(t, PlacedInit::MmShort(..)));
        }
        if has_matmul {
            let (cb_a, cb_b) = data
                .ops
                .iter()
                .find_map(|&op| match classify_tile_cfg(kernel, &self.cb.map, data, &self.cb.consumers, op) {
                    Some(Cfg::Matmul(a, b)) => Some((a, b)),
                    _ => None,
                })
                .expect("tenstorrent2: matmul flag without a matmul op");
            top.insert(0, PlacedInit::MmTop(cb_a, cb_b));
        }
        self.tl.at_op = at_op;
        self.tl.top = top;
        self.tl.has_matmul = has_matmul;
    }

    /// Generate the reader (dataflow movement) kernel from the reader op
    /// list: pure dispatch — every arm calls exactly one emitter
    /// method, no strings here. Entry: all CBs free; exit: settled.
    #[allow(unused_must_use)]
    fn generate_reader(
        &mut self,
        kernel: &Kernel,
        reader_data: &SectionData,
        params: &[OpId],
        ordinals: &[u32],
    ) -> Result<(), BackendError> {
        let ops = &reader_data.ops;
        let mut em = VarEmitter::new(kernel);
        let mut src = String::new();
        let mut indent = String::from("  ");
        self.noc.begin_section(params);
        self.cb.reset_all(CBState::Popped);
        let mut scope_level = 0u8;
        writeln!(src, "#include <cstdint>");
        writeln!(src, "#include \"api/dataflow/dataflow_api.h\"");
        writeln!(src, "#include \"api/dataflow/noc.h\"");
        writeln!(src, "#include \"api/dataflow/circular_buffer.h\"");
        writeln!(src, "#include \"api/tensor/noc_traits.h\"");
        writeln!(src, "#include \"api/debug/device_print.h\"");
        writeln!(src, "void kernel_main() {{");
        // Every shared CB is declared (matches v1 output); the section
        // only pushes the ones it fills.
        self.cb.declare_all(&mut src, &indent);
        let n = ops.len();
        for i in 0..n {
            let op_id = ops[i];
            self.cb.open_events(&mut src, &indent, op_id);
            match kernel.ops[op_id].op {
                Op::Param { dtype, kind, .. } => match kind {
                    ParamKind::Global => {
                        self.noc.declare_global(&mut src, &indent, op_id);
                    }
                    ParamKind::Variable => {
                        em.declare_variable(&mut src, &indent, reader_data, &self.noc, dtype, op_id, scope_level);
                    }
                    ParamKind::GlobalMut => {
                        self.noc.declare_global_mut(&mut src, &indent, op_id);
                    }
                },
                Op::Storage { scope, .. } => match scope {
                    MemScope::Circular | MemScope::Register => {
                        // Circular CBs are declared up front; Register
                        // accs thread through `tile_map`. Neither emits
                        // traffic.
                    }
                    MemScope::Local => unreachable!(
                        "tenstorrent does not have local threads; local indices should have been converted to loops by the opt_tenstorrent_tile optimization pass"
                    ),
                    MemScope::Global => todo!("tenstorrent2 reader storage scope"),
                },
                Op::Load { .. } => {
                    // Handled at the consuming store (global to local with
                    // no ops in between, as below).
                }
                Op::Store { ref dst, src: ref store_src, index: st_index, layout: st_layout } => {
                    let Op::Load { src: ld_src, index: ld_idx, layout: ld_layout } = kernel.ops[*store_src].op else {
                        return Err(BackendError {
                            status: ErrorStatus::KernelCompilation,
                            context: format!(
                                "tenstorrent2: reader supports only global to local stores, op {op_id} has ops in between"
                            )
                            .into(),
                        });
                    };
                    let Op::Param { kind: ParamKind::Global, .. } = kernel.ops[ld_src].op else {
                        return Err(BackendError {
                            status: ErrorStatus::KernelCompilation,
                            context: format!("tenstorrent2: reader load op {op_id} is not from a Global param").into(),
                        });
                    };
                    let Op::Storage { dtype, scope: MemScope::Circular, .. } = kernel.ops[*dst].op else {
                        return Err(BackendError {
                            status: ErrorStatus::KernelCompilation,
                            context: format!("tenstorrent2: reader store op {op_id} does not target a Circular CB").into(),
                        });
                    };
                    let Some(&cb) = self.cb.map.get(dst) else {
                        return Err(BackendError {
                            status: ErrorStatus::KernelCompilation,
                            context: format!("tenstorrent2: reader store op {op_id} targets unmapped CB").into(),
                        });
                    };
                    match (ld_layout, st_layout) {
                        (MemLayout::Tile { x, y, .. }, MemLayout::Tile { .. }) => {
                            let elem_size = dtype.bit_size() as u32 / 8;
                            let tile_bytes = x as u32 * y as u32 * elem_size;
                            let idx = em.resolve_idx(kernel, reader_data, ld_idx, scope_level, "reader")?;
                            let b = self.cb.batch(op_id).expect("tenstorrent2: pre-pass assigned a batch to every traffic op");
                            // The sync ops anchor at the transaction's
                            // open/close events; this text fills the
                            // block at its slot.
                            let off = self.cb.slot_offset(kernel, &mut em, reader_data, st_index, b, scope_level, "reader")?;
                            self.cb.record_move(b.cb, b.per_op);
                            self.noc.async_read_tile(&mut src, &indent, op_id, ld_src, &idx, elem_size, tile_bytes, cb, &off);
                        }
                        _ => todo!("tenstorrent2 reader only supports tile stores"),
                    }
                }
                Op::Loop { .. } => {
                    em.loop_begin(&mut src, &mut indent, kernel, reader_data, op_id, &mut scope_level)?;
                }
                Op::EndLoop => {
                    loop_end(&mut src, &mut indent, &mut scope_level);
                }
                Op::If { condition } => {
                    em.if_begin(&mut src, &mut indent, kernel, condition, &mut scope_level)?;
                }
                Op::EndIf => {
                    if_end(&mut src, &mut indent, &mut scope_level);
                }
                Op::Barrier => {}
                Op::Range { .. } => {
                    em.emit_op(&mut src, &indent, op_id, reader_data, &self.noc, scope_level)?;
                }
                Op::Const(_) => {
                    // Inlined as literals at uses; no declaration emitted.
                }
                Op::Binary { .. } | Op::Mad { .. } | Op::Cast { .. } => {
                    em.emit_op(&mut src, &indent, op_id, reader_data, &self.noc, scope_level)?;
                }
                Op::Unary { .. }
                | Op::Bitcast { .. }
                | Op::Stack { .. }
                | Op::Index { .. }
                | Op::Wmma { .. }
                | Op::ReduceTile { .. }
                | Op::MatmulTile { .. }
                | Op::TransposeTile { .. }
                | Op::BroadcastTile { .. }
                | Op::Asm { .. }
                | Op::Move { .. }
                | Op::Reduce { .. } => todo!("tenstorrent2 reader op {op_id}: {:?}", kernel.ops[op_id].op),
            }
            self.cb.close_events(&mut src, &indent, op_id);
        }
        self.noc.final_read_barrier(&mut src, &indent);
        writeln!(src, "}}");
        self.cb.assert_settled("reader");
        self.reader = TTKernel::Reader { src, ordinals: ordinals.to_vec() };
        Ok(())
    }

    /// Generate the compute kernel from the compute op list: pure
    /// dispatch — every arm calls exactly one emitter method, no
    /// strings here. CB loads stream through DST slots (wait + copy +
    /// pop in the load arm); fused tile ops (`matmul`, `reduce`) do
    /// their own wait/op/pop traffic; stores pack to CBs or thread
    /// Register accs. Entry: loaded CBs pushed, stored-only CBs free;
    /// exit: settled. Anything beyond the closed shapes (tiled ALU, CB
    /// copies outside the streaming pattern, DRAM touched from
    /// compute) is a loud halt.
    #[allow(unused_must_use)]
    fn generate_compute(
        &mut self,
        kernel: &Kernel,
        compute_data: &SectionData,
        params: &[OpId],
        ordinals: &[u32],
    ) -> Result<(), BackendError> {
        // Placement first: every config op's plan must exist before
        // its arm emits.
        self.compute_init_plans(kernel, compute_data);
        // Seed the unpack tracking from the kernel-top block (it
        // executes before every body op): a trailing `MmTop`
        // programs matmul geometry for `cb_a`. Only plain copy
        // inits may follow it (proven to program no format);
        // anything else leaves the seed empty (a missing seed only
        // ever skips a reconfig the old code never did either).
        if let Some(top) = self.tl.top.iter().rev().find_map(|init| match *init {
            PlacedInit::MmTop(a, _) => Some(Some(a)),
            PlacedInit::Full(Cfg::Copy(_)) => None,
            _ => Some(None),
        }) {
            if let Some(a) = top {
                let fmt = self.cb.config[a].0;
                self.tl.note_unpack(a, fmt);
            }
        }
        let mut em = VarEmitter::new(kernel);
        let mut src = String::new();
        let mut indent = String::from("  ");
        self.noc.begin_section(params);
        let mut scope_level = 0u8;
        writeln!(src, "#include <cstdint>");
        writeln!(src, "#include \"api/compute/common.h\"");
        writeln!(src, "#include \"api/compute/compute_kernel_api.h\"");
                writeln!(src, "#include \"api/compute/eltwise_binary_sfpu.h\"");
        writeln!(src, "#include \"api/compute/eltwise_unary/binop_with_scalar.h\"");        writeln!(src, "#include \"api/compute/tile_move_copy.h\"");
        writeln!(src, "#include \"api/compute/eltwise_unary/eltwise_unary.h\"");
        writeln!(src, "#include \"api/compute/eltwise_unary/trigonometry.h\"");
        writeln!(src, "#include \"api/compute/eltwise_unary/exp.h\"");
        writeln!(src, "#include \"api/compute/eltwise_unary/recip.h\"");
        writeln!(src, "#include \"api/compute/eltwise_unary/rsqrt.h\"");
        writeln!(src, "#include \"api/compute/eltwise_unary/sqrt.h\"");
        writeln!(src, "#include \"api/compute/eltwise_unary/rounding.h\"");
        writeln!(src, "#include \"api/compute/eltwise_unary/negative.h\"");
        writeln!(src, "#include \"api/compute/eltwise_unary/bitwise_not.h\"");
        writeln!(src, "#include \"api/compute/eltwise_unary/typecast.h\"");
        writeln!(src, "#include \"api/compute/eltwise_unary/logical_not.h\"");
        writeln!(src, "#include \"api/compute/binary_max_min.h\"");
        writeln!(src, "#include \"api/compute/binary_shift.h\"");
        writeln!(src, "#include \"api/compute/eltwise_unary/fill.h\"");
        writeln!(src, "#include \"api/compute/matmul.h\"");
        writeln!(src, "#include \"api/compute/bcast.h\"");
        writeln!(src, "#include \"api/compute/reduce.h\"");
        writeln!(src, "#include \"api/compute/transpose_wh.h\"");
        writeln!(src, "#include \"api/compute/reconfig_data_format.h\"");
        writeln!(src, "#include \"api/dataflow/circular_buffer.h\"");
        writeln!(src, "#include \"api/debug/device_print.h\"");
        writeln!(src, "void kernel_main() {{");
        // Every shared CB is declared (matches reader/writer).
        self.cb.declare_all(&mut src, &indent);
        // Entry states: CBs this section loads start pushed (produced
        // upstream); stored-only CBs start free. A CB both stored and
        // loaded in-section (mid-compute roundtrip) starts free: the
        // store pushes, the later load waits. First-touch load/store
        // order also resolves the startup triple.
        let mut loaded_here: Set<CBId> = Set::default();
        let mut loaded_order: Vec<CBId> = Vec::new();
        let mut stored_here: Set<CBId> = Set::default();
        let mut stored_first: Option<CBId> = None;
        for &op_id in &compute_data.ops {
            if let Op::Load { src, layout: MemLayout::Tile { .. }, .. } = kernel.ops[op_id].op {
                if let Some(&cb) = self.cb.map.get(&src) {
                    if loaded_here.insert(cb) {
                        loaded_order.push(cb);
                    }
                }
            }
            if let Op::Store { dst, layout: MemLayout::Tile { .. }, .. } = kernel.ops[op_id].op {
                if let Some(&cb) = self.cb.map.get(&dst) {
                    stored_here.insert(cb);
                    if stored_first.is_none() {
                        stored_first = Some(cb);
                    }
                }
            }
        }
        self.cb.reset_all(CBState::Popped);
        for cb in loaded_here.difference(&stored_here) {
            self.cb.set(*cb, CBState::Pushed { avail: self.cb.produced(*cb) });
        }
        // Startup triple `[in0, in1, out]`: single-input kernels repeat
        // in0 (the two-arg overload does the same). Kernels with no
        // loads or no stores (pure movement) need no startup.
        if let (Some(&in0), Some(&out)) = (loaded_order.first(), stored_first.as_ref()) {
            let in1 = loaded_order.get(1).copied().unwrap_or(in0);
            self.tl.set_startup([in0, in1, out]);
        }

        // Anchor for the hoisted tile-op inits: the common method
        // prepends them here, ahead of all loops, after the walk.
        let init_anchor = src.len();

        for &op_id in &compute_data.ops {
            self.cb.open_events(&mut src, &indent, op_id);
            match kernel.ops[op_id].op {
                Op::Param { dtype, kind, .. } => match kind {
                    ParamKind::Variable => {
                        em.declare_variable(&mut src, &indent, compute_data, &self.noc, dtype, op_id, scope_level);
                    }
                    ParamKind::Global | ParamKind::GlobalMut => {
                        return Err(BackendError {
                            status: ErrorStatus::ComputeAccessesDram,
                            context: format!("tenstorrent2: compute touches DRAM through param {op_id}").into(),
                        });
                    }
                },
                Op::Const(_) => {
                    // Inlined as literals at uses; no declaration emitted.
                }
                Op::Cast { x, .. } => {
                    if matches!(compute_data.dtypes[&op_id].1, MemLayout::Tile { .. }) {
                        // Tiled cast, in place: the operand must be a
                        // single-use DST slot (multi-use needs a re-copy,
                        // unimplemented).
                        let tile = self.tl.tile_map.get(&x).copied().ok_or_else(|| BackendError {
                            status: ErrorStatus::KernelCompilation,
                            context: format!("tenstorrent2: tiled cast reads a value with no DST slot, op {op_id}").into(),
                        })?;
                        if compute_data.rcs[&x] != 1 {
                            todo!("tenstorrent2 multi-use tiled cast operand, op {op_id}");
                        }
                        let in_dt = compute_data.dtypes[&x].0;
                        let out_dt = compute_data.dtypes[&op_id].0;
                        self.tl.cast(&mut src, &indent, op_id, tile, in_dt, out_dt)?;
                    } else {
                        em.emit_op(&mut src, &indent, op_id, compute_data, &self.noc, scope_level)?;
                    }
                }
                Op::Bitcast { x, .. } => {
                    if matches!(compute_data.dtypes[&op_id].1, MemLayout::Tile { .. }) {
                        // Tiled bitcast is a no-op: same bits, new dtype.
                        // The slot aliases forward.
                        let tile = self.tl.tile_map.get(&x).copied().ok_or_else(|| BackendError {
                            status: ErrorStatus::KernelCompilation,
                            context: format!("tenstorrent2: tiled bitcast reads a value with no DST slot, op {op_id}").into(),
                        })?;
                        self.tl.tile_map.insert(op_id, tile);
                    } else {
                        em.emit_op(&mut src, &indent, op_id, compute_data, &self.noc, scope_level)?;
                    }
                }
                Op::Unary { x, uop } => {
                    if matches!(compute_data.dtypes[&op_id].1, MemLayout::Tile { .. }) {
                        // Tiled unary, in place: single-use operand only.
                        let tile = self.tl.tile_map.get(&x).copied().ok_or_else(|| BackendError {
                            status: ErrorStatus::KernelCompilation,
                            context: format!("tenstorrent2: tiled unary reads a value with no DST slot, op {op_id}").into(),
                        })?;
                        if compute_data.rcs[&x] != 1 {
                            todo!("tenstorrent2 multi-use tiled unary operand, op {op_id}");
                        }
                        self.tl.unary(&mut src, &indent, op_id, tile, uop);
                    } else {
                        em.emit_op(&mut src, &indent, op_id, compute_data, &self.noc, scope_level)?;
                    }
                }
                Op::Binary { x, y, bop } => {
                    if matches!(compute_data.dtypes[&op_id].1, MemLayout::Tile { .. }) {
                        // A BroadcastTile-marked side fuses into the
                        // CB-based broadcast form; otherwise the
                        // DST-register form.
                        let marker = |side: OpId| match kernel.ops[side].op {
                            Op::BroadcastTile { x: mx, kind } => Some((kind, mx)),
                            _ => None,
                        };
                        let plain_cb = |side: OpId| match kernel.ops[side].op {
                            Op::Load { src: lsrc, .. } => self.cb.map.get(&lsrc).copied(),
                            _ => None,
                        };
                        match (marker(x), marker(y)) {
                            (Some((kind, mx)), None) => {
                                let (Some(cb_b), Some(cb_a)) = (plain_cb(mx), plain_cb(y)) else {
                                    return Err(BackendError {
                                        status: ErrorStatus::KernelCompilation,
                                        context: format!("tenstorrent2: broadcast op {op_id} side is no CB tile load").into(),
                                    });
                                };
                                let rc = compute_data.rcs[&op_id];
                                self.tl.bcast(&mut src, &indent, &mut self.cb, op_id, cb_a, cb_b, bop, kind, rc);
                            }
                            (None, Some((kind, my))) => {
                                let (Some(cb_b), Some(cb_a)) = (plain_cb(my), plain_cb(x)) else {
                                    return Err(BackendError {
                                        status: ErrorStatus::KernelCompilation,
                                        context: format!("tenstorrent2: broadcast op {op_id} side is no CB tile load").into(),
                                    });
                                };
                                let rc = compute_data.rcs[&op_id];
                                self.tl.bcast(&mut src, &indent, &mut self.cb, op_id, cb_a, cb_b, bop, kind, rc);
                            }
                            (Some(_), Some(_)) => {
                                return Err(BackendError {
                                    status: ErrorStatus::KernelCompilation,
                                    context: format!("tenstorrent2: broadcast op {op_id} marks both sides").into(),
                                });
                            }
                            (None, None) => {
                                // Const side folds into the immediate
                                // form: (call name, tile side). Add/Mul
                                // commute; Sub picks sub/rsub by side;
                                // const-first Div has no call (loud).
                                let xc = const_f32_bits(kernel, x);
                                let yc = const_f32_bits(kernel, y);
                                let scalar = match (xc, yc) {
                                    (None, Some(bits)) => Some(("tile", bits, x)),
                                    (Some(bits), None) => Some(("const", bits, y)),
                                    _ => None,
                                };
                                if let Some((side, bits, tile_op)) = scalar {
                                    let name = match (bop, side) {
                                        (BOp::Add, _) => "add_unary_tile",
                                        (BOp::Mul, _) => "mul_unary_tile",
                                        (BOp::Sub, "tile") => "sub_unary_tile",
                                        (BOp::Sub, _) => "rsub_unary_tile",
                                        (BOp::Div, "tile") => "div_unary_tile",
                                        _ => {
                                            return Err(BackendError {
                                                status: ErrorStatus::KernelCompilation,
                                                context: format!("tenstorrent2: const-first {bop:?} has no scalar call, op {op_id}").into(),
                                            });
                                        }
                                    };
                                    let t = self.tl.tile_map.get(&tile_op).copied().ok_or_else(|| BackendError {
                                        status: ErrorStatus::KernelCompilation,
                                        context: format!("tenstorrent2: scalar binary reads a value with no DST slot, op {op_id}")
                                            .into(),
                                    })?;
                                    self.tl.bin_scalar(&mut src, &indent, op_id, t, name, bits);
                                } else {
                                    // Tiled binary: three-operand form, inputs stay
                                    // live, result in a fresh slot.
                                    let ta = self.tl.tile_map.get(&x).copied().ok_or_else(|| BackendError {
                                        status: ErrorStatus::KernelCompilation,
                                        context: format!("tenstorrent2: tiled binary reads a value with no DST slot, op {op_id}")
                                            .into(),
                                    })?;
                                    let tb = self.tl.tile_map.get(&y).copied().ok_or_else(|| BackendError {
                                        status: ErrorStatus::KernelCompilation,
                                        context: format!("tenstorrent2: tiled binary reads a value with no DST slot, op {op_id}")
                                            .into(),
                                    })?;
                                    let rc = compute_data.rcs[&op_id];
                                    self.tl.binary(&mut src, &indent, op_id, ta, tb, bop, rc);
                                }
                            }
                        }
                    } else {
                        em.emit_op(&mut src, &indent, op_id, compute_data, &self.noc, scope_level)?;
                    }
                }
                Op::BroadcastTile { .. } => {
                    // Marker: no traffic of its own. The marked load
                    // drains at the consuming fused binary; the plain
                    // binary path never sees this op.
                }
                Op::Mad { .. } => {
                    if matches!(compute_data.dtypes[&op_id].1, MemLayout::Tile { .. }) {
                        todo!("tenstorrent2 tiled mad, op {op_id}");
                    } else {
                        em.emit_op(&mut src, &indent, op_id, compute_data, &self.noc, scope_level)?;
                    }
                }
                Op::Stack { .. } => todo!(),
                Op::Storage { scope, .. } => match scope {
                    MemScope::Circular => {
                        // Circular CBs are declared up front; no traffic here.
                    }
                    MemScope::Register => {
                        // Acc declaration takes the MATH lock (which
                        // zeroes the file: the seed) and allocates the
                        // slot: numbering follows IR declaration order,
                        // not first-use accidents. Later takes in the
                        // cone keep. Compute-only: other sections must
                        // not emit MATH traffic.
                        self.tl.math_lock(&mut src, &indent);
                        self.tl.acc_tile(op_id, compute_data.rcs[&op_id]);
                    }
                    MemScope::Local => unreachable!(
                        "tenstorrent does not have local threads; local indices should have been converted to loops by the opt_tenstorrent_tile optimization pass"
                    ),
                    MemScope::Global => todo!("tenstorrent2 compute storage scope"),
                },
                Op::Store { ref dst, src: ref store_src, layout: st_layout, .. } => {
                    if !matches!(st_layout, MemLayout::Tile { .. }) {
                        todo!("tenstorrent2 compute only supports tile stores");
                    }
                    if let Op::Storage { scope: MemScope::Register, .. } = kernel.ops[*dst].op {
                        // Acc-threading store: alias the dst storage to
                        // the src tile. Seeds (src not yet a tile) emit
                        // and record nothing: the lazy acquire zeroes
                        // the file on first tile-op use.
                        if let Some(&tile) = self.tl.tile_map.get(store_src) {
                            self.tl.tile_map.insert(*dst, tile);
                        } else if let Op::Load { src: lsrc, .. } = kernel.ops[*store_src].op {
                            if let Some(&tile) = self.tl.tile_map.get(&lsrc) {
                                self.tl.tile_map.insert(*dst, tile);
                            }
                        } else {
                            return Err(BackendError {
                                status: ErrorStatus::KernelCompilation,
                                context: format!("tenstorrent2: compute acc store reads a tile with no DST slot, op {op_id}")
                                    .into(),
                            });
                        }
                    } else {
                        // Pack path: the src tile drains to a CB.
                        // `pack` closes any open reduce cone, commits
                        // MATH, then runs the reserve/wait/pack/push
                        // /release drain.
                        let Some(&out_cb) = self.cb.map.get(dst) else {
                            return Err(BackendError {
                                status: ErrorStatus::KernelCompilation,
                                context: format!("tenstorrent2: compute store targets unmapped CB, op {op_id}").into(),
                            });
                        };
                        let tile = if let Some(&tile) = self.tl.tile_map.get(store_src) {
                            tile
                        } else if let Op::Load { src: lsrc, .. } = kernel.ops[*store_src].op {
                            self.tl.tile_map.get(&lsrc).copied().ok_or_else(|| BackendError {
                                status: ErrorStatus::KernelCompilation,
                                context: format!("tenstorrent2: compute store reads a tile with no DST slot, op {op_id}").into(),
                            })?
                        } else {
                            return Err(BackendError {
                                status: ErrorStatus::KernelCompilation,
                                context: format!("tenstorrent2: compute store reads a tile with no DST slot, op {op_id}").into(),
                            });
                        };
                        let b = self.cb.batch(op_id).expect("tenstorrent2: pre-pass assigned a batch to every traffic op");
                        // The sync ops anchor at the transaction's
                        // open/close events; this text packs one tile
                        // per execution sequentially.
                        self.tl.pack(&mut src, &indent, &mut self.cb, tile, out_cb);
                        self.cb.record_move(b.cb, b.per_op);
                    }
                }
                Op::Load { src: ref load_src, layout, .. } => {
                    if !matches!(layout, MemLayout::Tile { .. }) {
                        todo!("tenstorrent2 compute only supports tile loads");
                    }
                    // Register acc loads resolve to the declared slot
                    // (allocated at the Storage op): no CB, no traffic.
                    if matches!(kernel.ops[*load_src].op, Op::Storage { scope: MemScope::Register, .. }) {
                        let tile = self.tl.tile_map.get(load_src).copied().ok_or_else(|| BackendError {
                            status: ErrorStatus::KernelCompilation,
                            context: format!("tenstorrent2: compute acc load reads an undeclared acc, op {op_id}").into(),
                        })?;
                        self.tl.tile_map.insert(op_id, tile);
                    } else {
                        let Some(&cb) = self.cb.map.get(load_src) else {
                            return Err(BackendError {
                                status: ErrorStatus::KernelCompilation,
                                context: format!("tenstorrent2: compute load targets unmapped CB, op {op_id}").into(),
                            });
                        };
                        // Loads feeding only fused tile ops resolve to CB
                        // ids at the consuming op (no copy_tile — the
                        // reference has none). Every other load streams
                        // through a DST slot: wait, copy, pop.
                        let mut fused_only = false;
                        for &consumer in &compute_data.ops {
                            if kernel.ops[consumer].op.parameters().any(|p| p == op_id) {
                                if match kernel.ops[consumer].op {
                                    Op::ReduceTile { .. } | Op::MatmulTile { .. } | Op::TransposeTile { .. } | Op::BroadcastTile { .. } => true,
                                    Op::Binary { x, y, .. } => {
                                        matches!(kernel.ops[x].op, Op::BroadcastTile { .. })
                                            || matches!(kernel.ops[y].op, Op::BroadcastTile { .. })
                                    }
                                    _ => false,
                                } {
                                    fused_only = true;
                                } else {
                                    fused_only = false;
                                    break;
                                }
                            }
                        }
                        if !fused_only {
                            let rc = compute_data.rcs[&op_id];
                            let Op::Load { index: ld_idx, .. } = kernel.ops[op_id].op else {
                                unreachable!()
                            };
                            let b = self.cb.batch(op_id).expect("tenstorrent2: pre-pass assigned a batch to every traffic op");
                            // The sync ops anchor at the transaction's
                            // open/close events; this text copies its
                            // block slot.
                            let off = self.cb.slot_offset(kernel, &mut em, compute_data, ld_idx, b, scope_level, "compute")?;
                            self.cb.record_move(b.cb, b.per_op);
                            let fmt = self.cb.config[cb].0;
                            self.tl.copy(&mut src, &indent, op_id, cb, rc, &off, fmt);
                        }
                    }
                }
                Op::Range { .. } => {
                    em.emit_op(&mut src, &indent, op_id, compute_data, &self.noc, scope_level)?;
                }
                Op::Loop { .. } => {
                    // A placement keyed by this boundary emits ahead
                    // of the whole scope (the pass hoisted exactly one
                    // occupant outward here). Hoisted copy inits get
                    // the same stale-format treatment as inline ones;
                    // shorts update the unpack tracking.
                    if let Some(init) = self.tl.pop(op_id) {
                        match init {
                            PlacedInit::Full(Cfg::Copy(c)) => {
                                let fmt = self.cb.config[c].0;
                                writeln!(src, "{indent}{}", self.tl.copy_init(c, fmt));
                            }
                            PlacedInit::MmShort(a, _, _) => {
                                writeln!(src, "{indent}{}", self.tl.line(init, None)?);
                                let fmt = self.cb.config[a].0;
                                self.tl.note_unpack(a, fmt);
                            }
                            _ => {
                                writeln!(src, "{indent}{}", self.tl.line(init, None)?);
                            }
                        }
                    }
                    em.loop_begin(&mut src, &mut indent, kernel, compute_data, op_id, &mut scope_level)?;
                }
                Op::EndLoop => {
                    // No open cone across the back-edge: the body is
                    // emitted once but runs N times, so a PACK-held
                    // file here would deadlock iteration 2+ on an
                    // acquire whose release sits past the loop.
                    self.tl.flush_pack(&mut src, &indent);
                    loop_end(&mut src, &mut indent, &mut scope_level);
                }
                Op::If { condition } => {
                    // Same boundary rule as loops: a placement keyed
                    // here runs unconditionally ahead of the branch
                    // (inits program engines without data side
                    // effects, so the untaken path is harmless).
                    // Plain brace scope otherwise: no lock/CB
                    // interaction (those pair with loops, never ifs).
                    if let Some(init) = self.tl.pop(op_id) {
                        match init {
                            PlacedInit::Full(Cfg::Copy(c)) => {
                                let fmt = self.cb.config[c].0;
                                writeln!(src, "{indent}{}", self.tl.copy_init(c, fmt));
                            }
                            PlacedInit::MmShort(a, _, _) => {
                                writeln!(src, "{indent}{}", self.tl.line(init, None)?);
                                let fmt = self.cb.config[a].0;
                                self.tl.note_unpack(a, fmt);
                            }
                            _ => {
                                writeln!(src, "{indent}{}", self.tl.line(init, None)?);
                            }
                        }
                    }
                    em.if_begin(&mut src, &mut indent, kernel, condition, &mut scope_level)?;
                }
                Op::EndIf => {
                    if_end(&mut src, &mut indent, &mut scope_level);
                }
                Op::Index { .. } => todo!(),
                Op::MatmulTile { x, y, acc } => {
                    // Sides are CB tile loads (shapes proven by
                    // `check_matmul_closed`); the acc load threads a
                    // Register acc. The fused op does the wait/op/pop
                    // traffic into the acc slot.
                    let Op::Load { src: la, layout: MemLayout::Tile { .. }, .. } = kernel.ops[x].op else {
                        return Err(BackendError {
                            status: ErrorStatus::KernelCompilation,
                            context: format!("tenstorrent2: matmul side op {x} is no CB tile load").into(),
                        });
                    };
                    let Some(&cb_a) = self.cb.map.get(&la) else {
                        return Err(BackendError {
                            status: ErrorStatus::KernelCompilation,
                            context: format!("tenstorrent2: matmul side op {x} targets unmapped CB").into(),
                        });
                    };
                    let Op::Load { src: lb, layout: MemLayout::Tile { .. }, .. } = kernel.ops[y].op else {
                        return Err(BackendError {
                            status: ErrorStatus::KernelCompilation,
                            context: format!("tenstorrent2: matmul side op {y} is no CB tile load").into(),
                        });
                    };
                    let Some(&cb_b) = self.cb.map.get(&lb) else {
                        return Err(BackendError {
                            status: ErrorStatus::KernelCompilation,
                            context: format!("tenstorrent2: matmul side op {y} targets unmapped CB").into(),
                        });
                    };
                    let Op::Load { src: lacc, .. } = kernel.ops[acc].op else {
                        return Err(BackendError {
                            status: ErrorStatus::KernelCompilation,
                            context: format!("tenstorrent2: matmul acc op {acc} is no acc tile load").into(),
                        });
                    };
                    let rc = compute_data.rcs[&op_id];
                    let tile = self.tl.acc_tile(lacc, rc);
                    let out = self.tl.startup.map(|[_, _, o]| o).expect("tenstorrent2: matmul kernel without startup triple");
                    self.tl.matmul(&mut src, &indent, &mut self.cb, op_id, cb_a, cb_b, out, tile);
                }
                Op::TransposeTile { x } => {
                    // x is a tile load from a live input CB. The op
                    // streams the tile through transpose_wh into a
                    // fresh DST slot (wait/op/pop traffic, no acc).
                    let Op::Load { src: lx, layout: MemLayout::Tile { x: wx, y: hx, .. }, .. } = kernel.ops[x].op else {
                        return Err(BackendError {
                            status: ErrorStatus::KernelCompilation,
                            context: format!("tenstorrent2: transpose side op {x} is no CB tile load").into(),
                        });
                    };
                    if wx as u32 != 32 || hx as u32 != 32 {
                        return Err(BackendError {
                            status: ErrorStatus::KernelCompilation,
                            context: format!("tenstorrent2: transpose is fixed 32x32, op {x} is {wx}x{hx}").into(),
                        });
                    }
                    let Some(&cb) = self.cb.map.get(&lx) else {
                        return Err(BackendError {
                            status: ErrorStatus::KernelCompilation,
                            context: format!("tenstorrent2: transpose side op {x} targets unmapped CB").into(),
                        });
                    };
                    let out = self.tl.startup.map(|[_, _, o]| o).expect("tenstorrent2: transpose kernel without startup triple");
                    let rc = compute_data.rcs[&op_id];
                    self.tl.transpose(&mut src, &indent, &mut self.cb, op_id, cb, out, rc);
                }
                Op::ReduceTile { x, scaler, acc, rop, kind } => {
                    // x is a tile load from a live input CB, scaler a
                    // tile load from the scaler CB, acc a load threading
                    // a Register acc. The fused op does the
                    // wait/op/pop traffic into the acc slot.
                    let Op::Load { src: lx, layout: MemLayout::Tile { x: wx, y: hx, .. }, .. } = kernel.ops[x].op else {
                        return Err(BackendError {
                            status: ErrorStatus::KernelCompilation,
                            context: format!("tenstorrent2: reduce side op {x} is no CB tile load").into(),
                        });
                    };
                    if wx as u32 != 32 || hx as u32 != 32 {
                        return Err(BackendError {
                            status: ErrorStatus::KernelCompilation,
                            context: format!("tenstorrent2: reduce is fixed 32x32, op {x} is {wx}x{hx}").into(),
                        });
                    }
                    let Some(&cb_in) = self.cb.map.get(&lx) else {
                        return Err(BackendError {
                            status: ErrorStatus::KernelCompilation,
                            context: format!("tenstorrent2: reduce side op {x} targets unmapped CB").into(),
                        });
                    };
                    let Op::Load { src: la, layout: MemLayout::Tile { .. }, .. } = kernel.ops[acc].op else {
                        return Err(BackendError {
                            status: ErrorStatus::KernelCompilation,
                            context: format!("tenstorrent2: reduce acc op {acc} is no acc tile load").into(),
                        });
                    };
                    if !matches!(kernel.ops[la].op, Op::Storage { scope: MemScope::Register, .. }) {
                        return Err(BackendError {
                            status: ErrorStatus::KernelCompilation,
                            context: format!("tenstorrent2: reduce acc op {acc} does not thread a Register acc").into(),
                        });
                    }
                    let Op::Load { src: ls, layout: MemLayout::Tile { .. }, .. } = kernel.ops[scaler].op else {
                        return Err(BackendError {
                            status: ErrorStatus::KernelCompilation,
                            context: format!("tenstorrent2: reduce scaler op {scaler} is no scaler tile load").into(),
                        });
                    };
                    let Some(&cb_sc) = self.cb.map.get(&ls) else {
                        return Err(BackendError {
                            status: ErrorStatus::KernelCompilation,
                            context: format!("tenstorrent2: reduce scaler op {scaler} targets unmapped CB").into(),
                        });
                    };
                    for &s in &[lx, ls] {
                        let Op::Storage { dtype: DType::F16 | DType::BF16, .. } = kernel.ops[s].op else {
                            return Err(BackendError {
                                status: ErrorStatus::KernelCompilation,
                                context: format!("tenstorrent2: reduce tiles are F16/BF16, op {op_id} is not").into(),
                            });
                        };
                    }
                    let rc = compute_data.rcs[&op_id];
                    let tile = self.tl.acc_tile(la, rc);
                    self.tl.reduce(&mut src, &indent, &mut self.cb, op_id, cb_in, cb_sc, tile, rop, kind, rc)?;
                }
                Op::Asm { ref asm, ref ops } => {
                    // Statement template (CUDA expression form generalized):
                    // `{i}` substitutes the i-th operand — circular CBs
                    // render as their CB index, materialized tiles as
                    // their DST slot. Anything else is a loud error.
                    let mut rendered: String = asm.as_str().into();
                    for (i, &operand) in ops.iter().enumerate() {
                        let var = match kernel.ops[operand].op {
                            Op::Storage { scope: MemScope::Circular, .. } => {
                                let Some(&cb) = self.cb.map.get(&operand) else {
                                    return Err(BackendError {
                                        status: ErrorStatus::KernelCompilation,
                                        context: format!("tenstorrent2: asm operand {operand} targets unmapped CB").into(),
                                    });
                                };
                                format!("{cb}")
                            }
                            _ => {
                                let Some(&slot) = self.tl.tile_map.get(&operand) else {
                                    return Err(BackendError {
                                        status: ErrorStatus::KernelCompilation,
                                        context: format!("tenstorrent2: asm operand {operand} is not a CB or live tile").into(),
                                    });
                                };
                                format!("{slot}")
                            }
                        };
                        rendered = rendered.replace(&format!("{{{i}}}"), &var);
                    }
                    writeln!(src, "{indent}{rendered};");
                }
                Op::Barrier => unreachable!("should've been filtered by kernel sections decomposition"),
                Op::Move { .. } | Op::Reduce { .. } => unreachable!("should've been lowered by linearize"),
                Op::Wmma { .. } => unreachable!("tt does not support wmma, use Op::MatmulTile instead"),
            }
            self.cb.close_events(&mut src, &indent, op_id);
        }
        // Kernel-top block (startup/mm_init plus any top-hoisted
        // init) goes ahead of all loops, at the recorded anchor.
        self.tl.prepend_compute_inits(&mut src, init_anchor)?;

        self.tl.flush_pack(&mut src, &indent);
        writeln!(src, "}}");
        self.cb.assert_settled("compute");
        assert_eq!(
            self.tl.state,
            TileState::Unlocked,
            "tenstorrent2: compute ends with DST in {:?}, every lock must pair",
            self.tl.state
        );
        self.compute = TTKernel::Compute { src, ordinals: ordinals.to_vec() };

        Ok(())
    }

    /// Generate the writer (dataflow drain) kernel from the writer op
    /// list: pure dispatch — every arm calls exactly one emitter
    /// method, no strings here. Entry: drained CBs pushed, the rest
    /// free; exit: settled.
    #[allow(unused_must_use)]
    fn generate_writer(
        &mut self,
        kernel: &Kernel,
        writer_data: &SectionData,
        params: &[OpId],
        ordinals: &[u32],
    ) -> Result<(), BackendError> {
        let ops = &writer_data.ops;
        let mut em = VarEmitter::new(kernel);
        let mut src = String::new();
        let mut indent = String::from("  ");
        self.noc.begin_section(params);
        let mut scope_level = 0u8;
        writeln!(src, "#include <cstdint>");
        writeln!(src, "#include \"api/dataflow/dataflow_api.h\"");
        writeln!(src, "#include \"api/dataflow/noc.h\"");
        writeln!(src, "#include \"api/dataflow/circular_buffer.h\"");
        writeln!(src, "#include \"api/tensor/noc_traits.h\"");
        writeln!(src, "#include \"api/debug/dprint.h\"");
        writeln!(src, "void kernel_main() {{");
        self.cb.declare_all(&mut src, &indent);
        // Entry states: CBs this section drains start pushed; the rest
        // start free.
        let mut loaded_here: Set<CBId> = Set::default();
        for &op_id in ops {
            if let Op::Load { src, layout: MemLayout::Tile { .. }, .. } = kernel.ops[op_id].op {
                if let Some(&cb) = self.cb.map.get(&src) {
                    loaded_here.insert(cb);
                }
            }
        }
        self.cb.reset_all(CBState::Popped);
        for cb in loaded_here {
            self.cb.set(cb, CBState::Pushed { avail: self.cb.produced(cb) });
        }
        // Accessors only for the GlobalMut params this section writes.
        for &op_id in ops {
            if let Op::Param { kind: ParamKind::GlobalMut, .. } = kernel.ops[op_id].op {
                self.noc.declare_writer_out(&mut src, &indent, op_id);
            }
        }
        let n = ops.len();
        for i in 0..n {
            let op_id = ops[i];
            self.cb.open_events(&mut src, &indent, op_id);
            match kernel.ops[op_id].op {
                Op::Param { kind: ParamKind::GlobalMut, .. } => {
                    // Accessor emitted up front (see above).
                }
                Op::Param { kind: ParamKind::Variable, .. } => {
                    let Op::Param { dtype, .. } = kernel.ops[op_id].op else {
                        unreachable!("tenstorrent2 param changed under us");
                    };
                    em.declare_variable(&mut src, &indent, writer_data, &self.noc, dtype, op_id, scope_level);
                }
                Op::Storage { scope, .. } => match scope {
                    MemScope::Circular | MemScope::Register => {
                        // Circular CBs are declared up front; Register
                        // accs thread through `tile_map`. Neither emits
                        // traffic.
                    }
                    MemScope::Local => unreachable!(
                        "tenstorrent does not have local threads; local indices should have been converted to loops by the opt_tenstorrent_tile optimization pass"
                    ),
                    MemScope::Global => todo!("tenstorrent2 writer storage scope"),
                },
                Op::Load { .. } => {
                    // Consumed at the draining store below.
                }
                Op::Store { ref dst, src: ref store_src, index: st_idx, layout: st_layout } => {
                    let Op::Load { src: cb_src, index: ld_idx, layout: ld_layout } = kernel.ops[*store_src].op else {
                        return Err(BackendError {
                            status: ErrorStatus::KernelCompilation,
                            context: format!(
                                "tenstorrent2: writer supports only CB to DRAM stores, op {op_id} has ops in between"
                            )
                            .into(),
                        });
                    };
                    let Some(&cb) = self.cb.map.get(&cb_src) else {
                        return Err(BackendError {
                            status: ErrorStatus::KernelCompilation,
                            context: format!("tenstorrent2: writer load op {op_id} targets unmapped CB").into(),
                        });
                    };
                    let Op::Param { dtype, kind: ParamKind::GlobalMut, .. } = kernel.ops[*dst].op else {
                        return Err(BackendError {
                            status: ErrorStatus::KernelCompilation,
                            context: format!("tenstorrent2: writer store dst must be a GlobalMut param, op {op_id}").into(),
                        });
                    };
                    match (ld_layout, st_layout) {
                        (MemLayout::Tile { x, y, .. }, MemLayout::Tile { .. }) => {
                            let elem_size = dtype.bit_size() as u32 / 8;
                            let tile_bytes = x as u32 * y as u32 * elem_size;
                            let idx = em.resolve_idx(kernel, writer_data, st_idx, scope_level, "writer")?;
                            let b = self.cb.batch(op_id).expect("tenstorrent2: pre-pass assigned a batch to every traffic op");
                            // The sync ops anchor at the transaction's
                            // open/close events; this text drains the
                            // block at its slot.
                            let off = self.cb.slot_offset(kernel, &mut em, writer_data, ld_idx, b, scope_level, "writer")?;
                            self.cb.record_move(b.cb, b.per_op);
                            self.noc.async_write_tile(&mut src, &indent, op_id, *dst, &idx, elem_size, tile_bytes, cb, &off);
                        }
                        _ => todo!("tenstorrent2 writer only supports tile stores"),
                    }
                }
                Op::Loop { .. } => {
                    em.loop_begin(&mut src, &mut indent, kernel, writer_data, op_id, &mut scope_level)?;
                }
                Op::EndLoop => {
                    loop_end(&mut src, &mut indent, &mut scope_level);
                }
                Op::If { condition } => {
                    em.if_begin(&mut src, &mut indent, kernel, condition, &mut scope_level)?;
                }
                Op::EndIf => {
                    if_end(&mut src, &mut indent, &mut scope_level);
                }
                Op::Barrier => {}
                Op::Range { .. } => {
                    em.emit_op(&mut src, &indent, op_id, writer_data, &self.noc, scope_level)?;
                }
                Op::Const(_) => {
                    // Inlined as literals at uses; no declaration emitted.
                }
                Op::Binary { .. } | Op::Mad { .. } | Op::Cast { .. } => {
                    em.emit_op(&mut src, &indent, op_id, writer_data, &self.noc, scope_level)?;
                }
                Op::Param { .. } => {
                    todo!("tenstorrent2 writer Global param")
                }
                Op::Unary { .. }
                | Op::Bitcast { .. }
                | Op::Stack { .. }
                | Op::Index { .. }
                | Op::Wmma { .. }
                | Op::ReduceTile { .. }
                | Op::MatmulTile { .. }
                | Op::TransposeTile { .. }
                | Op::BroadcastTile { .. }
                | Op::Asm { .. }
                | Op::Move { .. }
                | Op::Reduce { .. } => todo!("tenstorrent2 writer op"),
            }
            self.cb.close_events(&mut src, &indent, op_id);
        }
        writeln!(src, "}}");
        self.cb.assert_settled("writer");
        self.writer = TTKernel::Writer { src, ordinals: ordinals.to_vec() };
        Ok(())
    }
}

impl VarEmitter<'_> {
    /// Shared basic-op emission across all sections: identical text
    /// everywhere, no section argument. Dispatched on layout: scalar in
    /// emits scalar C++, tiled in emits the TT tile op on DST slots.
    /// One register file (`var_map`/`vars`) holds both: `MemLayout` on
    /// the slot tells them apart, scalars format as `r{reg}`, tiled
    /// values as the bare DST slot id. Tiled ALU only ever appears in
    /// compute; reader/writer closures hold scalar index math. The tile
    /// arms insert into the init sets as they emit; the caller prepends
    /// the collected `*_init` calls pre-loop after the walk. Loads,
    /// stores, and loops stay in each section generator.
    #[allow(unused_must_use)]
    fn emit_op(
        &mut self,
        src: &mut String,
        indent: &str,
        op_id: OpId,
        data: &SectionData,
        noc: &NocEmitter,
        scope_level: u8,
    ) -> Result<(), BackendError> {
        let kernel = self.kernel;
        let var_map = &mut self.var_map;
        let vars = &mut self.vars;
        /// Standard allocation into `vars`: reuse a dead slot with
        /// matching dtype/layout from another block, else push a new
        /// one. Returns the `r{reg}` index and records `op_id`.
        fn new_var(
            op_id: OpId,
            var_map: &mut Map<OpId, u32>,
            vars: &mut Vec<VarSlot>,
            dtype: DType,
            layout: MemLayout,
            rcs: &Map<OpId, u32>,
            scope_level: u8,
        ) -> u32 {
            let rc: u32 = rcs[&op_id];
            for (i, s) in vars.iter_mut().enumerate() {
                if s.rc == 0 && s.dtype == dtype && s.layout == layout && s.scope_level != scope_level {
                    s.dtype = dtype;
                    s.layout = layout;
                    s.rc = rc;
                    s.scope_level = scope_level;
                    var_map.insert(op_id, i as u32);
                    return i as u32;
                }
            }
            let r = vars.len() as u32;
            vars.push(VarSlot { dtype, layout, rc, scope_level });
            var_map.insert(op_id, r);
            r
        }
        /// Shared operand resolution: constants inline as literals,
        /// loop/param/vars names resolve through `var_map` with the
        /// house use-site rule (same-level use decrements, deeper
        /// uses stay live). Anything else is a compilation error.
        /// Scalars format as `r{reg}`; tiled values (DST slots) as the
        /// bare slot id. Tiled entries never decrement: one static slot
        /// per use-site, reused across loop trips.
        fn get_var(
            kernel: &Kernel,
            id: OpId,
            var_map: &Map<OpId, u32>,
            vars: &mut [VarSlot],
            scope_level: u8,
        ) -> Result<String, BackendError> {
            if let Op::Const(c) = &kernel.ops[id].op {
                return Ok(format!("{}", c.c_code()));
            }
            if let Some(&r) = var_map.get(&id) {
                let s = &mut vars[r as usize];
                if matches!(s.layout, MemLayout::Tile { .. }) {
                    return Ok(format!("{r}"));
                }
                if s.scope_level == scope_level {
                    debug_assert!(s.rc > 0);
                    s.rc -= 1;
                }
                return Ok(format!("r{r}"));
            }
            Err(BackendError {
                status: ErrorStatus::KernelCompilation,
                context: format!("tenstorrent2: operand {id} not found in constants or registers").into(),
            })
        }
        match &kernel.ops[op_id].op {
            Op::Const(_) => unreachable!("tenstorrent2 consts inline as literals at their uses"),
            Op::Binary { x, y, bop } => {
                let dt = data.dtypes[&op_id].0;
                let rlay = data.dtypes[&op_id].1;
                let xlay = data.dtypes[x].1;
                let ylay = data.dtypes[y].1;
                if !matches!(xlay, MemLayout::Scalar) || !matches!(ylay, MemLayout::Scalar) {
                    todo!("tenstorrent2 binary over non-scalar layout");
                }
                let x = get_var(kernel, *x, var_map, vars, scope_level)?;
                let y = get_var(kernel, *y, var_map, vars, scope_level)?;
                let reg = new_var(op_id, &mut *var_map, &mut *vars, dt, rlay, &data.rcs, scope_level);
                let _ = match bop {
                    BOp::Add => writeln!(src, "{indent}{} r{reg} = {x} + {y};", dt.c_type()),
                    BOp::Sub => writeln!(src, "{indent}{} r{reg} = {x} - {y};", dt.c_type()),
                    BOp::Mul => writeln!(src, "{indent}{} r{reg} = {x} * {y};", dt.c_type()),
                    BOp::Div => writeln!(src, "{indent}{} r{reg} = {x} / {y};", dt.c_type()),
                    BOp::Mod => writeln!(src, "{indent}{} r{reg} = {x} % {y};", dt.c_type()),
                    BOp::Max => writeln!(src, "{indent}{} r{reg} = {x} > {y} ? {x} : {y};", dt.c_type()),
                    BOp::Cmplt => writeln!(src, "{indent}{} r{reg} = {x} < {y};", dt.c_type()),
                    BOp::Cmpgt => writeln!(src, "{indent}{} r{reg} = {x} > {y};", dt.c_type()),
                    BOp::Cmpge => writeln!(src, "{indent}{} r{reg} = {x} >= {y};", dt.c_type()),
                    BOp::Eq => writeln!(src, "{indent}{} r{reg} = {x} == {y};", dt.c_type()),
                    BOp::NotEq => writeln!(src, "{indent}{} r{reg} = {x} != {y};", dt.c_type()),
                    BOp::And => writeln!(src, "{indent}{} r{reg} = {x} && {y};", dt.c_type()),
                    BOp::Or => writeln!(src, "{indent}{} r{reg} = {x} || {y};", dt.c_type()),
                    BOp::BitXor => writeln!(src, "{indent}{} r{reg} = {x} ^ {y};", dt.c_type()),
                    BOp::BitOr => writeln!(src, "{indent}{} r{reg} = {x} | {y};", dt.c_type()),
                    BOp::BitAnd => writeln!(src, "{indent}{} r{reg} = {x} & {y};", dt.c_type()),
                    BOp::BitShiftLeft => writeln!(src, "{indent}{} r{reg} = {x} << {y};", dt.c_type()),
                    BOp::BitShiftRight => writeln!(src, "{indent}{} r{reg} = {x} >> {y};", dt.c_type()),
                    BOp::Pow => todo!("tenstorrent2 scalar pow"),
                };
            }
            Op::Mad { x, y, z } => {
                let dt = data.dtypes[&op_id].0;
                let rlay = data.dtypes[&op_id].1;
                let xlay = data.dtypes[x].1;
                let ylay = data.dtypes[y].1;
                let zlay = data.dtypes[z].1;
                if !matches!(xlay, MemLayout::Scalar)
                    || !matches!(ylay, MemLayout::Scalar)
                    || !matches!(zlay, MemLayout::Scalar)
                    || !matches!(rlay, MemLayout::Scalar)
                {
                    todo!("tenstorrent2 mad over non-scalar layout");
                }
                let x = get_var(kernel, *x, var_map, vars, scope_level)?;
                let y = get_var(kernel, *y, var_map, vars, scope_level)?;
                let z = get_var(kernel, *z, var_map, vars, scope_level)?;
                let reg = new_var(op_id, &mut *var_map, &mut *vars, dt, rlay, &data.rcs, scope_level);
                writeln!(src, "{indent}{} r{reg} = {x} * {y} + {z};", dt.c_type());
            }
            Op::Cast { x, dtype } => {
                let dt = data.dtypes[&op_id].0;
                let rlay = data.dtypes[&op_id].1;
                debug_assert_eq!(dt, *dtype);
                let xlay = data.dtypes[x].1;
                if !matches!(xlay, MemLayout::Scalar) {
                    todo!("tenstorrent2 cast over non-scalar layout");
                }
                let x = get_var(kernel, *x, var_map, vars, scope_level)?;
                let reg = new_var(op_id, &mut *var_map, &mut *vars, dt, rlay, &data.rcs, scope_level);
                writeln!(src, "{indent}{} r{reg} = ({}){x};", dtype.c_type(), dtype.c_type());
            }
            Op::Loop { len } => {
                let bound = get_var(kernel, *len, var_map, vars, scope_level)?;
                let dt = data.dtypes[&op_id].0;
                let rlay = data.dtypes[&op_id].1;
                debug_assert!(matches!(rlay, MemLayout::Scalar));
                let reg = new_var(op_id, &mut *var_map, &mut *vars, dt, rlay, &data.rcs, scope_level);
                writeln!(src, "{indent}for (uint32_t r{reg} = 0; r{reg} < {bound}; r{reg}++) {{");
            }
            Op::Range { axis, kind } => match kind {
                RangeKind::Group(_) => {
                    let arg = noc.group_arg(*axis);
                    let dt = data.dtypes[&op_id].0;
                    let rlay = data.dtypes[&op_id].1;
                    debug_assert!(matches!(rlay, MemLayout::Scalar));
                    let reg = new_var(op_id, &mut *var_map, &mut *vars, dt, rlay, &data.rcs, scope_level);
                    writeln!(src, "{indent}uint32_t r{reg} = get_arg_val<uint32_t>({arg});");
                }
                RangeKind::Local(_) => {
                    unreachable!(
                        "tenstorrent does not have local threads; local indices should have been converted to loops by the opt_tenstorrent_tile optimization pass"
                    )
                }
                RangeKind::Warp(_) => {
                    unreachable!("tenstorrent has no warps; warp ranges are gpu-only")
                }
            },
            Op::Unary { .. } if matches!(data.dtypes[&op_id].1, MemLayout::Tile { .. }) => {
                todo!("tenstorrent2 tiled unary op");
            }
            Op::Param { .. }
            | Op::Storage { .. }
            | Op::Load { .. }
            | Op::Store { .. }
            | Op::EndLoop
            | Op::If { .. }
            | Op::EndIf
            | Op::Index { .. }
            | Op::Barrier
            | Op::Unary { .. }
            | Op::Bitcast { .. }
            | Op::Stack { .. }
            | Op::Wmma { .. }
            | Op::ReduceTile { .. }
            | Op::MatmulTile { .. }
            | Op::TransposeTile { .. }
            | Op::BroadcastTile { .. }
            | Op::Asm { .. }
            | Op::Move { .. }
            | Op::Reduce { .. } => todo!("tenstorrent2 scalar op"),
        }
        Ok(())
    }
}

/// tt-metal `tt::DataFormat` values used on the tile path.
/// Discriminants mirror `tt_backend_api_types.hpp` exactly (do NOT
/// confuse with the CB descriptor codes F32=0/F16=1/BF16=2, which are
/// a different enum). zyx F16 rides `Float16_b` — 0.72 has no
/// plain-Float16 SFPU kernel (see typecast.h supported list).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum TtDataFormat {
    F32 = 0,
    F16B = 5,
    I32 = 8,
    U16 = 9,
    U32 = 24,
}

impl std::fmt::Display for TtDataFormat {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", *self as u32)
    }
}

/// TT `DataFormat` for a zyx dtype on the tile path.
fn tt_fmt(dt: DType) -> Result<TtDataFormat, BackendError> {
    match dt {
        DType::F32 => Ok(TtDataFormat::F32),
        DType::F16 | DType::BF16 => Ok(TtDataFormat::F16B),
        DType::U16 => Ok(TtDataFormat::U16),
        DType::U32 => Ok(TtDataFormat::U32),
        DType::I32 => Ok(TtDataFormat::I32),
        dt => Err(BackendError {
            status: ErrorStatus::KernelCompilation,
            context: format!("tenstorrent2: dtype {dt:?} has no tt tile format").into(),
        }),
    }
}
