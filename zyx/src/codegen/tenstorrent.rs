use crate::{
    DType, Map, Set,
    error::{BackendError, ErrorStatus},
    kernel::{BOp, IDX_T, Kernel, MMADType, MemLayout, MemScope, Op, OpId, ParamKind, RangeKind, TileReduceKind, UOp},
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
/// The cycle is free (`Popped`) --reserve--> `Reserved` --push--> `Pushed`
/// --wait--> `Waiting` --pop--> free (`Popped`). `Popped` doubles as the
/// initial free state: the first transition on a fresh CB is always
/// `reserve`. At every section end all CBs must be settled (`Popped` or
/// `Pushed`); a `Reserved`/`Waiting` remainder is a leaked transaction.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CBState {
    /// Free: no live transaction, `reserve` may be called.
    Popped,
    /// Space reserved, not yet pushed.
    Reserved,
    /// Live data, may be waited on.
    Pushed,
    /// Waited on, must be popped.
    Waiting,
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
                Op::Cast { x, .. } | Op::Bitcast { x, .. } | Op::Unary { x, .. } => {
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

    /// Shared index resolution: consts inline as literals, every other
    /// value (including Variable params, slot-named at declaration)
    /// must already sit in `var_map`. Same-level uses decrement the
    /// refcount (house use-site rule).
    fn resolve_idx(&mut self, kernel: &Kernel, _data: &SectionData, idx_op: OpId, scope_level: u8, who: &str) -> Result<String, BackendError> {
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
    /// write pointer.
    fn async_read_tile(&self, src: &mut String, indent: &str, op_id: OpId, ld_src: OpId, idx: &str, elem_size: u32, tile_bytes: u32, cb: CBId) {
        writeln!(src, "{indent}uint64_t rnoc{op_id} = p{ld_src}.get_noc_addr((uint32_t)(({idx}*{elem_size})/{TT_DRAM_PAGE_BYTES}), (uint32_t)(({idx}*{elem_size})%{TT_DRAM_PAGE_BYTES}));");
        writeln!(src, "{indent}noc_async_read(rnoc{op_id}, cb{cb}.get_write_ptr(), {tile_bytes});");
        writeln!(src, "{indent}noc_async_read_barrier();");
    }

    /// `noc_async_write` of one tile plus its barrier: CB read pointer
    /// to DRAM address `wnoc{op_id}` in accessor `p_out{dst}`.
    fn async_write_tile(&self, src: &mut String, indent: &str, op_id: OpId, dst: OpId, idx: &str, elem_size: u32, tile_bytes: u32, cb: CBId) {
        writeln!(src, "{indent}uint64_t wnoc{op_id} = p_out{dst}.get_noc_addr((uint32_t)(({idx}*{elem_size})/{TT_DRAM_PAGE_BYTES}), (uint32_t)(({idx}*{elem_size})%{TT_DRAM_PAGE_BYTES}));");
        writeln!(src, "{indent}noc_async_write(cb{cb}.get_read_ptr(), wnoc{op_id}, {tile_bytes});");
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
    /// Runtime CB config: (tt format, tile bytes) per CB. Format and
    /// tile bytes follow the CB storage dtype; an unmappable dtype is
    /// a compilation error, never a silent default.
    pub(crate) config: Slab<CBId, (u32, u32)>,
    /// Runtime state per CB, in id order.
    states: Slab<CBId, CBState>,
}

#[allow(unused_must_use)]
impl CBEmitter {
    /// Full CB allocation from a kernel: first-touch ids, page/L1
    /// validity, hardware count fit, runtime config. The single point
    /// that assigns CB ids; anything unmappable is a compilation
    /// error. Every mapped CB registers in the state slab (free
    /// state), in id order so the slab index matches the [`CBId`].
    fn new(kernel: &Kernel) -> Result<Self, BackendError> {
        // Single walk: first-touch CB ids. Registration order is
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
        let mut config: Slab<CBId, (u32, u32)> = Slab::new();
        for (cb, op) in cb_ops {
            let Op::Storage { dtype, .. } = &kernel.ops[op].op else {
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
            let pushed = config.push((fmt, tb));
            debug_assert_eq!(pushed, cb, "tenstorrent2: CB config out of sync with allocation");
        }
        Ok(Self { map, config, states })
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
                matches!(self.states[id], CBState::Popped | CBState::Pushed),
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

    /// `cb.reserve_back(1)`: only from the free state.
    fn reserve_back(&mut self, src: &mut String, indent: &str, cb: CBId) {
        assert_eq!(self.states[cb], CBState::Popped, "tenstorrent2: reserve_back on CB{cb} in {:?}, must be free", self.states[cb]);
        writeln!(src, "{indent}cb{cb}.reserve_back(1);");
        self.states[cb] = CBState::Reserved;
    }

    /// `cb.push_back(1)`: only from `Reserved`.
    fn push_back(&mut self, src: &mut String, indent: &str, cb: CBId) {
        assert_eq!(self.states[cb], CBState::Reserved, "tenstorrent2: push_back on CB{cb} in {:?}, must be Reserved", self.states[cb]);
        writeln!(src, "{indent}cb{cb}.push_back(1);");
        self.states[cb] = CBState::Pushed;
    }

    /// `cb.wait_front(1)`: only on live data.
    fn wait_front(&mut self, src: &mut String, indent: &str, cb: CBId) {
        assert_eq!(self.states[cb], CBState::Pushed, "tenstorrent2: wait_front on CB{cb} in {:?}, must be Pushed", self.states[cb]);
        writeln!(src, "{indent}cb{cb}.wait_front(1);");
        self.states[cb] = CBState::Waiting;
    }

    /// `cb.pop_front(1)`: only after a wait.
    fn pop_front(&mut self, src: &mut String, indent: &str, cb: CBId) {
        assert_eq!(self.states[cb], CBState::Waiting, "tenstorrent2: pop_front on CB{cb} in {:?}, must be Waiting", self.states[cb]);
        writeln!(src, "{indent}cb{cb}.pop_front(1);");
        self.states[cb] = CBState::Popped;
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
    /// `mm_init` CB pairs seen, for the hoisted init block.
    mm_inits: Vec<(CBId, CBId)>,
    /// Reduce triples seen, for the hoisted `reduce_init` block.
    /// The tile emits per use; `reduce_uninit` goes out once per
    /// cone at the consuming pack (`math_unlock`).
    reduce_inits: Vec<(&'static str, &'static str, CBId, CBId, TileId<DSTBF16>)>,
    /// Transpose CB pairs seen, for the hoisted `transpose_wh_init`
    /// block (0.72 has no split `transpose_init`; the `_wh` long init
    /// owns the full programming like `mm_init` does).
    transpose_inits: Vec<(CBId, CBId)>,
    /// A reduce cone is open: the next `pack` closes it with
    /// `reduce_uninit` before the commit.
    reduce_pending: bool,
    /// Compute-kernel startup triple `[in0, in1, out]` for
    /// `compute_kernel_hw_startup`: recorded at compute entry by the
    /// generator, emitted with the hoisted inits at the anchor.
    startup: Option<[CBId; 3]>,
    /// Tile unary ops seen, for the hoisted init block.
    unary_inits: Set<UOp>,
    /// Tile binary ops seen, for the hoisted init block.
    binary_inits: Set<BOp>,
    /// Tile cast format pairs seen, for the hoisted init block.
    typecast_inits: Vec<(u32, u32)>,
    /// Copy CBs seen, for the hoisted `copy_tile_init` block.
    copy_inits: Set<CBId>,
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
            unary_inits: Set::default(),
            binary_inits: Set::default(),
            typecast_inits: Vec::new(),
            copy_inits: Set::default(),
            mm_inits: Vec::new(),
            reduce_inits: Vec::new(),
            transpose_inits: Vec::new(),
            reduce_pending: false,
            startup: None,
        }
    }

    /// Record the compute-kernel startup triple (generator setup, no
    /// strings): `[in0, in1, out]`.
    fn set_startup(&mut self, triple: [CBId; 3]) {
        assert!(self.startup.is_none(), "tenstorrent2: compute startup triple set twice");
        self.startup = Some(triple);
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
    /// content and emit nothing. Taking from `PackLock` is a loud
    /// error (matrix X): the caller must release PACK first.
    fn math_lock(&mut self, src: &mut String, indent: &str) {
        if self.state == TileState::MathLock {
            return;
        }
        assert_eq!(self.state, TileState::Unlocked, "tenstorrent2: math op with DST in {:?}, must be Unlocked", self.state);
        writeln!(src, "{indent}tile_regs_acquire();");
        self.state.math_lock();
    }

    /// `tile_regs_commit`: MATH releases the file. Unlocking an
    /// unlocked file is a loud error: every lock must pair.
    fn math_unlock(&mut self, src: &mut String, indent: &str) {
        assert_eq!(self.state, TileState::MathLock, "tenstorrent2: math_unlock with DST in {:?}, every lock must pair", self.state);
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
        assert_eq!(self.state, TileState::PackLock, "tenstorrent2: pack_unlock with DST in {:?}, every lock must pair", self.state);
        writeln!(src, "{indent}tile_regs_release();");
        self.state.pack_unlock();
    }

    /// Fused matmul: waits both input CBs, emits the single
    /// `matmul_tiles` (input tile ids alias the acc slot: the op
    /// sources data from the CBs and accumulates into the one tile),
    /// then pops both CBs. Runs under the MATH lock (lazily acquired).
    /// `mm_init` goes out hoisted (one per CB pair, with the output
    /// CB from the startup triple). Records the result op in `tile_map`.
    fn matmul(
        &mut self,
        src: &mut String,
        indent: &str,
        cb_em: &mut CBEmitter,
        op_id: OpId,
        cb_a: CBId,
        cb_b: CBId,
        acc: TileId<DSTBF16>,
    ) {
        if !self.mm_inits.contains(&(cb_a, cb_b)) {
            self.mm_inits.push((cb_a, cb_b));
        }
        self.math_lock(src, indent);
        debug_assert_eq!(self.state, TileState::MathLock, "tenstorrent2: matmul without MATH lock");
        cb_em.wait_front(src, indent, cb_a);
        cb_em.wait_front(src, indent, cb_b);
        writeln!(src, "{indent}matmul_tiles({cb_a}, {cb_b}, {acc}, {acc}, {acc});");
        cb_em.pop_front(src, indent, cb_a);
        cb_em.pop_front(src, indent, cb_b);
        self.tile_map.insert(op_id, acc);
    }

    /// Streaming copy in: waits the CB, takes the MATH lock, copies
    /// the tile into a fresh DST slot, pops the CB. Records the load
    /// op in `tile_map`. The `copy_tile_init` goes out hoisted (one
    /// per CB).
    fn copy(
        &mut self,
        src: &mut String,
        indent: &str,
        cb_em: &mut CBEmitter,
        op_id: OpId,
        cb: CBId,
        rc: u32,
    ) -> TileId<DSTBF16> {
        cb_em.wait_front(src, indent, cb);
        self.math_lock(src, indent);
        debug_assert_eq!(self.state, TileState::MathLock, "tenstorrent2: copy without MATH lock");
        let slot = self.alloc(rc);
        self.copy_inits.insert(cb);
        writeln!(src, "{indent}copy_tile({cb}, 0, {slot});");
        cb_em.pop_front(src, indent, cb);
        self.tile_map.insert(op_id, slot);
        slot
    }

    /// Pack out: closes any open reduce cone, commits MATH, reserves
    /// the CB, takes the PACK lock, packs the slot, pushes the CB,
    /// releases the file. The whole drain sequences here, in emission
    /// order: uninit, commit, reserve, wait, [reconfig,] pack, push,
    /// release. The packer reconfig goes out only for non-native pack
    /// targets: the JIT programs the mode-native triple by
    /// construction, so reprogramming it is redundant there.
    fn pack(&mut self, src: &mut String, indent: &str, cb_em: &mut CBEmitter, slot: TileId<DSTBF16>, cb: CBId) {
        if self.reduce_pending {
            writeln!(src, "{indent}reduce_uninit();");
            self.reduce_pending = false;
        }
        self.math_unlock(src, indent);
        cb_em.reserve_back(src, indent, cb);
        self.pack_lock(src, indent);
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
        cb_em.push_back(src, indent, cb);
        self.pack_unlock(src, indent);
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
    /// the acc slot, pops both CBs. The init goes out hoisted, the
    /// uninit at the consuming pack. Runs under the MATH lock
    /// (lazily acquired). Records the result op in `tile_map`.
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
        kind: TileReduceKind,
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
        let dim_name = match kind {
            TileReduceKind::Row => "ReduceDim::REDUCE_ROW",
            TileReduceKind::Col => "ReduceDim::REDUCE_COL",
            TileReduceKind::Scalar => "ReduceDim::REDUCE_SCALAR",
        };
        let params = (op_name, dim_name, cb_in, cb_sc, acc);
        if !self.reduce_inits.contains(&params) {
            self.reduce_inits.push(params);
        }
        self.math_lock(src, indent);
        debug_assert_eq!(self.state, TileState::MathLock, "tenstorrent2: reduce without MATH lock");
        cb_em.wait_front(src, indent, cb_in);
        cb_em.wait_front(src, indent, cb_sc);
        writeln!(src, "{indent}reduce_tile<{op_name}, {dim_name}>({cb_in}, {cb_sc}, 0, 0, {acc});");
        self.reduce_pending = true;
        cb_em.pop_front(src, indent, cb_in);
        cb_em.pop_front(src, indent, cb_sc);
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
        self.binary_inits.insert(bop);
        let odst = self.alloc(rc);
        writeln!(src, "{indent}{name}({x}, {y}, {odst});");
        self.tile_map.insert(op_id, odst);
        odst
    }

    /// Tiled unary ALU: SFPU ops are in-place (`op(x)` transforms the
    /// slot), so the result aliases the operand slot. Takes the MATH
    /// lock and records the hoisted init.
    #[allow(dead_code)]
    fn unary(
        &mut self,
        src: &mut String,
        indent: &str,
        op_id: OpId,
        x: TileId<DSTBF16>,
        uop: UOp,
    ) -> TileId<DSTBF16> {
        let name = match uop {
            UOp::Neg => "negative_tile",
            UOp::BitNot => "bitwise_not_tile",
            UOp::Exp => "exp_tile",
            UOp::Exp2 => "exp2_tile",
            UOp::Log2 => "log_tile",
            UOp::Sin => "sin_tile",
            UOp::Cos => "cos_tile",
            UOp::Reciprocal => "recip_tile",
            UOp::Sqrt => "sqrt_tile",
            UOp::Floor => "floor_tile",
            UOp::Trunc => "trunc_tile",
            UOp::Abs => "abs_tile",
            UOp::Not => "logical_not_tile",
            UOp::Ln => unreachable!("ln is lowered to log2 before codegen"),
        };
        self.math_lock(src, indent);
        debug_assert_eq!(self.state, TileState::MathLock, "tenstorrent2: unary without MATH lock");
        self.unary_inits.insert(uop);
        writeln!(src, "{indent}{name}({x});");
        self.tile_map.insert(op_id, x);
        x
    }

    /// Tiled cast: in-place like unary (`typecast_tile<IN,OUT>(x)`),
    /// formats are tt::DataFormat values. Records the parameterized
    /// hoisted init (deduped: re-init of the same pair is harmless but
    /// noisy).
    #[allow(dead_code)]
    fn cast(
        &mut self,
        src: &mut String,
        indent: &str,
        op_id: OpId,
        x: TileId<DSTBF16>,
        in_fmt: u32,
        out_fmt: u32,
    ) -> TileId<DSTBF16> {
        self.math_lock(src, indent);
        debug_assert_eq!(self.state, TileState::MathLock, "tenstorrent2: cast without MATH lock");
        if !self.typecast_inits.contains(&(in_fmt, out_fmt)) {
            self.typecast_inits.push((in_fmt, out_fmt));
        }
        writeln!(src, "{indent}typecast_tile<{in_fmt}, {out_fmt}>({x});");
        self.tile_map.insert(op_id, x);
        x
    }

    /// Streaming transpose: waits the CB, takes the MATH lock,
    /// transposes the 32x32 tile into a fresh DST slot, pops the CB.
    /// Records the load op in `tile_map`. The `transpose_wh_init`
    /// goes out hoisted (one per input/output CB pair).
    fn transpose(
        &mut self,
        src: &mut String,
        indent: &str,
        cb_em: &mut CBEmitter,
        op_id: OpId,
        cb: CBId,
        out: CBId,
        rc: u32,
    ) -> TileId<DSTBF16> {
        cb_em.wait_front(src, indent, cb);
        self.math_lock(src, indent);
        debug_assert_eq!(self.state, TileState::MathLock, "tenstorrent2: transpose without MATH lock");
        let slot = self.alloc(rc);
        if !self.transpose_inits.contains(&(cb, out)) {
            self.transpose_inits.push((cb, out));
        }
        writeln!(src, "{indent}transpose_wh_tile({cb}, 0, {slot});");
        cb_em.pop_front(src, indent, cb);
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

    /// Common pre-loop init emission for compute tile ops (v1-proven
    /// hoist, official eltwise shape): `compute_kernel_hw_startup`
    /// once, then one `*_init` per collected kind, inserted at `pos`
    /// — the anchor the caller records ahead of all loops. Startup
    /// comes first (the header requires it exactly once at the
    /// beginning, before any op init; no separate SFPU init exists);
    /// FP32 mode enables 32-bit DST right after. Matmul kernels skip
    /// startup: 0.72 `mm_init` owns the full UNPACK/MATH/PACK
    /// programming (the installed example calls nothing else).
    /// Per-iteration init reprograms live packer state. The walk inserts
    /// into the sets as it emits; this method only formats into the
    /// section source. Always at function-scope indent: the anchor sits
    /// at base indent by construction.
    fn prepend_compute_inits(&self, src: &mut String, pos: usize) {
        let indent = "  ";
        let mut inits = String::new();
        if self.mm_inits.is_empty() {
            if let Some([in0, in1, out]) = self.startup {
                let _ = std::fmt::Write::write_fmt(
                    &mut inits,
                    format_args!("{indent}compute_kernel_hw_startup({in0}, {in1}, {out});\n"),
                );
                if !DSTBF16 {
                    let _ = std::fmt::Write::write_fmt(&mut inits, format_args!("{indent}enable_fp32_dest_acc();\n"));
                }
            }
        } else {
            // Matmul kernels: mm_init instead of startup (0.72 API).
            if !DSTBF16 {
                let _ = std::fmt::Write::write_fmt(&mut inits, format_args!("{indent}enable_fp32_dest_acc();\n"));
            }
            let out = self.startup.map(|[_, _, o]| o).expect("tenstorrent2: matmul kernel without startup triple");
            for &(cb_a, cb_b) in &self.mm_inits {
                let _ = std::fmt::Write::write_fmt(&mut inits, format_args!("{indent}mm_init({cb_a}, {cb_b}, {out});\n"));
            }
        }
        for &cb in &self.copy_inits {
            let _ = std::fmt::Write::write_fmt(&mut inits, format_args!("{indent}copy_tile_init({cb});\n"));
        }
        for &(cb_in, cb_out) in &self.transpose_inits {
            let _ = std::fmt::Write::write_fmt(&mut inits, format_args!("{indent}transpose_wh_init({cb_in}, {cb_out});\n"));
        }
        for &uop in &self.unary_inits {
            let init = match uop {
                UOp::Neg => "negative_tile_init();",
                UOp::BitNot => "bitwise_not_tile_init();",
                UOp::Exp => "exp_tile_init();",
                UOp::Exp2 => "exp2_tile_init();",
                UOp::Log2 => "log_tile_init();",
                UOp::Reciprocal => "recip_tile_init();",
                UOp::Sqrt => "sqrt_tile_init();",
                UOp::Sin => "sin_tile_init();",
                UOp::Cos => "cos_tile_init();",
                UOp::Floor | UOp::Trunc => "rounding_op_tile_init();",
                UOp::Abs => "abs_tile_init();",
                UOp::Not => "logical_not_tile_init();",
                UOp::Ln => unreachable!("ln is lowered to log2 before codegen"),
            };
            let _ = std::fmt::Write::write_fmt(&mut inits, format_args!("{indent}{init}\n"));
        }
        for &bop in &self.binary_inits {
            let init = match bop {
                BOp::Add => Some("add_binary_tile_init();"),
                BOp::Sub => Some("sub_binary_tile_init();"),
                BOp::Mul => Some("mul_binary_tile_init();"),
                BOp::Div => Some("div_binary_tile_init();"),
                BOp::Max => Some("binary_max_tile_init();"),
                BOp::BitShiftLeft => Some("binary_shift_tile_init();"),
                BOp::BitShiftRight => Some("binary_shift_tile_init();"),
                BOp::BitAnd => Some("bitwise_and_tile_init();"),
                _ => None,
            };
            if let Some(init) = init {
                let _ = std::fmt::Write::write_fmt(&mut inits, format_args!("{indent}{init}\n"));
            }
        }
        for &(in_fmt, out_fmt) in &self.typecast_inits {
            let _ = std::fmt::Write::write_fmt(&mut inits, format_args!("{indent}typecast_tile_init<{in_fmt}, {out_fmt}>();\n"));
        }
        for &(op_name, dim_name, cb_in, cb_sc, acc) in &self.reduce_inits {
            let _ = std::fmt::Write::write_fmt(&mut inits, format_args!("{indent}reduce_init<{op_name}, {dim_name}>({cb_in}, {cb_sc}, {acc});\n"));
        }
        src.insert_str(pos, &inits);
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
                Op::Store { ref dst, src: ref store_src, layout: st_layout, .. } => {
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
                            self.cb.reserve_back(&mut src, &indent, cb);
                            self.noc.async_read_tile(&mut src, &indent, op_id, ld_src, &idx, elem_size, tile_bytes, cb);
                            self.cb.push_back(&mut src, &indent, cb);
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
                | Op::Asm { .. }
                | Op::Move { .. }
                | Op::Reduce { .. } => todo!("tenstorrent2 reader op"),
            }
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
        let mut em = VarEmitter::new(kernel);
        let mut src = String::new();
        let mut indent = String::from("  ");
        self.noc.begin_section(params);
        let mut scope_level = 0u8;
        writeln!(src, "#include <cstdint>");
        writeln!(src, "#include \"api/compute/common.h\"");
        writeln!(src, "#include \"api/compute/compute_kernel_api.h\"");
        writeln!(src, "#include \"api/compute/eltwise_binary_sfpu.h\"");
        writeln!(src, "#include \"api/compute/tile_move_copy.h\"");
        writeln!(src, "#include \"api/compute/eltwise_unary/eltwise_unary.h\"");
        writeln!(src, "#include \"api/compute/eltwise_unary/trigonometry.h\"");
        writeln!(src, "#include \"api/compute/eltwise_unary/exp.h\"");
        writeln!(src, "#include \"api/compute/eltwise_unary/recip.h\"");
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
        writeln!(src, "#include \"api/compute/reduce.h\"");
        writeln!(src, "#include \"api/compute/transpose_wh.h\"");
        writeln!(src, "#include \"api/compute/reconfig_data_format.h\"");
        writeln!(src, "#include \"api/dataflow/circular_buffer.h\"");
        writeln!(src, "#include \"api/debug/device_print.h\"");
        writeln!(src, "void kernel_main() {{");
        // Every shared CB is declared (matches reader/writer).
        self.cb.declare_all(&mut src, &indent);
        // Entry states: CBs this section loads start pushed (produced
        // upstream); stored-only CBs start free. First-touch load/store
        // order also resolves the startup triple.
        let mut loaded_here: Set<CBId> = Set::default();
        let mut loaded_order: Vec<CBId> = Vec::new();
        let mut stored_first: Option<CBId> = None;
        for &op_id in &compute_data.ops {
            if let Op::Load { src, layout: MemLayout::Tile { .. }, .. } = kernel.ops[op_id].op {
                if let Some(&cb) = self.cb.map.get(&src) {
                    if loaded_here.insert(cb) {
                        loaded_order.push(cb);
                    }
                }
            }
            if stored_first.is_none() {
                if let Op::Store { dst, layout: MemLayout::Tile { .. }, .. } = kernel.ops[op_id].op {
                    if let Some(&cb) = self.cb.map.get(&dst) {
                        stored_first = Some(cb);
                    }
                }
            }
        }
        self.cb.reset_all(CBState::Popped);
        for cb in loaded_here {
            self.cb.set(cb, CBState::Pushed);
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
                        let in_fmt = tt_fmt(compute_data.dtypes[&x].0)?;
                        let out_fmt = tt_fmt(compute_data.dtypes[&op_id].0)?;
                        self.tl.cast(&mut src, &indent, op_id, tile, in_fmt, out_fmt);
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
                        // Tiled binary: three-operand form, inputs stay
                        // live, result in a fresh slot.
                        let ta = self.tl.tile_map.get(&x).copied().ok_or_else(|| BackendError {
                            status: ErrorStatus::KernelCompilation,
                            context: format!("tenstorrent2: tiled binary reads a value with no DST slot, op {op_id}").into(),
                        })?;
                        let tb = self.tl.tile_map.get(&y).copied().ok_or_else(|| BackendError {
                            status: ErrorStatus::KernelCompilation,
                            context: format!("tenstorrent2: tiled binary reads a value with no DST slot, op {op_id}").into(),
                        })?;
                        let rc = compute_data.rcs[&op_id];
                        self.tl.binary(&mut src, &indent, op_id, ta, tb, bop, rc);
                    } else {
                        em.emit_op(&mut src, &indent, op_id, compute_data, &self.noc, scope_level)?;
                    }
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
                        // zeroes the file: the seed). Later takes in the
                        // cone keep. Compute-only: other sections must
                        // not emit MATH traffic.
                        self.tl.math_lock(&mut src, &indent);
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
                                context: format!("tenstorrent2: compute acc store reads a tile with no DST slot, op {op_id}").into(),
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
                        self.tl.pack(&mut src, &indent, &mut self.cb, tile, out_cb);
                    }
                }
                Op::Load { src: ref load_src, layout, .. } => {
                    if !matches!(layout, MemLayout::Tile { .. }) {
                        todo!("tenstorrent2 compute only supports tile loads");
                    }
                    // Register acc loads thread the SSA value: no CB, no
                    // traffic. Anything else must be a mapped CB.
                    if matches!(kernel.ops[*load_src].op, Op::Storage { scope: MemScope::Register, .. }) {
                        // Emit nothing.
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
                                if matches!(kernel.ops[consumer].op, Op::ReduceTile { .. } | Op::MatmulTile { .. } | Op::TransposeTile { .. }) {
                                    fused_only = true;
                                } else {
                                    fused_only = false;
                                    break;
                                }
                            }
                        }
                        if !fused_only {
                            let rc = compute_data.rcs[&op_id];
                            self.tl.copy(&mut src, &indent, &mut self.cb, op_id, cb, rc);
                        }
                    }
                }
                Op::Range { .. } => {
                    em.emit_op(&mut src, &indent, op_id, compute_data, &self.noc, scope_level)?;
                }
                Op::Loop { .. } => {
                    em.loop_begin(&mut src, &mut indent, kernel, compute_data, op_id, &mut scope_level)?;
                }
                Op::EndLoop => {
                    loop_end(&mut src, &mut indent, &mut scope_level);
                }
                Op::If { condition } => {
                    // Plain brace scope: no lock/CB interaction (those
                    // pair with loops, never ifs).
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
                    self.tl.matmul(&mut src, &indent, &mut self.cb, op_id, cb_a, cb_b, tile);
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
                        let Op::Storage { dtype: DType::F16, .. } = kernel.ops[s].op else {
                            return Err(BackendError {
                                status: ErrorStatus::KernelCompilation,
                                context: format!("tenstorrent2: reduce tiles are F16, op {op_id} is not").into(),
                            });
                        };
                    }
                    let rc = compute_data.rcs[&op_id];
                    let tile = self.tl.acc_tile(la, rc);
                    self.tl.reduce(&mut src, &indent, &mut self.cb, op_id, cb_in, cb_sc, tile, rop, kind, rc)?;
                }
                Op::Asm { .. } => todo!(),
                Op::Barrier => unreachable!("should've been filtered by kernel sections decomposition"),
                Op::Move { .. } | Op::Reduce { .. } => unreachable!("should've been lowered by linearize"),
                Op::Wmma { .. } => unreachable!("tt does not support wmma, use Op::MatmulTile instead"),
            }
        }
        // Hoisted tile-op inits land at the anchor, ahead of all loops.
        self.tl.prepend_compute_inits(&mut src, init_anchor);

        writeln!(src, "}}");
        self.cb.assert_settled("compute");
        assert_eq!(self.tl.state, TileState::Unlocked, "tenstorrent2: compute ends with DST in {:?}, every lock must pair", self.tl.state);
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
            self.cb.set(cb, CBState::Pushed);
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
                    let Op::Load { src: cb_src, index: _ld_idx, layout: ld_layout } = kernel.ops[*store_src].op else {
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
                            self.cb.wait_front(&mut src, &indent, cb);
                            self.noc.async_write_tile(&mut src, &indent, op_id, *dst, &idx, elem_size, tile_bytes, cb);
                            self.cb.pop_front(&mut src, &indent, cb);
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
                | Op::Asm { .. }
                | Op::Move { .. }
                | Op::Reduce { .. } => todo!("tenstorrent2 writer op"),
            }
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
            | Op::Asm { .. }
            | Op::Move { .. }
            | Op::Reduce { .. } => todo!("tenstorrent2 scalar op"),
        }
        Ok(())
    }
}

/// TT `DataFormat` constant for a zyx dtype on the tile path.
/// tt-metal 0.72 has no plain-Float16 SFPU kernel, so zyx F16
/// rides Float16_b (see typecast.h supported list).
fn tt_fmt(dt: DType) -> Result<u32, BackendError> {
    match dt {
        DType::F32 => Ok(0),
        DType::F16 | DType::BF16 => Ok(5),
        DType::U16 => Ok(9),
        DType::U32 => Ok(24),
        DType::I32 => Ok(8),
        dt => Err(BackendError {
            status: ErrorStatus::KernelCompilation,
            context: format!("tenstorrent2: dtype {dt:?} has no tt tile format").into(),
        }),
    }
}
