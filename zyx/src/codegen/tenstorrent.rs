use crate::{
    DType, Map,
    error::{BackendError, ErrorStatus},
    kernel::{BOp, IDX_T, Kernel, MMADType, MemLayout, MemScope, Op, OpId, ParamKind, RangeKind},
};

use nanoserde::{DeBin, SerBin};
use std::fmt::Write;

use crate::slab::SlabId;

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

/// DST slot ID for Tenstorrent codegen v2.
///
/// This is a unique identifier for a compute-core register tile slot
/// within one section kernel.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, SerBin, DeBin)]
pub struct DstId(pub(crate) u32);

impl DstId {
    /// NULL
    pub const NULL: Self = Self(u32::MAX);

    /// Check if this DstId is null.
    pub const fn is_null(self) -> bool {
        self.0 == u32::MAX
    }
}

impl std::fmt::Display for DstId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::Display::fmt(&self.0, f)
    }
}

impl From<usize> for DstId {
    fn from(value: usize) -> Self {
        DstId(value as u32)
    }
}

impl From<DstId> for usize {
    fn from(value: DstId) -> usize {
        value.0 as usize
    }
}

impl SlabId for DstId {
    const ZERO: Self = Self(0);
    const NULL: Self = Self(u32::MAX);

    fn inc(&mut self) {
        self.0 += 1;
    }
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
        params: Vec<OpId>,
    },
    Compute {
        src: String,
        params: Vec<OpId>,
        dst_slots: Map<OpId, Vec<DstId>>,
        next_slot: DstId,
        /// Scratch CBs (stored in compute, never loaded by the writer):
        /// no DRAM traffic is emitted for them in any section.
        scratch_cbs: Map<CBId, ()>,
    },
    Writer {
        src: String,
        params: Vec<OpId>,
    },
    None,
}

/// Output of [`Kernel::generate_tenstorrent`]: the three section kernels
/// plus every kernel-derived table the backend needs.
///
/// CB ids and section params are calculated here and only here; the
/// backend consumes these tables without re-deriving them.
pub(crate) struct TtCompile {
    /// Emitted reader source with its section params.
    pub(crate) reader: TTKernel,
    /// Compute kernel (codegen not yet implemented: empty source, no params).
    pub(crate) compute: TTKernel,
    /// Emitted writer source with its section params.
    pub(crate) writer: TTKernel,
    /// Runtime CB config in id order: (id, tt format, tile bytes).
    /// Format and tile bytes follow the CB storage dtype; an
    /// unmappable dtype is a compilation error, never a silent default.
    pub(crate) cb_config: Vec<(u32, u32, u32)>,
    /// Per-section param ordinals (global head order): each section's
    /// runtime args.
    pub(crate) section_params: [Vec<u32>; 3],
    /// Global head-order ordinal of every param (all kinds).
    pub(crate) param_ordinal_of: Map<OpId, u32>,
    /// Global params in head order (kernel inputs).
    pub(crate) input_dtypes: Vec<DType>,
    /// GlobalMut params in head order (kernel outputs).
    pub(crate) output_dtypes: Vec<DType>,
}

impl Kernel {
    /// Generate TT Metalium reader, compute, and writer kernels from zyx IR.
    ///
    /// Single calculation point for CB ids and section params: builds one
    /// [`Compiler`] (shared cb_map plus per-section op lists), runs the
    /// section/CB/balance checks, then generates each section from its
    /// closed op list. The hardware CB limit comes from the kernel's bound
    /// device.
    ///
    /// # Returns
    /// The section kernels plus the kernel-derived tables the backend
    /// needs, or a compilation error if any check fails.
    #[allow(unused_must_use)]
    pub(crate) fn generate_tenstorrent(&self) -> Result<TtCompile, BackendError> {
        let mut compiler = Compiler::new(self);
        compiler.check_sections(self)?;
        compiler.check_cb_count()?;
        compiler.check_cb_validity(self)?;
        compiler.check_balance(self)?;
        compiler.generate_reader(self)?;
        compiler.generate_writer(self)?;

        // Per-section param ordinals (global head order): each section's
        // params are in IR order, so the ordinals ascend already.
        let section_params = [&compiler.reader, &compiler.compute, &compiler.writer].map(|kernel| {
            let params = match kernel {
                TTKernel::Reader { params, .. } => params,
                TTKernel::Compute { params, .. } => params,
                TTKernel::Writer { params, .. } => params,
                TTKernel::None => unreachable!("tenstorrent2 section kernel missing after generation"),
            };
            let ordinals: Vec<u32> = params.iter().map(|p| compiler.param_ordinal_of[p]).collect();
            debug_assert!(
                ordinals.windows(2).all(|w| w[0] < w[1]),
                "tenstorrent2 section params not in head order"
            );
            ordinals
        });
        // Runtime CB config in id order. Format and tile bytes follow the
        // CB storage dtype; anything else is a compilation error.
        let mut cb_ops: Vec<(CBId, OpId)> = compiler.cb_map.iter().map(|(&op, &cb)| (cb, op)).collect();
        cb_ops.sort_by_key(|&(cb, _)| cb);
        let mut cb_config: Vec<(u32, u32, u32)> = Vec::with_capacity(cb_ops.len());
        for (cb, op) in cb_ops {
            let Op::Storage { dtype, .. } = &self.ops[op].op else {
                unreachable!("tenstorrent2: cb_map entry {op} passed check_cb_validity but is not a storage op")
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
            cb_config.push((cb.0, fmt, tb));
        }
        Ok(TtCompile {
            reader: compiler.reader,
            compute: compiler.compute,
            writer: compiler.writer,
            cb_config,
            section_params,
            param_ordinal_of: compiler.param_ordinal_of,
            input_dtypes: compiler.input_dtypes,
            output_dtypes: compiler.output_dtypes,
        })
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
        let mut structural: Map<OpId, ()> = Map::default();
        let mut scan = self.head;
        for _ in 0..10_000 {
            if scan.is_null() {
                break;
            }
            // Barriers always track the section; every other arm carries
            // a guard so the op matches only inside the target section.
            match self.ops[scan].op {
                Op::Barrier => {
                    if section == tt_section {
                        structural.insert(scan, ());
                    }
                    section.advance();
                }
                Op::Store { .. } if section == tt_section => {
                    stores.push(scan);
                }
                Op::Loop { len } if section == tt_section => {
                    structural.insert(scan, ());
                    starters.push(len);
                }
                Op::Range { kind, .. } if section == tt_section => {
                    structural.insert(scan, ());
                    match kind {
                        RangeKind::Group(len) | RangeKind::Warp(len) => starters.push(len),
                        RangeKind::Local(_) => {}
                    }
                }
                Op::EndLoop | Op::If { .. } | Op::EndIf if section == tt_section => {
                    structural.insert(scan, ());
                }
                _ => {}
            }
            scan = self.next_op(scan);
        }
        if !scan.is_null() {
            panic!("get_needed_ops did not finish in 10000 steps");
        }
        // Phase 2: transitive data-dependency closure over the stores.
        let mut needed: Map<OpId, ()> = Map::default();
        let mut stack: Vec<OpId> = starters;
        for &store in &stores {
            needed.insert(store, ());
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
            if id.is_null() || needed.contains_key(&id) {
                continue;
            }
            needed.insert(id, ());
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
                Op::ReduceTile { x, .. } => {
                    stack.push(x);
                }
                Op::MatmulTile { x, y } => {
                    stack.push(x);
                    stack.push(y);
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
            if needed.contains_key(&op_id) || structural.contains_key(&op_id) {
                ops.push(op_id);
                match self.ops[op_id].op {
                    Op::Move { .. } | Op::Reduce { .. } | Op::ReduceTile { .. } => {
                        unreachable!()
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
                    Op::MatmulTile { x, y } => {
                        dtypes.insert(op_id, dtypes[&x]);
                        *rcs.entry(x).or_insert(0) += 1;
                        *rcs.entry(y).or_insert(0) += 1;
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

/// Tenstorrent v2 compiler: per-section code generation over closed op
/// lists (see [`Kernel::get_needed_ops`]) against one shared [`CBId`] map.
///
/// All state lives here (ptx.rs pattern); shared op emission is a method
/// on this struct so reader/compute/writer cannot diverge.
struct Compiler {
    /// Hardware circular buffer limit (DeviceInfo::num_circular_buffers).
    num_circular_buffers: u32,
    /// One CBId per Circular storage, shared by all three sections.
    cb_map: Map<OpId, CBId>,
    /// Next free CBId after the map build.
    next_cb: CBId,
    /// Global head-order ordinal of every param (all kinds).
    param_ordinal_of: Map<OpId, u32>,
    /// Global params in head order (kernel inputs).
    input_dtypes: Vec<DType>,
    /// GlobalMut params in head order (kernel outputs).
    output_dtypes: Vec<DType>,
    /// Generated section kernels (filled by generation).
    reader: TTKernel,
    compute: TTKernel,
    writer: TTKernel,
    /// Param op → runtime arg position within the current section.
    arg_pos: Map<OpId, u32>,
    /// Current C++ indent level.
    indent: String,
}

impl Compiler {
    /// Build compiler state from a kernel: the shared cb_map, the param
    /// ordinals and input/output dtypes, plus one closed op list per
    /// section.
    fn new(kernel: &Kernel) -> Self {
        let num_circular_buffers = kernel.device_info().num_circular_buffers;
        // - CB ids: first touch in any section registers, in kernel order.
        // - Param ordinals (all kinds, head order) and Global/GlobalMut
        //   dtypes: same single walk.
        let mut cb_map: Map<OpId, CBId> = Map::default();
        let mut next_cb = CBId::ZERO;
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
            if let Op::Load { src, .. } = &kernel.ops[scan].op {
                if let Op::Storage { scope: MemScope::Circular, .. } = kernel.ops[*src].op {
                    if !cb_map.contains_key(src) {
                        cb_map.insert(*src, next_cb);
                        next_cb.inc();
                    }
                }
            }
            if let Op::Store { dst, .. } = &kernel.ops[scan].op {
                if let Op::Storage { scope: MemScope::Circular, .. } = kernel.ops[*dst].op {
                    if !cb_map.contains_key(dst) {
                        cb_map.insert(*dst, next_cb);
                        next_cb.inc();
                    }
                }
            }
            scan = kernel.next_op(scan);
        }
        if !scan.is_null() {
            panic!("tenstorrent2 cb_map scan did not finish in 10000 steps");
        }
        Self {
            num_circular_buffers,
            cb_map,
            next_cb,
            param_ordinal_of,
            input_dtypes,
            output_dtypes,
            reader: TTKernel::None,
            compute: TTKernel::Compute {
                src: String::new(),
                params: Vec::new(),
                dst_slots: Map::default(),
                next_slot: DstId::ZERO,
                scratch_cbs: Map::default(),
            },
            writer: TTKernel::None,
            arg_pos: Map::default(),
            indent: String::from("  "),
        }
    }

    /// Exactly 2 barriers delimiting reader/compute/writer, else a
    /// compilation error.
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

    /// Shared cb_map fits the hardware CB count, else a compilation error.
    fn check_cb_count(&self) -> Result<(), BackendError> {
        if self.cb_map.len() > self.num_circular_buffers as usize {
            return Err(BackendError {
                status: ErrorStatus::TooManyCircularBuffers,
                context: format!(
                    "tenstorrent2: kernel needs {} circular buffers, device holds {}",
                    self.cb_map.len(),
                    self.num_circular_buffers
                )
                .into(),
            });
        }
        Ok(())
    }

    /// Every mapped CB holds whole 2048B pages within the single-core
    /// L1 budget, else a compilation error.
    fn check_cb_validity(&self, kernel: &Kernel) -> Result<(), BackendError> {
        for (&storage, &cb) in self.cb_map.iter() {
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
        Ok(())
    }

    /// Simplified push/pop balance: per CB, trip-weighted tile pushes
    /// equal trip-weighted tile pops, else a compilation error. One rule
    /// covers streaming (reader pushes == compute/writer pops) and
    /// compute-local scratch (seed + per-iteration push/pop pairs drain
    /// to zero) with no special cases.
    fn check_balance(&self, kernel: &Kernel) -> Result<(), BackendError> {
        let mut pushes: Map<CBId, i64> = Map::default();
        let mut pops: Map<CBId, i64> = Map::default();
        let mut trips: Vec<i64> = Vec::new();
        let mut scan = kernel.head;
        for _ in 0..10_000 {
            if scan.is_null() {
                break;
            }
            match &kernel.ops[scan].op {
                Op::Loop { len } => {
                    let Op::Const(c) = kernel.ops[*len].op else {
                        return Err(BackendError {
                            status: ErrorStatus::KernelCompilation,
                            context: format!("tenstorrent2: loop trip count op {len} is not const").into(),
                        });
                    };
                    trips.push(c.as_dim().expect("tenstorrent2 loop trip count must be a concrete dim"));
                }
                Op::EndLoop => {
                    trips.pop().expect("tenstorrent2 EndLoop without Loop");
                }
                Op::Store { dst, layout: MemLayout::Tile { .. }, .. } => {
                    if let Some(&cb) = self.cb_map.get(dst) {
                        let trip: i64 = trips.iter().product();
                        *pushes.entry(cb).or_insert(0) += trip;
                    }
                }
                Op::Load { src, layout: MemLayout::Tile { .. }, .. } => {
                    if let Some(&cb) = self.cb_map.get(src) {
                        let trip: i64 = trips.iter().product();
                        *pops.entry(cb).or_insert(0) += trip;
                    }
                }
                _ => {}
            }
            scan = kernel.next_op(scan);
        }
        if !scan.is_null() {
            panic!("tenstorrent2 balance scan did not finish in 10000 steps");
        }
        for (&storage, &cb) in self.cb_map.iter() {
            let pushed = pushes.get(&cb).copied().unwrap_or(0);
            let popped = pops.get(&cb).copied().unwrap_or(0);
            if pushed != popped {
                return Err(BackendError {
                    status: ErrorStatus::CircularBufferImbalance,
                    context: format!(
                        "tenstorrent2: CB{cb} imbalance: {pushed} pushed but {popped} popped (storage op {storage})"
                    )
                    .into(),
                });
            }
        }
        Ok(())
    }

    /// Generate the reader (dataflow movement) kernel from the reader op list.
    #[allow(unused_must_use)]
    fn generate_reader(&mut self, kernel: &Kernel) -> Result<(), BackendError> {
        let SectionData { ops, .. } = kernel.get_needed_ops(TtSection::Reader);
        // Scratch CBs live on the compute kernel but gate reader traffic:
        // stored in compute, never loaded by the writer.
        let compute_data = kernel.get_needed_ops(TtSection::Compute);
        let writer_data = kernel.get_needed_ops(TtSection::Writer);
        let mut stored_in_compute: Map<CBId, ()> = Map::default();
        for &op_id in &compute_data.ops {
            if let Op::Store { dst, .. } = kernel.ops[op_id].op {
                if let Some(&cb) = self.cb_map.get(&dst) {
                    stored_in_compute.insert(cb, ());
                }
            }
        }
        let mut read_by_writer: Map<CBId, ()> = Map::default();
        for &op_id in &writer_data.ops {
            if let Op::Load { src, .. } = kernel.ops[op_id].op {
                if let Some(&cb) = self.cb_map.get(&src) {
                    read_by_writer.insert(cb, ());
                }
            }
        }
        let mut scratch_cbs: Map<CBId, ()> = Map::default();
        for (&cb, _) in stored_in_compute.iter() {
            if !read_by_writer.contains_key(&cb) {
                scratch_cbs.insert(cb, ());
            }
        }
        let TTKernel::Compute { scratch_cbs: target, .. } = &mut self.compute else {
            unreachable!("tenstorrent2 compute state missing");
        };
        *target = scratch_cbs;
        // Section params in list order become this section's runtime args.
        let mut params: Vec<OpId> = Vec::new();
        for &op_id in &ops {
            if matches!(kernel.ops[op_id].op, Op::Param { .. }) {
                params.push(op_id);
            }
        }
        self.arg_pos = Map::default();
        for (i, &p) in params.iter().enumerate() {
            self.arg_pos.insert(p, i as u32);
        }
        self.indent = String::from("  ");
        let mut prev_accessor: Option<String> = None;
        let mut src = String::new();
        writeln!(src, "#include <cstdint>");
        writeln!(src, "#include \"api/dataflow/dataflow_api.h\"");
        writeln!(src, "#include \"api/dataflow/noc.h\"");
        writeln!(src, "#include \"api/dataflow/circular_buffer.h\"");
        writeln!(src, "#include \"api/tensor/noc_traits.h\"");
        writeln!(src, "#include \"api/debug/device_print.h\"");
        writeln!(src, "void kernel_main() {{");
        // Every shared CB is declared (matches v1 output); the section
        // only pushes the ones it fills.
        let mut cbs: Vec<CBId> = self.cb_map.iter().map(|(_, &cb)| cb).collect();
        cbs.sort();
        for cb in cbs {
            writeln!(src, "{}CircularBuffer cb{cb}(tt::CBIndex::c_{cb});", self.indent);
        }
        let n = ops.len();
        for i in 0..n {
            let op_id = ops[i];
            match &kernel.ops[op_id].op {
                Op::Param { dtype, kind, .. } => match kind {
                    ParamKind::Global => {
                        let arg = self.arg_pos.get(&op_id).copied().expect("tenstorrent2 reader param missing from section args");
                        writeln!(src, "{}uint32_t src{} = get_arg_val<uint32_t>({});", self.indent, op_id, arg);
                        let cta = match &prev_accessor {
                            None => String::from("0"),
                            Some(prev) => format!("{prev}.next_compile_time_args_offset()"),
                        };
                        writeln!(src, "{}auto args{} = TensorAccessorArgs<{}>({});", self.indent, op_id, cta, arg);
                        writeln!(
                            src,
                            "{}auto p{} = TensorAccessor(args{}, src{}, {});",
                            self.indent, op_id, op_id, op_id, TT_DRAM_PAGE_BYTES
                        );
                        prev_accessor = Some(format!("args{op_id}"));
                    }
                    ParamKind::Variable => {
                        let arg = self.arg_pos.get(&op_id).copied().expect("tenstorrent2 reader param missing from section args");
                        writeln!(
                            src,
                            "{}{} r{} = ({})get_arg_val<uint32_t>({});",
                            self.indent,
                            dtype.c_type(),
                            op_id,
                            dtype.c_type(),
                            arg
                        );
                    }
                    ParamKind::GlobalMut => {
                        let arg = self.arg_pos.get(&op_id).copied().expect("tenstorrent2 reader param missing from section args");
                        writeln!(src, "{}uint32_t dst{} = get_arg_val<uint32_t>({});", self.indent, op_id, arg);
                        let cta = match &prev_accessor {
                            None => String::from("0"),
                            Some(prev) => format!("{prev}.next_compile_time_args_offset()"),
                        };
                        writeln!(src, "{}auto args{} = TensorAccessorArgs<{}>({});", self.indent, op_id, cta, arg);
                        writeln!(
                            src,
                            "{}auto p{} = TensorAccessor(args{}, dst{}, {});",
                            self.indent, op_id, op_id, op_id, TT_DRAM_PAGE_BYTES
                        );
                        prev_accessor = Some(format!("args{op_id}"));
                    }
                },
                Op::Storage { scope: MemScope::Circular, .. } => {
                    // Declared up front for every shared CB (see driver).
                }
                Op::Storage { scope: MemScope::Local, .. } => {
                    unreachable!(
                        "tenstorrent does not have local threads; local indices should have been converted to loops by the opt_tenstorrent_tile optimization pass"
                    )
                }
                Op::Storage { .. } => {
                    todo!("tenstorrent2 reader storage scope")
                }
                Op::Load { .. } => {
                    // Handled at the consuming store (global to local with
                    // no ops in between, as below).
                }
                Op::Store { dst, src: store_src, layout: st_layout, .. } => {
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
                        unreachable!("tenstorrent2 reader loads only from Global params");
                    };
                    let Op::Storage { dtype, scope: MemScope::Circular, .. } = kernel.ops[*dst].op else {
                        unreachable!("tenstorrent2 reader stores only target Circular CBs");
                    };
                    let Some(&cb) = self.cb_map.get(dst) else {
                        unreachable!("tenstorrent2 reader store targets unmapped CB");
                    };
                    let is_scratch =
                        matches!(&self.compute, TTKernel::Compute { scratch_cbs, .. } if scratch_cbs.contains_key(&cb));
                    if !is_scratch {
                        match (ld_layout, st_layout) {
                            (MemLayout::Tile { x, y, .. }, MemLayout::Tile { .. }) => {
                                let elem_size = dtype.bit_size() as u32 / 8;
                                let tile_bytes = x as u32 * y as u32 * elem_size;
                                let page_size = TT_DRAM_PAGE_BYTES;
                                writeln!(src, "{}cb{}.reserve_back(1);", self.indent, cb);
                                writeln!(
                                    src,
                                    "{}uint64_t rnoc{} = p{}.get_noc_addr((uint32_t)((r{}*{})/{}), (uint32_t)((r{}*{})%{}));",
                                    self.indent, op_id, ld_src, ld_idx, elem_size, page_size, ld_idx, elem_size, page_size
                                );
                                writeln!(
                                    src,
                                    "{}noc_async_read(rnoc{}, cb{}.get_write_ptr(), {});",
                                    self.indent, op_id, cb, tile_bytes
                                );
                                writeln!(src, "{}noc_async_read_barrier();", self.indent);
                                writeln!(src, "{}cb{}.push_back(1);", self.indent, cb);
                            }
                            _ => todo!("tenstorrent2 reader only supports tile stores"),
                        }
                    }
                }
                Op::Loop { len } => {
                    writeln!(src, "{}for (uint32_t r{} = 0; r{} < r{}; r{}++) {{", self.indent, op_id, op_id, len, op_id);
                    self.indent.push_str("  ");
                }
                Op::EndLoop => {
                    self.indent.pop();
                    self.indent.pop();
                    writeln!(src, "{}}}", self.indent);
                }
                Op::Barrier => {}
                Op::Range { axis, kind, .. } => match kind {
                    RangeKind::Group(_) => {
                        let arg = self.arg_pos.len() as u32 + *axis;
                        writeln!(src, "{}uint32_t r{} = get_arg_val<uint32_t>({});", self.indent, op_id, arg);
                    }
                    RangeKind::Local(_) => {
                        unreachable!(
                            "tenstorrent does not have local threads; local indices should have been converted to loops by the opt_tenstorrent_tile optimization pass"
                        )
                    }
                    RangeKind::Warp(_) => todo!("tenstorrent2 reader warp range"),
                },
                Op::Const(_) | Op::Binary { .. } | Op::Mad { .. } | Op::Cast { .. } => {
                    let s = self.emit_scalar_op(kernel, op_id)?;
                    src.push_str(&s);
                }
                Op::Unary { .. }
                | Op::Bitcast { .. }
                | Op::Stack { .. }
                | Op::Index { .. }
                | Op::If { .. }
                | Op::EndIf
                | Op::Wmma { .. }
                | Op::ReduceTile { .. }
                | Op::MatmulTile { .. }
                | Op::TransposeTile { .. }
                | Op::Asm { .. }
                | Op::Move { .. }
                | Op::Reduce { .. } => todo!("tenstorrent2 reader op"),
            }
        }
        writeln!(src, "{}noc_async_read_barrier();", self.indent);
        writeln!(src, "}}");
        self.reader = TTKernel::Reader { src, params };
        Ok(())
    }

    /// Generate the writer (dataflow drain) kernel from the writer op list.
    #[allow(unused_must_use)]
    fn generate_writer(&mut self, kernel: &Kernel) -> Result<(), BackendError> {
        let SectionData { ops, .. } = kernel.get_needed_ops(TtSection::Writer);
        // Section params in list order become this section's runtime args.
        let mut params: Vec<OpId> = Vec::new();
        for &op_id in &ops {
            if matches!(kernel.ops[op_id].op, Op::Param { .. }) {
                params.push(op_id);
            }
        }
        self.arg_pos = Map::default();
        for (i, &p) in params.iter().enumerate() {
            self.arg_pos.insert(p, i as u32);
        }
        self.indent = String::from("  ");
        let mut prev_accessor: Option<String> = None;
        // CBs read (drained) by this section: waited upfront at loop
        // boundaries (see Loop arm).
        let mut writer_loop_cbs: Vec<CBId> = Vec::new();
        for &op_id in &ops {
            if let Op::Store { src, .. } = kernel.ops[op_id].op {
                if let Op::Load { src: cb_src, .. } = kernel.ops[src].op {
                    if let Some(&cb) = self.cb_map.get(&cb_src) {
                        if !writer_loop_cbs.contains(&cb) {
                            writer_loop_cbs.push(cb);
                        }
                    }
                }
            }
        }
        writer_loop_cbs.sort();
        let mut src = String::new();
        writeln!(src, "#include <cstdint>");
        writeln!(src, "#include \"api/dataflow/dataflow_api.h\"");
        writeln!(src, "#include \"api/dataflow/noc.h\"");
        writeln!(src, "#include \"api/dataflow/circular_buffer.h\"");
        writeln!(src, "#include \"api/tensor/noc_traits.h\"");
        writeln!(src, "#include \"api/debug/dprint.h\"");
        writeln!(src, "void kernel_main() {{");
        let mut cbs: Vec<CBId> = self.cb_map.iter().map(|(_, &cb)| cb).collect();
        cbs.sort();
        for cb in cbs {
            writeln!(src, "{}CircularBuffer cb{cb}(tt::CBIndex::c_{cb});", self.indent);
        }
        // Accessors only for the GlobalMut params this section writes.
        for &op_id in &ops {
            if let Op::Param { kind: ParamKind::GlobalMut, .. } = kernel.ops[op_id].op {
                let arg = self.arg_pos.get(&op_id).copied().expect("tenstorrent2 writer param missing from section args");
                writeln!(src, "{}uint32_t out{} = get_arg_val<uint32_t>({});", self.indent, op_id, arg);
                let cta = match &prev_accessor {
                    None => String::from("0"),
                    Some(prev) => format!("{prev}.next_compile_time_args_offset()"),
                };
                writeln!(src, "{}auto args_out{} = TensorAccessorArgs<{}>({});", self.indent, op_id, cta, arg);
                writeln!(
                    src,
                    "{}auto p_out{} = TensorAccessor(args_out{}, out{}, {});",
                    self.indent, op_id, op_id, op_id, TT_DRAM_PAGE_BYTES
                );
                prev_accessor = Some(format!("args_out{op_id}"));
            }
        }
        let mut loop_depth = 0u32;
        let mut loop_popped: Map<CBId, ()> = Map::default();
        let n = ops.len();
        for i in 0..n {
            let op_id = ops[i];
            match &kernel.ops[op_id].op {
                Op::Param { kind: ParamKind::GlobalMut, .. } => {
                    // Accessor emitted up front (see above).
                }
                Op::Param { kind: ParamKind::Variable, .. } => {
                    let arg = self.arg_pos.get(&op_id).copied().expect("tenstorrent2 writer param missing from section args");
                    let Op::Param { dtype, .. } = kernel.ops[op_id].op else {
                        unreachable!("tenstorrent2 param changed under us");
                    };
                    writeln!(
                        src,
                        "{}{} r{} = ({})get_arg_val<uint32_t>({});",
                        self.indent,
                        dtype.c_type(),
                        op_id,
                        dtype.c_type(),
                        arg
                    );
                }
                Op::Storage { scope: MemScope::Circular, .. } => {
                    // Declared up front for every shared CB (see driver).
                }
                Op::Storage { scope: MemScope::Local, .. } => {
                    unreachable!(
                        "tenstorrent does not have local threads; local indices should have been converted to loops by the opt_tenstorrent_tile optimization pass"
                    )
                }
                Op::Storage { .. } => {
                    todo!("tenstorrent2 writer storage scope")
                }
                Op::Load { .. } => {
                    // Consumed at the draining store below.
                }
                Op::Store { dst, src: store_src, index: st_idx, layout: st_layout } => {
                    let Op::Load { src: cb_src, index: _ld_idx, layout: ld_layout } = kernel.ops[*store_src].op else {
                        return Err(BackendError {
                            status: ErrorStatus::KernelCompilation,
                            context: format!(
                                "tenstorrent2: writer supports only CB to DRAM stores, op {op_id} has ops in between"
                            )
                            .into(),
                        });
                    };
                    let Some(&cb) = self.cb_map.get(&cb_src) else {
                        unreachable!("tenstorrent2 writer load targets unmapped CB");
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
                            let page_size = TT_DRAM_PAGE_BYTES;
                            writeln!(src, "{}cb{}.wait_front(1);", self.indent, cb);
                            writeln!(
                                src,
                                "{}uint64_t wnoc{} = p_out{}.get_noc_addr((uint32_t)((r{}*{})/{}), (uint32_t)((r{}*{})%{}));",
                                self.indent, op_id, dst, st_idx, elem_size, page_size, st_idx, elem_size, page_size
                            );
                            writeln!(
                                src,
                                "{}noc_async_write(cb{}.get_read_ptr(), wnoc{}, {});",
                                self.indent, cb, op_id, tile_bytes
                            );
                            writeln!(src, "{}noc_async_write_barrier();", self.indent);
                            writeln!(src, "{}cb{}.pop_front(1);", self.indent, cb);
                            if loop_depth > 0 {
                                loop_popped.insert(cb, ());
                            }
                        }
                        _ => todo!("tenstorrent2 writer only supports tile stores"),
                    }
                }
                Op::Loop { len } => {
                    if loop_depth == 0 {
                        for cb in &writer_loop_cbs {
                            writeln!(src, "{}cb{}.wait_front(1);", self.indent, cb);
                            writeln!(src, "{}uint32_t wbase{} = cb{}.get_read_ptr();", self.indent, cb, cb);
                        }
                    }
                    writeln!(src, "{}for (uint32_t r{} = 0; r{} < r{}; r{}++) {{", self.indent, op_id, op_id, len, op_id);
                    self.indent.push_str("  ");
                    loop_depth += 1;
                }
                Op::EndLoop => {
                    self.indent.pop();
                    self.indent.pop();
                    writeln!(src, "{}}}", self.indent);
                    if loop_depth == 1 {
                        writeln!(src, "{}noc_async_write_barrier();", self.indent);
                        for cb in &writer_loop_cbs {
                            // Tile streaming pops per iteration (see the
                            // tile Store above); popping again here would
                            // over-pop and stall.
                            if loop_popped.contains_key(cb) {
                                continue;
                            }
                            writeln!(src, "{}cb{}.pop_front(1);", self.indent, cb);
                        }
                    }
                    loop_depth -= 1;
                }
                Op::Barrier => {}
                Op::Range { axis, kind, .. } => match kind {
                    RangeKind::Group(_) => {
                        let arg = self.arg_pos.len() as u32 + *axis;
                        writeln!(src, "{}uint32_t r{} = get_arg_val<uint32_t>({});", self.indent, op_id, arg);
                    }
                    RangeKind::Local(_) => {
                        unreachable!(
                            "tenstorrent does not have local threads; local indices should have been converted to loops by the opt_tenstorrent_tile optimization pass"
                        )
                    }
                    RangeKind::Warp(_) => todo!("tenstorrent2 writer warp range"),
                },
                Op::Const(_) | Op::Binary { .. } | Op::Mad { .. } | Op::Cast { .. } => {
                    let s = self.emit_scalar_op(kernel, op_id)?;
                    src.push_str(&s);
                }
                Op::Param { .. } => {
                    todo!("tenstorrent2 writer Global param")
                }
                Op::Unary { .. }
                | Op::Bitcast { .. }
                | Op::Stack { .. }
                | Op::Index { .. }
                | Op::If { .. }
                | Op::EndIf
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
        self.writer = TTKernel::Writer { src, params };
        Ok(())
    }

    /// Shared scalar ALU emission across all sections: identical text
    /// everywhere, no section argument.
    #[allow(unused_must_use)]
    fn emit_scalar_op(&self, kernel: &Kernel, op_id: OpId) -> Result<String, BackendError> {
        let mut out = String::new();
        match &kernel.ops[op_id].op {
            Op::Const(val) => {
                writeln!(out, "{}{} r{} = {};", self.indent, val.dtype().c_type(), op_id, val.c_code());
            }
            Op::Binary { x, y, bop } => {
                let dt = kernel.dtype(op_id);
                let _ = match bop {
                    BOp::Add => writeln!(out, "{}{} r{} = r{} + r{};", self.indent, dt.c_type(), op_id, x, y),
                    BOp::Sub => writeln!(out, "{}{} r{} = r{} - r{};", self.indent, dt.c_type(), op_id, x, y),
                    BOp::Mul => writeln!(out, "{}{} r{} = r{} * r{};", self.indent, dt.c_type(), op_id, x, y),
                    BOp::Div => writeln!(out, "{}{} r{} = r{} / r{};", self.indent, dt.c_type(), op_id, x, y),
                    BOp::Mod => writeln!(out, "{}{} r{} = r{} % r{};", self.indent, dt.c_type(), op_id, x, y),
                    BOp::Max => writeln!(out, "{}{} r{} = r{} > r{} ? r{} : r{};", self.indent, dt.c_type(), op_id, x, y, x, y),
                    BOp::Cmplt => writeln!(out, "{}{} r{} = r{} < r{};", self.indent, dt.c_type(), op_id, x, y),
                    BOp::Cmpgt => writeln!(out, "{}{} r{} = r{} > r{};", self.indent, dt.c_type(), op_id, x, y),
                    BOp::Cmpge => writeln!(out, "{}{} r{} = r{} >= r{};", self.indent, dt.c_type(), op_id, x, y),
                    BOp::Eq => writeln!(out, "{}{} r{} = r{} == r{};", self.indent, dt.c_type(), op_id, x, y),
                    BOp::NotEq => writeln!(out, "{}{} r{} = r{} != r{};", self.indent, dt.c_type(), op_id, x, y),
                    BOp::And => writeln!(out, "{}{} r{} = r{} && r{};", self.indent, dt.c_type(), op_id, x, y),
                    BOp::Or => writeln!(out, "{}{} r{} = r{} || r{};", self.indent, dt.c_type(), op_id, x, y),
                    BOp::BitXor => writeln!(out, "{}{} r{} = r{} ^ r{};", self.indent, dt.c_type(), op_id, x, y),
                    BOp::BitOr => writeln!(out, "{}{} r{} = r{} | r{};", self.indent, dt.c_type(), op_id, x, y),
                    BOp::BitAnd => writeln!(out, "{}{} r{} = r{} & r{};", self.indent, dt.c_type(), op_id, x, y),
                    BOp::BitShiftLeft => writeln!(out, "{}{} r{} = r{} << r{};", self.indent, dt.c_type(), op_id, x, y),
                    BOp::BitShiftRight => writeln!(out, "{}{} r{} = r{} >> r{};", self.indent, dt.c_type(), op_id, x, y),
                    BOp::Pow => todo!("tenstorrent2 scalar pow"),
                };
            }
            Op::Mad { x, y, z } => {
                let dt = kernel.dtype(op_id);
                writeln!(out, "{}{} r{} = r{} * r{} + r{};", self.indent, dt.c_type(), op_id, x, y, z);
            }
            Op::Cast { x, dtype } => {
                writeln!(out, "{}{} r{} = ({})r{};", self.indent, dtype.c_type(), op_id, dtype.c_type(), x);
            }
            Op::Param { .. }
            | Op::Storage { .. }
            | Op::Load { .. }
            | Op::Store { .. }
            | Op::Range { .. }
            | Op::Loop { .. }
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
        Ok(out)
    }

    /// Tiled SFPU emission (compute only).
    fn emit_tiled_op(&mut self, _kernel: &Kernel, _op_id: OpId) -> Result<String, BackendError> {
        todo!("tenstorrent2 tiled emission")
    }
}
