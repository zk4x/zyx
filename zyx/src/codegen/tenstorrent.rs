use crate::{
    DType, Map, Set,
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
        /// Global head-order ordinals: this section's runtime args.
        ordinals: Vec<u32>,
    },
    Compute {
        src: String,
        params: Vec<OpId>,
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
    pub(crate) fn generate_tenstorrent(&self) -> Result<Compiler, BackendError> {
        let mut compiler = Compiler::new(self);
        compiler.check_sections(self)?;
        let reader_data = self.get_needed_ops(TtSection::Reader);
        let compute_data = self.get_needed_ops(TtSection::Compute);
        let writer_data = self.get_needed_ops(TtSection::Writer);
        let [reader_params, compute_params, writer_params] =
            compiler.section_param_lists(self, &reader_data, &compute_data, &writer_data);
        // Per-section param ordinals (global head order): each section's
        // params are in IR order, so the ordinals ascend already.
        let [reader_ord, compute_ord, writer_ord] = [&reader_params, &compute_params, &writer_params].map(|params| {
            let ordinals: Vec<u32> = params.iter().map(|p| compiler.param_ordinal_of[p]).collect();
            debug_assert!(ordinals.windows(2).all(|w| w[0] < w[1]), "tenstorrent2 section params not in head order");
            ordinals
        });
        compiler.allocate_cbs(self)?;

        compiler.check_balance(self)?;
        compiler.generate_reader(self, &reader_data, &reader_params, &reader_ord)?;
        compiler.generate_compute(self, &compute_data, &compute_params, &compute_ord)?;
        compiler.generate_writer(self, &writer_data, &writer_params, &writer_ord)?;

        Ok(compiler)
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
                    if section == tt_section {
                        structural.insert(scan);
                    }
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
            if needed.contains(&op_id) || structural.contains(&op_id) {
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
/// [`Kernel::generate_tenstorrent`] returns this struct to the backend,
/// which reads the section kernels, the CB config, and the dtypes off it.
pub(crate) struct Compiler {
    /// One CBId per Circular storage, shared by all three sections.
    cb_map: Map<OpId, CBId>,
    /// Runtime CB config in id order: (id, tt format, tile bytes).
    /// Format and tile bytes follow the CB storage dtype; an
    /// unmappable dtype is a compilation error, never a silent default.
    pub(crate) cb_config: Vec<(u32, u32, u32)>,
    /// Global head-order ordinal of every param (all kinds).
    pub(crate) param_ordinal_of: Map<OpId, u32>,
    /// Global params in head order (kernel inputs).
    pub(crate) input_dtypes: Vec<DType>,
    /// GlobalMut params in head order (kernel outputs).
    pub(crate) output_dtypes: Vec<DType>,
    /// Generated section kernels (filled by generation).
    pub(crate) reader: TTKernel,
    /// Generated section kernels (filled by generation).
    pub(crate) compute: TTKernel,
    /// Generated section kernels (filled by generation).
    pub(crate) writer: TTKernel,
    /// Scratch CBs (stored in compute, never loaded by the writer):
    /// no DRAM traffic is emitted for them in any section.
    scratch_cbs: Set<CBId>,
}

impl Compiler {
    /// Build compiler state from a kernel: the param ordinals and
    /// input/output dtypes. The CB map is filled by [`Compiler::allocate_cbs`].
    fn new(kernel: &Kernel) -> Self {
        // Param ordinals (all kinds, head order) and Global/GlobalMut
        // dtypes: one walk.
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
        Self {
            cb_map: Map::default(),
            cb_config: Vec::new(),
            param_ordinal_of,
            input_dtypes,
            output_dtypes,
            reader: TTKernel::None,
            compute: TTKernel::None,
            writer: TTKernel::None,
            scratch_cbs: Set::default(),
        }
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

    /// Full CB allocation resolution into `self.cb_map` (storing the
    /// runtime config in `self.cb_config`): first-touch ids, page/L1
    /// validity, hardware count fit. The single point that assigns CB
    /// ids; anything unmappable is a compilation error.
    fn allocate_cbs(&mut self, kernel: &Kernel) -> Result<(), BackendError> {
        // Single walk: first-touch CB ids plus scratch classification.
        // Scratch (stored in compute, never read by the writer) is read
        // lexically here: cross-section tile flow goes through CBs, so a
        // section's closure stores/loads are exactly its lexical
        // stores/loads. Registration order is load-then-store per op,
        // same as before.
        let mut next_cb = CBId::ZERO;
        let mut section = TtSection::Reader;
        let mut stored_in_compute: Set<CBId> = Set::default();
        let mut read_by_writer: Set<CBId> = Set::default();
        let mut scan = kernel.head;
        for _ in 0..10_000 {
            if scan.is_null() {
                break;
            }
            match kernel.ops[scan].op {
                Op::Barrier => section.advance(),
                Op::Load { ref src, .. } => {
                    if let Op::Storage { scope: MemScope::Circular, .. } = kernel.ops[*src].op {
                        if !self.cb_map.contains_key(src) {
                            self.cb_map.insert(*src, next_cb);
                            next_cb.inc();
                        }
                        if section == TtSection::Writer {
                            read_by_writer.insert(self.cb_map[src]);
                        }
                    }
                }
                Op::Store { ref dst, .. } => {
                    if let Op::Storage { scope: MemScope::Circular, .. } = kernel.ops[*dst].op {
                        if !self.cb_map.contains_key(dst) {
                            self.cb_map.insert(*dst, next_cb);
                            next_cb.inc();
                        }
                        if section == TtSection::Compute {
                            stored_in_compute.insert(self.cb_map[dst]);
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
        let mut scratch_cbs: Set<CBId> = Set::default();
        for &cb in stored_in_compute.iter() {
            if !read_by_writer.contains(&cb) {
                scratch_cbs.insert(cb);
            }
        }
        self.scratch_cbs = scratch_cbs;
        // Hardware CB count fit.
        let num_circular_buffers = kernel.device_info().num_circular_buffers;
        if self.cb_map.len() > num_circular_buffers as usize {
            return Err(BackendError {
                status: ErrorStatus::TooManyCircularBuffers,
                context: format!(
                    "tenstorrent2: kernel needs {} circular buffers, device holds {num_circular_buffers}",
                    self.cb_map.len()
                )
                .into(),
            });
        }
        // Every mapped CB holds whole 2048B pages within the
        // single-core L1 budget.
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
        // Runtime config in id order. Format and tile bytes follow the
        // CB storage dtype; anything else is a compilation error.
        let mut cb_ops: Vec<(CBId, OpId)> = self.cb_map.iter().map(|(&op, &cb)| (cb, op)).collect();
        cb_ops.sort_by_key(|&(cb, _)| cb);
        let mut cb_config: Vec<(u32, u32, u32)> = Vec::with_capacity(cb_ops.len());
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
            cb_config.push((cb.0, fmt, tb));
        }
        self.cb_config = cb_config;
        Ok(())
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
            match kernel.ops[scan].op {
                Op::Loop { ref len } => {
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
                Op::Store { ref dst, layout: MemLayout::Tile { .. }, .. } => {
                    if let Some(&cb) = self.cb_map.get(dst) {
                        let trip: i64 = trips.iter().product();
                        *pushes.entry(cb).or_insert(0) += trip;
                    }
                }
                Op::Load { ref src, layout: MemLayout::Tile { .. }, .. } => {
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
    fn generate_reader(
        &mut self,
        kernel: &Kernel,
        reader_data: &SectionData,
        params: &[OpId],
        ordinals: &[u32],
    ) -> Result<(), BackendError> {
        let ops = &reader_data.ops;
        // Section params in list order become this section's runtime args.
        let mut arg_pos: Map<OpId, u32> = Map::default();
        for (i, &p) in params.iter().enumerate() {
            arg_pos.insert(p, i as u32);
        }
        let mut indent = String::from("  ");
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
            writeln!(src, "{indent}CircularBuffer cb{cb}(tt::CBIndex::c_{cb});");
        }
        let n = ops.len();
        for i in 0..n {
            let op_id = ops[i];
            match kernel.ops[op_id].op {
                Op::Param { dtype, kind, .. } => match kind {
                    ParamKind::Global => {
                        let arg = arg_pos.get(&op_id).copied().expect("tenstorrent2 reader param missing from section args");
                        writeln!(src, "{indent}uint32_t src{op_id} = get_arg_val<uint32_t>({arg});");
                        let cta = match &prev_accessor {
                            None => String::from("0"),
                            Some(prev) => format!("{prev}.next_compile_time_args_offset()"),
                        };
                        writeln!(src, "{indent}auto args{op_id} = TensorAccessorArgs<{cta}>({arg});");
                        writeln!(src, "{indent}auto p{op_id} = TensorAccessor(args{op_id}, src{op_id}, {TT_DRAM_PAGE_BYTES});");
                        prev_accessor = Some(format!("args{op_id}"));
                    }
                    ParamKind::Variable => {
                        let arg = arg_pos.get(&op_id).copied().expect("tenstorrent2 reader param missing from section args");
                        writeln!(src, "{indent}{} r{op_id} = ({})get_arg_val<uint32_t>({arg});", dtype.c_type(), dtype.c_type());
                    }
                    ParamKind::GlobalMut => {
                        let arg = arg_pos.get(&op_id).copied().expect("tenstorrent2 reader param missing from section args");
                        writeln!(src, "{indent}uint32_t dst{op_id} = get_arg_val<uint32_t>({arg});");
                        let cta = match &prev_accessor {
                            None => String::from("0"),
                            Some(prev) => format!("{prev}.next_compile_time_args_offset()"),
                        };
                        writeln!(src, "{indent}auto args{op_id} = TensorAccessorArgs<{cta}>({arg});");
                        writeln!(src, "{indent}auto p{op_id} = TensorAccessor(args{op_id}, dst{op_id}, {TT_DRAM_PAGE_BYTES});");
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
                        unreachable!("tenstorrent2 reader loads only from Global params");
                    };
                    let Op::Storage { dtype, scope: MemScope::Circular, .. } = kernel.ops[*dst].op else {
                        unreachable!("tenstorrent2 reader stores only target Circular CBs");
                    };
                    let Some(&cb) = self.cb_map.get(dst) else {
                        unreachable!("tenstorrent2 reader store targets unmapped CB");
                    };
                    let is_scratch = self.scratch_cbs.contains(&cb);
                    if !is_scratch {
                        match (ld_layout, st_layout) {
                            (MemLayout::Tile { x, y, .. }, MemLayout::Tile { .. }) => {
                                let elem_size = dtype.bit_size() as u32 / 8;
                                let tile_bytes = x as u32 * y as u32 * elem_size;
                                let page_size = TT_DRAM_PAGE_BYTES;
                                writeln!(src, "{indent}cb{cb}.reserve_back(1);");
                                writeln!(
                                    src,
                                    "{indent}uint64_t rnoc{op_id} = p{ld_src}.get_noc_addr((uint32_t)((r{ld_idx}*{elem_size})/{page_size}), (uint32_t)((r{ld_idx}*{elem_size})%{page_size}));"
                                );
                                writeln!(src, "{indent}noc_async_read(rnoc{op_id}, cb{cb}.get_write_ptr(), {tile_bytes});");
                                writeln!(src, "{indent}noc_async_read_barrier();");
                                writeln!(src, "{indent}cb{cb}.push_back(1);");
                            }
                            _ => todo!("tenstorrent2 reader only supports tile stores"),
                        }
                    }
                }
                Op::Loop { len } => {
                    writeln!(src, "{indent}for (uint32_t r{op_id} = 0; r{op_id} < r{len}; r{op_id}++) {{");
                    indent.push_str("  ");
                }
                Op::EndLoop => {
                    indent.pop();
                    indent.pop();
                    writeln!(src, "{indent}}}");
                }
                Op::Barrier => {}
                Op::Range { axis, kind, .. } => match kind {
                    RangeKind::Group(_) => {
                        let arg = arg_pos.len() as u32 + axis;
                        writeln!(src, "{indent}uint32_t r{op_id} = get_arg_val<uint32_t>({arg});");
                    }
                    RangeKind::Local(_) => {
                        unreachable!(
                            "tenstorrent does not have local threads; local indices should have been converted to loops by the opt_tenstorrent_tile optimization pass"
                        )
                    }
                    RangeKind::Warp(_) => todo!("tenstorrent2 reader warp range"),
                },
                Op::Const(_) | Op::Binary { .. } | Op::Mad { .. } | Op::Cast { .. } => {
                    let s = self.emit_scalar_op(kernel, op_id, &indent)?;
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
        writeln!(src, "{indent}noc_async_read_barrier();");
        writeln!(src, "}}");
        self.reader = TTKernel::Reader { src, ordinals: ordinals.to_vec() };
        Ok(())
    }

    /// Generate the compute kernel from the compute op list: GEMM-only.
    /// One `mm_init` per distinct input/output CB triple up front, then
    /// the op walk: scalar bound arithmetic inline, CB loads waited (and
    /// popped at their loop level on the way out), real output stores
    /// packed, SSA accumulator copies coalesced onto DST slots (their
    /// loads/stores/adds emit nothing). DST acquire/release pairs with
    /// the pack scope (one output tile's live range), never the K loop.
    /// Anything beyond one accumulation cone per output (tiled ALU, CB
    /// copies, packs inside the K loop, DRAM touched from compute) is a
    /// loud halt.
    #[allow(unused_must_use)]
    fn generate_compute(
        &mut self,
        kernel: &Kernel,
        compute_data: &SectionData,
        params: &[OpId],
        ordinals: &[u32],
    ) -> Result<(), BackendError> {
        let ops = &compute_data.ops;
        // Section params in list order become this section's runtime args.
        let mut arg_pos: Map<OpId, u32> = Map::default();
        for (i, &p) in params.iter().enumerate() {
            arg_pos.insert(p, i as u32);
        }
        // Pack analysis, all driver locals: per real output store (a
        // store to a non-scratch CB) its DST slot (pack order) and its
        // matmuls; the crossed Adds are fiction the hardware folds.
        let mut pack_slot: Map<OpId, u32> = Map::default();
        let mut mm_slot: Map<OpId, u32> = Map::default();
        let mut pack_mms: Map<OpId, Vec<OpId>> = Map::default();
        let mut fiction_adds: Set<OpId> = Set::default();
        let mut triples: Vec<(CBId, CBId, CBId)> = Vec::new();
        let mut wide_dst = false;
        for &op_id in ops {
            let Op::Store { dst, src, .. } = kernel.ops[op_id].op else {
                continue;
            };
            if !matches!(kernel.ops[dst].op, Op::Storage { scope: MemScope::Circular, .. }) {
                continue;
            }
            // (non-Circular dests error in the main walk below)
            let Some(&dest_cb) = self.cb_map.get(&dst) else {
                unreachable!("tenstorrent2 compute store targets unmapped CB");
            };
            if self.scratch_cbs.contains(&dest_cb) {
                continue;
            }
            // Real output tile: chase the src cone through fiction Adds.
            // Loads sever the cone (CBs are the section boundary), so a
            // scratch reload splices in the in-section store to that CB
            // (the loop-carried chain, and the proof the adds are
            // fiction); a direct MatmulTile packs fresh DST content.
            let slot = pack_slot.len() as u32;
            let mut mms: Vec<OpId> = Vec::new();
            let mut acc_seen: Option<OpId> = None;
            let mut stack = vec![src];
            let mut seen: Set<OpId> = Set::default();
            while let Some(id) = stack.pop() {
                if !seen.insert(id) {
                    continue;
                }
                match kernel.ops[id].op {
                    Op::Binary { x, y, bop: BOp::Add } if compute_data.dtypes[&id].1 != MemLayout::Scalar => {
                        fiction_adds.insert(id);
                        stack.push(x);
                        stack.push(y);
                    }
                    Op::Load { src: ld_src, .. } => {
                        if self.cb_map.get(&ld_src).is_some_and(|&cb| self.scratch_cbs.contains(&cb)) {
                            if acc_seen.replace(ld_src).is_some_and(|prev| prev != ld_src) {
                                todo!("tenstorrent2 pack of two accumulators");
                            }
                            let mut found = false;
                            for &store in ops {
                                if let Op::Store { dst: st_dst, src: st_src, .. } = kernel.ops[store].op {
                                    if st_dst == ld_src {
                                        stack.push(st_src);
                                        found = true;
                                    }
                                }
                            }
                            if !found {
                                todo!("tenstorrent2 compute accumulator never stored in-section");
                            }
                        } else {
                            todo!("tenstorrent2 compute pack of non-accumulator tile");
                        }
                    }
                    Op::MatmulTile { .. } => {
                        mms.push(id);
                    }
                    Op::Const(_) | Op::Param { .. } | Op::Storage { .. } => {}
                    Op::Binary { .. }
                    | Op::Mad { .. }
                    | Op::Cast { .. }
                    | Op::Unary { .. }
                    | Op::Bitcast { .. }
                    | Op::Stack { .. }
                    | Op::Index { .. }
                    | Op::TransposeTile { .. }
                    | Op::ReduceTile { .. }
                    | Op::Asm { .. }
                    | Op::Move { .. }
                    | Op::Reduce { .. } => todo!("tenstorrent2 compute real epilogue op in pack cone"),
                    Op::Wmma { .. } => {
                        return Err(BackendError {
                            status: ErrorStatus::KernelCompilation,
                            context: "tenstorrent2: Wmma is a tensor-core op, not emittable here".into(),
                        });
                    }
                    Op::Store { .. }
                    | Op::Range { .. }
                    | Op::Loop { .. }
                    | Op::EndLoop
                    | Op::If { .. }
                    | Op::EndIf
                    | Op::Barrier => unreachable!("tenstorrent2 compute pack cone holds a non-value op"),
                }
            }
            if mms.is_empty() {
                todo!("tenstorrent2 compute pack store with no MatmulTile in its cone");
            }
            for &mm in &mms {
                if let Some(&prev) = mm_slot.get(&mm) {
                    if prev != slot {
                        todo!("tenstorrent2 MatmulTile output fanned out to two packs");
                    }
                } else {
                    mm_slot.insert(mm, slot);
                }
                let Op::MatmulTile { x, y } = kernel.ops[mm].op else {
                    unreachable!("tenstorrent2 mm list holds a non-matmul");
                };
                let Op::Load { src: a_src, .. } = kernel.ops[x].op else {
                    todo!("tenstorrent2 MatmulTile x not loaded from a CB");
                };
                let Op::Load { src: b_src, .. } = kernel.ops[y].op else {
                    todo!("tenstorrent2 MatmulTile y not loaded from a CB");
                };
                let (Some(&cb_a), Some(&cb_b)) = (self.cb_map.get(&a_src), self.cb_map.get(&b_src)) else {
                    unreachable!("tenstorrent2 MatmulTile input targets unmapped CB");
                };
                // Unpacker inputs must be trafficked: a scratch CB is never
                // pushed, so waiting on it hangs the core (deadlock, not an
                // error). An MM consuming an accumulator needs a real
                // intermediate push (the bmm c_24 pattern), which GEMM-only
                // lowering does not emit.
                if self.scratch_cbs.contains(&cb_a) || self.scratch_cbs.contains(&cb_b) {
                    return Err(BackendError {
                        status: ErrorStatus::CircularBufferImbalance,
                        context: format!("tenstorrent2: matmul {mm} reads scratch CB{cb_a}/CB{cb_b} with no traffic").into(),
                    });
                }
                let triple = (cb_a, cb_b, dest_cb);
                if !triples.contains(&triple) {
                    triples.push(triple);
                }
            }
            pack_mms.insert(op_id, mms);
            pack_slot.insert(op_id, slot);
            if matches!(kernel.ops[dst].op, Op::Storage { dtype: DType::F32, .. }) {
                wide_dst = true;
            }
        }
        // DST width: 8 F32 slots or 16 F16 slots; more packs do not fit.
        let dst_slots = if wide_dst { 8u32 } else { 16u32 };
        if pack_slot.len() as u32 > dst_slots {
            return Err(BackendError {
                status: ErrorStatus::KernelCompilation,
                context: format!("tenstorrent2: compute section packs {} outputs, DST holds {dst_slots}", pack_slot.len()).into(),
            });
        }
        // Loop structure (one walk): K-loops directly contain a
        // MatmulTile (commit/wait after the full accumulation); every
        // loop's parent (ancestry for the scope guard); every MM's K;
        // every pack store's scope. Acquire/release pair with the pack
        // scope (one output tile's lifetime), never the K loop: acquiring
        // per K iteration would zero the DST mid-accumulation.
        let mut k_loops: Set<OpId> = Set::default();
        let mut loop_parent: Map<OpId, Option<OpId>> = Map::default();
        let mut mm_k: Map<OpId, OpId> = Map::default();
        let mut pack_scope: Map<OpId, Option<OpId>> = Map::default();
        let mut loop_stack: Vec<OpId> = Vec::new();
        for &op_id in ops {
            match kernel.ops[op_id].op {
                Op::Loop { .. } => {
                    loop_parent.insert(op_id, loop_stack.last().copied());
                    loop_stack.push(op_id);
                }
                Op::EndLoop => {
                    loop_stack.pop().expect("tenstorrent2 compute EndLoop without Loop");
                }
                Op::MatmulTile { .. } => {
                    let Some(&k) = loop_stack.last() else {
                        todo!("tenstorrent2 loopless MatmulTile");
                    };
                    k_loops.insert(k);
                    mm_k.insert(op_id, k);
                }
                Op::Store { .. } if pack_slot.contains_key(&op_id) => {
                    pack_scope.insert(op_id, loop_stack.last().copied());
                }
                _ => {}
            }
        }
        let acquire_top = pack_scope.values().any(|s| s.is_none());
        let acquire_loops: Set<OpId> = pack_scope.values().filter_map(|&s| s).collect();
        let mut indent = String::from("  ");
        let mut src = String::new();
        writeln!(src, "#include <cstdint>");
        writeln!(src, "#include \"api/compute/tile_move_copy.h\"");
        writeln!(src, "#include \"api/compute/matmul.h\"");
        // Packer output-format reconfig (fp32_accuracy doc): without it
        // the packer treats DST as 16-bit even under fp32_dest_acc_en.
        writeln!(src, "#include \"api/compute/reconfig_data_format.h\"");
        writeln!(src, "#include \"hostdevcommon/kernel_structs.h\"");
        writeln!(src, "using std::uint32_t;");
        writeln!(src, "void kernel_main() {{");
        // Shared CB aliases: every traffic call below names cb{N}.
        let mut cbs: Vec<CBId> = self.cb_map.iter().map(|(_, &cb)| cb).collect();
        cbs.sort();
        for cb in cbs {
            writeln!(src, "{indent}constexpr uint32_t cb{cb} = {cb};");
        }
        // One mm_init per distinct triple (unpacker + packer config).
        for (a, b, o) in &triples {
            writeln!(src, "{indent}mm_init(cb{a}, cb{b}, cb{o});");
        }
        // Function-scope accumulation acquires once up front (pairs with
        // the end-of-function release below; every other scope acquires
        // at its loop entry in the walk).
        if acquire_top {
            writeln!(src, "{indent}tile_regs_acquire();");
        }
        let mut enclosing: Vec<OpId> = Vec::new();
        let mut loop_loads: Vec<Set<CBId>> = Vec::new();
        let n = ops.len();
        for i in 0..n {
            let op_id = ops[i];
            match kernel.ops[op_id].op {
                Op::Param { dtype, kind, .. } => match kind {
                    ParamKind::Variable => {
                        let arg = arg_pos.get(&op_id).copied().expect("tenstorrent2 compute param missing from section args");
                        writeln!(src, "{indent}{} r{op_id} = ({})get_arg_val<uint32_t>({arg});", dtype.c_type(), dtype.c_type());
                    }
                    ParamKind::Global | ParamKind::GlobalMut => {
                        return Err(BackendError {
                            status: ErrorStatus::ComputeAccessesDram,
                            context: format!("tenstorrent2: compute core cannot access DRAM param {op_id}").into(),
                        });
                    }
                },
                Op::Storage { scope: MemScope::Circular, .. } => {
                    // Aliased up front for every shared CB (see driver).
                }
                Op::Storage { scope: MemScope::Local, .. } => {
                    unreachable!(
                        "tenstorrent does not have local threads; local indices should have been converted to loops by the opt_tenstorrent_tile optimization pass"
                    )
                }
                Op::Storage { .. } => {
                    todo!("tenstorrent2 compute storage scope")
                }
                Op::Load { src: ld_src, .. } => {
                    if let Some(&cb) = self.cb_map.get(&ld_src) {
                        if self.scratch_cbs.contains(&cb) {
                            // Accumulator lives in DST, nothing to wait on.
                        } else {
                            writeln!(src, "{indent}cb_wait_front(cb{cb}, 1);");
                            let Some(top) = loop_loads.last_mut() else {
                                todo!("tenstorrent2 compute loopless CB load");
                            };
                            top.insert(cb);
                        }
                    } else if matches!(kernel.ops[ld_src].op, Op::Param { kind: ParamKind::Global | ParamKind::GlobalMut, .. }) {
                        return Err(BackendError {
                            status: ErrorStatus::ComputeAccessesDram,
                            context: format!("tenstorrent2: compute core cannot load DRAM param {ld_src}").into(),
                        });
                    } else {
                        todo!("tenstorrent2 compute scalar load");
                    }
                }
                Op::Store { dst: st_dst, .. } => {
                    if let Op::Storage { scope: MemScope::Circular, .. } = kernel.ops[st_dst].op {
                        let Some(&cb) = self.cb_map.get(&st_dst) else {
                            unreachable!("tenstorrent2 compute store targets unmapped CB");
                        };
                        if self.scratch_cbs.contains(&cb) {
                            // Fiction writeback into the DST slot, emits nothing.
                        } else {
                            let Some(&slot) = pack_slot.get(&op_id) else {
                                unreachable!("tenstorrent2 compute pack store missed by pack analysis");
                            };
                            if enclosing.iter().any(|l| k_loops.contains(l)) {
                                todo!("tenstorrent2 compute pack inside the K loop");
                            }
                            // Every feeding MM must execute inside the
                            // acquire scope (function scope holds all).
                            let scope = pack_scope[&op_id];
                            if scope.is_some() {
                                for &mm in &pack_mms[&op_id] {
                                    let mut k = mm_k[&mm];
                                    loop {
                                        if Some(k) == scope {
                                            break;
                                        }
                                        let Some(parent) = loop_parent[&k] else {
                                            todo!("tenstorrent2 compute MM outside pack scope");
                                        };
                                        k = parent;
                                    }
                                }
                            }
                            writeln!(src, "{indent}cb_reserve_back(cb{cb}, 1);");
                            writeln!(src, "{indent}pack_reconfig_data_format(cb{cb});");
                            writeln!(src, "{indent}pack_tile({slot}, cb{cb});");
                            writeln!(src, "{indent}cb_push_back(cb{cb}, 1);");
                        }
                    } else if matches!(kernel.ops[st_dst].op, Op::Param { kind: ParamKind::Global | ParamKind::GlobalMut, .. }) {
                        return Err(BackendError {
                            status: ErrorStatus::ComputeAccessesDram,
                            context: format!("tenstorrent2: compute core cannot store DRAM param {st_dst}").into(),
                        });
                    } else if matches!(kernel.ops[st_dst].op, Op::Storage { scope: MemScope::Local, .. }) {
                        unreachable!(
                            "tenstorrent does not have local threads; local indices should have been converted to loops by the opt_tenstorrent_tile optimization pass"
                        )
                    } else {
                        todo!("tenstorrent2 compute storage scope")
                    }
                }
                Op::MatmulTile { .. } => {
                    let Some(&slot) = mm_slot.get(&op_id) else {
                        todo!("tenstorrent2 dangling MatmulTile with no pack store");
                    };
                    let s = self.emit_tiled_op(kernel, op_id, &indent, slot)?;
                    src.push_str(&s);
                }
                Op::Loop { len } => {
                    writeln!(src, "{indent}for (uint32_t r{op_id} = 0; r{op_id} < r{len}; r{op_id}++) {{");
                    indent.push_str("  ");
                    enclosing.push(op_id);
                    loop_loads.push(Set::default());
                    if acquire_loops.contains(&op_id) {
                        writeln!(src, "{indent}tile_regs_acquire();");
                    }
                }
                Op::EndLoop => {
                    let loads = loop_loads.pop().expect("tenstorrent2 compute EndLoop without Loop");
                    let loop_id = enclosing.pop().expect("tenstorrent2 compute EndLoop without Loop");
                    let mut cbv: Vec<CBId> = loads.into_iter().collect();
                    cbv.sort();
                    for cb in cbv {
                        writeln!(src, "{indent}cb_pop_front(cb{cb}, 1);");
                    }
                    if acquire_loops.contains(&loop_id) {
                        writeln!(src, "{indent}tile_regs_release();");
                    }
                    indent.pop();
                    indent.pop();
                    writeln!(src, "{indent}}}");
                    // Commit once per accumulation, after the K loop closes:
                    // committing per iteration (before the brace) stalls the
                    // FPU sync protocol mid-accumulation (board wedge).
                    if k_loops.contains(&loop_id) {
                        writeln!(src, "{indent}tile_regs_commit();");
                        writeln!(src, "{indent}tile_regs_wait();");
                    }
                }
                Op::Barrier => {}
                Op::Range { axis, kind, .. } => match kind {
                    RangeKind::Group(_) => {
                        let arg = arg_pos.len() as u32 + axis;
                        writeln!(src, "{indent}uint32_t r{op_id} = get_arg_val<uint32_t>({arg});");
                    }
                    RangeKind::Local(_) => {
                        unreachable!(
                            "tenstorrent does not have local threads; local indices should have been converted to loops by the opt_tenstorrent_tile optimization pass"
                        )
                    }
                    RangeKind::Warp(_) => todo!("tenstorrent2 compute warp range"),
                },
                Op::Const(_)
                | Op::Binary { .. }
                | Op::Mad { .. }
                | Op::Cast { .. }
                | Op::Unary { .. }
                | Op::Bitcast { .. }
                | Op::Stack { .. }
                | Op::Index { .. }
                | Op::Asm { .. } => {
                    if fiction_adds.contains(&op_id) {
                        // Folded into the DST slot by matmul_tiles, emits nothing.
                    } else if compute_data.dtypes[&op_id].1 == MemLayout::Scalar {
                        let s = self.emit_scalar_op(kernel, op_id, &indent)?;
                        src.push_str(&s);
                    } else {
                        todo!("tenstorrent2 compute tiled ALU unimplemented");
                    }
                }
                Op::TransposeTile { .. } => todo!("tenstorrent2 compute TransposeTile unimplemented"),
                Op::Wmma { .. } => {
                    return Err(BackendError {
                        status: ErrorStatus::KernelCompilation,
                        context: "tenstorrent2: Wmma is a tensor-core op, not emittable here".into(),
                    });
                }
                Op::Move { .. } | Op::Reduce { .. } | Op::ReduceTile { .. } => {
                    unreachable!("tenstorrent2 compute closed over a removed op")
                }
                Op::If { .. } | Op::EndIf => todo!("tenstorrent2 compute branch"),
            }
        }
        // Function-scope accumulation releases after the walk.
        if acquire_top {
            writeln!(src, "{indent}tile_regs_release();");
        }
        writeln!(src, "}}");
        self.compute = TTKernel::Compute { src, params: params.to_vec(), ordinals: ordinals.to_vec() };
        Ok(())
    }

    /// Generate the writer (dataflow drain) kernel from the writer op list.
    #[allow(unused_must_use)]
    fn generate_writer(
        &mut self,
        kernel: &Kernel,
        writer_data: &SectionData,
        params: &[OpId],
        ordinals: &[u32],
    ) -> Result<(), BackendError> {
        let ops = &writer_data.ops;
        // Section params in list order become this section's runtime args.
        let mut arg_pos: Map<OpId, u32> = Map::default();
        for (i, &p) in params.iter().enumerate() {
            arg_pos.insert(p, i as u32);
        }
        let mut indent = String::from("  ");
        let mut prev_accessor: Option<String> = None;
        // CBs read (drained) by this section: waited upfront at loop
        // boundaries (see Loop arm).
        let mut writer_loop_cbs: Vec<CBId> = Vec::new();
        for &op_id in ops {
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
            writeln!(src, "{indent}CircularBuffer cb{cb}(tt::CBIndex::c_{cb});");
        }
        // Accessors only for the GlobalMut params this section writes.
        for &op_id in ops {
            if let Op::Param { kind: ParamKind::GlobalMut, .. } = kernel.ops[op_id].op {
                let arg = arg_pos.get(&op_id).copied().expect("tenstorrent2 writer param missing from section args");
                writeln!(src, "{indent}uint32_t out{op_id} = get_arg_val<uint32_t>({arg});");
                let cta = match &prev_accessor {
                    None => String::from("0"),
                    Some(prev) => format!("{prev}.next_compile_time_args_offset()"),
                };
                writeln!(src, "{indent}auto args_out{op_id} = TensorAccessorArgs<{cta}>({arg});");
                writeln!(src, "{indent}auto p_out{op_id} = TensorAccessor(args_out{op_id}, out{op_id}, {TT_DRAM_PAGE_BYTES});");
                prev_accessor = Some(format!("args_out{op_id}"));
            }
        }
        let mut loop_depth = 0u32;
        let mut loop_popped: Set<CBId> = Set::default();
        let n = ops.len();
        for i in 0..n {
            let op_id = ops[i];
            match kernel.ops[op_id].op {
                Op::Param { kind: ParamKind::GlobalMut, .. } => {
                    // Accessor emitted up front (see above).
                }
                Op::Param { kind: ParamKind::Variable, .. } => {
                    let arg = arg_pos.get(&op_id).copied().expect("tenstorrent2 writer param missing from section args");
                    let Op::Param { dtype, .. } = kernel.ops[op_id].op else {
                        unreachable!("tenstorrent2 param changed under us");
                    };
                    writeln!(src, "{indent}{} r{op_id} = ({})get_arg_val<uint32_t>({arg});", dtype.c_type(), dtype.c_type());
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
                            writeln!(src, "{indent}cb{cb}.wait_front(1);");
                            writeln!(
                                src,
                                "{indent}uint64_t wnoc{op_id} = p_out{dst}.get_noc_addr((uint32_t)((r{st_idx}*{elem_size})/{page_size}), (uint32_t)((r{st_idx}*{elem_size})%{page_size}));"
                            );
                            writeln!(src, "{indent}noc_async_write(cb{cb}.get_read_ptr(), wnoc{op_id}, {tile_bytes});");
                            writeln!(src, "{indent}noc_async_write_barrier();");
                            writeln!(src, "{indent}cb{cb}.pop_front(1);");
                            if loop_depth > 0 {
                                loop_popped.insert(cb);
                            }
                        }
                        _ => todo!("tenstorrent2 writer only supports tile stores"),
                    }
                }
                Op::Loop { len } => {
                    if loop_depth == 0 {
                        for cb in &writer_loop_cbs {
                            writeln!(src, "{indent}cb{cb}.wait_front(1);");
                            writeln!(src, "{indent}uint32_t wbase{cb} = cb{cb}.get_read_ptr();");
                        }
                    }
                    writeln!(src, "{indent}for (uint32_t r{op_id} = 0; r{op_id} < r{len}; r{op_id}++) {{");
                    indent.push_str("  ");
                    loop_depth += 1;
                }
                Op::EndLoop => {
                    indent.pop();
                    indent.pop();
                    writeln!(src, "{indent}}}");
                    if loop_depth == 1 {
                        writeln!(src, "{indent}noc_async_write_barrier();");
                        for cb in &writer_loop_cbs {
                            // Tile streaming pops per iteration (see the
                            // tile Store above); popping again here would
                            // over-pop and stall.
                            if loop_popped.contains(cb) {
                                continue;
                            }
                            writeln!(src, "{indent}cb{cb}.pop_front(1);");
                        }
                    }
                    loop_depth -= 1;
                }
                Op::Barrier => {}
                Op::Range { axis, kind, .. } => match kind {
                    RangeKind::Group(_) => {
                        let arg = arg_pos.len() as u32 + axis;
                        writeln!(src, "{indent}uint32_t r{op_id} = get_arg_val<uint32_t>({arg});");
                    }
                    RangeKind::Local(_) => {
                        unreachable!(
                            "tenstorrent does not have local threads; local indices should have been converted to loops by the opt_tenstorrent_tile optimization pass"
                        )
                    }
                    RangeKind::Warp(_) => todo!("tenstorrent2 writer warp range"),
                },
                Op::Const(_) | Op::Binary { .. } | Op::Mad { .. } | Op::Cast { .. } => {
                    let s = self.emit_scalar_op(kernel, op_id, &indent)?;
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
        self.writer = TTKernel::Writer { src, ordinals: ordinals.to_vec() };
        Ok(())
    }

    /// Shared scalar ALU emission across all sections: identical text
    /// everywhere, no section argument.
    #[allow(unused_must_use)]
    fn emit_scalar_op(&self, kernel: &Kernel, op_id: OpId, indent: &str) -> Result<String, BackendError> {
        let mut out = String::new();
        match &kernel.ops[op_id].op {
            Op::Const(val) => {
                writeln!(out, "{indent}{} r{op_id} = {};", val.dtype().c_type(), val.c_code());
            }
            Op::Binary { x, y, bop } => {
                let dt = kernel.dtype(op_id);
                let _ = match bop {
                    BOp::Add => writeln!(out, "{indent}{} r{op_id} = r{x} + r{y};", dt.c_type()),
                    BOp::Sub => writeln!(out, "{indent}{} r{op_id} = r{x} - r{y};", dt.c_type()),
                    BOp::Mul => writeln!(out, "{indent}{} r{op_id} = r{x} * r{y};", dt.c_type()),
                    BOp::Div => writeln!(out, "{indent}{} r{op_id} = r{x} / r{y};", dt.c_type()),
                    BOp::Mod => writeln!(out, "{indent}{} r{op_id} = r{x} % r{y};", dt.c_type()),
                    BOp::Max => writeln!(out, "{indent}{} r{op_id} = r{x} > r{y} ? r{x} : r{y};", dt.c_type()),
                    BOp::Cmplt => writeln!(out, "{indent}{} r{op_id} = r{x} < r{y};", dt.c_type()),
                    BOp::Cmpgt => writeln!(out, "{indent}{} r{op_id} = r{x} > r{y};", dt.c_type()),
                    BOp::Cmpge => writeln!(out, "{indent}{} r{op_id} = r{x} >= r{y};", dt.c_type()),
                    BOp::Eq => writeln!(out, "{indent}{} r{op_id} = r{x} == r{y};", dt.c_type()),
                    BOp::NotEq => writeln!(out, "{indent}{} r{op_id} = r{x} != r{y};", dt.c_type()),
                    BOp::And => writeln!(out, "{indent}{} r{op_id} = r{x} && r{y};", dt.c_type()),
                    BOp::Or => writeln!(out, "{indent}{} r{op_id} = r{x} || r{y};", dt.c_type()),
                    BOp::BitXor => writeln!(out, "{indent}{} r{op_id} = r{x} ^ r{y};", dt.c_type()),
                    BOp::BitOr => writeln!(out, "{indent}{} r{op_id} = r{x} | r{y};", dt.c_type()),
                    BOp::BitAnd => writeln!(out, "{indent}{} r{op_id} = r{x} & r{y};", dt.c_type()),
                    BOp::BitShiftLeft => writeln!(out, "{indent}{} r{op_id} = r{x} << r{y};", dt.c_type()),
                    BOp::BitShiftRight => writeln!(out, "{indent}{} r{op_id} = r{x} >> r{y};", dt.c_type()),
                    BOp::Pow => todo!("tenstorrent2 scalar pow"),
                };
            }
            Op::Mad { x, y, z } => {
                let dt = kernel.dtype(op_id);
                writeln!(out, "{indent}{} r{op_id} = r{x} * r{y} + r{z};", dt.c_type());
            }
            Op::Cast { x, dtype } => {
                writeln!(out, "{indent}{} r{op_id} = ({})r{x};", dtype.c_type(), dtype.c_type());
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

    /// Tiled compute emission (compute only): one tiled op to its C++
    /// text, given its DST slot. CB ids, slot assignment, traffic (CB
    /// handshake, pack/unpack), inits, and loop scaffolding live in
    /// `generate_compute`, not here.
    #[allow(unused_must_use)]
    fn emit_tiled_op(&self, kernel: &Kernel, op_id: OpId, indent: &str, slot: u32) -> Result<String, BackendError> {
        let mut out = String::new();
        match kernel.ops[op_id].op {
            Op::MatmulTile { x, y } => {
                let Op::Load { src: a_src, .. } = kernel.ops[x].op else {
                    todo!("tenstorrent2 MatmulTile x not loaded from a CB");
                };
                let Op::Load { src: b_src, .. } = kernel.ops[y].op else {
                    todo!("tenstorrent2 MatmulTile y not loaded from a CB");
                };
                let Some(&cb_a) = self.cb_map.get(&a_src) else {
                    unreachable!("tenstorrent2 MatmulTile input targets unmapped CB");
                };
                let Some(&cb_b) = self.cb_map.get(&b_src) else {
                    unreachable!("tenstorrent2 MatmulTile input targets unmapped CB");
                };
                writeln!(out, "{indent}matmul_tiles(cb{cb_a}, cb{cb_b}, {slot}, 0, 0);");
            }
            Op::TransposeTile { .. } => {
                todo!("tenstorrent2 TransposeTile emission unimplemented");
            }
            Op::ReduceTile { .. } => {
                todo!("tenstorrent2 ReduceTile emission unimplemented");
            }
            Op::Wmma { .. } => {
                return Err(BackendError {
                    status: ErrorStatus::KernelCompilation,
                    context: "tenstorrent2: Wmma is a tensor-core op, not emittable here".into(),
                });
            }
            Op::Const(_) => unreachable!("tenstorrent2 emit_tiled_op called with Const"),
            Op::Param { .. } => unreachable!("tenstorrent2 emit_tiled_op called with Param"),
            Op::Cast { .. } => todo!("tenstorrent2 tiled ALU emission unimplemented"),
            Op::Bitcast { .. } => todo!("tenstorrent2 tiled ALU emission unimplemented"),
            Op::Unary { .. } => todo!("tenstorrent2 tiled ALU emission unimplemented"),
            Op::Binary { .. } => todo!("tenstorrent2 tiled ALU emission unimplemented"),
            Op::Stack { .. } => todo!("tenstorrent2 tiled emission unimplemented"),
            Op::Storage { .. } => unreachable!("tenstorrent2 emit_tiled_op called with Storage"),
            Op::Store { .. } => unreachable!("tenstorrent2 emit_tiled_op called with Store"),
            Op::Load { .. } => unreachable!("tenstorrent2 emit_tiled_op called with Load"),
            Op::Range { .. } => unreachable!("tenstorrent2 emit_tiled_op called with Range"),
            Op::Loop { .. } => unreachable!("tenstorrent2 emit_tiled_op called with Loop"),
            Op::EndLoop => unreachable!("tenstorrent2 emit_tiled_op called with EndLoop"),
            Op::If { .. } => unreachable!("tenstorrent2 emit_tiled_op called with If"),
            Op::EndIf => unreachable!("tenstorrent2 emit_tiled_op called with EndIf"),
            Op::Mad { .. } => todo!("tenstorrent2 tiled ALU emission unimplemented"),
            Op::Index { .. } => todo!("tenstorrent2 tiled emission unimplemented"),
            Op::Barrier => unreachable!("tenstorrent2 emit_tiled_op called with Barrier"),
            Op::Asm { .. } => todo!("tenstorrent2 tiled emission unimplemented"),
            Op::Move { .. } => todo!("tenstorrent2 tiled emission unimplemented"),
            Op::Reduce { .. } => todo!("tenstorrent2 tiled emission unimplemented"),
        }
        Ok(out)
    }
}
