use crate::{
    DType, Map, Set,
    error::{BackendError, ErrorStatus},
    kernel::{BOp, IDX_T, Kernel, MMADType, MemLayout, MemScope, Op, OpId, ParamKind, RangeKind, UOp},
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
    pub(crate) fn generate_tenstorrent(&self) -> Result<Compiler, BackendError> {
        let mut compiler = Compiler::new(self);
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

        compiler.check_ir_tt(self)?;
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
                // Every listed op carries a section refcount, even when
                // nothing consumes it (zero uses). A missing entry
                // downstream is a phase-3 bug, never a default.
                rcs.entry(op_id).or_insert(0);
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

/// Shared per-section emission state: the kernel under codegen, the
/// output text, the indent and loop level, the section params, the
/// unified register file, and the tile-op init sets the walk collects
/// for the pre-loop hoist. One lives in each section generator,
/// initialized the same way in all three; `emit_op` is a method on it.
struct OpEmitter<'a> {
    /// Kernel under codegen, shared-borrowed for the section's life.
    kernel: &'a Kernel,
    /// Generated kernel source for this section.
    src: String,
    /// Current brace indent inside `kernel_main`.
    indent: String,
    /// Brace scope depth of the emission cursor: loops and ifs alike.
    scope_level: u8,
    /// Section params in list order become this section's runtime args.
    arg_pos: Map<OpId, u32>,
    /// Unified register file: scalars and DST slots alike, told apart
    /// by the `MemLayout` on each entry.
    var_map: Map<OpId, u32>,
    /// One entry per emitted value, in emission order.
    vars: Vec<VarSlot>,
    /// Tile unary ops seen, for the hoisted init block.
    unary_inits: Set<UOp>,
    /// Tile binary ops seen, for the hoisted init block.
    binary_inits: Set<BOp>,
    /// Tile cast format pairs seen, for the hoisted init block.
    typecast_inits: Vec<(u32, u32)>,
}

impl<'a> OpEmitter<'a> {
    /// Empty emitter over `kernel`: blank source, base indent, top
    /// level, no params, no registers, no inits. Each section
    /// generator fills in its own params right after.
    fn new(kernel: &'a Kernel) -> Self {
        Self {
            kernel,
            src: String::new(),
            indent: String::from("  "),
            scope_level: 0,
            arg_pos: Map::default(),
            var_map: Map::default(),
            vars: Vec::new(),
            unary_inits: Set::default(),
            binary_inits: Set::default(),
            typecast_inits: Vec::new(),
        }
    }

    /// Common pre-loop init emission for compute tile ops (v1-proven
    /// hoist): one `*_init` per collected op kind, inserted at `pos` —
    /// the anchor the caller records right after `init_sfpu`, ahead of
    /// all loops. Per-iteration init reprograms live packer state. Copy
    /// inits stay per-load (unpack config is per-CB, emitted at the
    /// load). The walk inserts into the sets as it emits; this method
    /// only formats. Always at function-scope indent: the anchor sits
    /// at base indent by construction.
    fn prepend_compute_inits(&mut self, pos: usize) {
        let indent = "  ";
        let mut inits = String::new();
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
                UOp::Not => todo!("tenstorrent tiled logical not"),
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
        self.src.insert_str(pos, &inits);
    }
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

    /// Closed-matmul shape gate: compute holds exactly one accumulation
    /// cone per output — acc-fold adds, acc loads/stores, one pack per
    /// output. Matmul is fixed 32x32 F16xF16. Every violation is a loud
    /// compilation error; SFPU coexistence especially so (see caller).
    fn check_matmul_closed(&self, kernel: &Kernel) -> Result<(), BackendError> {
        // Values stored to scratch in compute: fold candidates
        // (value op -> dst storage op).
        let mut scratch_vals: Map<OpId, OpId> = Map::default();
        let mut section = TtSection::Reader;
        let mut scan = kernel.head;
        for _ in 0..10_000 {
            if scan.is_null() {
                break;
            }
            match kernel.ops[scan].op {
                Op::Barrier => section.advance(),
                Op::Store { dst, src, .. } => {
                    if section == TtSection::Compute {
                        if let Some(&cb) = self.cb_map.get(&dst) {
                            if self.scratch_cbs.contains(&cb) {
                                scratch_vals.insert(src, dst);
                            }
                        }
                    }
                }
                _ => {}
            }
            scan = kernel.next_op(scan);
        }
        if !scan.is_null() {
            panic!("tenstorrent2 matmul scratch scan did not finish in 10000 steps");
        }
        let mut folds: Set<OpId> = Set::default();
        let mut chained: Set<OpId> = Set::default();
        section = TtSection::Reader;
        scan = kernel.head;
        for _ in 0..10_000 {
            if scan.is_null() {
                break;
            }
            match kernel.ops[scan].op {
                Op::Barrier => section.advance(),
                Op::MatmulTile { x, y } if section == TtSection::Compute => {
                    for &v in &[x, y] {
                        let Op::Load { src: lsrc, layout: MemLayout::Tile { x: w, y: h, .. }, .. } = kernel.ops[v].op
                        else {
                            return Err(BackendError {
                                status: ErrorStatus::KernelCompilation,
                                context: format!("tenstorrent2: matmul side op {v} is no CB tile load").into(),
                            });
                        };
                        if w as u32 != 32 || h as u32 != 32 {
                            return Err(BackendError {
                                status: ErrorStatus::KernelCompilation,
                                context: format!("tenstorrent2: matmul is fixed 32x32, op {v} is {w}x{h}").into(),
                            });
                        }
                        let Some(&cb) = self.cb_map.get(&lsrc) else {
                            return Err(BackendError {
                                status: ErrorStatus::KernelCompilation,
                                context: format!("tenstorrent2: matmul load op {v} targets unmapped CB").into(),
                            });
                        };
                        if self.scratch_cbs.contains(&cb) {
                            return Err(BackendError {
                                status: ErrorStatus::KernelCompilation,
                                context: format!("tenstorrent2: matmul side op {v} reads scratch").into(),
                            });
                        }
                        let Op::Storage { dtype: DType::F16, .. } = kernel.ops[lsrc].op else {
                            return Err(BackendError {
                                status: ErrorStatus::KernelCompilation,
                                context: format!("tenstorrent2: matmul inputs are F16, op {v} is not").into(),
                            });
                        };
                    }
                }
                Op::MatmulTile { .. } => {
                    return Err(BackendError {
                        status: ErrorStatus::KernelCompilation,
                        context: format!("tenstorrent2: matmul lives in compute, op {scan} is elsewhere").into(),
                    });
                }
                Op::Binary { x, y, bop } if section == TtSection::Compute => {
                    if !matches!(bop, BOp::Add) {
                        return Err(BackendError {
                            status: ErrorStatus::KernelCompilation,
                            context: format!("tenstorrent2: sfpu and matmul cannot share a kernel (op {scan})").into(),
                        });
                    }
                    // Fold sides: one acc load, one matmul, either order.
                    let mut acc = None;
                    let mut mm = None;
                    for &v in &[x, y] {
                        if matches!(kernel.ops[v].op, Op::MatmulTile { .. }) {
                            mm = Some(v);
                        } else if let Op::Load { src: lsrc, layout: MemLayout::Tile { .. }, .. } = kernel.ops[v].op {
                            match self.cb_map.get(&lsrc) {
                                Some(&cb) if self.scratch_cbs.contains(&cb) => acc = Some(lsrc),
                                _ => {
                                    return Err(BackendError {
                                        status: ErrorStatus::KernelCompilation,
                                        context: format!(
                                            "tenstorrent2: fold side op {v} is no acc tile load"
                                        )
                                        .into(),
                                    });
                                }
                            }
                        } else {
                            return Err(BackendError {
                                status: ErrorStatus::KernelCompilation,
                                context: format!("tenstorrent2: fold side op {v} is no acc/matmul tile").into(),
                            });
                        }
                    }
                    let (Some(s), Some(_)) = (acc, mm) else {
                        return Err(BackendError {
                            status: ErrorStatus::KernelCompilation,
                            context: format!("tenstorrent2: fold adds acc and matmul only, op {scan} does not")
                                .into(),
                        });
                    };
                    match scratch_vals.get(&scan) {
                        Some(&d) if d == s => {
                            folds.insert(scan);
                            chained.insert(s);
                        }
                        _ => {
                            return Err(BackendError {
                                status: ErrorStatus::KernelCompilation,
                                context: format!(
                                    "tenstorrent2: fold result op {scan} is not stored to its acc"
                                )
                                .into(),
                            });
                        }
                    }
                }
                Op::Store { dst, src, layout, .. } if section == TtSection::Compute => {
                    if !matches!(layout, MemLayout::Tile { .. }) {
                        return Err(BackendError {
                            status: ErrorStatus::KernelCompilation,
                            context: format!("tenstorrent2 compute only supports tile stores, op {scan}").into(),
                        });
                    }
                    let Some(&cb) = self.cb_map.get(&dst) else {
                        return Err(BackendError {
                            status: ErrorStatus::KernelCompilation,
                            context: format!("tenstorrent2: compute store op {scan} targets unmapped CB").into(),
                        });
                    };
                    if self.scratch_cbs.contains(&cb) {
                        if !folds.contains(&src) {
                            return Err(BackendError {
                                status: ErrorStatus::KernelCompilation,
                                context: format!(
                                    "tenstorrent2: compute scratch store op {scan} is not a validated fold"
                                )
                                .into(),
                            });
                        }
                    } else {
                        let Op::Load { src: lsrc, layout: MemLayout::Tile { .. }, .. } = kernel.ops[src].op
                        else {
                            return Err(BackendError {
                                status: ErrorStatus::KernelCompilation,
                                context: format!(
                                    "tenstorrent2: compute pack store op {scan} reads no acc tile"
                                )
                                .into(),
                            });
                        };
                        if !chained.contains(&lsrc) {
                            return Err(BackendError {
                                status: ErrorStatus::KernelCompilation,
                                context: format!(
                                    "tenstorrent2: compute pack store op {scan} packs an empty acc chain"
                                )
                                .into(),
                            });
                        }
                    }
                }
                Op::Load { layout, .. } if section == TtSection::Compute => {
                    if !matches!(layout, MemLayout::Tile { .. }) {
                        return Err(BackendError {
                            status: ErrorStatus::KernelCompilation,
                            context: format!("tenstorrent2 compute only supports tile loads, op {scan}").into(),
                        });
                    }
                    // Role (matmul input vs acc) is checked at consumers.
                }
                Op::Unary { .. } | Op::Cast { .. } | Op::Bitcast { .. } | Op::Mad { .. }
                    if section == TtSection::Compute =>
                {
                    return Err(BackendError {
                        status: ErrorStatus::KernelCompilation,
                        context: format!("tenstorrent2: sfpu and matmul cannot share a kernel (op {scan})").into(),
                    });
                }
                Op::Range { .. }
                | Op::Stack { .. }
                | Op::Index { .. }
                | Op::If { .. }
                | Op::EndIf
                | Op::TransposeTile { .. }
                | Op::ReduceTile { .. }
                | Op::Asm { .. }
                    if section == TtSection::Compute =>
                {
                    return Err(BackendError {
                        status: ErrorStatus::KernelCompilation,
                        context: format!("tenstorrent2: op {scan} is outside the closed matmul shape").into(),
                    });
                }
                _ => {}
            }
            scan = kernel.next_op(scan);
        }
        if !scan.is_null() {
            panic!("tenstorrent2 matmul check scan did not finish in 10000 steps");
        }
        Ok(())
    }

    /// Single IR validity gate for the TT path: section count, reader /
    /// writer store shapes, loop and group-index lengths, then CB
    /// push/pop balance. Everything here reads the input IR; nothing
    /// inspects emitted C++ text.
    fn check_ir_tt(&self, kernel: &Kernel) -> Result<(), BackendError> {
        self.check_sections(kernel)?;
        // Closed-matmul shape first: with a MatmulTile in the kernel,
        // compute holds exactly one accumulation cone per output. Any
        // SFPU sharing the kernel is rejected here (loud error, never
        // silent): mm_init plus SFPU inits in one kernel hung the board
        // during v1 bring-up, so the combination never reaches codegen.
        let mut is_mm = false;
        let mut pre = kernel.head;
        for _ in 0..10_000 {
            if pre.is_null() {
                break;
            }
            if matches!(kernel.ops[pre].op, Op::MatmulTile { .. }) {
                is_mm = true;
                break;
            }
            pre = kernel.next_op(pre);
        }
        if is_mm {
            self.check_matmul_closed(kernel)?;
        }
        let mut section = TtSection::Reader;
        let mut scan = kernel.head;
        for _ in 0..10_000 {
            if scan.is_null() {
                break;
            }
            match kernel.ops[scan].op {
                Op::Barrier => section.advance(),
                Op::Loop { len } => {
                    let Some(dim) = kernel.resolve_const(len).and_then(|c| c.as_dim()) else {
                        return Err(BackendError {
                            status: ErrorStatus::KernelCompilation,
                            context: format!("tenstorrent2: loop trip count op {len} is not resolvable").into(),
                        });
                    };
                    debug_assert!(dim >= 0, "tenstorrent2: negative loop length");
                    if dim < 0 {
                        return Err(BackendError {
                            status: ErrorStatus::KernelCompilation,
                            context: format!("tenstorrent2: loop length {dim} is negative").into(),
                        });
                    }
                }
                Op::Range { kind, .. } => match kind {
                    RangeKind::Group(len) | RangeKind::Warp(len) => {
                        let Some(dim) = kernel.resolve_const(len).and_then(|c| c.as_dim()) else {
                            return Err(BackendError {
                                status: ErrorStatus::KernelCompilation,
                                context: format!("tenstorrent2: group index length op {len} is not resolvable").into(),
                            });
                        };
                        debug_assert!(dim >= 0, "tenstorrent2: negative group length");
                        if dim < 0 {
                            return Err(BackendError {
                                status: ErrorStatus::KernelCompilation,
                                context: format!("tenstorrent2: group length {dim} is negative").into(),
                            });
                        }
                    }
                    RangeKind::Local(_) => {}
                },
                Op::Store { dst, src, layout: MemLayout::Tile { .. }, .. } => match section {
                    TtSection::Reader => {
                        let Op::Load { src: ld_src, layout: MemLayout::Tile { .. }, .. } = kernel.ops[src].op else {
                            return Err(BackendError {
                                status: ErrorStatus::KernelCompilation,
                                context: format!(
                                    "tenstorrent2: reader supports only global to CB tile stores, op {scan} has ops in between"
                                )
                                .into(),
                            });
                        };
                        if !matches!(kernel.ops[ld_src].op, Op::Param { kind: ParamKind::Global, .. }) {
                            return Err(BackendError {
                                status: ErrorStatus::KernelCompilation,
                                context: format!("tenstorrent2: reader tile load op {scan} is not from a Global param").into(),
                            });
                        }
                        if !matches!(kernel.ops[dst].op, Op::Storage { scope: MemScope::Circular, .. }) {
                            return Err(BackendError {
                                status: ErrorStatus::KernelCompilation,
                                context: format!("tenstorrent2: reader tile store op {scan} does not target a Circular CB")
                                    .into(),
                            });
                        }
                    }
                    TtSection::Writer => {
                        let Op::Load { src: cb_src, layout: MemLayout::Tile { .. }, .. } = kernel.ops[src].op else {
                            return Err(BackendError {
                                status: ErrorStatus::KernelCompilation,
                                context: format!(
                                    "tenstorrent2: writer supports only CB to DRAM tile stores, op {scan} has ops in between"
                                )
                                .into(),
                            });
                        };
                        if !self.cb_map.contains_key(&cb_src) {
                            return Err(BackendError {
                                status: ErrorStatus::KernelCompilation,
                                context: format!("tenstorrent2: writer tile load op {scan} targets unmapped CB").into(),
                            });
                        }
                        if !matches!(kernel.ops[dst].op, Op::Param { kind: ParamKind::GlobalMut, .. }) {
                            return Err(BackendError {
                                status: ErrorStatus::KernelCompilation,
                                context: format!("tenstorrent2: writer tile store op {scan} dst must be a GlobalMut param")
                                    .into(),
                            });
                        }
                    }
                    TtSection::Compute => {}
                },
                _ => {}
            }
            scan = kernel.next_op(scan);
        }
        if !scan.is_null() {
            panic!("tenstorrent2 ir check scan did not finish in 10000 steps");
        }
        self.check_balance(kernel)
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
                    let Some(dim) = kernel.resolve_const(*len).and_then(|c| c.as_dim()) else {
                        return Err(BackendError {
                            status: ErrorStatus::KernelCompilation,
                            context: format!("tenstorrent2: loop trip count op {len} is not resolvable").into(),
                        });
                    };
                    trips.push(dim);
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
            // Scratch accumulators (compute-stored, never writer-read)
            // drain through DST, not CB traffic: exempt from push/pop
            // equality (matmul acc seed + folds emit nothing).
            if self.scratch_cbs.contains(&cb) {
                continue;
            }
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
        let mut em = OpEmitter::new(kernel);
        // Section params in list order become this section's runtime args.
        for (i, &p) in params.iter().enumerate() {
            em.arg_pos.insert(p, i as u32);
        }
        let mut prev_accessor: Option<String> = None;
        // Owned working copy; re-synced from the emitter after every
        // mutation below. Never held across `em` method calls.
        let mut indent = em.indent.clone();
        writeln!(em.src, "#include <cstdint>");
        writeln!(em.src, "#include \"api/dataflow/dataflow_api.h\"");
        writeln!(em.src, "#include \"api/dataflow/noc.h\"");
        writeln!(em.src, "#include \"api/dataflow/circular_buffer.h\"");
        writeln!(em.src, "#include \"api/tensor/noc_traits.h\"");
        writeln!(em.src, "#include \"api/debug/device_print.h\"");
        writeln!(em.src, "void kernel_main() {{");
        // Every shared CB is declared (matches v1 output); the section
        // only pushes the ones it fills.
        let mut cbs: Vec<CBId> = self.cb_map.iter().map(|(_, &cb)| cb).collect();
        cbs.sort();
        for cb in cbs {
            writeln!(em.src, "{indent}CircularBuffer cb{cb}(tt::CBIndex::c_{cb});");
        }
        let n = ops.len();
        for i in 0..n {
            let op_id = ops[i];
            match kernel.ops[op_id].op {
                Op::Param { dtype, kind, .. } => match kind {
                    ParamKind::Global => {
                        let arg = em.arg_pos.get(&op_id).copied().expect("tenstorrent2 reader param missing from section args");
                        writeln!(em.src, "{indent}uint32_t src{op_id} = get_arg_val<uint32_t>({arg});");
                        let cta = match &prev_accessor {
                            None => String::from("0"),
                            Some(prev) => format!("{prev}.next_compile_time_args_offset()"),
                        };
                        writeln!(em.src, "{indent}auto args{op_id} = TensorAccessorArgs<{cta}>({arg});");
                        writeln!(em.src, "{indent}auto p{op_id} = TensorAccessor(args{op_id}, src{op_id}, {TT_DRAM_PAGE_BYTES});");
                        prev_accessor = Some(format!("args{op_id}"));
                    }
                    ParamKind::Variable => {
                        let arg = em.arg_pos.get(&op_id).copied().expect("tenstorrent2 reader param missing from section args");
                        writeln!(em.src, "{indent}{} r{op_id} = ({})get_arg_val<uint32_t>({arg});", dtype.c_type(), dtype.c_type());
                    }
                    ParamKind::GlobalMut => {
                        let arg = em.arg_pos.get(&op_id).copied().expect("tenstorrent2 reader param missing from section args");
                        writeln!(em.src, "{indent}uint32_t dst{op_id} = get_arg_val<uint32_t>({arg});");
                        let cta = match &prev_accessor {
                            None => String::from("0"),
                            Some(prev) => format!("{prev}.next_compile_time_args_offset()"),
                        };
                        writeln!(em.src, "{indent}auto args{op_id} = TensorAccessorArgs<{cta}>({arg});");
                        writeln!(em.src, "{indent}auto p{op_id} = TensorAccessor(args{op_id}, dst{op_id}, {TT_DRAM_PAGE_BYTES});");
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
                                // Index resolves through the section register map:
                                // consts inline, Variable params keep r{op_id},
                                // every other value must already sit in var_map.
                                let idx = if let Op::Const(c) = &kernel.ops[ld_idx].op {
                                    format!("{}", c.c_code())
                                } else if matches!(kernel.ops[ld_idx].op, Op::Param { kind: ParamKind::Variable, .. }) {
                                    format!("r{ld_idx}")
                                } else if let Some(&r) = em.var_map.get(&ld_idx) {
                                    if em.vars[r as usize].scope_level == em.scope_level {
                                        debug_assert!(em.vars[r as usize].rc > 0);
                                        em.vars[r as usize].rc -= 1;
                                    }
                                    format!("r{r}")
                                } else {
                                    return Err(BackendError {
                                        status: ErrorStatus::KernelCompilation,
                                        context: format!("tenstorrent2: reader index {ld_idx} not in registers").into(),
                                    });
                                };
                                writeln!(em.src, "{indent}cb{cb}.reserve_back(1);");
                                writeln!(
                                    em.src,
                                    "{indent}uint64_t rnoc{op_id} = p{ld_src}.get_noc_addr((uint32_t)(({idx}*{elem_size})/{page_size}), (uint32_t)(({idx}*{elem_size})%{page_size}));"
                                );
                                writeln!(em.src, "{indent}noc_async_read(rnoc{op_id}, cb{cb}.get_write_ptr(), {tile_bytes});");
                                writeln!(em.src, "{indent}noc_async_read_barrier();");
                                writeln!(em.src, "{indent}cb{cb}.push_back(1);");
                            }
                            _ => todo!("tenstorrent2 reader only supports tile stores"),
                        }
                    }
                }
                Op::Loop { .. } => {
                    em.emit_op(op_id, &reader_data)?;
                    em.indent.push_str("  ");
                    indent = em.indent.clone();
                    em.scope_level += 1;
                }
                Op::EndLoop => {
                    em.scope_level -= 1;
                    em.indent.pop();
                    em.indent.pop();
                    indent = em.indent.clone();
                    writeln!(em.src, "{indent}}}");
                }
                Op::If { condition } => {
                    // Condition is a prior boolean scalar: consts inline,
                    // registers resolve through the section map.
                    let cond = if let Op::Const(c) = &kernel.ops[condition].op {
                        format!("{}", c.c_code())
                    } else if let Some(&r) = em.var_map.get(&condition) {
                        format!("r{r}")
                    } else {
                        return Err(BackendError {
                            status: ErrorStatus::KernelCompilation,
                            context: format!(
                                "tenstorrent2: if condition op {condition} not in registers"
                            )
                            .into(),
                        });
                    };
                    writeln!(em.src, "{indent}if ({cond}) {{");
                    em.indent.push_str("  ");
                    indent = em.indent.clone();
                    em.scope_level += 1;
                }
                Op::EndIf => {
                    em.scope_level -= 1;
                    em.indent.pop();
                    em.indent.pop();
                    indent = em.indent.clone();
                    writeln!(em.src, "{indent}}}");
                }
                Op::Barrier => {}
                Op::Range { .. } => {
                    em.emit_op(op_id, &reader_data)?;
                }
                Op::Const(_) => {
                    // Inlined as literals at uses; no declaration emitted.
                }
                Op::Binary { .. } | Op::Mad { .. } | Op::Cast { .. } => {
                    em.emit_op(op_id, &reader_data)?;
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
        writeln!(em.src, "{indent}noc_async_read_barrier();");
        writeln!(em.src, "}}");
        self.reader = TTKernel::Reader { src: em.src, ordinals: ordinals.to_vec() };
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
        let mut em = OpEmitter::new(kernel);
        // Section params in list order become this section's runtime args.
        for (i, &p) in params.iter().enumerate() {
            em.arg_pos.insert(p, i as u32);
        }
        // Owned working copy; re-synced from the emitter after every
        // mutation below. Never held across `em` method calls.
        let mut indent = em.indent.clone();
        writeln!(em.src, "#include <cstdint>");
        writeln!(em.src, "#include \"api/compute/common.h\"");
        writeln!(em.src, "#include \"api/compute/compute_kernel_api.h\"");
        writeln!(em.src, "#include \"api/compute/eltwise_binary_sfpu.h\"");
        writeln!(em.src, "#include \"api/compute/tile_move_copy.h\"");
        writeln!(em.src, "#include \"api/compute/eltwise_unary/eltwise_unary.h\"");
        writeln!(em.src, "#include \"api/compute/eltwise_unary/trigonometry.h\"");
        writeln!(em.src, "#include \"api/compute/eltwise_unary/exp.h\"");
        writeln!(em.src, "#include \"api/compute/eltwise_unary/recip.h\"");
        writeln!(em.src, "#include \"api/compute/eltwise_unary/sqrt.h\"");
        writeln!(em.src, "#include \"api/compute/eltwise_unary/rounding.h\"");
        writeln!(em.src, "#include \"api/compute/eltwise_unary/negative.h\"");
        writeln!(em.src, "#include \"api/compute/eltwise_unary/bitwise_not.h\"");
        writeln!(em.src, "#include \"api/compute/eltwise_unary/typecast.h\"");
        writeln!(em.src, "#include \"api/compute/eltwise_unary/fill.h\"");
        writeln!(em.src, "#include \"api/compute/matmul.h\"");
        writeln!(em.src, "#include \"api/compute/reconfig_data_format.h\"");
        writeln!(em.src, "#include \"api/dataflow/circular_buffer.h\"");
        writeln!(em.src, "#include \"api/debug/device_print.h\"");
        writeln!(em.src, "void kernel_main() {{");
        // Every shared CB is declared (matches reader/writer); the section
        // only consumes the ones it reads.
        let mut cbs: Vec<CBId> = self.cb_map.iter().map(|(_, &cb)| cb).collect();
        cbs.sort();
        for cb in cbs {
            writeln!(em.src, "{indent}CircularBuffer cb{cb}(tt::CBIndex::c_{cb});");
        }
        // Input CBs in first-touch order: waited per iteration at the
        // outermost loop (official streaming handshake).
        let mut compute_input_cbs: Vec<CBId> = Vec::new();
        for &op_id in &compute_data.ops {
            if let Op::Load { src, layout: MemLayout::Tile { .. }, .. } = kernel.ops[op_id].op {
                if let Some(&cb) = self.cb_map.get(&src) {
                    if !compute_input_cbs.contains(&cb) {
                        compute_input_cbs.push(cb);
                    }
                }
            }
        }
        // First output CB in section order (pack target for init_sfpu).
        let mut compute_out_cb: Option<CBId> = None;
        for &op_id in &compute_data.ops {
            if let Op::Store { dst, layout: MemLayout::Tile { .. }, .. } = kernel.ops[op_id].op {
                if let Some(&cb) = self.cb_map.get(&dst) {
                    compute_out_cb = Some(cb);
                    break;
                }
            }
        }
        // Matmul kernels take the closed path (check_matmul_closed
        // already rejected anything beyond folds + one pack per output):
        // lazy tags, no copies, mm_init up front. Anything else keeps
        // the streaming sfpu path below, byte-identical.
        let is_mm = compute_data
            .ops
            .iter()
            .any(|&op_id| matches!(kernel.ops[op_id].op, Op::MatmulTile { .. }));
        // Unpacker init (v1-proven): the only unpack config on non-matmul
        // kernels. First compute tile-Load CB in, first tile-Store CB out.
        // Empty compute sections (pure movement) emit nothing. Matmul
        // kernels get mm_init at the anchor after the walk instead
        // (triples are complete only then); init_sfpu coexistence is
        // rejected by the IR check.
        if !is_mm {
            if let (Some(&in0), Some(out0)) = (compute_input_cbs.first(), compute_out_cb) {
                writeln!(em.src, "{indent}init_sfpu({in0}, {out0});");
            }
        }
        // Anchor for the hoisted tile-op inits: the common method
        // prepends them here, ahead of all loops, after the walk.
        let init_anchor = em.src.len();
        // Pack-scope acquire: loop-depths of non-scratch tile stores
        // (Ifs don't count: acquiring at the enclosing loop still
        // dominates a pack inside a branch). Acquire/release pairs with
        // the pack scope — one output tile's live range — never the
        // outermost loop (Mt>1 breaks hoisting). Multiple depths would
        // nest acquires (illegal DST): loud halt.
        let mut pack_depths: Set<u32> = Set::default();
        {
            let mut depth: u32 = 0;
            for &op_id in &compute_data.ops {
                match kernel.ops[op_id].op {
                    Op::Loop { .. } => depth += 1,
                    Op::EndLoop => depth -= 1,
                    Op::Store { ref dst, layout: MemLayout::Tile { .. }, .. } => {
                        if let Some(&cb) = self.cb_map.get(dst) {
                            if !self.scratch_cbs.contains(&cb) {
                                pack_depths.insert(depth);
                            }
                        }
                    }
                    _ => {}
                }
            }
        }
        if pack_depths.len() > 1 {
            return Err(BackendError {
                status: ErrorStatus::KernelCompilation,
                context: "tenstorrent2: packs at multiple loop depths would nest DST acquires".into(),
            });
        }
        // Degenerate pack scope (pack with no enclosing loop): acquire
        // up front, release before the close.
        if pack_depths.contains(&0) {
            writeln!(em.src, "{indent}tile_regs_acquire();");
        }
        // Per-loop output pushes (recorded at stores, emitted at EndLoop),
        // mirroring the proven v1 tail.
        let mut loop_pushes: Vec<Vec<CBId>> = Vec::new();
        // Loops-only depth (Ifs share scope_level but never own DST).
        let mut loop_depth: u32 = 0;
        // Closed-matmul state (empty on sfpu kernels): validated fold
        // binaries (binary -> acc storage, matmul), matmul input CB
        // pairs, matmul-fed CBs (popped inline, never blanket), one DST
        // slot per acc chain, distinct mm_init input pairs, and the pack
        // target (last circular store, v1 mm_out_cb rule).
        let mut folds: Map<OpId, (OpId, OpId)> = Map::default();
        let mut matmul_tags: Map<OpId, (CBId, CBId)> = Map::default();
        let mut fold_cbs: Set<CBId> = Set::default();
        let mut chain_slot: Map<OpId, u32> = Map::default();
        let mut mm_pairs: Vec<(CBId, CBId)> = Vec::new();
        let mut mm_out: Option<CBId> = None;

        for &op_id in &compute_data.ops {
            match kernel.ops[op_id].op {
                Op::Param { dtype, kind, .. } => match kind {
                    ParamKind::Variable => {
                        let arg = em.arg_pos.get(&op_id).copied().expect("tenstorrent2 compute param missing from section args");
                        writeln!(em.src, "{indent}{} r{op_id} = ({})get_arg_val<uint32_t>({arg});", dtype.c_type(), dtype.c_type());
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
                Op::Cast { .. } | Op::Bitcast { .. } | Op::Unary { .. } | Op::Binary { .. } | Op::Mad { .. } => {
                    if is_mm {
                        // Closed fold only: add(acc-load, matmul) records
                        // the chain slot and emits nothing; the acc store
                        // folds it onto DST accumulation. Everything else
                        // is SFPU sharing the kernel: loud halt.
                        let Op::Binary { x, y, bop } = kernel.ops[op_id].op else {
                            return Err(BackendError {
                                status: ErrorStatus::KernelCompilation,
                                context: format!(
                                    "tenstorrent2: sfpu and matmul cannot share a kernel (op {op_id})"
                                )
                                .into(),
                            });
                        };
                        if !matches!(bop, BOp::Add) {
                            return Err(BackendError {
                                status: ErrorStatus::KernelCompilation,
                                context: format!(
                                    "tenstorrent2: sfpu and matmul cannot share a kernel (op {op_id})"
                                )
                                .into(),
                            });
                        }
                        let mut acc = None;
                        let mut mm = None;
                        for &v in &[x, y] {
                            if matmul_tags.contains_key(&v) {
                                if mm.is_some() {
                                    return Err(BackendError {
                                        status: ErrorStatus::KernelCompilation,
                                        context: format!(
                                            "tenstorrent2: fold op {op_id} adds more than one matmul"
                                        )
                                        .into(),
                                    });
                                }
                                mm = Some(v);
                            } else if let Op::Load { src: lsrc, layout: MemLayout::Tile { .. }, .. } =
                                kernel.ops[v].op
                            {
                                match self.cb_map.get(&lsrc) {
                                    Some(&cb) if self.scratch_cbs.contains(&cb) => {
                                        if acc.is_some() {
                                            return Err(BackendError {
                                                status: ErrorStatus::KernelCompilation,
                                                context: format!(
                                                    "tenstorrent2: fold op {op_id} adds more than one acc"
                                                )
                                                .into(),
                                            });
                                        }
                                        acc = Some(lsrc);
                                    }
                                    _ => {
                                        return Err(BackendError {
                                            status: ErrorStatus::KernelCompilation,
                                            context: format!(
                                                "tenstorrent2: fold side op {v} is no acc tile load"
                                            )
                                            .into(),
                                        });
                                    }
                                }
                            } else {
                                return Err(BackendError {
                                    status: ErrorStatus::KernelCompilation,
                                    context: format!(
                                        "tenstorrent2: fold side op {v} is no acc/matmul tile"
                                    )
                                    .into(),
                                });
                            }
                        }
                        let (Some(storage), Some(m)) = (acc, mm) else {
                            return Err(BackendError {
                                status: ErrorStatus::KernelCompilation,
                                context: format!(
                                    "tenstorrent2: fold adds acc and matmul only, op {op_id} does not"
                                )
                                .into(),
                            });
                        };
                        // One static DST slot per acc chain, reused across
                        // trips (nested DST inherit, gemm precedent).
                        let slot = chain_slot.len() as u32;
                        debug_assert!(slot < 16, "tenstorrent DST holds 16 slots");
                        chain_slot.entry(storage).or_insert(slot);
                        folds.insert(op_id, (storage, m));
                        // Emit nothing.
                    } else {
                        em.emit_op(op_id, &compute_data)?;
                    }
                }
                Op::Stack { .. } => todo!(),
                Op::Storage { scope: MemScope::Circular, .. } => {
                    // Declared up front for every shared CB (see driver).
                }
                Op::Storage { scope: MemScope::Local, .. } => {
                    unreachable!(
                        "tenstorrent does not have local threads; local indices should have been converted to loops by the opt_tenstorrent_tile optimization pass"
                    )
                }
                Op::Storage { .. } => {
                    todo!("tenstorrent2 compute storage scope")
                }
                Op::Store { ref dst, src: ref store_src, layout: st_layout, .. } if is_mm => {
                    // Closed-matmul stores. Scratch stores are validated
                    // folds: wait/matmul/pop traffic only, the add rides
                    // DST accumulation for free. Pack stores drain the acc
                    // chain head: commit/wait/reserve/pack, push at
                    // EndLoop. No pack_reconfig: mm_init programmed the
                    // packer (official mm.cpp carries none either).
                    if !matches!(st_layout, MemLayout::Tile { .. }) {
                        return Err(BackendError {
                            status: ErrorStatus::KernelCompilation,
                            context: format!(
                                "tenstorrent2 compute only supports tile stores, op {op_id}"
                            )
                            .into(),
                        });
                    }
                    let Some(&out_cb) = self.cb_map.get(dst) else {
                        return Err(BackendError {
                            status: ErrorStatus::KernelCompilation,
                            context: format!("tenstorrent2: compute store targets unmapped CB, op {op_id}").into(),
                        });
                    };
                    if em.scope_level == 0 {
                        return Err(BackendError {
                            status: ErrorStatus::KernelCompilation,
                            context: format!(
                                "tenstorrent2: matmul traffic requires a loop, op {op_id} sits outside one"
                            )
                            .into(),
                        });
                    }
                    if self.scratch_cbs.contains(&out_cb) {
                        let Some(&(storage, m)) = folds.get(store_src) else {
                            return Err(BackendError {
                                status: ErrorStatus::KernelCompilation,
                                context: format!(
                                    "tenstorrent2: compute scratch store op {op_id} is not a validated fold"
                                )
                                .into(),
                            });
                        };
                        if *dst != storage {
                            return Err(BackendError {
                                status: ErrorStatus::KernelCompilation,
                                context: format!(
                                    "tenstorrent2: fold op {op_id} stores a different acc than it loads"
                                )
                                .into(),
                            });
                        }
                        let Some(&(cb_a, cb_b)) = matmul_tags.get(&m) else {
                            return Err(BackendError {
                                status: ErrorStatus::KernelCompilation,
                                context: format!("tenstorrent2: fold op {op_id} references untagged matmul").into(),
                            });
                        };
                        let Some(&slot) = chain_slot.get(&storage) else {
                            return Err(BackendError {
                                status: ErrorStatus::KernelCompilation,
                                context: format!("tenstorrent2: fold op {op_id} has no acc chain slot").into(),
                            });
                        };
                        writeln!(em.src, "{indent}cb{cb_a}.wait_front(1);");
                        writeln!(em.src, "{indent}cb{cb_b}.wait_front(1);");
                        writeln!(em.src, "{indent}matmul_tiles({cb_a}, {cb_b}, 0, 0, {slot});");
                        writeln!(em.src, "{indent}cb{cb_a}.pop_front(1);");
                        writeln!(em.src, "{indent}cb{cb_b}.pop_front(1);");
                    } else {
                        let Op::Load { src: lsrc, layout: MemLayout::Tile { .. }, .. } = kernel.ops[*store_src].op
                        else {
                            return Err(BackendError {
                                status: ErrorStatus::KernelCompilation,
                                context: format!(
                                    "tenstorrent2: compute pack store op {op_id} reads no acc tile"
                                )
                                .into(),
                            });
                        };
                        let Some(&slot) = chain_slot.get(&lsrc) else {
                            return Err(BackendError {
                                status: ErrorStatus::KernelCompilation,
                                context: format!(
                                    "tenstorrent2: compute pack store op {op_id} packs an empty acc chain"
                                )
                                .into(),
                            });
                        };
                        // Last circular store wins (v1 mm_out_cb rule).
                        mm_out = Some(out_cb);
                        writeln!(em.src, "{indent}tile_regs_commit();");
                        writeln!(em.src, "{indent}cb{out_cb}.reserve_back(1);");
                        writeln!(em.src, "{indent}tile_regs_wait();");
                        writeln!(em.src, "{indent}pack_tile({slot}, {out_cb});");
                        loop_pushes
                            .last_mut()
                            .expect("tenstorrent2 pack store outside loop body")
                            .push(out_cb);
                    }
                }
                Op::Store { ref dst, src: ref store_src, layout: st_layout, .. } => {
                    // Stored tile values flow through DST slots: every
                    // tiled value ever emitted sits in the unified
                    // register file. Anything else (scalar math,
                    // unloaded tiles) has no tiled entry and is rejected.
                    let Some(&slot) = em.var_map.get(store_src) else {
                        return Err(BackendError {
                            status: ErrorStatus::KernelCompilation,
                            context: format!(
                                "tenstorrent2: compute store reads a tile with no DST slot, op {op_id}"
                            )
                            .into(),
                        });
                    };
                    if !matches!(em.vars[slot as usize].layout, MemLayout::Tile { .. }) {
                        return Err(BackendError {
                            status: ErrorStatus::KernelCompilation,
                            context: format!(
                                "tenstorrent2: compute store reads a non-tiled value, op {op_id}"
                            )
                            .into(),
                        });
                    }
                    let Some(&out_cb) = self.cb_map.get(dst) else {
                        return Err(BackendError {
                            status: ErrorStatus::KernelCompilation,
                            context: format!("tenstorrent2: compute store targets unmapped CB, op {op_id}").into(),
                        });
                    };
                    if !matches!(st_layout, MemLayout::Tile { .. }) {
                        todo!("tenstorrent2 compute only supports tile stores");
                    }
                    if self.scratch_cbs.contains(&out_cb) {
                        return Err(BackendError {
                            status: ErrorStatus::KernelCompilation,
                            context: format!("tenstorrent2: compute copy targets scratch CB, op {op_id}").into(),
                        });
                    }
                    if em.scope_level == 0 {
                        return Err(BackendError {
                            status: ErrorStatus::KernelCompilation,
                            context: format!(
                                "tenstorrent2: compute streaming copy requires a loop, op {op_id} sits outside one"
                            )
                            .into(),
                        });
                    }
                    // Pack order: commit -> reserve -> wait -> reconfig ->
                    // pack. Push/pop/release emit at EndLoop (v1 tail:
                    // release -> push -> pop); record the push here.
                    writeln!(em.src, "{indent}tile_regs_commit();");
                    writeln!(em.src, "{indent}cb{out_cb}.reserve_back(1);");
                    writeln!(em.src, "{indent}tile_regs_wait();");
                    writeln!(em.src, "{indent}pack_reconfig_data_format({out_cb});");
                    writeln!(em.src, "{indent}pack_tile({slot}, {out_cb});");
                    loop_pushes
                        .last_mut()
                        .expect("tenstorrent2 streaming store outside loop body")
                        .push(out_cb);
                }
                Op::Load { src: ref load_src, layout, .. } if is_mm => {
                    // Resolve-only: matmul inputs resolve to CB ids at the
                    // consuming fold, acc loads to the chain slot at the
                    // pack store. No copy_tile under mm_init (a copy there
                    // stalls UNPACK: wedge guard).
                    if !matches!(layout, MemLayout::Tile { .. }) {
                        return Err(BackendError {
                            status: ErrorStatus::KernelCompilation,
                            context: format!("tenstorrent2 compute only supports tile loads, op {op_id}").into(),
                        });
                    }
                    if !self.cb_map.contains_key(load_src) {
                        return Err(BackendError {
                            status: ErrorStatus::KernelCompilation,
                            context: format!("tenstorrent2: compute load targets unmapped CB, op {op_id}").into(),
                        });
                    }
                    // Emit nothing.
                }
                Op::Load { src: ref load_src, layout, .. } => {
                    if !matches!(layout, MemLayout::Tile { .. }) {
                        todo!("tenstorrent2 compute only supports tile loads");
                    }
                    let Some(&cb) = self.cb_map.get(load_src) else {
                        return Err(BackendError {
                            status: ErrorStatus::KernelCompilation,
                            context: format!("tenstorrent2: compute load targets unmapped CB, op {op_id}").into(),
                        });
                    };
                    if self.scratch_cbs.contains(&cb) {
                        return Err(BackendError {
                            status: ErrorStatus::KernelCompilation,
                            context: format!("tenstorrent2: compute copy reads scratch CB, op {op_id}").into(),
                        });
                    }
                    // Unified register file: the vars index is the DST
                    // slot id. Tiled entries never decrement, so one
                    // static slot per use-site, reused across trips.
                    let slot = em.vars.len() as u32;
                    debug_assert!(slot < 16, "tenstorrent DST holds 16 slots");
                    em.vars.push(VarSlot {
                        dtype: compute_data.dtypes[&op_id].0,
                        layout,
                        rc: compute_data.rcs[&op_id],
                        scope_level: em.scope_level,
                    });
                    em.var_map.insert(op_id, slot);
                    writeln!(em.src, "{indent}copy_tile_init({cb});");
                    writeln!(em.src, "{indent}copy_tile({cb}, 0, {slot});");
                }
                Op::Range { .. } => {
                    if is_mm {
                        return Err(BackendError {
                            status: ErrorStatus::KernelCompilation,
                            context: format!(
                                "tenstorrent2: op {op_id} is outside the closed matmul shape"
                            )
                            .into(),
                        });
                    }
                    em.emit_op(op_id, &compute_data)?;
                }
                Op::Loop { .. } => {
                    let outermost = em.scope_level == 0;
                    em.emit_op(op_id, &compute_data)?;
                    em.indent.push_str("  ");
                    indent = em.indent.clone();
                    em.scope_level += 1;
                    loop_depth += 1;
                    if outermost {
                        // Streaming handshake waits stay outermost (sfpu
                        // path only; mm waits fire inline at each fold).
                        if !is_mm {
                            for &cb in &compute_input_cbs {
                                writeln!(em.src, "{indent}cb{cb}.wait_front(1);");
                            }
                        }
                    }
                    // Acquire pairs with the pack scope, not the
                    // outermost loop.
                    if pack_depths.contains(&loop_depth) {
                        writeln!(em.src, "{indent}tile_regs_acquire();");
                    }
                    loop_pushes.push(Vec::new());
                }
                Op::EndLoop => {
                    // v1 tail order: release, then push, then pop.
                    if pack_depths.contains(&loop_depth) {
                        writeln!(em.src, "{indent}tile_regs_release();");
                    }
                    loop_depth -= 1;
                    if let Some(pushes) = loop_pushes.pop() {
                        for cb_id in &pushes {
                            writeln!(em.src, "{indent}cb{cb_id}.push_back(1);");
                        }
                    }
                    for &cb in &compute_input_cbs {
                        // Fold inputs pop inline at their fold store (per
                        // inner iteration); scratch accs never pop.
                        // Blanket pops stay for the sfpu path only.
                        if is_mm && (fold_cbs.contains(&cb) || self.scratch_cbs.contains(&cb)) {
                            continue;
                        }
                        writeln!(em.src, "{indent}cb{cb}.pop_front(1);");
                    }
                    em.scope_level -= 1;
                    em.indent.pop();
                    em.indent.pop();
                    indent = em.indent.clone();
                    writeln!(em.src, "{indent}}}");
                }
                Op::If { condition } => {
                    // Plain brace scope: no acquire/push/pop interaction
                    // (those pair with loops, never ifs). Condition is a
                    // prior boolean scalar: consts inline, registers
                    // resolve through the section map.
                    let cond = if let Op::Const(c) = &kernel.ops[condition].op {
                        format!("{}", c.c_code())
                    } else if let Some(&r) = em.var_map.get(&condition) {
                        format!("r{r}")
                    } else {
                        return Err(BackendError {
                            status: ErrorStatus::KernelCompilation,
                            context: format!(
                                "tenstorrent2: if condition op {condition} not in registers"
                            )
                            .into(),
                        });
                    };
                    writeln!(em.src, "{indent}if ({cond}) {{");
                    em.indent.push_str("  ");
                    indent = em.indent.clone();
                    em.scope_level += 1;
                }
                Op::EndIf => {
                    em.scope_level -= 1;
                    em.indent.pop();
                    em.indent.pop();
                    indent = em.indent.clone();
                    writeln!(em.src, "{indent}}}");
                }
                Op::Index { .. } => todo!(),
                Op::MatmulTile { x, y } => {
                    if !is_mm {
                        return Err(BackendError {
                            status: ErrorStatus::KernelCompilation,
                            context: format!("tenstorrent2: matmul outside mm kernel, op {op_id}").into(),
                        });
                    }
                    // Both sides are tile loads from live (non-scratch)
                    // CBs; anything else is outside the closed shape. The
                    // op emits nothing: the consuming fold store does the
                    // wait/matmul/pop traffic (lazy, fewest locals).
                    let Op::Load { src: la, layout: MemLayout::Tile { .. }, .. } = kernel.ops[x].op else {
                        return Err(BackendError {
                            status: ErrorStatus::KernelCompilation,
                            context: format!("tenstorrent2: matmul side op {x} is no CB tile load").into(),
                        });
                    };
                    let Some(&cb_a) = self.cb_map.get(&la) else {
                        return Err(BackendError {
                            status: ErrorStatus::KernelCompilation,
                            context: format!("tenstorrent2: matmul side op {x} targets unmapped CB").into(),
                        });
                    };
                    if self.scratch_cbs.contains(&cb_a) {
                        return Err(BackendError {
                            status: ErrorStatus::KernelCompilation,
                            context: format!("tenstorrent2: matmul side op {x} reads scratch").into(),
                        });
                    }
                    let Op::Load { src: lb, layout: MemLayout::Tile { .. }, .. } = kernel.ops[y].op else {
                        return Err(BackendError {
                            status: ErrorStatus::KernelCompilation,
                            context: format!("tenstorrent2: matmul side op {y} is no CB tile load").into(),
                        });
                    };
                    let Some(&cb_b) = self.cb_map.get(&lb) else {
                        return Err(BackendError {
                            status: ErrorStatus::KernelCompilation,
                            context: format!("tenstorrent2: matmul side op {y} targets unmapped CB").into(),
                        });
                    };
                    if self.scratch_cbs.contains(&cb_b) {
                        return Err(BackendError {
                            status: ErrorStatus::KernelCompilation,
                            context: format!("tenstorrent2: matmul side op {y} reads scratch").into(),
                        });
                    }
                    fold_cbs.insert(cb_a);
                    fold_cbs.insert(cb_b);
                    matmul_tags.insert(op_id, (cb_a, cb_b));
                    let pair = (cb_a, cb_b);
                    if !mm_pairs.contains(&pair) {
                        mm_pairs.push(pair);
                    }
                }
                Op::TransposeTile { .. } => todo!(),
                Op::ReduceTile { .. } => todo!(),
                Op::Asm { .. } => todo!(),
                Op::Barrier => unreachable!("should've been filtered by kernel sections decomposition"),
                Op::Move { .. } | Op::Reduce { .. } => unreachable!("should've been lowered by linearize"),
                Op::Wmma { .. } => unreachable!("tt does not support wmma, use Op::MatmulTile instead"),
            }
        }
        // mm_init programs unpack+math+pack in one call (the only init
        // on matmul kernels): one per distinct input pair plus the pack
        // target, at the anchor ahead of all loops. Triples are complete
        // only after the walk, hence the deferred insert.
        if is_mm {
            let Some(out) = mm_out else {
                return Err(BackendError {
                    status: ErrorStatus::KernelCompilation,
                    context: "tenstorrent2: matmul kernel packs nowhere".into(),
                });
            };
            let mut init = String::new();
            for &(in0, in1) in &mm_pairs {
                writeln!(init, "  mm_init({in0}, {in1}, {out});");
            }
            em.src.insert_str(init_anchor, &init);
        }
        // Hoisted tile-op inits land at the anchor, ahead of all loops.
        em.prepend_compute_inits(init_anchor);

        // Degenerate pack scope: release the up-front acquire.
        if pack_depths.contains(&0) {
            writeln!(em.src, "  tile_regs_release();");
        }
        writeln!(em.src, "}}");
        self.compute = TTKernel::Compute { src: em.src, ordinals: ordinals.to_vec() };

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
        let mut em = OpEmitter::new(kernel);
        // Section params in list order become this section's runtime args.
        for (i, &p) in params.iter().enumerate() {
            em.arg_pos.insert(p, i as u32);
        }
        let mut prev_accessor: Option<String> = None;
        // Owned working copy; re-synced from the emitter after every
        // mutation below. Never held across `em` method calls.
        let mut indent = em.indent.clone();
        writeln!(em.src, "#include <cstdint>");
        writeln!(em.src, "#include \"api/dataflow/dataflow_api.h\"");
        writeln!(em.src, "#include \"api/dataflow/noc.h\"");
        writeln!(em.src, "#include \"api/dataflow/circular_buffer.h\"");
        writeln!(em.src, "#include \"api/tensor/noc_traits.h\"");
        writeln!(em.src, "#include \"api/debug/dprint.h\"");
        writeln!(em.src, "void kernel_main() {{");
        let mut cbs: Vec<CBId> = self.cb_map.iter().map(|(_, &cb)| cb).collect();
        cbs.sort();
        for cb in cbs {
            writeln!(em.src, "{indent}CircularBuffer cb{cb}(tt::CBIndex::c_{cb});");
        }
        // Accessors only for the GlobalMut params this section writes.
        for &op_id in ops {
            if let Op::Param { kind: ParamKind::GlobalMut, .. } = kernel.ops[op_id].op {
                let arg = em.arg_pos.get(&op_id).copied().expect("tenstorrent2 writer param missing from section args");
                writeln!(em.src, "{indent}uint32_t out{op_id} = get_arg_val<uint32_t>({arg});");
                let cta = match &prev_accessor {
                    None => String::from("0"),
                    Some(prev) => format!("{prev}.next_compile_time_args_offset()"),
                };
                writeln!(em.src, "{indent}auto args_out{op_id} = TensorAccessorArgs<{cta}>({arg});");
                writeln!(em.src, "{indent}auto p_out{op_id} = TensorAccessor(args_out{op_id}, out{op_id}, {TT_DRAM_PAGE_BYTES});");
                prev_accessor = Some(format!("args_out{op_id}"));
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
                    let arg = em.arg_pos.get(&op_id).copied().expect("tenstorrent2 writer param missing from section args");
                    let Op::Param { dtype, .. } = kernel.ops[op_id].op else {
                        unreachable!("tenstorrent2 param changed under us");
                    };
                    writeln!(em.src, "{indent}{} r{op_id} = ({})get_arg_val<uint32_t>({arg});", dtype.c_type(), dtype.c_type());
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
                            // Same register discipline as the reader: consts
                            // inline, Variable params keep r{op_id}, every
                            // other value must already sit in var_map.
                            let idx = if let Op::Const(c) = &kernel.ops[st_idx].op {
                                format!("{}", c.c_code())
                            } else if matches!(kernel.ops[st_idx].op, Op::Param { kind: ParamKind::Variable, .. }) {
                                format!("r{st_idx}")
                            } else if let Some(&r) = em.var_map.get(&st_idx) {
                                if em.vars[r as usize].scope_level == em.scope_level {
                                    debug_assert!(em.vars[r as usize].rc > 0);
                                    em.vars[r as usize].rc -= 1;
                                }
                                format!("r{r}")
                            } else {
                                return Err(BackendError {
                                    status: ErrorStatus::KernelCompilation,
                                    context: format!("tenstorrent2: writer index {st_idx} not in registers").into(),
                                });
                            };
                            writeln!(em.src, "{indent}cb{cb}.wait_front(1);");
                            writeln!(
                                em.src,
                                "{indent}uint64_t wnoc{op_id} = p_out{dst}.get_noc_addr((uint32_t)(({idx}*{elem_size})/{page_size}), (uint32_t)(({idx}*{elem_size})%{page_size}));"
                            );
                            writeln!(em.src, "{indent}noc_async_write(cb{cb}.get_read_ptr(), wnoc{op_id}, {tile_bytes});");
                            writeln!(em.src, "{indent}noc_async_write_barrier();");
                            writeln!(em.src, "{indent}cb{cb}.pop_front(1);");
                        }
                        _ => todo!("tenstorrent2 writer only supports tile stores"),
                    }
                }
                Op::Loop { .. } => {
                    em.emit_op(op_id, &writer_data)?;
                    em.indent.push_str("  ");
                    indent = em.indent.clone();
                    em.scope_level += 1;
                }
                Op::EndLoop => {
                    em.scope_level -= 1;
                    em.indent.pop();
                    em.indent.pop();
                    indent = em.indent.clone();
                    writeln!(em.src, "{indent}}}");
                }
                Op::If { condition } => {
                    // Condition is a prior boolean scalar: consts inline,
                    // registers resolve through the section map.
                    let cond = if let Op::Const(c) = &kernel.ops[condition].op {
                        format!("{}", c.c_code())
                    } else if let Some(&r) = em.var_map.get(&condition) {
                        format!("r{r}")
                    } else {
                        return Err(BackendError {
                            status: ErrorStatus::KernelCompilation,
                            context: format!(
                                "tenstorrent2: if condition op {condition} not in registers"
                            )
                            .into(),
                        });
                    };
                    writeln!(em.src, "{indent}if ({cond}) {{");
                    em.indent.push_str("  ");
                    indent = em.indent.clone();
                    em.scope_level += 1;
                }
                Op::EndIf => {
                    em.scope_level -= 1;
                    em.indent.pop();
                    em.indent.pop();
                    indent = em.indent.clone();
                    writeln!(em.src, "{indent}}}");
                }
                Op::Barrier => {}
                Op::Range { .. } => {
                    em.emit_op(op_id, &writer_data)?;
                }
                Op::Const(_) => {
                    // Inlined as literals at uses; no declaration emitted.
                }
                Op::Binary { .. } | Op::Mad { .. } | Op::Cast { .. } => {
                    em.emit_op(op_id, &writer_data)?;
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
        writeln!(em.src, "}}");
        self.writer = TTKernel::Writer { src: em.src, ordinals: ordinals.to_vec() };
        Ok(())
    }
}

impl OpEmitter<'_> {
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
    fn emit_op(&mut self, op_id: OpId, data: &SectionData) -> Result<(), BackendError> {
        let kernel = self.kernel;
        let indent = self.indent.clone();
        let scope_level = self.scope_level;
        let arg_pos = &self.arg_pos;
        let src = &mut self.src;
        let var_map = &mut self.var_map;
        let vars = &mut self.vars;
        let unary_inits = &mut self.unary_inits;
        let binary_inits = &mut self.binary_inits;
        let typecast_inits = &mut self.typecast_inits;
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
        /// Numeric twin of `get_var` for tile ops: the DST slot id of a
        /// tiled value. Anything else (scalars, missing entries) is a
        /// compilation error — tile calls take slot ids.
        fn tile_id(id: OpId, var_map: &Map<OpId, u32>, vars: &[VarSlot]) -> Result<u32, BackendError> {
            if let Some(&r) = var_map.get(&id) {
                if matches!(vars[r as usize].layout, MemLayout::Tile { .. }) {
                    return Ok(r);
                }
            }
            Err(BackendError {
                status: ErrorStatus::KernelCompilation,
                context: format!("tenstorrent2: operand {id} is not a tiled DST value").into(),
            })
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
        match &kernel.ops[op_id].op {
            Op::Const(_) => unreachable!("tenstorrent2 consts inline as literals at their uses"),
            Op::Binary { x, y, bop } => {
                let dt = data.dtypes[&op_id].0;
                let rlay = data.dtypes[&op_id].1;
                let xlay = data.dtypes[x].1;
                let ylay = data.dtypes[y].1;
                if matches!(rlay, MemLayout::Tile { .. }) {
                    // Tiled ALU: both operands alias DST slots, the result
                    // folds onto the lhs slot (v1 precedent). The init
                    // hoists pre-loop through the init sets.
                    let sx = tile_id(*x, var_map, vars)?;
                    let sy = tile_id(*y, var_map, vars)?;
                    var_map.insert(op_id, sx);
                    binary_inits.insert(*bop);
                    match bop {
                        BOp::Mul => writeln!(src, "{indent}mul_binary_tile({sx}, {sy}, {sx});"),
                        _ => todo!("tenstorrent2 tiled binary op"),
                    };
                    return Ok(());
                }
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
                if matches!(rlay, MemLayout::Tile { .. }) {
                    // Standalone tile cast (split-A shape): any slotted
                    // tile in, same DST slot out (v1 aliasing). The
                    // slotted-operand check is the validity proof; fused
                    // mixed-format math never reaches a slot. The init
                    // hoists pre-loop through the init sets.
                    let sx = tile_id(*x, var_map, vars)?;
                    var_map.insert(op_id, sx);
                    let in_fmt = tt_fmt(kernel.dtype(*x))?;
                    let out_fmt = tt_fmt(*dtype)?;
                    if in_fmt != out_fmt {
                        if !typecast_inits.contains(&(in_fmt, out_fmt)) {
                            typecast_inits.push((in_fmt, out_fmt));
                        }
                        writeln!(src, "{indent}typecast_tile<{in_fmt}, {out_fmt}>({sx});");
                    }
                    return Ok(());
                }
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
                    let arg = arg_pos.len() as u32 + *axis;
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
            Op::Unary { uop, .. } if matches!(data.dtypes[&op_id].1, MemLayout::Tile { .. }) => {
                unary_inits.insert(*uop);
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
