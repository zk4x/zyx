use crate::{
    DType, Map,
    error::BackendError,
    kernel::{IDX_T, Kernel, MMADType, MemLayout, MemScope, Op, OpId, RangeKind},
};

use nanoserde::{DeBin, SerBin};

use crate::slab::SlabId;

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

enum TTKernel {
    Reader { src: String, params: Vec<CBId> },
    Compute { src: String, params: Vec<CBId> },
    Writer { src: String, params: Vec<CBId> },
    None,
}

impl Kernel {
    /// Generate TT Metalium reader, compute, and writer C++ kernel sources from zyx IR.
    ///
    /// Walks the kernel three times — once per section (reader, compute, writer) —
    /// emitting TT Metalium dataflow and compute API calls for each op.
    ///
    /// # Parameters
    /// - `kernel` — zyx kernel IR (after tiling passes: pad, local, group, loop_local)
    /// - `debug_asm` — if true, print each generated source to stdout
    /// - `n_inputs` — number of Global params (GlobalMut params occupy the tail of the
    ///   head-order param list after the Global + Variable params)
    /// - `n_outputs` — number of writable global tensors
    /// - `reader_params` / `compute_params` / `writer_params` — per-section lists of
    ///   param ordinals (head order) that section needs. **Runtime-arg convention**
    ///   (identical for every section): the section's args are exactly
    ///   `[its Global + Variable params in head order] + [its GlobalMut params] +
    ///   [gidx0, gidx1]` — gidx0/gidx1 are the core's coordinates in the tensix grid,
    ///   different in each core. Tenstorrent has no SIMT threads, so there are no
    ///   local ranges — local indices must have been converted to loops by the
    ///   opt_tenstorrent_tile pass before codegen.
    /// - `cb_map` — maps each Circular storage op to its single CB index,
    ///   shared by reader, compute, and writer sections
    ///
    /// # Returns
    /// `(reader_source, compute_source, writer_source)` as C++ strings ready
    /// for the tt-metal JIT compiler.
    #[allow(unused_must_use)]
    pub(crate) fn generate_tenstorrent(&self) -> Result<(TTKernel, TTKernel, TTKernel), BackendError> {
        // One CBId per Circular storage, shared by all three sections:
        // a CB filled by one section and consumed by another must address
        // the same CB everywhere. Collected head order over the whole
        // kernel so ids are stable regardless of section.
        let mut cb_map: Map<OpId, CBId> = Map::default();
        let mut next_cb = CBId::ZERO;
        let mut scan = self.head;
        for _ in 0..10_000 {
            if scan.is_null() {
                break;
            }
            if let Op::Storage { scope: MemScope::Circular, .. } = self.ops[scan].op {
                if !cb_map.contains_key(&scan) {
                    cb_map.insert(scan, next_cb);
                    next_cb.inc();
                }
            }
            scan = self.next_op(scan);
        }
        if !scan.is_null() {
            panic!("tenstorrent2 cb_map scan did not finish in 10000 steps");
        }

        let reader = TTKernel::Reader { src: String::new(), params: Vec::new() };
        let compute = TTKernel::Reader { src: String::new(), params: Vec::new() };
        let writer = TTKernel::Reader { src: String::new(), params: Vec::new() };

        Ok((reader, compute, writer))
    }

    /// All ops needed by the stores inside the given section, in IR order,
    /// with their dtypes and section-local refcounts.
    ///
    /// The list holds the section's stores, the transitive closure of
    /// their data dependencies, and the structural ops (loops, branches,
    /// ranges, barriers) lexically inside the section. `dtypes`/`rcs`
    /// mirror [`Kernel::compute_dtypes_and_rcs`] restricted to this set:
    /// refcounts only count uses inside the section.
    fn get_needed_ops(&self, tt_section: TtSection) -> (Vec<OpId>, Map<OpId, (DType, MemLayout)>, Map<OpId, u32>) {
        // Phase 1: stores and structural ops lexically inside the section.
        let mut section = TtSection::Reader;
        let mut stores: Vec<OpId> = Vec::new();
        let mut structural: Map<OpId, ()> = Map::default();
        let mut scan = self.head;
        for _ in 0..10_000 {
            if scan.is_null() {
                break;
            }
            match &self.ops[scan].op {
                Op::Barrier => {
                    if section == tt_section {
                        structural.insert(scan, ());
                    }
                    section.advance();
                }
                Op::Store { .. } => {
                    if section == tt_section {
                        stores.push(scan);
                    }
                }
                Op::Loop { .. } | Op::EndLoop | Op::If { .. } | Op::EndIf | Op::Range { .. } => {
                    if section == tt_section {
                        structural.insert(scan, ());
                    }
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
        let mut stack: Vec<OpId> = Vec::new();
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
        let mut list: Vec<OpId> = Vec::new();
        let mut dtypes: Map<OpId, (DType, MemLayout)> = Map::default();
        let mut rcs: Map<OpId, u32> = Map::default();
        let mut op_id = self.head;
        for _ in 0..10_000 {
            if op_id.is_null() {
                break;
            }
            if needed.contains_key(&op_id) || structural.contains_key(&op_id) {
                list.push(op_id);
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
        (list, dtypes, rcs)
    }
}
