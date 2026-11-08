// Copyright (C) 2025 zk4x
// SPDX-License-Identifier: LGPL-3.0-only WITH Classpath-exception-2.0
use std::hash::{Hash, Hasher};

use nanoserde::{DeBin, SerBin};

use crate::backend::{Dev, GwsDim, ProgramId};
use crate::dtype::Constant;
use crate::kernel::MemScope;
use crate::shape::{Dim, UAxis};
use crate::slab::SlabId;
use crate::types::{TinyString, TinyVec};
use crate::{DType, Map};

/// Kernel parameter kind
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, SerBin)]
pub enum ParamKind {
    /// Single scalar variable
    Variable,
    /// Global read only buffer
    Global,
    /// Global read-write buffer
    GlobalMut,
}

/// Program source block: an opaque chunk of backend source text carried
/// through the IR (e.g. a hand-written kernel body). Boxed `str` (16 bytes)
/// keeps `Op` within its 24-byte budget; `String` (24 bytes inline) does not,
/// and `TinyString` caps at 255 bytes. `SerBin` is manual — `nanoserde` has
/// no `SerBin for str` — and serialize-only, matching `Op` (no `DeBin`).
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct SourceBlock(pub Box<str>);

impl SourceBlock {
    /// Source text.
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl SerBin for SourceBlock {
    fn ser_bin(&self, output: &mut Vec<u8>) {
        self.as_str().len().ser_bin(output);
        output.extend_from_slice(self.as_str().as_bytes());
    }
}

#[derive(Debug, Clone, SerBin)]
/// Kernel IR op: one node of the op list, walked `head` → `next_op`.
/// Pre-linearization ops are pure DAG views/reduces; post-linearization
/// they are linear SSA over `Param`/`Storage` escape hatches (see
/// [`Op::Param`], [`Op::Storage`]).
pub enum Op {
    // ops that exist in both
    /// A compile-time constant scalar.
    Const(Constant),
    /// A kernel parameter — one of the arguments passed to the compiled GPU
    /// kernel at launch.
    ///
    /// Pre-linearization the kernel is a pure DAG and Params are its leaves.
    /// Post-linearization the kernel is a linear SSA construct; Params remain
    /// as leaves and — together with [`Op::Storage`] — act as the SSA escape
    /// hatches: the mutable stuff values are read from and written to.
    /// Linearize only nulls a Param's `shape` field, so buffer sizes must be
    /// resolved BEFORE linearization (see `Kernel::alloc_buffers`).
    ///
    /// - [`ParamKind::Variable`] — a single scalar argument (e.g. a dynamic
    ///   dim), passed by value.
    /// - [`ParamKind::Global`] / [`ParamKind::GlobalMut`] — a read-only /
    ///   read-write buffer argument, passed by pointer.
    ///
    /// # Null operands
    ///
    /// Data operands (`x`, `y`, ...) of any op can never be null — only shape
    /// operands may be, where null means scalar shape (rank 0). A null data
    /// operand is always a bug.
    Param {
        /// Buffer element dtype.
        dtype: DType,
        /// How the kernel receives it (buffer pointer or scalar value).
        kind: ParamKind,
        /// Buffer shape (null means scalar shape, rank 0).
        shape: OpId,
        /// Buffer identity for egraph hashconsing: two params name the same
        /// buffer iff their `id`s agree. Kernel passes ignore it (params are
        /// bound positionally there); it is compared by `Eq` and included
        /// in `Hash` (canonicalized to head-order ordinals first, so
        /// program caches keep sharing across buffers).
        id: u32,
    },
    /// Cast `x` to `dtype` (value conversion; bit-preserving form is
    /// [`Op::Bitcast`]).
    Cast {
        /// Value to convert.
        x: OpId,
        /// Target dtype.
        dtype: DType,
    },
    /// Bitcast: reinterprets the raw bits of `x` as `dtype` without a value
    /// conversion. Requires equal bit widths of `x`'s dtype and `dtype`
    /// (`debug_assert` in `Kernel::bitcast`).
    Bitcast {
        /// Value whose bits to reinterpret.
        x: OpId,
        /// Target dtype (same bit width as `x`'s dtype).
        dtype: DType,
    },
    /// Elementwise unary op over `x`.
    Unary {
        /// Operand.
        x: OpId,
        /// The unary op.
        uop: UOp,
    },
    // For binary ops, next of x is y, then next of y is the binary op
    /// Elementwise binary op over `x` and `y`.
    Binary {
        /// Left operand.
        x: OpId,
        /// Right operand.
        y: OpId,
        /// The binary op.
        bop: BOp,
    },
    /// Branchless select: `cond ? if_true : if_false`. `cond` must be bool;
    /// result dtype and shape are `if_true`'s (`if_false` must match).
    /// Unlike the arithmetic `where` decomposition (`sel*a + (1-sel)*b`)
    /// this never multiplies by zero, so `±inf` on the dead side stays dead.
    /// Fusion handling is a later task: until then the kernelizer leaves
    /// this op alone (`todo!()`).
    Select {
        /// Boolean condition.
        cond: OpId,
        /// Value where `cond` is true.
        if_true: OpId,
        /// Value where `cond` is false.
        if_false: OpId,
    },
    // Vectorization, YAY!
    /// Vectorization: pack `ops` into one vector value.
    Stack {
        /// Members, in order.
        ops: Box<[OpId]>,
    },
    /// Select a single value from a vector.
    Index {
        /// The vector.
        vec: OpId,
        /// Position to select.
        idx: usize,
    }, // select a single value from a vector

    // ops that only exist after unfolding views and reduces
    /// Memory internal to the kernel — NOT a launch argument. Used for
    /// accumulators, shared/local memory, Tenstorrent circular buffers,
    /// arrays of values in registers, etc.
    ///
    /// Storage ops only exist post-linearization: in the linear SSA construct
    /// they — together with [`Op::Param`] — are the escape hatches, the
    /// mutable stuff that values are written to and read back from across the
    /// linear order. A `MemScope::Variable` storage holds a single scalar.
    Storage {
        /// Element dtype.
        dtype: DType,
        /// Memory scope (register file, circular buffer, global, ...).
        scope: MemScope,
        /// Element count.
        len: Dim,
        /// Allocation index: which object of this scope and size the storage
        /// is. `0` until the register allocator runs, so the storage is then
        /// only "a request for an allocation of this shape" — equal to every
        /// other request of the same `(dtype, scope, len)`, exactly as if the
        /// field did not exist. The allocator assigns the ids that fit the
        /// device's capacity (one per DST tile on Tenstorrent, one per
        /// coloured register on CUDA, ...) and, from then on, `id` is part of
        /// the storage's identity: `Eq`/`Hash` separate it from every other
        /// allocation, so post-allocation CSE cannot merge two registers.
        /// Backends that expose their file (Tenstorrent) read it directly as
        /// the physical slot; backends that hide it never look at it.
        id: u32,
    },
    /// Store an SSA value through an addressed location: `dst` is a
    /// [`Op::Param`] or [`Op::Storage`], `dst_idx` is arbitrary index
    /// arithmetic into it. Hard rule: Store is SSA→memory (never
    /// memory→memory within a kernel — that is [`Op::Copy`]). Effect-only:
    /// produces no SSA value, is a DCE root, is never CSE'd or
    /// LICM-hoisted.
    Store {
        /// Destination base ([`Op::Param`] or [`Op::Storage`]).
        dst: OpId,
        /// Index arithmetic into `dst`. Its shape is the access shape:
        /// a scalar index stores one element, an index over a lane
        /// vector stores a vector, an index over two distinct axes
        /// stores a tile.
        dst_idx: OpId,
        /// SSA value to store. Its shape must equal `dst_idx`'s.
        src: OpId,
    },
    /// Load an addressed location into an SSA value. Hard rule: Load is
    /// memory→SSA (the location stays in memory-space; only the value
    /// enters SSA).
    Load {
        /// Source base ([`Op::Param`] or [`Op::Storage`]).
        src: OpId,
        /// Index arithmetic into `src`. Its shape is the value shape:
        /// a scalar index loads one element, an index over a lane
        /// vector loads a vector, an index over two distinct axes
        /// loads a tile.
        src_idx: OpId,
    },
    /// Copy between addressed locations: moves the value at `src` to `dst`.
    /// Hard rule: both bases are [`Op::Param`]/[`Op::Storage`]s
    /// (post-linearize only), never SSA values. Effect-only
    /// like [`Op::Store`]: produces no SSA value, is a DCE root, is never
    /// CSE'd or LICM-hoisted. General: any backend may lower it (TT tile
    /// moves, CUDA DRAM-to-shared staging, ...).
    Copy {
        /// Source base ([`Op::Param`] or [`Op::Storage`]).
        src: OpId,
        /// Index arithmetic into `src`. Its shape is the access shape;
        /// must equal `dst_idx`'s.
        src_idx: OpId,
        /// Destination base ([`Op::Param`] or [`Op::Storage`]).
        dst: OpId,
        /// Index arithmetic into `dst`.
        dst_idx: OpId,
    },
    // Like loop, but for dimensions always executed in parallel
    /// A dimension always executed in parallel (never a trip-counted loop).
    Range {
        /// Axis index.
        axis: u32,
        /// What the range iterates (group grid, fixed span, ...).
        kind: RangeKind,
    },
    // Control flow: a counted loop, or a conditional when `len` is a
    // boolean (0-or-1 trip). Closed by `EndLoop` in both cases; use
    // `loop_over` / `if_over` to build them.
    /// Counted loop over `len` trips (a 0-or-1-trip conditional when
    /// `len` is boolean). Closed by [`Op::EndLoop`].
    Loop {
        /// Trip count, or the branch condition.
        len: OpId,
    },
    /// Closes the innermost open [`Op::Loop`].
    EndLoop,
    /// Ordering barrier: an operand-free effect and DCE root that pins
    /// program order across it.
    Barrier,
    // fused matmul, a, b, c are fragments, each is a vector, c is accumulator, returns new accumulated vector d
    /// Fused matmul on fragments: `d = a @ b + c`, returning the new
    /// accumulated vector `d`.
    Wmma {
        /// Fragment geometry.
        dims: MMADims,
        /// Fragment layout.
        layout: MMALayout,
        /// Fragment element dtype.
        dtype: MMADType,
        /// Left fragment.
        a: OpId,
        /// Right fragment.
        b: OpId,
        /// Accumulator fragment.
        c: OpId,
    },
    // For backend specific assembly
    /// Backend-specific assembly template: emitted verbatim with
    /// operands substituted.
    Asm {
        /// Template text (`{i}` substitutes the i-th operand).
        asm: TinyString,
        /// Operands.
        ops: TinyVec<OpId>,
    },

    // ops that exist before linearize and linearize converts them into these ops: index, loop and load
    /// Reshape to a new shape.
    Reshape {
        /// Input.
        x: OpId,
        /// New shape.
        shape: OpId,
    },
    /// Expand dimensions.
    Expand {
        /// Input.
        x: OpId,
        /// New shape.
        shape: OpId,
    },
    /// Permute axes.
    Permute {
        /// Input.
        x: OpId,
        /// Axis order.
        axes: TinyVec<UAxis>,
    },
    /// Flip axes
    Flip {
        /// Input.
        x: OpId,
        /// Axes to flip.
        axes: TinyVec<UAxis>,
    },
    /// Pad axis
    /// Pad with `lp` zeros on the left, to total axis length `len`. Right padding is `len - lp - orig_len`.
    Pad {
        /// Input.
        x: OpId,
        /// Axis to pad.
        axis: UAxis,
        /// Left pad width (zeros).
        lp: OpId,
        /// Total axis length after padding.
        len: OpId,
    },
    /// Slice axis
    Narrow {
        /// Input.
        x: OpId,
        /// Axis to slice.
        axis: UAxis,
        /// Slice start.
        start: OpId,
        /// Slice length.
        len: OpId,
    },
    /// Reduce `x` with `rop`, reducing away the named lanes: if
    /// `reduce_axis` is an `Op::Unroll`, its axis slot is zeroed,
    /// otherwise the last dim drops.
    Reduce {
        /// Input.
        x: OpId,
        /// Reduce op.
        rop: BOp,
        /// Axis expression naming the reduced lanes (see above).
        reduce_axis: OpId,
    },
    // Graph-only ops (former `Node` variants). They never appear in ordered
    // kernels: every kernel-side match arms them with `todo!()`.
    // NOTE: there is no `Assign` variant: graph assigns lower to a
    // whole-view `Op::Store` (bare `Param` dst, null bytes — `linearize`
    // it later).
    /// Ordering edge: `x` may not run before `dep` completes.
    After {
        /// Gated op.
        x: OpId,
        /// Prerequisite.
        dep: OpId,
    },
    /// Move `x` to `device`. `time` is measured launch timing, ignored by
    /// `Eq`/`Hash` (mirrors the former `Node::ToDevice`).
    ToDevice {
        /// Value to move.
        x: OpId,
        /// Target device.
        device: Dev,
        /// Measured launch timing (ignored by `Eq`/`Hash`).
        time: u64,
    },
    /// Fusion-break hint: forces `x` to materialize as a separate kernel
    /// output. The kernelizer never fuses through it.
    Contiguous {
        /// Value to materialize.
        x: OpId,
    },
    /// Autograd-break hint: value identity forward (`Detach(x) == x`), but
    /// the gradient does not propagate through it. The kernelizer erases it
    /// (remaps to `x`), so it never reaches ordered kernels.
    Detach {
        /// Value to detach from the backpropagation graph.
        x: OpId,
    },

    /// Unroll lane vector: the lane constants `0..len` as a single
    /// post-linearize value, tagged with the axis it was split from.
    /// Produced only by `split`; devectorized into a `Stack` of consts
    /// right before codegen. Shapes the ops it flows through: a binary
    /// of a scalar with an unroll is a vector, unrolls of two different
    /// axes in one expression form a tile.
    Unroll {
        /// The `Op::Range`/`Op::Loop` axis this lane vector was split from.
        axis: u32,
        /// Number of lanes: the vector holds values `0..len`.
        len: u16,
    },

    /// Program source block
    Source(SourceBlock),
    /// A compiled kernel boundary: `info` is the owning program and measured
    /// timing. Both `inputs` and `outputs` are `OpId`s of `Stack` ops holding
    /// the input/output classes — the lists are shared nodes, not per-kernel
    /// `Box` allocations. `info` is boxed so `Op` keeps its 24-byte budget.
    /// Timing is ignored by `Eq`/`Hash` (mirrors the former `Node::Kernel`).
    Program {
        /// Input classes (`Stack` op).
        inputs: OpId,
        /// Output classes (`Stack` op).
        outputs: OpId,
        /// Owning program and measured timing.
        info: Box<(ProgramId, u64)>,
    },
    /// A custom (user-built) kernel boundary, boxed to keep `Op` within
    /// its 24-byte budget. `outputs` triples are `(class, shape, dtype)`.
    /// `time` is measured timing, ignored by `Eq`/`Hash` (mirrors the
    /// former `Node::Custom`, which also never compares equal).
    Kernel(Box<CustomKernel>),

    // Backend specific extensions
    /// TT
    TT(TTOp),
    /// GPU launch configuration shared by all GPU backends (CUDA, HIP,
    /// OpenCL, Vulkan, ...). Pure value: equal contents compare equal so
    /// CSE shares them. Void leaf, no operands. Boxed so `Op` keeps its
    /// 24-byte budget (mirrors `Op::Kernel`).
    GPU(Box<GPUOp>),
    /// SPIR-V payload shared by Vulkan/wgpu: compiled word stream plus
    /// SPIR-V-specific annotations. Pure value: equal contents compare
    /// equal so CSE shares them. Void leaf, no operands. Boxed so `Op`
    /// keeps its 24-byte budget (mirrors `Op::GPU`).
    Spirv(Box<SpirvOp>),
    /// PTX payload for CUDA: assembled PTX bytes emitted directly at
    /// render (under `ZYX_PTX`), bypassing NVRTC at compile. Pure value:
    /// equal contents compare equal so CSE shares them. Void leaf, no
    /// operands. Boxed so `Op` keeps its 24-byte budget (mirrors
    /// `Op::Spirv`).
    PTX(Box<PTXOp>),
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, SerBin)]
/// GPU backend meta op: launch configuration shared by all GPU backends.
/// Lives inside [`Op::GPU`].
pub enum GPUOp {
    /// Local work size: work-items per work-group (OpenCL/Vulkan) /
    /// threads per block (CUDA/HIP), x/y/z. Concrete by codegen time;
    /// validated against device limits at launch.
    LocalWorkSize([u32; 3]),
    /// Global work size: one owned group-length expression per axis.
    /// GPUs are always 3D (missing axes are `Const(1)`); more than three
    /// group axes is a render-time error. Lowered once at render from the
    /// `Group` ranges; the launcher evaluates each against the launch args.
    Grid([GwsDim; 3]),
    /// Head-order `ParamKind` per kernel param (reads vs writes split
    /// and arity check at launch). Boxed to keep `Op` in budget.
    Params(Box<[ParamKind]>),
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, SerBin)]
/// SPIR-V backend meta op: compiled module plus SPIR-V-specific
/// annotations. Lives inside [`Op::Spirv`].
pub enum SpirvOp {
    /// Compiled SPIR-V word stream. Boxed to keep `Op` in budget.
    WordBytes(Box<[u32]>),
    /// Precomputed push-constant block size in bytes (Vulkan): laid out
    /// from the `Variable` params' dtypes at render, when the full kernel
    /// is still available. Zero means no variables.
    PushConstants(u32),
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, SerBin)]
/// PTX backend meta op: assembled PTX bytes. Lives inside [`Op::PTX`].
pub enum PTXOp {
    /// Assembled PTX assembly bytes, as handed to the driver (same shape
    /// as NVRTC output). Boxed to keep `Op` in budget.
    Bytes(Box<[u8]>),
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, SerBin)]
/// One page of the Tenstorrent reader list ([`TTOp::DmRead`]).
pub struct TTDmRead {
    /// Global head-order param ordinal (the launch arg index).
    pub param: u32,
    /// L1 byte address this page lands at.
    pub l1: u32,
    /// Bytes this page moves.
    pub len: u32,
    /// Byte offset of the page inside the param's buffer for every grid point,
    /// in launch order (`axis0 * extent1 + axis1`). The list is staged per
    /// core, so the address is arbitrary per grid point, never a stride.
    pub offsets: Box<[u32]>,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, SerBin)]
/// One page of the Tenstorrent writer list ([`TTOp::DmWrite`]).
pub struct TTDmWrite {
    /// Global head-order param ordinal (the launch arg index).
    pub param: u32,
    /// L1 byte address this page is read from.
    pub l1: u32,
    /// Bytes this page moves.
    pub len: u32,
    /// Byte offset of the page inside the param's buffer for every grid point,
    /// in launch order (`axis0 * extent1 + axis1`).
    pub offsets: Box<[u32]>,
    /// Writer-side narrowing: the L1 tile is FP32 and the param is 16-bit, so
    /// the host write-back converts (`TT_NARROW_*`).
    pub narrow: u8,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, SerBin)]
/// One tile-slot read of the reader list, in either shape the mover knows:
/// transposed ([`TTOp::DmReadTransposed`]) or with column 0 copied across
/// ([`TTOp::DmReadBroadcastCol`]). Both are `dm::op::READ_TRANSPOSED` /
/// `READ_BROADCAST_COL`: exactly one tile slot, so no length, and the
/// transpose scratch slot is the mover's, not the kernel's.
pub struct TTDmTileRead {
    /// Global head-order param ordinal (the launch arg index).
    pub param: u32,
    /// L1 byte address the shaped tile lands at.
    pub dst: u32,
    /// Byte offset of the tile inside the param's buffer for every grid point,
    /// in launch order (`axis0 * extent1 + axis1`).
    pub offsets: Box<[u32]>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, SerBin)]
/// One padding fill of a reader/writer list ([`TTOp::DmFill`]): every datum
/// of the tile slot at `dst` outside the first `bounds & 0xff` rows and
/// `bounds >> 8` columns takes `value` (each bound of `0` means 32).
pub struct TTDmFill {
    /// Datum written outside the bounds.
    pub value: u32,
    /// Packed bounds, as `dm::fill::param` packs them.
    pub bounds: u32,
    /// L1 byte address of the tile slot.
    pub dst: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, SerBin)]
/// One L1-to-L1 word copy of a list ([`TTOp::DmCopyWords`]):
/// `count` aligned words from `src` to `dst`, `src_stride`/`dst_stride`
/// bytes apart. The mover does no arithmetic: this is a bit-preserving
/// move inside the data arena.
pub struct TTDmCopyWords {
    /// L1 byte address copied from.
    pub src: u32,
    /// L1 byte address copied to.
    pub dst: u32,
    /// Words copied.
    pub count: u32,
    /// Byte stride between source words.
    pub src_stride: u32,
    /// Byte stride between destination words.
    pub dst_stride: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, SerBin)]
/// One indexed gather of a list ([`TTOp::DmIndexPick`]): B selects one datum
/// per row using resident I32 indices. All three slots are L1 addresses; the
/// source holds a 32-column chunk beginning at column `first`.
pub struct TTDmIndexPick {
    /// L1 byte address of the I32 indices.
    pub indices: u32,
    /// L1 byte address of the source tile.
    pub source: u32,
    /// L1 byte address of the gathered tile.
    pub output: u32,
    /// Indices outside `0..bound` report a domain error before any access.
    pub bound: u32,
    /// First source column of the chunk.
    pub first: u32,
    /// Rows gathered.
    pub rows: u32,
    /// Bytes per gathered datum.
    pub datum_bytes: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, SerBin)]
/// The list entry that runs the tile's roles ([`TTOp::DmKernel`]): the mover
/// waits for every move before it, points each role at its program image,
/// bumps the generation, and waits for the acknowledgement. Zero `at`/`len`
/// means "run what the host staged".
pub struct TTDmKernel {
    /// Generation counter the roles acknowledge.
    pub generation: u32,
    /// Program address per role (T0, T1, T2); zero means staged.
    pub at: [u32; 3],
    /// Program length in words per role (T0, T1, T2).
    pub len: [u32; 3],
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, SerBin)]
/// One row-level SFPU load ([`TTOp::SfpLoad`]): the `Dst` datum at address `at`
/// of the Register tile `slot` goes into lane register `reg`. `at` is the
/// address inside the tile, so codegen packs `slot.id * rows + at`; `fmt` is
/// the `Dst` format code (`tt_isa::sfpu::mod0_fmt`).
pub struct TTSfpLoad {
    /// Register tile read from.
    pub slot: OpId,
    /// `Dst` address inside the tile.
    pub at: u16,
    /// Lane register written (`L0..L15`; only `L0..L7` are writable).
    pub reg: u8,
    /// `Dst` format code (`mod0_fmt::FP32`, `INT32`, ...).
    pub fmt: u8,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, SerBin)]
/// One row-level SFPU store ([`TTOp::SfpStore`]): lane register `reg` goes to
/// the `Dst` address `at` of the Register tile `slot`. See [`TTSfpLoad`].
pub struct TTSfpStore {
    /// Register tile written to.
    pub slot: OpId,
    /// `Dst` address inside the tile.
    pub at: u16,
    /// Lane register read.
    pub reg: u8,
    /// `Dst` format code.
    pub fmt: u8,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, SerBin)]
/// One role's program ([`TTOp::Math`]): the instruction words codegen encoded,
/// plus the block repeats the role runner pushes. A loop entry is
/// `[start, len, count]`: push `words[start..start + len]` `count` times. This
/// is how a row loop body longer than the 32-word hardware replay buffer is
/// stored — one copy of the body in the program instead of 32 unrolled ones.
pub struct TTRoleProgram {
    /// The program's instruction words, in push order.
    pub words: Box<[u32]>,
    /// Block repeats over `words`, at most `mailbox::loops::MAX`.
    pub loops: Box<[[u32; 3]]>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, SerBin)]
/// One lane-register immediate ([`TTOp::SfpConst`]): `reg = bits`, in every
/// enabled lane. One `SFPLOADI` when the low half is zero, two otherwise.
pub struct TTSfpConst {
    /// Lane register written.
    pub reg: u8,
    /// The F32 bits (or the raw bits, for an integer-form program).
    pub bits: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, SerBin)]
/// One non-addressing SFPU instruction ([`TTOp::Sfp`]): the ISA form named by
/// its opcode, with the operand fields in the ISA's own order (high bits
/// first, as `InstructionDef::fields` lists them) and zero padding. Every
/// SFPU field is at most 16 bits wide (the widest is `Imm16`), so `u16`
/// holds every field value.
pub struct TTSfpInsn {
    /// ISA opcode (`tt_isa::isa::InstructionDef::opcode`).
    pub op: u8,
    /// Field values, in `fields()` order.
    pub args: [u16; 5],
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, SerBin)]
/// Tenstorrent backend op: the DRAM boundary stream, run-level compute
/// programs, and section markers. Lives inside [`Op::TT`].
///
/// These are the native operations the backend emits. tt-metal's
/// circular-buffer sync (`reserve_back`/`push_back`/`wait_front`/`pop_front`)
/// and DST locks (`tile_regs_*`) are not Tensix instructions, so they do not
/// exist here: the native builders carry their own semaphores and slots.
pub enum TTOp {
    /// TT section split: end of the reader section. Operand-free
    /// effect: a DCE root, never CSE'd or LICM-hoisted, program-ordered
    /// by scheduling. Inserted by lowering, never by render — section
    /// assignment is a placement decision. `tt_verify` enforces exactly
    /// one `EndReader` followed by one `EndCompute`, and rejects
    /// `Op::Barrier` in any kernel containing TT ops.
    EndReader,
    /// TT section split: end of the compute section. Same effect rules
    /// as [`TTOp::EndReader`].
    EndCompute,
    /// Rendered tensix grid: one owned group-length expression per
    /// axis, fixed 2D (missing axes are `Const(1)`); more than two
    /// group axes is a render-time error. Lowered once at render from
    /// the `Group` ranges; the launcher evaluates each against the
    /// launch args. Operand-free effect like [`TTOp::EndReader`].
    /// Boxed to keep `Op` in budget.
    TensixGrid(Box<[GwsDim; 2]>),
    /// One DRAM page of the reader list: `len` bytes from the buffer of param
    /// `param` (at the byte offset its per-grid-point table names, for the
    /// launch's grid point) into L1 at `l1`. Codegen emits one per 4 KB page,
    /// so the entry count — the stream's shape — is fixed at compile time; the
    /// launch only substitutes the chunk address and the grid point. Effect
    /// only: a DCE root, never deduplicated. Boxed to keep `Op` in budget.
    DmRead(Box<TTDmRead>),
    /// One DRAM page of the writer list (L1 -> DRAM). See [`TTOp::DmRead`]; the
    /// direction is the variant, never a launch-time choice.
    DmWrite(Box<TTDmWrite>),
    /// One transposed tile read of the reader list: the mover lands the tile
    /// slot in its scratch slot and writes the transpose to `dst`. Boxed to
    /// keep `Op` in budget.
    DmReadTransposed(Box<TTDmTileRead>),
    /// One broadcast tile read of the reader list: as [`TTOp::DmReadTransposed`],
    /// but every column of the tile takes the value of column 0.
    DmReadBroadcastCol(Box<TTDmTileRead>),
    /// One padding fill of a list: see `TTDmFill`. Boxed to keep `Op` in
    /// budget.
    DmFill(Box<TTDmFill>),
    /// One L1-to-L1 word copy of a list: see `TTDmCopyWords`. Boxed to keep
    /// `Op` in budget.
    DmCopyWords(Box<TTDmCopyWords>),
    /// One indexed gather of a list: see `TTDmIndexPick`. Boxed to keep `Op`
    /// in budget.
    DmIndexPick(Box<TTDmIndexPick>),
    /// The list entry that runs the three role programs: see `TTDmKernel`.
    /// Boxed to keep `Op` in budget.
    DmKernel(Box<TTDmKernel>),
    /// Wait for every move before it in the list to complete: the boundary
    /// between entries that reuse each other's L1 slots.
    DmWait,
    /// One row-level SFPU load: see [`TTSfpLoad`]. Boxed to keep `Op` in
    /// budget.
    SfpLoad(Box<TTSfpLoad>),
    /// One row-level SFPU store: see [`TTSfpStore`]. Boxed to keep `Op` in
    /// budget.
    SfpStore(Box<TTSfpStore>),
    /// One lane-register immediate: see [`TTSfpConst`]. Boxed to keep `Op` in
    /// budget.
    SfpConst(Box<TTSfpConst>),
    /// One non-addressing SFPU instruction: see [`TTSfpInsn`].
    Sfp(TTSfpInsn),
    /// Back-pressure before a `Post`: hold `before` and B1 until `sem`'s
    /// counter is below its maximum (`SEMWAIT` C1). A counting hand-off on a
    /// 4-bit counter would otherwise wrap when one role runs more than fifteen
    /// epochs ahead of the other; this is the producer's half of the bound.
    WaitBelowMax {
        /// Semaphore the wait is on.
        sem: u8,
        /// The producer role the wait holds back.
        before: u8,
        /// The role that runs the `SEMWAIT`.
        role: u8,
    },
    /// One recorded row loop (`REPLAY`): the next `len` instruction words of
    /// this role run once while being recorded, then `count - 1` more times.
    /// The body's last `Dst` access steps by two rows per iteration through
    /// the row address modifier, so the body is written for row group 0 and
    /// replayed for the rest. `len` is a **word** count (a 32-bit `SFPLOADI`
    /// constant is two words) and must fit the replay buffer; `count` is 32
    /// for a 64-row tile. Operand-free: the body follows this marker.
    Replay {
        /// Body length in instruction words, including two-word constants.
        len: u16,
        /// How many times the body runs (the recording run included).
        count: u16,
    },
    /// The compute core's T0 program: raw instruction words in emission
    /// order, already encoded by codegen, plus the block repeats the runner
    /// pushes (`mailbox::loops`) — a loop over tiles is one copy of the body
    /// and a table entry, like [`TTOp::Math`]'s. The launch pushes them into
    /// the role's program slot; nothing decodes them. Boxed to keep `Op` in
    /// budget.
    Unpack(Box<TTRoleProgram>),
    /// The compute core's T1 program: its instruction words plus the block
    /// repeats the runner pushes (`mailbox::loops`, at most
    /// `loops::MAX` entries of `[start, len, count]` word ranges). Codegen
    /// emits a row loop longer than the replay buffer this way: the body
    /// appears once and the runner repeats it, so a fused body of any length
    /// costs one copy in the program. Boxed to keep `Op` in budget.
    Math(Box<TTRoleProgram>),
    /// The compute core's T2 program. See [`TTOp::Unpack`].
    Pack(Box<TTRoleProgram>),
    /// The reader's program on RISCV B: RV32IM words, emitted by codegen,
    /// ending in a return. The loop is a branch in the code; the address is
    /// computed from the loop at run time. Boxed to keep `Op` in budget.
    Reader(Box<TTRoleProgram>),
    /// The writer's program on RISCV NC: see [`TTOp::Reader`].
    Writer(Box<TTRoleProgram>),
    /// CB sync, producer side: open `n` back slots of `cb` for writing. Placed
    /// by `tt_sync_cbs`; effect-only, like [`TTOp::EndReader`].
    ReserveBack {
        /// Circular buffer.
        cb: OpId,
        /// Slots to open.
        n: u8,
    },
    /// CB sync, producer side: publish `n` written slots.
    PushBack {
        /// Circular buffer.
        cb: OpId,
        /// Slots to publish.
        n: u8,
    },
    /// CB sync, consumer side: block until `n` front slots hold data.
    WaitFront {
        /// Circular buffer.
        cb: OpId,
        /// Slots to wait for.
        n: u8,
    },
    /// CB sync, consumer side: release `n` consumed slots.
    PopFront {
        /// Circular buffer.
        cb: OpId,
        /// Slots to release.
        n: u8,
    },
    /// TT copy: DRAM<->CB traffic, CB->DRAM drain, and the CB->DST
    /// unpack. Both indices are lane-invariant bases (unrolls stripped
    /// at lowering); the size carries the transfer, never the index
    /// shape. A Register `dst` is the unpack destination; `dst_idx` is
    /// unused there. Effect-only like [`Op::Copy`]: a DCE root, never
    /// CSE'd.
    Copy {
        /// Source buffer (DRAM param or Circular storage).
        src: OpId,
        /// Source base index (unroll-free).
        src_idx: OpId,
        /// Destination buffer (Circular storage or DRAM param).
        dst: OpId,
        /// Destination base index (unroll-free).
        dst_idx: OpId,
        /// Transfer size in bytes.
        bytes: u32,
    },
    /// `UNPACR`: unpacker `unit` (0 = SrcA, 1 = SrcB) moves `count` datums of
    /// the tile image at `src` (a Circular storage) into its `Src` bank,
    /// starting at the face the preceding [`TTOp::Setadczw`] selected. The
    /// base address is `src`'s L1 placement, so the operand stays symbolic.
    /// Effect-only: a DCE root, never deduplicated.
    Unpacr {
        /// Source circular buffer (the tile image in L1).
        src: OpId,
        /// Unpacker: 0 = SrcA (SEC0), 1 = SrcB (SEC1).
        unit: u8,
        /// Datums to unpack (`count - 1` in the instruction's ADC).
        count: u32,
        /// `FlipSrc`: hand the bank to the Matrix Unit when the operand is
        /// complete (`Banks::unpack_a`/`unpack_b` set it; a partial fill does
        /// not).
        flip: bool,
    },
    /// `SETADCZW`: select face `face` of the source tile for unpacker `unit`.
    Setadczw {
        /// Unpacker: 0 = SrcA, 1 = SrcB.
        unit: u8,
        /// Face index within the tile image.
        face: u32,
    },
    /// `SETADCX`: the datum range `first..first + count` unpacker `unit`
    /// walks, set once per tile rather than before every unpack.
    Setadcx {
        /// Unpacker: 0 = SrcA, 1 = SrcB.
        unit: u8,
        /// First datum (a multiple of four).
        first: u32,
        /// Datums in the run.
        count: u32,
    },
    /// `SETRWC`: point the `Dst` accumulator at `dst`'s row. `Setrwc` is also
    /// where the operand rows and the fidelity phase are set; zyx only ever
    /// moves the accumulator, so the rest stays at its prelude value.
    Setrwc {
        /// Accumulator (Register) storage.
        dst: OpId,
    },
    /// `ZEROACC`: the Matrix Unit's accumulator clear. `mode` 3 is `CLR_ALL`
    /// (every `Dst` row) — the fused-kernel hazard the notes track.
    Zeroacc {
        /// Zero mode.
        mode: u8,
        /// Row count for the ranged modes.
        imm10: u32,
    },
    /// `MVMUL`: one Matrix Unit multiply of `SrcA` by `SrcB` into `Dst` row
    /// `dst_row` of the current `SETRWC` base, advancing by `addr_mod`
    /// (fidelity phase, half, group, fi — see `crate::codegen::tenstorrent::matmul::MATH_AM_*`).
    Mvmul {
        /// `Dst` row within the accumulator base.
        dst_row: u32,
        /// Address modifier.
        addr_mod: u8,
        /// Hand `SrcA` back after the multiply (`Banks::mvmul_release_a`).
        flip_a: bool,
        /// Hand `SrcB` back after the multiply (`Banks::mvmul_release_b`).
        flip_b: bool,
        /// `BroadcastSrcBRow`: every lane reads `SrcB` row 0 (row broadcast).
        broadcast_b: bool,
    },
    /// `SETC16`: one `ThreadConfig` field. This is how the matmul's address
    /// modifiers, fidelity phase, and `ALU_ACC_CTRL` bits reach the math
    /// thread; `field` is the metalium `thread::*` field id.
    Setc16 {
        /// `ThreadConfig` field address (`tt_isa::cfg::thread::*`).
        field: u32,
        /// Value to write.
        value: u16,
    },
    /// `SETC16`-family word write for the `Config` fields that are not
    /// `ThreadConfigField`s (`alu::*`, `thcon::*`): the whole word `addr`
    /// staged to `value` (`ConfigWords::seed`, then the `WRCFG` program).
    /// The caller computes the word, so the IR stays raw.
    Setcfg {
        /// `Config` word address (`field.addr32()`).
        addr: u32,
        /// The complete word value.
        value: u32,
    },
    /// `SEMPOST`: post `sem` (index `0..8`) once `role`'s unit has drained,
    /// like `tt_isa::sync::post_after`. `role` names the stream the words join
    /// (0 = T0 unpack, 1 = T1 math, 2 = T2 pack): a semaphore op sits between
    /// the two roles it hands work between, so the stream cannot be inferred
    /// from the ops around it.
    Post {
        /// Semaphore index.
        sem: u8,
        /// Stream: 0 = T0, 1 = T1, 2 = T2.
        role: u8,
    },
    /// `SEMWAIT` + `SEMGET`: take `sem` — wait for it, then zero it —
    /// holding the `before` blocks until it was posted (`sync::take`).
    Wait {
        /// Semaphore index.
        sem: u8,
        /// `tt_isa::backend::Before` mask.
        before: u8,
        /// Stream: 0 = T0, 1 = T1, 2 = T2.
        role: u8,
    },
    /// Semaphore initial state for this run. Not an instruction: `compile`
    /// folds it into the program's semaphore init list (the concurrent
    /// schedule's), which is why it carries the value the role protocol
    /// expects.
    Semset {
        /// Semaphore index, `0..8`.
        sem: u8,
        /// Value the semaphore starts at.
        value: u8,
    },
    /// Lowered elementwise SFPU op. `kind` is the metalium opcode
    /// (`crate::codegen::tenstorrent::kind` or `crate::codegen::tenstorrent::sfpu::ops::kind_sfpu`), so one
    /// variant covers the whole elementwise op surface: adding an op is a
    /// table entry in lowering, not a new variant or a new match arm.
    /// `a` is the first DST input, `b` the second (`OpId::NULL` for a
    /// unary op), and `slot` the DST result. Effect-only: a DCE root,
    /// pinned by LICM, never deduplicated.
    Sfpu {
        /// Metalium elementwise opcode.
        kind: u32,
        /// First (Register) input.
        a: OpId,
        /// Second (Register) input, or `OpId::NULL` for a unary op.
        b: OpId,
        /// Result (Register) storage.
        slot: OpId,
    },
}

/// Boxed payload of [`Op::Kernel`]: a custom kernel boundary's inputs,
/// output `(class, shape, dtype)` triples, owning program, and measured
/// timing. Boxed so `Op` keeps its 24-byte budget.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, SerBin)]
pub struct CustomKernel {
    pub inputs: Box<[OpId]>,
    pub outputs: Box<[(OpId, OpId, DType)]>,
    pub program_id: ProgramId,
    pub time: u64,
}

impl Op {
    /// The [`GPUOp`] payload if this is an [`Op::GPU`] descriptor op.
    /// Backends decode rendered kernels positionally with this (`if let`
    /// rather than a 37-arm match: every non-GPU variant is `None` by
    /// construction, and a future variant is `None` too).
    pub fn as_gpu(&self) -> Option<&GPUOp> {
        if let Op::GPU(g) = self { Some(g) } else { None }
    }

    /// Position of the variant in declaration order. Used to keep the manual
    /// `Ord` identical to the previously derived one (existing variants keep
    /// declaration order; the graph-only variants are appended last).
    fn disc(&self) -> u8 {
        match self {
            Op::Const(_) => 0,
            Op::Param { .. } => 1,
            Op::Cast { .. } => 2,
            Op::Bitcast { .. } => 3,
            Op::Unary { .. } => 4,
            Op::Binary { .. } => 5,
            Op::Stack { .. } => 6,
            Op::Storage { .. } => 7,
            Op::Store { .. } => 8,
            Op::Load { .. } => 9,
            Op::Copy { .. } => 13,
            Op::Range { .. } => 10,
            Op::Loop { .. } => 11,
            Op::EndLoop => 12,
            Op::Index { .. } => 16,
            Op::Barrier => 17,
            Op::Wmma { .. } => 18,
            Op::TT(_) => 19,
            Op::Source { .. } => 20,
            Op::Asm { .. } => 23,
            Op::Reduce { .. } => 25,
            Op::After { .. } => 26,
            Op::ToDevice { .. } => 27,
            Op::Contiguous { .. } => 28,
            Op::Program { .. } => 29,
            Op::Kernel(_) => 30,
            Op::Reshape { .. } => 31,
            Op::Expand { .. } => 32,
            Op::Permute { .. } => 33,
            Op::Flip { .. } => 34,
            Op::Pad { .. } => 35,
            Op::Narrow { .. } => 36,
            Op::GPU(_) => 37,
            Op::Spirv(_) => 38,
            Op::PTX(_) => 39,
            // Appended past every existing discriminant: inserting an
            // in-declaration-order number would reorder existing variants.
            Op::Unroll { .. } => 41,
            Op::Select { .. } => 42,
            Op::Detach { .. } => 43,
        }
    }
}

impl PartialEq for Op {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Op::Const(a), Op::Const(b)) => a == b,
            (Op::Param { dtype: ad, kind: ak, shape: as_, id: ac }, Op::Param { dtype: bd, kind: bk, shape: bs, id: bc }) => {
                ad == bd && ak == bk && as_ == bs && ac == bc
            }
            (Op::Cast { x: a, dtype: ad }, Op::Cast { x: b, dtype: bd }) => a == b && ad == bd,
            (Op::Bitcast { x: a, dtype: ad }, Op::Bitcast { x: b, dtype: bd }) => a == b && ad == bd,
            (Op::Unary { x: a, uop: au }, Op::Unary { x: b, uop: bu }) => a == b && au == bu,
            (Op::Binary { x: a, y: ay, bop: ab }, Op::Binary { x: b, y: by, bop: bb }) => a == b && ay == by && ab == bb,
            (Op::Select { cond: a, if_true: at, if_false: af }, Op::Select { cond: b, if_true: bt, if_false: bf }) => {
                a == b && at == bt && af == bf
            }
            (Op::Stack { ops: a }, Op::Stack { ops: b }) => a == b,
            (Op::Storage { dtype: ad, scope: as_, len: al, id: ai }, Op::Storage { dtype: bd, scope: bs, len: bl, id: bi }) => {
                ad == bd && as_ == bs && al == bl && ai == bi
            }
            (Op::Store { dst: ad, dst_idx: adi, src: as_, .. }, Op::Store { dst: bd, dst_idx: bdi, src: bs, .. }) => {
                ad == bd && adi == bdi && as_ == bs
            }
            (Op::Load { src: a, src_idx: ai, .. }, Op::Load { src: b, src_idx: bi, .. }) => a == b && ai == bi,
            (
                Op::Copy { src: as_, src_idx: asi, dst: ad, dst_idx: adi, .. },
                Op::Copy { src: bs, src_idx: bsi, dst: bd, dst_idx: bdi, .. },
            ) => as_ == bs && asi == bsi && ad == bd && adi == bdi,
            (Op::Range { axis: aa, kind: ak }, Op::Range { axis: ba, kind: bk }) => aa == ba && ak == bk,
            (Op::Loop { len: a }, Op::Loop { len: b }) => a == b,
            (Op::EndLoop, Op::EndLoop) => true,
            (Op::Index { vec: a, idx: ai }, Op::Index { vec: b, idx: bi }) => a == b && ai == bi,
            (Op::Barrier, Op::Barrier) => true,
            (
                Op::Wmma { dims: ad, layout: al, dtype: at, a, b: ab, c: ac },
                Op::Wmma { dims: bd, layout: bl, dtype: bt, a: ba, b: bb, c: bc },
            ) => ad == bd && al == bl && at == bt && a == ba && ab == bb && ac == bc,
            (Op::TT(a), Op::TT(b)) => a == b,
            (Op::GPU(a), Op::GPU(b)) => a == b,
            (Op::Spirv(a), Op::Spirv(b)) => a == b,
            (Op::PTX(a), Op::PTX(b)) => a == b,
            (Op::Asm { asm: aa, ops: ao }, Op::Asm { asm: ba, ops: bo }) => aa == ba && ao == bo,
            (Op::Reduce { x: a, rop: ar, reduce_axis: aa }, Op::Reduce { x: b, rop: br, reduce_axis: ba }) => {
                a == b && ar == br && aa == ba
            }
            // Graph-only ops mirror the former `Node` equality: `After` never
            // merges; `time` is ignored on the boundary ops.
            (Op::After { .. }, Op::After { .. }) => false,
            (Op::ToDevice { x: a, device: ad, .. }, Op::ToDevice { x: b, device: bd, .. }) => a == b && ad == bd,
            (Op::Contiguous { x: a }, Op::Contiguous { x: b }) => a == b,
            (Op::Detach { x: a }, Op::Detach { x: b }) => a == b,
            (Op::Program { inputs: ai, outputs: ao, info: a }, Op::Program { inputs: bi, outputs: bo, info: b }) => {
                ai == bi && ao == bo && a.0 == b.0
            }
            (Op::Kernel(_), Op::Kernel(_)) => false,
            (Op::Reshape { x: a, shape: as_ }, Op::Reshape { x: b, shape: bs }) => a == b && as_ == bs,
            (Op::Expand { x: a, shape: as_ }, Op::Expand { x: b, shape: bs }) => a == b && as_ == bs,
            (Op::Permute { x: a, axes: aa }, Op::Permute { x: b, axes: ba }) => a == b && aa.as_slice() == ba.as_slice(),
            (Op::Flip { x: a, axes: aa }, Op::Flip { x: b, axes: ba }) => a == b && aa.as_slice() == ba.as_slice(),
            (Op::Pad { x: a, axis: aa, lp: al, len: alen }, Op::Pad { x: b, axis: ba, lp: bl, len: blen }) => {
                a == b && aa == ba && al == bl && alen == blen
            }
            (Op::Narrow { x: a, axis: aa, start: as_, len: al }, Op::Narrow { x: b, axis: ba, start: bs, len: bl }) => {
                a == b && aa == ba && as_ == bs && al == bl
            }
            (Op::Unroll { axis: aa, len: al }, Op::Unroll { axis: ba, len: bl }) => aa == ba && al == bl,
            // Different variants are never equal (Hash carries no
            // discriminant, so cross-variant collisions land here).
            _ => false,
        }
    }
}

impl Eq for Op {}

impl Hash for Op {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.disc().hash(state);
        match self {
            Op::Const(c) => c.hash(state),
            // `id` is included: it is canonicalized to head-order
            // ordinals before any cache lookup, so structurally
            // identical kernels hash identically and share programs.
            // `Eq` compares it too, which gives CSE same-id dedup.
            Op::Param { dtype, kind, shape, id } => {
                dtype.hash(state);
                kind.hash(state);
                shape.hash(state);
                id.hash(state);
            }
            Op::Cast { x, dtype } => {
                x.hash(state);
                dtype.hash(state);
            }
            Op::Bitcast { x, dtype } => {
                x.hash(state);
                dtype.hash(state);
            }
            Op::Unary { x, uop } => {
                x.hash(state);
                uop.hash(state);
            }
            Op::Binary { x, y, bop } => {
                x.hash(state);
                y.hash(state);
                bop.hash(state);
            }
            Op::Select { cond, if_true, if_false } => {
                cond.hash(state);
                if_true.hash(state);
                if_false.hash(state);
            }
            Op::Stack { ops } => ops.hash(state),
            Op::Storage { dtype, scope, len, id } => {
                dtype.hash(state);
                scope.hash(state);
                len.hash(state);
                // `id` is `0` until the allocator runs, so hashing it keeps
                // the pre-allocation identity ("a request of this shape") and
                // separates two registers after allocation.
                id.hash(state);
            }
            Op::Store { dst, dst_idx, src } => {
                dst.hash(state);
                dst_idx.hash(state);
                src.hash(state);
            }
            Op::Load { src, src_idx } => {
                src.hash(state);
                src_idx.hash(state);
            }
            Op::Copy { src, src_idx, dst, dst_idx } => {
                src.hash(state);
                src_idx.hash(state);
                dst.hash(state);
                dst_idx.hash(state);
            }
            Op::Range { axis, kind } => {
                axis.hash(state);
                kind.hash(state);
            }
            Op::Loop { len } => len.hash(state),
            Op::EndLoop | Op::Barrier => {}
            Op::Index { vec, idx } => {
                vec.hash(state);
                idx.hash(state);
            }
            Op::Wmma { dims, layout, dtype, a, b, c } => {
                dims.hash(state);
                layout.hash(state);
                dtype.hash(state);
                a.hash(state);
                b.hash(state);
                c.hash(state);
            }
            Op::TT(t) => t.hash(state),
            Op::GPU(g) => g.hash(state),
            Op::Spirv(s) => s.hash(state),
            Op::PTX(p) => p.hash(state),
            Op::Asm { asm, ops } => {
                asm.hash(state);
                ops.hash(state);
            }
            Op::Reduce { x, rop, reduce_axis } => {
                x.hash(state);
                rop.hash(state);
                reduce_axis.hash(state);
            }
            Op::After { x, dep } => {
                x.hash(state);
                dep.hash(state);
            }
            Op::ToDevice { x, device, .. } => {
                x.hash(state);
                device.hash(state);
            }
            Op::Contiguous { x } => x.hash(state),
            Op::Detach { x } => x.hash(state),
            Op::Unroll { axis, len } => {
                axis.hash(state);
                len.hash(state);
            }
            Op::Program { inputs, outputs, info } => {
                inputs.hash(state);
                outputs.hash(state);
                info.0.hash(state);
            }
            Op::Kernel(c) => {
                c.inputs.hash(state);
                c.outputs.hash(state);
                c.program_id.hash(state);
            }
            Op::Reshape { x, shape } => {
                x.hash(state);
                shape.hash(state);
            }
            Op::Expand { x, shape } => {
                x.hash(state);
                shape.hash(state);
            }
            Op::Permute { x, axes } => {
                x.hash(state);
                axes.as_slice().hash(state);
            }
            Op::Flip { x, axes } => {
                x.hash(state);
                axes.as_slice().hash(state);
            }
            Op::Pad { x, axis, lp, len } => {
                x.hash(state);
                axis.hash(state);
                lp.hash(state);
                len.hash(state);
            }
            Op::Narrow { x, axis, start, len } => {
                x.hash(state);
                axis.hash(state);
                start.hash(state);
                len.hash(state);
            }
            Op::Source(src) => {
                src.hash(state);
            }
        }
    }
}

impl PartialOrd for Op {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Op {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        match (self, other) {
            (Op::Const(a), Op::Const(b)) => a.cmp(b),
            (Op::Param { dtype: ad, kind: ak, shape: as_, id: ac }, Op::Param { dtype: bd, kind: bk, shape: bs, id: bc }) => {
                (ad, ak, as_, ac).cmp(&(bd, bk, bs, bc))
            }
            (Op::Cast { x: a, dtype: ad }, Op::Cast { x: b, dtype: bd }) => (a, ad).cmp(&(b, bd)),
            (Op::Bitcast { x: a, dtype: ad }, Op::Bitcast { x: b, dtype: bd }) => (a, ad).cmp(&(b, bd)),
            (Op::Unary { x: a, uop: au }, Op::Unary { x: b, uop: bu }) => (a, au).cmp(&(b, bu)),
            (Op::Binary { x: a, y: ay, bop: ab }, Op::Binary { x: b, y: by, bop: bb }) => (a, ay, ab).cmp(&(b, by, bb)),
            (Op::Select { cond: a, if_true: at, if_false: af }, Op::Select { cond: b, if_true: bt, if_false: bf }) => {
                (a, at, af).cmp(&(b, bt, bf))
            }
            (Op::Stack { ops: a }, Op::Stack { ops: b }) => a.cmp(b),
            (Op::Storage { dtype: ad, scope: as_, len: al, id: ai }, Op::Storage { dtype: bd, scope: bs, len: bl, id: bi }) => {
                (ad, as_, al, ai).cmp(&(bd, bs, bl, bi))
            }
            (Op::Store { dst: ad, dst_idx: adi, src: as_ }, Op::Store { dst: bd, dst_idx: bdi, src: bs }) => {
                (ad, adi, as_).cmp(&(bd, bdi, bs))
            }
            (Op::Load { src: a, src_idx: ai }, Op::Load { src: b, src_idx: bi }) => (a, ai).cmp(&(b, bi)),
            (
                Op::Copy { src: as_, src_idx: asi, dst: ad, dst_idx: adi },
                Op::Copy { src: bs, src_idx: bsi, dst: bd, dst_idx: bdi },
            ) => (as_, asi, ad, adi).cmp(&(bs, bsi, bd, bdi)),
            (Op::Range { axis: aa, kind: ak }, Op::Range { axis: ba, kind: bk }) => (aa, ak).cmp(&(ba, bk)),
            (Op::Loop { len: a }, Op::Loop { len: b }) => a.cmp(b),
            (Op::EndLoop, Op::EndLoop) | (Op::Barrier, Op::Barrier) => std::cmp::Ordering::Equal,
            (Op::Index { vec: a, idx: ai }, Op::Index { vec: b, idx: bi }) => (a, ai).cmp(&(b, bi)),
            (
                Op::Wmma { dims: ad, layout: al, dtype: at, a, b: ab, c: ac },
                Op::Wmma { dims: bd, layout: bl, dtype: bt, a: ba, b: bb, c: bc },
            ) => (ad, al, at, a, ab, ac).cmp(&(bd, bl, bt, ba, bb, bc)),
            (Op::TT(a), Op::TT(b)) => a.cmp(b),
            (Op::Asm { asm: aa, ops: ao }, Op::Asm { asm: ba, ops: bo }) => (aa, ao).cmp(&(ba, bo)),
            (Op::Reduce { x: a, rop: ar, reduce_axis: aa }, Op::Reduce { x: b, rop: br, reduce_axis: ba }) => {
                (a, ar, aa).cmp(&(b, br, ba))
            }
            // `Ord` ignores exactly what `Eq` ignores, so `Eq`-equal values
            // always compare `Equal`. `After`/`Custom` never compare equal,
            // so any two same-discriminant values order `Equal`.
            (Op::After { .. }, Op::After { .. }) => std::cmp::Ordering::Equal,
            (Op::ToDevice { x: a, device: ad, .. }, Op::ToDevice { x: b, device: bd, .. }) => (a, ad).cmp(&(b, bd)),
            (Op::Contiguous { x: a }, Op::Contiguous { x: b }) => a.cmp(b),
            (Op::Detach { x: a }, Op::Detach { x: b }) => a.cmp(b),
            (Op::Program { inputs: ai, outputs: ao, info: a }, Op::Program { inputs: bi, outputs: bo, info: b }) => {
                (ai, ao, a.0).cmp(&(bi, bo, b.0))
            }
            (Op::Kernel(_), Op::Kernel(_)) => std::cmp::Ordering::Equal,
            (Op::Reshape { x: a, shape: as_ }, Op::Reshape { x: b, shape: bs }) => (a, as_).cmp(&(b, bs)),
            (Op::Expand { x: a, shape: as_ }, Op::Expand { x: b, shape: bs }) => (a, as_).cmp(&(b, bs)),
            (Op::Permute { x: a, axes: aa }, Op::Permute { x: b, axes: ba }) => (a, aa.as_slice()).cmp(&(b, ba.as_slice())),
            (Op::Flip { x: a, axes: aa }, Op::Flip { x: b, axes: ba }) => (a, aa.as_slice()).cmp(&(b, ba.as_slice())),
            (Op::Pad { x: a, axis: aa, lp: al, len: alen }, Op::Pad { x: b, axis: ba, lp: bl, len: blen }) => {
                (a, aa, al, alen).cmp(&(b, ba, bl, blen))
            }
            (Op::Narrow { x: a, axis: aa, start: as_, len: al }, Op::Narrow { x: b, axis: ba, start: bs, len: bl }) => {
                (a, aa, as_, al).cmp(&(b, ba, bs, bl))
            }
            // Only different-variant pairs reach here: order by discriminant.
            (a, b) => a.disc().cmp(&b.disc()),
        }
    }
}

/// Which dimension a tile reduce collapses within each 32x32
/// tile (broadcast replication is [`Op::Expand`] now).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, SerBin, DeBin)]
pub enum TileDim {
    /// Reduce over rows: one value per row, carried in the result
    /// tile's first column.
    Row,
    /// Reduce over columns: one value per column, carried in the
    /// result tile's first row.
    Col,
    /// Whole tile to/from a single scalar.
    Scalar,
}

/// Scope of index. Index is like loop, but purely parallel acess
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, SerBin, DeBin)]
pub enum RangeKind {
    /// Group scope. Represents blocks in cuda, cores in CPU and tenstorrent.
    Group(OpId),
    /// Local scope. Represents cuda threads. Local axis 0 is the hardware
    /// warp (consecutive thread ids within a warp): its length is
    /// `DeviceInfo::warp_size` whenever lane semantics are used. There is no
    /// separate warp kind — the lane id is the axis-0 local value.
    Local(u32),
}

impl std::fmt::Display for RangeKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RangeKind::Group(x) => f.write_fmt(format_args!("group_r{x}")),
            RangeKind::Local(x) => f.write_fmt(format_args!("local_{x}")),
        }
    }
}

/// Unary operations for element-wise kernel transformations.
///
/// These operations are applied to a single input tensor.
///
/// # Variants
#[derive(Debug, PartialEq, Eq, PartialOrd, Ord, Clone, Copy, Hash, SerBin, DeBin)]
pub enum UOp {
    /// Negation: -x
    Neg,
    /// Logical NOT: !x
    Not,
    /// Bitwise NOT: ~x
    BitNot,
    /// Exponential with base 2: 2^x
    Exp2,
    /// Logarithm with base 2: log2(x)
    Log2,
    /// Reciprocal: 1/x
    Reciprocal,
    /// Square root: sqrt(x)
    Sqrt,
    /// Reciprocal square root: 1/sqrt(x)
    Rsqrt,
    /// Sine: sin(x)
    Sin,
    /// Cosine: cos(x)
    Cos,
    /// Floor: floor(x)
    Floor,
    /// Truncate toward zero: trunc(x)
    Trunc,
    /// Absolute value: |x|
    Abs,
}

#[derive(Debug, PartialEq, Eq, PartialOrd, Ord, Clone, Copy, Hash, SerBin, DeBin)]
/// Binary operations for element-wise or reduction kernel operations.
///
/// These operations take two input tensors and produce an output.
///
/// # Variants
pub enum BOp {
    /// Addition: x + y
    Add,
    /// Subtraction: x - y
    Sub,
    /// Multiplication: x * y
    Mul,
    /// Division: x / y
    Div,
    /// Power: x^y
    Pow,
    /// Modulo: x % y
    Mod,
    /// Compare less than: x < y
    Cmplt,
    /// Compare greater than: x > y
    Cmpgt,
    /// Compare greater than or equal: x >= y
    Cmpge,
    /// Maximum: max(x, y)
    Max,
    /// Bitwise OR: x | y
    Or,
    /// Bitwise AND: x & y
    And,
    /// Bitwise XOR: x ^ y
    BitXor,
    /// Bitwise OR: x | y
    BitOr,
    /// Bitwise AND: x & y
    BitAnd,
    /// Left shift: x << y
    BitShiftLeft,
    /// Right shift: x >> y
    BitShiftRight,
    /// Not equal: x != y
    NotEq,
    /// Equal: x == y
    Eq,
}

impl BOp {
    /// Returns true if the binary operation is associative:
    /// `(a op b) op c == a op (b op c)`.
    pub const fn is_associative(self) -> bool {
        use BOp::{Add, And, BitAnd, BitOr, BitShiftLeft, BitShiftRight, BitXor, Max, Mul, Or};
        matches!(self, Add | Mul | And | Or | BitXor | BitAnd | BitOr | BitShiftLeft | BitShiftRight | Max)
    }

    /// Returns true if the binary operation is commutative:
    /// `a op b == b op a`.
    pub const fn is_commutative(self) -> bool {
        use BOp::{Add, And, BitAnd, BitOr, BitXor, Max, Mul, Or};
        matches!(self, Add | Mul | And | Or | BitXor | BitAnd | BitOr | Max)
    }

    /// Returns true if the operation produces a boolean result.
    pub const fn returns_bool(self) -> bool {
        use BOp::{And, Cmpge, Cmpgt, Cmplt, Eq, NotEq, Or};
        matches!(self, Cmpgt | Cmpge | Cmplt | NotEq | Eq | And | Or)
    }
}

/// Matrix multiply dimensions for tensor core operations.
///
/// Represents the shape (m, n, k) for matrix multiplication.
#[allow(non_camel_case_types)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, SerBin, DeBin)]
pub enum MMADims {
    /// 8x8 with k=16
    m8n8k16,
    /// 16x8 with k=8
    m16n8k8,
    /// 16x8 with k=16
    m16n8k16,
    /// 32x8 with k=16 (int8)
    m32n8k16,
    /// 8x32 with k=16 (int8)
    m8n32k16,
    /// 8x8 with k=32 (int4)
    m8n8k32,
    /// 8x8 with k=128 (b1)
    m8n8k128,
}

impl MMADims {
    /// Decompose MMAD dimensions into m, n, k components.
    pub const fn decompose_mnk(self) -> (u64, u64, u64) {
        match self {
            MMADims::m8n8k16 => (8, 8, 16),
            MMADims::m16n8k8 => (16, 8, 8),
            MMADims::m16n8k16 => (16, 8, 16),
            MMADims::m32n8k16 => (32, 8, 16),
            MMADims::m8n32k16 => (8, 32, 16),
            MMADims::m8n8k32 => (8, 8, 32),
            MMADims::m8n8k128 => (8, 8, 128),
        }
    }
}

/// Memory layout for tensor core matrix operands.
///
/// Describes how matrix data is stored in memory.
#[allow(non_camel_case_types)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, SerBin, DeBin)]
pub enum MMALayout {
    /// Row-major for A, column-major for B — the only layout `mma.sync` accepts
    row_col,
}

/// Data type for matrix multiply operations.
#[allow(non_camel_case_types)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, SerBin, DeBin)]
pub enum MMADType {
    /// FP16 input with FP32 accumulator
    f16_f16_f16_f32,
    /// FP16 input with FP16 accumulator
    f16_f16_f16_f16,
    /// 8 bit signed integer input with 32 bit signed integer accumulator
    s8_s8_s32_s32,
    /// 4 bit signed integer input with 32 bit signed integer accumulator
    s4_s4_s32_s32,
    /// 1 bit input with 32 bit signed integer accumulator, XOR + popc reduction
    b1_b1_s32_xor_popc,
    /// 1 bit input with 32 bit signed integer accumulator, AND + popc reduction
    b1_b1_s32_and_popc,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, SerBin)]
pub struct OpLinked {
    pub prev: OpId,
    pub next: OpId, // Use Vec<OpId> instead for egraph
    pub op: Op,
}

const _: () = assert!(core::mem::size_of::<Op>() == 24);
const _: () = assert!(core::mem::size_of::<OpLinked>() == 32);

/// Operation ID for kernel operations.
///
/// This is a unique identifier for each operation in the kernel IR.
#[derive(Default, Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, SerBin, DeBin)]
pub struct OpId(pub(crate) u32);

impl OpId {
    /// NULL
    pub const NULL: Self = Self(u32::MAX);

    /// Check if this OpId is null.
    pub const fn is_null(self) -> bool {
        self.0 == u32::MAX
    }
}

impl std::fmt::Display for OpId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::Display::fmt(&self.0, f)
    }
}

impl From<usize> for OpId {
    fn from(value: usize) -> Self {
        OpId(value as u32)
    }
}

impl From<OpId> for usize {
    fn from(value: OpId) -> usize {
        value.0 as usize
    }
}

impl SlabId for OpId {
    const ZERO: Self = Self(0);
    const NULL: Self = Self(u32::MAX);

    fn inc(&mut self) {
        self.0 += 1;
    }
}

impl Op {
    // TODO use custom non allocating iterator instead of allocating a vec
    #[allow(clippy::match_same_arms)]
    pub(crate) fn parameters(&self) -> impl DoubleEndedIterator<Item = OpId> {
        match self {
            Op::Const { .. } | Op::Storage { .. } | Op::EndLoop | Op::Barrier => {
                vec![]
            }
            &Op::Param { shape, .. } => {
                // Shape is null after linearize
                if shape.is_null() { vec![] } else { vec![shape] }
            }
            &Op::Range { kind, .. } => match kind {
                RangeKind::Group(len) => vec![len],
                RangeKind::Local(_) => vec![],
            },
            &Op::Loop { len, .. } => vec![len],
            &Op::Reshape { x, shape, .. } | &Op::Expand { x, shape } => {
                // Null shape = scalar (see `shape_class`): no shape operand,
                // same as `Param` below.
                if shape.is_null() { vec![x] } else { vec![x, shape] }
            }
            &Op::Permute { x, .. } | &Op::Flip { x, .. } => vec![x],
            &Op::Pad { x, lp, len, .. } => vec![x, lp, len],
            &Op::Narrow { x, start, len, .. } => vec![x, start, len],
            Op::Reduce { x, reduce_axis, .. } => vec![*x, *reduce_axis],
            Op::Unroll { .. } => vec![],

            // dst/src are NEVER null (whole-view stores spell the bare
            // Param); only dst_idx can be null.
            &Op::Store { dst, dst_idx, src } => {
                if dst_idx.is_null() {
                    vec![dst, src]
                } else {
                    vec![dst, dst_idx, src]
                }
            }
            &Op::Copy { src, src_idx, dst, dst_idx } => {
                debug_assert!(!src_idx.is_null());
                vec![src, src_idx, dst, dst_idx]
            }
            Op::Cast { x, .. } => vec![*x],
            Op::Bitcast { x, .. } => vec![*x],
            Op::Unary { x, .. } => vec![*x],
            &Op::Binary { x, y, .. } => vec![x, y],
            &Op::Select { cond, if_true, if_false } => vec![cond, if_true, if_false],
            // src_idx is NEVER null: every load names its element.
            &Op::Load { src, src_idx } => vec![src, src_idx],
            Op::Asm { ops, .. } => ops.iter().copied().collect(),
            &Op::TT(TTOp::Copy { src, src_idx, dst, dst_idx, .. }) => vec![src, src_idx, dst, dst_idx],
            &Op::TT(TTOp::Unpacr { src, .. }) => vec![src],
            &Op::TT(TTOp::ReserveBack { cb, .. })
            | &Op::TT(TTOp::PushBack { cb, .. })
            | &Op::TT(TTOp::WaitFront { cb, .. })
            | &Op::TT(TTOp::PopFront { cb, .. }) => vec![cb],
            &Op::TT(TTOp::Setrwc { dst }) => vec![dst],
            Op::TT(TTOp::Setadczw { .. })
            | Op::TT(TTOp::Setadcx { .. })
            | Op::TT(TTOp::Zeroacc { .. })
            | Op::TT(TTOp::Mvmul { .. })
            | Op::TT(TTOp::Setc16 { .. })
            | Op::TT(TTOp::Setcfg { .. })
            | Op::TT(TTOp::Post { .. })
            | Op::TT(TTOp::Wait { .. })
            | Op::TT(TTOp::WaitBelowMax { .. })
            | Op::TT(TTOp::DmRead(_))
            | Op::TT(TTOp::DmWrite(_))
            | Op::TT(TTOp::DmReadTransposed(_))
            | Op::TT(TTOp::DmReadBroadcastCol(_))
            | Op::TT(TTOp::DmFill(_))
            | Op::TT(TTOp::DmCopyWords(_))
            | Op::TT(TTOp::DmIndexPick(_))
            | Op::TT(TTOp::DmKernel(_))
            | Op::TT(TTOp::DmWait)
            | Op::TT(TTOp::SfpConst(_))
            | Op::TT(TTOp::Sfp(_))
            | Op::TT(TTOp::Replay { .. })
            | Op::TT(TTOp::Semset { .. }) => vec![],
            &Op::TT(TTOp::SfpLoad(ref load)) => vec![load.slot],
            &Op::TT(TTOp::SfpStore(ref store)) => vec![store.slot],
            &Op::TT(TTOp::Sfpu { a, b, slot, .. }) => {
                if b.is_null() {
                    vec![a, slot]
                } else {
                    vec![a, b, slot]
                }
            }
            Op::Stack { ops } => ops.to_vec(),
            &Op::Index { vec, .. } => vec![vec],
            &Op::Wmma { a, b, c, .. } => vec![a, b, c],
            Op::TT(TTOp::EndReader)
            | Op::TT(TTOp::EndCompute)
            | Op::TT(TTOp::TensixGrid(_))
            | Op::TT(TTOp::Unpack(_))
            | Op::TT(TTOp::Math(_))
            | Op::TT(TTOp::Pack(_))
            | Op::TT(TTOp::Reader(_))
            | Op::TT(TTOp::Writer(_)) => {
                vec![]
            }
            Op::After { x, dep } => vec![*x, *dep],
            Op::ToDevice { x, .. } => vec![*x],
            Op::Contiguous { x } => vec![*x],
            Op::Detach { x } => vec![*x],
            Op::Source(_) | Op::Program { .. } | Op::Kernel(_) | Op::GPU(_) => {
                vec![]
            }
            Op::Spirv(_) | Op::PTX(_) => vec![],
        }
        .into_iter()
    }

    #[allow(clippy::match_same_arms)]
    pub(crate) fn parameters_mut(&mut self) -> impl DoubleEndedIterator<Item = &mut OpId> {
        match self {
            Op::Kernel(_)
            | Op::Program { .. }
            | Op::Source(_)
            | Op::GPU(_)
            | Op::Const { .. }
            | Op::Storage { .. }
            | Op::EndLoop
            | Op::Barrier => vec![],
            Op::Contiguous { x, .. } => vec![x],
            Op::Detach { x, .. } => vec![x],
            Op::Spirv(_) | Op::PTX(_) => vec![],
            Op::ToDevice { x, .. } => vec![x],
            Op::After { x, dep } => vec![x, dep],
            Op::Param { shape, .. } => {
                // Shape is null after linearize
                if shape.is_null() { vec![] } else { vec![shape] }
            }
            Op::Range { kind, .. } => match kind {
                RangeKind::Group(len) => vec![len],
                RangeKind::Local(_) => vec![],
            },
            Op::Loop { len, .. } => vec![len],
            Op::Reshape { x, shape, .. } | Op::Expand { x, shape } => {
                // Null shape = scalar (see `shape_class`): no shape operand,
                // same as `Param` below.
                if shape.is_null() { vec![x] } else { vec![x, shape] }
            }
            Op::Permute { x, .. } | Op::Flip { x, .. } => vec![x],
            Op::Pad { x, lp, len, .. } => vec![x, lp, len],
            Op::Narrow { x, start, len, .. } => vec![x, start, len],
            Op::Reduce { x, reduce_axis, .. } => vec![x, reduce_axis],
            Op::Store { dst, dst_idx, src: x } => {
                if dst_idx.is_null() {
                    vec![dst, x]
                } else {
                    vec![dst, dst_idx, x]
                }
            }
            Op::Copy { src, src_idx, dst, dst_idx } => {
                debug_assert!(!src_idx.is_null());
                vec![src, src_idx, dst, dst_idx]
            }
            Op::Cast { x, .. } => vec![x],
            Op::Bitcast { x, .. } => vec![x],
            Op::Unary { x, .. } => vec![x],
            Op::Binary { x, y, .. } => vec![x, y],
            Op::Select { cond, if_true, if_false } => vec![cond, if_true, if_false],
            // src_idx is NEVER null: every load names its element.
            Op::Load { src, src_idx } => vec![src, src_idx],
            Op::Stack { ops } => ops.iter_mut().collect(),
            Op::Unroll { .. } => vec![],
            Op::Index { vec, .. } => vec![vec],
            Op::Wmma { a, b, c, .. } => vec![a, b, c],
            Op::TT(TTOp::EndReader)
            | Op::TT(TTOp::EndCompute)
            | Op::TT(TTOp::TensixGrid(_))
            | Op::TT(TTOp::Unpack(_))
            | Op::TT(TTOp::Math(_))
            | Op::TT(TTOp::Pack(_))
            | Op::TT(TTOp::Reader(_))
            | Op::TT(TTOp::Writer(_)) => {
                vec![]
            }
            Op::Asm { ops, .. } => ops.iter_mut().collect(),
            Op::TT(TTOp::Copy { src, src_idx, dst, dst_idx, .. }) => vec![src, src_idx, dst, dst_idx],
            Op::TT(TTOp::Unpacr { src, .. }) => vec![src],
            Op::TT(TTOp::ReserveBack { cb, .. })
            | Op::TT(TTOp::PushBack { cb, .. })
            | Op::TT(TTOp::WaitFront { cb, .. })
            | Op::TT(TTOp::PopFront { cb, .. }) => vec![cb],
            Op::TT(TTOp::Setrwc { dst }) => vec![dst],
            Op::TT(TTOp::Setadczw { .. })
            | Op::TT(TTOp::Setadcx { .. })
            | Op::TT(TTOp::Zeroacc { .. })
            | Op::TT(TTOp::Mvmul { .. })
            | Op::TT(TTOp::Setc16 { .. })
            | Op::TT(TTOp::Setcfg { .. })
            | Op::TT(TTOp::Post { .. })
            | Op::TT(TTOp::Wait { .. })
            | Op::TT(TTOp::WaitBelowMax { .. })
            | Op::TT(TTOp::DmRead(_))
            | Op::TT(TTOp::DmWrite(_))
            | Op::TT(TTOp::DmReadTransposed(_))
            | Op::TT(TTOp::DmReadBroadcastCol(_))
            | Op::TT(TTOp::DmFill(_))
            | Op::TT(TTOp::DmCopyWords(_))
            | Op::TT(TTOp::DmIndexPick(_))
            | Op::TT(TTOp::DmKernel(_))
            | Op::TT(TTOp::DmWait)
            | Op::TT(TTOp::SfpConst(_))
            | Op::TT(TTOp::Sfp(_))
            | Op::TT(TTOp::Replay { .. })
            | Op::TT(TTOp::Semset { .. }) => vec![],
            &mut Op::TT(TTOp::SfpLoad(ref mut load)) => vec![&mut load.slot],
            &mut Op::TT(TTOp::SfpStore(ref mut store)) => vec![&mut store.slot],
            Op::TT(TTOp::Sfpu { a, b, slot, .. }) => {
                if b.is_null() {
                    vec![a, slot]
                } else {
                    vec![a, b, slot]
                }
            }
        }
        .into_iter()
    }

    /// Check if this operation is a constant.
    pub(crate) const fn is_const(&self) -> bool {
        matches!(self, Op::Cast { .. })
    }

    /// Check if this operation is a load.
    pub(crate) const fn is_load(&self) -> bool {
        matches!(self, Op::Load { .. })
    }

    /// Remap parameter IDs according to a mapping.
    pub(crate) fn remap_params(&mut self, remapping: &Map<OpId, OpId>) {
        for param in self.parameters_mut() {
            if let Some(remapped_id) = remapping.get(param) {
                *param = *remapped_id;
            }
        }
    }
}

impl std::fmt::Display for MemScope {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            MemScope::Global => "global",
            MemScope::Local => "local",
            MemScope::Register => "reg",
            MemScope::Circular => "cb",
        })
    }
}

impl std::fmt::Display for ParamKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            ParamKind::Variable => "var",
            ParamKind::Global => "global",
            ParamKind::GlobalMut => "global mut",
        })
    }
}
