use nanoserde::{DeBin, SerBin};

use crate::dtype::Constant;
use crate::kernel::{MemLayout, MemScope};
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

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, SerBin)]
pub enum Op {
    // ops that exist in both
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
        dtype: DType,
        kind: ParamKind,
        shape: OpId,
    },
    Cast {
        x: OpId,
        dtype: DType,
    },
    Unary {
        x: OpId,
        uop: UOp,
    },
    // For binary ops, next of x is y, then next of y is the binary op
    Binary {
        x: OpId,
        y: OpId,
        bop: BOp,
    },
    // Vectorization, YAY!
    Stack {
        ops: Box<[OpId]>,
    },

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
        dtype: DType,
        scope: MemScope,
        len: Dim,
    },
    Store {
        dst: OpId,
        src: OpId,
        index: OpId,
        layout: MemLayout,
    },
    Load {
        src: OpId,
        index: OpId,
        layout: MemLayout,
    },
    // Like loop, but for dimensions always executed in parallel
    Index {
        axis: u32,
        kind: IdxKind,
    },
    // Control flow
    Loop {
        len: OpId,
    },
    EndLoop,
    If {
        condition: OpId, // must be boolean variable
    },
    EndIf,
    // fused multiply add
    Mad {
        x: OpId,
        y: OpId,
        z: OpId,
    },
    Devectorize {
        vec: OpId,
        idx: usize,
    }, // select a single value from a vector
    Barrier,
    // fused matmul, a, b, c are fragments, each is a vector, c is accumulator, returns new accumulated vector d
    Wmma {
        dims: MMADims,
        layout: MMALayout,
        dtype: MMADType,
        a: OpId,
        b: OpId,
        c: OpId,
    },
    /// Hardware reduce_tile: collapses a 32x32 tile accumulator to a scalar.
    ReduceTile {
        x: OpId,
        rop: BOp,
        kind: TileReduceKind,
    },
    MatmulTile {
        x: OpId,
        y: OpId,
    },
    TransposeTile {
        x: OpId,
    },
    // For backend specific assembly
    Asm {
        asm: TinyString,
        ops: TinyVec<OpId>,
    },

    // ops that exist before linearize and linearize converts them into these ops: index, loop and load
    Move {
        x: OpId,
        mop: Box<MoveOp>,
    },
    Reduce {
        x: OpId,
        rop: BOp,
        reduce_axis: OpId,
    },
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, SerBin, DeBin)]
pub enum TileReduceKind {
    Row,
    Col,
    Scalar,
}

/// Scope of index. Index is like loop, but purely parallel acess
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, SerBin, DeBin)]
pub enum IdxKind {
    /// Group scope. Represents blocks in cuda, cores in CPU and tenstorrent.
    Group(OpId),
    /// Local scope. Represents cuda threads.
    Local(u32),
    /// Warp scope. Represents warps and wavefronts.
    Warp(u8),
}

impl std::fmt::Display for IdxKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            IdxKind::Group(x) => f.write_fmt(format_args!("group_r{x}")),
            IdxKind::Local(x) => f.write_fmt(format_args!("local_{x}")),
            IdxKind::Warp(x) => f.write_fmt(format_args!("warp_{x}")),
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
    /// Bitwise NOT: ~x
    BitNot,
    /// Exponential: e^x
    Exp,
    /// Exponential with base 2: 2^x
    Exp2,
    /// Natural logarithm: ln(x)
    Ln,
    /// Logarithm with base 2: log2(x)
    Log2,
    /// Reciprocal: 1/x
    Reciprocal,
    /// Square root: sqrt(x)
    Sqrt,
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

/// Movement operations for tensor shape transformations.
///
/// These operations change the shape of tensors without changing their data.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, SerBin)]
pub enum MoveOp {
    /// Reshape to a new shape.
    Reshape { shape: OpId },
    /// Expand dimensions.
    Expand { shape: OpId },
    /// Permute axes.
    Permute { axes: Box<[UAxis]> },
    /// Flip axes
    Flip { axes: Box<[UAxis]> },
    /// Pad axis
    /// Pad with `lp` zeros on the left, to total axis length `len`
    /// (tinygrad convention). Right padding is `len - lp - orig_len`.
    Pad { axis: UAxis, lp: OpId, len: OpId },
    /// Slice axis
    Narrow { axis: UAxis, start: OpId, len: OpId },
}

impl MoveOp {
    /// Returns a copy with all `OpId` references remapped through `op_map`.
    /// `fallback` is used for op ids not present in `op_map` (mirroring the
    /// assign replay's handling of the movement-chain head).
    pub(crate) fn remap(&self, op_map: &Map<OpId, OpId>) -> Box<Self> {
        match self {
            MoveOp::Reshape { shape } => Box::new(MoveOp::Reshape {
                shape: op_map.get(shape).copied().expect("MoveOp::remap: referenced op not in mapping"),
            }),
            MoveOp::Expand { shape } => Box::new(MoveOp::Expand {
                shape: op_map.get(shape).copied().expect("MoveOp::remap: referenced op not in mapping"),
            }),
            MoveOp::Permute { axes } => Box::new(MoveOp::Permute { axes: axes.clone() }),
            MoveOp::Pad { axis, lp, len } => Box::new(MoveOp::Pad {
                axis: *axis,
                lp: op_map.get(lp).copied().expect("MoveOp::remap: referenced op not in mapping"),
                len: op_map.get(len).copied().expect("MoveOp::remap: referenced op not in mapping"),
            }),
            MoveOp::Flip { axes } => Box::new(MoveOp::Flip { axes: axes.clone() }),
            MoveOp::Narrow { axis, start, len } => {
                let start = op_map.get(start).copied().expect("MoveOp::remap: referenced op not in mapping");
                let len = op_map.get(len).copied().expect("MoveOp::remap: referenced op not in mapping");
                Box::new(MoveOp::Narrow { axis: *axis, start, len })
            }
        }
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
}

impl MMADims {
    /// Decompose MMAD dimensions into m, n, k components.
    pub const fn decompose_mnk(self) -> (u64, u64, u64) {
        match self {
            MMADims::m8n8k16 => (8, 8, 16),
            MMADims::m16n8k8 => (16, 8, 8),
            MMADims::m16n8k16 => (16, 8, 16),
        }
    }
}

/// Memory layout for tensor core matrix operands.
///
/// Describes how matrix data is stored in memory.
#[allow(non_camel_case_types)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, SerBin, DeBin)]
pub enum MMALayout {
    /// Row-major for both matrices
    row_row,
    /// Row-major for A, column-major for B
    row_col,
    /// Column-major for A, row-major for B
    col_row,
    /// Column-major for both matrices
    col_col,
}

/// Data type for matrix multiply operations.
#[allow(non_camel_case_types)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, SerBin, DeBin)]
pub enum MMADType {
    /// FP16 input with FP32 accumulator
    f16_f16_f16_f32,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, SerBin)]
pub struct OpNode {
    pub prev: OpId,
    pub next: OpId, // Use Vec<OpId> instead for egraph
    pub op: Op,
}

/// Operation ID for kernel operations.
///
/// This is a unique identifier for each operation in the kernel IR.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, SerBin, DeBin)]
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

impl MemLayout {
    /// Get the number of elements in the memory layout.
    pub(crate) fn n_elements(self) -> Dim {
        match self {
            MemLayout::Scalar => 1,
            MemLayout::Vector(x) => x.into(),
            MemLayout::Tile { x, y, .. } => x as Dim * y as Dim,
        }
    }
}

impl std::fmt::Display for MemLayout {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MemLayout::Scalar => f.write_fmt(format_args!("Scalar")),
            MemLayout::Vector(x) => f.write_fmt(format_args!("Vec({x})")),
            MemLayout::Tile { x, y, stride } => f.write_fmt(format_args!("Tile({x}x{y} st={stride})")),
        }
    }
}

impl Op {
    // TODO use custom non allocating iterator instead of allocating a vec
    #[allow(clippy::match_same_arms)]
    pub(crate) fn parameters(&self) -> impl DoubleEndedIterator<Item = OpId> {
        match self {
            Op::Const { .. } | Op::Storage { .. } | Op::EndLoop | Op::Barrier | Op::EndIf => {
                vec![]
            }
            &Op::Param { shape, .. } => {
                // Shape is null after linearize
                if shape.is_null() { vec![] } else { vec![shape] }
            }
            &Op::Index { kind, .. } => match kind {
                IdxKind::Group(len) => vec![len],
                IdxKind::Local(_) => vec![],
                IdxKind::Warp(_) => vec![],
            },
            &Op::Loop { len, .. } => vec![len],
            &Op::Move { x, ref mop } => match mop.as_ref() {
                MoveOp::Reshape { shape, .. } | MoveOp::Expand { shape } => vec![x, *shape],
                MoveOp::Permute { .. } | MoveOp::Flip { .. } => vec![x],
                MoveOp::Pad { lp, len, .. } => vec![x, *lp, *len],
                MoveOp::Narrow { start, len, .. } => vec![x, *start, *len],
            },
            Op::Reduce { x, reduce_axis, .. } => vec![*x, *reduce_axis],
            Op::ReduceTile { x, .. } => vec![*x],
            &Op::Store { dst, src, index, .. } => {
                // Pre-linearize stores carry a NULL index (whole-view write).
                if index.is_null() {
                    vec![dst, src]
                } else {
                    vec![dst, src, index]
                }
            }
            Op::Cast { x, .. } => vec![*x],
            Op::Unary { x, .. } => vec![*x],
            &Op::Binary { x, y, .. } => vec![x, y],
            &Op::Load { src, index, .. } => vec![src, index],
            &Op::Mad { x, y, z } => vec![x, y, z],
            Op::Asm { ops, .. } => ops.iter().copied().collect(),
            Op::Stack { ops } => ops.iter().copied().collect(),
            &Op::Devectorize { vec, .. } => vec![vec],
            &Op::Wmma { a, b, c, .. } => vec![a, b, c],
            Op::If { condition } => vec![*condition],
            Op::MatmulTile { x, y } => vec![*x, *y],
            Op::TransposeTile { x } => vec![*x],
        }
        .into_iter()
    }

    #[allow(clippy::match_same_arms)]
    pub(crate) fn parameters_mut(&mut self) -> impl DoubleEndedIterator<Item = &mut OpId> {
        match self {
            Op::Const { .. } | Op::Storage { .. } | Op::EndLoop | Op::EndIf | Op::Barrier => vec![],
            Op::Param { shape, .. } => {
                // Shape is null after linearize
                if shape.is_null() { vec![] } else { vec![shape] }
            }
            Op::Index { kind, .. } => match kind {
                IdxKind::Group(len) => vec![len],
                IdxKind::Local(_) => vec![],
                IdxKind::Warp(_) => vec![],
            },
            Op::Loop { len, .. } => vec![len],
            Op::Move { x, mop } => match mop.as_mut() {
                MoveOp::Reshape { shape, .. } | MoveOp::Expand { shape } => vec![x, shape],
                MoveOp::Permute { .. } | MoveOp::Flip { .. } => vec![x],
                MoveOp::Pad { lp, len, .. } => vec![x, lp, len],
                MoveOp::Narrow { start, len, .. } => vec![x, start, len],
            },
            Op::Reduce { x, reduce_axis, .. } => vec![x, reduce_axis],
            Op::ReduceTile { x, .. } => vec![x],
            Op::Store { dst, src: x, index, .. } => {
                // Pre-linearize stores carry a NULL index (whole-view write).
                if index.is_null() { vec![dst, x] } else { vec![dst, x, index] }
            }
            Op::Cast { x, .. } => vec![x],
            Op::Unary { x, .. } => vec![x],
            Op::Binary { x, y, .. } => vec![x, y],
            Op::Load { src, index, .. } => vec![src, index],
            Op::Mad { x, y, z } => vec![x, y, z],
            Op::Stack { ops } => ops.iter_mut().collect(),
            Op::Devectorize { vec, .. } => vec![vec],
            Op::Wmma { a, b, c, .. } => vec![a, b, c],
            Op::If { condition } => vec![condition],
            Op::MatmulTile { x, y } => vec![x, y],
            Op::TransposeTile { x } => vec![x],
            Op::Asm { ops, .. } => ops.iter_mut().collect(),
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
            MemScope::CircularReader => "cb_reader",
            MemScope::CircularWriter => "cb_writer",
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
