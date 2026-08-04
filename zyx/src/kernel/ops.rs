use nanoserde::{DeBin, SerBin};

use crate::DType;
use crate::dtype::Constant;
use crate::kernel::{MemLayout, MemScope};
use crate::shape::{Dim, UAxis};
use crate::slab::SlabId;
use crate::view::View;

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, SerBin, DeBin)]
pub enum Op {
    // ops that exist in both
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

    // ops that only exist after unfolding views and reduces
    Const(Constant),
    Define {
        dtype: DType,
        scope: MemScope,
        ro: bool,
        len: Dim,
    }, // len is 0 for global stores
    Store {
        dst: OpId,
        x: OpId,
        index: OpId,
        layout: MemLayout,
    },
    Load {
        src: OpId,
        index: OpId,
        layout: MemLayout,
    },
    Index {
        len: Dim,
        axis: u32,
        scope: IdxScope,
    },
    // TODO add WarpIndex
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
    // Vectorization, YAY!
    Vectorize {
        ops: Vec<OpId>,
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

    // ops that exist only in kernelizer, basically they can be eventually removed.
    // TODO Get rid of the view, use whatever ops that are needed directly
    // and then use unfold movement ops function to convert it all into indices.
    // This will make Op smaller and Copy.
    // TODO Use MovementOp instead for all the movement.
    ConstView(Box<(Constant, View)>),
    LoadView(Box<(DType, View)>),
    StoreView {
        src: OpId,
        dtype: DType,
    },
    Move {
        x: OpId,
        mop: Box<MoveOp>,
    },
    Reduce {
        x: OpId,
        rop: BOp,
        n_axes: UAxis,
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
pub enum IdxScope {
    /// Group scope. Represents blocks in cuda, cores in CPU and tenstorrent.
    Group,
    /// Local scope. Represents cuda threads.
    Local,
    /// Warp scope. Represents warps and wavefronts.
    Warp,
}

impl std::fmt::Display for IdxScope {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            IdxScope::Group => "group",
            IdxScope::Local => "local",
            IdxScope::Warp => "warp",
        })
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
        use BOp::{And, Cmpgt, Cmplt, Eq, NotEq, Or};
        matches!(self, Cmpgt | Cmplt | NotEq | Eq | And | Or)
    }
}

/// Movement operations for tensor shape transformations.
///
/// These operations change the shape of tensors without changing their data.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, SerBin, DeBin)]
pub enum MoveOp {
    /// Reshape to a new shape.
    Reshape { shape: Vec<Dim> },
    /// Expand dimensions.
    Expand { shape: Vec<Dim> },
    /// Permute axes.
    Permute { axes: Vec<UAxis>, shape: Vec<Dim> },
    /// Pad dimensions.
    Pad { padding: Vec<(i64, i64)>, shape: Vec<Dim> },
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

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, SerBin, DeBin)]
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
    pub(crate) const NULL: Self = Self(u32::MAX);

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
