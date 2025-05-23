//! Graph node, each node is one operation. Nodes
//! represent the opset that is available on tensors.

use crate::{dtype::Constant, tensor::TensorId, DType, Scalar};
use half::{bf16, f16};

#[derive(Debug, PartialEq, Eq, PartialOrd, Ord, Clone, Copy, Hash)]
pub enum BOp {
    Add,
    Sub,
    Mul,
    Div,
    // Perhaps we will eventually remove this, but for now I am not certain
    #[allow(unused)]
    Pow,
    Mod,
    Cmplt,
    Cmpgt,
    Max,
    Or,
    And,
    BitXor,
    BitOr,
    BitAnd,
    BitShiftLeft,
    BitShiftRight,
    NotEq,
}

#[derive(Debug, PartialEq, Eq, PartialOrd, Ord, Clone, Copy, Hash)]
pub enum UOp {
    ReLU,
    Neg,
    Exp2,
    Log2,
    Reciprocal,
    Sqrt,
    Sin,
    Cos,
    Not,
}

#[derive(Debug, PartialEq, Eq, PartialOrd, Ord, Clone, Copy, Hash)]
pub enum ROp {
    Sum,
    Max,
}

#[derive(Debug, PartialEq, Eq, PartialOrd, Ord, Clone)]
pub enum Node {
    // Constant tensor baked into kernels
    Const {
        value: Constant,
    },
    // Tensor stored on device
    Leaf {
        dtype: DType,
    },
    Expand {
        x: TensorId,
        //shape: u32,
    },
    Permute {
        x: TensorId,
        //axes: Box<Vec<usize>>,
    },
    // Reshape can be sometimes axis split or axis join
    Reshape {
        x: TensorId,
        //shape: Box<Vec<usize>>,
    },
    Pad {
        x: TensorId,
        //padding: Box<Vec<(isize, isize)>>,
    },
    Reduce {
        x: TensorId,
        rop: ROp,
        //axes: Box<Vec<usize>>,
    },
    Cast {
        x: TensorId,
        dtype: DType,
    },
    Unary {
        x: TensorId,
        uop: UOp,
    },
    Binary {
        x: TensorId,
        y: TensorId,
        bop: BOp,
    },
}

pub struct NodeParametersIterator {
    parameters: [TensorId; 2],
    len: u8,
    idx: u8,
}

impl Iterator for NodeParametersIterator {
    type Item = TensorId;
    fn next(&mut self) -> Option<Self::Item> {
        if self.idx == self.len {
            return None;
        }
        let idx = self.idx;
        self.idx += 1;
        Some(self.parameters[idx as usize])
    }
}

impl Node {
    /// Get all parameters of self. This method does not allocate.
    pub const fn parameters(&self) -> impl Iterator<Item = TensorId> {
        match self {
            Node::Const { .. } | Node::Leaf { .. } => {
                NodeParametersIterator { parameters: [0, 0], idx: 0, len: 0 }
            }
            Node::Unary { x, .. }
            | Node::Cast { x, .. }
            | Node::Reshape { x, .. }
            | Node::Expand { x, .. }
            | Node::Permute { x, .. }
            | Node::Pad { x, .. }
            | Node::Reduce { x, .. } => {
                NodeParametersIterator { parameters: [*x, 0], idx: 0, len: 1 }
            }
            Node::Binary { x, y, .. } => {
                NodeParametersIterator { parameters: [*x, *y], idx: 0, len: 2 }
            }
        }
    }

    pub const fn num_parameters(&self) -> u8 {
        match self {
            Node::Const { .. } => 0,
            Node::Leaf { .. } => 0,
            Node::Expand { .. } => 1,
            Node::Permute { .. } => 1,
            Node::Reshape { .. } => 1,
            Node::Pad { .. } => 1,
            Node::Reduce { .. } => 1,
            Node::Cast { .. } => 1,
            Node::Unary { .. } => 1,
            Node::Binary { .. } => 2,
        }
    }

    pub const fn param1(&self) -> u32 {
        match *self {
            Node::Const { .. } => unreachable!(),
            Node::Leaf { .. } => unreachable!(),
            Node::Expand { x } => x,
            Node::Permute { x } => x,
            Node::Reshape { x } => x,
            Node::Pad { x } => x,
            Node::Reduce { x, .. } => x,
            Node::Cast { x, .. } => x,
            Node::Unary { x, .. } => x,
            Node::Binary { x, .. } => x,
        }
    }

    pub const fn param2(&self) -> (u32, u32) {
        match *self {
            Node::Binary { x, y, .. } => (x, y),
            _ => unreachable!(),
        }
    }

    /*pub(super) const fn is_movement(&self) -> bool {
        matches!(
            self,
            Node::Pad { .. } | Node::Reshape { .. } | Node::Expand { .. } | Node::Permute { .. }
        )
    }

    pub(super) const fn is_unary(&self) -> bool {
        matches!(self, Node::Unary { .. })
    }*/
}

trait CastDType: Scalar {
    fn cast_dtype(self, dtype: DType) -> Constant {
        match dtype {
            DType::BF16 => Constant::BF16(self.cast::<half::bf16>().to_bits()),
            DType::F16 => Constant::F16(self.cast::<half::f16>().to_bits()),
            DType::F32 => Constant::F32(self.cast::<f32>().to_bits()),
            DType::F64 => Constant::F64(self.cast::<f64>().to_bits()),
            DType::U8 => Constant::U8(self.cast()),
            DType::U16 => Constant::U16(self.cast()),
            DType::U32 => Constant::U32(self.cast()),
            DType::U64 => Constant::U64(self.cast()),
            DType::I8 => Constant::I8(self.cast()),
            DType::I16 => Constant::I16(self.cast()),
            DType::I32 => Constant::I32(self.cast()),
            DType::I64 => Constant::I64(self.cast()),
            DType::Bool => Constant::Bool(self.cast()),
        }
    }
}

impl<T: Scalar> CastDType for T {}

// TODO clean this up
impl Constant {
    pub(super) fn cast(self, dtype: DType) -> Constant {
        return match self {
            Constant::BF16(x) => half::bf16::from_bits(x).cast_dtype(dtype),
            Constant::F16(x) => half::f16::from_bits(x).cast_dtype(dtype),
            Constant::F32(x) => f32::from_bits(x).cast_dtype(dtype),
            Constant::F64(x) => f64::from_bits(x).cast_dtype(dtype),
            Constant::U8(x) => x.cast_dtype(dtype),
            Constant::I8(x) => x.cast_dtype(dtype),
            Constant::I16(x) => x.cast_dtype(dtype),
            Constant::U16(x) => x.cast_dtype(dtype),
            Constant::U32(x) => x.cast_dtype(dtype),
            Constant::U64(x) => x.cast_dtype(dtype),
            Constant::I32(x) => x.cast_dtype(dtype),
            Constant::I64(x) => x.cast_dtype(dtype),
            Constant::Bool(x) => x.cast_dtype(dtype),
        };
    }

    pub(super) fn unary(self, uop: UOp) -> Constant {
        use crate::Float;
        fn unary_func<T: Scalar>(x: T, uop: UOp) -> T {
            match uop {
                UOp::Exp2
                | UOp::Log2
                | UOp::Reciprocal
                | UOp::Sqrt
                | UOp::Sin
                | UOp::Cos => unreachable!(),
                UOp::ReLU => x.relu(),
                UOp::Neg => x.neg(),
                UOp::Not => x.not(),
            }
        }
        fn unary_func_float<T: Float>(x: T, uop: UOp) -> T {
            match uop {
                UOp::ReLU => x.relu(),
                UOp::Neg => x.neg(),
                UOp::Exp2 => x.exp2(),
                UOp::Log2 => x.log2(),
                UOp::Reciprocal => x.reciprocal(),
                UOp::Sqrt => x.sqrt(),
                UOp::Sin => x.sin(),
                UOp::Cos => x.cos(),
                UOp::Not => x.not(),
            }
        }
        match self {
            Constant::BF16(x) => {
                Constant::BF16(unary_func_float(half::bf16::from_bits(x), uop).to_bits())
            }
            Constant::F16(x) => {
                Constant::F16(unary_func_float(half::f16::from_bits(x), uop).to_bits())
            }
            Constant::F32(x) => Constant::F32(unary_func_float(f32::from_bits(x), uop).to_bits()),
            Constant::F64(x) => Constant::F64(unary_func_float(f64::from_bits(x), uop).to_bits()),
            Constant::U8(x) => Constant::U8(unary_func(x, uop)),
            Constant::U16(x) => Constant::U16(unary_func(x, uop)),
            Constant::U32(x) => Constant::U32(unary_func(x, uop)),
            Constant::U64(x) => Constant::U64(unary_func(x, uop)),
            Constant::I8(x) => Constant::I8(unary_func(x, uop)),
            Constant::I16(x) => Constant::I16(unary_func(x, uop)),
            Constant::I32(x) => Constant::I32(unary_func(x, uop)),
            Constant::I64(x) => Constant::I64(unary_func(x, uop)),
            Constant::Bool(x) => Constant::Bool(unary_func(x, uop)),
        }
    }

    // TODO binary constant evaluation
    // Assumes both constants are the same dtype
    pub(super) fn binary(x: Constant, y: Constant, bop: BOp) -> Constant {
        fn binary_func<T: Scalar>(x: T, y: T, bop: BOp) -> Constant {
            match bop {
                BOp::Add => Constant::new(x.add(y)),
                BOp::Sub => Constant::new(x.sub(y)),
                BOp::Mul => Constant::new(x.mul(y)),
                BOp::Div => Constant::new(x.div(y)),
                BOp::Pow => Constant::new(x.pow(y)),
                BOp::Mod => Constant::new(x.mod_(y)),
                BOp::Max => Constant::new(x.max(y)),
                BOp::Cmplt => Constant::new(x.cmplt(y)),
                BOp::Cmpgt => Constant::new(x.cmpgt(y)),
                BOp::Or => Constant::new(x.or(y)),
                BOp::And => Constant::new(x.and(y)),
                BOp::NotEq => Constant::new(x.noteq(y)),
                BOp::BitXor => Constant::new(x.bitxor(y)),
                BOp::BitOr => Constant::new(x.bitor(y)),
                BOp::BitAnd => Constant::new(x.bitand(y)),
                BOp::BitShiftLeft => Constant::new(x.bitshiftleft(y)),
                BOp::BitShiftRight => Constant::new(x.bitshiftright(y)),
            }
        }
        debug_assert_eq!(x.dtype(), y.dtype());
        match x {
            Constant::BF16(x) => {
                let Constant::BF16(y) = y else { unreachable!() };
                binary_func(bf16::from_bits(x), bf16::from_bits(y), bop)
            }
            Constant::F16(x) => {
                let Constant::F16(y) = y else { unreachable!() };
                binary_func(f16::from_bits(x), f16::from_bits(y), bop)
            }
            Constant::F32(x) => {
                let Constant::F32(y) = y else { unreachable!() };
                binary_func(f32::from_bits(x), f32::from_bits(y), bop)
            }
            Constant::F64(x) => {
                let Constant::F64(y) = y else { unreachable!() };
                binary_func(f64::from_bits(x), f64::from_bits(y), bop)
            }
            Constant::U8(x) => {
                let Constant::U8(y) = y else { unreachable!() };
                binary_func(x, y, bop)
            }
            Constant::U16(x) => {
                let Constant::U16(y) = y else { unreachable!() };
                binary_func(x, y, bop)
            }
            Constant::U32(x) => {
                let Constant::U32(y) = y else { unreachable!() };
                binary_func(x, y, bop)
            }
            Constant::U64(x) => {
                let Constant::U64(y) = y else { unreachable!() };
                binary_func(x, y, bop)
            }
            Constant::I8(x) => {
                let Constant::I8(y) = y else { unreachable!() };
                binary_func(x, y, bop)
            }
            Constant::I16(x) => {
                let Constant::I16(y) = y else { unreachable!() };
                binary_func(x, y, bop)
            }
            Constant::I32(x) => {
                let Constant::I32(y) = y else { unreachable!() };
                binary_func(x, y, bop)
            }
            Constant::I64(x) => {
                let Constant::I64(y) = y else { unreachable!() };
                binary_func(x, y, bop)
            }
            Constant::Bool(x) => {
                let Constant::Bool(y) = y else { unreachable!() };
                binary_func(x, y, bop)
            }
        }
    }
}
