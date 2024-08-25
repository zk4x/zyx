use crate::{dtype::Constant, tensor::TensorId, DType, Scalar};

#[derive(Debug, PartialEq, Eq, PartialOrd, Ord, Clone, Copy)]
pub(crate) enum BOp {
    Add,
    Sub,
    Mul,
    Div,
    Pow,
    Cmplt,
    Cmpgt,
    Max,
    Or,
}

#[derive(Debug, PartialEq, Eq, PartialOrd, Ord, Clone, Copy)]
pub(crate) enum UOp {
    Cast(DType),
    ReLU,
    Neg,
    Exp2,
    Log2,
    Inv,
    Sqrt,
    Sin,
    Cos,
    Not,
    Nonzero,
}

#[derive(Debug, PartialEq, Eq, PartialOrd, Ord, Clone, Copy)]
pub(crate) enum ROp {
    Sum,
    Max,
}

type Axis = usize;

type Dimension = usize;

#[derive(Debug, PartialEq, Eq, PartialOrd, Ord, Clone)]
pub(crate) enum Node {
    Const {
        value: Constant,
    },
    // Tensor stored on device
    Leaf {
        shape: Vec<Dimension>,
        dtype: DType,
    },
    // Constant tensor baked into kernels
    Expand {
        x: TensorId,
        shape: Vec<Dimension>,
    },
    Permute {
        x: TensorId,
        axes: Vec<Axis>,
        shape: Vec<usize>,
    },
    // Reshape can be sometimes axis split or axis join
    Reshape {
        x: TensorId,
        shape: Vec<Dimension>,
    },
    Pad {
        x: TensorId,
        padding: Vec<(isize, isize)>,
        shape: Vec<usize>,
    },
    Reduce {
        x: TensorId,
        axes: Vec<Axis>,
        shape: Vec<usize>,
        rop: ROp,
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

impl Default for Node {
    fn default() -> Self {
        Self::Const { value: Constant::Bool(false) }
    }
}

pub(crate) struct NodeParametersIterator {
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
        return Some(self.parameters[idx as usize]);
    }
}

impl Node {
    /// Get all parameters of self. This method does not allocate.
    pub const fn parameters(&self) -> impl Iterator<Item = TensorId> {
        return match self {
            Node::Const { .. } | Node::Leaf { .. } => NodeParametersIterator {
                parameters: [0, 0],
                idx: 0,
                len: 0,
            },
            Node::Unary { x, .. }
            | Node::Reshape { x, .. }
            | Node::Expand { x, .. }
            | Node::Permute { x, .. }
            | Node::Pad { x, .. }
            | Node::Reduce { x, .. } => NodeParametersIterator {
                parameters: [*x, 0],
                idx: 0,
                len: 1,
            },
            Node::Binary { x, y, .. } => NodeParametersIterator {
                parameters: [*x, *y],
                idx: 0,
                len: 2,
            },
        };
    }

    pub(crate) fn is_movement(&self) -> bool {
        matches!(
            self,
            Node::Pad { .. } | Node::Reshape { .. } | Node::Expand { .. } | Node::Permute { .. }
        )
    }

    pub(crate) fn is_unary(&self) -> bool {
        matches!(self, Node::Unary { .. })
    }
}

impl Constant {
    pub(crate) fn unary(self, uop: UOp) -> Constant {
        match uop {
            UOp::Cast(dtype) => {
                match self {
                    Constant::F32(x) => unsafe { std::mem::transmute::<_, f32>(x) }.cast_dtype(dtype),
                    Constant::F64(x) => unsafe { std::mem::transmute::<_, f64>(x) }.cast_dtype(dtype),
                    Constant::U8(x) => x.cast_dtype(dtype),
                    Constant::I8(x) => x.cast_dtype(dtype),
                    Constant::I16(x) => x.cast_dtype(dtype),
                    Constant::U32(_) => panic!(),
                    Constant::I32(x) => x.cast_dtype(dtype),
                    Constant::I64(x) => x.cast_dtype(dtype),
                    Constant::Bool(x) => x.cast_dtype(dtype),
                }
            }
            UOp::ReLU => todo!(),
            UOp::Neg => todo!(),
            UOp::Exp2 => todo!(),
            UOp::Log2 => todo!(),
            UOp::Inv => todo!(),
            UOp::Sqrt => todo!(),
            UOp::Sin => todo!(),
            UOp::Cos => todo!(),
            UOp::Not => todo!(),
            UOp::Nonzero => todo!(),
        }
    }

    // Assumes both constants are the same dtype
    pub(crate) fn binary(x: Constant, y: Constant, bop: BOp) -> Constant {
        todo!()
    }
}
