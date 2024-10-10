//! Graph node, each node is one operation. Nodes
//! represent the opset that is available on tensors.

use crate::{dtype::Constant, shape::Axis, tensor::TensorId, DType, Scalar};

#[cfg_attr(feature = "disk_cache", derive(bitcode::Encode, bitcode::Decode))]
#[derive(Debug, PartialEq, Eq, PartialOrd, Ord, Clone, Copy)]
pub(super) enum BOp {
    Add,
    Sub,
    Mul,
    Div,
    Pow,
    Cmplt,
    Cmpgt,
    Max,
    Or,
    And,
    BitXor,
    BitOr,
    BitAnd,
    NotEq,
}

#[cfg_attr(feature = "disk_cache", derive(bitcode::Encode, bitcode::Decode))]
#[derive(Debug, PartialEq, Eq, PartialOrd, Ord, Clone, Copy)]
pub(super) enum UOp {
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

#[cfg_attr(feature = "disk_cache", derive(bitcode::Encode, bitcode::Decode))]
#[derive(Debug, PartialEq, Eq, PartialOrd, Ord, Clone, Copy)]
pub(super) enum ROp {
    Sum,
    Max,
}

#[derive(Debug, PartialEq, Eq, PartialOrd, Ord, Clone)]
pub(super) enum Node {
    // Constant tensor baked into kernels
    Const {
        value: Constant,
    },
    // Tensor stored on device
    Leaf,
    Expand {
        x: TensorId,
    },
    Permute {
        x: TensorId,
        axes: Vec<Axis>,
    },
    // Reshape can be sometimes axis split or axis join
    Reshape {
        x: TensorId,
    },
    Pad {
        x: TensorId,
        padding: Vec<(isize, isize)>,
    },
    Reduce {
        x: TensorId,
        axes: Vec<Axis>,
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
        Self::Const {
            value: Constant::Bool(false),
        }
    }
}

pub(super) struct NodeParametersIterator {
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

    pub(super) fn is_movement(&self) -> bool {
        matches!(
            self,
            Node::Pad { .. } | Node::Reshape { .. } | Node::Expand { .. } | Node::Permute { .. }
        )
    }

    pub(super) fn is_unary(&self) -> bool {
        matches!(self, Node::Unary { .. })
    }
}

trait CastDType: Scalar {
    fn cast_dtype(self, dtype: DType) -> Constant {
        match dtype {
            DType::BF16 => {
                Constant::BF16(unsafe { std::mem::transmute(self.cast::<half::bf16>()) })
            }
            DType::F8 => todo!(),
            DType::F16 => Constant::F16(unsafe { std::mem::transmute(self.cast::<half::f16>()) }),
            DType::F32 => Constant::F32(unsafe { std::mem::transmute(self.cast::<f32>()) }),
            DType::F64 => Constant::F64(unsafe { std::mem::transmute(self.cast::<f64>()) }),
            #[cfg(feature = "complex")]
            DType::CF32 => todo!("Complex numbers"),
            #[cfg(feature = "complex")]
            DType::CF64 => todo!("Complex numbers"),
            DType::U8 => Constant::U8(self.cast()),
            DType::U32 => Constant::U32(self.cast()),
            DType::I8 => Constant::I8(self.cast()),
            DType::I16 => Constant::I16(self.cast()),
            DType::I32 => Constant::I32(self.cast()),
            DType::I64 => Constant::I64(self.cast()),
            DType::Bool => Constant::Bool(self.cast()),
        }
    }
}

impl<T: Scalar> CastDType for T {}

impl Constant {
    pub(super) fn unary(self, uop: UOp) -> Constant {
        use crate::Float;
        use std::mem::transmute as t;
        match uop {
            UOp::Cast(dtype) => match self {
                Constant::BF16(x) => unsafe { t::<_, half::bf16>(x) }.cast_dtype(dtype),
                Constant::F8(_) => panic!(),
                Constant::F16(x) => unsafe { t::<_, half::f16>(x) }.cast_dtype(dtype),
                Constant::F32(x) => unsafe { t::<_, f32>(x) }.cast_dtype(dtype),
                Constant::F64(x) => unsafe { t::<_, f64>(x) }.cast_dtype(dtype),
                #[cfg(feature = "complex")]
                Constant::CF32(..) => todo!("Complex numbers"),
                #[cfg(feature = "complex")]
                Constant::CF64(..) => todo!("Complex numbers"),
                Constant::U8(x) => x.cast_dtype(dtype),
                Constant::I8(x) => x.cast_dtype(dtype),
                Constant::I16(x) => x.cast_dtype(dtype),
                Constant::U32(_) => panic!(),
                Constant::I32(x) => x.cast_dtype(dtype),
                Constant::I64(x) => x.cast_dtype(dtype),
                Constant::Bool(x) => x.cast_dtype(dtype),
            },
            UOp::ReLU => match self {
                Constant::BF16(x) => Constant::F16(unsafe { t(t::<_, half::bf16>(x).relu()) }),
                Constant::F8(_) => panic!(),
                Constant::F16(x) => Constant::F16(unsafe { t(t::<_, half::f16>(x).relu()) }),
                Constant::F32(x) => Constant::F32(unsafe { t(t::<_, f32>(x).relu()) }),
                Constant::F64(x) => Constant::F64(unsafe { t(t::<_, f64>(x).relu()) }),
                #[cfg(feature = "complex")]
                Constant::CF32(..) => todo!("Complex numbers"),
                #[cfg(feature = "complex")]
                Constant::CF64(..) => todo!("Complex numbers"),
                Constant::U8(x) => Constant::U8(x.relu()),
                Constant::I8(x) => Constant::I8(x.relu()),
                Constant::I16(x) => Constant::I16(x.relu()),
                Constant::U32(_) => panic!(),
                Constant::I32(x) => Constant::I32(x.relu()),
                Constant::I64(x) => Constant::I64(x.relu()),
                Constant::Bool(x) => Constant::Bool(x.relu()),
            },
            UOp::Neg => match self {
                Constant::BF16(x) => Constant::F16(unsafe { t(t::<_, half::bf16>(x).neg()) }),
                Constant::F8(_) => panic!(),
                Constant::F16(x) => Constant::F16(unsafe { t(t::<_, half::f16>(x).neg()) }),
                Constant::F32(x) => Constant::F32(unsafe { t(t::<_, f32>(x).neg()) }),
                Constant::F64(x) => Constant::F64(unsafe { t(t::<_, f64>(x).neg()) }),
                #[cfg(feature = "complex")]
                Constant::CF32(..) => todo!("Complex numbers"),
                #[cfg(feature = "complex")]
                Constant::CF64(..) => todo!("Complex numbers"),
                Constant::U8(x) => Constant::U8(x.neg()),
                Constant::I8(x) => Constant::I8(x.neg()),
                Constant::I16(x) => Constant::I16(x.neg()),
                Constant::U32(_) => panic!(),
                Constant::I32(x) => Constant::I32(x.neg()),
                Constant::I64(x) => Constant::I64(x.neg()),
                Constant::Bool(x) => Constant::Bool(x.neg()),
            },
            UOp::Exp2 => match self {
                Constant::BF16(x) => Constant::F16(unsafe { t(t::<_, half::bf16>(x).exp2()) }),
                Constant::F8(_) => panic!(),
                Constant::F16(x) => Constant::F16(unsafe { t(t::<_, half::f16>(x).exp2()) }),
                Constant::F32(x) => Constant::F32(unsafe { t(t::<_, f32>(x).exp2()) }),
                Constant::F64(x) => Constant::F64(unsafe { t(t::<_, f64>(x).exp2()) }),
                #[cfg(feature = "complex")]
                Constant::CF32(..) => todo!("Complex numbers"),
                #[cfg(feature = "complex")]
                Constant::CF64(..) => todo!("Complex numbers"),
                Constant::U8(x) => Constant::U8(2.pow(x)),
                Constant::I8(x) => Constant::I8(2.pow(x)),
                Constant::I16(x) => Constant::I16(2.pow(x)),
                Constant::U32(_) => panic!(),
                Constant::I32(x) => Constant::I32(2.pow(x)),
                Constant::I64(x) => Constant::I64(2.pow(x)),
                Constant::Bool(_) => todo!(),
            },
            UOp::Log2 => match self {
                Constant::F8(_) => panic!(),
                Constant::F16(x) => Constant::F16(unsafe { t(t::<_, half::f16>(x).log2()) }),
                Constant::BF16(x) => Constant::F16(unsafe { t(t::<_, half::bf16>(x).log2()) }),
                Constant::F32(x) => Constant::F32(unsafe { t(t::<_, f32>(x).log2()) }),
                Constant::F64(x) => Constant::F64(unsafe { t(t::<_, f64>(x).log2()) }),
                #[cfg(feature = "complex")]
                Constant::CF32(..) => todo!("Complex numbers"),
                #[cfg(feature = "complex")]
                Constant::CF64(..) => todo!("Complex numbers"),
                Constant::U8(x) => Constant::U8(x.ilog2() as u8),
                Constant::I8(x) => Constant::I8(x.ilog2() as i8),
                Constant::I16(x) => Constant::I16(x.ilog2() as i16),
                Constant::U32(_) => panic!(),
                Constant::I32(x) => Constant::I32(x.ilog2() as i32),
                Constant::I64(x) => Constant::I64(x.ilog2() as i64),
                Constant::Bool(_) => todo!(),
            },
            UOp::Inv => match self {
                Constant::F8(_) => panic!(),
                Constant::F16(x) => {
                    Constant::F16(unsafe { t(half::f16::ONE / t::<_, half::f16>(x)) })
                }
                Constant::BF16(x) => {
                    Constant::F16(unsafe { t(half::bf16::ONE / t::<_, half::bf16>(x)) })
                }
                Constant::F32(x) => Constant::F32(unsafe { t(1f32 / t::<_, f32>(x)) }),
                Constant::F64(x) => Constant::F64(unsafe { t(1f64 / t::<_, f64>(x)) }),
                #[cfg(feature = "complex")]
                Constant::CF32(..) => todo!("Complex numbers"),
                #[cfg(feature = "complex")]
                Constant::CF64(..) => todo!("Complex numbers"),
                Constant::U8(x) => Constant::U8(1 / x),
                Constant::I8(x) => Constant::I8(1 / x),
                Constant::I16(x) => Constant::I16(1 / x),
                Constant::U32(_) => panic!(),
                Constant::I32(x) => Constant::I32(1 / x),
                Constant::I64(x) => Constant::I64(1 / x),
                Constant::Bool(_) => todo!(),
            },
            UOp::Sqrt => match self {
                Constant::F8(_) => panic!(),
                Constant::F16(x) => Constant::F16(unsafe { t(t::<_, half::f16>(x).relu()) }),
                Constant::BF16(x) => Constant::F16(unsafe { t(t::<_, half::bf16>(x).relu()) }),
                Constant::F32(x) => Constant::F32(unsafe { t(t::<_, f32>(x).sqrt()) }),
                Constant::F64(x) => Constant::F64(unsafe { t(t::<_, f64>(x).sqrt()) }),
                #[cfg(feature = "complex")]
                Constant::CF32(..) => todo!("Complex numbers"),
                #[cfg(feature = "complex")]
                Constant::CF64(..) => todo!("Complex numbers"),
                c => panic!("Unsupported dtype {}", c.dtype()),
            },
            UOp::Sin => match self {
                Constant::F8(_) => panic!(),
                Constant::F16(x) => Constant::F16(unsafe { t(t::<_, half::f16>(x).relu()) }),
                Constant::BF16(x) => Constant::F16(unsafe { t(t::<_, half::bf16>(x).relu()) }),
                Constant::F32(x) => Constant::F32(unsafe { t(t::<_, f32>(x).sin()) }),
                Constant::F64(x) => Constant::F64(unsafe { t(t::<_, f64>(x).sin()) }),
                #[cfg(feature = "complex")]
                Constant::CF32(..) => todo!("Complex numbers"),
                #[cfg(feature = "complex")]
                Constant::CF64(..) => todo!("Complex numbers"),
                c => panic!("Unsupported dtype {}", c.dtype()),
            },
            UOp::Cos => match self {
                Constant::F8(_) => panic!(),
                Constant::F16(x) => Constant::F16(unsafe { t(t::<_, half::f16>(x).relu()) }),
                Constant::BF16(x) => Constant::F16(unsafe { t(t::<_, half::bf16>(x).relu()) }),
                Constant::F32(x) => Constant::F32(unsafe { t(t::<_, f32>(x).cos()) }),
                Constant::F64(x) => Constant::F64(unsafe { t(t::<_, f64>(x).cos()) }),
                #[cfg(feature = "complex")]
                Constant::CF32(..) => todo!("Complex numbers"),
                #[cfg(feature = "complex")]
                Constant::CF64(..) => todo!("Complex numbers"),
                c => panic!("Unsupported dtype {}", c.dtype()),
            },
            UOp::Not => match self {
                Constant::F8(_) => panic!(),
                Constant::F16(x) => Constant::F16(unsafe { t(t::<_, half::f16>(x).relu()) }),
                Constant::BF16(x) => Constant::F16(unsafe { t(t::<_, half::bf16>(x).relu()) }),
                Constant::F32(x) => {
                    Constant::F32(unsafe { t(if t::<_, f32>(x) == 0f32 { 1f32 } else { 0f32 }) })
                }
                Constant::F64(x) => {
                    Constant::F64(unsafe { t(if t::<_, f64>(x) == 0f64 { 1f64 } else { 0f64 }) })
                }
                #[cfg(feature = "complex")]
                Constant::CF32(..) => todo!("Complex numbers"),
                #[cfg(feature = "complex")]
                Constant::CF64(..) => todo!("Complex numbers"),
                Constant::U8(x) => Constant::U8(!x),
                Constant::I8(x) => Constant::I8(!x),
                Constant::I16(x) => Constant::I16(!x),
                Constant::U32(_) => panic!(),
                Constant::I32(x) => Constant::I32(!x),
                Constant::I64(x) => Constant::I64(!x),
                Constant::Bool(_) => todo!(),
            },
            UOp::Nonzero => match self {
                Constant::F8(_) => panic!(),
                Constant::F16(x) => Constant::F16(unsafe { t(t::<_, half::f16>(x).relu()) }),
                Constant::BF16(x) => Constant::F16(unsafe { t(t::<_, half::bf16>(x).relu()) }),
                Constant::F32(x) => {
                    Constant::F32(unsafe { t((t::<_, f32>(x) != 0.) as i32 as f32) })
                }
                Constant::F64(x) => {
                    Constant::F64(unsafe { t((t::<_, f64>(x) != 0.) as i64 as f64) })
                }
                #[cfg(feature = "complex")]
                Constant::CF32(..) => todo!("Complex numbers"),
                #[cfg(feature = "complex")]
                Constant::CF64(..) => todo!("Complex numbers"),
                Constant::U8(x) => Constant::U8((x != 0) as u8),
                Constant::I8(x) => Constant::I8((x != 0) as i8),
                Constant::I16(x) => Constant::I16((x != 0) as i16),
                Constant::U32(_) => panic!(),
                Constant::I32(x) => Constant::I32((x != 0) as i32),
                Constant::I64(x) => Constant::I64((x != 0) as i64),
                Constant::Bool(_) => todo!(),
            },
        }
    }

    // TODO binary constant evaluation
    // Assumes both constants are the same dtype
    //pub(super) fn binary(x: Constant, y: Constant, bop: BOp) -> Constant { todo!() }
}
