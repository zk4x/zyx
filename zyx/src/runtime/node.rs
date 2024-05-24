use crate::DType;
use crate::runtime::TensorId;

/// Constant value
/// Floats must be bitcasted in order to implement Ord and Eq.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Constant {
    /// bf16 constant
    BF16(u16),
    /// f16 constant
    F16(u16),
    /// f32 constant
    F32(u32),
    /// f64 constant
    F64(u64),
    /// complex f32 constant
    CF32(u32),
    /// complex f64 constant
    CF64(u64),
    /// u8 constant
    U8(u8),
    /// i8 constant
    I8(i8),
    /// i16 constant
    I16(i16),
    /// i32 constant
    I32(i32),
    /// i64 constant
    I64(i64),
}

impl Constant {
    /// Get dtype of this constant
    pub fn dtype(&self) -> DType {
        return match self {
            Constant::BF16(..) => DType::BF16,
            Constant::F16(..) => DType::F16,
            Constant::F32(..) => DType::F32,
            Constant::F64(..) => DType::F64,
            Constant::CF32(..) => DType::CF32,
            Constant::CF64(..) => DType::CF64,
            Constant::U8(..) => DType::U8,
            Constant::I8(..) => DType::I8,
            Constant::I16(..) => DType::I16,
            Constant::I32(..) => DType::I32,
            Constant::I64(..) => DType::I64,
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub(super) enum Node {
    Const {
        value: Constant,
    },
    Leaf {
        len: usize,
        dtype: DType,
    },
    Cast {
        x: TensorId,
        dtype: DType,
    },
    ReLU {
        x: TensorId,
    },
    Neg {
        x: TensorId,
    },
    Inv {
        x: TensorId,
    },
    Cos {
        x: TensorId,
    },
    Sin {
        x: TensorId,
    },
    Exp {
        x: TensorId,
    },
    Ln {
        x: TensorId,
    },
    Sqrt {
        x: TensorId,
    },
    Tanh {
        x: TensorId,
    },
    Add {
        x: TensorId,
        y: TensorId,
    },
    Sub {
        x: TensorId,
        y: TensorId,
    },
    Mul {
        x: TensorId,
        y: TensorId,
    },
    Div {
        x: TensorId,
        y: TensorId,
    },
    Pow {
        x: TensorId,
        y: TensorId,
    },
    Cmplt {
        x: TensorId,
        y: TensorId,
    },
    Where {
        x: TensorId,
        y: TensorId,
        z: TensorId,
    },
    Reshape {
        x: TensorId,
        shape_id: u32,
    },
    Expand {
        x: TensorId,
        shape_id: u32,
    },
    Permute {
        x: TensorId,
        axes_id: u32,
        shape_id: u32,
    },
    Pad {
        x: TensorId,
        padding_id: u32,
        shape_id: u32,
    },
    Sum {
        x: TensorId,
        axes_id: u32,
        shape_id: u32,
    },
    Max {
        x: TensorId,
        axes_id: u32,
        shape_id: u32,
    },
}

/// Iterator over parameters of node which does not allocate on heap.
pub struct NodeParametersIterator {
    parameters: [TensorId; 3],
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
        return Some(self.parameters[idx as usize])
    }
}

impl Node {
    /// Get all parameters of self. This method does not allocate.
    pub const fn parameters(&self) -> impl Iterator<Item = TensorId> {
        return match self {
            Node::Const {..} | Node::Leaf {..} => NodeParametersIterator {
                parameters: [0; 3],
                idx: 0,
                len: 0,
            },
            Node::Cast {x, ..}
            | Node::Inv {x}
            | Node::Neg {x}
            | Node::ReLU {x}
            | Node::Exp {x}
            | Node::Ln {x}
            | Node::Sin {x}
            | Node::Cos {x}
            | Node::Sqrt {x}
            | Node::Tanh {x}
            | Node::Reshape {x, ..}
            | Node::Expand {x, ..}
            | Node::Permute {x, ..}
            | Node::Pad {x, ..}
            | Node::Sum {x, ..}
            | Node::Max {x, ..} => NodeParametersIterator {
                parameters: [*x, 0, 0],
                idx: 0,
                len: 1,
            },
            Node::Add {x, y}
            | Node::Sub {x, y}
            | Node::Mul {x, y}
            | Node::Div {x, y}
            | Node::Cmplt {x, y}
            | Node::Pow {x, y} => NodeParametersIterator {
                parameters: [*x, *y, 0],
                idx: 0,
                len: 2,
            },
            Node::Where {x, y, z} => NodeParametersIterator {
                parameters: [*x, *y, *z],
                idx: 0,
                len: 3,
            },
        }
    }
}