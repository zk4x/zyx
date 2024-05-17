use crate::DType;

type TensorId = u32;

/// Constant value
#[derive(Debug, Clone, PartialEq, PartialOrd)]
pub enum Constant {
    /// f32 constant
    F32(f32),
    /// f64 constant
    F64(f64),
    /// i32 constant
    I32(i32),
}

impl Constant {
    /// Get dtype of this constant
    pub fn dtype(&self) -> DType {
        match self {
            Constant::F32(..) => DType::F32,
            Constant::F64(..) => DType::F64,
            Constant::I32(..) => DType::I32,
        }
    }
}

#[derive(PartialEq, PartialOrd)]
pub(super) enum Node {
    Const(),
    Leaf(usize),
    Cast(TensorId),
    ReLU(TensorId),
    Neg(TensorId),
    Inv(TensorId),
    Cos(TensorId),
    Sin(TensorId),
    Exp(TensorId),
    Ln(TensorId),
    Sqrt(TensorId),
    Tanh(TensorId),
    Add(TensorId, TensorId),
    Sub(TensorId, TensorId),
    Mul(TensorId, TensorId),
    Div(TensorId, TensorId),
    Pow(TensorId, TensorId),
    Cmplt(TensorId, TensorId),
    Where(TensorId, TensorId, TensorId),
    Reshape(TensorId),
    Expand(TensorId),
    Permute(TensorId),
    Pad(TensorId),
    Sum(TensorId),
    Max(TensorId),
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
        Some(self.parameters[idx as usize])
    }
}

impl Node {
    /// Get all parameters of self. This method does not allocate.
    pub const fn parameters(&self) -> impl Iterator<Item = TensorId> {
        match self {
            Node::Const(..) | Node::Leaf(..) => NodeParametersIterator {
                parameters: [0; 3],
                idx: 0,
                len: 0,
            },
            Node::Cast(x, ..)
            | Node::Inv(x)
            | Node::Neg(x)
            | Node::ReLU(x)
            | Node::Exp(x)
            | Node::Ln(x)
            | Node::Sin(x)
            | Node::Cos(x)
            | Node::Sqrt(x)
            | Node::Tanh(x)
            | Node::Reshape(x, ..)
            | Node::Expand(x, ..)
            | Node::Permute(x, ..)
            | Node::Pad(x, ..)
            | Node::Sum(x, ..)
            | Node::Max(x, ..) => NodeParametersIterator {
                parameters: [*x, 0, 0],
                idx: 0,
                len: 1,
            },
            Node::Add(x, y)
            | Node::Sub(x, y)
            | Node::Mul(x, y)
            | Node::Div(x, y)
            | Node::Cmplt(x, y)
            | Node::Pow(x, y) => NodeParametersIterator {
                parameters: [*x, *y, 0],
                idx: 0,
                len: 2,
            },
            Node::Where(x, y, z) => NodeParametersIterator {
                parameters: [*x, *y, *z],
                idx: 0,
                len: 3,
            },
        }
    }
}