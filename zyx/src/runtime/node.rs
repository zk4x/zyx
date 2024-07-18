use alloc::vec::Vec;

use crate::{tensor::TensorId, DType, Device};

#[derive(Debug, PartialEq, Eq, PartialOrd, Ord, Clone, Copy)]
pub(crate) enum BOp {
    Add,
    Sub,
    Mul,
    Div,
    Pow,
    Cmplt,
    Max,
}

#[derive(Debug, PartialEq, Eq, PartialOrd, Ord, Clone, Copy)]
pub(crate) enum UOp {
    Noop,
    Cast(DType),
    ReLU,
    Neg,
    Exp,
    Ln,
    Tanh,
    Inv,
    Sqrt,
    Sin,
    Cos,
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
    // TODO later add constant nodes
    /*Const {
        value: enum {
            F32(u32),
            I32(i32),
            I64(i64),
        },
    },*/
    Leaf {
        shape: Vec<Dimension>,
        dtype: DType,
        device: Device,
    },
    // Can be later added for moving between devices
    /*ToDevice {
        x: TensorId,
        device: Device,
    },*/
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
        pad: Vec<(isize, isize)>,
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
    Where {
        x: TensorId,
        y: TensorId,
        z: TensorId,
    },
}

pub(crate) struct NodeParametersIterator {
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
        return Some(self.parameters[idx as usize]);
    }
}

impl Node {
    /// Get all parameters of self. This method does not allocate.
    pub const fn parameters(&self) -> impl Iterator<Item = TensorId> {
        return match self {
            Node::Leaf { .. } => NodeParametersIterator {
                parameters: [0, 0, 0],
                idx: 0,
                len: 0,
            },
            /*TODO Node::ToDevice { x, .. } => NodeParametersIterator {
            parameters: [*x, 0, 0],
            idx: 0,
            len: 1,
            },*/
            Node::Unary { x, .. }
            | Node::Reshape { x, .. }
            | Node::Expand { x, .. }
            | Node::Permute { x, .. }
            | Node::Pad { x, .. }
            | Node::Reduce { x, .. } => NodeParametersIterator {
                parameters: [*x, 0, 0],
                idx: 0,
                len: 1,
            },
            Node::Binary { x, y, .. } => NodeParametersIterator {
                parameters: [*x, *y, 0],
                idx: 0,
                len: 2,
            },
            Node::Where { x, y, z } => NodeParametersIterator {
                parameters: [*x, *y, *z],
                idx: 0,
                len: 3,
            },
        };
    }
}
