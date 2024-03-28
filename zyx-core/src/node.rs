extern crate alloc;
use crate::dtype::DType;
use crate::utils::get_shape;
use crate::{axes::Axes, shape::Shape, tensor::Id};
use crate::scalar::Scalar;
use alloc::boxed::Box;
use core::fmt::Formatter;
use std::hash::Hasher;

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

impl core::hash::Hash for Constant {
    fn hash<H: Hasher>(&self, state: &mut H) {
        match self {
            Constant::F32(x) => state.write(&x.to_le_bytes()),
            Constant::F64(x) => state.write(&x.to_le_bytes()),
            Constant::I32(x) => state.write_i32(*x),
        }
    }
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

/// Node representing different possible tensors
#[derive(PartialEq, PartialOrd, Hash)]
pub enum Node {
    /// Constant node that can be compiled into kernels
    Const(Constant),
    /// Detach tensor from tape
    Detach(Id),
    /// Leaf that is guaranteed to be evaluated
    Leaf(Shape, DType),
    /// Random normal distribution tensor
    Normal(Shape, DType),
    /// Random uniform distribution tensor
    Uniform(Shape, Constant, Constant),
    /// Cast to dtype unary op
    Cast(Id, DType),
    /// Neg unary op
    Neg(Id),
    /// ReLU unary op
    ReLU(Id),
    /// Sine unary op
    Sin(Id),
    /// Cosine unary op
    Cos(Id),
    /// Natural logarithm unary op
    Ln(Id),
    /// Exp unary op
    Exp(Id),
    /// Hyperbolic tangent unary op
    Tanh(Id),
    /// Square root unary op
    Sqrt(Id),
    /// Addition binary op
    Add(Id, Id),
    /// Subtraction binary op
    Sub(Id, Id),
    /// Multiplication binary op
    Mul(Id, Id),
    /// Division binary op
    Div(Id, Id),
    /// Exponentiation binary op
    Pow(Id, Id),
    /// Compare less than binary op
    Cmplt(Id, Id),
    /// Where op
    Where(Id, Id, Id),
    /// Reshape movement op
    Reshape(Id, Shape),
    /// Expand movement op
    Expand(Id, Shape),
    /// Permute movement op
    Permute(Id, Axes, Shape),
    /// Pad movement op
    Pad(Id, Box<[(i64, i64)]>, Shape),
    /// Sum reduce op
    Sum(Id, Axes, Shape),
    /// Max reduce op
    Max(Id, Axes, Shape),
}

impl core::fmt::Debug for Node {
    fn fmt(&self, f: &mut Formatter<'_>) -> core::fmt::Result {
        match self {
            Node::Detach(x) => f.write_fmt(format_args!("Detach({x})")),
            Node::Leaf(sh, dtype) => f.write_fmt(format_args!("Leaf({sh}, {dtype})")),
            Node::Const(x) => f.write_fmt(format_args!("Const({x:.2?})")),
            Node::Normal(sh, dtype) => f.write_fmt(format_args!("Normal({sh}, {dtype})")),
            Node::Uniform(sh, start, end) => f.write_fmt(format_args!("Uniform({sh}, {start:?}..{end:?})")),
            Node::Cast(x, dtype) => f.write_fmt(format_args!("Cast({x}, {dtype})")),
            Node::Neg(x) => f.write_fmt(format_args!("Neg({x})")),
            Node::ReLU(x) => f.write_fmt(format_args!("ReLU({x})")),
            Node::Sin(x) => f.write_fmt(format_args!("Sin({x})")),
            Node::Cos(x) => f.write_fmt(format_args!("Cos({x})")),
            Node::Ln(x) => f.write_fmt(format_args!("Ln({x})")),
            Node::Exp(x) => f.write_fmt(format_args!("Exp({x})")),
            Node::Tanh(x) => f.write_fmt(format_args!("Tanh({x})")),
            Node::Sqrt(x) => f.write_fmt(format_args!("Sqrt({x})")),
            Node::Add(x, y) => f.write_fmt(format_args!("Add({x}, {y})")),
            Node::Sub(x, y) => f.write_fmt(format_args!("Sub({x}, {y})")),
            Node::Mul(x, y) => f.write_fmt(format_args!("Mul({x}, {y})")),
            Node::Div(x, y) => f.write_fmt(format_args!("Div({x}, {y})")),
            Node::Pow(x, y) => f.write_fmt(format_args!("Pow({x}, {y})")),
            Node::Cmplt(x, y) => f.write_fmt(format_args!("Cmplt({x}, {y})")),
            Node::Where(x, y, z) => f.write_fmt(format_args!("Where({x}, {y}, {z})")),
            Node::Expand(x, sh) => f.write_fmt(format_args!("Expand({x}, {sh})")),
            Node::Reshape(x, sh) => f.write_fmt(format_args!("Reshape({x}, {sh})")),
            Node::Pad(x, padding, ..) => f.write_fmt(format_args!("Pad({x}, {padding:?})")),
            Node::Permute(x, ax, ..) => f.write_fmt(format_args!("Permute({x}, {ax})")),
            Node::Sum(x, ax, sh) => f.write_fmt(format_args!("Sum({x}, {ax}, {sh})")),
            Node::Max(x, ax, sh) => f.write_fmt(format_args!("Max({x}, {ax}, {sh})")),
        }
    }
}

/// Iterator over parameters of node which does not allocate on heap.
pub struct NodeParametersIterator {
    parameters: [Id; 3],
    len: u8,
    idx: u8,
}

impl Iterator for NodeParametersIterator {
    type Item = Id;
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
    /// Get number of parameters of self. This method does not allocate.
    pub const fn num_parameters(&self) -> u8 {
        match self {
            Node::Const(..) | Node::Leaf(..) | Node::Normal(..) | Node::Uniform(..) => 0,
            Node::Detach(..)
            | Node::Cast(..)
            | Node::Neg(..)
            | Node::ReLU(..)
            | Node::Exp(..)
            | Node::Ln(..)
            | Node::Sin(..)
            | Node::Cos(..)
            | Node::Sqrt(..)
            | Node::Tanh(..)
            | Node::Reshape(..)
            | Node::Expand(..)
            | Node::Permute(..)
            | Node::Pad(..)
            | Node::Sum(..)
            | Node::Max(..) => 1,
            Node::Add(..)
            | Node::Sub(..)
            | Node::Mul(..)
            | Node::Div(..)
            | Node::Cmplt(..)
            | Node::Pow(..) => 2,
            Node::Where(..) => 3,
        }
    }

    /// Get all parameters of self. This method does not allocate.
    pub const fn parameters(&self) -> impl Iterator<Item = Id> {
        match self {
            Node::Const(..) | Node::Leaf(..) | Node::Normal(..) | Node::Uniform(..) => NodeParametersIterator {
                parameters: [crate::tensor::id(0); 3],
                idx: 0,
                len: 0,
            },
            Node::Cast(x, ..)
            | Node::Detach(x)
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
                parameters: [*x, crate::tensor::id(0), crate::tensor::id(0)],
                idx: 0,
                len: 1,
            },
            Node::Add(x, y)
            | Node::Sub(x, y)
            | Node::Mul(x, y)
            | Node::Div(x, y)
            | Node::Cmplt(x, y)
            | Node::Pow(x, y) => NodeParametersIterator {
                parameters: [*x, *y, crate::tensor::id(0)],
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

    /// Get number of operations necessary to calculate this node
    pub fn flop(&self, nodes: &[Node]) -> usize {
        match self {
            Node::Detach(..)
            | Node::Const(..)
            | Node::Leaf(..)
            | Node::Reshape(..)
            | Node::Expand(..)
            | Node::Permute(..)
            | Node::Pad(..) => 0,
            Node::Normal(sh, ..)
            | Node::Uniform(sh, ..) => sh.numel(),
            Node::Where(x, ..)
            | Node::Add(x, _)
            | Node::Sub(x, _)
            | Node::Mul(x, _)
            | Node::Div(x, _)
            | Node::Cmplt(x, _)
            | Node::Pow(x, _) => get_shape(nodes, *x).numel(), // x and y are guaranteed to be same shape
            Node::Cast(x, ..)
            | Node::Neg(x)
            | Node::ReLU(x)
            | Node::Exp(x)
            | Node::Ln(x)
            | Node::Sin(x)
            | Node::Cos(x)
            | Node::Sqrt(x)
            | Node::Tanh(x) => get_shape(nodes, *x).numel(),
            Node::Sum(x, _, sh) | Node::Max(x, _, sh) => {
                let n = sh.numel();
                let rdim = get_shape(nodes, *x).numel() / n;
                rdim * n // technically (rdim-1)*n, but hardware needs to do rdim*n
            }
        }
    }

    /// Check if parameters of self contains nid.
    pub fn parameters_contain(&self, nid: Id) -> bool {
        match self {
            Node::Const(..) | Node::Leaf(..) | Node::Normal(..) | Node::Uniform(..) => false,
            Node::Detach(x)
            | Node::Cast(x, ..)
            | Node::Neg(x)
            | Node::ReLU(x)
            | Node::Exp(x)
            | Node::Ln(x)
            | Node::Sin(x)
            | Node::Cos(x)
            | Node::Sqrt(x)
            | Node::Tanh(x)
            | Node::Sum(x, ..)
            | Node::Max(x, ..)
            | Node::Reshape(x, ..)
            | Node::Expand(x, ..)
            | Node::Permute(x, ..)
            | Node::Pad(x, ..) => nid == *x,
            Node::Add(x, y)
            | Node::Sub(x, y)
            | Node::Mul(x, y)
            | Node::Div(x, y)
            | Node::Cmplt(x, y)
            | Node::Pow(x, y) => nid == *x || nid == *y,
            Node::Where(x, y, z) => nid == *x || nid == *y || nid == *z,
        }
    }

    /// Is this reduce node? (sum or max)
    pub fn is_reduce(&self) -> bool {
        matches!(self, Node::Sum(..) | Node::Max(..))
    }
}
