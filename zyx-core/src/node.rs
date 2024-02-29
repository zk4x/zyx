extern crate alloc;
use crate::utils::get_shape;
use crate::{axes::Axes, shape::Shape, tensor::Id};
use alloc::boxed::Box;
use core::fmt::Formatter;
use crate::dtype::DType;

/// Node representing different possible tensors
pub enum Node {
    /// IterF32 initializer
    IterF32(Box<dyn Iterator<Item = f32>>, Shape),
    /// IterF64 initializer
    IterF64(Box<dyn Iterator<Item = f64>>, Shape),
    /// IterI32 initializer
    IterI32(Box<dyn Iterator<Item = i32>>, Shape),
    /// Leaf that is guaranteed to be evaluated
    Leaf(Shape, DType),
    /// Uniform initializer for range 0..1
    Uniform(Shape, DType),
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
            Node::Leaf(sh, dtype) => f.write_fmt(format_args!("Leaf({sh}, {dtype})")),
            Node::IterF32(_, sh) => f.write_fmt(format_args!("Iter({sh}, F32)")),
            Node::IterF64(_, sh) => f.write_fmt(format_args!("Iter({sh}, F64)")),
            Node::IterI32(_, sh) => f.write_fmt(format_args!("Iter({sh}, I32)")),
            Node::Cast(x, dtype) => f.write_fmt(format_args!("Cast({x}, {dtype})")),
            Node::Uniform(sh, dtype) => f.write_fmt(format_args!("Uniform({sh}, {dtype})")),
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
            return None
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
            Node::Leaf(..)
            | Node::Uniform(..)
            | Node::IterF32(..)
            | Node::IterF64(..)
            | Node::IterI32(..) => 0,
            Node::Cast(..)
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
            Node::Leaf(..)
            | Node::Uniform(..)
            | Node::IterF32(..)
            | Node::IterF64(..)
            | Node::IterI32(..) => NodeParametersIterator { parameters: [crate::tensor::id(0); 3], idx: 0, len: 0 },
            Node::Cast(x, ..)
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
            | Node::Max(x, ..) => NodeParametersIterator { parameters: [*x, crate::tensor::id(0), crate::tensor::id(0)], idx: 0, len: 1 },
            Node::Add(x, y)
            | Node::Sub(x, y)
            | Node::Mul(x, y)
            | Node::Div(x, y)
            | Node::Cmplt(x, y)
            | Node::Pow(x, y) => NodeParametersIterator { parameters: [*x, *y, crate::tensor::id(0)], idx: 0, len: 2 },
            Node::Where(x, y, z) => NodeParametersIterator { parameters: [*x, *y, *z], idx: 0, len: 3 },
        }
    }

    /// Get number of operations necessary to calculate this node
    pub fn flop(&self, nodes: &[Node]) -> usize {
        match self {
            Node::Leaf(..)
            | Node::Uniform(..)
            | Node::IterF32(..)
            | Node::IterF64(..)
            | Node::IterI32(..)
            | Node::Reshape(..)
            | Node::Expand(..)
            | Node::Permute(..)
            | Node::Pad(..) => 0,
            Node::Where(x, ..)
            | Node::Add(x, _)
            | Node::Sub(x, _)
            | Node::Mul(x, _)
            | Node::Div(x, _)
            | Node::Cmplt(x, _)
            | Node::Pow(x, _) => get_shape(nodes, *x).numel() * 2, // x and y are guaranteed to be same shape
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
            Node::Leaf(..)
            | Node::Uniform(..)
            | Node::IterF32(..)
            | Node::IterF64(..)
            | Node::IterI32(..) => false,
            Node::Cast(x, ..)
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
