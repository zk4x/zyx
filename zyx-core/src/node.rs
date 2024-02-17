extern crate alloc;
use crate::utils::get_shape;
use crate::{axes::Axes, shape::Shape, tensor::Id};
use alloc::boxed::Box;
use core::fmt::Formatter;

/// Node representing different possible tensors
pub enum Node {
    /// F32 leaf that is guaranteed to be evaluated
    LeafF32(Shape),
    /// I32 leaf that is guaranteed to be evaluated
    LeafI32(Shape),
    /// UniformF32 initializer for range 0..1
    UniformF32(Shape),
    /// IterF32 initializer
    IterF32(Box<dyn Iterator<Item = f32>>, Shape),
    //IterF32(Box<dyn Iterator<Item = f32>>, Shape),
    /// IterI32 initializer
    IterI32(Box<dyn Iterator<Item = i32>>, Shape),
    //IterI32(Box<dyn Iterator<Item = i32>>, Shape),
    /// Cast to F32 unary op
    CastF32(Id),
    /// Cast to I32 unary op
    CastI32(Id),
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
            Node::LeafF32(sh) => f.write_fmt(format_args!("Leaf({sh}, F32)")),
            Node::LeafI32(sh) => f.write_fmt(format_args!("Leaf({sh}, I32)")),
            Node::IterF32(_, sh) => f.write_fmt(format_args!("Iter({sh}, F32)")),
            Node::IterI32(_, sh) => f.write_fmt(format_args!("Iter({sh}, I32)")),
            Node::CastF32(x) => f.write_fmt(format_args!("Cast({x}, F32)")),
            Node::CastI32(x) => f.write_fmt(format_args!("Cast({x}, I32)")),
            Node::UniformF32(sh) => f.write_fmt(format_args!("Uniform({sh}, F32)")),
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
            Node::Sum(x, ax, ..) => f.write_fmt(format_args!("Sum({x}, {ax})")),
            Node::Max(x, ax, ..) => f.write_fmt(format_args!("Max({x}, {ax})")),
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
            Node::LeafF32(..)
            | Node::LeafI32(..)
            | Node::UniformF32(..)
            | Node::IterF32(..)
            | Node::IterI32(..) => 0,
            Node::CastF32(x)
            | Node::CastI32(x)
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
            | Node::Max(x, ..) => 1,
            Node::Add(x, y)
            | Node::Sub(x, y)
            | Node::Mul(x, y)
            | Node::Div(x, y)
            | Node::Cmplt(x, y)
            | Node::Pow(x, y) => 2,
            Node::Where(x, y, z) => 3,
        }
    }

    /// Get all parameters of self. This method does not allocate.
    pub const fn parameters(&self) -> impl Iterator<Item = Id> {
        match self {
            Node::LeafF32(..)
            | Node::LeafI32(..)
            | Node::UniformF32(..)
            | Node::IterF32(..)
            | Node::IterI32(..) => NodeParametersIterator { parameters: [crate::tensor::id(0); 3], idx: 0, len: 0 },
            Node::CastF32(x)
            | Node::CastI32(x)
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
            Node::LeafF32(..)
            | Node::LeafI32(..)
            | Node::UniformF32(..)
            | Node::IterF32(..)
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
            Node::CastF32(x)
            | Node::CastI32(x)
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
            Node::LeafF32(..)
            | Node::LeafI32(..)
            | Node::UniformF32(..)
            | Node::IterF32(..)
            | Node::IterI32(..) => false,
            Node::CastF32(x)
            | Node::CastI32(x)
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
}
