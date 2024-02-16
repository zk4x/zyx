extern crate alloc;
use crate::utils::get_shape;
use crate::{axes::Axes, shape::Shape, tensor::Id};
use alloc::boxed::Box;
use alloc::vec::Vec;

/// Blah
pub trait Blah<T>: Iterator<Item = T> + core::fmt::Debug {}
impl<A: Iterator<Item = T> + core::fmt::Debug, T> Blah<T> for A {}

/// Node representing different possible tensors
#[derive(Debug)]
pub enum Node {
    /// F32 leaf that is guaranteed to be evaluated
    LeafF32(Shape),
    /// I32 leaf that is guaranteed to be evaluated
    LeafI32(Shape),
    /// UniformF32 initializer for range 0..1
    UniformF32(Shape),
    /// IterF32 initializer
    IterF32(Box<dyn Blah<f32>>, Shape),
    //IterF32(Box<dyn Iterator<Item = f32>>, Shape),
    /// IterI32 initializer
    IterI32(Box<dyn Blah<i32>>, Shape),
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

impl Node {
    /// Get all parameters of self
    pub fn parameters(&self) -> impl Iterator<Item = Id> {
        match self {
            Node::LeafF32(..)
            | Node::LeafI32(..)
            | Node::UniformF32(..)
            | Node::IterF32(..)
            | Node::IterI32(..) => Vec::new().into_iter(),
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
            | Node::Max(x, ..) => Vec::from([*x]).into_iter(),
            Node::Add(x, y)
            | Node::Sub(x, y)
            | Node::Mul(x, y)
            | Node::Div(x, y)
            | Node::Cmplt(x, y)
            | Node::Pow(x, y) => Vec::from([*x, *y]).into_iter(),
            Node::Where(x, y, z) => Vec::from([*x, *y, *z]).into_iter(),
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
