extern crate alloc;
use alloc::boxed::Box;
use alloc::vec::Vec;
use crate::{tensor::Id, axes::Axes, shape::Shape};

/// Node representing different possible tensors
pub enum Node {
    LeafF32(Shape),
    LeafI32(Shape),
    UniformF32(Shape),
    UniformI32(Shape),
    IterF32(Box<dyn Iterator<Item = f32>>, Shape),
    IterI32(Box<dyn Iterator<Item = i32>>, Shape),
    CastF32(Id),
    CastI32(Id),
    Neg(Id),
    ReLU(Id),
    Sin(Id),
    Cos(Id),
    Ln(Id),
    Exp(Id),
    Tanh(Id),
    Sqrt(Id),
    Add(Id, Id),
    Sub(Id, Id),
    Mul(Id, Id),
    Div(Id, Id),
    Pow(Id, Id),
    Cmplt(Id, Id),
    Reshape(Id, Shape),
    Expand(Id, Shape),
    Permute(Id, Axes, Shape),
    Sum(Id, Axes, Shape),
    Max(Id, Axes, Shape),
}

impl Node {
    pub fn parameters(&self) -> impl Iterator<Item = Id> {
        match self {
            Node::LeafF32(..)
            | Node::LeafI32(..)
            | Node::UniformF32(..)
            | Node::UniformI32(..)
            | Node::IterF32(..)
            | Node::IterI32(..) => Vec::new().into_iter(),
            Node::Add(x, y)
            | Node::Sub(x, y)
            | Node::Mul(x, y)
            | Node::Div(x, y)
            | Node::Cmplt(x, y)
            | Node::Pow(x, y) => Vec::from([*x, *y]).into_iter(),
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
            | Node::Sum(x, ..)
            | Node::Max(x, ..) => Vec::from([*x]).into_iter(),
        }
    }
}
