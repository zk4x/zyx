#![no_std]

pub mod autodiff;
pub mod axes;
pub mod backend;
pub mod dtype;
pub mod shape;
pub mod tensor;

#[cfg(not(feature = "heapless"))]
extern crate alloc;

use crate::axes::Axes;
use crate::shape::Shape;
use crate::tensor::Id;

type Vec<T> = alloc::vec::Vec<T>;
type Map<K, V> = alloc::collections::BTreeMap<K, V>;
type Set<V> = alloc::collections::BTreeSet<V>;

/// Node representing different possible tensors
pub enum Node {
    Leaf,
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
    Permute(Id, Axes),
    Sum(Id, Axes),
    Max(Id, Axes),
}
