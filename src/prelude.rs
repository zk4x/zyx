//! Includes useful traits, mainly for [operations](crate::ops) and [shape](crate::shape).

pub use crate::{
    ops::*,
    shape::Shape,
    nn::{module::Module, parameters::{Parameters, HasParameters}},
    device::{BufferFromSlice, BufferInit, ShapedBufferInit},
    optim::SGDStep,
};
