//! Includes useful traits, mainly for [operations](crate::ops) and [shape](crate::shape).

pub use crate::{
    device::{BufferFromSlice, BufferInit, ShapedBufferInit},
    nn::{
        parameters::{HasParameters, Parameters},
        ApplyModule, Module,
    },
    ops::*,
    optim::SGDStep,
    shape::Shape,
};
