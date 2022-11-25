//! Includes useful traits, mainly for [operations](crate::ops) and [shape](crate::shape).

pub use crate::{
    ops::*,
    module::{Module, Parameters},
    shape::Shape,
    init::{EyeInit, RandnInit, UniformInit},
    optim::SGDStep,
    tensor::IntoVariable
};
