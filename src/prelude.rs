//! Include useful traits, mainly for [operations](crate::ops) and [shape](crate::shape).

pub use crate::{
    ops::*,
    module::{Module, Apply},
    shape::{IntoShape, IntoDims},
    optim::Optimizer,
    init::{EyeInit, RandnInit, UniformInit},
    tensor::IntoVariable
};
