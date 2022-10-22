//! Include useful traits, mainly for operations and shape.

pub use crate::{
    ops::*,
    module::{Module, Apply},
    shape::{IntoShape, IntoDims},
    optim::Optimizer,
    init::{EyeInit, RandInit, UniformInit},
    tensor::IntoVariable
};
