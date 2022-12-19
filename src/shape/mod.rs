//! # Shape
//! 
//! This module contains definition of [Shape], it's axes/dimensions and some associated methods.

//mod bin_op;
//mod const_index;
pub mod shape;
pub mod axes;

pub use shape::{Shape, Sh0, Sh1, Sh2, Sh3, Sh4, Sh5, ReducableBy, PermutableBy, BinOpWith};
pub use axes::{Axes, Ax0, Ax1, Ax2, Ax3, Ax4, Ax5};

