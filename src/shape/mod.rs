//! # Shape
//! 
//! This module contains definition of [Shape], it's axes/dimensions and some associated methods.

//mod bin_op;
//mod const_index;
pub mod shape;
pub mod axes;

pub use shape::{Shape, Sh0, Sh1, Sh2, Sh3, Sh4, Sh5};
pub use axes::{Axes, Ax0, Ax1, Ax2, Ax3, Ax4, Ax5};

pub trait HasLastDim {
    const LAST_DIM: usize;
}

pub trait HasLast2Dims: HasLastDim {
    const LAST_DIM_2: usize;
}

/// This must be implemented for Axes
pub trait Argsortable {
    /// Ordered axes, these will never contain negative numbers
    type Argsort: Axes;
}

pub trait ReducableBy<Ax>
where
    Ax: Axes,
{
    type Output: Shape;
}

/// # PermutableBy
/// 
/// This trait is implemented for shapes that are permutable by the given axes.
///
/// ## Important note
/// 
/// We support permutation by Ax2<-1, -2> for all [shapes](Shape) with [rank](Shape::RANK) more than 1. This is transpose.
/// 
/// All other permutations are only implemented by axes with nonnegative numbers.
/// This is current limitation of stable rust - we have to write all combinations manually.
/// Even with nonnegative numbers it is 119 axes combinations for shape of RANK 5.
pub trait PermutableBy<Ax>
where
    Ax: Axes,
{
    type Output: Shape;
}

pub trait BinOpBy<YSh>
where
    YSh: Shape,
{
    type Output: Shape;
}

pub trait MatMulBy<YSh>
where
    YSh: Shape + HasLastDim,
    Self: HasLastDim,
{
    type Output: Shape + HasLastDim;
}

