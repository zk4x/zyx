//! # Shape
//! 
//! This module contains definition of [Shape], it's [Axes] and some associated methods.

//mod const_index;
pub mod shapes;
pub mod axes;

pub use shapes::{Shape, Sh0, Sh1, Sh2, Sh3, Sh4, Sh5};
pub use axes::{Axes, Ax0, Ax1, Ax2, Ax3, Ax4, Ax5};

/// HasLastDim
pub trait HasLastDim {
    /// Last dimension size
    const LAST_DIM: usize;
}

/// HasLast2Dims
pub trait HasLast2Dims: HasLastDim {
    /// Last dimension size
    const LAST_DIM_2: usize;
}

/// ReducableBy
pub trait ReducableBy<Ax>
where
    Ax: Axes,
{
    /// Output of reduce
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
/// This is current limitation of stable rust - we have to write all permutations manually.
/// Even with nonnegative numbers it is 119 axes permutations for shape of RANK 5.
pub trait PermutableBy<Ax>
where
    Ax: Axes,
{
    /// Output of permute
    type Output;
}

/// MatMulBy
pub trait MatMulBy<YSh>
where
    YSh: Shape + HasLastDim,
    Self: HasLastDim,
{
    /// Output shape of matmul
    type Output: Shape + HasLastDim;
}

