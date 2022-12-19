/*
Operations needed:
reduce, reshape, expand, permute

This const Shape is not quite there yet, but we will see how const generic expressions evolve
and then this will be ready.
*/

//! Shape module

use core::{ops::Index, fmt::{Display, Debug}};
//use crate::ops::Permute;

use super::axes::{Axes, Ax2, Ax3};

/// Shape trait
pub trait Shape: Copy + Clone + PartialEq + Eq + Debug + Display + Index<usize> + Index<i32> {
    /// Rank of Shape
    const RANK: usize;
    /// Output type when calling strides function
    type Strides; // This is [usize; RANK], just needed because you can't write it directly
    /// Get shape's strides
    fn strides() -> Self::Strides;
    /// Get shape's number of elements
    fn numel() -> usize;
    /// Check if the shape is empty, that is if Self::numel() == 0
    fn is_empty() -> bool { Self::numel() == 0 }
}

pub trait HasLastDim {
    const LAST_DIM: usize;
}

pub trait ReducableBy<Ax>
where
    Ax: Axes,
{
    type Output: Shape;
}

pub trait PermutableBy<Ax>
where
    Ax: Axes,
{
    type Output: Shape;
}

pub trait BinOpWith<YSh>
where
    YSh: Shape,
{
    type Output: Shape;
}

/// Shape with no dimensions. Used for scalars.
#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub struct Sh0 {}

impl Shape for Sh0 {
    const RANK: usize = 0;
    type Strides = [usize; 0];
    fn strides() -> Self::Strides { [] }
    fn numel() -> usize { 0 }
}

impl Display for Sh0 {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_fmt(format_args!("()"))
    }
}

impl Index<usize> for Sh0 {
    type Output = usize;
    fn index(&self, index: usize) -> &Self::Output {
        panic!("Index out of range, index is {}, but the length is 0", index)
    }
}

impl Index<i32> for Sh0 {
    type Output = usize;
    fn index(&self, index: i32) -> &Self::Output {
        panic!("Index out of range, index is {}, but the length is 0", index)
    }
}

/// Shape with 1 dimension
#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub struct Sh1<const D0: usize> {}

impl<const D0: usize> Shape for Sh1<D0> {
    const RANK: usize = 1;
    type Strides = [usize; 1];
    fn strides() -> Self::Strides { [1] }
    fn numel() -> usize { D0 }
}

impl<const D0: usize> Display for Sh1<D0> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_fmt(format_args!("{}", D0))
    }
}

impl<const D0: usize> Index<usize> for Sh1<D0> {
    type Output = usize;
    fn index(&self, index: usize) -> &Self::Output {
        match index {
            0 => &D0,
            _ => panic!("Index out of range, index is {}, but the length is 1", index),
        }
    }
}

impl<const D0: usize> Index<i32> for Sh1<D0> {
    type Output = usize;
    fn index(&self, index: i32) -> &Self::Output {
        match index {
            0 => &D0,
            -1 => &D0,
            _ => panic!("Index out of range, index is {}, but the length is 1", index),
        }
    }
}

impl<const D0: usize> HasLastDim for Sh1<D0> {
    const LAST_DIM: usize = D0;
}

/// Shape with 2 dimensions
#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub struct Sh2<const D0: usize, const D1: usize> {}

impl<const D0: usize, const D1: usize> Shape for Sh2<D0, D1> {
    const RANK: usize = 2;
    type Strides = [usize; 2];
    fn strides() -> Self::Strides { [D1, 1] }
    fn numel() -> usize { D0*D1 }
}

impl<const D0: usize, const D1: usize> Display for Sh2<D0, D1> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_fmt(format_args!("{}x{}", D0, D1))
    }
}

impl<const D0: usize, const D1: usize> Index<usize> for Sh2<D0, D1> {
    type Output = usize;
    fn index(&self, index: usize) -> &Self::Output {
        match index {
            0 => &D0,
            1 => &D1,
            _ => panic!("Index out of range, index is {}, but the length is 2", index),
        }
    }
}

impl<const D0: usize, const D1: usize> Index<i32> for Sh2<D0, D1> {
    type Output = usize;
    fn index(&self, index: i32) -> &Self::Output {
        match index {
            0 => &D0,
            1 => &D1,
            -1 => &D1,
            -2 => &D0,
            _ => panic!("Index out of range, index is {}, but the length is 2", index),
        }
    }
}

impl<const D0: usize, const D1: usize> HasLastDim for Sh2<D0, D1> {
    const LAST_DIM: usize = D1;
}

impl<const D0: usize, const D1: usize> PermutableBy<Ax2<1, 0>> for Sh2<D0, D1> { type Output = Sh2<D1, D0>; }
impl<const D0: usize, const D1: usize> PermutableBy<Ax2<-1, -2>> for Sh2<D0, D1> { type Output = Sh2<D1, D0>; }
impl<const D0: usize, const D1: usize> PermutableBy<Ax2<-1, 0>> for Sh2<D0, D1> { type Output = Sh2<D1, D0>; }
impl<const D0: usize, const D1: usize> PermutableBy<Ax2<1, -2>> for Sh2<D0, D1> { type Output = Sh2<D1, D0>; }

/// Shape with 3 dimensions
#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub struct Sh3<const D0: usize, const D1: usize, const D2: usize> {}

impl<const D0: usize, const D1: usize, const D2: usize> Shape for Sh3<D0, D1, D2> {
    const RANK: usize = 3;
    type Strides = [usize; 3];
    fn strides() -> Self::Strides { [D1*D2, D2, 1] }
    fn numel() -> usize { D0*D1*D2 }
}

impl<const D0: usize, const D1: usize, const D2: usize> Display for Sh3<D0, D1, D2> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_fmt(format_args!("{}x{}x{}", D0, D1, D2))
    }
}

impl<const D0: usize, const D1: usize, const D2: usize> Index<usize> for Sh3<D0, D1, D2> {
    type Output = usize;
    fn index(&self, index: usize) -> &Self::Output {
        match index {
            0 => &D0,
            1 => &D1,
            2 => &D2,
            _ => panic!("Index out of range, index is {}, but the length is 3", index),
        }
    }
}

impl<const D0: usize, const D1: usize, const D2: usize> Index<i32> for Sh3<D0, D1, D2> {
    type Output = usize;
    fn index(&self, index: i32) -> &Self::Output {
        match index {
            0 => &D0,
            1 => &D1,
            2 => &D2,
            -1 => &D2,
            -2 => &D1,
            -3 => &D0,
            _ => panic!("Index out of range, index is {}, but the length is 3", index),
        }
    }
}

impl<const D0: usize, const D1: usize, const D2: usize> HasLastDim for Sh3<D0, D1, D2> {
    const LAST_DIM: usize = D2;
}

impl<const D0: usize, const D1: usize, const D2: usize> PermutableBy<Ax2<-1, -2>> for Sh3<D0, D1, D2> { type Output = Sh3<D0, D2, D1>; }
impl<const D0: usize, const D1: usize, const D2: usize> PermutableBy<Ax2<-1, 1>> for Sh3<D0, D1, D2> { type Output = Sh3<D0, D2, D1>; }
impl<const D0: usize, const D1: usize, const D2: usize> PermutableBy<Ax2<2, -2>> for Sh3<D0, D1, D2> { type Output = Sh3<D0, D2, D1>; }
impl<const D0: usize, const D1: usize, const D2: usize> PermutableBy<Ax2<2, 1>> for Sh3<D0, D1, D2> { type Output = Sh3<D0, D2, D1>; }
impl<const D0: usize, const D1: usize, const D2: usize> PermutableBy<Ax3<0, -1, -2>> for Sh3<D0, D1, D2> { type Output = Sh3<D0, D2, D1>; }
impl<const D0: usize, const D1: usize, const D2: usize> PermutableBy<Ax3<0, -1, 1>> for Sh3<D0, D1, D2> { type Output = Sh3<D0, D2, D1>; }
impl<const D0: usize, const D1: usize, const D2: usize> PermutableBy<Ax3<0, 2, -2>> for Sh3<D0, D1, D2> { type Output = Sh3<D0, D2, D1>; }
impl<const D0: usize, const D1: usize, const D2: usize> PermutableBy<Ax3<0, 2, 1>> for Sh3<D0, D1, D2> { type Output = Sh3<D0, D2, D1>; }
impl<const D0: usize, const D1: usize, const D2: usize> PermutableBy<Ax3<-3, -1, -2>> for Sh3<D0, D1, D2> { type Output = Sh3<D0, D2, D1>; }
impl<const D0: usize, const D1: usize, const D2: usize> PermutableBy<Ax3<-3, -1, 1>> for Sh3<D0, D1, D2> { type Output = Sh3<D0, D2, D1>; }
impl<const D0: usize, const D1: usize, const D2: usize> PermutableBy<Ax3<-3, 2, -2>> for Sh3<D0, D1, D2> { type Output = Sh3<D0, D2, D1>; }
impl<const D0: usize, const D1: usize, const D2: usize> PermutableBy<Ax3<-3, 2, 1>> for Sh3<D0, D1, D2> { type Output = Sh3<D0, D2, D1>; }

impl<const D0: usize, const D1: usize, const D2: usize> PermutableBy<Ax2<1, 0>> for Sh3<D0, D1, D2> { type Output = Sh3<D1, D0, D2>; }
impl<const D0: usize, const D1: usize, const D2: usize> PermutableBy<Ax2<-2, 0>> for Sh3<D0, D1, D2> { type Output = Sh3<D1, D0, D2>; }
impl<const D0: usize, const D1: usize, const D2: usize> PermutableBy<Ax2<1, -3>> for Sh3<D0, D1, D2> { type Output = Sh3<D1, D0, D2>; }
impl<const D0: usize, const D1: usize, const D2: usize> PermutableBy<Ax2<-2, -3>> for Sh3<D0, D1, D2> { type Output = Sh3<D1, D0, D2>; }
impl<const D0: usize, const D1: usize, const D2: usize> PermutableBy<Ax3<1, 0, 2>> for Sh3<D0, D1, D2> { type Output = Sh3<D1, D0, D2>; }
impl<const D0: usize, const D1: usize, const D2: usize> PermutableBy<Ax3<-2, 0, 2>> for Sh3<D0, D1, D2> { type Output = Sh3<D1, D0, D2>; }
impl<const D0: usize, const D1: usize, const D2: usize> PermutableBy<Ax3<1, -3, 2>> for Sh3<D0, D1, D2> { type Output = Sh3<D1, D0, D2>; }
impl<const D0: usize, const D1: usize, const D2: usize> PermutableBy<Ax3<1, 0, -1>> for Sh3<D0, D1, D2> { type Output = Sh3<D1, D0, D2>; }
impl<const D0: usize, const D1: usize, const D2: usize> PermutableBy<Ax3<-2, -3, 2>> for Sh3<D0, D1, D2> { type Output = Sh3<D1, D0, D2>; }
impl<const D0: usize, const D1: usize, const D2: usize> PermutableBy<Ax3<1, -3, -1>> for Sh3<D0, D1, D2> { type Output = Sh3<D1, D0, D2>; }
impl<const D0: usize, const D1: usize, const D2: usize> PermutableBy<Ax3<-2, 0, -1>> for Sh3<D0, D1, D2> { type Output = Sh3<D1, D0, D2>; }
impl<const D0: usize, const D1: usize, const D2: usize> PermutableBy<Ax3<-2, -3, -1>> for Sh3<D0, D1, D2> { type Output = Sh3<D1, D0, D2>; }

//impl<const D0: usize, const D1: usize, const D2: usize> PermutableBy<Ax3<1, 2, 0>> for Sh3<D0, D1, D2> { type Output = Sh3<D1, D2, D0>; }

//impl<const D0: usize, const D1: usize, const D2: usize> PermutableBy<Ax2<1, 0>> for Sh3<D0, D1, D2> { type Output = Sh3<D2, D0, D1>; }

//impl<const D0: usize, const D1: usize, const D2: usize> PermutableBy<Ax2<1, 0>> for Sh3<D0, D1, D2> { type Output = Sh3<D2, D1, D0>; }

/// Shape with 4 dimensions
#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub struct Sh4<const D0: usize, const D1: usize, const D2: usize, const D3: usize> {}

impl<const D0: usize, const D1: usize, const D2: usize, const D3: usize> Shape for Sh4<D0, D1, D2, D3> {
    const RANK: usize = 4;
    type Strides = [usize; 4];
    fn strides() -> Self::Strides { [D1*D2*D3, D2*D3, D3, 1] }
    fn numel() -> usize { D0*D1*D2*D3 }
}

impl<const D0: usize, const D1: usize, const D2: usize, const D3: usize> Display for Sh4<D0, D1, D2, D3> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_fmt(format_args!("{}x{}x{}x{}", D0, D1, D2, D3))
    }
}

impl<const D0: usize, const D1: usize, const D2: usize, const D3: usize> Index<usize> for Sh4<D0, D1, D2, D3> {
    type Output = usize;
    fn index(&self, index: usize) -> &Self::Output {
        match index {
            0 => &D0,
            1 => &D1,
            2 => &D2,
            3 => &D3,
            _ => panic!("Index out of range, index is {}, but the length is 4", index),
        }
    }
}

impl<const D0: usize, const D1: usize, const D2: usize, const D3: usize> Index<i32> for Sh4<D0, D1, D2, D3> {
    type Output = usize;
    fn index(&self, index: i32) -> &Self::Output {
        match index {
            0 => &D0,
            1 => &D1,
            2 => &D2,
            3 => &D3,
            -1 => &D3,
            -2 => &D2,
            -3 => &D1,
            -4 => &D0,
            _ => panic!("Index out of range, index is {}, but the length is 4", index),
        }
    }
}

impl<const D0: usize, const D1: usize, const D2: usize, const D3: usize> HasLastDim for Sh4<D0, D1, D2, D3> {
    const LAST_DIM: usize = D3;
}

/// Shape with 5 dimensions
#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub struct Sh5<const D0: usize, const D1: usize, const D2: usize, const D3: usize, const D4: usize> {}

impl<const D0: usize, const D1: usize, const D2: usize, const D3: usize, const D4: usize> Shape for Sh5<D0, D1, D2, D3, D4> {
    const RANK: usize = 4;
    type Strides = [usize; 4];
    fn strides() -> Self::Strides { [D1*D2*D3, D2*D3, D3, 1] }
    fn numel() -> usize { D0*D1*D2*D3 }
}

impl<const D0: usize, const D1: usize, const D2: usize, const D3: usize, const D4: usize> Display for Sh5<D0, D1, D2, D3, D4> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_fmt(format_args!("{}x{}x{}x{}x{}", D0, D1, D2, D3, D4))
    }
}

impl<const D0: usize, const D1: usize, const D2: usize, const D3: usize, const D4: usize> Index<usize> for Sh5<D0, D1, D2, D3, D4> {
    type Output = usize;
    fn index(&self, index: usize) -> &Self::Output {
        match index {
            0 => &D0,
            1 => &D1,
            2 => &D2,
            3 => &D3,
            4 => &D4,
            _ => panic!("Index out of range, index is {}, but the length is 4", index),
        }
    }
}

impl<const D0: usize, const D1: usize, const D2: usize, const D3: usize, const D4: usize> Index<i32> for Sh5<D0, D1, D2, D3, D4> {
    type Output = usize;
    fn index(&self, index: i32) -> &Self::Output {
        match index {
            0 => &D0,
            1 => &D1,
            2 => &D2,
            3 => &D3,
            4 => &D4,
            -1 => &D4,
            -2 => &D3,
            -3 => &D2,
            -4 => &D1,
            -5 => &D0,
            _ => panic!("Index out of range, index is {}, but the length is 4", index),
        }
    }
}

impl<const D0: usize, const D1: usize, const D2: usize, const D3: usize, const D4: usize> HasLastDim for Sh5<D0, D1, D2, D3, D4> {
    const LAST_DIM: usize = D4;
}
