/*
Operations needed:
reduce, reshape, expand, permute

This const Shape is not quite there yet, but we will see how const generic expressions evolve
and then this will be ready.
*/

//! Shape module
// TODO DOCS

use core::{
    fmt::{Debug, Display},
    ops::{Index, IndexMut},
};
//use crate::ops::Permute;

use super::{Ax1, Ax2, Ax3, Ax4, Ax5};
use super::{HasLast2Dims, HasLastDim, MatMulBy, PermutableBy, ReducableBy};

/// Shape trait
// TODO DOCS
pub trait Shape:
    Default
    + Copy
    + Clone
    + PartialEq
    + Eq
    + Debug
    + Display
    + Index<usize, Output = usize>
    + Index<i32, Output = usize>
{
    /// Rank of Shape
    const RANK: usize;
    /// Output type when calling array and strides function
    type AsArray: Index<usize, Output = usize>
        + IndexMut<usize>
        + Debug
        + IntoIterator<Item = usize>; // This is [usize; RANK], just needed because you can't write it directly
    /// Get shape as array
    fn array() -> Self::AsArray;
    /// Get shape's strides
    fn strides() -> Self::AsArray;
    /// Get shape's number of elements
    const NUMEL: usize;
    /// Check if the shape is empty, that is if Self::NUMEL == 0
    const IS_EMPTY: bool = Self::NUMEL == 0;
    /// Acces dimension at give index
    fn at(dim: usize) -> usize;
}

/// Shape with no dimensions. Used for scalars.
// TODO DOCS
#[derive(Default, Debug, PartialEq, Eq, Clone, Copy)]
pub struct Sh0 {}

impl Shape for Sh0 {
    const RANK: usize = 0;
    const NUMEL: usize = 0;
    type AsArray = [usize; 0];
    fn array() -> Self::AsArray {
        []
    }
    fn strides() -> Self::AsArray {
        []
    }
    fn at(dim: usize) -> usize {
        panic!("Index out of range, index is {}, but the length is 0", dim)
    }
}

impl Display for Sh0 {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_fmt(format_args!("()"))
    }
}

impl Index<usize> for Sh0 {
    type Output = usize;
    fn index(&self, index: usize) -> &Self::Output {
        panic!(
            "Index out of range, index is {}, but the length is 0",
            index
        )
    }
}

impl Index<i32> for Sh0 {
    type Output = usize;
    fn index(&self, index: i32) -> &Self::Output {
        panic!(
            "Index out of range, index is {}, but the length is 0",
            index
        )
    }
}

/// Shape with 1 dimension
// TODO DOCS
#[derive(Default, Debug, PartialEq, Eq, Clone, Copy)]
pub struct Sh1<const D0: usize> {}

impl<const D0: usize> Shape for Sh1<D0> {
    const RANK: usize = 1;
    const NUMEL: usize = D0;
    type AsArray = [usize; 1];
    fn array() -> Self::AsArray {
        [D0]
    }
    fn strides() -> Self::AsArray {
        [1]
    }
    fn at(dim: usize) -> usize {
        match dim {
            0 => D0,
            _ => panic!("Index out of range, index is {}, but the length is 1", dim),
        }
    }
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
            _ => panic!(
                "Index out of range, index is {}, but the length is 1",
                index
            ),
        }
    }
}

impl<const D0: usize> Index<i32> for Sh1<D0> {
    type Output = usize;
    fn index(&self, index: i32) -> &Self::Output {
        match index {
            0 => &D0,
            -1 => &D0,
            _ => panic!(
                "Index out of range, index is {}, but the length is 1",
                index
            ),
        }
    }
}

impl<const D0: usize> HasLastDim for Sh1<D0> {
    const LAST_DIM: usize = D0;
}

impl<const D0: usize> ReducableBy<Ax1<0>> for Sh1<D0> {
    type Output = Sh1<1>;
}

/// Shape with 2 dimensions
// TODO DOCS
#[derive(Default, Debug, PartialEq, Eq, Clone, Copy)]
pub struct Sh2<const D0: usize, const D1: usize> {}

impl<const D0: usize, const D1: usize> Shape for Sh2<D0, D1> {
    const RANK: usize = 2;
    const NUMEL: usize = D0 * D1;
    type AsArray = [usize; 2];
    fn array() -> Self::AsArray {
        [D0, D1]
    }
    fn strides() -> Self::AsArray {
        [D1, 1]
    }
    fn at(dim: usize) -> usize {
        match dim {
            0 => D0,
            1 => D1,
            _ => panic!("Index out of range, index is {}, but the length is 2", dim),
        }
    }
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
            _ => panic!(
                "Index out of range, index is {}, but the length is 2",
                index
            ),
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
            _ => panic!(
                "Index out of range, index is {}, but the length is 2",
                index
            ),
        }
    }
}

impl<const D0: usize, const D1: usize> HasLastDim for Sh2<D0, D1> {
    const LAST_DIM: usize = D1;
}
impl<const D0: usize, const D1: usize> HasLast2Dims for Sh2<D0, D1> {
    const LAST_DIM_2: usize = D0;
}

impl<const D0: usize, const D1: usize> PermutableBy<Ax2<-1, -2>> for Sh2<D0, D1> {
    type Output = Sh2<D1, D0>;
}
impl<const D0: usize, const D1: usize> PermutableBy<Ax2<-2, -1>> for Sh2<D0, D1> {
    type Output = Sh2<D0, D1>;
}
impl<const D0: usize, const D1: usize> PermutableBy<Ax2<0, 1>> for Sh2<D0, D1> {
    type Output = Sh2<D0, D1>;
}
impl<const D0: usize, const D1: usize> PermutableBy<Ax2<1, 0>> for Sh2<D0, D1> {
    type Output = Sh2<D1, D0>;
}

impl<const D0: usize, const D1: usize> ReducableBy<Ax1<0>> for Sh2<D0, D1> {
    type Output = Sh2<1, D1>;
}
impl<const D0: usize, const D1: usize> ReducableBy<Ax1<1>> for Sh2<D0, D1> {
    type Output = Sh2<D0, 1>;
}
impl<const D0: usize, const D1: usize> ReducableBy<Ax2<0, 1>> for Sh2<D0, D1> {
    type Output = Sh2<1, 1>;
}

impl<const M: usize, const K: usize, const N: usize> MatMulBy<Sh2<K, N>> for Sh2<M, K> {
    type Output = Sh2<M, N>;
}

/// Shape with 3 dimensions
// TODO DOCS
#[derive(Default, Debug, PartialEq, Eq, Clone, Copy)]
pub struct Sh3<const D0: usize, const D1: usize, const D2: usize> {}

impl<const D0: usize, const D1: usize, const D2: usize> Shape for Sh3<D0, D1, D2> {
    const RANK: usize = 3;
    const NUMEL: usize = D0 * D1 * D2;
    type AsArray = [usize; 3];
    fn array() -> Self::AsArray {
        [D0, D1, D2]
    }
    fn strides() -> Self::AsArray {
        [D1 * D2, D2, 1]
    }
    fn at(dim: usize) -> usize {
        match dim {
            0 => D0,
            1 => D1,
            2 => D2,
            _ => panic!("Index out of range, index is {}, but the length is 3", dim),
        }
    }
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
            _ => panic!(
                "Index out of range, index is {}, but the length is 3",
                index
            ),
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
            _ => panic!(
                "Index out of range, index is {}, but the length is 3",
                index
            ),
        }
    }
}

impl<const D0: usize, const D1: usize, const D2: usize> HasLastDim for Sh3<D0, D1, D2> {
    const LAST_DIM: usize = D2;
}
impl<const D0: usize, const D1: usize, const D2: usize> HasLast2Dims for Sh3<D0, D1, D2> {
    const LAST_DIM_2: usize = D1;
}

impl<const D0: usize, const D1: usize, const D2: usize> PermutableBy<Ax2<-1, -2>>
    for Sh3<D0, D1, D2>
{
    type Output = Sh3<D0, D2, D1>;
}
impl<const D0: usize, const D1: usize, const D2: usize> PermutableBy<Ax2<-2, -1>>
    for Sh3<D0, D1, D2>
{
    type Output = Sh3<D0, D1, D2>;
}
impl<const D0: usize, const D1: usize, const D2: usize> PermutableBy<Ax3<0, 1, 2>>
    for Sh3<D0, D1, D2>
{
    type Output = Sh3<D0, D1, D2>;
}
impl<const D0: usize, const D1: usize, const D2: usize> PermutableBy<Ax3<0, 2, 1>>
    for Sh3<D0, D1, D2>
{
    type Output = Sh3<D0, D2, D1>;
}
impl<const D0: usize, const D1: usize, const D2: usize> PermutableBy<Ax3<1, 0, 2>>
    for Sh3<D0, D1, D2>
{
    type Output = Sh3<D1, D0, D2>;
}
impl<const D0: usize, const D1: usize, const D2: usize> PermutableBy<Ax3<1, 2, 0>>
    for Sh3<D0, D1, D2>
{
    type Output = Sh3<D1, D2, D0>;
}
impl<const D0: usize, const D1: usize, const D2: usize> PermutableBy<Ax3<2, 0, 1>>
    for Sh3<D0, D1, D2>
{
    type Output = Sh3<D2, D0, D1>;
}
impl<const D0: usize, const D1: usize, const D2: usize> PermutableBy<Ax3<2, 1, 0>>
    for Sh3<D0, D1, D2>
{
    type Output = Sh3<D2, D1, D0>;
}

impl<const D0: usize, const D1: usize, const D2: usize> ReducableBy<Ax1<0>> for Sh3<D0, D1, D2> {
    type Output = Sh3<1, D1, D2>;
}
impl<const D0: usize, const D1: usize, const D2: usize> ReducableBy<Ax1<1>> for Sh3<D0, D1, D2> {
    type Output = Sh3<D0, 1, D2>;
}
impl<const D0: usize, const D1: usize, const D2: usize> ReducableBy<Ax1<2>> for Sh3<D0, D1, D2> {
    type Output = Sh3<D0, D1, 1>;
}
impl<const D0: usize, const D1: usize, const D2: usize> ReducableBy<Ax2<0, 1>> for Sh3<D0, D1, D2> {
    type Output = Sh3<1, 1, D2>;
}
impl<const D0: usize, const D1: usize, const D2: usize> ReducableBy<Ax2<0, 2>> for Sh3<D0, D1, D2> {
    type Output = Sh3<1, D1, 1>;
}
impl<const D0: usize, const D1: usize, const D2: usize> ReducableBy<Ax2<1, 2>> for Sh3<D0, D1, D2> {
    type Output = Sh3<D0, 1, 1>;
}
impl<const D0: usize, const D1: usize, const D2: usize> ReducableBy<Ax3<0, 1, 2>>
    for Sh3<D0, D1, D2>
{
    type Output = Sh3<1, 1, 1>;
}

impl<const D2: usize, const M: usize, const K: usize, const N: usize> MatMulBy<Sh3<D2, K, N>>
    for Sh3<D2, M, K>
{
    type Output = Sh3<D2, M, N>;
}

/// Shape with 4 dimensions
// TODO DOCS
#[derive(Default, Debug, PartialEq, Eq, Clone, Copy)]
pub struct Sh4<const D0: usize, const D1: usize, const D2: usize, const D3: usize> {}

impl<const D0: usize, const D1: usize, const D2: usize, const D3: usize> Shape
    for Sh4<D0, D1, D2, D3>
{
    const RANK: usize = 4;
    const NUMEL: usize = D0 * D1 * D2 * D3;
    type AsArray = [usize; 4];
    fn array() -> Self::AsArray {
        [D0, D1, D2, D3]
    }
    fn strides() -> Self::AsArray {
        [D1 * D2 * D3, D2 * D3, D3, 1]
    }
    fn at(dim: usize) -> usize {
        match dim {
            0 => D0,
            1 => D1,
            2 => D2,
            3 => D3,
            _ => panic!("Index out of range, index is {}, but the length is 4", dim),
        }
    }
}

impl<const D0: usize, const D1: usize, const D2: usize, const D3: usize> Display
    for Sh4<D0, D1, D2, D3>
{
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_fmt(format_args!("{}x{}x{}x{}", D0, D1, D2, D3))
    }
}

impl<const D0: usize, const D1: usize, const D2: usize, const D3: usize> Index<usize>
    for Sh4<D0, D1, D2, D3>
{
    type Output = usize;
    fn index(&self, index: usize) -> &Self::Output {
        match index {
            0 => &D0,
            1 => &D1,
            2 => &D2,
            3 => &D3,
            _ => panic!(
                "Index out of range, index is {}, but the length is 4",
                index
            ),
        }
    }
}

impl<const D0: usize, const D1: usize, const D2: usize, const D3: usize> Index<i32>
    for Sh4<D0, D1, D2, D3>
{
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
            _ => panic!(
                "Index out of range, index is {}, but the length is 4",
                index
            ),
        }
    }
}

impl<const D0: usize, const D1: usize, const D2: usize, const D3: usize> HasLastDim
    for Sh4<D0, D1, D2, D3>
{
    const LAST_DIM: usize = D3;
}
impl<const D0: usize, const D1: usize, const D2: usize, const D3: usize> HasLast2Dims
    for Sh4<D0, D1, D2, D3>
{
    const LAST_DIM_2: usize = D2;
}

impl<const D0: usize, const D1: usize, const D2: usize, const D3: usize> PermutableBy<Ax2<-1, -2>>
    for Sh4<D0, D1, D2, D3>
{
    type Output = Sh4<D0, D1, D3, D2>;
}
impl<const D0: usize, const D1: usize, const D2: usize, const D3: usize> PermutableBy<Ax2<-2, -1>>
    for Sh4<D0, D1, D2, D3>
{
    type Output = Sh4<D0, D1, D2, D3>;
}
impl<const D0: usize, const D1: usize, const D2: usize, const D3: usize>
    PermutableBy<Ax4<0, 1, 2, 3>> for Sh4<D0, D1, D2, D3>
{
    type Output = Sh4<D0, D1, D2, D3>;
}
impl<const D0: usize, const D1: usize, const D2: usize, const D3: usize>
    PermutableBy<Ax4<0, 1, 3, 2>> for Sh4<D0, D1, D2, D3>
{
    type Output = Sh4<D0, D1, D3, D2>;
}
impl<const D0: usize, const D1: usize, const D2: usize, const D3: usize>
    PermutableBy<Ax4<0, 2, 1, 3>> for Sh4<D0, D1, D2, D3>
{
    type Output = Sh4<D0, D2, D1, D3>;
}
impl<const D0: usize, const D1: usize, const D2: usize, const D3: usize>
    PermutableBy<Ax4<0, 2, 3, 1>> for Sh4<D0, D1, D2, D3>
{
    type Output = Sh4<D0, D2, D3, D1>;
}
impl<const D0: usize, const D1: usize, const D2: usize, const D3: usize>
    PermutableBy<Ax4<0, 3, 1, 2>> for Sh4<D0, D1, D2, D3>
{
    type Output = Sh4<D0, D3, D1, D2>;
}
impl<const D0: usize, const D1: usize, const D2: usize, const D3: usize>
    PermutableBy<Ax4<0, 3, 2, 1>> for Sh4<D0, D1, D2, D3>
{
    type Output = Sh4<D0, D3, D2, D1>;
}
impl<const D0: usize, const D1: usize, const D2: usize, const D3: usize>
    PermutableBy<Ax4<1, 0, 2, 3>> for Sh4<D0, D1, D2, D3>
{
    type Output = Sh4<D1, D0, D2, D3>;
}
impl<const D0: usize, const D1: usize, const D2: usize, const D3: usize>
    PermutableBy<Ax4<1, 0, 3, 2>> for Sh4<D0, D1, D2, D3>
{
    type Output = Sh4<D1, D0, D3, D2>;
}
impl<const D0: usize, const D1: usize, const D2: usize, const D3: usize>
    PermutableBy<Ax4<1, 2, 0, 3>> for Sh4<D0, D1, D2, D3>
{
    type Output = Sh4<D1, D2, D0, D3>;
}
impl<const D0: usize, const D1: usize, const D2: usize, const D3: usize>
    PermutableBy<Ax4<1, 2, 3, 0>> for Sh4<D0, D1, D2, D3>
{
    type Output = Sh4<D1, D2, D3, D0>;
}
impl<const D0: usize, const D1: usize, const D2: usize, const D3: usize>
    PermutableBy<Ax4<1, 3, 0, 2>> for Sh4<D0, D1, D2, D3>
{
    type Output = Sh4<D1, D3, D0, D2>;
}
impl<const D0: usize, const D1: usize, const D2: usize, const D3: usize>
    PermutableBy<Ax4<1, 3, 2, 0>> for Sh4<D0, D1, D2, D3>
{
    type Output = Sh4<D1, D3, D2, D0>;
}
impl<const D0: usize, const D1: usize, const D2: usize, const D3: usize>
    PermutableBy<Ax4<2, 0, 1, 3>> for Sh4<D0, D1, D2, D3>
{
    type Output = Sh4<D2, D0, D1, D3>;
}
impl<const D0: usize, const D1: usize, const D2: usize, const D3: usize>
    PermutableBy<Ax4<2, 0, 3, 1>> for Sh4<D0, D1, D2, D3>
{
    type Output = Sh4<D2, D0, D3, D1>;
}
impl<const D0: usize, const D1: usize, const D2: usize, const D3: usize>
    PermutableBy<Ax4<2, 1, 0, 3>> for Sh4<D0, D1, D2, D3>
{
    type Output = Sh4<D2, D1, D0, D3>;
}
impl<const D0: usize, const D1: usize, const D2: usize, const D3: usize>
    PermutableBy<Ax4<2, 1, 3, 0>> for Sh4<D0, D1, D2, D3>
{
    type Output = Sh4<D2, D1, D3, D0>;
}
impl<const D0: usize, const D1: usize, const D2: usize, const D3: usize>
    PermutableBy<Ax4<2, 3, 0, 1>> for Sh4<D0, D1, D2, D3>
{
    type Output = Sh4<D2, D3, D0, D1>;
}
impl<const D0: usize, const D1: usize, const D2: usize, const D3: usize>
    PermutableBy<Ax4<2, 3, 1, 0>> for Sh4<D0, D1, D2, D3>
{
    type Output = Sh4<D2, D3, D1, D0>;
}
impl<const D0: usize, const D1: usize, const D2: usize, const D3: usize>
    PermutableBy<Ax4<3, 0, 1, 2>> for Sh4<D0, D1, D2, D3>
{
    type Output = Sh4<D3, D0, D1, D2>;
}
impl<const D0: usize, const D1: usize, const D2: usize, const D3: usize>
    PermutableBy<Ax4<3, 0, 2, 1>> for Sh4<D0, D1, D2, D3>
{
    type Output = Sh4<D3, D0, D2, D1>;
}
impl<const D0: usize, const D1: usize, const D2: usize, const D3: usize>
    PermutableBy<Ax4<3, 1, 0, 2>> for Sh4<D0, D1, D2, D3>
{
    type Output = Sh4<D3, D1, D0, D2>;
}
impl<const D0: usize, const D1: usize, const D2: usize, const D3: usize>
    PermutableBy<Ax4<3, 1, 2, 0>> for Sh4<D0, D1, D2, D3>
{
    type Output = Sh4<D3, D1, D2, D0>;
}
impl<const D0: usize, const D1: usize, const D2: usize, const D3: usize>
    PermutableBy<Ax4<3, 2, 0, 1>> for Sh4<D0, D1, D2, D3>
{
    type Output = Sh4<D3, D2, D0, D1>;
}
impl<const D0: usize, const D1: usize, const D2: usize, const D3: usize>
    PermutableBy<Ax4<3, 2, 1, 0>> for Sh4<D0, D1, D2, D3>
{
    type Output = Sh4<D3, D2, D1, D0>;
}

impl<const D0: usize, const D1: usize, const D2: usize, const D3: usize> ReducableBy<Ax1<0>>
    for Sh4<D0, D1, D2, D3>
{
    type Output = Sh4<1, D1, D2, D3>;
}
impl<const D0: usize, const D1: usize, const D2: usize, const D3: usize> ReducableBy<Ax1<1>>
    for Sh4<D0, D1, D2, D3>
{
    type Output = Sh4<D0, 1, D2, D3>;
}
impl<const D0: usize, const D1: usize, const D2: usize, const D3: usize> ReducableBy<Ax1<2>>
    for Sh4<D0, D1, D2, D3>
{
    type Output = Sh4<D0, D1, 1, D3>;
}
impl<const D0: usize, const D1: usize, const D2: usize, const D3: usize> ReducableBy<Ax1<3>>
    for Sh4<D0, D1, D2, D3>
{
    type Output = Sh4<D0, D1, D2, 1>;
}
impl<const D0: usize, const D1: usize, const D2: usize, const D3: usize> ReducableBy<Ax2<0, 1>>
    for Sh4<D0, D1, D2, D3>
{
    type Output = Sh4<1, 1, D2, D3>;
}
impl<const D0: usize, const D1: usize, const D2: usize, const D3: usize> ReducableBy<Ax2<0, 2>>
    for Sh4<D0, D1, D2, D3>
{
    type Output = Sh4<1, D1, 1, D3>;
}
impl<const D0: usize, const D1: usize, const D2: usize, const D3: usize> ReducableBy<Ax2<0, 3>>
    for Sh4<D0, D1, D2, D3>
{
    type Output = Sh4<1, D1, D2, 1>;
}
impl<const D0: usize, const D1: usize, const D2: usize, const D3: usize> ReducableBy<Ax2<1, 2>>
    for Sh4<D0, D1, D2, D3>
{
    type Output = Sh4<D0, 1, 1, D3>;
}
impl<const D0: usize, const D1: usize, const D2: usize, const D3: usize> ReducableBy<Ax2<1, 3>>
    for Sh4<D0, D1, D2, D3>
{
    type Output = Sh4<D0, 1, D2, 1>;
}
impl<const D0: usize, const D1: usize, const D2: usize, const D3: usize> ReducableBy<Ax2<2, 3>>
    for Sh4<D0, D1, D2, D3>
{
    type Output = Sh4<D0, D1, 1, 1>;
}
impl<const D0: usize, const D1: usize, const D2: usize, const D3: usize> ReducableBy<Ax3<0, 1, 2>>
    for Sh4<D0, D1, D2, D3>
{
    type Output = Sh4<1, 1, 1, D3>;
}
impl<const D0: usize, const D1: usize, const D2: usize, const D3: usize> ReducableBy<Ax3<0, 1, 3>>
    for Sh4<D0, D1, D2, D3>
{
    type Output = Sh4<1, 1, D2, 1>;
}
impl<const D0: usize, const D1: usize, const D2: usize, const D3: usize> ReducableBy<Ax3<0, 2, 3>>
    for Sh4<D0, D1, D2, D3>
{
    type Output = Sh4<1, D1, 1, 1>;
}
impl<const D0: usize, const D1: usize, const D2: usize, const D3: usize> ReducableBy<Ax3<1, 2, 3>>
    for Sh4<D0, D1, D2, D3>
{
    type Output = Sh4<D0, 1, 1, 1>;
}
impl<const D0: usize, const D1: usize, const D2: usize, const D3: usize>
    ReducableBy<Ax4<0, 1, 2, 3>> for Sh4<D0, D1, D2, D3>
{
    type Output = Sh4<1, 1, 1, 1>;
}

impl<const D3: usize, const D2: usize, const M: usize, const K: usize, const N: usize>
    MatMulBy<Sh4<D3, D2, K, N>> for Sh4<D3, D2, M, K>
{
    type Output = Sh4<D3, D2, M, N>;
}

/// Shape with 5 dimensions
// TODO DOCS
#[derive(Default, Debug, PartialEq, Eq, Clone, Copy)]
pub struct Sh5<const D0: usize, const D1: usize, const D2: usize, const D3: usize, const D4: usize>
{}

impl<const D0: usize, const D1: usize, const D2: usize, const D3: usize, const D4: usize> Shape
    for Sh5<D0, D1, D2, D3, D4>
{
    const RANK: usize = 5;
    const NUMEL: usize = D0 * D1 * D2 * D3 * D4;
    type AsArray = [usize; 5];
    fn array() -> Self::AsArray {
        [D0, D1, D2, D3, D4]
    }
    fn strides() -> Self::AsArray {
        [D1 * D2 * D3 * D4, D2 * D3 * D4, D3 * D4, D4, 1]
    }
    fn at(dim: usize) -> usize {
        match dim {
            0 => D0,
            1 => D1,
            2 => D2,
            3 => D3,
            4 => D4,
            _ => panic!("Index out of range, index is {}, but the length is 4", dim),
        }
    }
}

impl<const D0: usize, const D1: usize, const D2: usize, const D3: usize, const D4: usize> Display
    for Sh5<D0, D1, D2, D3, D4>
{
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_fmt(format_args!("{}x{}x{}x{}x{}", D0, D1, D2, D3, D4))
    }
}

impl<const D0: usize, const D1: usize, const D2: usize, const D3: usize, const D4: usize>
    Index<usize> for Sh5<D0, D1, D2, D3, D4>
{
    type Output = usize;
    fn index(&self, index: usize) -> &Self::Output {
        match index {
            0 => &D0,
            1 => &D1,
            2 => &D2,
            3 => &D3,
            4 => &D4,
            _ => panic!(
                "Index out of range, index is {}, but the length is 4",
                index
            ),
        }
    }
}

impl<const D0: usize, const D1: usize, const D2: usize, const D3: usize, const D4: usize> Index<i32>
    for Sh5<D0, D1, D2, D3, D4>
{
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
            _ => panic!(
                "Index out of range, index is {}, but the length is 4",
                index
            ),
        }
    }
}

impl<const D0: usize, const D1: usize, const D2: usize, const D3: usize, const D4: usize> HasLastDim
    for Sh5<D0, D1, D2, D3, D4>
{
    const LAST_DIM: usize = D4;
}
impl<const D0: usize, const D1: usize, const D2: usize, const D3: usize, const D4: usize>
    HasLast2Dims for Sh5<D0, D1, D2, D3, D4>
{
    const LAST_DIM_2: usize = D4;
}

impl<const D0: usize, const D1: usize, const D2: usize, const D3: usize, const D4: usize>
    PermutableBy<Ax2<-1, -2>> for Sh5<D0, D1, D2, D3, D4>
{
    type Output = Sh5<D0, D1, D2, D4, D3>;
}
impl<const D0: usize, const D1: usize, const D2: usize, const D3: usize, const D4: usize>
    PermutableBy<Ax2<-2, -1>> for Sh5<D0, D1, D2, D3, D4>
{
    type Output = Sh5<D0, D1, D2, D3, D4>;
}
impl<const D0: usize, const D1: usize, const D2: usize, const D3: usize, const D4: usize>
    PermutableBy<Ax5<0, 1, 2, 3, 4>> for Sh5<D0, D1, D2, D3, D4>
{
    type Output = Sh5<D0, D1, D2, D3, D4>;
}
impl<const D0: usize, const D1: usize, const D2: usize, const D3: usize, const D4: usize>
    PermutableBy<Ax5<0, 1, 2, 4, 3>> for Sh5<D0, D1, D2, D3, D4>
{
    type Output = Sh5<D0, D1, D2, D4, D3>;
}

impl<const D0: usize, const D1: usize, const D2: usize, const D3: usize, const D4: usize>
    ReducableBy<Ax1<0>> for Sh5<D0, D1, D2, D3, D4>
{
    type Output = Sh5<1, D1, D2, D3, D4>;
}
impl<const D0: usize, const D1: usize, const D2: usize, const D3: usize, const D4: usize>
    ReducableBy<Ax1<1>> for Sh5<D0, D1, D2, D3, D4>
{
    type Output = Sh5<D0, 1, D2, D3, D4>;
}
impl<const D0: usize, const D1: usize, const D2: usize, const D3: usize, const D4: usize>
    ReducableBy<Ax1<2>> for Sh5<D0, D1, D2, D3, D4>
{
    type Output = Sh5<D0, D1, 1, D3, D4>;
}
impl<const D0: usize, const D1: usize, const D2: usize, const D3: usize, const D4: usize>
    ReducableBy<Ax1<3>> for Sh5<D0, D1, D2, D3, D4>
{
    type Output = Sh5<D0, D1, D2, 1, D4>;
}
impl<const D0: usize, const D1: usize, const D2: usize, const D3: usize, const D4: usize>
    ReducableBy<Ax1<4>> for Sh5<D0, D1, D2, D3, D4>
{
    type Output = Sh5<D0, D1, D2, D3, 1>;
}
impl<const D0: usize, const D1: usize, const D2: usize, const D3: usize, const D4: usize>
    ReducableBy<Ax2<0, 1>> for Sh5<D0, D1, D2, D3, D4>
{
    type Output = Sh5<1, 1, D2, D3, D4>;
}
impl<const D0: usize, const D1: usize, const D2: usize, const D3: usize, const D4: usize>
    ReducableBy<Ax2<0, 2>> for Sh5<D0, D1, D2, D3, D4>
{
    type Output = Sh5<1, D1, 1, D3, D4>;
}
impl<const D0: usize, const D1: usize, const D2: usize, const D3: usize, const D4: usize>
    ReducableBy<Ax2<0, 3>> for Sh5<D0, D1, D2, D3, D4>
{
    type Output = Sh5<1, D1, D2, 1, D4>;
}
impl<const D0: usize, const D1: usize, const D2: usize, const D3: usize, const D4: usize>
    ReducableBy<Ax2<0, 4>> for Sh5<D0, D1, D2, D3, D4>
{
    type Output = Sh5<1, D1, D2, D3, 1>;
}
impl<const D0: usize, const D1: usize, const D2: usize, const D3: usize, const D4: usize>
    ReducableBy<Ax2<1, 2>> for Sh5<D0, D1, D2, D3, D4>
{
    type Output = Sh5<D0, 1, 1, D3, D4>;
}
impl<const D0: usize, const D1: usize, const D2: usize, const D3: usize, const D4: usize>
    ReducableBy<Ax2<1, 3>> for Sh5<D0, D1, D2, D3, D4>
{
    type Output = Sh5<D0, 1, D2, 1, D4>;
}
impl<const D0: usize, const D1: usize, const D2: usize, const D3: usize, const D4: usize>
    ReducableBy<Ax2<1, 4>> for Sh5<D0, D1, D2, D3, D4>
{
    type Output = Sh5<D0, 1, D2, D3, 1>;
}
impl<const D0: usize, const D1: usize, const D2: usize, const D3: usize, const D4: usize>
    ReducableBy<Ax2<2, 3>> for Sh5<D0, D1, D2, D3, D4>
{
    type Output = Sh5<D0, D1, 1, 1, D4>;
}
impl<const D0: usize, const D1: usize, const D2: usize, const D3: usize, const D4: usize>
    ReducableBy<Ax2<2, 4>> for Sh5<D0, D1, D2, D3, D4>
{
    type Output = Sh5<D0, D1, 1, D3, 1>;
}
impl<const D0: usize, const D1: usize, const D2: usize, const D3: usize, const D4: usize>
    ReducableBy<Ax2<3, 4>> for Sh5<D0, D1, D2, D3, D4>
{
    type Output = Sh5<D0, D1, D2, 1, 1>;
}
impl<const D0: usize, const D1: usize, const D2: usize, const D3: usize, const D4: usize>
    ReducableBy<Ax3<0, 1, 2>> for Sh5<D0, D1, D2, D3, D4>
{
    type Output = Sh5<1, 1, 1, D3, D4>;
}
impl<const D0: usize, const D1: usize, const D2: usize, const D3: usize, const D4: usize>
    ReducableBy<Ax3<0, 1, 3>> for Sh5<D0, D1, D2, D3, D4>
{
    type Output = Sh5<1, 1, D2, 1, D4>;
}
impl<const D0: usize, const D1: usize, const D2: usize, const D3: usize, const D4: usize>
    ReducableBy<Ax3<0, 1, 4>> for Sh5<D0, D1, D2, D3, D4>
{
    type Output = Sh5<1, 1, D2, D3, 1>;
}
impl<const D0: usize, const D1: usize, const D2: usize, const D3: usize, const D4: usize>
    ReducableBy<Ax3<0, 2, 3>> for Sh5<D0, D1, D2, D3, D4>
{
    type Output = Sh5<1, D1, 1, 1, D4>;
}
impl<const D0: usize, const D1: usize, const D2: usize, const D3: usize, const D4: usize>
    ReducableBy<Ax3<0, 2, 4>> for Sh5<D0, D1, D2, D3, D4>
{
    type Output = Sh5<1, D1, 1, D3, 1>;
}
impl<const D0: usize, const D1: usize, const D2: usize, const D3: usize, const D4: usize>
    ReducableBy<Ax3<0, 3, 4>> for Sh5<D0, D1, D2, D3, D4>
{
    type Output = Sh5<1, D1, D2, 1, 1>;
}
impl<const D0: usize, const D1: usize, const D2: usize, const D3: usize, const D4: usize>
    ReducableBy<Ax3<1, 2, 3>> for Sh5<D0, D1, D2, D3, D4>
{
    type Output = Sh5<D0, 1, 1, 1, D4>;
}
impl<const D0: usize, const D1: usize, const D2: usize, const D3: usize, const D4: usize>
    ReducableBy<Ax3<1, 2, 4>> for Sh5<D0, D1, D2, D3, D4>
{
    type Output = Sh5<D0, 1, 1, D3, 1>;
}
impl<const D0: usize, const D1: usize, const D2: usize, const D3: usize, const D4: usize>
    ReducableBy<Ax3<1, 3, 4>> for Sh5<D0, D1, D2, D3, D4>
{
    type Output = Sh5<D0, 1, D2, 1, 1>;
}
impl<const D0: usize, const D1: usize, const D2: usize, const D3: usize, const D4: usize>
    ReducableBy<Ax3<2, 3, 4>> for Sh5<D0, D1, D2, D3, D4>
{
    type Output = Sh5<D0, D1, 1, 1, 1>;
}
impl<const D0: usize, const D1: usize, const D2: usize, const D3: usize, const D4: usize>
    ReducableBy<Ax4<0, 1, 2, 3>> for Sh5<D0, D1, D2, D3, D4>
{
    type Output = Sh5<1, 1, 1, 1, D4>;
}
impl<const D0: usize, const D1: usize, const D2: usize, const D3: usize, const D4: usize>
    ReducableBy<Ax4<0, 1, 2, 4>> for Sh5<D0, D1, D2, D3, D4>
{
    type Output = Sh5<1, 1, 1, D3, 1>;
}
impl<const D0: usize, const D1: usize, const D2: usize, const D3: usize, const D4: usize>
    ReducableBy<Ax4<0, 1, 3, 4>> for Sh5<D0, D1, D2, D3, D4>
{
    type Output = Sh5<1, 1, D2, 1, 1>;
}
impl<const D0: usize, const D1: usize, const D2: usize, const D3: usize, const D4: usize>
    ReducableBy<Ax4<0, 2, 3, 4>> for Sh5<D0, D1, D2, D3, D4>
{
    type Output = Sh5<1, D1, 1, 1, 1>;
}
impl<const D0: usize, const D1: usize, const D2: usize, const D3: usize, const D4: usize>
    ReducableBy<Ax4<1, 2, 3, 4>> for Sh5<D0, D1, D2, D3, D4>
{
    type Output = Sh5<D0, 1, 1, 1, 1>;
}
impl<const D0: usize, const D1: usize, const D2: usize, const D3: usize, const D4: usize>
    ReducableBy<Ax5<0, 1, 2, 3, 4>> for Sh5<D0, D1, D2, D3, D4>
{
    type Output = Sh5<1, 1, 1, 1, 1>;
}

impl<
        const D4: usize,
        const D3: usize,
        const D2: usize,
        const M: usize,
        const K: usize,
        const N: usize,
    > MatMulBy<Sh5<D4, D3, D2, K, N>> for Sh5<D4, D3, D2, M, K>
{
    type Output = Sh5<D4, D3, D2, M, N>;
}
