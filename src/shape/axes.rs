//! Axes module

use core::{fmt::{Debug, Display}, ops::Index};

/// Axes trait
pub trait Axes: Copy + Clone + PartialEq + Eq + Debug + Display + Index<usize> + Index<i32>  {
    /// Rank
    const RANK: usize;
    // Ordered axes
    //type Argsort: Axes;
}

/// Type that represents all axes
/*#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub struct AxA {}*/

/// Single axis
#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub struct Ax1<const A0: i32> {}

impl<const A0: i32> Axes for Ax1<A0> {
    const RANK: usize = 1;
    //type Argsort = Ax1<A0>;
}

impl<const A0: i32> Display for Ax1<A0> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_fmt(format_args!("({})", A0))
    }
}

impl<const A0: i32> Index<usize> for Ax1<A0> {
    type Output = i32;
    fn index(&self, index: usize) -> &Self::Output {
        match index {
            0 => &A0,
            _ => panic!("Index out of range, index is {}, but the length is 1", index),
        }
    }
}

impl<const A0: i32> Index<i32> for Ax1<A0> {
    type Output = i32;
    fn index(&self, index: i32) -> &Self::Output {
        match index {
            0 => &A0,
            -1 => &A0,
            _ => panic!("Index out of range, index is {}, but the length is 1", index),
        }
    }
}

/// Two axes
#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub struct Ax2<const A0: i32, const A1: i32> {}

impl<const A0: i32, const A1: i32> Axes for Ax2<A0, A1> {
    const RANK: usize = 2;
}

impl<const A0: i32, const A1: i32> Display for Ax2<A0, A1> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_fmt(format_args!("({}, {})", A0, A1))
    }
}

impl<const A0: i32, const A1: i32> Index<usize> for Ax2<A0, A1> {
    type Output = i32;
    fn index(&self, index: usize) -> &Self::Output {
        match index {
            0 => &A0,
            1 => &A1,
            _ => panic!("Index out of range, index is {}, but the length is 2", index),
        }
    }
}

impl<const A0: i32, const A1: i32> Index<i32> for Ax2<A0, A1> {
    type Output = i32;
    fn index(&self, index: i32) -> &Self::Output {
        match index {
            0 => &A0,
            1 => &A1,
            -1 => &A1,
            -2 => &A0,
            _ => panic!("Index out of range, index is {}, but the length is 2", index),
        }
    }
}

/// Three axes
#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub struct Ax3<const A0: i32, const A1: i32, const A2: i32> {}

impl<const A0: i32, const A1: i32, const A2: i32> Axes for Ax3<A0, A1, A2> {
    const RANK: usize = 3;
    //type Argsort = Ax3<0, 1, 2>;
}

impl<const A0: i32, const A1: i32, const A2: i32> Display for Ax3<A0, A1, A2> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_fmt(format_args!("({}, {}, {})", A0, A1, A2))
    }
}

impl<const A0: i32, const A1: i32, const A2: i32> Index<usize> for Ax3<A0, A1, A2> {
    type Output = i32;
    fn index(&self, index: usize) -> &Self::Output {
        match index {
            0 => &A0,
            1 => &A1,
            2 => &A2,
            _ => panic!("Index out of range, index is {}, but the length is 3", index),
        }
    }
}

impl<const A0: i32, const A1: i32, const A2: i32> Index<i32> for Ax3<A0, A1, A2> {
    type Output = i32;
    fn index(&self, index: i32) -> &Self::Output {
        match index {
            0 => &A0,
            1 => &A1,
            2 => &A2,
            -1 => &A2,
            -2 => &A1,
            -3 => &A0,
            _ => panic!("Index out of range, index is {}, but the length is 3", index),
        }
    }
}

/// Four axes
#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub struct Ax4<const A0: i32, const A1: i32, const A2: i32, const A3: i32> {}

impl<const A0: i32, const A1: i32, const A2: i32, const A3: i32> Display for Ax4<A0, A1, A2, A3> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_fmt(format_args!("({}, {}, {}, {})", A0, A1, A2, A3))
    }
}

impl<const A0: i32, const A1: i32, const A2: i32, const A3: i32> Index<usize> for Ax4<A0, A1, A2, A3> {
    type Output = i32;
    fn index(&self, index: usize) -> &Self::Output {
        match index {
            0 => &A0,
            1 => &A1,
            2 => &A2,
            3 => &A3,
            _ => panic!("Index out of range, index is {}, but the length is 4", index),
        }
    }
}

impl<const A0: i32, const A1: i32, const A2: i32, const A3: i32> Index<i32> for Ax4<A0, A1, A2, A3> {
    type Output = i32;
    fn index(&self, index: i32) -> &Self::Output {
        match index {
            0 => &A0,
            1 => &A1,
            2 => &A2,
            3 => &A3,
            -1 => &A3,
            -2 => &A2,
            -3 => &A1,
            -4 => &A0,
            _ => panic!("Index out of range, index is {}, but the length is 4", index),
        }
    }
}

/// Five axes
#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub struct Ax5<const A0: i32, const A1: i32, const A2: i32, const A3: i32, const A4: i32> {}

impl<const A0: i32, const A1: i32, const A2: i32, const A3: i32, const A4: i32> Display for Ax5<A0, A1, A2, A3, A4> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_fmt(format_args!("({}, {}, {}, {}, {})", A0, A1, A2, A3, A4))
    }
}

impl<const A0: i32, const A1: i32, const A2: i32, const A3: i32, const A4: i32> Index<usize> for Ax5<A0, A1, A2, A3, A4> {
    type Output = i32;
    fn index(&self, index: usize) -> &Self::Output {
        match index {
            0 => &A0,
            1 => &A1,
            2 => &A2,
            3 => &A3,
            4 => &A4,
            _ => panic!("Index out of range, index is {}, but the length is 5", index),
        }
    }
}

impl<const A0: i32, const A1: i32, const A2: i32, const A3: i32, const A4: i32> Index<i32> for Ax5<A0, A1, A2, A3, A4> {
    type Output = i32;
    fn index(&self, index: i32) -> &Self::Output {
        match index {
            0 => &A0,
            1 => &A1,
            2 => &A2,
            3 => &A3,
            4 => &A4,
            -1 => &A4,
            -2 => &A3,
            -3 => &A2,
            -4 => &A1,
            -5 => &A0,
            _ => panic!("Index out of range, index is {}, but the length is 5", index),
        }
    }
}