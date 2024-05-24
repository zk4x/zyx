use alloc::vec::Vec;
use core::fmt::Debug;
use core::ops::{Add, RangeInclusive};

pub trait IntoShape: Clone {
    fn into_shape(self) -> impl Iterator<Item = usize>;
    fn rank(&self) -> usize;
}

impl IntoShape for usize {
    fn into_shape(self) -> impl Iterator<Item=usize> {
        [self].into_iter()
    }

    fn rank(&self) -> usize {
        1
    }
}

impl<const N: usize> IntoShape for [usize; N] {
    fn into_shape(self) -> impl Iterator<Item = usize> {
        self.into_iter()
    }

    fn rank(&self) -> usize {
        N
    }
}

impl IntoShape for Vec<usize> {
    fn into_shape(self) -> impl Iterator<Item = usize> {
        self.into_iter()
    }

    fn rank(&self) -> usize {
        self.len()
    }
}

fn to_axis<T>(axis: T, rank: usize) -> usize
where
    usize: TryInto<T>,
    T: TryInto<usize>,
    T: Add<Output = T>,
    <usize as TryInto<T>>::Error: Debug,
    <T as TryInto<usize>>::Error: Debug,
{
    let t = axis + rank.try_into().unwrap();
    let t = <T as TryInto<usize>>::try_into(t).unwrap();
    let t = t % rank;
    return t
}

pub trait IntoAxes: Clone {
    fn into_axes(self, rank: usize) -> impl Iterator<Item = usize>;
}

impl<const N: usize> IntoAxes for [isize; N] {
    fn into_axes(self, rank: usize) -> impl Iterator<Item=usize> {
        self.into_iter().map(move |a| to_axis(a, rank))
    }
}

impl IntoAxes for RangeInclusive<isize> {
    fn into_axes(self, rank: usize) -> impl Iterator<Item=usize> {
        to_axis(*self.start(), rank)..to_axis(*self.end(), rank)
    }
}
