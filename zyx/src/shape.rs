use alloc::vec::Vec;
use core::fmt::Debug;
use core::ops::{Add, Range, RangeInclusive};

pub trait IntoShape: Clone + Debug {
    fn into_shape(self) -> impl Iterator<Item = usize>;
    fn rank(&self) -> usize;
}

impl IntoShape for usize {
    fn into_shape(self) -> impl Iterator<Item = usize> {
        return [self].into_iter();
    }

    fn rank(&self) -> usize {
        return 1;
    }
}

impl<const N: usize> IntoShape for [usize; N] {
    fn into_shape(self) -> impl Iterator<Item = usize> {
        return self.into_iter();
    }

    fn rank(&self) -> usize {
        return N;
    }
}

impl IntoShape for &[usize] {
    fn into_shape(self) -> impl Iterator<Item = usize> {
        return self.into_iter().copied();
    }

    fn rank(&self) -> usize {
        return self.len();
    }
}

impl IntoShape for Vec<usize> {
    fn into_shape(self) -> impl Iterator<Item = usize> {
        return self.into_iter();
    }

    fn rank(&self) -> usize {
        return self.len();
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
    return t;
}

pub trait IntoAxes: Clone {
    fn into_axes(self, rank: usize) -> impl Iterator<Item = usize>;
    fn len(&self) -> usize;
}

impl IntoAxes for isize {
    fn into_axes(self, rank: usize) -> impl Iterator<Item = usize> {
        return [to_axis(self, rank)].into_iter();
    }

    fn len(&self) -> usize {
        1
    }
}

impl<const N: usize> IntoAxes for [isize; N] {
    fn into_axes(self, rank: usize) -> impl Iterator<Item = usize> {
        return self.into_iter().map(move |a| to_axis(a, rank));
    }

    fn len(&self) -> usize {
        N
    }
}

impl IntoAxes for Range<isize> {
    fn into_axes(self, rank: usize) -> impl Iterator<Item = usize> {
        return to_axis(self.start, rank)..to_axis(self.end, rank);
    }

    fn len(&self) -> usize {
        (self.end - self.start) as usize
    }
}

impl IntoAxes for RangeInclusive<isize> {
    fn into_axes(self, rank: usize) -> impl Iterator<Item = usize> {
        return to_axis(*self.start(), rank)..to_axis(*self.end(), rank);
    }

    fn len(&self) -> usize {
        (self.end() - self.start() + 1) as usize
    }
}
