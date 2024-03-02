extern crate alloc;
use alloc::{boxed::Box, vec::Vec};
use core::fmt::{Display, Formatter};
use core::ops::Range;
use std::ops::RangeInclusive;

/// Axes used in reduce and permute operations.
/// Just Box<[usize]>.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct Axes(pub(crate) Box<[usize]>);

impl Axes {
    /// Number of axes
    pub fn len(&self) -> usize {
        self.0.len()
    }

    /// Is there no axes?
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    /// Iterate over axes
    pub fn iter(&self) -> impl Iterator<Item = &usize> + '_ {
        self.into_iter()
    }

    /// Returns the indices that sort axes in ascending order by value.
    pub fn argsort(&self) -> Axes {
        let mut axes: Box<[(usize, usize)]> = self.0.iter().copied().enumerate().collect();
        axes.sort_by_key(|(_, v)| *v);
        Axes(axes.iter().map(|(k, _)| *k).collect())
    }

    /// Does self contain axis a?
    pub fn contains(&self, a: usize) -> bool {
        self.0.contains(&a)
    }

    /// Get self as vector i64
    #[must_use]
    pub fn vi64(&self) -> Vec<i64> {
        self.0.iter().map(|x| *x as i64).collect()
    }

    /// Permute shape's dimensions with axes
    /// # Panics
    /// Panics if axes is incorrect.
    #[must_use]
    pub fn permute(&self, axes: &Axes) -> Self {
        debug_assert!(axes.iter().all(|a| *a < self.len()));
        Self(axes.into_iter().map(|axis| self.0[*axis]).collect())
    }
}

impl Display for Axes {
    fn fmt(&self, f: &mut Formatter<'_>) -> core::fmt::Result {
        f.write_fmt(format_args!("{self:?}"))
    }
}

impl From<Axes> for Vec<usize> {
    fn from(val: Axes) -> Self {
        val.0.into()
    }
}

impl<'a> IntoIterator for &'a Axes {
    type Item = &'a usize;
    type IntoIter = <&'a [usize] as IntoIterator>::IntoIter;
    fn into_iter(self) -> Self::IntoIter {
        self.0.iter()
    }
}

/// # `IntoAxes`
/// Convert value into axes
/// () is all axes
#[allow(clippy::module_name_repetitions)]
pub trait IntoAxes {
    /// Convert value into axes
    /// ```
    /// use zyx_core::axes::IntoAxes;
    /// let ax = [2, 3, 1];
    /// let axes = ax.into_axes(4);
    /// ```
    fn into_axes(self, rank: usize) -> Axes;
}

impl IntoAxes for Axes {
    fn into_axes(self, rank: usize) -> Axes {
        debug_assert!(self.iter().all(|a| *a < rank));
        self
    }
}

impl IntoAxes for &Axes {
    fn into_axes(self, rank: usize) -> Axes {
        debug_assert!(self.iter().all(|a| *a < rank));
        self.clone()
    }
}

impl IntoAxes for Vec<usize> {
    fn into_axes(self, rank: usize) -> Axes {
        debug_assert!(self.iter().all(|a| *a < rank));
        Axes(self.into_boxed_slice())
    }
}

impl IntoAxes for Box<[usize]> {
    fn into_axes(self, rank: usize) -> Axes {
        debug_assert!(self.iter().all(|a| *a < rank));
        Axes(self)
    }
}

impl IntoAxes for Range<usize> {
    fn into_axes(mut self, rank: usize) -> Axes {
        debug_assert!(self.all(|a| a < rank));
        Axes(self.collect())
    }
}

impl IntoAxes for Range<i64> {
    fn into_axes(mut self, rank: usize) -> Axes {
        debug_assert!(self.all(|a| if a > 0 { (a as usize) < rank } else { ((-a) as usize) <= rank }));
        Axes((((self.start + i64::try_from(rank).unwrap()) as usize % rank)..((self.end + i64::try_from(rank).unwrap()) as usize % rank)).collect())
    }
}

impl IntoAxes for RangeInclusive<i64> {
    fn into_axes(self, rank: usize) -> Axes {
        (&*(self.collect::<Box<[i64]>>())).into_axes(rank)
    }
}

impl IntoAxes for &RangeInclusive<i64> {
    fn into_axes(self, rank: usize) -> Axes {
        (&*(self.clone().collect::<Box<[i64]>>())).into_axes(rank)
    }
}

impl IntoAxes for () {
    fn into_axes(self, rank: usize) -> Axes {
        Axes((0..rank).collect())
    }
}

impl IntoAxes for &[i64] {
    fn into_axes(self, rank: usize) -> Axes {
        debug_assert!(self.iter().all(|a| if *a > 0 { (*a as usize) < rank } else { ((-*a) as usize) <= rank }));
        Axes(
            self.iter()
                .map(|x| (x + i64::try_from(rank).unwrap()) as usize % rank)
                .collect(),
        )
    }
}

impl IntoAxes for i64 {
    fn into_axes(self, rank: usize) -> Axes {
        debug_assert!(if self > 0 { (self as usize) < rank } else { ((-self) as usize) <= rank });
        [self].into_axes(rank)
    }
}

impl<const N: usize> IntoAxes for [i64; N] {
    fn into_axes(self, rank: usize) -> Axes {
        let axes: &[i64] = &self;
        axes.into_axes(rank)
    }
}

impl core::ops::Index<i64> for Axes {
    type Output = usize;
    fn index(&self, index: i64) -> &Self::Output {
        debug_assert!(if index > 0 { (index as usize) < self.len() } else { (-index) as usize <= self.len() });
        let rank = self.len();
        self.0.get((index + rank as i64) as usize % rank).unwrap()
    }
}
