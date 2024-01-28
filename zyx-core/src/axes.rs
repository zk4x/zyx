extern crate alloc;
use alloc::boxed::Box;

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
    /// let ax = (2, 3, 1);
    /// let axes = ax.into_axes(3);
    /// ```
    fn into_axes(self, rank: usize) -> Axes;
}

impl IntoAxes for Axes {
    fn into_axes(self, rank: usize) -> Axes {
        Axes(self.iter().copied().filter(|a| *a < rank).collect())
    }
}

impl IntoAxes for Box<[usize]> {
    fn into_axes(self, rank: usize) -> Axes {
        Axes(self.iter().copied().filter(|a| *a < rank).collect())
    }
}

impl IntoAxes for &[i64] {
    fn into_axes(self, rank: usize) -> Axes {
        Axes(
            self.iter()
                .map(|x| (x + i64::try_from(rank).unwrap()) as usize % rank)
                .collect(),
        )
    }
}

impl IntoAxes for i64 {
    fn into_axes(self, rank: usize) -> Axes {
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
        let rank = self.len();
        self.0.get((index + rank as i64) as usize % rank).unwrap()
    }
}
