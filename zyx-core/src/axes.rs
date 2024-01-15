extern crate alloc;
use alloc::boxed::Box;

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct Axes(pub(crate) Box<[usize]>);

impl Axes {
    /// Iterate over axes
    pub fn iter(&self) -> impl Iterator<Item = &usize> + '_ {
        self.into_iter()
    }

    pub fn argsort(&self) -> Axes {
        let mut axes: Box<[(usize, usize)]> = self.0.iter().copied().enumerate().collect();
        axes.sort_by_key(|(_, v)| *v);
        Axes(axes.iter().map(|(k, _)| *k).collect())
    }
}

impl<'a> IntoIterator for &'a Axes {
    type IntoIter = <&'a [usize] as IntoIterator>::IntoIter;
    type Item = &'a usize;
    fn into_iter(self) -> Self::IntoIter {
        self.0.iter()
    }
}

/// # `IntoAxes`
/// Convert value into axes
/// () is all axes
#[allow(clippy::module_name_repetitions)]
pub trait IntoAxes: Clone {
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

impl IntoAxes for &[i64] {
    fn into_axes(self, ndim: usize) -> Axes {
        Axes(
            self.iter()
                .map(|x| (x + i64::try_from(ndim).unwrap()) as usize % ndim)
                .collect(),
        )
    }
}

impl<const N: usize> IntoAxes for [i64; N] {
    fn into_axes(self, ndim: usize) -> Axes {
        let axes: &[i64] = &self;
        axes.into_axes(ndim)
    }
}
