//! # Axes

extern crate alloc;
use alloc::boxed::Box;

/// # Axes
///
/// Axes is passed into some operations on tensors, such as permute.
/// ```
/// use zyx::context::Context;
/// let mut ctx = Context::new();
/// let x = ctx.tensor([[2, 3, 4], [5, 6, 1]]);
/// let mut y = x.permute((-1, -2));
/// y.realize()?;
/// assert_eq!(y, [[2, 5], [3, 6], [4, 1]]);
/// # Ok::<(), zyx::OutOfMemoryError>(())
/// ```
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Axes(pub(crate) Box<[usize]>);

impl Axes {
    #[cfg(feature = "opencl")]
    pub(crate) fn len(&self) -> usize {
        self.0.len()
    }

    /// Check if axes contains axis.
    pub(crate) fn contains(&self, axis: usize) -> bool {
        self.0.contains(&axis)
    }

    pub(crate) fn argsort(&self) -> Axes {
        let mut axes: Box<[(usize, usize)]> = self.0.iter().copied().enumerate().collect();
        axes.sort_by_key(|(_, v)| *v);
        Axes(axes.iter().map(|(k, _)| *k).collect())
    }
}

impl core::fmt::Display for Axes {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let res = alloc::format!("{:?}", self.0)
            .replace('[', "(")
            .replace(']', ")");
        f.write_str(&res)
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
#[allow(clippy::module_name_repetitions)]
pub trait IntoAxes: Clone {
    /// Convert value into axes
    /// ```
    /// use zyx::axes::IntoAxes;
    /// let ax = (2, 3, 1);
    /// let axes = ax.into_axes(3);
    /// ```
    fn into_axes(self, rank: usize) -> Axes;
}

impl IntoAxes for &[i32] {
    fn into_axes(self, ndim: usize) -> Axes {
        Axes(
            self.iter()
                .map(|x| (x + ndim as i32) as usize % ndim)
                .collect(),
        )
    }
}

impl<const N: usize> IntoAxes for [i32; N] {
    fn into_axes(self, ndim: usize) -> Axes {
        Axes(
            self.iter()
                .map(|x| (x + ndim as i32) as usize % ndim)
                .collect(),
        )
    }
}

impl IntoAxes for () {
    fn into_axes(self, ndim: usize) -> Axes {
        Axes((0..ndim).collect())
    }
}

impl IntoAxes for i32 {
    fn into_axes(self, ndim: usize) -> Axes {
        Axes(Box::new([(self + ndim as i32) as usize % ndim]))
    }
}

impl IntoAxes for (i32, i32) {
    fn into_axes(self, ndim: usize) -> Axes {
        Axes(Box::new([
            (self.0 + ndim as i32) as usize % ndim,
            (self.1 + ndim as i32) as usize % ndim,
        ]))
    }
}

impl IntoAxes for (i32, i32, i32) {
    fn into_axes(self, ndim: usize) -> Axes {
        Axes(Box::new([
            (self.0 + ndim as i32) as usize % ndim,
            (self.1 + ndim as i32) as usize % ndim,
            (self.2 + ndim as i32) as usize % ndim,
        ]))
    }
}

impl IntoAxes for (i32, i32, i32, i32) {
    fn into_axes(self, ndim: usize) -> Axes {
        Axes(Box::new([
            (self.0 + ndim as i32) as usize % ndim,
            (self.1 + ndim as i32) as usize % ndim,
            (self.2 + ndim as i32) as usize % ndim,
            (self.3 + ndim as i32) as usize % ndim,
        ]))
    }
}

impl IntoAxes for (i32, i32, i32, i32, i32) {
    fn into_axes(self, ndim: usize) -> Axes {
        Axes(Box::new([
            (self.0 + ndim as i32) as usize % ndim,
            (self.1 + ndim as i32) as usize % ndim,
            (self.2 + ndim as i32) as usize % ndim,
            (self.3 + ndim as i32) as usize % ndim,
            (self.4 + ndim as i32) as usize % ndim,
        ]))
    }
}

impl IntoAxes for (i32, i32, i32, i32, i32, i32) {
    fn into_axes(self, ndim: usize) -> Axes {
        Axes(Box::new([
            (self.0 + ndim as i32) as usize % ndim,
            (self.1 + ndim as i32) as usize % ndim,
            (self.2 + ndim as i32) as usize % ndim,
            (self.3 + ndim as i32) as usize % ndim,
            (self.4 + ndim as i32) as usize % ndim,
            (self.5 + ndim as i32) as usize % ndim,
        ]))
    }
}

impl IntoAxes for (i32, i32, i32, i32, i32, i32, i32) {
    fn into_axes(self, ndim: usize) -> Axes {
        Axes(Box::new([
            (self.0 + ndim as i32) as usize % ndim,
            (self.1 + ndim as i32) as usize % ndim,
            (self.2 + ndim as i32) as usize % ndim,
            (self.3 + ndim as i32) as usize % ndim,
            (self.4 + ndim as i32) as usize % ndim,
            (self.5 + ndim as i32) as usize % ndim,
            (self.6 + ndim as i32) as usize % ndim,
        ]))
    }
}

impl IntoAxes for (i32, i32, i32, i32, i32, i32, i32, i32) {
    fn into_axes(self, ndim: usize) -> Axes {
        Axes(Box::new([
            (self.0 + ndim as i32) as usize % ndim,
            (self.1 + ndim as i32) as usize % ndim,
            (self.2 + ndim as i32) as usize % ndim,
            (self.3 + ndim as i32) as usize % ndim,
            (self.4 + ndim as i32) as usize % ndim,
            (self.5 + ndim as i32) as usize % ndim,
            (self.6 + ndim as i32) as usize % ndim,
            (self.7 + ndim as i32) as usize % ndim,
        ]))
    }
}