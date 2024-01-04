extern crate alloc;
use alloc::boxed::Box;

pub struct Axes(pub(crate) Box<[usize]>);

impl Axes {
    /// Iterate over axes
    pub fn iter(&self) -> impl Iterator + '_ {
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
