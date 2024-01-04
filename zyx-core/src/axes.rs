extern crate alloc;
use alloc::boxed::Box;

pub struct Axes(Box<[usize]>);

impl Axes {
    fn iter(&self) -> impl Iterator + '_ {
        self.into_iter()
    }
}

impl<'a> IntoIterator for &'a Axes {
    type IntoIter = <&'a [usize] as IntoIterator>::IntoIter;
    type Item = &'a usize;
    fn into_iter(self) -> Self::IntoIter {
        self.0.iter()
    }
}
