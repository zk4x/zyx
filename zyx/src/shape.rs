pub trait IntoShape: Clone {
    fn into_shape(self) -> impl Iterator<Item = usize>;
}

impl<const N: usize> IntoShape for [usize; N] {
    fn into_shape(self) -> impl Iterator<Item = usize> {
        self.into_iter()
    }
}

pub trait IntoAxes: Clone {
    fn into_axes(self, rank: usize) -> impl Iterator<Item = usize>;
}

impl<const N: usize> IntoAxes for [isize; N] {
    fn into_axes(self, rank: usize) -> impl Iterator<Item=usize> {
        self.into_iter().map(move |a| (a + rank as isize) as usize % rank)
    }
}
