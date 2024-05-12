
pub trait IntoShape {}

impl<const N: usize> IntoShape for [usize; N] {}

pub trait IntoAxes {}

impl<const N: usize> IntoShape for [isize; N] {}
