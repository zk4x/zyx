// use crate::{ops::Ones, tensor::{Variable, Gradient}, shape::Shape};

// /// Initialize tensor filled with zeros
// TODO determine whether this is usefull at all
/*impl<S, Sh> Ones for Variable<S>
where
    S: Ones<Sh = Sh>,
    Sh: Shape<D = usize>,
{
    type Sh = Sh;

    fn ones(shape: Self::Sh) -> Self {
        Self {
            data: S::ones(shape),
            grad: Gradient::new(),
        }
    }
}*/
