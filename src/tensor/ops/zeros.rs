//use crate::{ops::Zeros, tensor::{Variable, Gradient}, shape::Shape};

// /// Initialize tensor filled with zeros
// TODO determine whether this is usefull at all
/*impl<S, Sh> Zeros for Variable<S>
where
    S: Zeros<Sh = Sh>,
    Sh: Shape<D = usize>,
{
    type Sh = Sh;

    fn zeros(shape: Self::Sh) -> Self {
        Self {
            data: S::zeros(shape),
            grad: Gradient::new(),
        }
    }
}*/
