use core::marker::PhantomData;

use crate::{ops::{Summable, Expandable, HasShape}, tensor::{Variable, Tensor, Backward, GradientRef, GradAcc}, shape::{Shape, Axes}};

#[derive(Debug, Clone)]
pub struct SummableBackwardV<'g, G> {
    grad: GradientRef<'g, G>,
}

impl<S, G> Backward<S> for SummableBackwardV<'_, G>
where
    S: Expandable<G::Sh>,
    G: HasShape + GradAcc<<S as Expandable<G::Sh>>::Output>,
{
    fn backward(self, res_grad: S) {
        self.grad.accumulate(res_grad._expand());
    }
}

impl<'g, S, Dims> Summable<Dims> for &'g Variable<S>
where
    S: Clone + Summable<Dims>,
    Dims: Axes,
{
    type Output = Tensor<<S as Summable<Dims>>::Output, SummableBackwardV<'g, S>>;
    fn _sum(self) -> Self::Output {
        Tensor {
            data: (*self.data()).clone()._sum(),
            grad_fn: SummableBackwardV {
                grad: GradientRef::new(&self.grad),
            }
        }
    }
}

#[derive(Debug, Clone)]
pub struct SummableBackwardT<F, Sh> {
    grad_fn: F,
    shape: PhantomData<Sh>,
}

impl<S, F, Sh> Backward<S> for SummableBackwardT<F, Sh>
where
    Sh: Shape,
    S: Expandable<Sh>,
    F: Backward<<S as Expandable<Sh>>::Output>,
{
    fn backward(self, res_grad: S) {
        self.grad_fn.backward(res_grad._expand());
    }
}

impl<S, F, Dims> Summable<Dims> for Tensor<S, F>
where
    S: Clone + Summable<Dims> + HasShape,
    Dims: Axes,
{
    type Output = Tensor<<S as Summable<Dims>>::Output, SummableBackwardT<F, <S as HasShape>::Sh>>;
    fn _sum(self) -> Self::Output {
        //let shape = self.data.shape();
        Tensor {
            data: self.data._sum(),
            grad_fn: SummableBackwardT {
                grad_fn: self.grad_fn,
                shape: PhantomData,
            }
        }
    }
}

#[test]
fn sum() {
    // TODO finish all variations
    use crate::prelude::*;
    use crate::shape::{Sh3, Ax2};
    use crate::accel::cpu::Buffer;

    extern crate alloc;

    let vec = alloc::vec![3, 1, 2, 4, 1, 0, 4, 3, 5];
    let x = Buffer::<_, Sh3<3, 3, 1>>::from_slice(&vec);
    let y = x.sum::<Ax2<0, 1>>();

    let x = Buffer::<_, Sh3<3, 3, 1>>::from_slice(&vec).with_grad();
    //let x = Buffer::<_, Sh5<1, 3, 1, 3, 1>>::from_slice(&vec).with_grad();
    let y = (&x).sum::<Ax2<0, 1>>();

    assert_eq!([6, 5, 12].to_vec(), y.to_vec());
}
