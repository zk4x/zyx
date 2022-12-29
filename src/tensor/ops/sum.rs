use core::marker::PhantomData;

use crate::{ops::{Summable, Expandable, HasShape}, tensor::{Variable, Tensor, Backward, GradientRef, GradAcc}, shape::{Shape, Axes, ReducableBy}};

#[derive(Debug, Clone)]
pub struct SummableBackwardV<'g, G, Sh, Ax> {
    grad: GradientRef<'g, G>,
    shape: PhantomData<Sh>,
    axes: PhantomData<Ax>,
}

impl<S, G, Sh, Ax> Backward<S> for SummableBackwardV<'_, G, Sh, Ax>
where
    Sh: Shape,
    Ax: Axes,
    S: Expandable<Sh, Ax>,
    Sh: ReducableBy<Ax, Output = <S as HasShape>::Sh>,
    G: GradAcc<<S as Expandable<Sh, Ax>>::Output>,
{
    fn backward(self, res_grad: S) {
        self.grad.accumulate(res_grad._expand());
    }
}

impl<'g, S, Dims> Summable<Dims> for &'g Variable<S>
where
    S: Clone + Summable<Dims> + HasShape,
    Dims: Axes,
{
    type Output = Tensor<<S as Summable<Dims>>::Output, SummableBackwardV<'g, S, <S as HasShape>::Sh, Dims>>;
    fn _sum(self) -> Self::Output {
        Tensor {
            data: (*self.data()).clone()._sum(),
            grad_fn: SummableBackwardV {
                grad: GradientRef::new(&self.grad),
                shape: PhantomData,
                axes: PhantomData,
            }
        }
    }
}

#[derive(Debug, Clone)]
pub struct SummableBackwardT<F, Sh, Ax> {
    grad_fn: F,
    shape: PhantomData<Sh>,
    axes: PhantomData<Ax>,
}

impl<S, F, Sh, Ax> Backward<S> for SummableBackwardT<F, Sh, Ax>
where
    Sh: Shape,
    Ax: Axes,
    S: Expandable<Sh, Ax>,
    Sh: ReducableBy<Ax, Output = <S as HasShape>::Sh>,
    F: Backward<<S as Expandable<Sh, Ax>>::Output>,
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
    type Output = Tensor<<S as Summable<Dims>>::Output, SummableBackwardT<F, <S as HasShape>::Sh, Dims>>;
    fn _sum(self) -> Self::Output {
        Tensor {
            data: self.data._sum(),
            grad_fn: SummableBackwardT {
                grad_fn: self.grad_fn,
                shape: PhantomData,
                axes: PhantomData,
            }
        }
    }
}

#[test]
fn sum() {
    // TODO finish all variations
    use crate::prelude::*;
    use crate::shape::Ax2;
    use crate::device::cpu;

    let device = cpu::Device::default();

    let x = device.buffer([[[[2, 3, 1], [3, 4, 5]], [[5, 6, 7], [7, 8, 9]]], [[[3, 8, 9], [4, 5, 3]], [[3, 2, 1], [6, 5, 3]]]]);
    let y = x.sum::<Ax2<0, 2>>().reshape();
    assert_eq!(y, [[12, 20, 18], [21, 21, 20]]);

    //panic!();

    let x = device.buffer([[[3], [1], [2]], [[4], [1], [0]], [[4], [3], [5]]]).with_grad();
    let y = (&x).sum::<Ax2<1, 2>>().reshape();

    assert_eq!(y.data().clone(), [6, 5, 12]);

    y.backward();
}
