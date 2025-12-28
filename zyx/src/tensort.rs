use std::marker::PhantomData;

use crate::{IntoShape, RT, Scalar, ZyxError, graph::UOp, shape::Dim, tensor::TensorId};

trait Shape {
    fn shape() -> impl IntoShape;
}

struct S1<const D0: Dim> {}

impl<const D0: Dim> Shape for S1<D0> {
    fn shape() -> impl IntoShape {
        D0
    }
}

struct S2<const D0: Dim, const D1: Dim> {}

impl<const D0: Dim, const D1: Dim> Shape for S2<D0, D1> {
    fn shape() -> impl IntoShape {
        [D0, D1]
    }
}

struct S4<const D0: Dim, const D1: Dim, const D3: Dim, const D4: Dim> {}

impl<const D0: Dim, const D1: Dim, const D2: Dim, const D3: Dim> Shape for S4<D0, D1, D2, D3> {
    fn shape() -> impl IntoShape {
        [D0, D1, D2, D3]
    }
}

pub struct Tensor<T: Scalar, S: Shape> {
    pub(super) id: TensorId,
    _dtype: PhantomData<T>,
    _shape: PhantomData<S>,
}

impl<T: Scalar, S: Shape> Clone for Tensor<T, S> {
    fn clone(&self) -> Self {
        RT.lock().retain(self.id);
        Tensor { id: self.id, _dtype: Default::default(), _shape: Default::default() }
    }
}

impl<T: Scalar, S: Shape> Drop for Tensor<T, S> {
    fn drop(&mut self) {
        //std::println!("dropping");
        if let Some(mut rt) = RT.try_lock() {
            rt.release(self.id);
        } else {
            println!("Warning: Unable to drop Tensor due to runtime mutex lock.");
        }
    }
}

impl<T: Scalar, S: Shape> Tensor<T, S> {
    fn rand() -> Result<Tensor<T, S>, ZyxError> {
        let t = crate::tensor::Tensor::rand(S::shape(), T::dtype())?;
        Ok(Tensor { id: t.id(), _dtype: Default::default(), _shape: Default::default() })
    }

    fn exp2(&self) -> Tensor<T, S> {
        let id = RT.lock().unary(self.id, UOp::Exp2);
        Tensor { id, _dtype: Default::default(), _shape: Default::default() }
    }
}

impl<T: Scalar, const D0: Dim, const D1: Dim> Tensor<T, S2<D0, D1>> {
    pub fn matmul<const D2: Dim>(
        &self,
        rhs: impl Into<Tensor<T, S2<D1, D2>>>,
    ) -> Result<Tensor<T, S2<D0, D2>>, ZyxError> {
        todo!()
    }
}

pub trait ConvDims {
    const C_IN: usize;
    const GROUPS: usize;
    const C_IN_PER_GROUP: usize = Self::C_IN / Self::GROUPS;
}

pub struct ConvShape<const C_IN: usize, const GROUPS: usize>;

impl<const C_IN: usize, const GROUPS: usize> ConvDims for ConvShape<C_IN, GROUPS> {
    const C_IN: usize = C_IN;
    const GROUPS: usize = GROUPS;
}

pub trait ConvOut<
    const H_IN: usize,
    const W_IN: usize,
    const K_H: usize,
    const K_W: usize,
    const PAD_H: usize,
    const PAD_W: usize,
    const STRIDE_H: usize,
    const STRIDE_W: usize,
    const DIL_H: usize,
    const DIL_W: usize,
>
{
    const H_OUT: usize = (H_IN + 2 * PAD_H - DIL_H * (K_H - 1) - 1) / STRIDE_H + 1;
    const W_OUT: usize = (W_IN + 2 * PAD_W - DIL_W * (K_W - 1) - 1) / STRIDE_W + 1;
}

impl<
    const H_IN: usize,
    const W_IN: usize,
    const K_H: usize,
    const K_W: usize,
    const PAD_H: usize,
    const PAD_W: usize,
    const STRIDE_H: usize,
    const STRIDE_W: usize,
    const DIL_H: usize,
    const DIL_W: usize,
> ConvOut<H_IN, W_IN, K_H, K_W, PAD_H, PAD_W, STRIDE_H, STRIDE_W, DIL_H, DIL_W> for ()
{
}

// THIS IS WHY WE CAN'T HAVE FULLY COMPTIME VERIFIED TENSOR SHAPES
pub fn conv<
    T: Scalar,
    const N: usize,
    const C_IN: usize,
    const C_OUT: usize,
    const H_IN: usize,
    const W_IN: usize,
    const K_H: usize,
    const K_W: usize,
    const PAD_H: usize,
    const PAD_W: usize,
    const STRIDE_H: usize,
    const STRIDE_W: usize,
    const DIL_H: usize,
    const DIL_W: usize,
>(
    input: &Tensor<T, S4<N, C_IN, H_IN, W_IN>>,
    weight: &Tensor<T, S4<C_OUT, C_IN, K_H, K_W>>,
) -> Tensor<
    T,
    S4<
        N,
        C_OUT,
        { <() as ConvOut<H_IN, W_IN, K_H, K_W, PAD_H, PAD_W, STRIDE_H, STRIDE_W, DIL_H, DIL_W>>::H_OUT },
        { <() as ConvOut<H_IN, W_IN, K_H, K_W, PAD_H, PAD_W, STRIDE_H, STRIDE_W, DIL_H, DIL_W>>::W_OUT },
    >,
> {
    // impl
}

#[test]
fn t0() {
    let x = Tensor::<f32, S2<1024, 512>>::rand().unwrap();
    let z = x.exp2();
}
