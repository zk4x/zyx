//! Structs that implement trait [Module](crate::module::Module) for anything that it makes sense to implement this trait.
//! These include zyx::ops, as well as layers, such as [Linear].
//!
//! This module is expected to get most stuff added.
//! It will contain functors, layers, models, cells, simply anything that can have .forward(input) function.
//!

use crate::{module::Module, ops::{self, GetShape, Pow, MatMul, Zeros, Ones}, tensor::{IntoVariable, Variable}, init::UniformInit, shape::IntoDims};
use std::ops::{Neg, Add, Sub, Div, Mul};

/// ReLU operation
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ReLU;

impl<Input> Module<'_, Input> for ReLU
where
    Input: ops::ReLU,
{
    type Output = Input::Output;
    type Params = ();

    fn forward(&self, x: Input) -> Self::Output {
        x.relu()
    }

    fn parameters(&mut self) -> Self::Params {}
}

/// Exp operation
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Exp;

impl<Input> Module<'_, Input> for Exp
where
    Input: ops::Exp,
{
    type Output = Input::Output;
    type Params = ();

    fn forward(&self, x: Input) -> Self::Output {
        x.exp()
    }

    fn parameters(&mut self) -> Self::Params {}
}

/// Ln operation
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Ln;

impl<Input> Module<'_, Input> for Ln
where
    Input: ops::Ln,
{
    type Output = Input::Output;
    type Params = ();

    fn forward(&self, x: Input) -> Self::Output {
        x.ln()
    }

    fn parameters(&mut self) -> Self::Params {}
}

/// Tanh operation
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Tanh;

impl<Input> Module<'_, Input> for Tanh
where
    Input: ops::Tanh,
{
    type Output = Input::Output;
    type Params = ();

    fn forward(&self, x: Input) -> Self::Output {
        x.tanh()
    }

    fn parameters(&mut self) -> Self::Params {}
}

/// Sigmoid operation
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Sigmoid;

impl<Input> Module<'_, Input> for Sigmoid
where
    Input: Neg,
    <Input as Neg>::Output: ops::Exp,
    i32: Add<<<Input as Neg>::Output as ops::Exp>::Output>,
    i32: Div<<i32 as Add<<<Input as Neg>::Output as ops::Exp>::Output>>::Output>,
{
    type Output = <i32 as Div<<i32 as Add<<<Input as Neg>::Output as ops::Exp>::Output>>::Output>>::Output;
    type Params = ();

    fn forward(&self, x: Input) -> Self::Output {
        use ops::Exp;
        1/(1+(-x).exp())
    }

    fn parameters(&mut self) -> Self::Params {}
}

/// Softmax operation
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct SoftMax<D>
where
    D: IntoDims,
{
    /// [Dimensions](crate::shape::IntoDims) to calculate softmax across
    pub dims: D
}

impl<Input, D> Module<'_, Input> for SoftMax<D>
where
    D: IntoDims + Clone,
    Input: Clone + ops::Max + Sub<<Input as ops::Max>::Output>,
    <Input as Sub<<Input as ops::Max>::Output>>::Output: ops::Exp,
    <<Input as Sub<<Input as ops::Max>::Output>>::Output as ops::Exp>::Output: Clone + ops::Sum,
    <<Input as Sub<<Input as ops::Max>::Output>>::Output as ops::Exp>::Output: Div<<<<Input as Sub<<Input as ops::Max>::Output>>::Output as ops::Exp>::Output as ops::Sum>::Output>,
{
    type Output = <<<Input as Sub<<Input as ops::Max>::Output>>::Output as ops::Exp>::Output as Div<<<<Input as Sub<<Input as ops::Max>::Output>>::Output as ops::Exp>::Output as ops::Sum>::Output>>::Output;
    type Params = ();

    fn forward(&self, x: Input) -> Self::Output {
        use crate::ops::{Exp, Sum};
        // TODO: Check if cloning Tensors (that is also cloning their grad_fn)
        // has any effect on correctness of this function's gradient calculation
        let temp = (x.clone() - x.max(())).exp();
        temp.clone() / temp.sum(self.dims.clone())
    }

    fn parameters(&mut self) -> Self::Params {}
}

#[test]
fn softmax_test() {
    use crate::prelude::*;
    use crate::accel::cpu::Buffer;

    let x = Buffer::<f32>::cfrom([[3., 2., 4.], [4., 2., 5.]]).with_grad();

    /*let dim = -1;
    let e_x = ((&x).data().clone() - (&x).max(())).exp();
    println!("\n{}", e_x);
    let y = e_x.clone() / e_x.sum(dim);
    println!("\n{}", y);
    y.backward();
    println!("\n{}", x.grad());*/

    let sm = SoftMax { dims: -1 };
    let y = sm.forward(&x);
    println!("\n{}", y);
    y.backward();
    println!("\n{}", x);
    //panic!();
}

/// Sum operation
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Sum<D>
where
    D: IntoDims,
{
    /// [Dimensions](crate::shape::IntoDims) to sum
    pub dims: D,
}

impl<Input, D> Module<'_, Input> for Sum<D>
where
    Input: ops::Sum,
    D: IntoDims + Clone,
{
    type Output = Input::Output;
    type Params = ();

    fn forward(&self, x: Input) -> Self::Output {
        x.sum(self.dims.clone())
    }
    
    fn parameters(&mut self) -> Self::Params {}
}

/// Max operation
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Max<D>
where
    D: IntoDims,
{
    /// [Dimensions](crate::shape::IntoDims) to max
    pub dims: D,
}

impl<Input, D> Module<'_, Input> for Max<D>
where
    Input: ops::Max,
    D: IntoDims + Clone,
{
    type Output = Input::Output;
    type Params = ();

    fn forward(&self, x: Input) -> Self::Output {
        x.max(self.dims.clone())
    }
    
    fn parameters(&mut self) -> Self::Params {}
}

/// Min operation
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Min<D> {
    /// [Dimensions](crate::shape::IntoDims) to min
    pub dims: D,
}

impl<Input, D> Module<'_, Input> for Min<D>
where
    Input: ops::Min,
    D: IntoDims + Clone,
{
    type Output = Input::Output;
    type Params = ();

    fn forward(&self, x: Input) -> Self::Output {
        x.min(self.dims.clone())
    }

    fn parameters(&mut self) -> Self::Params {}
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
/// Mean operation
pub struct Mean<D>
where
    D: IntoDims,
{
    /// [Dimensions](crate::shape::IntoDims) for mean
    pub dims: D
}

impl<Input, D> Module<'_, Input> for Mean<D>
where
    D: IntoDims + Clone,
    Input: GetShape + ops::Sum,
    <Input as ops::Sum>::Output: Div<usize>,
{
    type Output = <<Input as ops::Sum>::Output as Div<usize>>::Output;
    type Params = ();

    fn forward(&self, x: Input) -> Self::Output {
        let n = x.shape().numel();
        x.sum(self.dims.clone())/n
    }

    fn parameters(&mut self) -> Self::Params {}
}

/// MSE loss
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct MSELoss;

impl<Y, YP> Module<'_, (Y, YP)> for MSELoss
where
    Y: Sub<YP>,
    <Y as Sub<YP>>::Output: Pow<i32>,
{
    type Output = <<Y as Sub<YP>>::Output as Pow<i32>>::Output;
    type Params = ();

    fn forward(&self, x: (Y, YP)) -> Self::Output {
        (x.0 - x.1).pow(2)
    }

    fn parameters(&mut self) -> Self::Params {}
}

//pub struct STD {}

/*#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct NormLayer {}

impl<Input> Module<Input> for &NormLayer {
    type Output = Self;
    fn forward(self, x: Input) -> Self::Output {
        (x - x.apply(&Mean))/x.apply(&STD)
    }
}*/

/// Linear layer
#[derive(Debug, Clone)]
pub struct Linear<W, WG, B, BG> {
    w: Variable<W, WG>,
    b: Variable<B, BG>,
}

impl<W, WG, B, BG> Linear<W, WG, B, BG> {
    /// Create new [Linear layer](Linear) with given in_features and out_features dimensions
    pub fn new<T>(in_features: usize, out_features: usize) -> Self
    where
        T: Zeros + Ones,
        W: UniformInit<T>,
        B: UniformInit<T>,
    {
        Self {
            w: W::uniform([in_features, out_features], T::zeros(()), T::ones(())).with_grad(),
            b: B::uniform([1, out_features], T::zeros(()), T::ones(())).with_grad(),
        }
    }
}

impl<'p, W, WG, B, BG, Input> Module<'p, Input> for Linear<W, WG, B, BG>
where
    W: Clone + Sub<<WG as Mul<f64>>::Output, Output = W>,
    WG: Clone + Mul<f64>,
    B: Clone + Sub<<BG as Mul<f64>>::Output, Output = B>,
    BG: Clone + Mul<f64>,

    W: 'p,
    WG: 'p,
    B: 'p,
    BG: 'p,

    Input: MatMul<&'p Variable<W, WG>>,
    <Input as MatMul<&'p Variable<W, WG>>>::Output: Add<&'p Variable<B, BG>>,
{
    type Output = <<Input as MatMul<&'p Variable<W, WG>>>::Output as Add<&'p Variable<B, BG>>>::Output;
    type Params = (&'p mut Variable<W, WG>, &'p mut Variable<B, BG>);

    fn forward(&'p self, x: Input) -> Self::Output {
        x.matmul(&self.w) + &self.b
    }

    fn parameters(&'p mut self) -> Self::Params {
        (&mut self.w, &mut self.b)
    }
}

/// RNNCell
// TODO: Should we rewrite this as two linear layers?
#[derive(Debug, Clone)]
pub struct RNNCell<WI, WIG, BI, BIG, WH, WHG, BH, BHG> {
    wih: Variable<WI, WIG>,
    bih: Variable<BI, BIG>,
    whh: Variable<WH, WHG>,
    bhh: Variable<BH, BHG>,
}

impl<WI, WIG, BI, BIG, WH, WHG, BH, BHG> RNNCell<WI, WIG, BI, BIG, WH, WHG, BH, BHG> {
    /// Create new [RNNCell] with given input_size and hidden_size dimensions
    pub fn new<T>(input_size: usize, hidden_size: usize) -> Self
    where
        T: Zeros + Ones,
        WI: UniformInit<T>,
        BI: UniformInit<T>,
        WH: UniformInit<T>,
        BH: UniformInit<T>,
    {
        Self {
            wih: WI::uniform([input_size, hidden_size], T::zeros(()), T::ones(())).with_grad(),
            bih: BI::uniform([1, hidden_size], T::zeros(()), T::ones(())).with_grad(),
            whh: WH::uniform([hidden_size, hidden_size], T::zeros(()), T::ones(())).with_grad(),
            bhh: BH::uniform([1, hidden_size], T::zeros(()), T::ones(())).with_grad(),
        }
    }
}

impl<'a, WI, WIG, BI, BIG, WH, WHG, BH, BHG, I, H> Module<'a, (I, H)> for RNNCell<WI, WIG, BI, BIG, WH, WHG, BH, BHG>
where
    WI: 'a + Clone + Sub<<WIG as Mul<f64>>::Output, Output = WI>,
    WIG: 'a + Clone + Mul<f64>,
    BI: 'a + Clone + Sub<<BIG as Mul<f64>>::Output, Output = BI>,
    BIG: 'a + Clone + Mul<f64>,
    WH: 'a + Clone + Sub<<WHG as Mul<f64>>::Output, Output = WH>,
    WHG: 'a + Clone + Mul<f64>,
    BH: 'a + Clone + Sub<<BHG as Mul<f64>>::Output, Output = BH>,
    BHG: 'a + Clone + Mul<f64>,

    I: MatMul<&'a Variable<WI, WIG>>,
    <I as MatMul<&'a Variable<WI, WIG>>>::Output: Add<&'a Variable<BI, BIG>>,
    H: MatMul<&'a Variable<WH, WHG>>,
    <H as MatMul<&'a Variable<WH, WHG>>>::Output: Add<&'a Variable<BH, BHG>>,
    <<I as MatMul<&'a Variable<WI, WIG>>>::Output as Add<&'a Variable<BI, BIG>>>::Output: Add<<<H as MatMul<&'a Variable<WH, WHG>>>::Output as Add<&'a Variable<BH, BHG>>>::Output>,
{
    type Output = <<<I as MatMul<&'a Variable<WI, WIG>>>::Output as Add<&'a Variable<BI, BIG>>>::Output as Add<<<H as MatMul<&'a Variable<WH, WHG>>>::Output as Add<&'a Variable<BH, BHG>>>::Output>>::Output;
    type Params = (&'a mut Variable<WI, WIG>, &'a mut Variable<BI, BIG>, &'a mut Variable<WH, WHG>, &'a mut Variable<BH, BHG>);

    fn forward(&'a self, x: (I, H)) -> Self::Output {
        (x.0.matmul(&self.wih) + &self.bih) + (x.1.matmul(&self.whh) + &self.bhh)
    }

    fn parameters(&'a mut self) -> Self::Params {
        (&mut self.wih, &mut self.bih, &mut self.whh, &mut self.bhh)
    }
}
