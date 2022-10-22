//! Structs that implement trait Module for anything that it makes sense to implement this trait.
//! These include zyx::ops, as well as layers, such as Linear.
//!

use crate::{module::Module, ops::{self, ConvertFrom, GetShape, Pow}, tensor::Variable, init::UniformInit, ops::Zeros, shape::IntoDims};
use std::ops::{Neg, Add, Sub, Mul, Div};

/// ReLU operation
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct ReLU;

impl<Input> Module<Input> for &ReLU
where
    Input: ops::ReLU,
{
    type Output = Input::Output;
    type Params = ();

    fn forward(self, x: Input) -> Self::Output {
        x.relu()
    }

    fn parameters(self) -> Self::Params {
        ()
    }
}

/// Exp operation
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct Exp;

impl<Input> Module<Input> for &Exp
where
    Input: ops::Exp,
{
    type Output = Input::Output;
    type Params = ();

    fn forward(self, x: Input) -> Self::Output {
        x.exp()
    }

    fn parameters(self) -> Self::Params {
        ()
    }
}

/// Ln operation
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct Ln;

impl<Input> Module<Input> for &Ln
where
    Input: ops::Ln,
{
    type Output = Input::Output;
    type Params = ();

    fn forward(self, x: Input) -> Self::Output {
        x.ln()
    }

    fn parameters(self) -> Self::Params {
        ()
    }
}

/// Tanh operation
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct Tanh;

impl<Input> Module<Input> for &Tanh
where
    Input: ops::Tanh,
{
    type Output = Input::Output;
    type Params = ();

    fn forward(self, x: Input) -> Self::Output {
        x.tanh()
    }

    fn parameters(self) -> Self::Params {
        ()
    }
}

/// Sigmoid operation
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct Sigmoid;

impl<Input> Module<Input> for &Sigmoid
where
    Input:
        Clone +
        Neg +
        Add<<<Input as Neg>::Output as ops::Exp>::Output> +
        Div<<Input as Add<<<Input as Neg>::Output as ops::Exp>::Output>>::Output>,
    <Input as Neg>::Output: ops::Exp,
    i32: Add<<<Input as Neg>::Output as ops::Exp>::Output> + Div<<i32 as Add<<<Input as Neg>::Output as ops::Exp>::Output>>::Output>,
{
    type Output = <i32 as Div<<i32 as Add<<<Input as Neg>::Output as ops::Exp>::Output>>::Output>>::Output;
    type Params = ();

    fn forward(self, x: Input) -> Self::Output {
        use ops::Exp;
        1/(1+(-x).exp())
    }

    fn parameters(self) -> Self::Params {
        ()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
/// Softmax operation
pub struct SoftMax<D>
where
    D: IntoDims,
{
    /// Dimensions to calculate softmax across
    pub dims: D
}

impl<Input, D> Module<Input> for &SoftMax<D>
where
    D: IntoDims + Clone,
    Input: Clone + ops::Max + Sub<<Input as ops::Max>::Output>,
    <Input as Sub<<Input as ops::Max>::Output>>::Output: ops::Exp,
    <<Input as Sub<<Input as ops::Max>::Output>>::Output as ops::Exp>::Output: Clone + ops::Sum,
    <<Input as Sub<<Input as ops::Max>::Output>>::Output as ops::Exp>::Output: Div<<<<Input as Sub<<Input as ops::Max>::Output>>::Output as ops::Exp>::Output as ops::Sum>::Output>,
{
    type Output = <<<Input as Sub<<Input as ops::Max>::Output>>::Output as ops::Exp>::Output as Div<<<<Input as Sub<<Input as ops::Max>::Output>>::Output as ops::Exp>::Output as ops::Sum>::Output>>::Output;
    type Params = ();

    fn forward(self, x: Input) -> Self::Output {
        use crate::ops::{Exp, Sum};
        // TODO: Check if cloning Tensors (that is also cloning their grad_fn)
        // has any effect on correctness of this function's gradient calculation
        let temp = (x.clone() - x.max(())).exp();
        temp.clone() / temp.sum(self.dims.clone())
    }

    fn parameters(self) -> Self::Params {
        ()
    }
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

    let y = (&x).apply(&SoftMax { dims: -1 });
    println!("\n{}", y);
    y.backward();
    println!("\n{}", x);
    //panic!();
}

/// Sum operation
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct Sum<D>
where
    D: IntoDims,
{
    /// Dimensions to sum
    pub dims: D,
}

impl<Input, D> Module<Input> for &Sum<D>
where
    Input: ops::Sum,
    D: IntoDims + Clone,
{
    type Output = Input::Output;
    type Params = ();

    fn forward(self, x: Input) -> Self::Output {
        x.sum(self.dims.clone())
    }
    
    fn parameters(self) -> Self::Params {
        ()
    }
}

/// Max operation
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct Max<D>
where
    D: IntoDims,
{
    /// Dimensions to max
    pub dims: D,
}

impl<Input, D> Module<Input> for &Max<D>
where
    Input: ops::Max,
    D: IntoDims + Clone,
{
    type Output = Input::Output;
    type Params = ();

    fn forward(self, x: Input) -> Self::Output {
        x.max(self.dims.clone())
    }
    
    fn parameters(self) -> Self::Params {
        ()
    }
}

/// Min operation
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct Min<D> {
    /// Dimensions to min
    pub dims: D,
}

impl<Input, D> Module<Input> for &Min<D>
where
    Input: ops::Min,
    D: IntoDims + Clone,
{
    type Output = Input::Output;
    type Params = ();

    fn forward(self, x: Input) -> Self::Output {
        x.min(self.dims.clone())
    }

    fn parameters(self) -> Self::Params {
        ()
    }
}

/// Mean operation
pub struct Mean<D>
where
    D: IntoDims,
{
    dims: D
}

impl<Input, D> Module<Input> for &Mean<D>
where
    D: IntoDims + Clone,
    Input: ops::Sum + ConvertFrom<usize> + GetShape,
    <Input as ops::Sum>::Output: Div<Input>,
{
    type Output = <<Input as ops::Sum>::Output as Div<Input>>::Output;
    type Params = ();

    fn forward(self, x: Input) -> Self::Output {
        // TODO: We can't do Input::cfrom, because we don't know what type Input has and whether that type implements ConvertFrom<usize>
        let n = Input::cfrom(x.shape().numel());
        x.sum(self.dims.clone())/n
    }

    fn parameters(self) -> Self::Params {
        ()
    }
}

/// MSE loss
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct MSELoss;

impl<Y, YP> Module<(Y, YP)> for &MSELoss
where
    Y: Sub<YP>,
    <Y as Sub<YP>>::Output: Pow<i32>,
{
    type Output = <<Y as Sub<YP>>::Output as Pow<i32>>::Output;
    type Params = ();

    fn forward(self, x: (Y, YP)) -> Self::Output {
        (x.0 - x.1).pow(2)
    }

    fn parameters(self) -> Self::Params {
        ()
    }
}

//pub struct STD {}

/*#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct NormLayer {}

impl<Input> Module<Input> for &NormLayer {
    type Output = Self;
    fn forward(self, x: Input) -> Self::Output {
        (x - x.apply(Mean))/x.apply(STD)
    }
}*/

/// Linear layer
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct Linear<S> {
    w: Variable<S>,
    b: Variable<S>,
}

impl<S> Linear<S> {
    /// Create new Linear layer with given in_features and out_features dimensions
    pub fn new<T>(in_features: usize, out_features: usize) -> Self
    where
        S: ops::FromVec<T> + ops::Zeros,
        T: Clone + ops::Zeros + ops::Ones + rand::distributions::uniform::SampleUniform,
    {
        Self {
            w: Variable::<S>::uniform([in_features, out_features], T::zeros(()), T::ones(())),
            b: Variable::<S>::uniform([1, out_features], T::zeros(()), T::ones(())),
        }
    }
}

impl<'a, S, Input> Module<Input> for &'a Linear<S>
where
    S: 'a + Default + Zeros + Clone + Sub<Output = S> + Mul<Output = S> + Mul<f64, Output = S>,
    Input: ops::MatMul<&'a Variable<S>>,
    <Input as ops::MatMul<&'a Variable<S>>>::Output: std::ops::Add<&'a Variable<S>>,
{
    type Output = <<Input as ops::MatMul<&'a Variable<S>>>::Output as std::ops::Add<&'a Variable<S>>>::Output;
    type Params = (&'a Variable<S>, &'a Variable<S>);

    fn forward(self, x: Input) -> Self::Output {
        x.matmul(&self.w) + &self.b
    }

    fn parameters(self) -> Self::Params {
        (&self.w, &self.b)
    }
}

/// RNNCell
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct RNNCell<S> {
    wih: Variable<S>,
    bih: Variable<S>,
    whh: Variable<S>,
    bhh: Variable<S>,
}

impl<S> RNNCell<S> {
    /// Create new RNNCell with given input_size and hidden_size dimensions
    pub fn new<T>(input_size: usize, hidden_size: usize) -> Self
    where
        S: ops::FromVec<T> + ops::Zeros,
        T: Clone + ops::Zeros + ops::Ones + rand::distributions::uniform::SampleUniform,
    {
        Self {
            wih: Variable::<S>::uniform([input_size, hidden_size], T::zeros(()), T::ones(())),
            bih: Variable::<S>::uniform([1, hidden_size], T::zeros(()), T::ones(())),
            whh: Variable::<S>::uniform([hidden_size, hidden_size], T::zeros(()), T::ones(())),
            bhh: Variable::<S>::uniform([1, hidden_size], T::zeros(()), T::ones(())),
        }
    }
}

use ops::MatMul;
impl<'a, S, X, H> Module<(X, H)> for &'a RNNCell<S>
where
    S: Zeros + Clone + Default + Sub<Output = S> + Mul<Output = S> + Mul<f64, Output = S>,
    X: MatMul<&'a Variable<S>>,
    H: MatMul<&'a Variable<S>>,
    <X as MatMul<&'a Variable<S>>>::Output: Add<&'a Variable<S>>,
    <<X as MatMul<&'a Variable<S>>>::Output as Add<&'a Variable<S>>>::Output: Add<<H as MatMul<&'a Variable<S>>>::Output>,
    <<<X as MatMul<&'a Variable<S>>>::Output as Add<&'a Variable<S>>>::Output as Add<<H as MatMul<&'a Variable<S>>>::Output>>::Output: Add<&'a Variable<S>>,
{
    type Output = <<<<X as MatMul<&'a Variable<S>>>::Output as Add<&'a Variable<S>>>::Output as Add<<H as MatMul<&'a Variable<S>>>::Output>>::Output as Add<&'a Variable<S>>>::Output;
    type Params = (&'a Variable<S>, &'a Variable<S>, &'a Variable<S>, &'a Variable<S>);

    fn forward(self, x: (X, H)) -> Self::Output {
        x.0.matmul(&self.wih) + &self.bih + x.1.matmul(&self.whh) + &self.bhh
    }

    fn parameters(self) -> Self::Params {
        (&self.wih, &self.bih, &self.whh, &self.bhh)
    }
}
