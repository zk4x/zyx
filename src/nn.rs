//! Structs that implement trait [Module](crate::module::Module) for anything that it makes sense to implement this trait.
//! These include zyx::ops, as well as layers, such as [Linear].
//!
//! This module is expected to get most stuff added.
//! It will contain functors, layers, models, cells, simply anything that can have .forward(input) function.
//!

use crate::{module::Module, ops::{self, GetShape, Pow, MatMul, Zeros, Ones}, tensor::{IntoVariable, Variable}, init::UniformInit, shape::Shape};
use core::ops::{Neg, Add, Sub, Div, Mul};

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
pub struct SoftMax<Dims>
where
    Dims: Shape<D = i32>,
{
    /// Dimension to calculate softmax across
    pub dims: Dims
}

impl<Input, D> Module<'_, Input> for SoftMax<D>
where
    D: Clone + Shape<D = i32>,
    Input: Clone + ops::Max<D> + Sub<<Input as ops::Max<D>>::Output>,
    <Input as Sub<<Input as ops::Max<D>>::Output>>::Output: ops::Exp,
    <<Input as Sub<<Input as ops::Max<D>>::Output>>::Output as ops::Exp>::Output: Clone + ops::Sum<D>,
    <<Input as Sub<<Input as ops::Max<D>>::Output>>::Output as ops::Exp>::Output: Div<<<<Input as Sub<<Input as ops::Max<D>>::Output>>::Output as ops::Exp>::Output as ops::Sum<D>>::Output>,
{
    type Output = <<<Input as Sub<<Input as ops::Max<D>>::Output>>::Output as ops::Exp>::Output as Div<<<<Input as Sub<<Input as ops::Max<D>>::Output>>::Output as ops::Exp>::Output as ops::Sum<D>>::Output>>::Output;
    type Params = ();

    fn forward(&self, x: Input) -> Self::Output {
        use crate::ops::{Exp, Sum};
        // TODO: Check if cloning Tensors (that is also cloning their grad_fn)
        // has any effect on correctness of this function's gradient calculation
        let temp = (x.clone() - x.max(self.dims)).exp();
        temp.clone() / temp.sum(self.dims)
    }

    fn parameters(&mut self) -> Self::Params {}
}

#[test]
fn softmax_test() {
    use crate::prelude::*;
    use crate::accel::cpu::Buffer;

    let x = Buffer::<f32, _>::cfrom([[3., 2., 4.], [4., 2., 5.]]).with_grad();

    /*let dim = -1;
    let e_x = ((&x).data().clone() - (&x).max(())).exp();
    println!("\n{}", e_x);
    let y = e_x.clone() / e_x.sum(dim);
    println!("\n{}", y);
    y.backward();
    println!("\n{}", x.grad());*/

    let sm = SoftMax { dims: -1 };
    let y = sm.forward(&x);
    //println!("\n{}", y);
    y.backward();
    //println!("\n{}", x);
    //panic!();
}

/// Sum operation
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Sum<Dims>
where
    Dims: Shape<D = i32>,
{
    /// [Dimensions](crate::shape::IntoDims) to sum
    pub dims: Dims,
}

impl<Input, Dims> Module<'_, Input> for Sum<Dims>
where
    Input: ops::Sum<Dims>,
    Dims: Shape<D = i32> + Clone,
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
pub struct Max<Dims>
where
    Dims: Shape<D = i32>,
{
    /// [Dimensions](crate::shape::IntoDims) to max
    pub dims: Dims,
}

impl<Input, Dims> Module<'_, Input> for Max<Dims>
where
    Input: ops::Max<Dims>,
    Dims: Shape<D = i32> + Clone,
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
pub struct Min<Dims>
where
    Dims: Shape<D = i32>,
{
    /// [Dimensions](crate::shape::IntoDims) to min
    pub dims: Dims,
}

impl<Input, Dims> Module<'_, Input> for Min<Dims>
where
    Input: ops::Min<Dims>,
    Dims: Clone + Shape<D = i32>,
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
pub struct Mean<Dims>
where
    Dims: Shape<D = i32>,
{
    /// [Dimensions](crate::shape::IntoDims) for mean
    pub dims: Dims
}

impl<Input, Dims> Module<'_, Input> for Mean<Dims>
where
    Dims: Clone + Shape<D = i32>,
    Input: GetShape + ops::Sum<Dims>,
    <Input as ops::Sum<Dims>>::Output: Div<usize>,
{
    type Output = <<Input as ops::Sum<Dims>>::Output as Div<usize>>::Output;
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
pub struct Linear<W, B> {
    w: Variable<W>,
    b: Variable<B>,
}

impl<W, B> Linear<W, B> {
    /// Create new [Linear layer](Linear) with given in_features and out_features dimensions
    pub fn new<T>(in_features: usize, out_features: usize) -> Self
    where
        T: Zeros<Sh = ()> + Ones<Sh = ()>,
        W: UniformInit<T = T, Sh = (usize, usize)>,
        B: UniformInit<T = T, Sh = (usize, usize)>,
    {
        Self {
            w: W::uniform((in_features, out_features), T::zeros(()), T::ones(())).with_grad(),
            b: B::uniform((1, out_features), T::zeros(()), T::ones(())).with_grad(),
        }
    }
}

impl<'p, W, B, Input> Module<'p, Input> for Linear<W, B>
where
    W: Clone + Sub<<W as Mul<f64>>::Output, Output = W> + Mul<f64>,
    B: Clone + Sub<<B as Mul<f64>>::Output, Output = B> + Mul<f64>,

    W: 'p,
    B: 'p,

    Input: MatMul<&'p Variable<W>>,
    <Input as MatMul<&'p Variable<W>>>::Output: Add<&'p Variable<B>>,
{
    type Output = <<Input as MatMul<&'p Variable<W>>>::Output as Add<&'p Variable<B>>>::Output;
    type Params = (&'p mut Variable<W>, &'p mut Variable<B>);

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
pub struct RNNCell<WI, BI, WH, BH> {
    wih: Variable<WI>,
    bih: Variable<BI>,
    whh: Variable<WH>,
    bhh: Variable<BH>,
}

impl<WI, BI, WH, BH> RNNCell<WI, BI, WH, BH> {
    /// Create new [RNNCell] with given input_size and hidden_size dimensions
    pub fn new<T>(input_size: usize, hidden_size: usize) -> Self
    where
        T: Zeros<Sh = ()> + Ones<Sh = ()>,
        WI: UniformInit<T = T, Sh = (usize, usize)>,
        BI: UniformInit<T = T, Sh = (usize, usize)>,
        WH: UniformInit<T = T, Sh = (usize, usize)>,
        BH: UniformInit<T = T, Sh = (usize, usize)>,
    {
        Self {
            wih: WI::uniform((input_size, hidden_size), T::zeros(()), T::ones(())).with_grad(),
            bih: BI::uniform((1, hidden_size), T::zeros(()), T::ones(())).with_grad(),
            whh: WH::uniform((hidden_size, hidden_size), T::zeros(()), T::ones(())).with_grad(),
            bhh: BH::uniform((1, hidden_size), T::zeros(()), T::ones(())).with_grad(),
        }
    }
}

impl<'a, WI, BI, WH, BH, I, H> Module<'a, (I, H)> for RNNCell<WI, BI, WH, BH>
where
    WI: 'a + Clone + Sub<<WI as Mul<f64>>::Output, Output = WI> + Mul<f64>,
    BI: 'a + Clone + Sub<<BI as Mul<f64>>::Output, Output = BI> + Mul<f64>,
    WH: 'a + Clone + Sub<<WH as Mul<f64>>::Output, Output = WH> + Mul<f64>,
    BH: 'a + Clone + Sub<<BH as Mul<f64>>::Output, Output = BH> + Mul<f64>,

    I: MatMul<&'a Variable<WI>>,
    <I as MatMul<&'a Variable<WI>>>::Output: Add<&'a Variable<BI>>,
    H: MatMul<&'a Variable<WH>>,
    <H as MatMul<&'a Variable<WH>>>::Output: Add<&'a Variable<BH>>,
    <<I as MatMul<&'a Variable<WI>>>::Output as Add<&'a Variable<BI>>>::Output: Add<<<H as MatMul<&'a Variable<WH>>>::Output as Add<&'a Variable<BH>>>::Output>,
{
    type Output = <<<I as MatMul<&'a Variable<WI>>>::Output as Add<&'a Variable<BI>>>::Output as Add<<<H as MatMul<&'a Variable<WH>>>::Output as Add<&'a Variable<BH>>>::Output>>::Output;
    type Params = (&'a mut Variable<WI>, &'a mut Variable<BI>, &'a mut Variable<WH>, &'a mut Variable<BH>);

    fn forward(&'a self, x: (I, H)) -> Self::Output {
        (x.0.matmul(&self.wih) + &self.bih) + (x.1.matmul(&self.whh) + &self.bhh)
    }

    fn parameters(&'a mut self) -> Self::Params {
        (&mut self.wih, &mut self.bih, &mut self.whh, &mut self.bhh)
    }
}

/*#[derive(Debug, Clone)]
struct Attention<W, WG, F, FG, FB, FBG> {
    num_heads: usize,
    scale: f64,
    qkv: Variable<W, WG>,
    fc_out: Linear<F, FG, FB, FBG>,
}

impl<'p, W, WG, F, FG, FB, FBG, QKV> Attention<'p, QKV> for Attention<W, WG, F, WG, FB, FBG>
where
    W: 'p + Clone + Sub<<WG as Mul<f64>>::Output, Output = W>,
    WG: 'a + Clone + Mul<f64>,
    F: 'a + Clone + Sub<<FG as Mul<f64>>::Output, Output = F>,
    FG: 'a + Clone + Mul<f64>,
    FB: 'a + Clone + Sub<<FBG as Mul<f64>>::Output, Output = FB>,
    FBG: 'a + Clone + Mul<f64>,

{
    type Output = ();
    type Params = (&'p mut Variable<W, WG>, (&'p mut Variable<F, FG>, &'p mut Variable<FB, FBG>));

    fn forward(&'a self, x: QKV) -> Self::Output {
        // All that is needed for attention mechanism
        let s = x.shape();
        let (B, N, C) = (s[0], s[1], s[2]);
        qkv = x.matmul(self.qkv).reshape(B, B, 3, self.num_heads, C/self.num_heads).permute(2, 0, 3, 1, 4);
        let (q, k, v) = qkv.unbind(0); // we don't have unbind yet, so we need to implement it and then we will have working attention

        attn = (q.matmul(k.transpose())) * self.scale;
        attn = SoftMax { dim = -1 }.forward(attn);
        x = (attn.matmul(v)).permute(1, 2).reshape(B, N, C);

        self.fc_out.forward(x)
    }

    fn parameters(&'a mut self) -> Self::Params {
        (&mut self.qkv, self.fc_out.parameters())
    }
}*/
