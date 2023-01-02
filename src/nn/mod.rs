//! Structs that implement trait [Module](crate::nn::Module) for anything that it makes sense to implement this trait.
//! These include zyx::ops, as well as layers, such as [Linear].
//!
//! This module is expected to get most stuff added.
//! It will contain functors, layers, models, cells, simply anything that can have .forward(input) function.
//!

mod module;
pub mod parameters;

pub use module::{ApplyModule, Module};

use crate::{
    nn::parameters::HasParameters,
    ops::{self, HasDType, HasShape, IntoVariable, MatMul, Pow, ZerosLike},
    shape::{Axes, Sh2, Shape},
    tensor::Variable,
};
use core::{
    marker::PhantomData,
    ops::{Add, Div, Neg, Sub},
};

/// ReLU operation
///
/// Applies the rectified linear unit function element-wise:
///
/// &emsp; y = (x)<sup>+</sup> = max(0, x)
///
/// Parameters:
///
/// > ()
///
/// Input:
///
/// > anything that implements [ReLU](crate::ops::ReLU)
///
/// Example:
/// ```
/// # use zyx::prelude::*;
/// # use zyx::device::cpu;
/// # use zyx::nn::ReLU;
/// # let device = cpu::Device::default();
/// let x = device.buffer([-1., 2.]);
/// let y = x.apply(&ReLU {});
/// assert_eq!(y, [0., 2.]);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ReLU;

impl<Input> Module<'_, Input> for ReLU
where
    Input: ops::ReLU,
{
    type Output = Input::Output;
    fn forward(&self, x: Input) -> Self::Output {
        x.relu()
    }
}

impl<'p> HasParameters<'p> for ReLU {
    type Params = ();
    fn parameters(&'p mut self) -> Self::Params {}
}

/// Exp operation
///
/// Returns a new tensor with the exponential of the elements of the input tensor.
///
/// &emsp; y = e<sup>x</sup>
///
/// Parameters:
///
/// > ()
///
/// Input:
///
/// > anything that implements [Exp](crate::ops::Exp)
///
/// Example:
/// ```
/// # use zyx::prelude::*;
/// # use zyx::device::cpu;
/// # use zyx::nn::Exp;
/// # let device = cpu::Device::default();
/// let x = device.buffer([0., 2.]);
/// let y = x.apply(&Exp {});
/// assert_eq!(y, [1., 7.38905609893065]);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Exp;

impl<Input> Module<'_, Input> for Exp
where
    Input: ops::Exp,
{
    type Output = Input::Output;
    fn forward(&self, x: Input) -> Self::Output {
        x.exp()
    }
}

impl<'p> HasParameters<'p> for Exp {
    type Params = ();
    fn parameters(&'p mut self) -> Self::Params {}
}

/// Ln operation
///
/// Returns a new tensor with the natural logarithm of the elements of input.
///
/// &emsp; y = log<sub>e</sub>(x)
///
/// Parameters:
///
/// > ()
///
/// Input:
///
/// > anything that implements [Ln](crate::ops::Ln)
///
/// Example:
/// ```
/// # use zyx::prelude::*;
/// # use zyx::device::cpu;
/// # use zyx::nn::Ln;
/// # let device = cpu::Device::default();
/// let x = device.buffer([1., 2.]);
/// let y = x.apply(&Ln {});
/// assert_eq!(y, [0., 0.6931471805599453]);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Ln;

impl<Input> Module<'_, Input> for Ln
where
    Input: ops::Ln,
{
    type Output = Input::Output;
    fn forward(&self, x: Input) -> Self::Output {
        x.ln()
    }
}

impl<'p> HasParameters<'p> for Ln {
    type Params = ();
    fn parameters(&'p mut self) -> Self::Params {}
}

/// Tanh operation
///
/// Applies the Hyperbolic Tangent (Tanh) function element-wise.
///
/// &emsp; y = tanh(x)
///
/// Parameters:
///
/// > ()
///
/// Input:
///
/// > anything that implements [Tanh](crate::ops::Tanh)
///
/// Example:
/// ```
/// # use zyx::prelude::*;
/// # use zyx::device::cpu;
/// # use zyx::nn::Tanh;
/// # let device = cpu::Device::default();
/// let x = device.buffer([0., 2.]);
/// let y = x.apply(&Tanh {});
/// assert_eq!(y, [0., 0.9640275800758169]);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Tanh;

impl<Input> Module<'_, Input> for Tanh
where
    Input: ops::Tanh,
{
    type Output = Input::Output;
    fn forward(&self, x: Input) -> Self::Output {
        x.tanh()
    }
}

impl<'p> HasParameters<'p> for Tanh {
    type Params = ();
    fn parameters(&'p mut self) -> Self::Params {}
}

/// Sigmoid operation
///
/// Applies the sigmoid function element-wise.
///
/// &emsp; y = σ(x)
///
/// Parameters:
///
/// > ()
///
/// Input:
///
/// > any [SType](crate::device::SType)
///
/// Example:
/// ```
/// # use zyx::prelude::*;
/// # use zyx::device::cpu;
/// # use zyx::nn::Sigmoid;
/// # let device = cpu::Device::default();
/// let x = device.buffer([0., 2.]);
/// let y = x.apply(&Sigmoid {});
/// assert_eq!(y, [0.5, 0.8807970779778823]);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Sigmoid;

impl<Input> Module<'_, Input> for Sigmoid
where
    Input: Neg + HasDType,
    Input::T: num_traits::One,
    <Input as Neg>::Output: ops::Exp,
    <<Input as Neg>::Output as ops::Exp>::Output: Add<Input::T>,
    <<<Input as Neg>::Output as ops::Exp>::Output as Add<Input::T>>::Output: Pow<i32>,
{
    type Output = <<<<Input as Neg>::Output as ops::Exp>::Output as Add<Input::T>>::Output as Pow<i32>>::Output;
    fn forward(&self, x: Input) -> Self::Output {
        use num_traits::One;
        use ops::Exp;
        ((-x).exp() + Input::T::one()).pow(-1)
    }
}

impl<'p> HasParameters<'p> for Sigmoid {
    type Params = ();
    fn parameters(&'p mut self) -> Self::Params {}
}

/*
/// Softmax operation
// TODO DOCS
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct SoftMax<Dims>
where
    Dims: Axes,
{
    /// Dimension to calculate softmax across
    pub dims: Dims
}

impl<Input, D> Module<'_, Input> for SoftMax<D>
where
    D: Clone + Axes,
    Input: Clone + ops::Maximizable<D> + Sub<<Input as ops::Maximizable<D>>::Output>,
    <Input as Sub<<Input as ops::Maximizable<D>>::Output>>::Output: ops::Exp,
    <<Input as Sub<<Input as ops::Maximizable<D>>::Output>>::Output as ops::Exp>::Output: Clone + ops::Summable<D>,
    <<Input as Sub<<Input as ops::Maximizable<D>>::Output>>::Output as ops::Exp>::Output: Div<<<<Input as Sub<<Input as ops::Maximizable<D>>::Output>>::Output as ops::Exp>::Output as ops::Summable<D>>::Output>,
{
    type Output = <<<Input as Sub<<Input as ops::Max<D>>::Output>>::Output as ops::Exp>::Output as Div<<<<Input as Sub<<Input as ops::Max<D>>::Output>>::Output as ops::Exp>::Output as ops::Sum<D>>::Output>>::Output;
    type Params = ();

    fn forward(&self, x: Input) -> Self::Output {
        use crate::ops::{Exp, Sum};
        // TODO: Check if cloning Tensors (that is also cloning their grad_fn)
        // has any effect on correctness of this function's gradient calculation
        let temp = (x.clone() - x.max()).exp();
        temp.clone() / temp.sum()
    }

    fn parameters(&mut self) -> Self::Params {}
}*/

#[test]
fn softmax_test() {
    //use crate::prelude::*;
    //use crate::device::cpu::Buffer;

    //let x = Buffer::<f32, _>::cfrom([[3., 2., 4.], [4., 2., 5.]]).with_grad();

    /*let dim = -1;
    let e_x = ((&x).data().clone() - (&x).max(())).exp();
    println!("\n{}", e_x);
    let y = e_x.clone() / e_x.sum(dim);
    println!("\n{}", y);
    y.backward();
    println!("\n{}", x.grad());*/

    /*let sm = SoftMax { dims: -1 };
    let y = sm.forward(&x);
    //println!("\n{}", y);
    y.backward();
    //println!("\n{}", x);
    //panic!();*/
}

/// Sum operation
// TODO DOCS
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Sum<Dims>
where
    Dims: Axes,
{
    dims: PhantomData<Dims>,
}

impl<Dims> Sum<Dims>
where
    Dims: Axes,
{
    /// Create new Sum functor
    // TODO DOCS
    pub fn new() -> Self {
        Self { dims: PhantomData }
    }
}

impl<Input, Dims> Module<'_, Input> for Sum<Dims>
where
    Input: ops::Summable<Dims>,
    Dims: Axes,
{
    type Output = Input::Output;
    fn forward(&self, x: Input) -> Self::Output {
        x._sum()
    }
}

impl<'p, Dims> HasParameters<'p> for Sum<Dims>
where
    Dims: Axes,
{
    type Params = ();
    fn parameters(&'p mut self) -> Self::Params {}
}

/// Max operation
// TODO DOCS
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Max<Dims>
where
    Dims: Axes,
{
    // [Dimensions](crate::shape::Shape) to max
    dims: PhantomData<Dims>,
}

impl<Dims> Max<Dims>
where
    Dims: Axes,
{
    /// Create new Max functor
    // TODO DOCS
    pub fn new() -> Self {
        Self { dims: PhantomData }
    }
}

impl<Input, Dims> Module<'_, Input> for Max<Dims>
where
    Input: ops::Maximizable<Dims>,
    Dims: Axes,
{
    type Output = (Input::Values, Input::Indices);
    fn forward(&self, x: Input) -> Self::Output {
        x._max()
    }
}

impl<'p, Dims> HasParameters<'p> for Max<Dims>
where
    Dims: Axes,
{
    type Params = ();
    fn parameters(&'p mut self) -> Self::Params {}
}

/// Minimizable operation
// TODO DOCS
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Min<Dims>
where
    Dims: Axes,
{
    // [Dimensions](crate::shape::Shape) to min
    dims: PhantomData<Dims>,
}

impl<Dims> Min<Dims>
where
    Dims: Axes,
{
    /// Create new Min functor
    // TODO DOCS
    pub fn new() -> Self {
        Self { dims: PhantomData }
    }
}

impl<Input, Dims> Module<'_, Input> for Min<Dims>
where
    Input: ops::Minimizable<Dims>,
    Dims: Axes,
{
    type Output = (Input::Values, Input::Indices);
    fn forward(&self, x: Input) -> Self::Output {
        x._min()
    }
}

impl<'p, Dims> HasParameters<'p> for Min<Dims>
where
    Dims: Axes,
{
    type Params = ();
    fn parameters(&'p mut self) -> Self::Params {}
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
/// Mean operation
// TODO DOCS
pub struct Mean<Dims>
where
    Dims: Axes,
{
    // [Dimensions](crate::shape::Shape) for mean
    dims: PhantomData<Dims>,
}

impl<Dims> Mean<Dims>
where
    Dims: Axes,
{
    /// Create new Mean functor
    // TODO DOCS
    pub fn new() -> Self {
        Self { dims: PhantomData }
    }
}

impl<Input, Dims> Module<'_, Input> for Mean<Dims>
where
    Dims: Axes,
    Input: HasShape + ops::Summable<Dims>,
    <Input as ops::Summable<Dims>>::Output: Div<i32>,
{
    type Output = <<Input as ops::Summable<Dims>>::Output as Div<i32>>::Output;
    fn forward(&self, x: Input) -> Self::Output {
        x._sum() / (Input::Sh::NUMEL as i32)
    }
}

impl<'p, Dims> HasParameters<'p> for Mean<Dims>
where
    Dims: Axes,
{
    type Params = ();
    fn parameters(&'p mut self) -> Self::Params {}
}

/// MSE loss
// TODO DOCS
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct MSELoss;

impl<Y, YP> Module<'_, (Y, YP)> for MSELoss
where
    Y: Sub<YP>,
    <Y as Sub<YP>>::Output: Pow<i32>,
{
    type Output = <<Y as Sub<YP>>::Output as Pow<i32>>::Output;
    fn forward(&self, x: (Y, YP)) -> Self::Output {
        (x.0 - x.1).pow(2)
    }
}

impl<'p> HasParameters<'p> for MSELoss {
    type Params = ();
    fn parameters(&'p mut self) -> Self::Params {}
}

//pub struct STD {}

/*#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct NormLayer {}

impl<Input> Module<Input> for NormLayer {
    type Output = Self;
    fn forward(&self, x: Input) -> Self::Output {
        (x - x.apply(&Mean))/x.apply(&STD)
    }
}*/

/*use self::parameters::HasParameters;
/// Linear layer
// TODO DOCS
#[derive(Debug, Clone)]
pub struct Linear<
    'd,
    const IN_FEATURES: usize,
    const OUT_FEATURES: usize,
    W = Buffer<'d, Sh2<IN_FEATURES, OUT_FEATURES>>,
    B = Buffer<'d, Sh2<1, OUT_FEATURES>>,
> where
    W: HasShape,
    B: HasShape,
{
    dev: PhantomData<&'d W>,
    w: Variable<W>,
    b: Variable<B>,
}

impl<'d, W, B, const IN_FEATURES: usize, const OUT_FEATURES: usize>
    Linear<'d, IN_FEATURES, OUT_FEATURES, W, B>
where
    W: HasShape + HasDType,
    W::T: num_traits::Zero + num_traits::One,
    B: HasShape + HasDType,
    B::T: num_traits::Zero + num_traits::One,
{
    /// Create new [Linear layer](Linear) with given in_features and out_features dimensions
    /// with parameters stored on given device.
    pub fn new<Dev>(device: &'d Dev) -> Self
    where
        Dev: BufferFromSlice<'d, W> + BufferFromSlice<'d, B>,
        W: 'd + IntoVariable,
        W::T: rand::distributions::uniform::SampleUniform,
        B: 'd + IntoVariable,
        B::T: rand::distributions::uniform::SampleUniform,
    {
        use num_traits::{One, Zero};
        Self {
            dev: PhantomData,
            w: <Dev as BufferInit<'d, W>>::uniform(device, W::T::zero(), W::T::one()).with_grad(),
            b: <Dev as BufferInit<'d, B>>::uniform(device, B::T::zero(), B::T::one()).with_grad(),
        }
    }
}

impl<'p, Input, W, B, const IN_FEATURES: usize, const OUT_FEATURES: usize> Module<'p, Input>
    for Linear<'_, IN_FEATURES, OUT_FEATURES, W, B>
where
    W: 'p + HasShape + HasDType,
    W::T: num_traits::Zero + num_traits::One,
    B: 'p + HasShape + HasDType,
    B::T: num_traits::Zero + num_traits::One,
    Input: MatMul<&'p Variable<W>>,
    <Input as MatMul<&'p Variable<W>>>::Output: Add<&'p Variable<B>>,
{
    type Output = <<Input as MatMul<&'p Variable<W>>>::Output as Add<&'p Variable<B>>>::Output;
    fn forward(&'p self, x: Input) -> Self::Output {
        x.matmul(&self.w) + &self.b
    }
}

impl<'p, W, B, const IN_FEATURES: usize, const OUT_FEATURES: usize> HasParameters<'p>
    for Linear<'_, IN_FEATURES, OUT_FEATURES, W, B>
where
    W: 'p + HasShape + HasDType + ZerosLike,
    B: 'p + HasShape + HasDType + ZerosLike,
{
    type Params = (&'p mut Variable<W>, &'p mut Variable<B>);
    fn parameters(&'p mut self) -> Self::Params {
        (&mut self.w, &mut self.b)
    }
}

/// RNNCell
// TODO DOCS
// TODO: Should we rewrite this as two linear layers?
#[derive(Debug, Clone)]
pub struct RNNCell<
    'd,
    const INPUT_SIZE: usize,
    const HIDDEN_SIZE: usize,
    WI = Buffer<'d, Sh2<INPUT_SIZE, HIDDEN_SIZE>>,
    BI = Buffer<'d, Sh2<1, HIDDEN_SIZE>>,
    WH = Buffer<'d, Sh2<HIDDEN_SIZE, HIDDEN_SIZE>>,
    BH = Buffer<'d, Sh2<1, HIDDEN_SIZE>>,
> where
    WI: HasShape,
    BI: HasShape,
    WH: HasShape,
    BH: HasShape,
{
    dev: PhantomData<&'d WI>,
    wih: Variable<WI>,
    bih: Variable<BI>,
    whh: Variable<WH>,
    bhh: Variable<BH>,
}

impl<'d, WI, BI, WH, BH, const INPUT_SIZE: usize, const HIDDEN_SIZE: usize>
    RNNCell<'d, INPUT_SIZE, HIDDEN_SIZE, WI, BI, WH, BH>
where
    WI: HasShape + HasDType,
    WI::T: num_traits::Zero + num_traits::One,
    BI: HasShape + HasDType,
    BI::T: num_traits::Zero + num_traits::One,
    WH: HasShape + HasDType,
    WH::T: num_traits::Zero + num_traits::One,
    BH: HasShape + HasDType,
    BH::T: num_traits::Zero + num_traits::One,
{
    /// Create new [RNNCell] with given input_size and output_size dimensions
    /// with parameters stored on given device.
    pub fn new<Dev>(device: &'d Dev) -> Self
    where
        Dev: BufferFromSlice<'d, WI>
            + BufferFromSlice<'d, BI>
            + BufferFromSlice<'d, WH>
            + BufferFromSlice<'d, BH>,
        WI: 'd + IntoVariable,
        WI::T: rand::distributions::uniform::SampleUniform,
        BI: 'd + IntoVariable,
        BI::T: rand::distributions::uniform::SampleUniform,
        WH: 'd + IntoVariable,
        WH::T: rand::distributions::uniform::SampleUniform,
        BH: 'd + IntoVariable,
        BH::T: rand::distributions::uniform::SampleUniform,
    {
        use num_traits::{One, Zero};
        Self {
            dev: PhantomData,
            wih: <Dev as BufferInit<'d, WI>>::uniform(device, WI::T::zero(), WI::T::one())
                .with_grad(),
            bih: <Dev as BufferInit<'d, BI>>::uniform(device, BI::T::zero(), BI::T::one())
                .with_grad(),
            whh: <Dev as BufferInit<'d, WH>>::uniform(device, WH::T::zero(), WH::T::one())
                .with_grad(),
            bhh: <Dev as BufferInit<'d, BH>>::uniform(device, BH::T::zero(), BH::T::one())
                .with_grad(),
        }
    }
}

impl<'p, I, H, WI, BI, WH, BH, const INPUT_SIZE: usize, const HIDDEN_SIZE: usize> Module<'p, (I, H)>
    for RNNCell<'_, INPUT_SIZE, HIDDEN_SIZE, WI, BI, WH, BH>
where
    WI: 'p + HasShape + HasDType,
    WI::T: num_traits::Zero + num_traits::One,
    BI: 'p + HasShape + HasDType,
    BI::T: num_traits::Zero + num_traits::One,
    WH: 'p + HasShape + HasDType,
    WH::T: num_traits::Zero + num_traits::One,
    BH: 'p + HasShape + HasDType,
    BH::T: num_traits::Zero + num_traits::One,
    I: MatMul<&'p Variable<WI>>,
    <I as MatMul<&'p Variable<WI>>>::Output: Add<&'p Variable<BI>>,
    H: MatMul<&'p Variable<WH>>,
    <H as MatMul<&'p Variable<WH>>>::Output: Add<&'p Variable<BH>>,
    <<I as MatMul<&'p Variable<WI>>>::Output as Add<&'p Variable<BI>>>::Output:
        Add<<<H as MatMul<&'p Variable<WH>>>::Output as Add<&'p Variable<BH>>>::Output>,
{
    type Output =
        <<<I as MatMul<&'p Variable<WI>>>::Output as Add<&'p Variable<BI>>>::Output as Add<
            <<H as MatMul<&'p Variable<WH>>>::Output as Add<&'p Variable<BH>>>::Output,
        >>::Output;
    fn forward(&'p self, x: (I, H)) -> Self::Output {
        (x.0.matmul(&self.wih) + &self.bih) + (x.1.matmul(&self.whh) + &self.bhh)
    }
}

impl<'p, WI, BI, WH, BH, const IN_FEATURES: usize, const OUT_FEATURES: usize> HasParameters<'p>
    for RNNCell<'_, IN_FEATURES, OUT_FEATURES, WI, BI, WH, BH>
where
    WI: 'p + HasShape + HasDType + ZerosLike,
    BI: 'p + HasShape + HasDType + ZerosLike,
    WH: 'p + HasShape + HasDType + ZerosLike,
    BH: 'p + HasShape + HasDType + ZerosLike,
{
    type Params = (
        &'p mut Variable<WI>,
        &'p mut Variable<BI>,
        &'p mut Variable<WH>,
        &'p mut Variable<BH>,
    );
    fn parameters(&'p mut self) -> Self::Params {
        (&mut self.wih, &mut self.bih, &mut self.whh, &mut self.bhh)
    }
}*/

/*impl<WI, BI, WH, BH> RNNCell<WI, BI, WH, BH> {
    /// Create new [RNNCell] with given input_size and hidden_size dimensions
    pub fn new<T>(input_size: usize, hidden_size: usize) -> Self
    where
        T: Zeros<Sh = Ax0> + Ones<Sh = Ax0>,
        WI: UniformInit<T = T, Sh = (usize, usize)>,
        BI: UniformInit<T = T, Sh = (usize, usize)>,
        WH: UniformInit<T = T, Sh = (usize, usize)>,
        BH: UniformInit<T = T, Sh = (usize, usize)>,
    {
        Self {
            wih: WI::uniform((input_size, hidden_size), T::zeros(), T::ones()).with_grad(),
            bih: BI::uniform((1, hidden_size), T::zeros(), T::ones()).with_grad(),
            whh: WH::uniform((hidden_size, hidden_size), T::zeros(), T::ones()).with_grad(),
            bhh: BH::uniform((1, hidden_size), T::zeros(), T::ones()).with_grad(),
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
}*/

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
