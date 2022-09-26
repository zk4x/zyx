//! Structs that implement trait Module for anything that it makes sense to implement this trait.
//! These include zyx::ops, as well as layers, such as Linear.
//!

use crate::{module::Module, ops, tensor::{Tensor, TensorGrad}, prelude::ModuleParams};

#[derive(Debug)]
pub struct ReLU;

impl<Input> Module<Input> for &ReLU
where
    Input: ops::ReLU,
{
    type Output = Input::Output;
    fn forward(self, x: Input) -> Self::Output {
        x.relu()
    }
}

impl<'a, S> ModuleParams<'a, S> for ReLU {
    fn parameters(&self) -> Vec<&'a TensorGrad<S>> {
        Vec::new()
    }
}

#[derive(Debug)]
pub struct Exp;

impl<Input> Module<Input> for &Exp
where
    Input: ops::Exp,
{
    type Output = Input::Output;
    fn forward(self, x: Input) -> Self::Output {
        x.exp()
    }
}

impl<'a, S> ModuleParams<'a, S> for Exp {
    fn parameters(&self) -> Vec<&'a TensorGrad<S>> {
        Vec::new()
    }
}

#[derive(Debug)]
pub struct Ln;

impl<Input> Module<Input> for &Ln
where
    Input: ops::Ln,
{
    type Output = Input::Output;
    fn forward(self, x: Input) -> Self::Output {
        x.ln()
    }
}

impl<'a, S> ModuleParams<'a, S> for Ln {
    fn parameters(&self) -> Vec<&'a TensorGrad<S>> {
        Vec::new()
    }
}

#[derive(Debug)]
pub struct Tanh;

impl<Input> Module<Input> for &Tanh
where
    Input: ops::Tanh,
{
    type Output = Input::Output;
    fn forward(self, x: Input) -> Self::Output {
        x.tanh()
    }
}

impl<'a, S> ModuleParams<'a, S> for Tanh {
    fn parameters(&self) -> Vec<&'a TensorGrad<S>> {
        Vec::new()
    }
}

#[derive(Debug)]
pub struct Sum<'a> {
    pub dims: &'a [i32],
}

impl<'a, Input> Module<Input> for &Sum<'a>
where
    Input: ops::Sum,
{
    type Output = Input::Output;
    fn forward(self, x: Input) -> Self::Output {
        x.sum(self.dims)
    }
}

impl<'a, S> ModuleParams<'a, S> for Sum<'a> {
    fn parameters(&self) -> Vec<&'a TensorGrad<S>> {
        Vec::new()
    }
}

#[derive(Debug)]
pub struct Max<'a> {
    pub dims: &'a [i32],
}

impl<'a, Input> Module<Input> for &Max<'a>
where
    Input: ops::Max,
{
    type Output = Input::Output;
    fn forward(self, x: Input) -> Self::Output {
        x.max(self.dims)
    }
}

impl<'a, S> ModuleParams<'a, S> for Max<'a> {
    fn parameters(&self) -> Vec<&'a TensorGrad<S>> {
        Vec::new()
    }
}

#[derive(Debug)]
pub struct Min<'a> {
    pub dims: &'a [i32],
}

impl<'a, Input> Module<Input> for &Min<'a>
where
    Input: ops::Min,
{
    type Output = Input::Output;
    fn forward(self, x: Input) -> Self::Output {
        x.min(self.dims)
    }
}

impl<'a, S> ModuleParams<'a, S> for Min<'a> {
    fn parameters(&self) -> Vec<&'a TensorGrad<S>> {
        Vec::new()
    }
}

#[derive(Debug)]
pub struct Linear<S> {
    w: TensorGrad<S>,
    b: TensorGrad<S>,
}

impl<S> Linear<S> {
    pub fn new<T>(in_features: usize, out_features: usize) -> Self
    where
        for<'a> S: ops::ConvertFrom<&'a crate::buffer::cpu::Buffer<T>>,
        T: Clone + ops::Zeros + ops::Ones + rand::distributions::uniform::SampleUniform,
    {
        use ops::ConvertFrom;
        Self {
            w: TensorGrad::<S>::convert_from(Tensor::uniform(&[in_features, out_features], T::zeros(&[]), T::ones(&[])).with_grad()),
            b: TensorGrad::<S>::convert_from(Tensor::uniform(&[1, out_features], T::zeros(&[]), T::ones(&[])).with_grad()),
        }
    }
}

impl<'a, S, Input> Module<Input> for &'a Linear<S>
where
    Input: ops::MatMul<&'a TensorGrad<S>>,
    <Input as ops::MatMul<&'a TensorGrad<S>>>::Output: std::ops::Add<&'a TensorGrad<S>>,
{
    //type Output = TensorFunc<S, impl FnOnce(S)>;
    type Output = <<Input as ops::MatMul<&'a TensorGrad<S>>>::Output as std::ops::Add<&'a TensorGrad<S>>>::Output;
    fn forward(self, x: Input) -> Self::Output
    {
        x.matmul(&self.w) + &self.b
    }
}

impl<'a, S> ModuleParams<'a, S> for Linear<S> {
    fn parameters(&'a self) -> Vec<&'a TensorGrad<S>> {
        vec![&self.w, &self.b]
    }
}
