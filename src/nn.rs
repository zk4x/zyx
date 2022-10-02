//! Structs that implement trait Module for anything that it makes sense to implement this trait.
//! These include zyx::ops, as well as layers, such as Linear.
//!

use crate::{module::Module, ops, ops::MatMul, tensor::Variable, prelude::ModuleParams, init::UniformInit};
use std::ops::Add;

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
    fn parameters(&self) -> Vec<&'a Variable<S>> {
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
    fn parameters(&self) -> Vec<&'a Variable<S>> {
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
    fn parameters(&self) -> Vec<&'a Variable<S>> {
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
    fn parameters(&self) -> Vec<&'a Variable<S>> {
        Vec::new()
    }
}

#[derive(Debug)]
pub struct Sigmoid;

/*impl<Input> Module<Input> for &Sigmoid
where
    Input: std::ops::Neg,
    <Input as std::ops::Neg>::Output: ops::Exp,
//std::ops::Div
{
    type Output = <<<Input as std::ops::Neg>::Output as ops::Exp>::Output as std::ops::Div>::Output;
    fn forward(self, x: Input) -> Self::Output {
        use crate::ops::Ones;
        Buffer::ones(&[])/(Buffer::ones(&[])+(-x).exp())
    }
}*/

impl<'a, S> ModuleParams<'a, S> for Sigmoid {
    fn parameters(&self) -> Vec<&'a Variable<S>> {
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
    fn parameters(&self) -> Vec<&'a Variable<S>> {
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
    fn parameters(&self) -> Vec<&'a Variable<S>> {
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
    fn parameters(&self) -> Vec<&'a Variable<S>> {
        Vec::new()
    }
}

#[derive(Debug)]
pub struct Linear<S> {
    w: Variable<S>,
    b: Variable<S>,
}

impl<S> Linear<S> {
    pub fn new<T>(in_features: usize, out_features: usize) -> Self
    where
        S: ops::FromVec<T> + ops::Zeros,
        T: Clone + ops::Zeros + ops::Ones + rand::distributions::uniform::SampleUniform,
    {
        Self {
            w: Variable::<S>::uniform(&[in_features, out_features], T::zeros(&[]), T::ones(&[])),
            b: Variable::<S>::uniform(&[1, out_features], T::zeros(&[]), T::ones(&[])),
        }
    }
}

impl<'a, S, Input> Module<Input> for &'a Linear<S>
where
    Input: ops::MatMul<&'a Variable<S>>,
    <Input as ops::MatMul<&'a Variable<S>>>::Output: std::ops::Add<&'a Variable<S>>,
{
    type Output = <<Input as ops::MatMul<&'a Variable<S>>>::Output as std::ops::Add<&'a Variable<S>>>::Output;
    fn forward(self, x: Input) -> Self::Output {
        x.matmul(&self.w) + &self.b
    }
}

impl<'a, S> ModuleParams<'a, S> for Linear<S> {
    fn parameters(&'a self) -> Vec<&'a Variable<S>> {
        vec![&self.w, &self.b]
    }
}

pub struct RNNCell<S> {
    wih: Variable<S>,
    bih: Variable<S>,
    whh: Variable<S>,
    bhh: Variable<S>,
}

impl<S> RNNCell<S> {
    pub fn new<T>(input_size: usize, hidden_size: usize) -> Self
    where
        S: ops::FromVec<T> + ops::Zeros,
        T: Clone + ops::Zeros + ops::Ones + rand::distributions::uniform::SampleUniform,
    {
        Self {
            wih: Variable::<S>::uniform(&[input_size, hidden_size], T::zeros(&[]), T::ones(&[])),
            bih: Variable::<S>::uniform(&[1, hidden_size], T::zeros(&[]), T::ones(&[])),
            whh: Variable::<S>::uniform(&[hidden_size, hidden_size], T::zeros(&[]), T::ones(&[])),
            bhh: Variable::<S>::uniform(&[1, hidden_size], T::zeros(&[]), T::ones(&[])),
        }
    }
}

impl<'a, S, X, H> Module<(X, H)> for &'a RNNCell<S>
where
    X: MatMul<&'a Variable<S>>,
    H: MatMul<&'a Variable<S>>,
    <X as MatMul<&'a Variable<S>>>::Output: Add<&'a Variable<S>>,
    <<X as MatMul<&'a Variable<S>>>::Output as Add<&'a Variable<S>>>::Output: Add<<H as MatMul<&'a Variable<S>>>::Output>,
    <<<X as MatMul<&'a Variable<S>>>::Output as Add<&'a Variable<S>>>::Output as Add<<H as MatMul<&'a Variable<S>>>::Output>>::Output: Add<&'a Variable<S>>,
{
    type Output = <<<<X as MatMul<&'a Variable<S>>>::Output as Add<&'a Variable<S>>>::Output as Add<<H as MatMul<&'a Variable<S>>>::Output>>::Output as Add<&'a Variable<S>>>::Output;
    fn forward(self, x: (X, H)) -> Self::Output {
        x.0.matmul(&self.wih) + &self.bih + x.1.matmul(&self.whh) + &self.bhh
    }
}

impl<'a, S> ModuleParams<'a, S> for RNNCell<S> {
    fn parameters(&'a self) -> Vec<&'a Variable<S>> {
        vec![&self.wih, &self.bih, &self.whh, &self.bhh]
    }
}
