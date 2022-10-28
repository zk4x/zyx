//! [Module] is generic trait that implements only one method: [forward](Module::forward()).
//! It is meant to be implemented mainly for functors and layers.
//!
//! This module also contains implementation of [Apply], that is automatically
//! implemented for anything that implements [Module] and allows users to use both
//! module.forward(input) notation as well as
//! input.apply(module) notation
//!
//! Basically all functions, layers and models should have [module.forward(input)](Module::forward()) function and this module also provides [input.apply(module)](Apply::apply()) function.
//! We think it is usefull for the user to have access to both standard [module.forward(input)] type of API and API with monads.
//! 

// TODO: use macros to make this DRY

use crate::{tensor::Variable, ops::{Zeros, GetShape}, optim::Optimizer};
use std::ops::{Sub, Mul};

// We can just store all Variables in tuple and implement some trait for this tuple that will take input and call
// all the required methods - update_data, zero_grad.
// This way there is no need for dyn.

/// # Parameters trait
/// 
/// Implemented for different arrays/tuples/vecs of [Variables](crate::tensor::Variable).
/// These can then be used by [optimizers](crate::optim).
pub trait Parameters {
    /// Update [Parameter's](Parameters) data
    fn update_data<Optim>(&self, optim: &Optim)
    where
        Optim: Optimizer; // these functions should be callable only by optimizers
    /// Zero [Parameter's](Parameters) gradients
    fn zero_grad(&self);
}

impl Parameters for () {
    fn update_data<Optim>(&self, _: &Optim)
    where
        Optim: Optimizer {}
    fn zero_grad(&self) {}
}

impl<S, const N: usize> Parameters for [&Variable<S>; N]
where
    S: Zeros + Clone + Default + Sub<Output = S> + Mul<Output = S> + Mul<f64, Output = S> + GetShape,
{
    fn update_data<Optim>(&self, optim: &Optim)
    where
        Optim: Optimizer
    {
        self.iter().for_each(|x| x.update_data(optim));
    }

    fn zero_grad(&self) {
        self.iter().for_each(|x| x.zero_grad());
    }
}

impl<S> Parameters for Vec<&Variable<S>>
where
    S: Zeros + Clone + Default + Sub<Output = S> + Mul<Output = S> + Mul<f64, Output = S> + GetShape,
{
    fn update_data<Optim>(&self, optim: &Optim)
    where
        Optim: Optimizer
    {
        self.iter().for_each(|x| x.update_data(optim));
    }

    fn zero_grad(&self) {
        self.iter().for_each(|x| x.zero_grad());
    }
}

impl<Params1, Params2> Parameters for (Params1, Params2)
where
    Params1: Parameters,
    Params2: Parameters,
{
    fn update_data<Optim>(&self, optim: &Optim)
    where
        Optim: Optimizer,
    {
        self.0.update_data(optim);
        self.1.update_data(optim);
    }

    fn zero_grad(&self) {
        self.0.zero_grad();
        self.1.zero_grad();
    }
}

impl<Params1, Params2, Params3> Parameters for (Params1, Params2, Params3)
where
    Params1: Parameters,
    Params2: Parameters,
    Params3: Parameters,
{
    fn update_data<Optim>(&self, optim: &Optim)
    where
        Optim: Optimizer,
    {
        self.0.update_data(optim);
        self.1.update_data(optim);
        self.2.update_data(optim);
    }

    fn zero_grad(&self) {
        self.0.zero_grad();
        self.1.zero_grad();
        self.2.zero_grad();
    }
}

impl<Params1, Params2, Params3, Params4> Parameters for (Params1, Params2, Params3, Params4)
where
    Params1: Parameters,
    Params2: Parameters,
    Params3: Parameters,
    Params4: Parameters,
{
    fn update_data<Optim>(&self, optim: &Optim)
    where
        Optim: Optimizer,
    {
        self.0.update_data(optim);
        self.1.update_data(optim);
        self.2.update_data(optim);
        self.3.update_data(optim);
    }

    fn zero_grad(&self) {
        self.0.zero_grad();
        self.1.zero_grad();
        self.2.zero_grad();
        self.3.zero_grad();
    }
}

impl<Params1, Params2, Params3, Params4, Params5> Parameters for (Params1, Params2, Params3, Params4, Params5)
where
    Params1: Parameters,
    Params2: Parameters,
    Params3: Parameters,
    Params4: Parameters,
    Params5: Parameters,
{
    fn update_data<Optim>(&self, optim: &Optim)
    where
        Optim: Optimizer,
    {
        self.0.update_data(optim);
        self.1.update_data(optim);
        self.2.update_data(optim);
        self.3.update_data(optim);
        self.4.update_data(optim);
    }

    fn zero_grad(&self) {
        self.0.zero_grad();
        self.1.zero_grad();
        self.2.zero_grad();
        self.3.zero_grad();
        self.4.zero_grad();
    }
}

impl<Params1, Params2, Params3, Params4, Params5, Params6> Parameters for (Params1, Params2, Params3, Params4, Params5, Params6)
where
    Params1: Parameters,
    Params2: Parameters,
    Params3: Parameters,
    Params4: Parameters,
    Params5: Parameters,
    Params6: Parameters,
{
    fn update_data<Optim>(&self, optim: &Optim)
    where
        Optim: Optimizer,
    {
        self.0.update_data(optim);
        self.1.update_data(optim);
        self.2.update_data(optim);
        self.3.update_data(optim);
        self.4.update_data(optim);
        self.5.update_data(optim);
    }

    fn zero_grad(&self) {
        self.0.zero_grad();
        self.1.zero_grad();
        self.2.zero_grad();
        self.3.zero_grad();
        self.4.zero_grad();
        self.5.zero_grad();
    }
}

impl<Params1, Params2, Params3, Params4, Params5, Params6, Params7> Parameters for (Params1, Params2, Params3, Params4, Params5, Params6, Params7)
where
    Params1: Parameters,
    Params2: Parameters,
    Params3: Parameters,
    Params4: Parameters,
    Params5: Parameters,
    Params6: Parameters,
    Params7: Parameters,
{
    fn update_data<Optim>(&self, optim: &Optim)
    where
        Optim: Optimizer,
    {
        self.0.update_data(optim);
        self.1.update_data(optim);
        self.2.update_data(optim);
        self.3.update_data(optim);
        self.4.update_data(optim);
        self.5.update_data(optim);
        self.6.update_data(optim);
    }

    fn zero_grad(&self) {
        self.0.zero_grad();
        self.1.zero_grad();
        self.2.zero_grad();
        self.3.zero_grad();
        self.4.zero_grad();
        self.5.zero_grad();
        self.6.zero_grad();
    }
}

impl<Params1, Params2, Params3, Params4, Params5, Params6, Params7, Params8> Parameters for (Params1, Params2, Params3, Params4, Params5, Params6, Params7, Params8)
where
    Params1: Parameters,
    Params2: Parameters,
    Params3: Parameters,
    Params4: Parameters,
    Params5: Parameters,
    Params6: Parameters,
    Params7: Parameters,
    Params8: Parameters,
{
    fn update_data<Optim>(&self, optim: &Optim)
    where
        Optim: Optimizer,
    {
        self.0.update_data(optim);
        self.1.update_data(optim);
        self.2.update_data(optim);
        self.3.update_data(optim);
        self.4.update_data(optim);
        self.5.update_data(optim);
        self.6.update_data(optim);
        self.7.update_data(optim);
    }

    fn zero_grad(&self) {
        self.0.zero_grad();
        self.1.zero_grad();
        self.2.zero_grad();
        self.3.zero_grad();
        self.4.zero_grad();
        self.5.zero_grad();
        self.6.zero_grad();
        self.7.zero_grad();
    }
}

impl<Params1, Params2, Params3, Params4, Params5, Params6, Params7, Params8, Params9> Parameters for (Params1, Params2, Params3, Params4, Params5, Params6, Params7, Params8, Params9)
where
    Params1: Parameters,
    Params2: Parameters,
    Params3: Parameters,
    Params4: Parameters,
    Params5: Parameters,
    Params6: Parameters,
    Params7: Parameters,
    Params8: Parameters,
    Params9: Parameters,
{
    fn update_data<Optim>(&self, optim: &Optim)
    where
        Optim: Optimizer,
    {
        self.0.update_data(optim);
        self.1.update_data(optim);
        self.2.update_data(optim);
        self.3.update_data(optim);
        self.4.update_data(optim);
        self.5.update_data(optim);
        self.6.update_data(optim);
        self.7.update_data(optim);
        self.8.update_data(optim);
    }

    fn zero_grad(&self) {
        self.0.zero_grad();
        self.1.zero_grad();
        self.2.zero_grad();
        self.3.zero_grad();
        self.4.zero_grad();
        self.5.zero_grad();
        self.6.zero_grad();
        self.7.zero_grad();
        self.8.zero_grad();
    }
}

/// # Module trait
/// 
/// Module can be implemented for anything that can have forward function.
/// Forward simply takes any input, applies some operation and returns output.
/// Every module also has some or zero [parameters](Parameters) that can be optimized
/// by [optimizers](crate::optim).
pub trait Module<Input> {
    /// Output of forward operation on [Module]
    type Output;
    /// [Parameters] of [Module]
    type Params: Parameters;
    /// Forward operation on [Module]
    fn forward(self, x: Input) -> Self::Output;
    /// Get parameters of [Module]
    fn parameters(self) -> Self::Params;
    // Set [parameters](Parameters) of [Module] (This is primarily used for loading models)
    //fn set_parameters(self, parameters: Self::Params) -> Self;
}

/// Apply trait allows us to use monads
pub trait Apply<Operation> {
    /// Output of forward operation on [Module]
    type Output;
    /// Forward operation on [Module]
    fn apply(self, x: Operation) -> Self::Output;
}

impl<Input, Operation> Apply<Operation> for Input
where
    Operation: Module<Input>
{
    type Output = Operation::Output;
    fn apply(self, x: Operation) -> Self::Output {
        x.forward(self)
    }
}

// Functions are modules
// Layers are modules
// Tuples of modules are modules
impl<'a, Input, M0, M1> Module<Input> for &'a (M0, M1)
where
    &'a M0: Module<Input>,
    &'a M1: Module<<&'a M0 as Module<Input>>::Output>,
{
    type Output = <&'a M1 as Module<<&'a M0 as Module<Input>>::Output>>::Output;
    type Params = (<&'a M0 as Module<Input>>::Params, <&'a M1 as Module<<&'a M0 as Module<Input>>::Output>>::Params);

    fn forward(self, x: Input) -> Self::Output {
        self.1.forward(self.0.forward(x))
    }

    fn parameters(self) -> Self::Params {
        (self.0.parameters(), self.1.parameters())
    }
}

impl<'a, Input, M0, M1, M2> Module<Input> for &'a (M0, M1, M2)
where
    &'a M0: Module<Input>,
    &'a M1: Module<<&'a M0 as Module<Input>>::Output>,
    &'a M2: Module<<&'a M1 as Module<<&'a M0 as Module<Input>>::Output>>::Output>,
{
    type Output = <&'a M2 as Module<<&'a M1 as Module<<&'a M0 as Module<Input>>::Output>>::Output>>::Output;
    type Params = (
        <&'a M0 as Module<Input>>::Params,
        <&'a M1 as Module<<&'a M0 as Module<Input>>::Output>>::Params,
        <&'a M2 as Module<<&'a M1 as Module<<&'a M0 as Module<Input>>::Output>>::Output>>::Params,
    );

    fn forward(self, x: Input) -> Self::Output {
        self.2.forward(self.1.forward(self.0.forward(x)))
    }

    fn parameters(self) -> Self::Params {
        (self.0.parameters(), self.1.parameters(), self.2.parameters())
    }
}

impl<'a, Input, M0, M1, M2, M3> Module<Input> for &'a (M0, M1, M2, M3)
where
    &'a M0: Module<Input>,
    &'a M1: Module<<&'a M0 as Module<Input>>::Output>,
    &'a M2: Module<<&'a M1 as Module<<&'a M0 as Module<Input>>::Output>>::Output>,
    &'a M3: Module<<&'a M2 as Module<<&'a M1 as Module<<&'a M0 as Module<Input>>::Output>>::Output>>::Output>,
{
    type Output = <&'a M3 as Module<<&'a M2 as Module<<&'a M1 as Module<<&'a M0 as Module<Input>>::Output>>::Output>>::Output>>::Output;
    type Params = (
        <&'a M0 as Module<Input>>::Params,
        <&'a M1 as Module<<&'a M0 as Module<Input>>::Output>>::Params,
        <&'a M2 as Module<<&'a M1 as Module<<&'a M0 as Module<Input>>::Output>>::Output>>::Params,
        <&'a M3 as Module<<&'a M2 as Module<<&'a M1 as Module<<&'a M0 as Module<Input>>::Output>>::Output>>::Output>>::Params,
    );

    fn forward(self, x: Input) -> Self::Output {
        self.3.forward(self.2.forward(self.1.forward(self.0.forward(x))))
    }

    fn parameters(self) -> Self::Params {
        (self.0.parameters(), self.1.parameters(), self.2.parameters(), self.3.parameters())
    }
}

impl<'a, Input, M0, M1, M2, M3, M4> Module<Input> for &'a (M0, M1, M2, M3, M4)
where
    &'a M0: Module<Input>,
    &'a M1: Module<<&'a M0 as Module<Input>>::Output>,
    &'a M2: Module<<&'a M1 as Module<<&'a M0 as Module<Input>>::Output>>::Output>,
    &'a M3: Module<<&'a M2 as Module<<&'a M1 as Module<<&'a M0 as Module<Input>>::Output>>::Output>>::Output>,
    &'a M4: Module<<&'a M3 as Module<<&'a M2 as Module<<&'a M1 as Module<<&'a M0 as Module<Input>>::Output>>::Output>>::Output>>::Output>,
{
    type Output = <&'a M4 as Module<<&'a M3 as Module<<&'a M2 as Module<<&'a M1 as Module<<&'a M0 as Module<Input>>::Output>>::Output>>::Output>>::Output>>::Output;
    type Params = (
        <&'a M0 as Module<Input>>::Params,
        <&'a M1 as Module<<&'a M0 as Module<Input>>::Output>>::Params,
        <&'a M2 as Module<<&'a M1 as Module<<&'a M0 as Module<Input>>::Output>>::Output>>::Params,
        <&'a M3 as Module<<&'a M2 as Module<<&'a M1 as Module<<&'a M0 as Module<Input>>::Output>>::Output>>::Output>>::Params,
        <&'a M4 as Module<<&'a M3 as Module<<&'a M2 as Module<<&'a M1 as Module<<&'a M0 as Module<Input>>::Output>>::Output>>::Output>>::Output>>::Params,
    );

    fn forward(self, x: Input) -> Self::Output {
        self.4.forward(self.3.forward(self.2.forward(self.1.forward(self.0.forward(x)))))
    }

    fn parameters(self) -> Self::Params {
        (self.0.parameters(), self.1.parameters(), self.2.parameters(), self.3.parameters(), self.4.parameters())
    }
}

impl<'a, Input, M0, M1, M2, M3, M4, M5> Module<Input> for &'a (M0, M1, M2, M3, M4, M5)
where
    &'a M0: Module<Input>,
    &'a M1: Module<<&'a M0 as Module<Input>>::Output>,
    &'a M2: Module<<&'a M1 as Module<<&'a M0 as Module<Input>>::Output>>::Output>,
    &'a M3: Module<<&'a M2 as Module<<&'a M1 as Module<<&'a M0 as Module<Input>>::Output>>::Output>>::Output>,
    &'a M4: Module<<&'a M3 as Module<<&'a M2 as Module<<&'a M1 as Module<<&'a M0 as Module<Input>>::Output>>::Output>>::Output>>::Output>,
    &'a M5: Module<<&'a M4 as Module<<&'a M3 as Module<<&'a M2 as Module<<&'a M1 as Module<<&'a M0 as Module<Input>>::Output>>::Output>>::Output>>::Output>>::Output>,
{
    type Output = <&'a M5 as Module<<&'a M4 as Module<<&'a M3 as Module<<&'a M2 as Module<<&'a M1 as Module<<&'a M0 as Module<Input>>::Output>>::Output>>::Output>>::Output>>::Output>>::Output;
    type Params = (
        <&'a M0 as Module<Input>>::Params,
        <&'a M1 as Module<<&'a M0 as Module<Input>>::Output>>::Params,
        <&'a M2 as Module<<&'a M1 as Module<<&'a M0 as Module<Input>>::Output>>::Output>>::Params,
        <&'a M3 as Module<<&'a M2 as Module<<&'a M1 as Module<<&'a M0 as Module<Input>>::Output>>::Output>>::Output>>::Params,
        <&'a M4 as Module<<&'a M3 as Module<<&'a M2 as Module<<&'a M1 as Module<<&'a M0 as Module<Input>>::Output>>::Output>>::Output>>::Output>>::Params,
        <&'a M5 as Module<<&'a M4 as Module<<&'a M3 as Module<<&'a M2 as Module<<&'a M1 as Module<<&'a M0 as Module<Input>>::Output>>::Output>>::Output>>::Output>>::Output>>::Params,
    );

    fn forward(self, x: Input) -> Self::Output {
        self.5.forward(self.4.forward(self.3.forward(self.2.forward(self.1.forward(self.0.forward(x))))))
    }

    fn parameters(self) -> Self::Params {
        (self.0.parameters(), self.1.parameters(), self.2.parameters(), self.3.parameters(), self.4.parameters(), self.5.parameters())
    }
}

impl<'a, Input, M0, M1, M2, M3, M4, M5, M6> Module<Input> for &'a (M0, M1, M2, M3, M4, M5, M6)
where
    &'a M0: Module<Input>,
    &'a M1: Module<<&'a M0 as Module<Input>>::Output>,
    &'a M2: Module<<&'a M1 as Module<<&'a M0 as Module<Input>>::Output>>::Output>,
    &'a M3: Module<<&'a M2 as Module<<&'a M1 as Module<<&'a M0 as Module<Input>>::Output>>::Output>>::Output>,
    &'a M4: Module<<&'a M3 as Module<<&'a M2 as Module<<&'a M1 as Module<<&'a M0 as Module<Input>>::Output>>::Output>>::Output>>::Output>,
    &'a M5: Module<<&'a M4 as Module<<&'a M3 as Module<<&'a M2 as Module<<&'a M1 as Module<<&'a M0 as Module<Input>>::Output>>::Output>>::Output>>::Output>>::Output>,
    &'a M6: Module<<&'a M5 as Module<<&'a M4 as Module<<&'a M3 as Module<<&'a M2 as Module<<&'a M1 as Module<<&'a M0 as Module<Input>>::Output>>::Output>>::Output>>::Output>>::Output>>::Output>,
{
    type Output = <&'a M6 as Module<<&'a M5 as Module<<&'a M4 as Module<<&'a M3 as Module<<&'a M2 as Module<<&'a M1 as Module<<&'a M0 as Module<Input>>::Output>>::Output>>::Output>>::Output>>::Output>>::Output>>::Output;
    type Params = (
        <&'a M0 as Module<Input>>::Params,
        <&'a M1 as Module<<&'a M0 as Module<Input>>::Output>>::Params,
        <&'a M2 as Module<<&'a M1 as Module<<&'a M0 as Module<Input>>::Output>>::Output>>::Params,
        <&'a M3 as Module<<&'a M2 as Module<<&'a M1 as Module<<&'a M0 as Module<Input>>::Output>>::Output>>::Output>>::Params,
        <&'a M4 as Module<<&'a M3 as Module<<&'a M2 as Module<<&'a M1 as Module<<&'a M0 as Module<Input>>::Output>>::Output>>::Output>>::Output>>::Params,
        <&'a M5 as Module<<&'a M4 as Module<<&'a M3 as Module<<&'a M2 as Module<<&'a M1 as Module<<&'a M0 as Module<Input>>::Output>>::Output>>::Output>>::Output>>::Output>>::Params,
        <&'a M6 as Module<<&'a M5 as Module<<&'a M4 as Module<<&'a M3 as Module<<&'a M2 as Module<<&'a M1 as Module<<&'a M0 as Module<Input>>::Output>>::Output>>::Output>>::Output>>::Output>>::Output>>::Params,
    );

    fn forward(self, x: Input) -> Self::Output {
        self.6.forward(self.5.forward(self.4.forward(self.3.forward(self.2.forward(self.1.forward(self.0.forward(x)))))))
    }

    fn parameters(self) -> Self::Params {
        (self.0.parameters(), self.1.parameters(), self.2.parameters(), self.3.parameters(), self.4.parameters(), self.5.parameters(), self.6.parameters())
    }
}

impl<'a, Input, M0, M1, M2, M3, M4, M5, M6, M7> Module<Input> for &'a (M0, M1, M2, M3, M4, M5, M6, M7)
where
    &'a M0: Module<Input>,
    &'a M1: Module<<&'a M0 as Module<Input>>::Output>,
    &'a M2: Module<<&'a M1 as Module<<&'a M0 as Module<Input>>::Output>>::Output>,
    &'a M3: Module<<&'a M2 as Module<<&'a M1 as Module<<&'a M0 as Module<Input>>::Output>>::Output>>::Output>,
    &'a M4: Module<<&'a M3 as Module<<&'a M2 as Module<<&'a M1 as Module<<&'a M0 as Module<Input>>::Output>>::Output>>::Output>>::Output>,
    &'a M5: Module<<&'a M4 as Module<<&'a M3 as Module<<&'a M2 as Module<<&'a M1 as Module<<&'a M0 as Module<Input>>::Output>>::Output>>::Output>>::Output>>::Output>,
    &'a M6: Module<<&'a M5 as Module<<&'a M4 as Module<<&'a M3 as Module<<&'a M2 as Module<<&'a M1 as Module<<&'a M0 as Module<Input>>::Output>>::Output>>::Output>>::Output>>::Output>>::Output>,
    &'a M7: Module<<&'a M6 as Module<<&'a M5 as Module<<&'a M4 as Module<<&'a M3 as Module<<&'a M2 as Module<<&'a M1 as Module<<&'a M0 as Module<Input>>::Output>>::Output>>::Output>>::Output>>::Output>>::Output>>::Output>,
{
    type Output = <&'a M7 as Module<<&'a M6 as Module<<&'a M5 as Module<<&'a M4 as Module<<&'a M3 as Module<<&'a M2 as Module<<&'a M1 as Module<<&'a M0 as Module<Input>>::Output>>::Output>>::Output>>::Output>>::Output>>::Output>>::Output>>::Output;
    type Params = (
        <&'a M0 as Module<Input>>::Params,
        <&'a M1 as Module<<&'a M0 as Module<Input>>::Output>>::Params,
        <&'a M2 as Module<<&'a M1 as Module<<&'a M0 as Module<Input>>::Output>>::Output>>::Params,
        <&'a M3 as Module<<&'a M2 as Module<<&'a M1 as Module<<&'a M0 as Module<Input>>::Output>>::Output>>::Output>>::Params,
        <&'a M4 as Module<<&'a M3 as Module<<&'a M2 as Module<<&'a M1 as Module<<&'a M0 as Module<Input>>::Output>>::Output>>::Output>>::Output>>::Params,
        <&'a M5 as Module<<&'a M4 as Module<<&'a M3 as Module<<&'a M2 as Module<<&'a M1 as Module<<&'a M0 as Module<Input>>::Output>>::Output>>::Output>>::Output>>::Output>>::Params,
        <&'a M6 as Module<<&'a M5 as Module<<&'a M4 as Module<<&'a M3 as Module<<&'a M2 as Module<<&'a M1 as Module<<&'a M0 as Module<Input>>::Output>>::Output>>::Output>>::Output>>::Output>>::Output>>::Params,
        <&'a M7 as Module<<&'a M6 as Module<<&'a M5 as Module<<&'a M4 as Module<<&'a M3 as Module<<&'a M2 as Module<<&'a M1 as Module<<&'a M0 as Module<Input>>::Output>>::Output>>::Output>>::Output>>::Output>>::Output>>::Output>>::Params,
    );

    fn forward(self, x: Input) -> Self::Output {
        self.7.forward(self.6.forward(self.5.forward(self.4.forward(self.3.forward(self.2.forward(self.1.forward(self.0.forward(x))))))))
    }

    fn parameters(self) -> Self::Params {
        (self.0.parameters(), self.1.parameters(), self.2.parameters(), self.3.parameters(), self.4.parameters(), self.5.parameters(), self.6.parameters(), self.7.parameters())
    }
}

// Arrays of modules are modules (although inputs and outputs must be the same type)
/*impl<Input, M, const N: usize> Module<Input> for [M; N]
where
    M: Module<Input, Output = Input>,
{
    type Output = Input;
    fn forward(self, mut x: Input) -> Self::Output {
        for module in self {
            x = module.forward(x);
        }
        x
    }
}

// Closures are modules
impl<Input, Output, Function> Module<Input> for Function
where
    Function: Fn(Input) -> Output
{
    type Output = Output;
    fn forward(self, x: Input) -> Self::Output {
        self(x)
    }
}*/
