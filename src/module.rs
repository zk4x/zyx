//! [Module] is generic trait that implements only one method: [forward](Module::forward()).
//! It is meant to be implemented mainly for functors and layers.
//!
// This module also contains implementation of [Apply], that is automatically
// implemented for anything that implements [Module] and allows users to use both
// module.forward(input) notation as well as
// input.apply(module) notation
//
// Basically all functions, layers and models should have [module.forward(input)](Module::forward()) function and this module also provides [input.apply(module)](Apply::apply()) function.
// We think it is usefull for the user to have access to both standard [module.forward(input)] type of API and API with monads.
// 

// TODO: use macros to make this DRY

/// # Parameters trait
/// 
/// Implemented for different tuples of [Variables](crate::tensor::Variable).
/// These can then be used by [optimizers](crate::optim).
/// Parameters are just a tuple of mutable references to [Variables](crate::tensor::Variable).
pub trait Parameters {
    /// Zero [Parameter's](Parameters) gradients
    fn zero_grad(&mut self);
}

// Parameters should be implemented differently
// there should be only two functions.
// one should take mutable reference to gradient - for optimizers and zero_grad
// and the other should take mutable reference to data - for optimizers and setters, loading from files, initialization etc.

// The optimizer and loader and function that zeros gradients should be directly implemented for Variable and all tuples of Variables.
// If this can't be done in standard rust, we should create macro for that.
//
// trait SGDStep {
//     fn step<Optim>(&mut self, optim: &SGD);
// }
//
// impl<S> SGDStep for &mut Variable<S>
// where
//     S: Add,
// {
//     fn step(self, optim: &SGD)
//     { todo!() }
// }
//
// This should be automatically derived;
// impl<V1, V2> SGDStep for (V1, V2)
// where
//     V1: SGDStep,
//     V2: SGDStep,
// {
//     fn step(self, optim: &SGD) {
//          self.0.step(optim);
//          self.1.step(optim);
//     }
// }

//impl<X, S1, S2> X for (Variable<S1>, Variable<S2>) where X is implemented for &mut Variable

/// Datatypes implementing this trait can update values of [parameters](Parameters)
/*pub trait ParametersSetter {
    /// Update values of [Parameters]
    fn update_data<S>(&mut self, data: &mut S);
}*/

/*struct UniformDistribution<T> {
    pub low: T,
    pub high: T,
}

impl<T> ParametersSetter for UniformDistribution<T> {
    fn update_data<S>(&mut self, data: &mut S)
    where
        S: UniformInit<T> + crate::ops::GetShape,
    {
        *data = S::uniform(data.shape(), self.low, self.high);
    }
}*/

// If we want to take path in some ParametersSetter, we should take AsRef<Path>

impl Parameters for () {
    fn zero_grad(&mut self) {}
}

impl<Params1, Params2> Parameters for (Params1, Params2)
where
    Params1: Parameters,
    Params2: Parameters,
{
    fn zero_grad(&mut self) {
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
    fn zero_grad(&mut self) {
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
    fn zero_grad(&mut self) {
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
    fn zero_grad(&mut self) {
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
    fn zero_grad(&mut self) {
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
    fn zero_grad(&mut self) {
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
    fn zero_grad(&mut self) {
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
    fn zero_grad(&mut self) {
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
pub trait Module<'p, Input> {
    /// Output of forward operation on [Module]
    type Output;
    /// [Parameters] of [Module]
    type Params: Parameters;
    /// Forward operation on [Module]
    fn forward(&'p self, x: Input) -> Self::Output;
    /// Get parameters of [Module]
    fn parameters(&'p mut self) -> Self::Params;
}

// Apply trait allows us to use monads
/*pub trait Apply<'p, Operation> {
    /// Output of forward operation on [Module]
    type Output;
    /// Forward operation on [Module]
    fn apply(self, x: Operation) -> Self::Output;
}

impl<'p, Input, Operation> Apply<'p, Operation> for Input
where
    Operation: Module<'p, Input>
{
    type Output = <Operation as Module<'p, Input>>::Output;
    fn apply(self, x: Operation) -> Self::Output {
        x.forward(self)
    }
}*/

// Functions are modules
// Layers are modules
// Tuples of modules are modules
impl<'p, Input, M0, M1> Module<'p, Input> for (M0, M1)
where
    M0: Module<'p, Input>,
    M1: Module<'p, <M0 as Module<'p, Input>>::Output>,
{
    type Output = <M1 as Module<'p, <M0 as Module<'p, Input>>::Output>>::Output;
    type Params = (<M0 as Module<'p, Input>>::Params, <M1 as Module<'p, <M0 as Module<'p, Input>>::Output>>::Params);

    fn forward(&'p self, x: Input) -> Self::Output {
        self.1.forward(self.0.forward(x))
    }

    fn parameters(&'p mut self) -> Self::Params {
        (self.0.parameters(), self.1.parameters())
    }
}

impl<'p, Input, M0, M1, M2> Module<'p, Input> for (M0, M1, M2)
where
    M0: Module<'p, Input>,
    M1: Module<'p, <M0 as Module<'p, Input>>::Output>,
    M2: Module<'p, <M1 as Module<'p, <M0 as Module<'p, Input>>::Output>>::Output>,
{
    type Output = <M2 as Module<'p, <M1 as Module<'p, <M0 as Module<'p, Input>>::Output>>::Output>>::Output;
    type Params = (
        <M0 as Module<'p, Input>>::Params,
        <M1 as Module<'p, <M0 as Module<'p, Input>>::Output>>::Params,
        <M2 as Module<'p, <M1 as Module<'p, <M0 as Module<'p, Input>>::Output>>::Output>>::Params,
    );

    fn forward(&'p self, x: Input) -> Self::Output {
        self.2.forward(self.1.forward(self.0.forward(x)))
    }

    fn parameters(&'p mut self) -> Self::Params {
        (self.0.parameters(), self.1.parameters(), self.2.parameters())
    }
}

impl<'p, Input, M0, M1, M2, M3> Module<'p, Input> for (M0, M1, M2, M3)
where
    M0: Module<'p, Input>,
    M1: Module<'p, <M0 as Module<'p, Input>>::Output>,
    M2: Module<'p, <M1 as Module<'p, <M0 as Module<'p, Input>>::Output>>::Output>,
    M3: Module<'p, <M2 as Module<'p, <M1 as Module<'p, <M0 as Module<'p, Input>>::Output>>::Output>>::Output>,
{
    type Output = <M3 as Module<'p, <M2 as Module<'p, <M1 as Module<'p, <M0 as Module<'p, Input>>::Output>>::Output>>::Output>>::Output;
    type Params = (
        <M0 as Module<'p, Input>>::Params,
        <M1 as Module<'p, <M0 as Module<'p, Input>>::Output>>::Params,
        <M2 as Module<'p, <M1 as Module<'p, <M0 as Module<'p, Input>>::Output>>::Output>>::Params,
        <M3 as Module<'p, <M2 as Module<'p, <M1 as Module<'p, <M0 as Module<'p, Input>>::Output>>::Output>>::Output>>::Params,
    );

    fn forward(&'p self, x: Input) -> Self::Output {
        self.3.forward(self.2.forward(self.1.forward(self.0.forward(x))))
    }

    fn parameters(&'p mut self) -> Self::Params {
        (self.0.parameters(), self.1.parameters(), self.2.parameters(), self.3.parameters())
    }
}

impl<'p, Input, M0, M1, M2, M3, M4> Module<'p, Input> for (M0, M1, M2, M3, M4)
where
    M0: Module<'p, Input>,
    M1: Module<'p, <M0 as Module<'p, Input>>::Output>,
    M2: Module<'p, <M1 as Module<'p, <M0 as Module<'p, Input>>::Output>>::Output>,
    M3: Module<'p, <M2 as Module<'p, <M1 as Module<'p, <M0 as Module<'p, Input>>::Output>>::Output>>::Output>,
    M4: Module<'p, <M3 as Module<'p, <M2 as Module<'p, <M1 as Module<'p, <M0 as Module<'p, Input>>::Output>>::Output>>::Output>>::Output>,
{
    type Output = <M4 as Module<'p, <M3 as Module<'p, <M2 as Module<'p, <M1 as Module<'p, <M0 as Module<'p, Input>>::Output>>::Output>>::Output>>::Output>>::Output;
    type Params = (
        <M0 as Module<'p, Input>>::Params,
        <M1 as Module<'p, <M0 as Module<'p, Input>>::Output>>::Params,
        <M2 as Module<'p, <M1 as Module<'p, <M0 as Module<'p, Input>>::Output>>::Output>>::Params,
        <M3 as Module<'p, <M2 as Module<'p, <M1 as Module<'p, <M0 as Module<'p, Input>>::Output>>::Output>>::Output>>::Params,
        <M4 as Module<'p, <M3 as Module<'p, <M2 as Module<'p, <M1 as Module<'p, <M0 as Module<'p, Input>>::Output>>::Output>>::Output>>::Output>>::Params,
    );

    fn forward(&'p self, x: Input) -> Self::Output {
        self.4.forward(self.3.forward(self.2.forward(self.1.forward(self.0.forward(x)))))
    }

    fn parameters(&'p mut self) -> Self::Params {
        (self.0.parameters(), self.1.parameters(), self.2.parameters(), self.3.parameters(), self.4.parameters())
    }
}

impl<'p, Input, M0, M1, M2, M3, M4, M5> Module<'p, Input> for (M0, M1, M2, M3, M4, M5)
where
    M0: Module<'p, Input>,
    M1: Module<'p, <M0 as Module<'p, Input>>::Output>,
    M2: Module<'p, <M1 as Module<'p, <M0 as Module<'p, Input>>::Output>>::Output>,
    M3: Module<'p, <M2 as Module<'p, <M1 as Module<'p, <M0 as Module<'p, Input>>::Output>>::Output>>::Output>,
    M4: Module<'p, <M3 as Module<'p, <M2 as Module<'p, <M1 as Module<'p, <M0 as Module<'p, Input>>::Output>>::Output>>::Output>>::Output>,
    M5: Module<'p, <M4 as Module<'p, <M3 as Module<'p, <M2 as Module<'p, <M1 as Module<'p, <M0 as Module<'p, Input>>::Output>>::Output>>::Output>>::Output>>::Output>,
{
    type Output = <M5 as Module<'p, <M4 as Module<'p, <M3 as Module<'p, <M2 as Module<'p, <M1 as Module<'p, <M0 as Module<'p, Input>>::Output>>::Output>>::Output>>::Output>>::Output>>::Output;
    type Params = (
        <M0 as Module<'p, Input>>::Params,
        <M1 as Module<'p, <M0 as Module<'p, Input>>::Output>>::Params,
        <M2 as Module<'p, <M1 as Module<'p, <M0 as Module<'p, Input>>::Output>>::Output>>::Params,
        <M3 as Module<'p, <M2 as Module<'p, <M1 as Module<'p, <M0 as Module<'p, Input>>::Output>>::Output>>::Output>>::Params,
        <M4 as Module<'p, <M3 as Module<'p, <M2 as Module<'p, <M1 as Module<'p, <M0 as Module<'p, Input>>::Output>>::Output>>::Output>>::Output>>::Params,
        <M5 as Module<'p, <M4 as Module<'p, <M3 as Module<'p, <M2 as Module<'p, <M1 as Module<'p, <M0 as Module<'p, Input>>::Output>>::Output>>::Output>>::Output>>::Output>>::Params,
    );

    fn forward(&'p self, x: Input) -> Self::Output {
        self.5.forward(self.4.forward(self.3.forward(self.2.forward(self.1.forward(self.0.forward(x))))))
    }

    fn parameters(&'p mut self) -> Self::Params {
        (self.0.parameters(), self.1.parameters(), self.2.parameters(), self.3.parameters(), self.4.parameters(), self.5.parameters())
    }
}

impl<'p, Input, M0, M1, M2, M3, M4, M5, M6> Module<'p, Input> for (M0, M1, M2, M3, M4, M5, M6)
where
    M0: Module<'p, Input>,
    M1: Module<'p, <M0 as Module<'p, Input>>::Output>,
    M2: Module<'p, <M1 as Module<'p, <M0 as Module<'p, Input>>::Output>>::Output>,
    M3: Module<'p, <M2 as Module<'p, <M1 as Module<'p, <M0 as Module<'p, Input>>::Output>>::Output>>::Output>,
    M4: Module<'p, <M3 as Module<'p, <M2 as Module<'p, <M1 as Module<'p, <M0 as Module<'p, Input>>::Output>>::Output>>::Output>>::Output>,
    M5: Module<'p, <M4 as Module<'p, <M3 as Module<'p, <M2 as Module<'p, <M1 as Module<'p, <M0 as Module<'p, Input>>::Output>>::Output>>::Output>>::Output>>::Output>,
    M6: Module<'p, <M5 as Module<'p, <M4 as Module<'p, <M3 as Module<'p, <M2 as Module<'p, <M1 as Module<'p, <M0 as Module<'p, Input>>::Output>>::Output>>::Output>>::Output>>::Output>>::Output>,
{
    type Output = <M6 as Module<'p, <M5 as Module<'p, <M4 as Module<'p, <M3 as Module<'p, <M2 as Module<'p, <M1 as Module<'p, <M0 as Module<'p, Input>>::Output>>::Output>>::Output>>::Output>>::Output>>::Output>>::Output;
    type Params = (
        <M0 as Module<'p, Input>>::Params,
        <M1 as Module<'p, <M0 as Module<'p, Input>>::Output>>::Params,
        <M2 as Module<'p, <M1 as Module<'p, <M0 as Module<'p, Input>>::Output>>::Output>>::Params,
        <M3 as Module<'p, <M2 as Module<'p, <M1 as Module<'p, <M0 as Module<'p, Input>>::Output>>::Output>>::Output>>::Params,
        <M4 as Module<'p, <M3 as Module<'p, <M2 as Module<'p, <M1 as Module<'p, <M0 as Module<'p, Input>>::Output>>::Output>>::Output>>::Output>>::Params,
        <M5 as Module<'p, <M4 as Module<'p, <M3 as Module<'p, <M2 as Module<'p, <M1 as Module<'p, <M0 as Module<'p, Input>>::Output>>::Output>>::Output>>::Output>>::Output>>::Params,
        <M6 as Module<'p, <M5 as Module<'p, <M4 as Module<'p, <M3 as Module<'p, <M2 as Module<'p, <M1 as Module<'p, <M0 as Module<'p, Input>>::Output>>::Output>>::Output>>::Output>>::Output>>::Output>>::Params,
    );

    fn forward(&'p self, x: Input) -> Self::Output {
        self.6.forward(self.5.forward(self.4.forward(self.3.forward(self.2.forward(self.1.forward(self.0.forward(x)))))))
    }

    fn parameters(&'p mut self) -> Self::Params {
        (self.0.parameters(), self.1.parameters(), self.2.parameters(), self.3.parameters(), self.4.parameters(), self.5.parameters(), self.6.parameters())
    }
}

impl<'p, Input, M0, M1, M2, M3, M4, M5, M6, M7> Module<'p, Input> for (M0, M1, M2, M3, M4, M5, M6, M7)
where
    M0: Module<'p, Input>,
    M1: Module<'p, <M0 as Module<'p, Input>>::Output>,
    M2: Module<'p, <M1 as Module<'p, <M0 as Module<'p, Input>>::Output>>::Output>,
    M3: Module<'p, <M2 as Module<'p, <M1 as Module<'p, <M0 as Module<'p, Input>>::Output>>::Output>>::Output>,
    M4: Module<'p, <M3 as Module<'p, <M2 as Module<'p, <M1 as Module<'p, <M0 as Module<'p, Input>>::Output>>::Output>>::Output>>::Output>,
    M5: Module<'p, <M4 as Module<'p, <M3 as Module<'p, <M2 as Module<'p, <M1 as Module<'p, <M0 as Module<'p, Input>>::Output>>::Output>>::Output>>::Output>>::Output>,
    M6: Module<'p, <M5 as Module<'p, <M4 as Module<'p, <M3 as Module<'p, <M2 as Module<'p, <M1 as Module<'p, <M0 as Module<'p, Input>>::Output>>::Output>>::Output>>::Output>>::Output>>::Output>,
    M7: Module<'p, <M6 as Module<'p, <M5 as Module<'p, <M4 as Module<'p, <M3 as Module<'p, <M2 as Module<'p, <M1 as Module<'p, <M0 as Module<'p, Input>>::Output>>::Output>>::Output>>::Output>>::Output>>::Output>>::Output>,
{
    type Output = <M7 as Module<'p, <M6 as Module<'p, <M5 as Module<'p, <M4 as Module<'p, <M3 as Module<'p, <M2 as Module<'p, <M1 as Module<'p, <M0 as Module<'p, Input>>::Output>>::Output>>::Output>>::Output>>::Output>>::Output>>::Output>>::Output;
    type Params = (
        <M0 as Module<'p, Input>>::Params,
        <M1 as Module<'p, <M0 as Module<'p, Input>>::Output>>::Params,
        <M2 as Module<'p, <M1 as Module<'p, <M0 as Module<'p, Input>>::Output>>::Output>>::Params,
        <M3 as Module<'p, <M2 as Module<'p, <M1 as Module<'p, <M0 as Module<'p, Input>>::Output>>::Output>>::Output>>::Params,
        <M4 as Module<'p, <M3 as Module<'p, <M2 as Module<'p, <M1 as Module<'p, <M0 as Module<'p, Input>>::Output>>::Output>>::Output>>::Output>>::Params,
        <M5 as Module<'p, <M4 as Module<'p, <M3 as Module<'p, <M2 as Module<'p, <M1 as Module<'p, <M0 as Module<'p, Input>>::Output>>::Output>>::Output>>::Output>>::Output>>::Params,
        <M6 as Module<'p, <M5 as Module<'p, <M4 as Module<'p, <M3 as Module<'p, <M2 as Module<'p, <M1 as Module<'p, <M0 as Module<'p, Input>>::Output>>::Output>>::Output>>::Output>>::Output>>::Output>>::Params,
        <M7 as Module<'p, <M6 as Module<'p, <M5 as Module<'p, <M4 as Module<'p, <M3 as Module<'p, <M2 as Module<'p, <M1 as Module<'p, <M0 as Module<'p, Input>>::Output>>::Output>>::Output>>::Output>>::Output>>::Output>>::Output>>::Params,
    );

    fn forward(&'p self, x: Input) -> Self::Output {
        self.7.forward(self.6.forward(self.5.forward(self.4.forward(self.3.forward(self.2.forward(self.1.forward(self.0.forward(x))))))))
    }

    fn parameters(&'p mut self) -> Self::Params {
        (self.0.parameters(), self.1.parameters(), self.2.parameters(), self.3.parameters(), self.4.parameters(), self.5.parameters(), self.6.parameters(), self.7.parameters())
    }
}

// Closures are modules
//
// But they can not have any parameters. And this must be assured by the user.
// So for now we won't use this, as it does not look like a good thing (we want to at least warn user, if he does something shady).
/*impl<Input, Output, Function> Module<'_, Input> for Function
where
    Function: Fn(Input) -> Output
{
    type Output = Output;
    type Params = ();

    fn forward(&self, x: Input) -> Self::Output {
        self(x)
    }

    fn parameters(&mut self) -> Self::Params {}
}*/

// TODO: Should arrays of modules be modules?
// Arrays of modules are modules (although inputs and outputs must be the same type)
/*impl<Input, M, const N: usize> Module<'p, Input> for [M; N]
where
    M: Module<'p, Input, Output = Input>,
{
    type Output = Input;
    fn forward(self, mut x: Input) -> Self::Output {
        for module in self {
            x = module.forward(x);
        }
        x
    }
}*/
