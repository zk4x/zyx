//! Module is generic trait that implements only one method: forward.
//! It can be implemented for anything.
//! This module also contains implementation of Apply trait, that is automatically
//! implemented for anything that implements Module and allows users to use both
//! module.forward(input) notation as well as
//! input.apply(module) notation
//! 

use crate::tensor::Variable;

/// # ModuleParams trait
/// 
/// Every module that stores some parameters should implement this trait.
/// Also every module that can be used by optimizers must implement this trait.
/// Thus this trait is implemented by most thing in zyx::nn, such as layers and functors.
pub trait ModuleParams<'a, S> {
    /// Get parameters of module
    fn parameters(&'a self) -> Vec<&'a Variable<S>>;
}

/// # Module trait
/// 
/// Module can be implemented for anything that can have forward function.
/// Forward simply takes any input, applies some operation and returns output.
pub trait Module<Input> {
    /// Output of forward operation on Module
    type Output;
    /// Forward operation on Module
    fn forward(self, x: Input) -> Self::Output;
}

/// Apply trait allows us to use monads
pub trait Apply<Operation> {
    /// Output of forward operation on Module
    type Output;
    /// Forward operation on Module
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

// Modules are modules

// Functions are modules

// Layers are modules

// Tuples of modules are modules
impl<'a, Input, M0, M1> Module<Input> for &'a (M0, M1)
where
    &'a M0: Module<Input>,
    &'a M1: Module<<&'a M0 as Module<Input>>::Output>,
{
    type Output = <&'a M1 as Module<<&'a M0 as Module<Input>>::Output>>::Output;
    fn forward(self, x: Input) -> Self::Output {
        self.1.forward(self.0.forward(x))
    }
}

impl<'a, S, M0, M1> ModuleParams<'a, S> for (M0, M1)
where
    M0: ModuleParams<'a, S>,
    M1: ModuleParams<'a, S>,
{
    fn parameters(&'a self) -> Vec<&'a Variable<S>> {
        self.0.parameters().into_iter()
            .chain(self.1.parameters().into_iter()).collect()
    }
}

impl<'a, Input, M0, M1, M2> Module<Input> for &'a (M0, M1, M2)
where
    &'a M0: Module<Input>,
    &'a M1: Module<<&'a M0 as Module<Input>>::Output>,
    &'a M2: Module<<&'a M1 as Module<<&'a M0 as Module<Input>>::Output>>::Output>,
{
    type Output = <&'a M2 as Module<<&'a M1 as Module<<&'a M0 as Module<Input>>::Output>>::Output>>::Output;
    fn forward(self, x: Input) -> Self::Output {
        self.2.forward(self.1.forward(self.0.forward(x)))
    }
}

impl<'a, S, M0, M1, M2> ModuleParams<'a, S> for (M0, M1, M2)
where
    M0: ModuleParams<'a, S>,
    M1: ModuleParams<'a, S>,
    M2: ModuleParams<'a, S>,
{
    fn parameters(&'a self) -> Vec<&'a Variable<S>> {
        self.0.parameters().into_iter()
            .chain(self.1.parameters().into_iter())
            .chain(self.2.parameters().into_iter()).collect()
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
    fn forward(self, x: Input) -> Self::Output {
        self.3.forward(self.2.forward(self.1.forward(self.0.forward(x))))
    }
}

impl<'a, S, M0, M1, M2, M3> ModuleParams<'a, S> for (M0, M1, M2, M3)
where
    M0: ModuleParams<'a, S>,
    M1: ModuleParams<'a, S>,
    M2: ModuleParams<'a, S>,
    M3: ModuleParams<'a, S>,
{
    fn parameters(&'a self) -> Vec<&'a Variable<S>> {
        self.0.parameters().into_iter()
            .chain(self.1.parameters().into_iter())
            .chain(self.2.parameters().into_iter())
            .chain(self.3.parameters().into_iter()).collect()
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
    fn forward(self, x: Input) -> Self::Output {
        self.4.forward(self.3.forward(self.2.forward(self.1.forward(self.0.forward(x)))))
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
    fn forward(self, x: Input) -> Self::Output {
        self.5.forward(self.4.forward(self.3.forward(self.2.forward(self.1.forward(self.0.forward(x))))))
    }
}

impl<'a, S, M0, M1, M2, M3, M4, M5> ModuleParams<'a, S> for (M0, M1, M2, M3, M4, M5)
where
    M0: ModuleParams<'a, S>,
    M1: ModuleParams<'a, S>,
    M2: ModuleParams<'a, S>,
    M3: ModuleParams<'a, S>,
    M4: ModuleParams<'a, S>,
    M5: ModuleParams<'a, S>,
{
    fn parameters(&'a self) -> Vec<&'a Variable<S>> {
        self.0.parameters().into_iter()
            .chain(self.1.parameters().into_iter())
            .chain(self.2.parameters().into_iter())
            .chain(self.3.parameters().into_iter())
            .chain(self.4.parameters().into_iter())
            .chain(self.5.parameters().into_iter()).collect()
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
    fn forward(self, x: Input) -> Self::Output {
        self.6.forward(self.5.forward(self.4.forward(self.3.forward(self.2.forward(self.1.forward(self.0.forward(x)))))))
    }
}

impl<'a, S, M0, M1, M2, M3, M4, M5, M6> ModuleParams<'a, S> for (M0, M1, M2, M3, M4, M5, M6)
where
    M0: ModuleParams<'a, S>,
    M1: ModuleParams<'a, S>,
    M2: ModuleParams<'a, S>,
    M3: ModuleParams<'a, S>,
    M4: ModuleParams<'a, S>,
    M5: ModuleParams<'a, S>,
    M6: ModuleParams<'a, S>,
{
    fn parameters(&'a self) -> Vec<&'a Variable<S>> {
        self.0.parameters().into_iter()
            .chain(self.1.parameters().into_iter())
            .chain(self.2.parameters().into_iter())
            .chain(self.3.parameters().into_iter())
            .chain(self.4.parameters().into_iter())
            .chain(self.5.parameters().into_iter())
            .chain(self.6.parameters().into_iter()).collect()
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
    fn forward(self, x: Input) -> Self::Output {
        self.7.forward(self.6.forward(self.5.forward(self.4.forward(self.3.forward(self.2.forward(self.1.forward(self.0.forward(x))))))))
    }
}

impl<'a, S, M0, M1, M2, M3, M4, M5, M6, M7> ModuleParams<'a, S> for (M0, M1, M2, M3, M4, M5, M6, M7)
where
    M0: ModuleParams<'a, S>,
    M1: ModuleParams<'a, S>,
    M2: ModuleParams<'a, S>,
    M3: ModuleParams<'a, S>,
    M4: ModuleParams<'a, S>,
    M5: ModuleParams<'a, S>,
    M6: ModuleParams<'a, S>,
    M7: ModuleParams<'a, S>,
{
    fn parameters(&'a self) -> Vec<&'a Variable<S>> {
        self.0.parameters().into_iter()
            .chain(self.1.parameters().into_iter())
            .chain(self.2.parameters().into_iter())
            .chain(self.3.parameters().into_iter())
            .chain(self.4.parameters().into_iter())
            .chain(self.5.parameters().into_iter())
            .chain(self.6.parameters().into_iter())
            .chain(self.7.parameters().into_iter()).collect()
    }
}

// Arrays of modules are modules (although inputs and outputs must be the same type)
impl<Input, M0, const N: usize> Module<Input> for [M0; N]
where
    M0: Module<Input, Output = Input>,
{
    type Output = Input;
    fn forward(self, mut x: Input) -> Self::Output {
        for module in self {
            x = module.forward(x);
        }
        x
    }
}

impl<Input, Output, Function> Module<Input> for Function
where
    Function: Fn(Input) -> Output
{
    type Output = Output;
    fn forward(self, x: Input) -> Self::Output {
        self(x)
    }
}
