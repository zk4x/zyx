//! [Module] is generic trait that implements only one method: [forward](Module::forward()).
//! It is meant to be implemented mainly for functors and layers.
//!
//! This module also contains implementation of [ApplyModule], that is automatically
//! implemented for anything that implements [Module] and allows users to use both
//! module.forward(input) notation as well as
//! input.apply(module) notation
//!
//! Basically all functions, layers and models should have [module.forward(input)](Module::forward()) function and this module also provides [input.apply(module)](ApplyModule::apply()) function.
//! We think it is usefull to have access to both standard [module.forward(input)] type of API and API with monads.
//!

// TODO: use macros to make this DRY

/// # Module trait
///
/// Module can be implemented for anything that can have forward function.
/// Forward simply takes any input, applies some operation and returns output.
/// Every module also has some or zero [parameters](super::parameters::Parameters) that can be optimized
/// by [optimizers](crate::optim).
pub trait Module<'p, Input> {
    // TODO 'p is probably not needed anymore, because Device has it's lifetime parameter, although is it needed for deviceless datatypes?
    /// Output of forward operation on [Module]
    type Output;
    /// Forward operation on [Module]
    fn forward(&'p self, x: Input) -> Self::Output;
}

/// Apply trait allows us to use monads
pub trait ApplyModule<'p, Operation> {
    /// Output of forward operation on [Module]
    type Output;
    /// Forward operation on [Module]
    fn apply(self, x: &'p Operation) -> Self::Output;
}

impl<'p, Input, Operation> ApplyModule<'p, Operation> for Input
where
    Operation: Module<'p, Input>,
{
    type Output = <Operation as Module<'p, Input>>::Output;
    fn apply(self, x: &'p Operation) -> Self::Output {
        x.forward(self)
    }
}

// Functions are modules
// Layers are modules
// Tuples of modules are modules
impl<'p, Input, M0, M1> Module<'p, Input> for (M0, M1)
where
    M0: Module<'p, Input>,
    M1: Module<'p, <M0 as Module<'p, Input>>::Output>,
{
    type Output = <M1 as Module<'p, <M0 as Module<'p, Input>>::Output>>::Output;
    fn forward(&'p self, x: Input) -> Self::Output {
        self.1.forward(self.0.forward(x))
    }
}

impl<'p, Input, M0, M1, M2> Module<'p, Input> for (M0, M1, M2)
where
    M0: Module<'p, Input>,
    M1: Module<'p, <M0 as Module<'p, Input>>::Output>,
    M2: Module<'p, <M1 as Module<'p, <M0 as Module<'p, Input>>::Output>>::Output>,
{
    type Output =
        <M2 as Module<'p, <M1 as Module<'p, <M0 as Module<'p, Input>>::Output>>::Output>>::Output;
    fn forward(&'p self, x: Input) -> Self::Output {
        self.2.forward(self.1.forward(self.0.forward(x)))
    }
}

impl<'p, Input, M0, M1, M2, M3> Module<'p, Input> for (M0, M1, M2, M3)
where
    M0: Module<'p, Input>,
    M1: Module<'p, <M0 as Module<'p, Input>>::Output>,
    M2: Module<'p, <M1 as Module<'p, <M0 as Module<'p, Input>>::Output>>::Output>,
    M3: Module<
        'p,
        <M2 as Module<'p, <M1 as Module<'p, <M0 as Module<'p, Input>>::Output>>::Output>>::Output,
    >,
{
    type Output = <M3 as Module<
        'p,
        <M2 as Module<'p, <M1 as Module<'p, <M0 as Module<'p, Input>>::Output>>::Output>>::Output,
    >>::Output;
    fn forward(&'p self, x: Input) -> Self::Output {
        self.3
            .forward(self.2.forward(self.1.forward(self.0.forward(x))))
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
    fn forward(&'p self, x: Input) -> Self::Output {
        self.4.forward(self.3.forward(self.2.forward(self.1.forward(self.0.forward(x)))))
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
    fn forward(&'p self, x: Input) -> Self::Output {
        self.5.forward(self.4.forward(self.3.forward(self.2.forward(self.1.forward(self.0.forward(x))))))
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
    fn forward(&'p self, x: Input) -> Self::Output {
        self.6.forward(self.5.forward(self.4.forward(self.3.forward(self.2.forward(self.1.forward(self.0.forward(x)))))))
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
    fn forward(&'p self, x: Input) -> Self::Output {
        self.7.forward(self.6.forward(self.5.forward(self.4.forward(self.3.forward(self.2.forward(self.1.forward(self.0.forward(x))))))))
    }
}

// TODO: Should closures be modules?
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
