//! Module is generic trait that implements only one method: forward.
//! It can be implemented for anything.
//! This module also contains implementation of Apply trait, that is automatically
//! implemented for anything that implements Module and allows users to use both
//! module.forward(input) notation as well as
//! input.apply(module) notation
//! 

pub trait Module<Input> {
    type Output;
    fn forward(self, x: Input) -> Self::Output;
}

pub trait Apply<Operation> {
    type Output;
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

// TODO write these in similar way as the previous ones:
/*impl<Input, M0, M1, M2, M3> Module<Input> for (M0, M1, M2, M3)
where
    M0: Module<Input>,
    M1: Module<M0::Output>,
    M2: Module<M1::Output>,
    M3: Module<M2::Output>,
{
    type Output = M3::Output;
    fn forward(self, x: Input) -> Self::Output {
        x.apply(self.0).apply(self.1).apply(self.2).apply(self.3)
    }
}

impl<Input, M0, M1, M2, M3, M4> Module<Input> for (M0, M1, M2, M3, M4)
where
    M0: Module<Input>,
    M1: Module<M0::Output>,
    M2: Module<M1::Output>,
    M3: Module<M2::Output>,
    M4: Module<M3::Output>,
{
    type Output = M4::Output;
    fn forward(self, x: Input) -> Self::Output {
        x.apply(self.0).apply(self.1).apply(self.2).apply(self.3).apply(self.4)
    }
}*/

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
