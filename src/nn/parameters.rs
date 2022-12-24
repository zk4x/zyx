//! Parameters module

/// # Parameters trait
/// 
/// Implemented for different tuples of [Variables](crate::tensor::Variable).
/// These can then be used by [optimizers](crate::optim).
/// Parameters are just a tuple of mutable references to [Variables](crate::tensor::Variable).
pub trait Parameters {
    /// Zero [Parameter's](Parameters) gradients
    fn zero_grad(&mut self);
}

// TODO: use macros to make this DRY

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
        S: UniformInit<T> + crate::ops::HasShape,
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