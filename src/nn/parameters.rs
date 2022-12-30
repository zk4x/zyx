//! Parameters module

/// # HasParameters
pub trait HasParameters<'p> {
    /// [Parameters](crate::nn::parameters::Parameters) of [Module](crate::nn::Module)
    type Params: super::parameters::Parameters;
    /// Get parameters of [Module](crate::nn::Module)
    fn parameters(&'p mut self) -> Self::Params;
}

impl<'p, M0, M1> HasParameters<'p> for (M0, M1)
where
    M0: HasParameters<'p>,
    M1: HasParameters<'p>,
{
    type Params = (M0::Params, M1::Params);
    fn parameters(&'p mut self) -> Self::Params {
        (
            self.0.parameters(),
            self.1.parameters(),
        )
    }
}

impl<'p, M0, M1, M2> HasParameters<'p> for (M0, M1, M2)
where
    M0: HasParameters<'p>,
    M1: HasParameters<'p>,
    M2: HasParameters<'p>,
{
    type Params = (M0::Params, M1::Params, M2::Params);
    fn parameters(&'p mut self) -> Self::Params {
        (
            self.0.parameters(),
            self.1.parameters(),
            self.2.parameters(),
        )
    }
}

impl<'p, M0, M1, M2, M3> HasParameters<'p> for (M0, M1, M2, M3)
where
    M0: HasParameters<'p>,
    M1: HasParameters<'p>,
    M2: HasParameters<'p>,
    M3: HasParameters<'p>,
{
    type Params = (M0::Params, M1::Params, M2::Params, M3::Params);
    fn parameters(&'p mut self) -> Self::Params {
        (
            self.0.parameters(),
            self.1.parameters(),
            self.2.parameters(),
            self.3.parameters(),
        )
    }
}

impl<'p, M0, M1, M2, M3, M4> HasParameters<'p> for (M0, M1, M2, M3, M4)
where
    M0: HasParameters<'p>,
    M1: HasParameters<'p>,
    M2: HasParameters<'p>,
    M3: HasParameters<'p>,
    M4: HasParameters<'p>,
{
    type Params = (M0::Params, M1::Params, M2::Params, M3::Params, M4::Params);
    fn parameters(&'p mut self) -> Self::Params {
        (
            self.0.parameters(),
            self.1.parameters(),
            self.2.parameters(),
            self.3.parameters(),
            self.4.parameters(),
        )
    }
}

impl<'p, M0, M1, M2, M3, M4, M5> HasParameters<'p> for (M0, M1, M2, M3, M4, M5)
where
    M0: HasParameters<'p>,
    M1: HasParameters<'p>,
    M2: HasParameters<'p>,
    M3: HasParameters<'p>,
    M4: HasParameters<'p>,
    M5: HasParameters<'p>,
{
    type Params = (M0::Params, M1::Params, M2::Params, M3::Params, M4::Params, M5::Params);
    fn parameters(&'p mut self) -> Self::Params {
        (
            self.0.parameters(),
            self.1.parameters(),
            self.2.parameters(),
            self.3.parameters(),
            self.4.parameters(),
            self.5.parameters(),
        )
    }
}

impl<'p, M0, M1, M2, M3, M4, M5, M6> HasParameters<'p> for (M0, M1, M2, M3, M4, M5, M6)
where
    M0: HasParameters<'p>,
    M1: HasParameters<'p>,
    M2: HasParameters<'p>,
    M3: HasParameters<'p>,
    M4: HasParameters<'p>,
    M5: HasParameters<'p>,
    M6: HasParameters<'p>,
{
    type Params = (M0::Params, M1::Params, M2::Params, M3::Params, M4::Params, M5::Params, M6::Params);
    fn parameters(&'p mut self) -> Self::Params {
        (
            self.0.parameters(),
            self.1.parameters(),
            self.2.parameters(),
            self.3.parameters(),
            self.4.parameters(),
            self.5.parameters(),
            self.6.parameters(),
        )
    }
}

impl<'p, M0, M1, M2, M3, M4, M5, M6, M7> HasParameters<'p> for (M0, M1, M2, M3, M4, M5, M6, M7)
where
    M0: HasParameters<'p>,
    M1: HasParameters<'p>,
    M2: HasParameters<'p>,
    M3: HasParameters<'p>,
    M4: HasParameters<'p>,
    M5: HasParameters<'p>,
    M6: HasParameters<'p>,
    M7: HasParameters<'p>,
{
    type Params = (M0::Params, M1::Params, M2::Params, M3::Params, M4::Params, M5::Params, M6::Params, M7::Params);
    fn parameters(&'p mut self) -> Self::Params {
        (
            self.0.parameters(),
            self.1.parameters(),
            self.2.parameters(),
            self.3.parameters(),
            self.4.parameters(),
            self.5.parameters(),
            self.6.parameters(),
            self.7.parameters(),
        )
    }
}

impl<'p, M0, M1, M2, M3, M4, M5, M6, M7, M8> HasParameters<'p> for (M0, M1, M2, M3, M4, M5, M6, M7, M8)
where
    M0: HasParameters<'p>,
    M1: HasParameters<'p>,
    M2: HasParameters<'p>,
    M3: HasParameters<'p>,
    M4: HasParameters<'p>,
    M5: HasParameters<'p>,
    M6: HasParameters<'p>,
    M7: HasParameters<'p>,
    M8: HasParameters<'p>,
{
    type Params = (M0::Params, M1::Params, M2::Params, M3::Params, M4::Params, M5::Params, M6::Params, M7::Params, M8::Params);
    fn parameters(&'p mut self) -> Self::Params {
        (
            self.0.parameters(),
            self.1.parameters(),
            self.2.parameters(),
            self.3.parameters(),
            self.4.parameters(),
            self.5.parameters(),
            self.6.parameters(),
            self.7.parameters(),
            self.8.parameters(),
        )
    }
}

impl<'p, M0, M1, M2, M3, M4, M5, M6, M7, M8, M9> HasParameters<'p> for (M0, M1, M2, M3, M4, M5, M6, M7, M8, M9)
where
    M0: HasParameters<'p>,
    M1: HasParameters<'p>,
    M2: HasParameters<'p>,
    M3: HasParameters<'p>,
    M4: HasParameters<'p>,
    M5: HasParameters<'p>,
    M6: HasParameters<'p>,
    M7: HasParameters<'p>,
    M8: HasParameters<'p>,
    M9: HasParameters<'p>,
{
    type Params = (M0::Params, M1::Params, M2::Params, M3::Params, M4::Params, M5::Params, M6::Params, M7::Params, M8::Params, M9::Params);
    fn parameters(&'p mut self) -> Self::Params {
        (
            self.0.parameters(),
            self.1.parameters(),
            self.2.parameters(),
            self.3.parameters(),
            self.4.parameters(),
            self.5.parameters(),
            self.6.parameters(),
            self.7.parameters(),
            self.8.parameters(),
            self.9.parameters(),
        )
    }
}

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

impl<Params0, Params1, Params2, Params3, Params4, Params5, Params6, Params7, Params8, Params9> Parameters for (Params0, Params1, Params2, Params3, Params4, Params5, Params6, Params7, Params8, Params9)
where
    Params0: Parameters,
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
        self.9.zero_grad();
    }
}