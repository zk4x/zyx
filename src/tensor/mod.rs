//! Description of generic tensor types. All tensors are immutable!
//! Mutability is allowed only for calculating gradients and optimizer parameters.
//!
//! There are two tensor types:
//! > 1. [Variable]   - stores datatype and also it's gradient, passed around by reference
//! > 2. [Tensor]     - stores datatype and functions necessary to calculate gradient of [Variable], passed around by cloning
//!
//! Buffer and [Variable] are leaf tensors. [Tensor] is strictly non-leaf and therefore it doesn't store it's gradient.
//!
//! # Example
//!
//! Basic buffer
//! ```
//! # use zyx::prelude::*;
//! # use zyx::{device::cpu::{self, Buffer}, tensor::{Variable, Tensor}, shape::Sh1};
//! let mut device = cpu::Device::default();
//! let x: Buffer<Sh1<3>, f32> = device.buffer([2., 1., 4.]);
//! ```
//!
//! By calling [.with_grad()](crate::ops::IntoVariable::with_grad()) on any datatype, you get [Variable], which adds gradient to the buffer.
//! ```
//! # use zyx::prelude::*;
//! # use zyx::{device::cpu::{self, Buffer}, tensor::{Variable, Tensor}, shape::Sh1};
//! # let mut device = cpu::Device::default();
//! # let x: Buffer<Sh1<3>, f32> = device.buffer([2., 1., 4.]);
//! let y: Variable<Buffer<Sh1<3>, f32>> = x.clone().with_grad();
//! ```
//!
//! Applying function to [Variable] returns [Tensor].
//! [Tensor] stores references to [Variable's](Variable) gradients and some data buffers used during gradient calculation.
//! ```
//! # use zyx::prelude::*;
//! # use zyx::{device::cpu::{self, Buffer}, tensor::{Variable, Tensor}, shape::Sh1};
//! # let mut device = cpu::Device::default();
//! # let x: Buffer<Sh1<3>, f32> = device.buffer([2., 1., 4.]);
//! # let y: Variable<Buffer<Sh1<3>, f32>> = x.clone().with_grad();
//! let z: Tensor<Buffer<Sh1<3>, f32>, _/* ReLUBackwardT<_> */> = y.relu();
//! ```
//!
//! Applying function to buffer simply returns buffer.
//! ```
//! # use zyx::prelude::*;
//! # use zyx::{device::cpu::{self, Buffer}, tensor::{Variable, Tensor}, shape::Sh1};
//! # let mut device = cpu::Device::default();
//! # let x: Buffer<Sh1<3>, f32> = device.buffer([2., 1., 4.]);
//! let z: Buffer<Sh1<3>, f32> = x.relu();
//! ```
//!

// This file contains tensor definitions, getters and setters for tensors.

mod ops;

// How this works (for contributors)
//
// The Buffer and Variable are leaf tensors. Buffer does not have grad, while Variable does (obviously).
// Tensor is non leaf tensor.
// When an operation is performed with Buffer, a new Buffer is returned with result. Nothing magical happens.
// When an operation is performed with Variable, a new Tensor is returned that holds weak pointer
// to Variable's gradient. This Tensor contains a closure, that hold's this pointer and calculates
// the gradient when backward is called on it.
// We can imagine neural network as a tree, where leafs are Buffers and Variables and root/roots
// are Tensor. When an operation is performed with Tensor, it's consumed and it's func is moved to the resulting
// Tensor. So the last Tensor is the root of the tree and is the only Tensor in existence
// and it holds all the closures with &RefCell<S> pointers to the Variable's gradients.
// If you want to have more the one Tensor to call .backward() on, you need to clone this Tensor
// or any of the intermediate Tensors. In this case, the library performs cloning of the closures.

// Tensors are moved into operations, while Variables are passed by reference!

/// # Variable
/// 
/// Variable holds data and it's gradient.
#[derive(Debug, Clone, Default)]
pub struct Variable<S> {
    pub(crate) data: S,
    // Gradient has the same type and shape as data
    grad: Gradient<S>,
}

/// Gradient of [Variable]
/// 
/// User has read-only access to [Gradient] with to_vec and to_string.
// This is using UnsafeCell to store value inside gradient.
// Since UnsafeCell is not Sync, there is just few times that we access gradient,
// so it is not difficult to make it safe to use.
// The only mutable accesses are [Self::zero], which does not use unsafe
// and accumulate, which uses unsafe, since [Variable] can be passed into multiple
// functions and all of them need to be able to accumulate gradient.
// So we just make sure that backward can not be called on buffer that is borrowed
// and that is it.
#[derive(Default, Debug)]
pub struct Gradient<S>(core::cell::UnsafeCell<S>);

impl<S> Clone for Gradient<S>
where
    S: Clone,
{
    fn clone(&self) -> Self {
        // Safe, read only access
        unsafe { Self(core::cell::UnsafeCell::new((*self.0.get()).clone())) }
    }
}

impl<G> Gradient<G> {
    fn new(data: G) -> Self {
        Self(core::cell::UnsafeCell::new(data))
    }

    /// Get value stored inside of the gradient
    pub fn buffer(&self) -> &G {
        unsafe { &*self.0.get() }
    }

    fn zero(&mut self)
    where
        G: crate::ops::ZerosLike,
    {
        self.0.get_mut().zeros_like();
    }
}

trait GradAcc<G>: core::ops::Add<G, Output = Self> + Clone {}
impl<G, T> GradAcc<G> for T where T: core::ops::Add<G, Output = Self> + Clone {}

#[derive(Debug, Clone, Copy)]
struct GradientRef<'g, S>(&'g Gradient<S>);

impl<'g, S> GradientRef<'g, S> {
    fn new(gradient: &'g Gradient<S>) -> Self {
        Self(gradient)
    }

    fn accumulate<G>(&self, value: G)
    where
        S: GradAcc<G>,
    {
        // Accumulate is called by backward function to accumulate gradients. This is needed in batch processing.
        // Unsafe is needed, because we need multiple functions accessing the same gradient.
        unsafe { *self.0.0.get() = (*self.0.0.get()).clone() + value; }
    }
}

/*impl<S, Rhs> PartialEq<Rhs> for Gradient<S>
where
   S: PartialEq<Rhs>,
{
    fn eq(&self, rhs: &Rhs) -> bool {
        unsafe { &*self.0.get() == rhs }
    }
}*/

impl<G> core::fmt::Display for Gradient<G>
where
    G: core::fmt::Display,
{
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        extern crate alloc;
        use alloc::string::ToString;
        // This is save, beacause it is read only access
        let grad = unsafe { &*self.0.get() };
        f.write_str(&grad.to_string())
    }
}

/// # Tensor
/// 
/// Tensor holds data and grad_fn to calculate gradients of [Variables](Variable).
/// Tensor is only created as a result of some operations on at least one [Variable].
/// Tensor does not store it's gradient, but the gradient can be accessed during backward
/// pass by using [GradHookT].
#[derive(Debug, Clone, Default, PartialEq, Eq, PartialOrd, Ord)]
pub struct Tensor<S, GradFn> {
    data: S,
    grad_fn: GradFn,
}

/// # Display Variable
/// 
/// Shows [Variable] and it's gradient.
impl<S> core::fmt::Display for Variable<S>
where
    S: core::fmt::Display,
{
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        extern crate alloc;
        use alloc::format;
        f.write_str(&format!(
            "{} with grad:\n{}",
            self.data,
            &self.grad,
        ))
    }
}

/// # Display Tensor
/// 
/// Shows [Tensor] and it's grad_fn.
impl<S, GradFn> core::fmt::Display for Tensor<S, GradFn>
where
    S: core::fmt::Display,
    GradFn: core::fmt::Debug,
{
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        // TODO: this is ugly solution, please just print the name of the function, don't require GradFn debug
        // and creating debug string of all the buffers stored in grad_fn
        extern crate alloc;
        use alloc::format;
        f.write_str(&format!("{} with grad_fn: {}", self.data, format!("{:?}", self.grad_fn).split_once(" {").unwrap().0))
    }
}

impl<S> Variable<S> {
    /// # Variable backward
    /// 
    /// Calls backward function on [Variable].
    /// This results in [Variable's](Variable) gradient being increased by one.
    /// We take [Variable] with mutable reference, so that we have exclusive access during mutation of gradient.
    /// 
    /// ```txt
    /// x.grad += 1;
    /// ```
    ///
    /// # Example
    ///
    /// ```
    /// use zyx::prelude::*;
    /// let mut x = 3.with_grad();
    /// println!("Grad: {}", x.grad());
    /// x.backward();
    /// println!("Grad: {}", x.grad());
    /// assert_eq!(x.grad(), &1);
    /// ```
    /// And gradient gets accumulated if we call backward again.
    /// ```
    /// # use zyx::prelude::*;
    /// # let mut x = 3.with_grad();
    /// # x.backward();
    /// x.backward();
    /// assert_eq!(x.grad(), &2);
    /// ```
    pub fn backward(&mut self)
    where
        S: crate::ops::HasDType + core::ops::Add<S::T, Output = S> + Clone,
        S::T: crate::ops::One,
    {
        use crate::ops::One;
        GradientRef(&self.grad).accumulate(S::T::one());
    }
}

/// # Backward trait
/// 
/// This trait is implemented by all functions that allow us to calculate gradients.
pub trait Backward<S> {
    /// Calls backward on a grad_fn, passing calculated output's gradient as parameter.
    fn backward(self, res_grad: S);
}

impl<S, F> Tensor<S, F> {
    /// # Tensor backward
    /// 
    /// Calls backward function on [Tensor].
    /// Computes gradient of all [Variables](Variable) that were used as inputs to operations that resulted in creation of this [Tensor].
    /// This function accumulates gradients in those [Variables](Variable). If you want to clear [Variable's](Variable) gradients, call [.zero_grad()](crate::nn::parameters::Parameters::zero_grad()) on that [Variable].
    ///
    /// # Example
    /// ```
    /// # use zyx::prelude::*;
    /// # use zyx::device::cpu;
    /// let mut device = cpu::Device::default();
    /// let x = device.buffer([2., 3., 1.]).with_grad();
    /// let y = x.exp();
    /// ```
    /// y is now [Tensor], so we can call backward on it.
    /// ```
    /// # use zyx::prelude::*;
    /// # use zyx::device::cpu;
    /// # let mut device = cpu::Device::default();
    /// # let x = device.buffer([2., 3., 1.]).with_grad();
    /// # let y = x.exp();
    /// y.backward();
    /// ```
    /// and the gradient of [Variable] x will now get populated:
    /// ```
    /// # use zyx::prelude::*;
    /// # use zyx::device::cpu;
    /// # let mut device = cpu::Device::default();
    /// # let x = device.buffer([2., 3., 1.]).with_grad();
    /// # let y = x.exp();
    /// # y.backward();
    /// assert_eq!(x.grad(), &x.data().clone().exp());
    /// ```
    pub fn backward(self)
    where
        S: crate::ops::HasDType + crate::ops::ZerosLike + core::ops::Add<S::T>,
        S::T: crate::ops::One,
        F: Backward<<S as core::ops::Add<S::T>>::Output>,
    {
        // NOTE: right now backward call is recursive.
        // Shall this pose a problem, we can switch to iterative version.
        use crate::ops::One;
        self.grad_fn.backward(self.data.zeros_like() + S::T::one());
    }
}

/// Create new [Variable] that requires gradient
impl<S> crate::ops::IntoVariable for S
where
    S: crate::ops::ZerosLike,
{
    fn with_grad(self) -> Variable<Self> {
        Variable {
            grad: Gradient::new(self.zeros_like()),
            data: self,
        }
    }
}

impl<S> Variable<S> {
    /// Access [Variable's](Variable) data buffer
    pub fn data(&self) -> &S {
        &self.data
    }
}

impl<S, GradFn> Tensor<S, GradFn> {
    /// Access [Tensor's](Tensor) data buffer
    pub fn data(&self) -> &S {
        &self.data
    }
}

impl<S> Variable<S> {
    /// Access [Tensor's](Tensor) grad buffer
    pub fn grad(&self) -> &S {
        self.grad.buffer()
    }
}

/*impl<S, GradFn> Tensor<S, GradFn> {
    /// Access [Tensor's](Tensor) backward function
    pub fn grad_fn(&self) -> &GradFn {
        &self.grad_fn
    }
}*/

/// Gradient hook for [Variable]
#[derive(Debug, Clone, Copy)]
pub struct GradHookV<'g, G, Hook> {
    grad: GradientRef<'g, G>,
    hook: Hook,
}

impl<S, G, HOOK> Backward<S> for GradHookV<'_, G, HOOK>
where
    S: Clone + crate::ops::HasShape,
    G: Clone + GradAcc<S>,
    HOOK: FnOnce(S),
{
    fn backward(self, res_grad: S) {
        (self.hook)(res_grad.clone());
        self.grad.accumulate(res_grad);
    }
}

impl<S> Variable<S> {
    /// Add custom FnOnce closure that will receive Buffer's gradient during backward pass.
    /// The hook is stored in the result, so make sure to do all operations on this result,
    /// otherwise your hook will not be called.
    pub fn register_hook<HOOK>(&self, hook: HOOK) -> Tensor<S, GradHookV<'_, S, HOOK>>
    where
        S: Clone,
        HOOK: FnOnce(S), // not necessary to put this requirement here, but seems like a good idea
    {
        Tensor {
            data: (*self.data()).clone(),
            grad_fn: GradHookV {
                grad: GradientRef::new(&self.grad),
                hook,
            },
        }
    }
}

/// Gradient hook for [Tensor]
#[derive(Debug, Clone, Copy)]
pub struct GradHookT<GradFn, HOOK> {
    grad_fn: GradFn,
    hook: HOOK,
}

impl<S, GradFn, HOOK> Backward<S> for GradHookT<GradFn, HOOK>
where
    GradFn: Backward<S>,
    HOOK: FnOnce(S),
    S: Clone,
{
    fn backward(self, res_grad: S) {
        (self.hook)(res_grad.clone());
        self.grad_fn.backward(res_grad);
    }
}

impl<S, GradFn> Tensor<S, GradFn> {
    /// Add custom FnOnce closure that will receive Buffer's gradient during backward pass
    /// The hook is stored in the result, so make sure to do all operations on this result,
    /// otherwise your hook will not be called.
    pub fn register_hook<HOOK>(self, hook: HOOK) -> Tensor<S, GradHookT<GradFn, HOOK>>
    where
        HOOK: FnOnce(S), // not necessary to put this requirement here, but seems like a good idea
    {
        Tensor {
            data: self.data,
            grad_fn: GradHookT {
                grad_fn: self.grad_fn,
                hook,
            },
        }
    }
}

/// Conversions between devices and types
// NOTE: you need to move the Variable into required device and type
// before using it in optimizer
impl<S, S2> crate::ops::ConvertFrom<Variable<S2>> for Variable<S>
where
    S: crate::ops::ConvertFrom<S2> + crate::ops::ZerosLike,
    S2: Clone,
{
    fn cfrom(x: Variable<S2>) -> Self {
        let data = S::cfrom(x.data);
        Self {
            grad: Gradient::new(data.zeros_like()),
            data,
        }
    }
}

/// Conversions between devices and types
// We usually don't want to move across devices inside the model, but we want want to implement changing dtypes,
// so here is an implementation of ConvertFrom, but keep in mind the performance implications of calling
// this function, especially if you are changing devices on the fly.
impl<S, S2, GradFn> crate::ops::ConvertFrom<Tensor<S2, GradFn>> for Tensor<S, GradFn>
where
    S: crate::ops::ConvertFrom<S2>,
{
    fn cfrom(x: Tensor<S2, GradFn>) -> Self {
        Self {
            data: S::cfrom(x.data),
            grad_fn: x.grad_fn,
        }
    }
}

impl<S> crate::nn::parameters::Parameters for &mut Variable<S>
where
    S: crate::ops::ZerosLike,
{
    fn zero_grad(&mut self) {
        self.grad.zero();
    }
}
