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
//! # use zyx::{accel::cpu::Buffer, tensor::{Variable, Tensor}};
//! # use zyx::prelude::*;
//! let x: Buffer<_> = Buffer::cfrom([2., 1., 4.]);
//! ```
//!
//! By calling [IntoVariable::with_grad()] on any datatype, you get [Variable], which adds gradient to the buffer.
//! ```
//! # use zyx::{accel::cpu::Buffer, tensor::{Variable, Tensor}};
//! # use zyx::prelude::*;
//! # let x: Buffer<_> = Buffer::cfrom([2., 1., 4.]);
//! let y: Variable<_, ()> = x.clone().with_grad();
//! ```
//!
//! Applying function to [Variable] returns [Tensor].
//! [Tensor] stores references to [Variable's](Variable) gradients and some data buffers used during gradient calculation.
//! ```
//! # use zyx::{accel::cpu::Buffer, tensor::{Variable, Tensor}};
//! # use zyx::prelude::*;
//! # let y: Variable<_, ()> = Buffer::cfrom([2., 1., 4.]).with_grad();
//! let z: Tensor<_, _> = y.relu();
//! ```
//!
//! Applying function to buffer simply returns buffer.
//! ```
//! # use zyx::{accel::cpu::Buffer, tensor::{Variable, Tensor}};
//! # use zyx::prelude::*;
//! # let x: Buffer<_> = Buffer::cfrom([2., 1., 4.]);
//! let z: Buffer<_> = x.relu();
//! ```
//!

// This file contains tensor definitions, getters and setters for tensors.

mod ops;

use crate::{ops::GetShape, module::Parameters, optim::Optimizer};

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
pub struct Variable<S, G> {
    data: S,
    // Theoretically gradient should be the same shape and type as data, but practically
    // it just needs to implement operations required by optimizers so there is no need
    // to arbitrarily constrain it. This also simplyfies some performance optimizations.
    // One side effect is, that the type of the gradient does not need to be user known
    // at the time of creation, because compiler will determine it when some operation
    // is applied to this Variable.
    grad: Gradient<G>,
}

/// Gradient of [Variable]
/// 
/// User has read-only access to [Gradient] with to_vec and to_string.
// This is using UnsafeCell to store value inside gradient.
// Since UnsafeCell is not Sync, there is just few times that we access gradient,
// so it is not difficult to make it safe to use.
// The only mutable accesses are [Self::zero], which does not use unsafe
// and accumulate, which uses unsafe, since [Variable] can be passed into multiple
// functions and all of them need to be able to accumulate gradient, also maybe needed
// for batch?
#[derive(Default, Debug)]
pub struct Gradient<G>(std::cell::UnsafeCell<Option<G>>);

impl<G> Clone for Gradient<G>
where
    G: Clone,
{
    fn clone(&self) -> Self {
        unsafe { Self(std::cell::UnsafeCell::new((*self.0.get()).clone())) }
    }
}

impl<G> Gradient<G> {
    fn new() -> Self {
        Self(std::cell::UnsafeCell::new(None))
    }

    fn zero(&self) {
        // TODO this unsafe can be removed by making this function take mutable reference,
        // that however requires that parameters are passed around mutably
        // (they can not be stored in optimizers)
        unsafe { *self.0.get() = None; }
    }
}

#[derive(Debug, Clone, Copy)]
struct GradientRef<'g, G>(&'g Gradient<G>);

impl<'g, G> GradientRef<'g, G> {
    fn new(gradient: &'g Gradient<G>) -> Self {
        Self(gradient)
    }

    fn accumulate(&self, value: G)
    where
        G: std::ops::Add<Output = G>,
    {
        let mut x = None;
        unsafe { core::ptr::swap(self.0.0.get(), &mut x); }
        let grad = Some(match x {
            Some(grad) => grad + value,
            None => value,
        });
        unsafe { *self.0.0.get() = grad; }
    }
}

impl<G, T> crate::ops::IntoVec<T> for Gradient<G>
where
    G: crate::ops::IntoVec<T>,
{
    fn to_vec(&self) -> Vec<T> {
        if let Some(grad) = unsafe { &*self.0.get() } {
            grad.to_vec()
        } else {
            Vec::new()
        }
    }
}

impl<G> std::fmt::Display for Gradient<G>
where
    G: std::fmt::Display,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let grad = unsafe { &*self.0.get() };
        let s = if grad.is_some() {
            grad.as_ref().unwrap().to_string()
        } else {
            "None".into()
        };
        f.write_str(&s)
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
impl<S, G> std::fmt::Display for Variable<S, G>
where
    S: std::fmt::Display,
    G: std::fmt::Display,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
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
impl<S, GradFn> std::fmt::Display for Tensor<S, GradFn>
where
    S: std::fmt::Display,
    GradFn: std::fmt::Debug,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // TODO: this is ugly solution, please just print the name of the function, don't require GradFn debug
        // and creating debug string of all the buffers stored in grad_fn
        f.write_str(&format!("{} with grad_fn: {}", self.data, format!("{:?}", self.grad_fn).split_once(" {").unwrap().0))
    }
}

impl<S> Variable<S, i32> {
    /// # Variable backward
    /// 
    /// Calls backward function on [Variable].
    /// This results in [Variable's](Variable) gradient being increased by one.
    /// 
    /// ```txt
    /// x.grad += 1;
    /// ```
    pub fn backward(&self) {
        GradientRef(&self.grad).accumulate(1);
    }
}

/// # Backward trait
/// 
/// This trait is implemented by all functions that allow us to calculate gradients.
pub trait Backward<S> {
    /// Calls backward on a grad_fn, passing calculated output's gradient as parameter.
    fn backward(self, res_grad: S);
}

impl<S, F> Tensor<S, F>
where
    S: crate::ops::Ones + GetShape,
    F: Backward<S>,
{
    /// # Tensor backward
    /// 
    /// Calls backward function on [Tensor].
    /// Computes gradient of all [Variables](Variable) that were used as inputs to operations that resulted in creation of this [Tensor].
    /// This function accumulates gradients in those [Variables](Variable). If you want to clear [Variable's](Variable) gradients, call [Parameters::zero_grad()] on that [Variable].
    ///
    /// # Example
    /// ```
    /// # use zyx::prelude::*;
    /// # use zyx::accel::cpu;
    /// let x = cpu::Buffer::cfrom([2., 3., 1.]).with_grad::<cpu::Buffer<f32>>();
    /// let y = x.exp();
    /// ```
    /// y is now [Tensor], so we can call backward on it.
    /// ```
    /// # use zyx::prelude::*;
    /// # use zyx::accel::cpu;
    /// # let x = cpu::Buffer::cfrom([2., 3., 1.]).with_grad();
    /// # let y = x.exp();
    /// y.backward();
    /// ```
    /// and the gradient of [Variable] x will now get populated:
    /// ```
    /// # use zyx::prelude::*;
    /// # use zyx::accel::cpu;
    /// # let x = cpu::Buffer::cfrom([2., 3., 1.]).with_grad();
    /// # let y = x.exp();
    /// # y.backward();
    /// assert_eq!(x.grad().to_vec(), x.data().clone().exp().to_vec());
    /// ```
    pub fn backward(self) {
        // NOTE: right now backward call is recursive.
        // Shall this pose a problem, we can switch to iterative version.
        let shape = self.shape();
        self.grad_fn.backward(S::ones(shape));
    }
}

/// Turn any datatype into [Variable].
pub trait IntoVariable {
    /// Calling this function turns input into [Variable] adding gradient in the process.
    fn with_grad<G>(self) -> Variable<Self, G>
    where
        Self: Sized;
}

/// Create new [Variable] that requires gradient
impl<S> IntoVariable for S {
    fn with_grad<G>(self) -> Variable<Self, G> {
        Variable {
            data: self,
            grad: Gradient::new(),
        }
    }
}

impl<S, G> Variable<S, G> {
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

impl<S, G> Variable<S, G> {
    /// Access [Tensor's](Tensor) grad buffer
    pub fn grad(&self) -> &Gradient<G> {
        &self.grad
    }
}

impl<S, GradFn> Tensor<S, GradFn> {
    /// Access [Tensor's](Tensor) backward function
    pub fn grad_fn(&self) -> &GradFn {
        &self.grad_fn
    }
}

/// Gradient hook for [Variable]
#[derive(Debug, Clone, Copy)]
pub struct GradHookV<'g, G, Hook> {
    grad: GradientRef<'g, G>,
    hook: Hook,
}

impl<G, HOOK> Backward<G> for GradHookV<'_, G, HOOK>
where
    G: Clone + std::ops::Add<G, Output = G>,
    HOOK: FnOnce(G),
{
    fn backward(self, res_grad: G) {
        (self.hook)(res_grad.clone());
        self.grad.accumulate(res_grad);
    }
}

impl<S, G> Variable<S, G> {
    /// Add custom FnOnce closure that will receive Buffer's gradient during backward pass.
    /// The hook is stored in the result, so make sure to do all operations on this result,
    /// otherwise your hook will not be called.
    pub fn register_hook<HOOK>(&self, hook: HOOK) -> Tensor<S, GradHookV<'_, G, HOOK>>
    where
        S: Clone,
        HOOK: FnOnce(G), // not necessary to put this requirement here, but seems like a good idea
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
impl<S, S2, G> crate::ops::ConvertFrom<Variable<S2, G>> for Variable<S, G>
where
    S: crate::ops::ConvertFrom<S2>,
    S2: Clone,
{
    fn cfrom(x: Variable<S2, G>) -> Self {
        Self {
            data: S::cfrom(x.data),
            grad: Gradient::new(),
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


// Update Variable's data. It is used by optimizer.step().
impl<S, G> Parameters for &Variable<S, G>
//where
    //S: Default + Clone + Sub<Output = S> + Mul<Output = S> + Mul<f64, Output = S> + GetShape,
{
    fn update_data<Optim>(&self, _optim: &Optim)
    where
        Optim: Optimizer
    {
        //let grad = self.grad().clone();
        //let data = self.data.take();
        //self.data.replace(optim.update_data(data, grad));
    }

    fn zero_grad(&self) {
        self.grad.zero();
    }
}
