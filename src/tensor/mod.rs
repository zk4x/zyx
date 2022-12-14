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
//! # use zyx::{device::cpu::{self, Buffer}, tensor::{Variable, Tensor, ReLUBackwardV}, shape::Sh1};
//! # let mut device = cpu::Device::default();
//! # let x: Buffer<Sh1<3>, f32> = device.buffer([2., 1., 4.]);
//! # let y: Variable<Buffer<Sh1<3>, f32>> = x.clone().with_grad();
//! let z: Tensor<Buffer<Sh1<3>, f32>, ReLUBackwardV<_, _>> = y.relu();
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
//! How this works
//!
//! The Buffer and Variable are leaf tensors. Buffer does not have grad, while Variable does (obviously).
//! Tensor is non leaf tensor.
//!
//! When an operation is performed with Buffer, a new Buffer is returned with result. Nothing magical happens.
//!
//! When an operation is performed with Variable, a new Tensor is returned that holds reference to Variable's gradient.
//! This Tensor contains a struct with name \[Op\]Backward\[Operands\], that hold's this pointer and calculates
//! the gradient when backward is called on it.
//!
//! We can imagine neural network as a tree, where leafs are Buffers and Variables and root/roots
//! are Tensors. When an operation is performed with Tensor, it's consumed and it's grad_fn is moved to the resulting
//! Tensor. So the last [Tensor] is the root of the tree and it holds all the closures with RefGradient references to [Variable's](Variable) gradients.
//! If you want to have more the one [Tensor] to call .backward() on, you need to clone this Tensor
//! or any of the intermediate Tensors. In this case, the library performs cloning of the closures.
//!
//! Tensors are moved into operations, while Variables are passed by reference!
//!
//! This file contains definitions, getters and setters for [Variable] and [Tensor].

mod ops;

// re export backward ops
pub use ops::{
    add::{
        AddBackwardSV, AddBackwardTT, AddBackwardTV, AddBackwardVS, AddBackwardVT, AddBackwardVV,
    },
    //sub::{SubBackwardSV, SubBackwardVV, SubBackwardVT, SubBackwardTT, SubBackwardTV, SubBackwardVS},
    //div::{DivBackwardTS, DivBackwardTT, DivBackwardTV, DivBackwardVS, DivBackwardVT, DivBackwardVV},
    //mul::{MulBackwardTS, MulBackwardTT, MulBackwardTV, MulBackwardVS, MulBackwardVT, MulBackwardVV},
    matmul::{
        MatMulBackwardST, MatMulBackwardSV, MatMulBackwardTS, MatMulBackwardTT, MatMulBackwardTV,
        MatMulBackwardVS, MatMulBackwardVT, MatMulBackwardVV,
    },
    relu::{ReLUBackwardT, ReLUBackwardV},
    tanh::{TanhBackwardT, TanhBackwardV},
};

/// # Variable
///
/// Variable holds data and it's gradient.
///
/// ## Gradient of [Variable]
///
/// User has read-only access to gradient.
/// This is using UnsafeCell to store value inside gradient.
/// Since UnsafeCell is not Sync, there is just few times that we access gradient,
/// so it is not difficult to make it safe to use.
/// The only mutable accesses are zeroing gradients, and accumulate, which use unsafe, since [Variable] can be passed into multiple
/// functions and all of them need to be able to accumulate gradient.
/// So we just make sure that backward can not be called on buffer that is borrowed and that is it.
#[derive(Debug, Clone, Default)]
pub struct Variable<B> {
    pub(crate) data: B,
    // Gradient has the same type and shape as data
    grad: Gradient<B>,
}

#[derive(Default, Debug)]
struct Gradient<B>(core::cell::UnsafeCell<B>);

impl<B> Clone for Gradient<B>
where
    B: Clone,
{
    fn clone(&self) -> Self {
        // Safe, read only access
        unsafe { Self(core::cell::UnsafeCell::new((*self.0.get()).clone())) }
    }
}

impl<B> Gradient<B> {
    fn new(data: B) -> Self {
        Self(core::cell::UnsafeCell::new(data))
    }

    // Get value stored inside of the gradient
    /*fn buffer(&self) -> &G {
        unsafe { &*self.0.get() }
    }*/

    fn zero(&mut self)
    where
        B: crate::ops::ZerosLike,
    {
        self.0.get_mut().zeros_like();
    }
}

trait GradAcc<B>: core::ops::Add<B, Output = Self> + Clone {}
impl<G, B> GradAcc<G> for B where B: core::ops::Add<G, Output = Self> + Clone {}

#[derive(Debug, Clone, Copy)]
struct GradientRef<'g, B>(&'g Gradient<B>);

impl<'g, B> GradientRef<'g, B> {
    fn new(gradient: &'g Gradient<B>) -> Self {
        Self(gradient)
    }

    fn accumulate<G>(&self, value: G)
    where
        B: GradAcc<G>,
    {
        // Accumulate is called by backward function to accumulate gradients. This is needed in batch processing.
        // Unsafe is needed, because we need multiple functions accessing the same gradient.
        unsafe {
            *self.0 .0.get() = (*self.0 .0.get()).clone() + value;
        }
    }
}

/// # Tensor
///
/// Tensor holds data and grad_fn to calculate gradients of [Variables](Variable).
/// Tensor is only created as a result of some operations on at least one [Variable].
/// Tensor does not store it's gradient, but the gradient can be accessed during backward
/// pass by using [GradHookT].
// TODO DOCS
#[derive(Debug, Clone, Default, PartialEq, Eq, PartialOrd, Ord)]
pub struct Tensor<S, GradFn> {
    data: S,
    grad_fn: GradFn,
}

/// # Display Variable
///
/// Shows [Variable] and it's gradient.
impl<B> core::fmt::Display for Variable<B>
where
    B: core::fmt::Display,
{
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        extern crate alloc;
        use alloc::format;
        f.write_str(&format!("{} with grad:\n{}", self.data, self.grad(),))
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
        f.write_str(&format!(
            "{} with grad_fn: {}",
            self.data,
            format!("{:?}", self.grad_fn).split_once(" {").unwrap().0
        ))
    }
}

impl<B> Variable<B> {
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
        B: crate::ops::HasDType + core::ops::Add<B::T, Output = B> + Clone,
        B::T: num_traits::One,
    {
        use num_traits::One;
        GradientRef(&self.grad).accumulate(B::T::one());
    }
}

/// # Backward trait
///
/// This trait is implemented by all functions that allow us to calculate gradients.
///
/// These functions are stored inside [Tensor] as it's grad_fn.
/// But you can't directly access these functions.
/// They are created by calling operations on [Variables](Variable) and [Tensors](Tensor).
pub trait Backward<B> {
    /// Calls backward on a grad_fn, passing calculated output's gradient as parameter.
    fn backward(self, res_grad: B);
}

impl<B, F> Tensor<B, F> {
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
        B: crate::ops::HasDType,
        B::T: num_traits::One,
        F: Backward<B::T>,
    {
        // NOTE: right now backward call is recursive.
        // Shall this pose a problem, we can switch to iterative version.
        use num_traits::One;
        self.grad_fn.backward(B::T::one());
    }
}

/// Create new [Variable] that requires gradient
impl<B> crate::ops::IntoVariable for B
where
    B: crate::ops::ZerosLike,
{
    fn with_grad(self) -> Variable<Self> {
        Variable {
            grad: Gradient::new(self.zeros_like()),
            data: self,
        }
    }
}

impl<B> Variable<B> {
    /// Access [Variable's](Variable) data buffer
    pub fn data(&self) -> &B {
        &self.data
    }
}

impl<B, GradFn> Tensor<B, GradFn> {
    /// Access [Tensor's](Tensor) data buffer
    pub fn data(&self) -> &B {
        &self.data
    }
}

impl<B> Variable<B> {
    /// Access [Tensor's](Tensor) grad buffer
    pub fn grad(&self) -> &B {
        // unsafe access, but read only, may be a problem in certain cases, we need more testing
        unsafe { &*self.grad.0.get() }
    }
}

/*impl<S, GradFn> Tensor<S, GradFn> {
    /// Access [Tensor's](Tensor) backward function
    pub fn grad_fn(&self) -> &GradFn {
        &self.grad_fn
    }
}*/

/// Gradient hook for [Variable]
// TODO DOCS
#[derive(Debug, Clone, Copy)]
pub struct GradHookV<'g, G, Hook> {
    grad: GradientRef<'g, G>,
    hook: Hook,
}

impl<B, G, HOOK> Backward<B> for GradHookV<'_, G, HOOK>
where
    B: Clone + crate::ops::HasShape,
    G: Clone + GradAcc<B>,
    HOOK: FnOnce(B),
{
    fn backward(self, res_grad: B) {
        (self.hook)(res_grad.clone());
        self.grad.accumulate(res_grad);
    }
}

impl<B> Variable<B> {
    /// Add custom FnOnce closure that will receive Buffer's gradient during backward pass.
    /// The hook is stored in the result, so make sure to do all operations on this result,
    /// otherwise your hook will not be called.
    pub fn register_hook<HOOK>(&self, hook: HOOK) -> Tensor<B, GradHookV<'_, B, HOOK>>
    where
        B: Clone,
        HOOK: FnOnce(B), // not necessary to put this requirement here, but seems like a good idea
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
// TODO DOCS
#[derive(Debug, Clone, Copy)]
pub struct GradHookT<GradFn, HOOK> {
    grad_fn: GradFn,
    hook: HOOK,
}

impl<B, GradFn, HOOK> Backward<B> for GradHookT<GradFn, HOOK>
where
    GradFn: Backward<B>,
    HOOK: FnOnce(B),
    B: Clone,
{
    fn backward(self, res_grad: B) {
        (self.hook)(res_grad.clone());
        self.grad_fn.backward(res_grad);
    }
}

impl<B, GradFn> Tensor<B, GradFn> {
    /// Add custom FnOnce closure that will receive Buffer's gradient during backward pass
    /// The hook is stored in the result, so make sure to do all operations on this result,
    /// otherwise your hook will not be called.
    // TODO DOCS
    pub fn register_hook<HOOK>(self, hook: HOOK) -> Tensor<B, GradHookT<GradFn, HOOK>>
    where
        HOOK: FnOnce(B), // not necessary to put this requirement here, but seems like a good idea
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
impl<B, S2> crate::ops::ConvertFrom<Variable<S2>> for Variable<B>
where
    B: crate::ops::ConvertFrom<S2> + crate::ops::ZerosLike,
    S2: Clone,
{
    fn cfrom(x: Variable<S2>) -> Self {
        let data = B::cfrom(x.data);
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
impl<B, S2, GradFn> crate::ops::ConvertFrom<Tensor<S2, GradFn>> for Tensor<B, GradFn>
where
    B: crate::ops::ConvertFrom<S2>,
{
    fn cfrom(x: Tensor<S2, GradFn>) -> Self {
        Self {
            data: B::cfrom(x.data),
            grad_fn: x.grad_fn,
        }
    }
}

impl<B> crate::nn::parameters::Parameters for &mut Variable<B>
where
    B: crate::ops::ZerosLike,
{
    fn zero_grad(&mut self) {
        self.grad.zero();
    }
}
