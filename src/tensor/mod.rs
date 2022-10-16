//! Description of generic Buffer types. All Buffers are immutable!
//! Mutability is allowed only for calculating gradients and optimizer parameters.
//! There are three Buffer types.
//! 1. Buffer     - only stores datatype inside Rc, passed around by cloning
//! 2. Variable - stores datatype and also it's gradient, passed around by reference
//! 3. Tensor - stores datatype and function necessary to calculate gradient of Variable, passed around by cloning
//!
//! Buffer and Variable are leaf Buffers. Tensor is strictly non-leaf and therefore it doesn't store it's gradient.
//!
//! # Example
//!
//! ```
//! use zyx::{accel::cpu::Buffer, tensor::{Variable, Tensor}};
//! use zyx::prelude::*;
//!
//! let x: Buffer<_> = Buffer::cfrom([2., 1., 4.]);        // basic Buffer
//! let y: Variable<_> = x.clone().with_grad();            // return Variable
//! let z: Tensor<_, _> = y.relu();                        // applying any function on Variable returns Tensor
//! let z: Buffer<_> = x.relu();                           // applying function to Buffer returns Buffer
//! ```
//!

mod init;
mod ops;

use crate::ops::GetShape;
use ops::RefCellReplaceTake;
use std::cell::{Ref, RefCell};

// How this works (for contributors)
//
// The Buffer and Variable are leaf Buffers. Buffer does not have grad, while Variable does (obviously).
// Tensor is non leaf Buffer.
// When an operation is performed with Buffer, a new Buffer is returned with result. Nothing magical happens.
// When an operation is performed with Variable, a new Tensor is returned that holds weak pointer
// to Variable's gradient. This Tensor contains a closure, that hold's this pointer and calculates
// the gradient when backward is called on it.
// We can imagine neural network as a tree, where leafs are Buffers and Variables and root/roots
// are Tensor. When an operation is performed with Tensor, it's consumed and it's func is moved to the resulting
// Tensor. So the last Tensor is the root of the tree and is the only Tensor in existence
// and it holds all the closures with Rc<RefCell<S>> pointers to the Variable's gradients.
// If you want to have more the one Tensor to call .backward() on, you need to clone this Tensor
// or any of the intermediate Tensors. In this case, the library performs cloning of the closures.
// This is a conscious decision to not store the closures in Rc and just clone them if needed,
// because we find the RAM usage of cloned closures (basically &RefCell<S> to gradients and S of some data)
// less concerning than the performance implications of using Rc<FnOnce(S)> closures.

// Tensors are moved into operations, while Variables are passed by reference!

/// # B<S>
///
/// This is used as a placeholder for custom storage, since some operations like addition are foreign traits,
/// so we need to have our type to wrap foreign storage types inside.
/// B<S> does not have gradient.
pub struct B<S>(pub S);

/// # Variable<S>
/// 
/// Variable holds data and it's gradient.
#[derive(Debug, Clone, Default, PartialEq, Eq, PartialOrd, Ord)]
pub struct Variable<S> {
    data: RefCell<S>, // RefCell here is needed for optimizer.step() function
    grad: RefCell<S>, // RefCell needed for .backward() gradient calculation
}

/// # Tensor<S, GradFn>
/// 
/// Tensor holds data and grad_fn to calculate gradients of Variables.
/// Tensor is only created as a result of some operations on at least one Variable.
/// Tensor does not store it's gradient, but the gradient can be accessed during backward
/// pass by using GradHookT. This is a FnOnce closure: x.register_hook(|grad| { // do something with grad })
#[derive(Debug, Clone, Default, PartialEq, Eq, PartialOrd, Ord)]
pub struct Tensor<S, GradFn> {
    data: S,
    func: GradFn, // Cell needed for .backward() freeing of buffers by making func None
}

/// # Display Variable
/// 
/// Shows Variable and it's gradient.
impl<S> std::fmt::Display for Variable<S>
where
    S: std::fmt::Display,
{
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        f.write_str(&format!(
            "{} with_grad\n{}",
            self.data.borrow(),
            self.grad.borrow()
        ))
    }
}

/// # Display Tensor
/// 
/// Shows Tensor and it's grad_fn.
impl<S, GradFn> std::fmt::Display for Tensor<S, GradFn>
where
    S: std::fmt::Display,
    GradFn: std::fmt::Debug,
{
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        f.write_str(&format!("{} with_func {:?}", self.data, self.func,))
    }
}

impl<S> Variable<S>
where
    S: Default + crate::ops::Ones + std::ops::Add<Output = S> + GetShape,
{
    pub fn backward(&self) {
        self.grad
            .replace_take(|grad| grad + S::ones(self.data().shape()));
    }
}

pub trait Backward<S> {
    fn backward(self, res_grad: S);
}

impl<S, F> Tensor<S, F>
where
    S: crate::ops::Ones + GetShape,
    F: Backward<S>,
{
    pub fn backward(self) {
        let shape = self.data.shape();
        // NOTE: right now backward call is recursive
        // shall this pose a problem, we can switch to iterative version
        self.func.backward(S::ones(shape));
    }
}

pub trait IntoVariable {
    fn with_grad(self) -> Variable<Self>
    where
        Self: Sized;
}

/// Create new Variable that requires gradient
impl<S> IntoVariable for S
where
    S: crate::ops::Zeros + GetShape,
{
    fn with_grad(self) -> Variable<Self> {
        let shape = self.shape();
        Variable {
            data: RefCell::new(self),
            grad: RefCell::new(S::zeros(shape)),
        }
    }
}

/// Drop Tensor's gradient function
impl<S, GradFn> Tensor<S, GradFn> {
    pub fn detach(self) -> S {
        self.data
    }
}

/// Access Variable's data buffer
impl<S> Variable<S> {
    pub fn data(&self) -> Ref<S> {
        self.data.borrow()
    }
}

/// Access Tensor's data buffer
impl<S, GradFn> Tensor<S, GradFn> {
    pub fn data(&self) -> &S {
        &self.data
    }
}

/// Access Buffer's grad buffer
impl<S> Variable<S> {
    pub fn grad(&self) -> Ref<S> {
        self.grad.borrow()
    }
}

/// Access Buffer's backward function
impl<S, GradFn> Tensor<S, GradFn> {
    pub fn grad_fn(&self) -> &GradFn {
        &self.func
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct GradHookV<'g, S, Hook> {
    grad: &'g RefCell<S>,
    hook: Hook,
}

impl<'g, S, HOOK> Backward<S> for GradHookV<'g, S, HOOK>
where
    S: Default + Clone + std::ops::Add<Output = S>,
    HOOK: FnOnce(S),
{
    fn backward(self, res_grad: S) {
        (self.hook)(res_grad.clone());
        self.grad.replace_take(|grad| grad + res_grad);
    }
}

/// Add custom FnOnce closure that will receive Buffer's gradient during backward pass
/// The hook is stored in the result, so make sure to do all operations on this result,
/// otherwise your hook will not be called.
impl<S> Variable<S> {
    pub fn register_hook<'g, HOOK>(&'g self, hook: HOOK) -> Tensor<S, GradHookV<'g, S, HOOK>>
    where
        S: 'g + Clone,
        HOOK: FnOnce(S), // not necessary to put this requirement here, but seems like a good idea
    {
        Tensor {
            data: (*self.data()).clone(),
            func: GradHookV {
                grad: &self.grad,
                hook,
            },
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct GradHookT<GradFn, HOOK> {
    func: GradFn,
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
        self.func.backward(res_grad);
    }
}

/// Add custom FnOnce closure that will receive Buffer's gradient during backward pass
/// The hook is stored in the result, so make sure to do all operations on this result,
/// otherwise your hook will not be called.
impl<S, GradFn> Tensor<S, GradFn> {
    pub fn register_hook<HOOK>(self, hook: HOOK) -> Tensor<S, GradHookT<GradFn, HOOK>>
    where
        HOOK: FnOnce(S), // not necessary to put this requirement here, but seems like a good idea
    {
        Tensor {
            data: self.data,
            func: GradHookT {
                func: self.func,
                hook,
            },
        }
    }
}

impl<S> Variable<S>
where
    S: Default,
{
    // users should be able to access Variables only immutably,
    // this is meant for optimizers
    pub(crate) fn update_data<F>(&self, f: F)
    where
        F: Fn(S) -> S,
    {
        self.data.replace_take(f);
    }
}

impl<S> Variable<S>
where
    S: Default,
{
    pub fn zero_grad(&self) {
        self.grad.replace(S::default());
    }
}

// conversions between devices and types
// NOTE: you need to move the Variable into required device and type
// before using it in optimizer
impl<S, S2> crate::ops::ConvertFrom<Variable<S2>> for Variable<S>
where
    S: crate::ops::ConvertFrom<S2>,
    S2: Clone,
{
    fn cfrom(x: Variable<S2>) -> Self {
        Self {
            data: RefCell::new(S::cfrom((*x.data.borrow()).clone())),
            grad: RefCell::new(S::cfrom((*x.grad.borrow()).clone())),
        }
    }
}

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
            func: x.func,
        }
    }
}
