//! Description of generic tensor types. All tensors are immutable!
//! Mutability is allowed only for calculating gradients and optimizer parameters.
//! There are three tensor types.
//! 1. Tensor     - only stores datatype inside Rc
//! 2. TensorGrad - stores datatype and also it's gradient
//! 3. TensorFunc - stores datatype and function necessary to calculate gradient of TensorGrad
//!
//! Tensor and TensorGrad are leaf tensors. TensorFunc is strictly non-leaf and therefore it doesn't store it's gradient.
//! 
//! # Example
//!
//! ```
//! use zyx::tensor::{Tensor, TensorGrad, TensorFunc};
//! use zyx::prelude::*;
//!
//! let x: Tensor<_> = Tensor::from([2., 1., 4.]);        // basic Tensor
//! let y: TensorGrad<_> = x.with_grad();                 // return TensorGrad
//! let z: TensorFunc<_, _> = y.relu();                   // applying any function on TensorGrad returns TensorFunc
//! let z: Tensor<_> = x.relu();                          // applying function to Tensor returns Tensor
//! ```
//!

mod init;
mod ops;

use crate::ops::GetShape;
use std::{
    cell::RefCell,
    rc::Rc,
};

// How this works (for contributors)
//
// The Tensor and TensorGrad are leaf tensors. Tensor does not have grad, while TensorGrad does (obviously).
// TensorFunc is non leaf tensor.
// When an operation is performed with Tensor, a new Tensor is returned with result. Nothing magical happens.
// When an operation is performed with TensorGrad, a new TensorFunc is returned that holds weak pointer
// to TensorGrad's gradient. This TensorFunc contains a closure, that hold's this pointer and calculates
// the gradient when backward is called on it.
// We can imagine neural network as a tree, where leafs are Tensors and TensorGrads and root/roots
// are TensorFunc. When an operation is performed with TensorFunc, it's consumed and it's func is moved to the resulting
// TensorFunc. So the last TensorFunc is the root of the tree and is the only TensorFunc in existence
// and it holds all the closures with Rc<RefCell<S>> pointers to the TensorGrad's gradients.
// If you want to have more the one TensorFunc to call .backward() on, you need to clone this TensorFunc
// or any of the intermediate TensorFuncs. In this case, the library performs cloning of the closures.
// This is a conscious decision to not store the closures in Rc and just clone them if needed,
// because we find the RAM usage of cloned closures (basically cloned Rc<RefCell<S>> to gradients and Rc<S> to some data)
// less concerning than the performance implications of using Rc<FnOnce(S)> closures.

// Tensors and TensorFuncs are moved into operations, while TensorGrads are passed by reference!

#[derive(Debug, Default, PartialEq, Eq, PartialOrd, Ord)]
pub struct Tensor<S> {
    pub data: Rc<S>,
}

// If needed, Rc can be change to Arc and RefCell to Mutex/RwLock

#[derive(Debug, Default, PartialEq, Eq, PartialOrd, Ord)]
pub struct TensorGrad<S> {
    pub data: RefCell<Rc<S>>, // RefCell here is needed for optimizer.step() function
    grad: RefCell<S>, // RefCell needed for .backward() gradient calculation
}

#[derive(Debug, Default, PartialEq, Eq, PartialOrd, Ord)]
pub struct TensorFunc<S, F> {
    pub data: Rc<S>,
    func: F, // Cell needed for .backward() freeing of buffers by making func None
}

// Automatic implementations of Clone trait try to clone underlying S instead of cloning Rc
impl<S> Clone for Tensor<S> {
    fn clone(&self) -> Self {
        Self {
            data: Rc::clone(&self.data),
        }
    }
}

impl<S> Clone for TensorGrad<S>
where
    S: Clone,
{
    fn clone(&self) -> Self {
        Self {
            data: RefCell::new(Rc::clone(&self.data.borrow())),
            grad: self.grad.clone(),
        }
    }
}

impl<S, F> Clone for TensorFunc<S, F>
where
    F: Clone,
{
    fn clone(&self) -> Self {
        Self {
            data: Rc::clone(&self.data),
            func: self.func.clone(),
        }
    }
}

impl<S> std::fmt::Display for Tensor<S>
where
    S: std::fmt::Display,
{
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        f.write_str(&format!("{}", self.data,))
    }
}

impl<S> std::fmt::Display for TensorGrad<S>
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

impl<S, F> std::fmt::Display for TensorFunc<S, F>
where
    S: std::fmt::Display,
{
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        f.write_str(&format!(
            "{} with_func {}",
            self.data,
            std::any::type_name::<F>()
        ))
    }
}

impl<S> TensorGrad<S>
where
    for<'a> &'a S: GetShape,
{
    pub fn backward(&self)
    where
        S: crate::ops::Ones,
        for<'a> &'a S: std::ops::Add<Output = S>,
    {
        self.grad
            .replace_with(|grad| &*grad + &S::ones(&self.data().shape()));
    }
}

impl<S, F> TensorFunc<S, F>
where
    for<'a> &'a S: GetShape,
{
    pub fn backward(self)
    where
        S: crate::ops::Ones,
        F: FnOnce(S),
    {
        let shape = self.data.shape();
        (self.func)(S::ones(&shape));
    }
}

impl<S> Tensor<S> {
    pub fn with_grad(&self) -> TensorGrad<S>
    where
        S: crate::ops::Zeros,
        for<'a> &'a S: GetShape,
    {
        TensorGrad {
            data: RefCell::new(Rc::clone(&self.data)),
            grad: RefCell::new(S::zeros(&self.data.shape())),
        }
    }
}

/// Drop tensor's gradient
impl<S> TensorGrad<S> {
    pub fn detach(&self) -> Tensor<S> {
        Tensor {
            data: Rc::clone(&self.data.borrow()),
        }
    }
}

/// Drop tensor's gradient
impl<S, F> TensorFunc<S, F> {
    pub fn detach(&self) -> Tensor<S> {
        Tensor {
            data: Rc::clone(&self.data),
        }
    }
}

impl<S> Tensor<S> {
    pub fn data(&self) -> Rc<S> {
        Rc::clone(&self.data)
    }
}

impl<S> TensorGrad<S> {
    pub fn data(&self) -> Rc<S> {
        Rc::clone(&self.data.borrow())
    }
}

impl<S, F> TensorFunc<S, F> {
    pub fn data(&self) -> Rc<S> {
        Rc::clone(&self.data)
    }
}

impl<S> TensorGrad<S> {
    pub fn grad(&self) -> &RefCell<S> {
        &self.grad
    }
}

impl<S> TensorGrad<S> {
    pub fn update_data(&self, f: &dyn Fn(&mut Rc<S>) -> Rc<S>) {
        self.data.replace_with(f);
    }
}

impl<S> TensorGrad<S>
where
    S: Default,
{
    pub fn zero_grad(&self) {
        self.grad.replace(S::default());
    }
}

// conversions between devices and types
// NOTE: you need to move the tensor into required device and type
// before using it in optimizer
impl<S, S2> crate::ops::ConvertFrom<Tensor<S2>> for Tensor<S>
where
    for<'a> S: crate::ops::ConvertFrom<&'a S2>,
{
    fn convert_from(x: Tensor<S2>) -> Self {
        Self {
            data: Rc::new(S::convert_from(x.data.as_ref())),
        }
    }
}

impl<S, S2> crate::ops::ConvertFrom<TensorGrad<S2>> for TensorGrad<S>
where
    for<'a> S: crate::ops::ConvertFrom<&'a S2>,
{
    fn convert_from(x: TensorGrad<S2>) -> Self {
        Self {
            data: RefCell::new(Rc::new(S::convert_from(x.data.borrow().as_ref()))),
            grad: RefCell::new(S::convert_from(&x.grad.borrow())),
        }
    }
}

// We usually don't want to move across devices inside the model, but we want want to implement changing dtypes,
// so here is an implementation of ConvertFrom, but keep in mind the performance implications of calling
// this function, especially if you are changing devices on the fly.
impl<S, S2, F> crate::ops::ConvertFrom<TensorFunc<S2, F>> for TensorFunc<S, F>
where
    for<'a> S: crate::ops::ConvertFrom<&'a S2>,
{
    fn convert_from(x: TensorFunc<S2, F>) -> Self {
        Self {
            data: Rc::new(S::convert_from(x.data.as_ref())),
            func: x.func,
        }
    }
}
