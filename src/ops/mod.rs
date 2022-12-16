//! # Tensor operations
//! 
//! Traits for different operations you can to with tensors.
//!
//! ## Operations are separated into categories
//! 
//! ```txt
//! Initialization ops:   ConvertFrom, Zeros, Ones, FromVec
//! Getters:              IntoVec, GetShape
//! Unary ops:            ReLU, DReLU, Exp, Ln, Tanh
//! Reduce ops:           Sum, Max, Min
//! Movement ops:         Reshape, Expand, Permute
//! Binary ops:           Pow
//! Processing ops:       MatMul, Conv
//! ```
//! 
//! ## List of all operations
//! 
//! This list excludes ops that are automatically implemented.
//! 
//! - [ConvertFrom]
//! - [Zeros]
//! - [Ones]
//! - [IntoVec]
//! - [FromVec]
//! - [GetShape]
//! - [ReLU]
//! - [DReLU]
//! - [Exp]
//! - [Ln]
//! - [Tanh]
//! - [Sum]
//! - [Max]
//! - [Min]
//! - [Reshape]
//! - [Expand]
//! - [Permute]
//! - [Pow]
//! - [MatMul]
//! - [Conv]
//! 

mod convert_from;
mod zeros;
mod ones;
mod get_shape;
mod relu;
mod drelu;
mod exp;
mod ln;
mod tanh;
mod min;
mod pow;
mod max;

use crate::shape::Shape;

/// ## Convert between devices and types
/// 
/// Create new tensor on given device with given type
// Needed because we can't use core::convert::From
// because it's foreign trait and it doesn't work
// when T == Self
pub trait ConvertFrom<T> {
    /// Converts input into output type
    fn cfrom(x: T) -> Self;
}

/// ## Convert into given type
/// 
/// This trait is automatically implemented for everything that implements ConvertFrom
pub trait ConvertInto<T> {
    /// Converts input into output type
    fn cinto(self) -> T;
}

impl<T, R> ConvertInto<R> for T
where
    R: ConvertFrom<T>,
{
    fn cinto(self) -> R {
        R::cfrom(self)
    }
}

/// ## Zeros operation
/// 
/// Create new tensor initialized with zeros.
/// ### Example
/// ```
/// use zyx::prelude::*;
/// use zyx::accel::cpu::Buffer;
/// let x = Buffer::<i32, _>::zeros((2usize, 3, 1));
/// ```
/// ### Output
/// ```txt
/// [0
///  0
///  0]
/// [0
///  0
///  0]
/// ```
pub trait Zeros {
    /// Shape of the resulting tensor
    type Sh: Shape<D = usize>;
    /// Create new tensor initialized with zeros.
    fn zeros(shape: Self::Sh) -> Self;
}

/// ## Ones operation
/// 
/// Create new tensor initialized with ones.
/// ### Example
/// ```
/// use zyx::prelude::*;
/// use zyx::accel::cpu::Buffer;
/// let x = Buffer::<i32, _>::ones((2usize, 3, 1));
/// let y = x.shape();
/// ```
/// ### Output
/// ```txt
/// [1
///  1
///  1]
/// [1
///  1
///  1]
/// ```
pub trait Ones {
    /// Shape of the resulting tensor
    type Sh: Shape<D = usize>;
    /// Create new tensor initialized with ones.
    fn ones(shape: Self::Sh) -> Self;
}

/// ## GetShape operation
/// 
/// Returns the shape of tensor as a Shape struct.
/// 
/// ### Example
/// ```
/// use zyx::{accel::cpu::Buffer, ops::{GetShape, ConvertFrom}};
/// let x = Buffer::cfrom([2, 3, 1]);
/// let y = x.shape();
/// assert_eq!(y, 3);
/// ```
pub trait GetShape {
    /// Type of the shape
    type Output: Shape<D = usize>;
    /// Get shape of input tensor.
    fn shape(&self) -> Self::Output;
}

// Unary ops
/// ## ReLU operation
/// 
/// Applies the rectified linear unit function
/// DReLU(x)=max⁡(0,x)
/// 
/// ### Example
/// ```
/// use zyx::ops::ReLU;
/// let x: i32 = 1;
/// let y = x.relu();
/// assert_eq!(y, 1);
/// ```
pub trait ReLU {
    /// Output of the ReLU operation.
    type Output;
    /// Apply ReLU operation on given input.
    fn relu(self) -> Self::Output;
}

/// ## DReLU operation
/// 
/// Applies the derivative of the rectified linear unit function
/// DReLU(x) = if self < 0. { 0. } else { 1. }
/// 
/// ### Example
/// ```
/// use zyx::ops::DReLU;
/// let x: i32 = 2;
/// let y = x.drelu();
/// assert_eq!(y, 1)
/// ```
pub trait DReLU {
    /// Output of the DReLU operation.
    type Output;
    /// Apply DReLU operation on given input.
    fn drelu(self) -> Self::Output;
}

/// ## Exp operation
/// 
/// Returns the exponential of the input
/// Exp(x) = x.exp()
/// 
/// ### Example
/// ```
/// use zyx::ops::Exp;
/// let x = 2.;
/// let y = x.exp();
/// ```
pub trait Exp {
    /// Output of the Exp operation.
    type Output;
    /// Apply Exp operation on given input.
    fn exp(self) -> Self::Output;
}

/// ## Ln operation
/// 
/// Returns the natural logarithm of the input
/// Ln(x) = x.ln()
/// 
/// ### Example
/// ```
/// use zyx::ops::Ln;
/// let x = 2.;
/// let y = x.ln();
/// ```
pub trait Ln {
    /// Output of the Ln operation.
    type Output;
    /// Apply Ln operation on given input.
    fn ln(self) -> Self::Output;
}

/// ## Tanh operation
/// 
/// Returns the hyperbolic tangent of the input
/// Tanh(x) = x.tanh()
/// 
/// ### Example
/// ```
/// use zyx::ops::Tanh;
/// let x = 2.;
/// let y = x.tanh();
/// ```
pub trait Tanh {
    /// Output of the Tanh operation.
    type Output;
    /// Apply Tanh operation on given input.
    fn tanh(self) -> Self::Output;
}

/// ## Summation operation
/// 
/// This operation reduces input across one or multiple dimensions.
/// All reduce operations (sum, max) take given dimensions and set them to one, applying operation accordingly.
/// The result's dimensions are not squeezed.
/// 
/// ### Example
/// 
/// ```
/// use zyx::prelude::*;
/// use zyx::accel::cpu::Buffer;
/// 
/// let x = Buffer::cfrom([[3, 2, 1], [4, 2, 1]]);
/// let y = x.sum([0]);
/// println!("{}", y);
/// ```
/// ### Output
/// ```txt
/// [7 4 2]
/// ```
/// 
pub trait Sum<Sh>
where
    Sh: Shape<D = i32>,
{
    /// Output of the Sum operation.
    type Output;
    /// Apply Sum operation on given input.
    fn sum(self, dims: Sh) -> Self::Output;
}

/// ## Max operation
/// 
/// This operation reduces input across one or multiple dimensions.
/// All reduce operations (sum, max) take given dimensions and set them to one, applying operation accordingly.
/// The result's dimensions are not squeezed.
/// 
/// ### Example
/// 
/// ```
/// use zyx::prelude::*;
/// use zyx::accel::cpu::Buffer;
/// 
/// let x = Buffer::cfrom([[3, 2, 1], [4, 2, 1]]);
/// let y = x.max([0]);
/// println!("{}", y);
/// ```
/// ### Output
/// ```txt
/// [4 2 1]
/// ```
/// 
pub trait Max<Sh>
where
    Sh: Shape<D = i32>,
{
    /// Output of the Max operation.
    type Output;
    /// Apply Max operation on given input.
    fn max(self, dims: Sh) -> Self::Output;
}

/// ## Min operation
/// 
/// This operation reduces input across one or multiple dimensions.
/// All reduce operations (sum, max) take given dimensions and set them to one, applying operation accordingly.
/// The result's dimensions are not squeezed.
/// 
/// ### Example
/// 
/// ```
/// use zyx::prelude::*;
/// use zyx::accel::cpu::Buffer;
/// 
/// let x = Buffer::cfrom([[3, 2, 1], [4, 2, 1]]);
/// let y = x.min([0]);
/// println!("{}", y);
/// ```
/// ### Output
/// ```txt
/// [3 2 1]
/// ```
/// 
pub trait Min<Sh>
where
    Sh: Shape<D = i32>,
{
    /// Output of the Min operation.
    type Output;
    /// Apply Min operation on given input.
    fn min(self, dims: Sh) -> Self::Output;
}

// Reshape simply changes shape of the tensor.
// Permute also changes it's data ordering.
// Expand expands to given shape if some dimensions are one.
// PERMUTE, PAD, SHRINK, EXPAND, FLIP,
// Reshape, Permute, Slice, Expand, Flip   # movement ops

// Movement ops
/// ## Reshape tensor
/// 
/// Reshaping changes tensor's shape, while leaving data untouched.
/// 
/// ### Example
/// ```
/// use zyx::accel::cpu::Buffer;
/// use zyx::prelude::*;
/// 
/// let x = Buffer::cfrom([[[3, 2, 4], [3, 4, 2]], [[1, 4, 2], [5, 1, 6]]]);
/// let x = x.reshape([2usize, 1, 6]);
/// println!("{}", x);
/// ```
/// 
/// ### Output
/// ```txt
/// [[3 2 4 2 4 2]
///  [1 4 2 5 1 6]]
/// ```
/// 
pub trait Reshape<Sh>
where
    Sh: Shape<D = usize>,
{
    /// Output of the Reshape operation.
    type Output;
    /// Apply Reshape operation on given input.
    fn reshape(self, shape: Sh) -> Self::Output;
}

/// ## Expand tensor
/// 
/// Expands tensor to given shape, if some dimensions are 1. Data is cloned to fill the required size.
/// 
/// ### Example
/// ```
/// use zyx::accel::cpu;
/// use zyx::prelude::*;
/// 
/// let x = cpu::Buffer::cfrom([[[3, 2, 4]], [[1, 4, 2]]]);
/// let x = x.expand((2usize, 3, 3));
/// println!("{}", x);
/// ```
/// 
/// ### Output
/// ```txt
/// [[3 2 4
///   3 2 4
///   3 2 4]
///  [1 4 2
///   1 4 2
///   1 4 2]]
/// ```
/// 
pub trait Expand<Sh>
where
    Sh: Shape<D = usize>,
{
    /// Output of the Expand operation.
    type Output;
    /// Apply Expand operation on given input.
    fn expand(self, shape: Sh) -> Self::Output;
}

/// ## Permute tensor
/// 
/// Shuffles tensors's dimensions in given order.
/// 
/// ### Example
/// ```
/// use zyx::accel::cpu::Buffer;
/// use zyx::prelude::*;
/// 
/// let x = Buffer::cfrom([[[3, 2, 4]], [[1, 4, 2]]]);
/// let x = x.permute((2, 0, 1));
/// assert_eq!(&x.to_vec(), &[3, 1, 2, 4, 4, 2]);
/// assert_eq!(x.shape(), (3, 2, 1));
/// println!("{}", x);
/// ```
/// 
/// ### Output
/// ```txt
/// [[3
///   1]
///  [2
///   4]
///  [4
///   2]]
/// ```
pub trait Permute<Sh>
where
    Sh: Shape<D = i32>,
{
    /// Output of the Permute operation.
    type Output;
    /// Apply Permute operation on given input.
    fn permute(self, dims: Sh) -> Self::Output;
}

// TODO: this is only API proposal, it is yet to be finalized
// Extracts only given dimensions, setting remaining dimensions to 1
/*pub trait Slice<SH, const N: usize>
where
    SH: Shape<N>,
{
    type Output;
    fn slice(self, dims: SH) -> Self::Output;
}*/

/// # Transpose tensor
///
/// Transpose is a subset of permute.
/// It is equivalent to x.permute((-1, -2))
///
/// ### Example
/// ```
/// use zyx::accel::cpu::Buffer;
/// use zyx::prelude::*;
///
/// let x = Buffer::cfrom([[3, 2, 4], [1, 4, 2]]);
/// let x = x.transpose();
/// assert_eq!(&x.to_vec(), &[3, 1, 2, 4, 4, 2]);
/// assert_eq!(x.shape(), (3, 2));
/// println!("{}", x);
/// ```
///
/// ### Output
/// ```txt
/// [3 1
///  2 4
///  4 2]
/// ```
pub trait Transpose {
    /// Output of the Transpose operation.
    type Output;
    /// Apply Transpose operation on given input.
    fn transpose(self) -> Self::Output;
}

impl<T> Transpose for T
where
    T: Permute<(i32, i32)>,
{
    type Output = T::Output;
    fn transpose(self) -> Self::Output {
        self.permute((-1, -2))
    }
}

// Binary ops are Add, Sub, Mul, Div, Pow, all with same size tensors,
// use core::ops to implement them (except for Pow)

/// Pow operation
/// 
/// Calculate the power of the input tensor to the given exponent tensor.
/// As with all binary operations, both left and right hand side can be also scalar.
///
/// ### Example
/// ```
/// use zyx::accel::cpu::Buffer;
/// use zyx::prelude::*;
///
/// let x = Buffer::cfrom([[3., 2., 4.], [1., 4., 2.]]);
/// let z = x.pow(2);
/// println!("{}", z);
/// ```
///
/// ### Output
/// ```txt
/// [ 9  4 16
///   1 16  4]
/// ```
pub trait Pow<Rhs = Self> {
    /// Output of the Pow operation.
    type Output;
    /// Apply Pow operation on given input.
    fn pow(self, rhs: Rhs) -> Self::Output;
}

/// ## Mathematical multiplication
/// 
/// Calculates matrix product.
/// 
/// ### Example
/// ```
/// use zyx::accel::cpu::Buffer;
/// use zyx::prelude::*;
/// 
/// let x = Buffer::cfrom([[3., 2., 4.], [1., 4., 2.]]);
/// let y = Buffer::cfrom([[3., 2.], [4., 1.], [4., 2.]]);
/// let z = x.matmul(y);
/// println!("{}", z);
/// ```
/// 
/// ### Output
/// ```txt
/// [33 16
///  27 10]
/// ```
pub trait MatMul<Rhs = Self> {
    /// Output of the MatMul operation.
    type Output;
    /// Apply MatMul operation on given input.
    fn matmul(self, rhs: Rhs) -> Self::Output;
}

// TODO: conv2d
/// ## 2D Convolution
/// 
/// Calculates 2D convodution.
/// 
/// NOTE: This API is not yet stable and may be subject to change
pub trait Conv<Pd, Kernel = Self>
where
    Pd: Shape<D = usize>,
{
    /// Output of the Conv operation.
    type Output;
    /// Apply Conv operation on given input.
    fn conv(self, kernel: Kernel, padding: Pd) -> Self::Output;
}

/// ## FromVec operation
/// 
/// Creates new tensor from given array and shape. Array is assumed to be in row-major order.
/// 
/// ### Example
/// ```
/// use zyx::prelude::*;
/// use zyx::accel::cpu::Buffer;
/// let x = Buffer::from_vec(&[2, 3, 1, 3], [2usize, 2]);
/// println!("{}", x);
/// ```
/// ### Output
/// [2 3
///  1 3]
pub trait FromVec {
    /// dtype of values in resulting tensor
    type T;
    /// Shape of the resulting tensor
    type Sh: Shape<D = usize>;
    /// Create new tensor from Vec and Shape.
    fn from_vec(data: &[Self::T], shape: Self::Sh) -> Self;
}

extern crate alloc;
/// ## IntoVec operation
/// 
/// Returns values from tensor as a Vec.
/// It must have row major order.
pub trait IntoVec<T> {
    /// Returns values from tensor as a Vec with row-major order.
    fn to_vec(&self) -> alloc::vec::Vec<T>;
}
