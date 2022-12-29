//! # Tensor operations
//! 
//! Traits for different operations you can to with tensors.
//!
//! ## Operations are separated into categories
//! 
//! ```txt
//! Initialization ops:   ConvertFrom, Zeros, Ones
//! Getters:              IntoVariable, IntoVec, HasShape
//! Unary ops:            ReLU, DReLU, Exp, Ln, Tanh
//! Reduce ops:           Sum, Max, Min
//! Movement ops:         Reshape, Expand, Permute
//! Binary ops:           Pow
//! Processing ops:       MatMul, Conv
//! ```
//! 

mod convert_from;
mod zero;
mod one;
mod relu;
mod drelu;
mod exp;
mod ln;
mod tanh;
mod pow;
mod has_min;
mod has_max;
mod zeros_like;

use crate::shape::{Shape, Axes, ReducableBy};

/// # HasDevice
pub trait HasDevice {
    /// Type of device that tensor is stored on
    type Dev: crate::device::Device;
    /// Device that tensor is stored on
    fn device(&self) -> &Self::Dev;
}

/// # HasDType
pub trait HasDType {
    /// Type of tensor
    type T: crate::dtype::DType;
}

/// # HasShape
/// 
/// Stores the shape of the tensor.
/// 
/// ## Example
/// ```
/// use zyx::prelude::*;
/// use zyx::device::cpu;
/// let mut device = cpu::Device::default();
/// let x = device.buffer([2, 3, 1]);
/// assert_eq!(x.shape(), [3]);
/// ```
pub trait HasShape {
    /// Shape of tensor
    type Sh: Shape;

    /// Get the shape as array
    fn shape(&self) -> <Self::Sh as Shape>::AsArray {
        Self::Sh::array()
    }
}

/// # HasMax
pub trait HasMax {
    /// Global maximum of tensor
    fn max() -> Self;
}

/// # HasMin
pub trait HasMin {
    /// Global minimum of tensor
    fn min() -> Self;
}

/// # ZerosLike
///
/// Returns a tensor filled with the scalar value 0, with the same size as input.
pub trait ZerosLike {
    /// Returns a tensor filled with the scalar value 0, with the same size as input.
    fn zeros_like(&self) -> Self;
}

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

/// ## Zero operation
/// 
/// Create new tensor initialized with zeros.
/// ### Example
/// ```
/// use zyx::prelude::*;
/// use zyx::device::cpu;
/// use zyx::shape::Sh3;
///
/// let mut device = cpu::Device::default();
///
/// let x: cpu::Buffer<'_, Sh3<2, 3, 1>> = device.zeros();
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
pub trait Zero {
    /// Create new tensor initialized with zeros.
    fn zero() -> Self;
}

/// ## One operation
/// 
/// Create new tensor initialized with ones.
/// ### Example
/// ```
/// use zyx::prelude::*;
/// use zyx::device::cpu::{Device, Buffer};
/// use zyx::shape::Sh3;
///
/// let mut device = Device::default();
/// 
/// let x: Buffer<'_, Sh3<2, 3, 1>, i32> = device.ones();
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
pub trait One {
    /// Create new tensor initialized with ones.
    fn one() -> Self;
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
/// use zyx::device::cpu;
/// use zyx::shape::Ax1;
///
/// let mut device = cpu::Device::default();
/// 
/// let x = device.buffer([[3, 2, 1], [4, 2, 1]]);
/// let y = x.sum::<Ax1<0>>();
/// println!("{}", y);
/// ```
/// ### Output
/// ```txt
/// [7 4 2]
/// ```
pub trait Sum {
    /// Sum over dims
    fn sum<Dims>(self) -> Self::Output
    where
        Dims: Axes,
        Self: Summable<Dims>;
}

impl<T> Sum for T {
    fn sum<Dims>(self) -> T::Output
    where
        Dims: Axes,
        T: Summable<Dims>,
    {
        self._sum()
    }
}

/// Summable
pub trait Summable<Dims>
where
    Dims: Axes,
{
    /// Output of the Sum operation.
    type Output;
    /// Apply Sum operation on given input.
    fn _sum(self) -> Self::Output;
}

/// ## Max operation
/// 
/// This operation reduces input across one or multiple dimensions.
/// All reduce operations (sum, max) take given dimensions and set them to one, applying operation accordingly.
/// The result's dimensions are not squeezed.
/// 
/// ### Example
/// 
/// ```ignore
/// use zyx::prelude::*;
/// use zyx::device::cpu::Buffer;
/// use zyx::shape::Ax1;
/// 
/// let x = Buffer::cfrom([[3, 2, 1], [4, 2, 1]]);
/// let y = x.max::<Ax1<0>>();
/// println!("{}, {}", y.0, y.1);
/// ```
/// ### Output
/// ```txt
/// [[4 2 1]], [[1 0 0]]
/// ```
pub trait Max {
    /// Max over dims
    fn max<Dims>(self) -> (Self::Values, Self::Indices)
    where
        Dims: Axes,
        Self: Maximizable<Dims>;
}

impl<T> Max for T {
    fn max<Dims>(self) -> (T::Values, T::Indices)
    where
        Dims: Axes,
        T: Maximizable<Dims>
    {
        self._max()
    }
}

/// Maximizable
pub trait Maximizable<Dims>
where
    Dims: Axes,
{
    /// Output of the Max operation.
    type Values;
    /// Indices of Values.
    type Indices;
    /// Apply Max operation on given input.
    fn _max(self) -> (Self::Values, Self::Indices);
}

/// ## Min operation
/// 
/// This operation reduces input across one or multiple dimensions.
/// All reduce operations (sum, max) take given dimensions and set them to one, applying operation accordingly.
/// The result's dimensions are not squeezed.
/// 
/// ### Example
/// 
/// ```ignore
/// use zyx::prelude::*;
/// use zyx::device::cpu::Buffer;
/// use zyx::shape::Ax1;
/// 
/// let x = Buffer::cfrom([[3, 2, 1], [4, 2, 1]]);
/// let y = x.min::<Ax1<0>>();
/// println!("{}", y.0);
/// ```
/// ### Output
/// ```txt
/// [[3 2 1]]
/// ```
pub trait Min {
    /// Minimize over dims
    fn min<Dims>(self) -> (Self::Values, Self::Indices)
    where
        Dims: Axes,
        Self: Minimizable<Dims>;
}

impl<T> Min for T {
    fn min<Dims>(self) -> (T::Values, T::Indices)
    where
        Dims: Axes,
        T: Minimizable<Dims>
    {
        self._min()
    }
}

/// Minimizable
pub trait Minimizable<Dims>
where
    Dims: Axes,
{
    /// Output of the Min operation.
    type Values;
    /// Indices of Values.
    type Indices;
    /// Apply Min operation on given input.
    fn _min(self) -> (Self::Values, Self::Indices);
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
/// use zyx::device::cpu;
/// use zyx::prelude::*;
/// use zyx::shape::Sh3;
///
/// let mut device = cpu::Device::default();
/// 
/// let x = device.buffer([[[3, 2, 4], [3, 4, 2]], [[1, 4, 2], [5, 1, 6]]]);
/// let x = x.reshape::<Sh3<2, 1, 6>>();
/// println!("{}", x);
/// ```
/// 
/// ### Output
/// ```txt
/// [[3 2 4 2 4 2]
///  [1 4 2 5 1 6]]
/// ```
pub trait Reshape {
    /// Reshape to Sh
    fn reshape<Sh>(self) -> Self::Output
    where
        Sh: Shape,
        Self: Reshapable<Sh>;
}

impl<T> Reshape for T {
    fn reshape<Sh>(self) -> T::Output
    where
        Sh: Shape,
        T: Reshapable<Sh>
    {
        self._reshape()
    }
}

/// Reshapable
pub trait Reshapable<Sh>
where
    // TODO add check Sh::NUMEL == Self::Sh::NUMEL when stable rust supports it,
    // for now it is checked using static_assertions const_assert at buffers
    Sh: Shape,
{
    /// Output of the Reshape operation.
    type Output;
    /// Apply Reshape operation on given input.
    fn _reshape(self) -> Self::Output;
}

/// ## Expand tensor
/// 
/// Expands tensor to given shape, if some dimensions are 1.
/// These dimensions must be specified as second generic argument.
/// It is enforced at compile time that they will be correct.
/// For example, if you passed Ax1<0> in the following example,
/// the program would not compile.
/// Data is cloned to fill the required size.
/// 
/// ### Example
/// ```
/// use zyx::prelude::*;
/// use zyx::device::cpu;
/// use zyx::shape::{Sh3, Ax1};
///
/// let mut device = cpu::Device::default();
/// 
/// let x = device.buffer([[[3, 2, 4]], [[1, 4, 2]]]);
/// let x = x.expand::<Sh3<2, 3, 3>, Ax1<1>>();
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
pub trait Expand {
    /// Expand to Sh
    fn expand<Sh, Ax>(self) -> Self::Output
    where
        Sh: Shape,
        Ax: Axes,
        Self: HasShape,
        Sh: ReducableBy<Ax, Output = Self::Sh>,
        Self: Expandable<Sh, Ax>;
}

// For this, as well as [Permute] and so on we need to differentiate public and private API due to compiler reasons
impl<T> Expand for T {
    fn expand<Sh, Ax>(self) -> T::Output
    where
        Sh: Shape,
        Ax: Axes,
        Self: HasShape,
        Sh: ReducableBy<Ax, Output = <Self as HasShape>::Sh>,
        T: Expandable<Sh, Ax>,
    {
        self._expand()
    }
}

/// Expandable
pub trait Expandable<Sh, Ax>
where
    Sh: Shape,
    Ax: Axes,
    Self: HasShape,
    Sh: ReducableBy<Ax, Output = Self::Sh>,
{
    /// Output of the Expand operation.
    type Output;
    /// Apply Expand operation on given input.
    fn _expand(self) -> Self::Output;
}

/// ## Permute tensor
/// 
/// Shuffles tensors's dimensions in given order.
/// 
/// ### Example
/// ```
/// use zyx::device::cpu;
/// use zyx::prelude::*;
/// use zyx::shape::Ax3;
///
/// let mut device = cpu::Device::default();
/// 
/// let x = device.buffer([[[3, 2, 4]], [[1, 4, 2]]]);
/// let x = x.permute::<Ax3<2, 0, 1>>();
/// println!("{}", x);
/// # assert_eq!(&x.to_vec(), &[3, 1, 2, 4, 4, 2]);
/// # assert_eq!(x.shape(), [3, 2, 1]);
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
pub trait Permute {
    /// Permute shape with dims
    fn permute<Dims>(self) -> Self::Output
    where
        Dims: Axes,
        Self: Permutable<Dims>;
}

impl<T> Permute for T {
    fn permute<Dims>(self) -> T::Output
    where
        Dims: Axes,
        T: Permutable<Dims>
    {
        self._permute()
    }
}

/// Permutable
pub trait Permutable<Dims>
where
    Dims: Axes,
{
    /// Output of the Permute operation.
    type Output;
    /// Apply Permute operation on given input.
    fn _permute(self) -> Self::Output;
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
/// use zyx::prelude::*;
/// use zyx::device::cpu;
///
/// let mut device = cpu::Device::default();
///
/// let x = device.buffer([[3, 2, 4], [1, 4, 2]]);
/// let x = x.transpose();
/// println!("{}", x);
/// # assert_eq!(&x.to_vec(), &[3, 1, 2, 4, 4, 2]);
/// # assert_eq!(x.shape(), [3, 2]);
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
    T: Permutable<crate::shape::Ax2<-1, -2>>,
{
    type Output = T::Output;
    fn transpose(self) -> Self::Output {
        self.permute()
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
/// use zyx::device::cpu;
/// use zyx::prelude::*;
///
/// let mut device = cpu::Device::default();
///
/// let x = device.buffer([[3., 2., 4.], [1., 4., 2.]]);
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
/// use zyx::device::cpu;
/// use zyx::prelude::*;
///
/// let mut device = cpu::Device::default();
///
/// let x = device.buffer([[3., 2., 4.], [1., 4., 2.]]);
/// let y = device.buffer([[3., 2.], [4., 1.], [4., 2.]]);
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
pub trait Conv<const N: usize, const M: usize, Kernel = Self> {
    /// Output of the Conv operation.
    type Output;
    /// Apply Conv operation on given input.
    fn conv(self, kernel: Kernel, padding: crate::shape::Sh2<N, M>) -> Self::Output;
}

// This is only operation that requires alloc.
// Maybe figure a way how to do this without alloc?
// Can we return slice?
// Can gpu buffer return slice?
extern crate alloc;
/// ## IntoVec operation
/// 
/// Returns values from tensor as a Vec.
/// It must have row major order.
pub trait IntoVec<T> {
    /// Returns values from tensor as a Vec with row-major order.
    fn to_vec(&self) -> alloc::vec::Vec<T>;
}

/// Turn any datatype into [crate::tensor::Variable].
pub trait IntoVariable {
    /// Calling this function turns input into [crate::tensor::Variable] adding gradient in the process.
    fn with_grad(self) -> crate::tensor::Variable<Self>
    where
        Self: Sized;
}
