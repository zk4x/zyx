//! Traits for different operations you can to with Buffers.
//! Currently under development, but we take inspiration about which operations are most important from tinygrad.
//!

mod relu;
mod drelu;
mod exp;
mod ln;
mod tanh;
mod pow;
mod zeros;
mod ones;
mod min;
mod max;
mod get_shape;

/// ## Convert between devices and types
/// 
/// Create new Buffer on given device with given type
// needed because we can't use std::convert::From
// because it's foreign trait and it doesn't work
// when T == Self
pub trait ConvertFrom<T> {
    fn cfrom(x: T) -> Self;
}

pub trait ConvertInto<T> {
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
/// Create new Buffer initialized with zeros.
pub trait Zeros {
    fn zeros(shape: &[usize]) -> Self;
}

/// ## Ones operation
/// 
/// Create new Buffer initialized with ones.
pub trait Ones {
    fn ones(shape: &[usize]) -> Self;
}

/// ## IntoVec operation
/// 
/// Returns values from Buffer as a Vec. This accesses raw storage,
/// with the buffer::Cpu it will have row major order.
/// 
pub trait IntoVec<T> {
    fn to_vec(&self) -> Vec<T>;
}

/// ## FromVec operation
/// 
/// Creates new Buffer from given Vec and shape.
/// 
/// ### Example
/// ```
/// use zyx::prelude::*;
/// use zyx::accel::cpu::Buffer;
/// let x = Buffer::from_vec([2, 3, 1, 3].to_vec(), &[2, 2]);
/// println!("{}", x);
/// ```
/// ### Output
/// [2 3
///  1 3]
/// 
pub trait FromVec<T> {
    fn from_vec(data: Vec<T>, shape: &[usize]) -> Self;
}

/// ## IntoShape operation
/// 
/// Returns the shape of Buffer as an array of dimensions.
/// 
/// ### Example
/// ```
/// use zyx::{accel::cpu::Buffer, ops::{IntoShape, ConvertFrom}};
/// let x = Buffer::cfrom([2, 3, 1]);
/// let y = x.shape();
/// assert_eq!(y, [3]);
/// ```
pub trait IntoShape {
    fn shape(&self) -> Vec<usize>;
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
    type Output;
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
    type Output;
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
    type Output;
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
    type Output;
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
    type Output;
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
/// let y = x.sum(&[0]);
/// println!("{}", y);
/// ```
/// ### Output
/// ```txt
/// [7 4 2]
/// ```
/// 
pub trait Sum {
    type Output;
    fn sum(self, dims: &[i32]) -> Self::Output;
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
/// let y = x.max(&[0]);
/// println!("{}", y);
/// ```
/// ### Output
/// ```txt
/// [4 2 1]
/// ```
/// 
pub trait Max {
    type Output;
    fn max(self, dims: &[i32]) -> Self::Output;
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
/// let y = x.min(&[0]);
/// println!("{}", y);
/// ```
/// ### Output
/// ```txt
/// [3 2 1]
/// ```
/// 
pub trait Min {
    type Output;
    fn min(self, dims: &[i32]) -> Self::Output;
}

// Reshape simply changes shape of the Buffer.
// Permute also changes it's data ordering.
// Expand expands to given shape if some dimensions are one.
// PERMUTE, PAD, SHRINK, EXPAND, FLIP,
// Reshape, Permute, Slice, Expand, Flip   # movement ops

// Movement ops
/// ## Reshape Buffer
/// 
/// Reshaping changes Buffers's shape, while leaving data untouched.
/// 
/// ### Example
/// ```
/// use zyx::accel::cpu::Buffer;
/// use zyx::prelude::*;
/// 
/// let x = Buffer::cfrom([[[3, 2, 4], [3, 4, 2]], [[1, 4, 2], [5, 1, 6]]]);
/// let x = x.reshape(&[2, 1, 6]);
/// println!("{}", x);
/// ```
/// 
/// ### Output
/// ```txt
/// [[3 2 4 2 4 2]
///  [1 4 2 5 1 6]]
/// ```
/// 
pub trait Reshape {
    type Output;
    fn reshape(self, shape: &[usize]) -> Self::Output;
}

/// ## Expand Buffer
/// 
/// Expands Buffer to given shape, if some dimensions are 1. Data is cloned to fill the required size.
/// 
/// ### Example
/// ```
/// use zyx::accel::cpu;
/// use zyx::prelude::*;
/// 
/// let x = cpu::Buffer::cfrom([[[3, 2, 4]], [[1, 4, 2]]]);
/// let x = x.expand(&[2, 3, 3]);
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
    type Output;
    fn expand(self, shape: &[usize]) -> Self::Output;
}

/// ## Permute Buffer
/// 
/// Shuffles Buffer's dimensions in given order.
/// 
/// ### Example
/// ```
/// use zyx::accel::cpu::Buffer;
/// use zyx::prelude::*;
/// 
/// let x = Buffer::cfrom([[[3, 2, 4]], [[1, 4, 2]]]);
/// let x = x.permute(&[2, 0, 1]);
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
/// 
pub trait Permute {
    type Output;
    fn permute(self, dims: &[i32]) -> Self::Output;
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

/// Transpose is a subset of permute.
pub trait Transpose {
    type Output;
    fn transpose(self) -> Self::Output;
}

impl<T> Transpose for T
where
    T: Permute,
{
    type Output = T::Output;
    fn transpose(self) -> Self::Output {
        self.permute(&[-1, -2])
    }
}

// Binary ops are Add, Sub, Mul, Div, Pow, all with same size Buffers,
// use std::ops to implement them (except for Pow)

/// Pow operation
/// 
/// Calculate the power of the input Buffer to the given exponent Buffer.
pub trait Pow<Rhs = Self> {
    type Output;
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
/// 
pub trait MatMul<Rhs = Self> {
    type Output;
    fn matmul(self, rhs: Rhs) -> Self::Output;
}

// TODO: conv2d
/// ## 2D Convolution
/// 
/// Calculates 2D convodution.
/// 
/// NOTE: This API is not yet stable and may be subject to change
pub trait Conv<Kernel = Self> {
    type Output;
    fn conv(self, kernel: Kernel, padding: &[usize]) -> Self::Output;
}
