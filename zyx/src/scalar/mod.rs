mod scalar_bf16;
mod scalar_cf32;
mod scalar_cf64;
mod scalar_f16;
mod scalar_f32;
mod scalar_f64;
mod scalar_i16;
mod scalar_i32;
mod scalar_i64;
mod scalar_i8;
mod scalar_u8;

use crate::dtype::DType;
use half::{bf16, f16};

/// Scalar trait is implemented for all [dtypes](DType)
pub trait Scalar: Clone + Sized + core::fmt::Debug + 'static {
    /// From bf16
    fn from_bf16(t: bf16) -> Self;
    /// From f16
    fn from_f16(t: f16) -> Self;
    /// From f32
    fn from_f32(t: f32) -> Self;
    /// From f64
    fn from_f64(t: f64) -> Self;
    /// From u8
    fn from_u8(t: u8) -> Self;
    /// From i8
    fn from_i8(t: i8) -> Self;
    /// From i16
    fn from_i16(t: i16) -> Self;
    /// From i32
    fn from_i32(t: i32) -> Self;
    /// From i64
    fn from_i64(t: i64) -> Self;
    /// From little endian bytes
    fn from_le_bytes(bytes: &[u8]) -> Self;
    /// Get dtype of Self
    fn dtype() -> DType;
    /// Get zero of Self
    fn zero() -> Self;
    /// Get one of Self
    fn one() -> Self;
    /// Bute size of Self
    fn byte_size() -> usize;
    /// Convert self into f32
    fn into_f32(self) -> f32;
    /// Convert self into f64
    fn into_f64(self) -> f64;
    /// Convert self into i32
    fn into_i32(self) -> i32;
    /// Absolute value of self
    fn abs(self) -> Self;
    /// 1/self
    fn reciprocal(self) -> Self;
    /// Neg
    fn neg(self) -> Self;
    /// ReLU
    fn relu(self) -> Self;
    /// Sin
    fn sin(self) -> Self;
    /// Cos
    fn cos(self) -> Self;
    /// Ln
    fn ln(self) -> Self;
    /// Exp
    fn exp(self) -> Self;
    /// Tanh
    fn tanh(self) -> Self;
    /// Square root of this scalar.
    /// That this function may be imprecise.
    fn sqrt(self) -> Self;
    /// Add
    fn add(self, rhs: Self) -> Self;
    /// Sub
    fn sub(self, rhs: Self) -> Self;
    /// Mul
    fn mul(self, rhs: Self) -> Self;
    /// Div
    fn div(self, rhs: Self) -> Self;
    /// Pow
    fn pow(self, rhs: Self) -> Self;
    /// Compare less than
    fn cmplt(self, rhs: Self) -> Self;
    /// Max of two numbers
    fn max(self, rhs: Self) -> Self;
    /// Max value of this dtype
    fn max_value() -> Self;
    /// Min value of this dtype
    fn min_value() -> Self;
    /// Very small value of scalar, very close to zero
    fn epsilon() -> Self;
    /// Comparison for scalars,
    /// if they are floats, this checks for diffs > Self::epsilon()
    fn is_equal(self, rhs: Self) -> bool;
}
