#[cfg(feature = "half")]
mod scalar_bf16;
mod scalar_bool;
#[cfg(feature = "complex")]
mod scalar_cf32;
#[cfg(feature = "complex")]
mod scalar_cf64;
#[cfg(feature = "half")]
mod scalar_f16;
mod scalar_f32;
mod scalar_f64;
mod scalar_i16;
mod scalar_i32;
mod scalar_i64;
mod scalar_i8;
mod scalar_u8;

#[cfg(feature = "complex")]
use num_complex::Complex;

#[cfg(feature = "half")]
use half::{bf16, f16};

use crate::dtype::DType;

/// Scalar trait is implemented for all [dtypes](DType)
pub trait Scalar: Copy + Clone + Sized + core::fmt::Debug + 'static {
    /// From bf16
    #[cfg(feature = "half")]
    fn from_bf16(t: bf16) -> Self;
    /// From f16
    #[cfg(feature = "half")]
    fn from_f16(t: f16) -> Self;
    /// From f32
    fn from_f32(t: f32) -> Self;
    /// From f64
    fn from_f64(t: f64) -> Self;
    /// From complex f32
    #[cfg(feature = "complex")]
    fn from_cf32(t: Complex<f32>) -> Self;
    /// From complex f64
    #[cfg(feature = "complex")]
    fn from_cf64(t: Complex<f64>) -> Self;
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
    /// From bool
    fn from_bool(t: bool) -> Self;
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
    /// Absolute value of self
    fn abs(self) -> Self;
    /// 1/self
    fn reciprocal(self) -> Self;
    /// Round down
    fn floor(self) -> Self;
    /// Neg
    fn neg(self) -> Self;
    /// ReLU
    fn relu(self) -> Self;
    /// Sin
    fn sin(self) -> Self;
    /// Cos
    fn cos(self) -> Self;
    /// Exp 2
    fn exp2(self) -> Self;
    /// Log 2
    fn log2(self) -> Self;
    /// Inv
    fn inv(self) -> Self;
    /// Not
    fn not(self) -> Self;
    /// Nonzero
    fn nonzero(self) -> Self;
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
    /// Compare less than
    fn cmpgt(self, rhs: Self) -> Self;
    /// Compare less than
    fn or(self, rhs: Self) -> Self;
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
    /// Cast into different dtype
    fn cast<T: Scalar>(self) -> T {
        use core::mem::transmute_copy as t;
        return unsafe {
            match Self::dtype() {
                #[cfg(feature = "half")]
                DType::BF16 => T::from_bf16(t(&self)),
                #[cfg(feature = "half")]
                DType::F16 => T::from_f16(t(&self)),
                DType::F32 => T::from_f32(t(&self)),
                DType::F64 => T::from_f64(t(&self)),
                #[cfg(feature = "complex")]
                DType::CF32 => T::from_cf32(t(&self)),
                #[cfg(feature = "complex")]
                DType::CF64 => T::from_cf64(t(&self)),
                DType::U8 => T::from_u8(t(&self)),
                DType::I8 => T::from_i8(t(&self)),
                DType::I16 => T::from_i16(t(&self)),
                DType::I32 => T::from_i32(t(&self)),
                DType::I64 => T::from_i64(t(&self)),
                DType::Bool => T::from_bool(t(&self)),
            }
        };
    }
}
