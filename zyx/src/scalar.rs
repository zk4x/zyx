//! Trait describing required operations on scalar values

mod scalar_bf16;
mod scalar_bool;
mod scalar_f16;
mod scalar_f32;
mod scalar_f64;
mod scalar_f8;
mod scalar_i16;
mod scalar_i32;
mod scalar_i64;
mod scalar_i8;
mod scalar_u32;
mod scalar_u8;

use float8::F8E4M3;
use half::{bf16, f16};

use crate::dtype::DType;

/// Scalar trait is implemented for all [dtypes](DType)
pub trait Scalar: Copy + Clone + Sized + core::fmt::Debug + 'static + PartialEq {
    /// From bf16
    #[must_use]
    fn from_bf16(t: bf16) -> Self;
    /// From f8
    #[must_use]
    fn from_f8(t: F8E4M3) -> Self;
    /// From f16
    #[must_use]
    fn from_f16(t: f16) -> Self;
    /// From f32
    #[must_use]
    fn from_f32(t: f32) -> Self;
    /// From f64
    #[must_use]
    fn from_f64(t: f64) -> Self;
    /// From u8
    #[must_use]
    fn from_u8(t: u8) -> Self;
    /// From u32
    #[must_use]
    fn from_u32(t: u32) -> Self;
    /// From i8
    #[must_use]
    fn from_i8(t: i8) -> Self;
    /// From i16
    fn from_i16(t: i16) -> Self;
    #[must_use]
    /// From i32
    fn from_i32(t: i32) -> Self;
    /// From i64
    #[must_use]
    fn from_i64(t: i64) -> Self;
    /// From bool
    #[must_use]
    fn from_bool(t: bool) -> Self;
    /// From little endian bytes
    #[must_use]
    fn from_le_bytes(bytes: &[u8]) -> Self;
    /// Get dtype of Self
    #[must_use]
    fn dtype() -> DType;
    /// Get zero of Self
    #[must_use]
    fn zero() -> Self;
    /// Get one of Self
    #[must_use]
    fn one() -> Self;
    /// Bute size of Self
    #[must_use]
    fn byte_size() -> usize;
    /// Absolute value of self
    #[must_use]
    fn abs(self) -> Self;
    /// Neg
    #[must_use]
    fn neg(self) -> Self;
    /// `ReLU`
    #[must_use]
    fn relu(self) -> Self;
    /// Not
    #[must_use]
    fn not(self) -> Self;
    /// Nonzero
    #[must_use]
    fn nonzero(self) -> Self;
    /// Add
    #[must_use]
    fn add(self, rhs: Self) -> Self;
    /// Sub
    #[must_use]
    fn sub(self, rhs: Self) -> Self;
    /// Mul
    #[must_use]
    fn mul(self, rhs: Self) -> Self;
    /// Div
    #[must_use]
    fn div(self, rhs: Self) -> Self;
    /// Pow
    #[must_use]
    fn pow(self, rhs: Self) -> Self;
    /// Compare less than
    #[must_use]
    fn cmplt(self, rhs: Self) -> bool;
    /// Compare less than
    #[must_use]
    fn cmpgt(self, rhs: Self) -> bool;
    /// Compare less than
    #[must_use]
    fn or(self, rhs: Self) -> bool;
    /// And
    #[must_use]
    fn and(self, rhs: Self) -> bool;
    /// Max of two numbers
    #[must_use]
    fn max(self, rhs: Self) -> Self;
    /// Max value of this dtype
    #[must_use]
    fn max_value() -> Self;
    /// Min value of this dtype
    #[must_use]
    fn min_value() -> Self;
    /// Comparison for scalars,
    /// if they are floats, this checks for diffs > `Self::epsilon()`
    #[must_use]
    fn is_equal(self, rhs: Self) -> bool;
    /// Cast into different dtype
    #[must_use]
    fn cast<T: Scalar>(self) -> T {
        use core::mem::transmute_copy as t;
        unsafe {
            match Self::dtype() {
                DType::BF16 => T::from_bf16(t(&self)),
                DType::F8 => T::from_f8(t(&self)),
                DType::F16 => T::from_f16(t(&self)),
                DType::F32 => T::from_f32(t(&self)),
                DType::F64 => T::from_f64(t(&self)),
                DType::U8 => T::from_u8(t(&self)),
                DType::U32 => T::from_u32(t(&self)),
                DType::I8 => T::from_i8(t(&self)),
                DType::I16 => T::from_i16(t(&self)),
                DType::I32 => T::from_i32(t(&self)),
                DType::I64 => T::from_i64(t(&self)),
                DType::Bool => T::from_bool(t(&self)),
            }
        }
    }
    /// Very small value of scalar, very close to zero, zero in case of integers
    #[must_use]
    fn epsilon() -> Self {
        Self::zero()
    }
}

/// Float dtype
pub trait Float: Scalar {
    /// Round down
    #[must_use]
    fn floor(self) -> Self;
    /// 1/self
    #[must_use]
    fn reciprocal(self) -> Self;
    /// Sin
    #[must_use]
    fn sin(self) -> Self;
    /// Cos
    #[must_use]
    fn cos(self) -> Self;
    /// Exp 2
    #[must_use]
    fn exp2(self) -> Self;
    /// Log 2
    #[must_use]
    fn log2(self) -> Self;
    /// Square root of this scalar.
    #[must_use]
    fn sqrt(self) -> Self;
}