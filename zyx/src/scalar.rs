// Copyright (C) 2025 zk4x
// SPDX-License-Identifier: LGPL-3.0-only WITH Classpath-exception-2.0

//! Trait describing required operations on scalar values

use crate::dtype::DType;
use core::ops::{Add, Div, Mul, Neg, Rem, Sub};
use std::fmt;

#[allow(non_camel_case_types)]
/// bfloat16 (1 sign, 8 exponent, 7 mantissa)
#[derive(Clone, Copy, PartialEq, PartialOrd, Debug)]
pub struct bf16(pub u16);

#[allow(non_camel_case_types)]
/// IEEE half-precision float (1 sign, 5 exponent, 10 mantissa)
#[derive(Clone, Copy, PartialEq, PartialOrd, Debug)]
pub struct f16(pub u16);

impl bf16 {
    /// zero
    pub const ZERO: Self = Self(0);
    /// one
    pub const ONE: Self = Self(0x3f80);
    /// min
    pub const MIN: Self = Self(0xff7f);
    /// max
    pub const MAX: Self = Self(0x7f7f);
    /// min positive
    pub const MIN_POSITIVE: Self = Self(0x0080);

    /// to f32
    pub fn to_f32(self) -> f32 {
        f32::from_bits((self.0 as u32) << 16)
    }

    /// to f64
    pub fn to_f64(self) -> f64 {
        self.to_f32() as f64
    }

    /// from f32
    pub fn from_f32(x: f32) -> Self {
        Self((x.to_bits() >> 16) as u16)
    }

    /// from f64
    pub fn from_f64(x: f64) -> Self {
        Self::from_f32(x as f32)
    }

    /// to le bytes
    pub const fn to_le_bytes(self) -> [u8; 2] {
        self.0.to_le_bytes()
    }

    /// from le bytes
    pub fn from_le_bytes(bytes: [u8; 2]) -> Self {
        Self(u16::from_le_bytes(bytes))
    }

    /// to bits
    pub fn to_bits(self) -> u16 {
        self.0
    }

    /// is nan
    pub fn is_nan(self) -> bool {
        self.0 & 0x7fff > 0x7f80
    }

    /// is infinite
    pub fn is_infinite(self) -> bool {
        self.0 & 0x7fff == 0x7f80
    }

    /// abs
    pub fn abs(self) -> Self {
        Self(self.0 & 0x7fff)
    }

    /// max
    pub fn max(self, other: Self) -> Self {
        if self.to_f32() >= other.to_f32() { self } else { other }
    }
}

impl fmt::Display for bf16 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.to_f32())
    }
}

impl From<bf16> for f32 {
    fn from(x: bf16) -> Self {
        x.to_f32()
    }
}
impl From<bf16> for f64 {
    fn from(x: bf16) -> Self {
        x.to_f64()
    }
}

impl Neg for bf16 {
    type Output = Self;
    fn neg(self) -> Self {
        Self(self.0 ^ 0x8000)
    }
}
impl Add for bf16 {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        Self::from_f32(self.to_f32() + rhs.to_f32())
    }
}
impl Sub for bf16 {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        Self::from_f32(self.to_f32() - rhs.to_f32())
    }
}
impl Mul for bf16 {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self {
        Self::from_f32(self.to_f32() * rhs.to_f32())
    }
}
impl Div for bf16 {
    type Output = Self;
    fn div(self, rhs: Self) -> Self {
        Self::from_f32(self.to_f32() / rhs.to_f32())
    }
}
impl Rem for bf16 {
    type Output = Self;
    fn rem(self, rhs: Self) -> Self {
        Self::from_f32(self.to_f32() % rhs.to_f32())
    }
}

impl f16 {
    /// zero
    pub const ZERO: Self = Self(0x0000);
    /// one
    pub const ONE: Self = Self(0x3c00);
    /// min
    pub const MIN: Self = Self(0xfbff);
    /// max
    pub const MAX: Self = Self(0x7bff);
    /// epsilon
    pub const EPSILON: Self = Self(0x1400);

    /// to f32
    pub fn to_f32(self) -> f32 {
        let bits = self.0;
        let sign = if (bits >> 15) != 0 { -1.0f32 } else { 1.0f32 };
        let exp = (bits >> 10) & 0x1f;
        let mant = (bits & 0x3ff) as f32;
        if exp == 0 {
            if mant == 0.0 {
                return 0.0_f32.copysign(sign);
            }
            return sign * 2.0f32.powi(-14) * mant / 1024.0;
        }
        if exp == 31 {
            if mant == 0.0 {
                return if sign > 0.0 { f32::INFINITY } else { f32::NEG_INFINITY };
            }
            return f32::NAN;
        }
        sign * 2.0f32.powi(exp as i32 - 15) * (1.0 + mant / 1024.0)
    }

    /// to f64
    pub fn to_f64(self) -> f64 {
        self.to_f32() as f64
    }

    /// from f32
    pub fn from_f32(x: f32) -> Self {
        if x.is_nan() {
            return Self(0x7e00);
        }
        if x.is_infinite() {
            return if x.is_sign_positive() { Self(0x7c00) } else { Self(0xfc00) };
        }
        let sign = if x.is_sign_negative() { 0x8000u16 } else { 0x0000 };
        let x = x.abs();
        if x == 0.0 {
            return Self(sign);
        }
        if x < 2.0f32.powi(-24) {
            return Self(sign);
        }
        let exp = x.log2().floor() as i32;
        if exp < -14 {
            // Subnormal range (x < 2^-14): round mantissa; a carry
            // reaches the smallest normal (0x0400).
            if x < 2.0f32.powi(-24) {
                return Self(sign);
            }
            let m = (x / 2.0f32.powi(-24) + 0.5) as u16;
            if m >= 1024 {
                return Self(sign | 0x0400);
            }
            return Self(sign | (m & 0x3ff));
        }
        let exp = exp.clamp(-14, 15);
        let mant = x / 2.0f32.powi(exp);
        // Round mantissa; a carry of 1024 increments the exponent (it
        // must not be masked off by & 0x3ff). Saturating `as` keeps
        // huge inputs finite until the explicit inf clamp below.
        let m = ((mant - 1.0) * 1024.0 + 0.5) as i32;
        let (exp, m) = if m >= 1024 { (exp + 1, 0) } else { (exp, m) };
        if exp > 15 {
            return Self(sign | 0x7c00);
        }
        Self(sign | (((exp + 15) as u16) << 10) | (m as u16 & 0x3ff))
    }

    /// from f64
    pub fn from_f64(x: f64) -> Self {
        Self::from_f32(x as f32)
    }

    /// to le bytes
    pub const fn to_le_bytes(self) -> [u8; 2] {
        self.0.to_le_bytes()
    }

    /// from le bytes
    pub fn from_le_bytes(bytes: [u8; 2]) -> Self {
        Self(u16::from_le_bytes(bytes))
    }

    /// to bits
    pub fn to_bits(self) -> u16 {
        self.0
    }

    /// from bits
    pub fn from_bits(bits: u16) -> Self {
        Self(bits)
    }

    /// is nan
    pub fn is_nan(self) -> bool {
        self.0 & 0x7c00 == 0x7c00 && self.0 & 0x03ff != 0
    }

    /// is infinite
    pub fn is_infinite(self) -> bool {
        self.0 & 0x7fff == 0x7c00
    }

    /// abs
    pub fn abs(self) -> Self {
        Self(self.0 & 0x7fff)
    }

    /// max
    pub fn max(self, other: Self) -> Self {
        if self.to_f32() >= other.to_f32() { self } else { other }
    }

    /// min
    pub fn min(self, other: Self) -> Self {
        if self.to_f32() <= other.to_f32() { self } else { other }
    }
}

impl fmt::Display for f16 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.to_f32())
    }
}

impl From<f16> for f32 {
    fn from(x: f16) -> Self {
        x.to_f32()
    }
}
impl From<f16> for f64 {
    fn from(x: f16) -> Self {
        x.to_f64()
    }
}

impl Neg for f16 {
    type Output = Self;
    fn neg(self) -> Self {
        Self(self.0 ^ 0x8000)
    }
}
impl Add for f16 {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        Self::from_f32(self.to_f32() + rhs.to_f32())
    }
}
impl Sub for f16 {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        Self::from_f32(self.to_f32() - rhs.to_f32())
    }
}
impl Mul for f16 {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self {
        Self::from_f32(self.to_f32() * rhs.to_f32())
    }
}
impl Div for f16 {
    type Output = Self;
    fn div(self, rhs: Self) -> Self {
        Self::from_f32(self.to_f32() / rhs.to_f32())
    }
}
impl Rem for f16 {
    type Output = Self;
    fn rem(self, rhs: Self) -> Self {
        Self::from_f32(self.to_f32() % rhs.to_f32())
    }
}

#[allow(non_camel_case_types)]
/// FP8 E4M3 (1 sign, 4 exponent, 3 mantissa, bias 7). Finite-only:
/// no infinities (overflow saturates), NaN is 0x7F/0xFF.
#[derive(Clone, Copy, PartialEq, PartialOrd, Debug)]
pub struct f8e4m3(pub u8);

#[allow(non_camel_case_types)]
/// FP8 E5M2 (1 sign, 5 exponent, 2 mantissa, bias 15). IEEE inf/NaN.
#[derive(Clone, Copy, PartialEq, PartialOrd, Debug)]
pub struct f8e5m2(pub u8);

impl f8e4m3 {
    /// zero
    pub const ZERO: Self = Self(0x00);
    /// one
    pub const ONE: Self = Self(0x38);
    /// min (most negative)
    pub const MIN: Self = Self(0xfe);
    /// max
    pub const MAX: Self = Self(0x7e);

    /// to f32
    pub fn to_f32(self) -> f32 {
        let bits = self.0;
        let sign = if (bits >> 7) != 0 { -1.0f32 } else { 1.0f32 };
        let exp = (bits >> 3) & 0x0f;
        let mant = (bits & 0x07) as f32;
        if exp == 0 {
            if mant == 0.0 {
                return 0.0_f32.copysign(sign);
            }
            return sign * 2.0f32.powi(-6) * mant / 8.0;
        }
        if exp == 15 {
            // Only mantissa 7 is NaN; anything else is a normal 2^8 value.
            if mant == 7.0 {
                return f32::NAN;
            }
            return sign * 2.0f32.powi(8) * (1.0 + mant / 8.0);
        }
        sign * 2.0f32.powi(exp as i32 - 7) * (1.0 + mant / 8.0)
    }

    /// to f64
    pub fn to_f64(self) -> f64 {
        self.to_f32() as f64
    }

    /// from f32 (overflow saturates, never inf)
    pub fn from_f32(x: f32) -> Self {
        if x.is_nan() {
            return Self(0x7f);
        }
        let sign = if x.is_sign_negative() { 0x80u8 } else { 0x00 };
        let x = x.abs();
        if x == 0.0 {
            return Self(sign);
        }
        if x >= 448.0 {
            return Self(sign | 0x7e);
        }
        let exp = x.log2().floor() as i32;
        if exp < -6 {
            // Subnormal range: round mantissa; a carry reaches 0x08.
            let m = (x / 2.0f32.powi(-9) + 0.5) as u8;
            if m >= 8 {
                return Self(sign | 0x08);
            }
            return Self(sign | (m & 0x07));
        }
        let exp = exp.clamp(-6, 8);
        let m = ((x / 2.0f32.powi(exp) - 1.0) * 8.0 + 0.5) as i32;
        let (exp, m) = if m >= 8 { (exp + 1, 0) } else { (exp, m) };
        if exp > 8 {
            return Self(sign | 0x7e);
        }
        // Field 15 with mantissa 7 is NaN: saturate instead.
        if exp == 8 && m >= 7 {
            return Self(sign | 0x7e);
        }
        Self(sign | (((exp + 7) as u8) << 3) | (m as u8 & 0x07))
    }

    /// from f64
    pub fn from_f64(x: f64) -> Self {
        Self::from_f32(x as f32)
    }

    /// to le bytes
    pub const fn to_le_bytes(self) -> [u8; 1] {
        self.0.to_le_bytes()
    }

    /// from le bytes
    pub fn from_le_bytes(bytes: [u8; 1]) -> Self {
        Self(u8::from_le_bytes(bytes))
    }

    /// to bits
    pub const fn to_bits(self) -> u8 {
        self.0
    }

    /// from bits
    pub fn from_bits(bits: u8) -> Self {
        Self(bits)
    }

    /// is nan
    pub fn is_nan(self) -> bool {
        self.0 == 0x7f || self.0 == 0xff
    }

    /// is infinite (never true: E4M3 saturates instead)
    pub fn is_infinite(self) -> bool {
        false
    }

    /// abs
    pub fn abs(self) -> Self {
        Self(self.0 & 0x7f)
    }

    /// max
    pub fn max(self, other: Self) -> Self {
        if self.to_f32() >= other.to_f32() { self } else { other }
    }

    /// min
    pub fn min(self, other: Self) -> Self {
        if self.to_f32() <= other.to_f32() { self } else { other }
    }
}

impl fmt::Display for f8e4m3 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.to_f32())
    }
}

impl From<f8e4m3> for f32 {
    fn from(x: f8e4m3) -> Self {
        x.to_f32()
    }
}
impl From<f8e4m3> for f64 {
    fn from(x: f8e4m3) -> Self {
        x.to_f64()
    }
}

impl Neg for f8e4m3 {
    type Output = Self;
    fn neg(self) -> Self {
        Self(self.0 ^ 0x80)
    }
}
impl Add for f8e4m3 {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        Self::from_f32(self.to_f32() + rhs.to_f32())
    }
}
impl Sub for f8e4m3 {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        Self::from_f32(self.to_f32() - rhs.to_f32())
    }
}
impl Mul for f8e4m3 {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self {
        Self::from_f32(self.to_f32() * rhs.to_f32())
    }
}
impl Div for f8e4m3 {
    type Output = Self;
    fn div(self, rhs: Self) -> Self {
        Self::from_f32(self.to_f32() / rhs.to_f32())
    }
}
impl Rem for f8e4m3 {
    type Output = Self;
    fn rem(self, rhs: Self) -> Self {
        Self::from_f32(self.to_f32() % rhs.to_f32())
    }
}

impl f8e5m2 {
    /// zero
    pub const ZERO: Self = Self(0x00);
    /// one
    pub const ONE: Self = Self(0x3c);
    /// min (most negative)
    pub const MIN: Self = Self(0xfb);
    /// max
    pub const MAX: Self = Self(0x7b);

    /// to f32
    pub fn to_f32(self) -> f32 {
        let bits = self.0;
        let sign = if (bits >> 7) != 0 { -1.0f32 } else { 1.0f32 };
        let exp = (bits >> 2) & 0x1f;
        let mant = (bits & 0x03) as f32;
        if exp == 0 {
            if mant == 0.0 {
                return 0.0_f32.copysign(sign);
            }
            return sign * 2.0f32.powi(-14) * mant / 4.0;
        }
        if exp == 31 {
            if mant == 0.0 {
                return if sign > 0.0 { f32::INFINITY } else { f32::NEG_INFINITY };
            }
            return f32::NAN;
        }
        sign * 2.0f32.powi(exp as i32 - 15) * (1.0 + mant / 4.0)
    }

    /// to f64
    pub fn to_f64(self) -> f64 {
        self.to_f32() as f64
    }

    /// from f32
    pub fn from_f32(x: f32) -> Self {
        if x.is_nan() {
            return Self(0x7f);
        }
        if x.is_infinite() {
            return if x.is_sign_positive() { Self(0x7c) } else { Self(0xfc) };
        }
        let sign = if x.is_sign_negative() { 0x80u8 } else { 0x00 };
        let x = x.abs();
        if x == 0.0 {
            return Self(sign);
        }
        if x < 2.0f32.powi(-17) {
            return Self(sign);
        }
        let exp = x.log2().floor() as i32;
        if exp < -14 {
            // Subnormal range: round mantissa; a carry reaches 0x04.
            let m = (x / 2.0f32.powi(-16) + 0.5) as u8;
            if m >= 4 {
                return Self(sign | 0x04);
            }
            return Self(sign | (m & 0x03));
        }
        let exp = exp.clamp(-14, 15);
        let m = ((x / 2.0f32.powi(exp) - 1.0) * 4.0 + 0.5) as i32;
        let (exp, m) = if m >= 4 { (exp + 1, 0) } else { (exp, m) };
        if exp > 15 {
            return Self(sign | 0x7c);
        }
        Self(sign | (((exp + 15) as u8) << 2) | (m as u8 & 0x03))
    }

    /// from f64
    pub fn from_f64(x: f64) -> Self {
        Self::from_f32(x as f32)
    }

    /// to le bytes
    pub const fn to_le_bytes(self) -> [u8; 1] {
        self.0.to_le_bytes()
    }

    /// from le bytes
    pub fn from_le_bytes(bytes: [u8; 1]) -> Self {
        Self(u8::from_le_bytes(bytes))
    }

    /// to bits
    pub const fn to_bits(self) -> u8 {
        self.0
    }

    /// from bits
    pub fn from_bits(bits: u8) -> Self {
        Self(bits)
    }

    /// is nan
    pub fn is_nan(self) -> bool {
        self.0 & 0x7c == 0x7c && self.0 & 0x03 != 0
    }

    /// is infinite
    pub fn is_infinite(self) -> bool {
        self.0 & 0x7f == 0x7c
    }

    /// abs
    pub fn abs(self) -> Self {
        Self(self.0 & 0x7f)
    }

    /// max
    pub fn max(self, other: Self) -> Self {
        if self.to_f32() >= other.to_f32() { self } else { other }
    }

    /// min
    pub fn min(self, other: Self) -> Self {
        if self.to_f32() <= other.to_f32() { self } else { other }
    }
}

impl fmt::Display for f8e5m2 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.to_f32())
    }
}

impl From<f8e5m2> for f32 {
    fn from(x: f8e5m2) -> Self {
        x.to_f32()
    }
}
impl From<f8e5m2> for f64 {
    fn from(x: f8e5m2) -> Self {
        x.to_f64()
    }
}

impl Neg for f8e5m2 {
    type Output = Self;
    fn neg(self) -> Self {
        Self(self.0 ^ 0x80)
    }
}
impl Add for f8e5m2 {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        Self::from_f32(self.to_f32() + rhs.to_f32())
    }
}
impl Sub for f8e5m2 {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        Self::from_f32(self.to_f32() - rhs.to_f32())
    }
}
impl Mul for f8e5m2 {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self {
        Self::from_f32(self.to_f32() * rhs.to_f32())
    }
}
impl Div for f8e5m2 {
    type Output = Self;
    fn div(self, rhs: Self) -> Self {
        Self::from_f32(self.to_f32() / rhs.to_f32())
    }
}
impl Rem for f8e5m2 {
    type Output = Self;
    fn rem(self, rhs: Self) -> Self {
        Self::from_f32(self.to_f32() % rhs.to_f32())
    }
}

/// Scalar trait is implemented for all [dtypes](DType)
pub trait Scalar: Copy + Clone + Sized + core::fmt::Debug + 'static + PartialEq + Send + Sync + PartialOrd {
    /// From bf16
    #[must_use]
    fn from_bf16(t: bf16) -> Self;
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
    /// From u16
    #[must_use]
    fn from_u16(t: u16) -> Self;
    /// From u32
    #[must_use]
    fn from_u32(t: u32) -> Self;
    /// From u64
    #[must_use]
    fn from_u64(t: u64) -> Self;
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
    /// To native endian bytes
    #[must_use]
    #[allow(clippy::ptr_as_ptr)]
    fn to_ne_bytes(&self) -> &[u8];
    /// Get size in bits
    #[must_use]
    fn bit_size() -> u8 {
        Self::dtype().bit_size()
    }
    /// Get dtype of Self
    #[must_use]
    fn dtype() -> DType;
    /// Get zero of Self
    #[must_use]
    fn zero() -> Self;
    /// Get one of Self
    #[must_use]
    fn one() -> Self;
    /// Absolute value of self
    #[must_use]
    fn abs(self) -> Self;
    /// Neg
    #[must_use]
    fn neg(self) -> Self;
    /// Exp 2
    #[must_use]
    fn exp2(self) -> Self;
    /// Log 2
    #[must_use]
    fn log2(self) -> Self;
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
    /// Mod
    #[must_use]
    fn mod_(self, rhs: Self) -> Self;
    /// Compare less than
    #[must_use]
    fn cmplt(self, rhs: Self) -> bool;
    /// Compare less than
    #[must_use]
    fn cmpgt(self, rhs: Self) -> bool;
    /// Noteq
    #[must_use]
    fn noteq(self, rhs: Self) -> bool;
    /// Compare less than
    #[must_use]
    fn or(self, rhs: Self) -> bool;
    /// Bitxor
    #[must_use]
    fn bitxor(self, rhs: Self) -> Self;
    /// Bitor
    #[must_use]
    fn bitor(self, rhs: Self) -> Self;
    /// Bitand
    #[must_use]
    fn bitand(self, rhs: Self) -> Self;
    /// Bit shift left
    #[must_use]
    fn bitshiftleft(self, rhs: Self) -> Self;
    /// Bit shift rigt
    #[must_use]
    fn bitshiftright(self, rhs: Self) -> Self;
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
                DType::F16 => T::from_f16(t(&self)),
                DType::F32 => T::from_f32(t(&self)),
                DType::F64 => T::from_f64(t(&self)),
                DType::F8E4M3 => T::from_f32(f8e4m3::to_f32(t(&self))),
                DType::F8E5M2 => T::from_f32(f8e5m2::to_f32(t(&self))),
                DType::U8 => T::from_u8(t(&self)),
                DType::U16 => T::from_u16(t(&self)),
                DType::U32 => T::from_u32(t(&self)),
                DType::U64 => T::from_u64(t(&self)),
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
    /// Square root of this scalar.
    #[must_use]
    fn sqrt(self) -> Self;
    /// Truncate towards zero
    #[must_use]
    fn trunc(self) -> Self;
    /// Natural exponent e^x
    #[must_use]
    fn exp(self) -> Self;
    /// Natural logarithm
    #[must_use]
    fn ln(self) -> Self;
}

impl Scalar for bf16 {
    fn from_bf16(t: bf16) -> Self {
        t
    }

    fn from_f16(t: f16) -> Self {
        bf16::from_f32(t.into())
    }

    fn from_f32(t: f32) -> Self {
        bf16::from_f32(t)
    }

    fn from_f64(t: f64) -> Self {
        bf16::from_f64(t)
    }

    fn from_u8(t: u8) -> Self {
        bf16::from_f32(f32::from(t))
    }

    fn from_u16(t: u16) -> Self {
        bf16::from_f32(f32::from(t))
    }

    fn from_u32(t: u32) -> Self {
        bf16::from_f64(f64::from(t))
    }

    #[allow(clippy::cast_precision_loss)]
    fn from_u64(t: u64) -> Self {
        bf16::from_f64(t as f64)
    }

    fn from_i8(t: i8) -> Self {
        bf16::from_f32(f32::from(t))
    }

    fn from_i16(t: i16) -> Self {
        bf16::from_f32(f32::from(t))
    }

    #[allow(clippy::cast_possible_truncation)]
    fn from_i32(t: i32) -> Self {
        bf16::from_f32(t as f32)
    }

    #[allow(clippy::cast_possible_truncation)]
    fn from_i64(t: i64) -> Self {
        bf16::from_f32(t as f32)
    }

    fn from_bool(t: bool) -> Self {
        bf16::from_f32(f32::from(t))
    }

    fn from_le_bytes(bytes: &[u8]) -> Self {
        bf16::from_le_bytes([bytes[0], bytes[1]])
    }

    fn to_ne_bytes(&self) -> &[u8] {
        let x: *const Self = self;
        unsafe { std::slice::from_raw_parts(x.cast(), 2) }
    }

    fn dtype() -> DType {
        DType::BF16
    }

    fn zero() -> Self {
        bf16::ZERO
    }

    fn one() -> Self {
        bf16::ONE
    }

    fn abs(self) -> Self {
        self.max(-self)
    }

    fn neg(self) -> Self {
        -self
    }

    fn exp2(self) -> Self {
        bf16::from_f64(f64::from(self).exp2())
    }

    fn log2(self) -> Self {
        bf16::from_f64(f64::from(self).log2())
    }

    fn relu(self) -> Self {
        Scalar::max(self, Self::ZERO)
    }

    fn not(self) -> Self {
        bf16::from_f32(if f64::from(self) == 0.0 { 0.0 } else { 1.0 })
    }

    fn nonzero(self) -> Self {
        bf16::from_f32(if f64::from(self) == 0.0 { 0.0 } else { 1.0 })
    }

    fn add(self, rhs: Self) -> Self {
        self + rhs
    }

    fn sub(self, rhs: Self) -> Self {
        self - rhs
    }

    fn mul(self, rhs: Self) -> Self {
        self * rhs
    }

    fn div(self, rhs: Self) -> Self {
        self / rhs
    }

    fn pow(self, rhs: Self) -> Self {
        bf16::from_f64(f64::from(self).powf(f64::from(rhs)))
    }

    fn mod_(self, rhs: Self) -> Self {
        self % rhs
    }

    fn cmplt(self, rhs: Self) -> bool {
        self < rhs
    }

    fn cmpgt(self, rhs: Self) -> bool {
        self > rhs
    }

    fn noteq(self, rhs: Self) -> bool {
        self != rhs
    }

    fn or(self, rhs: Self) -> bool {
        self != Self::ZERO || rhs != Self::ZERO
    }

    fn bitxor(self, rhs: Self) -> Self {
        let a = f64::from(self) as i64;
        let b = f64::from(rhs) as i64;
        bf16::from_f32((a ^ b) as f32)
    }

    fn bitor(self, rhs: Self) -> Self {
        let a = f64::from(self) as i64;
        let b = f64::from(rhs) as i64;
        bf16::from_f32((a | b) as f32)
    }

    fn bitand(self, rhs: Self) -> Self {
        let a = f64::from(self) as i64;
        let b = f64::from(rhs) as i64;
        bf16::from_f32((a & b) as f32)
    }

    fn bitshiftleft(self, rhs: Self) -> Self {
        let a = f64::from(self) as i64;
        let b = f64::from(rhs) as i64;
        bf16::from_f32((a << b) as f32)
    }

    fn bitshiftright(self, rhs: Self) -> Self {
        let a = f64::from(self) as i64;
        let b = f64::from(rhs) as i64;
        bf16::from_f32((a >> b) as f32)
    }

    fn and(self, rhs: Self) -> bool {
        self != Self::ZERO && rhs != Self::ZERO
    }

    fn max(self, rhs: Self) -> Self {
        self.max(rhs)
    }

    fn max_value() -> Self {
        bf16::MAX
    }

    fn min_value() -> Self {
        bf16::MIN
    }

    fn is_equal(self, rhs: Self) -> bool {
        let a = self;
        let b = rhs;
        if a.is_nan() && b.is_nan() {
            return true;
        }
        if a == b {
            return true;
        }
        let diff = (a - b).abs();
        let max_abs = a.abs().max(b.abs());
        let rel_tol = bf16::from_f32(0.01) * max_abs;
        let abs_tol = bf16::from_f32(0.001);
        diff < rel_tol || diff < abs_tol
    }

    fn epsilon() -> Self {
        bf16::MIN_POSITIVE
    }
}

impl Float for bf16 {
    fn reciprocal(self) -> Self {
        bf16::ONE / self
    }

    fn floor(self) -> Self {
        bf16::from_f32(self.to_f32().floor())
    }

    fn sin(self) -> Self {
        bf16::from_f32(self.to_f32().sin())
    }

    fn cos(self) -> Self {
        bf16::from_f32(self.to_f32().cos())
    }

    fn sqrt(self) -> Self {
        bf16::from_f32(self.to_f32().sqrt())
    }

    fn trunc(self) -> Self {
        bf16::from_f32(self.to_f32().trunc())
    }

    fn exp(self) -> Self {
        bf16::from_f32(self.to_f32().exp())
    }

    fn ln(self) -> Self {
        bf16::from_f32(self.to_f32().ln())
    }
}

impl Scalar for f16 {
    fn from_bf16(t: bf16) -> Self {
        f16::from_f32(t.to_f32())
    }

    fn from_f16(t: f16) -> Self {
        f16::from_f32(t.to_f32())
    }

    fn from_f32(t: f32) -> Self {
        f16::from_f32(t)
    }

    fn from_f64(t: f64) -> Self {
        f16::from_f64(t)
    }

    fn from_u8(t: u8) -> Self {
        f16::from_f32(t as f32)
    }

    fn from_u16(t: u16) -> Self {
        f16::from_f32(t as f32)
    }

    fn from_u32(t: u32) -> Self {
        f16::from_f64(t.into())
    }

    fn from_u64(t: u64) -> Self {
        f16::from_f64(t as f64)
    }

    fn from_i8(t: i8) -> Self {
        f16::from_f32(t as f32)
    }

    fn from_i16(t: i16) -> Self {
        f16::from_f32(t as f32)
    }

    #[allow(clippy::cast_lossless)]
    fn from_i32(t: i32) -> Self {
        f16::from_f64(t as f64)
    }

    #[allow(clippy::cast_precision_loss)]
    fn from_i64(t: i64) -> Self {
        f16::from_f64(t as f64)
    }

    #[allow(clippy::cast_lossless)]
    fn from_bool(t: bool) -> Self {
        f16::from_f64(t as i8 as f64)
    }

    fn from_le_bytes(bytes: &[u8]) -> Self {
        f16::from_le_bytes([bytes[0], bytes[1]])
    }

    fn to_ne_bytes(&self) -> &[u8] {
        let i: *const Self = self;
        unsafe { std::slice::from_raw_parts(i.cast::<u8>(), std::mem::size_of::<Self>()) }
    }

    fn dtype() -> DType {
        DType::F16
    }

    fn zero() -> Self {
        f16::ZERO
    }

    fn one() -> Self {
        f16::ONE
    }

    fn abs(self) -> Self {
        self.max(-self)
    }

    fn neg(self) -> Self {
        -self
    }

    fn exp2(self) -> Self {
        f16::from_f32(self.to_f32().exp2())
    }

    fn log2(self) -> Self {
        f16::from_f32(self.to_f32().log2())
    }

    fn relu(self) -> Self {
        self.max(f16::ZERO)
    }

    fn not(self) -> Self {
        f16::from_f32(if f32::from(self) == 0.0 { 0.0 } else { 1.0 })
    }

    fn nonzero(self) -> Self {
        f16::from_f32(if f32::from(self) == 0.0 { 0.0 } else { 1.0 })
    }

    fn add(self, rhs: Self) -> Self {
        self + rhs
    }

    fn sub(self, rhs: Self) -> Self {
        self - rhs
    }

    fn mul(self, rhs: Self) -> Self {
        self * rhs
    }

    fn div(self, rhs: Self) -> Self {
        self / rhs
    }

    fn pow(self, rhs: Self) -> Self {
        f16::from_f32(self.to_f32().pow(rhs.to_f32()))
    }

    fn mod_(self, rhs: Self) -> Self {
        self % rhs
    }

    fn cmplt(self, rhs: Self) -> bool {
        self < rhs
    }

    fn cmpgt(self, rhs: Self) -> bool {
        self > rhs
    }

    fn noteq(self, rhs: Self) -> bool {
        self != rhs
    }

    fn or(self, rhs: Self) -> bool {
        self != f16::ZERO || rhs != f16::ZERO
    }

    fn bitxor(self, rhs: Self) -> Self {
        let ix = self.to_bits() ^ rhs.to_bits();
        f16::from_le_bytes([ix as u8, (ix >> 8) as u8])
    }

    fn bitor(self, rhs: Self) -> Self {
        let ix = self.to_bits() | rhs.to_bits();
        f16::from_le_bytes([ix as u8, (ix >> 8) as u8])
    }

    fn bitand(self, rhs: Self) -> Self {
        let ix = self.to_bits() & rhs.to_bits();
        f16::from_le_bytes([ix as u8, (ix >> 8) as u8])
    }

    fn bitshiftleft(self, rhs: Self) -> Self {
        let lhs_f32 = self.to_f32();
        let rhs_f32 = rhs.to_f32();
        let lhs_bits = lhs_f32.to_bits() as i32;
        let rhs_bits = rhs_f32.to_bits() as i32;
        let result = f32::from_bits((lhs_bits << rhs_bits) as u32);
        f16::from_f32(result)
    }

    fn bitshiftright(self, rhs: Self) -> Self {
        let lhs_f32 = self.to_f32();
        let rhs_f32 = rhs.to_f32();
        let lhs_bits = lhs_f32.to_bits() as i32;
        let rhs_bits = rhs_f32.to_bits() as i32;
        let result = f32::from_bits((lhs_bits >> rhs_bits) as u32);
        f16::from_f32(result)
    }

    fn and(self, rhs: Self) -> bool {
        self != f16::ZERO && rhs != f16::ZERO
    }

    fn max(self, rhs: Self) -> Self {
        f16::max(self, rhs)
    }

    fn max_value() -> Self {
        f16::MAX
    }

    fn min_value() -> Self {
        f16::MIN
    }

    fn is_equal(self, rhs: Self) -> bool {
        let a = self;
        let b = rhs;
        if a.is_nan() && b.is_nan() {
            return true;
        }
        if a == b {
            return true;
        }
        let diff = (a - b).abs();
        let max_abs = a.abs().max(b.abs());
        let rel_tol = f16::from_f32(0.01) * max_abs;
        let abs_tol = f16::from_f32(0.001);
        diff < rel_tol || diff < abs_tol
    }

    fn epsilon() -> Self {
        f16::EPSILON
    }
}

impl Float for f16 {
    fn reciprocal(self) -> Self {
        f16::ONE / self
    }

    fn sin(self) -> Self {
        f16::from_f32(self.to_f32().sin())
    }

    fn cos(self) -> Self {
        f16::from_f32(self.to_f32().cos())
    }

    fn sqrt(self) -> Self {
        f16::from_f32(self.to_f32().sqrt())
    }

    fn floor(self) -> Self {
        f16::from_f32(self.to_f32().floor())
    }

    fn trunc(self) -> Self {
        f16::from_f32(self.to_f32().trunc())
    }

    fn exp(self) -> Self {
        f16::from_f32(self.to_f32().exp())
    }

    fn ln(self) -> Self {
        f16::from_f32(self.to_f32().ln())
    }
}

impl Scalar for f8e4m3 {
    fn from_bf16(t: bf16) -> Self {
        f8e4m3::from_f32(t.to_f32())
    }

    fn from_f16(t: f16) -> Self {
        f8e4m3::from_f32(t.to_f32())
    }

    fn from_f32(t: f32) -> Self {
        f8e4m3::from_f32(t)
    }

    fn from_f64(t: f64) -> Self {
        f8e4m3::from_f64(t)
    }

    fn from_u8(t: u8) -> Self {
        f8e4m3::from_f32(t as f32)
    }

    fn from_u16(t: u16) -> Self {
        f8e4m3::from_f32(t as f32)
    }

    fn from_u32(t: u32) -> Self {
        f8e4m3::from_f64(t.into())
    }

    fn from_u64(t: u64) -> Self {
        f8e4m3::from_f64(t as f64)
    }

    fn from_i8(t: i8) -> Self {
        f8e4m3::from_f32(t as f32)
    }

    fn from_i16(t: i16) -> Self {
        f8e4m3::from_f32(t as f32)
    }

    #[allow(clippy::cast_lossless)]
    fn from_i32(t: i32) -> Self {
        f8e4m3::from_f64(t as f64)
    }

    #[allow(clippy::cast_precision_loss)]
    fn from_i64(t: i64) -> Self {
        f8e4m3::from_f64(t as f64)
    }

    #[allow(clippy::cast_lossless)]
    fn from_bool(t: bool) -> Self {
        f8e4m3::from_f64(t as i8 as f64)
    }

    fn from_le_bytes(bytes: &[u8]) -> Self {
        f8e4m3::from_le_bytes([bytes[0]])
    }

    fn to_ne_bytes(&self) -> &[u8] {
        let i: *const Self = self;
        unsafe { std::slice::from_raw_parts(i.cast::<u8>(), std::mem::size_of::<Self>()) }
    }

    fn dtype() -> DType {
        DType::F8E4M3
    }

    fn zero() -> Self {
        f8e4m3::ZERO
    }

    fn one() -> Self {
        f8e4m3::ONE
    }

    fn abs(self) -> Self {
        self.max(-self)
    }

    fn neg(self) -> Self {
        -self
    }

    fn exp2(self) -> Self {
        f8e4m3::from_f32(self.to_f32().exp2())
    }

    fn log2(self) -> Self {
        f8e4m3::from_f32(self.to_f32().log2())
    }

    fn relu(self) -> Self {
        self.max(f8e4m3::ZERO)
    }

    fn not(self) -> Self {
        f8e4m3::from_f32(if f32::from(self) == 0.0 { 0.0 } else { 1.0 })
    }

    fn nonzero(self) -> Self {
        f8e4m3::from_f32(if f32::from(self) == 0.0 { 0.0 } else { 1.0 })
    }

    fn add(self, rhs: Self) -> Self {
        self + rhs
    }

    fn sub(self, rhs: Self) -> Self {
        self - rhs
    }

    fn mul(self, rhs: Self) -> Self {
        self * rhs
    }

    fn div(self, rhs: Self) -> Self {
        self / rhs
    }

    fn pow(self, rhs: Self) -> Self {
        f8e4m3::from_f32(self.to_f32().powf(rhs.to_f32()))
    }

    fn mod_(self, rhs: Self) -> Self {
        self % rhs
    }

    fn cmplt(self, rhs: Self) -> bool {
        self < rhs
    }

    fn cmpgt(self, rhs: Self) -> bool {
        self > rhs
    }

    fn noteq(self, rhs: Self) -> bool {
        self != rhs
    }

    fn or(self, rhs: Self) -> bool {
        self != f8e4m3::ZERO || rhs != f8e4m3::ZERO
    }

    fn bitxor(self, rhs: Self) -> Self {
        f8e4m3::from_bits(self.to_bits() ^ rhs.to_bits())
    }

    fn bitor(self, rhs: Self) -> Self {
        f8e4m3::from_bits(self.to_bits() | rhs.to_bits())
    }

    fn bitand(self, rhs: Self) -> Self {
        f8e4m3::from_bits(self.to_bits() & rhs.to_bits())
    }

    fn bitshiftleft(self, rhs: Self) -> Self {
        f8e4m3::from_bits(self.to_bits().wrapping_shl(rhs.to_bits() as u32))
    }

    fn bitshiftright(self, rhs: Self) -> Self {
        f8e4m3::from_bits(self.to_bits().wrapping_shr(rhs.to_bits() as u32))
    }

    fn and(self, rhs: Self) -> bool {
        self != f8e4m3::ZERO && rhs != f8e4m3::ZERO
    }

    fn max(self, rhs: Self) -> Self {
        self.max(rhs)
    }

    fn max_value() -> Self {
        f8e4m3::MAX
    }

    fn min_value() -> Self {
        f8e4m3::MIN
    }

    fn is_equal(self, rhs: Self) -> bool {
        let a = self;
        let b = rhs;
        if a.is_nan() && b.is_nan() {
            return true;
        }
        if a == b {
            return true;
        }
        let diff = (a - b).abs();
        let max_abs = a.abs().max(b.abs());
        let rel_tol = f8e4m3::from_f32(0.05) * max_abs;
        let abs_tol = f8e4m3::from_f32(0.02);
        diff < rel_tol || diff < abs_tol
    }
}

impl Float for f8e4m3 {
    fn reciprocal(self) -> Self {
        f8e4m3::ONE / self
    }

    fn floor(self) -> Self {
        f8e4m3::from_f32(self.to_f32().floor())
    }

    fn sin(self) -> Self {
        f8e4m3::from_f32(self.to_f32().sin())
    }

    fn cos(self) -> Self {
        f8e4m3::from_f32(self.to_f32().cos())
    }

    fn sqrt(self) -> Self {
        f8e4m3::from_f32(self.to_f32().sqrt())
    }

    fn trunc(self) -> Self {
        f8e4m3::from_f32(self.to_f32().trunc())
    }

    fn exp(self) -> Self {
        f8e4m3::from_f32(self.to_f32().exp())
    }

    fn ln(self) -> Self {
        f8e4m3::from_f32(self.to_f32().ln())
    }
}

impl Scalar for f8e5m2 {
    fn from_bf16(t: bf16) -> Self {
        f8e5m2::from_f32(t.to_f32())
    }

    fn from_f16(t: f16) -> Self {
        f8e5m2::from_f32(t.to_f32())
    }

    fn from_f32(t: f32) -> Self {
        f8e5m2::from_f32(t)
    }

    fn from_f64(t: f64) -> Self {
        f8e5m2::from_f64(t)
    }

    fn from_u8(t: u8) -> Self {
        f8e5m2::from_f32(t as f32)
    }

    fn from_u16(t: u16) -> Self {
        f8e5m2::from_f32(t as f32)
    }

    fn from_u32(t: u32) -> Self {
        f8e5m2::from_f64(t.into())
    }

    fn from_u64(t: u64) -> Self {
        f8e5m2::from_f64(t as f64)
    }

    fn from_i8(t: i8) -> Self {
        f8e5m2::from_f32(t as f32)
    }

    fn from_i16(t: i16) -> Self {
        f8e5m2::from_f32(t as f32)
    }

    #[allow(clippy::cast_lossless)]
    fn from_i32(t: i32) -> Self {
        f8e5m2::from_f64(t as f64)
    }

    #[allow(clippy::cast_precision_loss)]
    fn from_i64(t: i64) -> Self {
        f8e5m2::from_f64(t as f64)
    }

    #[allow(clippy::cast_lossless)]
    fn from_bool(t: bool) -> Self {
        f8e5m2::from_f64(t as i8 as f64)
    }

    fn from_le_bytes(bytes: &[u8]) -> Self {
        f8e5m2::from_le_bytes([bytes[0]])
    }

    fn to_ne_bytes(&self) -> &[u8] {
        let i: *const Self = self;
        unsafe { std::slice::from_raw_parts(i.cast::<u8>(), std::mem::size_of::<Self>()) }
    }

    fn dtype() -> DType {
        DType::F8E5M2
    }

    fn zero() -> Self {
        f8e5m2::ZERO
    }

    fn one() -> Self {
        f8e5m2::ONE
    }

    fn abs(self) -> Self {
        self.max(-self)
    }

    fn neg(self) -> Self {
        -self
    }

    fn exp2(self) -> Self {
        f8e5m2::from_f32(self.to_f32().exp2())
    }

    fn log2(self) -> Self {
        f8e5m2::from_f32(self.to_f32().log2())
    }

    fn relu(self) -> Self {
        self.max(f8e5m2::ZERO)
    }

    fn not(self) -> Self {
        f8e5m2::from_f32(if f32::from(self) == 0.0 { 0.0 } else { 1.0 })
    }

    fn nonzero(self) -> Self {
        f8e5m2::from_f32(if f32::from(self) == 0.0 { 0.0 } else { 1.0 })
    }

    fn add(self, rhs: Self) -> Self {
        self + rhs
    }

    fn sub(self, rhs: Self) -> Self {
        self - rhs
    }

    fn mul(self, rhs: Self) -> Self {
        self * rhs
    }

    fn div(self, rhs: Self) -> Self {
        self / rhs
    }

    fn pow(self, rhs: Self) -> Self {
        f8e5m2::from_f32(self.to_f32().powf(rhs.to_f32()))
    }

    fn mod_(self, rhs: Self) -> Self {
        self % rhs
    }

    fn cmplt(self, rhs: Self) -> bool {
        self < rhs
    }

    fn cmpgt(self, rhs: Self) -> bool {
        self > rhs
    }

    fn noteq(self, rhs: Self) -> bool {
        self != rhs
    }

    fn or(self, rhs: Self) -> bool {
        self != f8e5m2::ZERO || rhs != f8e5m2::ZERO
    }

    fn bitxor(self, rhs: Self) -> Self {
        f8e5m2::from_bits(self.to_bits() ^ rhs.to_bits())
    }

    fn bitor(self, rhs: Self) -> Self {
        f8e5m2::from_bits(self.to_bits() | rhs.to_bits())
    }

    fn bitand(self, rhs: Self) -> Self {
        f8e5m2::from_bits(self.to_bits() & rhs.to_bits())
    }

    fn bitshiftleft(self, rhs: Self) -> Self {
        f8e5m2::from_bits(self.to_bits().wrapping_shl(rhs.to_bits() as u32))
    }

    fn bitshiftright(self, rhs: Self) -> Self {
        f8e5m2::from_bits(self.to_bits().wrapping_shr(rhs.to_bits() as u32))
    }

    fn and(self, rhs: Self) -> bool {
        self != f8e5m2::ZERO && rhs != f8e5m2::ZERO
    }

    fn max(self, rhs: Self) -> Self {
        self.max(rhs)
    }

    fn max_value() -> Self {
        f8e5m2::MAX
    }

    fn min_value() -> Self {
        f8e5m2::MIN
    }

    fn is_equal(self, rhs: Self) -> bool {
        let a = self;
        let b = rhs;
        if a.is_nan() && b.is_nan() {
            return true;
        }
        if a == b {
            return true;
        }
        let diff = (a - b).abs();
        let max_abs = a.abs().max(b.abs());
        let rel_tol = f8e5m2::from_f32(0.1) * max_abs;
        let abs_tol = f8e5m2::from_f32(0.05);
        diff < rel_tol || diff < abs_tol
    }
}

impl Float for f8e5m2 {
    fn reciprocal(self) -> Self {
        f8e5m2::ONE / self
    }

    fn floor(self) -> Self {
        f8e5m2::from_f32(self.to_f32().floor())
    }

    fn sin(self) -> Self {
        f8e5m2::from_f32(self.to_f32().sin())
    }

    fn cos(self) -> Self {
        f8e5m2::from_f32(self.to_f32().cos())
    }

    fn sqrt(self) -> Self {
        f8e5m2::from_f32(self.to_f32().sqrt())
    }

    fn trunc(self) -> Self {
        f8e5m2::from_f32(self.to_f32().trunc())
    }

    fn exp(self) -> Self {
        f8e5m2::from_f32(self.to_f32().exp())
    }

    fn ln(self) -> Self {
        f8e5m2::from_f32(self.to_f32().ln())
    }
}

impl Scalar for f32 {
    fn from_bf16(t: bf16) -> Self {
        t.into()
    }

    fn from_f16(t: f16) -> Self {
        t.into()
    }

    fn from_f32(t: f32) -> Self {
        t
    }

    #[allow(clippy::cast_possible_truncation)]
    fn from_f64(t: f64) -> Self {
        t as Self
    }

    fn from_u8(t: u8) -> Self {
        f32::from(t)
    }

    fn from_u16(t: u16) -> Self {
        t.into()
    }

    #[allow(clippy::cast_precision_loss)]
    fn from_u32(t: u32) -> Self {
        t as f32
    }

    #[allow(clippy::cast_precision_loss)]
    fn from_u64(t: u64) -> Self {
        t as f32
    }

    fn from_i8(t: i8) -> Self {
        f32::from(t)
    }

    fn from_i16(t: i16) -> Self {
        f32::from(t)
    }

    #[allow(clippy::cast_precision_loss)]
    fn from_i32(t: i32) -> Self {
        t as f32
    }

    #[allow(clippy::cast_precision_loss)]
    fn from_i64(t: i64) -> Self {
        t as f32
    }

    fn from_bool(t: bool) -> Self {
        f32::from(i8::from(t))
    }

    fn from_le_bytes(bytes: &[u8]) -> Self {
        f32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]])
    }

    fn to_ne_bytes(&self) -> &[u8] {
        let i: *const Self = self;
        unsafe { std::slice::from_raw_parts(i.cast::<u8>(), std::mem::size_of::<Self>()) }
    }

    fn dtype() -> DType {
        DType::F32
    }

    fn zero() -> Self {
        0.
    }

    fn one() -> Self {
        1.
    }

    fn abs(self) -> Self {
        self.abs()
    }

    fn neg(self) -> Self {
        -self
    }

    fn exp2(self) -> Self {
        f32::exp2(self)
    }

    fn log2(self) -> Self {
        self.log2()
    }

    fn relu(self) -> Self {
        self.max(0.)
    }

    fn not(self) -> Self {
        if self == 0. { 1. } else { 0. }
    }

    fn nonzero(self) -> Self {
        f32::from(i8::from(self != 0.))
    }

    fn add(self, rhs: Self) -> Self {
        self + rhs
    }

    fn sub(self, rhs: Self) -> Self {
        self - rhs
    }

    fn mul(self, rhs: Self) -> Self {
        self * rhs
    }

    fn div(self, rhs: Self) -> Self {
        self / rhs
    }

    fn pow(self, rhs: Self) -> Self {
        self.powf(rhs)
    }

    fn mod_(self, rhs: Self) -> Self {
        self % rhs
    }

    fn cmplt(self, rhs: Self) -> bool {
        self < rhs
    }

    fn cmpgt(self, rhs: Self) -> bool {
        self > rhs
    }

    fn noteq(self, rhs: Self) -> bool {
        !self.is_equal(rhs)
    }

    fn or(self, rhs: Self) -> bool {
        self != 0. || rhs != 0.
    }

    fn bitxor(self, rhs: Self) -> Self {
        let rhs_bits = rhs.to_bits();
        f32::from_bits(self.to_bits() ^ rhs_bits)
    }

    fn bitor(self, rhs: Self) -> Self {
        let rhs_bits = rhs.to_bits();
        f32::from_bits(self.to_bits() | rhs_bits)
    }

    fn bitand(self, rhs: Self) -> Self {
        let rhs_bits = rhs.to_bits();
        f32::from_bits(self.to_bits() & rhs_bits)
    }

    fn bitshiftleft(self, rhs: Self) -> Self {
        let rhs_shift = rhs.to_bits() & 0xFF;
        let ix = (self.to_bits() as u64) << rhs_shift;
        f32::from_bits(ix as u32)
    }

    fn bitshiftright(self, rhs: Self) -> Self {
        let rhs_shift = rhs.to_bits() & 0xFF;
        let ix = self.to_bits() >> rhs_shift;
        f32::from_bits(ix)
    }

    fn and(self, rhs: Self) -> bool {
        self != 0. && rhs != 0.
    }

    fn max(self, rhs: Self) -> Self {
        f32::max(self, rhs)
    }

    fn max_value() -> Self {
        f32::MAX
    }

    fn min_value() -> Self {
        f32::MIN
    }

    fn is_equal(self, rhs: Self) -> bool {
        let a = self;
        let b = rhs;
        if a.is_nan() && b.is_nan() {
            return true;
        }
        #[allow(clippy::float_cmp)]
        if a == b {
            return true;
        }
        let diff = (a - b).abs();
        let max_abs = a.abs().max(b.abs());
        let rel_tol = 1e-3 * max_abs; // relative tolerance for large numbers
        let abs_tol = 2e-7; // absolute tolerance for tiny numbers
        diff < rel_tol || diff < abs_tol
    }

    fn epsilon() -> Self {
        0.0001
    }
}

impl Float for f32 {
    fn sin(self) -> Self {
        //libm::sinf(self)
        //let b = 4f32 / PI;
        //let c = -4f32 / (PI * PI);
        //return -(b * self + c * self * if self < 0. { -self } else { self });
        f32::sin(self)
    }

    fn floor(self) -> Self {
        f32::floor(self)
    }

    fn cos(self) -> Self {
        //libm::cosf(self)
        //let mut x = self;
        //x *= 1. / (2. * PI);
        //x -= 0.25 + (x + 0.25).floor();
        //x *= 16.0 * (x.abs() - 0.5);
        //x += 0.225 * x * (x.abs() - 1.0);
        //return x;
        f32::cos(self)
    }

    fn sqrt(self) -> Self {
        // good enough (error of ~ 5%)
        /*if self >= 0. {
            Self::from_bits((self.to_bits() + 0x3f80_0000) >> 1)
        } else {
            Self::NAN
        }*/
        f32::sqrt(self)
    }

    fn reciprocal(self) -> Self {
        1.0 / self
    }

    fn trunc(self) -> Self {
        f32::trunc(self)
    }

    fn exp(self) -> Self {
        f32::exp(self)
    }

    fn ln(self) -> Self {
        f32::ln(self)
    }
}

impl Scalar for f64 {
    fn from_bf16(t: bf16) -> Self {
        t.into()
    }

    fn from_f16(t: f16) -> Self {
        t.into()
    }

    fn from_f32(t: f32) -> Self {
        f64::from(t)
    }

    fn from_f64(t: f64) -> Self {
        t
    }

    fn from_u8(t: u8) -> Self {
        f64::from(t)
    }

    fn from_u16(t: u16) -> Self {
        t.into()
    }

    fn from_u32(t: u32) -> Self {
        t.into()
    }

    #[allow(clippy::cast_precision_loss)]
    fn from_u64(t: u64) -> Self {
        t as f64
    }

    fn from_i8(t: i8) -> Self {
        t.into()
    }

    fn from_i16(t: i16) -> Self {
        t.into()
    }

    fn from_i32(t: i32) -> Self {
        t.into()
    }

    #[allow(clippy::cast_precision_loss)]
    fn from_i64(t: i64) -> Self {
        t as f64
    }

    fn from_bool(t: bool) -> Self {
        t.into()
    }

    fn from_le_bytes(bytes: &[u8]) -> Self {
        f64::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5], bytes[6], bytes[7]])
    }

    fn to_ne_bytes(&self) -> &[u8] {
        let i: *const Self = self;
        unsafe { std::slice::from_raw_parts(i.cast::<u8>(), std::mem::size_of::<Self>()) }
    }

    fn dtype() -> DType {
        DType::F64
    }

    fn zero() -> Self {
        0.
    }

    fn one() -> Self {
        1.
    }

    fn abs(self) -> Self {
        self.abs()
    }

    fn neg(self) -> Self {
        -self
    }

    fn exp2(self) -> Self {
        f64::exp2(self)
    }

    fn log2(self) -> Self {
        self.log2()
    }

    fn relu(self) -> Self {
        self.max(0.)
    }

    fn not(self) -> Self {
        if self == 0. { 1. } else { 0. }
    }

    fn nonzero(self) -> Self {
        u8::from(self != 0.).into()
    }

    fn add(self, rhs: Self) -> Self {
        self + rhs
    }

    fn sub(self, rhs: Self) -> Self {
        self - rhs
    }

    fn mul(self, rhs: Self) -> Self {
        self * rhs
    }

    fn div(self, rhs: Self) -> Self {
        self / rhs
    }

    fn pow(self, rhs: Self) -> Self {
        self.powf(rhs)
    }

    fn mod_(self, rhs: Self) -> Self {
        self % rhs
    }

    fn cmplt(self, rhs: Self) -> bool {
        self < rhs
    }

    fn cmpgt(self, rhs: Self) -> bool {
        self > rhs
    }

    fn noteq(self, rhs: Self) -> bool {
        !self.is_equal(rhs)
    }

    fn or(self, rhs: Self) -> bool {
        self != 0. || rhs != 0.
    }

    fn bitxor(self, rhs: Self) -> Self {
        f64::from_bits(self.to_bits() ^ rhs.to_bits())
    }

    fn bitor(self, rhs: Self) -> Self {
        f64::from_bits(self.to_bits() | rhs.to_bits())
    }

    fn bitand(self, rhs: Self) -> Self {
        f64::from_bits(self.to_bits() & rhs.to_bits())
    }

    fn bitshiftleft(self, rhs: Self) -> Self {
        let rhs_shift = (rhs.to_bits() & 0xFF) as u32;
        let ix = self.to_bits() << rhs_shift;
        f64::from_bits(ix)
    }

    fn bitshiftright(self, rhs: Self) -> Self {
        let rhs_shift = (rhs.to_bits() & 0xFF) as i32;
        let ix = (self.to_bits() >> rhs_shift) as u32;
        f64::from_bits(ix as u64)
    }

    fn and(self, rhs: Self) -> bool {
        self != 0. && rhs != 0.
    }

    fn max(self, rhs: Self) -> Self {
        f64::max(self, rhs)
    }

    fn max_value() -> Self {
        f64::MAX
    }

    fn min_value() -> Self {
        f64::MIN
    }

    fn is_equal(self, rhs: Self) -> bool {
        // Less than 0.1% error is OK
        (self == -f64::INFINITY && rhs == -f64::INFINITY) || (self - rhs).abs() <= self.abs() * 0.001
    }

    fn epsilon() -> Self {
        0.00001
    }
}

impl Float for f64 {
    fn reciprocal(self) -> Self {
        1.0 / self
    }

    fn floor(self) -> Self {
        self.floor()
    }

    fn sin(self) -> Self {
        f64::sin(self)
    }

    fn cos(self) -> Self {
        f64::cos(self)
    }

    fn sqrt(self) -> Self {
        f64::sqrt(self)
    }

    fn trunc(self) -> Self {
        self.trunc()
    }

    fn exp(self) -> Self {
        f64::exp(self)
    }

    fn ln(self) -> Self {
        f64::ln(self)
    }
}

impl Scalar for i8 {
    #[allow(clippy::cast_possible_truncation)]
    fn from_bf16(t: bf16) -> Self {
        let t: f32 = t.into();
        t as Self
    }

    #[allow(clippy::cast_possible_truncation)]
    fn from_f16(t: f16) -> Self {
        let t: f32 = t.into();
        t as Self
    }

    #[allow(clippy::cast_possible_truncation)]
    fn from_f32(t: f32) -> Self {
        t as Self
    }

    #[allow(clippy::cast_possible_truncation)]
    fn from_f64(t: f64) -> Self {
        t as Self
    }

    fn from_u8(t: u8) -> Self {
        t.try_into().unwrap()
    }

    fn from_u16(t: u16) -> Self {
        t.try_into().unwrap()
    }

    fn from_u32(t: u32) -> Self {
        t.try_into().unwrap()
    }

    fn from_u64(t: u64) -> Self {
        t.try_into().unwrap()
    }

    fn from_i8(t: i8) -> Self {
        t
    }

    fn from_i16(t: i16) -> Self {
        Self::try_from(t).unwrap()
    }

    fn from_i32(t: i32) -> Self {
        Self::try_from(t).unwrap()
    }

    fn from_i64(t: i64) -> Self {
        Self::try_from(t).unwrap()
    }

    fn from_bool(t: bool) -> Self {
        Self::from(t)
    }

    fn from_le_bytes(bytes: &[u8]) -> Self {
        i8::from_le_bytes([bytes[0]])
    }

    fn to_ne_bytes(&self) -> &[u8] {
        let i: *const Self = self;
        unsafe { std::slice::from_raw_parts(i.cast::<u8>(), std::mem::size_of::<Self>()) }
    }

    fn dtype() -> DType {
        DType::I8
    }

    fn zero() -> Self {
        0
    }

    fn one() -> Self {
        1
    }

    fn abs(self) -> Self {
        self.abs()
    }

    fn neg(self) -> Self {
        -self
    }

    fn exp2(self) -> Self {
        2i32.pow(self as u32) as i8
    }

    fn log2(self) -> Self {
        f64::from(self).log2() as i8
    }

    fn relu(self) -> Self {
        Scalar::max(self, 0)
    }

    fn not(self) -> Self {
        i8::from(self == 0)
    }

    fn nonzero(self) -> Self {
        i8::from(self != 0)
    }

    fn add(self, rhs: Self) -> Self {
        self + rhs
    }

    fn sub(self, rhs: Self) -> Self {
        self - rhs
    }

    fn mul(self, rhs: Self) -> Self {
        self * rhs
    }

    fn div(self, rhs: Self) -> Self {
        self / rhs
    }

    fn pow(self, rhs: Self) -> Self {
        if rhs >= 0 {
            return self.pow(rhs as u32);
        }
        if self == 1 {
            return 1;
        }
        if self == -1 {
            return if rhs % 2 == 0 { 1 } else { -1 };
        }
        0
    }

    fn mod_(self, rhs: Self) -> Self {
        self % rhs
    }

    fn cmplt(self, rhs: Self) -> bool {
        self < rhs
    }

    fn cmpgt(self, rhs: Self) -> bool {
        self > rhs
    }

    fn noteq(self, rhs: Self) -> bool {
        self != rhs
    }

    fn or(self, rhs: Self) -> bool {
        self != 0 || rhs != 0
    }

    fn bitxor(self, rhs: Self) -> Self {
        self ^ rhs
    }

    fn bitor(self, rhs: Self) -> Self {
        self | rhs
    }

    fn bitand(self, rhs: Self) -> Self {
        self & rhs
    }

    fn bitshiftleft(self, rhs: Self) -> Self {
        self.wrapping_shl(rhs as u32)
    }

    fn bitshiftright(self, rhs: Self) -> Self {
        self.wrapping_shr(rhs as u32)
    }

    fn and(self, rhs: Self) -> bool {
        self != 0 && rhs != 0
    }

    fn max(self, rhs: Self) -> Self {
        <i8 as Ord>::max(self, rhs)
    }

    fn max_value() -> Self {
        i8::MAX
    }

    fn min_value() -> Self {
        i8::MIN
    }

    fn is_equal(self, rhs: Self) -> bool {
        self == rhs
    }

    fn epsilon() -> Self {
        0
    }
}

impl Scalar for i16 {
    #[allow(clippy::cast_possible_truncation)]
    fn from_bf16(t: bf16) -> Self {
        t.to_f32() as i16
    }

    #[allow(clippy::cast_possible_truncation)]
    fn from_f16(t: f16) -> Self {
        t.to_f32() as i16
    }

    #[allow(clippy::cast_possible_truncation)]
    fn from_f32(t: f32) -> Self {
        t as i16
    }

    #[allow(clippy::cast_possible_truncation)]
    fn from_f64(t: f64) -> Self {
        t as i16
    }

    fn from_u8(t: u8) -> Self {
        t.into()
    }

    #[allow(clippy::cast_possible_truncation)]
    #[allow(clippy::cast_possible_wrap)]
    fn from_u16(t: u16) -> Self {
        t as i16
    }

    #[allow(clippy::cast_possible_truncation)]
    fn from_u32(t: u32) -> Self {
        t as i16
    }

    fn from_u64(t: u64) -> Self {
        t.try_into().unwrap()
    }

    fn from_i8(t: i8) -> Self {
        t.into()
    }

    fn from_i16(t: i16) -> Self {
        t
    }

    #[allow(clippy::cast_possible_truncation)]
    fn from_i32(t: i32) -> Self {
        t as i16
    }

    #[allow(clippy::cast_possible_truncation)]
    fn from_i64(t: i64) -> Self {
        t as i16
    }

    fn from_bool(t: bool) -> Self {
        t.into()
    }

    fn from_le_bytes(bytes: &[u8]) -> Self {
        i16::from_le_bytes([bytes[0], bytes[1]])
    }

    fn to_ne_bytes(&self) -> &[u8] {
        let i: *const Self = self;
        unsafe { std::slice::from_raw_parts(i.cast::<u8>(), std::mem::size_of::<Self>()) }
    }

    fn dtype() -> DType {
        DType::I16
    }

    fn zero() -> Self {
        0
    }

    fn one() -> Self {
        1
    }

    fn abs(self) -> Self {
        self.abs()
    }

    fn neg(self) -> Self {
        -self
    }

    fn exp2(self) -> Self {
        2i32.pow(self as u32) as i16
    }

    fn log2(self) -> Self {
        f64::from(self).log2() as i16
    }

    fn relu(self) -> Self {
        Scalar::max(self, 0)
    }

    fn not(self) -> Self {
        i16::from(self == 0)
    }

    fn nonzero(self) -> Self {
        i16::from(self != 0)
    }

    fn add(self, rhs: Self) -> Self {
        self + rhs
    }

    fn sub(self, rhs: Self) -> Self {
        self - rhs
    }

    fn mul(self, rhs: Self) -> Self {
        self * rhs
    }

    fn div(self, rhs: Self) -> Self {
        self / rhs
    }

    fn pow(self, rhs: Self) -> Self {
        if rhs >= 0 {
            return self.pow(rhs as u32);
        }
        if self == 1 {
            return 1;
        }
        if self == -1 {
            return if rhs % 2 == 0 { 1 } else { -1 };
        }
        0
    }

    fn mod_(self, rhs: Self) -> Self {
        self % rhs
    }

    fn cmplt(self, rhs: Self) -> bool {
        self < rhs
    }

    fn cmpgt(self, rhs: Self) -> bool {
        self > rhs
    }

    fn noteq(self, rhs: Self) -> bool {
        self != rhs
    }

    fn or(self, rhs: Self) -> bool {
        self != 0 || rhs != 0
    }

    fn bitxor(self, rhs: Self) -> Self {
        self ^ rhs
    }

    fn bitor(self, rhs: Self) -> Self {
        self | rhs
    }

    fn bitand(self, rhs: Self) -> Self {
        self & rhs
    }

    fn bitshiftleft(self, rhs: Self) -> Self {
        self.wrapping_shl(rhs as u32)
    }

    fn bitshiftright(self, rhs: Self) -> Self {
        self.wrapping_shr(rhs as u32)
    }

    fn and(self, rhs: Self) -> bool {
        self != 0 && rhs != 0
    }

    fn max(self, rhs: Self) -> Self {
        Ord::max(self, rhs)
    }

    fn max_value() -> Self {
        i16::MAX
    }

    fn min_value() -> Self {
        i16::MIN
    }

    fn is_equal(self, rhs: Self) -> bool {
        self == rhs
    }

    fn epsilon() -> Self {
        0
    }
}

impl Scalar for i32 {
    fn from_bf16(t: bf16) -> Self {
        t.to_f32() as i32
    }

    fn from_f16(t: f16) -> Self {
        t.to_f32() as i32
    }

    #[allow(clippy::cast_possible_truncation)]
    fn from_f32(t: f32) -> Self {
        t as i32
    }

    #[allow(clippy::cast_possible_truncation)]
    fn from_f64(t: f64) -> Self {
        t as i32
    }

    fn from_u8(t: u8) -> Self {
        t.into()
    }

    fn from_u16(t: u16) -> Self {
        t.into()
    }

    fn from_u32(t: u32) -> Self {
        i32::try_from(t).unwrap()
    }

    fn from_u64(t: u64) -> Self {
        t.try_into().unwrap()
    }

    fn from_i8(t: i8) -> Self {
        t.into()
    }

    fn from_i16(t: i16) -> Self {
        t.into()
    }

    fn from_i32(t: i32) -> Self {
        t
    }

    #[allow(clippy::cast_possible_truncation)]
    fn from_i64(t: i64) -> Self {
        t as i32
    }

    fn from_bool(t: bool) -> Self {
        t.into()
    }

    fn from_le_bytes(bytes: &[u8]) -> Self {
        i32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]])
    }

    fn to_ne_bytes(&self) -> &[u8] {
        let i: *const i32 = self;
        unsafe { std::slice::from_raw_parts(i.cast::<u8>(), std::mem::size_of::<i32>()) }
    }

    fn dtype() -> DType {
        DType::I32
    }

    fn zero() -> Self {
        0
    }

    fn one() -> Self {
        1
    }

    fn abs(self) -> Self {
        self.abs()
    }

    fn neg(self) -> Self {
        -self
    }

    fn exp2(self) -> Self {
        2i32.pow(self as u32)
    }

    fn log2(self) -> Self {
        f64::from(self).log2() as i32
    }

    fn relu(self) -> Self {
        Scalar::max(self, 0)
    }

    fn not(self) -> Self {
        i32::from(self == 0)
    }

    fn nonzero(self) -> Self {
        i32::from(self != 0)
    }

    fn add(self, rhs: Self) -> Self {
        self + rhs
    }

    fn sub(self, rhs: Self) -> Self {
        self - rhs
    }

    fn mul(self, rhs: Self) -> Self {
        self * rhs
    }

    fn div(self, rhs: Self) -> Self {
        self / rhs
    }

    fn pow(self, rhs: Self) -> Self {
        i32::pow(self, u32::try_from(rhs).unwrap())
    }

    fn mod_(self, rhs: Self) -> Self {
        self % rhs
    }

    fn cmplt(self, rhs: Self) -> bool {
        self < rhs
    }

    fn cmpgt(self, rhs: Self) -> bool {
        self > rhs
    }

    fn noteq(self, rhs: Self) -> bool {
        self != rhs
    }

    fn or(self, rhs: Self) -> bool {
        self != 0 || rhs != 0
    }

    fn bitxor(self, rhs: Self) -> Self {
        self ^ rhs
    }

    fn bitor(self, rhs: Self) -> Self {
        self | rhs
    }

    fn bitand(self, rhs: Self) -> Self {
        self & rhs
    }

    fn bitshiftleft(self, rhs: Self) -> Self {
        self.wrapping_shl(rhs as u32)
    }

    fn bitshiftright(self, rhs: Self) -> Self {
        self.wrapping_shr(rhs as u32)
    }

    fn and(self, rhs: Self) -> bool {
        self != 0 && rhs != 0
    }

    fn max(self, rhs: Self) -> Self {
        <i32 as Ord>::max(self, rhs)
    }

    fn max_value() -> Self {
        i32::MAX
    }

    fn min_value() -> Self {
        i32::MIN
    }

    fn is_equal(self, rhs: Self) -> bool {
        self == rhs
    }

    fn epsilon() -> Self {
        0
    }
}

impl Scalar for i64 {
    fn from_bf16(t: bf16) -> Self {
        t.to_f32() as i64
    }

    fn from_f16(t: f16) -> Self {
        t.to_f32() as i64
    }

    #[allow(clippy::cast_possible_truncation)]
    fn from_f32(t: f32) -> Self {
        t as Self
    }

    #[allow(clippy::cast_possible_truncation)]
    fn from_f64(t: f64) -> Self {
        t as Self
    }

    fn from_u8(t: u8) -> Self {
        t.into()
    }

    fn from_u16(t: u16) -> Self {
        t.into()
    }

    fn from_u32(t: u32) -> Self {
        t.into()
    }

    fn from_u64(t: u64) -> Self {
        t.try_into().unwrap()
    }

    fn from_i8(t: i8) -> Self {
        t.into()
    }

    fn from_i16(t: i16) -> Self {
        t.into()
    }

    fn from_i32(t: i32) -> Self {
        t.into()
    }

    fn from_i64(t: i64) -> Self {
        t
    }

    fn from_bool(t: bool) -> Self {
        t.into()
    }

    fn from_le_bytes(bytes: &[u8]) -> Self {
        i64::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5], bytes[6], bytes[7]])
    }

    fn to_ne_bytes(&self) -> &[u8] {
        let i: *const Self = self;
        unsafe { std::slice::from_raw_parts(i.cast::<u8>(), std::mem::size_of::<Self>()) }
    }

    fn dtype() -> DType {
        DType::I64
    }

    fn zero() -> Self {
        0
    }

    fn one() -> Self {
        1
    }

    fn abs(self) -> Self {
        self.abs()
    }

    fn neg(self) -> Self {
        -self
    }

    fn exp2(self) -> Self {
        2i64.pow(self as u32)
    }

    fn log2(self) -> Self {
        self as f64 as i64
    }

    fn relu(self) -> Self {
        Scalar::max(self, 0)
    }

    fn not(self) -> Self {
        i64::from(self == 0)
    }

    fn nonzero(self) -> Self {
        i64::from(self != 0)
    }

    fn add(self, rhs: Self) -> Self {
        self + rhs
    }

    fn sub(self, rhs: Self) -> Self {
        self - rhs
    }

    fn mul(self, rhs: Self) -> Self {
        self * rhs
    }

    fn div(self, rhs: Self) -> Self {
        self / rhs
    }

    fn pow(self, rhs: Self) -> Self {
        i64::pow(self, u32::try_from(rhs).unwrap())
    }

    fn mod_(self, rhs: Self) -> Self {
        self % rhs
    }

    fn cmplt(self, rhs: Self) -> bool {
        self < rhs
    }

    fn cmpgt(self, rhs: Self) -> bool {
        self > rhs
    }

    fn noteq(self, rhs: Self) -> bool {
        self != rhs
    }

    fn or(self, rhs: Self) -> bool {
        self != 0 || rhs != 0
    }

    fn bitxor(self, rhs: Self) -> Self {
        self ^ rhs
    }

    fn bitor(self, rhs: Self) -> Self {
        self | rhs
    }

    fn bitand(self, rhs: Self) -> Self {
        self & rhs
    }

    fn bitshiftleft(self, rhs: Self) -> Self {
        self.wrapping_shl(rhs as u32)
    }

    fn bitshiftright(self, rhs: Self) -> Self {
        self.wrapping_shr(rhs as u32)
    }

    fn and(self, rhs: Self) -> bool {
        self != 0 && rhs != 0
    }

    fn max(self, rhs: Self) -> Self {
        <i64 as Ord>::max(self, rhs)
    }

    fn max_value() -> Self {
        Self::MAX
    }

    fn min_value() -> Self {
        Self::MIN
    }

    fn is_equal(self, rhs: Self) -> bool {
        self == rhs
    }

    fn epsilon() -> Self {
        0
    }
}

impl Scalar for u8 {
    fn from_bf16(t: bf16) -> Self {
        t.to_f32() as u32 as u8
    }

    fn from_f16(t: f16) -> Self {
        t.to_f32() as u32 as u8
    }

    fn from_f32(t: f32) -> Self {
        t as u32 as u8
    }

    fn from_f64(t: f64) -> Self {
        t as u32 as u8
    }

    fn from_u8(t: u8) -> Self {
        t
    }

    fn from_u16(t: u16) -> Self {
        t.try_into().unwrap()
    }

    fn from_u32(t: u32) -> Self {
        t.try_into().unwrap()
    }

    fn from_u64(t: u64) -> Self {
        t.try_into().unwrap()
    }

    fn from_i8(t: i8) -> Self {
        t.try_into().unwrap()
    }

    fn from_i16(t: i16) -> Self {
        t.try_into().unwrap()
    }

    fn from_i32(t: i32) -> Self {
        t.try_into().unwrap()
    }

    fn from_i64(t: i64) -> Self {
        t.try_into().unwrap()
    }

    fn from_bool(t: bool) -> Self {
        t.into()
    }

    fn from_le_bytes(bytes: &[u8]) -> Self {
        u8::from_le_bytes([bytes[0]])
    }

    fn to_ne_bytes(&self) -> &[u8] {
        let i: *const Self = self;
        unsafe { std::slice::from_raw_parts(i.cast::<u8>(), std::mem::size_of::<Self>()) }
    }

    fn dtype() -> DType {
        DType::U8
    }

    fn zero() -> Self {
        0
    }

    fn one() -> Self {
        1
    }

    fn abs(self) -> Self {
        self
    }

    fn neg(self) -> Self {
        self.wrapping_neg()
    }

    fn exp2(self) -> Self {
        if self <= 31 { 2u32.pow(self as u32) as u8 } else { 255 }
    }

    fn log2(self) -> Self {
        self.ilog2() as u8
    }

    fn relu(self) -> Self {
        self
    }

    fn not(self) -> Self {
        u8::from(self == 0)
    }

    fn nonzero(self) -> Self {
        u8::from(self != 0)
    }

    fn add(self, rhs: Self) -> Self {
        self + rhs
    }

    fn sub(self, rhs: Self) -> Self {
        self - rhs
    }

    fn mul(self, rhs: Self) -> Self {
        self * rhs
    }

    fn div(self, rhs: Self) -> Self {
        self / rhs
    }

    fn pow(self, rhs: Self) -> Self {
        Self::pow(self, u32::from(rhs))
    }

    fn mod_(self, rhs: Self) -> Self {
        self % rhs
    }

    fn cmplt(self, rhs: Self) -> bool {
        self < rhs
    }

    fn cmpgt(self, rhs: Self) -> bool {
        self > rhs
    }

    fn noteq(self, rhs: Self) -> bool {
        self != rhs
    }

    fn or(self, rhs: Self) -> bool {
        self != 0 || rhs != 0
    }

    fn bitxor(self, rhs: Self) -> Self {
        self ^ rhs
    }

    fn bitor(self, rhs: Self) -> Self {
        self | rhs
    }

    fn bitand(self, rhs: Self) -> Self {
        self & rhs
    }

    fn bitshiftleft(self, rhs: Self) -> Self {
        self.wrapping_shl(rhs as u32)
    }

    fn bitshiftright(self, rhs: Self) -> Self {
        self.wrapping_shr(rhs as u32)
    }

    fn and(self, rhs: Self) -> bool {
        self != 0 && rhs != 0
    }

    fn max(self, rhs: Self) -> Self {
        Ord::max(self, rhs)
    }

    fn max_value() -> Self {
        u8::MAX
    }

    fn min_value() -> Self {
        u8::MIN
    }

    fn is_equal(self, rhs: Self) -> bool {
        self == rhs
    }

    fn epsilon() -> Self {
        0
    }
}

impl Scalar for u16 {
    fn from_bf16(t: bf16) -> Self {
        t.to_f32() as u32 as u16
    }

    fn from_f16(t: f16) -> Self {
        t.to_f32() as u32 as u16
    }

    fn from_f32(t: f32) -> Self {
        t as u32 as u16
    }

    fn from_f64(t: f64) -> Self {
        t as u32 as u16
    }

    fn from_u8(t: u8) -> Self {
        t.into()
    }

    fn from_u16(t: u16) -> Self {
        t
    }

    fn from_u32(t: u32) -> Self {
        t.try_into().unwrap()
    }

    fn from_u64(t: u64) -> Self {
        t.try_into().unwrap()
    }

    fn from_i8(t: i8) -> Self {
        t.try_into().unwrap()
    }

    fn from_i16(t: i16) -> Self {
        t.try_into().unwrap()
    }

    fn from_i32(t: i32) -> Self {
        t.try_into().unwrap()
    }

    fn from_i64(t: i64) -> Self {
        t.try_into().unwrap()
    }

    fn from_bool(t: bool) -> Self {
        t.into()
    }

    fn from_le_bytes(bytes: &[u8]) -> Self {
        Self::from_le_bytes([bytes[0], bytes[1]])
    }

    fn to_ne_bytes(&self) -> &[u8] {
        let i: *const Self = self;
        unsafe { std::slice::from_raw_parts(i.cast::<u8>(), std::mem::size_of::<Self>()) }
    }

    fn dtype() -> DType {
        DType::U16
    }

    fn zero() -> Self {
        0
    }

    fn one() -> Self {
        1
    }

    fn abs(self) -> Self {
        self
    }

    fn neg(self) -> Self {
        self.wrapping_neg()
    }

    fn exp2(self) -> Self {
        if self <= 31 { 2u32.pow(self as u32) as u16 } else { 65535 }
    }

    fn log2(self) -> Self {
        self.ilog2() as u16
    }

    fn relu(self) -> Self {
        self
    }

    fn not(self) -> Self {
        u16::from(self == 0)
    }

    fn nonzero(self) -> Self {
        u16::from(self != 0)
    }

    fn add(self, rhs: Self) -> Self {
        self + rhs
    }

    fn sub(self, rhs: Self) -> Self {
        self - rhs
    }

    fn mul(self, rhs: Self) -> Self {
        self * rhs
    }

    fn div(self, rhs: Self) -> Self {
        self / rhs
    }

    fn pow(self, rhs: Self) -> Self {
        Self::pow(self, u32::from(rhs))
    }

    fn mod_(self, rhs: Self) -> Self {
        self % rhs
    }

    fn cmplt(self, rhs: Self) -> bool {
        self < rhs
    }

    fn cmpgt(self, rhs: Self) -> bool {
        self > rhs
    }

    fn noteq(self, rhs: Self) -> bool {
        self != rhs
    }

    fn or(self, rhs: Self) -> bool {
        self != 0 || rhs != 0
    }

    fn bitxor(self, rhs: Self) -> Self {
        self ^ rhs
    }

    fn bitor(self, rhs: Self) -> Self {
        self | rhs
    }

    fn bitand(self, rhs: Self) -> Self {
        self & rhs
    }

    fn bitshiftleft(self, rhs: Self) -> Self {
        self.wrapping_shl(rhs as u32)
    }

    fn bitshiftright(self, rhs: Self) -> Self {
        self.wrapping_shr(rhs as u32)
    }

    fn and(self, rhs: Self) -> bool {
        self != 0 && rhs != 0
    }

    fn max(self, rhs: Self) -> Self {
        Ord::max(self, rhs)
    }

    fn max_value() -> Self {
        Self::MAX
    }

    fn min_value() -> Self {
        Self::MIN
    }

    fn is_equal(self, rhs: Self) -> bool {
        self == rhs
    }

    fn epsilon() -> Self {
        0
    }

    fn cast<T: Scalar>(self) -> T {
        use core::mem::transmute_copy as t;
        unsafe {
            match Self::dtype() {
                DType::BF16 => T::from_bf16(t(&self)),
                DType::F16 => T::from_f16(t(&self)),
                DType::F32 => T::from_f32(t(&self)),
                DType::F64 => T::from_f64(t(&self)),
                DType::F8E4M3 => T::from_f32(f8e4m3::to_f32(t(&self))),
                DType::F8E5M2 => T::from_f32(f8e5m2::to_f32(t(&self))),
                DType::U8 => T::from_u8(t(&self)),
                DType::U16 => T::from_u16(t(&self)),
                DType::U32 => T::from_u32(t(&self)),
                DType::U64 => T::from_u64(t(&self)),
                DType::I8 => T::from_i8(t(&self)),
                DType::I16 => T::from_i16(t(&self)),
                DType::I32 => T::from_i32(t(&self)),
                DType::I64 => T::from_i64(t(&self)),
                DType::Bool => T::from_bool(t(&self)),
            }
        }
    }
}

impl Scalar for u32 {
    fn from_bf16(t: bf16) -> Self {
        t.to_f32() as u32
    }

    fn from_f16(t: f16) -> Self {
        t.to_f32() as Self
    }

    fn from_f32(t: f32) -> Self {
        t as Self
    }

    fn from_f64(t: f64) -> Self {
        t as Self
    }

    fn from_u8(t: u8) -> Self {
        t.into()
    }

    fn from_u16(t: u16) -> Self {
        t.into()
    }

    fn from_u32(t: u32) -> Self {
        t
    }

    fn from_u64(t: u64) -> Self {
        t.try_into().unwrap()
    }

    fn from_i8(t: i8) -> Self {
        t.try_into().unwrap()
    }

    fn from_i16(t: i16) -> Self {
        t.try_into().unwrap()
    }

    fn from_i32(t: i32) -> Self {
        t.try_into().unwrap()
    }

    fn from_i64(t: i64) -> Self {
        t.try_into().unwrap()
    }

    fn from_bool(t: bool) -> Self {
        t.into()
    }

    fn from_le_bytes(bytes: &[u8]) -> Self {
        Self::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]])
    }

    fn to_ne_bytes(&self) -> &[u8] {
        let i: *const Self = self;
        unsafe { std::slice::from_raw_parts(i.cast::<u8>(), std::mem::size_of::<Self>()) }
    }

    fn dtype() -> DType {
        DType::U32
    }

    fn zero() -> Self {
        0
    }

    fn one() -> Self {
        1
    }

    fn abs(self) -> Self {
        self
    }

    fn neg(self) -> Self {
        self.wrapping_neg()
    }

    fn exp2(self) -> Self {
        if self <= 31 { 2u32.pow(self) } else { u32::MAX }
    }

    fn log2(self) -> Self {
        self.ilog2()
    }

    fn relu(self) -> Self {
        self
    }

    fn not(self) -> Self {
        u32::from(self == 0)
    }

    fn nonzero(self) -> Self {
        u32::from(self != 0)
    }

    fn add(self, rhs: Self) -> Self {
        self.wrapping_add(rhs)
    }

    fn sub(self, rhs: Self) -> Self {
        self.wrapping_sub(rhs)
    }

    fn mul(self, rhs: Self) -> Self {
        self * rhs
    }

    fn div(self, rhs: Self) -> Self {
        self / rhs
    }

    fn pow(self, rhs: Self) -> Self {
        u32::pow(self, rhs)
    }

    fn mod_(self, rhs: Self) -> Self {
        self % rhs
    }

    fn cmplt(self, rhs: Self) -> bool {
        self < rhs
    }

    fn cmpgt(self, rhs: Self) -> bool {
        self > rhs
    }

    fn noteq(self, rhs: Self) -> bool {
        self != rhs
    }

    fn or(self, rhs: Self) -> bool {
        self != 0 || rhs != 0
    }

    fn bitxor(self, rhs: Self) -> Self {
        self ^ rhs
    }

    fn bitor(self, rhs: Self) -> Self {
        self | rhs
    }

    fn bitand(self, rhs: Self) -> Self {
        self & rhs
    }

    fn bitshiftleft(self, rhs: Self) -> Self {
        self << rhs
    }

    fn bitshiftright(self, rhs: Self) -> Self {
        self >> rhs
    }

    fn and(self, rhs: Self) -> bool {
        self != 0 && rhs != 0
    }

    fn max(self, rhs: Self) -> Self {
        Ord::max(self, rhs)
    }

    fn max_value() -> Self {
        Self::MAX
    }

    fn min_value() -> Self {
        Self::MIN
    }

    fn is_equal(self, rhs: Self) -> bool {
        self == rhs
    }

    fn epsilon() -> Self {
        0
    }

    fn cast<T: Scalar>(self) -> T {
        use core::mem::transmute_copy as t;
        unsafe {
            match Self::dtype() {
                DType::BF16 => T::from_bf16(t(&self)),
                DType::F16 => T::from_f16(t(&self)),
                DType::F32 => T::from_f32(t(&self)),
                DType::F64 => T::from_f64(t(&self)),
                DType::F8E4M3 => T::from_f32(f8e4m3::to_f32(t(&self))),
                DType::F8E5M2 => T::from_f32(f8e5m2::to_f32(t(&self))),
                DType::U8 => T::from_u8(t(&self)),
                DType::U16 => T::from_u16(t(&self)),
                DType::U32 => T::from_u32(t(&self)),
                DType::U64 => T::from_u64(t(&self)),
                DType::I8 => T::from_i8(t(&self)),
                DType::I16 => T::from_i16(t(&self)),
                DType::I32 => T::from_i32(t(&self)),
                DType::I64 => T::from_i64(t(&self)),
                DType::Bool => T::from_bool(t(&self)),
            }
        }
    }
}

impl Scalar for u64 {
    fn from_bf16(t: bf16) -> Self {
        t.to_f32() as u64
    }

    fn from_f16(t: f16) -> Self {
        t.to_f32() as Self
    }

    fn from_f32(t: f32) -> Self {
        t as u64
    }

    fn from_f64(t: f64) -> Self {
        t as Self
    }

    fn from_u8(t: u8) -> Self {
        t.into()
    }

    fn from_u16(t: u16) -> Self {
        t.into()
    }

    fn from_u32(t: u32) -> Self {
        t.into()
    }

    fn from_u64(t: u64) -> Self {
        t
    }

    fn from_i8(t: i8) -> Self {
        t.try_into().unwrap()
    }

    fn from_i16(t: i16) -> Self {
        t.try_into().unwrap()
    }

    fn from_i32(t: i32) -> Self {
        t.try_into().unwrap()
    }

    fn from_i64(t: i64) -> Self {
        t.try_into().unwrap()
    }

    fn from_bool(t: bool) -> Self {
        t.into()
    }

    fn from_le_bytes(bytes: &[u8]) -> Self {
        Self::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5], bytes[6], bytes[7]])
    }

    fn to_ne_bytes(&self) -> &[u8] {
        let i: *const Self = self;
        unsafe { std::slice::from_raw_parts(i.cast::<u8>(), std::mem::size_of::<Self>()) }
    }

    fn dtype() -> DType {
        DType::U64
    }

    fn zero() -> Self {
        0
    }

    fn one() -> Self {
        1
    }

    fn abs(self) -> Self {
        self
    }

    fn neg(self) -> Self {
        self.wrapping_neg()
    }

    fn exp2(self) -> Self {
        if self <= 63 { 2u64.pow(self as u32) } else { u64::MAX }
    }

    fn log2(self) -> Self {
        self.ilog2() as u64
    }

    fn relu(self) -> Self {
        self
    }

    fn not(self) -> Self {
        u64::from(self == 0)
    }

    fn nonzero(self) -> Self {
        u64::from(self != 0)
    }

    fn add(self, rhs: Self) -> Self {
        self + rhs
    }

    fn sub(self, rhs: Self) -> Self {
        self.wrapping_sub(rhs)
    }

    fn mul(self, rhs: Self) -> Self {
        self * rhs
    }

    fn div(self, rhs: Self) -> Self {
        self / rhs
    }

    fn pow(self, rhs: Self) -> Self {
        i64::pow(self as i64, u32::try_from(rhs).unwrap()) as u64
    }

    fn mod_(self, rhs: Self) -> Self {
        self % rhs
    }

    fn cmplt(self, rhs: Self) -> bool {
        self < rhs
    }

    fn cmpgt(self, rhs: Self) -> bool {
        self > rhs
    }

    fn noteq(self, rhs: Self) -> bool {
        self != rhs
    }

    fn or(self, rhs: Self) -> bool {
        self != 0 || rhs != 0
    }

    fn bitxor(self, rhs: Self) -> Self {
        self ^ rhs
    }

    fn bitor(self, rhs: Self) -> Self {
        self | rhs
    }

    fn bitand(self, rhs: Self) -> Self {
        self & rhs
    }

    fn bitshiftleft(self, rhs: Self) -> Self {
        self << rhs
    }

    fn bitshiftright(self, rhs: Self) -> Self {
        self >> rhs
    }

    fn and(self, rhs: Self) -> bool {
        self != 0 && rhs != 0
    }

    fn max(self, rhs: Self) -> Self {
        Ord::max(self, rhs)
    }

    fn max_value() -> Self {
        Self::MAX
    }

    fn min_value() -> Self {
        Self::MIN
    }

    fn is_equal(self, rhs: Self) -> bool {
        self == rhs
    }

    fn epsilon() -> Self {
        0
    }
}

impl Scalar for bool {
    fn from_bf16(t: bf16) -> Self {
        t != bf16::ZERO
    }

    fn from_f16(t: f16) -> Self {
        t != f16::ZERO
    }

    fn from_f32(t: f32) -> Self {
        t != 0.
    }

    fn from_f64(t: f64) -> Self {
        t != 0.
    }

    fn from_u8(t: u8) -> Self {
        t != 0
    }

    fn from_u16(t: u16) -> Self {
        t != 0
    }

    fn from_u32(t: u32) -> Self {
        t != 0
    }

    fn from_u64(t: u64) -> Self {
        t != 0
    }

    fn from_i8(t: i8) -> Self {
        t != 0
    }

    fn from_i16(t: i16) -> Self {
        t != 0
    }

    fn from_i32(t: i32) -> Self {
        t != 0
    }

    fn from_i64(t: i64) -> Self {
        t != 0
    }

    fn from_bool(t: bool) -> Self {
        t
    }

    fn from_le_bytes(bytes: &[u8]) -> Self {
        bytes[0] != 0
    }

    fn to_ne_bytes(&self) -> &[u8] {
        let i: *const Self = self;
        unsafe { std::slice::from_raw_parts(i.cast::<u8>(), std::mem::size_of::<Self>()) }
    }

    fn dtype() -> DType {
        DType::Bool
    }

    fn zero() -> Self {
        false
    }

    fn one() -> Self {
        true
    }

    fn abs(self) -> Self {
        self
    }

    fn neg(self) -> Self {
        panic!()
    }

    fn exp2(self) -> Self {
        panic!()
    }

    fn log2(self) -> Self {
        panic!()
    }

    fn relu(self) -> Self {
        panic!()
    }

    fn not(self) -> Self {
        !self
    }

    fn nonzero(self) -> Self {
        self
    }

    fn add(self, rhs: Self) -> Self {
        self | rhs
    }

    fn sub(self, rhs: Self) -> Self {
        let _ = rhs;
        panic!()
    }

    fn mul(self, rhs: Self) -> Self {
        self & rhs
    }

    fn div(self, rhs: Self) -> Self {
        let _ = rhs;
        panic!()
    }

    fn pow(self, rhs: Self) -> Self {
        let _ = rhs;
        panic!()
    }

    fn mod_(self, rhs: Self) -> Self {
        let _ = rhs;
        panic!()
    }

    fn cmplt(self, rhs: Self) -> Self {
        !self & rhs
    }

    fn cmpgt(self, rhs: Self) -> Self {
        self && !rhs
    }

    fn noteq(self, rhs: Self) -> bool {
        self != rhs
    }

    fn or(self, rhs: Self) -> Self {
        self || rhs
    }

    fn bitxor(self, rhs: Self) -> Self {
        self ^ rhs
    }

    fn bitor(self, rhs: Self) -> Self {
        self | rhs
    }

    fn bitand(self, rhs: Self) -> Self {
        self & rhs
    }

    fn bitshiftleft(self, _rhs: Self) -> Self {
        self
    }

    fn bitshiftright(self, _rhs: Self) -> Self {
        false
    }

    fn and(self, rhs: Self) -> bool {
        self && rhs
    }

    fn max(self, rhs: Self) -> Self {
        <bool as Ord>::max(self, rhs)
    }

    fn max_value() -> Self {
        true
    }

    fn min_value() -> Self {
        false
    }

    fn is_equal(self, rhs: Self) -> bool {
        self == rhs
    }

    fn epsilon() -> Self {
        false
    }

    fn cast<T: Scalar>(self) -> T {
        use core::mem::transmute_copy as t;
        unsafe {
            match Self::dtype() {
                DType::BF16 => T::from_bf16(t(&self)),
                DType::F16 => T::from_f16(t(&self)),
                DType::F32 => T::from_f32(t(&self)),
                DType::F64 => T::from_f64(t(&self)),
                DType::F8E4M3 => T::from_f32(f8e4m3::to_f32(t(&self))),
                DType::F8E5M2 => T::from_f32(f8e5m2::to_f32(t(&self))),
                DType::U8 => T::from_u8(t(&self)),
                DType::U16 => T::from_u16(t(&self)),
                DType::U32 => T::from_u32(t(&self)),
                DType::U64 => T::from_u64(t(&self)),
                DType::I8 => T::from_i8(t(&self)),
                DType::I16 => T::from_i16(t(&self)),
                DType::I32 => T::from_i32(t(&self)),
                DType::I64 => T::from_i64(t(&self)),
                DType::Bool => T::from_bool(t(&self)),
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{f8e4m3, f8e5m2};

    #[test]
    fn f8_roundtrip() {
        // E4M3 known bit patterns (OCP: bias 7, NaN-only 0x7F/0xFF).
        assert_eq!(f8e4m3::from_bits(0x38).to_f32(), 1.0);
        assert_eq!(f8e4m3::from_bits(0x3c).to_f32(), 1.5);
        assert_eq!(f8e4m3::from_bits(0x7e).to_f32(), 448.0);
        assert_eq!(f8e4m3::from_bits(0xfe).to_f32(), -448.0);
        assert!(f8e4m3::from_bits(0x7f).is_nan());
        assert!(f8e4m3::from_bits(0xff).is_nan());
        assert_eq!(f8e4m3::from_f32(1.0).to_bits(), 0x38);
        assert_eq!(f8e4m3::from_f32(-1.5).to_bits(), 0xbc);
        // Overflow saturates, never inf.
        assert_eq!(f8e4m3::from_f32(1000.0).to_bits(), 0x7e);
        assert_eq!(f8e4m3::from_f32(f32::INFINITY).to_bits(), 0x7e);
        assert!(!f8e4m3::from_f32(1000.0).is_infinite());
        // E5M2 known bit patterns (bias 15, IEEE inf/NaN).
        assert_eq!(f8e5m2::from_bits(0x3c).to_f32(), 1.0);
        assert_eq!(f8e5m2::from_bits(0x7b).to_f32(), 57344.0);
        assert_eq!(f8e5m2::from_bits(0xfb).to_f32(), -57344.0);
        assert!(f8e5m2::from_bits(0x7c).is_infinite());
        assert!(f8e5m2::from_bits(0x7e).is_nan());
        assert_eq!(f8e5m2::from_f32(1.0).to_bits(), 0x3c);
        assert_eq!(f8e5m2::from_f32(f32::INFINITY).to_bits(), 0x7c);
        // Arithmetic round-trips through f32.
        assert_eq!((f8e4m3::from_f32(1.5) + f8e4m3::from_f32(2.25)).to_f32(), 3.75);
        assert_eq!((f8e5m2::from_f32(2.0) * f8e5m2::from_f32(3.0)).to_f32(), 6.0);
        assert_eq!(f8e4m3::from_f32(7.0).to_f32(), 7.0);
    }
}
