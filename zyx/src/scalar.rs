//! Trait describing required operations on scalar values

use crate::dtype::DType;
use half::{bf16, f16};

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
    fn to_ne_bytes(&self) -> &[u8];
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

    fn from_u64(t: u64) -> Self {
        let _ = t;
        todo!()
    }

    fn from_i8(t: i8) -> Self {
        bf16::from_f32(f32::from(t))
    }

    fn from_i16(t: i16) -> Self {
        bf16::from_f32(f32::from(t))
    }

    fn from_i32(t: i32) -> Self {
        bf16::from_f32(f32::from(u16::try_from(t).unwrap()))
    }

    fn from_i64(t: i64) -> Self {
        bf16::from_f32(f32::from(u16::try_from(t).unwrap()))
    }

    fn from_bool(t: bool) -> Self {
        bf16::from_f32(f32::from(t))
    }

    fn from_le_bytes(bytes: &[u8]) -> Self {
        bf16::from_le_bytes([bytes[0], bytes[1]])
    }

    fn to_ne_bytes(&self) -> &[u8] {
        todo!()
    }

    fn dtype() -> DType {
        //DType::BF16
        todo!()
    }

    fn zero() -> Self {
        bf16::ZERO
    }

    fn one() -> Self {
        bf16::ONE
    }

    fn byte_size() -> usize {
        2
    }

    fn abs(self) -> Self {
        self.max(-self)
    }

    fn neg(self) -> Self {
        -self
    }

    fn exp2(self) -> Self {
        todo!()
    }

    fn log2(self) -> Self {
        todo!()
    }

    fn relu(self) -> Self {
        self.max(bf16::ZERO)
    }

    fn not(self) -> Self {
        todo!()
    }

    fn nonzero(self) -> Self {
        todo!()
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

    fn pow(self, _rhs: Self) -> Self {
        todo!()
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
        let _ = rhs;
        //self | rhs
        todo!()
    }

    fn bitor(self, rhs: Self) -> Self {
        let _ = rhs;
        //self ^ rhs
        todo!()
    }

    fn bitand(self, rhs: Self) -> Self {
        let _ = rhs;
        //self & rhs
        todo!()
    }

    fn bitshiftleft(self, rhs: Self) -> Self {
        let _ = rhs;
        todo!()
    }

    fn bitshiftright(self, rhs: Self) -> Self {
        let _ = rhs;
        todo!()
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
        self == rhs
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
        todo!()
    }

    fn sin(self) -> Self {
        todo!()
    }

    fn cos(self) -> Self {
        todo!()
    }

    fn sqrt(self) -> Self {
        todo!()
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
        let _ = t;
        todo!()
    }

    fn from_u16(t: u16) -> Self {
        let _ = t;
        todo!()
    }

    fn from_u32(t: u32) -> Self {
        f16::from_f64(t.into())
    }

    fn from_u64(t: u64) -> Self {
        let _ = t;
        todo!()
    }

    fn from_i8(t: i8) -> Self {
        let _ = t;
        todo!()
    }

    fn from_i16(t: i16) -> Self {
        let _ = t;
        todo!()
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
        todo!()
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

    fn byte_size() -> usize {
        2
    }

    fn abs(self) -> Self {
        todo!()
    }

    fn neg(self) -> Self {
        -self
    }

    fn exp2(self) -> Self {
        todo!()
    }

    fn log2(self) -> Self {
        todo!()
    }

    fn relu(self) -> Self {
        self.max(f16::ZERO)
    }

    fn not(self) -> Self {
        todo!()
    }

    fn nonzero(self) -> Self {
        todo!()
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
        self != Self::ZERO || rhs != Self::ZERO
    }

    fn bitxor(self, rhs: Self) -> Self {
        let _ = rhs;
        //self ^ rhs
        todo!()
    }

    fn bitor(self, rhs: Self) -> Self {
        let _ = rhs;
        //self | rhs
        todo!()
    }

    fn bitand(self, rhs: Self) -> Self {
        let _ = rhs;
        //self & rhs
        todo!()
    }

    fn bitshiftleft(self, rhs: Self) -> Self {
        let _ = rhs;
        todo!()
    }

    fn bitshiftright(self, rhs: Self) -> Self {
        let _ = rhs;
        todo!()
    }

    fn and(self, rhs: Self) -> bool {
        self != Self::ZERO && rhs != Self::ZERO
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
        (self == -Self::INFINITY && rhs == -Self::INFINITY)
            || (self.is_nan() && rhs.is_nan())
            || self.sub(rhs).abs() < self.abs() * f16::from_f32(0.0001)
    }

    fn epsilon() -> Self {
        f16::from_f32(0.00001)
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
        unsafe { std::slice::from_raw_parts(i as *const u8, std::mem::size_of::<Self>()) }
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

    fn byte_size() -> usize {
        4
    }

    fn abs(self) -> Self {
        self.abs()
    }

    fn neg(self) -> Self {
        -self
    }

    fn exp2(self) -> Self {
        todo!()
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
        let _ = rhs;
        //self ^ rhs
        todo!()
    }

    fn bitor(self, rhs: Self) -> Self {
        let _ = rhs;
        //self | rhs
        todo!()
    }

    fn bitand(self, rhs: Self) -> Self {
        let _ = rhs;
        //self & rhs
        todo!()
    }

    fn bitshiftleft(self, rhs: Self) -> Self {
        let _ = rhs;
        todo!()
    }

    fn bitshiftright(self, rhs: Self) -> Self {
        let _ = rhs;
        todo!()
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

    fn from_u64(t: u64) -> Self {
        let _ = t;
        todo!()
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
        f64::from_le_bytes([
            bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5], bytes[6], bytes[7],
        ])
    }

    fn to_ne_bytes(&self) -> &[u8] {
        let i: *const Self = self;
        unsafe { std::slice::from_raw_parts(i as *const u8, std::mem::size_of::<Self>()) }
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

    fn byte_size() -> usize {
        8
    }

    fn abs(self) -> Self {
        self.abs()
    }

    fn neg(self) -> Self {
        -self
    }

    fn exp2(self) -> Self {
        todo!()
    }

    fn log2(self) -> Self {
        todo!()
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
        let _ = rhs;
        //self ^ rhs
        todo!()
    }

    fn bitor(self, rhs: Self) -> Self {
        let _ = rhs;
        //self | rhs
        todo!()
    }

    fn bitand(self, rhs: Self) -> Self {
        let _ = rhs;
        //self & rhs
        todo!()
    }

    fn bitshiftleft(self, rhs: Self) -> Self {
        let _ = rhs;
        todo!()
    }

    fn bitshiftright(self, rhs: Self) -> Self {
        let _ = rhs;
        todo!()
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
        unsafe { std::slice::from_raw_parts(i as *const u8, std::mem::size_of::<Self>()) }
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

    fn byte_size() -> usize {
        1
    }

    fn abs(self) -> Self {
        self.abs()
    }

    fn neg(self) -> Self {
        -self
    }

    fn exp2(self) -> Self {
        todo!()
    }

    fn log2(self) -> Self {
        todo!()
    }

    fn relu(self) -> Self {
        Ord::max(self, 0)
    }

    fn not(self) -> Self {
        todo!()
    }

    fn nonzero(self) -> Self {
        todo!()
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
        let _ = rhs;
        todo!()
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
        let _ = rhs;
        todo!()
    }

    fn bitshiftright(self, rhs: Self) -> Self {
        let _ = rhs;
        todo!()
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
        unsafe { std::slice::from_raw_parts(i as *const u8, std::mem::size_of::<Self>()) }
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

    fn byte_size() -> usize {
        2
    }

    fn abs(self) -> Self {
        todo!()
    }

    fn neg(self) -> Self {
        -self
    }

    fn exp2(self) -> Self {
        todo!()
    }

    fn log2(self) -> Self {
        todo!()
    }

    fn relu(self) -> Self {
        Ord::max(self, 0)
    }

    fn not(self) -> Self {
        todo!()
    }

    fn nonzero(self) -> Self {
        todo!()
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
        let _ = rhs;
        todo!()
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
        let _ = rhs;
        todo!()
    }

    fn bitshiftright(self, rhs: Self) -> Self {
        let _ = rhs;
        todo!()
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
        let _ = t;
        todo!()
    }

    fn from_f16(t: f16) -> Self {
        let _ = t;
        todo!()
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
        unsafe { std::slice::from_raw_parts(i as *const u8, std::mem::size_of::<i32>()) }
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

    fn byte_size() -> usize {
        4
    }

    fn abs(self) -> Self {
        todo!()
    }

    fn neg(self) -> Self {
        -self
    }

    fn exp2(self) -> Self {
        todo!()
    }

    fn log2(self) -> Self {
        todo!()
    }

    fn relu(self) -> Self {
        <i32 as Ord>::max(self, 0)
    }

    fn not(self) -> Self {
        todo!()
    }

    fn nonzero(self) -> Self {
        todo!()
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
        let _ = rhs;
        todo!()
    }

    fn bitshiftright(self, rhs: Self) -> Self {
        let _ = rhs;
        todo!()
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
        let _ = t;
        todo!()
    }

    fn from_f16(t: f16) -> Self {
        let _ = t;
        todo!()
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
        i64::from_le_bytes([
            bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5], bytes[6], bytes[7],
        ])
    }

    fn to_ne_bytes(&self) -> &[u8] {
        let i: *const Self = self;
        unsafe { std::slice::from_raw_parts(i as *const u8, std::mem::size_of::<Self>()) }
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

    fn byte_size() -> usize {
        8
    }

    fn abs(self) -> Self {
        todo!()
    }

    fn neg(self) -> Self {
        -self
    }

    fn exp2(self) -> Self {
        todo!()
    }

    fn log2(self) -> Self {
        todo!()
    }

    fn relu(self) -> Self {
        <i64 as Ord>::max(self, 0)
    }

    fn not(self) -> Self {
        todo!()
    }

    fn nonzero(self) -> Self {
        todo!()
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
        let _ = rhs;
        todo!()
    }

    fn bitshiftright(self, rhs: Self) -> Self {
        let _ = rhs;
        todo!()
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
        let _ = t;
        todo!()
    }

    fn from_f16(t: f16) -> Self {
        let _ = t;
        todo!()
    }

    fn from_f32(t: f32) -> Self {
        let _ = t;
        todo!()
    }

    fn from_f64(t: f64) -> Self {
        let _ = t;
        todo!()
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
        unsafe { std::slice::from_raw_parts(i as *const u8, std::mem::size_of::<Self>()) }
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

    fn byte_size() -> usize {
        1
    }

    fn abs(self) -> Self {
        todo!()
    }

    fn neg(self) -> Self {
        todo!()
    }

    fn exp2(self) -> Self {
        todo!()
    }

    fn log2(self) -> Self {
        todo!()
    }

    fn relu(self) -> Self {
        todo!()
    }

    fn not(self) -> Self {
        todo!()
    }

    fn nonzero(self) -> Self {
        todo!()
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
        let _ = rhs;
        todo!()
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
        let _ = rhs;
        todo!()
    }

    fn bitshiftright(self, rhs: Self) -> Self {
        let _ = rhs;
        todo!()
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
        let _ = t;
        todo!()
    }

    fn from_f16(t: f16) -> Self {
        let _ = t;
        todo!()
    }

    fn from_f32(t: f32) -> Self {
        let _ = t;
        todo!()
    }

    fn from_f64(t: f64) -> Self {
        let _ = t;
        todo!()
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
        unsafe { std::slice::from_raw_parts(i as *const u8, std::mem::size_of::<Self>()) }
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

    fn byte_size() -> usize {
        4
    }

    fn abs(self) -> Self {
        todo!()
    }

    fn neg(self) -> Self {
        todo!()
    }

    fn exp2(self) -> Self {
        todo!()
    }

    fn log2(self) -> Self {
        todo!()
    }

    fn relu(self) -> Self {
        todo!()
    }

    fn not(self) -> Self {
        todo!()
    }

    fn nonzero(self) -> Self {
        todo!()
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
        let _ = rhs;
        todo!()
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
        let _ = rhs;
        todo!()
    }

    fn bitshiftright(self, rhs: Self) -> Self {
        let _ = rhs;
        todo!()
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
        let _ = t;
        todo!()
    }

    fn from_f16(t: f16) -> Self {
        let _ = t;
        todo!()
    }

    fn from_f32(t: f32) -> Self {
        let _ = t;
        todo!()
    }

    fn from_f64(t: f64) -> Self {
        let _ = t;
        todo!()
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
        unsafe { std::slice::from_raw_parts(i as *const u8, std::mem::size_of::<Self>()) }
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

    fn byte_size() -> usize {
        4
    }

    fn abs(self) -> Self {
        todo!()
    }

    fn neg(self) -> Self {
        todo!()
    }

    fn exp2(self) -> Self {
        todo!()
    }

    fn log2(self) -> Self {
        self.ilog2()
    }

    fn relu(self) -> Self {
        todo!()
    }

    fn not(self) -> Self {
        todo!()
    }

    fn nonzero(self) -> Self {
        todo!()
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
        let _ = rhs;
        todo!()
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
        let _ = rhs;
        todo!()
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
        let _ = t;
        todo!()
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
        Self::from_le_bytes([
            bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5], bytes[6], bytes[7],
        ])
    }

    fn to_ne_bytes(&self) -> &[u8] {
        let i: *const Self = self;
        unsafe { std::slice::from_raw_parts(i as *const u8, std::mem::size_of::<Self>()) }
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

    fn byte_size() -> usize {
        4
    }

    fn abs(self) -> Self {
        todo!()
    }

    fn neg(self) -> Self {
        todo!()
    }

    fn exp2(self) -> Self {
        todo!()
    }

    fn log2(self) -> Self {
        todo!()
    }

    fn relu(self) -> Self {
        todo!()
    }

    fn not(self) -> Self {
        todo!()
    }

    fn nonzero(self) -> Self {
        todo!()
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
        let _ = rhs;
        todo!()
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
        let _ = rhs;
        todo!()
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
        unsafe { std::slice::from_raw_parts(i as *const u8, std::mem::size_of::<Self>()) }
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

    fn byte_size() -> usize {
        1
    }

    fn abs(self) -> Self {
        self
    }

    fn neg(self) -> Self {
        panic!()
    }

    fn exp2(self) -> Self {
        todo!()
    }

    fn log2(self) -> Self {
        todo!()
    }

    fn relu(self) -> Self {
        panic!()
    }

    fn not(self) -> Self {
        todo!()
    }

    fn nonzero(self) -> Self {
        todo!()
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
        //self % rhs
        todo!()
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

    fn bitshiftleft(self, rhs: Self) -> Self {
        let _ = rhs;
        todo!()
    }

    fn bitshiftright(self, rhs: Self) -> Self {
        let _ = rhs;
        todo!()
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
