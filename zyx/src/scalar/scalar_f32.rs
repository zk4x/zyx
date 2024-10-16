use crate::dtype::DType;
use crate::scalar::{Scalar, Float};
use half::{bf16, f16};
#[cfg(feature = "complex")]
use num_complex::Complex;
use float8::F8E4M3;

impl Scalar for f32 {
    fn from_bf16(t: bf16) -> Self {
        t.into()
    }

    fn from_f8(t: F8E4M3) -> Self {
        t.into()
    }

    fn from_f16(t: f16) -> Self {
        t.into()
    }

    fn from_f32(t: f32) -> Self {
        t
    }

    fn from_f64(t: f64) -> Self {
        t as f32
    }

    #[cfg(feature = "complex")]
    fn from_cf32(t: Complex<f32>) -> Self {
        t.re
    }

    #[cfg(feature = "complex")]
    fn from_cf64(t: Complex<f64>) -> Self {
        t.re as f32
    }

    fn from_u8(t: u8) -> Self {
        t as f32
    }

    fn from_u32(t: u32) -> Self {
        t as f32
    }

    fn from_i8(t: i8) -> Self {
        t as f32
    }

    fn from_i16(t: i16) -> Self {
        t as f32
    }

    fn from_i32(t: i32) -> Self {
        t as f32
    }

    fn from_i64(t: i64) -> Self {
        t as f32
    }

    fn from_bool(t: bool) -> Self {
        t as i32 as f32
    }

    fn from_le_bytes(bytes: &[u8]) -> Self {
        f32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]])
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

    fn relu(self) -> Self {
        self.max(0.)
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

    fn cmplt(self, rhs: Self) -> Self {
        (self < rhs) as i32 as f32
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

    fn epsilon() -> Self {
        0.0001
    }

    fn is_equal(self, rhs: Self) -> bool {
        // Less than 1% error is OK
        (self == -f32::INFINITY && rhs == -f32::INFINITY)
            || (self.is_nan() && rhs.is_nan())
            || (self - rhs).abs() < Self::epsilon()
            || (self - rhs).abs() < self.abs() * 0.01
    }

    fn not(self) -> Self {
        if self != 0. { 0. } else { 1. }
    }

    fn nonzero(self) -> Self {
        (self != 0.) as i32 as f32
    }

    fn cmpgt(self, rhs: Self) -> Self {
        (self > rhs) as i32 as f32
    }

    fn or(self, rhs: Self) -> Self {
        (self != 0. || rhs != 0.) as i32 as f32
    }
}

impl Float for f32 {
    fn exp2(self) -> Self {
        self.exp2()
    }

    fn log2(self) -> Self {
        self.log2()
    }

    fn sin(self) -> Self {
        //libm::sinf(self)
        //let b = 4f32 / PI;
        //let c = -4f32 / (PI * PI);
        //return -(b * self + c * self * if self < 0. { -self } else { self });
        f32::sin(self)
    }

    fn floor(self) -> Self {
        let i = self as i32 as f32;
        return i - (i > self) as i32 as f32;
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
