use core::f32::consts::{E, PI};

use crate::dtype::DType;
use crate::scalar::Scalar;
#[cfg(feature = "half")]
use half::{bf16, f16};
#[cfg(feature = "complex")]
use num_complex::Complex;

impl Scalar for f32 {
    #[cfg(feature = "half")]
    fn from_bf16(t: bf16) -> Self {
        let _ = t;
        todo!()
    }

    #[cfg(feature = "half")]
    fn from_f16(t: f16) -> Self {
        let _ = t;
        todo!()
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
        todo!()
    }

    fn reciprocal(self) -> Self {
        1.0 / self
    }

    fn neg(self) -> Self {
        -self
    }

    fn relu(self) -> Self {
        self.max(0.)
    }

    fn sin(self) -> Self {
        //libm::sinf(self)
        let b = 4f32 / PI;
        let c = -4f32 / (PI * PI);
        return -(b * self + c * self * if self < 0. { -self } else { self });
    }

    fn floor(self) -> Self {
        let i = self as i32 as f32;
        return i - (i > self) as i32 as f32;
    }

    fn cos(self) -> Self {
        //libm::cosf(self)
        let mut x = self;
        x *= 1. / (2. * PI);
        x -= 0.25 + (x + 0.25).floor();
        x *= 16.0 * (x.abs() - 0.5);
        //x += 0.225 * x * (x.abs() - 1.0);
        return x;
    }

    fn ln(self) -> Self {
        //libm::logf(self)
        todo!()
    }

    fn exp(self) -> Self {
        //libm::expf(self)
        todo!()
    }

    fn tanh(self) -> Self {
        //libm::tanhf(self)
        let e2x = E.pow(2.0 * self);
        return (e2x - 1.0) / (e2x + 1.0);
    }

    fn sqrt(self) -> Self {
        // good enough (error of ~ 5%)
        if self >= 0. {
            Self::from_bits((self.to_bits() + 0x3f80_0000) >> 1)
        } else {
            Self::NAN
        }
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
        //libm::powf(self, rhs)
        todo!()
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
        0.00001
    }

    fn is_equal(self, rhs: Self) -> bool {
        // Less than 1% error is OK
        (self == -f32::INFINITY && rhs == -f32::INFINITY)
            || (self - rhs).abs() < Self::epsilon()
            || (self - rhs).abs() < self.abs() * 0.01
    }
    
    fn exp2(self) -> Self {
        todo!()
    }
    
    fn log2(self) -> Self {
        todo!()
    }
    
    fn log(self) -> Self {
        todo!()
    }
}
