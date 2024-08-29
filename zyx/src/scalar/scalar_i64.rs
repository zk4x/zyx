use crate::dtype::DType;
use crate::scalar::Scalar;
#[cfg(feature = "half")]
use half::{bf16, f16};
#[cfg(feature = "complex")]
use num_complex::Complex;

impl Scalar for i64 {
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
        t as Self
    }

    fn from_f64(t: f64) -> Self {
        t as Self
    }

    #[cfg(feature = "complex")]
    fn from_cf32(t: Complex<f32>) -> Self {
        t.re as Self
    }

    #[cfg(feature = "complex")]
    fn from_cf64(t: Complex<f64>) -> Self {
        t.re as Self
    }

    fn from_u8(t: u8) -> Self {
        t.into()
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
        t as i64
    }

    fn from_le_bytes(bytes: &[u8]) -> Self {
        i64::from_le_bytes([
            bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5], bytes[6], bytes[7],
        ])
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

    fn reciprocal(self) -> Self {
        1 / self
    }

    fn floor(self) -> Self {
        self
    }

    fn neg(self) -> Self {
        -self
    }

    fn relu(self) -> Self {
        <i64 as Ord>::max(self, 0)
    }

    fn sin(self) -> Self {
        f64::sin(self as f64) as i64
    }

    fn cos(self) -> Self {
        f64::cos(self as f64) as i64
    }

    fn ln(self) -> Self {
        f64::ln(self as f64) as i64
    }

    fn exp(self) -> Self {
        f64::exp(self as f64) as i64
    }

    fn tanh(self) -> Self {
        f64::tanh(self as f64) as i64
    }

    fn sqrt(self) -> Self {
        (self as f64).sqrt() as i64
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
        i64::pow(self, rhs as u32)
    }

    fn cmplt(self, rhs: Self) -> Self {
        (self < rhs) as Self
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

    fn epsilon() -> Self {
        0
    }

    fn is_equal(self, rhs: Self) -> bool {
        self == rhs
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
