use crate::DType;

use super::Scalar;
#[cfg(feature = "half")]
use half::{bf16, f16};
#[cfg(feature = "complex")]
use num_complex::Complex;

impl Scalar for bool {
    #[cfg(feature = "half")]
    fn from_bf16(t: bf16) -> Self {
        t != bf16::ZERO
    }

    #[cfg(feature = "half")]
    fn from_f16(t: f16) -> Self {
        t != f16::ZERO
    }

    fn from_f32(t: f32) -> Self {
        t != 0.
    }

    fn from_f64(t: f64) -> Self {
        t != 0.
    }

    #[cfg(feature = "complex")]
    fn from_cf32(t: Complex<f32>) -> Self {
        t != Complex::new(0., 0.)
    }

    #[cfg(feature = "complex")]
    fn from_cf64(t: Complex<f64>) -> Self {
        t != Complex::new(0., 0.)
    }

    fn from_u8(t: u8) -> Self {
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

    fn reciprocal(self) -> Self {
        panic!()
    }

    fn floor(self) -> Self {
        self
    }

    fn neg(self) -> Self {
        panic!()
    }

    fn relu(self) -> Self {
        panic!()
    }

    fn sin(self) -> Self {
        panic!()
    }

    fn cos(self) -> Self {
        panic!()
    }

    fn sqrt(self) -> Self {
        panic!()
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

    fn cmplt(self, rhs: Self) -> Self {
        self < rhs
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

    fn epsilon() -> Self {
        false
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
    
    fn inv(self) -> Self {
        todo!()
    }
    
    fn not(self) -> Self {
        todo!()
    }
    
    fn nonzero(self) -> Self {
        todo!()
    }
    
    fn cmpgt(self, rhs: Self) -> Self {
        self > rhs
    }
    
    fn or(self, rhs: Self) -> Self {
        self || rhs
    }
}
