use crate::dtype::DType;
use crate::scalar::Scalar;
use half::{bf16, f16};

#[cfg(feature = "complex")]
use num_complex::Complex;

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

    #[cfg(feature = "complex")]
    fn from_cf32(t: Complex<f32>) -> Self {
        bf16::from_f32(t.re)
    }

    #[cfg(feature = "complex")]
    fn from_cf64(t: Complex<f64>) -> Self {
        bf16::from_f64(t.re)
    }

    fn from_u8(t: u8) -> Self {
        bf16::from_f32(t as f32)
    }

    fn from_i8(t: i8) -> Self {
        bf16::from_f32(t as f32)
    }

    fn from_i16(t: i16) -> Self {
        bf16::from_f32(t as f32)
    }

    fn from_i32(t: i32) -> Self {
        bf16::from_f32(t as f32)
    }

    fn from_i64(t: i64) -> Self {
        bf16::from_f32(t as f32)
    }

    fn from_bool(t: bool) -> Self {
        bf16::from_f32(t as i32 as f32)
    }

    fn from_le_bytes(bytes: &[u8]) -> Self {
        bf16::from_le_bytes([bytes[0], bytes[1]])
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

    fn byte_size() -> usize {
        2
    }

    fn abs(self) -> Self {
        self.max(-self)
    }

    fn reciprocal(self) -> Self {
        bf16::ONE / self
    }

    fn neg(self) -> Self {
        -self
    }

    fn floor(self) -> Self {
        todo!()
    }

    fn relu(self) -> Self {
        self.max(bf16::ZERO)
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

    fn cmplt(self, rhs: Self) -> Self {
        if self < rhs {
            bf16::ONE
        } else {
            bf16::ZERO
        }
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

    fn epsilon() -> Self {
        bf16::MIN_POSITIVE
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
        todo!()
    }
    
    fn or(self, rhs: Self) -> Self {
        todo!()
    }
}

impl Float for bf16 {}

