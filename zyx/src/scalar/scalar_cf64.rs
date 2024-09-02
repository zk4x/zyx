use crate::dtype::DType;
use crate::scalar::Scalar;
#[cfg(feature = "half")]
use half::{bf16, f16};
use num_complex::Complex;

impl Scalar for Complex<f64> {
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
        let _ = t;
        todo!()
    }

    fn from_f64(t: f64) -> Self {
        let _ = t;
        todo!()
    }

    fn from_cf32(t: Complex<f32>) -> Self {
        let _ = t;
        todo!()
    }

    fn from_cf64(t: Complex<f64>) -> Self {
        let _ = t;
        todo!()
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
        let _ = t;
        todo!()
    }

    fn from_i64(t: i64) -> Self {
        let _ = t;
        todo!()
    }

    fn from_bool(t: bool) -> Self {
        let _ = t;
        todo!()
    }

    fn from_le_bytes(bytes: &[u8]) -> Self {
        Complex::<f64>::new(
            f64::from_le_bytes([
                bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5], bytes[6], bytes[7],
            ]),
            f64::from_le_bytes([
                bytes[8], bytes[9], bytes[10], bytes[11], bytes[12], bytes[13], bytes[14],
                bytes[15],
            ]),
        )
    }

    fn dtype() -> DType {
        todo!()
    }

    fn zero() -> Self {
        todo!()
    }

    fn one() -> Self {
        todo!()
    }

    fn byte_size() -> usize {
        todo!()
    }

    fn abs(self) -> Self {
        todo!()
    }

    fn reciprocal(self) -> Self {
        todo!()
    }

    fn floor(self) -> Self {
        todo!()
    }

    fn neg(self) -> Self {
        todo!()
    }

    fn relu(self) -> Self {
        todo!()
    }

    fn sin(self) -> Self {
        todo!()
    }

    fn cos(self) -> Self {
        todo!()
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

    fn pow(self, rhs: Self) -> Self {
        let _ = rhs;
        todo!()
    }

    fn cmplt(self, rhs: Self) -> Self {
        let _ = rhs;
        todo!()
    }
    
    fn cmpgt(self, rhs: Self) -> Self {
        todo!()
    }
    
    fn or(self, rhs: Self) -> Self {
        todo!()
    }
    
    fn max(self, rhs: Self) -> Self {
        let _ = rhs;
        todo!()
    }
    
    fn max_value() -> Self {
        todo!()
    }
    
    fn min_value() -> Self {
        todo!()
    }
    
    fn epsilon() -> Self {
        todo!()
    }
    
    fn is_equal(self, rhs: Self) -> bool {
        self == rhs
    }
}
