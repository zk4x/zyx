use crate::dtype::DType;
use crate::scalar::Scalar;
use half::{bf16, f16};
use float8::F8E4M3;
#[cfg(feature = "complex")]
use num_complex::Complex;

impl Scalar for i8 {
    fn from_bf16(t: bf16) -> Self {
        let t: f32 = t.into();
        t as Self
    }

    fn from_f8(t: F8E4M3) -> Self {
        let t: F8E4M3 = t.into();
        t.to_f32() as Self
    }

    fn from_f16(t: f16) -> Self {
        let t: f32 = t.into();
        t as Self
    }

    fn from_f32(t: f32) -> Self {
        t as Self
    }

    fn from_f64(t: f64) -> Self {
        t as Self
    }

    #[cfg(feature = "complex")]
    fn from_cf32(t: Complex<f32>) -> Self {
        let _ = t;
        todo!()
    }

    #[cfg(feature = "complex")]
    fn from_cf64(t: Complex<f64>) -> Self {
        let _ = t;
        todo!()
    }

    fn from_u8(t: u8) -> Self {
        t as Self
    }

    fn from_u32(t: u32) -> Self {
        t as Self
    }

    fn from_i8(t: i8) -> Self {
        t as Self
    }

    fn from_i16(t: i16) -> Self {
        t as Self
    }

    fn from_i32(t: i32) -> Self {
        t as Self
    }

    fn from_i64(t: i64) -> Self {
        t as Self
    }

    fn from_bool(t: bool) -> Self {
        t as Self
    }

    fn from_le_bytes(bytes: &[u8]) -> Self {
        i8::from_le_bytes([bytes[0]])
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

    fn relu(self) -> Self {
        Ord::max(self, 0)
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

    fn max(self, rhs: Self) -> Self {
        let _ = rhs;
        todo!()
    }

    fn max_value() -> Self {
        i8::MAX
    }

    fn min_value() -> Self {
        i8::MIN
    }

    fn epsilon() -> Self {
        0
    }

    fn is_equal(self, rhs: Self) -> bool {
        self == rhs
    }
    
    fn not(self) -> Self {
        todo!()
    }
    
    fn nonzero(self) -> Self {
        todo!()
    }
    
    fn cmpgt(self, rhs: Self) -> Self {
        (self > rhs) as i8
    }
    
    fn or(self, rhs: Self) -> Self {
        (self != 0 || rhs != 0) as i8
    }
}
