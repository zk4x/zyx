use crate::dtype::DType;
use crate::scalar::Scalar;
use half::{bf16, f16};

impl Scalar for i8 {
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

    fn into_f32(self) -> f32 {
        todo!()
    }

    fn into_f64(self) -> f64 {
        todo!()
    }

    fn into_i32(self) -> i32 {
        todo!()
    }

    fn abs(self) -> Self {
        todo!()
    }

    fn reciprocal(self) -> Self {
        1/self
    }

    fn neg(self) -> Self {
        -self
    }

    fn relu(self) -> Self {
        Ord::max(self, 0)
    }

    fn sin(self) -> Self {
        todo!()
    }

    fn cos(self) -> Self {
        todo!()
    }

    fn ln(self) -> Self {
        todo!()
    }

    fn exp(self) -> Self {
        todo!()
    }

    fn tanh(self) -> Self {
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
}
