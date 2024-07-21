use crate::dtype::DType;
use crate::scalar::Scalar;
use half::{bf16, f16};

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
        bf16::from_f32(t.into_f32())
    }

    fn from_i8(t: i8) -> Self {
        bf16::from_f32(t.into_f32())
    }

    fn from_i16(t: i16) -> Self {
        bf16::from_f32(t.into_f32())
    }

    fn from_i32(t: i32) -> Self {
        bf16::from_f32(t.into_f32())
    }

    fn from_i64(t: i64) -> Self {
        bf16::from_f32(t.into_f32())
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

    fn into_f32(self) -> f32 {
        self.into()
    }

    fn into_f64(self) -> f64 {
        self.into()
    }

    fn into_i32(self) -> i32 {
        self.into_f32().into_i32()
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
        bf16::from_f32(self.into_f32().pow(rhs.into_f32()))
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
}
