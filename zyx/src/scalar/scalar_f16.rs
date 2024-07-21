use crate::dtype::DType;
use crate::scalar::Scalar;
use half::{bf16, f16};

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

    fn from_i8(t: i8) -> Self {
        let _ = t;
        todo!()
    }

    fn from_i16(t: i16) -> Self {
        let _ = t;
        todo!()
    }

    fn from_i32(t: i32) -> Self {
        f16::from_f32(t as f32)
    }

    fn from_i64(t: i64) -> Self {
        f16::from_f64(t as f64)
    }

    fn from_le_bytes(bytes: &[u8]) -> Self {
        f16::from_le_bytes([bytes[0], bytes[1]])
    }

    fn dtype() -> DType {
        DType::F32
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

    fn into_f32(self) -> f32 {
        self.to_f32()
    }

    fn into_f64(self) -> f64 {
        self.to_f64()
    }

    fn into_i32(self) -> i32 {
        self.to_f32() as i32
    }

    fn abs(self) -> Self {
        todo!()
    }

    fn reciprocal(self) -> Self {
        f16::ONE / self
    }

    fn floor(self) -> Self {
        todo!()
    }

    fn neg(self) -> Self {
        -self
    }

    fn relu(self) -> Self {
        self.max(f16::ZERO)
    }

    fn sin(self) -> Self {
        f16::from_f32(self.to_f32().sin())
    }

    fn cos(self) -> Self {
        f16::from_f32(self.to_f32().cos())
    }

    fn ln(self) -> Self {
        f16::from_f32(self.to_f32().ln())
    }

    fn exp(self) -> Self {
        f16::from_f32(self.to_f32().exp())
    }

    fn tanh(self) -> Self {
        f16::from_f32(self.to_f32().tanh())
    }

    fn sqrt(self) -> Self {
        f16::from_f32(self.to_f32().sqrt())
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

    fn cmplt(self, rhs: Self) -> Self {
        f16::from_f32((self < rhs) as i32 as f32)
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

    fn epsilon() -> Self {
        f16::from_f32(0.00001)
    }

    fn is_equal(self, rhs: Self) -> bool {
        self.to_f32() == rhs.to_f32()
    }
}
