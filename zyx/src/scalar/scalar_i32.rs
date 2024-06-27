use crate::dtype::DType;
use crate::scalar::Scalar;
use half::{bf16, f16};

impl Scalar for i32 {
    fn from_bf16(t: bf16) -> Self {
        let _ = t;
        todo!()
    }

    fn from_f16(t: f16) -> Self {
        let _ = t;
        todo!()
    }

    fn from_f32(t: f32) -> Self {
        t as i32
    }

    fn from_f64(t: f64) -> Self {
        t as i32
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
        t
    }

    fn from_i64(t: i64) -> Self {
        t as i32
    }

    fn from_le_bytes(bytes: &[u8]) -> Self {
        i32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]])
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

    fn into_f32(self) -> f32 {
        self as f32
    }

    fn into_f64(self) -> f64 {
        self as f64
    }

    fn into_i32(self) -> i32 {
        self
    }

    fn abs(self) -> Self {
        todo!()
    }

    fn reciprocal(self) -> Self {
        1 / self
    }

    fn neg(self) -> Self {
        -self
    }

    fn relu(self) -> Self {
        <i32 as Ord>::max(self, 0)
    }

    fn sin(self) -> Self {
        f32::sin(self as f32) as i32
    }

    fn cos(self) -> Self {
        f32::cos(self as f32) as i32
    }

    fn ln(self) -> Self {
        f32::ln(self as f32) as i32
    }

    fn exp(self) -> Self {
        f32::exp(self as f32) as i32
    }

    fn tanh(self) -> Self {
        f32::tanh(self as f32) as i32
    }

    fn sqrt(self) -> Self {
        (self as f32).sqrt() as i32
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
        i32::pow(self, rhs as u32)
    }

    fn cmplt(self, rhs: Self) -> Self {
        (self < rhs) as i32
    }

    fn max(self, rhs: Self) -> Self {
        <i32 as Ord>::max(self, rhs)
    }

    fn max_value() -> Self {
        i32::MAX
    }

    fn min_value() -> Self {
        i32::MIN
    }

    fn epsilon() -> Self {
        0
    }

    fn is_equal(self, rhs: Self) -> bool {
        self == rhs
    }
}
