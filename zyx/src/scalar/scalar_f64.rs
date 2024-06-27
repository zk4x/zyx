use crate::dtype::DType;
use crate::scalar::Scalar;
use half::{bf16, f16};

impl Scalar for f64 {
    fn from_bf16(t: bf16) -> Self {
        t.into()
    }

    fn from_f16(t: f16) -> Self {
        t.into()
    }

    fn from_f32(t: f32) -> Self {
        t as f64
    }

    fn from_f64(t: f64) -> Self {
        t
    }

    fn from_u8(t: u8) -> Self {
        t as f64
    }

    fn from_i8(t: i8) -> Self {
        t as f64
    }

    fn from_i16(t: i16) -> Self {
        t as f64
    }

    fn from_i32(t: i32) -> Self {
        t as f64
    }

    fn from_i64(t: i64) -> Self {
        t as f64
    }

    fn from_le_bytes(bytes: &[u8]) -> Self {
        f64::from_le_bytes([
            bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5], bytes[6], bytes[7],
        ])
    }

    fn dtype() -> DType {
        DType::F64
    }

    fn zero() -> Self {
        0.
    }

    fn one() -> Self {
        1.
    }

    fn byte_size() -> usize {
        8
    }

    fn into_f32(self) -> f32 {
        self as f32
    }

    fn into_f64(self) -> f64 {
        self
    }

    fn into_i32(self) -> i32 {
        self as i32
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
        libm::sin(self)
    }

    fn cos(self) -> Self {
        libm::cos(self)
    }

    fn ln(self) -> Self {
        libm::log(self)
    }

    fn exp(self) -> Self {
        libm::exp(self)
    }

    fn tanh(self) -> Self {
        libm::tanh(self)
    }

    fn sqrt(self) -> Self {
        libm::sqrt(self)
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
        libm::pow(self, rhs)
    }

    fn cmplt(self, rhs: Self) -> Self {
        (self < rhs) as i32 as f64
    }

    fn max(self, rhs: Self) -> Self {
        f64::max(self, rhs)
    }

    fn max_value() -> Self {
        f64::MAX
    }

    fn min_value() -> Self {
        f64::MIN
    }

    fn epsilon() -> Self {
        0.00001
    }

    fn is_equal(self, rhs: Self) -> bool {
        // Less than 1% error is OK
        (self == -f64::INFINITY && rhs == -f64::INFINITY)
            || (self - rhs).abs() < Self::epsilon()
            || (self - rhs).abs() < self.abs() * 0.01
    }
}
