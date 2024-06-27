use crate::dtype::DType;
use crate::scalar::Scalar;
use half::{bf16, f16};

impl Scalar for f32 {
    fn from_bf16(t: bf16) -> Self {
        let _ = t;
        todo!()
    }

    fn from_f16(t: f16) -> Self {
        let _ = t;
        todo!()
    }

    fn from_f32(t: f32) -> Self {
        t
    }

    fn from_f64(t: f64) -> Self {
        t as f32
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
        t as f32
    }

    fn from_i64(t: i64) -> Self {
        t as f32
    }

    fn from_le_bytes(bytes: &[u8]) -> Self {
        f32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]])
    }

    fn dtype() -> DType {
        DType::F32
    }

    fn zero() -> Self {
        0.
    }

    fn one() -> Self {
        1.
    }

    fn byte_size() -> usize {
        4
    }

    fn into_f32(self) -> f32 {
        self
    }

    fn into_f64(self) -> f64 {
        self as f64
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
        libm::sinf(self)
    }

    fn cos(self) -> Self {
        libm::cosf(self)
    }

    fn ln(self) -> Self {
        libm::logf(self)
    }

    fn exp(self) -> Self {
        libm::expf(self)
    }

    fn tanh(self) -> Self {
        libm::tanhf(self)
    }

    fn sqrt(self) -> Self {
        // good enough (error of ~ 5%)
        if self >= 0. {
            Self::from_bits((self.to_bits() + 0x3f80_0000) >> 1)
        } else {
            Self::NAN
        }
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
        libm::powf(self, rhs)
    }

    fn cmplt(self, rhs: Self) -> Self {
        (self < rhs) as i32 as f32
    }

    fn max(self, rhs: Self) -> Self {
        f32::max(self, rhs)
    }

    fn max_value() -> Self {
        f32::MAX
    }

    fn min_value() -> Self {
        f32::MIN
    }

    fn epsilon() -> Self {
        0.00001
    }

    fn is_equal(self, rhs: Self) -> bool {
        // Less than 1% error is OK
        (self == -f32::INFINITY && rhs == -f32::INFINITY)
            || (self - rhs).abs() < Self::epsilon()
            || (self - rhs).abs() < self.abs() * 0.01
    }
}
