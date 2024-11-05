use crate::dtype::DType;
use crate::scalar::Scalar;
use float8::F8E4M3;
use half::{bf16, f16};

impl Scalar for i16 {
    #[allow(clippy::cast_possible_truncation)]
    fn from_bf16(t: bf16) -> Self {
        t.to_f32() as i16
    }

    #[allow(clippy::cast_possible_truncation)]
    fn from_f8(t: F8E4M3) -> Self {
        t.to_f32() as i16
    }

    #[allow(clippy::cast_possible_truncation)]
    fn from_f16(t: f16) -> Self {
        t.to_f32() as i16
    }

    #[allow(clippy::cast_possible_truncation)]
    fn from_f32(t: f32) -> Self {
        t as i16
    }

    #[allow(clippy::cast_possible_truncation)]
    fn from_f64(t: f64) -> Self {
        t as i16
    }

    fn from_u8(t: u8) -> Self {
        t.into()
    }

    #[allow(clippy::cast_possible_truncation)]
    fn from_u32(t: u32) -> Self {
        t as i16
    }

    fn from_i8(t: i8) -> Self {
        t.into()
    }

    fn from_i16(t: i16) -> Self {
        t
    }

    #[allow(clippy::cast_possible_truncation)]
    fn from_i32(t: i32) -> Self {
        t as i16
    }

    #[allow(clippy::cast_possible_truncation)]
    fn from_i64(t: i64) -> Self {
        t as i16
    }

    fn from_bool(t: bool) -> Self {
        t.into()
    }

    fn from_le_bytes(bytes: &[u8]) -> Self {
        i16::from_le_bytes([bytes[0], bytes[1]])
    }

    fn dtype() -> DType {
        DType::I16
    }

    fn zero() -> Self {
        0
    }

    fn one() -> Self {
        1
    }

    fn byte_size() -> usize {
        2
    }

    fn abs(self) -> Self {
        todo!()
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
        (self < rhs).into()
    }

    fn max(self, rhs: Self) -> Self {
        Ord::max(self, rhs)
    }

    fn max_value() -> Self {
        i16::MAX
    }

    fn min_value() -> Self {
        i16::MIN
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
        (self > rhs).into()
    }

    fn or(self, rhs: Self) -> Self {
        (self != 0 || rhs != 0).into()
    }
}
