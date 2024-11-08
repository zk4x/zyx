use crate::dtype::DType;
use crate::scalar::Scalar;
use float8::F8E4M3;
use half::{bf16, f16};

impl Scalar for u8 {
    fn from_bf16(t: bf16) -> Self {
        let _ = t;
        todo!()
    }

    fn from_f8(t: F8E4M3) -> Self {
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
        t
    }

    fn from_u32(t: u32) -> Self {
        t.try_into().unwrap()
    }

    fn from_i8(t: i8) -> Self {
        t.try_into().unwrap()
    }

    fn from_i16(t: i16) -> Self {
        t.try_into().unwrap()
    }

    fn from_i32(t: i32) -> Self {
        t.try_into().unwrap()
    }

    fn from_i64(t: i64) -> Self {
        t.try_into().unwrap()
    }

    fn from_bool(t: bool) -> Self {
        t.into()
    }

    fn from_le_bytes(bytes: &[u8]) -> Self {
        u8::from_le_bytes([bytes[0]])
    }

    fn dtype() -> DType {
        DType::U8
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
        todo!()
    }

    fn neg(self) -> Self {
        todo!()
    }

    fn relu(self) -> Self {
        todo!()
    }

    fn not(self) -> Self {
        todo!()
    }

    fn nonzero(self) -> Self {
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

    fn cmplt(self, rhs: Self) -> bool {
        self < rhs
    }

    fn cmpgt(self, rhs: Self) -> bool {
        self > rhs
    }

    fn or(self, rhs: Self) -> bool {
        self != 0 || rhs != 0
    }

    fn and(self, rhs: Self) -> bool {
        self != 0 && rhs != 0
    }

    fn max(self, rhs: Self) -> Self {
        Ord::max(self, rhs)
    }

    fn max_value() -> Self {
        u8::MAX
    }

    fn min_value() -> Self {
        u8::MIN
    }

    fn is_equal(self, rhs: Self) -> bool {
        self == rhs
    }

    fn epsilon() -> Self {
        0
    }
}
