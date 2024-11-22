use crate::dtype::DType;
use crate::scalar::Scalar;
use float8::F8E4M3;
use half::{bf16, f16};

impl Scalar for i32 {
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

    #[allow(clippy::cast_possible_truncation)]
    fn from_f32(t: f32) -> Self {
        t as i32
    }

    #[allow(clippy::cast_possible_truncation)]
    fn from_f64(t: f64) -> Self {
        t as i32
    }

    fn from_u8(t: u8) -> Self {
        t.into()
    }

    fn from_u32(t: u32) -> Self {
        i32::try_from(t).unwrap()
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

    #[allow(clippy::cast_possible_truncation)]
    fn from_i64(t: i64) -> Self {
        t as i32
    }

    fn from_bool(t: bool) -> Self {
        t.into()
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

    fn abs(self) -> Self {
        todo!()
    }

    fn neg(self) -> Self {
        -self
    }

    fn relu(self) -> Self {
        <i32 as Ord>::max(self, 0)
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
        i32::pow(self, u32::try_from(rhs).unwrap())
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
        <i32 as Ord>::max(self, rhs)
    }

    fn max_value() -> Self {
        i32::MAX
    }

    fn min_value() -> Self {
        i32::MIN
    }

    fn is_equal(self, rhs: Self) -> bool {
        self == rhs
    }

    fn epsilon() -> Self {
        0
    }

    fn mod_(self, rhs: Self) -> Self {
        self % rhs
    }

    fn noteq(self, rhs: Self) -> bool {
        self != rhs
    }

    fn bitxor(self, rhs: Self) -> Self {
        self ^ rhs
    }

    fn bitor(self, rhs: Self) -> Self {
        self | rhs
    }

    fn bitand(self, rhs: Self) -> Self {
        self & rhs
    }
}
