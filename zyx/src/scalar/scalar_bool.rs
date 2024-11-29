use crate::DType;

use super::Scalar;
use float8::F8E4M3;
use half::{bf16, f16};

impl Scalar for bool {
    fn from_bf16(t: bf16) -> Self {
        t != bf16::ZERO
    }

    fn from_f8(t: F8E4M3) -> Self {
        t != F8E4M3::ZERO
    }

    fn from_f16(t: f16) -> Self {
        t != f16::ZERO
    }

    fn from_u64(t: u64) -> Self {
        t != 0
    }

    fn from_f32(t: f32) -> Self {
        t != 0.
    }

    fn from_f64(t: f64) -> Self {
        t != 0.
    }

    fn from_u8(t: u8) -> Self {
        t != 0
    }

    fn from_u16(t: u16) -> Self {
        t != 0
    }

    fn from_u32(t: u32) -> Self {
        t != 0
    }

    fn from_i8(t: i8) -> Self {
        t != 0
    }

    fn from_i16(t: i16) -> Self {
        t != 0
    }

    fn from_i32(t: i32) -> Self {
        t != 0
    }

    fn from_i64(t: i64) -> Self {
        t != 0
    }

    fn from_bool(t: bool) -> Self {
        t
    }

    fn from_le_bytes(bytes: &[u8]) -> Self {
        bytes[0] != 0
    }

    fn dtype() -> DType {
        DType::Bool
    }

    fn zero() -> Self {
        false
    }

    fn one() -> Self {
        true
    }

    fn byte_size() -> usize {
        1
    }

    fn abs(self) -> Self {
        self
    }

    fn neg(self) -> Self {
        panic!()
    }

    fn relu(self) -> Self {
        panic!()
    }

    fn not(self) -> Self {
        todo!()
    }

    fn nonzero(self) -> Self {
        todo!()
    }

    fn add(self, rhs: Self) -> Self {
        self | rhs
    }

    fn sub(self, rhs: Self) -> Self {
        let _ = rhs;
        panic!()
    }

    fn mul(self, rhs: Self) -> Self {
        self & rhs
    }

    fn div(self, rhs: Self) -> Self {
        let _ = rhs;
        panic!()
    }

    fn pow(self, rhs: Self) -> Self {
        let _ = rhs;
        panic!()
    }

    fn cmplt(self, rhs: Self) -> Self {
        !self & rhs
    }

    fn cmpgt(self, rhs: Self) -> Self {
        self && !rhs
    }

    fn or(self, rhs: Self) -> Self {
        self || rhs
    }

    fn and(self, rhs: Self) -> bool {
        self && rhs
    }

    fn max(self, rhs: Self) -> Self {
        <bool as Ord>::max(self, rhs)
    }

    fn max_value() -> Self {
        true
    }

    fn min_value() -> Self {
        false
    }

    fn is_equal(self, rhs: Self) -> bool {
        self == rhs
    }

    fn epsilon() -> Self {
        false
    }

    fn mod_(self, rhs: Self) -> Self {
        let _ = rhs;
        //self % rhs
        todo!()
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
