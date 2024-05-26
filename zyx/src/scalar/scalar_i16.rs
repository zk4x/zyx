use crate::{DType, Scalar};
use half::{bf16, f16};

impl Scalar for i16 {
    fn from_bf16(t: bf16) -> Self {
        todo!()
    }

    fn from_f16(t: f16) -> Self {
        todo!()
    }

    fn from_f32(t: f32) -> Self {
        todo!()
    }

    fn from_f64(t: f64) -> Self {
        todo!()
    }

    fn from_u8(t: u8) -> Self {
        todo!()
    }

    fn from_i8(t: i8) -> Self {
        todo!()
    }

    fn from_i16(t: i16) -> Self {
        todo!()
    }

    fn from_i32(t: i32) -> Self {
        todo!()
    }

    fn from_i74(t: i32) -> Self {
        todo!()
    }

    fn from_le_bytes(bytes: &[u8]) -> Self {
        todo!()
    }

    fn dtype() -> DType {
        todo!()
    }

    fn zero() -> Self {
        todo!()
    }

    fn one() -> Self {
        todo!()
    }

    fn byte_size() -> usize {
        todo!()
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
        todo!()
    }

    fn neg(self) -> Self {
        todo!()
    }

    fn relu(self) -> Self {
        todo!()
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
        todo!()
    }

    fn sub(self, rhs: Self) -> Self {
        todo!()
    }

    fn mul(self, rhs: Self) -> Self {
        todo!()
    }

    fn div(self, rhs: Self) -> Self {
        todo!()
    }

    fn pow(self, rhs: Self) -> Self {
        todo!()
    }

    fn cmplt(self, rhs: Self) -> Self {
        todo!()
    }

    fn max(self, rhs: Self) -> Self {
        todo!()
    }

    fn max_value() -> Self {
        todo!()
    }

    fn min_value() -> Self {
        todo!()
    }

    fn epsilon() -> Self {
        todo!()
    }

    fn is_equal(self, rhs: Self) -> bool {
        todo!()
    }
}
