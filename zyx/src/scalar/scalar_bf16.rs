use crate::dtype::DType;
use crate::scalar::{Float, Scalar};
use float8::F8E4M3;
use half::{bf16, f16};

impl Scalar for bf16 {
    fn from_bf16(t: bf16) -> Self {
        t
    }

    fn from_f8(t: F8E4M3) -> Self {
        bf16::from_f32(t.into())
    }

    fn from_f16(t: f16) -> Self {
        bf16::from_f32(t.into())
    }

    fn from_u64(t: u64) -> Self {
        let _ = t;
        todo!()
    }

    fn from_f32(t: f32) -> Self {
        bf16::from_f32(t)
    }

    fn from_f64(t: f64) -> Self {
        bf16::from_f64(t)
    }

    fn from_u8(t: u8) -> Self {
        bf16::from_f32(f32::from(t))
    }

    fn from_u16(t: u16) -> Self {
        bf16::from_f32(f32::from(t))
    }

    fn from_u32(t: u32) -> Self {
        bf16::from_f64(f64::from(t))
    }

    fn from_i8(t: i8) -> Self {
        bf16::from_f32(f32::from(t))
    }

    fn from_i16(t: i16) -> Self {
        bf16::from_f32(f32::from(t))
    }

    fn from_i32(t: i32) -> Self {
        bf16::from_f32(f32::from(u16::try_from(t).unwrap()))
    }

    fn from_i64(t: i64) -> Self {
        bf16::from_f32(f32::from(u16::try_from(t).unwrap()))
    }

    fn from_bool(t: bool) -> Self {
        bf16::from_f32(f32::from(t))
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

    fn abs(self) -> Self {
        self.max(-self)
    }

    fn neg(self) -> Self {
        -self
    }

    fn relu(self) -> Self {
        self.max(bf16::ZERO)
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

    fn pow(self, _rhs: Self) -> Self {
        todo!()
    }

    fn cmplt(self, rhs: Self) -> bool {
        self < rhs
    }

    fn cmpgt(self, rhs: Self) -> bool {
        self > rhs
    }

    fn or(self, rhs: Self) -> bool {
        self != Self::ZERO || rhs != Self::ZERO
    }

    fn and(self, rhs: Self) -> bool {
        self != Self::ZERO && rhs != Self::ZERO
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

    fn is_equal(self, rhs: Self) -> bool {
        self == rhs
    }

    fn epsilon() -> Self {
        bf16::MIN_POSITIVE
    }

    fn mod_(self, rhs: Self) -> Self {
        self % rhs
    }

    fn noteq(self, rhs: Self) -> bool {
        self != rhs
    }

    fn bitxor(self, rhs: Self) -> Self {
        let _ = rhs;
        //self | rhs
        todo!()
    }

    fn bitor(self, rhs: Self) -> Self {
        let _ = rhs;
        //self ^ rhs
        todo!()
    }

    fn bitand(self, rhs: Self) -> Self {
        let _ = rhs;
        //self & rhs
        todo!()
    }
}

impl Float for bf16 {
    fn reciprocal(self) -> Self {
        bf16::ONE / self
    }

    fn floor(self) -> Self {
        todo!()
    }

    fn sin(self) -> Self {
        todo!()
    }

    fn cos(self) -> Self {
        todo!()
    }

    fn sqrt(self) -> Self {
        todo!()
    }

    fn exp2(self) -> Self {
        todo!()
    }

    fn log2(self) -> Self {
        todo!()
    }
}
