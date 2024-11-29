use crate::dtype::DType;
use crate::scalar::{Float, Scalar};
use float8::F8E4M3;
use half::{bf16, f16};

impl Scalar for F8E4M3 {
    fn from_bf16(t: bf16) -> Self {
        Self::from_f32(t.to_f32())
    }

    fn from_f8(t: F8E4M3) -> Self {
        t
    }

    fn from_f16(t: f16) -> Self {
        Self::from_f32(t.to_f32())
    }

    fn from_u64(t: u64) -> Self {
        let _ = t;
        todo!()
    }

    fn from_f32(t: f32) -> Self {
        Self::from_f32(t)
    }

    fn from_f64(t: f64) -> Self {
        Self::from_f64(t)
    }

    fn from_u8(t: u8) -> Self {
        let _ = t;
        todo!()
    }

    fn from_u16(t: u16) -> Self {
        let _ = t;
        todo!()
    }

    fn from_u32(t: u32) -> Self {
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

    #[allow(clippy::cast_precision_loss)]
    fn from_i32(t: i32) -> Self {
        Self::from_f32(t as f32)
    }

    #[allow(clippy::cast_precision_loss)]
    fn from_i64(t: i64) -> Self {
        Self::from_f64(t as f64)
    }

    fn from_bool(t: bool) -> Self {
        Self::from_f64(u8::from(t).into())
    }

    fn from_le_bytes(bytes: &[u8]) -> Self {
        let _ = bytes;
        //Self::from_le_bytes(&[bytes[0]])
        todo!()
    }

    fn dtype() -> DType {
        DType::F32
    }

    fn zero() -> Self {
        Self::ZERO
    }

    fn one() -> Self {
        Self::ONE
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
        self.max(Self::ZERO)
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
        Self::from_f32(self.to_f32().pow(rhs.to_f32()))
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
        Self::max(self, rhs)
    }

    fn max_value() -> Self {
        Self::MAX
    }

    fn min_value() -> Self {
        Self::MIN
    }

    fn is_equal(self, rhs: Self) -> bool {
        (self == -Self::INFINITY && rhs == -Self::INFINITY)
            || (self.is_nan() && rhs.is_nan())
            || self.sub(rhs).abs() < self.abs() * Self::from_f32(0.0001)
    }

    fn epsilon() -> Self {
        Self::from_f32(0.00001)
    }

    fn mod_(self, rhs: Self) -> Self {
        self % rhs
    }

    fn noteq(self, rhs: Self) -> bool {
        self != rhs
    }

    fn bitxor(self, rhs: Self) -> Self {
        let _ = rhs;
        //self ^ rhs
        todo!()
    }

    fn bitor(self, rhs: Self) -> Self {
        let _ = rhs;
        //self | rhs
        todo!()
    }

    fn bitand(self, rhs: Self) -> Self {
        let _ = rhs;
        //self & rhs
        todo!()
    }
}

impl Float for F8E4M3 {
    fn reciprocal(self) -> Self {
        Self::ONE / self
    }

    fn floor(self) -> Self {
        todo!()
    }

    fn sin(self) -> Self {
        Self::from_f32(self.to_f32().sin())
    }

    fn cos(self) -> Self {
        Self::from_f32(self.to_f32().cos())
    }

    fn sqrt(self) -> Self {
        Self::from_f32(self.to_f32().sqrt())
    }

    fn exp2(self) -> Self {
        Self::from_f32(self.to_f32().exp2())
    }

    fn log2(self) -> Self {
        Self::from_f32(self.to_f32().log2())
    }
}
