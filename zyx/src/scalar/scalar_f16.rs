use crate::dtype::DType;
use crate::scalar::{Float, Scalar};
use half::{bf16, f16};

impl Scalar for f16 {
    fn from_bf16(t: bf16) -> Self {
        f16::from_f32(t.to_f32())
    }

    fn from_f16(t: f16) -> Self {
        f16::from_f32(t.to_f32())
    }

    fn from_u64(t: u64) -> Self {
        let _ = t;
        todo!()
    }

    fn from_f32(t: f32) -> Self {
        f16::from_f32(t)
    }

    fn from_f64(t: f64) -> Self {
        f16::from_f64(t)
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
        f16::from_f64(t.into())
    }

    fn from_i8(t: i8) -> Self {
        let _ = t;
        todo!()
    }

    fn from_i16(t: i16) -> Self {
        let _ = t;
        todo!()
    }

    #[allow(clippy::cast_lossless)]
    fn from_i32(t: i32) -> Self {
        f16::from_f64(t as f64)
    }

    #[allow(clippy::cast_precision_loss)]
    fn from_i64(t: i64) -> Self {
        f16::from_f64(t as f64)
    }

    #[allow(clippy::cast_lossless)]
    fn from_bool(t: bool) -> Self {
        f16::from_f64(t as i8 as f64)
    }

    fn from_le_bytes(bytes: &[u8]) -> Self {
        f16::from_le_bytes([bytes[0], bytes[1]])
    }

    fn dtype() -> DType {
        DType::F16
    }

    fn zero() -> Self {
        f16::ZERO
    }

    fn one() -> Self {
        f16::ONE
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
        self.max(f16::ZERO)
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
        f16::from_f32(self.to_f32().pow(rhs.to_f32()))
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
        f16::max(self, rhs)
    }

    fn max_value() -> Self {
        f16::MAX
    }

    fn min_value() -> Self {
        f16::MIN
    }

    fn is_equal(self, rhs: Self) -> bool {
        (self == -Self::INFINITY && rhs == -Self::INFINITY)
            || (self.is_nan() && rhs.is_nan())
            || self.sub(rhs).abs() < self.abs() * f16::from_f32(0.0001)
    }

    fn epsilon() -> Self {
        f16::from_f32(0.00001)
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
    
    fn bitshiftleft(self, rhs: Self) -> Self {
        let _ = rhs;
        todo!()
    }
    
    fn bitshiftright(self, rhs: Self) -> Self {
        let _ = rhs;
        todo!()
    }
}

impl Float for f16 {
    fn reciprocal(self) -> Self {
        f16::ONE / self
    }

    fn sin(self) -> Self {
        f16::from_f32(self.to_f32().sin())
    }

    fn cos(self) -> Self {
        f16::from_f32(self.to_f32().cos())
    }

    fn sqrt(self) -> Self {
        f16::from_f32(self.to_f32().sqrt())
    }

    fn exp2(self) -> Self {
        f16::from_f32(self.to_f32().exp2())
    }

    fn log2(self) -> Self {
        f16::from_f32(self.to_f32().log2())
    }

    fn floor(self) -> Self {
        f16::from_f32(self.to_f32().floor())
    }
}
