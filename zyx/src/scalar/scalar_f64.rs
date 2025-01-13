use crate::dtype::DType;
use crate::scalar::{Float, Scalar};
use float8::F8E4M3;
use half::{bf16, f16};

impl Scalar for f64 {
    fn from_bf16(t: bf16) -> Self {
        t.into()
    }

    fn from_f8(t: F8E4M3) -> Self {
        t.into()
    }

    fn from_f16(t: f16) -> Self {
        t.into()
    }

    fn from_u64(t: u64) -> Self {
        let _ = t;
        todo!()
    }

    fn from_f32(t: f32) -> Self {
        f64::from(t)
    }

    fn from_f64(t: f64) -> Self {
        t
    }

    fn from_u8(t: u8) -> Self {
        f64::from(t)
    }

    fn from_u16(t: u16) -> Self {
        t.into()
    }

    fn from_u32(t: u32) -> Self {
        t.into()
    }

    fn from_i8(t: i8) -> Self {
        t.into()
    }

    fn from_i16(t: i16) -> Self {
        t.into()
    }

    fn from_i32(t: i32) -> Self {
        t.into()
    }

    #[allow(clippy::cast_precision_loss)]
    fn from_i64(t: i64) -> Self {
        t as f64
    }

    fn from_bool(t: bool) -> Self {
        t.into()
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

    fn abs(self) -> Self {
        self.abs()
    }

    fn neg(self) -> Self {
        -self
    }

    fn relu(self) -> Self {
        self.max(0.)
    }

    fn not(self) -> Self {
        if self == 0. {
            1.
        } else {
            0.
        }
    }

    fn nonzero(self) -> Self {
        u8::from(self != 0.).into()
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
        self.powf(rhs)
    }

    fn cmplt(self, rhs: Self) -> bool {
        self < rhs
    }

    fn cmpgt(self, rhs: Self) -> bool {
        self > rhs
    }

    fn or(self, rhs: Self) -> bool {
        self != 0. || rhs != 0.
    }

    fn and(self, rhs: Self) -> bool {
        self != 0. && rhs != 0.
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

    fn is_equal(self, rhs: Self) -> bool {
        // Less than 0.1% error is OK
        (self == -f64::INFINITY && rhs == -f64::INFINITY)
            || (self - rhs).abs() <= self.abs() * 0.001
    }

    fn epsilon() -> Self {
        0.00001
    }

    fn mod_(self, rhs: Self) -> Self {
        self % rhs
    }

    fn noteq(self, rhs: Self) -> bool {
        !self.is_equal(rhs)
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

impl Float for f64 {
    fn exp2(self) -> Self {
        self.exp2()
    }

    fn log2(self) -> Self {
        self.log2()
    }

    fn reciprocal(self) -> Self {
        1.0 / self
    }

    fn floor(self) -> Self {
        self.floor()
    }

    fn sin(self) -> Self {
        f64::sin(self)
    }

    fn cos(self) -> Self {
        f64::cos(self)
    }

    fn sqrt(self) -> Self {
        f64::sqrt(self)
    }
}
