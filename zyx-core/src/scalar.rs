use crate::dtype::DType;

/// Scalar trait is implemented for all [dtypes](DType)
pub trait Scalar: Clone + Sized + core::fmt::Debug + 'static {
    /// From f32
    fn from_f32(t: f32) -> Self;
    /// From f64
    fn from_f64(t: f64) -> Self;
    /// From i32
    fn from_i32(t: i32) -> Self;
    /// From little endian bytes
    fn from_le_bytes(bytes: &[u8]) -> Self;
    /// Get dtype of Self
    fn dtype() -> DType;
    /// Get zero of Self
    fn zero() -> Self;
    /// Get one of Self
    fn one() -> Self;
    /// Bute size of Self
    fn byte_size() -> usize;
    /// Convert self into f32
    fn into_f32(self) -> f32;
    /// Convert self into f64
    fn into_f64(self) -> f64;
    /// Convert self into i32
    fn into_i32(self) -> i32;
    /// 1/self
    fn reciprocal(self) -> Self;
    /// Neg
    fn neg(self) -> Self;
    /// ReLU
    fn relu(self) -> Self;
    /// Sin
    fn sin(self) -> Self;
    /// Cos
    fn cos(self) -> Self;
    /// Ln
    fn ln(self) -> Self;
    /// Exp
    fn exp(self) -> Self;
    /// Tanh
    fn tanh(self) -> Self;
    /// Square root of this scalar.
    /// That this function may be imprecise.
    fn sqrt(self) -> Self;
    /// Add
    fn add(self, rhs: Self) -> Self;
    /// Sub
    fn sub(self, rhs: Self) -> Self;
    /// Mul
    fn mul(self, rhs: Self) -> Self;
    /// Div
    fn div(self, rhs: Self) -> Self;
    /// Pow
    fn pow(self, rhs: Self) -> Self;
    /// Compare less than
    fn cmplt(self, rhs: Self) -> Self;
    /// Max of two numbers
    fn max(self, rhs: Self) -> Self;
    /// Max value of this dtype
    fn max_value() -> Self;
    /// Min value of this dtype
    fn min_value() -> Self;
    /// Very small value of scalar, very close to zero
    fn epsilon() -> Self;
    /// Comparison for scalars,
    /// if they are floats, this checks for diffs > Self::epsilon()
    fn is_equal(self, rhs: Self) -> bool;
}

impl Scalar for f32 {
    fn from_f32(t: f32) -> Self {
        t
    }

    fn from_f64(t: f64) -> Self {
        t as f32
    }

    fn from_i32(t: i32) -> Self {
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
        f32::sin(self)
    }

    fn cos(self) -> Self {
        f32::cos(self)
    }

    fn exp(self) -> Self {
        f32::exp(self)
    }

    fn ln(self) -> Self {
        f32::ln(self)
    }

    fn tanh(self) -> Self {
        f32::tanh(self)
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
        f32::powf(self, rhs)
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

impl Scalar for f64 {
    fn from_f32(t: f32) -> Self {
        t as f64
    }

    fn from_f64(t: f64) -> Self {
        t
    }

    fn from_i32(t: i32) -> Self {
        t as f64
    }

    fn from_le_bytes(bytes: &[u8]) -> Self {
        f64::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5], bytes[6], bytes[7]])
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

    fn into_f32(self) -> f32 {
        self as f32
    }

    fn into_f64(self) -> f64 {
        self
    }

    fn into_i32(self) -> i32 {
        self as i32
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
        f64::sin(self)
    }

    fn cos(self) -> Self {
        f64::cos(self)
    }

    fn exp(self) -> Self {
        f64::exp(self)
    }

    fn ln(self) -> Self {
        f64::ln(self)
    }

    fn tanh(self) -> Self {
        f64::tanh(self)
    }

    fn sqrt(self) -> Self {
        f64::sqrt(self)
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
        f64::powf(self, rhs)
    }

    fn cmplt(self, rhs: Self) -> Self {
        (self < rhs) as i32 as f64
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

    fn epsilon() -> Self {
        0.00001
    }

    fn is_equal(self, rhs: Self) -> bool {
        // Less than 1% error is OK
        (self == -f64::INFINITY && rhs == -f64::INFINITY)
            || (self - rhs).abs() < Self::epsilon()
            || (self - rhs).abs() < self.abs() * 0.01
    }
}

impl Scalar for i32 {
    fn from_f32(t: f32) -> Self {
        t as i32
    }

    fn from_f64(t: f64) -> Self {
        t as i32
    }

    fn from_i32(t: i32) -> Self {
        t
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

    fn into_f32(self) -> f32 {
        self as f32
    }

    fn into_f64(self) -> f64 {
        self as f64
    }

    fn into_i32(self) -> i32 {
        self
    }

    fn reciprocal(self) -> Self {
        1 / self
    }

    fn neg(self) -> Self {
        -self
    }

    fn relu(self) -> Self {
        <i32 as Ord>::max(self, 0)
    }

    fn sin(self) -> Self {
        f32::sin(self as f32) as i32
    }

    fn cos(self) -> Self {
        f32::cos(self as f32) as i32
    }

    fn exp(self) -> Self {
        f32::exp(self as f32) as i32
    }

    fn ln(self) -> Self {
        f32::ln(self as f32) as i32
    }

    fn tanh(self) -> Self {
        f32::tanh(self as f32) as i32
    }

    fn sqrt(self) -> Self {
        (self as f32).sqrt() as i32
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
        i32::pow(self, rhs as u32)
    }

    fn cmplt(self, rhs: Self) -> Self {
        (self < rhs) as i32
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

    fn epsilon() -> Self {
        0
    }

    fn is_equal(self, rhs: Self) -> bool {
        self == rhs
    }
}
