use crate::dtype::DType;

/// Scalar trait is implemented for all [dtypes](DType)
pub trait Scalar: Clone + core::fmt::Debug + 'static {
    /// Get dtype of Self
    fn dtype() -> DType;
    /// Get zero of Self
    fn zero() -> Self;
    /// Bute size of Self
    fn byte_size() -> usize;
    /// Convert self into f32
    fn into_f32(self) -> f32;
    /// Convert self into i32
    fn into_i32(self) -> i32;
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
    /// Comparison for scalars,
    /// if they are floats, this checks for diffs > 0.000001
    fn is_equal(self, rhs: Self) -> bool;
}

impl Scalar for f32 {
    fn dtype() -> DType {
        DType::F32
    }

    fn zero() -> Self {
        0.
    }

    fn byte_size() -> usize {
        4
    }

    fn into_f32(self) -> f32 {
        self
    }

    fn into_i32(self) -> i32 {
        self as i32
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

    fn is_equal(self, rhs: Self) -> bool {
        (self - rhs).abs() < 0.000001
    }
}

impl Scalar for i32 {
    fn dtype() -> DType {
        DType::I32
    }

    fn zero() -> Self {
        0
    }

    fn byte_size() -> usize {
        4
    }

    fn into_f32(self) -> f32 {
        self as f32
    }

    fn into_i32(self) -> i32 {
        self
    }

    fn neg(self) -> Self {
        -self
    }

    fn relu(self) -> Self {
        self.max(0)
    }

    fn sin(self) -> Self {
        panic!()
    }

    fn cos(self) -> Self {
        panic!()
    }

    fn exp(self) -> Self {
        panic!()
    }

    fn ln(self) -> Self {
        panic!()
    }

    fn tanh(self) -> Self {
        panic!()
    }

    fn sqrt(self) -> Self {
        (self as f32).sqrt() as i32
    }

    fn is_equal(self, rhs: Self) -> bool {
        self == rhs
    }
}
