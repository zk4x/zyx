use crate::dtype::DType;

pub trait Scalar: Clone + 'static {
    fn dtype() -> DType;
    fn zero() -> Self;
    fn byte_size() -> usize;
    fn into_f32(self) -> f32;
    fn into_i32(self) -> i32;
    /// Square root of this scalar.
    /// Note that this function may be imprecise.
    fn sqrt(self) -> Self;
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

    fn sqrt(self) -> Self {
        // good enough (error of ~ 5%)
        if self >= 0. {
            Self::from_bits((self.to_bits() + 0x3f80_0000) >> 1)
        } else {
            Self::NAN
        }
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

    fn sqrt(self) -> Self {
        (self as f32).sqrt() as i32
    }
}
