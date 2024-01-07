use crate::dtype::DType;

pub trait Scalar: Clone {
    fn dtype() -> DType;
    fn into_f32(self) -> f32;
    fn into_i32(self) -> i32;
}

impl Scalar for f32 {
    fn dtype() -> DType {
        DType::F32
    }

    fn into_f32(self) -> f32 {
        self
    }

    fn into_i32(self) -> i32 {
        self as i32
    }
}

impl Scalar for i32 {
    fn dtype() -> DType {
        DType::I32
    }

    fn into_f32(self) -> f32 {
        self as f32
    }

    fn into_i32(self) -> i32 {
        self
    }
}
