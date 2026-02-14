use std::ops::{Add, BitAnd, BitOr, BitXor, Div, Mul, Sub};
use super::Tensor;
use half::{bf16, f16};
use crate::{RT, graph::BOp};

impl<IT: Into<Tensor>> Add<IT> for Tensor {
    type Output = Tensor;
    fn add(self, rhs: IT) -> Self::Output {
        let (x, y) = Tensor::broadcast(self, rhs).unwrap();
        #[allow(clippy::let_and_return)] // otherwise it deadlocks
        let tensor = Tensor { id: RT.lock().binary(x.id, y.id, BOp::Add) };
        tensor
    }
}

impl<IT: Into<Tensor>> Add<IT> for &Tensor {
    type Output = Tensor;
    fn add(self, rhs: IT) -> Self::Output {
        let (x, y) = Tensor::broadcast(self.clone(), rhs).unwrap();
        #[allow(clippy::let_and_return)] // otherwise it deadlocks
        let tensor = Tensor { id: RT.lock().binary(x.id, y.id, BOp::Add) };
        tensor
    }
}

impl<IT: Into<Tensor>> Sub<IT> for Tensor {
    type Output = Tensor;
    fn sub(self, rhs: IT) -> Self::Output {
        let (x, y) = Tensor::broadcast(self, rhs).unwrap();
        #[allow(clippy::let_and_return)] // otherwise it deadlocks
        let tensor = Tensor { id: RT.lock().binary(x.id, y.id, BOp::Sub) };
        tensor
    }
}

impl<IT: Into<Tensor>> Sub<IT> for &Tensor {
    type Output = Tensor;
    fn sub(self, rhs: IT) -> Self::Output {
        let (x, y) = Tensor::broadcast(self.clone(), rhs).unwrap();
        #[allow(clippy::let_and_return)] // otherwise it deadlocks
        let tensor = Tensor { id: RT.lock().binary(x.id, y.id, BOp::Sub) };
        tensor
    }
}

impl<IT: Into<Tensor>> Mul<IT> for Tensor {
    type Output = Tensor;
    fn mul(self, rhs: IT) -> Self::Output {
        let rhs = rhs.into();
        let (x, y) = Tensor::broadcast(self, rhs).unwrap();
        #[allow(clippy::let_and_return)] // otherwise it deadlocks
        let tensor = Tensor { id: RT.lock().binary(x.id, y.id, BOp::Mul) };
        tensor
    }
}

impl<IT: Into<Tensor>> Mul<IT> for &Tensor {
    type Output = Tensor;
    fn mul(self, rhs: IT) -> Self::Output {
        let rhs = rhs.into();
        let (x, y) = Tensor::broadcast(self.clone(), rhs).unwrap();
        #[allow(clippy::let_and_return)] // otherwise it deadlocks
        let tensor = Tensor { id: RT.lock().binary(x.id, y.id, BOp::Mul) };
        tensor
    }
}

impl<IT: Into<Tensor>> Div<IT> for Tensor {
    type Output = Tensor;
    fn div(self, rhs: IT) -> Self::Output {
        let (x, y) = Tensor::broadcast(self, rhs).unwrap();
        #[allow(clippy::let_and_return)] // otherwise it deadlocks
        let tensor = Tensor { id: RT.lock().binary(x.id, y.id, BOp::Div) };
        tensor
    }
}

impl<IT: Into<Tensor>> Div<IT> for &Tensor {
    type Output = Tensor;
    fn div(self, rhs: IT) -> Self::Output {
        let (x, y) = Tensor::broadcast(self.clone(), rhs).unwrap();
        #[allow(clippy::let_and_return)] // otherwise it deadlocks
        let tensor = Tensor { id: RT.lock().binary(x.id, y.id, BOp::Div) };
        tensor
    }
}

impl<IT: Into<Tensor>> BitOr<IT> for Tensor {
    type Output = Tensor;
    fn bitor(self, rhs: IT) -> Self::Output {
        let (x, y) = Tensor::broadcast(self, rhs).unwrap();
        #[allow(clippy::let_and_return)] // otherwise it deadlocks
        let tensor = Tensor { id: RT.lock().binary(x.id, y.id, BOp::BitOr) };
        tensor
    }
}

impl<IT: Into<Tensor>> BitOr<IT> for &Tensor {
    type Output = Tensor;
    fn bitor(self, rhs: IT) -> Self::Output {
        let (x, y) = Tensor::broadcast(self.clone(), rhs).unwrap();
        #[allow(clippy::let_and_return)] // otherwise it deadlocks
        let tensor = Tensor { id: RT.lock().binary(x.id, y.id, BOp::BitOr) };
        tensor
    }
}

impl<IT: Into<Tensor>> BitXor<IT> for Tensor {
    type Output = Tensor;
    fn bitxor(self, rhs: IT) -> Self::Output {
        let (x, y) = Tensor::broadcast(self, rhs).unwrap();
        #[allow(clippy::let_and_return)] // otherwise it deadlocks
        let tensor = Tensor { id: RT.lock().binary(x.id, y.id, BOp::BitXor) };
        tensor
    }
}

impl<IT: Into<Tensor>> BitXor<IT> for &Tensor {
    type Output = Tensor;
    fn bitxor(self, rhs: IT) -> Self::Output {
        let (x, y) = Tensor::broadcast(self.clone(), rhs).unwrap();
        #[allow(clippy::let_and_return)] // otherwise it deadlocks
        let tensor = Tensor { id: RT.lock().binary(x.id, y.id, BOp::BitXor) };
        tensor
    }
}

impl<IT: Into<Tensor>> BitAnd<IT> for Tensor {
    type Output = Tensor;
    fn bitand(self, rhs: IT) -> Self::Output {
        let (x, y) = Tensor::broadcast(self, rhs).unwrap();
        #[allow(clippy::let_and_return)] // otherwise it deadlocks
        let tensor = Tensor { id: RT.lock().binary(x.id, y.id, BOp::BitAnd) };
        tensor
    }
}

impl<IT: Into<Tensor>> BitAnd<IT> for &Tensor {
    type Output = Tensor;
    fn bitand(self, rhs: IT) -> Self::Output {
        let (x, y) = Tensor::broadcast(self.clone(), rhs).unwrap();
        #[allow(clippy::let_and_return)] // otherwise it deadlocks
        let tensor = Tensor { id: RT.lock().binary(x.id, y.id, BOp::BitAnd) };
        tensor
    }
}

macro_rules! impl_trait {
    ($trait:ident for $type:ty, $fn_name:ident) => {
        impl $trait<Tensor> for $type {
            type Output = Tensor;
            fn $fn_name(self, rhs: Tensor) -> Self::Output {
                Tensor::from(self).$fn_name(rhs)
            }
        }

        impl $trait<&Tensor> for $type {
            type Output = Tensor;
            fn $fn_name(self, rhs: &Tensor) -> Self::Output {
                Tensor::from(self).$fn_name(rhs)
            }
        }
    };
}

impl_trait!(Add for bf16, add);
impl_trait!(Add for f16, add);
impl_trait!(Add for f32, add);
impl_trait!(Add for f64, add);
impl_trait!(Add for u8, add);
impl_trait!(Add for u32, add);
impl_trait!(Add for i8, add);
impl_trait!(Add for i16, add);
impl_trait!(Add for i32, add);
impl_trait!(Add for i64, add);
impl_trait!(Add for bool, add);

impl_trait!(Sub for bf16, sub);
impl_trait!(Sub for f16, sub);
impl_trait!(Sub for f32, sub);
impl_trait!(Sub for f64, sub);
impl_trait!(Sub for u8, sub);
impl_trait!(Sub for u32, sub);
impl_trait!(Sub for i8, sub);
impl_trait!(Sub for i16, sub);
impl_trait!(Sub for i32, sub);
impl_trait!(Sub for i64, sub);
impl_trait!(Sub for bool, sub);

impl_trait!(Mul for bf16, mul);
impl_trait!(Mul for f16, mul);
impl_trait!(Mul for f32, mul);
impl_trait!(Mul for f64, mul);
impl_trait!(Mul for u8, mul);
impl_trait!(Mul for u32, mul);
impl_trait!(Mul for i8, mul);
impl_trait!(Mul for i16, mul);
impl_trait!(Mul for i32, mul);
impl_trait!(Mul for i64, mul);
impl_trait!(Mul for bool, mul);

impl_trait!(Div for bf16, div);
impl_trait!(Div for f16, div);
impl_trait!(Div for f32, div);
impl_trait!(Div for f64, div);
impl_trait!(Div for u8, div);
impl_trait!(Div for u32, div);
impl_trait!(Div for i8, div);
impl_trait!(Div for i16, div);
impl_trait!(Div for i32, div);
impl_trait!(Div for i64, div);
impl_trait!(Div for bool, div);

impl_trait!(BitXor for bf16, bitxor);
impl_trait!(BitXor for f16, bitxor);
impl_trait!(BitXor for f32, bitxor);
impl_trait!(BitXor for f64, bitxor);
impl_trait!(BitXor for u8, bitxor);
impl_trait!(BitXor for u32, bitxor);
impl_trait!(BitXor for i8, bitxor);
impl_trait!(BitXor for i16, bitxor);
impl_trait!(BitXor for i32, bitxor);
impl_trait!(BitXor for i64, bitxor);
impl_trait!(BitXor for bool, bitxor);

impl_trait!(BitOr for bf16, bitor);
impl_trait!(BitOr for f16, bitor);
impl_trait!(BitOr for f32, bitor);
impl_trait!(BitOr for f64, bitor);
impl_trait!(BitOr for u8, bitor);
impl_trait!(BitOr for u32, bitor);
impl_trait!(BitOr for i8, bitor);
impl_trait!(BitOr for i16, bitor);
impl_trait!(BitOr for i32, bitor);
impl_trait!(BitOr for i64, bitor);
impl_trait!(BitOr for bool, bitor);

impl_trait!(BitAnd for bf16, bitand);
impl_trait!(BitAnd for f16, bitand);
impl_trait!(BitAnd for f32, bitand);
impl_trait!(BitAnd for f64, bitand);
impl_trait!(BitAnd for u8, bitand);
impl_trait!(BitAnd for u32, bitand);
impl_trait!(BitAnd for i8, bitand);
impl_trait!(BitAnd for i16, bitand);
impl_trait!(BitAnd for i32, bitand);
impl_trait!(BitAnd for i64, bitand);
impl_trait!(BitAnd for bool, bitand);
