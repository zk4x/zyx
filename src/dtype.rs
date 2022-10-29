use crate::ops::{ConvertFrom, FromVec};
use duplicate::duplicate_item;

// # DType traits
//
// This just differentiates between types that can be used inside this library and other types
pub trait DType {}

#[duplicate_item( dtype; [f32]; [f64]; [i8]; [i16]; [i32]; [i64]; [i128]; [isize]; [u8]; [u16]; [u32]; [u64]; [u128]; [usize]; [bool];)]
impl DType for dtype {}
#[duplicate_item( dtype; [f32]; [f64]; [i8]; [i16]; [i32]; [i64]; [i128]; [isize]; [u8]; [u16]; [u32]; [u64]; [u128]; [usize]; [bool];)]
impl DType for crate::accel::cpu::Buffer<dtype> {}

pub trait ScalarType {}

#[duplicate_item( dtype; [f32]; [f64]; [i8]; [i16]; [i32]; [i64]; [i128]; [isize]; [u8]; [u16]; [u32]; [u64]; [u128]; [usize]; [bool];)]
impl ScalarType for dtype {}

pub(crate) trait NDimType {}

#[duplicate_item( dtype; [f32]; [f64]; [i8]; [i16]; [i32]; [i64]; [i128]; [isize]; [u8]; [u16]; [u32]; [u64]; [u128]; [usize]; [bool];)]
impl NDimType for crate::accel::cpu::Buffer<dtype> {}

// TODO: implement DType for NDArray

#[duplicate_item( dtype; [f32]; [f64]; [i8]; [i16]; [i32]; [i64]; [i128]; [isize]; [u8]; [u16]; [u32]; [u64]; [u128]; [usize]; [bool];)]

impl<S> ConvertFrom<dtype> for S
where
    S: FromVec<dtype>,
{
    fn cfrom(x: dtype) -> Self {
        S::from_vec(vec![x], [1])
    }
}