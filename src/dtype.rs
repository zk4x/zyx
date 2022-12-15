use duplicate::duplicate_item;
use crate::shape::Shape;

// # DType traits
//
// This just differentiates between types that can be used inside this library and other types
pub trait DType {}

//use crate::accel::cpu;
#[duplicate_item( dtype; [f32]; [f64]; [i8]; [i16]; [i32]; [i64]; [i128]; [isize]; [u8]; [u16]; [u32]; [u64]; [u128]; [usize]; [bool];)]
impl DType for dtype {}

impl<T, Sh> DType for crate::accel::cpu::Buffer<T, Sh> {}

pub trait ScalarType {}

#[duplicate_item( dtype; [f32]; [f64]; [i8]; [i16]; [i32]; [i64]; [i128]; [isize]; [u8]; [u16]; [u32]; [u64]; [u128]; [usize]; [bool];)]
impl ScalarType for dtype {}

pub(crate) trait NDimType<Sh>
where
    Sh: Shape<D = usize>,
{}

#[duplicate_item( dtype; [f32]; [f64]; [i8]; [i16]; [i32]; [i64]; [i128]; [isize]; [u8]; [u16]; [u32]; [u64]; [u128]; [usize]; [bool];)]
impl<Sh> NDimType<Sh> for crate::accel::cpu::Buffer<dtype, Sh>
where
    Sh: Shape<D = usize>,
{}

#[cfg(feature = "ndarray")]
impl<A, D> DType for ArrayBase<A, D>
where
    A: RawData,
{}

#[cfg(feature = "ndarray")]
use ndarray::{ArrayBase, RawData};

#[cfg(feature = "ndarray")]
impl<A, D> NDimType for ArrayBase<A, D>
where
    A: RawData,
{}
