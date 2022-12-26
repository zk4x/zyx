use duplicate::duplicate_item;
use crate::{shape::Sh0, ops::{HasShape, HasDType}};

/// # DType
///
/// DType is any type that can be stored inside NDType.
/// It is sometimes necessary to limit some operations to take only DType
/// in order to avoid some cycles.
/// 
/// There are no requirements on DType, except it must implement Clone.
/// Although cloning is not necessarily required for every operation with DType,
/// it is used quite often.
/// That also means that when you implement DType for your custom type, cloning
/// should be pretty fast, otherwise you will face serious performance issues with [cpu::Buffer](crate::device::cpu::Buffer),
/// since this type assumes that passing DType by value is faster than passing it by reference.
/// 
/// If you want to use sparse tensors with chunky DTypes, please create your own deviceerator.
pub trait DType: Clone {}

//use crate::device::cpu;
#[duplicate_item( dtype; [f32]; [f64]; [i8]; [i16]; [i32]; [i64]; [i128]; [isize]; [u8]; [u16]; [u32]; [u64]; [u128]; [usize]; [bool];)]
impl DType for dtype {}

impl<T> HasShape for T
where
    T: DType,
{
    type Sh = Sh0;
}

impl<T> HasDType for T
where
    T: DType,
{
    type T = T;
}

/// Storage type is implemented for every type that can have added gradient
pub trait SType {}

#[duplicate_item( dtype; [f32]; [f64]; [i8]; [i16]; [i32]; [i64]; [i128]; [isize]; [u8]; [u16]; [u32]; [u64]; [u128]; [usize]; [bool];)]
impl SType for dtype {}

impl<Sh, T> SType for crate::device::cpu::Buffer<Sh, T>
where
    Sh: crate::shape::Shape,
    T: DType
{}

/*
/// # NDType
/// 
/// NDType is any type that has more than one dimension.
/// Each NDType has [Shape] and some [DType] stored inside some type of collection.
pub(crate) trait NDType {
    type T: DType;
    type Sh: Shape;
}

impl<T, Sh> NDType for crate::device::cpu::Buffer<T, Sh> {
    type T = T;
    type Sh = Sh;
}
*/

//impl<T, Sh> DType for crate::device::cpu::Buffer<T, Sh> {}

/*pub trait ScalarType {}

#[duplicate_item( dtype; [f32]; [f64]; [i8]; [i16]; [i32]; [i64]; [i128]; [isize]; [u8]; [u16]; [u32]; [u64]; [u128]; [usize]; [bool];)]
impl ScalarType for dtype {}*/

/*#[cfg(feature = "ndarray")]
impl<A, D> DType for ArrayBase<A, D>
where
    A: RawData,
{}

#[cfg(feature = "ndarray")]
use ndarray::{ArrayBase, RawData};

#[cfg(feature = "ndarray")]
impl<A, D> NDimType<Sh> for ArrayBase<A, D>
where
    A: RawData,
{}*/
