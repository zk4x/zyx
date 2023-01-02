use crate::{
    ops::{HasDType, HasShape},
    shape::Sh0,
};
use duplicate::duplicate_item;

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
/// If you want to use sparse tensors with chunky DTypes, please provide your own device for best performance.
pub trait DType: ocl::OclPrm {
    /// Get this type as static string
    const DTYPE_STR: &'static str;
}

impl DType for f32 {
    const DTYPE_STR: &'static str = "f32";
}
impl DType for f64 {
    const DTYPE_STR: &'static str = "f64";
}
impl DType for i8 {
    const DTYPE_STR: &'static str = "i8";
}
impl DType for i16 {
    const DTYPE_STR: &'static str = "i16";
}
impl DType for i32 {
    const DTYPE_STR: &'static str = "i32";
}
impl DType for i64 {
    const DTYPE_STR: &'static str = "i64";
}
/*impl DType for i128 {
    const DTYPE_STR: &'static str = "i128";
}*/
impl DType for isize {
    const DTYPE_STR: &'static str = "isize";
}
impl DType for u8 {
    const DTYPE_STR: &'static str = "u8";
}
impl DType for u16 {
    const DTYPE_STR: &'static str = "u16";
}
impl DType for u32 {
    const DTYPE_STR: &'static str = "u32";
}
impl DType for u64 {
    const DTYPE_STR: &'static str = "u64";
}
/*impl DType for u128 {
    const DTYPE_STR: &'static str = "u128";
}*/
impl DType for usize {
    const DTYPE_STR: &'static str = "usize";
}
/*impl DType for bool {
    const DTYPE_STR: &'static str = "bool";
}*/

/*impl<T> HasShape for T
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
}*/

/// Storage type is implemented for every type that can have added gradient.
/// That is every type that supports all tensor backward operations.
///
/// [DType] is every rust primitive - f32, f64, i32, i64 etc.
/// [SType] is every rust primitive and also [cpu::Buffer](crate::device::cpu::Buffer) and [opencl::Buffer](crate::device::opencl::Buffer).
///
/// That is [SType] is also implemented for ndimensional types.
pub trait SType {}

#[duplicate_item( dtype; [f32]; [f64]; [i8]; [i16]; [i32]; [i64]; [i128]; [isize]; [u8]; [u16]; [u32]; [u64]; [u128]; [usize]; [bool];)]
impl SType for dtype {}

/*impl<Sh, T> SType for crate::device::cpu::Buffer<'_, Sh, T>
where
    Sh: crate::shape::Shape,
    T: DType,
{
}

impl<Sh, T> SType for crate::device::opencl::Buffer<'_, Sh, T>
where
    Sh: crate::shape::Shape,
    T: DType + ocl::OclPrm,
{
}*/

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
