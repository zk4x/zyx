use duplicate::duplicate_item;

// # DType traits
//
// This just differentiates between types that can be used inside this library and other types
pub trait DType {}

use crate::accel::cpu;
#[duplicate_item( dtype; [f32]; [f64]; [i8]; [i16]; [i32]; [i64]; [i128]; [isize]; [u8]; [u16]; [u32]; [u64]; [u128]; [usize]; [bool];
    [cpu::Buffer<f32>]; [cpu::Buffer<f64>]; [cpu::Buffer<i32>]; [cpu::Buffer<i64>]; [cpu::Buffer<i128>];
    [cpu::Buffer<u8>]; [cpu::Buffer<u16>]; [cpu::Buffer<u32>]; [cpu::Buffer<u64>]; [cpu::Buffer<u128>]; [cpu::Buffer<bool>];)]
impl DType for dtype {}

pub trait ScalarType {}

#[duplicate_item( dtype; [f32]; [f64]; [i8]; [i16]; [i32]; [i64]; [i128]; [isize]; [u8]; [u16]; [u32]; [u64]; [u128]; [usize]; [bool];)]
impl ScalarType for dtype {}

pub(crate) trait NDimType {}

#[duplicate_item( dtype; [f32]; [f64]; [i8]; [i16]; [i32]; [i64]; [i128]; [isize]; [u8]; [u16]; [u32]; [u64]; [u128]; [usize]; [bool];)]
impl NDimType for crate::accel::cpu::Buffer<dtype> {}

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
