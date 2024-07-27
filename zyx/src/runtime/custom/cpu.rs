use crate::dtype::DType;
use crate::runtime::custom::{Custom, CustomError};
use crate::scalar::Scalar;
use alloc::vec::Vec;

#[cfg(feature = "half")]
use half::{bf16, f16};

#[cfg(feature = "complex")]
use num_complex::Complex;

pub(crate) struct CPU {}

pub(crate) enum CPUBuffer {
    #[cfg(feature = "half")]
    BF16(Vec<bf16>),
    #[cfg(feature = "half")]
    F16(Vec<f16>),
    F32(Vec<f32>),
    F64(Vec<f64>),
    #[cfg(feature = "complex")]
    CF32(Vec<Complex<f32>>),
    #[cfg(feature = "complex")]
    CF64(Vec<Complex<f64>>),
    U8(Vec<u8>),
    I8(Vec<i8>),
    I16(Vec<i16>),
    I32(Vec<i32>),
    I64(Vec<i64>),
    Bool(Vec<bool>),
}

impl CPUBuffer {
    fn len(&self) -> usize {
        return match self {
            #[cfg(feature = "half")]
            CPUBuffer::BF16(x) => x.len(),
            #[cfg(feature = "half")]
            CPUBuffer::F16(x) => x.len(),
            CPUBuffer::F32(x) => x.len(),
            CPUBuffer::F64(x) => x.len(),
            #[cfg(feature = "complex")]
            CPUBuffer::CF32(x) => x.len(),
            #[cfg(feature = "complex")]
            CPUBuffer::CF64(x) => x.len(),
            CPUBuffer::U8(x) => x.len(),
            CPUBuffer::I8(x) => x.len(),
            CPUBuffer::I16(x) => x.len(),
            CPUBuffer::I32(x) => x.len(),
            CPUBuffer::I64(x) => x.len(),
            CPUBuffer::Bool(x) => x.len(),
        };
    }

    fn dtype(&self) -> DType {
        return match self {
            #[cfg(feature = "half")]
            CPUBuffer::BF16(_) => DType::BF16,
            #[cfg(feature = "half")]
            CPUBuffer::F16(_) => DType::F16,
            CPUBuffer::F32(_) => DType::F32,
            CPUBuffer::F64(_) => DType::F64,
            #[cfg(feature = "complex")]
            CPUBuffer::CF32(_) => DType::CF32,
            #[cfg(feature = "complex")]
            CPUBuffer::CF64(_) => DType::CF64,
            CPUBuffer::U8(_) => DType::U8,
            CPUBuffer::I8(_) => DType::I8,
            CPUBuffer::I16(_) => DType::I16,
            CPUBuffer::I32(_) => DType::I32,
            CPUBuffer::I64(_) => DType::I64,
            CPUBuffer::Bool(_) => DType::Bool,
        };
    }
}

impl Custom for CPU {
    type Buffer = CPUBuffer;
    fn initialize() -> Result<Self, CustomError> {
        Ok(Self {})
    }

    fn store_mem<T: Scalar>(&self, data: Vec<T>) -> Result<Self::Buffer, CustomError> {
        Ok(match T::dtype() {
            #[cfg(feature = "half")]
            DType::BF16 => CPUBuffer::BF16(unsafe { core::mem::transmute(data) }),
            #[cfg(feature = "half")]
            DType::F16 => CPUBuffer::F16(unsafe { core::mem::transmute(data) }),
            DType::F32 => CPUBuffer::F32(unsafe { core::mem::transmute(data) }),
            DType::F64 => CPUBuffer::F64(unsafe { core::mem::transmute(data) }),
            #[cfg(feature = "complex")]
            DType::CF32 => CPUBuffer::CF32(unsafe { core::mem::transmute(data) }),
            #[cfg(feature = "complex")]
            DType::CF64 => CPUBuffer::CF64(unsafe { core::mem::transmute(data) }),
            DType::U8 => CPUBuffer::U8(unsafe { core::mem::transmute(data) }),
            DType::I8 => CPUBuffer::I8(unsafe { core::mem::transmute(data) }),
            DType::I16 => CPUBuffer::I16(unsafe { core::mem::transmute(data) }),
            DType::I32 => CPUBuffer::I32(unsafe { core::mem::transmute(data) }),
            DType::I64 => CPUBuffer::I64(unsafe { core::mem::transmute(data) }),
            DType::Bool => CPUBuffer::Bool(unsafe { core::mem::transmute(data) }),
        })
    }

    fn load_mem<T: Scalar>(
        &self,
        buffer: &Self::Buffer,
        length: usize,
    ) -> Result<Vec<T>, CustomError> {
        debug_assert_eq!(buffer.len(), length);
        if T::dtype() != buffer.dtype() {
            return Err(CustomError::BufferDoesNotExist);
        }
        let data: &Vec<T> = match buffer {
            #[cfg(feature = "half")]
            CPUBuffer::BF16(data) => unsafe { core::mem::transmute(data) },
            #[cfg(feature = "half")]
            CPUBuffer::F16(data) => unsafe { core::mem::transmute(data) },
            CPUBuffer::F32(data) => unsafe { core::mem::transmute(data) },
            CPUBuffer::F64(data) => unsafe { core::mem::transmute(data) },
            #[cfg(feature = "complex")]
            CPUBuffer::CF32(data) => unsafe { core::mem::transmute(data) },
            #[cfg(feature = "complex")]
            CPUBuffer::CF64(data) => unsafe { core::mem::transmute(data) },
            CPUBuffer::U8(data) => unsafe { core::mem::transmute(data) },
            CPUBuffer::I8(data) => unsafe { core::mem::transmute(data) },
            CPUBuffer::I16(data) => unsafe { core::mem::transmute(data) },
            CPUBuffer::I32(data) => unsafe { core::mem::transmute(data) },
            CPUBuffer::I64(data) => unsafe { core::mem::transmute(data) },
            CPUBuffer::Bool(data) => unsafe { core::mem::transmute(data) },
        };
        return Ok(data.clone());
    }

    fn deallocate_memory(&mut self, buffer: Self::Buffer) -> Result<(), CustomError> {
        // or nothing at all, but lets be explicit
        core::mem::drop(buffer);
        Ok(())
    }
}
