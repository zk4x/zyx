use crate::runtime::interpreter::{Interpreter, InterpreterError};
use crate::dtype::DType;
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
        }
    }
}

impl Interpreter for CPU {
    type Buffer = CPUBuffer;
    fn initialize() -> Result<Self, InterpreterError> {
        todo!()
    }
    fn load_mem<T: Scalar>(
        &self,
        buffer: &Self::Buffer,
        length: usize,
    ) -> Result<Vec<T>, InterpreterError> {
        debug_assert_eq!(buffer.len(), length);
        match T::dtype() {
            #[cfg(feature = "half")]
            DType::BF16 => {
                if let CPUBuffer::BF16(data) = buffer {
                    return Ok(unsafe { core::mem::transmute(data.clone()) });
                } else {
                    return Err(InterpreterError::BufferDoesNotExist);
                }
            }
            #[cfg(feature = "half")]
            DType::F16 => {
                if let CPUBuffer::F16(data) = buffer {
                    return Ok(unsafe { core::mem::transmute(data.clone()) });
                } else {
                    return Err(InterpreterError::BufferDoesNotExist);
                }
            }
            DType::F32 => {
                if let CPUBuffer::F32(data) = buffer {
                    return Ok(unsafe { core::mem::transmute(data.clone()) });
                } else {
                    return Err(InterpreterError::BufferDoesNotExist);
                }
            }
            DType::F64 => {
                if let CPUBuffer::F64(data) = buffer {
                    return Ok(unsafe { core::mem::transmute(data.clone()) });
                } else {
                    return Err(InterpreterError::BufferDoesNotExist);
                }
            }
            #[cfg(feature = "complex")]
            DType::CF32 => {
                if let CPUBuffer::CF32(data) = buffer {
                    return Ok(unsafe { core::mem::transmute(data.clone()) });
                } else {
                    return Err(InterpreterError::BufferDoesNotExist);
                }
            }
            #[cfg(feature = "complex")]
            DType::CF64 => {
                if let CPUBuffer::CF64(data) = buffer {
                    return Ok(unsafe { core::mem::transmute(data.clone()) });
                } else {
                    return Err(InterpreterError::BufferDoesNotExist);
                }
            }
            DType::U8 => {
                if let CPUBuffer::U8(data) = buffer {
                    return Ok(unsafe { core::mem::transmute(data.clone()) });
                } else {
                    return Err(InterpreterError::BufferDoesNotExist);
                }
            }
            DType::I8 => {
                if let CPUBuffer::I8(data) = buffer {
                    return Ok(unsafe { core::mem::transmute(data.clone()) });
                } else {
                    return Err(InterpreterError::BufferDoesNotExist);
                }
            }
            DType::I16 => {
                if let CPUBuffer::I16(data) = buffer {
                    return Ok(unsafe { core::mem::transmute(data.clone()) });
                } else {
                    return Err(InterpreterError::BufferDoesNotExist);
                }
            }
            DType::I32 => {
                if let CPUBuffer::I32(data) = buffer {
                    return Ok(unsafe { core::mem::transmute(data.clone()) });
                } else {
                    return Err(InterpreterError::BufferDoesNotExist);
                }
            }
            DType::I64 => {
                if let CPUBuffer::I64(data) = buffer {
                    return Ok(unsafe { core::mem::transmute(data.clone()) });
                } else {
                    return Err(InterpreterError::BufferDoesNotExist);
                }
            }
        }
    }

    fn deallocate_memory(&mut self, buffer: Self::Buffer) -> Result<(), InterpreterError> {
        let _ = buffer;
        todo!()
    }
}
