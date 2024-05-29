use crate::runtime::interpreter::{Interpreter, InterpreterError};
use crate::{DType, Scalar};
use alloc::vec::Vec;
use half::{bf16, f16};
use num_complex::Complex;

pub(crate) struct CPU {}

pub(crate) enum CPUBuffer {
    BF16(Vec<bf16>),
    F16(Vec<f16>),
    F32(Vec<f32>),
    F64(Vec<f64>),
    CF32(Vec<Complex<f32>>),
    CF64(Vec<Complex<f64>>),
    U8(Vec<u8>),
    I8(Vec<i8>),
    I16(Vec<i16>),
    I32(Vec<i32>),
    I64(Vec<i64>),
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
        match T::dtype() {
            DType::BF16 => {
                if let CPUBuffer::BF16(data) = buffer {
                    return Ok(unsafe { core::mem::transmute(data.clone()) });
                } else {
                    return Err(InterpreterError::BufferDoesNotExist);
                }
            }
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
            DType::CF32 => {
                if let CPUBuffer::CF32(data) = buffer {
                    return Ok(unsafe { core::mem::transmute(data.clone()) });
                } else {
                    return Err(InterpreterError::BufferDoesNotExist);
                }
            }
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
        todo!()
    }
}
