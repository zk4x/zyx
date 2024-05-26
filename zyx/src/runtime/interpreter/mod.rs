use crate::runtime::TensorId;
use crate::scalar::Scalar;
use alloc::collections::BTreeMap;
use alloc::vec::Vec;

pub(super) mod cpu;

pub(super) struct InterpretedBackend<I: Interpreter> {
    interpreter: I,
    buffers: BTreeMap<TensorId, I::Buffer>,
}

#[derive(Debug)]
pub(crate) enum InterpreterError {
    InitializationFailure,
    MemoryAllocationFailure,
    BufferDoesNotExist,
}

pub(super) trait Interpreter: Sized {
    type Buffer;
    fn initialize() -> Result<Self, InterpreterError>;
    fn load_mem<T: Scalar>(
        &self,
        buffer: &Self::Buffer,
        length: usize,
    ) -> Result<Vec<T>, InterpreterError>;
}

impl<I: Interpreter> InterpretedBackend<I> {
    pub(super) fn initialize() -> Result<Self, InterpreterError> {
        Ok(Self {
            interpreter: I::initialize()?,
            buffers: BTreeMap::new(),
        })
    }

    pub(super) fn is_realized(&self, x: TensorId) -> bool {
        self.buffers.contains_key(&x)
    }

    pub(super) fn store<T: Scalar>(&mut self, x: TensorId, data: &[T]) -> Result<(), InterpreterError> {
        todo!()
    }

    // Load values at x, if x is not evaluated, it will return error
    pub(super) fn load<T: Scalar>(
        &self,
        x: TensorId,
        length: usize,
    ) -> Result<Vec<T>, InterpreterError> {
        if let Some(buffer) = self.buffers.get(&x) {
            return self.interpreter.load_mem(buffer, length);
        }
        return Err(InterpreterError::BufferDoesNotExist);
    }
}
