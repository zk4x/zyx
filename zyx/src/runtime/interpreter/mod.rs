use crate::scalar::Scalar;

pub(super) mod cpu;

pub(super) struct InterpretedBackend<I> {
    interpreter: I,
}

pub(crate) enum InterpreterError {
    InitializationFailure,
    MemoryAllocationFailure,
}

pub(super) trait Interpreter: Sized {
    fn initialize() -> Result<Self, InterpreterError>;
}

impl<I: Interpreter> InterpretedBackend<I> {
    pub(super) fn initialize() -> Result<Self, InterpreterError> {
        Ok(Self {
            interpreter: I::initialize()?,
        })
    }

    pub(super) fn store<T: Scalar>(&mut self, data: &[T]) -> Result<(), InterpreterError> {
        todo!()
    }
}
