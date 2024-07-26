use crate::runtime::TensorId;
use crate::scalar::Scalar;
use alloc::collections::{BTreeMap, BTreeSet};
use alloc::vec::Vec;

use super::graph::Graph;

pub(super) mod cpu;

pub(super) struct InterpretedBackend<I: Interpreter> {
    interpreter: I,
    buffers: BTreeMap<TensorId, I::Buffer>,
}

#[derive(Debug)]
pub enum InterpreterError {
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
    fn deallocate_memory(&mut self, buffer: Self::Buffer) -> Result<(), InterpreterError>;
}

impl<I: Interpreter> InterpretedBackend<I> {
    pub(super) fn initialize() -> Result<Self, InterpreterError> {
        return Ok(Self {
            interpreter: I::initialize()?,
            buffers: BTreeMap::new(),
        });
    }

    pub(super) fn is_realized(&self, x: TensorId) -> bool {
        return self.buffers.contains_key(&x);
    }

    pub(super) fn store<T: Scalar>(
        &mut self,
        x: TensorId,
        data: Vec<T>,
    ) -> Result<(), InterpreterError> {
        let _ = x;
        let _ = data;
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

    pub(super) fn remove(&mut self, x: TensorId) -> Result<(), InterpreterError> {
        if let Some(buffer) = self.buffers.remove(&x) {
            return self.interpreter.deallocate_memory(buffer);
        }
        return Ok(());
    }

    pub(super) fn interpret_graph(
        &mut self,
        graph: &Graph,
        to_eval: &BTreeSet<TensorId>,
    ) -> Result<(), InterpreterError> {
        todo!()
    }
}
