use crate::runtime::TensorId;
use crate::scalar::Scalar;
use alloc::collections::{BTreeMap, BTreeSet};
use alloc::vec::Vec;

use super::graph::Graph;

pub(super) mod cpu;

pub(super) struct InterpretedBackend<I: Custom> {
    interpreter: I,
    buffers: BTreeMap<TensorId, I::Buffer>,
}

pub(super) trait Custom: Sized {
    type Buffer;
    type Error: core::fmt::Debug;
    fn initialize() -> Result<Self, Self::Error>;
    fn store_mem<T: Scalar>(&self, data: Vec<T>) -> Result<Self::Buffer, Self::Error>;
    fn load_mem<T: Scalar>(
        &self,
        buffer: &Self::Buffer,
        length: usize,
    ) -> Result<Vec<T>, Self::Error>;
    fn deallocate_memory(&mut self, buffer: Self::Buffer) -> Result<(), Self::Error>;
    /*fn unary(&self, x: &Self::Buffer, x_view: &View, uop: UOp) -> Self::Buffer;
    fn binary(&self, x: &Self::Buffer, y: &Self::Buffer, bop: BOp) -> Self::Buffer;
    fn where_(&self, x: &Self::Buffer, y: &Self::Buffer, z: &Self::Buffer) -> Self::Buffer;
    fn dot(
        &self,
        x: &Self::Buffer,
        y: &Self::Buffer,
        transpose_x: bool,
        transpose_y: bool,
        ) -> Self::Buffer;*/
}

impl<C: Custom> InterpretedBackend<C> {
    pub(super) fn initialize() -> Result<Self, C::Error> {
        return Ok(Self {
            interpreter: C::initialize()?,
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
    ) -> Result<(), C::Error> {
        self.buffers.insert(x, self.interpreter.store_mem(data)?);
        Ok(())
    }

    // Load values at x, if x is not evaluated, it will return error
    pub(super) fn load<T: Scalar>(
        &self,
        x: TensorId,
        length: usize,
    ) -> Result<Vec<T>, C::Error> {
        if let Some(buffer) = self.buffers.get(&x) {
            return self.interpreter.load_mem(buffer, length);
        } else {
            panic!("Custom buffer does not exists. Internal bug.");
        }
    }

    pub(super) fn remove(&mut self, x: TensorId) -> Result<(), C::Error> {
        if let Some(buffer) = self.buffers.remove(&x) {
            return self.interpreter.deallocate_memory(buffer);
        }
        return Ok(());
    }

    pub(super) fn interpret_graph(
        &mut self,
        mut graph: Graph,
        to_eval: &BTreeSet<TensorId>,
    ) -> Result<(), C::Error> {
        let order = graph.execution_order(to_eval);
        let _ = order;
        todo!()
    }
}
