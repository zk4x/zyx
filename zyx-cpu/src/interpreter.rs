use alloc::{
    vec::Vec,
    collections::{BTreeSet, BTreeMap},
    boxed::Box,
};
use zyx_core::{
    view::View,
    node::Node,
    scalar::Scalar,
    error::ZyxError,
    tensor::Id,
    runtime::RuntimeBackend,
};

enum Data {
    F32(Box<[f32]>),
    I32(Box<[i32]>),
}

pub struct Interpreter {
    data: BTreeMap<Id, Data>,
    views: BTreeMap<Id, Data>,
}

impl Interpreter {
    pub(crate) fn new() -> Self {
        Self {
            data: Vec::new(),
            views: Vec::new(),
        }
    }
}

impl RuntimeBackend for Interpreter {
    fn is_evaluated(&self, x: Id) -> bool {
        self.data.contains_key(&x)
    }

    fn remove(&mut self, x: Id) -> Result<(), ZyxError> {
        self.data.remove(x);
        self.views.remove(x);
        Ok(())
    }

    fn load<T: Scalar>(&mut self, x: Id, numel: usize) -> Result<Vec<T>, ZyxError> {
        todo!()
    }

    fn evaluate(&mut self, to_eval: BTreeSet<Id>, order: &[Id], nodes: &mut [Node]) -> Result<(), ZyxError> {
        todo!()
    }
}
