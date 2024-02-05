use alloc::{boxed::Box, collections::BTreeMap, vec::Vec};
use zyx_core::{
    error::ZyxError, node::Node, runtime::RuntimeBackend, scalar::Scalar, tensor::Id, view::View,
};

enum Data {
    F32(Box<[f32]>),
    I32(Box<[i32]>),
}

impl Data {
    unsafe fn as_type<T: Scalar>(&self) -> &[T] {
        match self {
            Data::F32(data) => core::mem::transmute(data.as_ref()),
            Data::I32(data) => core::mem::transmute(data.as_ref()),
        }
    }
}

pub struct Interpreter {
    buffers: BTreeMap<Id, Data>,
    views: BTreeMap<Id, View>,
}

impl Interpreter {
    pub(crate) fn new() -> Self {
        Self {
            buffers: BTreeMap::new(),
            views: BTreeMap::new(),
        }
    }
}

impl RuntimeBackend for Interpreter {
    fn is_evaluated(&self, x: Id) -> bool {
        self.buffers.contains_key(&x)
    }

    fn remove(&mut self, x: Id) -> Result<(), ZyxError> {
        self.buffers.remove(&x);
        self.views.remove(&x);
        Ok(())
    }

    fn load<T: Scalar>(&mut self, x: Id, numel: usize) -> Result<Vec<T>, ZyxError> {
        let view = &self.views[&x];
        let data = unsafe { self.buffers[&x].as_type::<T>() };
        Ok((0..numel).map(|i| data[view.get_idx(i)].clone()).collect())
    }

    fn evaluate(
        &mut self,
        mut rcs: BTreeMap<Id, u8>,
        order: &[Id],
        nodes: &mut [Node],
    ) -> Result<(), ZyxError> {
        for nid in order.iter().copied() {
            match &mut nodes[nid.i()] {
                Node::LeafF32(..)
                | Node::LeafI32(..)
                | Node::UniformF32(..)
                | Node::CastF32(..)
                | Node::CastI32(..)
                | Node::Neg(..)
                | Node::ReLU(..)
                | Node::Sin(..)
                | Node::Cos(..)
                | Node::Ln(..)
                | Node::Exp(..)
                | Node::Tanh(..)
                | Node::Sqrt(..)
                | Node::Add(..)
                | Node::Sub(..)
                | Node::Mul(..)
                | Node::Div(..)
                | Node::Pow(..)
                | Node::Cmplt(..)
                | Node::Reshape(..)
                | Node::Expand(..)
                | Node::Permute(..)
                | Node::Pad(..)
                | Node::Sum(..)
                | Node::Max(..) => {
                    todo!()
                }
                Node::IterF32(_, shape) => {
                    let mut new_node = Node::LeafF32(shape.clone());
                    core::mem::swap(&mut nodes[nid.i()], &mut new_node);
                    if let Node::IterF32(iter, _) = new_node {
                        self.buffers.insert(nid, Data::F32(iter.collect()));
                    }
                }
                Node::IterI32(_, shape) => {
                    let mut new_node = Node::LeafI32(shape.clone());
                    core::mem::swap(&mut nodes[nid.i()], &mut new_node);
                    if let Node::IterI32(iter, _) = new_node {
                        self.buffers.insert(nid, Data::I32(iter.collect()));
                    }
                }
            }
            for p in nodes[nid.i()].parameters() {
                rcs.entry(p).and_modify(|rc| *rc -= 1);
                self.remove(p)?;
            }
        }
        Ok(())
    }
}
