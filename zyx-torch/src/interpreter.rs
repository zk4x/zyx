use alloc::{boxed::Box, collections::{BTreeSet, BTreeMap, btree_map::Entry}, vec::Vec};
use zyx_core::{
    error::ZyxError, node::Node, runtime::RuntimeBackend, scalar::Scalar, tensor::Id, view::View,
};
use zyx_core::dtype::DType;
use zyx_core::view::ViewType;

enum Data {
    F32(Box<[f32]>),
    F64(Box<[f64]>),
    I32(Box<[i32]>),
}

impl Data {
    fn as_type<T: Scalar>(&self) -> &[T] {
        /*match self {
            Data::F32(data) => core::mem::transmute(data.as_ref()),
            Data::F64(data) => core::mem::transmute(data.as_ref()),
            Data::I32(data) => core::mem::transmute(data.as_ref()),
        }*/
        todo!()
    }
}

pub struct Interpreter {
    buffers: BTreeMap<Id, Data>,
    views: BTreeMap<Id, (View, Id)>,
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
        self.views.remove(&x);
        // TODO only remove buffers if no view points to it anymore
        self.buffers.remove(&x);
        Ok(())
    }

    fn load<T: Scalar>(&mut self, x: Id, numel: usize) -> Result<Vec<T>, ZyxError> {
        todo!()
    }

    fn evaluate(
        &mut self,
        _to_eval: BTreeSet<Id>,
        mut rcs: BTreeMap<Id, u8>,
        order: &[Id],
        nodes: &mut [Node],
    ) -> Result<(), ZyxError> {
        for nid in order.iter().copied() {
            match &mut nodes[nid.i()] {
                Node::Leaf(..)
                | Node::Uniform(..)
                | Node::Cast(..)
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
                | Node::Where(..)
                | Node::Reshape(..)
                | Node::Expand(..)
                | Node::Permute(..)
                | Node::Pad(..)
                | Node::Sum(..)
                | Node::Max(..) => {
                    todo!()
                }
                Node::IterF32(_, shape) => {
                    let mut new_node = Node::Leaf(shape.clone(), DType::F32);
                    core::mem::swap(&mut nodes[nid.i()], &mut new_node);
                    if let Node::IterF32(iter, _) = new_node {
                        self.buffers.insert(nid, Data::F32(iter.collect()));
                    }
                }
                Node::IterF64(_, shape) => {
                    let mut new_node = Node::Leaf(shape.clone(), DType::F64);
                    core::mem::swap(&mut nodes[nid.i()], &mut new_node);
                    if let Node::IterF64(iter, _) = new_node {
                        self.buffers.insert(nid, Data::F64(iter.collect()));
                    }
                }
                Node::IterI32(_, shape) => {
                    let mut new_node = Node::Leaf(shape.clone(), DType::I32);
                    core::mem::swap(&mut nodes[nid.i()], &mut new_node);
                    if let Node::IterI32(iter, _) = new_node {
                        self.buffers.insert(nid, Data::I32(iter.collect()));
                    }
                }
            }
            for p in nodes[nid.i()].parameters() {
                if let Entry::Occupied(e) = rcs.entry(p).and_modify(|rc| *rc -= 1) {
                    if *e.get() == 0 {
                        self.remove(p)?;
                    }
                }
            }
        }
        Ok(())
    }
}
