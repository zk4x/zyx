extern crate alloc;
use crate::{graph::Node, node_id::NodeId, OutOfMemoryError, dtype::DType};
use alloc::{
    vec::Vec,
    boxed::Box,
    collections::{BTreeMap, BTreeSet},
};
use tch::Tensor;
use super::Storage;

trait GetConst {
    fn c(&self, i: NodeId) -> &Storage;
}

impl GetConst for BTreeMap<NodeId, (usize, Node)> {
    fn c(&self, i: NodeId) -> &Storage {
        if let Node::Const(storage) = &self.get(&i).unwrap().1 {
            storage
        } else {
            panic!()
        }
    }
}

#[derive(Debug)]
pub(crate) struct TorchDev;

impl TorchDev {
    pub(crate) fn new() -> Self {
        Self
    }

    #[allow(clippy::unused_self)]
    pub(super) fn load_f32(&self, storage: &Tensor) -> Box<[f32]> {
        let v: alloc::vec::Vec<f32> = storage.data().flatten(0, -1).try_into().unwrap();
        v.into()
    }

    #[allow(clippy::unused_self)]
    pub(super) fn load_i32(&self, storage: &Tensor) -> Box<[i32]> {
        let v: alloc::vec::Vec<i32> = storage.data().flatten(0, -1).try_into().unwrap();
        v.into()
    }

    #[allow(clippy::unnecessary_wraps)]
    pub(super) fn realize(
        &mut self,
        graph: &mut BTreeMap<NodeId, (usize, Node)>, // id, refcount and Node
        order: &[NodeId],                            // recommended realization order
        _nodes: &BTreeSet<NodeId>,                    // which nodes need to be realized
    ) -> Result<(), OutOfMemoryError> {
        'a: for node_id in order {
            let node = &graph.get(node_id).unwrap().1;
            match node {
                Node::None | Node::Leaf | Node::Const(..) => continue 'a,
                _ => {}
            }
            let res = match node {
                Node::None | Node::Leaf | Node::Const(..) => panic!(),
                Node::StoreF32(data, shape) => Storage::TorchF32(<Vec<f32> as TryInto<Tensor>>::try_into(data.to_vec()).unwrap().reshape(shape.vi64())),
                Node::StoreI32(data, shape) => Storage::TorchI32(<Vec<i32> as TryInto<Tensor>>::try_into(data.to_vec()).unwrap().reshape(shape.vi64())),
                Node::Add(x, y) => binary_op(graph.c(*x), graph.c(*y), "+"),
                Node::Sub(x, y) => binary_op(graph.c(*x), graph.c(*y), "-"),
                Node::Mul(x, y) => binary_op(graph.c(*x), graph.c(*y), "*"),
                Node::Div(x, y) => binary_op(graph.c(*x), graph.c(*y), "/"),
                Node::Pow(x, y) => binary_op(graph.c(*x), graph.c(*y), "pow"),
                Node::Cmplt(x, y) => binary_op(graph.c(*x), graph.c(*y), "<"),
                Node::TDot(x, y, _) => binary_op(graph.c(*x), graph.c(*y), "tdot"),
                Node::Expand(x, eshape) => match graph.c(*x) {
                    Storage::TorchF32(data) => Storage::TorchF32(data.expand(eshape.vi64(), true)),
                    Storage::TorchI32(data) => Storage::TorchI32(data.expand(eshape.vi64(), true)),
                    #[cfg(any(feature = "opencl", feature = "torch"))]
                    _ => panic!(),
                },
                Node::Reshape(x, eshape) => match graph.c(*x) {
                    Storage::TorchF32(data) => Storage::TorchF32(data.reshape(eshape.vi64())),
                    Storage::TorchI32(data) => Storage::TorchI32(data.reshape(eshape.vi64())),
                    #[cfg(any(feature = "opencl", feature = "torch"))]
                    _ => panic!(),
                },
                Node::Permute(x, axes, _) => match graph.c(*x) {
                    Storage::TorchF32(data) => Storage::TorchF32(data.permute(axes.vi64())),
                    Storage::TorchI32(data) => Storage::TorchI32(data.permute(axes.vi64())),
                    #[cfg(any(feature = "opencl", feature = "torch"))]
                    _ => panic!(),
                },
                Node::Sum(x, axes, _) => match graph.c(*x) {
                    Storage::TorchF32(data) => Storage::TorchF32(data.sum_dim_intlist(axes.vi64(), true, None)),
                    Storage::TorchI32(data) => Storage::TorchI32(data.sum_dim_intlist(axes.vi64(), true, None)),
                    #[cfg(any(feature = "opencl", feature = "torch"))]
                    _ => panic!(),
                },
                Node::Max(x, axes, _) => match graph.c(*x) {
                    Storage::TorchF32(data) => Storage::TorchF32(data.amax(axes.vi64(), true)),
                    Storage::TorchI32(data) => Storage::TorchI32(data.amax(axes.vi64(), true)),
                    #[cfg(any(feature = "opencl", feature = "torch"))]
                    _ => panic!(),
                },
                Node::Neg(x) => unary_op(graph.c(*x), "neg"),
                Node::ReLU(x) => unary_op(graph.c(*x), "relu"),
                Node::Exp(x) => unary_op(graph.c(*x), "exp"),
                Node::Ln(x) => unary_op(graph.c(*x), "ln"),
                Node::Sin(x) => unary_op(graph.c(*x), "sin"),
                Node::Cos(x) => unary_op(graph.c(*x), "cos"),
                Node::Sqrt(x) => unary_op(graph.c(*x), "sqrt"),
                Node::Tanh(x) => unary_op(graph.c(*x), "tanh"),
                // Dropout does not use seed here
                Node::Dropout(x, _, prob) => match graph.c(*x) {
                    Storage::TorchF32(data) => Storage::TorchF32(data.dropout((*prob).into(), true)),
                    Storage::TorchI32(data) => Storage::TorchI32(data.dropout((*prob).into(), true)),
                    #[cfg(any(feature = "opencl", feature = "torch"))]
                    _ => panic!(),
                },
                Node::Cast(x, dtype) => match graph.c(*x) {
                    Storage::TorchF32(data) => match dtype {
                        DType::F32 => Storage::TorchF32(data.data()),
                        DType::I32 => Storage::TorchI32(data.to_kind(tch::kind::Kind::Int)),
                    },
                    Storage::TorchI32(data) => match dtype {
                        DType::F32 => Storage::TorchF32(data.to_kind(tch::kind::Kind::Float)),
                        DType::I32 => Storage::TorchI32(data.data()),
                    },
                    #[cfg(any(feature = "opencl", feature = "torch"))]
                    _ => todo!(),
                },
                //_ => todo!("Op {node:?}"),
            };
            let parameters = node.parameters();
            graph.get_mut(node_id).unwrap().1 = Node::Const(res);
            for parameter in &*parameters {
                let val = graph.get_mut(parameter).unwrap();
                val.0 -= 1;
                if val.0 == 0 {
                    val.1 = Node::None;
                }
            }
        }
        Ok(())
    }
}

fn unary_op(data: &Storage, op: &str) -> Storage {
    match data {
        Storage::TorchF32(data) => {
            Storage::TorchF32(match op {
                "neg" => -data,
                "relu" => data.relu(),
                "exp" => data.exp(),
                "ln" => data.log(),
                "sin" => data.sin(),
                "cos" => data.cos(),
                "sqrt" => data.sqrt(),
                "tanh" => data.tanh(),
                _ => panic!()
            })
        }
        Storage::TorchI32(data) => {
            Storage::TorchI32(match op {
                "neg" => -data,
                "relu" => data.relu(),
                _ => panic!()
            })
        }
        _ => panic!()
    }
}

fn binary_op(data_x: &Storage, data_y: &Storage, op: &str) -> Storage {
    match data_x {
        Storage::TorchF32(data_x) => {
            if let Storage::TorchF32(data_y) = data_y {
                match op {
                    "+" => Storage::TorchF32(data_x + data_y),
                    "-" => Storage::TorchF32(data_x - data_y),
                    "*" => Storage::TorchF32(data_x * data_y),
                    "/" => Storage::TorchF32(data_x / data_y),
                    "pow" => Storage::TorchF32(data_x.pow(data_y)),
                    "<" => Storage::TorchF32(data_x.lt_tensor(data_y).to_kind(tch::kind::Kind::Float)),
                    "tdot" => {
                        // k, m @ k, n -> m, n
                        Storage::TorchF32(data_x.transpose(-2, -1).matmul(data_y))
                    }
                    _ => panic!(),
                }
            } else {
                panic!()
            }
        }
        Storage::TorchI32(data_x) => {
            if let Storage::TorchI32(data_y) = data_y {
                match op {
                    "+" => Storage::TorchI32(data_x + data_y),
                    "-" => Storage::TorchI32(data_x - data_y),
                    "*" => Storage::TorchI32(data_x * data_y),
                    "/" => Storage::TorchI32(data_x / data_y),
                    "pow" => Storage::TorchI32(data_x.pow(data_y)),
                    "<" => Storage::TorchI32(data_x.lt_tensor(data_y).to_kind(tch::kind::Kind::Int)),
                    "tdot" => {
                        // k, m @ k, n -> m, n
                        Storage::TorchI32(data_x.transpose(-2, -1).matmul(data_y))
                    }
                    _ => panic!(),
                }
            } else {
                panic!()
            }
        }
        #[cfg(any(feature = "opencl", feature = "torch"))]
        _ => panic!(),
    }
}

