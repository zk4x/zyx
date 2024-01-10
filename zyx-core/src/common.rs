extern crate alloc;

use alloc::boxed::Box;
use crate::node::Node;
use crate::tensor::Id;
use alloc::collections::{BTreeMap, BTreeSet};
use alloc::vec::Vec;
use core::ffi::c_void;
use crate::dtype::DType;
use crate::shape::Shape;

pub trait Autograd {
    fn nodes(&self) -> &[Node];
    fn order(&self) -> &[Id];
    fn shape(&self, x: Id) -> &Shape;
    fn dtype(&self, x: Id) -> DType;
    fn push(&mut self, node: Node) -> Id;
    fn release(&mut self, x: Id);
    fn retain(&mut self, x: Id);
}

/// Common autograd engine, currently used by all backends
pub fn backward(backend: &mut impl Autograd, x: Id, sources: &BTreeSet<Id>) -> BTreeMap<Id, Id> {
    fn build_topo(x: Id, sources: &BTreeSet<Id>, nodes: &[Node], order: &[Id]) -> Vec<Id> {
        // First we need to know which nodes require gradient
        let mut req_grad = BTreeSet::new();
        for i in order {
            for p in nodes[i.i()].parameters() {
                if sources.contains(&p) || req_grad.contains(&p) {
                    req_grad.insert(i);
                }
            }
        }
        // Here we build topo
        let mut topo = Vec::new();
        if !req_grad.contains(&x) {
            return topo;
        }
        let mut visited = BTreeSet::new();
        let mut params = alloc::collections::VecDeque::with_capacity(32);
        params.push_back(x);
        while let Some(p) = params.pop_front() {
            if req_grad.contains(&p) {
                topo.push(p);
                if visited.insert(p) {
                    params.extend(nodes[p.i()].parameters());
                }
            }
        }
        topo
    }

    let topo = build_topo(x, sources, backend.nodes(), backend.order());
    let req_grad: BTreeSet<Id> = topo
        .iter()
        .copied()
        .chain(sources.iter().copied())
        .collect();
    //extern crate std;
    //std::println!("These nodes require gradient: {:?}", req_grad);
    // Node -> Grad
    let mut grads: BTreeMap<Id, Id> = BTreeMap::new();
    // Initial gradient of ones
    let grad1 = match backend.dtype(x) {
        DType::F32 => backend.push(Node::IterF32(
            Box::new([1.].into_iter()),
            backend.shape(x).clone(),
        )),
        DType::I32 => backend.push(Node::IterF32(
            Box::new([1.].into_iter()),
            backend.shape(x).clone(),
        )),
    };
    grads.insert(x, backend.push(Node::Expand(grad1, backend.shape(x).clone())));
    backend.release(grad1);
    // backpropagate
    // TODO this is not very clean code. Can we make it cleaner?
    for nid in topo {
        let grad = grads[&nid];
        match backend.nodes()[nid.i()] {
            Node::LeafF32(..)
            | Node::LeafI32(..)
            | Node::UniformF32(..)
            | Node::UniformI32(..)
            | Node::IterF32(..)
            | Node::IterI32(..) => {}
            Node::Add(x, y) => {
                if req_grad.contains(&x) && grads.insert(x, grad).is_none() {
                    backend.retain(grad);
                }
                if req_grad.contains(&y) && grads.insert(y, grad).is_none() {
                    backend.retain(grad);
                }
            }
            Node::Sub(x, y) => {
                if req_grad.contains(&x) && grads.insert(x, grad).is_none() {
                    backend.retain(grad);
                }
                if req_grad.contains(&y) && !grads.contains_key(&y) {
                    grads.insert(y, backend.push(Node::Neg(grad)));
                }
            }
            Node::Mul(x, y) => {
                if req_grad.contains(&x) && !grads.contains_key(&x) {
                    grads.insert(x, backend.push(Node::Mul(y, grad)));
                }
                if req_grad.contains(&y) && !grads.contains_key(&y) {
                    grads.insert(y, backend.push(Node::Mul(x, grad)));
                }
            }
            Node::Div(x, y) => {
                if req_grad.contains(&x) && !grads.contains_key(&x) {
                    grads.insert(x, backend.push(Node::Div(grad, y)));
                }
                if req_grad.contains(&y) && !grads.contains_key(&y) {
                    // -grad*x/(y^2)
                    let two = match backend.dtype(y) {
                        DType::F32 => {
                            backend.push(Node::IterF32(Box::new([2.].into_iter()), 1.into()))
                        }
                        DType::I32 => backend.push(Node::IterI32(Box::new([2].into_iter()), 1.into())),
                    };
                    let two_e = backend.push(Node::Expand(two, backend.shape(y).clone()));
                    backend.release(two);
                    let two_2 = backend.push(Node::Pow(y, two_e));
                    backend.release(two_e);
                    let temp = backend.push(Node::Mul(x, grad));
                    let temp_neg = backend.push(Node::Neg(temp));
                    backend.release(temp);
                    let y_grad = backend.push(Node::Div(temp_neg, two_2));
                    backend.release(temp_neg);
                    backend.release(two_2);
                    grads.insert(y, y_grad);
                }
            }
            Node::Pow(x, y) => {
                if req_grad.contains(&x) && !grads.contains_key(&x) {
                    // grad * y * x.pow(y-1)
                    let one = match backend.dtype(y) {
                        DType::F32 => {
                            backend.push(Node::IterF32(Box::new([1.].into_iter()), 1.into()))
                        }
                        DType::I32 => backend.push(Node::IterI32(Box::new([1].into_iter()), 1.into())),
                    };
                    let one1 = backend.push(Node::Expand(one, backend.shape(y).clone()));
                    backend.release(one);
                    let y_1 = backend.push(Node::Sub(y, one1));
                    backend.release(one1);
                    let pow_y_1 = backend.push(Node::Pow(x, y_1));
                    backend.release(y_1);
                    let y_mul = backend.push(Node::Mul(y, pow_y_1));
                    backend.release(pow_y_1);
                    let x_grad = backend.push(Node::Mul(grad, y_mul));
                    backend.release(y_mul);
                    grads.insert(x, x_grad);
                }
                if req_grad.contains(&y) && !grads.contains_key(&y) {
                    // grad * x.pow(y) * ln(x)
                    let temp1 = backend.push(Node::Ln(x));
                    let temp2 = backend.push(Node::Mul(nid, temp1));
                    backend.release(temp1);
                    let y_grad = backend.push(Node::Mul(grad, temp2));
                    backend.release(temp2);
                    grads.insert(y, y_grad);
                }
            }
            Node::Cmplt(..) => {
                panic!("Compare less than (operator <) is not differentiable operation.");
            }
            Node::ReLU(x) => {
                // TODO is grads.contains_key useless for unary ops?
                grads.entry(x).or_insert_with(|| {
                    let zero = match backend.dtype(x) {
                        DType::F32 => {
                            backend.push(Node::IterF32(Box::new([0.].into_iter()), 1.into()))
                        }
                        DType::I32 => backend.push(Node::IterI32(Box::new([0].into_iter()), 1.into())),
                    };
                    let zeros = backend.push(Node::Expand(zero, backend.shape(x).clone()));
                    backend.release(zero);
                    let zl = backend.push(Node::Cmplt(zeros, x));
                    backend.release(zeros);
                    let x_grad = backend.push(Node::Mul(zl, grad));
                    backend.release(zl);
                    x_grad
                });
            }
            Node::Exp(x) => {
                grads
                    .entry(x)
                    .or_insert_with(|| backend.push(Node::Mul(nid, grad)));
            }
            Node::Ln(x) => {
                grads
                    .entry(x)
                    .or_insert_with(|| backend.push(Node::Div(grad, x)));
            }
            Node::Sin(x) => {
                grads.entry(x).or_insert_with(|| {
                    let x_temp = backend.push(Node::Cos(x));
                    let x_grad = backend.push(Node::Mul(x_temp, grad));
                    backend.release(x_temp);
                    x_grad
                });
            }
            Node::Cos(x) => {
                grads.entry(x).or_insert_with(|| {
                    let x_temp1 = backend.push(Node::Sin(x));
                    let x_temp = backend.push(Node::Neg(x_temp1));
                    backend.release(x_temp1);
                    let x_grad = backend.push(Node::Mul(x_temp, grad));
                    backend.release(x_temp);
                    x_grad
                });
            }
            Node::Sqrt(x) => {
                grads.entry(x).or_insert_with(|| {
                    // x_grad = grad/(2*sqrt(x))
                    let x_shape = backend.shape(x).clone();
                    let two1 = backend.push(Node::IterF32(Box::new([2.].into_iter()), 1.into()));
                    let two2 = backend.push(Node::Expand(two1, x_shape));
                    backend.release(two1);
                    let x_temp = backend.push(Node::Mul(two2, nid));
                    backend.release(two2);
                    let x_grad = backend.push(Node::Div(grad, x_temp));
                    backend.release(x_temp);
                    x_grad
                });
            }
            Node::CastF32(x) => {
                grads.entry(x).or_insert_with(|| match backend.dtype(x) {
                    DType::F32 => backend.push(Node::CastF32(grad)),
                    DType::I32 => backend.push(Node::CastI32(grad)),
                });
            }
            Node::CastI32(x) => {
                grads.entry(x).or_insert_with(|| match backend.dtype(x) {
                    DType::F32 => backend.push(Node::CastF32(grad)),
                    DType::I32 => backend.push(Node::CastI32(grad)),
                });
            }
            Node::Neg(x) => {
                grads.entry(x).or_insert_with(|| backend.push(Node::Neg(grad)));
            }
            Node::Tanh(x) => {
                grads.entry(x).or_insert_with(|| {
                    // 1 - tanh^2(x)
                    let shape = backend.shape(x).clone();
                    let (two1, one1) = match backend.dtype(x) {
                        DType::F32 => (
                            backend.push(Node::IterF32(Box::new([2.].into_iter()), 1.into())),
                            backend.push(Node::IterF32(Box::new([1.].into_iter()), 1.into())),
                        ),
                        DType::I32 => (
                            backend.push(Node::IterI32(Box::new([2].into_iter()), 1.into())),
                            backend.push(Node::IterI32(Box::new([1].into_iter()), 1.into())),
                        ),
                    };
                    let two2 = backend.push(Node::Expand(two1, shape.clone()));
                    backend.release(two1);
                    let two = backend.push(Node::Pow(nid, two2));
                    backend.release(two2);
                    let one2 = backend.push(Node::Expand(one1, shape));
                    backend.release(one1);
                    let one = backend.push(Node::Sub(one2, two));
                    backend.release(one2);
                    backend.release(two);
                    let x_grad = backend.push(Node::Mul(one, grad));
                    backend.release(one);
                    x_grad
                });
            }
            Node::Reshape(x, ..) => {
                grads
                    .entry(x)
                    .or_insert_with(|| backend.push(Node::Reshape(grad, backend.shape(x).clone())));
            }
            Node::Expand(x, ref shape) => {
                if !grads.contains_key(&x) {
                    let org_shape = backend.shape(x).clone();
                    let axes = org_shape.expand_axes(shape);
                    let temp = backend.push(Node::Sum(grad, axes, org_shape.clone()));
                    let x_grad = backend.push(Node::Reshape(temp, org_shape));
                    backend.release(temp);
                    grads.insert(x, x_grad);
                }
            }
            Node::Permute(x, ref axes, _) => {
                if !grads.contains_key(&x) {
                    let shape = backend.shape(x);
                    grads.insert(
                        x,
                        backend.push(Node::Permute(grads[&nid], axes.argsort(), shape.clone())),
                    );
                }
            }
            Node::Sum(x, ..) => {
                grads
                    .entry(x)
                    .or_insert_with(|| backend.push(Node::Expand(grad, backend.shape(x).clone())));
            }
            Node::Max(x, ..) => {
                grads.entry(x).or_insert_with(|| {
                    // x_grad = (1 - (x < z.expand(x.shape()))) * grad
                    let x_shape = backend.shape(x).clone();
                    let z_temp = backend.push(Node::Expand(nid, x_shape.clone()));
                    let cmp_t = backend.push(Node::Cmplt(x, z_temp));
                    backend.release(z_temp);
                    let one1 = backend.push(Node::IterF32(Box::new([1.].into_iter()), 1.into()));
                    let one2 = backend.push(Node::Expand(one1, x_shape));
                    backend.release(one1);
                    let max_1s = backend.push(Node::Sub(one2, cmp_t));
                    backend.release(one2);
                    backend.release(cmp_t);
                    let x_grad = backend.push(Node::Mul(max_1s, grad));
                    backend.release(max_1s);
                    x_grad
                });
            }
        }
    }
    grads
        .into_iter()
        .flat_map(|x| {
            if sources.contains(&x.0) {
                Some(x)
            } else {
                backend.release(x.1);
                None
            }
        })
        .collect()
}
