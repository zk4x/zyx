extern crate alloc;

use crate::dtype::DType;
use crate::node::Node;
use crate::shape::Shape;
use crate::tensor::Id;
use alloc::boxed::Box;
use alloc::collections::{BTreeMap, BTreeSet};
use alloc::vec::Vec;

pub trait Autograd {
    fn nodes(&self) -> &[Node];
    fn order(&self) -> &[Id];
    fn shape(&self, x: Id) -> &Shape;
    fn dtype(&self, x: Id) -> DType;
    fn push(&mut self, node: Node) -> Id;
    fn release(&mut self, x: Id);
    fn retain(&mut self, x: Id);

    /// Common autograd engine, currently used by all backends
    fn backward(&mut self, x: Id, sources: &BTreeSet<Id>) -> BTreeMap<Id, Id> {
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

        let topo = build_topo(x, sources, self.nodes(), self.order());
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
        let grad1 = match self.dtype(x) {
            DType::F32 => self.push(Node::IterF32(
                Box::new([1.].into_iter()),
                self.shape(x).clone(),
            )),
            DType::I32 => self.push(Node::IterF32(
                Box::new([1.].into_iter()),
                self.shape(x).clone(),
            )),
        };
        grads.insert(
            x,
            self.push(Node::Expand(grad1, self.shape(x).clone())),
        );
        self.release(grad1);
        // backpropagate
        // TODO this is not very clean code. Can we make it cleaner?
        for nid in topo {
            let grad = grads[&nid];
            match self.nodes()[nid.i()] {
                Node::LeafF32(..)
                | Node::LeafI32(..)
                | Node::UniformF32(..)
                | Node::UniformI32(..)
                | Node::IterF32(..)
                | Node::IterI32(..) => {}
                Node::Add(x, y) => {
                    if req_grad.contains(&x) && grads.insert(x, grad).is_none() {
                        self.retain(grad);
                    }
                    if req_grad.contains(&y) && grads.insert(y, grad).is_none() {
                        self.retain(grad);
                    }
                }
                Node::Sub(x, y) => {
                    if req_grad.contains(&x) && grads.insert(x, grad).is_none() {
                        self.retain(grad);
                    }
                    if req_grad.contains(&y) && !grads.contains_key(&y) {
                        grads.insert(y, self.push(Node::Neg(grad)));
                    }
                }
                Node::Mul(x, y) => {
                    if req_grad.contains(&x) && !grads.contains_key(&x) {
                        grads.insert(x, self.push(Node::Mul(y, grad)));
                    }
                    if req_grad.contains(&y) && !grads.contains_key(&y) {
                        grads.insert(y, self.push(Node::Mul(x, grad)));
                    }
                }
                Node::Div(x, y) => {
                    if req_grad.contains(&x) && !grads.contains_key(&x) {
                        grads.insert(x, self.push(Node::Div(grad, y)));
                    }
                    if req_grad.contains(&y) && !grads.contains_key(&y) {
                        // -grad*x/(y^2)
                        let two = match self.dtype(y) {
                            DType::F32 => {
                                self.push(Node::IterF32(Box::new([2.].into_iter()), 1.into()))
                            }
                            DType::I32 => {
                                self.push(Node::IterI32(Box::new([2].into_iter()), 1.into()))
                            }
                        };
                        let two_e = self.push(Node::Expand(two, self.shape(y).clone()));
                        self.release(two);
                        let two_2 = self.push(Node::Pow(y, two_e));
                        self.release(two_e);
                        let temp = self.push(Node::Mul(x, grad));
                        let temp_neg = self.push(Node::Neg(temp));
                        self.release(temp);
                        let y_grad = self.push(Node::Div(temp_neg, two_2));
                        self.release(temp_neg);
                        self.release(two_2);
                        grads.insert(y, y_grad);
                    }
                }
                Node::Pow(x, y) => {
                    if req_grad.contains(&x) && !grads.contains_key(&x) {
                        // grad * y * x.pow(y-1)
                        let one = match self.dtype(y) {
                            DType::F32 => {
                                self.push(Node::IterF32(Box::new([1.].into_iter()), 1.into()))
                            }
                            DType::I32 => {
                                self.push(Node::IterI32(Box::new([1].into_iter()), 1.into()))
                            }
                        };
                        let one1 = self.push(Node::Expand(one, self.shape(y).clone()));
                        self.release(one);
                        let y_1 = self.push(Node::Sub(y, one1));
                        self.release(one1);
                        let pow_y_1 = self.push(Node::Pow(x, y_1));
                        self.release(y_1);
                        let y_mul = self.push(Node::Mul(y, pow_y_1));
                        self.release(pow_y_1);
                        let x_grad = self.push(Node::Mul(grad, y_mul));
                        self.release(y_mul);
                        grads.insert(x, x_grad);
                    }
                    if req_grad.contains(&y) && !grads.contains_key(&y) {
                        // grad * x.pow(y) * ln(x)
                        let temp1 = self.push(Node::Ln(x));
                        let temp2 = self.push(Node::Mul(nid, temp1));
                        self.release(temp1);
                        let y_grad = self.push(Node::Mul(grad, temp2));
                        self.release(temp2);
                        grads.insert(y, y_grad);
                    }
                }
                Node::Cmplt(..) => {
                    panic!("Compare less than (operator <) is not differentiable operation.");
                }
                Node::ReLU(x) => {
                    // TODO is grads.contains_key useless for unary ops?
                    grads.entry(x).or_insert_with(|| {
                        let zero = match self.dtype(x) {
                            DType::F32 => {
                                self.push(Node::IterF32(Box::new([0.].into_iter()), 1.into()))
                            }
                            DType::I32 => {
                                self.push(Node::IterI32(Box::new([0].into_iter()), 1.into()))
                            }
                        };
                        let zeros = self.push(Node::Expand(zero, self.shape(x).clone()));
                        self.release(zero);
                        let zl = self.push(Node::Cmplt(zeros, x));
                        self.release(zeros);
                        let x_grad = self.push(Node::Mul(zl, grad));
                        self.release(zl);
                        x_grad
                    });
                }
                Node::Exp(x) => {
                    grads
                        .entry(x)
                        .or_insert_with(|| self.push(Node::Mul(nid, grad)));
                }
                Node::Ln(x) => {
                    grads
                        .entry(x)
                        .or_insert_with(|| self.push(Node::Div(grad, x)));
                }
                Node::Sin(x) => {
                    grads.entry(x).or_insert_with(|| {
                        let x_temp = self.push(Node::Cos(x));
                        let x_grad = self.push(Node::Mul(x_temp, grad));
                        self.release(x_temp);
                        x_grad
                    });
                }
                Node::Cos(x) => {
                    grads.entry(x).or_insert_with(|| {
                        let x_temp1 = self.push(Node::Sin(x));
                        let x_temp = self.push(Node::Neg(x_temp1));
                        self.release(x_temp1);
                        let x_grad = self.push(Node::Mul(x_temp, grad));
                        self.release(x_temp);
                        x_grad
                    });
                }
                Node::Sqrt(x) => {
                    grads.entry(x).or_insert_with(|| {
                        // x_grad = grad/(2*sqrt(x))
                        let x_shape = self.shape(x).clone();
                        let two1 = self.push(Node::IterF32(Box::new([2.].into_iter()), 1.into()));
                        let two2 = self.push(Node::Expand(two1, x_shape));
                        self.release(two1);
                        let x_temp = self.push(Node::Mul(two2, nid));
                        self.release(two2);
                        let x_grad = self.push(Node::Div(grad, x_temp));
                        self.release(x_temp);
                        x_grad
                    });
                }
                Node::CastF32(x) => {
                    grads.entry(x).or_insert_with(|| match self.dtype(x) {
                        DType::F32 => self.push(Node::CastF32(grad)),
                        DType::I32 => self.push(Node::CastI32(grad)),
                    });
                }
                Node::CastI32(x) => {
                    grads.entry(x).or_insert_with(|| match self.dtype(x) {
                        DType::F32 => self.push(Node::CastF32(grad)),
                        DType::I32 => self.push(Node::CastI32(grad)),
                    });
                }
                Node::Neg(x) => {
                    grads
                        .entry(x)
                        .or_insert_with(|| self.push(Node::Neg(grad)));
                }
                Node::Tanh(x) => {
                    grads.entry(x).or_insert_with(|| {
                        // 1 - tanh^2(x)
                        let shape = self.shape(x).clone();
                        let (two1, one1) = match self.dtype(x) {
                            DType::F32 => (
                                self.push(Node::IterF32(Box::new([2.].into_iter()), 1.into())),
                                self.push(Node::IterF32(Box::new([1.].into_iter()), 1.into())),
                            ),
                            DType::I32 => (
                                self.push(Node::IterI32(Box::new([2].into_iter()), 1.into())),
                                self.push(Node::IterI32(Box::new([1].into_iter()), 1.into())),
                            ),
                        };
                        let two2 = self.push(Node::Expand(two1, shape.clone()));
                        self.release(two1);
                        let two = self.push(Node::Pow(nid, two2));
                        self.release(two2);
                        let one2 = self.push(Node::Expand(one1, shape));
                        self.release(one1);
                        let one = self.push(Node::Sub(one2, two));
                        self.release(one2);
                        self.release(two);
                        let x_grad = self.push(Node::Mul(one, grad));
                        self.release(one);
                        x_grad
                    });
                }
                Node::Reshape(x, ..) => {
                    grads
                        .entry(x)
                        .or_insert_with(|| self.push(Node::Reshape(grad, self.shape(x).clone())));
                }
                Node::Expand(x, ref shape) => {
                    if !grads.contains_key(&x) {
                        let org_shape = self.shape(x).clone();
                        let axes = org_shape.expand_axes(shape);
                        let temp = self.push(Node::Sum(grad, axes, org_shape.clone()));
                        let x_grad = self.push(Node::Reshape(temp, org_shape));
                        self.release(temp);
                        grads.insert(x, x_grad);
                    }
                }
                Node::Permute(x, ref axes, _) => {
                    if !grads.contains_key(&x) {
                        let shape = self.shape(x);
                        grads.insert(
                            x,
                            self.push(Node::Permute(grads[&nid], axes.argsort(), shape.clone())),
                        );
                    }
                }
                Node::Sum(x, ..) => {
                    grads
                        .entry(x)
                        .or_insert_with(|| self.push(Node::Expand(grad, self.shape(x).clone())));
                }
                Node::Max(x, ..) => {
                    grads.entry(x).or_insert_with(|| {
                        // x_grad = (1 - (x < z.expand(x.shape()))) * grad
                        let x_shape = self.shape(x).clone();
                        let z_temp = self.push(Node::Expand(nid, x_shape.clone()));
                        let cmp_t = self.push(Node::Cmplt(x, z_temp));
                        self.release(z_temp);
                        let one1 = self.push(Node::IterF32(Box::new([1.].into_iter()), 1.into()));
                        let one2 = self.push(Node::Expand(one1, x_shape));
                        self.release(one1);
                        let max_1s = self.push(Node::Sub(one2, cmp_t));
                        self.release(one2);
                        self.release(cmp_t);
                        let x_grad = self.push(Node::Mul(max_1s, grad));
                        self.release(max_1s);
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
                    self.release(x.1);
                    None
                }
            })
            .collect()
    }
}
