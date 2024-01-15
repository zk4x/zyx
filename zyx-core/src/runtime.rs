use crate::{
    axes::Axes,
    dtype::DType,
    node::Node,
    scalar::Scalar,
    shape::Shape,
    tensor::{id, Id},
};
use alloc::{
    boxed::Box,
    collections::{BTreeMap, BTreeSet},
    vec::Vec,
};
use crate::utils::{dtype, shape};

pub(crate) trait RuntimeBackend {
    fn is_evaluated(&self, x: Id) -> bool;
    fn remove(&mut self, x: Id);
    /// Load evaluated x
    fn load<T: Scalar>(&mut self, x: Id, numel: usize) -> Vec<T>;
    fn evaluate(&mut self, to_eval: BTreeSet<Id>, order: &[Id], nodes: &mut [Node]);
}

pub struct Runtime<R: RuntimeBackend> {
    rng: rand::rngs::SmallRng,
    rcs: Vec<u8>,
    order: Vec<Id>,
    nodes: Vec<Node>,
    leafs: BTreeSet<Id>, // these do not need backward graph
    runtime_backend: R,
}

impl<R: RuntimeBackend> Runtime<R> {
    pub fn new(runtime_backend: R) -> Self {
        use rand::SeedableRng;
        Self {
            rng: rand::rngs::SmallRng::seed_from_u64(420_694_206_942_069),
            rcs: Vec::new(),
            order: Vec::new(),
            nodes: Vec::new(),
            leafs: BTreeSet::new(),
            runtime_backend,
        }
    }

    pub fn randn(&mut self, shape: Shape, dtype: DType) -> Id {
        let shape: Shape = shape.into();
        use rand::Rng;
        let n = shape.numel();
        let mut rng = self.rng.clone();
        use rand::distributions::Standard;
        let data = match dtype {
            DType::F32 => self.push(Node::IterF32(
                Box::new((0..n).map(move |_| rng.sample(Standard))),
                shape,
            )),
            DType::I32 => self.push(Node::IterI32(
                Box::new((0..n).map(move |_| rng.sample(Standard))),
                shape,
            )),
        };
        // change the state of the random seed in rng
        for _ in 0..n {
            self.rng.sample::<f32, _>(Standard);
        }
        data
    }

    pub fn uniform<T: Scalar>(&mut self, shape: Shape, low: T, high: T) -> Id {
        match T::dtype() {
            DType::F32 => self.push(Node::UniformF32(shape, low.into_f32(), high.into_f32())),
            DType::I32 => self.push(Node::UniformI32(shape, low.into_i32(), high.into_i32())),
        }
    }

    pub fn full<T: Scalar>(&mut self, shape: Shape, value: T) -> Id {
        match T::dtype() {
            DType::F32 => self.push(Node::IterF32(
                Box::new(core::iter::repeat(value.into_f32()).take(shape.numel())),
                shape,
            )),
            DType::I32 => self.push(Node::IterI32(
                Box::new(core::iter::repeat(value.into_i32()).take(shape.numel())),
                shape,
            )),
        }
    }

    pub fn eye(&mut self, n: usize, dtype: DType) -> Id {
        match dtype {
            DType::F32 => self.push(Node::IterF32(
                Box::new(
                    (0..n).flat_map(move |i| (0..n).map(move |j| if j == i { 1. } else { 0. })),
                ),
                [n, n].into(),
            )),
            DType::I32 => self.push(Node::IterI32(
                Box::new((0..n).flat_map(move |i| (0..n).map(move |j| if j == i { 1 } else { 0 }))),
                [n, n].into(),
            )),
        }
    }

    pub fn shape(&self, x: Id) -> &Shape {
        shape(&self.nodes, x)
    }

    pub fn dtype(&self, x: Id) -> DType {
        dtype(&self.nodes, x)
    }

    pub fn load<T: Scalar>(&mut self, x: Id) -> Vec<T> {
        // This may need to evaluate, therefore we need to take mutable reference to self
        if !self.runtime_backend.is_evaluated(x) {
            // TODO also check if these are only movements ops,
            // in which case we can directly return iterator with view
            self.evaluate(BTreeSet::from([x]));
        }
        self.runtime_backend.load(x, shape(&self.nodes, x).numel())
    }

    pub fn set_leaf(&mut self, x: Id) {
        self.leafs.insert(x);
    }

    pub fn push(&mut self, node: Node) -> Id {
        for nid in node.parameters() {
            self.rcs[nid.i()] += 1;
        }
        let (i, new_node) = if let Some(i) = self.rcs.iter().position(|rc| *rc == 0) {
            (id(i), false)
        } else {
            (id(self.rcs.len()), true)
        };
        if new_node {
            self.rcs.push(1);
            self.nodes.push(node);
            self.order.push(i);
        } else {
            self.rcs[i.i()] = 1;
            self.nodes[i.i()] = node;
            // Keep the ordering, this is probably as fast as it gets
            // (i. e. better track the ordering here then reconstruct
            // the whole tree during evaluation)
            let prev = self.order[i.i()];
            for x in self.order.iter_mut() {
                if *x > prev {
                    *x -= 1;
                }
            }
            self.order[i.i()] = id(self.order.len() - 1);
        }
        i
    }

    pub fn release(&mut self, x: Id) {
        let mut params = Vec::with_capacity(10);
        params.push(x);
        while let Some(p) = params.pop() {
            self.rcs[p.i()] -= 1;
            if self.rcs[p.i()] == 0 {
                params.extend(self.nodes[p.i()].parameters());
                self.leafs.remove(&p);
                self.runtime_backend.remove(p);
            }
        }
    }

    pub fn retain(&mut self, x: Id) {
        self.rcs[x.i()] += 1;
    }

    pub fn evaluate(&mut self, nodes: BTreeSet<Id>) {
        // TODO we are probably going too many times back and forth in the graph.
        // First we go back to create graph of all nodes that need to be evaluated.
        // Then we go forward to find which nodes are kernel subgraphs.
        // Then we go back again to create subgraphs for individual kernels.
        // Then we go forward again to create the kernel itself.

        // Find all needed parameters for calculation of nodes
        let mut params: Vec<Id> = nodes.iter().copied().collect();
        let mut rcs: BTreeMap<Id, u8> = BTreeMap::new();
        while let Some(nid) = params.pop() {
            rcs.entry(nid).and_modify(|rc| *rc += 1).or_insert_with(|| {
                params.extend(self.nodes[nid.i()].parameters());
                1
            });
        }
        let mut order: Vec<Id> = rcs.keys().copied().collect();
        order.sort_by_cached_key(|nid| self.order[nid.i()]);

        self.runtime_backend.evaluate(nodes, &order, &mut self.nodes);

        // Release parts of graph that are not needed for backpropagation
        while let Some(leaf) = self.leafs.pop_last() {
            //std::println!("Releasing leaf {leaf}");
            let shape = shape(&self.nodes, leaf).clone();
            let mut node = match dtype(&self.nodes, leaf) {
                DType::F32 => Node::LeafF32(shape),
                DType::I32 => Node::LeafI32(shape),
            };
            core::mem::swap(self.nodes.get_mut(leaf.i()).unwrap(), &mut node);
            for nid in node.parameters() {
                self.release(nid);
            }
        }
    }

    /// Common autograd engine, currently used by all backends
    pub fn backward(&mut self, x: Id, sources: &BTreeSet<Id>) -> BTreeMap<Id, Id> {
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

        let topo = build_topo(x, sources, &self.nodes, &self.order);
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
        let grad1 = match dtype(&self.nodes, x) {
            DType::F32 => self.push(Node::IterF32(
                Box::new([1.].into_iter()),
                shape(&self.nodes, x).clone(),
            )),
            DType::I32 => self.push(Node::IterF32(
                Box::new([1.].into_iter()),
                shape(&self.nodes, x).clone(),
            )),
        };
        grads.insert(x, self.push(Node::Expand(grad1, shape(&self.nodes, x).clone())));
        self.release(grad1);
        // backpropagate
        // TODO this is not very clean code. Can we make it cleaner?
        for nid in topo {
            let grad = grads[&nid];
            match self.nodes[nid.i()] {
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
                        let two = match dtype(&self.nodes, y) {
                            DType::F32 => {
                                self.push(Node::IterF32(Box::new([2.].into_iter()), 1.into()))
                            }
                            DType::I32 => {
                                self.push(Node::IterI32(Box::new([2].into_iter()), 1.into()))
                            }
                        };
                        let two_e = self.push(Node::Expand(two, shape(&self.nodes, y).clone()));
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
                        let one = match dtype(&self.nodes, y) {
                            DType::F32 => {
                                self.push(Node::IterF32(Box::new([1.].into_iter()), 1.into()))
                            }
                            DType::I32 => {
                                self.push(Node::IterI32(Box::new([1].into_iter()), 1.into()))
                            }
                        };
                        let one1 = self.push(Node::Expand(one, shape(&self.nodes, y).clone()));
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
                        let zero = match dtype(&self.nodes, x) {
                            DType::F32 => {
                                self.push(Node::IterF32(Box::new([0.].into_iter()), 1.into()))
                            }
                            DType::I32 => {
                                self.push(Node::IterI32(Box::new([0].into_iter()), 1.into()))
                            }
                        };
                        let zeros = self.push(Node::Expand(zero, shape(&self.nodes, x).clone()));
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
                        let x_shape = shape(&self.nodes, x).clone();
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
                    grads.entry(x).or_insert_with(|| match dtype(&self.nodes, x) {
                        DType::F32 => self.push(Node::CastF32(grad)),
                        DType::I32 => self.push(Node::CastI32(grad)),
                    });
                }
                Node::CastI32(x) => {
                    grads.entry(x).or_insert_with(|| match dtype(&self.nodes, x) {
                        DType::F32 => self.push(Node::CastF32(grad)),
                        DType::I32 => self.push(Node::CastI32(grad)),
                    });
                }
                Node::Neg(x) => {
                    grads.entry(x).or_insert_with(|| self.push(Node::Neg(grad)));
                }
                Node::Tanh(x) => {
                    grads.entry(x).or_insert_with(|| {
                        // 1 - tanh^2(x)
                        let shape = shape(&self.nodes, x).clone();
                        let (two1, one1) = match dtype(&self.nodes, x) {
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
                        .or_insert_with(|| self.push(Node::Reshape(grad, shape(&self.nodes, x).clone())));
                }
                Node::Expand(x, ref sh) => {
                    if !grads.contains_key(&x) {
                        let org_shape = shape(&self.nodes, x).clone();
                        let axes = org_shape.expand_axes(sh);
                        let temp = self.push(Node::Sum(grad, axes, org_shape.clone()));
                        let x_grad = self.push(Node::Reshape(temp, org_shape));
                        self.release(temp);
                        grads.insert(x, x_grad);
                    }
                }
                Node::Permute(x, ref axes, _) => {
                    if !grads.contains_key(&x) {
                        let shape = shape(&self.nodes, x);
                        grads.insert(
                            x,
                            self.push(Node::Permute(grads[&nid], axes.argsort(), shape.clone())),
                        );
                    }
                }
                Node::Sum(x, ..) => {
                    grads
                        .entry(x)
                        .or_insert_with(|| self.push(Node::Expand(grad, shape(&self.nodes, x).clone())));
                }
                Node::Max(x, ..) => {
                    grads.entry(x).or_insert_with(|| {
                        // x_grad = (1 - (x < z.expand(x.shape()))) * grad
                        let x_shape = shape(&self.nodes, x).clone();
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
