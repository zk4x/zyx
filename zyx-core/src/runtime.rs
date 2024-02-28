use crate::error::ZyxError;
use crate::utils::{get_dtype, get_shape};
use crate::{
    dtype::DType,
    node::Node,
    scalar::Scalar,
    shape::Shape,
    tensor::{self, Id},
};
use alloc::collections::btree_map::Entry;
use alloc::{
    boxed::Box,
    collections::{BTreeMap, BTreeSet},
    vec::Vec,
};
use core::ops::Range;
use rand::distributions::Uniform;

/// RuntimeBackend is a good plug in point for backend developers.
/// Use Runtime::new(YourOwnStructThatImplementsRuntimeBackend::new()) to write your
/// own backend which needs to implement only evaluation of graph.
/// Used by torch and native backends.
pub trait RuntimeBackend {
    /// Is tensor x evaluated?
    fn is_evaluated(&self, x: Id) -> bool;
    /// Delete all memory used by tensor x.
    fn remove(&mut self, x: Id) -> Result<(), ZyxError>;
    /// Load evaluated tensor x.
    fn load<T: Scalar>(&mut self, x: Id, numel: usize) -> Result<Vec<T>, ZyxError>;
    /// Evaluate tensors to_eval with given graph of nodes and recommended
    /// order of evaluation.
    fn evaluate(
        &mut self,
        to_eval: BTreeSet<Id>,
        rcs: BTreeMap<Id, u8>,
        order: &[Id],
        nodes: &mut [Node],
    ) -> Result<(), ZyxError>;
}

/// Runtime with autograd engine.
/// This runtime uses [Node] enum as representation of tensors.
pub struct Runtime<R: RuntimeBackend> {
    rng: rand::rngs::SmallRng,
    rcs: Vec<u8>,
    nodes: Vec<Node>,
    leafs: BTreeSet<Id>, // these do not need backward graph
    runtime_backend: R,
}

impl<R: RuntimeBackend> Runtime<R> {
    /// Initialize new runtime.
    pub fn new(runtime_backend: R) -> Self {
        use rand::SeedableRng;
        Self {
            rng: rand::rngs::SmallRng::seed_from_u64(420_694_206_942_069),
            rcs: Vec::new(),
            nodes: Vec::new(),
            leafs: BTreeSet::new(),
            runtime_backend,
        }
    }

    /// Create tensor initialized from normal distribution.
    pub fn randn(&mut self, shape: Shape, dtype: DType) -> Id {
        use rand::Rng;
        let n = shape.numel();
        let mut rng = self.rng.clone();
        use rand::distributions::Standard;
        let data = match dtype {
            DType::F32 => self.push(Node::IterF32(
                Box::new((0..n).map(move |_| rng.sample(Standard))),
                shape,
            )),
            DType::F64 => self.push(Node::IterF64(
                Box::new((0..n).map(move |_| rng.sample(Standard))),
                shape,
            )),
            DType::I32 => self.push(Node::IterI32(
                Box::new((0..n).map(move |_| rng.sample(Standard))),
                shape,
            )),
        }
        .unwrap(); // Can't fail, as this does not call backend
                   // change the state of the random seed in rng
        for _ in 0..n {
            self.rng.sample::<f32, _>(Standard);
        }
        data
    }

    /// Create uniform tensor from range low..high
    pub fn uniform<T: Scalar>(&mut self, shape: Shape, range: Range<T>) -> Id {
        // TODO for f32 in range 0.0..1.0 switch to Node::UniformF32 for better performance
        use rand::Rng;
        let n = shape.numel();
        let mut rng = self.rng.clone();
        use rand::distributions::Standard;
        let data = match T::dtype() {
            DType::F32 => self.push(Node::IterF32(
                Box::new((0..n).map(move |_| {
                    rng.sample(Uniform::new(
                        range.start.clone().into_f32(),
                        range.end.clone().into_f32(),
                    ))
                })),
                shape,
            )),
            DType::F64 => self.push(Node::IterF64(
                Box::new((0..n).map(move |_| {
                    rng.sample(Uniform::new(
                        range.start.clone().into_f64(),
                        range.end.clone().into_f64(),
                    ))
                })),
                shape,
            )),
            DType::I32 => self.push(Node::IterI32(
                Box::new((0..n).map(move |_| {
                    rng.sample(Uniform::new(
                        range.start.clone().into_i32(),
                        range.end.clone().into_i32(),
                    ))
                })),
                shape,
            )),
        }
        .unwrap(); // Can't fail, as this does not call backend
                   // change the state of the random seed in rng
        for _ in 0..n {
            self.rng.sample::<f32, _>(Standard);
        }
        data
    }

    /// Get shape of tensor x
    pub fn shape(&self, x: Id) -> &Shape {
        get_shape(self.nodes.as_slice(), x)
    }

    /// Get dtype of tensor x
    pub fn dtype(&self, x: Id) -> DType {
        get_dtype(self.nodes.as_slice(), x)
    }

    /// Load tensor x
    pub fn load<T: Scalar>(&mut self, x: Id) -> Result<Vec<T>, ZyxError> {
        // This may need to evaluate, therefore we need to take mutable reference to self
        if !self.runtime_backend.is_evaluated(x) {
            // TODO also check if these are only movements ops,
            // in which case we can directly return iterator with view
            self.evaluate(BTreeSet::from([x]))?;
        }
        let numel = get_shape(self.nodes.as_slice(), x).numel();
        //std::println!("Reading buffer with {numel} elements.");
        self.runtime_backend.load(x, numel)
    }

    /// Set tensor x as leaf. Leaf tensors do not need to store graph of operations,
    /// because user does not hold references to any of it's predecessors.
    pub fn set_leaf(&mut self, x: Id) {
        self.leafs.insert(x);
    }

    /// Push new Node into the graph creating new tensor.
    /// This function does ZERO verification that the node is correct, but it optimizes
    /// out useless operations (like reshaping to the same shape)
    pub fn push(&mut self, node: Node) -> Result<Id, ZyxError> {
        // get rid of noops :)
        match node {
            Node::Reshape(x, ref shape) | Node::Expand(x, ref shape) => {
                if shape == self.shape(x) {
                    self.retain(x);
                    return Ok(x);
                }
            }
            Node::Sum(x, ref axes, ..) | Node::Max(x, ref axes, ..) => {
                if axes.len() == 0 {
                    self.retain(x);
                    return Ok(x);
                }
            }
            _ => {}
        }
        for nid in node.parameters() {
            self.rcs[nid.i()] += 1;
        }
        let (i, new_node) = if let Some(i) = self.rcs.iter().position(|rc| *rc == 0) {
            (tensor::id(i), false)
        } else {
            (tensor::id(self.rcs.len()), true)
        };
        if new_node {
            self.rcs.push(1);
            self.nodes.push(node);
        } else {
            self.rcs[i.i()] = 1;
            self.nodes[i.i()] = node;
        }
        // TODO This evaluates last leafs if there are too many nodes,
        // is it better to find repeating nodes and evaluate those?
        if self.nodes.len() > 10000 {
            // Roughly 500 KiB of Nodes
            self.evaluate(
                self.leafs
                    .iter()
                    .copied()
                    .filter(|nid| self.is_user_id(*nid))
                    .collect::<BTreeSet<Id>>(),
            )?;
        }
        Ok(i)
    }

    fn is_user_id(&self, nid: Id) -> bool {
        let mut rc = self.rcs[nid.i()];
        for node in &self.nodes {
            for p in node.parameters() {
                if p == nid {
                    rc -= 1;
                }
            }
        }
        rc > 0
    }

    /// Decrease reference count of x. If x's reference count reaches zero, this function will delete
    /// x and release all of it's predecessors in the graph.
    pub fn release(&mut self, x: Id) -> Result<(), ZyxError> {
        let mut params = Vec::with_capacity(10);
        params.push(x);
        while let Some(p) = params.pop() {
            self.rcs[p.i()] -= 1;
            if self.rcs[p.i()] == 0 {
                params.extend(self.nodes[p.i()].parameters());
                self.leafs.remove(&p);
                self.runtime_backend.remove(p)?;
            }
        }
        Ok(())
    }

    /// Increase reference count of tensor x.
    pub fn retain(&mut self, x: Id) {
        self.rcs[x.i()] += 1;
    }

    /// Evaluate specified nodes.
    pub fn evaluate(&mut self, mut nodes: BTreeSet<Id>) -> Result<(), ZyxError> {
        // TODO in order to be more efficient, we can optimize the graph
        // by reordering nodes and removing unnecessary nodes

        // This creation of graph that needs to be evaluated runs in linear time,
        // max once per node in self.nodes

        // Make a list of visited nodes and their reference counts.
        let mut params: Vec<Id> = nodes.iter().copied().collect();
        let mut rcs: BTreeMap<Id, u8> = BTreeMap::new();
        while let Some(nid) = params.pop() {
            rcs.entry(nid).and_modify(|rc| *rc += 1).or_insert_with(|| {
                if !self.runtime_backend.is_evaluated(nid) {
                    params.extend(self.nodes[nid.i()].parameters());
                }
                1
            });
        }

        // Order them using rcs reference counts
        let mut order = Vec::new();
        let mut internal_rcs: BTreeMap<Id, u8> = BTreeMap::new();
        let mut params: Vec<Id> = nodes.iter().copied().collect();
        while let Some(nid) = params.pop() {
            if let Some(rc) = rcs.get(&nid) {
                if *rc
                    == *internal_rcs
                        .entry(nid)
                        .and_modify(|rc| *rc += 1)
                        .or_insert(1)
                {
                    order.push(nid);
                    params.extend(self.nodes[nid.i()].parameters());
                }
            }
        }
        let order: Vec<Id> = order.into_iter().rev().collect();
        //std::println!("Order: {order:?}");

        // TODO should we increase refcount of some other nodes to keep them evaluated in memory?
        // Like if they are referenced by the user and in the graph that needs to be evaluated?
        // TODO this memory <=> performance tradeoff should be decided by the user, with some setting.
        for nid in &order {
            if matches!(
                self.nodes[nid.i()],
                Node::Leaf(..)
                    | Node::IterF32(..)
                    | Node::IterI32(..)
                    | Node::IterF64(..)
                    | Node::Uniform(..)
            ) {
                *rcs.get_mut(nid).unwrap() += 1;
                nodes.insert(*nid);
            }
        }

        for leaf in self.leafs.iter() {
            if let Some(rc) = rcs.get_mut(leaf) {
                *rc += 1;
                nodes.insert(*leaf);
            }
        }

        /*std::println!("");
        std::println!("");
        for nid in &order {
            std::println!("{nid} x {}: {:?}", rcs[nid], self.nodes[nid.i()]);
        }
        std::println!("");
        std::println!("");*/

        self.runtime_backend
            .evaluate(nodes, rcs, &order, self.nodes.as_mut())?;

        // Release parts of graph that are not needed for backpropagation
        for leaf in self.leafs.clone() {
            if self.runtime_backend.is_evaluated(leaf) {
                self.leafs.remove(&leaf);
                //std::println!("Releasing leaf {leaf}");
                let shape = get_shape(self.nodes.as_ref(), leaf).clone();
                let mut node = Node::Leaf(shape, get_dtype(self.nodes.as_ref(), leaf));
                core::mem::swap(self.nodes.get_mut(leaf.i()).unwrap(), &mut node);
                for nid in node.parameters() {
                    self.release(nid)?;
                }
            }
        }
        Ok(())
    }

    /// Plot dot graph in dot format between given nodes
    pub fn plot_graph_dot(&self, ids: &[Id]) -> alloc::string::String {
        // Make a list of visited nodes and their reference counts.
        let mut params: Vec<Id> = ids.into();
        let mut rcs: BTreeMap<Id, u8> = BTreeMap::new();
        while let Some(nid) = params.pop() {
            rcs.entry(nid).and_modify(|rc| *rc += 1).or_insert_with(|| {
                params.extend(self.nodes[nid.i()].parameters());
                1
            });
        }
        // Order them using rcs reference counts
        let mut order = Vec::new();
        let mut internal_rcs: BTreeMap<Id, u8> = BTreeMap::new();
        let mut params: Vec<Id> = ids.into();
        while let Some(nid) = params.pop() {
            if rcs[&nid]
                == *internal_rcs
                    .entry(nid)
                    .and_modify(|rc| *rc += 1)
                    .or_insert(1)
            {
                order.push(nid);
                if rcs.contains_key(&nid) {
                    params.extend(self.nodes[nid.i()].parameters());
                }
            }
        }
        // Build topo, this way it ensures that grad is not used in backprop
        // before it was insert_or_add by all parents.
        let mut topo: BTreeSet<Id> = ids.iter().copied().collect();
        for nid in order.into_iter().rev() {
            for p in self.nodes[nid.i()].parameters() {
                if topo.contains(&p) {
                    topo.insert(nid);
                }
            }
        }

        crate::utils::plot_graph_dot(&topo, &self.nodes, &self.rcs)
    }

    /// Common autograd engine, currently used by all backends.
    pub fn backward(
        &mut self,
        x: Id,
        sources: &BTreeSet<Id>,
    ) -> Result<BTreeMap<Id, Id>, ZyxError> {
        fn build_topo(x: Id, sources: &BTreeSet<Id>, nodes: &[Node]) -> Vec<Id> {
            // Make a list of visited nodes and their reference counts.
            let mut params: Vec<Id> = alloc::vec![x];
            let mut rcs: BTreeMap<Id, u8> = BTreeMap::new();
            while let Some(nid) = params.pop() {
                rcs.entry(nid).and_modify(|rc| *rc += 1).or_insert_with(|| {
                    if !sources.contains(&nid) {
                        params.extend(nodes[nid.i()].parameters());
                    }
                    1
                });
            }
            // Order them using rcs reference counts
            let mut order = Vec::new();
            let mut internal_rcs: BTreeMap<Id, u8> = BTreeMap::new();
            let mut params: Vec<Id> = alloc::vec![x];
            while let Some(nid) = params.pop() {
                if let Some(rc) = rcs.get(&nid) {
                    if *rc
                        == *internal_rcs
                            .entry(nid)
                            .and_modify(|rc| *rc += 1)
                            .or_insert(1)
                    {
                        order.push(nid);
                        params.extend(nodes[nid.i()].parameters());
                    }
                }
            }
            // Build topo, this way it ensures that grad is not used in backprop
            // before it was insert_or_add by all parents.
            let mut topo = Vec::new();
            let mut req_grad = sources.clone();
            let mut visited = BTreeSet::new();
            for nid in order.into_iter().rev() {
                for p in nodes[nid.i()].parameters() {
                    if req_grad.contains(&p) && visited.insert(nid) {
                        req_grad.insert(nid);
                        topo.push(nid);
                    }
                }
            }
            topo.reverse();
            topo
        }

        let topo = build_topo(x, sources, &self.nodes);
        //std::println!("Topo: {topo:?}");

        let req_grad: BTreeSet<Id> = topo
            .iter()
            .copied()
            .chain(sources.iter().copied())
            .collect();
        // Node -> Grad
        let mut grads: BTreeMap<Id, Id> = BTreeMap::new();
        // Initial gradient of ones
        let grad1 = match get_dtype(&self.nodes, x) {
            DType::F32 => self.push(Node::IterF32(Box::new([1.].into_iter()), [1].into()))?,
            DType::F64 => self.push(Node::IterF64(Box::new([1.].into_iter()), [1].into()))?,
            DType::I32 => self.push(Node::IterI32(Box::new([1].into_iter()), [1].into()))?,
        };
        let sh = get_shape(&self.nodes, x).clone();
        grads.insert(x, self.push(Node::Expand(grad1, sh))?);
        self.release(grad1)?;
        //std::println!("{:?}", self.nodes.last().unwrap());

        fn insert_or_add_grad<B: RuntimeBackend>(
            r: &mut Runtime<B>,
            grads: &mut BTreeMap<Id, Id>,
            x: Id,
            grad: Id,
        ) -> Result<(), ZyxError> {
            match grads.entry(x) {
                Entry::Vacant(e) => {
                    e.insert(grad);
                }
                Entry::Occupied(e) => {
                    let (k, prev_grad) = e.remove_entry();
                    grads.insert(k, r.push(Node::Add(prev_grad, grad))?);
                    r.release(prev_grad)?;
                    r.release(grad)?;
                }
            }
            Ok(())
        }

        // backpropagate
        // TODO this is not very clean code. Can we make it cleaner?
        for nid in topo {
            let grad = grads[&nid];
            match self.nodes[nid.i()] {
                Node::Leaf(..)
                | Node::Uniform(..)
                | Node::IterF32(..)
                | Node::IterF64(..)
                | Node::IterI32(..) => {}
                Node::Add(x, y) => {
                    if req_grad.contains(&x) {
                        self.retain(grad);
                        insert_or_add_grad(self, &mut grads, x, grad)?;
                    }
                    if req_grad.contains(&y) {
                        self.retain(grad);
                        insert_or_add_grad(self, &mut grads, y, grad)?;
                    }
                }
                Node::Sub(x, y) => {
                    if req_grad.contains(&x) {
                        self.retain(grad);
                        insert_or_add_grad(self, &mut grads, x, grad)?;
                    }
                    if req_grad.contains(&y) {
                        let grad = self.push(Node::Neg(grad))?;
                        insert_or_add_grad(self, &mut grads, y, grad)?;
                    }
                }
                Node::Mul(x, y) => {
                    if req_grad.contains(&x) {
                        let grad = self.push(Node::Mul(y, grad))?;
                        insert_or_add_grad(self, &mut grads, x, grad)?;
                    }
                    if req_grad.contains(&y) {
                        let grad = self.push(Node::Mul(x, grad))?;
                        insert_or_add_grad(self, &mut grads, y, grad)?;
                    }
                }
                Node::Div(x, y) => {
                    if req_grad.contains(&x) {
                        grads.insert(x, self.push(Node::Div(grad, y))?);
                        insert_or_add_grad(self, &mut grads, x, grad)?;
                    }
                    if req_grad.contains(&y) {
                        // -grad*x/(y^2)
                        let two = match get_dtype(&self.nodes, y) {
                            DType::F32 => {
                                self.push(Node::IterF32(Box::new([2.].into_iter()), 1.into()))
                            }
                            DType::F64 => {
                                self.push(Node::IterF64(Box::new([2.].into_iter()), 1.into()))
                            }
                            DType::I32 => {
                                self.push(Node::IterI32(Box::new([2].into_iter()), 1.into()))
                            }
                        }?;
                        let two_e =
                            self.push(Node::Expand(two, get_shape(&self.nodes, y).clone()))?;
                        self.release(two)?;
                        let two_2 = self.push(Node::Pow(y, two_e))?;
                        self.release(two_e)?;
                        let temp = self.push(Node::Mul(x, grad))?;
                        let temp_neg = self.push(Node::Neg(temp))?;
                        self.release(temp)?;
                        let y_grad = self.push(Node::Div(temp_neg, two_2))?;
                        self.release(temp_neg)?;
                        self.release(two_2)?;
                        grads.insert(y, y_grad);
                        insert_or_add_grad(self, &mut grads, y, grad)?;
                    }
                }
                Node::Pow(x, y) => {
                    if req_grad.contains(&x) {
                        // grad * y * x.pow(y-1)
                        let one = match get_dtype(&self.nodes, y) {
                            DType::F32 => {
                                self.push(Node::IterF32(Box::new([1.].into_iter()), 1.into()))
                            }
                            DType::F64 => {
                                self.push(Node::IterF64(Box::new([1.].into_iter()), 1.into()))
                            }
                            DType::I32 => {
                                self.push(Node::IterI32(Box::new([1].into_iter()), 1.into()))
                            }
                        }?;
                        let one1 =
                            self.push(Node::Expand(one, get_shape(&self.nodes, y).clone()))?;
                        self.release(one)?;
                        let y_1 = self.push(Node::Sub(y, one1))?;
                        self.release(one1)?;
                        let pow_y_1 = self.push(Node::Pow(x, y_1))?;
                        self.release(y_1)?;
                        let y_mul = self.push(Node::Mul(y, pow_y_1))?;
                        self.release(pow_y_1)?;
                        let x_grad = self.push(Node::Mul(grad, y_mul))?;
                        self.release(y_mul)?;
                        insert_or_add_grad(self, &mut grads, x, x_grad)?;
                    }
                    if req_grad.contains(&y) {
                        // grad * x.pow(y) * ln(x)
                        let temp1 = self.push(Node::Ln(x))?;
                        let temp2 = self.push(Node::Mul(nid, temp1))?;
                        self.release(temp1)?;
                        let y_grad = self.push(Node::Mul(grad, temp2))?;
                        self.release(temp2)?;
                        insert_or_add_grad(self, &mut grads, y, y_grad)?;
                    }
                }
                Node::Cmplt(..) => {
                    panic!("Compare less than (operator <) is not a differentiable operation.");
                }
                Node::Where(x, y, z) => {
                    //return None, \
                    //self.x.e(TernaryOps.WHERE, grad_output, grad_output.const(0)) if self.needs_input_grad[1] else None, \
                    //self.x.e(TernaryOps.WHERE, grad_output.const(0), grad_output) if self.needs_input_grad[2] else None
                    if req_grad.contains(&y) {
                        let zero = match get_dtype(&self.nodes, x) {
                            DType::F32 => {
                                self.push(Node::IterF32(Box::new([0.].into_iter()), 1.into()))
                            }
                            DType::F64 => {
                                self.push(Node::IterF64(Box::new([0.].into_iter()), 1.into()))
                            }
                            DType::I32 => {
                                self.push(Node::IterI32(Box::new([0].into_iter()), 1.into()))
                            }
                        }?;
                        let zeros =
                            self.push(Node::Expand(zero, get_shape(&self.nodes, x).clone()))?;
                        self.release(zero)?;
                        let y_grad = self.push(Node::Where(x, grad, zeros))?;
                        self.release(zeros)?;
                        insert_or_add_grad(self, &mut grads, y, y_grad)?;
                    }
                    if req_grad.contains(&z) {
                        let zero = match get_dtype(&self.nodes, x) {
                            DType::F32 => {
                                self.push(Node::IterF32(Box::new([0.].into_iter()), 1.into()))
                            }
                            DType::F64 => {
                                self.push(Node::IterF64(Box::new([0.].into_iter()), 1.into()))
                            }
                            DType::I32 => {
                                self.push(Node::IterI32(Box::new([0].into_iter()), 1.into()))
                            }
                        }?;
                        let zeros =
                            self.push(Node::Expand(zero, get_shape(&self.nodes, x).clone()))?;
                        self.release(zero)?;
                        let z_grad = self.push(Node::Where(x, zeros, grad))?;
                        self.release(zeros)?;
                        insert_or_add_grad(self, &mut grads, z, z_grad)?;
                    }
                }
                Node::ReLU(x) => {
                    let zero = match get_dtype(&self.nodes, x) {
                        DType::F32 => {
                            self.push(Node::IterF32(Box::new([0.].into_iter()), 1.into()))
                        }
                        DType::F64 => {
                            self.push(Node::IterF64(Box::new([0.].into_iter()), 1.into()))
                        }
                        DType::I32 => self.push(Node::IterI32(Box::new([0].into_iter()), 1.into())),
                    }?;
                    let zeros = self.push(Node::Expand(zero, get_shape(&self.nodes, x).clone()))?;
                    self.release(zero)?;
                    let zl = self.push(Node::Cmplt(zeros, x))?;
                    self.release(zeros)?;
                    let x_grad = self.push(Node::Mul(zl, grad))?;
                    self.release(zl)?;
                    insert_or_add_grad(self, &mut grads, x, x_grad)?;
                }
                Node::Exp(x) => {
                    let grad = self.push(Node::Mul(nid, grad))?;
                    insert_or_add_grad(self, &mut grads, x, grad)?;
                }
                Node::Ln(x) => {
                    let grad = self.push(Node::Div(grad, x))?;
                    insert_or_add_grad(self, &mut grads, x, grad)?;
                }
                Node::Sin(x) => {
                    let x_temp = self.push(Node::Cos(x))?;
                    let grad = self.push(Node::Mul(x_temp, grad))?;
                    self.release(x_temp)?;
                    insert_or_add_grad(self, &mut grads, x, grad)?;
                }
                Node::Cos(x) => {
                    let x_temp1 = self.push(Node::Sin(x))?;
                    let x_temp = self.push(Node::Neg(x_temp1))?;
                    self.release(x_temp1)?;
                    let grad = self.push(Node::Mul(x_temp, grad))?;
                    self.release(x_temp)?;
                    insert_or_add_grad(self, &mut grads, x, grad)?;
                }
                Node::Sqrt(x) => {
                    // x_grad = grad/(2*sqrt(x))
                    let x_shape = get_shape(&self.nodes, x).clone();
                    let two1 = self.push(Node::IterF32(Box::new([2.].into_iter()), 1.into()))?;
                    let two2 = self.push(Node::Expand(two1, x_shape))?;
                    self.release(two1)?;
                    let x_temp = self.push(Node::Mul(two2, nid))?;
                    self.release(two2)?;
                    let grad = self.push(Node::Div(grad, x_temp))?;
                    self.release(x_temp)?;
                    insert_or_add_grad(self, &mut grads, x, grad)?;
                }
                Node::Cast(x, _) => {
                    let grad = self.push(Node::Cast(grad, get_dtype(&self.nodes, x)))?;
                    insert_or_add_grad(self, &mut grads, x, grad)?;
                }
                Node::Neg(x) => {
                    let grad = self.push(Node::Neg(grad))?;
                    insert_or_add_grad(self, &mut grads, x, grad)?;
                }
                Node::Tanh(x) => {
                    // 1 - tanh^2(x)
                    let shape = get_shape(&self.nodes, x).clone();
                    let (two1, one1) = match get_dtype(&self.nodes, x) {
                        DType::F32 => (
                            self.push(Node::IterF32(Box::new([2.].into_iter()), 1.into()))?,
                            self.push(Node::IterF32(Box::new([1.].into_iter()), 1.into()))?,
                        ),
                        DType::F64 => (
                            self.push(Node::IterF64(Box::new([2.].into_iter()), 1.into()))?,
                            self.push(Node::IterF64(Box::new([1.].into_iter()), 1.into()))?,
                        ),
                        DType::I32 => (
                            self.push(Node::IterI32(Box::new([2].into_iter()), 1.into()))?,
                            self.push(Node::IterI32(Box::new([1].into_iter()), 1.into()))?,
                        ),
                    };
                    let two2 = self.push(Node::Expand(two1, shape.clone()))?;
                    self.release(two1)?;
                    let two = self.push(Node::Pow(nid, two2))?;
                    self.release(two2)?;
                    let one2 = self.push(Node::Expand(one1, shape))?;
                    self.release(one1)?;
                    let one = self.push(Node::Sub(one2, two))?;
                    self.release(one2)?;
                    self.release(two)?;
                    let grad = self.push(Node::Mul(one, grad))?;
                    self.release(one)?;
                    insert_or_add_grad(self, &mut grads, x, grad)?;
                }
                Node::Reshape(x, ..) => {
                    let grad = self.push(Node::Reshape(grad, get_shape(&self.nodes, x).clone()))?;
                    insert_or_add_grad(self, &mut grads, x, grad)?;
                }
                Node::Expand(x, ref sh) => {
                    let org_shape = get_shape(&self.nodes, x).clone();
                    let axes = org_shape.expand_axes(sh);
                    let temp = self.push(Node::Sum(grad, axes, org_shape.clone()))?;
                    let grad = self.push(Node::Reshape(temp, org_shape))?;
                    self.release(temp)?;
                    insert_or_add_grad(self, &mut grads, x, grad)?;
                }
                Node::Permute(x, ref axes, _) => {
                    let shape = get_shape(&self.nodes, x);
                    let grad = self.push(Node::Permute(grad, axes.argsort(), shape.clone()))?;
                    insert_or_add_grad(self, &mut grads, x, grad)?;
                }
                Node::Pad(x, ref padding, _) => {
                    let sh = get_shape(&self.nodes, x).clone();
                    let inv_padding = padding.iter().map(|(lp, rp)| (-lp, -rp)).collect();
                    let grad = self.push(Node::Pad(grad, inv_padding, sh))?;
                    insert_or_add_grad(self, &mut grads, x, grad)?;
                }
                Node::Sum(x, ..) => {
                    let grad = self.push(Node::Expand(grad, get_shape(&self.nodes, x).clone()))?;
                    insert_or_add_grad(self, &mut grads, x, grad)?;
                }
                Node::Max(x, ..) => {
                    // x_grad = (1 - (x < z.expand(x.shape()))) * grad
                    let x_shape = get_shape(&self.nodes, x).clone();
                    let z_temp = self.push(Node::Expand(nid, x_shape.clone()))?;
                    let cmp_t = self.push(Node::Cmplt(x, z_temp))?;
                    self.release(z_temp)?;
                    let one1 = self.push(Node::IterF32(Box::new([1.].into_iter()), 1.into()))?;
                    let one2 = self.push(Node::Expand(one1, x_shape))?;
                    self.release(one1)?;
                    let max_1s = self.push(Node::Sub(one2, cmp_t))?;
                    self.release(one2)?;
                    self.release(cmp_t)?;
                    let grad = self.push(Node::Mul(max_1s, grad))?;
                    self.release(max_1s)?;
                    insert_or_add_grad(self, &mut grads, x, grad)?;
                }
            }
        }
        let mut res = BTreeMap::new();
        for (k, v) in grads.into_iter() {
            if sources.contains(&k) {
                res.insert(k, v);
            } else {
                self.release(v)?;
            }
        }
        Ok(res)
    }
}
