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
    /// Check if there are no more buffers on id x
    fn is_free_id(&self, x: Id) -> bool;
    /// Delete all memory used by tensor x.
    fn remove(&mut self, x: Id) -> Result<(), ZyxError>;
    /// Store iterator into runtime backend
    fn store<T: Scalar, IT>(&mut self, x: Id, iter: IT) -> Result<(), ZyxError>
    where
        IT: IntoIterator<Item = T>,
        IT::IntoIter: ExactSizeIterator;
    /// Load evaluated tensor x.
    fn load<T: Scalar>(&mut self, x: Id, numel: usize) -> Result<Vec<T>, ZyxError>;
    /// Evaluate tensors to_eval with given graph of nodes and recommended
    /// order of evaluation.
    fn evaluate(
        &mut self,
        rcs: BTreeMap<Id, u32>,
        order: &[Id],
        nodes: &[Node],
    ) -> Result<(), ZyxError>;
}

/// Runtime with autograd engine.
/// This runtime uses [Node] enum as representation of tensors.
pub struct Runtime<R: RuntimeBackend> {
    rng: rand::rngs::SmallRng,
    rcs: Vec<u32>,
    nodes: Vec<Node>,
    backprop_nodes_count: usize,
    runtime_backend: R,
}

impl<R: RuntimeBackend> Runtime<R> {
    /// Initialize new runtime.
    #[must_use]
    pub fn new(runtime_backend: R) -> Self {
        use rand::SeedableRng;
        Self {
            rng: rand::rngs::SmallRng::seed_from_u64(420_694_206_942_069),
            rcs: Vec::new(),
            nodes: Vec::new(),
            backprop_nodes_count: 0,
            runtime_backend,
        }
    }

    /// Create tensor initialized from normal distribution.
    #[must_use]
    pub fn randn(&mut self, shape: Shape, dtype: DType) -> Result<Id, ZyxError> {
        use rand::Rng;
        let n = shape.numel();
        let mut rng = self.rng.clone();
        use rand::distributions::Standard;
        let data1 = match dtype {
            DType::F32 => self.store::<f32, _>((0..n).map(move |_| rng.sample(Standard))),
            DType::F64 => self.store::<f64, _>((0..n).map(move |_| rng.sample(Standard))),
            DType::I32 => self.store::<i32, _>((0..n).map(move |_| rng.sample(Standard))),
        }?;
        let data = self.push(Node::Reshape(data1, shape))?;
        self.release(data1)?;
        // change the state of the random seed in rng
        for _ in 0..n {
            self.rng.sample::<f32, _>(Standard);
        }
        Ok(data)
    }

    /// Create uniform tensor from range low..high
    #[must_use]
    pub fn uniform<T: Scalar>(&mut self, shape: Shape, range: Range<T>) -> Result<Id, ZyxError> {
        // TODO for f32 in range 0.0..1.0 switch to Node::UniformF32 for better performance
        use rand::Rng;
        let n = shape.numel();
        let mut rng = self.rng.clone();
        use rand::distributions::Standard;
        let data1 = match T::dtype() {
            DType::F32 => self.store((0..n).map(move |_| {
                rng.sample(Uniform::new(
                    range.start.clone().into_f32(),
                    range.end.clone().into_f32(),
                ))
            })),
            DType::F64 => self.store((0..n).map(move |_| {
                rng.sample(Uniform::new(
                    range.start.clone().into_f64(),
                    range.end.clone().into_f64(),
                ))
            })),
            DType::I32 => self.store((0..n).map(move |_| {
                rng.sample(Uniform::new(
                    range.start.clone().into_i32(),
                    range.end.clone().into_i32(),
                ))
            })),
        }?;
        let data = self.push(Node::Reshape(data1, shape))?;
        self.release(data1)?;
        // change the state of the random seed in rng
        for _ in 0..n {
            self.rng.sample::<f32, _>(Standard);
        }
        Ok(data)
    }

    /// Get shape of tensor x
    #[must_use]
    pub fn shape(&self, x: Id) -> &Shape {
        get_shape(self.nodes.as_slice(), x)
    }

    /// Get dtype of tensor x
    #[must_use]
    pub fn dtype(&self, x: Id) -> DType {
        get_dtype(self.nodes.as_slice(), x)
    }

    /// Load tensor x
    #[must_use]
    pub fn load<T: Scalar>(&mut self, x: Id) -> Result<Vec<T>, ZyxError> {
        if !self.runtime_backend.is_evaluated(x) {
            self.evaluate(BTreeSet::from([x]))?;
        }
        let numel = get_shape(self.nodes.as_slice(), x).numel();
        //std::println!("Reading buffer with {numel} elements.");
        self.runtime_backend.load(x, numel)
    }

    /// Store iterator into runtime as tensor
    #[must_use]
    pub fn store<T: Scalar, IT>(&mut self, iter: IT) -> Result<Id, ZyxError>
    where
        IT: IntoIterator<Item = T>,
        IT::IntoIter: ExactSizeIterator,
    {
        // TODO optimizations for scalars and very small tensors, by using Node::Scalar(...) or Node::SmallTensor(..)
        // With those optimizations, these can be compiled into kernels for better performance.
        let iter = iter.into_iter();
        let len = iter.len();
        let node = Node::Leaf(len.into(), T::dtype());
        let id = if let Some(i) = self
            .rcs
            .iter()
            .enumerate()
            .position(|(i, rc)| *rc == 0 && self.runtime_backend.is_free_id(tensor::id(i)))
        {
            let id = tensor::id(i);
            self.rcs[i] = 1;
            self.nodes[i] = node;
            id
        } else {
            let id = tensor::id(self.rcs.len());
            self.rcs.push(1);
            self.nodes.push(node);
            id
        };
        //if id.i() == 1 { panic!("break") }
        self.runtime_backend.store(id, iter)?;
        self.backprop_nodes_count += 1;
        std::println!("Storing {id}, {:?}", self.rcs);
        Ok(id)
    }

    /// Push new Node into the graph creating new tensor.
    /// This function does ZERO verification that the node is correct, but it optimizes
    /// out useless operations (like reshaping to the same shape)
    #[must_use]
    pub fn push(&mut self, node: Node) -> Result<Id, ZyxError> {
        std::println!("Pushing {node:?}, len: {}", self.nodes.len());
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
            self.retain(nid);
        }
        let id = if let Some(i) = self
            .rcs
            .iter()
            .enumerate()
            .position(|(i, rc)| *rc == 0 && self.runtime_backend.is_free_id(tensor::id(i)))
        {
            let id = tensor::id(i);
            self.rcs[i] = 1;
            self.nodes[i] = node;
            id
        } else {
            let id = tensor::id(self.rcs.len());
            self.rcs.push(1);
            if self.rcs.len() > 4000000000 {
                panic!("Maximum number of tensors has been reached. Zyx supports up to 4 billion tensors. \
                Please check your code for memory leaks. If you really need to use more tensors, please raise an issue: https://github.com/zk4x/zyx");
            }
            self.nodes.push(node);
            id
        };
        std::println!("Assigned id: {id}");
        self.backprop_nodes_count += 1;
        // This regulates caching, 1000 tensors per batch seems like a good default
        if self.backprop_nodes_count > 1000 {
            // Approx. 50 KiB of Nodes
            self.evaluate([id].into_iter().collect::<BTreeSet<Id>>())?;
        }
        Ok(id)
    }

    /// Decrease reference count of x. If x's reference count reaches zero, this function will delete
    /// x and release all of it's predecessors in the graph.
    pub fn release(&mut self, x: Id) -> Result<(), ZyxError> {
        let mut params = Vec::with_capacity(10);
        params.push(x);
        while let Some(x) = params.pop() {
            self.rcs[x.i()] -= 1;
            //std::println!("Releasing {x} {:?}", self.rcs);
            if self.rcs[x.i()] == 0 {
                params.extend(self.nodes[x.i()].parameters());
                self.runtime_backend.remove(x)?;
            }
        }
        //std::println!("After release rcs {:?}", self.rcs);
        Ok(())
    }

    /// Increase reference count of tensor x.
    pub fn retain(&mut self, x: Id) {
        //std::println!("Retaining {x}, rcs: {:?}", self.rcs);
        //panic!();
        debug_assert!(
            self.rcs[x.i()] < u32::MAX,
            "Reference count of tensor {x} has been exceeded,\
        This is zyx bug. please report it at: https://github.com/zk4x/zyx"
        );
        self.rcs[x.i()] += 1;
    }

    /// Evaluate specified nodes.
    pub fn evaluate(&mut self, nodes: BTreeSet<Id>) -> Result<(), ZyxError> {
        // This whole function is needed so that we can batch ops together.
        // This aleviates the cost of keeping intermediate buffers for backpropagation,
        // as this function runs independently from backpropagation and if some tensors
        // are dropped after backpropagation, this function optimizes those away.
        // Basically the difference between immediate and lazy execution with caching.
        // We simply wait to get more information about the graph structure before
        // we push it to the device.

        //std::println!("Evaluating {nodes:?}, rcs: {:?}", self.rcs);

        // TODO in order to be more efficient, we can optimize the graph
        // by reordering nodes and removing unnecessary nodes
        // TODO should we decrease refcount of some other nodes to drop them from memory?
        // This memory <=> performance tradeoff should be decided by the user, with some setting.
        // TODO simplify this function if possible

        // Creation of graph (DFS) runs in linear time, max once per node in self.nodes.
        // Make a list of visited nodes and their reference counts.
        let mut temp_rcs: BTreeMap<Id, u32> = BTreeMap::new();
        let mut params: Vec<Id> = nodes.iter().copied().collect();
        params.reserve(100);
        while let Some(nid) = params.pop() {
            if !self.runtime_backend.is_evaluated(nid) {
                temp_rcs
                    .entry(nid)
                    .and_modify(|rc| *rc += 1)
                    .or_insert_with(|| {
                        params.extend(self.nodes[nid.i()].parameters());
                        1
                    });
            }
        }
        // Order them using rcs reference counts.
        let mut order = Vec::new();
        let mut rcs: BTreeMap<Id, u32> = BTreeMap::new();
        let mut params: Vec<Id> = nodes.iter().copied().collect();
        params.reserve(100);
        while let Some(nid) = params.pop() {
            if let Some(temp_rc) = temp_rcs.get(&nid) {
                let rc = rcs.entry(nid).and_modify(|rc| *rc += 1).or_insert(1);
                if *temp_rc == *rc {
                    order.push(nid);
                    params.extend(self.nodes[nid.i()].parameters());
                }
            }
        }
        order.reverse();
        //std::println!("Order {order:?}");

        // Just create another DFS that goes all the way to Node::Leaf and adds branches to drop_nodes
        // if needed.
        let mut drop_nodes = BTreeSet::new();
        let mut temp_rcs: BTreeMap<Id, u32> = BTreeMap::new();
        let mut params: Vec<Id> = nodes.iter().copied().collect();
        params.reserve(100);
        while let Some(nid) = params.pop() {
            if !matches!(self.nodes[nid.i()], Node::Leaf(..)) {
                temp_rcs
                    .entry(nid)
                    .and_modify(|rc| *rc += 1)
                    .or_insert_with(|| {
                        params.extend(self.nodes[nid.i()].parameters());
                        1
                    });
            } else {
                let rc = temp_rcs.entry(nid).and_modify(|rc| *rc += 1).or_insert(1);
                if *rc == self.rcs[nid.i()] && !nodes.contains(&nid) {
                    drop_nodes.insert(nid);
                }
            }
        }
        let mut new_order = Vec::new();
        let mut new_rcs: BTreeMap<Id, u32> = BTreeMap::new();
        let mut params: Vec<Id> = nodes.iter().copied().collect();
        params.reserve(100);
        while let Some(nid) = params.pop() {
            if let Some(temp_rc) = temp_rcs.get(&nid) {
                let rc = new_rcs.entry(nid).and_modify(|rc| *rc += 1).or_insert(1);
                if *temp_rc == *rc {
                    new_order.push(nid);
                    params.extend(self.nodes[nid.i()].parameters());
                }
            }
        }
        new_order.reverse();

        std::println!("RCS: {:?}", self.rcs);
        // This must go over the graph from the previous loop!
        let mut new_leafs = BTreeSet::new();
        for nid in &new_order {
            if new_rcs[nid] == self.rcs[nid.i()] && !nodes.contains(nid) {
                for p in self.nodes[nid.i()].parameters() {
                    if drop_nodes.contains(&p) {
                        drop_nodes.insert(*nid);
                    }
                }
            } else {
                if self.nodes[nid.i()]
                    .parameters()
                    .any(|p| drop_nodes.contains(&p))
                {
                    new_leafs.insert(*nid);
                }
            }
        }

        std::println!("Drop nodes: {drop_nodes:?}");
        std::println!("New leafs {new_leafs:?}");

        //self.backprop_nodes_count -= drop_nodes.len();
        //std::println!("Order: {order:?}");
        /*std::println!("");
        for nid in &order {
            std::println!("{nid} x {}: {:?}", rcs[nid], self.nodes[nid.i()]);
        }
        std::println!("");*/
        self.runtime_backend.evaluate(rcs, &order, &self.nodes)?;

        // TODO all evaluated Detach nodes should be renamed to leafs also.
        for nid in new_leafs {
            self.nodes[nid.i()] = Node::Leaf(
                get_shape(&self.nodes, nid).clone(),
                get_dtype(&self.nodes, nid),
            );
        }
        for nid in drop_nodes {
            self.rcs[nid.i()] -= 1;
            if self.rcs[nid.i()] == 0 {
                //std::println!("Removing {nid}");
                self.runtime_backend.remove(nid)?;
            }
        }

        if self.backprop_nodes_count > 2000000000 {
            panic!("Maximum number of tensors in gradient tape has been reached. Zyx supports up to 2 billion tensors on the tape.\
            This error can be raised for example in RNNs. Please detach gradient tape (Tensor::detach) from some tensors.\
            If you really need to use more tensors, please raise an issue: https://github.com/zk4x/zyx");
        }
        Ok(())
    }

    /// Plot dot graph in dot format between given nodes
    #[must_use]
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
                    if !sources.contains(&nid) && !matches!(nodes[nid.i()], Node::Detach(..)) {
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
            DType::F32 => self.store([1f32]),
            DType::F64 => self.store([1f64]),
            DType::I32 => self.store([1i32]),
        }?;
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
                Node::Detach(..) | Node::Leaf(..) | Node::Uniform(..) => {}
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
                            DType::F32 => self.store([2f32]),
                            DType::F64 => self.store([2f64]),
                            DType::I32 => self.store([2i32]),
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
                            DType::F32 => self.store([1f32]),
                            DType::F64 => self.store([1f64]),
                            DType::I32 => self.store([1i32]),
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
                    panic!(
                        "Compare less than (cmplt, operator <) is not a differentiable operation."
                    );
                }
                Node::Where(x, y, z) => {
                    //return None, \
                    //self.x.e(TernaryOps.WHERE, grad_output, grad_output.const(0)) if self.needs_input_grad[1] else None, \
                    //self.x.e(TernaryOps.WHERE, grad_output.const(0), grad_output) if self.needs_input_grad[2] else None
                    if req_grad.contains(&y) {
                        let zero = match get_dtype(&self.nodes, x) {
                            DType::F32 => self.store([0f32]),
                            DType::F64 => self.store([0f64]),
                            DType::I32 => self.store([0i32]),
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
                            DType::F32 => self.store([0f32]),
                            DType::F64 => self.store([0f64]),
                            DType::I32 => self.store([0i32]),
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
                        DType::F32 => self.store([0f32]),
                        DType::F64 => self.store([0f64]),
                        DType::I32 => self.store([0i32]),
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
                    let two1 = match get_dtype(&self.nodes, x) {
                        DType::F32 => self.store([2f32]),
                        DType::F64 => self.store([2f64]),
                        DType::I32 => self.store([2i32]),
                    }?;
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
                        DType::F32 => (self.store([2f32])?, self.store([1f32])?),
                        DType::F64 => (self.store([2f64])?, self.store([1f64])?),
                        DType::I32 => (self.store([2i32])?, self.store([1i32])?),
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
                    let one1 = match get_dtype(&self.nodes, x) {
                        DType::F32 => self.store([1f32]),
                        DType::F64 => self.store([1f64]),
                        DType::I32 => self.store([1i32]),
                    }?;
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
