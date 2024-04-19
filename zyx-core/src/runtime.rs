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
use std::hash::Hash;
use half::f16;

/// RuntimeBackend is a good plug in point for backend developers.
/// Use Runtime::new(YourOwnStructThatImplementsRuntimeBackend::new()) to write your
/// own backend which needs to implement only evaluation of graph.
/// Used by torch and native backends.
pub trait RuntimeBackend {
    /// Compiled graph of nodes
    type CompiledGraph;
    /// Can this id be used by new nodes?
    fn is_empty(&self, x: Id) -> bool;
    /// Returns all evaluated node ids
    fn evaluated_nodes(&self) -> BTreeSet<Id>;
    /// Store iterator into runtime backend
    fn store<T: Scalar, IT>(&mut self, x: Id, iter: IT) -> Result<(), ZyxError>
    where
        IT: IntoIterator<Item = T>,
        IT::IntoIter: ExactSizeIterator;
    /// Load evaluated tensor x.
    fn load<T: Scalar>(&mut self, x: Id, numel: usize) -> Result<Vec<T>, ZyxError>;
    /// Delete all memory used by tensor x.
    fn remove(&mut self, x: Id) -> Result<(), ZyxError>;
    /// Compile graph of nodes into CompiledGraph, apply all optimizations
    fn compile_graph(&mut self, rcs: &[u32], nodes: &[Node], to_eval: &BTreeSet<Id>) -> Result<Self::CompiledGraph, ZyxError>;
    /// Launch compiled graph
    fn launch_graph(&mut self, graph: &Self::CompiledGraph) -> Result<(), ZyxError>;
}

/// Runtime with autograd engine.
/// This runtime uses [Node] enum as representation of tensors.
pub struct Runtime<R: RuntimeBackend> {
    rcs: Vec<u32>,
    nodes: Vec<Node>,
    unrealized_nodes_count: usize,
    runtime_backend: R,
    compiled_graphs: BTreeMap<u64, R::CompiledGraph>,
    /// Rng seed for use by backends
    pub rng_seed: u128,
}

impl<R: RuntimeBackend> Runtime<R> {
    /// Initialize new runtime.
    #[must_use]
    pub fn new(runtime_backend: R) -> Self {
        Self {
            rcs: Vec::new(),
            nodes: Vec::new(),
            unrealized_nodes_count: 0,
            runtime_backend,
            compiled_graphs: BTreeMap::new(),
            rng_seed: 42069,
        }
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
    pub fn load<T: Scalar>(&mut self, x: Id) -> Result<Vec<T>, ZyxError> {
        if self.runtime_backend.is_empty(x) {
            self.realize(BTreeSet::from([x]))?;
        }
        let numel = get_shape(self.nodes.as_slice(), x).numel();
        //std::println!("Reading buffer with {numel} elements.");
        self.runtime_backend.load(x, numel)
    }

    /// Store iterator into runtime as tensor
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
            .position(|(i, rc)| *rc == 0 && self.runtime_backend.is_empty(tensor::id(i)))
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
        //std::println!("Storing {id}, {:?}", self.rcs);
        Ok(id)
    }

    /// Push new Node into the graph creating new tensor.
    /// This function does ZERO verification that the node is correct, but it optimizes
    /// out useless operations (like reshaping to the same shape)
    pub fn push(&mut self, node: Node) -> Result<Id, ZyxError> {
        //std::println!("Pushing {node:?}, len: {}, rcs: {:?}", self.nodes.len(), self.rcs);
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
            .position(|(i, rc)| *rc == 0 && self.runtime_backend.is_empty(tensor::id(i)))
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
        //std::println!("Assigned id: {id}, rcs {:?}", self.rcs);
        self.unrealized_nodes_count += 1;
        // This regulates caching, 256 tensors per batch seems like a good default
        if self.unrealized_nodes_count > 10000 {
            self.realize([id].into_iter().collect::<BTreeSet<Id>>())?;
            //std::println!("Num tensors: {}", self.nodes.len());
        }
        Ok(id)
    }

    /// Decrease reference count of x. If x's reference count reaches zero, this function will delete
    /// x and release all of it's predecessors in the graph.
    pub fn release(&mut self, x: Id) -> Result<(), ZyxError> {
        //std::println!("Releasing {x}");
        let mut params = Vec::with_capacity(10);
        params.push(x);
        while let Some(x) = params.pop() {
            self.rcs[x.i()] -= 1;
            //std::println!("Releasing {x} {:?}", self.rcs);
            if self.rcs[x.i()] == 0 {
                params.extend(self.nodes[x.i()].parameters());
                self.runtime_backend.remove(x)?;
                // We count only non leaf nodes
                if !matches!(self.nodes[x.i()], Node::Leaf(..)) {
                    self.unrealized_nodes_count -= 1;
                }
            }
        }
        //std::println!("After released {x} rcs {:?}", self.rcs);
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

    /// Debug print all nodes
    pub fn debug_graph(&self) {
        for (id, node) in self.nodes.iter().enumerate() {
            std::println!("{id:>5} x{:>3} -> {node:?}", self.rcs[id]);
        }
    }

    /// Evaluate specified nodes.
    pub fn realize(&mut self, nodes: BTreeSet<Id>) -> Result<(), ZyxError> {
        use core::hash::Hasher;
        // TODO currently two different graphs can be hashed into the same hash,
        // so make sure that does not happen,
        // We can just clone hashed nodes into Vec, combine it with to_eval BTreeSet
        // and uses that as a key into compiled_graphs, but is it worth the space taken?
        // TODO replace this with our own no-std hasher
        let mut hash_state = core::hash::SipHasher::new();
        // Depth first search hashing all nodes required to evaluate subgraph
        let mut visited = BTreeSet::new();
        let mut params: Vec<Id> = nodes.iter().copied().collect();
        while let Some(param) = params.pop() {
            if visited.insert(param) {
                let node = &self.nodes[param.i()];
                node.hash(&mut hash_state);
                if self.runtime_backend.is_empty(param) {
                    params.extend(node.parameters());
                }
            }
        }
        for nid in &nodes {
            nid.hash(&mut hash_state);
        }
        let graph_hash = hash_state.finish();

        // If this graph was already compiled, use compiled graph
        let compiled_graph = if let Some(compiled_graph) = self.compiled_graphs.get(&graph_hash) {
            compiled_graph
        } else {
            let compiled_graph = self.runtime_backend.compile_graph(&self.rcs, &self.nodes, &nodes)?;
            self.compiled_graphs.entry(graph_hash).or_insert(compiled_graph)
        };
        self.runtime_backend.launch_graph(compiled_graph)?;

        // TODO delete nodes that are not needed for backpropagation anymore

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
                    if !sources.contains(&nid) && !matches!(nodes[nid.i()], Node::Detach(..) | Node::Cmplt(..)) {
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
            DType::F16 => self.store([f16::from_f32(1.0)]),
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
        // Should we just use Tensors here directly instead of Ids?
        // It will make it cleaner, because we do not have to put in all release calls,
        // but it is NOT worth the overhead, since we still need to do
        // all the req_grad.contains checks and such.
        for nid in topo {
            let grad = grads[&nid];
            match self.nodes[nid.i()] {
                Node::Const(..) | Node::Detach(..) | Node::Leaf(..) => {}
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
                            DType::F16 => self.store([f16::ONE + f16::ONE]),
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
                            DType::F16 => self.store([f16::ONE]),
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
                            DType::F16 => self.store([f16::ZERO]),
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
                            DType::F16 => self.store([f16::ZERO]),
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
                        DType::F16 => self.store([f16::ZERO]),
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
                        DType::F16 => self.store([f16::ONE + f16::ONE]),
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
                        DType::F16 => (self.store([f16::ONE+f16::ONE])?, self.store([f16::ONE])?),
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
                        DType::F16 => self.store([f16::ONE]),
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
