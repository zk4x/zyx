extern crate alloc;

use crate::node_id::NodeId;
use crate::device::{Device, Storage};
use crate::{axes::Axes, dtype::DType, shape::Shape, tensor::Tensor, OutOfMemoryError};
use alloc::boxed::Box;
use alloc::collections::BTreeMap;
use alloc::collections::BTreeSet;
use alloc::vec;
use alloc::vec::Vec;
use alloc::{format, string::String};
use rand::RngCore;

pub(super) enum Node {
    None,
    Const(Storage), // Only needed for evaluation in devices
    Leaf,
    StoreF32(Box<[f32]>, Shape),
    StoreI32(Box<[i32]>, Shape),
    Add(NodeId, NodeId),
    Sub(NodeId, NodeId),
    Mul(NodeId, NodeId),
    Div(NodeId, NodeId),
    Pow(NodeId, NodeId),
    TDot(NodeId, NodeId, Shape), // Matmul A x B with B transposed
    Cast(NodeId, DType),
    Neg(NodeId),
    Ln(NodeId),
    Sin(NodeId),
    Sqrt(NodeId),
    Exp(NodeId),
    ReLU(NodeId),
    DReLU(NodeId),
    Tanh(NodeId),
    Dropout(NodeId, u64, f32), // parameter, random seed, probability
    Reshape(NodeId, Shape),
    Expand(NodeId, Shape),
    Permute(NodeId, Axes, Shape),
    Sum(NodeId, Axes, Shape),
    Max(NodeId, Axes, Shape),
}

impl Clone for Node {
    fn clone(&self) -> Self {
        match self {
            Node::Const(..) | Node::StoreF32(..) | Node::StoreI32(..) => panic!(),
            Node::None => Node::None,
            Node::Leaf => Node::Leaf,
            Node::Add(x, y) => Node::Add(*x, *y),
            Node::Sub(x, y) => Node::Sub(*x, *y),
            Node::Mul(x, y) => Node::Mul(*x, *y),
            Node::Div(x, y) => Node::Div(*x, *y),
            Node::Pow(x, y) => Node::Pow(*x, *y),
            Node::TDot(x, y, s) => Node::TDot(*x, *y, s.clone()),
            Node::Cast(x, d) => Node::Cast(*x, *d),
            Node::Neg(x) => Node::Neg(*x),
            Node::Ln(x) => Node::Ln(*x),
            Node::Sin(x) => Node::Sin(*x),
            Node::Sqrt(x) => Node::Sqrt(*x),
            Node::ReLU(x) => Node::ReLU(*x),
            Node::DReLU(x) => Node::DReLU(*x),
            Node::Exp(x) => Node::Exp(*x),
            Node::Tanh(x) => Node::Tanh(*x),
            Node::Dropout(x, seed, prob) => Node::Dropout(*x, *seed, *prob),
            Node::Reshape(x, s) => Node::Reshape(*x, s.clone()),
            Node::Expand(x, s) => Node::Expand(*x, s.clone()),
            Node::Permute(x, a, s) => Node::Permute(*x, a.clone(), s.clone()),
            Node::Sum(x, a, s) => Node::Sum(*x, a.clone(), s.clone()),
            Node::Max(x, a, s) => Node::Max(*x, a.clone(), s.clone()),
        }
    }
}

impl core::fmt::Debug for Node {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Node::Const(..) => f.write_str("\x1b[31mConst\x1b[0m"),
            Node::None => f.write_str("\x1b[31mNone\x1b[0m"),
            Node::Leaf => f.write_str("\x1b[31mLeaf\x1b[0m"),
            Node::StoreF32(_, shape) => {
                f.write_fmt(format_args!("\x1b[31mStoreF32\x1b[0m() -> {shape}"))
            }
            Node::StoreI32(_, shape) => {
                f.write_fmt(format_args!("\x1b[31mStoreI32\x1b[0m() -> {shape})"))
            }
            Node::Add(x, y) => f.write_fmt(format_args!("\x1b[31mAdd\x1b[0m({x}, {y})")),
            Node::Sub(x, y) => f.write_fmt(format_args!("\x1b[31mSub\x1b[0m({x}, {y})")),
            Node::Mul(x, y) => f.write_fmt(format_args!("\x1b[31mMul\x1b[0m({x}, {y})")),
            Node::Div(x, y) => f.write_fmt(format_args!("\x1b[31mDiv\x1b[0m({x}, {y})")),
            Node::Pow(x, y) => f.write_fmt(format_args!("\x1b[31mPow\x1b[0m({x}, {y})")),
            Node::TDot(x, y, _) => f.write_fmt(format_args!("\x1b[31mTDot\x1b[0m({x}, {y})")),
            Node::Cast(x, dtype) => {
                f.write_fmt(format_args!("\x1b[31mCast\x1b[0m({x}) -> {dtype}"))
            }
            Node::Neg(x) => f.write_fmt(format_args!("\x1b[31mNeg\x1b[0m({x})")),
            Node::ReLU(x) => f.write_fmt(format_args!("\x1b[31mReLU\x1b[0m({x})")),
            Node::DReLU(x) => f.write_fmt(format_args!("\x1b[31mDReLU\x1b[0m({x})")),
            Node::Exp(x) => f.write_fmt(format_args!("\x1b[31mExp\x1b[0m({x})")),
            Node::Ln(x) => f.write_fmt(format_args!("\x1b[31mLn\x1b[0m({x})")),
            Node::Sin(x) => f.write_fmt(format_args!("\x1b[31mSin\x1b[0m({x})")),
            Node::Sqrt(x) => f.write_fmt(format_args!("\x1b[31mSqrt\x1b[0m({x})")),
            Node::Tanh(x) => f.write_fmt(format_args!("\x1b[31mTanh\x1b[0m({x})")),
            Node::Dropout(x,_, prob) => f.write_fmt(format_args!("\x1b[31mDropout\x1b[0m({x}, prob={prob})")),
            Node::Reshape(x, ..) => f.write_fmt(format_args!("\x1b[31mReshape\x1b[0m({x})")),
            Node::Expand(x, ..) => f.write_fmt(format_args!("\x1b[31mExpand\x1b[0m({x})")),
            Node::Permute(x, axes, _) => {
                f.write_fmt(format_args!("\x1b[31mPermute\x1b[0m({x}) -> {axes}"))
            }
            Node::Sum(x, axes, _) => f.write_fmt(format_args!("\x1b[31mSum\x1b[0m({x}) -> {axes}")),
            Node::Max(x, axes, _) => f.write_fmt(format_args!("\x1b[31mMax\x1b[0m({x}) -> {axes}")),
        }
    }
}

impl Node {
    pub(crate) fn parameters(&self) -> Box<[NodeId]> {
        match self {
            Node::None | Node::Leaf | Node::Const(..) | Node::StoreF32(..) | Node::StoreI32(..) => {
                Box::new([])
            }
            Node::Add(x, y)
            | Node::Sub(x, y)
            | Node::Mul(x, y)
            | Node::Div(x, y)
            | Node::Pow(x, y)
            | Node::TDot(x, y, _) => Box::new([*x, *y]),
            Node::Cast(x, ..)
            | Node::Neg(x)
            | Node::ReLU(x)
            | Node::DReLU(x)
            | Node::Exp(x)
            | Node::Ln(x)
            | Node::Sin(x)
            | Node::Sqrt(x)
            | Node::Tanh(x)
            | Node::Dropout(x, ..)
            | Node::Reshape(x, ..)
            | Node::Expand(x, ..)
            | Node::Permute(x, ..)
            | Node::Sum(x, ..)
            | Node::Max(x, ..) => Box::new([*x]),
        }
    }

    #[cfg(feature = "debug1")]
    fn flop(&self, graph: &Graph) -> usize {
        match self {
            Node::None |
            Node::Leaf |
            Node::Const(..) |
            Node::StoreF32(..) |
            Node::StoreI32(..) |
            Node::Expand(..) | Node::Reshape(..) | Node::Permute(..) => 0,
            Node::Exp(x) |
            Node::ReLU(x) |
            Node::DReLU(x) |
            Node::Ln(x) |
            Node::Sin(x) |
            Node::Sqrt(x) |
            Node::Neg(x) |
            Node::Cast(x, ..) |
            Node::Dropout(x, ..) |
            Node::Tanh(x) => graph.shape(*x).numel(),
            Node::Add(x, y) |
            Node::Sub(x, y) |
            Node::Mul(x, y) |
            Node::Div(x, y) |
            Node::Pow(x, y) => graph.shape(*x).numel() + graph.shape(*y).numel(),
            Node::TDot(x, _, shape) => 2 * shape.numel() * (graph.shape(*x)[-1] - 1usize),
            Node::Sum(x, axes, shape) | Node::Max(x, axes, shape) => {
                let shapex = graph.shape(*x);
                let stridesx = shapex.strides();
                shapex.into_iter().zip(&stridesx).enumerate().map(|(a, (d, st))| axes.contains(a) as usize * (d - 1) * st).product::<usize>() * shape.numel()
            }
        }
    }
}

// Nodes are kept small
#[derive(Debug)]
pub(super) struct Graph {
    #[cfg(feature = "rand")]
    rng: rand::rngs::SmallRng,
    pub(super) devices: Vec<Device>,
    pub(super) default_device: usize,
    rc: Vec<u8>,                        // reference count of nodes
    nodes: Vec<Node>,                   // all nodes
    buffers: BTreeMap<NodeId, Storage>, // realized nodes
    labels: BTreeMap<NodeId, String>,   // labels
    leafs: BTreeSet<NodeId>,            // nodes which parameters can be dropped after realization
}

impl Default for Graph {
    fn default() -> Self {
        #[cfg(feature = "rand")]
        use rand::SeedableRng;
        Self {
            #[cfg(feature = "rand")]
            rng: rand::rngs::SmallRng::seed_from_u64(420694206942069),
            devices: vec![Device::CPU],
            default_device: 0,
            rc: Vec::with_capacity(128),
            nodes: Vec::with_capacity(128),
            buffers: BTreeMap::new(),
            labels: BTreeMap::new(),
            leafs: BTreeSet::new(),
        }
    }
}

impl Graph {
    pub(super) fn push(&mut self, node: Node) -> NodeId {
        for nid in &*node.parameters() {
            self.rc[nid.i()] += 1;
        }
        if let Some(id) = self.rc.iter().position(|rc| *rc == 0) {
            self.rc[id] = 1;
            self.nodes[id] = node;
            NodeId::new(id)
        } else {
            let id = self.rc.len();
            self.rc.push(1);
            self.nodes.push(node);
            NodeId::new(id)
        }
    }

    pub(super) fn retain(&mut self, id: NodeId) {
        self.rc[id.i()] += 1;
    }

    pub(super) fn release(&mut self, id: NodeId) {
        let mut params = Vec::with_capacity(10);
        params.push(id);
        while let Some(p) = params.pop() {
            self.rc[p.i()] -= 1;
            if self.rc[p.i()] == 0 {
                params.extend(self.nodes[p.i()].parameters().into_iter());
                self.nodes[p.i()] = Node::None;
                self.buffers.remove(&p);
                self.labels.remove(&p);
                self.leafs.remove(&p);
            }
        }
    }

    pub(super) fn shape(&self, mut id: NodeId) -> &Shape {
        loop {
            if let Some(storage) = self.buffers.get(&id) {
                return storage.shape();
            }
            match &self.nodes[id.i()] {
                Node::StoreF32(.., shape)
                | Node::StoreI32(.., shape)
                | Node::TDot(.., shape)
                | Node::Reshape(_, shape)
                | Node::Expand(_, shape)
                | Node::Permute(.., shape)
                | Node::Sum(.., shape)
                | Node::Max(.., shape) => return shape,
                _ => {}
            }
            //std::println!("{:?}", self.buffers.keys());
            //std::println!("Asking for shape of node {id} {:?}", self.nodes[id.i()]);
            id = self.nodes[id.i()].parameters()[0];
        }
    }

    pub(super) fn set_leaf(&mut self, id: NodeId) {
        self.leafs.insert(id);
    }

    pub(super) fn dtype(&self, id: NodeId) -> DType {
        let mut param = id;
        loop {
            if let Some(storage) = self.buffers.get(&param) {
                return storage.dtype();
            }
            match &self.nodes[param.i()] {
                Node::StoreF32(..) => return DType::F32,
                Node::StoreI32(..) => return DType::I32,
                Node::Cast(.., dtype) => return *dtype,
                _ => {}
            }
            param = self.nodes[param.i()].parameters()[0];
        }
    }

    pub(super) fn load_f32(&mut self, id: NodeId) -> Option<Box<[f32]>> {
        self.buffers
            .get(&id)
            .map(|storage| self.devices[self.default_device].load_f32(storage))
    }

    pub(super) fn load_i32(&mut self, id: NodeId) -> Option<Box<[i32]>> {
        self.buffers
            .get(&id)
            .map(|storage| self.devices[self.default_device].load_i32(storage))
    }

    pub(super) fn num_nodes(&self) -> usize {
        self.nodes.len()
    }

    #[allow(clippy::cast_precision_loss)]
    pub(super) fn realize(&mut self, nodes: &[NodeId]) -> Result<(), OutOfMemoryError> {
        // TODO write optimizer that will rearrange ops to make it easier for devices.
        // For example multiple permutes can be merged together, similar with other movement ops.
        // Also remove duplicates, i. e. in softmax there are two exp ops with same input, so rewrite it.
        // Also add caching, i. e. if there are parts of graph that are repeated multiple times (constant folding),
        // add those to nodes, so that they don't need to be recalculated every time.
        
        // TODO don't log as much when using debug1. The same graph should be printed only once.

        /*
        This function is complex (easy to introduce bugs), so here is what it does:
        1. calculate order of realization of nodes
        2. clone nodes that will be realized into graph, consts and stores are moved into graph
        3. realize graph and move constants to buffers
        4. drop all nodes required only for realization and not for backward pass,
           i. e. parameters of all self.leafs
         */

        // This block determines order of execution of nodes
        let mut params = nodes.to_vec();
        params.reserve(self.nodes.len());
        let mut order = Vec::new();
        while let Some(node_id) = params.pop() {
            order.push(node_id);
            if !self.buffers.contains_key(&node_id) {
                params.extend(&*self.nodes[node_id.i()].parameters());
            }
        }
        {
            order.reverse();
            // deduplicate
            let mut uniques = BTreeSet::new();
            order.retain(|x| uniques.insert(*x));
        }

        #[cfg(feature = "debug1")]
        std::println!("     Label |  NID |  RC |           Shape |  Node     realizing graph:");
        #[cfg(feature = "debug1")]
        for node_id in &order {
            use alloc::string::ToString;
            let mut sh = self.shape(*node_id).to_string();
            while sh.len() < 15 {
                sh.insert(0, ' ');
            }
            let mut lb = self.label(*node_id).unwrap_or(&String::new()).to_string();
            while lb.len() < 10 {
                lb.insert(0, ' ');
            }
            std::println!(
                "{} | {:>4} | {:>3} | {} |  {:?}",
                lb,
                node_id.i(),
                self.rc[node_id.i()],
                sh,
                self.nodes[node_id.i()]
            );
        }
        #[cfg(feature = "debug1")]
        let mut flop: u128 = 0;
        #[cfg(feature = "debug1")]
        for id in &order {
            flop += self.nodes[id.i()].flop(self) as u128;
        }
        #[cfg(feature = "debug1")]
        let mut mem: u128 = 0;
        #[cfg(feature = "debug1")]
        for id in &order {
            if matches!(self.nodes[id.i()], Node::Leaf | Node::StoreF32(..) | Node::StoreI32(..)) {
                mem += (self.shape(*id).numel() * self.dtype(*id).byte_size()) as u128;
            }
        }

        // Here we clone nodes into graph that will be executed by the device,
        // while also adding reference counts for nodes.
        let mut params = nodes.to_vec();
        let mut graph: BTreeMap<NodeId, (usize, Node)> = BTreeMap::new();
        let mut nodes: BTreeSet<NodeId> = nodes.iter().copied().collect();
        while let Some(node_id) = params.pop() {
            if let Some((rc, ..)) = graph.get_mut(&node_id) {
                *rc += 1;
            } else if let Some(storage) = self.buffers.remove(&node_id) {
                graph.insert(node_id, (2, Node::Const(storage)));
                nodes.insert(node_id);
            } else if matches!(
                self.nodes[node_id.i()],
                Node::StoreF32(..) | Node::StoreI32(..)
            ) {
                let mut node = Node::Leaf;
                core::mem::swap(&mut self.nodes[node_id.i()], &mut node);
                graph.insert(node_id, (2, node));
                nodes.insert(node_id);
            } else {
                graph.insert(node_id, (1, self.nodes[node_id.i()].clone()));
                params.extend(&*self.nodes[node_id.i()].parameters());
            }
        }

        #[cfg(feature = "debug1")]
        let begin = std::time::Instant::now();

        self.devices[self.default_device].realize(&mut graph, &order, &nodes)?;

        #[cfg(feature = "debug1")]
        let elapsed = begin.elapsed();
        #[cfg(feature = "debug1")]
        std::println!(
            "Elapsed {:}ms, {:.2} GFLOPS, {:.2} GB/s\n",
            elapsed.as_millis(),
            flop as f64 / elapsed.as_nanos() as f64,
            mem as f64 / elapsed.as_nanos() as f64
        );

        for id in nodes {
            let Node::Const(storage) = graph.remove(&id).unwrap().1 else {
                panic!()
            };
            self.buffers.insert(id, storage);
        }

        // Release parts of graph that are not needed for backpropagation
        while let Some(leaf) = self.leafs.pop_last() {
            let mut node = Node::Leaf;
            core::mem::swap(&mut self.nodes[leaf.i()], &mut node);
            for nid in &*node.parameters() {
                self.release(*nid);
            }
        }

        Ok(())
    }

    // Returns set of all nodes that require grad, if sources require grad
    pub(super) fn grad_nodes(&self, sources: &[&mut Tensor]) -> BTreeSet<NodeId> {
        let mut sources: BTreeSet<NodeId> = sources.iter().map(|tensor| tensor.data).collect();
        let mut n = 0;
        while n != sources.len() {
            n = sources.len();
            for (i, node) in self.nodes.iter().enumerate() {
                for param in &*node.parameters() {
                    if sources.contains(param) {
                        sources.insert(NodeId::new(i));
                    }
                }
            }
        }
        sources
    }

    #[allow(clippy::match_on_vec_items)]
    #[allow(clippy::too_many_lines)]
    pub(super) fn backward(
        &mut self,
        id: NodeId,
        grad: NodeId,
        sources: &mut [&mut Tensor],
        grad_nodes: &BTreeSet<NodeId>,
        visited: &mut BTreeSet<NodeId>,
    ) {
        // TODO this function needs to be simplified
        // TODO make this iterative instead of recursive
        if visited.contains(&id) { return }
        visited.insert(id);
        if let Some(idx) = sources.iter().position(|t| t.id() == id.i()) {
            if let Some(gradient) = &mut sources[idx].grad {
                let old_grad = *gradient;
                *gradient = self.push(Node::Add(*gradient, grad));
                //std::println!("grad rc {}", self.rc[grad.id()]);
                self.release(old_grad);
                //self.release(grad);
            } else {
                self.retain(grad);
                sources[idx].grad = Some(grad);
            }
        }
        //std::println!("Backwara on {:?}", self.nodes[id.id()]);
        match self.nodes[id.i()] {
            Node::None |
            Node::DReLU(..) |
            Node::Const(..)
            => panic!("Internal bug running backward on .."),
            Node::Leaf | Node::StoreF32(..) | Node::StoreI32(..) => {}
            Node::Add(x, y) => {
                if grad_nodes.contains(&x) {
                    self.backward(x, grad, sources, grad_nodes, visited);
                }
                if grad_nodes.contains(&y) {
                    self.backward(y, grad, sources, grad_nodes, visited);
                }
            }
            Node::Sub(x, y) => {
                if grad_nodes.contains(&x) {
                    self.backward(x, grad, sources, grad_nodes, visited);
                }
                if grad_nodes.contains(&y) {
                    let y_grad = self.push(Node::Neg(grad));
                    self.backward(y, y_grad, sources, grad_nodes, visited);
                    self.release(y_grad);
                }
            }
            Node::Mul(x, y) => {
                if grad_nodes.contains(&x) {
                    let x_grad = self.push(Node::Mul(y, grad));
                    self.backward(x, x_grad, sources, grad_nodes, visited);
                    self.release(x_grad);
                }
                if grad_nodes.contains(&y) {
                    let y_grad = self.push(Node::Mul(x, grad));
                    self.backward(y, y_grad, sources, grad_nodes, visited);
                    self.release(y_grad);
                }
            }
            Node::Div(x, y) => {
                if grad_nodes.contains(&x) {
                    let x_grad = self.push(Node::Div(grad, y));
                    self.backward(x, x_grad, sources, grad_nodes, visited);
                    self.release(x_grad);
                }
                if grad_nodes.contains(&y) {
                    // -grad*x/(y^2)
                    let two = match self.dtype(y) {
                        DType::F32 => self.tensor_from_iter_f32(1.into(), [2.]),
                        DType::I32 => self.tensor_from_iter_i32(1.into(), [2]),
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
                    self.backward(y, y_grad, sources, grad_nodes, visited);
                    self.release(y_grad);
                }
            }
            Node::Pow(x, y) => {
                if grad_nodes.contains(&x) {
                    // grad * y * x.pow(y-1)
                    let one = match self.dtype(y) {
                        DType::F32 => self.tensor_from_iter_f32(1.into(), [1.]),
                        DType::I32 => self.tensor_from_iter_i32(1.into(), [1]),
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
                    self.backward(x, x_grad, sources, grad_nodes, visited);
                    self.release(x_grad);
                }
                if grad_nodes.contains(&y) {
                    // grad * x.pow(y) * ln(x)
                    let temp1 = self.push(Node::Ln(x));
                    let temp2 = self.push(Node::Mul(id, temp1));
                    self.release(temp1);
                    let y_grad = self.push(Node::Mul(grad, temp2));
                    self.release(temp2);
                    self.backward(y, y_grad, sources, grad_nodes, visited);
                    self.release(y_grad);
                }
            }
            Node::TDot(x, y, _) => {
                // x_grad += grad @ y_data.T
                // y_grad += x_data.T @ grad
                //      z  k, m @ k, n -> m, n
                // grad x  n, k @ n, m -> k, m
                // grad y  m, k @ m, n -> k, n
                if grad_nodes.contains(&x) {
                    let grad_shape = self.shape(grad).clone();
                    let grad_temp = self.push(Node::Permute(grad, grad_shape.transpose_axes(), grad_shape.transpose()));
                    let y_shape = self.shape(y).clone();
                    let y_temp = self.push(Node::Permute(y, y_shape.transpose_axes(), y_shape.transpose()));
                    let x_grad = self.push(Node::TDot(y_temp, grad_temp, self.shape(x).clone()));
                    self.release(grad_temp);
                    self.release(y_temp);
                    self.backward(x, x_grad, sources, grad_nodes, visited);
                    self.release(x_grad);
                }
                if grad_nodes.contains(&y) {
                    let x_shape = self.shape(x).clone();
                    let x_temp = self.push(Node::Permute(x, x_shape.transpose_axes(), x_shape.transpose()));
                    let y_grad = self.push(Node::TDot(x_temp, grad, self.shape(y).clone()));
                    self.release(x_temp);
                    self.backward(y, y_grad, sources, grad_nodes, visited);
                    self.release(y_grad);
                }
            }
            Node::ReLU(x) => {
                let drelu = self.push(Node::DReLU(x));
                let x_grad = self.push(Node::Mul(drelu, grad));
                self.backward(x, x_grad, sources, grad_nodes, visited);
                self.release(x_grad);
            }
            Node::Exp(x) => {
                let x_grad = self.push(Node::Mul(id, grad));
                self.backward(x, x_grad, sources, grad_nodes, visited);
                self.release(x_grad);
            }
            Node::Ln(x) => {
                let x_grad = self.push(Node::Div(grad, x));
                self.backward(x, x_grad, sources, grad_nodes, visited);
                self.release(x_grad);
            }
            Node::Sin(x) => {
                todo!()
                //let x_grad = self.push(Node::Div(grad, x));
                //self.backward(x, x_grad, sources, grad_nodes, visited);
                //self.release(x_grad);
            }
            Node::Sqrt(x) => {
                todo!()
                //let x_grad = self.push(Node::Div(grad, x));
                //self.backward(x, x_grad, sources, grad_nodes, visited);
                //self.release(x_grad);
            }
            Node::Cast(x, _) => {
                let x_grad = self.push(Node::Cast(grad, self.dtype(x)));
                self.backward(x, x_grad, sources, grad_nodes, visited);
                self.release(x_grad);
            }
            Node::Neg(x) => {
                let x_grad = self.push(Node::Neg(grad));
                self.backward(x, x_grad, sources, grad_nodes, visited);
                self.release(x_grad);
            }
            Node::Dropout(x, ..) => {
                let x_grad = self.push(Node::Neg(grad));
                self.backward(x, x_grad, sources, grad_nodes, visited);
                self.release(x_grad);
            }
            Node::Tanh(x) => {
                // 1 - tanh^2(x)
                let shape = self.shape(x).clone();
                match self.dtype(x) {
                    DType::F32 => {
                        let two1 = self.tensor_from_iter_f32(1.into(), [2.]);
                        let two2 = self.push(Node::Expand(two1, shape.clone()));
                        self.release(two1);
                        let two = self.push(Node::Pow(id, two2));
                        self.release(two2);
                        let one1 = self.tensor_from_iter_f32(1.into(), [1.]);
                        let one2 = self.push(Node::Expand(one1, shape));
                        self.release(one1);
                        let one = self.push(Node::Sub(one2, two));
                        self.release(one2);
                        self.release(two);
                        let x_grad = self.push(Node::Mul(one, grad));
                        self.release(one);
                        self.backward(x, x_grad, sources, grad_nodes, visited);
                        self.release(x_grad);
                    }
                    DType::I32 => {
                        let two1 = self.tensor_from_iter_i32(1.into(), [2]);
                        let two2 = self.push(Node::Expand(two1, shape.clone()));
                        self.release(two1);
                        let two = self.push(Node::Pow(id, two2));
                        self.release(two2);
                        let one1 = self.tensor_from_iter_i32(1.into(), [1]);
                        let one2 = self.push(Node::Expand(one1, shape));
                        self.release(one1);
                        let one = self.push(Node::Sub(one2, two));
                        self.release(one2);
                        self.release(two);
                        let x_grad = self.push(Node::Mul(one, grad));
                        self.release(one);
                        self.backward(x, x_grad, sources, grad_nodes, visited);
                        self.release(x_grad);
                    }
                }
            }
            Node::Reshape(x, ..) => {
                let x_grad = self.push(Node::Reshape(grad, self.shape(x).clone()));
                self.backward(x, x_grad, sources, grad_nodes, visited);
                self.release(x_grad);
            }
            Node::Expand(x, ref shape) => {
                let org_shape = self.shape(x).clone();
                let axes = org_shape.expand_axes(shape);
                let shape = shape.clone().reduce(&axes);
                let temp = self.push(Node::Sum(grad, axes, shape));
                let x_grad = self.push(Node::Reshape(temp, org_shape));
                self.release(temp);
                self.backward(x, x_grad, sources, grad_nodes, visited);
                self.release(x_grad);
            }
            Node::Permute(x, ref axes, _) => {
                let x_grad = self.push(Node::Permute(grad, axes.argsort(), self.shape(x).clone()));
                self.backward(x, x_grad, sources, grad_nodes, visited);
                self.release(x_grad);
            }
            Node::Sum(x, ..) => {
                let org_shape = self.shape(x).clone();
                let x_grad = self.push(Node::Expand(grad, org_shape));
                self.backward(x, x_grad, sources, grad_nodes, visited);
                self.release(x_grad);
            }
            Node::Max(..) => {
                todo!("max backward is not yet implemented");
            }
        }
    }

    pub(super) fn debug_nodes(&self) -> alloc::vec::Vec<String> {
        use alloc::string::ToString;
        let mut res = alloc::vec::Vec::new();
        for (i, node) in self.nodes.iter().enumerate() {
            if matches!(node, Node::None) {
                continue;
            }
            //res.push(format!("{i:>4} -> {node:?}"));
            let node_id = NodeId::new(i);
            let mut sh = self.shape(node_id).to_string();
            while sh.len() < 11 {
                sh.insert(0, ' ');
            }
            let mut lb = self.label(node_id).unwrap_or(&String::new()).to_string();
            while lb.len() < 6 {
                lb.insert(0, ' ');
            }
            res.push(format!(
                "{} | {:>4} | {:>3} | {} |  {:?}",
                lb,
                node_id.i(),
                self.rc[i],
                sh,
                node
            ));
        }
        res
    }

    pub(super) fn label(&self, id: NodeId) -> Option<&String> {
        self.labels.get(&id)
    }

    pub(super) fn rand_u64(&mut self) -> u64 {
        self.rng.next_u64()
    }

    pub(super) fn set_label(&mut self, id: NodeId, label: &str) {
        self.labels.insert(id, label.into());
    }

    pub(super) fn show_graph(&self) -> String {
        use core::fmt::Write;
        let mut user_rc = self.rc.clone();
        for node in &self.nodes {
            for param in &*node.parameters() {
                user_rc[param.i()] -= 1;
            }
        }
        //std::println!("User {:?}", user_rc);
        let mut res = String::from("strict digraph {\n  ordering=in\n  rank=source\n");
        let mut add_node = |id: usize, text: &str, shape: &str| {
            let fillcolor = if user_rc[id] > 0 { "lightblue" } else { "grey" };
            if let Some(label) = self.labels.get(&NodeId::new(id)) {
                write!(res, "  {id}[label=\"{}NL{} x {}NL{}NL{}\", shape={}, fillcolor=\"{}\", style=filled]",
                    label, id, self.rc[id], text, self.shape(NodeId::new(id)), shape, fillcolor).unwrap();
            } else {
                write!(
                    res,
                    "  {id}[label=\"{} x {}NL{}NL{}\", shape={}, fillcolor=\"{}\", style=filled]",
                    id,
                    self.rc[id],
                    text,
                    self.shape(NodeId::new(id)),
                    shape,
                    fillcolor
                )
                .unwrap();
            }
            writeln!(res).unwrap();
        };
        let mut edges = String::new();
        for (id, node) in self.nodes.iter().enumerate() {
            match node {
                Node::Const(..) => add_node(id, "Const", "box"),
                Node::None => {}
                Node::Leaf => add_node(id, "Leaf", "box"),
                Node::StoreI32(..) => add_node(id, "StoreI32", "box"),
                Node::StoreF32(..) => add_node(id, "StoreF32", "box"),
                Node::Add(x, y) => add_node(id, &format!("Add({x}, {y})"), "oval"),
                Node::Sub(x, y) => add_node(id, &format!("Sub({x}, {y})"), "oval"),
                Node::Mul(x, y) => add_node(id, &format!("Mul({x}, {y})"), "oval"),
                Node::Div(x, y) => add_node(id, &format!("Div({x}, {y})"), "oval"),
                Node::Pow(x, y) => add_node(id, &format!("Pow({x}, {y})"), "oval"),
                Node::TDot(x, y, ..) => add_node(id, &format!("TDot({x}, {y})"), "oval"),
                Node::Neg(x) => add_node(id, &format!("Neg({x})"), "oval"),
                Node::Exp(x) => add_node(id, &format!("Exp({x})"), "oval"),
                Node::ReLU(x) => add_node(id, &format!("ReLU({x})"), "oval"),
                Node::DReLU(x) => add_node(id, &format!("DReLU({x})"), "oval"),
                Node::Dropout(x, _, prob) => add_node(id, &format!("Dropout({x}, prob={prob})"), "oval"),
                Node::Ln(x) => add_node(id, &format!("Ln({x})"), "oval"),
                Node::Sin(x) => add_node(id, &format!("Sin({x})"), "oval"),
                Node::Sqrt(x) => add_node(id, &format!("Sqrt({x})"), "oval"),
                Node::Tanh(x) => add_node(id, &format!("Tanh({x})"), "oval"),
                Node::Expand(x, ..) => add_node(id, &format!("Expand({x})"), "oval"),
                Node::Cast(x, dtype) => add_node(id, &format!("Cast({x} -> {dtype})"), "oval"),
                Node::Reshape(x, ..) => add_node(id, &format!("Reshape({x})"), "oval"),
                Node::Permute(x, axes, ..) => {
                    add_node(id, &format!("Permute({x}, axes {axes})"), "oval");
                }
                Node::Sum(x, axes, ..) => add_node(id, &format!("Sum({x}, axes {axes})"), "oval"),
                Node::Max(x, axes, ..) => add_node(id, &format!("Max({x}, axes {axes})"), "oval"),
            }
            for param in &*node.parameters() {
                writeln!(edges, "  {} -> {id}", param.i()).unwrap();
            }
        }
        res = res.replace("NL", "\n");
        write!(res, "{edges}}}").unwrap();
        res
    }
}

// Boring boilerplate methods
impl Graph {
    #[cfg(feature = "rand")]
    pub(super) fn randn_f32(&mut self, shape: Shape) -> NodeId {
        use rand::Rng;
        let n = shape.numel();
        let node = Node::StoreF32(
            (0..n)
                .map(|_| self.rng.sample(rand::distributions::Standard))
                .collect(),
            shape,
        );
        self.push(node)
    }

    #[cfg(feature = "rand")]
    pub(super) fn randn_i32(&mut self, shape: Shape) -> NodeId {
        use rand::Rng;
        let n = shape.numel();
        let node = Node::StoreI32(
            (0..n)
                .map(|_| self.rng.sample(rand::distributions::Standard))
                .collect(),
            shape,
        );
        self.push(node)
    }

    #[cfg(feature = "rand")]
    pub(super) fn uniform_f32(&mut self, shape: Shape, range: core::ops::Range<f32>) -> NodeId {
        use rand::Rng;
        let n = shape.numel();
        let dist = rand::distributions::Uniform::from(range);
        let node = Node::StoreF32((0..n).map(|_| self.rng.sample(dist)).collect(), shape);
        self.push(node)
    }

    #[cfg(feature = "rand")]
    pub(super) fn uniform_i32(&mut self, shape: Shape, range: core::ops::Range<i32>) -> NodeId {
        use rand::Rng;
        let n = shape.numel();
        let dist = rand::distributions::Uniform::from(range);
        let node = Node::StoreI32((0..n).map(|_| self.rng.sample(dist)).collect(), shape);
        self.push(node)
    }

    pub(super) fn tensor_from_iter_f32(
        &mut self,
        shape: Shape,
        iter: impl IntoIterator<Item = f32>,
    ) -> NodeId {
        let n = shape.numel();
        let node = Node::StoreF32(iter.into_iter().take(n).collect(), shape);
        self.push(node)
    }

    pub(super) fn tensor_from_iter_i32(
        &mut self,
        shape: Shape,
        iter: impl IntoIterator<Item = i32>,
    ) -> NodeId {
        let n = shape.numel();
        let node = Node::StoreI32(iter.into_iter().take(n).collect(), shape);
        self.push(node)
    }
}