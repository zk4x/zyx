use crate::autograd::Autograd;
use crate::axes::Axes;
use crate::backend::BufferView;
use crate::dtype::DType;
use crate::node::Node;
use crate::scalar::Scalar;
use crate::shape::Shape;
use crate::tensor::{id, Id};
use alloc::boxed::Box;
use alloc::collections::{BTreeMap, BTreeSet};
use alloc::vec::Vec;

pub trait Runtime {
    type Buffer;
    type Program;
    fn store<T>(&mut self, iter: Box<dyn Iterator<Item = T>>) -> Self::Buffer;
    fn load<T>(&mut self, buffer: &Self::Buffer) -> BufferView;
    fn compile(&mut self, kernel: OpKernel) -> Self::Program;
    fn launch(&mut self, program: &Self::Program, args: &[&Self::Buffer]) -> Self::Buffer;
}

pub struct CompiledBackend<R: Runtime> {
    rng: rand::rngs::SmallRng,
    rcs: Vec<u8>,
    order: Vec<Id>,
    nodes: Vec<Node>,
    leafs: BTreeSet<Id>, // these do not need backward graph
    runtime: R,
    buffers: BTreeMap<Id, R::Buffer>,
    programs: BTreeMap<OpKernel, R::Program>,
}

// These are all IDs into ops, leafs have IDs into args
#[derive(PartialEq, Eq, PartialOrd, Ord)]
enum Op {
    Leaf(usize),
    CastF32(usize),
    Exp(usize),
    Add(usize, usize),
    Reshape(Shape),
    Expand(Shape),
    Permute(Axes),
    //Pad(Box<[(i64, i64)]>),
    Sum(Axes),
    Max(Axes),
}

#[derive(PartialEq, Eq, PartialOrd, Ord)]
pub struct OpKernel {
    args: Box<[(Shape, DType)]>,
    ops: Box<[Op]>,
}

impl<R: Runtime> CompiledBackend<R> {
    pub fn new(runtime: R) -> Self {
        use rand::SeedableRng;
        Self {
            rng: rand::rngs::SmallRng::seed_from_u64(420_694_206_942_069),
            rcs: Vec::new(),
            order: Vec::new(),
            nodes: Vec::new(),
            leafs: BTreeSet::new(),
            runtime,
            buffers: BTreeMap::new(),
            programs: BTreeMap::new(),
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

    pub fn load(&mut self, x: Id) -> BufferView {
        // This may need to evaluate, therefore we need to take mutable reference to self
        if !self.buffers.contains_key(&x) {
            // TODO also check if these are only movements ops,
            // in which case we can directly return iterator with view
            self.evaluate(BTreeSet::from([x]));
        }
        todo!()
    }

    pub fn set_leaf(&mut self, x: Id) {
        self.leafs.insert(x);
    }

    fn evaluate(&mut self, nodes: BTreeSet<Id>) {
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

        for nid in order {
            match &mut self.nodes[nid.i()] {
                Node::LeafF32(..)
                | Node::LeafI32(..)
                | Node::UniformF32(..)
                | Node::UniformI32(..)
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
                | Node::Permute(..)
                | Node::Sum(..)
                | Node::Max(..) => {}
                Node::IterF32(_, shape) => {
                    let mut new_node = Node::LeafF32(shape.clone());
                    core::mem::swap(&mut self.nodes[nid.i()], &mut new_node);
                    if let Node::IterF32(iter, _) = new_node {
                        self.runtime.store(iter);
                    }
                }
                Node::IterI32(_, shape) => {
                    let mut new_node = Node::LeafI32(shape.clone());
                    core::mem::swap(&mut self.nodes[nid.i()], &mut new_node);
                    if let Node::IterI32(iter, _) = new_node {
                        self.runtime.store(iter);
                    }
                }
                Node::Expand(x, _) => {
                    // if reduce operation preceded expand, we call evaluate_buffer
                    let mut params = alloc::vec![*x];
                    while let Some(p) = params.pop() {
                        // TODO check that there is no more than one reduce
                        if matches!(self.nodes[p.i()], Node::Sum(..) | Node::Max(..)) {
                            self.evaluate_buffer(p);
                            break;
                        }
                        params.extend(self.nodes[p.i()].parameters());
                    }
                }
            }
            // TODO release nodes that are no longer needed.
            // And release intermediate buffers.
        }

        // Release parts of graph that are not needed for backpropagation
        //while let Some(leaf) = self.leafs.pop_last() {
        //std::println!("Releasing leaf {leaf}");
        //let mut node = Node::Leaf(self.dtype(leaf));
        //let shape = self.shape(leaf);
        //self.shapes.insert(leaf, shape);
        //core::mem::swap(self.nodes.get_mut(leaf.i()).unwrap(), &mut node);
        //for nid in &*node.parameters() {
        //self.release(*nid);
        //}
        //}
    }

    /// This function evaluates concrete buffer that we know can be directly evaluated,
    /// that is it all of it's leafs are already evaluated.
    fn evaluate_buffer(&mut self, x: Id) {
        // create ordered list of nodes that need to be evaluated
        let mut temp = alloc::vec![x];
        let mut porder = Vec::new();
        while let Some(nid) = temp.pop() {
            if self.buffers.contains_key(&nid) {
                continue;
            }
            porder.extend(self.nodes[nid.i()].parameters())
        }
        porder.sort_by_cached_key(|nid| self.order[nid.i()]);

        // Convert these to Kernel
        let mut args = Vec::new();
        //let mut kernel_args = Vec::new();
        for nid in porder {
            if let Some(x) = self.buffers.get(&nid) {
                args.push((self.shape(nid), self.dtype(nid)));
                //kernel_args.push(x.mem);
            } else {
                todo!()
            }
        }
        todo!()
    }
}

impl<C: Runtime> Autograd for CompiledBackend<C> {
    fn nodes(&self) -> &[Node] {
        &self.nodes
    }

    fn order(&self) -> &[Id] {
        &self.order
    }

    fn shape(&self, mut x: Id) -> &Shape {
        loop {
            let node = self.nodes.get(x.i()).unwrap();
            match node {
                Node::LeafF32(shape)
                | Node::IterF32(_, shape)
                | Node::UniformF32(shape, ..)
                | Node::LeafI32(shape)
                | Node::IterI32(_, shape)
                | Node::UniformI32(shape, ..)
                | Node::Reshape(_, shape)
                | Node::Expand(_, shape)
                | Node::Permute(.., shape)
                | Node::Sum(.., shape)
                | Node::Max(.., shape) => return shape,
                _ => x = node.parameters().next().unwrap(),
            }
        }
    }

    fn dtype(&self, mut x: Id) -> DType {
        loop {
            let node = self.nodes.get(x.i()).unwrap();
            match node {
                Node::LeafF32(..)
                | Node::IterF32(..)
                | Node::UniformF32(..)
                | Node::CastF32(..) => return DType::F32,
                Node::LeafI32(..)
                | Node::IterI32(..)
                | Node::UniformI32(..)
                | Node::CastI32(..) => return DType::I32,
                _ => x = node.parameters().next().unwrap(),
            }
        }
    }

    fn push(&mut self, node: Node) -> Id {
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

    fn release(&mut self, x: Id) {
        let mut params = Vec::with_capacity(10);
        params.push(x);
        while let Some(p) = params.pop() {
            self.rcs[p.i()] -= 1;
            if self.rcs[p.i()] == 0 {
                params.extend(self.nodes[p.i()].parameters());
                self.leafs.remove(&p);
                self.buffers.remove(&p);
            }
        }
    }

    fn retain(&mut self, x: Id) {
        self.rcs[x.i()] += 1;
    }
}
