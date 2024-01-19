use crate::{
    dtype::DType,
    node::Node,
    runtime::RuntimeBackend,
    scalar::Scalar,
    tensor::Id,
    utils::{dtype, shape},
    view::View,
};
use alloc::{
    boxed::Box,
    collections::{BTreeMap, BTreeSet},
    vec::Vec,
};

pub trait Compiler {
    type Buffer;
    type Program;
    fn store<T: Scalar>(&mut self, iter: impl Iterator<Item = T>) -> Self::Buffer;
    fn load<T: Scalar>(&mut self, buffer: &Self::Buffer, numel: usize) -> Vec<T>;
    fn launch(&mut self, program: &Self::Program, args: &[&Self::Buffer]) -> Self::Buffer;
    fn compile(&mut self, ast: &AST) -> Self::Program;
}

// These are all IDs into ops, leafs have IDs into args
#[derive(PartialEq, Eq, PartialOrd, Ord)]
pub enum Op {
    Leaf(usize),
    CastF32(usize),
    Exp(usize),
    Add(usize, usize),
}

#[derive(PartialEq, Eq, PartialOrd, Ord)]
pub enum ROp {
    None,
    Sum,
    Max,
}

#[derive(PartialEq, Eq, PartialOrd, Ord)]
pub struct AST {
    args: Box<[(View, DType)]>,
    ops: Box<[Op]>,
    reduce: ROp,
    ar_ops: Box<[Op]>,
}

impl AST {
    pub fn args(&self) -> &[(View, DType)] {
        &self.args
    }

    pub fn ops(&self) -> &[Op] {
        &self.ops
    }

    pub fn reduce(&self) -> &ROp {
        &self.reduce
    }

    // Ops applied after reduce
    pub fn ar_ops(&self) -> &[Op] {
        &self.ar_ops
    }
}

pub struct CompiledBackend<C: Compiler> {
    compiler: C,
    buffers: BTreeMap<Id, C::Buffer>,
    programs: BTreeMap<AST, C::Program>,
}

impl<C: Compiler> RuntimeBackend for CompiledBackend<C> {
    fn is_evaluated(&self, x: Id) -> bool {
        self.buffers.contains_key(&x)
    }

    fn remove(&mut self, x: Id) {
        self.buffers.remove(&x);
    }

    fn load<T: Scalar>(&mut self, x: Id, numel: usize) -> Vec<T> {
        self.compiler.load(&self.buffers[&x], numel)
    }

    fn evaluate(&mut self, to_eval: BTreeSet<Id>, order: &[Id], nodes: &mut [Node]) {
        for nid in order.iter().copied() {
            match &mut nodes[nid.i()] {
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
                    core::mem::swap(&mut nodes[nid.i()], &mut new_node);
                    if let Node::IterF32(iter, _) = new_node {
                        self.buffers.insert(nid, self.compiler.store(iter));
                    }
                }
                Node::IterI32(_, shape) => {
                    let mut new_node = Node::LeafI32(shape.clone());
                    core::mem::swap(&mut nodes[nid.i()], &mut new_node);
                    if let Node::IterI32(iter, _) = new_node {
                        self.buffers.insert(nid, self.compiler.store(iter));
                    }
                }
                Node::Expand(x, _) => {
                    // if reduce operation preceded expand, we call evaluate_buffer
                    let mut params = alloc::vec![*x];
                    while let Some(p) = params.pop() {
                        // TODO check that there is no more than one reduce
                        if matches!(nodes[p.i()], Node::Sum(..) | Node::Max(..)) {
                            self.evaluate_buffer(p, order, nodes);
                            break;
                        }
                        params.extend(nodes[p.i()].parameters());
                    }
                }
            }
            if to_eval.contains(&nid) && !self.buffers.contains_key(&nid) {
                self.evaluate_buffer(nid, order, nodes);
            }
            // TODO release nodes that are no longer needed.
            // And release intermediate buffers.
        }
    }
}

impl<C: Compiler> CompiledBackend<C> {
    pub fn new(compiler: C) -> Self {
        Self {
            compiler,
            buffers: BTreeMap::new(),
            programs: BTreeMap::new(),
        }
    }

    /// This function evaluates concrete buffer that we know can be directly evaluated,
    /// that is we know that all of it's leafs are already evaluated and stored in device.
    fn evaluate_buffer(&mut self, x: Id, order: &[Id], nodes: &[Node]) {
        // Create ordered list of nodes that need to be evaluated
        let mut temp = alloc::vec![x];
        let mut porder = Vec::new();
        while let Some(nid) = temp.pop() {
            if self.buffers.contains_key(&nid) {
                continue;
            }
            porder.extend(nodes[nid.i()].parameters())
        }
        porder.sort_by_cached_key(|nid| order[nid.i()]);
        // Convert this list to kernel
        let mut program_args = Vec::new();
        let mut args = Vec::new();
        let mut ops = Vec::new();
        let mut ar_ops = Vec::new();
        let mut after_reduce = false;
        let mut reduce = ROp::None;
        let mut mapping = BTreeMap::new();
        for nid in porder {
            mapping.insert(nid, ops.len());
            if let Some(x) = self.buffers.get(&nid) {
                args.push((View::new(shape(&nodes, nid).clone()), dtype(&nodes, nid)));
                program_args.push(x);
                if after_reduce {
                    ar_ops.push(Op::Leaf(args.len() - 1));
                } else {
                    ops.push(Op::Leaf(args.len() - 1));
                }
            } else {
                match &nodes[nid.i()] {
                    Node::IterF32(..) | Node::IterI32(..) => {},
                    Node::Exp(x) => {
                        if after_reduce {
                            ar_ops.push(Op::Exp(mapping[x]));
                        } else {
                            ops.push(Op::Exp(mapping[x]));
                        }
                    }
                    Node::Expand(x, sh) => {
                        let mut params = alloc::vec![*x];
                        while let Some(p) = params.pop() {
                            if let Op::Leaf(a) = ops[p.i()] {
                                args[a].0.expand(sh);
                            }
                            params.extend(nodes[x.i()].parameters());
                        }
                    }
                    _ => todo!("Op is not implemented yet."),
                }
            };
        }
        if matches!(reduce, ROp::None) {
            for (view, _) in &mut args {
                *view = view.reshape(&view.numel().into());
            }
        }
        let ast = AST {
            args: args.into_boxed_slice(),
            ops: ops.into_boxed_slice(),
            reduce,
            ar_ops: Box::new([]),
        };
        // Used cached program or compile new program
        let program = if let Some(program) = self.programs.get(&ast) {
            program
        } else {
            let program = self.compiler.compile(&ast);
            self.programs.entry(ast).or_insert(program)
        };
        // Run the program
        self.buffers
            .insert(x, self.compiler.launch(program, &program_args));
    }
}
