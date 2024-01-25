use crate::error::ZyxError;
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

/// Implement this trait for compiled backends
pub trait Compiler {
    /// Buffer holds actual values in memory
    type Buffer;
    /// Program is kernel executable on the device, can be compiled at runtime
    type Program;
    /// Store iter into buffer
    fn store<T: Scalar>(&mut self, iter: impl Iterator<Item = T>)
        -> Result<Self::Buffer, ZyxError>;
    /// Load buffer into vec
    fn load<T: Scalar>(&mut self, buffer: &Self::Buffer, numel: usize) -> Result<Vec<T>, ZyxError>;
    /// Drop Buffer
    fn drop_buffer(&mut self, buffer: &mut Self::Buffer) -> Result<(), ZyxError>;
    /// Drop Program
    fn drop_program(&mut self, program: &mut Self::Program) -> Result<(), ZyxError>;
    /// Launch program with args
    fn launch(
        &mut self,
        program: &Self::Program,
        args: &[&Self::Buffer],
    ) -> Result<Self::Buffer, ZyxError>;
    /// Compile ast into program
    fn compile(&mut self, ast: &AST) -> Result<Self::Program, ZyxError>;
}

/// Op executable on device with compiled backend
/// usize are all IDs into ops, leafs have IDs into args
#[derive(Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Op {
    /// Leaf (holds data, id to kernel arg)
    Leaf(usize),
    // TODO uniform generators should also take shape into consideration
    // and repeat the same random number if this shape is expanded.
    /// Shaped uniform F32 generator of numbers from 0. to 1.
    UniformF32(View),
    /// Cast into F32 unary op
    CastF32(usize),
    /// Cast into I32 unary op
    CastI32(usize),
    /// Neg unary op
    Neg(usize),
    /// ReLU unary op
    ReLU(usize),
    /// Sin unary op
    Sin(usize),
    /// Cos unary op
    Cos(usize),
    /// Ln unary op
    Ln(usize),
    /// Exp unary op
    Exp(usize),
    /// Tanh unary op
    Tanh(usize),
    /// Sqrt unary op
    Sqrt(usize),
    /// Addition binary op
    Add(usize, usize),
    /// Substitution binary op
    Sub(usize, usize),
    /// Multiplication binary op
    Mul(usize, usize),
    /// Division binary op
    Div(usize, usize),
    /// Exponentiation binary op
    Pow(usize, usize),
    /// Compare less than binary op
    Cmplt(usize, usize),
}

/// Possible reduce ops
#[derive(Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum ROp {
    /// No reduce
    None,
    /// Sum reduce op
    Sum,
    /// Max reduce op
    Max,
}

/// Abstract syntax tree that can be compiled into program.
/// Consists of kernel arguments, elementwise ops, optional reduce op
/// and elementwise ops after reduce.
#[derive(Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct AST {
    args: Box<[(View, DType)]>,
    ops: Box<[Op]>,
    reduce: ROp,
    ar_ops: Box<[Op]>,
}

impl AST {
    /// Get AST arguments metadata
    pub fn args(&self) -> &[(View, DType)] {
        &self.args
    }

    /// Get AST ops
    pub fn ops(&self) -> &[Op] {
        &self.ops
    }

    /// Get AST reduce op
    pub fn reduce(&self) -> &ROp {
        &self.reduce
    }

    /// Ops applied after reduce
    pub fn ar_ops(&self) -> &[Op] {
        &self.ar_ops
    }
}

/// Compiled backend that holds compiler, buffers and programs
pub struct CompiledBackend<C: Compiler> {
    compiler: C,
    buffers: BTreeMap<Id, C::Buffer>,
    programs: BTreeMap<AST, C::Program>,
}

impl<C: Compiler> RuntimeBackend for CompiledBackend<C> {
    fn is_evaluated(&self, x: Id) -> bool {
        self.buffers.contains_key(&x)
    }

    fn remove(&mut self, x: Id) -> Result<(), ZyxError> {
        if let Some(mut buf) = self.buffers.remove(&x) {
            self.compiler.drop_buffer(&mut buf)?;
        }
        Ok(())
    }

    fn load<T: Scalar>(&mut self, x: Id, numel: usize) -> Result<Vec<T>, ZyxError> {
        self.compiler.load(&self.buffers[&x], numel)
    }

    fn evaluate(
        &mut self,
        to_eval: BTreeSet<Id>,
        order: &[Id],
        nodes: &mut [Node],
    ) -> Result<(), ZyxError> {
        for nid in order.iter().copied() {
            match &mut nodes[nid.i()] {
                Node::LeafF32(..)
                | Node::LeafI32(..)
                | Node::UniformF32(..)
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
                | Node::Pad(..)
                | Node::Sum(..)
                | Node::Max(..) => {}
                Node::IterF32(_, shape) => {
                    let mut new_node = Node::LeafF32(shape.clone());
                    core::mem::swap(&mut nodes[nid.i()], &mut new_node);
                    if let Node::IterF32(iter, _) = new_node {
                        self.buffers.insert(nid, self.compiler.store(iter)?);
                    }
                }
                Node::IterI32(_, shape) => {
                    let mut new_node = Node::LeafI32(shape.clone());
                    core::mem::swap(&mut nodes[nid.i()], &mut new_node);
                    if let Node::IterI32(iter, _) = new_node {
                        self.buffers.insert(nid, self.compiler.store(iter)?);
                    }
                }
                Node::Expand(x, _) => {
                    // if reduce operation preceded expand, we call evaluate_buffer
                    let mut params = alloc::vec![*x];
                    while let Some(p) = params.pop() {
                        // TODO check that there is no more than one reduce
                        if matches!(nodes[p.i()], Node::Sum(..) | Node::Max(..)) {
                            self.evaluate_buffer(p, order, nodes)?;
                            break;
                        }
                        params.extend(nodes[p.i()].parameters());
                    }
                }
            }
            if to_eval.contains(&nid) && !self.buffers.contains_key(&nid) {
                self.evaluate_buffer(nid, order, nodes)?;
            }
            // TODO release nodes that are no longer needed.
            // And release intermediate buffers.
        }
        Ok(())
    }
}

impl<C: Compiler> CompiledBackend<C> {
    /// Initialize new compiled backend using provided compiler
    pub fn new(compiler: C) -> Self {
        Self {
            compiler,
            buffers: BTreeMap::new(),
            programs: BTreeMap::new(),
        }
    }

    /// This function evaluates concrete buffer that we know can be directly evaluated,
    /// that is we know that all of it's leafs are already evaluated and stored in device.
    fn evaluate_buffer(&mut self, x: Id, order: &[Id], nodes: &[Node]) -> Result<(), ZyxError> {
        // Create ordered list of nodes that need to be evaluated
        let mut temp = alloc::vec![x];
        let mut porder = Vec::new();
        while let Some(nid) = temp.pop() {
            porder.push(nid);
            if self.buffers.contains_key(&nid) {
                continue;
            }
            temp.extend(nodes[nid.i()].parameters());
        }
        porder.sort_by_cached_key(|nid| order[nid.i()]);
        // Convert this list to kernel
        let mut program_args = Vec::new();
        let mut args = Vec::new();
        let mut ops = Vec::new();
        let mut ar_ops = Vec::new();
        let ar = false;
        let reduce = ROp::None;
        let mut mapping = BTreeMap::new();
        for nid in porder {
            mapping.insert(nid, ops.len());
            if let Some(x) = self.buffers.get(&nid) {
                args.push((View::new(shape(&nodes, nid).clone()), dtype(&nodes, nid)));
                program_args.push(x);
                if ar {
                    ar_ops.push(Op::Leaf(args.len() - 1));
                } else {
                    ops.push(Op::Leaf(args.len() - 1));
                }
            } else {
                match &nodes[nid.i()] {
                    Node::LeafF32(..) | Node::LeafI32(..) | Node::IterF32(..) | Node::IterI32(..) => {}
                    Node::UniformF32(sh) => {
                        if ar { &mut ar_ops } else { &mut ops }.push(Op::UniformF32(View::new(sh.clone())));
                    }
                    Node::CastF32(x) => {
                        if ar { &mut ar_ops } else { &mut ops }.push(Op::CastF32(mapping[x]));
                    }
                    Node::CastI32(x) => {
                        if ar { &mut ar_ops } else { &mut ops }.push(Op::CastI32(mapping[x]));
                    }
                    Node::Neg(x) => {
                        if ar { &mut ar_ops } else { &mut ops }.push(Op::Neg(mapping[x]));
                    }
                    Node::ReLU(x) => {
                        if ar { &mut ar_ops } else { &mut ops }.push(Op::ReLU(mapping[x]));
                    }
                    Node::Sin(x) => {
                        if ar { &mut ar_ops } else { &mut ops }.push(Op::Sin(mapping[x]));
                    }
                    Node::Cos(x) => {
                        if ar { &mut ar_ops } else { &mut ops }.push(Op::Cos(mapping[x]));
                    }
                    Node::Ln(x) => {
                        if ar { &mut ar_ops } else { &mut ops }.push(Op::Ln(mapping[x]));
                    }
                    Node::Exp(x) => {
                        if ar { &mut ar_ops } else { &mut ops }.push(Op::Exp(mapping[x]));
                    }
                    Node::Tanh(x) => {
                        if ar { &mut ar_ops } else { &mut ops }.push(Op::Tanh(mapping[x]));
                    }
                    Node::Sqrt(x) => {
                        if ar { &mut ar_ops } else { &mut ops }.push(Op::Sqrt(mapping[x]));
                    }
                    // TODO also apply movement ops to UniformF32
                    Node::Expand(x, sh) => {
                        let mut params = alloc::vec![*x];
                        while let Some(p) = params.pop() {
                            if let Op::Leaf(a) = ops[p.i()] {
                                args[a].0 = args[a].0.expand(sh);
                            }
                            params.extend(nodes[x.i()].parameters());
                        }
                    }
                    Node::Reshape(x, sh) => {
                        let mut params = alloc::vec![*x];
                        while let Some(p) = params.pop() {
                            if let Op::Leaf(a) = ops[p.i()] {
                                args[a].0 = args[a].0.reshape(sh);
                            }
                            params.extend(nodes[x.i()].parameters());
                        }
                    }
                    Node::Pad(x, padding) => {
                        let mut params = alloc::vec![*x];
                        while let Some(p) = params.pop() {
                            if let Op::Leaf(a) = ops[p.i()] {
                                args[a].0 = args[a].0.pad(padding);
                            }
                            params.extend(nodes[x.i()].parameters());
                        }
                    }
                    Node::Permute(x, axes, _) => {
                        let mut params = alloc::vec![*x];
                        while let Some(p) = params.pop() {
                            if let Op::Leaf(a) = ops[p.i()] {
                                args[a].0 = args[a].0.permute(axes);
                            }
                            params.extend(nodes[x.i()].parameters());
                        }
                    }
                    Node::Add(x, y) => {
                        if ar { &mut ar_ops } else { &mut ops }
                            .push(Op::Add(mapping[x], mapping[y]));
                    }
                    Node::Sub(x, y) => {
                        if ar { &mut ar_ops } else { &mut ops }
                            .push(Op::Sub(mapping[x], mapping[y]));
                    }
                    Node::Mul(x, y) => {
                        if ar { &mut ar_ops } else { &mut ops }
                            .push(Op::Mul(mapping[x], mapping[y]));
                    }
                    Node::Div(x, y) => {
                        if ar { &mut ar_ops } else { &mut ops }
                            .push(Op::Div(mapping[x], mapping[y]));
                    }
                    Node::Pow(x, y) => {
                        if ar { &mut ar_ops } else { &mut ops }
                            .push(Op::Pow(mapping[x], mapping[y]));
                    }
                    Node::Cmplt(x, y) => {
                        if ar { &mut ar_ops } else { &mut ops }
                            .push(Op::Cmplt(mapping[x], mapping[y]));
                    }
                    Node::Sum(x, axes, rshape) => {
                        todo!()
                    }
                    Node::Max(x, axes, rshape) => {
                        todo!()
                    }
                }
            };
        }
        if matches!(reduce, ROp::None) {
            for (view, _) in &mut args {
                let n = view.numel();
                *view = view
                    .reshape(&n.into())
                    .pad(&[(0, if n % 8 != 0 { (8 - n % 8) as i64 } else { 0 })]);
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
            let program = self.compiler.compile(&ast)?;
            self.programs.entry(ast).or_insert(program)
        };
        //panic!("Number of args: {}", program_args.len());
        // Run the program
        self.buffers
            .insert(x, self.compiler.launch(program, &program_args)?);
        Ok(())
    }
}
