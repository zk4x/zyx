use crate::axes::Axes;
use crate::error::ZyxError;
use crate::{
    dtype::DType,
    node::Node,
    runtime::RuntimeBackend,
    scalar::Scalar,
    tensor::Id,
    utils::{get_dtype, get_shape},
    view::View,
};
use alloc::{
    boxed::Box,
    collections::{btree_map::Entry, BTreeMap, BTreeSet},
    vec::Vec,
};
use std::println;
use crate::shape::Shape;

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
    /// Where op
    Where(usize, usize, usize),
    /// Sum reduce op
    Sum(usize),
    /// Max reduce op
    Max(usize),
}

impl Op {
    fn access_parameters(&self, mut op: impl FnMut(usize)) {
        match self {
            Op::Leaf(..) | Op::UniformF32(..) => {}
            Op::CastF32(x)
            | Op::CastI32(x)
            | Op::Neg(x)
            | Op::ReLU(x)
            | Op::Sin(x)
            | Op::Cos(x)
            | Op::Ln(x)
            | Op::Exp(x)
            | Op::Tanh(x)
            | Op::Sqrt(x)
            | Op::Sum(x)
            | Op::Max(x) => op(*x),
            Op::Add(x, y)
            | Op::Sub(x, y)
            | Op::Mul(x, y)
            | Op::Div(x, y)
            | Op::Pow(x, y)
            | Op::Cmplt(x, y) => {
                op(*x);
                op(*y);
            }
            Op::Where(x, y, z) => {
                op(*x);
                op(*y);
                op(*z);
            }
        }
    }
}

/// Abstract syntax tree that can be compiled into program.
/// Consists of kernel arguments, elementwise ops, optional reduce op
/// and elementwise ops after reduce.
/// This struct is immutable.
#[derive(Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct AST {
    /// Get AST arguments metadata
    pub args: Box<[(View, DType)]>,
    /// Get AST ops
    pub ops: Box<[Op]>,
    /// View of the result, for reduce kernel, this is after the reduce op
    pub view: View,
    /// Reduce dimension size, if any
    pub rdim: Option<usize>,
    /// Operations to execute this AST
    pub flop: usize,
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
        mut rcs: BTreeMap<Id, u8>,
        order: &[Id],
        nodes: &mut [Node],
    ) -> Result<(), ZyxError> {
        for (i, nid) in order.iter().copied().enumerate() {
            println!("Node {:?}", nodes[nid.i()]);
            if to_eval.contains(&nid) {
                self.evaluate_buffer(nid, order, nodes)?;
            }
            match &mut nodes[nid.i()] {
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
                | Node::Where(..)
                | Node::Reshape(..)
                | Node::Permute(..)
                | Node::Expand(..)
                | Node::Pad(..) => {}
                Node::Sum(..)
                | Node::Max(..) => {
                    // Search forward for first expand, in to_eval or rc > 1, then evaluate that.
                    // That way we fuse as much as possible :)
                    let mut p = nid;
                    for x in &order[i..] {
                        let node = &nodes[x.i()];
                        if node.parameters_contain(p) {
                            if matches!(node, Node::Expand(..)) {
                                self.evaluate_buffer(p, order, nodes)?;
                            }
                            if to_eval.contains(x) || rcs[x] > 1 {
                                self.evaluate_buffer(*x, order, nodes)?;
                            }
                            p = *x;
                        }
                    }
                }
            }
            for p in nodes[nid.i()].parameters() {
                if let Entry::Occupied(e) = rcs.entry(p).and_modify(|rc| *rc -= 1) {
                    if *e.get() == 0 {
                        self.remove(p)?;
                    }
                }
            }
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
        //println!("Evaluating buffer {x}");
        if self.is_evaluated(x) {
            return Ok(())
        }
        // Create ordered list of nodes that need to be evaluated
        //extern crate std;
        //println!("x: {x}\norder: {order:?}\nnodes: {nodes:?}");
        let mut reduce_nid = None;
        let mut temp = alloc::vec![x];
        let mut porder = BTreeSet::new();
        while let Some(nid) = temp.pop() {
            porder.insert(nid);
            if self.is_evaluated(nid) {
                continue;
            }
            if matches!(nodes[nid.i()], Node::Sum(..) | Node::Max(..)) {
                reduce_nid = Some(nid);
            }
            temp.extend(nodes[nid.i()].parameters());
        }
        /*let mut br_leafs = Vec::new();
        if let Some(reduce_nid) = reduce_nid {
            let mut temp = alloc::vec![reduce_nid];
            while let Some(nid) = temp.pop() {
                if self.is_evaluated(nid) {
                    br_leafs.push(nid);
                    continue;
                }
                temp.extend(nodes[nid.i()].parameters());
            }
        }*/
        let p_order: Vec<Id> = order
            .iter()
            .copied()
            .filter(|nid| porder.contains(nid))
            .collect();
        let porder: Vec<Id> = porder
            .iter()
            .copied()
            .filter(|nid| !p_order.contains(nid))
            .chain(p_order.clone())
            .collect();
        //std::println!("porder {porder:?}");
        // Convert this list to kernel
        let mut program_args = Vec::new();
        let mut args = Vec::new();
        let mut ar = false;
        let mut first_ar_arg = usize::MAX;
        let mut ops = Vec::new();
        let mut raxes = None;
        let mut mapping = BTreeMap::new();
        let mut rshape = [].into();
        let mut flop = 0;
        // TODO don't load the same buffer twice if used in different positions unless it
        // uses different views for those loads, this can be done as deduplication in the end
        // of this function.
        //println!();
        // TODO reorder in such a way, that first are all before reduce loads,
        // then before reduce ops, then after reduce loads, then after reduce ops.
        // If it is not a reduce kernel, then all leafs should be first.
        for nid in porder {
            //println!("ops: {ops:?}");
            flop += nodes[nid.i()].flop(nodes);
            let mut mapped = true;
            if let Some(x) = self.buffers.get(&nid) {
                if ar {
                    if first_ar_arg != usize::MAX {
                        first_ar_arg = args.len();
                    }
                }
                args.push((
                    View::new(get_shape(nodes, nid).clone()),
                    get_dtype(nodes, nid),
                ));
                program_args.push(x);
                ops.push(Op::Leaf(args.len() - 1));
            } else {
                match &nodes[nid.i()] {
                    Node::LeafF32(..)
                    | Node::LeafI32(..)
                    | Node::IterF32(..)
                    | Node::IterI32(..) => {}
                    Node::UniformF32(sh) => ops.push(Op::UniformF32(View::new(sh.clone()))),
                    Node::CastF32(x) => ops.push(Op::CastF32(mapping[x])),
                    Node::CastI32(x) => ops.push(Op::CastI32(mapping[x])),
                    Node::Neg(x) => ops.push(Op::Neg(mapping[x])),
                    Node::ReLU(x) => ops.push(Op::ReLU(mapping[x])),
                    Node::Sin(x) => ops.push(Op::Sin(mapping[x])),
                    Node::Cos(x) => ops.push(Op::Cos(mapping[x])),
                    Node::Ln(x) => ops.push(Op::Ln(mapping[x])),
                    Node::Exp(x) => ops.push(Op::Exp(mapping[x])),
                    Node::Tanh(x) => ops.push(Op::Tanh(mapping[x])),
                    Node::Sqrt(x) => ops.push(Op::Sqrt(mapping[x])),
                    // TODO also apply movement ops to UniformF32
                    Node::Expand(x, sh) => {
                        mapped = false;
                        mapping.insert(nid, mapping[x]);
                        let mut params = alloc::vec![mapping[x]];
                        while let Some(p) = params.pop() {
                            if let Op::Leaf(a) = ops[p] {
                                if ar {
                                    if a >= first_ar_arg {
                                        args[a].0 = args[a].0.expand(sh);
                                    }
                                } else {
                                    if a < first_ar_arg {
                                        args[a].0 = args[a].0.expand(sh);
                                    }
                                }
                            }
                            ops[p].access_parameters(|x| params.push(x));
                        }
                    }
                    Node::Reshape(x, sh) => {
                        mapped = false;
                        mapping.insert(nid, mapping[x]);
                        let mut params = alloc::vec![mapping[x]];
                        while let Some(p) = params.pop() {
                            if let Op::Leaf(a) = ops[p] {
                                if ar {
                                    if a >= first_ar_arg {
                                        args[a].0 = args[a].0.reshape(sh);
                                    }
                                } else {
                                    if a < first_ar_arg {
                                        args[a].0 = args[a].0.reshape(sh);
                                    }
                                }
                            }
                            ops[p].access_parameters(|x| params.push(x));
                        }
                    }
                    Node::Pad(x, padding, _) => {
                        mapped = false;
                        mapping.insert(nid, mapping[x]);
                        let mut params = alloc::vec![mapping[x]];
                        while let Some(p) = params.pop() {
                            if let Op::Leaf(a) = ops[p] {
                                if ar {
                                    if a >= first_ar_arg {
                                        args[a].0 = args[a].0.pad(padding);
                                    }
                                } else {
                                    if a < first_ar_arg {
                                        args[a].0 = args[a].0.pad(padding);
                                    }
                                }
                            }
                            ops[p].access_parameters(|x| params.push(x));
                        }
                    }
                    Node::Permute(x, axes, _) => {
                        mapped = false;
                        mapping.insert(nid, mapping[x]);
                        let mut params = alloc::vec![mapping[x]];
                        while let Some(p) = params.pop() {
                            if let Op::Leaf(a) = ops[p] {
                                if ar {
                                    if a >= first_ar_arg {
                                        args[a].0 = args[a].0.permute(axes);
                                    }
                                } else {
                                    if a < first_ar_arg {
                                        args[a].0 = args[a].0.permute(axes);
                                    }
                                }
                            }
                            ops[p].access_parameters(|x| params.push(x));
                        }
                    }
                    Node::Add(x, y) => ops.push(Op::Add(mapping[x], mapping[y])),
                    Node::Sub(x, y) => ops.push(Op::Sub(mapping[x], mapping[y])),
                    Node::Mul(x, y) => ops.push(Op::Mul(mapping[x], mapping[y])),
                    Node::Div(x, y) => ops.push(Op::Div(mapping[x], mapping[y])),
                    Node::Pow(x, y) => ops.push(Op::Pow(mapping[x], mapping[y])),
                    Node::Cmplt(x, y) => ops.push(Op::Cmplt(mapping[x], mapping[y])),
                    Node::Where(x, y, z) => ops.push(Op::Where(mapping[x], mapping[y], mapping[z])),
                    Node::Sum(x, axes, _) => {
                        ops.push(Op::Sum(mapping[x]));
                        raxes = Some(axes.clone());
                        rshape = get_shape(nodes, *x).clone();
                        ar = true;
                    }
                    Node::Max(x, axes, _) => {
                        ops.push(Op::Max(mapping[x]));
                        raxes = Some(axes.clone());
                        rshape = get_shape(nodes, *x).clone();
                        ar = true;
                    }
                }
            };
            if mapped {
                mapping.insert(nid, ops.len() - 1);
            }
        }
        let view;
        let rdim = if let Some(raxes) = raxes {
            //println!("rshape: {rshape:?}");
            //println!("raxes: {raxes:?}");
            //std::println!("s1: {s1}");
            let s0: usize = rshape
                .iter()
                .enumerate()
                .filter(|(a, _)| raxes.0.contains(a))
                .map(|(_, d)| *d)
                .product();
            let s1: usize = rshape
                .iter()
                .enumerate()
                .filter(|(a, _)| !raxes.0.contains(a))
                .map(|(_, d)| *d)
                .product();
            let axes: (Vec<usize>, Vec<usize>) =
                (0..rshape.rank()).partition(|a| raxes.0.contains(a));
            let axes = Axes(axes.0.into_iter().chain(axes.1).collect());
            let new_shape: Shape = [s0, s1].into();
            let ar_new_shape: Shape = [1, s1].into();
            // Join raxes together to be second last dimension
            // permute first and then reshape
            //println!("Permute axes {axes:?}");
            //println!("New shape before reduce: {new_shape:?}");
            //println!("New shape after reduce: {ar_new_shape:?}");
            let ar_rshape = rshape.clone().reduce(&axes);
            let apply_padding = |sh: View| sh.pad(
                &new_shape
                    .iter()
                    .rev()
                    .map(|d| {
                        if *d != 1 && d % 8 != 0 {
                            (0, (8 - d % 8) as i64)
                        } else {
                            (0, 0)
                        }
                    })
                    .collect::<Vec<(i64, i64)>>(),
            );
            for (a, (aview, _)) in args.iter_mut().enumerate() {
                //std::println!("aview: {aview:?}");
                *aview = apply_padding(aview
                    .reshape(if a < first_ar_arg { &rshape } else { &ar_rshape })
                    .permute(&axes)
                    .reshape(if a < first_ar_arg { &new_shape } else { &ar_new_shape }));
            }
            // view after the reduce op
            view = apply_padding(View::new(ar_rshape.clone())
                .permute(&axes)
                .reshape(&ar_new_shape));
            //std::println!("view: {view:?}");
            Some(if s0 != 1 && s0 % 8 != 0 {
                (s0 / 8 + 1) * 8
            } else {
                s0
            })
        } else {
            let sh = get_shape(nodes, x);
            let n = sh.numel();
            view = View::new(sh.clone())
                .reshape(&n.into())
                .pad(&[(0, if n % 8 != 0 { (8 - n % 8) as i64 } else { 0 })]);
            for (aview, _) in &mut args {
                let n = aview.numel();
                *aview = aview
                    .reshape(&n.into())
                    .pad(&[(0, if n % 8 != 0 { (8 - n % 8) as i64 } else { 0 })]);
            }
            None
        };
        let ast = AST {
            args: args.into_boxed_slice(),
            ops: ops.into_boxed_slice(),
            view,
            rdim,
            flop,
        };
        for op in &*ast.ops {
            println!("{op:?}");
        }
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
