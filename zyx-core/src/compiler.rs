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
    collections::{BTreeMap, BTreeSet},
    vec::Vec,
};
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
    /// Shaped uniform generator of numbers from 0. to 1.
    Uniform(View, DType),
    /// Cast into dtype unary op
    Cast(usize, DType),
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
            Op::Leaf(..) | Op::Uniform(..) => {}
            Op::Cast(x, ..)
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
        rcs: BTreeMap<Id, u8>,
        order: &[Id],
        mut nodes: &mut [Node],
    ) -> Result<(), ZyxError> {
        let mut found_reduce = None;
        for nid in order.iter().copied() {
            std::println!("{nid}: {:?}", nodes[nid.i()]);
            self.store(nid, &mut nodes)?;
            if nodes[nid.i()].is_reduce() {
                if let Some(_) = found_reduce {
                    self.evaluate_buffer(nid, Some(nid), nodes)?;
                } else {
                    found_reduce = Some((nid, nid));
                }
            }
            let mut nullify_found_reduce = false;
            if let Some((rid, p)) = &mut found_reduce {
                let node = &nodes[nid.i()];
                if node.parameters_contain(*p) {
                    if matches!(node, Node::Expand(..)) {
                        self.evaluate_buffer(*p, Some(*rid), nodes)?;
                        nullify_found_reduce = true;
                    }
                    if to_eval.contains(&nid) || rcs[&nid] > 1 {
                        self.evaluate_buffer(nid, Some(*rid), nodes)?;
                        nullify_found_reduce = true;
                    }
                    *p = nid;
                }
            }
            if nullify_found_reduce {
                found_reduce = None;
            }
            if to_eval.contains(&nid) {
                self.evaluate_buffer(nid, None, nodes)?;
            }
            // TODO removing buffers must be done properly
            /*for p in nodes[nid.i()].parameters() {
                if let Entry::Occupied(e) = rcs.entry(p).and_modify(|rc| *rc -= 1) {
                    if *e.get() == 0 {
                        self.remove(p)?;
                    }
                }
            }*/
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

    fn store(&mut self, x: Id, nodes: &mut [Node]) -> Result<(), ZyxError> {
        match &nodes[x.i()] {
            Node::IterF32(_, shape) => {
                let mut new_node = Node::Leaf(shape.clone(), DType::F32);
                core::mem::swap(&mut nodes[x.i()], &mut new_node);
                if let Node::IterF32(iter, _) = new_node {
                    self.buffers.insert(x, self.compiler.store(iter)?);
                }
            }
            Node::IterF64(_, shape) => {
                let mut new_node = Node::Leaf(shape.clone(), DType::F64);
                core::mem::swap(&mut nodes[x.i()], &mut new_node);
                if let Node::IterF64(iter, _) = new_node {
                    self.buffers.insert(x, self.compiler.store(iter)?);
                }
            }
            Node::IterI32(_, shape) => {
                let mut new_node = Node::Leaf(shape.clone(), DType::I32);
                core::mem::swap(&mut nodes[x.i()], &mut new_node);
                if let Node::IterI32(iter, _) = new_node {
                    self.buffers.insert(x, self.compiler.store(iter)?);
                }
            }
            _ => {}
        }
        Ok(())
    }

    /// This function evaluates concrete buffer that we know can be directly evaluated,
    /// that is we know that all of it's leafs are already evaluated and stored in device.
    fn evaluate_buffer(&mut self, x: Id, reduce_id: Option<Id>, nodes: &[Node]) -> Result<(), ZyxError> {
        //std::println!("Evaluating buffer {x}, reduce_id: {reduce_id:?}");
        if self.is_evaluated(x) {
            return Ok(())
        }

        let topological_order = |x: Id, check_reduce: bool| {
            let mut params: Vec<Id> = alloc::vec![x];
            let mut rcs: BTreeMap<Id, u8> = BTreeMap::new();
            //std::println!("Evaluated: {}", self.is_evaluated(crate::tensor::id(2)));
            while let Some(nid) = params.pop() {
                //std::println!("{nid}");
                rcs.entry(nid).and_modify(|rc| *rc += 1).or_insert_with(|| {
                    if !self.is_evaluated(nid) {
                        for p in nodes[nid.i()].parameters() {
                            //std::println!("Adding param: {p}, reduce_id: {reduce_id:?}");
                            if check_reduce {
                                if p != reduce_id.unwrap() {
                                    params.push(p);
                                }
                            } else {
                                params.push(p);
                            }
                        }
                    }
                    1
                });
            }
            let mut order = Vec::new();
            let mut internal_rcs: BTreeMap<Id, u8> = BTreeMap::new();
            let mut params: Vec<Id> = alloc::vec![x];
            while let Some(nid) = params.pop() {
                if rcs[&nid] == *internal_rcs.entry(nid).and_modify(|rc| *rc += 1).or_insert(1) {
                    order.push(nid);
                    if rcs.contains_key(&nid) && !self.is_evaluated(nid) {
                        for p in nodes[nid.i()].parameters() {
                            if check_reduce {
                                if p != reduce_id.unwrap() {
                                    params.push(p);
                                }
                            } else {
                                params.push(p);
                            }
                        }
                    }
                }
            }
            order.into_iter().rev()
        };

        let order: Vec<Id> = if let Some(reduce_id) = reduce_id {
            let temp = topological_order(reduce_id, false);
            if reduce_id != x {
                //std::println!("Adding AR ops.");
                temp.chain(topological_order(x, true)).collect()
            } else {
                temp.collect()
            }
        } else {
            topological_order(x, false).collect()
        };

        //std::println!("order {order:?}");
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
        let mut reduce_type_max = false;
        // TODO reorder in such a way, that first are all before reduce loads,
        // then before reduce ops, then after reduce loads, then after reduce ops.
        // If it is not a reduce kernel, then all leafs should be first.
        for nid in order {
            //std::println!("ops: {ops:?}");
            //std::println!("{nid}: {:?}", nodes[nid.i()]);
            flop += nodes[nid.i()].flop(nodes);
            let mut mapped = true;
            if let Some(x) = self.buffers.get(&nid) {
                if ar {
                    if first_ar_arg == usize::MAX {
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
                    Node::Leaf(..)
                    | Node::IterF32(..)
                    | Node::IterF64(..)
                    | Node::IterI32(..) => {}
                    Node::Uniform(sh, dtype) => ops.push(Op::Uniform(View::new(sh.clone()), *dtype)),
                    Node::Cast(x, dtype) => ops.push(Op::Cast(mapping[x], *dtype)),
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
                                if a < first_ar_arg {
                                    args[a].0 = args[a].0.expand(sh);
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
                                    args[a].0 = args[a].0.reshape(sh);
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
                                    args[a].0 = args[a].0.pad(padding);
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
                                    args[a].0 = args[a].0.permute(axes);
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
                        reduce_type_max = true;
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
        let padding_width = 4;
        let view;
        let rdim = if let Some(raxes) = raxes {
            //std::println!("rshape: {rshape:?}");
            //std::println!("raxes: {raxes:?}");
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
            //std::println!("Permute axes {axes:?}");
            //std::println!("New shape before reduce: {new_shape:?}");
            //std::println!("New shape after reduce: {ar_new_shape:?}");
            let ar_rshape = rshape.clone().reduce(&axes);
            // Padding is not applied, in max kernel, because in max op
            // we can not pad with zeros, because it is incorrect!!!
            let apply_padding = |sh: View| sh.pad(
                &new_shape
                    .iter()
                    .rev()
                    .map(|d| {
                        if d % padding_width != 0 {
                            (0, (padding_width - d % padding_width) as i64)
                        } else {
                            (0, 0)
                        }
                    })
                    .collect::<Vec<(i64, i64)>>(),
            );
            for (a, (aview, _)) in args.iter_mut().enumerate() {
                //std::println!("aview: {aview:?}");
                let temp = aview
                    .reshape(if a < first_ar_arg { &rshape } else { &ar_rshape })
                    .permute(&axes)
                    .reshape(if a < first_ar_arg { &new_shape } else { &ar_new_shape });
                if reduce_type_max {
                    *aview = temp;
                } else {
                    *aview = apply_padding(temp);
                }
            }
            // view after the reduce op
            let temp = View::new(ar_rshape.clone())
                .permute(&axes)
                .reshape(&ar_new_shape);
            if reduce_type_max {
                view = temp;
            } else {
                view = apply_padding(temp);
            }
            //std::println!("view: {view:?}");
            Some(if s0 != 1 && s0 % padding_width != 0 && !reduce_type_max {
                (s0 / padding_width + 1) * padding_width
            } else {
                s0
            })
        } else {
            let sh = get_shape(nodes, x);
            let n = sh.numel();
            view = View::new(sh.clone())
                .reshape(&n.into())
                .pad(&[(0, if n % padding_width != 0 { (padding_width - n % padding_width) as i64 } else { 0 })]);
            for (aview, _) in &mut args {
                let n = aview.numel();
                *aview = aview
                    .reshape(&n.into())
                    .pad(&[(0, if n % padding_width != 0 { (padding_width - n % padding_width) as i64 } else { 0 })]);
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
        for op in &*ast.ops { std::println!("{op:?}"); }
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
