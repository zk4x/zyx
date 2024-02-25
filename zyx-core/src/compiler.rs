use crate::axes::Axes;
use crate::error::ZyxError;
use crate::{
    dtype::DType,
    node::Node,
    runtime::RuntimeBackend,
    scalar::Scalar,
    tensor::Id,
    view::View,
};
use alloc::{
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
        flop: usize
    ) -> Result<Self::Buffer, ZyxError>;
    /// Compile ast into program
    fn compile(&mut self, ast: &AST) -> Result<Self::Program, ZyxError>;
}

/// Op executable on device with compiled backend
/// usize are all IDs into ops, leafs have IDs into args
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub enum Op {
    // We are not gonna have so gigantic kernels with more than 256 arguments or ops.
    // OpenCL does not support more than 255 args.
    /// Leaf (holds data, id to kernel arg)
    Leaf(u8),
    // TODO uniform generators should also take shape into consideration
    // and repeat the same random number if this shape is expanded.
    // Shaped uniform generator of numbers from 0. to 1.
    //Uniform(View, DType),
    /// Cast into dtype unary op
    Cast(u8, DType),
    /// Neg unary op
    Neg(u8),
    /// ReLU unary op
    ReLU(u8),
    /// Sin unary op
    Sin(u8),
    /// Cos unary op
    Cos(u8),
    /// Ln unary op
    Ln(u8),
    /// Exp unary op
    Exp(u8),
    /// Tanh unary op
    Tanh(u8),
    /// Sqrt unary op
    Sqrt(u8),
    /// Addition binary op
    Add(u8, u8),
    /// Substitution binary op
    Sub(u8, u8),
    /// Multiplication binary op
    Mul(u8, u8),
    /// Division binary op
    Div(u8, u8),
    /// Exponentiation binary op
    Pow(u8, u8),
    /// Compare less than binary op
    Cmplt(u8, u8),
    /// Where op
    Where(u8, u8, u8),
    /// Sum reduce op
    Sum(u8),
    /// Max reduce op
    Max(u8),
}

impl Op {
    fn access_parameters(&mut self, mut op: impl FnMut(&mut u8)) {
        match self {
            Op::Leaf(..) => {}
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
            | Op::Max(x) => op(x),
            Op::Add(x, y)
            | Op::Sub(x, y)
            | Op::Mul(x, y)
            | Op::Div(x, y)
            | Op::Pow(x, y)
            | Op::Cmplt(x, y) => {
                op(x);
                op(y);
            }
            Op::Where(x, y, z) => {
                op(x);
                op(y);
                op(z);
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
    /// AST argument views
    pub arg_views: Vec<View>,
    /// AST argument dtypes
    pub arg_dtypes: Vec<DType>,
    /// AST ops
    pub ops: Vec<Op>,
    /// Shape of the result
    pub shape: Shape,
    /// DType of the result
    pub dtype: DType,
    /// Reduce axes, if any
    pub reduce_axes: Option<Axes>,
}

#[derive(Debug, Clone, PartialEq, Eq, Ord, PartialOrd)]
struct Buffer {
    program_args: Vec<Id>,
    arg_views: Vec<View>,
    arg_dtypes: Vec<DType>,
    ops: Vec<Op>,
    reduce_axes: Option<Axes>,
    shape: Shape,
    dtype: DType,
    flop: usize,
}

impl Buffer {
    fn leaf(x: Id, shape: &Shape, dtype: &DType) -> Self {
        Self {
            program_args: alloc::vec![x],
            arg_views: alloc::vec![View::new(shape.clone())],
            arg_dtypes: alloc::vec![dtype.clone()],
            ops: alloc::vec![Op::Leaf(0)],
            reduce_axes: None,
            shape: shape.clone(),
            dtype: *dtype,
            flop: 0,
        }
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
        rcs: BTreeMap<Id, u8>,
        order: &[Id],
        mut nodes: &mut [Node],
    ) -> Result<(), ZyxError> {
        let mut buffers: BTreeMap<Id, Buffer> = BTreeMap::new();
        // TODO calculate flops for buffers :D
        for nid in order.iter().copied() {
            self.store(nid, &mut nodes)?;
            let buffer = match &nodes[nid.i()] {
                Node::IterF32(..) | Node::IterF64(..) | Node::IterI32(..) => { panic!() }
                Node::Leaf(sh, dtype) => {
                    Buffer::leaf(nid, sh, dtype)
                }
                Node::Uniform(..) => { todo!() }
                Node::Cast(x, dtype) => {
                    let mut buffer = buffers[&x].clone();
                    buffer.ops.push(Op::Cast(buffer.ops.len() as u8 - 1, *dtype));
                    buffer.dtype = *dtype;
                    buffer
                }
                Node::Ln(x) => {
                    let mut buffer = buffers[&x].clone();
                    buffer.ops.push(Op::Ln(buffer.ops.len() as u8 - 1));
                    buffer
                }
                Node::Exp(x) => {
                    let mut buffer = buffers[&x].clone();
                    buffer.ops.push(Op::Exp(buffer.ops.len() as u8 - 1));
                    buffer
                }
                Node::Add(x, y) => {
                    self.binary_buffer(*x, *y, &mut buffers)?
                }
                Node::Reshape(x, sh) => {
                    let mut buffer = buffers[&x].clone();
                    for view in &mut buffer.arg_views {
                        *view = view.reshape(sh);
                    }
                    buffer.shape = sh.clone();
                    buffer
                }
                Node::Expand(x, sh) => {
                    let mut buffer = buffers[&x].clone();
                    for view in &mut buffer.arg_views {
                        *view = view.expand(sh);
                    }
                    buffer.shape = sh.clone();
                    buffer
                }
                Node::Permute(x, ax, sh) => {
                    let mut buffer = buffers[&x].clone();
                    for view in &mut buffer.arg_views {
                        *view = view.permute(ax);
                    }
                    buffer.shape = sh.clone();
                    buffer
                }
                Node::Pad(x, padding, sh) => {
                    let mut buffer = buffers[&x].clone();
                    for view in &mut buffer.arg_views {
                        *view = view.pad(padding);
                    }
                    buffer.shape = sh.clone();
                    buffer
                }
                Node::Sum(x, ax, sh) => {
                    let mut buffer = buffers[&x].clone();
                    if buffer.reduce_axes.is_some() {
                        buffer = self.evaluate_buffer(*x, &mut buffers)?.clone();
                        buffer.reduce_axes = Some(ax.clone());
                        buffer.ops.push(Op::Sum(0));
                        buffer.shape = sh.clone();
                    } else {
                        buffer.ops.push(Op::Sum(buffer.ops.len() as u8));
                        buffer.reduce_axes = Some(ax.clone());
                    }
                    buffer
                }
                Node::Max(x, ax, sh) => {
                    let mut buffer = buffers[&x].clone();
                    if buffer.reduce_axes.is_some() {
                        self.evaluate_buffer(*x, &mut buffers)?;
                        buffer = Buffer::leaf(*x, &buffer.shape, &buffer.dtype);
                        buffer.reduce_axes = Some(ax.clone());
                        buffer.ops.push(Op::Max(0));
                        buffer.shape = sh.clone();
                    } else {
                        buffer.ops.push(Op::Max(buffer.ops.len() as u8));
                        buffer.reduce_axes = Some(ax.clone());
                    }
                    buffer
                }
                _ => { todo!() }
            };
            buffers.insert(nid, buffer);

           if to_eval.contains(&nid) || (rcs[&nid] > 1 && buffers[&nid].program_args.len() > 1) {
                self.evaluate_buffer(nid, &mut buffers)?;
            }

            /*for p in nodes[nid.i()].parameters() {
                if let Entry::Occupied(e) = rcs.entry(p).and_modify(|rc| *rc -= 1) {
                    if *e.get() == 0 {
                        self.remove(p)?;
                    }
                }
            }*/
            // i. e. if they contain other than just unary ops.
            /*if self.is_evaluated(nid) {
                let mut params: Vec<Id> = nodes[nid.i()].parameters().collect();
                while let Some(p) = params.pop() {
                    //std::println!("Param: {p}, rc: {}", rcs[&p]);
                    if let Entry::Occupied(e) = rcs.entry(p).and_modify(|rc| *rc -= 1) {
                        if *e.get() == 0 {
                            e.remove_entry();
                            self.remove(p)?;
                            params.extend(nodes[p.i()].parameters());
                        }
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

    fn evaluate_buffer<'a>(&mut self, x: Id, buffers: &'a mut BTreeMap<Id, Buffer>) -> Result<&'a Buffer, ZyxError> {
        let Buffer { program_args, arg_views, arg_dtypes, ops, reduce_axes, shape, dtype, flop } = buffers[&x].clone();
        // TODO reshape and permute reduce axes
        let ast = AST {
            arg_views,
            arg_dtypes,
            ops,
            shape: shape.clone(),
            dtype,
            reduce_axes,
        };
        std::println!("Ops");
        for op in &*ast.ops { std::println!("{op:?}"); }
        // Used cached program or compile new program
        let program = if let Some(program) = self.programs.get(&ast) {
            program
        } else {
            let program = self.compiler.compile(&ast)?;
            self.programs.entry(ast).or_insert(program)
        };
        let program_args: Vec<&C::Buffer> = program_args.into_iter().map(|nid| &self.buffers[&nid]).collect();
        // Run the program
        self.buffers.insert(x, self.compiler.launch(program, &program_args, flop)?);
        Ok(buffers.entry(x).or_insert(Buffer::leaf(x, &shape, &dtype)))
    }

    fn binary_buffer(&mut self, x: Id, y: Id, buffers: &mut BTreeMap<Id, Buffer>) -> Result<Buffer, ZyxError> {
        let reduce_axes = match (buffers[&x].reduce_axes.clone(), buffers[&y].reduce_axes.clone()) {
            (Some(x_ax), Some(_)) => {
                self.evaluate_buffer(y, buffers)?;
                Some(x_ax)
            }
            (Some(x_ax), None) => {
                Some(x_ax)
            }
            (None, Some(y_ax)) => {
                Some(y_ax)
            }
            (None, None) => {
                None
            }
        };
        let y_buffer = &buffers[&y];
        let x_buffer = &buffers[&x];
        let n = x_buffer.ops.len() as u8;
        Ok(Buffer {
            program_args: x_buffer.program_args.iter().chain(y_buffer.program_args.iter()).copied().collect(),
            arg_views: x_buffer.arg_views.iter().chain(y_buffer.arg_views.iter()).cloned().collect(),
            arg_dtypes: x_buffer.arg_dtypes.iter().chain(y_buffer.arg_dtypes.iter()).copied().collect(),
            ops: x_buffer.ops.iter().cloned().chain(y_buffer.ops.iter().cloned().map(|mut op| {
                op.access_parameters(|x| *x += n);
                op
            })).collect(),
            reduce_axes,
            shape: x_buffer.shape.clone(),
            dtype: x_buffer.dtype,
            flop: x_buffer.flop + y_buffer.flop,
        })
    }
}
