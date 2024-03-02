use crate::axes::Axes;
use crate::error::ZyxError;
use crate::shape::Shape;
use crate::{
    dtype::DType, node::Node, runtime::RuntimeBackend, scalar::Scalar, tensor::Id,
    utils::get_dtype, view::View,
};
use alloc::{
    collections::{btree_map::Entry, BTreeMap},
    vec::Vec,
};

/// Implement this trait for compiled backends
pub trait Compiler {
    /// Buffer holds actual values in memory
    type Buffer;
    /// Program is kernel executable on the device, can be compiled at runtime
    type Program;
    /// Store iter into buffer
    fn store<T: Scalar>(&mut self, iter: impl IntoIterator<Item = T>) -> Result<Self::Buffer, ZyxError>;
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
        flop: usize,
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
    /// Shape of the result, this is before any reduce ops
    pub shape: Shape,
    /// DType of the result
    pub dtype: DType,
    /// Reduce axes, if any
    pub reduce_axes: Option<Axes>,
    /// DType of accumulated elements, if any
    pub reduce_dtype: Option<DType>,
}

#[derive(Debug, Clone, PartialEq, Eq, Ord, PartialOrd)]
struct Kernel {
    program_args: Vec<Id>,
    arg_views: Vec<View>,
    arg_dtypes: Vec<DType>,
    ops: Vec<Op>,
    reduce_axes: Option<Axes>,
    reduce_dtype: Option<DType>,
    shape: Shape,
    dtype: DType,
    flop: usize,
}

impl Kernel {
    fn leaf(x: Id, shape: &Shape, dtype: &DType) -> Self {
        Self {
            program_args: alloc::vec![x],
            arg_views: alloc::vec![View::new(shape.clone())],
            arg_dtypes: alloc::vec![dtype.clone()],
            ops: alloc::vec![Op::Leaf(0)],
            reduce_axes: None,
            reduce_dtype: None,
            shape: shape.clone(),
            dtype: *dtype,
            flop: 0,
        }
    }
}

/// Compiled backend that holds compiler, buffers and programs
pub struct CompiledBackend<C: Compiler> {
    compiler: C,
    kernels: BTreeMap<Id, Kernel>,
    buffers: BTreeMap<Id, C::Buffer>,
    programs: BTreeMap<AST, C::Program>,
}

impl<C: Compiler> RuntimeBackend for CompiledBackend<C> {
    fn is_evaluated(&self, x: Id) -> bool {
        self.kernels.contains_key(&x)
    }

    fn is_free_id(&self, x: Id) -> bool {
        !(self.buffers.contains_key(&x) || self.kernels.contains_key(&x))
    }

    fn remove(&mut self, x: Id) -> Result<(), ZyxError> {
        std::println!("Compiler removing {x}");
        if self.kernels.remove(&x).is_some() {
            //std::println!("Kernels {:?}", self.kernels);
            if !self.kernels.values().any(|kernel| kernel.program_args.contains(&x)) {
                if let Some(mut buffer) = self.buffers.remove(&x) {
                    std::println!("Dropping buffer {x}");
                    self.compiler.drop_buffer(&mut buffer)?;
                }
            }
        }
        Ok(())
    }

    fn store<T: Scalar, IT>(&mut self, x: Id, iter: IT) -> Result<(), ZyxError>
    where
        IT: IntoIterator<Item=T>,
        IT::IntoIter: ExactSizeIterator,
    {
        //std::println!("Storing {x}");
        let iter = iter.into_iter();
        self.kernels.insert(x, Kernel::leaf(x, &iter.len().into(), &T::dtype()));
        self.buffers.insert(x, self.compiler.store(iter)?);
        Ok(())
    }

    fn load<T: Scalar>(&mut self, x: Id, numel: usize) -> Result<Vec<T>, ZyxError> {
        //std::println!("Loading {x}");
        if let Some(buffer) = self.buffers.get(&x) {
            self.compiler.load(buffer, numel)
        } else {
            self.evaluate_kernel(x)?;
            self.compiler.load(&self.buffers[&x], numel)
        }
    }

    fn evaluate(
        &mut self,
        mut rcs: BTreeMap<Id, u32>,
        order: &[Id],
        nodes: &[Node],
    ) -> Result<(), ZyxError> {
        //std::println!("Evaluating rcs {:?}", rcs);
        // TODO must_eval are currently new_leafs from runtime, but this may not
        // be the case later and then this won't work, so fix it.
        // TODO calculate flops for kernels :D
        for nid in order.iter().copied() {
            std::println!("Compiling {nid}: {:?} x {}", nodes[nid.i()], rcs[&nid]);
            let kernel = match &nodes[nid.i()] {
                Node::Leaf(sh, dtype) => Kernel::leaf(nid, sh, dtype),
                Node::Uniform(..) => {
                    todo!()
                }
                Node::Cast(x, dtype) => {
                    let mut buffer = self.kernels[&x].clone();
                    buffer
                        .ops
                        .push(Op::Cast(buffer.ops.len() as u8 - 1, *dtype));
                    buffer.dtype = *dtype;
                    buffer
                }
                Node::Detach(x) => {
                    self.kernels[&x].clone()
                }
                Node::Neg(x) => {
                    let mut buffer = self.kernels[&x].clone();
                    buffer.ops.push(Op::Neg(buffer.ops.len() as u8 - 1));
                    buffer
                }
                Node::ReLU(x) => {
                    let mut buffer = self.kernels[&x].clone();
                    buffer.ops.push(Op::ReLU(buffer.ops.len() as u8 - 1));
                    buffer
                }
                Node::Exp(x) => {
                    let mut buffer = self.kernels[&x].clone();
                    buffer.ops.push(Op::Exp(buffer.ops.len() as u8 - 1));
                    buffer
                }
                Node::Ln(x) => {
                    let mut buffer = self.kernels[&x].clone();
                    buffer.ops.push(Op::Ln(buffer.ops.len() as u8 - 1));
                    buffer
                }
                Node::Sin(x) => {
                    let mut buffer = self.kernels[&x].clone();
                    buffer.ops.push(Op::Sin(buffer.ops.len() as u8 - 1));
                    buffer
                }
                Node::Cos(x) => {
                    let mut buffer = self.kernels[&x].clone();
                    buffer.ops.push(Op::Cos(buffer.ops.len() as u8 - 1));
                    buffer
                }
                Node::Sqrt(x) => {
                    let mut buffer = self.kernels[&x].clone();
                    buffer.ops.push(Op::Sqrt(buffer.ops.len() as u8 - 1));
                    buffer
                }
                Node::Tanh(x) => {
                    let mut kernel = self.kernels[&x].clone();
                    kernel.ops.push(Op::Tanh(kernel.ops.len() as u8 - 1));
                    kernel
                }
                Node::Add(x, y) => self.binary_kernel(*x, *y, |x, y| Op::Add(x, y))?,
                Node::Sub(x, y) => self.binary_kernel(*x, *y, |x, y| Op::Sub(x, y))?,
                Node::Mul(x, y) => self.binary_kernel(*x, *y, |x, y| Op::Mul(x, y))?,
                Node::Div(x, y) => self.binary_kernel(*x, *y, |x, y| Op::Div(x, y))?,
                Node::Pow(x, y) => self.binary_kernel(*x, *y, |x, y| Op::Pow(x, y))?,
                Node::Cmplt(x, y) => self.binary_kernel(*x, *y, |x, y| Op::Cmplt(x, y))?,
                Node::Where(..) => {
                    // TODO fix this for x == y == z or any combination of those
                    todo!()
                }
                Node::Reshape(x, sh) => {
                    let mut buffer = if self.kernels[&x].reduce_axes.is_some() {
                        self.evaluate_kernel(*x)?.clone()
                    } else {
                        self.kernels[&x].clone()
                    };
                    for view in &mut buffer.arg_views {
                        *view = view.reshape(sh);
                    }
                    buffer.shape = sh.clone();
                    buffer
                }
                Node::Expand(x, sh) => {
                    let mut kernel = if self.kernels[&x].reduce_axes.is_some() {
                        self.evaluate_kernel(*x)?.clone()
                    } else {
                        self.kernels[&x].clone()
                    };
                    for view in &mut kernel.arg_views {
                        *view = view.expand(sh);
                    }
                    kernel.shape = sh.clone();
                    kernel
                }
                Node::Permute(x, ax, sh) => {
                    let mut kernel = self.kernels[&x].clone();
                    for view in &mut kernel.arg_views {
                        *view = view.permute(ax);
                    }
                    if let Some(reduce_axes) = &mut kernel.reduce_axes {
                        *reduce_axes = reduce_axes.permute(ax);
                    }
                    kernel.shape = sh.clone();
                    kernel
                }
                Node::Pad(x, padding, sh) => {
                    let mut kernel = if self.kernels[&x].reduce_axes.is_some() {
                        self.evaluate_kernel(*x)?.clone()
                    } else {
                        self.kernels[&x].clone()
                    };
                    for view in &mut kernel.arg_views {
                        *view = view.pad(padding);
                    }
                    kernel.shape = sh.clone();
                    kernel
                }
                Node::Sum(x, ax, _) => {
                    let mut kernel = self.kernels[&x].clone();
                    if kernel.reduce_axes.is_some() {
                        kernel = self.evaluate_kernel(*x)?.clone();
                        kernel.reduce_axes = Some(ax.clone());
                        kernel.reduce_dtype = Some(get_dtype(nodes, nid));
                        kernel.ops.push(Op::Sum(0));
                    } else {
                        kernel.reduce_axes = Some(ax.clone());
                        kernel.reduce_dtype = Some(get_dtype(nodes, nid));
                        kernel.ops.push(Op::Sum(kernel.ops.len() as u8 - 1));
                    }
                    kernel
                }
                Node::Max(x, ax, _) => {
                    let mut kernel = self.kernels[&x].clone();
                    if kernel.reduce_axes.is_some() {
                        kernel = self.evaluate_kernel(*x)?.clone();
                        kernel.reduce_axes = Some(ax.clone());
                        kernel.reduce_dtype = Some(get_dtype(nodes, nid));
                        kernel.ops.push(Op::Max(0));
                    } else {
                        kernel.reduce_axes = Some(ax.clone());
                        kernel.reduce_dtype = Some(get_dtype(nodes, nid));
                        kernel.ops.push(Op::Max(kernel.ops.len() as u8 - 1));
                    }
                    kernel
                }
            };
            std::println!("Inserting kernel {nid}");
            self.kernels.insert(nid, kernel);

            if self.kernels[&nid].ops.len() > 200
                || (rcs[&nid] > 1 && self.kernels[&nid].program_args.len() > 1)
            {
                //std::println!("Forcing evaluation of {nid}");
                self.evaluate_kernel(nid)?;
            }
            //std::println!("Kernel {:?}, len of buffers: {}", self.kernels[&nid], self.buffers.len());

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
            kernels: BTreeMap::new(),
            buffers: BTreeMap::new(),
            programs: BTreeMap::new(),
        }
    }

    fn evaluate_kernel(
        &mut self,
        x: Id,
    ) -> Result<&Kernel, ZyxError> {
        std::println!("Evaluating kernel {x}");
        if self.buffers.contains_key(&x) {
            //std::println!("Accessing kernel {x}, {:?} {:?}", self.buffers.keys(), self.kernels);
            return Ok(&self.kernels[&x]);
        }
        let Kernel {
            program_args,
            arg_views,
            arg_dtypes,
            ops,
            reduce_axes,
            reduce_dtype,
            shape,
            dtype,
            flop,
        } = self.kernels[&x].clone();
        let r_shape = if let Some(reduce_axes) = &reduce_axes {
            shape.clone().reduce(reduce_axes)
        } else {
            shape.clone()
        };
        let ast = AST {
            arg_views,
            arg_dtypes,
            ops,
            shape,
            dtype,
            reduce_axes,
            reduce_dtype,
        };
        //std::println!("Ops");
        //for op in &*ast.ops { std::println!("{op:?}"); }
        // Used cached program or compile new program
        let program = if let Some(program) = self.programs.get(&ast) {
            program
        } else {
            // TODO optimize ast as much as possible here
            // for example deduplicate args
            let program = self.compiler.compile(&ast)?;
            self.programs.entry(ast).or_insert(program)
        };
        let program_args: Vec<&C::Buffer> = program_args
            .into_iter()
            .map(|nid| &self.buffers[&nid])
            .collect();
        // Run the program
        self.kernels.insert(x, Kernel::leaf(x, &r_shape, &dtype));
        self.buffers.insert(x, self.compiler.launch(program, &program_args, flop)?);
        Ok(&self.kernels[&x])
    }

    fn binary_kernel(
        &mut self,
        x: Id,
        y: Id,
        op: impl Fn(u8, u8) -> Op,
    ) -> Result<Kernel, ZyxError> {
        let (reduce_axes, reduce_dtype) = if x != y {
            match (
                self.kernels[&x].reduce_axes.clone(),
                self.kernels[&y].reduce_axes.clone(),
            ) {
                (Some(x_ax), Some(_)) => {
                    self.evaluate_kernel(y)?;
                    (Some(x_ax), Some(self.kernels[&x].dtype))
                }
                (Some(x_ax), None) => (Some(x_ax), Some(self.kernels[&x].dtype)),
                (None, Some(y_ax)) => (Some(y_ax), Some(self.kernels[&y].dtype)),
                (None, None) => (None, None),
            }
        } else {
            //(self.kernels[&x].reduce_axes.clone(), self.kernels[&x].reduce_dtype)
            // TODO if it is one reduce, it could be merged, but it is lot of work
            let mut buffer = if self.kernels[&x].reduce_axes.is_some() {
                self.evaluate_kernel(x)?.clone()
            } else {
                self.kernels[&x].clone()
            };
            let n = buffer.ops.len() as u8 - 1;
            buffer.ops.push(op(n, n));
            //(None, None)
            return Ok(buffer);
        };
        let x_buffer = &self.kernels[&x];
        let y_buffer = &self.kernels[&y];
        let n = x_buffer.ops.len() as u8;
        Ok(Kernel {
            program_args: x_buffer
                .program_args
                .iter()
                .chain(y_buffer.program_args.iter())
                .copied()
                .collect(),
            arg_views: x_buffer
                .arg_views
                .iter()
                .chain(y_buffer.arg_views.iter())
                .cloned()
                .collect(),
            arg_dtypes: x_buffer
                .arg_dtypes
                .iter()
                .chain(y_buffer.arg_dtypes.iter())
                .copied()
                .collect(),
            ops: x_buffer
                .ops
                .iter()
                .cloned()
                .chain(y_buffer.ops.iter().cloned().map(|mut op| {
                    match &mut op {
                        Op::Leaf(x) => *x += x_buffer.arg_views.len() as u8,
                        Op::Cast(x, ..)
                        | Op::Neg(x)
                        | Op::ReLU(x)
                        | Op::Exp(x)
                        | Op::Ln(x)
                        | Op::Tanh(x)
                        | Op::Sin(x)
                        | Op::Cos(x)
                        | Op::Sqrt(x)
                        | Op::Sum(x)
                        | Op::Max(x) => *x += n,
                        Op::Add(x, y)
                        | Op::Sub(x, y)
                        | Op::Mul(x, y)
                        | Op::Div(x, y)
                        | Op::Pow(x, y)
                        | Op::Cmplt(x, y) => {
                            *x += n;
                            *y += n;
                        }
                        Op::Where(x, y, z) => {
                            *x += n;
                            *y += n;
                            *z += n;
                        }
                    }
                    op
                }))
                .chain([op(n - 1, n + y_buffer.ops.len() as u8 - 1)])
                .collect(),
            reduce_axes,
            reduce_dtype,
            shape: x_buffer.shape.clone(),
            dtype: x_buffer.dtype,
            flop: x_buffer.flop + y_buffer.flop,
        })
    }
}
