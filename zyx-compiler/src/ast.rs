use crate::{ASTOp, CompiledBackend, Compiler, AST, ASTUOp, ASTBOp, ASTROp};
use alloc::{
    collections::{btree_map::Entry, BTreeMap},
    vec::Vec,
};
use zyx_core::axes::Axes;
use zyx_core::dtype::DType;
use zyx_core::error::ZyxError;
use zyx_core::node::Node;
use zyx_core::runtime::RuntimeBackend;
use zyx_core::scalar::Scalar;
use zyx_core::shape::Shape;
use zyx_core::tensor::Id;
use zyx_core::utils::get_dtype;
use zyx_core::view::View;

#[derive(Debug, Clone, PartialEq, Eq, Ord, PartialOrd)]
pub(super) struct Kernel {
    program_args: Vec<Id>,
    arg_views: Vec<View>,
    arg_dtypes: Vec<DType>,
    ops: Vec<ASTOp>,
    reduce_axes: Option<Axes>,
    reduce_dtype: Option<DType>,
    shape: Shape,
    dtype: DType,
    flop: usize,
    bytes: usize,
}

impl Kernel {
    fn leaf(x: Id, shape: &Shape, dtype: &DType) -> Self {
        Self {
            program_args: alloc::vec![x],
            arg_views: alloc::vec![View::new(shape.clone())],
            arg_dtypes: alloc::vec![dtype.clone()],
            ops: alloc::vec![ASTOp::Leaf(0)],
            reduce_axes: None,
            reduce_dtype: None,
            shape: shape.clone(),
            dtype: *dtype,
            flop: 0,
            bytes: shape.numel() * dtype.byte_size(),
        }
    }
}

impl<C: Compiler> RuntimeBackend for CompiledBackend<C> {
    fn is_evaluated(&self, x: Id) -> bool {
        self.kernels.contains_key(&x)
    }

    fn is_free_id(&self, x: Id) -> bool {
        !(self.buffers.contains_key(&x) || self.kernels.contains_key(&x))
    }

    fn remove(&mut self, x: Id) -> Result<(), ZyxError> {
        if let Some(Kernel { program_args, .. }) = self.kernels.remove(&x) {
            for p in program_args.iter().chain([&x]) {
                if !self
                    .kernels
                    .values()
                    .any(|k| k.program_args.contains(&p))
                {
                    if let Some(mut buffer) = self.buffers.remove(&p) {
                        //std::println!("Dropping buffer {p} out of total {} buffers", self.buffers.len());
                        self.compiler.drop_buffer(&mut buffer)?;
                    }
                }
            }
        }
        Ok(())
    }

    fn store<T: Scalar, IT>(&mut self, x: Id, iter: IT) -> Result<(), ZyxError>
    where
        IT: IntoIterator<Item = T>,
        IT::IntoIter: ExactSizeIterator,
    {
        //std::println!("Storing {x}");
        let iter = iter.into_iter();
        self.kernels
            .insert(x, Kernel::leaf(x, &iter.len().into(), &T::dtype()));
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
        for nid in order.iter().copied() {
            //std::println!("Compiling {nid}: {:?} x {}", nodes[nid.i()], rcs[&nid]);
            let mut kernel = match &nodes[nid.i()] {
                Node::Leaf(sh, dtype) => Kernel::leaf(nid, sh, dtype),
                Node::Uniform(..) => {
                    todo!()
                }
                Node::Cast(x, dtype) => {
                    let mut buffer = self.kernels[&x].clone();
                    buffer
                        .ops
                        .push(ASTOp::Unary(buffer.ops.len() as u8 - 1, ASTUOp::Cast(*dtype)));
                    buffer.dtype = *dtype;
                    buffer
                }
                Node::Detach(x) => self.kernels[&x].clone(),
                Node::Neg(x) => {
                    let mut buffer = self.kernels[&x].clone();
                    buffer.ops.push(ASTOp::Unary(buffer.ops.len() as u8 - 1, ASTUOp::Neg));
                    buffer
                }
                Node::ReLU(x) => {
                    let mut buffer = self.kernels[&x].clone();
                    buffer.ops.push(ASTOp::Unary(buffer.ops.len() as u8 - 1, ASTUOp::ReLU));
                    buffer
                }
                Node::Exp(x) => {
                    let mut buffer = self.kernels[&x].clone();
                    buffer.ops.push(ASTOp::Unary(buffer.ops.len() as u8 - 1, ASTUOp::Exp));
                    buffer
                }
                Node::Ln(x) => {
                    let mut buffer = self.kernels[&x].clone();
                    buffer.ops.push(ASTOp::Unary(buffer.ops.len() as u8 - 1, ASTUOp::Ln));
                    buffer
                }
                Node::Sin(x) => {
                    let mut buffer = self.kernels[&x].clone();
                    buffer.ops.push(ASTOp::Unary(buffer.ops.len() as u8 - 1, ASTUOp::Sin));
                    buffer
                }
                Node::Cos(x) => {
                    let mut buffer = self.kernels[&x].clone();
                    buffer.ops.push(ASTOp::Unary(buffer.ops.len() as u8 - 1, ASTUOp::Cos));
                    buffer
                }
                Node::Sqrt(x) => {
                    let mut buffer = self.kernels[&x].clone();
                    buffer.ops.push(ASTOp::Unary(buffer.ops.len() as u8 - 1, ASTUOp::Sqrt));
                    buffer
                }
                Node::Tanh(x) => {
                    let mut kernel = self.kernels[&x].clone();
                    kernel.ops.push(ASTOp::Unary(kernel.ops.len() as u8 - 1, ASTUOp::Tanh));
                    kernel
                }
                Node::Add(x, y) => self.binary_kernel(*x, *y, |x, y| ASTOp::Binary(x, y, ASTBOp::Add))?,
                Node::Sub(x, y) => self.binary_kernel(*x, *y, |x, y| ASTOp::Binary(x, y, ASTBOp::Sub))?,
                Node::Mul(x, y) => self.binary_kernel(*x, *y, |x, y| ASTOp::Binary(x, y, ASTBOp::Mul))?,
                Node::Div(x, y) => self.binary_kernel(*x, *y, |x, y| ASTOp::Binary(x, y, ASTBOp::Div))?,
                Node::Pow(x, y) => self.binary_kernel(*x, *y, |x, y| ASTOp::Binary(x, y, ASTBOp::Pow))?,
                Node::Cmplt(x, y) => self.binary_kernel(*x, *y, |x, y| ASTOp::Binary(x, y, ASTBOp::Cmplt))?,
                Node::Where(..) => {
                    // TODO fix this for x == y == z or any combination of those
                    todo!()
                }
                Node::Reshape(x, sh) => {
                    let mut buffer = if self.kernels[&x].reduce_axes.is_some() {
                        // TODO this should not always evaluate, because dot reshapes
                        // result and then we should still be able to merge it with unary
                        // and binary ops.
                        // TODO reshape can be applied, but no transpose afterwards? Perhaps?
                        // TODO it is perhaps easier to just reorder reshape to come later
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
                        kernel.ops.push(ASTOp::Reduce(0, ASTROp::Sum));
                    } else {
                        kernel.reduce_axes = Some(ax.clone());
                        kernel.reduce_dtype = Some(get_dtype(nodes, nid));
                        kernel.ops.push(ASTOp::Reduce(kernel.ops.len() as u8 - 1, ASTROp::Sum));
                    }
                    kernel
                }
                Node::Max(x, ax, _) => {
                    let mut kernel = self.kernels[&x].clone();
                    if kernel.reduce_axes.is_some() {
                        kernel = self.evaluate_kernel(*x)?.clone();
                        kernel.reduce_axes = Some(ax.clone());
                        kernel.reduce_dtype = Some(get_dtype(nodes, nid));
                        kernel.ops.push(ASTOp::Reduce(0, ASTROp::Max));
                    } else {
                        kernel.reduce_axes = Some(ax.clone());
                        kernel.reduce_dtype = Some(get_dtype(nodes, nid));
                        kernel.ops.push(ASTOp::Reduce(kernel.ops.len() as u8 - 1, ASTROp::Max));
                    }
                    kernel
                }
            };
            kernel.flop += nodes[nid.i()].flop(&nodes);
            //std::println!("Inserting kernel {nid}");
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

    fn evaluate_kernel(&mut self, x: Id) -> Result<&Kernel, ZyxError> {
        //kstd::println!("Evaluating kernel {x}");
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
            bytes,
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
            let ir = crate::ir::ast_to_ir(&ast, 256, 256*1024*8, 64);
            let program = self.compiler.compile(&ir)?;
            self.programs.entry(ast).or_insert(program)
        };
        let program_args: Vec<&C::Buffer> = program_args
            .into_iter()
            .map(|nid| &self.buffers[&nid])
            .collect();
        // Run the program
        self.buffers.insert(
            x,
            self.compiler.launch(program, &program_args, flop, bytes)?,
        );

        // We need to remove unused kernel and possibly drop its args!
        if let Some(kernel) = self.kernels.insert(x, Kernel::leaf(x, &r_shape, &dtype)) {
            for p in kernel.program_args {
                if !self
                    .kernels
                    .values()
                    .any(|k| k.program_args.contains(&p))
                {
                    if let Some(mut buffer) = self.buffers.remove(&p) {
                        //std::println!("Dropping buffer {p} out of total {} buffers", self.buffers.len());
                        self.compiler.drop_buffer(&mut buffer)?;
                    }
                }
            }
        }
        Ok(&self.kernels[&x])
    }

    fn binary_kernel(
        &mut self,
        x: Id,
        y: Id,
        op: impl Fn(u8, u8) -> ASTOp,
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
                        ASTOp::Leaf(x) => *x += x_buffer.arg_views.len() as u8,
                        ASTOp::Unary(x, ..) | ASTOp::Reduce(x, ..) => *x += n,
                        ASTOp::Binary(x, y, ..) => {
                            *x += n;
                            *y += n;
                        }
                        ASTOp::Where(x, y, z) => {
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
            bytes: x_buffer.bytes + y_buffer.bytes,
        })
    }
}
