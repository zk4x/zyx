use crate::dtype::Constant;
use crate::runtime::TensorId;
use crate::scalar::Scalar;
use crate::DType;
use alloc::collections::{BTreeMap, BTreeSet};
use alloc::format as f;
use alloc::string::String;
use alloc::vec;
use alloc::vec::Vec;
use core::fmt::{Display, Formatter};

#[cfg(feature = "debug1")]
use std::println;

use super::graph::Graph;
use super::node::{BOp, Node, ROp, UOp};

#[cfg(feature = "cuda")]
pub(super) mod cuda;

#[cfg(feature = "hsa")]
pub(super) mod hsa;

#[cfg(feature = "opencl")]
pub(super) mod opencl;

#[cfg(feature = "wgsl")]
pub(super) mod wgsl;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum Scope {
    Global,
    Local,
    Register,
}

impl Display for Scope {
    fn fmt(&self, f: &mut Formatter<'_>) -> core::fmt::Result {
        f.write_str(match self {
            Self::Global => "g",
            Self::Local => "l",
            Self::Register => "r",
        })
    }
}

pub(super) trait Compiler: Sized {
    type Buffer;
    type Program;
    fn initialize() -> Result<Self, CompilerError>;
    fn hardware_information(&mut self) -> Result<HWInfo, CompilerError>;
    fn allocate_memory(&mut self, byte_size: usize) -> Result<Self::Buffer, CompilerError>;
    fn store_memory<T: Scalar>(
        &mut self,
        buffer: &mut Self::Buffer,
        data: &[T],
    ) -> Result<(), CompilerError>;
    fn load_memory<T: Scalar>(
        &mut self,
        buffer: &Self::Buffer,
        length: usize,
    ) -> Result<Vec<T>, CompilerError>;
    fn deallocate_memory(&mut self, buffer: Self::Buffer) -> Result<(), CompilerError>;
    fn compile_program(&mut self, kernel: &IRKernel) -> Result<Self::Program, CompilerError>;
    fn launch_program(
        &mut self,
        program: &Self::Program,
        args: &mut [Self::Buffer],
    ) -> Result<(), CompilerError>;
    fn release_program(&mut self, program: Self::Program) -> Result<(), CompilerError>;
}

pub(super) struct CompiledBackend<C: Compiler> {
    compiler: C,
    buffers: BTreeMap<TensorId, C::Buffer>,
    compiled_graphs: BTreeMap<Graph, CompiledGraph<C::Program>>,
    hwinfo: HWInfo,
}

impl<C: Compiler> Drop for CompiledBackend<C> {
    fn drop(&mut self) {
        while let Some((_, buffer)) = self.buffers.pop_last() {
            self.compiler.deallocate_memory(buffer).unwrap();
        }
        while let Some((_, graph)) = self.compiled_graphs.pop_last() {
            for (_, program) in graph.programs {
                self.compiler.release_program(program).unwrap();
            }
        }
    }
}

/// Compiled graph
pub(super) struct CompiledGraph<Program> {
    // Ordered programs and arguments to them
    programs: Vec<(Vec<TensorId>, Program)>,
    //flop: usize,
    //bytes: usize,
}

#[derive(Debug)]
pub enum CompilerError {
    InitializationFailure(&'static str),
    OutOfDeviceMemory(&'static str),
    OutOfHostMemory(&'static str),
    BufferDoesNotExist(&'static str),
    // For all unknown errors
    GeneralExecutionError(&'static str),
}

/// Hardware information needed for applying optimizations
#[derive(Debug)]
pub struct HWInfo {
    /// Biggest kernel dimensions
    pub max_work_item_sizes: Vec<usize>,
    /// Maximum local work size threads
    pub max_work_group_size: usize,
    /// Preferred vector size in bytes
    pub preferred_vector_size: usize,
    /// Is half supported?
    pub f16_support: bool,
    /// Is double supported?
    pub f64_support: bool,
    /// Is fused multiply add supported?
    pub fmadd: bool,
    /// Global (VRAM, RAM) memory size in bytes
    pub global_mem_size: usize,
    /// Maximum memory allocation for single buffer in bytes
    pub max_mem_alloc: usize,
    /// Alignment for data types in bytes
    pub mem_align: usize,
    /// Page size (base address alignment) in bytes
    pub page_size: usize,
    /// Local memory size in bytes
    pub local_mem_size: usize,
    /// Number of registers per thread
    pub num_registers: usize,
    /// Does this hardware support native matmul of 16x16 local tiles?
    pub native_mm16x16_support: bool,
}

impl<C: Compiler> CompiledBackend<C> {
    pub(super) fn initialize() -> Result<Self, CompilerError> {
        let mut compiler = C::initialize()?;
        Ok(Self {
            hwinfo: compiler.hardware_information()?,
            compiler,
            buffers: BTreeMap::new(),
            compiled_graphs: BTreeMap::new(),
        })
    }

    pub(super) fn is_realized(&self, x: TensorId) -> bool {
        self.buffers.contains_key(&x)
    }

    pub(super) fn store<T: Scalar>(
        &mut self,
        x: TensorId,
        data: &[T],
    ) -> Result<(), CompilerError> {
        let mut buffer = self.compiler.allocate_memory(data.len() * T::byte_size())?;
        self.compiler.store_memory(&mut buffer, data)?;
        self.buffers.insert(x, buffer);
        return Ok(());
    }

    // Load values at x, if x is not evaluated, it will return error
    pub(super) fn load<T: Scalar>(
        &mut self,
        x: TensorId,
        length: usize,
    ) -> Result<Vec<T>, CompilerError> {
        #[cfg(feature = "debug1")]
        println!("Attempting to load buffer with id {x}");
        if let Some(buffer) = self.buffers.get(&x) {
            return self.compiler.load_memory(buffer, length);
        }
        return Err(CompilerError::BufferDoesNotExist(
            "Buffer with given id does not exist.",
        ));
    }

    pub(super) fn remove(&mut self, x: TensorId) -> Result<(), CompilerError> {
        if let Some(buffer) = self.buffers.remove(&x) {
            return self.compiler.deallocate_memory(buffer);
        }
        return Ok(());
    }

    /// Compiles and caches graph
    pub(super) fn compile_graph(
        &mut self,
        org_graph: &Graph,
        to_eval: BTreeSet<TensorId>,
    ) -> Result<(), CompilerError> {
        //#[cfg(feature = "debug1")]
        //libc_print::libc_println!("Evaluating {to_eval:?}");
        if self.compiled_graphs.contains_key(&org_graph) {
            return Ok(());
        }
        let mut graph = org_graph.clone();
        let order = graph.execution_order(&to_eval);
        let mut kernels = generate_kernels(&graph, &order, &to_eval);

        let mut programs = Vec::new();
        #[cfg(feature = "debug1")]
        println!("Compiling kernels");
        // Rewrite tiled representation to IR representation and compile it for HW device
        // TODO kernels can be compiled in parallel
        for kernel in &mut kernels {
            // Get the number of loops before any other operation
            let num_loops = kernel
                .ops
                .iter()
                .position(|kernel| !matches!(kernel, VOp::Loop { .. }))
                .unwrap();

            // If there is more loops than 3, pick first three loops as global loops,
            // rest is register loops.
            if num_loops > 3 {
                todo!()
            }
            // If there is less than three loops, add loops with dimension 1
            if num_loops < 3 {
                todo!()
            }

            // Then split those first three loops into global and local loops.
            // Reshape kernels to 6d (global, local) with some register loops
            // should be shape: [gws[0], lws[0], gws[1], lws[1], gws[2], lws[2]]
            let mut gws = [1; 3];
            let lws = [1; 3]; // TODO
            for op in &kernel.ops {
                if let VOp::Loop { axis, dimension } = op {
                    if *axis > 2 {
                        break;
                    }
                    gws[*axis] = *dimension;
                }
            }

            kernel.split_axis(0, &[gws[0], lws[0]]);
            kernel.split_axis(2, &[gws[1], lws[1]]);
            kernel.split_axis(4, &[gws[2], lws[2]]);

            #[cfg(feature = "debug1")]
            {
                println!();
                for op in &kernel.ops {
                    println!("{op:?}");
                }
                println!();
            }

            let ir_kernel = compile_ir(
                &graph,
                gws,
                lws,
                &kernel.inputs,
                &kernel.outputs,
                &kernel.ops,
                &self.hwinfo,
            );

            let args: Vec<TensorId> = ir_kernel.args.keys().copied().collect();
            programs.push((args.clone(), self.compiler.compile_program(&ir_kernel)?));

            // Allocate memory for intermediate args and results
            for arg in args.iter().copied() {
                if !self.buffers.contains_key(&arg) {
                    self.buffers.insert(
                        arg,
                        self.compiler.allocate_memory(
                            graph.shape(arg).iter().product::<usize>()
                                * graph.dtype(arg).byte_size(),
                        )?,
                    );
                }
            }
        }

        self.compiled_graphs.insert(
            org_graph.clone(),
            CompiledGraph {
                programs,
                //flop: 0,
                //bytes: 0,
            },
        );

        return Ok(());
    }

    pub(super) fn launch_graph(&mut self, graph: &Graph) -> Result<(), CompilerError> {
        let graph = self.compiled_graphs.get(graph).unwrap();

        for (args, program) in &graph.programs {
            let mut buffers = Vec::with_capacity(args.len());
            for arg in args {
                //libc_print::libc_println!("Argument: {arg}");
                let buffer = self.buffers.remove(arg).unwrap();
                buffers.push(buffer);
            }
            self.compiler.launch_program(&program, &mut buffers)?; // graph.flop, graph.bytes)
            for arg in args.iter().copied().rev() {
                self.buffers.insert(arg, buffers.pop().unwrap());
            }
        }

        return Ok(());
    }
}

type Axis = usize;
type Dimension = usize;
type Stride = usize;

#[derive(Debug)]
struct Kernel {
    // Current shape of the kernel after all current ops
    shape: Vec<Dimension>,
    // Global loads
    inputs: BTreeSet<TensorId>,
    // Global stores
    outputs: BTreeSet<TensorId>,
    // Register variables
    vars: BTreeSet<TensorId>,
    ops: Vec<VOp>,
}

impl Kernel {
    fn split_axis(&mut self, op_id: usize, dimensions: &[usize]) {
        // First split loop at op_id
        let VOp::Loop { axis, dimension } = &mut self.ops[op_id] else {
            panic!()
        };
        *dimension = dimensions[0];
        let axis = *axis;
        let mut temp_axis = axis;
        let mut id = op_id;
        for dim in dimensions[1..].iter() {
            id += 1;
            temp_axis += 1;
            self.ops.insert(
                id,
                VOp::Loop {
                    axis: temp_axis,
                    dimension: *dim,
                },
            )
        }
        let axis_shift = dimensions.len() - 1;
        let mut loop_ends = 0;
        for i in id + 1..self.ops.len() {
            match &mut self.ops[i] {
                // Then change axis ids for all following loops
                VOp::Loop { axis, .. } => {
                    *axis += axis_shift;
                }
                VOp::Reduce { axis, .. } => {
                    if loop_ends == axis_shift {
                        break;
                    }
                    loop_ends += 1;
                    *axis += axis_shift;
                }
                // Then change all load and store operations in this
                // loop in the same way.
                VOp::Load { view, .. } => {
                    let mut stride = view.0[axis].stride;
                    let mut temp_axis = axis + dimensions.len();
                    for dim in dimensions[1..].iter().copied().rev() {
                        temp_axis -= 1;
                        view.0.insert(
                            axis + 1,
                            ViewDim {
                                axis: temp_axis,
                                dim,
                                stride,
                                // TODO this will need to be different if len and shift weren't equal to dim before
                                len: dim,
                                shift: dim,
                            },
                        );
                        stride *= dim;
                    }
                    view.0[axis] = ViewDim {
                        axis,
                        dim: dimensions[0],
                        stride,
                        len: dimensions[0],
                        shift: dimensions[0],
                    };
                }
                VOp::Store { strides, .. } => {
                    // Example of axis split
                    // shape
                    //  2, 6,    2
                    //  2, 3, 2, 2
                    // strides
                    // 12, 2,    1
                    // 12, 4, 2, 1
                    let mut stride = strides[axis];
                    for dim in dimensions[1..].iter().rev() {
                        strides.insert(axis, stride);
                        stride *= dim;
                    }
                }
                _ => {}
            }
        }
    }
}

#[derive(Debug)]
struct View(Vec<ViewDim>);

impl View {
    fn new(shape: &[usize]) -> Self {
        let mut stride = 1;
        let mut view: Vec<ViewDim> = shape
            .iter()
            .enumerate()
            .rev()
            .map(|(axis, dim)| {
                let temp = stride;
                stride *= dim;
                ViewDim {
                    axis,
                    stride: temp,
                    dim: *dim,
                    len: *dim,
                    shift: *dim,
                }
            })
            .collect();
        view.reverse();
        return Self(view);
    }

    fn rank(&self) -> usize {
        self.0.len()
    }

    fn numel(&self) -> usize {
        self.0.iter().map(|dim| dim.dim).product()
    }

    fn permute(&mut self, axes: &[usize]) {
        assert_eq!(self.0.len(), axes.len());
        self.0 = axes.iter().map(|axis| self.0[*axis]).collect();
    }

    fn index(&self) -> Index {
        // TODO add index for padded views
        Index::Strided {
            dims: self.0.iter().map(|dim| (dim.axis, dim.stride)).collect(),
        }
    }
}

#[derive(Debug, Clone, Copy)]
struct ViewDim {
    axis: Axis,
    dim: Dimension,
    stride: Stride,
    len: usize,
    shift: usize,
}

#[derive(Debug)]
enum VOp {
    Load {
        z: TensorId,
        x: TensorId,
        view: View,
    },
    Store {
        z: TensorId,
        strides: Vec<Stride>,
    },
    Loop {
        axis: Axis,
        dimension: Dimension,
    },
    Accumulator {
        z: TensorId,
        rop: ROp,
    },
    Reduce {
        axis: Axis,
        rop: ROp,
        z: TensorId,
        x: TensorId,
    },
    Unary {
        z: TensorId,
        x: TensorId,
        uop: UOp,
    },
    Binary {
        z: TensorId,
        x: TensorId,
        y: TensorId,
        bop: BOp,
    },
}

fn generate_kernels(
    graph: &Graph,
    order: &[TensorId],
    to_eval: &BTreeSet<TensorId>,
) -> Vec<Kernel> {
    let mut kernels: Vec<Kernel> = Vec::new();
    for nid in order.iter().copied() {
        let node = &graph[nid];
        match node {
            Node::Leaf { shape, .. } => {
                let view = View::new(shape);
                let load_op = VOp::Load {
                    z: nid,
                    x: nid,
                    view,
                };
                if let Some(kernel) = kernels.iter_mut().find(|kernel| &kernel.shape == shape) {
                    kernel.ops.push(load_op);
                    kernel.inputs.insert(nid);
                    kernel.vars.insert(nid);
                } else {
                    let mut ops: Vec<VOp> = shape
                        .iter()
                        .copied()
                        .enumerate()
                        .map(|(axis, dimension)| VOp::Loop { axis, dimension })
                        .collect();
                    ops.push(load_op);
                    kernels.push(Kernel {
                        shape: shape.clone(),
                        inputs: BTreeSet::from([nid]),
                        outputs: BTreeSet::new(),
                        vars: BTreeSet::from([nid]),
                        ops,
                    });
                }
            }
            Node::Expand { x, shape } => {
                // Expand can just add loops
                // Expand means that global buffer is accessed multiple times. Thus we need to add caching (local, register) here.
                // Expand increases axes with dimension of 1 to bigger dimension
                // and sets strides in those axes to 0 for both loads and stores
                if let Some(kernel) = kernels.iter_mut().find(|kernel| kernel.vars.contains(x)) {
                    assert_eq!(kernel.shape.len(), shape.len());
                    let mut expand_axes = BTreeSet::new();
                    for a in 0..kernel.shape.len() {
                        if kernel.shape[a] != shape[a] {
                            assert_eq!(kernel.shape[a], 1);
                            kernel.shape[a] = shape[a];
                            expand_axes.insert(a);
                        }
                    }
                    // We go over ops in reverse, increasing last loops dimension
                    let mut done_expanding = BTreeSet::new();
                    for op in kernel.ops.iter_mut().rev() {
                        match op {
                            VOp::Loop { axis, dimension } => {
                                if expand_axes.contains(axis) && done_expanding.insert(*axis) {
                                    assert_eq!(*dimension, 1);
                                    *dimension = shape[*axis];
                                }
                            }
                            VOp::Load { view, .. } => {
                                // Done expanding marks which loops are behind us,
                                // so we need to only adjust strides to 0 in axes for those axes that are not behind us yet.
                                for a in expand_axes.difference(&done_expanding) {
                                    view.0[*a].dim = shape[*a];
                                    view.0[*a].stride = 0;
                                }
                            }
                            VOp::Store { strides, .. } => {
                                for a in expand_axes.difference(&done_expanding) {
                                    // TODO This will do multiple writes of the same value, so this would probably be better solved in different way,
                                    // perhaps doing only single write during the whole loop
                                    strides[*a] = 0;
                                }
                            }
                            _ => {}
                        }
                    }
                    kernel.ops.push(VOp::Unary {
                        z: nid,
                        x: *x,
                        uop: UOp::Noop,
                    });
                    kernel.vars.insert(nid);
                } else {
                    panic!()
                }
            }
            Node::Permute { x, axes, .. } => {
                // Permute shuffles load and store strides
                if let Some(kernel) = kernels.iter_mut().find(|kernel| kernel.vars.contains(x)) {
                    for op in &mut kernel.ops {
                        match op {
                            VOp::Load { view, .. } => {
                                view.permute(&axes[..view.rank()]);
                            }
                            VOp::Store { strides, .. } => {
                                *strides = axes[..strides.len()]
                                    .iter()
                                    .map(|axis| strides[*axis])
                                    .collect();
                            }
                            _ => {}
                        }
                    }
                    kernel.ops.push(VOp::Unary {
                        z: nid,
                        x: *x,
                        uop: UOp::Noop,
                    });
                    kernel.vars.insert(nid);
                } else {
                    panic!()
                }
            }
            Node::Reshape { x, shape } => {
                // Reshape always creates new kernel, as squeeze, unsqueeze and axis split
                // are already separate operations
                // If we really want, we can get reshape working with loads and stores
                // simply by using view for loads
                // But for now it is much simpler to just add new kernel

                if let Some(kernel) = kernels.iter_mut().find(|kernel| kernel.vars.contains(x)) {
                    // First if this is unsqueeze, add new loop with dimension 1 to the end of ops
                    // TODO

                    // else if we can split axes, split axes by replacing one loop with two loops
                    // TODO

                    // else create new kernel after storing results of previous kernel
                    let strides = shape_to_strides(graph.shape(*x));
                    kernel.ops.push(VOp::Store { z: *x, strides });
                    kernel.outputs.insert(*x);
                    kernels.push(Kernel {
                        shape: shape.clone(),
                        inputs: BTreeSet::from([*x]),
                        outputs: BTreeSet::new(),
                        vars: BTreeSet::from([nid]),
                        ops: vec![VOp::Load {
                            z: nid,
                            x: *x,
                            view: View::new(shape),
                        }],
                    });
                } else {
                    panic!()
                }
            }
            Node::Pad { x, pad, .. } => {
                // Pad shrinks or expands dimension of axes, but if there is store,
                // then it creates new kernel
                todo!()
            }
            Node::Reduce { x, axes, rop, .. } => {
                // Reduce removes loops and adds accumulator before those loops that it removes
                if let Some(kernel) = kernels.iter_mut().find(|kernel| kernel.vars.contains(x)) {
                    let mut axes_order = Vec::new();
                    // Add accumulator
                    let mut i = kernel.ops.len();
                    while i > 0 {
                        i -= 1;
                        if let VOp::Loop { axis, .. } = &kernel.ops[i] {
                            if axes.contains(axis) {
                                axes_order.push(*axis);
                                kernel.ops.insert(i, VOp::Accumulator { z: nid, rop: *rop });
                            }
                        }
                    }
                    // End loop
                    for axis in axes_order {
                        kernel.ops.push(VOp::Reduce {
                            axis,
                            rop: *rop,
                            z: nid,
                            x: *x,
                        });
                    }
                } else {
                    panic!()
                }
            }
            Node::Unary { x, uop } => {
                if let Some(kernel) = kernels.iter_mut().find(|kernel| kernel.vars.contains(x)) {
                    kernel.ops.push(VOp::Unary {
                        z: nid,
                        x: *x,
                        uop: *uop,
                    });
                    kernel.vars.insert(nid);
                } else {
                    panic!()
                }
            }
            Node::Binary { x, y, bop } => {
                // Binary ops may allow us to join two kernels together
                if let Some(kernel) = kernels
                    .iter_mut()
                    .find(|kernel| kernel.vars.contains(x) && kernel.vars.contains(y))
                {
                    // If both inputs are in the same kernel
                    kernel.ops.push(VOp::Binary {
                        z: nid,
                        x: *x,
                        y: *y,
                        bop: *bop,
                    });
                    kernel.vars.insert(nid);
                } else if let Some(kernel_x_id) =
                    kernels.iter().position(|kernel| kernel.vars.contains(x))
                {
                    if let Some(kernel_y_id) =
                        kernels.iter().position(|kernel| kernel.vars.contains(y))
                    {
                        // Two separate kernels contain our inputs, so we join them together
                        // TODO do some checks that this join is always valid
                        let kernel_y = kernels.remove(kernel_y_id);
                        let kernel_x = &mut kernels[kernel_x_id];
                        assert_eq!(kernel_x.shape, kernel_y.shape);
                        // TODO Here we must do something about the loops
                        // We cannot have both loops from kernel_x and kernel_y
                        // We have to remove one set of loops
                        kernel_x.ops.extend(kernel_y.ops);
                        kernel_x.ops.push(VOp::Binary {
                            z: nid,
                            x: *x,
                            y: *y,
                            bop: *bop,
                        })
                    } else {
                        panic!()
                    }
                } else {
                    panic!()
                }
            }
            Node::Where { x, y, z } => {
                todo!()
            }
        }
        if to_eval.contains(&nid) {
            if let Some(kernel) = kernels.iter_mut().find(|kernel| kernel.vars.contains(&nid)) {
                kernel.ops.push(VOp::Store {
                    z: nid,
                    strides: shape_to_strides(graph.shape(nid)),
                });
            } else {
                panic!()
            }
        }
    }
    return kernels;
}

fn shape_to_strides(shape: &[usize]) -> Vec<usize> {
    let mut stride = 1;
    let mut strides: Vec<usize> = shape
        .iter()
        .rev()
        .map(|d| {
            let temp = stride;
            stride *= d;
            temp
        })
        .collect();
    strides.reverse();
    return strides;
}

#[derive(Debug)]
pub(in crate::runtime) struct IRKernel {
    pub(super) global_work_size: [usize; 3],
    pub(super) local_work_size: [usize; 3],
    pub(super) args: BTreeMap<TensorId, IRArg>,
    pub(super) ops: Vec<IROp>,
}

#[derive(Debug, Clone, Copy)]
pub(super) struct IRArg {
    pub(super) dtype: DType,
    pub(super) read_only: bool,
}

#[derive(Debug, Clone)]
pub(super) enum IRMem {
    Const(Constant),
    Var {
        id: usize,
        scope: Scope,
        index: Index,
    },
}

impl IRMem {
    pub(super) fn to_str(&self, temp_id: u32) -> (Vec<String>, String) {
        match self {
            IRMem::Const(value) => {
                return (
                    Vec::new(),
                    match value {
                        Constant::F32(value) => {
                            f!("{}", unsafe { core::mem::transmute::<u32, f32>(*value) })
                        }
                        Constant::I32(value) => f!("{}", value),
                        _ => todo!(),
                    },
                )
            }
            IRMem::Var { id, scope, index } => match index {
                Index::Contiguous { dims } | Index::Strided { dims } => {
                    let mut res = String::new();
                    for (id, mul) in dims {
                        res += &f!("i{id}*{mul}+");
                    }
                    res.pop();
                    return (Vec::new(), f!("{}{}[{res}]", scope, id));
                }
                /*Index::Reshaped { dims, reshapes, .. } => {
                let mut res = String::new();
                for (id, mul) in dims {
                    res += &f!("i{id}*{mul}+");
                }
                res.pop();
                let mut res = vec![res];
                for reshape in reshapes[..reshapes.len() - 1].iter() {
                    let mut idx = String::new();
                    for (div, m, mul) in reshape.iter() {
                        idx += &f!("t{temp_id}/{div}%{m}*{mul}+");
                    }
                    idx.pop();
                    res.push(idx);
                }
                let mut idx = String::new();
                for (div, m, mul) in reshapes.last().unwrap().iter() {
                    idx += &f!("t{temp_id}/{div}%{m}*{mul}+");
                }
                idx.pop();
                return (res, f!("{}{}[{idx}]", scope, id));
                }*/
                Index::None => return (Vec::new(), f!("{}{}", scope, id)),
            },
        }
    }
}

/// IROp for direct translation to hardware kernels
#[derive(Debug, Clone)]
pub(super) enum IROp {
    // All variables are 1d, so that it is easier for implementors
    DeclareMem {
        id: usize,
        scope: Scope,
        dtype: DType,
        read_only: bool,
        len: usize,
        // Initialization is mostly for accumulators
        init: Option<Constant>,
    },
    AssignMem {
        z: IRMem,
        x: IRMem,
    },
    /// Multiple successive unary ops on register variables
    Unary {
        z: IRMem,
        x: IRMem,
        ops: Vec<UOp>,
    },
    /// Single binary op on register variables, x is scalar
    Binary {
        z: IRMem,
        x: IRMem,
        y: IRMem,
        op: BOp,
    },
    /// Register loop, len is number of iterations, step is 1
    Loop {
        id: usize,
        len: usize,
    },
    /// End of register loop
    EndLoop,
    /// Synchronization barrier
    Barrier {
        scope: Scope,
    },
}

// Movement op, simply changes the view of this buffer. This means moving things around in memory
// and thus is extremely expensive. We should use memory caching here if possible.
// Things can be also moved between different memory scopes.

// Optimation instructions, implementation is hardware specific and thus is up to the compiler
// Matmul of two 16x16 tiles, result is also 16x16 tile stored in local memory

/// Rewrite tiled representation to ir representation, optionally fuse some kernels if possible
/// (if they have the same work size)
pub(crate) fn compile_ir(
    graph: &Graph,
    global_work_size: [usize; 3],
    local_work_size: [usize; 3],
    inputs: &BTreeSet<TensorId>,
    outputs: &BTreeSet<TensorId>,
    vops: &[VOp],
    hwinfo: &HWInfo,
) -> IRKernel {
    // Here tiles get rewritten into tiles and loops, dimensions get bound
    // and optimizations applied. At this stage, all movement and reduce ops are removed.
    // Also, there will be special instructions for applying optimizations on like 4x4x4
    // matmuls (like strassen or tensor cores) or 16x16x16 matmul (wmma).
    // These optimizations are hardware dependent.
    let _ = hwinfo;
    let gws = global_work_size;
    let lws = local_work_size;

    let mut ops = Vec::new();

    // Remove first 6 loops, these are global loops.

    for vop in &vops[6..] {
        match vop {
            VOp::Load { z, x, view } => {
                ops.push(IROp::DeclareMem {
                    id: *z,
                    scope: Scope::Register,
                    dtype: graph.dtype(*z),
                    read_only: false,
                    len: 0,
                    init: None,
                });
                ops.push(IROp::AssignMem {
                    z: IRMem::Var {
                        id: *z,
                        scope: Scope::Register,
                        index: Index::None,
                    },
                    x: IRMem::Var {
                        id: *x,
                        scope: Scope::Global,
                        index: view.index(),
                    },
                });
            }
            VOp::Store { z, strides } => {
                ops.push(IROp::AssignMem {
                    z: IRMem::Var {
                        id: *z,
                        scope: Scope::Global,
                        index: Index::Strided {
                            dims: strides.iter().copied().enumerate().collect(),
                        },
                    },
                    x: IRMem::Var {
                        id: *z,
                        scope: Scope::Register,
                        index: Index::None,
                    },
                });
            }
            VOp::Loop { axis, dimension } => {
                ops.push(IROp::Loop {
                    id: *axis,
                    len: *dimension,
                });
            }
            VOp::Accumulator { z, rop } => {
                let dtype = graph.dtype(*z);
                ops.push(IROp::DeclareMem {
                    id: *z,
                    scope: Scope::Register,
                    dtype,
                    read_only: false,
                    len: 0,
                    init: Some(match rop {
                        ROp::Sum => dtype.zero_constant(),
                        ROp::Max => dtype.min_constant(),
                    }),
                });
            }
            VOp::Reduce { axis, rop, z, x } => todo!(),
            VOp::Unary { z, x, uop } => {
                ops.push(IROp::DeclareMem {
                    id: *z,
                    scope: Scope::Register,
                    dtype: graph.dtype(*z),
                    read_only: false,
                    len: 0,
                    init: None,
                });
                ops.push(IROp::Unary {
                    z: IRMem::Var {
                        id: *z,
                        scope: Scope::Register,
                        index: Index::None,
                    },
                    x: IRMem::Var {
                        id: *x,
                        scope: Scope::Register,
                        index: Index::None,
                    },
                    ops: vec![*uop],
                });
            }
            VOp::Binary { z, x, y, bop } => {
                ops.push(IROp::DeclareMem {
                    id: *z,
                    scope: Scope::Register,
                    dtype: graph.dtype(*z),
                    read_only: false,
                    len: 0,
                    init: None,
                });
                ops.push(IROp::Binary {
                    z: IRMem::Var {
                        id: *z,
                        scope: Scope::Register,
                        index: Index::None,
                    },
                    x: IRMem::Var {
                        id: *x,
                        scope: Scope::Register,
                        index: Index::None,
                    },
                    y: IRMem::Var {
                        id: *y,
                        scope: Scope::Register,
                        index: Index::None,
                    },
                    op: *bop,
                });
            }
        }
    }

    let mut args = BTreeMap::new();
    for x in inputs {
        args.insert(
            *x,
            IRArg {
                dtype: graph.dtype(*x),
                read_only: true,
            },
        );
    }
    for x in outputs {
        args.insert(
            *x,
            IRArg {
                dtype: graph.dtype(*x),
                read_only: false,
            },
        );
    }

    return IRKernel {
        global_work_size,
        local_work_size,
        ops,
        args,
    };
}

// With this representation of index, we can find repeating
// multipliers and extract them out into common factors.
// However this would be a bit of micro-optimization, as OpenCL, CUDA, WGPU
// and most other compilers extract them automatically.
// This will be needed if we want to directly generate SPIR or PTX IR.

/// Virtual representation of index into view
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Index {
    /// For variables that only have single element (scalars),
    /// such as most register variables.
    None,
    /// Pairs of index id and multiplier.
    /// Can use wide loads directly with pointer casts.
    Contiguous {
        /// Dimension and multiplier
        dims: BTreeMap<usize, usize>,
        // When should the padding get applied?
        //padding_condition: String,
    },
    /// Expanded and/or permuted
    /// Pairs of index id and multiplier.
    /// Wide loads are possible only if we can transpose it in the kernel
    Strided {
        /// Dimension and multiplier
        dims: BTreeMap<usize, usize>,
        // When should the padding get applied?
        //padding_condition: String,
    },
    // Expanded, permuted and/or padded
    // Only if reshape could not be merged.
    /*Padded {
    /// Multiple dimension and multipliers
    dims: BTreeMap<usize, usize>,
    /// When should the padding get applied?
    padding_condition: String,
    },*/
}
