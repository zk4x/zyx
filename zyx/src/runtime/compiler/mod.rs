use crate::runtime::compiler::ir::IRKernel;
use crate::runtime::TensorId;
use crate::scalar::Scalar;
use alloc::collections::{BTreeMap, BTreeSet};
use alloc::vec::Vec;
use core::fmt::{Display, Formatter};

#[cfg(feature = "debug1")]
use libc_print::std_name::println;

use super::graph::Graph;
use super::node::{BOp, Node, ROp, UOp};

mod ir;

#[cfg(feature = "cuda")]
pub(super) mod cuda;

#[cfg(feature = "opencl")]
pub(super) mod opencl;

#[cfg(feature = "wgpu")]
pub(super) mod wgpu;

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
    flop: usize,
    bytes: usize,
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
        graph: &Graph,
        to_eval: BTreeSet<TensorId>,
    ) -> Result<(), CompilerError> {
        //#[cfg(feature = "debug1")]
        //libc_print::libc_println!("Evaluating {to_eval:?}");
        if self.compiled_graphs.contains_key(&graph) {
            return Ok(());
        }
        let mut graph = graph.clone();
        let order = graph.execution_order(&to_eval);
        let kernels = generate_kernels(&graph, &order, &to_eval);

        #[cfg(feature = "debug1")]
        {
            for kernel in kernels {
                println!();
                for op in &kernel.ops {
                    println!("{op:?}");
                }
                println!();
            }
        }

        let _ = kernels;

        // Now we need to optimize shapes to 8d non reduce and 10d in reduce loops
        /*for kernel in &mut kernels {
            let mut kernel_view = View::from(&kernel.shape);

            // Make all kernels 3d
            let mut sh = kernel.shape.clone();
            while sh.len() < 3 {
                sh.insert(0, 1);
            }
            let rank = sh.len();
            if rank > 3 {
                let sh = [sh[..rank - 2].iter().product(), sh[rank - 2], sh[rank - 1]];
                kernel_view.reshape(&sh);
            } else {
                kernel_view.reshape(&sh);
            }
            kernel_view.optimize_local_mem_size_and_work_per_thread(&self.hwinfo);
            let sh = kernel_view.shape();
            kernel.shape = sh.clone();

            let mut reduce_data: Option<(Vec<usize>, Vec<usize>)> = None;
            for op in &mut kernel.ops {
                // Make all loads outside of reduce loops 3d, in reduce loops 4d
                match op {
                    compile::Op::Load { view, .. } => {
                        if let Some((shape, axes)) = &reduce_data {
                            view.permute(axes);
                            view.reshape(shape);
                        } else {
                            view.reshape(&sh);
                        }
                    }
                    compile::Op::ReduceLoop {
                        axes,
                        shape_before_reduce,
                        ..
                    } => {
                        let all_axes: Vec<usize> = (0..shape_before_reduce.len())
                            .filter(|a| !axes.contains(a))
                            .chain(axes.iter().copied())
                            .collect();

                        let mut view = View::from(&shape_before_reduce);
                        view.permute(&all_axes);
                        let sh = view.shape();

                        let r_dim = axes.iter().map(|a| shape_before_reduce[*a]).product();
                        // Reshape to join reduce axes and make it 4d
                        let mut sh: Vec<usize> = sh[..sh.len() - axes.len()]
                            .iter()
                            .copied()
                            .chain([r_dim])
                            .collect();
                        while sh.len() < 4 {
                            sh.insert(0, 1);
                        }
                        let rank = sh.len();
                        if rank > 4 {
                            let sh = [
                                sh[..rank - 3].iter().product(),
                                sh[rank - 3],
                                sh[rank - 2],
                                sh[rank - 1],
                            ];
                            view.reshape(&sh);
                        } else {
                            view.reshape(&sh);
                        }
                        view.optimize_local_mem_size_and_work_per_thread(&self.hwinfo);
                        let sh = view.shape();
                        // Set shape_before_reduce to be just reduce dims, now it is over last dimensions
                        *shape_before_reduce = alloc::vec![sh[8], sh[9]];
                        reduce_data = Some((sh, all_axes));
                    }
                    compile::Op::Reduce { .. } => {
                        reduce_data = None;
                    }
                    _ => {}
                }
            }
        }*/

        // TODO there are some advanced optimizations that can be additionally applied.
        // Namely some movement operations could be done without the need for global temporary
        // variable, but these can be added in later.

        /*let mut programs = Vec::new();

        // Rewrite tiled representation to IR representation and compile it for HW device
        #[cfg(feature = "debug1")]
        println!("Compiling kernels");
        // TODO kernels can be compiled in parallel
        for kernel in kernels {
            let ir_kernel = ir::tiled_to_ir(
                kernel.global_work_size,
                kernel.local_work_size,
                kernel.ops,
                &self.hwinfo,
            );

            let args: Vec<TensorId> = ir_kernel.args.keys().copied().collect();

            // Compile kernel
            //libc_print::libc_println!("Program with args: {:?}", args);
            // BEWARE in which order compiled kernels are pushed into programs,
            // as lanuching graph executes them in this same order FIFO
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
            org_graph.nodes.clone(),
            CompiledGraph {
                programs,
                flop: 0,
                bytes: 0,
            },
        );*/

        Ok(())
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

#[derive(Debug)]
struct View(Vec<ViewDim>);

#[derive(Debug)]
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
        strides: Vec<(Axis, Stride)>,
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
                if let Some(kernel) = kernels.iter_mut().find(|kernel| &kernel.shape == shape) {
                    let mut stride = 1;
                    let mut strides: Vec<(Axis, Dimension)> = shape
                        .iter()
                        .enumerate()
                        .rev()
                        .map(|(axis, dimension)| {
                            let temp = stride;
                            stride *= dimension;
                            (axis, temp)
                        })
                        .collect();
                    strides.reverse();
                    kernel.ops.push(VOp::Load {
                        z: nid,
                        x: nid,
                        strides,
                    });
                    kernel.inputs.insert(nid);
                    kernel.vars.insert(nid);
                } else {
                    let mut ops: Vec<VOp> = shape
                        .iter()
                        .copied()
                        .enumerate()
                        .map(|(axis, dimension)| VOp::Loop { axis, dimension })
                        .collect();
                    let mut stride = 1;
                    let mut strides: Vec<(Axis, Dimension)> = shape
                        .iter()
                        .enumerate()
                        .rev()
                        .map(|(axis, dimension)| {
                            let temp = stride;
                            stride *= dimension;
                            (axis, temp)
                        })
                        .collect();
                    strides.reverse();
                    ops.push(VOp::Load {
                        z: nid,
                        x: nid,
                        strides,
                    });
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
                todo!()
            }
            Node::Permute { x, axes, .. } => {
                // Permute shuffles load and store strides
                if let Some(kernel) = kernels.iter_mut().find(|kernel| kernel.vars.contains(x)) {
                    for op in kernel.ops {
                        match op {
                            VOp::Load { z, x, view } => {
                                view.permute();
                            }
                            VOp::Store { z: (), strides: () } => {
                                strides.permute();
                            }
                            _ => {}
                        }
                    }
                } else {
                    panic!()
                }
            }
            /*Node::Split { axis, dimensions } => {
            // Split axis into multiple axes
            if let Some(kernel) = kernels.iter_mut().find(|kernel| kernel.vars.contains(x)) {
                for axis in axes {
                    kernel.ops.push(Op::Loop {
                        axis: *axis,
                        dimension: 1,
                    });
                }
            } else {
                panic!()
            }
            }*/
            Node::Reshape { x, shape } => {
                // Reshape always creates new kernel, as squeeze, unsqueeze and axis split
                // are already separate operations
                // If we really want, we can get reshape working with loads and stores
                // simply by using view for loads
                // But for now it is much simpler to just add new kernel

                // First if we can split axes, split axes

                // else create new kernel
                todo!()
            }
            Node::Pad { x, pad, .. } => {
                // Pad shrinks or expands dimension of axes, but if there is store,
                // then it creates new kernel
                todo!()
            }
            Node::Unary { x, uop } => {
                if let Some(kernel) = kernels.iter_mut().find(|kernel| kernel.vars.contains(x)) {
                    kernel.ops.push(VOp::Unary {
                        z: nid,
                        x: *x,
                        uop: UOp::Exp,
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
                        // Here we must do something about the loops
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
            Node::Reduce { x, axes, rop, .. } => {
                // Reduce removes loops and adds accumulator before those loops that it removes
                todo!()
            }
        }
    }
    return kernels;
}
