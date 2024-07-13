use crate::runtime::compiler::ir::IRKernel;
use crate::runtime::node::Node;
use crate::runtime::view::View;
use crate::runtime::{Subgraph, TensorId};
use crate::scalar::Scalar;
use crate::DType;
use alloc::collections::{BTreeMap, BTreeSet};
use alloc::vec::Vec;
use core::fmt::{Display, Formatter};

#[cfg(feature = "debug1")]
use libc_print::std_name::println;

mod compile;
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
    compiled_graphs: BTreeMap<BTreeMap<TensorId, Node>, CompiledGraph<C::Program>>,
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
        org_graph: &Subgraph,
        to_eval: BTreeSet<TensorId>,
    ) -> Result<(), CompilerError> {
        //#[cfg(feature = "debug1")]
        //libc_print::libc_println!("Evaluating {to_eval:?}");
        if self.compiled_graphs.contains_key(&org_graph.nodes) {
            return Ok(());
        }
        let mut graph = org_graph.clone();
        //println!("{subgraph:?}");
        //println!("{hw_info:?}");
        // Find the best order of execution of nodes
        let rcs = calculate_graph_rcs(&graph, &to_eval);
        let mut order = calculate_graph_execution_order(&graph, &to_eval, &rcs);

        // Reorder nodes in such a way, that movement ops are as late as possible,
        // after all unary ops just before reduce ops. (Do not reorder it after binary ops though.)
        let mut node_swap = true;
        while node_swap {
            node_swap = false;
            for nid in order.iter().take(order.len() - 1) {
                if graph.nodes[nid].is_movement() && graph.nodes[&(nid + 1)].is_unary() {
                    //libc_print::libc_println!("Reordering movement and unary ops, swap {} and {}", nid, nid+1);
                    graph.swap_nodes(*nid, nid + 1);
                    node_swap = true;
                }
            }
        }

        let mut kernels = compile::create_kernels(
            &self.buffers.keys().copied().collect(),
            &graph,
            to_eval,
            &order,
        );

        for kernel in &kernels {
            #[cfg(feature = "debug1")]
            println!("\n{kernel:?}\n");
        }

        // Create kernels function merges all mergeable ops together, creates final groups of ops
        // and includes reduce loops. Shapes are still original and not changed.

        // Now we need to optimize shapes to 8d non reduce and 10d in reduce loops
        for kernel in &mut kernels {
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
        }

        for kernel in &kernels {
            #[cfg(feature = "debug1")]
            println!("\n{kernel:?}\n");
        }

        // TODO there are some advanced optimizations that can be additionally applied.
        // Namely some movement operations could be done without the need for global temporary
        // variable, but these can be added in later.

        let mut programs = Vec::new();

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
        );

        Ok(())
    }

    pub(super) fn launch_graph(
        &mut self,
        graph: &BTreeMap<TensorId, Node>,
    ) -> Result<(), CompilerError> {
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

fn calculate_graph_rcs(
    subgraph: &Subgraph,
    to_eval: &BTreeSet<TensorId>,
) -> BTreeMap<TensorId, u32> {
    // Depth first search through graph. Number of visits of each node are reference counts.
    let mut visited_rcs: BTreeMap<TensorId, u32> = BTreeMap::new();
    let mut params: Vec<TensorId> = to_eval.iter().copied().collect();
    params.reserve(100);
    while let Some(nid) = params.pop() {
        //std::println!("{nid} is evaluated: {}", self.runtime_backend.is_evaluated(nid));
        visited_rcs
            .entry(nid)
            .and_modify(|rc| *rc += 1)
            .or_insert_with(|| {
                params.extend(subgraph[nid].parameters());
                1
            });
    }
    //println!("Temp: {visited_rcs:?}");
    return visited_rcs;
}

fn calculate_graph_execution_order(
    graph: &Subgraph,
    to_eval: &BTreeSet<TensorId>,
    temp_rcs: &BTreeMap<TensorId, u32>,
) -> Vec<TensorId> {
    // Calculates dependency graph of nodes and viable execution order, which is not currently
    // optimized. It is depth first search. On each visit of the node, rc is increased. Once
    // rc of the node A reaches A's rc in the whole graph, then A gets added to the order,
    // that is, there are no more nodes that node A depends on, i.e. there are no nodes that
    // need to be evaluated before A.
    let mut order = Vec::new();
    let mut rcs: BTreeMap<TensorId, u32> = BTreeMap::new();
    let mut params: Vec<TensorId> = to_eval.iter().copied().collect();
    params.reserve(100);
    while let Some(nid) = params.pop() {
        if let Some(temp_rc) = temp_rcs.get(&nid) {
            let rc = rcs.entry(nid).and_modify(|rc| *rc += 1).or_insert(1);
            if *temp_rc == *rc {
                order.push(nid);
                params.extend(graph[nid].parameters());
            }
        }
    }
    order.reverse();
    return order;
}

// This is basically just a kernel represented as bunch of tiles with common global and local work
// size.
struct TiledKernel {
    global_work_size: [usize; 3],
    local_work_size: [usize; 3],
    tiles: Vec<Tile>,
}

// Movement operation can be skipped for now, perhaps later added for some register ->
// local -> register movement operations without storing to global, but that is complex.
// Load and reduce are currently only operations that contain movement ops.
#[derive(Debug, Clone)]
enum Tile {
    // Load from global to register
    Load {
        z: TensorId,
        x: TensorId,
        view: View,
        dtype: DType,
    },
    // Store from register to global, contiguous
    Store {
        z: TensorId,
        dtype: DType,
    },
    // Multiple unary ops applied on x, resulting in z
    Unary {
        z: TensorId,
        x: TensorId,
        z_dtype: DType,
        ops: Vec<UOp>,
    },
    // Binary operation
    Binary {
        z: TensorId,
        x: TensorId,
        y: TensorId,
        op: BOp,
    },
    // Reduce operation begin loop, z is accumulator
    ReduceBegin {
        z: TensorId,
        z_dtype: DType,
        op: ROp,
        r_dim: usize,
    },
    // Reduce operation end loop, z is accumulator
    ReduceEnd {
        z: TensorId,
        x: TensorId,
        op: ROp,
    },
}

#[derive(Debug, Clone, Copy)]
enum UOp {
    Inv,
    Neg,
    Sin,
    Cos,
    Exp,
    Ln,
    Sqrt,
    Cast(DType),
}

#[derive(Debug, Clone, Copy)]
enum BOp {
    Add,
    Sub,
    Mul,
    Div,
    Pow,
    Cmplt,
    Max, // for ReLU and max reduce
}

#[derive(Debug, Clone, Copy)]
enum ROp {
    Sum,
    Max,
}
