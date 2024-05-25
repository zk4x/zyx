use alloc::collections::{BTreeMap, BTreeSet};
use alloc::vec;
use crate::scalar::Scalar;
use alloc::vec::Vec;
use crate::DType;
use crate::runtime::compiler::ir::IRKernel;
use crate::runtime::node::Node;
use crate::runtime::{Graph, TensorId};
use crate::runtime::view::View;

pub(super) mod opencl;
pub(super) mod cuda;
pub(super) mod wgpu;
mod ir;

trait Compiler: Sized {
    type Buffer;
    type Program;
    fn initialize() -> Result<Self, CompilerError>;
    fn hwinfo(&mut self) -> Result<HWInfo, CompilerError>;
    fn allocate_memory(&mut self, byte_size: usize) -> Result<Self::Buffer, CompilerError>;
    fn store_memory<T>(&mut self, buffer: &mut Self::Buffer, data: &[T]) -> Result<(), CompilerError>;
    fn load_mem<T>(&self, buffer: &Self::Buffer, length: usize) -> Result<Vec<T>, CompilerError>;
    fn deallocate_memory(&mut self, buffer: Self::Buffer);
    fn compile_program(&mut self, kernel: &IRKernel) -> Result<Self::Program, CompilerError>;
    fn launch_program(&mut self, program: &Self::Program, args: &[&mut Self::Buffer]) -> Result<(), CompilerError>;
    fn drop_program(&mut self, program: Self::Program);
}

pub(super) struct CompiledBackend<C: Compiler> {
    compiler: C,
    buffers: BTreeMap<TensorId, C::Buffer>,
    compiled_graphs: BTreeMap<BTreeMap<TensorId, Node>, CompiledGraph<C::Program>>,
}

/// Compiled graph
pub(super) struct CompiledGraph<Program> {
    args: Vec<TensorId>,
    program: Program,
    flop: usize,
    bytes: usize,
}

#[derive(Debug)]
pub(crate) enum CompilerError {
    InitializationFailure,
    DeviceOutOfMemory,
    HostOutOfMemory,
    BufferDoesNotExist,
    // For all unknown errors
    GeneralExecutionError,
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
}

impl<C: Compiler> CompiledBackend<C> {
    pub(super) fn initialize() -> Result<Self, CompilerError> {
        Ok(Self {
            compiler: C::initialize()?,
            buffers: BTreeMap::new(),
            compiled_graphs: BTreeMap::new(),
        })
    }

    pub(super) fn is_realized(&self, x: TensorId) -> bool {
        self.buffers.contains_key(&x)
    }

    pub(super) fn store<T: Scalar>(&mut self, data: &[T]) -> Result<(), CompilerError> {
        todo!()
    }

    // Load values at x, if x is not evaluated, it will return error
    pub(super) fn load<T: Scalar>(&self, x: TensorId, length: usize) -> Result<Vec<T>, CompilerError> {
        if let Some(buffer) = self.buffers.get(&x) {
            return self.compiler.load_mem(buffer, length)
        }
        return Err(CompilerError::BufferDoesNotExist)
    }

    pub(super) fn remove(&mut self, x: TensorId) {
        if let Some(buffer) = self.buffers.remove(&x) {
            self.compiler.deallocate_memory(buffer);
        }
    }

    /// Compiles and caches graph
    pub(super) fn compile_graph(&mut self, graph: &Graph, subgraph: BTreeMap<TensorId, Node>, to_eval: BTreeSet<TensorId>) -> Result<(), CompilerError> {
        //std::println!("{hw_info:?}");
        // Find the best order of execution of nodes
        let (order, rcs) = {
            let mut temp_rcs: BTreeMap<TensorId, u32> = BTreeMap::new();
            let mut params: Vec<TensorId> = to_eval.iter().copied().collect();
            params.reserve(100);
            while let Some(nid) = params.pop() {
                //std::println!("{nid} is evaluated: {}", self.runtime_backend.is_evaluated(nid));
                temp_rcs
                    .entry(nid)
                    .and_modify(|rc| *rc += 1)
                    .or_insert_with(|| {
                        if graph.is_empty(nid) {
                            params.extend(subgraph[&nid].parameters());
                        }
                        1
                    });
            }
            // Order them using rcs reference counts.
            let mut order = Vec::new();
            let mut rcs: BTreeMap<TensorId, u32> = BTreeMap::new();
            let mut params: Vec<TensorId> = to_eval.iter().copied().collect();
            params.reserve(100);
            while let Some(nid) = params.pop() {
                if let Some(temp_rc) = temp_rcs.get(&nid) {
                    let rc = rcs.entry(nid).and_modify(|rc| *rc += 1).or_insert(1);
                    if *temp_rc == *rc {
                        order.push(nid);
                        params.extend(subgraph[&nid].parameters());
                    }
                }
            }
            order.reverse();
            (order, rcs)
        };

        // First reorder nodes in such a way, that movement ops are as late as possible,
        // after all unary ops just before reduce ops. (Do not reorder it after binary ops though.)
        // Permutes after reduce ops can be also moved before reduce ops!

        // Global work sizes are known as the shapes of the reduce kernels!

        let mut tiles = BTreeMap::new();

        for nid in order.iter().copied() {
            //std::println!("{:?}", nodes[nid.i()]);
            if self.buffers.contains_key(&nid) {
                let dtype = graph.dtype(nid);
                tiles.insert(nid, Tile {
                    view: View::from(&graph.shape(nid)),
                    dtype,
                    scope: 2,
                    first_op: FirstOp::Load { dtype },
                    ops: Vec::new(),
                    can_be_fused: rcs[&nid] < 2,
                });
                continue;
            }
            match &subgraph[&nid] {
                Node::Const { .. } => {}
                Node::Leaf { .. } => {}
                Node::Cast { .. } => {}
                Node::Exp { x } => {
                    //libc_print::libc_println!("{x}");
                    let mut tile = tiles[x].clone();
                    tile.ops.push(UOp::Exp);
                    tiles.insert(nid, tile);
                }
                Node::Add { x, y } => {
                    let tile = Tile {
                        view: View::from(&graph.shape(nid)),
                        dtype: graph.dtype(nid),
                        scope: 2,
                        first_op: FirstOp::Binary {
                            x: *x,
                            y: *y,
                            op: BOp::Add,
                        },
                        ops: vec![],
                        can_be_fused: rcs[&nid] < 2,
                    };
                    tiles.insert(nid, tile);
                }
                Node::Reshape { x, shape_id } => {
                    match &tiles[x].first_op {
                        FirstOp::Reduce { .. } => {
                            let tile = Tile {
                                view: View::from(graph._shape(*shape_id)),
                                dtype: graph.dtype(nid),
                                scope: 2,
                                first_op: FirstOp::Movement { x: *x },
                                ops: vec![],
                                can_be_fused: rcs[&nid] < 2,
                            };
                            tiles.insert(nid, tile);
                        }
                        _ => {
                            let mut tile = tiles[x].clone();
                            tile.view.reshape(graph._shape(*shape_id));
                            tiles.insert(nid, tile);
                        }
                    }
                }
                Node::Expand { x, shape_id } => {
                    let mut view = View::from(&graph.shape(*x));
                    view.expand(graph._shape(*shape_id));
                    let tile = Tile {
                        view,
                        dtype: graph.dtype(nid),
                        scope: 2,
                        first_op: FirstOp::Movement { x: *x },
                        ops: vec![],
                        can_be_fused: rcs[&nid] < 2,
                    };
                    tiles.insert(nid, tile);
                }
                Node::Permute { .. } => {
                    // Permute permutes all views.
                    // In case of reduce kernel, permute also axes and shape before reduce.
                }
                Node::Pad { x, shape_id, padding_id } => {
                    let padding = graph._padding(*padding_id);
                    match &tiles[x].first_op {
                        FirstOp::Reduce { .. } => {
                            let mut view = View::from(&graph.shape(*x));
                            view.pad(padding);
                            let tile = Tile {
                                view,
                                dtype: graph.dtype(nid),
                                scope: 2,
                                first_op: FirstOp::Movement { x: *x },
                                ops: vec![],
                                can_be_fused: rcs[&nid] < 2,
                            };
                            tiles.insert(nid, tile);
                        }
                        _ => {
                            let mut tile = tiles[x].clone();
                            tile.view.pad(padding);
                            tiles.insert(nid, tile);
                        }
                    }
                }
                Node::Sum { x, axes_id, shape_id } => {
                    let tile = Tile {
                        view: View::from(graph._shape(*shape_id)),
                        dtype: graph.dtype(nid),
                        scope: 2,
                        first_op: FirstOp::Reduce {
                            x: *x,
                            shape: graph._shape(*shape_id).into(),
                            axes: graph._axes(*axes_id).into(),
                            op: ROp::Sum
                        },
                        ops: vec![],
                        can_be_fused: rcs[&nid] < 2,
                    };
                    tiles.insert(nid, tile);
                }
                _ => {
                    todo!()
                }
            }
        }

        // Print AST
        /*for tile in &tiles {
            std::println!("{tile:?}");
        }*/

        // Reshape and permute tiles to use exactly work 3 dimensions
        // and at most single reduce loop over the last dimension>

        for (id, tile) in &mut tiles {
            match tile.view.rank() {
                1 => match &mut tile.first_op {
                    FirstOp::Reduce { axes, .. } => {
                        // reshape to 4d, last dim reduce
                        debug_assert_eq!(axes.len(), 1);
                        *axes = vec![3];
                        tile.view.reshape(&[1, 1, 1, tile.view.numel()]);
                    }
                    _ => {
                        // reshape to 3d
                        tile.view.reshape(&[1, 1, tile.view.numel()]);
                    }
                }
                2 => match &tile.first_op {
                    FirstOp::Reduce { x, shape, axes, op } => {
                        // permute to join reduce axes together in last dimension
                        // reshape to 4d, last dim reduce
                        let sh = tile.view.shape();
                        let d0 = sh[0];
                        let d1 = sh[1];
                        if axes.contains(&0) {
                            if axes.contains(&1) {
                                tile.view.reshape(&[1, 1, 1, d0*d1]);
                            } else {
                                tile.view.permute(&[1, 0]);
                                tile.view.reshape(&[1, 1, d1, d0]);
                            }
                        } else if axes.contains(&1) {
                            tile.view.reshape(&[1, 1, d0, d1]);
                        }
                    }
                    _ => {
                        // reshape to 3d
                        let sh = tile.view.shape();
                        tile.view.reshape(&[1, sh[0], sh[1]]);
                    }
                }
                3 => match &mut tile.first_op {
                    FirstOp::Reduce { x, shape, axes, op } => {
                        // permute to join reduce axes together in last dimension
                        // and possibly reshape to 4d, last dim reduce
                        let all_axes: Vec<usize> = (0..tile.view.rank()).filter(|a| !axes.contains(a)).chain(axes.iter().copied()).collect();
                        tile.view.permute(&all_axes);
                        let sh = tile.view.shape();
                        let r = sh.len();
                        let d1 = if r - axes.len() > 2 { sh[r-axes.len()-2] } else { 1 };
                        let d2 = if r - axes.len() > 1 { sh[r-axes.len()-1] } else { 1 };
                        let d3 = sh[r-axes.len()..r].iter().product();
                        let d0 = sh.iter().product::<usize>()/(d1*d2*d3);
                        tile.view.reshape(&[d0, d1, d2, d3]);
                    }
                    _ => {}
                }
                _ => match &tile.first_op {
                    FirstOp::Reduce { x, shape, axes, op } => {
                        // permute to join reduce axes together in last dimension
                        // reshape to 4d, last dim reduce
                        let all_axes: Vec<usize> = (0..tile.view.rank()).filter(|a| !axes.contains(a)).chain(axes.iter().copied()).collect();
                        tile.view.permute(&all_axes);
                        let sh = tile.view.shape();
                        let r = sh.len();
                        let d1 = if r - axes.len() > 2 { sh[r-axes.len()-2] } else { 1 };
                        let d2 = if r - axes.len() > 1 { sh[r-axes.len()-1] } else { 1 };
                        let d3 = sh[r-axes.len()..r].iter().product();
                        let d0 = sh.iter().product::<usize>()/(d1*d2*d3);
                        tile.view.reshape(&[d0, d1, d2, d3]);
                    }
                    _ => {
                        // reshape to 3d
                        let sh = tile.view.shape();
                        let n = sh.len();
                        let d1 = sh[n-2];
                        let d2 = sh[n-1];
                        tile.view.reshape(&[tile.view.numel()/(d1*d2), d1, d2]);
                    }
                }
            }
        }

        //std::println!();
        //std::println!();

        // Print AST
        /*for tile in &tiles {
            std::println!("{tile:?}");
        }*/

        // Rewrite tiled representation to looped representation
        let ir = ir::tiled_to_ir(tiles, &order);
        //std::println!("{looped:#?}");

        let mut to_compile = to_eval.clone();
        let mut programs = Vec::new();
        while let Some(id) = to_compile.pop_last() {
            // Go backward from to_eval, compile those kernels and then compile all kernels
            // that are used as arguments to those kernels.
            // Search over
            programs.push(self.compiler.compile_program(&ir[&id]));
        }

        panic!();

        // Check work sizes (view.shapes) of tiles and add appropriate loops.
        // Global (and local) work size is maximum shape of input and output (to_eval) tiles.
        // Any tile expanded beyond this maximum size starts a reduce loops,
        // that is closed by reduce tile (i.e. we know that there is a reduce tile)

        // Kernel
        /*let mut kernel = VirtKernel {
            indices: vec![],
            mems: vec![],
            instructions: vec![],
        };

        Ok(CompiledGraph {
            args: vec![],
            program: self.compiler.compile_program(&kernel)?,
            flop: 0,
            bytes: 0,
        })*/
    }

    pub(super) fn launch_graph(&mut self, graph: &BTreeMap<TensorId, Node>) -> Result<(), CompilerError> {
        let graph = self.compiled_graphs.get(graph).unwrap();
        let mut buffers = Vec::with_capacity(graph.args.len());
        // We can move those buffers out of self.buffers and then move them back,
        // or we can pass &mut self.buffers to launch program. Which one is better?
        /*for arg in &graph.args {
            buffers.push(self.buffers.get_mut(arg).unwrap());
        }*/
        return self.compiler.launch_program(&graph.program, &buffers) //, graph.flop, graph.bytes)
    }
}

// Includes Noop for copying between tiles of various scopes
#[derive(Debug, Clone, Copy)]
enum UOp {
    Noop,
    Neg,
    Sin,
    Cos,
    Exp,
    Ln,
    Tanh,
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

#[derive(Debug, Clone)]
struct Tile {
    pub(crate) view: View,
    dtype: DType,
    scope: u8,
    pub(crate) first_op: FirstOp,
    // Note that order of these ops does not matter for correctness,
    // but may matter for performance
    pub(crate) ops: Vec<UOp>,
    can_be_fused: bool, // true by default, false if this tile is used by more than one tile (rc>1)
}

impl Tile {
    fn is_reduce(&self) -> bool {
        matches!(self.first_op, FirstOp::Reduce { .. })
    }
}

#[derive(Debug, Clone)]
enum ROp {
    Sum,
    Max,
}

// It is better if these ops represent breaks between tiles,
// from performance perspective, for example we do not want to run
// expanded tensor in a million threads if it was a tensor with just 10 scalars.
// But these are also fused (if possible) later when converted to looped representation.
#[derive(Debug, Clone)]
enum FirstOp {
    // Load existing buffer from memory
    Load {
        dtype: DType,
    },
    Binary {
        x: TensorId,
        y: TensorId,
        op: BOp,
    },
    Reduce {
        x: TensorId,
        shape: Vec<usize>, // Shape before reduce
        axes: Vec<usize>,
        op: ROp,
    },
    // Some movement ops can not be fused into existing kernel
    // Permute can always be fused.
    // Expand can never be fused.
    // Reshape and pad can not be fused with reduce kernel.
    // Technically reshaped and pad could be fused with some reduce kernels,
    // but we can keep this simple here and fuse them in looped kernel.
    Movement {
        x: TensorId,
    },
}
