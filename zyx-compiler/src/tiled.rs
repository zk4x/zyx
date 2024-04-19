//! Here work size and number of kernels is determined by creating tiles.
//! Tiles with the same work size get fused together.

use std::collections::{BTreeMap, BTreeSet};
use std::prelude::rust_2015::Vec;
use zyx_core::error::ZyxError;
use zyx_core::node::Node;
use zyx_core::runtime::RuntimeBackend;
use zyx_core::scalar::Scalar;
use zyx_core::tensor::Id;
use crate::{CompiledBackend, Compiler, looped};
use alloc::vec;
use zyx_core::axes::{Axes, IntoAxes};
use zyx_core::dtype::DType;
use zyx_core::shape::Shape;
use zyx_core::utils::{get_dtype, get_shape};
use zyx_core::view::View;

/// Compiled graph
pub struct CompiledGraph<Program> {
    args: Vec<Id>,
    program: Program,
    flop: usize,
    bytes: usize,
}

impl<C: Compiler> RuntimeBackend for CompiledBackend<C> {
    type CompiledGraph = CompiledGraph<C::Program>;

    fn is_empty(&self, x: Id) -> bool {
        !self.buffers.contains_key(&x)
    }

    fn evaluated_nodes(&self) -> BTreeSet<Id> {
        self.buffers.keys().copied().collect()
    }

    fn store<T: Scalar, IT>(&mut self, x: Id, iter: IT) -> Result<(), ZyxError>
        where
            IT: IntoIterator<Item=T>,
            IT::IntoIter: ExactSizeIterator,
    {
        //std::println!("Storing {x}");
        let data = iter.into_iter();
        let mut buffer = self.compiler.allocate_mem(data.len(), T::dtype())?;
        self.compiler.store_mem(&mut buffer, data)?;
        self.buffers.insert(x, buffer);
        Ok(())
    }

    fn load<T: Scalar>(&mut self, x: Id, numel: usize) -> Result<Vec<T>, ZyxError> {
        //std::println!("Loading {x}");
        if let Some(buffer) = self.buffers.get(&x) {
            self.compiler.load_mem(buffer, numel)
        } else {
            panic!("Buffer not evaluated");
        }
    }

    fn remove(&mut self, x: Id) -> Result<(), ZyxError> {
        // TODO we can later optimize this by not deallocating memory if it can be reused later
        if let Some(mut buffer) = self.buffers.remove(&x) {
            //std::println!("Dropping buffer {p} out of total {} buffers", self.buffers.len());
            self.compiler.deallocate_mem(&mut buffer)?;
        }
        Ok(())
    }

    fn compile_graph(&mut self, _global_rcs: &[u32], nodes: &[Node], to_eval: &BTreeSet<Id>) -> Result<Self::CompiledGraph, ZyxError> {
        let hw_info = self.compiler.hardware_info();
        std::println!("{hw_info:?}");
        // Find the best order of execution of nodes
        let (order, rcs) = {
            let mut temp_rcs: BTreeMap<Id, u32> = BTreeMap::new();
            let mut params: Vec<Id> = to_eval.iter().copied().collect();
            params.reserve(100);
            while let Some(nid) = params.pop() {
                //std::println!("{nid} is evaluated: {}", self.runtime_backend.is_evaluated(nid));
                temp_rcs
                    .entry(nid)
                    .and_modify(|rc| *rc += 1)
                    .or_insert_with(|| {
                        if self.is_empty(nid) {
                            params.extend(nodes[nid.i()].parameters());
                        }
                        1
                    });
            }
            // Order them using rcs reference counts.
            let mut order = Vec::new();
            let mut rcs: BTreeMap<Id, u32> = BTreeMap::new();
            let mut params: Vec<Id> = to_eval.iter().copied().collect();
            params.reserve(100);
            while let Some(nid) = params.pop() {
                if let Some(temp_rc) = temp_rcs.get(&nid) {
                    let rc = rcs.entry(nid).and_modify(|rc| *rc += 1).or_insert(1);
                    if *temp_rc == *rc {
                        order.push(nid);
                        params.extend(nodes[nid.i()].parameters());
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
                let dtype = get_dtype(nodes, nid);
                tiles.insert(nid, Tile {
                    view: View::from(get_shape(nodes, nid)),
                    dtype,
                    scope: 2,
                    first_op: FirstOp::Load { dtype },
                    ops: Vec::new(),
                });
                continue;
            }
            match &nodes[nid.i()] {
                Node::Const(_) => {}
                Node::Detach(_) => {}
                Node::Leaf(_, _) => {}
                Node::Cast(_, _) => {}
                Node::Exp(x) => {
                    std::println!("{x}");
                    let mut tile = tiles[x].clone();
                    tile.ops.push(UOp::Exp);
                    tiles.insert(nid, tile);
                }
                Node::Add(x, y) => {
                    let tile = Tile {
                        view: View::from(get_shape(nodes, nid)),
                        dtype: get_dtype(nodes, nid),
                        scope: 2,
                        first_op: FirstOp::Binary {
                            x: *x,
                            y: *y,
                            op: BOp::Add,
                        },
                        ops: vec![],
                    };
                    tiles.insert(nid, tile);
                }
                Node::Reshape(x, sh) => {
                    match &tiles[x].first_op {
                        FirstOp::Reduce { .. } => {
                            let tile = Tile {
                                view: View::from(sh),
                                dtype: get_dtype(nodes, nid),
                                scope: 2,
                                first_op: FirstOp::Movement { x: *x },
                                ops: vec![],
                            };
                            tiles.insert(nid, tile);
                        }
                        _ => {
                            let mut tile = tiles[x].clone();
                            tile.view.reshape(sh);
                            tiles.insert(nid, tile);
                        }
                    }
                }
                Node::Expand(x, sh) => {
                    let tile = Tile {
                        view: View::from(sh),
                        dtype: get_dtype(nodes, nid),
                        scope: 2,
                        first_op: FirstOp::Movement { x: *x },
                        ops: vec![],
                    };
                    tiles.insert(nid, tile);
                }
                Node::Permute(_, _, _) => {
                    // Permute permutes all views.
                    // In case of reduce kernel, permute also axes and shape before reduce.
                }
                Node::Pad(x, padding, _) => {
                    match &tiles[x].first_op {
                        FirstOp::Reduce { .. } => {
                            let mut view = View::from(get_shape(nodes, *x));
                            view.pad(padding);
                            let tile = Tile {
                                view,
                                dtype: get_dtype(nodes, nid),
                                scope: 2,
                                first_op: FirstOp::Movement { x: *x },
                                ops: vec![],
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
                Node::Sum(x, axes, sh) => {
                    let tile = Tile {
                        view: View::from(sh),
                        dtype: get_dtype(nodes, nid),
                        scope: 2,
                        first_op: FirstOp::Reduce {
                            x: *x,
                            shape: get_shape(nodes, *x).clone(),
                            axes: axes.clone(),
                            op: ROp::Sum
                        },
                        ops: vec![],
                    };
                    tiles.insert(nid, tile);
                }
                _ => {
                    todo!()
                }
            }
        }

        // Print AST
        for tile in &tiles {
            std::println!("{tile:?}");
        }

        // Reshape and permute tiles to use exactly work 3 dimensions
        // and at most single reduce loop over the last dimension>

        for (id, tile) in &mut tiles {
            match tile.view.rank() {
                1 => match &mut tile.first_op {
                    FirstOp::Reduce { axes, .. } => {
                        // permute to join reduce axes together in last dimension
                        // reshape to 4d, last dim reduce
                        debug_assert_eq!(axes.len(), 1);
                        *axes = 3i64.into_axes(tile.view.rank());
                        tile.view.reshape(&[1, 1, 1, tile.view.numel()].into());
                    }
                    _ => {
                        // reshape to 3d
                        tile.view.reshape(&[1, 1, tile.view.numel()].into());
                    }
                }
                2 => match &tile.first_op {
                    FirstOp::Reduce { x, shape, axes, op } => {
                        // permute to join reduce axes together in last dimension
                        // reshape to 4d, last dim reduce
                        let sh = tile.view.shape();
                        let d0 = sh[0];
                        let d1 = sh[1];
                        if axes.contains(0) {
                            if axes.contains(1) {
                                tile.view.reshape(&[1, 1, 1, d0*d1].into());
                            } else {
                                tile.view.permute(&[1, 0].into_axes(2));
                                tile.view.reshape(&[1, 1, d1, d0].into());
                            }
                        } else if axes.contains(1) {
                            tile.view.reshape(&[1, 1, d0, d1].into());
                        }
                    }
                    _ => {
                        // reshape to 3d
                        let sh = tile.view.shape();
                        tile.view.reshape(&[1, sh[0], sh[1]].into());
                    }
                }
                3 => match &mut tile.first_op {
                    FirstOp::Reduce { x, shape, axes, op } => {
                        // permute to join reduce axes together in last dimension
                        // and possibly reshape to 4d, last dim reduce
                        todo!()
                    }
                    _ => {}
                }
                _ => match &tile.first_op {
                    FirstOp::Reduce { x, shape, axes, op } => {
                        // permute to join reduce axes together in last dimension
                        // reshape to 4d, last dim reduce
                        let all_axes: Vec<usize> = (0..tile.view.rank()).filter(|a| !axes.contains(*a)).chain(axes.iter().copied()).collect();
                        tile.view.permute(&all_axes.into_axes(tile.view.rank()));
                        let sh = tile.view.shape();
                        let r = sh.rank();
                        let d1 = if r - axes.len() > 2 { sh[r-axes.len()-2] } else { 1 };
                        let d2 = if r - axes.len() > 1 { sh[r-axes.len()-1] } else { 1 };
                        let d3 = sh[(r-axes.len()) as i64..r as i64].iter().product();
                        let d0 = sh.numel()/(d1*d2*d3);
                        tile.view.reshape(&[d0, d1, d2, d3].into());
                    }
                    _ => {
                        // reshape to 3d
                        let sh = tile.view.shape();
                        let d1 = sh[-2];
                        let d2 = sh[-1];
                        tile.view.reshape(&[tile.view.numel()/(d1*d2), d1, d2].into());
                    }
                }
            }
        }

        // Print AST
        for tile in &tiles {
            std::println!("{tile:?}");
        }


        // Rewrite tiled representation to looped representation
        let looped = looped::tiled_to_looped(tiles, &order);



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

    fn launch_graph(&mut self, graph: &Self::CompiledGraph) -> Result<(), ZyxError> {
        let buffers: Vec<&C::Buffer> = graph.args.iter().map(|id| self.buffers.get(id).unwrap()).collect();
        self.compiler.launch_program(&graph.program, &buffers, graph.flop, graph.bytes)
    }
}

#[derive(Debug, Clone)]
pub(crate) struct Tile {
    pub(crate) view: View,
    dtype: DType,
    scope: u8,
    pub(crate) first_op: FirstOp,
    // Note that order of these ops does not matter for correctness,
    // but may matter for performance
    pub(crate) ops: Vec<UOp>,
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

#[derive(Debug, Clone)]
enum UOp {
    Exp,
    Neg,
    Tanh,
}

#[derive(Debug, Clone)]
enum BOp {
    Add,
    Sub,
    Mul,
    Div,
    Pow,
    Cmplt,
}

// It is better if these ops represent breaks between tiles,
// from performance perspective, for example we do not want to run
// expanded tensor in a million threads if it was a tensor with just 10 scalars.
// But these are also fused (if possible) later when converted to looped representation.
#[derive(Debug, Clone)]
pub(crate) enum FirstOp {
    // Load existing buffer from memory
    Load {
        dtype: DType,
    },
    Binary {
        x: Id,
        y: Id,
        op: BOp,
    },
    Reduce {
        x: Id,
        shape: Shape, // Shape before reduce
        axes: Axes,
        op: ROp,
    },
    // Some movement ops can not be fused into existing kernel
    // Permute can always be fused.
    // Expand can never be fused.
    // Reshape and pad can not be fused with reduce kernel.
    // Technically reshaped and pad could be fused with some reduce kernels,
    // but we can keep this simple here and fuse them in looped kernel.
    Movement {
        x: Id,
    },
}

// Higher level abstraction, basically with tiles, views, bound dimensions and loops
// On this level we can do ops reordering, reshapes, permutes and pads can be moved even
// with binary ops, while expands can be only reordered between unary ops.
//
// So we need to add loops,
// mark movement of leafs between global, local and register tiles
// Add loops for expands bigger than work size,
// those must loops end with reduce ops, since initial work size (global and local loops)
// is calculated with output size in mind.
// Apply movement ops directly on tiles,
// Leave unary and binary ops as they are.
// Here we can do all the local and register tile optimizations and have special instructions
// for things like 4x4x4 matmuls, where we can just use tensor cores, wmma cores or like strassen

// For more low level (this needs to be rewritten once the more higher level approach is finalized
// Nodes like this (don't worry to make some assumptions)
//
// If values is in global scope
//   -> create new register mem
//   -> copy data from global into register (TODO use wide loads if it increases performance)
// Exp, Tanh, Neg, Sin, Cos
//   -> create new register mem
//   -> add instruction for unary op
//   -> decrease reference count from mems
// ReLU
//   -> same as unary, just rewrite as binary max with const
// Add, Sub, Mul, Div, Pow, Cmplt
//   -> same as unary
// Sum, Max
//   -> mark reduce start before first movement op on any of the leafs
//   -> create reduce loop
//   -> apply
//

// As opencl kernel
//
// let x = dev.randn([1024, 2048], DType::F32)?;
// let z = (&x + x.exp()).sum(-1);
//
// AST (simple version, no optimizations):
//  0 loop global 1024
//  1 loop global 2048
//  2 move global to register from id 0, view contiguous
//  3 exp 2
//  4 add 2, 3
//  5 loop
//  this seems very complicated
//  4
//  3 move global to register from id 0, view contiguous, bind existing ids
//  2 add 4, 3
//  1 sum (end loop) mark bind idx1 to 1, redefine id 1
//  0 move register to global, view contiguous (mark bind idx0 to dimension 0, idx1 to 1)
//
//
// float rmem0 = data0[];
// float rmem1 = exp(rmem0);
// // x has the same view on both sides of binary, so no need for second load
// float rmem2 = rmem0 + rmem1;
//

// Perhaps just do standard ASTs as in 0.12.1 and just join them together?
//
