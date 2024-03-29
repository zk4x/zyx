use crate::{CompiledBackend, Compiler, ir::ast_to_ir};
use alloc::{
    collections::BTreeMap,
    vec::Vec,
};
use std::collections::BTreeSet;
use zyx_core::axes::Axes;
use zyx_core::dtype::DType;
use zyx_core::error::ZyxError;
use zyx_core::node::{Constant, Node};
use zyx_core::runtime::RuntimeBackend;
use zyx_core::scalar::Scalar;
use zyx_core::shape::Shape;
use zyx_core::tensor::Id;
use zyx_core::utils::{get_shape, get_dtype};

// From the graph, we first abstract into AST, where ops are applied on tiles.
// AST is then compiled into IR.
// AST uses custom IDs, so that it can be used for caching and so that it can use
// different number of tiles then the original graph has number of nodes.
// AST has optimization passes to give us better data locality.

#[derive(Debug)]
pub(crate) struct Dimension {
    size: usize,
    stride: usize,
    // if left_mask < size, its indexing, else it's padding
    // left_mask goes in reverse direction, from biggest to smallest value
    left_mask: usize,
    // if right_mask < size, its indexing, else it's padding
    right_mask: usize,
}

#[derive(Debug)]
pub(crate) enum Scope {
    Global,
    Local,
    Private,
}

pub(crate) type ASTId = usize;

#[derive(Debug)]
pub(crate) enum ASTUOp {
    Noop, // Just a copy for moving between local, global and registers
    Exp,
    Tanh,
}

#[derive(Debug)]
pub(crate) enum ASTBOp {
    Add,
    Sub,
}

#[derive(Debug)]
pub(crate) enum ASTROp {
    Sum,
    Max,
}

#[derive(Debug)]
enum MOp {
    Expand(Shape),
    Pad(Vec<(usize, usize)>),
    Permute(Axes),
    Reshape(Shape),
}

#[derive(Debug)]
pub(crate) enum ASTOp {
    Leaf {
        id: Id, // Id into self.buffers
        shape: Vec<Vec<Dimension>>, // Vec of InnerView essentially
        dtype: DType,
        scope: Scope,
    },
    Unary(ASTId, ASTUOp),
    Binary(ASTId, ASTId, ASTBOp),
    Where(ASTId, ASTId, ASTId),
    Reduce(ASTId, ASTROp),
}

impl ASTOp {
    fn parameters(&self) -> impl Iterator<Item = ASTId> {
        match self {
            ASTOp::Leaf { .. } => {
                Vec::new().into_iter()
            }
            ASTOp::Unary(x, _) => {
                alloc::vec![x].into_iter()
            }
            ASTOp::Binary(x, y, _) => {
                alloc::vec![x, y].into_iter()
            }
            ASTOp::Where(x, y, z) => {
                alloc::vec![x, y, z].into_iter()
            }
        }
    }
}

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
        IT: IntoIterator<Item = T>,
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
        if let Some(mut buffer) = self.buffers.remove(&x) {
            //std::println!("Dropping buffer {p} out of total {} buffers", self.buffers.len());
            self.compiler.deallocate_mem(&mut buffer)?;
        }
        Ok(())
    }

    fn compile_graph(&mut self, global_rcs: &[u32], nodes: &[Node], to_eval: &BTreeSet<Id>) -> Result<Self::CompiledGraph, ZyxError> {
        // Find the best order of execution of nodes
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

        // TODO Reorder movement ops as to be late as possible (before reduce ops)
        // movement ops must be grouped together!

        // Calculate correct work size for the graph,
        // reshape so that it is always 3d kernel,
        // where first dimension is batch.
        // Work size should be maximum of output buffers dimensions
        let mut max_shape: Shape = 1.into();
        for nid in to_eval {
            let shape = get_shape(nodes, *nid);
            if shape.numel() > max_shape.numel() {
                max_shape = shape.clone();
            }
        }
        let mut global_work_size: Vec<usize> = alloc::vec![1, 1, 1];

        // Map from Id into ASTId
        let mut idmap = BTreeMap::new();
        let mut ops = Vec::new();

        // Process graph
        for nid in &order {
            std::println!("{nid:>3}x{}  {:?}", rcs[nid], nodes[nid.i()]);
            if self.buffers.contains_key(nid) {
                idmap.insert(*nid, ops.len());
                let sh = get_shape(nodes, *nid);
                let shape = Vec::with_capacity(sh.rank());
                ops.push(ASTOp::Leaf {
                    id: *nid,
                    shape,
                    dtype: get_dtype(nodes, *nid),
                    scope: Scope::Global,
                });
                continue;
            }
            std::println!("{ops:?}");
            match &nodes[nid.i()] {
                Node::Leaf(sh, dtype) => {}
                Node::Exp(x) => {
                    idmap.insert(*nid, ops.len());
                    ops.push(ASTOp::Unary(idmap[x], ASTUOp::Exp));
                }
                Node::Add(x, y) => {
                    idmap.insert(*nid, ops.len());
                    ops.push(ASTOp::Binary(idmap[x], idmap[y], ASTBOp::Add));
                }
                Node::Expand(x, sh) => {
                    apply_movement_op(&mut ops, *nid, MOp::Expand(sh.clone()));
                }
                Node::Reshape(x, sh) => {
                    todo!()
                }
                Node::Sum(x, axes, shape) => {
                    todo!()
                }
                _ => {
                    todo!()
                }
            }
        }

        // TODO Second pass for memory tiling

        // TODO Possible optimization passes, like fusing mul and add into madd
        
        let ir = ast_to_ir(&ops, 256, 256*1024, 80);
        let program = self.compiler.compile_program(&ir)?;

        Ok(Self::CompiledGraph {
            args: Vec::new(),
            program,
            flop: 0,
            bytes: 0,
        })
    }

    fn launch_graph(&mut self, graph: &Self::CompiledGraph) -> Result<(), ZyxError> {
        let buffers: Vec<&C::Buffer> = graph.args.iter().map(|id| self.buffers.get(id).unwrap()).collect();
        self.compiler.launch_program(&graph.program, &buffers, graph.flop, graph.bytes)
    }
}

fn apply_movement_op(ops: &mut [ASTOp], nid: Id, mop: MOp) {
    // DFS search and apply movement op to all AST leafs
    let mut params = alloc::vec![nid];
    let mut visited = BTreeSet::new();
    while let Some(param) = params.pop() {
        if visited.insert(param) {
            if let ASTOp::Leaf { id, shape, dtype, scope } = &ops[param.i()] {
                shape.apply(mop);
            } else {
                params.extend(ops[param.i()].parameters());
            }
        }
    }
}
