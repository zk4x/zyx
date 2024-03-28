use crate::{CompiledBackend, Compiler, ir::ast_to_ir};
use alloc::{
    collections::BTreeMap,
    vec::Vec,
};
use zyx_core::dtype::DType;
use zyx_core::error::ZyxError;
use zyx_core::node::Node;
use zyx_core::runtime::{Graph, RuntimeBackend};
use zyx_core::scalar::Scalar;
use zyx_core::tensor::Id;
use zyx_core::utils::{get_shape, get_dtype};

impl<C: Compiler> CompiledBackend<C> {
    /// Initialize new compiled backend using provided compiler
    pub fn new(compiler: C) -> Self {
        Self {
            compiler,
            buffers: BTreeMap::new(),
            programs: BTreeMap::new(),
        }
    }
}

// From the graph, we first abstract into AST, where ops are applied on tiles.
// AST is then compiled into IR.
// AST uses custom IDs, so that it can be used for caching and so that it can use
// different number of tiles then the original graph has number of nodes.
// AST has optimization passes to give us better data locality.

#[derive(PartialEq, Eq, PartialOrd, Ord)]
pub(crate) struct AST {
    tiles: Vec<ASTTile>,
    ops: Vec<ASTOp>,
}

#[derive(PartialEq, Eq, PartialOrd, Ord)]
struct Dimension {
    size: usize,
    stride: usize,
    // if left_mask < size, its indexing, else it's padding
    // left_mask goes in reverse direction, from biggest to smallest value
    left_mask: usize,
    // if right_mask < size, its indexing, else it's padding
    right_mask: usize,
}

#[derive(PartialEq, Eq, PartialOrd, Ord)]
enum Scope {
    Global,
    Local,
    Private,
}

#[derive(PartialEq, Eq, PartialOrd, Ord)]
struct ASTTile {
    shape: Vec<Dimension>,
    dtype: DType, // not as important, but still useful
    scope: Scope,
    read_only: bool,
}

type ASTId = u32;

#[derive(PartialEq, Eq, PartialOrd, Ord)]
enum ASTUOp {
    Exp,
    Tanh,
}

#[derive(PartialEq, Eq, PartialOrd, Ord)]
enum ASTBOp {
    Add,
    Sub,
}

#[derive(PartialEq, Eq, PartialOrd, Ord)]
enum ASTROp {
    Sum,
    Max,
}

#[derive(PartialEq, Eq, PartialOrd, Ord)]
struct Padding {
    left_padding: Vec<usize>,
    right_padding: Vec<usize>,
}

#[derive(PartialEq, Eq, PartialOrd, Ord)]
enum ASTOp {
    Leaf(ASTId),
    Unary(ASTId, ASTUOp),
    Binary(ASTId, ASTId, ASTBOp),
    Where(ASTId, ASTId, ASTId),
    //Reshape(ASTId, Shape),
    //Expand(ASTId, Shape),
    //Pad(ASTId, Padding),
    //Permute(ASTId, Axes),
    //Reduce(ASTId, Axes, ASTROp),
}

impl<C: Compiler> RuntimeBackend for CompiledBackend<C> {
    fn is_free_id(&self, x: Id) -> bool {
        !self.buffers.contains_key(&x)
    }

    fn remove(&mut self, x: Id) -> Result<(), ZyxError> {
        if let Some(mut buffer) = self.buffers.remove(&x) {
            //std::println!("Dropping buffer {p} out of total {} buffers", self.buffers.len());
            self.compiler.deallocate_mem(&mut buffer)?;
        }
        Ok(())
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

    fn compile_graph(&mut self, rcs: &[u32], nodes: &[Node]) -> Result<Self::CompiledGraph, ZyxError> {
        let mut rcs = graph.rcs.clone();
        // Calculate correct work size for the graph,
        // reshape so that it is always 3d kernel,
        // where first dimension is batch.
        let mut global_work_size: Vec<usize> = Vec::new();

        // Map from Id into ASTId
        let mut idmap = BTreeMap::new();
        // AST tiles
        let mut tiles = Vec::new();
        let mut ops = Vec::new();
        // Create input nodes
        for nid in graph.order.iter().copied() {
            for p in graph.nodes[nid].parameters() {
                *rcs.get_mut(&p).unwrap() -= 1;
            }
            if let Node::Leaf(sh, dtype) = &graph.nodes[nid] {
                idmap.insert(nid, tiles.len());
                ops.push(ASTOp::Leaf(tiles.len() as u32));
                let mut shape = Vec::new();
                let mut stride = 1;
                for d in sh {
                    shape.push(Dimension {
                        size: *d,
                        stride,
                        left_mask: *d,
                        right_mask: *d,
                    });
                    stride *= d;
                }
                shape.reverse();
                tiles.push(ASTTile {
                    shape,
                    dtype: *dtype,
                    scope: Scope::Global,
                    read_only: true,
                });
            }
        }
        // Create output nodes
        for (id, rc) in rcs.iter() {
            if *rc > 0 {
                idmap.insert(*id, tiles.len());
                ops.push(ASTOp::Leaf(tiles.len() as u32));
                let mut shape = Vec::new();
                let mut stride = 1;
                for d in get_shape(nodes, *id) {
                    shape.push(Dimension {
                        size: *d,
                        stride,
                        left_mask: *d,
                        right_mask: *d,
                    });
                    stride *= d;
                }
                shape.reverse();
                tiles.push(ASTTile {
                    shape,
                    dtype: get_dtype(nodes, *id),
                    scope: Scope::Global,
                    read_only: true,
                });
            }
        }

        // Process graph in backward direction
        for nid in order.iter().copied().rev() {
            std::println!("{nid:>3}x{}  {:?}", rcs[&nid], nodes[nid.i()]);
            //ops.push();
        }

        // Possible optimization passes, like fusing mul and add into madd
        
        // Send AST to IR compiler and compile it, if it is not cached
        // TODO currently we are caching AST, be we really should be caching the whole graph
        let ast = AST { tiles, ops };
        let program = if let Some(program) = self.programs.get(&ast) {
            program
        } else {
            let ir = ast_to_ir(&ast, 256, 256*1024, 80);
            let program = self.compiler.compile_program(&ir)?;
            self.programs.entry(ast).or_insert(program)
        };

        //self.compiler.launch();

        Ok(())
    }
}
