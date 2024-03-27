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

struct AST {
    tiles: Vec<Tile>,
}

struct Dimension {
    size: usize,
    stride: usize,
    // if left_pad < size, its indexing, else it's padding
    // left_pad goes in reverse direction, from biggest to smallest value
    left_pad: usize,
    // if right < size, its indexing, else it's padding
    right_pad: usize,
}

struct Tile {
    shape: Vec<Dimension>,
    dtype: DType, // not as important, but still useful
}

impl<C: Compiler> RuntimeBackend for CompiledBackend<C> {
    fn is_evaluated(&self, x: Id) -> bool {
        self.kernels.contains_key(&x)
    }

    fn is_free_id(&self, x: Id) -> bool {
        !self.buffers.contains_key(&x)
    }

    fn remove(&mut self, x: Id) -> Result<(), ZyxError> {
        if let Some(mut buffer) = self.buffers.remove(&p) {
            //std::println!("Dropping buffer {p} out of total {} buffers", self.buffers.len());
            self.compiler.drop_buffer(&mut buffer)?;
        }
        Ok(())
    }

    fn store<T: Scalar, IT>(&mut self, x: Id, iter: IT) -> Result<(), ZyxError>
    where
        IT: IntoIterator<Item = T>,
        IT::IntoIter: ExactSizeIterator,
    {
        //std::println!("Storing {x}");
        self.buffers.insert(x, self.compiler.store(iter.into_iter())?);
        Ok(())
    }

    fn load<T: Scalar>(&mut self, x: Id, numel: usize) -> Result<Vec<T>, ZyxError> {
        //std::println!("Loading {x}");
        if let Some(buffer) = self.buffers.get(&x) {
            self.compiler.load(buffer, numel)
        } else {
            panic!("Buffer not evaluated");
        }
    }

    fn evaluate(
        &mut self,
        mut rcs: BTreeMap<Id, u32>,
        order: &[Id],
        nodes: &[Node],
    ) -> Result<(), ZyxError> {
        for nid in order.iter().copied().rev() {
        }
        Ok(())
    }
}
