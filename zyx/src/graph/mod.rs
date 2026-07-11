use crate::{
    DType, Map,
    backend::ProgramId,
    dtype::Constant,
    kernel::{BOp, DeviceId, UOp},
    runtime::{KernelId, ShapeId},
    shape::{Dim, UAxis},
    slab::{Slab, SlabId},
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(crate) struct NodeId(pub u32);

impl From<usize> for NodeId {
    fn from(v: usize) -> Self {
        Self(v as u32)
    }
}
impl From<NodeId> for usize {
    fn from(v: NodeId) -> usize {
        v.0 as usize
    }
}

impl SlabId for NodeId {
    const ZERO: Self = Self(0);
    const NULL: Self = Self(u32::MAX);
    fn inc(&mut self) {
        self.0 += 1;
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(crate) struct ClassId(pub u32);

impl From<usize> for ClassId {
    fn from(v: usize) -> Self {
        Self(v as u32)
    }
}
impl From<ClassId> for usize {
    fn from(v: ClassId) -> usize {
        v.0 as usize
    }
}

impl SlabId for ClassId {
    const ZERO: Self = Self(0);
    const NULL: Self = Self(u32::MAX);
    fn inc(&mut self) {
        self.0 += 1;
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub(crate) enum ENode {
    Const(Constant),
    Leaf {
        dtype: DType,
        shape: ShapeId,
    },
    Expand {
        x: ClassId,
        shape: ShapeId,
    },
    Permute {
        x: ClassId,
        axes: Box<[UAxis]>,
    },
    Reshape {
        x: ClassId,
        shape: ShapeId,
    },
    PadZeros {
        x: ClassId,
        padding: Box<[(i64, i64)]>,
    },
    Reduce {
        x: ClassId,
        bop: BOp,
        axes: Box<[UAxis]>,
    },
    Cast {
        x: ClassId,
        dtype: DType,
    },
    Unary {
        x: ClassId,
        uop: UOp,
    },
    Binary {
        x: ClassId,
        y: ClassId,
        bop: BOp,
    },
    ToDevice {
        x: ClassId,
        device: DeviceId,
    },
    Kernel {
        inputs: Box<[ClassId]>,
        outputs: Box<[ClassId]>,
        program_id: ProgramId,
    },
}

#[derive(Debug)]
pub(crate) struct EClass {
    pub nodes: Vec<NodeId>,
    pub parents: Vec<(NodeId, usize)>,
    pub shape: Box<[Dim]>,
    pub dtype: DType,
}

pub(crate) struct EGraph {
    nodes: Slab<NodeId, ENode>,
    classes: Slab<ClassId, EClass>,
    class_of: Vec<ClassId>,
    class_parent: Vec<ClassId>,
    class_rank: Vec<u8>,
    hashcons: Map<ENode, NodeId>,
    // Node -> Kernel, cost
    kernel_map: Map<NodeId, (KernelId, u64)>,
}
