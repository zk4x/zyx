use crate::{dtype::Constant, ir::Scope, node::{BOp, ROp, UOp}, shape::{Axis, Dimension}, view::View, DType};


#[cfg_attr(feature = "disk_cache", derive(bitcode::Encode, bitcode::Decode))]
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(super) struct Kernel {
    pub(super) max_id: TId,
    pub(super) ops: Vec<Op>,
}

// Tensor id in a kernel
pub(super) type TId = u16;

// TODO this needs to be smaller, since it's stored on the disk
#[cfg_attr(feature = "disk_cache", derive(bitcode::Encode, bitcode::Decode))]
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(super) enum Op {
    Loop {
        axis: Axis,
        len: Dimension,
    },
    // End the latest loop
    EndLoop,
    Const {
        z: TId,
        value: Constant,
        view: View,
    },
    Load {
        z: TId,
        zscope: Scope,
        zview: View,
        xscope: Scope,
        xview: View,
        xdtype: DType,
    },
    Store {
        z: TId,
        zscope: Scope,
        zview: View,
        zdtype: DType,
        xscope: Scope,
        xview: View,
    },
    Accumulator {
        z: TId,
        rop: ROp,
        view: View,
        dtype: DType,
    },
    // Move is noop, just a marker for easy debugging
    // and to keep track of tensor ids
    Move {
        z: TId,
        x: TId,
        mop: MOp,
    },
    Unary {
        z: TId,
        x: TId,
        uop: UOp,
    },
    Binary {
        z: TId,
        x: TId,
        y: TId,
        bop: BOp,
    },
    // Synchronization for local and global memory
    Barrier {
        scope: Scope,
    },
}

#[cfg_attr(feature = "disk_cache", derive(bitcode::Encode, bitcode::Decode))]
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum MOp {
    Expa,
    Perm,
    Resh,
    Padd,
}

impl Kernel {
    pub(super) fn empty() -> Kernel {
        Kernel {
            max_id: 0,
            ops: Vec::new(),
        }
    }

    pub(super) fn constant(value: Constant) -> Kernel {
        let mut ops = Vec::with_capacity(50);
        ops.push(Op::Loop { axis: 0, len: 1 });
        ops.push(Op::Const {
            z: 0,
            value,
            view: View::contiguous(&[1]),
        });
        Kernel { max_id: 0, ops }
    }

    pub(super) fn leaf(shape: &[usize], dtype: DType) -> Kernel {
        let mut ops = Vec::with_capacity(50);
        for (axis, dimension) in shape.iter().copied().enumerate() {
            ops.push(Op::Loop {
                axis,
                len: dimension,
            });
        }
        ops.push(Op::Load {
            z: 0,
            zscope: Scope::Register,
            zview: View::none(),
            xscope: Scope::Global,
            xview: View::contiguous(&shape),
            xdtype: dtype,
        });
        Kernel { max_id: 0, ops }
    }

    pub(super) fn store(&mut self, z: TId, zview: View, zdtype: DType) {
        if let Some(&Op::Store {
            z: nz,
            zview: ref nzview,
            ..
        }) = self.ops.last()
        {
            if z == nz && &zview == nzview {
                return;
            }
        }
        debug_assert!(zview.numel() < 1024 * 1024 * 1024, "Too big store.");
        let store_op = Op::Store {
            z,
            zview,
            zscope: Scope::Global,
            zdtype,
            xscope: Scope::Register,
            xview: View::none(),
        };
        self.ops.push(store_op);
    }
}