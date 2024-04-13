// Here tiles get rewritten into tiles and loops, dimensions get bound
// and optimizations applied. At this stage, all movement and reduce ops are removed.
// Also, there will be special instructions for applying optimizations on like 4x4x4
// matmuls (like strassen or tensor cores) or 16x16x16 matmul (wmma).
// These optimizations are hardware dependent.

use zyx_core::dtype::DType;
use zyx_core::tensor::Id;
use alloc::collections::BTreeMap;
use alloc::collections::BTreeSet;
use alloc::vec::Vec;
use alloc::vec;
use zyx_core::view::View;
use crate::tiled::{FirstOp, Tile};

// Includes Noop for copying between tiles of various scopes
enum UOp {
    Noop,
    Neg,
    Sin,
    Cos,
    Exp,
    Ln,
    Tanh,
    Sqrt,
}

enum BOp {
    Add,
    Sub,
    Mul,
    Div,
    Pow,
    Cmplt,
    Max, // for ReLU and max reduce
}

enum Op {
    // Argument outside of kernel (appears in function arguments)
    Arg {
        id: Id,
    },
    // Movement op, simply changes the view of this buffer. This means moving things around in memory
    // and thus is extremely expensive. We should use memory caching here if possible.
    Movement {
        view: View,
    },
    // Cheap op (from performance perspective)
    Cast {
        x: usize,
        dtype: DType,
    },
    // Cheap op
    Unary {
        x: usize,
        op: UOp,
    },
    // Even binary ops are cheap
    Binary {
        x: usize,
        y: usize,
        op: BOp,
    },
    Loop {
        iters: usize,
        scope: u8,
    },
    EndLoop,
}

pub(crate) struct LoopedKernel {
    fused: BTreeSet<Id>, // which tiled kernels were fused into this looped one?
    ops: Vec<Op>,
}

pub(crate) fn tiled_to_looped(tiles: BTreeMap<Id, Tile>, order: &[Id]) -> Vec<LoopedKernel> {
    let mut looped = Vec::new();

    for nid in order {
        match tiles[nid].first_op {
            FirstOp::Load { dtype } => {
                looped.push(LoopedKernel {
                    fused: [*nid].into(),
                    ops: vec![],
                    //ops: vec![Op::Loop, Op::Arg, Op::Movement {}],
                });
            }
            // These tiled kernels can be fused with previous kernels if reduce and expand
            // kernels exist back to back (with some binary kernels in between and the final
            // work size is the same as the beginning work size.
            FirstOp::Reduce { .. } => {}
            FirstOp::Movement { .. } => {}
            // Binary tile fuses two looped kernels together
            FirstOp::Binary { .. } => {}
        }
    }

    looped
}