// Here tiles get rewritten into tiles and loops, dimensions get bound
// and optimizations applied. At this stage, all movement and reduce ops are removed.
// Also, there will be special instructions for applying optimizations on like 4x4x4
// matmuls (like strassen or tensor cores) or 16x16x16 matmul (wmma).
// These optimizations are hardware dependent.

use crate::runtime::compiler::{BOp, FirstOp, HWInfo, Tile, UOp};
use crate::runtime::view::{Index, View};
use crate::runtime::TensorId;
use crate::DType;
use alloc::collections::BTreeMap;
use alloc::vec;
use alloc::vec::Vec;

#[derive(Debug)]
pub(super) struct IRKernel {
    pub(super) ops: Vec<IROp>,
}

#[derive(Debug)]
struct IRMem {
    id: u32,
    scope: u8,
    index: Option<u32>,
}

#[derive(Debug)]
enum IRIdx {
    Index(u32),
    Const(u32),
}

#[derive(Debug)]
enum IdxBOp {
    Add,
    Mul,
    Div,
    Mod,
}

/// IROp for direct translation to hardware kernels
/// Scope:
/// 0 - global
/// 1 - local
/// 2 - register
#[derive(Debug)]
pub(super) enum IROp {
    // All variables are 1d, so that it is easier for implementors
    InitMem {
        id: u32,
        scope: u8,
        dtype: DType,
        len: usize,
    },
    AssignMem {
        left: IRMem,
        right: IRMem,
    },
    UnaryMem {
        z: IRMem,
        x: IRMem,
        op: UOp,
    },
    BinaryMem {
        z: IRMem,
        x: IRMem,
        y: IRMem,
        op: BOp,
    },
    InitIdx {
        id: u32,
        value: usize,
    },
    BinaryIdx {
        z: IRIdx,
        x: IRIdx,
        y: IRIdx,
        op: IdxBOp,
    },
    Loop {
        id: u32,
        iters: usize,
        scope: u8,
    },
    EndLoop,
}

/*#[derive(Debug, Clone)]
pub(super) enum IROp {
    // Global kernel argument load
    Load { buffer_id: TensorId, dtype: DType },
    // Global kerneL argument store
    Store { buffer_id: TensorId, dtype: DType },
    // Movement op, simply changes the view of this buffer. This means moving things around in memory
    // and thus is extremely expensive. We should use memory caching here if possible.
    // Things can be also moved between different memory scopes.
    Movement { x: usize, scope: u8, view: View },
    // Unary and binary ops always take and return variables in registers
    // Cheap op
    Unary { x: usize, op: UOp },
    // Even binary ops are cheap
    Binary { x: usize, y: usize, op: BOp },
    // TODO we also need
    // Loop, global, local, register ...
    Loop { iters: usize, scope: u8 },
    EndLoop,
    // Optimation instructions, implementation is hardware specific and thus is up to the compiler
    // Matmul of two 16x16 tiles, result is also 16x16 tile stored in local memory
    NativeMM16x16 { x: usize, y: usize },
}*/

pub(crate) fn tiled_to_ir(
    tiles: BTreeMap<TensorId, Tile>,
    order: &[TensorId],
    hwinfo: &HWInfo,
) -> BTreeMap<TensorId, IRKernel> {
    let mut kernels = BTreeMap::new();

    // At this point every kernel is already 3d, reduce kernels are 4d, with last dim reduce

    // First we write version with only global and register loops, without caching

    for nid in order {
        match tiles[nid].first_op {
            FirstOp::Load { dtype, buffer_id } => {
                let tile = &tiles[nid];
                let sh = tile.view.shape();
                let mut ops = vec![
                    // Global work size
                    IROp::Loop {
                        id: 0,
                        iters: sh[0],
                        scope: 0,
                    },
                    IROp::Loop {
                        id: 1,
                        iters: sh[1],
                        scope: 0,
                    },
                    IROp::Loop {
                        id: 2,
                        iters: sh[2],
                        scope: 0,
                    },
                    IROp::Loop {
                        id: 3,
                        iters: 1,
                        scope: 1,
                    },
                    IROp::Loop {
                        id: 4,
                        iters: 1,
                        scope: 1,
                    },
                    IROp::Loop {
                        id: 5,
                        iters: 1,
                        scope: 1,
                    },
                    IROp::Load { buffer_id, dtype },
                ];
                if !tile.view.is_contiguous() {
                    ops.push(IROp::Movement {
                        x: 3,
                        scope: 0,
                        view: tile.view.clone(),
                    });
                }
                for op in &tile.ops {
                    ops.push(IROp::Unary {
                        x: ops.len() - 1,
                        op: *op,
                    });
                }
                kernels.insert(*nid, IRKernel { ops });
            }
            // These tiled kernels can be fused with previous kernels if reduce and expand
            // kernels exist back to back (with some binary kernels in between and the final
            // work size is the same as the beginning work size.
            FirstOp::Reduce { .. } => {}
            FirstOp::Movement { .. } => {}
            // Binary tile fuses two ir kernels together
            FirstOp::Binary { x, y, op } => {
                let kernel_x = &kernels[&x];
                let kernel_y = &kernels[&y];
                let tile = &tiles[nid];
                let sh = tile.view.shape();
                // Add ops from input tiles
                // Copy directly from tile_x
                let mut ops = kernel_x.ops.clone();
                let n = ops.len() - 3; // 3 is the number of loops in kernel_y that are removed
                                       // Reindex ops from tile_y
                for op in &kernel_y.ops {
                    match op {
                        IROp::Movement { x, scope, view } => {
                            ops.push(IROp::Movement {
                                x: x + n,
                                scope: *scope,
                                view: view.clone(),
                            });
                        }
                        IROp::Unary { x, op } => {
                            ops.push(IROp::Unary { x: x + n, op: *op });
                        }
                        IROp::Binary { x, y, op } => {
                            ops.push(IROp::Binary {
                                x: x + n,
                                y: y + n,
                                op: *op,
                            });
                        }
                        IROp::Loop { .. } => {}
                        _ => {
                            ops.push(op.clone());
                        }
                    };
                }
                // Add ops from binary tile
                ops.push(IROp::Binary {
                    x: n - 1,
                    y: ops.len() - 1,
                    op: BOp::Add,
                });
                for op in &tile.ops {
                    ops.push(IROp::Unary {
                        x: ops.len() - 1,
                        op: *op,
                    });
                }
                kernels.insert(*nid, IRKernel { ops });
            }
        }
    }
    // End global and local loops
    for kernel in kernels.values_mut() {
        for _ in 0..6 {
            kernel.ops.push(IROp::EndLoop);
        }
    }

    // TODO optimize ir_kernels using memory tiling and such

    kernels
}
