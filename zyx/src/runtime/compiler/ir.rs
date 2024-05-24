// Here tiles get rewritten into tiles and loops, dimensions get bound
// and optimizations applied. At this stage, all movement and reduce ops are removed.
// Also, there will be special instructions for applying optimizations on like 4x4x4
// matmuls (like strassen or tensor cores) or 16x16x16 matmul (wmma).
// These optimizations are hardware dependent.

use alloc::collections::BTreeMap;
use alloc::vec;
use alloc::vec::Vec;
use crate::DType;
use crate::runtime::compiler::{BOp, FirstOp, Tile, UOp};
use crate::runtime::TensorId;
use crate::runtime::view::View;

pub(super) struct IRKernel {
    ops: Vec<IROp>,
}

#[derive(Debug, Clone)]
enum IROp {
    // Global kernel argument load
    Load {
        id: TensorId,
        dtype: DType,
    },
    // Global kerneL argument store
    Store {
        id: TensorId,
        dtype: DType,
    },
    // Movement op, simply changes the view of this buffer. This means moving things around in memory
    // and thus is extremely expensive. We should use memory caching here if possible.
    // Things can be also moved between different memory scopes.
    Movement {
        x: usize,
        scope: u8,
        view: View,
    },
    // Unary and binary ops always take and return variables in registers
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
    // TODO we also need
    // Loop, global, local, register ...
    Loop {
        iters: usize,
        scope: u8,
    },
    EndLoop,
    // Optimation instructions, implementation is hardware specific and thus is up to the compiler
    // Matmul of two 16x16 tiles, result is also 16x16 tile stored in local memory
    NativeMM16x16 {
        x: usize,
        y: usize,
    },
}

pub(crate) fn tiled_to_ir(tiles: BTreeMap<TensorId, Tile>, order: &[TensorId]) -> BTreeMap<TensorId, IRKernel> {
    let mut kernels = BTreeMap::new();

    // At this point every kernel is already 3d, reduce kernels are 4d, with last dim reduce

    // First we write version with only global and register loops, without caching

    for nid in order {
        match tiles[nid].first_op {
            FirstOp::Load { dtype } => {
                let tile = &tiles[nid];
                let sh = tile.view.shape();
                let mut ops = vec![
                    IROp::Loop { iters: sh[0], scope: 0 },
                    IROp::Loop { iters: sh[1], scope: 0 },
                    IROp::Loop { iters: sh[2], scope: 0 },
                    IROp::Load { id: *nid, dtype },
                    IROp::Movement { x: 3, scope: 0, view: tile.view.clone() },
                ];
                for op in &tile.ops {
                    ops.push(IROp::Unary { x: ops.len() - 1, op: *op });
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
                            ops.push(IROp::Unary {
                                x: x + n,
                                op: *op,
                            });
                        }
                        IROp::Binary { x, y, op } => {
                            ops.push(IROp::Binary {
                                x: x + n,
                                y: y + n,
                                op: *op,
                            });
                        }
                        IROp::Loop { .. } => {}
                        _ => { ops.push(op.clone()); }
                    };
                }
                // Add ops from binary tile
                ops.push(IROp::Binary { x: n-1, y: ops.len() - 1, op: BOp::Add });
                for op in &tile.ops {
                    ops.push(IROp::Unary { x: ops.len() - 1, op: *op });
                }
                kernels.insert(*nid, IRKernel { ops });
            }
        }
    }
    // End global loops
    for kernel in kernels.values_mut() {
        for _ in 0..3 {
            kernel.ops.push(IROp::EndLoop);
        }
    }
    kernels
}
