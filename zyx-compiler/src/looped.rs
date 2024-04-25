// Here tiles get rewritten into tiles and loops, dimensions get bound
// and optimizations applied. At this stage, all movement and reduce ops are removed.
// Also, there will be special instructions for applying optimizations on like 4x4x4
// matmuls (like strassen or tensor cores) or 16x16x16 matmul (wmma).
// These optimizations are hardware dependent.

use zyx_core::tensor::Id;
use alloc::collections::BTreeMap;
use alloc::vec;
use crate::{IRKernel, Op};
use crate::tiled::{BOp, FirstOp, Tile};

pub(crate) fn tiled_to_ir(tiles: BTreeMap<Id, Tile>, order: &[Id]) -> BTreeMap<Id, IRKernel> {
    let mut kernels = BTreeMap::new();

    // At this point every kernel is already 3d, reduce kernels are 4d, with last dim reduce

    // First we write version with only global and register loops, without caching

    for nid in order {
        match tiles[nid].first_op {
            FirstOp::Load { dtype } => {
                let tile = &tiles[nid];
                let sh = tile.view.shape();
                let mut ops = vec![
                    Op::Loop { iters: sh[0], scope: 0 },
                    Op::Loop { iters: sh[1], scope: 0 },
                    Op::Loop { iters: sh[2], scope: 0 },
                    Op::Arg { id: *nid, dtype },
                    Op::Movement { x: 3, view: tile.view.clone() },
                ];
                for op in &tile.ops {
                    ops.push(Op::Unary { x: ops.len() - 1, op: *op });
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
                        Op::Movement { x, view } => {
                            ops.push(Op::Movement {
                                x: x + n,
                                view: view.clone(),
                            });
                        }
                        Op::Unary { x, op } => {
                            ops.push(Op::Unary {
                                x: x + n,
                                op: *op,
                            });
                        }
                        Op::Binary { x, y, op } => {
                            ops.push(Op::Binary {
                                x: x + n,
                                y: y + n,
                                op: *op,
                            });
                        }
                        Op::Loop { .. } => {}
                        _ => { ops.push(op.clone()); }
                    };
                }
                // Add ops from binary tile
                ops.push(Op::Binary { x: n-1, y: ops.len() - 1, op: BOp::Add });
                for op in &tile.ops {
                    ops.push(Op::Unary { x: ops.len() - 1, op: *op });
                }
                kernels.insert(*nid, IRKernel { ops });
            }
        }
    }
    // End global loops
    for kernel in kernels.values_mut() {
        for _ in 0..3 {
            kernel.ops.push(Op::EndLoop);
        }
    }
    kernels
}
