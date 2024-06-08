// Here tiles get rewritten into tiles and loops, dimensions get bound
// and optimizations applied. At this stage, all movement and reduce ops are removed.
// Also, there will be special instructions for applying optimizations on like 4x4x4
// matmuls (like strassen or tensor cores) or 16x16x16 matmul (wmma).
// These optimizations are hardware dependent.

use crate::runtime::compiler::{BOp, FirstOp, HWInfo, Scope, Tile, UOp};
use crate::runtime::TensorId;
use crate::DType;
use alloc::collections::BTreeMap;
use alloc::vec;
use alloc::vec::Vec;
use core::fmt::{Display, Formatter, Write};
use crate::runtime::view::{Index, View};

#[derive(Debug)]
pub(super) struct IRKernel {
    pub(super) global_work_size: [usize; 3],
    pub(super) local_work_size: [usize; 3],
    pub(super) args: Vec<IRKernelArg>,
    pub(super) ops: Vec<IROp>,
}

#[derive(Debug)]
pub(super) struct IRKernelArg {
    pub(super) dtype: DType,
    pub(super) read_only: bool,
}

#[derive(Debug, Clone)]
pub(super) struct IRMem {
    id: u32,
    scope: Scope,
    index: Option<Index>,
}

impl Display for IRMem {
    fn fmt(&self, f: &mut Formatter<'_>) -> core::fmt::Result {
        return if let Some(idx) = &self.index {
            f.write_fmt(format_args!("{}mem{}[{idx}]", self.scope, self.id))
        } else {
            f.write_fmt(format_args!("{}mem{}", self.scope, self.id))
        }
    }
}

/// IROp for direct translation to hardware kernels
/// Scope:
/// 0 - global
/// 1 - local
/// 2 - register
#[derive(Debug, Clone)]
pub(super) enum IROp {
    // All variables are 1d, so that it is easier for implementors
    InitMem {
        id: u32,
        scope: Scope,
        dtype: DType,
        read_only: bool,
        len: usize,
    },
    AssignMem {
        z: IRMem,
        x: IRMem,
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
    Loop {
        id: u32,
        max: usize,
    },
    EndLoop,
}

// Movement op, simply changes the view of this buffer. This means moving things around in memory
// and thus is extremely expensive. We should use memory caching here if possible.
// Things can be also moved between different memory scopes.

// Optimation instructions, implementation is hardware specific and thus is up to the compiler
// Matmul of two 16x16 tiles, result is also 16x16 tile stored in local memory

/// Rewrite tiled representation to ir representation, optionally fuse some kernels if possible
/// (if they have the same work size)
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
                let mut first_dtype = dtype;
                let mut dtype = dtype;
                let tile = &tiles[nid];
                let sh = tile.view.shape();
                let mut ops = vec![
                    IROp::InitMem {
                        id: 0,
                        scope: Scope::Register,
                        dtype,
                        read_only: false,
                        len: 0,
                    },
                    IROp::AssignMem {
                        z: IRMem {
                            id: 0,
                            scope: Scope::Register,
                            index: None,
                        },
                        x: IRMem {
                            id: 0,
                            scope: Scope::Global,
                            index: Some(tile.view.ir_index(&[0, 1, 2, 3, 4, 5])),
                        },
                    },
                ];
                // id for the last register value
                let mut id = 0;
                for op in &tile.ops {
                    if let UOp::Cast(inner_dtype) = *op {
                        dtype = inner_dtype;
                    };
                    let source_id = id;
                    id += 1;
                    ops.push(IROp::InitMem {
                        id,
                        scope: Scope::Register,
                        dtype,
                        read_only: false,
                        len: 1,
                    });
                    ops.push(IROp::UnaryMem {
                        z: IRMem {
                            id,
                            scope: Scope::Register,
                            index: None,
                        },
                        x: IRMem {
                            id: source_id,
                            scope: Scope::Register,
                            index: None,
                        },
                        op: *op,
                    });
                }
                // store result to global
                ops.push(IROp::AssignMem {
                    z: IRMem {
                        id: id + 1,
                        scope: Scope::Global,
                        index: Some(View::from(&[sh[0], sh[1], sh[2], sh[3], sh[4], sh[5]]).ir_index(&[0, 1, 2, 3, 4, 5])),
                    },
                    x: IRMem {
                        id,
                        scope: Scope::Register,
                        index: None,
                    },
                });
                kernels.insert(*nid, IRKernel {
                    global_work_size: [sh[0], sh[2], sh[4]],
                    local_work_size: [sh[1], sh[3], sh[5]],
                    args: vec![IRKernelArg { dtype: first_dtype, read_only: true }, IRKernelArg { dtype, read_only: false }],
                    ops
                });
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
                /*for op in &kernel_y.ops {
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
                }*/
                //kernels.insert(*nid, IRKernel { ops });
            }
        }
    }

    // TODO optimize ir_kernels using memory tiling and such

    kernels
}
