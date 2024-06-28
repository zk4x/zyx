// Here tiles get rewritten into tiles and loops, dimensions get bound
// and optimizations applied. At this stage, all movement and reduce ops are removed.
// Also, there will be special instructions for applying optimizations on like 4x4x4
// matmuls (like strassen or tensor cores) or 16x16x16 matmul (wmma).
// These optimizations are hardware dependent.

use crate::runtime::compiler::{BOp, FirstOp, HWInfo, Scope, Tile, UOp};
use crate::runtime::TensorId;
use crate::runtime::node::Constant;
use crate::DType;
use alloc::collections::BTreeMap;
use alloc::vec;
use alloc::vec::Vec;
use alloc::string::String;
use crate::runtime::view::{Index, View};
use alloc::format as f;

#[derive(Debug)]
pub(in crate::runtime) struct IRKernel {
    pub(super) global_work_size: [usize; 3],
    pub(super) local_work_size: [usize; 3],
    pub(super) args: Vec<IRArg>,
    pub(super) ops: Vec<IROp>,
}

#[derive(Debug, Clone, Copy)]
pub(super) struct IRArg {
    pub(super) dtype: DType,
    pub(super) read_only: bool,
}

#[derive(Debug, Clone)]
pub(super) enum IRMem {
    Const(Constant),
    Var {
        id: u32,
        scope: Scope,
        index: Option<Index>,
    },
}

impl IRMem {
    pub(super) fn to_str(&self, temp_id: u32) -> (Vec<String>, String) {
        match self {
            IRMem::Const(value) => {
                return (Vec::new(), match value {
                    Constant::F32(value) => f!("{}", unsafe { core::mem::transmute::<u32, f32>(*value) }),
                    Constant::I32(value) => f!("{}", value),
                    _ => todo!(),
                })
            }
            IRMem::Var { id, scope, index } => {
                if let Some(idx) = &index {
                    match idx {
                        Index::Contiguous { dims } | Index::Strided { dims } => {
                            let mut res = String::new();
                            for (id, mul) in dims {
                                res += &f!("i{id}*{mul}+");
                            }
                            res.pop();
                            return (Vec::new(), f!("{}{}[{res}]", scope, id))
                        },
                        Index::Reshaped { dims, reshapes, .. } => {
                            let mut res = String::new();
                            for (id, mul) in dims {
                                res += &f!("i{id}*{mul}+");
                            }
                            res.pop();
                            let mut res = vec![res];
                            for reshape in reshapes[..reshapes.len()-1].iter() {
                                let mut idx = String::new();
                                for (div, m, mul) in reshape.iter() {
                                    idx += &f!("t{temp_id}/{div}%{m}*{mul}+");
                                }
                                idx.pop();
                                res.push(idx);
                            }
                            let mut idx = String::new();
                            for (div, m, mul) in reshapes.last().unwrap().iter() {
                                idx += &f!("t{temp_id}/{div}%{m}*{mul}+");
                            }
                            idx.pop();
                            return (res, f!("{}{}[{idx}]", scope, id))
                        },
                    }
                } else {
                    return (Vec::new(), f!("{}{}", scope, id))
                }
            }
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
    DeclareMem {
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
    Barrier {
        scope: Scope,
    },
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
) -> BTreeMap<TensorId, (Vec<TensorId>, IRKernel)> {
    let _ = hwinfo;
    let mut kernels = BTreeMap::new();

    // At this point every kernel is already 8d, reduce kernels are 10d, with last dim reduce
    // and added local loops for first 3 dims and register loops for last 2 dims and reduce dim

    for nid in order {
        let tile = &tiles[nid];
        match tile.first_op {
            FirstOp::Load { dtype, buffer_id } => {
                //libc_print::libc_println!("Nid {nid}");
                let sh = tile.view.shape();
                let (ops, args) = create_unary_kernel(dtype, &sh, &tile.view, &tile.ops);
                kernels.insert(*nid, (vec![buffer_id, *nid], IRKernel {
                    global_work_size: [sh[0], sh[2], sh[5]],
                    local_work_size: [sh[1], sh[3], sh[6]],
                    args,
                    ops
                }));
            }
            FirstOp::Movement { x } => {
                // New movement operation that could not be merged in tiled version,
                // but it perhaps can be merged in IR version (as in IR everything
                // with the same global and local work size gets merged)
                let sh = tile.view.shape();
                let gws = &kernels[&x].1.global_work_size;
                let lws = &kernels[&x].1.local_work_size;
                if gws[0] != sh[0]
                    || lws[0] != sh[1]
                    || gws[1] != sh[2]
                    || lws[1] != sh[3]
                    || gws[2] != sh[5]
                    || lws[3] != sh[6] {
                    // If kernel can not be fused
                    let (ops, args) = create_unary_kernel(tile.dtype, &sh, &tile.view, &tile.ops);
                    kernels.insert(*nid, (
                        vec![x, *nid],
                        IRKernel {
                            global_work_size: [sh[0], sh[2], sh[5]],
                            local_work_size: [sh[1], sh[3], sh[6]],
                            args,
                            ops
                        }
                    ));
                } else {
                    // If kernel can be fused
                    //let ops = kernels[&x].1.ops.clone();
                    //let args = kernels[&x].1.args.clone();
                    todo!();
                    //kernels.insert(*nid, ());
                }
            }
            // These tiled kernels can be fused with previous kernels if reduce and expand
            // kernels exist back to back (with some binary kernels in between and the final
            // work size is the same as the beginning work size.
            FirstOp::Reduce { x, ref shape, ref axes, op } => {
                let sh = tile.view.shape();
                let (ops, args) = create_reduce_kernel(tile.dtype, &sh, &tile.view, &tile.ops);
                kernels.insert(*nid, (
                    vec![x, *nid],
                    IRKernel {
                        global_work_size: [sh[0], sh[2], sh[5]],
                        local_work_size: [sh[1], sh[3], sh[6]],
                        args,
                        ops
                    }
                ));
            }
            // Binary tile fuses two ir kernels together
            FirstOp::Binary { .. } => {
                //let kernel_x = &kernels[&x];
                //let kernel_y = &kernels[&y];
                //let tile = &tiles[nid];
                //let sh = tile.view.shape();
                // Add ops from input tiles
                // Copy directly from tile_x
                //let mut ops = kernel_x.ops.clone();
                //let n = ops.len() - 3; // 3 is the number of loops in kernel_y that are removed
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

    for (_, (_, kernel)) in &mut kernels {
        // Reorder local memory initialization to be first
        for i in 0..kernel.ops.len() {
            if let IROp::DeclareMem { scope, .. } = kernel.ops[i] {
                if scope == Scope::Local {
                    let op = kernel.ops.remove(i);
                    kernel.ops.insert(0, op);
                }
            }
        }
    }

    return kernels
}

fn create_unary_kernel(mut dtype: DType, sh: &[usize], view: &View, uops: &[UOp]) -> (Vec<IROp>, Vec<IRArg>) {
    let first_dtype = dtype;
    let mut ops = Vec::new();
    let l_view = if view.is_expanded() {
        // Add local memory tiling for expanded buffers
        // Dimensions for local tiles are register work size * local work size,
        // that is global index change means load of new tile.
        let strides = view.strides();
        let len = sh.iter().zip(strides.iter()).enumerate()
            .map(|(i, (d, st))| if *st == 0 || [0, 2, 5, 8].contains(&i) { 1 } else { *d })
            .product();
        if len > 1 {
            #[cfg(feature = "debug1")]
            libc_print::libc_println!("Adding local memory tile.");
            ops.insert(0, IROp::DeclareMem {
                id: 0,
                scope: Scope::Local,
                dtype,
                read_only: false,
                // skip expanded dimensions and global work size,
                // use only local * register work size
                len,
            });
        }
        // load from global into local memory
        // if tile is expanded in some dimension that is local,
        // then use threads from that dimension to load different local dimension
        // of this tile.
        for i in [4, 7] {
            if strides[i] != 0 {
                ops.push(IROp::Loop { id: i as u32, max: sh[i] });
            } else {
                ops.push(IROp::Loop { id: i as u32, max: 1 });
            }
        }
        let mut l_sh = [sh[1], sh[3], sh[4], sh[6], sh[7]];
        for (st, d) in strides.iter().zip(&mut l_sh) {
            if *st == 0 {
                *d = 1;
            }
        }
        // TODO change l_view indices for tiles with expanded dimensions
        // being work local dimensions such that these other local threads
        // help load different unexpanded dimension of the tile.
        let l_view = View::from(&l_sh);
        ops.push(IROp::AssignMem {
            z: IRMem::Var {
                id: 0,
                scope: Scope::Local,
                index: Some(l_view.ir_index(&[1, 3, 4, 6, 7])),
            },
            x: IRMem::Var {
                id: 0,
                scope: Scope::Global,
                index: Some(view.ir_index(&[0, 1, 2, 3, 4, 5, 6, 7])),
            },
        });
        ops.push(IROp::EndLoop);
        ops.push(IROp::EndLoop);
        ops.push(IROp::Barrier { scope: Scope::Local });
        Some(l_view)
    } else {
        None
    };
    let mut id = 0;
    // add register loops (for more work per thread)
    ops.push(IROp::Loop { id: 4, max: sh[4] });
    ops.push(IROp::Loop { id: 7, max: sh[7] });
    ops.push(IROp::DeclareMem {
        id: 0,
        scope: Scope::Register,
        dtype,
        read_only: false,
        len: 0,
    });
    ops.push(IROp::AssignMem {
        z: IRMem::Var {
            id: 0,
            scope: Scope::Register,
            index: None,
        },
        x: if let Some(l_view) = l_view {
            IRMem::Var {
                id: 0,
                scope: Scope::Local,
                index: Some(l_view.ir_index(&[1, 3, 4, 6, 7])),
            }
        } else {
            IRMem::Var {
                id: 0,
                scope: Scope::Global,
                index: Some(view.ir_index(&[0, 1, 2, 3, 4, 5, 6, 7])),
            }
        },
    });
    for op in uops {
        let source_id = id;
        if let UOp::Cast(inner_dtype) = *op {
            dtype = inner_dtype;
            id += 1;
            ops.push(IROp::DeclareMem {
                id,
                scope: Scope::Register,
                dtype,
                read_only: false,
                len: 1,
            });
        };
        ops.push(IROp::UnaryMem {
            z: IRMem::Var {
                id,
                scope: Scope::Register,
                index: None,
            },
            x: IRMem::Var {
                id: source_id,
                scope: Scope::Register,
                index: None,
            },
            op: *op,
        });
    }
    // store result to global
    ops.push(IROp::AssignMem {
        z: IRMem::Var {
            id: 1,
            scope: Scope::Global,
            index: Some(View::from(&[sh[0], sh[1], sh[2], sh[3], sh[4], sh[5], sh[6], sh[7]]).ir_index(&[0, 1, 2, 3, 4, 5, 6, 7])),
        },
        x: IRMem::Var {
            id,
            scope: Scope::Register,
            index: None,
        },
    });
    ops.push(IROp::EndLoop);
    ops.push(IROp::EndLoop);
    return (ops, vec![IRArg { dtype: first_dtype, read_only: true }, IRArg { dtype, read_only: false }])
}

fn create_reduce_kernel(mut dtype: DType, sh: &[usize], view: &View, uops: &[UOp]) -> (Vec<IROp>, Vec<IRArg>) {
    let first_dtype = dtype;
    let mut ops = Vec::new();
    let l_view = if view.is_expanded() {
        // Add local memory tiling for expanded buffers
        // Dimensions for local tiles are register work size * local work size,
        // that is global index change means load of new tile.
        let strides = view.strides();
        let len = sh.iter().zip(strides.iter()).enumerate()
            .map(|(i, (d, st))| if *st == 0 || [0, 2, 5, 8].contains(&i) { 1 } else { *d })
            .product();
        if len > 1 {
            #[cfg(feature = "debug1")]
            libc_print::libc_println!("Adding local memory tile.");
            ops.insert(0, IROp::DeclareMem {
                id: 0,
                scope: Scope::Local,
                dtype,
                read_only: false,
                // skip expanded dimensions and global work size,
                // use only local * register work size
                len,
            });
        }
        // load from global into local memory
        // if tile is expanded in some dimension that is local,
        // then use threads from that dimension to load different local dimension
        // of this tile.
        for i in [4, 7] {
            if strides[i] != 0 {
                ops.push(IROp::Loop { id: i as u32, max: sh[i] });
            } else {
                ops.push(IROp::Loop { id: i as u32, max: 1 });
            }
        }
        let mut l_sh = [sh[1], sh[3], sh[4], sh[6], sh[7]];
        for (st, d) in strides.iter().zip(&mut l_sh) {
            if *st == 0 {
                *d = 1;
            }
        }
        // TODO change l_view indices for tiles with expanded dimensions
        // being work local dimensions such that these other local threads
        // help load different unexpanded dimension of the tile.
        let l_view = View::from(&l_sh);
        ops.push(IROp::AssignMem {
            z: IRMem::Var {
                id: 0,
                scope: Scope::Local,
                index: Some(l_view.ir_index(&[1, 3, 4, 6, 7, 9])),
            },
            x: IRMem::Var {
                id: 0,
                scope: Scope::Global,
                index: Some(view.ir_index(&[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])),
            },
        });
        ops.push(IROp::EndLoop);
        ops.push(IROp::EndLoop);
        ops.push(IROp::Barrier { scope: Scope::Local });
        Some(l_view)
    } else {
        None
    };
    let mut id = 0;
    // Create accumulator, size is equal to register work size
    ops.push(IROp::DeclareMem {
        id: 1,
        scope: Scope::Register,
        dtype,
        read_only: false,
        len: sh[4]*sh[7],
    });
    let a_view = View::from(&[sh[4], sh[7]]);
    // Initialize accumulator
    ops.push(IROp::Loop { id: 4, max: sh[4] });
    ops.push(IROp::Loop { id: 7, max: sh[7] });
    ops.push(IROp::AssignMem {
        z: IRMem::Var {
            id: 1,
            scope: Scope::Register,
            index: Some(a_view.ir_index(&[4, 7])),
        },
        x: IRMem::Const(match dtype {
            DType::F32 => Constant::F32(unsafe { core::mem::transmute(0f32) }),
            DType::I32 => Constant::I32(0),
            _ => todo!(),
        }),
    });
    ops.push(IROp::EndLoop);
    ops.push(IROp::EndLoop);
    // Global reduce thread
    ops.push(IROp::Loop { id: 8, max: sh[8] });
    // add register loops (for more work per thread)
    ops.push(IROp::Loop { id: 4, max: sh[4] });
    ops.push(IROp::Loop { id: 7, max: sh[7] });
    ops.push(IROp::Loop { id: 9, max: sh[9] });
    ops.push(IROp::DeclareMem {
        id: 0,
        scope: Scope::Register,
        dtype,
        read_only: false,
        len: 0,
    });
    ops.push(IROp::AssignMem {
        z: IRMem::Var {
            id: 0,
            scope: Scope::Register,
            index: None,
        },
        x: if let Some(l_view) = l_view {
            IRMem::Var {
                id: 0,
                scope: Scope::Local,
                index: Some(l_view.ir_index(&[1, 3, 4, 6, 7, 9])),
            }
        } else {
            IRMem::Var {
                id: 0,
                scope: Scope::Global,
                index: Some(view.ir_index(&[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])),
            }
        },
    });
    for op in uops {
        let source_id = id;
        if let UOp::Cast(inner_dtype) = *op {
            dtype = inner_dtype;
            id += 1;
            ops.push(IROp::DeclareMem {
                id,
                scope: Scope::Register,
                dtype,
                read_only: false,
                len: 1,
            });
        };
        ops.push(IROp::UnaryMem {
            z: IRMem::Var {
                id,
                scope: Scope::Register,
                index: None,
            },
            x: IRMem::Var {
                id: source_id,
                scope: Scope::Register,
                index: None,
            },
            op: *op,
        });
    }
    ops.push(IROp::BinaryMem {
        z: IRMem::Var {
            id: 1,
            scope: Scope::Register,
            index: Some(a_view.ir_index(&[4, 7])),
        },
        x: IRMem::Var {
            id: 0,
            scope: Scope::Register,
            index: None,
        },
        y: IRMem::Var {
            id: 1,
            scope: Scope::Register,
            index: Some(a_view.ir_index(&[4, 7])),
        },
        op: BOp::Add,
    });
    ops.push(IROp::EndLoop);
    ops.push(IROp::EndLoop);
    ops.push(IROp::EndLoop);
    ops.push(IROp::EndLoop);
    ops.push(IROp::Loop { id: 4, max: sh[4] });
    ops.push(IROp::Loop { id: 7, max: sh[7] });
    // store result to global
    ops.push(IROp::AssignMem {
        z: IRMem::Var {
            id: 1,
            scope: Scope::Global,
            index: Some(View::from(&[sh[0], sh[1], sh[2], sh[3], sh[4], sh[5], sh[6], sh[7]]).ir_index(&[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])),
        },
        x: IRMem::Var {
            id: 1,
            scope: Scope::Register,
            index: Some(a_view.ir_index(&[4, 7])),
        },
    });
    ops.push(IROp::EndLoop);
    ops.push(IROp::EndLoop);
    return (ops, vec![IRArg { dtype: first_dtype, read_only: true }, IRArg { dtype, read_only: false }])
}
