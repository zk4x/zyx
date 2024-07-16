// Here tiles get rewritten into tiles and loops, dimensions get bound
// and optimizations applied. At this stage, all movement and reduce ops are removed.
// Also, there will be special instructions for applying optimizations on like 4x4x4
// matmuls (like strassen or tensor cores) or 16x16x16 matmul (wmma).
// These optimizations are hardware dependent.

use crate::dtype::Constant;
use crate::runtime::compiler::{HWInfo, Scope};
use crate::runtime::graph::Graph;
use crate::runtime::node::UOp;
use crate::runtime::view::Index;
use crate::runtime::TensorId;
use crate::DType;
use alloc::collections::{BTreeMap, BTreeSet};
use alloc::format as f;
use alloc::string::String;
use alloc::vec;
use alloc::vec::Vec;

#[cfg(feature = "debug1")]
use libc_print::std_name::println;

use super::VOp;

#[derive(Debug)]
pub(in crate::runtime) struct IRKernel {
    pub(super) global_work_size: [usize; 3],
    pub(super) local_work_size: [usize; 3],
    pub(super) args: BTreeMap<TensorId, IRArg>,
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
    Var { id: u32, scope: Scope, index: Index },
}

impl IRMem {
    pub(super) fn to_str(&self, temp_id: u32) -> (Vec<String>, String) {
        match self {
            IRMem::Const(value) => {
                return (
                    Vec::new(),
                    match value {
                        Constant::F32(value) => {
                            f!("{}", unsafe { core::mem::transmute::<u32, f32>(*value) })
                        }
                        Constant::I32(value) => f!("{}", value),
                        _ => todo!(),
                    },
                )
            }
            IRMem::Var { id, scope, index } => match index {
                Index::Contiguous { dims } | Index::Strided { dims } => {
                    let mut res = String::new();
                    for (id, mul) in dims {
                        res += &f!("i{id}*{mul}+");
                    }
                    res.pop();
                    return (Vec::new(), f!("{}{}[{res}]", scope, id));
                }
                Index::Reshaped { dims, reshapes, .. } => {
                    let mut res = String::new();
                    for (id, mul) in dims {
                        res += &f!("i{id}*{mul}+");
                    }
                    res.pop();
                    let mut res = vec![res];
                    for reshape in reshapes[..reshapes.len() - 1].iter() {
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
                    return (res, f!("{}{}[{idx}]", scope, id));
                }
                Index::None => return (Vec::new(), f!("{}{}", scope, id)),
            },
        }
    }
}

#[derive(Debug, Clone)]
pub(crate) enum BOp {
    Add,
    Sub,
    Mul,
    Div,
    Max,
    Pow,
    Cmplt,
}

/// IROp for direct translation to hardware kernels
#[derive(Debug, Clone)]
pub(super) enum IROp {
    // All variables are 1d, so that it is easier for implementors
    DeclareMem {
        id: u32,
        scope: Scope,
        dtype: DType,
        read_only: bool,
        len: usize,
        // Initialization is mostly for accumulators
        init: Option<Constant>,
    },
    AssignMem {
        z: IRMem,
        x: IRMem,
    },
    /// Multiple successive unary ops on register variables
    Unary {
        z: IRMem,
        x: IRMem,
        ops: Vec<UOp>,
    },
    /// Single binary op on register variables, x is scalar
    Binary {
        z: IRMem,
        x: IRMem,
        y: IRMem,
        op: BOp,
    },
    /// Register loop, len is number of iterations, step is 1
    Loop {
        id: u32,
        len: usize,
    },
    /// End of register loop
    EndLoop,
    /// Synchronization barrier
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
pub(crate) fn compile_ir(
    graph: &Graph,
    global_work_size: [usize; 3],
    local_work_size: [usize; 3],
    inputs: &BTreeSet<TensorId>,
    outputs: &BTreeSet<TensorId>,
    ops: &[VOp],
    hwinfo: &HWInfo,
) -> IRKernel {
    let _ = hwinfo;
    let gws = global_work_size;
    let lws = local_work_size;

    /*let mut ops = Vec::new();

    // At this point every kernel is already 8d, reduce kernels are 10d, with last dim reduce
    // and added local loops for first 3 dims and register loops for last 2 dims and reduce dim

    // A map of global, local and register variables and their views, or probably just indices.
    // The second value is true if read only variables, else it is false.
    let mut vars: BTreeMap<(TensorId, Scope), (DType, Index, bool)> = BTreeMap::new();
    let mut last_register_loops = [0; 2];
    // First of these is "global" reduce, second is register
    let mut last_reduce_loops = [0; 2];

    for tile in tiles {
        match tile {
            Tile::Load { z, x, dtype, view } => {
                let sh = view.shape();
                let index = if sh.len() == 8 {
                    // It isn't reduce load
                    view.ir_index(&[0, 1, 2, 3, 4, 5, 6, 7])
                } else {
                    // It is reduce load
                    // Add global reduce loop if it does not exist yet
                    if last_reduce_loops != [sh[8], sh[9]] {
                        if last_reduce_loops != [0, 0] {
                            ops.push(IROp::EndLoop);
                            ops.push(IROp::EndLoop);
                        }
                        // Declare accumulator
                        let dtype = todo!();
                        ops.push(IROp::DeclareMem {
                            id: todo!(),
                            dtype,
                            scope: Scope::Register,
                            // TODO set to dtype.min() if it is ROp::Max
                            init: Some(dtype.zero()),
                            len: sh[8] * sh[9],
                            read_only: false,
                        });
                        // Global reduce loop
                        ops.push(IROp::Loop { id: 8, max: sh[8] });
                        last_reduce_loops = [sh[8], sh[9]];

                        // Here come local memory loads
                    }
                    view.ir_index(&[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
                };
                // End previous register loops
                if last_register_loops != [sh[4], sh[7]] {
                    // Add register loops
                    ops.push(IROp::Loop { id: 4, max: sh[4] });
                    ops.push(IROp::Loop { id: 7, max: sh[7] });

                    if sh.len() == 10 {
                        // Register reduce loop
                        ops.push(IROp::Loop { id: 9, max: sh[9] });
                    }
                }
                // Declare register variable for x
                vars.insert((z, Scope::Register), (dtype, Index::None, false));
                ops.push(IROp::DeclareMem {
                    id: z,
                    scope: Scope::Register,
                    dtype,
                    read_only: false,
                    len: 0,
                    init: None,
                });
                // Load from global to register variable
                vars.insert((x, Scope::Global), (dtype, index.clone(), true));
                ops.push(IROp::AssignMem {
                    z_id: z,
                    z_scope: Scope::Register,
                    z_index: Index::None,
                    x_id: x,
                    x_scope: Scope::Global,
                    x_index: index,
                });
            }
            Tile::Unary {
                z,
                x,
                z_dtype,
                ops: uops,
            } => {
                vars.insert((z, Scope::Register), (z_dtype, Index::None, false));
                ops.push(IROp::DeclareMem {
                    id: z,
                    scope: Scope::Register,
                    dtype: z_dtype,
                    read_only: false,
                    len: 0,
                    init: None,
                });
                ops.push(IROp::Unary {
                    z,
                    x,
                    index: vars[&(x, Scope::Register)].1.clone(),
                    ops: uops,
                });
            }
            // Binary tile fuses two ir kernels together
            Tile::Binary { z, x, y, op } => {
                // Even two reduce kernels are fusable now that we can add global
                // temporary variables.
                //println!("Kernel x ops {:?}", kernels[&x].1.ops);
                //println!("Kernel y ops {:?}", kernels[&y].1.ops);
                // Thus we can remove assignements to global variables
                // and directly apply binary operation on register variables.
                // If inputs are reduce kernels, we need to work with accumulators.
                todo!()
            }
            // These tiled kernels can be fused with previous kernels if reduce and expand
            // kernels exist back to back (with some binary kernels in between and the final
            // work size is the same as the beginning work size.
            Tile::ReduceEnd { z, x, op } => {
                // Apply reduce op to the accumulator
                match op {
                    ROp::Sum => {
                        ops.push(IROp::Binary {
                            z,
                            x,
                            y: z,
                            zy_index: vars[&(z, Scope::Register)].1.clone(),
                            op: BOp::Add,
                        });
                    }
                    ROp::Max => {
                        todo!()
                    }
                }
                // End register reduce loop
                ops.push(IROp::EndLoop);
                // End register loops
                ops.push(IROp::EndLoop);
                ops.push(IROp::EndLoop);
                // End global reduce loop
                ops.push(IROp::EndLoop);
                last_register_loops = [0, 0];
                last_reduce_loops = [0, 0];
            }
            Tile::Store { z, dtype } => {
                // TODO get correct register loop sizes for this view
                let index = View::from(&[gws[0], lws[0], gws[1], lws[1], 1, gws[2], lws[2], 1])
                    .ir_index(&[0, 1, 2, 3, 4, 5, 6, 7]);
                // Add it to kernel arguments
                vars.insert((z, Scope::Global), (dtype, index.clone(), false));
                // Store from register to global variable
                ops.push(IROp::AssignMem {
                    z_id: z,
                    z_scope: Scope::Global,
                    z_index: index,
                    x_id: z,
                    x_scope: Scope::Register,
                    x_index: vars[&(z, Scope::Register)].1.clone(),
                });
            }
            _ => {}
        }
    }

    // Add register loop endings
    ops.push(IROp::EndLoop);
    ops.push(IROp::EndLoop);

    // Reorder local memory initialization to be first
    for i in 0..ops.len() {
        if let IROp::DeclareMem { scope, .. } = ops[i] {
            if scope == Scope::Local {
                let op = ops.remove(i);
                ops.insert(0, op);
            }
        }
    }

    // Get global variables from all variables
    let mut args = BTreeMap::new();
    for ((id, scope), (dtype, _, read_only)) in vars {
        if scope == Scope::Global {
            if !read_only || !args.contains_key(&id) {
                args.insert(id, IRArg { dtype, read_only });
            }
        }
    }

    return IRKernel {
        global_work_size,
        local_work_size,
        ops,
        args,
        };*/
    todo!()
}

/*fn create_unary_kernel(mut dtype: DType, sh: &[usize], view: &View, uops: &[UOp]) -> (Vec<IROp>, Vec<IRArg>) {
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
            println!("Adding local memory tile.");
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
}*/
