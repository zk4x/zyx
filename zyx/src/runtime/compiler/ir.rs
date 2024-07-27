use crate::dtype::Constant;
use crate::runtime::graph::Graph;
use crate::runtime::node::{BOp, ROp, UOp};
use crate::tensor::TensorId;
use crate::DType;
use alloc::collections::BTreeMap;
use alloc::collections::BTreeSet;
use alloc::format as f;
use alloc::string::String;
use alloc::vec;
use alloc::vec::Vec;

use super::HWInfo;
use super::Scope;
use super::VOp;

#[derive(Debug)]
pub(crate) struct IRKernel {
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
    Var {
        id: usize,
        scope: Scope,
        index: Index,
    },
}

impl IRMem {
    pub(super) fn to_str(&self, _temp_id: u32) -> (Vec<String>, String) {
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
                /*Index::Reshaped { dims, reshapes, .. } => {
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
                }*/
                Index::None => return (Vec::new(), f!("{}{}", scope, id)),
            },
        }
    }
}

/// IROp for direct translation to hardware kernels
#[derive(Debug, Clone)]
pub(super) enum IROp {
    // All variables are 1d, so that it is easier for implementors
    DeclareMem {
        id: usize,
        scope: Scope,
        dtype: DType,
        read_only: bool,
        len: usize,
        // Initialization is mostly for accumulators
        init: Option<Constant>,
    },
    /// Multiple successive unary ops on register variables
    Unary { z: IRMem, x: IRMem, ops: Vec<UOp> },
    /// Single binary op on register variables, x is scalar
    Binary {
        z: IRMem,
        x: IRMem,
        y: IRMem,
        op: BOp,
    },
    /// Register loop, len is number of iterations, step is 1
    Loop { id: usize, len: usize },
    /// End of register loop
    EndLoop,
    /// Synchronization barrier
    Barrier { scope: Scope },
}

// Movement op, simply changes the view of this buffer. This means moving things around in memory
// and thus is extremely expensive. We should use memory caching here if possible.
// Things can be also moved between different memory scopes.

// Optimation instructions, implementation is hardware specific and thus is up to the compiler
// Matmul of two 16x16 tiles, result is also 16x16 tile stored in local memory

/// Rewrite tiled representation to ir representation, optionally fuse some kernels if possible
/// (if they have the same work size)
pub(super) fn compile_ir(
    graph: &Graph,
    global_work_size: [usize; 3],
    local_work_size: [usize; 3],
    inputs: &BTreeSet<TensorId>,
    outputs: &BTreeSet<TensorId>,
    vops: &[VOp],
    hwinfo: &HWInfo,
) -> IRKernel {
    // Here tiles get rewritten into tiles and loops, dimensions get bound
    // and optimizations applied. At this stage, all movement and reduce ops are removed.
    // Also, there will be special instructions for applying optimizations on like 4x4x4
    // matmuls (like strassen or tensor cores) or 16x16x16 matmul (wmma).
    // These optimizations are hardware dependent.
    let _ = hwinfo;
    let gws = global_work_size;
    let lws = local_work_size;

    let mut ops = Vec::new();

    // Remove first 6 loops, these are global loops.
    for vop in &vops[6..] {
        match vop {
            VOp::Load { z, x, view } => {
                ops.push(IROp::DeclareMem {
                    id: *z,
                    scope: Scope::Register,
                    dtype: graph.dtype(*z),
                    read_only: false,
                    len: 0,
                    init: None,
                });
                ops.push(IROp::Unary {
                    z: IRMem::Var {
                        id: *z,
                        scope: Scope::Register,
                        index: Index::None,
                    },
                    x: IRMem::Var {
                        id: *x,
                        scope: Scope::Global,
                        index: view.index(),
                    },
                    ops: vec![UOp::Noop],
                });
            }
            VOp::Store { z, strides } => {
                ops.push(IROp::Unary {
                    z: IRMem::Var {
                        id: *z,
                        scope: Scope::Global,
                        index: Index::Strided {
                            dims: strides.iter().copied().enumerate().collect(),
                        },
                    },
                    x: IRMem::Var {
                        id: *z,
                        scope: Scope::Register,
                        index: Index::None,
                    },
                    ops: vec![UOp::Noop],
                });
            }
            VOp::Loop { axis, dimension } => {
                ops.push(IROp::Loop {
                    id: *axis,
                    len: *dimension,
                });
            }
            VOp::Accumulator { z, rop } => {
                let dtype = graph.dtype(*z);
                ops.push(IROp::DeclareMem {
                    id: *z,
                    scope: Scope::Register,
                    dtype,
                    read_only: false,
                    len: 0,
                    init: Some(match rop {
                        ROp::Sum => dtype.zero_constant(),
                        ROp::Max => dtype.min_constant(),
                    }),
                });
            }
            VOp::Reduce {
                num_axes,
                rop,
                z,
                x,
            } => {
                let z_var = IRMem::Var {
                    id: *z,
                    scope: Scope::Register,
                    index: Index::None,
                };
                ops.push(IROp::Binary {
                    z: z_var.clone(),
                    x: IRMem::Var {
                        id: *x,
                        scope: Scope::Register,
                        index: Index::None,
                    },
                    y: z_var,
                    op: match rop {
                        ROp::Sum => BOp::Add,
                        ROp::Max => BOp::Max,
                    },
                });
                for _ in 0..*num_axes {
                    ops.push(IROp::EndLoop);
                }
            }
            VOp::Unary { z, x, uop } => {
                ops.push(IROp::DeclareMem {
                    id: *z,
                    scope: Scope::Register,
                    dtype: graph.dtype(*z),
                    read_only: false,
                    len: 0,
                    init: None,
                });
                ops.push(IROp::Unary {
                    z: IRMem::Var {
                        id: *z,
                        scope: Scope::Register,
                        index: Index::None,
                    },
                    x: IRMem::Var {
                        id: *x,
                        scope: Scope::Register,
                        index: Index::None,
                    },
                    ops: vec![*uop],
                });
            }
            VOp::Binary { z, x, y, bop } => {
                ops.push(IROp::DeclareMem {
                    id: *z,
                    scope: Scope::Register,
                    dtype: graph.dtype(*z),
                    read_only: false,
                    len: 0,
                    init: None,
                });
                ops.push(IROp::Binary {
                    z: IRMem::Var {
                        id: *z,
                        scope: Scope::Register,
                        index: Index::None,
                    },
                    x: IRMem::Var {
                        id: *x,
                        scope: Scope::Register,
                        index: Index::None,
                    },
                    y: IRMem::Var {
                        id: *y,
                        scope: Scope::Register,
                        index: Index::None,
                    },
                    op: *bop,
                });
            }
        }
    }

    // Add loop ends
    let mut loop_ends_count = 0;
    for op in &ops {
        match op {
            IROp::Loop { .. } => loop_ends_count += 1,
            IROp::EndLoop { .. } => loop_ends_count -= 1,
            _ => {}
        }
    }
    for _ in 0..loop_ends_count {
        ops.push(IROp::EndLoop);
    }

    let mut args = BTreeMap::new();
    for x in inputs {
        args.insert(
            *x,
            IRArg {
                dtype: graph.dtype(*x),
                read_only: true,
            },
        );
    }
    for x in outputs {
        args.insert(
            *x,
            IRArg {
                dtype: graph.dtype(*x),
                read_only: false,
            },
        );
    }

    return IRKernel {
        global_work_size,
        local_work_size,
        ops,
        args,
    };
}

// With this representation of index, we can find repeating
// multipliers and extract them out into common factors.
// However this would be a bit of micro-optimization, as OpenCL, CUDA, WGPU
// and most other compilers extract them automatically.
// This will be needed if we want to directly generate SPIR or PTX IR.

/// Virtual representation of index into view
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Index {
    /// For variables that only have single element (scalars),
    /// such as most register variables.
    None,
    /// Pairs of index id and multiplier.
    /// Can use wide loads directly with pointer casts.
    Contiguous {
        /// Dimension and multiplier
        dims: BTreeMap<usize, usize>,
        // When should the padding get applied?
        //padding_condition: String,
    },
    /// Expanded and/or permuted
    /// Pairs of index id and multiplier.
    /// Wide loads are possible only if we can transpose it in the kernel
    Strided {
        /// Dimension and multiplier
        dims: BTreeMap<usize, usize>,
        // When should the padding get applied?
        //padding_condition: String,
    },
    // Expanded, permuted and/or padded
    // Only if reshape could not be merged.
    /*Padded {
    /// Multiple dimension and multipliers
    dims: BTreeMap<usize, usize>,
    /// When should the padding get applied?
    padding_condition: String,
    },*/
}
