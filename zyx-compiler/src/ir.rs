use crate::{AST, ASTOp};
use alloc::vec::Vec;
use alloc::collections::{BTreeSet, BTreeMap};
use core::fmt::{Display, Formatter};
use zyx_core::axes::{Axes, IntoAxes};
use zyx_core::dtype::DType;
use zyx_core::shape::Shape;
use zyx_core::view::{Index, View};

/// Variable in IR
pub enum Var {
    Local {
        id: u8,
        index: String,
    },
    Register {
        id: u8,
        index: Option<String>,
    },
    ConstF32(f32),
    ConstF64(f64),
    ConstI32(i32),
    ConstI64(i64),
}

impl Display for Var {
    fn fmt(&self, f: &mut Formatter<'_>) -> core::fmt::Result {
        match self {
            Var::Local { id, index } => {
                f.write_fmt(format_args!("lmem{id}[{index}]"))
            }
            Var::Register { id, index } => {
                if let Some(index) = index {
                    f.write_fmt(format_args!("rmem{id}[{index}]"))
                } else {
                    f.write_fmt(format_args!("rmem{id}"))
                }
            }
            Var::ConstF32(_) => { todo!() }
            Var::ConstF64(_) => { todo!() }
            Var::ConstI32(_) => { todo!() }
            Var::ConstI64(_) => { todo!() }
        }
    }
}

pub enum Op {
    LoadGlobal {
        res: Var,
        arg: u8,
        index: Index,
    },
    StoreGlobal {
        res: u8,
        index: Index,
        arg: Var,
    },
    DeclareVar {
        dtype: DType,
        id: u8,
        len: Option<u8>,
    },
    Exp { res: Var, args: [Var; 1] },
    Add { res: Var, args: [Var; 2] },
    Max { res: Var, args: [Var; 2] },
    AddIdx { res: Index, args: [Index; 2] },
    MulIdx { res: Index, args: [Index; 2] },
    // Loop inside kernel (register/private)
    Loop {
        id: u8,
        upper_bound: usize,
        step: usize,
    },
    EndLoop,
}

/// Intermediate representation for compilers
pub struct IR {
    pub global_work_size: Vec<usize>,
    pub local_work_size: Vec<usize>,
    pub kernel_args: Vec<(DType, bool)>, // dtype and read_only
    pub ops: Vec<Op>,
    pub res_byte_size: usize,
}

/// Calculates arg_views, reduce dim size, global, local and register work sizes,
/// each across three dimensions
fn calculate_work_sizes(
    ast_reduce_axes: &Option<Axes>,
    ast_shape: &Shape,
    ast_arg_views: Vec<View>,
    max_local_work_size: usize,
    max_register_work_size: usize,
) -> (Vec<View>, Option<usize>, Vec<usize>, Vec<usize>, Vec<usize>) {
    let (arg_views, shape, reduce_dim) = if let Some(reduce_axes) = &ast_reduce_axes {
        let mut arg_views = ast_arg_views.clone();
        let rank = ast_shape.rank();
        let permute_axes = (0..rank as i64)
            .filter(|a| !reduce_axes.contains(*a as usize))
            .chain(reduce_axes.iter().map(|a| *a as i64))
            .collect::<Box<_>>()
            .into_axes(rank);
        let shape = if rank > 4 || reduce_axes.len() > 1 {
            let d1: usize = ast_shape
                .iter()
                .enumerate()
                .filter_map(|(a, d)| {
                    if reduce_axes.contains(a) {
                        Some(*d)
                    } else {
                        None
                    }
                })
                .product();
            let d0 = ast_shape.numel() / d1;
            let shape: Shape = [d0, d1].into();
            for view in &mut arg_views {
                *view = view.permute(&permute_axes).reshape(&shape);
            }
            shape
        } else {
            for view in &mut arg_views {
                *view = view.permute(&permute_axes);
            }
            ast_shape.permute(&permute_axes)
        };
        let reduce_dim = shape[-1];
        (arg_views, shape, Some(reduce_dim))
    } else {
        let mut arg_views = ast_arg_views.clone();
        let shape = if ast_shape.rank() > 3 {
            let n = ast_shape.numel();
            for view in &mut arg_views {
                *view = view.reshape(&[n].into());
            }
            [n].into()
        } else {
            ast_shape.clone()
        };
        (arg_views, shape, None)
    };
    let mut lws = 1;
    let rank = shape.rank();
    let mut register_work_size = Vec::new();
    let mut global_work_size: Vec<usize> = shape
        .iter()
        .enumerate()
        .filter_map(|(i, d)| {
            if reduce_dim.is_some() && i == rank - 1 {
                None
            } else {
                let mut d = *d;
                register_work_size.push(1);
                while d % 2 == 0 && register_work_size[i] * 2 <= max_register_work_size {
                    register_work_size[i] *= 2;
                    d /= 2;
                }
                Some(d)
            }
        })
        .collect();
    let mut full_reduce = false; // reduce across all axes
    if global_work_size.len() == 0 {
        full_reduce = true;
        global_work_size.push(1);
    }
    // Runtimes are horrible at inferring local work sizes, we just have to give it our
    let local_work_size: Vec<usize> = global_work_size
        .iter()
        .rev()
        .map(|d| {
            let mut x = 1;
            while d % (x * 2) == 0 && x * lws < max_local_work_size {
                x *= 2;
            }
            lws *= x;
            x
        })
        .collect::<Vec<_>>()
        .into_iter()
        .rev()
        .collect();
    (
        arg_views,
        reduce_dim,
        global_work_size,
        local_work_size,
        register_work_size,
    )
}

// Single most important function and one of the most difficult
// functions to write. All of this is cached, so take your time to optimize
// these kernels.
pub(super) fn ast_to_ir(ast: &AST, max_local_work_size: usize, max_register_work_size: usize, max_register_tiled_leafs: usize) -> IR {
    // Register tile size in reduce dimension
    let rtsr = 4u8;
    let res_byte_size = if let Some(reduce_axes) = &ast.reduce_axes {
        ast.shape.clone().reduce(reduce_axes).numel() * ast.dtype.byte_size()
    } else {
        ast.shape.numel() * ast.dtype.byte_size()
    };
    // TODO Repeat these functions with different parameters (autotuning)
    let (arg_views,
         reduce_dim,
         global_work_size,
         local_work_size,
         register_work_size) =
        calculate_work_sizes(
            &ast.reduce_axes,
            &ast.shape,
            ast.arg_views.clone(),
            max_local_work_size,
            max_register_work_size,
        );
    let mut kernel_args = Vec::new();
    for dtype in &ast.arg_dtypes {
        kernel_args.push((*dtype, true));
    }
    // Push result buffer as arg
    kernel_args.push((ast.dtype, false));

    let mut ops = Vec::new();

    // TODO Load loops for local memory tiles

    let mut register_tiled_leafs = BTreeMap::new();

    // Id of the currently processed op
    let mut res_id = 0;
    // Determine number and order of loops
    // First global reduce loop if any
    if let Some(reduce_dim) = reduce_dim {
        ops.push(Op::Loop {
            id: 0,
            upper_bound: reduce_dim/rtsr as usize,
            step: rtsr as usize,
        });

        // Determine which leafs should be register tiled
        let mut reduce_op_i= 0;
        for (i, op) in ast.ops.iter().enumerate() {
            match op {
                ASTOp::Leaf(id) => {
                    if register_tiled_leafs.len() < max_register_tiled_leafs {
                        let view = &ast.arg_views[(*id) as usize];
                        let mut expanded_axes = BTreeSet::new();
                        for a in 0..view.shape().rank() {
                            if view.is_expanded_axis(a) {
                                expanded_axes.insert(a);
                            }
                        }
                        register_tiled_leafs.insert(*id, expanded_axes);
                    }
                }
                ASTOp::Sum(_) | ASTOp::Max(_) => {
                    reduce_op_i = i;
                    break;
                }
                _ => {}
            }
        }

        // Declare all register tiled leafs
        for (id, expanded_axes) in register_tiled_leafs.iter() {
            let dtype = ast.arg_dtypes[(*id) as usize];
            let mut len = 1 as u8;
            for a in expanded_axes {
                len *= register_work_size[*a] as u8;
            }
            ops.push(Op::DeclareVar {
                dtype,
                id: *id,
                len: Some(len * rtsr),
            });
        }

        // Load all register tiled leafs before reduce
        for (id, _) in register_tiled_leafs.iter() {
            let view = &ast.arg_views[(*id) as usize];
            // Create indexes for view
            ops.push(Op::LoadGlobal {
                res: Var::Register { id: *id, index: None },
                arg: *id,
                index: view.cidx(),
            })
        }

        // Apply ops before reduce
        for op in &ast.ops[..reduce_op_i] {
            add_elementwise_op(res_id, op, &mut ops, &register_tiled_leafs, None);
            res_id += 1;
        }

        // Apply reduce op
        match &ast.ops[reduce_op_i] {
            ASTOp::Sum(_) => {}
            ASTOp::Max(_) => {}
            _ => panic!(),
        }
        res_id += 1;

        // End reduce loop after reduce op was applied
        ops.push(Op::EndLoop);

        // Apply ops after reduce
        for op in &ast.ops[reduce_op_i+1..] {
            add_elementwise_op(res_id, op, &mut ops, &register_tiled_leafs, None);
            res_id += 1;
        }
    } else {
        // Not a reduce kernel, so apply all ops in order
        for op in &ast.ops {
            add_elementwise_op(res_id, op, &mut ops, &register_tiled_leafs, None);
            res_id += 1;
        }
    }

    IR {
        global_work_size,
        local_work_size,
        kernel_args,
        ops,
        res_byte_size,
    }
}

// Same op can be applied multiple times with different register_index
fn add_elementwise_op(res_id: u8, op: &ASTOp, ops: &mut Vec<Op>, register_tiled_leafs: &BTreeMap<u8, BTreeSet<usize>>, register_index: Option<String>) {
    match op {
        ASTOp::Leaf(_) => {
            // TODO if not in register_tiled_leafs, then load
        }
        ASTOp::Cast(_, _) => {}
        ASTOp::Neg(_) => {}
        ASTOp::ReLU(_) => {}
        ASTOp::Sin(_) => {}
        ASTOp::Cos(_) => {}
        ASTOp::Ln(_) => {}
        ASTOp::Exp(x) => {
            if register_tiled_leafs.contains_key(x) {
                let index = register_index;
                ops.push(Op::Exp { res: Var::Register { id: res_id, index: index.clone(), }, args: [Var::Register { id: *x, index }] })
            } else {
                ops.push(Op::Exp { res: Var::Register { id: res_id, index: None }, args: [Var::Register { id: *x, index: None }] })
            }
        }
        ASTOp::Tanh(_) => {}
        ASTOp::Sqrt(_) => {}
        ASTOp::Add(_, _) => {}
        ASTOp::Sub(_, _) => {}
        ASTOp::Mul(_, _) => {}
        ASTOp::Div(_, _) => {}
        ASTOp::Pow(_, _) => {}
        ASTOp::Cmplt(_, _) => {}
        ASTOp::Where(_, _, _) => {}
        ASTOp::Sum(_) | ASTOp::Max(_) => {
            panic!()
        }
    }
}
