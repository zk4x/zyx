use alloc::{vec::Vec, collections::BTreeSet, string::String};
use zyx_core::dtype::DType;
use zyx_core::shape::Shape;
use zyx_core::view::View;
use crate::{ASTOp, ASTROp, BOp, Op};
use crate::ir::{apply_elementwise_op, Var};
use alloc::format;

#[allow(dead_code)]
pub(super) fn compile_tiled_reduce_kernel(
    ast_ops: &[ASTOp],
    arg_views: Vec<View>,
    arg_dtypes: Vec<DType>,
    reduce_dim: usize,
    local_work_size: &[usize],
    register_work_size: &[usize],
    res_shape: Shape,
    reduce_dtype: DType,
    tiled_buffers: BTreeSet<u8>,
    _tiling_axes: BTreeSet<usize>,
) -> Vec<Op> {
    let mut ops = Vec::new();
    let rank = register_work_size.len();

    std::println!("Register work size: {register_work_size:?}");

    // Declare local memory tiles
    for id in &tiled_buffers {
        let view = &arg_views[*id as usize];
        let len: usize = (0..view.shape().rank()).filter_map(|a| if !view.is_expanded_axis(a) { Some(local_work_size[a]) } else { None }).product();
        ops.push(Op::DeclareLocalVar {
            id: *id,
            dtype: arg_dtypes[*id as usize],
            len: len,
        });
    }

    let (reduce_op_i, is_sum_reduce) = ast_ops.iter().enumerate().find(|(_, op)| matches!(op, ASTOp::Reduce(..))).map(|(i, op)| {
        if let ASTOp::Reduce(_, rop) = op {
            if *rop == ASTROp::Sum {
                (i, true)
            } else {
                (i, false)
            }
        } else {
            panic!()
        }
    }).unwrap();

    // Initiliaze accumulator
    ops.push(Op::InitAccumulator {
        id: reduce_op_i as u8,
        dtype: reduce_dtype,
        is_sum_reduce,
        len: None,
    });

    // Reduce loop
    ops.push(Op::Loop {
        id: 3,
        upper_bound: reduce_dim/local_work_size[rank-1],
        step: 1,
    });

    // Add indexes for ops in reduce loop
    for (a, d) in local_work_size.iter().enumerate() {
        if a == rank - 1 {
            break;
        }
        ops.push(Op::InitIndex {
            id: a as u8,
            value: format!("gid{a}*{d}+lid{a}"),
        });
    }

    // Load tiled_buffers into local memory tiles
    ops.push(Op::DeclareIndex {
        id: (rank - 1) as u8
    });
    for id in &tiled_buffers {
        let mut res_index = String::new();
        let view = &arg_views[*id as usize];
        for a in 0..rank-1 {
            if view.is_expanded_axis(a) {
                ops.push(Op::SetIndex {
                    id: (rank - 1) as u8,
                    value: format!("idx{a}"),
                });
                res_index += &format!("lid{a}+");
            } else {
                res_index += &format!("lid{a}*{}+", local_work_size[a]);
            }
        }
        res_index.pop();
        ops.push(Op::LoadGlobalIntoLocal {
            res: *id,
            res_index: res_index,
            arg: *id,
            arg_index: arg_views[*id as usize].cidx(),
        });
    }

    // Synchronize local memory after loading tiled buffers
    ops.push(Op::LocalBarrier);

    // Local tiling reduce loop
    ops.push(Op::Loop {
        id: 4,
        upper_bound: local_work_size[rank-1],
        step: 1,
    });

    // Apply AST ops before reduce
    let mut res_dtype = DType::F32;
    let mut res_id = 0;
    while res_id < reduce_op_i as u8 {
        let op = &ast_ops[res_id as usize];
        match op {
            ASTOp::Leaf(id) => {
                res_dtype = arg_dtypes[*id as usize];
                ops.push(Op::DeclareVar {
                    dtype: res_dtype,
                    id: res_id,
                    len: None,
                });
                let view = &arg_views[*id as usize];
                if !tiled_buffers.contains(id) {
                    ops.push(Op::LoadGlobal {
                        res: Var::Register {
                            id: res_id,
                            index: None,
                        },
                        arg: *id,
                        index: view.cidx(),
                    })
                } else {
                    let mut index = String::new();
                    for a in 0..rank-1 {
                        if !view.is_expanded_axis(a) {
                            // I think adding 1 here gets rid of local memory bank conflicts.
                            index += &format!("lid{a}*{}+", local_work_size[a]);
                        }
                    }
                    index += &format!("rid4");
                    ops.push(Op::LoadLocal {
                        res: Var::Register {
                            id: res_id,
                            index: None,
                        },
                        arg: *id,
                        index,
                    })
                }
            }
            _ => {
                ops.extend(apply_elementwise_op(res_id, &mut res_dtype, op));
            }
        }
        res_id += 1;
    }

    // Apply reduce op
    if is_sum_reduce {
        ops.push(Op::Binary {
            res: Var::Register {
                id: reduce_op_i as u8,
                index: None,
            },
            x: Var::Register {
                id: res_id - 1,
                index: None,
            },
            y: Var::Register {
                id: reduce_op_i as u8,
                index: None,
            },
            op: BOp::Add,
        });
    } else {
        ops.push(Op::Binary {
            res: Var::Register {
                id: reduce_op_i as u8,
                index: None,
            },
            x: Var::Register {
                id: res_id - 1,
                index: None,
            },
            y: Var::Register {
                id: reduce_op_i as u8,
                index: None,
            },
            op: BOp::Max,
        });
    }
    res_id += 1;

    // End local memory tiling loop and synchronize
    ops.push(Op::EndLoop);
    ops.push(Op::LocalBarrier);

    // End reduce loop after reduce op was applied
    ops.push(Op::EndLoop);

    // Add indexes for ops
    for (a, d) in local_work_size.iter().enumerate() {
        if a == rank - 1 {
            break;
        }
        ops.push(Op::InitIndex {
            id: a as u8,
            value: format!("gid{a}*{d}+lid{a}"),
        });
    }

    // Apply ops after reduce
    while res_id < ast_ops.len() as u8 {
        let op = &ast_ops[res_id as usize];
        match op {
            ASTOp::Leaf(id) => {
                res_dtype = arg_dtypes[*id as usize];
                ops.push(Op::DeclareVar {
                    dtype: res_dtype,
                    id: res_id,
                    len: None,
                });
                ops.push(Op::LoadGlobal {
                    res: Var::Register {
                        id: res_id,
                        index: None,
                    },
                    arg: *id,
                    index: arg_views[*id as usize].cidx(),
                })
            }
            _ => {
                ops.extend(apply_elementwise_op(res_id, &mut res_dtype, op));
            }
        }
        res_id += 1;
    }

    // Store result
    ops.push(Op::StoreGlobal {
        res: arg_dtypes.len() as u8,
        index: View::new(res_shape[0..-1].into()).cidx(),
        arg: Var::Register {
            id: res_id - 1,
            index: None,
        },
    });
    ops
}
