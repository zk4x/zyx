use alloc::vec::Vec;
use std::collections::BTreeMap;
use zyx_core::dtype::DType;
use zyx_core::shape::Shape;
use zyx_core::view::{Index, View};
use crate::{ASTOp, ASTROp, BOp, Op, UOp};
use crate::ir::{apply_elementwise_op, Var};

pub(in crate::ir) fn compile_reduce_kernel(
    ast_ops: &[ASTOp],
    arg_views: Vec<View>,
    arg_dtypes: Vec<DType>,
    reduce_dtype: DType,
    reduce_dim: usize,
    local_work_size: &[usize],
    res_shape: Shape,
) -> Vec<Op> {
    let mut ops = Vec::new();
    let rank = res_shape.rank();

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

    // Declare local memory tiles
    ops.push(Op::DeclareLocalVar {
        id: reduce_op_i as u8,
        dtype: reduce_dtype,
        len: local_work_size[2],
    });

    // Add indexes for ops after reduce
    for (a, d) in local_work_size.iter().enumerate() {
        if a != rank - 1 {
            ops.push(Op::InitIndex {
                id: a as u8,
                value: alloc::format!("gid{a}*{d}+lid{a}"),
            });
        }
    }

    // Initiliaze accumulator
    ops.push(Op::InitAccumulator {
        id: reduce_op_i as u8,
        dtype: reduce_dtype,
        is_sum_reduce,
        len: None,
    });

    // First reduce loop
    ops.push(Op::Loop {
        name: "rid0".into(),
        upper_bound: reduce_dim/local_work_size[2],
        step: 1,
    });

    // Indices in reduce loop
    ops.push(Op::InitIndex {
        id: (rank - 1) as u8,
        value: alloc::string::String::from("rid0"),
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
                ops.push(Op::LoadGlobal {
                    res: Var::Register {
                        id: res_id,
                        index: None,
                    },
                    arg: *id,
                    index: view.cidx(),
                })
            }
            _ => {
                ops.extend(apply_elementwise_op(res_id, &mut res_dtype, op, &BTreeMap::new()));
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

    // End reduce loop after reduce op was applied
    ops.push(Op::EndLoop);

    ops.push(Op::Unary {
        res: Var::Local {
            id: reduce_op_i as u8,
            index: "lid2".into(),
        },
        x: Var::Register {
            id: reduce_op_i as u8,
            index: None,
        },
        op: UOp::Noop,
    });

    ops.push(Op::LocalBarrier);

    ops.push(Op::IfBlock { condition: "lid2 < 1".into() });

    // Second reduce loop
    ops.push(Op::Loop {
        name: "rid0".into(),
        upper_bound: local_work_size[2]-1,
        step: 1,
    });

    // Apply reduce op
    if is_sum_reduce {
        ops.push(Op::Binary {
            res: Var::Register {
                id: reduce_op_i as u8,
                index: None,
            },
            x: Var::Local {
                id: res_id - 1,
                index: "rid0+1".into(),
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
            x: Var::Local {
                id: res_id - 1,
                index: "rid0".into(),
            },
            y: Var::Register {
                id: reduce_op_i as u8,
                index: None,
            },
            op: BOp::Max,
        });
    }

    ops.push(Op::EndLoop);

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
                ops.extend(apply_elementwise_op(res_id, &mut res_dtype, op, &BTreeMap::new()));
            }
        }
        res_id += 1;
    }

    // Store result
    ops.push(Op::StoreGlobal {
        res: arg_dtypes.len() as u8,
        index: Index::Normal("0".into()),
        arg: Var::Register {
            id: res_id - 1,
            index: None,
        },
    });

    ops.push(Op::EndIf);

    ops
}
