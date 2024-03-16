use alloc::vec::Vec;
use zyx_core::dtype::DType;
use zyx_core::shape::Shape;
use zyx_core::view::View;
use crate::{AST, ASTOp, Op};
use crate::ir::{apply_elementwise_op, Var};
use alloc::format as f;
use std::collections::BTreeMap;

pub(super) fn compile_elementwise_kernel(
    ast: &AST,
    local_work_size: &[usize],
    arg_views: Vec<View>,
    res_shape: Shape,
) -> Vec<Op> {
    let mut ops = Vec::new();
    // Add indexes
    for (a, d) in local_work_size.iter().enumerate() {
        ops.push(Op::InitIndex {
            id: a as u8,
            value: f!("gid{a}*{d}+lid{a}"),
        });
    }
    // Compile AST ops
    let mut res_dtype = DType::F32;
    let mut res_id = 0;
    while res_id < ast.ops.len() as u8 {
        let op = &ast.ops[res_id as usize];
        match op {
            ASTOp::Leaf(id) => {
                res_dtype = ast.arg_dtypes[*id as usize];
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
        res: ast.arg_dtypes.len() as u8,
        index: View::new(res_shape.clone()).cidx(),
        arg: Var::Register {
            id: res_id - 1,
            index: None,
        },
    });
    ops
}

