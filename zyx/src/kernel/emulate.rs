// Copyright (C) 2025 zk4x
// SPDX-License-Identifier: GPL-2.0-only

use crate::{
    Map,
    dtype::Constant,
    kernel::{IDX_T, Kernel, Op, OpId, Scope},
};

impl Constant {
    /// Converts a Vec<Vec<f32>> into Vec<Vec<Constant>>
    pub fn from_f32_matrix(matrix: Vec<Vec<f32>>) -> Vec<Vec<Constant>> {
        matrix
            .into_iter()
            .map(|row| row.into_iter().map(|v| Constant::F32(v.to_le_bytes())).collect())
            .collect()
    }

    /// Converts a Vec<Vec<i32>> into Vec<Vec<Constant>>
    pub fn from_i32_matrix(matrix: Vec<Vec<i32>>) -> Vec<Vec<Constant>> {
        matrix
            .into_iter()
            .map(|row| row.into_iter().map(|v| Constant::I32(v)).collect())
            .collect()
    }
}

impl Kernel {
    pub fn emulate(&self, params: Vec<Vec<Constant>>) {
        self.debug();

        // Spce for each number in the final table
        let num_space = 5;

        // --- 1. Collect index dimensions ---
        let mut gws = vec![1; 3];
        let mut lws = vec![1; 3];
        for node in self.ops.values() {
            if let Op::Index { len, scope, axis } = node.op {
                let axis = axis as usize;
                match scope {
                    Scope::Global => gws[axis] = len,
                    Scope::Local => lws[axis] = len,
                    Scope::Register => unreachable!(),
                }
            }
        }

        let n_threads = gws.iter().product::<usize>() * lws.iter().product::<usize>();

        // Print header
        print!("Simulating gws={gws:?} lws={lws:?}  ");
        for i in 0..n_threads {
            print!("{:>num_space$}", format!("T{i}"));
        }
        println!();

        // Print indices
        let mut index_combinations = vec![vec![0usize; n_threads]; 6];
        let mut i = 0;
        for gidx0 in 0..gws[0] {
            for gidx1 in 0..gws[1] {
                for gidx2 in 0..gws[2] {
                    for lidx0 in 0..lws[0] {
                        for lidx1 in 0..lws[1] {
                            for lidx2 in 0..lws[2] {
                                index_combinations[0][i] = gidx0;
                                index_combinations[1][i] = gidx1;
                                index_combinations[2][i] = gidx2;
                                index_combinations[3][i] = lidx0;
                                index_combinations[4][i] = lidx1;
                                index_combinations[5][i] = lidx2;
                                i += 1;
                            }
                        }
                    }
                }
            }
        }

        let mut reg_arrays: Map<OpId, Vec<Vec<Constant>>> = Map::default();
        let mut globals_map = Map::default();

        let mut regs: Map<OpId, Vec<Constant>> = Map::default();
        let mut dtypes = Map::default();

        // Print ops
        let mut op_id = self.head;
        while !op_id.is_null() {
            let (text, dtype, reg_values) = match self.ops[op_id].op {
                Op::Cast { x, dtype } => todo!(),
                Op::Unary { x, uop } => todo!(),
                Op::Binary { x, y, bop } => todo!(),
                Op::Const(constant) => {
                    let dtype = constant.dtype();
                    (format!("r{op_id}: {dtype} = {constant}"), dtype, vec![constant; n_threads])
                }
                Op::Define { dtype, scope, ro, len } => {
                    match scope {
                        Scope::Global => {
                            globals_map.insert(op_id, globals_map.len());
                        }
                        Scope::Local => todo!(),
                        Scope::Register => {
                            reg_arrays.insert(op_id, vec![vec![dtype.zero_constant(); len]; n_threads]);
                        }
                    }
                    (format!("r{op_id}: {dtype} = def {scope}"), dtype, Vec::new())
                }
                Op::Store { dst, x, index, vlen } => {
                    let dtype = dtypes[&x];
                    (format!("r{op_id}: {dtype} = r{x}[{index}]"), dtype, Vec::new())
                }
                Op::Load { src, index, vlen } => todo!(),
                Op::Index { len, scope, axis } => (
                    format!("r{op_id}: {IDX_T} = gidx{axis}"),
                    IDX_T,
                    match scope {
                        Scope::Global => index_combinations[axis as usize]
                            .iter()
                            .map(|&i| Constant::idx(i as u64))
                            .collect(),
                        Scope::Local => todo!(),
                        Scope::Register => todo!(),
                    },
                ),
                Op::Loop { len, axis } => {
                    let dtype = IDX_T;
                    //(format!("r{op_id}: {IDX_T} = gidx{axis}"), dtype, vec![; n_threads])
                    todo!()
                }
                Op::EndLoop => todo!(),
                Op::Mad { x, y, z } => todo!(),
                Op::WMMA { dims, layout, dtype, a, b, c } => todo!(),
                Op::Vectorize { ref ops } => todo!(),
                Op::Devectorize { vec, idx } => todo!(),
                _ => todo!(),
            };

            print!("{text:>40}");

            for val in &reg_values {
                print!("{:>num_space$}", val.to_string());
            }

            println!();

            dtypes.insert(op_id, dtype);
            regs.insert(op_id, reg_values);
            op_id = self.next_op(op_id);
        }
    }
}
