// Copyright (C) 2025 zk4x
// SPDX-License-Identifier: LGPL-3.0-only

use std::collections::BTreeMap;

use crate::{
    Set,
    dtype::Constant,
    kernel::{BOp, IDX_T, Kernel, MoveOp, Op, OpId, Scope},
    shape::{Dim, UAxis},
};

impl Kernel {
    pub fn unfold_movement_ops(&mut self) {
        // Apply movement ops on views
        let mut op_id = self.head;
        while !op_id.is_null() {
            if let Op::Move { x, ref mop } = self.ops[op_id].op {
                self.recursively_move(x, &mop.clone(), &mut Set::default(), 0);
            }
            op_id = self.next_op(op_id);
        }
        // Drop movement ops
        let mut op_id = self.head;
        while !op_id.is_null() {
            let next_op_id = self.next_op(op_id);
            if let Op::Move { x, .. } = self.ops[op_id].op {
                self.remap(op_id, x);
                self.remove_op(op_id);
            }
            op_id = next_op_id;
        }

        // Add global ids
        let shape = self.shape();
        let mut axis = shape.len() as u32;
        for len in shape.into_iter().rev() {
            axis -= 1;
            self.insert_before(self.head, Op::Index { len, scope: Scope::Global, axis });
        }

        self.verify();

        self.unfold_reduces();
        self.unfold_views();
    }

    pub fn recursively_move(&mut self, op_id: OpId, move_op: &MoveOp, visited: &mut Set<OpId>, n_reduce_axes: UAxis) {
        if !visited.insert(op_id) {
            return;
        }
        match &mut self.ops[op_id].op {
            Op::LoadView(x) => match move_op {
                MoveOp::Reshape { shape } => {
                    x.1.reshape(0..x.1.rank() - n_reduce_axes, shape);
                }
                MoveOp::Expand { shape } => x.1.expand(shape),
                MoveOp::Permute { axes, .. } => x.1.permute(axes),
                MoveOp::Pad { padding, .. } => x.1.pad(x.1.rank() - n_reduce_axes, padding),
            },
            Op::ConstView(x) => match move_op {
                MoveOp::Reshape { shape } => x.1.reshape(0..x.1.rank() - n_reduce_axes, shape),
                MoveOp::Expand { shape } => x.1.expand(shape),
                MoveOp::Permute { axes, .. } => x.1.permute(axes),
                MoveOp::Pad { padding, .. } => x.1.pad(x.1.rank() - n_reduce_axes, padding),
            },
            &mut Op::Reduce { x, n_axes, .. } => {
                self.recursively_move(x, move_op, visited, n_reduce_axes + n_axes);
            }
            &mut Op::Cast { x, .. } | &mut Op::Unary { x, .. } | &mut Op::Move { x, .. } => {
                self.recursively_move(x, move_op, visited, n_reduce_axes);
            }
            &mut Op::Binary { x, y, .. } => {
                self.recursively_move(x, move_op, visited, n_reduce_axes);
                self.recursively_move(y, move_op, visited, n_reduce_axes);
            }
            _ => {}
        }
    }

    pub fn reduce_dims(&self, op_id: OpId) -> Vec<Dim> {
        let mut params = vec![op_id];
        let mut n_reduce_axes = 0;
        let mut visited = Set::default();
        while let Some(param) = params.pop() {
            if visited.insert(param) {
                match self.at(param) {
                    Op::ConstView(x) => {
                        let view = &x.1;
                        let n = view.rank();
                        return view.shape()[n - n_reduce_axes..].into();
                    }
                    Op::LoadView(x) => {
                        let view = &x.1;
                        let n = view.rank();
                        return view.shape()[n - n_reduce_axes..].into();
                    }
                    Op::Reduce { n_axes, .. } => n_reduce_axes += n_axes,
                    Op::Move { mop, .. } => match mop.as_ref() {
                        MoveOp::Reshape { shape, .. }
                        | MoveOp::Expand { shape }
                        | MoveOp::Permute { shape, .. }
                        | MoveOp::Pad { shape, .. } => {
                            return shape[shape.len() - n_reduce_axes..].into();
                        }
                    },
                    _ => {}
                }
                params.extend(self.at(param).parameters());
            }
        }
        unreachable!();
    }

    pub fn unfold_reduces(&mut self) {
        let mut reduce_op_ids: Vec<OpId> = Vec::new();
        let mut op_id = self.head;
        while !op_id.is_null() {
            if let Op::Reduce { .. } = self.at(op_id) {
                reduce_op_ids.push(op_id);
            }
            op_id = self.next_op(op_id);
        }

        while let Some(reduce_op_id) = reduce_op_ids.pop() {
            let Op::Reduce { x, rop, n_axes } = self.ops[reduce_op_id].op else { unreachable!() };

            let mut reduce_loop_ops_set = Set::default();
            let mut params = vec![x];
            let mut acc_dtype = None;
            while let Some(param) = params.pop() {
                if reduce_loop_ops_set.insert(param) {
                    params.extend(self.at(param).parameters());
                    if acc_dtype.is_none() {
                        match self.at(param) {
                            &Op::Define { dtype, .. } | &Op::Cast { dtype, .. } => acc_dtype = Some(dtype),
                            Op::ConstView(x) => acc_dtype = Some(x.0.dtype()),
                            Op::LoadView(x) => acc_dtype = Some(x.0),
                            _ => {}
                        }
                    }
                }
            }
            let acc_dtype = acc_dtype.unwrap();
            // Sort reduce loop ops by original order
            let mut op_id = self.head;
            let mut loop_start = OpId::NULL;
            while !op_id.is_null() {
                if reduce_loop_ops_set.contains(&op_id) {
                    loop_start = op_id;
                    break;
                }
                op_id = self.next_op(op_id);
            }

            // Add const zero
            let const_zero = self.insert_before(loop_start, Op::Const(Constant::idx(0)));

            // Add accumulator
            let acc_init_id = self.insert_before(
                loop_start,
                Op::Const(match rop {
                    BOp::Add => acc_dtype.zero_constant(),
                    BOp::Max => acc_dtype.min_constant(),
                    BOp::Mul => acc_dtype.one_constant(),
                    _ => unreachable!(),
                }),
            );

            let acc = self.insert_before(
                loop_start,
                Op::Define { dtype: acc_dtype, scope: Scope::Register, ro: false, len: 1 },
            );

            // Zero the accumulator
            self.insert_before(loop_start, Op::Store { dst: acc, x: acc_init_id, index: const_zero, vlen: 1 });

            // Add Loops for the reduce
            for &dim in &self.reduce_dims(reduce_op_id)[..n_axes] {
                self.insert_before(loop_start, Op::Loop { len: dim });
            }

            // Add reduction operation, load from acc, accumulate, store to acc
            let load_acc = self.insert_before(reduce_op_id, Op::Load { src: acc, index: const_zero, vlen: 1 });
            let bin_acc = self.insert_before(reduce_op_id, Op::Binary { x, y: load_acc, bop: rop });
            self.insert_before(reduce_op_id, Op::Store { dst: acc, x: bin_acc, index: const_zero, vlen: 1 });

            // Close the reduce loop
            for _ in 0..n_axes {
                self.insert_before(reduce_op_id, Op::EndLoop);
            }

            // Replace old reduce op with the acc load op
            self.ops[reduce_op_id].op = Op::Load { src: acc, index: const_zero, vlen: 1 };
        }

        self.verify();
    }

    fn new_op(&mut self, op_iter: &mut OpId, op: Op) -> OpId {
        let op_id = self.insert_after(*op_iter, op);
        *op_iter = op_id;
        op_id
    }

    pub fn unfold_views(&mut self) {
        let mut axes: BTreeMap<u32, OpId> = BTreeMap::default();
        let start = self.head;
        let mut op_id = self.head;
        while !op_id.is_null() {
            match self.ops[op_id].op {
                Op::ConstView(ref x) => {
                    let value = x.0;
                    // With padding, right padding does not affect offset
                    // offset = (a0-lp0)*st0 + a1*st1
                    // Padding condition, negative right padding does not affect it
                    // pc = a0 > lp0-1 && a0 < d0-rp0
                    // pc = pc.cast(dtype)
                    // x = pc * value[offset]
                    let view = x.1.clone();
                    let axes: Vec<OpId> = axes.values().copied().collect();

                    //println!("Unfolding view: {view}");

                    let mut opi = self.prev_op(op_id);
                    let opi = &mut opi;
                    let mut pc = self.new_op(opi, Op::Const(Constant::Bool(true)));
                    let constant_zero = self.new_op(opi, Op::Const(Constant::idx(0)));

                    let mut offset;

                    let mut old_offset: Option<OpId> = None;
                    //println!("View");
                    //for inner in self.0.iter() { println!("{inner:?}") }
                    //println!();
                    for inner in view.0.iter().rev() {
                        //println!("\n{inner:?}");
                        // a = offset / ost % dim
                        let mut ost = 1;
                        offset = constant_zero;
                        let mut ax = inner.len();
                        // TODO check if we can remove the rev and iterate forward
                        for dim in inner.iter().rev() {
                            ax -= 1;
                            //println!("ax={ax} axes={axes:?} dim={dim:?}");
                            let loop_id = if let Some(old_offset) = old_offset {
                                let t_ost = ost;
                                ost *= dim.d as u64;
                                let x = if t_ost == 1 {
                                    old_offset
                                } else {
                                    let ost_c = self.new_op(opi, Op::Const(Constant::idx(t_ost)));
                                    self.new_op(opi, Op::Binary { x: old_offset, y: ost_c, bop: BOp::Div })
                                };
                                if dim.d == 1 {
                                    constant_zero
                                } else {
                                    let dimd_c = self.new_op(opi, Op::Const(Constant::idx(dim.d as u64)));
                                    self.new_op(opi, Op::Binary { x, y: dimd_c, bop: BOp::Mod })
                                }
                            } else if dim.d == 1 {
                                self.new_op(opi, Op::Const(Constant::idx(0u64)))
                            } else {
                                axes[ax]
                            };
                            //println!("loop_id={loop_id} ax={ax} axes={axes:?} dim={dim:?}");
                            //println!("ost: {ost}, a: {a:?}, {dim:?}");
                            // Offset
                            let t = if dim.lp != 0 {
                                let lp = self.new_op(opi, Op::Const(Constant::idx(dim.lp.unsigned_abs() as u64)));
                                if dim.lp > 0 {
                                    self.new_op(opi, Op::Binary { x: loop_id, y: lp, bop: BOp::Sub })
                                } else {
                                    self.new_op(opi, Op::Binary { x: loop_id, y: lp, bop: BOp::Add })
                                }
                            } else {
                                loop_id
                            };

                            if dim.st != 0 {
                                let stride = self.new_op(opi, Op::Const(Constant::idx(dim.st as u64)));
                                //let x = self.new_op(opi, Op::Binary { x: t, y: stride, bop: BOp::Mul });
                                //offset = self.new_op(opi, Op::Binary { x, y: offset, bop: BOp::Add });
                                offset = self.new_op(opi, Op::Mad { x: t, y: stride, z: offset });
                            }

                            // Padding condition
                            if dim.lp > 0 {
                                let lp = self.new_op(opi, Op::Const(Constant::idx((dim.lp - 1) as u64)));
                                let t = self.new_op(opi, Op::Binary { x: loop_id, y: lp, bop: BOp::Cmpgt });
                                pc = self.new_op(opi, Op::Binary { x: t, y: pc, bop: BOp::And });
                            }
                            if dim.rp > 0 {
                                let rp = self.new_op(opi, Op::Const(Constant::idx((dim.d as i64 - dim.rp) as u64)));
                                let t = self.new_op(opi, Op::Binary { x: loop_id, y: rp, bop: BOp::Cmplt });
                                pc = self.new_op(opi, Op::Binary { x: t, y: pc, bop: BOp::And });
                            }
                        }
                        old_offset = Some(offset);
                    }

                    let z = self.new_op(opi, Op::Const(value));

                    let dtype = value.dtype();
                    let pcd = self.new_op(opi, Op::Cast { x: pc, dtype });

                    // Nullify z if padding condition is false (if there is padding at that index)
                    self.ops[op_id].op = Op::Binary { x: pcd, y: z, bop: BOp::Mul };
                    // this is now the new op_id
                }
                Op::LoadView(ref x) => {
                    let dtype = x.0;
                    // With padding, right padding does not affect offset
                    // offset = (a0-lp0)*st0 + a1*st1 + a2*st2 + (a3-lp3)*st3 + ...
                    // Padding condition, negative right padding does not affect it
                    // pc = a0 > lp0-1 && a0 < d0-rp0
                    // pc = pc.cast(dtype)
                    // x = pc * value[offset]
                    let view = x.1.clone();
                    let axes: Vec<OpId> = axes.values().copied().collect();

                    let mut opi = self.prev_op(op_id);
                    let opi = &mut opi;
                    let mut pc = self.new_op(opi, Op::Const(Constant::Bool(true)));
                    let constant_zero = self.new_op(opi, Op::Const(Constant::idx(0)));
                    let mut offset = constant_zero;
                    let mut old_offset: Option<OpId> = None;
                    //println!("axes={axes:?}");
                    //for inner in self.0.iter() { println!("{inner:?}") }
                    //println!();
                    for inner in view.0.iter().rev() {
                        //println!("\n{inner:?}");
                        // a = offset / ost % dim
                        let mut ost = 1;
                        offset = constant_zero;
                        let mut ax = inner.len();
                        // TODO check if we can remove the rev and iterate forward
                        for dim in inner.iter().rev() {
                            ax -= 1;
                            let loop_id = if let Some(old_offset) = old_offset {
                                /*let ost_c = new_op(ops, Op::Const(Constant::U32(ost)));
                                ost *= dim.d as u32;
                                let x = new_op(ops, Op::Binary { x: old_offset, y: ost_c, bop: BOp::Div });
                                let dimd_c = new_op(ops, Op::Const(Constant::U32(dim.d as u32)));
                                new_op(ops, Op::Binary { x, y: dimd_c, bop: BOp::Mod })*/
                                let t_ost = ost;
                                ost *= dim.d as u64;
                                let x = if t_ost == 1 {
                                    old_offset
                                } else {
                                    let ost_c = self.new_op(opi, Op::Const(Constant::idx(t_ost)));
                                    self.new_op(opi, Op::Binary { x: old_offset, y: ost_c, bop: BOp::Div })
                                };
                                if dim.d == 1 {
                                    constant_zero
                                } else {
                                    let dimd_c = self.new_op(opi, Op::Const(Constant::idx(dim.d as u64)));
                                    self.new_op(opi, Op::Binary { x, y: dimd_c, bop: BOp::Mod })
                                }
                            } else if dim.d == 1 {
                                constant_zero
                            } else {
                                axes[ax]
                            };
                            //println!("loop_id={loop_id} ax={ax} axes={axes:?} dim={dim:?}");

                            //println!("ost: {ost}, a: {a:?}, {dim:?}");
                            // Offset
                            let padded_loop_id = if dim.lp != 0 {
                                let lp = self.new_op(opi, Op::Const(Constant::idx(dim.lp.unsigned_abs() as u64)));
                                if dim.lp > 0 {
                                    self.new_op(opi, Op::Binary { x: loop_id, y: lp, bop: BOp::Sub })
                                } else {
                                    self.new_op(opi, Op::Binary { x: loop_id, y: lp, bop: BOp::Add })
                                }
                            } else {
                                loop_id
                            };

                            if dim.st != 0 {
                                let stride = self.new_op(opi, Op::Const(Constant::idx(dim.st as u64)));
                                //let x = self.new_op(opi, Op::Binary { x: padded_loop_id, y: stride, bop: BOp::Mul });
                                //offset = self.new_op(opi, Op::Binary { x, y: offset, bop: BOp::Add });
                                offset = self.new_op(opi, Op::Mad { x: padded_loop_id, y: stride, z: offset });
                            }

                            // Padding condition
                            if dim.lp > 0 {
                                let lp = self.new_op(opi, Op::Const(Constant::idx((dim.lp - 1) as u64)));
                                let t = self.new_op(opi, Op::Binary { x: loop_id, y: lp, bop: BOp::Cmpgt });
                                pc = self.new_op(opi, Op::Binary { x: t, y: pc, bop: BOp::And });
                            }
                            if dim.rp > 0 {
                                let rp = self.new_op(opi, Op::Const(Constant::idx((dim.d as i64 - dim.rp) as u64)));
                                let t = self.new_op(opi, Op::Binary { x: loop_id, y: rp, bop: BOp::Cmplt });
                                pc = self.new_op(opi, Op::Binary { x: t, y: pc, bop: BOp::And });
                            }
                        }
                        old_offset = Some(offset);
                    }

                    let pcu = self.new_op(opi, Op::Cast { x: pc, dtype: IDX_T });
                    let offset = self.new_op(opi, Op::Binary { x: pcu, y: offset, bop: BOp::Mul });

                    let src = self.insert_before(
                        start,
                        Op::Define { dtype, scope: Scope::Global, ro: true, len: view.original_numel() as u64 },
                    );
                    let z = self.new_op(opi, Op::Load { src, index: offset, vlen: 1 });

                    let pcd = self.new_op(opi, Op::Cast { x: pc, dtype });
                    // Nullify z if padding condition is false (if there is padding at that index)
                    self.ops[op_id].op = Op::Binary { x: pcd, y: z, bop: BOp::Mul };
                }
                Op::StoreView { dtype, src } => {
                    let mut st = 1;
                    let mut strides = Vec::new();
                    for (_, &ax_id) in axes.iter().rev() {
                        match self.ops[ax_id].op {
                            Op::Index { len, scope, .. } => {
                                debug_assert_eq!(scope, Scope::Global);
                                strides.push((len, st, ax_id));
                                st *= len;
                            }
                            Op::Loop { len, .. } => {
                                strides.push((len, st, ax_id));
                                st *= len;
                            }
                            _ => {}
                        }
                    }

                    let mut opi = self.prev_op(op_id);
                    let opi = &mut opi;
                    let mut index = self.new_op(opi, Op::Const(Constant::idx(0u64)));
                    let mut len = 1;
                    for (dim, st, ax_id) in strides.into_iter().rev() {
                        let y = self.new_op(opi, Op::Const(Constant::idx(st as u64)));
                        index = self.new_op(opi, Op::Mad { x: ax_id, y, z: index });
                        //index = self.new_op(opi, Op::Binary { x, y: index, bop: BOp::Add });
                        len *= dim;
                    }

                    let dst = self.insert_before(start, Op::Define { dtype, scope: Scope::Global, ro: false, len });
                    self.ops[op_id].op = Op::Store { dst, x: src, index, vlen: 1 };
                }
                Op::Index { axis, .. } => {
                    axes.insert(axis, op_id);
                }
                Op::Loop { .. } => {
                    axes.insert(axes.last_key_value().map_or(0, |x| x.0 + 1), op_id);
                }
                Op::EndLoop => {
                    axes.pop_last();
                }
                _ => {}
            }
            op_id = self.next_op(op_id);
        }

        // Reorder defines of global args so that stores are after loads
        let mut op_id = self.prev_op(start);
        let mut last_load = OpId::NULL;
        while !op_id.is_null() {
            let Op::Define { ro, .. } = self.ops[op_id].op else { unreachable!() };
            if ro {
                if last_load.is_null() {
                    last_load = op_id;
                }
            } else {
                if !last_load.is_null() {
                    self.move_op_after(op_id, last_load);
                }
            }
            op_id = self.prev_op(op_id);
        }

        self.verify();
    }

    pub fn is_preceded_by_reduce(&self, x: OpId) -> bool {
        if self.ops.values().filter(|node| matches!(node.op, Op::Reduce { .. })).count() > 1 {
            return true;
        }
        let mut params = vec![x];
        let mut reduce_params = Vec::new();
        while let Some(param) = params.pop() {
            if let &Op::Reduce { x, .. } = self.at(param) {
                reduce_params.push(x);
                break;
            }
            params.extend(self.ops[param].op.parameters());
        }
        // If there is a load (non constant reduce) or multiple reduces, return true
        while let Some(param) = reduce_params.pop() {
            if matches!(self.at(param), Op::LoadView(_) | Op::Reduce { .. }) {
                return true;
            }
            params.extend(self.ops[param].op.parameters());
        }
        false
    }
}
