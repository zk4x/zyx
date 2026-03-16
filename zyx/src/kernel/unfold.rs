use std::collections::BTreeMap;

use crate::{
    Map, Set,
    dtype::Constant,
    kernel::{BOp, IDX_T, Kernel, MoveOp, Op, OpId, Scope},
    shape::{Dim, UAxis},
};

impl Kernel {
    /// Apply  movement ops on views.
    /// Generates indices on views and unfolds reduce ops.
    pub fn unfold_movement_ops(&mut self) {
        #[derive(Debug, Clone, Copy, PartialEq, Eq)]
        enum Axis {
            Index,
            Loop,
        }
        let mut running_dims: Map<OpId, Vec<(u32, OpId, Dim, Axis)>> = Map::default();
        let mut order = 0;
        let mut op_id = self.head;
        while !op_id.is_null() {
            order += 1;
            match self.ops[op_id].op {
                Op::ConstView(ref x) => {
                    let view = &x.1;
                    let shape = view.shape();
                    let mut rdims = Vec::new();
                    for &dim in &shape {
                        rdims.push((order, op_id, dim, Axis::Index));
                    }
                    running_dims.insert(op_id, rdims);
                }
                Op::LoadView(ref x) => {
                    let view = &x.1;
                    let shape = view.shape();
                    let mut rdims = Vec::new();
                    for &dim in &shape {
                        rdims.push((order, op_id, dim, Axis::Index));
                    }
                    running_dims.insert(op_id, rdims);
                }
                Op::Move { x, ref mop } => {
                    running_dims.insert(op_id, running_dims[&x].clone());
                    let rdims = running_dims.get_mut(&op_id).unwrap();
                    let mop = mop.clone();
                    match mop.as_ref() {
                        MoveOp::Reshape { shape } => {
                            let preceded_by_reduce = self.is_preceded_by_reduce(x);
                            let &(rorder, rid, _, _) = rdims.iter().min_by_key(|x| x.0).unwrap();
                            rdims.clear();
                            for &d in shape {
                                if d == 1 && preceded_by_reduce {
                                    rdims.push((order, op_id, d, Axis::Loop));
                                } else {
                                    rdims.push((rorder, rid, d, Axis::Index));
                                }
                            }
                        }
                        MoveOp::Expand { shape } => {
                            debug_assert_eq!(rdims.len(), shape.len());
                            let preceded_by_reduce = self.is_preceded_by_reduce(x);
                            let mut axis = 0;
                            for (rdim, &len) in rdims.iter_mut().zip(shape) {
                                if rdim.2 != len {
                                    debug_assert_eq!(rdim.2, 1);
                                    if preceded_by_reduce {
                                        match rdim.3 {
                                            Axis::Index => {
                                                self.insert_before(
                                                    rdim.1,
                                                    Op::Index { len: 1, scope: Scope::Global, axis },
                                                );
                                            }
                                            Axis::Loop => {
                                                self.insert_before(rdim.1, Op::Loop { len: 1, axis });
                                                self.insert_before(op_id, Op::EndLoop);
                                            }
                                        }
                                        rdim.0 = order;
                                        rdim.1 = op_id;
                                        rdim.3 = Axis::Loop;
                                    }
                                    rdim.2 = len;
                                }
                                axis += 1;
                            }
                        }
                        MoveOp::Permute { axes, .. } => {
                            debug_assert_eq!(rdims.len(), axes.len());
                            *rdims = crate::shape::permute(rdims, axes);
                        }
                        MoveOp::Pad { padding, .. } => {
                            debug_assert_eq!(rdims.len(), padding.len());
                            for (rdim, &(lp, rp)) in rdims.iter_mut().zip(padding) {
                                rdim.2 = (rdim.2 as i64 + lp as i64 + rp as i64) as Dim;
                            }
                        }
                    }
                    self.recursively_move(x, &mop, &mut Set::default(), 0);
                }
                Op::Reduce { n_axes, x, .. } => {
                    running_dims.insert(op_id, running_dims[&x].clone());
                    let rdims = running_dims.get_mut(&op_id).unwrap();
                    let mut axis = rdims.len() as u32;
                    for _ in 0..n_axes {
                        axis -= 1;
                        let (_, before_id, len, _) = rdims.pop().unwrap();
                        //println!("reduce before_id={before_id}");
                        self.insert_before(before_id, Op::Loop { len, axis });
                    }
                }
                Op::Unary { x, .. } | Op::Cast { x, .. } | Op::StoreView { src: x, .. } => {
                    running_dims.insert(op_id, running_dims[&x].clone());
                }
                Op::Binary { x, y, .. } => {
                    //println!("{x} {y}, {:?} {:?}", running_dims[&x], running_dims[&y]);
                    // If running dims from x and y use indices vs loops, only loops can remain in the running_dims
                    // and indices must be destoyed.
                    let rdims_x = &running_dims[&x];
                    let rdims_y = &running_dims[&y];

                    let mut rdims = Vec::new();
                    debug_assert_eq!(rdims_x.len(), rdims_y.len());
                    for (&rdx, rdy) in rdims_x.iter().zip(rdims_y) {
                        debug_assert_eq!(rdx.2, rdy.2);
                        match (rdx.3, rdy.3) {
                            (Axis::Index, Axis::Index) | (Axis::Loop, Axis::Loop) => {
                                if rdx.0 < rdy.0 {
                                    rdims.push((rdx.0, rdx.1, rdx.2, rdx.3));
                                } else {
                                    rdims.push((rdy.0, rdy.1, rdx.2, rdx.3));
                                }
                            }
                            (Axis::Index, Axis::Loop) => {
                                if rdx.0 < rdy.0 {
                                    rdims.push((rdx.0, rdx.1, rdy.2, Axis::Loop));
                                } else {
                                    rdims.push((rdy.0, rdy.1, rdy.2, Axis::Loop));
                                }
                            }
                            (Axis::Loop, Axis::Index) => {
                                if rdx.0 < rdy.0 {
                                    rdims.push((rdx.0, rdx.1, rdx.2, Axis::Loop));
                                } else {
                                    rdims.push((rdy.0, rdy.1, rdx.2, Axis::Loop));
                                }
                            }
                        }
                    }
                    //println!("rdims={rdims:?}");

                    running_dims.insert(op_id, rdims);
                }
                _ => {}
            }
            op_id = self.next_op(op_id);
        }

        // Add indices or loops as needed
        let rdims = &running_dims[&self.tail];
        let mut axis = 0;
        for &(_, before_id, len, raxis) in rdims {
            //println!("before_id={before_id}, raxis={raxis:?}");
            match raxis {
                Axis::Index => {
                    self.insert_before(self.head, Op::Index { len, scope: Scope::Global, axis });
                }
                Axis::Loop => {
                    self.insert_before(before_id, Op::Loop { len, axis });
                    self.insert_after(self.tail, Op::EndLoop);
                }
            }
            axis += 1;
        }

        // Remove all movement ops as no longer needed
        let mut op_id = self.head;
        while !op_id.is_null() {
            let next_op_id = self.next_op(op_id);
            if let Op::Move { x, .. } = self.ops[op_id].op {
                self.remap(op_id, x);
                self.remove_op(op_id);
            }
            op_id = next_op_id;
        }

        #[cfg(debug_assertions)]
        self.verify();

        self.unfold_reduces();
        self.unfold_views();

        // TODO remove this from here
        self.swap_commutative();
        self.constant_folding();
        self.common_subexpression_elimination();
        self.dead_code_elimination();
        self.swap_commutative();
        self.constant_folding();
        self.common_subexpression_elimination();
        self.dead_code_elimination();
    }

    pub fn is_preceded_by_reduce(&self, x: OpId) -> bool {
        let mut params = vec![x];
        while let Some(param) = params.pop() {
            if let Op::Reduce { .. } = self.at(param) {
                return true;
            } else {
                params.extend(self.ops[param].op.parameters());
            }
        }
        false
    }

    /// TODO this function likely needs to be removed and the stuff needs to be applied more directly
    /// in unfold_movement_ops function
    pub fn recursively_move(&mut self, op_id: OpId, move_op: &MoveOp, visited: &mut Set<OpId>, n_reduce_axes: UAxis) {
        if !visited.insert(op_id) {
            return;
        }
        match &mut self.ops[op_id].op {
            Op::LoadView(x) => match move_op {
                MoveOp::Reshape { shape } => {
                    x.1.reshape(0..x.1.rank() - n_reduce_axes, &shape);
                }
                MoveOp::Expand { shape } => x.1.expand(&shape),
                MoveOp::Permute { axes, .. } => x.1.permute(&axes),
                MoveOp::Pad { padding, .. } => x.1.pad(x.1.rank() - n_reduce_axes, &padding),
            },
            Op::ConstView(x) => match move_op {
                MoveOp::Reshape { shape } => x.1.reshape(0..x.1.rank() - n_reduce_axes, &shape),
                MoveOp::Expand { shape } => x.1.expand(&shape),
                MoveOp::Permute { axes, .. } => x.1.permute(&axes),
                MoveOp::Pad { padding, .. } => x.1.pad(x.1.rank() - n_reduce_axes, &padding),
            },
            &mut Op::Reduce { x, n_axes, .. } => match move_op {
                MoveOp::Reshape { .. } => {}
                MoveOp::Expand { .. } => {}
                MoveOp::Permute { .. } => {
                    self.recursively_move(x, move_op, visited, n_reduce_axes + n_axes);
                }
                MoveOp::Pad { .. } => {
                    self.recursively_move(x, move_op, visited, n_reduce_axes + n_axes);
                }
            },
            &mut Op::Cast { x, .. } | &mut Op::Unary { x, .. } => {
                self.recursively_move(x, move_op, visited, n_reduce_axes);
            }
            &mut Op::Binary { x, y, .. } => {
                self.recursively_move(x, move_op, visited, n_reduce_axes);
                self.recursively_move(y, move_op, visited, n_reduce_axes);
            }
            &mut Op::Move { x, .. } => {
                self.recursively_move(x, move_op, visited, n_reduce_axes);
            }
            _ => {}
        }
    }

    pub fn unfold_reduces(&mut self) {
        let mut dtypes = Map::default();
        let mut loops = Vec::new();
        let mut op_id = self.head;
        while !op_id.is_null() {
            match self.ops[op_id].op {
                Op::ConstView(ref x) => _ = dtypes.insert(op_id, x.0.dtype()),
                Op::LoadView(ref x) => _ = dtypes.insert(op_id, x.0),
                Op::Define { dtype, .. } => _ = dtypes.insert(op_id, dtype),
                Op::Const(constant) => _ = dtypes.insert(op_id, constant.dtype()),
                Op::Load { src, .. } => _ = dtypes.insert(op_id, dtypes[&src]),
                Op::Cast { dtype, .. } => _ = dtypes.insert(op_id, dtype),
                Op::Unary { x, .. } => _ = dtypes.insert(op_id, dtypes[&x]),
                Op::Binary { x, .. } => _ = dtypes.insert(op_id, dtypes[&x]),
                Op::Loop { .. } => {
                    dtypes.insert(op_id, IDX_T);
                    loops.push(op_id);
                }
                Op::EndLoop => _ = loops.pop(),
                Op::Reduce { x, rop, n_axes } => {
                    let dtype = dtypes[&x];
                    dtypes.insert(op_id, dtype);
                    let loop_id = loops[loops.len() - n_axes];
                    let Op::Loop { .. } = self.ops[loop_id].op else { unreachable!() };
                    let index = self.insert_before(loop_id, Op::Const(Constant::idx(0)));
                    let acc_init_const = self.insert_before(loop_id, Op::Const(dtype.init_for_rop(rop)));
                    let acc =
                        self.insert_before(loop_id, Op::Define { dtype, scope: Scope::Register, ro: false, len: 1 });
                    self.insert_before(loop_id, Op::Store { dst: acc, x: acc_init_const, index, vlen: 1 });
                    let acc_load = self.insert_before(op_id, Op::Load { src: acc, index, vlen: 1 });
                    let acc_op = self.insert_before(op_id, Op::Binary { x, y: acc_load, bop: rop });
                    self.insert_before(op_id, Op::Store { dst: acc, x: acc_op, index, vlen: 1 });
                    for _ in 0..n_axes {
                        self.insert_before(op_id, Op::EndLoop);
                        loops.pop();
                    }
                    self.ops[op_id].op = Op::Load { src: acc, index, vlen: 1 };
                }
                Op::Index { .. } => {
                    dtypes.insert(op_id, IDX_T);
                }
                Op::StoreView { .. } | Op::Store { .. } => {}
                Op::Move { .. } => unreachable!("Can't unfold reduces before unfolding movement ops"),
                Op::Mad { x, .. } => _ = dtypes.insert(op_id, dtypes[&x]),
                ref op => todo!("{op:?}"),
            }
            op_id = self.next_op(op_id);
        }

        let mut op_id = self.head;
        while !op_id.is_null() {
            op_id = self.next_op(op_id);
        }

        #[cfg(debug_assertions)]
        self.verify();
    }

    fn new_op(&mut self, op_iter: &mut OpId, op: Op) -> OpId {
        let op_id = self.insert_after(*op_iter, op);
        *op_iter = op_id;
        op_id
    }

    pub fn unfold_views(&mut self) {
        let mut axes = BTreeMap::default();
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
                        let mut ax = inner.len() as u32;
                        // TODO check if we can remove the rev and iterate forward
                        for dim in inner.iter().rev() {
                            ax -= 1;
                            let a = if let Some(old_offset) = old_offset {
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
                                if let Some(&id) = axes.get(&ax) {
                                    id
                                } else {
                                    constant_zero
                                }
                            };
                            ax += 1;
                            //println!("ost: {ost}, a: {a:?}, {dim:?}");
                            // Offset
                            let t = if dim.lp != 0 {
                                let lp = self.new_op(opi, Op::Const(Constant::idx(dim.lp.unsigned_abs() as u64)));
                                if dim.lp > 0 {
                                    self.new_op(opi, Op::Binary { x: a, y: lp, bop: BOp::Sub })
                                } else {
                                    self.new_op(opi, Op::Binary { x: a, y: lp, bop: BOp::Add })
                                }
                            } else {
                                a
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
                                let t = self.new_op(opi, Op::Binary { x: a, y: lp, bop: BOp::Cmpgt });
                                pc = self.new_op(opi, Op::Binary { x: t, y: pc, bop: BOp::And });
                            }
                            if dim.rp > 0 {
                                let rp = self.new_op(opi, Op::Const(Constant::idx((dim.d as i32 - dim.rp) as u64)));
                                let t = self.new_op(opi, Op::Binary { x: a, y: rp, bop: BOp::Cmplt });
                                pc = self.new_op(opi, Op::Binary { x: t, y: pc, bop: BOp::And });
                            }
                        }
                        old_offset = Some(offset);
                    }

                    let z = self.new_op(opi, Op::Const(value));

                    let dtype = value.dtype();
                    let pcd = self.new_op(opi, Op::Cast { x: pc, dtype });

                    // Nullify z if padding condition is false (if there is padding at that index)
                    self.ops[op_id].op = Op::Binary { x: pcd, y: z, bop: BOp::Mul }; // this is now the new op_id
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

                    let mut opi = self.prev_op(op_id);
                    let opi = &mut opi;
                    let mut pc = self.new_op(opi, Op::Const(Constant::Bool(true)));
                    let constant_zero = self.new_op(opi, Op::Const(Constant::idx(0)));
                    let mut offset = constant_zero;
                    let mut old_offset: Option<OpId> = None;
                    //println!("View");
                    //for inner in self.0.iter() { println!("{inner:?}") }
                    //println!();
                    for inner in view.0.iter().rev() {
                        //println!("\n{inner:?}");
                        // a = offset / ost % dim
                        let mut ost = 1;
                        offset = constant_zero;
                        let mut ax = inner.len() as u32;
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
                                //println!("ax={ax}, axes={axes:?}");
                                if let Some(&id) = axes.get(&ax) {
                                    id
                                } else {
                                    constant_zero
                                }
                            };
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
                                let rp = self.new_op(opi, Op::Const(Constant::idx((dim.d as i32 - dim.rp) as u64)));
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
                        Op::Define { dtype, scope: Scope::Global, ro: true, len: view.original_numel() },
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
                Op::Index { axis, .. } | Op::Loop { axis, .. } => {
                    axes.insert(axis, op_id);
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

        #[cfg(debug_assertions)]
        self.verify();
    }
}
