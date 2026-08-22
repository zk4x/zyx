use crate::{
    Map, Set,
    dtype::Constant,
    graph::{ClassId, Graph, GraphId, Node},
    kernel::{BOp, UOp},
    runtime::Runtime,
    shape::{Dim, UAxis},
    slab::SlabId,
    tensor::TensorId,
};
use std::collections::BTreeSet;

impl Runtime {
    pub(crate) fn gradient(&mut self, target: TensorId, sources: Set<TensorId>, graph_id: GraphId) -> Map<TensorId, TensorId> {
        let target_class = self.tensors[target].class_id;
        if target_class.is_null() {
            panic!("gradient on non-graph tensor");
        }
        let source_classes: Set<ClassId> = sources.iter().map(|tid| self.tensors[*tid].class_id).collect();
        for class_id in &source_classes {
            if class_id.is_null() {
                panic!("one of the sources is non-graph tensor");
            }
        }

        let output_set: BTreeSet<ClassId> = [target_class].into();
        let topo = self.graphs[graph_id].build_topo(&output_set, &source_classes);

        let mut grads: Map<ClassId, ClassId> = Map::default();

        // Seed gradient: ones expanded to the target's shape. Never a bare
        // const — a const class has no producer path, so it cannot be realized
        // as a tape output on its own.
        let target_dtype = self.graphs[graph_id].dtype(target_class);
        let one_cid = self.push_const(graph_id, Constant::new(1u8).cast(target_dtype));
        let target_dims = self.graphs[graph_id].shape(target_class);
        let shape_class = match target_dims.len() {
            0 => ClassId::NULL,
            1 => target_dims[0],
            _ => self.push_node(graph_id, Node::Stack { ops: target_dims.into_boxed_slice() }).1,
        };
        let ones = self.push_node(graph_id, Node::Expand { x: one_cid, shape: shape_class }).1;
        grads.insert(target_class, ones);

        for &cid in &topo {
            let Some(&grad) = grads.get(&cid) else {
                continue;
            };

            let nid = match self.graphs[graph_id].classes[cid].nodes.iter().copied().find(|&nid| {
                !matches!(&self.graphs[graph_id].nodes[nid].node, Node::Leaf { .. } | Node::Const { .. } | Node::Kernel { .. })
            }) {
                Some(nid) => nid,
                None => continue,
            };

            match self.graphs[graph_id].nodes[nid].node {
                Node::Unary { x, uop } => match uop {
                    UOp::Neg => {
                        let g = self.push_node(graph_id, Node::Unary { x: grad, uop: UOp::Neg }).1;
                        accum_grad(self, graph_id, &mut grads, x, g);
                    }
                    UOp::Reciprocal => {
                        let z_sq = self.push_binary_node(graph_id, cid, cid, BOp::Mul);
                        let neg_z_sq = self.push_node(graph_id, Node::Unary { x: z_sq, uop: UOp::Neg }).1;
                        let g = self.push_binary_node(graph_id, grad, neg_z_sq, BOp::Mul);
                        accum_grad(self, graph_id, &mut grads, x, g);
                    }
                    UOp::Exp2 => {
                        let ln2_cid = self.push_const(graph_id, Constant::new(std::f64::consts::LN_2));
                        let z_ln2 = self.push_binary_node(graph_id, cid, ln2_cid, BOp::Mul);
                        let g = self.push_binary_node(graph_id, grad, z_ln2, BOp::Mul);
                        accum_grad(self, graph_id, &mut grads, x, g);
                    }
                    UOp::Log2 => {
                        let ln2_cid = self.push_const(graph_id, Constant::new(std::f64::consts::LN_2));
                        let x_ln2 = self.push_binary_node(graph_id, x, ln2_cid, BOp::Mul);
                        let g = self.push_binary_node(graph_id, grad, x_ln2, BOp::Div);
                        accum_grad(self, graph_id, &mut grads, x, g);
                    }
                    UOp::Sqrt => {
                        let two_cid = self.push_const(graph_id, Constant::new(2));
                        let z2 = self.push_binary_node(graph_id, cid, two_cid, BOp::Mul);
                        let g = self.push_binary_node(graph_id, grad, z2, BOp::Div);
                        accum_grad(self, graph_id, &mut grads, x, g);
                    }
                    UOp::Sin => {
                        let cos_x = self.push_node(graph_id, Node::Unary { x, uop: UOp::Cos }).1;
                        let g = self.push_binary_node(graph_id, grad, cos_x, BOp::Mul);
                        accum_grad(self, graph_id, &mut grads, x, g);
                    }
                    UOp::Cos => {
                        let sin_x = self.push_node(graph_id, Node::Unary { x, uop: UOp::Sin }).1;
                        let neg_sin = self.push_node(graph_id, Node::Unary { x: sin_x, uop: UOp::Neg }).1;
                        let g = self.push_binary_node(graph_id, grad, neg_sin, BOp::Mul);
                        accum_grad(self, graph_id, &mut grads, x, g);
                    }
                    UOp::Exp => {
                        let exp_x = self.push_node(graph_id, Node::Unary { x, uop: UOp::Exp }).1;
                        let g = self.push_binary_node(graph_id, grad, exp_x, BOp::Mul);
                        accum_grad(self, graph_id, &mut grads, x, g);
                    }
                    UOp::Ln => {
                        let g = self.push_binary_node(graph_id, grad, x, BOp::Div);
                        accum_grad(self, graph_id, &mut grads, x, g);
                    }
                    UOp::Abs => {
                        let zero = self.push_const(graph_id, Constant::new(0u8));
                        let one = self.push_const(graph_id, Constant::new(1u8));
                        let neg_one = self.push_const(graph_id, Constant::new(-1i8));
                        let is_pos = self.push_binary_node(graph_id, x, zero, BOp::Cmpgt);
                        let is_neg = self.push_binary_node(graph_id, x, zero, BOp::Cmplt);
                        let sign_pos = self.push_binary_node(graph_id, is_pos, one, BOp::Mul);
                        let sign_neg = self.push_binary_node(graph_id, is_neg, neg_one, BOp::Mul);
                        let sign = self.push_binary_node(graph_id, sign_pos, sign_neg, BOp::Add);
                        let g = self.push_binary_node(graph_id, grad, sign, BOp::Mul);
                        accum_grad(self, graph_id, &mut grads, x, g);
                    }
                    UOp::Floor | UOp::Trunc | UOp::BitNot => {}
                },
                Node::Binary { x, y, bop } => match bop {
                    BOp::Add => {
                        accum_grad(self, graph_id, &mut grads, x, grad);
                        accum_grad(self, graph_id, &mut grads, y, grad);
                    }
                    BOp::Sub => {
                        accum_grad(self, graph_id, &mut grads, x, grad);
                        let neg_grad = self.push_node(graph_id, Node::Unary { x: grad, uop: UOp::Neg }).1;
                        accum_grad(self, graph_id, &mut grads, y, neg_grad);
                    }
                    BOp::Mul => {
                        let gx = self.push_binary_node(graph_id, grad, y, BOp::Mul);
                        accum_grad(self, graph_id, &mut grads, x, gx);
                        let gy = self.push_binary_node(graph_id, grad, x, BOp::Mul);
                        accum_grad(self, graph_id, &mut grads, y, gy);
                    }
                    BOp::Div => {
                        let gx = self.push_binary_node(graph_id, grad, y, BOp::Div);
                        accum_grad(self, graph_id, &mut grads, x, gx);
                        let neg_grad = self.push_node(graph_id, Node::Unary { x: grad, uop: UOp::Neg }).1;
                        let x_mul = self.push_binary_node(graph_id, neg_grad, x, BOp::Mul);
                        let y_sq = self.push_binary_node(graph_id, y, y, BOp::Mul);
                        let gy = self.push_binary_node(graph_id, x_mul, y_sq, BOp::Div);
                        accum_grad(self, graph_id, &mut grads, y, gy);
                    }
                    BOp::Pow => {
                        let one = self.push_const(graph_id, Constant::new(1u8));
                        let y_1 = self.push_binary_node(graph_id, y, one, BOp::Sub);
                        let x_pow_ym1 = self.push_binary_node(graph_id, x, y_1, BOp::Pow);
                        let y_mul = self.push_binary_node(graph_id, y, x_pow_ym1, BOp::Mul);
                        let gx = self.push_binary_node(graph_id, grad, y_mul, BOp::Mul);
                        accum_grad(self, graph_id, &mut grads, x, gx);
                        let ln_x = self.push_node(graph_id, Node::Unary { x, uop: UOp::Ln }).1;
                        let z_lnx = self.push_binary_node(graph_id, cid, ln_x, BOp::Mul);
                        let gy = self.push_binary_node(graph_id, grad, z_lnx, BOp::Mul);
                        accum_grad(self, graph_id, &mut grads, y, gy);
                    }
                    BOp::Mod => {
                        accum_grad(self, graph_id, &mut grads, x, grad);
                        let x_div_y = self.push_binary_node(graph_id, x, y, BOp::Div);
                        let floored = self.push_node(graph_id, Node::Unary { x: x_div_y, uop: UOp::Floor }).1;
                        let neg_floor = self.push_node(graph_id, Node::Unary { x: floored, uop: UOp::Neg }).1;
                        let gy = self.push_binary_node(graph_id, neg_floor, grad, BOp::Mul);
                        accum_grad(self, graph_id, &mut grads, y, gy);
                    }
                    BOp::Max => {
                        let x_gt_y = self.push_binary_node(graph_id, x, y, BOp::Cmpgt);
                        let x_lt_y = self.push_binary_node(graph_id, x, y, BOp::Cmplt);
                        let gx = self.push_binary_node(graph_id, grad, x_gt_y, BOp::Mul);
                        accum_grad(self, graph_id, &mut grads, x, gx);
                        let gy = self.push_binary_node(graph_id, grad, x_lt_y, BOp::Mul);
                        accum_grad(self, graph_id, &mut grads, y, gy);
                    }
                    BOp::Cmplt
                    | BOp::Cmpgt
                    | BOp::Cmpge
                    | BOp::Eq
                    | BOp::NotEq
                    | BOp::Or
                    | BOp::And
                    | BOp::BitXor
                    | BOp::BitOr
                    | BOp::BitAnd
                    | BOp::BitShiftLeft
                    | BOp::BitShiftRight => {}
                },
                Node::Cast { x, .. } => {
                    let g = self.push_node(graph_id, Node::Cast { x: grad, dtype: self.graphs[graph_id].dtype(x) }).1;
                    accum_grad(self, graph_id, &mut grads, x, g);
                }
                Node::Reshape { .. } => {
                    /*let x_shape = self.graphs[graph_id].classes[x].shape;
                    let g = self
                        .push_node(
                            graph_id,
                            Node::Reshape { x: grad, shape: x_shape },
                            x_shape,
                            self.graphs[graph_id].classes[grad].dtype,
                        )
                        .1;
                    accum_grad(self, graph_id, &mut grads, x, g);*/
                    todo!()
                }
                Node::Expand { .. } => {
                    todo!()
                    /*let x_shape = self.graphs[graph_id].classes[x].shape;
                    let out_shape = self.shapes[self.graphs[graph_id].classes[cid].shape].clone();
                    let in_shape = self.shapes[x_shape].clone();
                    // Right-align the input against the expanded output per broadcast
                    // semantics. The input is broadcast to the output by (a) leading
                    // `pad` dims that the input did not have at all (implicitly size 1)
                    // and (b) trailing-aligned dims where the input is a singleton (1)
                    // broadcast to a larger output extent. The gradient of a broadcast
                    // must be summed over *all* of these axes to drop back to the
                    // input shape.
                    let pad = out_shape.len() - in_shape.len();
                    let sum_axes: Vec<UAxis> = (0..pad)
                        .chain(in_shape.iter().enumerate().filter_map(|(i, &xd)| {
                            if xd == 1 && out_shape[pad + i] > 1 {
                                Some((pad + i) as UAxis)
                            } else {
                                None
                            }
                        }))
                        .collect();
                    if sum_axes.is_empty() {
                        accum_grad(self, graph_id, &mut grads, x, grad);
                    } else {
                        let grad_shape = self.shapes[self.graphs[graph_id].classes[grad].shape].to_vec();
                        let reduce_shape_id = self.push_shape(crate::shape::reduce(&grad_shape, &sum_axes));
                        let reduced = self
                            .push_node(
                                graph_id,
                                Node::Reduce { x: grad, rop: BOp::Add, axes: sum_axes.into_boxed_slice() },
                                reduce_shape_id,
                                self.graphs[graph_id].classes[grad].dtype,
                            )
                            .1;
                        // The graph reduce drops the reduced dims; restore the
                        // original shape (keepdim) with an explicit reshape.
                        let reduced = if reduce_shape_id == x_shape {
                            reduced
                        } else {
                            self.push_node(
                                graph_id,
                                Node::Reshape { x: reduced, shape: x_shape },
                                x_shape,
                                self.graphs[graph_id].classes[grad].dtype,
                            )
                            .1
                        };
                        accum_grad(self, graph_id, &mut grads, x, reduced);
                    }*/
                }
                Node::Permute { x, ref axes } => {
                    let mut inv_axes: Vec<UAxis> = vec![0; axes.len()];
                    for (i, &a) in axes.iter().enumerate() {
                        inv_axes[a] = i as UAxis;
                    }
                    let g = self.push_node(graph_id, Node::Permute { x: grad, axes: inv_axes.into_boxed_slice() }).1;
                    accum_grad(self, graph_id, &mut grads, x, g);
                }
                Node::Pad { .. } => {
                    todo!()
                }
                Node::Narrow { .. } => {
                    todo!()
                }
                Node::Flip { x, ref axes } => {
                    // Flip is its own inverse: the gradient back-propagates by
                    // flipping along the same axes.
                    let g = self.push_node(graph_id, Node::Flip { x: grad, axes: axes.clone() }).1;
                    accum_grad(self, graph_id, &mut grads, x, g);
                }
                Node::Reduce { x: _, rop: bop, axes: _ } => {
                    //let axes = axes.clone();
                    match bop {
                        BOp::Add => {
                            todo!();
                            /*let x_shape_id = self.graphs[graph_id].classes[x].shape;
                            let x_shape_vec: Vec<Dim> = self.shapes[x_shape_id].clone();
                            let mut grad_shape_vec: Vec<Dim> = self.shapes[self.graphs[graph_id].classes[cid].shape].clone();
                            for &axis in axes.iter() {
                                grad_shape_vec.insert(axis, 1);
                            }
                            if axes.len() == x_shape_vec.len() {
                                grad_shape_vec.remove(0);
                            }
                            let gs = self.push_shape(grad_shape_vec);
                            let grad_r = self
                                .push_node(
                                    graph_id,
                                    Node::Reshape { x: grad, shape: gs },
                                    gs,
                                    self.graphs[graph_id].classes[grad].dtype,
                                )
                                .1;
                            let g = self
                                .push_node(
                                    graph_id,
                                    Node::Expand { x: grad_r, shape: x_shape_id },
                                    x_shape_id,
                                    self.graphs[graph_id].classes[grad].dtype,
                                )
                                .1;
                            accum_grad(self, graph_id, &mut grads, x, g);*/
                        }
                        BOp::Max => {
                            todo!();
                            /*let x_shape_id = self.graphs[graph_id].classes[x].shape;
                            let x_shape_vec: Vec<Dim> = self.shapes[x_shape_id].clone();

                            let mut z_shape_vec: Vec<Dim> = self.shapes[self.graphs[graph_id].classes[cid].shape].clone();
                            for &axis in axes.iter() {
                                z_shape_vec.insert(axis, 1);
                            }
                            if axes.len() == x_shape_vec.len() {
                                z_shape_vec.remove(0);
                            }
                            let zs = self.push_shape(z_shape_vec);
                            let z_reshaped = self
                                .push_node(
                                    graph_id,
                                    Node::Reshape { x: cid, shape: zs },
                                    zs,
                                    self.graphs[graph_id].classes[cid].dtype,
                                )
                                .1;
                            let z_broadcasted = self
                                .push_node(
                                    graph_id,
                                    Node::Expand { x: z_reshaped, shape: x_shape_id },
                                    x_shape_id,
                                    self.graphs[graph_id].classes[cid].dtype,
                                )
                                .1;
                            let cmp = self.push_binary_node(graph_id, x, z_broadcasted, BOp::Cmplt);
                            let cmp_f = self
                                .push_node(
                                    graph_id,
                                    Node::Cast { x: cmp, dtype: self.graphs[graph_id].classes[x].dtype },
                                    self.graphs[graph_id].classes[cmp].shape,
                                    self.graphs[graph_id].classes[x].dtype,
                                )
                                .1;
                            let one = self
                                .push_node(
                                    graph_id,
                                    Node::Const { cons_id: 0, value: Constant::new(1u8).cast(self.graphs[graph_id].classes[x].dtype) },
                                    scalar_shape,
                                    self.graphs[graph_id].classes[x].dtype,
                                )
                                .1;
                            let one_e = self
                                .push_node(
                                    graph_id,
                                    Node::Expand { x: one, shape: x_shape_id },
                                    x_shape_id,
                                    self.graphs[graph_id].classes[x].dtype,
                                )
                                .1;
                            let mask = self.push_binary_node(graph_id, one_e, cmp_f, BOp::Sub);

                            let mut grad_shape_vec: Vec<Dim> = self.shapes[self.graphs[graph_id].classes[grad].shape].clone();
                            for &axis in axes.iter() {
                                grad_shape_vec.insert(axis, 1);
                            }
                            if axes.len() == x_shape_vec.len() {
                                grad_shape_vec.remove(0);
                            }
                            let gs = self.push_shape(grad_shape_vec);
                            let grad_r = self
                                .push_node(
                                    graph_id,
                                    Node::Reshape { x: grad, shape: gs },
                                    gs,
                                    self.graphs[graph_id].classes[grad].dtype,
                                )
                                .1;
                            let grad_e = self
                                .push_node(
                                    graph_id,
                                    Node::Expand { x: grad_r, shape: x_shape_id },
                                    x_shape_id,
                                    self.graphs[graph_id].classes[grad].dtype,
                                )
                                .1;

                            let grad_x = self.push_binary_node(graph_id, mask, grad_e, BOp::Mul);
                            accum_grad(self, graph_id, &mut grads, x, grad_x);*/
                        }
                        _ => {}
                    }
                }
                Node::ToDevice { x, .. } => {
                    accum_grad(self, graph_id, &mut grads, x, grad);
                }
                Node::Contiguous { x } => {
                    accum_grad(self, graph_id, &mut grads, x, grad);
                }
                Node::Assign { dst: _, src, .. } => {
                    accum_grad(self, graph_id, &mut grads, src, grad);
                }
                Node::After { .. } => {
                    todo!()
                }
                Node::Leaf { .. } | Node::Const { .. } | Node::Stack { .. } | Node::Kernel { .. } => {}
            }
        }

        grads.retain(|k, _| source_classes.contains(k));

        let mut res = Map::default();
        for tid in sources {
            let grad_tid = match grads.get(&self.tensors[tid].class_id) {
                Some(&gcid) => self.new_graph_tensor(graph_id, gcid),
                None => {
                    let shape: Vec<Dim> = self.shape(tid).into();
                    let dtype = self.dtype(tid);
                    let zero_cid = self.push_const(graph_id, Constant::new(0u8).cast(dtype));
                    let ops: Box<[ClassId]> =
                        shape.iter().map(|&d| self.push_const(graph_id, Constant::idx(d))).collect();
                    let shape_cid = if ops.len() == 1 {
                        ops[0]
                    } else {
                        self.push_node(graph_id, Node::Stack { ops }).1
                    };
                    let (_, cid) = self.push_node(graph_id, Node::Expand { x: zero_cid, shape: shape_cid });
                    self.new_graph_tensor(graph_id, cid)
                }
            };
            res.insert(tid, grad_tid);
        }
        res
    }
}

impl Graph {
    pub fn build_topo(&self, outputs: &BTreeSet<ClassId>, sources: &Set<ClassId>) -> Vec<ClassId> {
        let mut stack: Vec<ClassId> = outputs.iter().copied().collect();
        let mut rcs: Map<ClassId, u32> = Map::default();
        while let Some(cid) = stack.pop() {
            rcs.entry(cid).and_modify(|rc| *rc += 1).or_insert_with(|| {
                for nid in &self.classes[cid].nodes {
                    let node = &self.nodes[*nid].node;
                    if matches!(
                        node,
                        Node::Binary {
                            bop: BOp::Cmpgt
                                | BOp::Cmplt
                                | BOp::Eq
                                | BOp::NotEq
                                | BOp::Or
                                | BOp::And
                                | BOp::BitAnd
                                | BOp::BitOr
                                | BOp::BitXor
                                | BOp::BitShiftLeft
                                | BOp::BitShiftRight,
                            ..
                        }
                    ) {
                        continue;
                    }
                    for p in node.class_params() {
                        if !stack.contains(&p) {
                            stack.push(p);
                        }
                    }
                }
                1
            });
        }

        let mut order = Vec::new();
        let mut internal_rcs: Map<ClassId, u32> = Map::default();
        let mut stack: Vec<ClassId> = outputs.iter().copied().collect();
        while let Some(cid) = stack.pop() {
            if let Some(&rc) = rcs.get(&cid)
                && rc == *internal_rcs.entry(cid).and_modify(|c| *c += 1).or_insert(1)
            {
                order.push(cid);
                for nid in &self.classes[cid].nodes {
                    for p in self.nodes[*nid].node.class_params() {
                        if !stack.contains(&p) {
                            stack.push(p);
                        }
                    }
                }
            }
        }

        let mut topo = Vec::new();
        let mut req_grad = sources.clone();
        let mut visited: Set<ClassId> = Set::default();
        for cid in order.into_iter().rev() {
            for nid in &self.classes[cid].nodes {
                for p in self.nodes[*nid].node.class_params() {
                    if req_grad.contains(&p) && visited.insert(cid) {
                        req_grad.insert(cid);
                        topo.push(cid);
                        break;
                    }
                }
                if visited.contains(&cid) {
                    break;
                }
            }
        }
        topo.reverse();
        topo
    }
}

fn accum_grad(rt: &mut Runtime, graph_id: GraphId, grads: &mut Map<ClassId, ClassId>, nid: ClassId, grad: ClassId) {
    match grads.entry(nid) {
        std::collections::hash_map::Entry::Vacant(e) => {
            e.insert(grad);
        }
        std::collections::hash_map::Entry::Occupied(mut e) => {
            let sum = rt.push_binary_node(graph_id, *e.get(), grad, BOp::Add);
            e.insert(sum);
        }
    }
}
