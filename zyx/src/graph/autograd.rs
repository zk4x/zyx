use crate::{
    Map, Set,
    dtype::Constant,
    graph::{ClassId, Graph, Node},
    kernel::{BOp, UOp},
    runtime::{Runtime, ShapeId, TensorData, TensorState},
    shape::{Dim, UAxis},
    slab::Slab,
    tensor::TensorId,
};
use std::collections::BTreeSet;

impl Runtime {
    pub(crate) fn gradient(&mut self, target: TensorId, sources: Set<TensorId>) -> Map<TensorId, TensorId> {
        let target_class = match self.tensors[target].state {
            TensorState::Graph { class_id, .. } => class_id,
            _ => unreachable!("gradient on non-graph tensor"),
        };
        let source_classes: Set<ClassId> = sources
            .iter()
            .filter_map(|tid| match self.tensors[*tid].state {
                TensorState::Graph { class_id, .. } => Some(class_id),
                _ => None,
            })
            .collect();
        let scalar_shape = self.push_shape(vec![1 as Dim]);
        let class_grads = self.graph.as_mut().unwrap().gradient(target_class, &source_classes, &self.shapes, scalar_shape);
        let mut res = Map::default();
        for tid in sources {
            let class_id = match self.tensors[tid].state {
                TensorState::Graph { class_id, .. } => class_id,
                _ => continue,
            };
            let grad_id = match class_grads.get(&class_id) {
                Some(&gcid) => {
                    let shape_id = self.graph.as_ref().unwrap().classes[gcid].shape;
                    let dtype = self.graph.as_ref().unwrap().classes[gcid].dtype;
                    let nid = self.graph.as_ref().unwrap().classes[gcid].nodes[0];
                    self.tensors.push(TensorData { shape_id, dtype, state: TensorState::Graph { node_id: nid, class_id: gcid, rc: 1 } })
                }
                None => {
                    let shape = self.shape(tid).into();
                    let dtype = self.dtype(tid);
                    self.new_full(shape, dtype.zero_constant())
                }
            };
            res.insert(tid, grad_id);
        }
        res
    }
}

impl Graph {
    pub(crate) fn gradient(
        &mut self,
        target: ClassId,
        _sources: &Set<ClassId>,
        shapes: &Slab<ShapeId, Vec<Dim>>,
        scalar_shape: ShapeId,
    ) -> Map<ClassId, ClassId> {
        let output_set: BTreeSet<ClassId> = [target].into();
        let order = self.topo_sort_classes(&output_set);

        let mut grads: Map<ClassId, ClassId> = Map::default();

        // Initial gradient: ones in the target shape
        let one_cid = self
            .push(Node::Const(Constant::new(1u8).cast(self.classes[target].dtype)), scalar_shape, self.classes[target].dtype)
            .1;
        let ones = self
            .push(
                Node::Expand { x: one_cid, shape: self.classes[target].shape },
                self.classes[target].shape,
                self.classes[target].dtype,
            )
            .1;
        grads.insert(target, ones);

        for &cid in order.iter().rev() {
            let Some(&grad) = grads.get(&cid) else {
                continue;
            };

            let nid = match self.classes[cid]
                .nodes
                .iter()
                .copied()
                .find(|&nid| !matches!(&self.nodes[nid].node, Node::Leaf { .. } | Node::Const(_) | Node::Kernel { .. }))
            {
                Some(nid) => nid,
                None => continue,
            };

            match self.nodes[nid].node {
                Node::Unary { x, uop } => match uop {
                    UOp::Neg => {
                        let g = self
                            .push(Node::Unary { x: grad, uop: UOp::Neg }, self.classes[grad].shape, self.classes[grad].dtype)
                            .1;
                        accum_grad(self, &mut grads, x, g);
                    }
                    UOp::Reciprocal => {
                        let z_sq = push_binary(self, cid, cid, BOp::Mul);
                        let neg_z_sq = self
                            .push(Node::Unary { x: z_sq, uop: UOp::Neg }, self.classes[z_sq].shape, self.classes[z_sq].dtype)
                            .1;
                        let g = push_binary(self, grad, neg_z_sq, BOp::Mul);
                        accum_grad(self, &mut grads, x, g);
                    }
                    UOp::Exp2 => {
                        let ln2 = Constant::new(std::f64::consts::LN_2).cast(self.classes[x].dtype);
                        let ln2_cid = self.push(Node::Const(ln2), scalar_shape, self.classes[x].dtype).1;
                        let ln2_e = self
                            .push(
                                Node::Expand { x: ln2_cid, shape: self.classes[cid].shape },
                                self.classes[cid].shape,
                                self.classes[cid].dtype,
                            )
                            .1;
                        let z_ln2 = push_binary(self, cid, ln2_e, BOp::Mul);
                        let g = push_binary(self, grad, z_ln2, BOp::Mul);
                        accum_grad(self, &mut grads, x, g);
                    }
                    UOp::Log2 => {
                        let ln2 = Constant::new(std::f64::consts::LN_2).cast(self.classes[x].dtype);
                        let ln2_cid = self.push(Node::Const(ln2), scalar_shape, self.classes[x].dtype).1;
                        let ln2_e = self
                            .push(
                                Node::Expand { x: ln2_cid, shape: self.classes[x].shape },
                                self.classes[x].shape,
                                self.classes[x].dtype,
                            )
                            .1;
                        let x_ln2 = push_binary(self, x, ln2_e, BOp::Mul);
                        let g = push_binary(self, grad, x_ln2, BOp::Div);
                        accum_grad(self, &mut grads, x, g);
                    }
                    UOp::Sqrt => {
                        let two = Constant::new(2u8).cast(self.classes[cid].dtype);
                        let two_cid = self.push(Node::Const(two), scalar_shape, self.classes[cid].dtype).1;
                        let two_e = self
                            .push(
                                Node::Expand { x: two_cid, shape: self.classes[cid].shape },
                                self.classes[cid].shape,
                                self.classes[cid].dtype,
                            )
                            .1;
                        let z2 = push_binary(self, cid, two_e, BOp::Mul);
                        let g = push_binary(self, grad, z2, BOp::Div);
                        accum_grad(self, &mut grads, x, g);
                    }
                    UOp::Sin => {
                        let cos_x =
                            self.push(Node::Unary { x: x, uop: UOp::Cos }, self.classes[x].shape, self.classes[x].dtype).1;
                        let g = push_binary(self, grad, cos_x, BOp::Mul);
                        accum_grad(self, &mut grads, x, g);
                    }
                    UOp::Cos => {
                        let sin_x =
                            self.push(Node::Unary { x: x, uop: UOp::Sin }, self.classes[x].shape, self.classes[x].dtype).1;
                        let neg_sin = self
                            .push(Node::Unary { x: sin_x, uop: UOp::Neg }, self.classes[sin_x].shape, self.classes[sin_x].dtype)
                            .1;
                        let g = push_binary(self, grad, neg_sin, BOp::Mul);
                        accum_grad(self, &mut grads, x, g);
                    }
                    UOp::Exp => {
                        let exp_x =
                            self.push(Node::Unary { x: x, uop: UOp::Exp }, self.classes[x].shape, self.classes[x].dtype).1;
                        let g = push_binary(self, grad, exp_x, BOp::Mul);
                        accum_grad(self, &mut grads, x, g);
                    }
                    UOp::Ln => {
                        let g = push_binary(self, grad, x, BOp::Div);
                        accum_grad(self, &mut grads, x, g);
                    }
                    UOp::Abs => {
                        let dtype = self.classes[x].dtype;
                        let zero = self.push(Node::Const(Constant::new(0u8).cast(dtype)), scalar_shape, dtype).1;
                        let one = self.push(Node::Const(Constant::new(1u8).cast(dtype)), scalar_shape, dtype).1;
                        let neg_one = self.push(Node::Const(Constant::new(-1i8).cast(dtype)), scalar_shape, dtype).1;
                        let zero_e =
                            self.push(Node::Expand { x: zero, shape: self.classes[x].shape }, self.classes[x].shape, dtype).1;
                        let one_e =
                            self.push(Node::Expand { x: one, shape: self.classes[x].shape }, self.classes[x].shape, dtype).1;
                        let neg_one_e =
                            self.push(Node::Expand { x: neg_one, shape: self.classes[x].shape }, self.classes[x].shape, dtype).1;
                        let is_pos = push_binary(self, x, zero_e, BOp::Cmpgt);
                        let is_neg = push_binary(self, x, zero_e, BOp::Cmplt);
                        let sign_pos = push_binary(self, is_pos, one_e, BOp::Mul);
                        let sign_neg = push_binary(self, is_neg, neg_one_e, BOp::Mul);
                        let sign = push_binary(self, sign_pos, sign_neg, BOp::Add);
                        let g = push_binary(self, grad, sign, BOp::Mul);
                        accum_grad(self, &mut grads, x, g);
                    }
                    UOp::Floor | UOp::Trunc | UOp::BitNot => {}
                },
                Node::Binary { x, y, bop } => match bop {
                    BOp::Add => {
                        accum_grad(self, &mut grads, x, grad);
                        accum_grad(self, &mut grads, y, grad);
                    }
                    BOp::Sub => {
                        accum_grad(self, &mut grads, x, grad);
                        let neg_grad = self
                            .push(Node::Unary { x: grad, uop: UOp::Neg }, self.classes[grad].shape, self.classes[grad].dtype)
                            .1;
                        accum_grad(self, &mut grads, y, neg_grad);
                    }
                    BOp::Mul => {
                        let gx = push_binary(self, grad, y, BOp::Mul);
                        accum_grad(self, &mut grads, x, gx);
                        let gy = push_binary(self, grad, x, BOp::Mul);
                        accum_grad(self, &mut grads, y, gy);
                    }
                    BOp::Div => {
                        let gx = push_binary(self, grad, y, BOp::Div);
                        accum_grad(self, &mut grads, x, gx);
                        let neg_grad = self
                            .push(Node::Unary { x: grad, uop: UOp::Neg }, self.classes[grad].shape, self.classes[grad].dtype)
                            .1;
                        let x_mul = push_binary(self, neg_grad, x, BOp::Mul);
                        let y_sq = push_binary(self, y, y, BOp::Mul);
                        let gy = push_binary(self, x_mul, y_sq, BOp::Div);
                        accum_grad(self, &mut grads, y, gy);
                    }
                    BOp::Pow => {
                        let dtype = self.classes[x].dtype;
                        let one = self.push(Node::Const(Constant::new(1u8).cast(dtype)), scalar_shape, dtype).1;
                        let one_e =
                            self.push(Node::Expand { x: one, shape: self.classes[y].shape }, self.classes[y].shape, dtype).1;
                        let y_1 = push_binary(self, y, one_e, BOp::Sub);
                        let x_pow_ym1 = push_binary(self, x, y_1, BOp::Pow);
                        let y_mul = push_binary(self, y, x_pow_ym1, BOp::Mul);
                        let gx = push_binary(self, grad, y_mul, BOp::Mul);
                        accum_grad(self, &mut grads, x, gx);
                        let ln_x = self.push(Node::Unary { x: x, uop: UOp::Ln }, self.classes[x].shape, self.classes[x].dtype).1;
                        let z_lnx = push_binary(self, cid, ln_x, BOp::Mul);
                        let gy = push_binary(self, grad, z_lnx, BOp::Mul);
                        accum_grad(self, &mut grads, y, gy);
                    }
                    BOp::Mod => {
                        accum_grad(self, &mut grads, x, grad);
                        let x_div_y = push_binary(self, x, y, BOp::Div);
                        let floored = self
                            .push(
                                Node::Unary { x: x_div_y, uop: UOp::Floor },
                                self.classes[x_div_y].shape,
                                self.classes[x_div_y].dtype,
                            )
                            .1;
                        let neg_floor = self
                            .push(
                                Node::Unary { x: floored, uop: UOp::Neg },
                                self.classes[floored].shape,
                                self.classes[floored].dtype,
                            )
                            .1;
                        let gy = push_binary(self, neg_floor, grad, BOp::Mul);
                        accum_grad(self, &mut grads, y, gy);
                    }
                    BOp::Max => {
                        let dtype = self.classes[x].dtype;
                        let x_gt_y = push_binary(self, x, y, BOp::Cmpgt);
                        let x_lt_y = push_binary(self, x, y, BOp::Cmplt);
                        let x_gt_f = self.push(Node::Cast { x: x_gt_y, dtype }, self.classes[x_gt_y].shape, dtype).1;
                        let x_lt_f = self.push(Node::Cast { x: x_lt_y, dtype }, self.classes[x_lt_y].shape, dtype).1;
                        let gx = push_binary(self, grad, x_gt_f, BOp::Mul);
                        accum_grad(self, &mut grads, x, gx);
                        let gy = push_binary(self, grad, x_lt_f, BOp::Mul);
                        accum_grad(self, &mut grads, y, gy);
                    }
                    BOp::Cmplt
                    | BOp::Cmpgt
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
                    let g = self
                        .push(
                            Node::Cast { x: grad, dtype: self.classes[x].dtype },
                            self.classes[grad].shape,
                            self.classes[x].dtype,
                        )
                        .1;
                    accum_grad(self, &mut grads, x, g);
                }
                Node::Reshape { x, shape } => {
                    let g = self.push(Node::Reshape { x: grad, shape: shape }, shape, self.classes[grad].dtype).1;
                    accum_grad(self, &mut grads, x, g);
                }
                Node::Expand { x, .. } => {
                    let x_shape = self.classes[x].shape;
                    let sum_axes: Vec<UAxis> = shapes[self.classes[cid].shape]
                        .iter()
                        .zip(shapes[x_shape].iter())
                        .enumerate()
                        .filter_map(|(i, (&od, &xd))| if od != xd { Some(i as UAxis) } else { None })
                        .collect();
                    if sum_axes.is_empty() {
                        accum_grad(self, &mut grads, x, grad);
                    } else {
                        let reduced = self
                            .push(
                                Node::Reduce { x: grad, bop: BOp::Add, axes: sum_axes.into_boxed_slice() },
                                x_shape,
                                self.classes[grad].dtype,
                            )
                            .1;
                        accum_grad(self, &mut grads, x, reduced);
                    }
                }
                Node::Permute { x, ref axes } => {
                    let mut inv_axes: Vec<UAxis> = vec![0; axes.len()];
                    for (i, &a) in axes.iter().enumerate() {
                        inv_axes[a as usize] = i as UAxis;
                    }
                    let g = self
                        .push(
                            Node::Permute { x: grad, axes: inv_axes.into_boxed_slice() },
                            self.classes[x].shape,
                            self.classes[grad].dtype,
                        )
                        .1;
                    accum_grad(self, &mut grads, x, g);
                }
                Node::PadZeros { x, .. } => {
                    accum_grad(self, &mut grads, x, grad);
                }
                Node::Reduce { x, bop, .. } => match bop {
                    BOp::Add => {
                        let x_shape = self.classes[x].shape;
                        let g = self.push(Node::Expand { x: grad, shape: x_shape }, x_shape, self.classes[grad].dtype).1;
                        accum_grad(self, &mut grads, x, g);
                    }
                    BOp::Max => {
                        let x_shape = self.classes[x].shape;
                        let z_broadcasted =
                            self.push(Node::Expand { x: cid, shape: x_shape }, x_shape, self.classes[cid].dtype).1;
                        let cmp = push_binary(self, x, z_broadcasted, BOp::Cmplt);
                        let cmp_f = self
                            .push(
                                Node::Cast { x: cmp, dtype: self.classes[x].dtype },
                                self.classes[cmp].shape,
                                self.classes[x].dtype,
                            )
                            .1;
                        let one = self
                            .push(
                                Node::Const(Constant::new(1u8).cast(self.classes[x].dtype)),
                                scalar_shape,
                                self.classes[x].dtype,
                            )
                            .1;
                        let one_e = self.push(Node::Expand { x: one, shape: x_shape }, x_shape, self.classes[x].dtype).1;
                        let mask = push_binary(self, one_e, cmp_f, BOp::Sub);
                        let grad_e = self.push(Node::Expand { x: grad, shape: x_shape }, x_shape, self.classes[grad].dtype).1;
                        let grad_x = push_binary(self, mask, grad_e, BOp::Mul);
                        accum_grad(self, &mut grads, x, grad_x);
                    }
                    _ => {}
                },
                Node::ToDevice { x, .. } => {
                    accum_grad(self, &mut grads, x, grad);
                }
                Node::Leaf { .. } | Node::Const(_) | Node::Kernel { .. } => {}
            }
        }

        grads
    }
}

fn push_binary(g: &mut Graph, x: ClassId, y: ClassId, bop: BOp) -> ClassId {
    g.push(Node::Binary { x, y, bop }, g.classes[x].shape, g.classes[x].dtype).1
}

fn accum_grad(g: &mut Graph, grads: &mut Map<ClassId, ClassId>, nid: ClassId, grad: ClassId) {
    match grads.entry(nid) {
        std::collections::hash_map::Entry::Vacant(e) => {
            e.insert(grad);
        }
        std::collections::hash_map::Entry::Occupied(mut e) => {
            let sum = push_binary(g, *e.get(), grad, BOp::Add);
            e.insert(sum);
        }
    }
}
