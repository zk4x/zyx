use crate::{
    Map, Set,
    dtype::Constant,
    graph::{ClassId, Graph, GraphId, Node},
    kernel::{BOp, UOp},
    runtime::{Runtime, TensorData, TensorState},
    shape::{Dim, UAxis},
    tensor::TensorId,
};
use std::collections::BTreeSet;

impl Runtime {
    pub(crate) fn gradient(&mut self, target: TensorId, sources: Set<TensorId>, graph_id: GraphId) -> Map<TensorId, TensorId> {
        let target_class = match self.tensors[target].state {
            TensorState::Graph { class_id, .. } => class_id,
            TensorState::Eager { .. } => panic!("gradient on non-graph tensor"),
        };
        let source_classes: Set<ClassId> = sources
            .iter()
            .filter_map(|tid| match self.tensors[*tid].state {
                TensorState::Graph { class_id, .. } => Some(class_id),
                _ => None,
            })
            .collect();
        let scalar_shape = self.push_shape(vec![1 as Dim]);

        let graph = &mut self.graphs[graph_id];
        let shapes = &mut self.shapes;

        let output_set: BTreeSet<ClassId> = [target_class].into();
        let topo = graph.build_topo(&output_set, &source_classes);

        let mut grads: Map<ClassId, ClassId> = Map::default();

        let one_cid = graph
            .push(
                Node::Const(Constant::new(1u8).cast(graph.classes[target_class].dtype)),
                scalar_shape,
                graph.classes[target_class].dtype,
            )
            .1;
        let ones = graph
            .push(
                Node::Expand { x: one_cid, shape: graph.classes[target_class].shape },
                graph.classes[target_class].shape,
                graph.classes[target_class].dtype,
            )
            .1;
        grads.insert(target_class, ones);

        for &cid in &topo {
            let Some(&grad) = grads.get(&cid) else {
                continue;
            };

            let nid = match graph.classes[cid]
                .nodes
                .iter()
                .copied()
                .find(|&nid| !matches!(&graph.nodes[nid].node, Node::Leaf { .. } | Node::Const(_) | Node::Kernel { .. }))
            {
                Some(nid) => nid,
                None => continue,
            };

            match graph.nodes[nid].node {
                Node::Unary { x, uop } => match uop {
                    UOp::Neg => {
                        let g = graph
                            .push(Node::Unary { x: grad, uop: UOp::Neg }, graph.classes[grad].shape, graph.classes[grad].dtype)
                            .1;
                        accum_grad(graph, &mut grads, x, g);
                    }
                    UOp::Reciprocal => {
                        let z_sq = push_binary(graph, cid, cid, BOp::Mul);
                        let neg_z_sq = graph
                            .push(Node::Unary { x: z_sq, uop: UOp::Neg }, graph.classes[z_sq].shape, graph.classes[z_sq].dtype)
                            .1;
                        let g = push_binary(graph, grad, neg_z_sq, BOp::Mul);
                        accum_grad(graph, &mut grads, x, g);
                    }
                    UOp::Exp2 => {
                        let ln2 = Constant::new(std::f64::consts::LN_2).cast(graph.classes[x].dtype);
                        let ln2_cid = graph.push(Node::Const(ln2), scalar_shape, graph.classes[x].dtype).1;
                        let ln2_e = graph
                            .push(
                                Node::Expand { x: ln2_cid, shape: graph.classes[cid].shape },
                                graph.classes[cid].shape,
                                graph.classes[cid].dtype,
                            )
                            .1;
                        let z_ln2 = push_binary(graph, cid, ln2_e, BOp::Mul);
                        let g = push_binary(graph, grad, z_ln2, BOp::Mul);
                        accum_grad(graph, &mut grads, x, g);
                    }
                    UOp::Log2 => {
                        let ln2 = Constant::new(std::f64::consts::LN_2).cast(graph.classes[x].dtype);
                        let ln2_cid = graph.push(Node::Const(ln2), scalar_shape, graph.classes[x].dtype).1;
                        let ln2_e = graph
                            .push(
                                Node::Expand { x: ln2_cid, shape: graph.classes[x].shape },
                                graph.classes[x].shape,
                                graph.classes[x].dtype,
                            )
                            .1;
                        let x_ln2 = push_binary(graph, x, ln2_e, BOp::Mul);
                        let g = push_binary(graph, grad, x_ln2, BOp::Div);
                        accum_grad(graph, &mut grads, x, g);
                    }
                    UOp::Sqrt => {
                        let two = Constant::new(2u8).cast(graph.classes[cid].dtype);
                        let two_cid = graph.push(Node::Const(two), scalar_shape, graph.classes[cid].dtype).1;
                        let two_e = graph
                            .push(
                                Node::Expand { x: two_cid, shape: graph.classes[cid].shape },
                                graph.classes[cid].shape,
                                graph.classes[cid].dtype,
                            )
                            .1;
                        let z2 = push_binary(graph, cid, two_e, BOp::Mul);
                        let g = push_binary(graph, grad, z2, BOp::Div);
                        accum_grad(graph, &mut grads, x, g);
                    }
                    UOp::Sin => {
                        let cos_x =
                            graph.push(Node::Unary { x, uop: UOp::Cos }, graph.classes[x].shape, graph.classes[x].dtype).1;
                        let g = push_binary(graph, grad, cos_x, BOp::Mul);
                        accum_grad(graph, &mut grads, x, g);
                    }
                    UOp::Cos => {
                        let sin_x =
                            graph.push(Node::Unary { x, uop: UOp::Sin }, graph.classes[x].shape, graph.classes[x].dtype).1;
                        let neg_sin = graph
                            .push(Node::Unary { x: sin_x, uop: UOp::Neg }, graph.classes[sin_x].shape, graph.classes[sin_x].dtype)
                            .1;
                        let g = push_binary(graph, grad, neg_sin, BOp::Mul);
                        accum_grad(graph, &mut grads, x, g);
                    }
                    UOp::Exp => {
                        let exp_x =
                            graph.push(Node::Unary { x, uop: UOp::Exp }, graph.classes[x].shape, graph.classes[x].dtype).1;
                        let g = push_binary(graph, grad, exp_x, BOp::Mul);
                        accum_grad(graph, &mut grads, x, g);
                    }
                    UOp::Ln => {
                        let g = push_binary(graph, grad, x, BOp::Div);
                        accum_grad(graph, &mut grads, x, g);
                    }
                    UOp::Abs => {
                        let dtype = graph.classes[x].dtype;
                        let zero = graph.push(Node::Const(Constant::new(0u8).cast(dtype)), scalar_shape, dtype).1;
                        let one = graph.push(Node::Const(Constant::new(1u8).cast(dtype)), scalar_shape, dtype).1;
                        let neg_one = graph.push(Node::Const(Constant::new(-1i8).cast(dtype)), scalar_shape, dtype).1;
                        let zero_e =
                            graph.push(Node::Expand { x: zero, shape: graph.classes[x].shape }, graph.classes[x].shape, dtype).1;
                        let one_e =
                            graph.push(Node::Expand { x: one, shape: graph.classes[x].shape }, graph.classes[x].shape, dtype).1;
                        let neg_one_e = graph
                            .push(Node::Expand { x: neg_one, shape: graph.classes[x].shape }, graph.classes[x].shape, dtype)
                            .1;
                        let is_pos = push_binary(graph, x, zero_e, BOp::Cmpgt);
                        let is_neg = push_binary(graph, x, zero_e, BOp::Cmplt);
                        let sign_pos = push_binary(graph, is_pos, one_e, BOp::Mul);
                        let sign_neg = push_binary(graph, is_neg, neg_one_e, BOp::Mul);
                        let sign = push_binary(graph, sign_pos, sign_neg, BOp::Add);
                        let g = push_binary(graph, grad, sign, BOp::Mul);
                        accum_grad(graph, &mut grads, x, g);
                    }
                    UOp::Floor | UOp::Trunc | UOp::BitNot => {}
                },
                Node::Binary { x, y, bop } => match bop {
                    BOp::Add => {
                        accum_grad(graph, &mut grads, x, grad);
                        accum_grad(graph, &mut grads, y, grad);
                    }
                    BOp::Sub => {
                        accum_grad(graph, &mut grads, x, grad);
                        let neg_grad = graph
                            .push(Node::Unary { x: grad, uop: UOp::Neg }, graph.classes[grad].shape, graph.classes[grad].dtype)
                            .1;
                        accum_grad(graph, &mut grads, y, neg_grad);
                    }
                    BOp::Mul => {
                        let gx = push_binary(graph, grad, y, BOp::Mul);
                        accum_grad(graph, &mut grads, x, gx);
                        let gy = push_binary(graph, grad, x, BOp::Mul);
                        accum_grad(graph, &mut grads, y, gy);
                    }
                    BOp::Div => {
                        let gx = push_binary(graph, grad, y, BOp::Div);
                        accum_grad(graph, &mut grads, x, gx);
                        let neg_grad = graph
                            .push(Node::Unary { x: grad, uop: UOp::Neg }, graph.classes[grad].shape, graph.classes[grad].dtype)
                            .1;
                        let x_mul = push_binary(graph, neg_grad, x, BOp::Mul);
                        let y_sq = push_binary(graph, y, y, BOp::Mul);
                        let gy = push_binary(graph, x_mul, y_sq, BOp::Div);
                        accum_grad(graph, &mut grads, y, gy);
                    }
                    BOp::Pow => {
                        let dtype = graph.classes[x].dtype;
                        let one = graph.push(Node::Const(Constant::new(1u8).cast(dtype)), scalar_shape, dtype).1;
                        let one_e =
                            graph.push(Node::Expand { x: one, shape: graph.classes[y].shape }, graph.classes[y].shape, dtype).1;
                        let y_1 = push_binary(graph, y, one_e, BOp::Sub);
                        let x_pow_ym1 = push_binary(graph, x, y_1, BOp::Pow);
                        let y_mul = push_binary(graph, y, x_pow_ym1, BOp::Mul);
                        let gx = push_binary(graph, grad, y_mul, BOp::Mul);
                        accum_grad(graph, &mut grads, x, gx);
                        let ln_x = graph.push(Node::Unary { x, uop: UOp::Ln }, graph.classes[x].shape, graph.classes[x].dtype).1;
                        let z_lnx = push_binary(graph, cid, ln_x, BOp::Mul);
                        let gy = push_binary(graph, grad, z_lnx, BOp::Mul);
                        accum_grad(graph, &mut grads, y, gy);
                    }
                    BOp::Mod => {
                        accum_grad(graph, &mut grads, x, grad);
                        let x_div_y = push_binary(graph, x, y, BOp::Div);
                        let floored = graph
                            .push(
                                Node::Unary { x: x_div_y, uop: UOp::Floor },
                                graph.classes[x_div_y].shape,
                                graph.classes[x_div_y].dtype,
                            )
                            .1;
                        let neg_floor = graph
                            .push(
                                Node::Unary { x: floored, uop: UOp::Neg },
                                graph.classes[floored].shape,
                                graph.classes[floored].dtype,
                            )
                            .1;
                        let gy = push_binary(graph, neg_floor, grad, BOp::Mul);
                        accum_grad(graph, &mut grads, y, gy);
                    }
                    BOp::Max => {
                        let dtype = graph.classes[x].dtype;
                        let x_gt_y = push_binary(graph, x, y, BOp::Cmpgt);
                        let x_lt_y = push_binary(graph, x, y, BOp::Cmplt);
                        let x_gt_f = graph.push(Node::Cast { x: x_gt_y, dtype }, graph.classes[x_gt_y].shape, dtype).1;
                        let x_lt_f = graph.push(Node::Cast { x: x_lt_y, dtype }, graph.classes[x_lt_y].shape, dtype).1;
                        let gx = push_binary(graph, grad, x_gt_f, BOp::Mul);
                        accum_grad(graph, &mut grads, x, gx);
                        let gy = push_binary(graph, grad, x_lt_f, BOp::Mul);
                        accum_grad(graph, &mut grads, y, gy);
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
                    let g = graph
                        .push(
                            Node::Cast { x: grad, dtype: graph.classes[x].dtype },
                            graph.classes[grad].shape,
                            graph.classes[x].dtype,
                        )
                        .1;
                    accum_grad(graph, &mut grads, x, g);
                }
                Node::Reshape { x, .. } => {
                    let x_shape = graph.classes[x].shape;
                    let g = graph.push(Node::Reshape { x: grad, shape: x_shape }, x_shape, graph.classes[grad].dtype).1;
                    accum_grad(graph, &mut grads, x, g);
                }
                Node::Expand { x, .. } => {
                    let x_shape = graph.classes[x].shape;
                    let sum_axes: Vec<UAxis> = shapes[graph.classes[cid].shape]
                        .iter()
                        .zip(shapes[x_shape].iter())
                        .enumerate()
                        .filter_map(|(i, (&od, &xd))| if od != xd { Some(i as UAxis) } else { None })
                        .collect();
                    if sum_axes.is_empty() {
                        accum_grad(graph, &mut grads, x, grad);
                    } else {
                        let reduced = graph
                            .push(
                                Node::Reduce { x: grad, bop: BOp::Add, axes: sum_axes.into_boxed_slice() },
                                x_shape,
                                graph.classes[grad].dtype,
                            )
                            .1;
                        accum_grad(graph, &mut grads, x, reduced);
                    }
                }
                Node::Permute { x, ref axes } => {
                    let mut inv_axes: Vec<UAxis> = vec![0; axes.len()];
                    for (i, &a) in axes.iter().enumerate() {
                        inv_axes[a as usize] = i as UAxis;
                    }
                    let g = graph
                        .push(
                            Node::Permute { x: grad, axes: inv_axes.into_boxed_slice() },
                            graph.classes[x].shape,
                            graph.classes[grad].dtype,
                        )
                        .1;
                    accum_grad(graph, &mut grads, x, g);
                }
                Node::PadZeros { x, .. } => {
                    accum_grad(graph, &mut grads, x, grad);
                }
                Node::Reduce { x, bop, ref axes } => {
                    let axes = axes.clone();
                    match bop {
                        BOp::Add => {
                            let x_shape_id = graph.classes[x].shape;
                            let x_shape_vec: Vec<Dim> = shapes[x_shape_id].clone();
                            let mut grad_shape_vec: Vec<Dim> = shapes[graph.classes[cid].shape].clone();
                            for &axis in axes.iter() {
                                grad_shape_vec.insert(axis as usize, 1);
                            }
                            if axes.len() == x_shape_vec.len() {
                                grad_shape_vec.remove(0);
                            }
                            let gs = shapes.push(grad_shape_vec);
                            let grad_r = graph.push(Node::Reshape { x: grad, shape: gs }, gs, graph.classes[grad].dtype).1;
                            let g = graph
                                .push(Node::Expand { x: grad_r, shape: x_shape_id }, x_shape_id, graph.classes[grad].dtype)
                                .1;
                            accum_grad(graph, &mut grads, x, g);
                        }
                        BOp::Max => {
                            let x_shape_id = graph.classes[x].shape;
                            let x_shape_vec: Vec<Dim> = shapes[x_shape_id].clone();

                            let mut z_shape_vec: Vec<Dim> = shapes[graph.classes[cid].shape].clone();
                            for &axis in axes.iter() {
                                z_shape_vec.insert(axis as usize, 1);
                            }
                            if axes.len() == x_shape_vec.len() {
                                z_shape_vec.remove(0);
                            }
                            let zs = shapes.push(z_shape_vec);
                            let z_reshaped = graph.push(Node::Reshape { x: cid, shape: zs }, zs, graph.classes[cid].dtype).1;
                            let z_broadcasted = graph
                                .push(Node::Expand { x: z_reshaped, shape: x_shape_id }, x_shape_id, graph.classes[cid].dtype)
                                .1;
                            let cmp = push_binary(graph, x, z_broadcasted, BOp::Cmplt);
                            let cmp_f = graph
                                .push(
                                    Node::Cast { x: cmp, dtype: graph.classes[x].dtype },
                                    graph.classes[cmp].shape,
                                    graph.classes[x].dtype,
                                )
                                .1;
                            let one = graph
                                .push(
                                    Node::Const(Constant::new(1u8).cast(graph.classes[x].dtype)),
                                    scalar_shape,
                                    graph.classes[x].dtype,
                                )
                                .1;
                            let one_e =
                                graph.push(Node::Expand { x: one, shape: x_shape_id }, x_shape_id, graph.classes[x].dtype).1;
                            let mask = push_binary(graph, one_e, cmp_f, BOp::Sub);

                            let mut grad_shape_vec: Vec<Dim> = shapes[graph.classes[grad].shape].clone();
                            for &axis in axes.iter() {
                                grad_shape_vec.insert(axis as usize, 1);
                            }
                            if axes.len() == x_shape_vec.len() {
                                grad_shape_vec.remove(0);
                            }
                            let gs = shapes.push(grad_shape_vec);
                            let grad_r = graph.push(Node::Reshape { x: grad, shape: gs }, gs, graph.classes[grad].dtype).1;
                            let grad_e = graph
                                .push(Node::Expand { x: grad_r, shape: x_shape_id }, x_shape_id, graph.classes[grad].dtype)
                                .1;

                            let grad_x = push_binary(graph, mask, grad_e, BOp::Mul);
                            accum_grad(graph, &mut grads, x, grad_x);
                        }
                        _ => {}
                    }
                }
                Node::ToDevice { x, .. } => {
                    accum_grad(graph, &mut grads, x, grad);
                }
                Node::Leaf { .. } | Node::Const(_) | Node::Kernel { .. } => {}
            }
        }

        grads.retain(|k, _| source_classes.contains(k));

        let mut res = Map::default();
        for tid in sources {
            let class_id = match self.tensors[tid].state {
                TensorState::Graph { class_id, .. } => class_id,
                _ => continue,
            };
            let grad_tid = match grads.get(&class_id) {
                Some(&gcid) => {
                    let graph = &self.graphs[graph_id];
                    let shape_id = graph.classes[gcid].shape;
                    let dtype = graph.classes[gcid].dtype;
                    self.tensors.push(TensorData {
                        shape_id,
                        dtype,
                        state: TensorState::Graph { class_id: gcid, rc: 1, graph_id },
                    })
                }
                None => {
                    let shape: Vec<Dim> = self.shape(tid).into();
                    let dtype = self.dtype(tid);
                    let one_shape = self.push_shape(vec![1]);
                    let full_shape_id = self.push_shape(shape);
let graph = &mut self.graphs[graph_id];
                    let (_, zero_cid) =
                        graph.push(Node::Const(Constant::new(0u8).cast(dtype)), one_shape, dtype);
                    let (_, cid) =
                        graph.push(Node::Expand { x: zero_cid, shape: full_shape_id }, full_shape_id, dtype);
                    self.tensors.push(TensorData {
                        shape_id: full_shape_id,
                        dtype,
                        state: TensorState::Graph { class_id: cid, rc: 1, graph_id },
                    })
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
            if let Some(&rc) = rcs.get(&cid) {
                if rc == *internal_rcs.entry(cid).and_modify(|c| *c += 1).or_insert(1) {
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
