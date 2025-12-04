use crate::{
    Map, RT, Set, Tensor,
    dtype::Constant,
    graph::{BOp, Node, ROp, UOp},
    runtime::{Runtime, deallocate_tensors},
    shape::{Dim, UAxis},
    tensor::TensorId,
};
use std::hash::BuildHasherDefault;

/// Gradient tape
///
/// Graph is always recorded, but when tensor is realized, it's graph is dropped.
/// When [`GradientTape`] is alive, graph is not dropped until [`GradientTape`] is dropped.
///
/// Unlike other deep learning frameworks, there is no need to specify which tensors
/// are differentiable nor is there need to specify multiple gradient tapes to calculate
/// higher order derivatives. In zyx as long as gradient tape is alive, derivatives
/// of all operations then occured since it's creation can be calculated.
///
/// Gradient tape is necessary because without it graph would grow
/// indefinitely with each iteration of training/inference loop.
/// By creating gradient tape in the beginning of each training loop and dropping
/// it at the end, the user ensures that graph of tensors is dropped after each
/// iteration of the training loop.
///
/// Since tensors are realized lazily, intermediate tensors needed for backpropagation
/// are not held in memory.
#[cfg_attr(feature = "py", pyo3::pyclass)]
pub struct GradientTape {}

impl Default for GradientTape {
    fn default() -> Self {
        Self::new()
    }
}

impl GradientTape {
    /// Create new gradient tape. Only one gradient tape can exist at a time.
    pub fn new() -> Self {
        let mut rt = RT.lock();
        rt.graph.gradient_tape_ref_count += 1;
        if rt.graph.gradient_tape.is_some() {
            //panic!("Only one gradient tape can exist at a time.");
            return Self {};
        }
        rt.graph.gradient_tape = Some(Set::with_capacity_and_hasher(100, BuildHasherDefault::default()));
        drop(rt);
        Self {}
    }

    /// Returns gradients of target derived w.r.t. sources
    /// Any ops following this function will not be traced.
    /// If you want to keep tracing, use [`gradient_persistent`](GradientTape::gradient_persistent)
    #[must_use]
    pub fn gradient<'a>(&self, target: &Tensor, sources: impl IntoIterator<Item = &'a Tensor>) -> Vec<Option<Tensor>> {
        let sources: Vec<TensorId> = sources.into_iter().map(Tensor::id).collect();
        //println!("Sources: {sources:?}");
        let mut rt = RT.lock();
        let grads: Map<TensorId, TensorId> = rt.gradient(target.id(), &sources.iter().copied().collect());
        rt.graph.gradient_tape_ref_count += 1;
        rt.drop_gradient_tape();
        drop(rt);
        sources
            .into_iter()
            .map(|x: TensorId| grads.get(&x).copied())
            .map(|id: Option<TensorId>| id.map(|id| Tensor { id }))
            .collect()
    }

    /// Returns gradients of target derived w.r.t. sources
    /// This persistent version keeps gradient tape alive and new ops will be traced until [`GradientTape`] gets destroyed.
    /// This function is useful for higher order derivatives or derivating multiple tensors.
    #[must_use]
    pub fn gradient_persistent<'a>(
        &self,
        target: &Tensor,
        sources: impl IntoIterator<Item = &'a Tensor>,
    ) -> Vec<Option<Tensor>> {
        let sources: Vec<TensorId> = sources.into_iter().map(Tensor::id).collect();
        //println!("Sources: {sources:?}");
        let grads: Map<TensorId, TensorId> = RT.lock().gradient(target.id(), &sources.iter().copied().collect());
        sources
            .into_iter()
            .map(|x: TensorId| grads.get(&x).copied())
            .map(|id: Option<TensorId>| id.map(|id| Tensor { id }))
            .collect()
    }
}

impl Drop for GradientTape {
    fn drop(&mut self) {
        //RT.lock().drop_gradient_tape();
        if let Ok(mut rt) = RT.try_lock() {
            rt.drop_gradient_tape();
        } else {
            println!("Warning: Unable to drop GradientTape due to runtime mutex lock.");
        }
    }
}

impl Runtime {
    pub(super) fn drop_gradient_tape(&mut self) {
        self.graph.gradient_tape_ref_count -= 1;
        if self.graph.gradient_tape_ref_count == 0 {
            if let Some(tape) = &self.graph.gradient_tape {
                // Remove parts of graph that are realized and were needed only for gradient tracing
                let realized_nodes: Set<TensorId> =
                    self.pools.iter().flat_map(|pool| pool.buffer_map.keys()).copied().collect();
                let mut to_release = Vec::new();
                for &nid in realized_nodes.intersection(tape) {
                    let shape = self.graph.shape(nid).into();
                    self.graph.shapes.insert(nid, shape);
                    let dtype = self.dtype(nid);
                    to_release.extend(self.graph[nid].parameters());
                    self.graph.nodes[nid].1 = Node::Leaf { dtype };
                }
                let to_remove = self.graph.release(&to_release);
                deallocate_tensors(&to_remove, &mut self.pools);
                self.graph.gradient_tape = None;
            }
        }
    }

    #[allow(clippy::similar_names)]
    pub(super) fn gradient(&mut self, x: TensorId, sources: &Set<TensorId>) -> Map<TensorId, TensorId> {
        fn insert_or_add_grad(r: &mut Runtime, grads: &mut Map<TensorId, TensorId>, nid: TensorId, grad: TensorId) {
            match grads.entry(nid) {
                std::collections::hash_map::Entry::Vacant(e) => {
                    e.insert(grad);
                }
                std::collections::hash_map::Entry::Occupied(e) => {
                    let (k, prev_grad) = e.remove_entry();
                    grads.insert(k, r.graph.push(Node::Binary { x: prev_grad, y: grad, bop: BOp::Add }));
                    // These can never fail as it just decreses ref count,
                    // there is no deallocation.
                    r.release(prev_grad);
                    r.release(grad);
                }
            }
        }
        //println!("sources={sources:?}");

        // Does not allocate new tensors, only constant and op nodes
        let topo = self.graph.build_topo(x, sources);
        //println!("Topo: {topo:?}");

        let req_grad: Set<TensorId> = topo.iter().copied().chain(sources.iter().copied()).collect();
        // Node -> Grad
        let mut grads: Map<TensorId, TensorId> = Map::with_capacity_and_hasher(100, BuildHasherDefault::default());

        // Initial gradient of ones
        grads.insert(x, self.ones(self.shape(x).into(), self.dtype(x)));
        //println!("{:?}", self.nodes.last().unwrap());

        // All releases that cannot fail use unwrap to catch incorrect refcounts immediatelly.
        // reverse gradient calculation
        for nid in topo {
            let grad = grads[&nid];
            match self.graph[nid] {
                Node::Const { .. } | Node::Leaf { .. } => {}
                Node::Binary { x, y, bop } => match bop {
                    BOp::Add => {
                        if req_grad.contains(&x) {
                            self.retain(grad);
                            insert_or_add_grad(self, &mut grads, x, grad);
                        }
                        if req_grad.contains(&y) {
                            self.retain(grad);
                            insert_or_add_grad(self, &mut grads, y, grad);
                        }
                    }
                    BOp::Sub => {
                        if req_grad.contains(&x) {
                            self.retain(grad);
                            insert_or_add_grad(self, &mut grads, x, grad);
                        }
                        if req_grad.contains(&y) {
                            let grad = self.unary(grad, UOp::Neg);
                            insert_or_add_grad(self, &mut grads, y, grad);
                        }
                    }
                    BOp::Mul => {
                        if req_grad.contains(&x) {
                            let grad = self.binary(y, grad, BOp::Mul);
                            insert_or_add_grad(self, &mut grads, x, grad);
                        }
                        if req_grad.contains(&y) {
                            let grad = self.binary(x, grad, BOp::Mul);
                            insert_or_add_grad(self, &mut grads, y, grad);
                        }
                    }
                    BOp::Div => {
                        if req_grad.contains(&x) {
                            let x_grad = self.binary(grad, y, BOp::Div);
                            insert_or_add_grad(self, &mut grads, x, x_grad);
                        }
                        if req_grad.contains(&y) {
                            // -(grad*x/(y*y))
                            let grad_neg = self.unary(grad, UOp::Neg);
                            let x_mul = self.binary(grad_neg, x, BOp::Mul);
                            self.release(grad_neg);
                            let y_squared = self.binary(y, y, BOp::Mul);
                            let y_grad = self.binary(x_mul, y_squared, BOp::Div);
                            self.release(y_squared);
                            insert_or_add_grad(self, &mut grads, y, y_grad);
                        }
                    }
                    BOp::Mod => {
                        if req_grad.contains(&x) {
                            self.retain(grad);
                            insert_or_add_grad(self, &mut grads, x, grad);
                        }
                        if req_grad.contains(&y) {
                            // (x/y).floor().neg() * grad
                            let x_div_y = self.binary(x, y, BOp::Div);
                            let floored = self.unary(x_div_y, UOp::Floor);
                            self.release(x_div_y);
                            let negated = self.unary(floored, UOp::Neg);
                            self.release(floored);
                            let grad = self.binary(negated, grad, BOp::Mul);
                            self.release(negated);
                            insert_or_add_grad(self, &mut grads, y, grad);
                        }
                    }
                    BOp::Pow => {
                        if req_grad.contains(&x) {
                            // grad * y * x.pow(y-1)
                            let ones = self.ones(self.shape(y).into(), self.dtype(y));
                            let y_1 = self.binary(y, ones, BOp::Sub);
                            self.release(ones);
                            let pow_y_1 = self.binary(x, y_1, BOp::Pow);
                            self.release(y_1);
                            let y_mul = self.binary(y, pow_y_1, BOp::Mul);
                            self.release(pow_y_1);
                            let x_grad = self.binary(grad, y_mul, BOp::Mul);
                            self.release(y_mul);
                            insert_or_add_grad(self, &mut grads, x, x_grad);
                        }
                        if req_grad.contains(&y) {
                            // grad * x.pow(y) * log2(x) * (1/E.log2)
                            let sh = self.shape(y).into();
                            let dtype = self.dtype(y);
                            let one_elog2 = self.graph.push(Node::Const {
                                value: Constant::new(1f64 / std::f64::consts::E.log2()).cast(dtype),
                            });
                            let one_elog2_ex = self.expand(one_elog2, sh).unwrap();
                            self.release(one_elog2);
                            let log2 = self.unary(x, UOp::Log2);
                            let log2_one_elog2 = self.binary(log2, one_elog2_ex, BOp::Mul);
                            self.release(log2);
                            self.release(one_elog2_ex);
                            let xpowy_log2_one_elog2 = self.binary(nid, log2_one_elog2, BOp::Mul);
                            self.release(log2_one_elog2);
                            let y_grad = self.binary(grad, xpowy_log2_one_elog2, BOp::Mul);
                            self.release(xpowy_log2_one_elog2);
                            insert_or_add_grad(self, &mut grads, y, y_grad);
                        }
                        /*if req_grad.contains(&x) {
                            // x_grad = grad * y * x^(y-1)
                            let ones = self.ones(self.shape(y).into(), self.dtype(y));
                            let y_1 = self.binary(y, ones, BOp::Sub);
                            self.release(ones);
                            // Cast x to float to ensure pow works correctly
                            let x_f = self.cast(x, crate::DType::F32);
                            let pow_y_1 = self.binary(x_f, y_1, BOp::Pow);
                            self.release(x_f);
                            self.release(y_1);
                            let y_mul = self.binary(y, pow_y_1, BOp::Mul);
                            self.release(pow_y_1);
                            let x_grad = self.binary(grad, y_mul, BOp::Mul);
                            self.release(y_mul);
                            insert_or_add_grad(self, &mut grads, x, x_grad);
                        }
                        if req_grad.contains(&y) {
                            let x_f = self.cast(x, crate::DType::F32);
                            let pow_xy = self.binary(x_f, y, BOp::Pow);
                            self.release(x_f);
                            let log2_x = self.unary(x_f, UOp::Log2);
                            self.release(pow_xy);
                            let one_elog2 = self.constant(std::f32::consts::LN_2);
                            let one_elog2_ex = self.expand(x, self.shape(x).into()).unwrap();
                            self.release(one_elog2);
                            let ln_x = self.binary(log2_x, one_elog2_ex, BOp::Mul);
                            self.release(log2_x);
                            self.release(one_elog2_ex);
                            let xpowy_log = self.binary(pow_xy, ln_x, BOp::Mul);
                            self.release(pow_xy);
                            self.release(ln_x);
                            let y_grad = self.binary(grad, xpowy_log, BOp::Mul);
                            insert_or_add_grad(self, &mut grads, y, y_grad);
                        }*/
                    }
                    BOp::Maximum => {
                        //# Create masks for where x > y, x < y, and x == y
                        //mask_x_greater = (x > y).to(grad_output.dtype)
                        //mask_y_greater = (x < y).to(grad_output.dtype)
                        //mask_equal = (x == y).to(grad_output.dtype)
                        //# When equal, split gradient equally
                        //grad_x = grad_output * (mask_x_greater + 0.5 * mask_equal)
                        //grad_y = grad_output * (mask_y_greater + 0.5 * mask_equal)

                        let dtype = self.dtype(x);
                        let c = self.graph.push(Node::Const { value: Constant::new(0.5).cast(dtype) });
                        let mask_eq = self.binary(x, y, BOp::Eq);
                        let eq = self.binary(mask_eq, c, BOp::Mul);
                        self.release(c);
                        self.release(mask_eq);
                        if req_grad.contains(&x) {
                            let mask_xgt = self.binary(x, y, BOp::Cmpgt);
                            let add = self.binary(mask_xgt, eq, BOp::Add);
                            self.release(mask_xgt);
                            let x_grad = self.binary(grad, add, BOp::Mul);
                            self.release(add);
                            insert_or_add_grad(self, &mut grads, x, x_grad);
                        }
                        if req_grad.contains(&y) {
                            let mask_ygt = self.binary(x, y, BOp::Cmplt);
                            let add = self.binary(mask_ygt, eq, BOp::Add);
                            self.release(mask_ygt);
                            let y_grad = self.binary(grad, add, BOp::Mul);
                            self.release(add);
                            insert_or_add_grad(self, &mut grads, y, y_grad);
                        }
                        self.release(eq);
                    }
                    BOp::Cmplt
                    | BOp::Cmpgt
                    | BOp::NotEq
                    | BOp::Eq
                    | BOp::Or
                    | BOp::And
                    | BOp::BitAnd
                    | BOp::BitOr
                    | BOp::BitXor
                    | BOp::BitShiftLeft
                    | BOp::BitShiftRight => {}
                },
                Node::Cast { x, .. } => {
                    let grad = self.cast(grad, self.dtype(x));
                    insert_or_add_grad(self, &mut grads, x, grad);
                }
                Node::Unary { x, uop } => match uop {
                    UOp::BitNot => todo!(),
                    UOp::Reciprocal => {
                        // -1/(x*x)
                        let x_2_inv = self.binary(nid, nid, BOp::Mul);
                        let x_grad = self.unary(x_2_inv, UOp::Neg);
                        self.release(x_2_inv);
                        insert_or_add_grad(self, &mut grads, x, x_grad);
                    }
                    UOp::ReLU => {
                        let zeros = self.zeros(self.shape(x).into(), self.dtype(x));
                        let zl = self.binary(zeros, x, BOp::Cmplt);
                        self.release(zeros);
                        let zl_cast = self.cast(zl, self.dtype(x));
                        self.release(zl);
                        let x_grad = self.binary(zl_cast, grad, BOp::Mul);
                        self.release(zl_cast);
                        insert_or_add_grad(self, &mut grads, x, x_grad);
                    }
                    UOp::Exp2 => {
                        let dtype = self.dtype(x);
                        let c = std::f64::consts::E.log2();
                        let temp = self.graph.push(Node::Const { value: Constant::new(c).cast(dtype) });
                        let temp1 = self.expand(temp, self.shape(x).into()).unwrap();
                        self.release(temp);
                        let temp2 = self.binary(nid, temp1, BOp::Mul);
                        self.release(temp1);
                        let grad = self.binary(nid, temp2, BOp::Mul);
                        self.release(temp2);
                        insert_or_add_grad(self, &mut grads, x, grad);
                    }
                    UOp::Log2 => {
                        let dtype = self.dtype(x);
                        let c = std::f64::consts::E.log2();
                        let temp = self.graph.push(Node::Const { value: Constant::new(c).cast(dtype) });
                        let temp1 = self.expand(temp, self.shape(x).into()).unwrap();
                        self.release(temp);
                        let temp2 = self.binary(x, temp1, BOp::Mul);
                        self.release(temp1);
                        let grad = self.binary(grad, temp2, BOp::Div);
                        self.release(temp2);
                        insert_or_add_grad(self, &mut grads, x, grad);
                    }
                    UOp::Sin => {
                        let x_temp = self.unary(x, UOp::Cos);
                        let grad = self.binary(x_temp, grad, BOp::Mul);
                        self.release(x_temp);
                        insert_or_add_grad(self, &mut grads, x, grad);
                    }
                    UOp::Cos => {
                        let x_temp1 = self.unary(x, UOp::Sin);
                        let x_temp = self.unary(x_temp1, UOp::Neg);
                        self.release(x_temp1);
                        let grad = self.binary(x_temp, grad, BOp::Mul);
                        self.release(x_temp);
                        insert_or_add_grad(self, &mut grads, x, grad);
                    }
                    UOp::Sqrt => {
                        // x_grad = grad/(2*sqrt(x))
                        let sqrt_x = self.unary(x, UOp::Sqrt);
                        let sqrtx_2 = self.binary(sqrt_x, sqrt_x, BOp::Add);
                        self.release(sqrt_x);
                        let grad = self.binary(grad, sqrtx_2, BOp::Div);
                        insert_or_add_grad(self, &mut grads, x, grad);
                    }
                    UOp::Neg => {
                        let grad = self.unary(grad, UOp::Neg);
                        insert_or_add_grad(self, &mut grads, x, grad);
                    }
                    UOp::Floor => {
                        let dtype = self.dtype(x);
                        let temp = self.graph.push(Node::Const { value: Constant::new(0).cast(dtype) });
                        let grad = self.expand(temp, self.shape(x).into()).unwrap();
                        self.release(temp);
                        insert_or_add_grad(self, &mut grads, x, grad);
                    } /*UOp::Tanh => {
                          // 1 - tanh^2(x)
                          let tanh_x_2 = self.mul(nid, nid);
                          let ones = self.ones(self.shape(x).into(), self.dtype(x));
                          let grad = self.sub(ones, tanh_x_2);
                          self.release(ones).unwrap();
                          self.release(tanh_x_2).unwrap();
                          insert_or_add_grad(self, &mut grads, x, grad);
                      }*/
                },
                Node::Reshape { x, .. } => {
                    let grad = self.reshape(grad, self.shape(x).into());
                    insert_or_add_grad(self, &mut grads, x, grad);
                }
                Node::Expand { x } => {
                    let sh = self.graph.shape(nid);
                    let x_shape: Vec<Dim> = self.shape(x).into();
                    debug_assert_eq!(sh.len(), x_shape.len());
                    let expand_axes: Vec<UAxis> = sh
                        .iter()
                        .zip(&x_shape)
                        .enumerate()
                        .filter_map(|(a, (&d, &e))| if d == e { None } else { Some(a as UAxis) })
                        .collect();
                    //println!("x shape {:?}, nid shape {:?}, expand_axes: {:?}", x_shape, sh, expand_axes);
                    debug_assert!(!expand_axes.is_empty());
                    let temp = self.sum_reduce(grad, expand_axes);
                    let grad = self.reshape(temp, x_shape);
                    self.release(temp);
                    insert_or_add_grad(self, &mut grads, x, grad);
                }
                Node::Permute { x } => {
                    let axes = self.graph.axes(nid);
                    let mut axes: Vec<(usize, UAxis)> = axes.iter().copied().enumerate().collect();
                    axes.sort_by_key(|(_, v)| *v);
                    let argsort_axes: Vec<UAxis> = axes.iter().map(|(k, _)| *k as UAxis).collect();
                    let grad = self.permute(grad, &argsort_axes);
                    insert_or_add_grad(self, &mut grads, x, grad);
                }
                Node::Pad { x } => {
                    let padding = self.graph.padding(x);
                    let inv_padding = padding.iter().map(|(lp, rp)| (-lp, -rp)).collect();
                    let grad = self.pad_zeros(grad, inv_padding);
                    insert_or_add_grad(self, &mut grads, x, grad);
                }
                Node::Reduce { x, rop } => match rop {
                    ROp::Sum => {
                        let x_shape: Vec<Dim> = self.shape(x).into();
                        let mut z_shape: Vec<Dim> = self.shape(nid).into();
                        //println!("Reduce backward, z shape: {z_shape:?}, x shape: {x_shape:?}, reduce axes: {:?}", self.graph.axes(nid));
                        for &axis in self.graph.axes(nid) {
                            z_shape.insert(axis as usize, 1);
                        }
                        if self.graph.axes(nid).len() == x_shape.len() {
                            z_shape.remove(0);
                        }
                        let temp = self.reshape(grad, z_shape);
                        let grad = self.expand(temp, x_shape).unwrap();
                        self.release(temp);
                        insert_or_add_grad(self, &mut grads, x, grad);
                    }
                    ROp::Max => {
                        // x_grad = (1 - (x < z.expand(x.shape()))) * grad
                        let x_shape: Vec<Dim> = self.shape(x).into();
                        let z_temp = self.expand(nid, x_shape.clone()).unwrap();
                        let cmp_t = self.binary(x, z_temp, BOp::Cmplt);
                        self.release(z_temp);
                        let ones = self.zeros(x_shape, self.dtype(x));
                        let max_1s = self.binary(ones, cmp_t, BOp::Sub);
                        self.release(ones);
                        self.release(cmp_t);
                        let grad = self.binary(max_1s, grad, BOp::Mul);
                        self.release(max_1s);
                        insert_or_add_grad(self, &mut grads, x, grad);
                    }
                },
            }
        }
        //println!("gradients: {grads:?}");
        let mut res = Map::with_capacity_and_hasher(10, BuildHasherDefault::default());
        for (k, v) in grads {
            if sources.contains(&k) {
                res.insert(k, v);
            } else {
                self.release(v);
            }
        }
        //println!("res: {res:?}");
        res
    }
}
