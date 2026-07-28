use std::collections::BTreeSet;

use crate::{
    DType, Map, Set,
    graph::{ClassId, EKernelData, EKernelId, Graph, Node, NodeData},
    kernel::{BOp, DeviceId, Kernel, MoveOp, Op, OpId, UOp},
    runtime::ShapeId,
    shape::{Dim, UAxis},
    slab::{Slab, SlabId},
    view::View,
};

impl Graph {
    /// Fuses remaining non-Kernel nodes into kernels so every output has an all-Kernel path to leaves.
    ///
    /// The zyx graph is very granular — a single matmul produces dozens of structural nodes (Expand,
    /// Reduce, Permute, Cast, etc.). Materializing each as a separate kernel is impossible (e.g.,
    /// Expand to 2048×2048×2048 would OOM). [`fill_remaining`] uses eager fusion to batch structural
    /// nodes into larger kernels, ensuring the [`extract`](Graph::extract) invariant holds:
    ///
    /// > A path composed exclusively of [`Node::Kernel`] and [`Node::ToDevice`] nodes must exist
    /// > from leaves (realized classes) to every output.
    ///
    /// Not every class needs a kernel — only those that lie on output computation paths. Dead graph
    /// regions without kernels are harmless. After this function returns, all classes that *are* on
    /// output paths must be covered by a kernel.
    ///
    /// # Reference Counts (rcs)
    ///
    /// Before processing, each class's reference count is computed: `rcs[cid]` is the number of
    /// times `cid` appears as a `class_params` input of another graph node **plus** 1 for each
    /// user-requested output class. The output classes are "consumed" by extraction — a terminal
    /// output has rcs = 1 rather than 0.
    ///
    /// When a class is produced (its operation is added to a kernel), the producer pushes exactly
    /// `rcs[cid]` copies of `cid` into the kernel's `outputs` list — one copy per consumer.
    /// Each consumer later calls [`remove_first_output`] to remove one copy, and decrements
    /// `rcs[cid]` by 1. When all copies are consumed (`rcs[cid] == 0` and `outputs` contains no
    /// more instances of `cid`), the class no longer holds the kernel open.
    ///
    /// # Storage and Load Kernels
    ///
    /// [`add_store`] stores a class's value into a kernel's `stores` list. On store:
    /// - All instances of the class are removed from the kernel's `outputs` (via `retain`).
    /// - If `rcs[cid] > 0` after the store (remaining consumers exist), a new **load kernel**
    ///   is created with `rcs[cid]` copies of `cid` in its `outputs`. This load kernel provides
    ///   the class's value for all remaining consumers via a reload from storage.
    /// - The class is removed from `visited`. If a load kernel was created, the class is
    ///   re-inserted into `visited` pointing to the load kernel.
    ///
    /// # Invariants (maintained at all times)
    ///
    /// 1. **Output count**: For each class `cid`, the total number of occurrences across all
    ///    kernels' `outputs` lists equals `rcs[cid]`. A class appears in at most one kernel.
    /// 2. **Visited residency**: Every class with `rcs[cid] > 0` that has been produced must have
    ///    exactly one entry in `visited` mapping it to the kernel where its computation lives.
    ///    [`add_store`] removes the entry and restores it via a load kernel if consumers remain.
    /// 3. **Sealing safety**: When a kernel is sealed (all output instances exhausted),
    ///    [`visited.retain`] kills all its visited entries. Every removed class either has
    ///    `rcs == 0` or already has a fresh visited entry pointing to a load kernel.
    pub fn fill_remaining(
        &mut self,
        outputs: &BTreeSet<ClassId>,
        shapes: &Slab<ShapeId, Vec<Dim>>,
        realized_nodes: Set<ClassId>,
    ) {
        let order = self.topo_sort_classes(outputs);

        let mut rcs: Map<ClassId, u32> = Map::default();
        for &cid in &order {
            for nid in &self.classes[cid].nodes {
                for child in self.nodes[*nid].node.class_params() {
                    *rcs.entry(child).or_default() += 1;
                }
            }
        }

        // User-requested outputs are consumers too — extraction needs a producer path for them.
        for &cid in outputs {
            *rcs.entry(cid).or_default() += 1;
        }

        let mut pending_stores: Set<ClassId> = realized_nodes;
        let mut visited: Map<ClassId, (EKernelId, OpId)> = Map::default();

        for (i, &cid) in order.iter().enumerate() {
            debug_assert!(!visited.contains_key(&cid), "class {cid:?} already visited");

            // If this class is in pending_stores (realized), create a load kernel.
            if pending_stores.contains(&cid) {
                let kid = self.new_load_kernel(cid, shapes);
                let rc = rcs.get(&cid).copied().unwrap_or(0) as usize;
                for _ in 0..rc {
                    self.ekernels[kid].outputs.push(cid);
                }
                visited.insert(cid, (kid, self.ekernels[kid].kernel.head));
                continue;
            }

            let nid = self.classes[cid].nodes[0];
            let node = &self.nodes[nid].node;

            println!("cid={} nid={} rc={}, {node:?}", cid.0, nid.0, rcs[&cid]);
            match node {
                Node::Leaf { .. } => unreachable!(),
                Node::Const(c) => {
                    let mut kernel = Kernel::new(DeviceId::AUTO);
                    let result_op = kernel.push_back(Op::ConstView(Box::new((*c, View::contiguous(&[1])))));
                    let kid =
                        self.ekernels.push(EKernelData { kernel, outputs: Vec::new(), loads: Vec::new(), stores: Vec::new() });
                    let n_consumers = rcs.get(&cid).copied().unwrap_or(0) as usize;
                    for _ in 0..n_consumers {
                        self.ekernels[kid].outputs.push(cid);
                    }
                    visited.insert(cid, (kid, result_op));
                }
                Node::Unary { x, uop } => {
                    self.add_unary(cid, *x, *uop, &mut visited, &mut rcs, shapes);
                }
                Node::Cast { x, dtype } => {
                    self.add_cast(cid, *x, *dtype, &mut visited, &mut rcs, shapes);
                }
                Node::Binary { x, y, bop } => {
                    self.add_binary(cid, *x, *y, *bop, &mut visited, &mut rcs, &mut pending_stores, shapes);
                }
                Node::Reduce { x, bop, axes } => {
                    let axes: Vec<UAxis> = axes.to_vec();
                    self.add_reduce(cid, *x, *bop, axes, &mut visited, &mut rcs, &mut pending_stores, shapes);
                }
                Node::Expand { x, shape } => {
                    self.add_expand(cid, *x, *shape, &mut visited, &mut rcs, &mut pending_stores, shapes);
                }
                Node::Permute { x, axes } => {
                    let axes: Vec<UAxis> = axes.to_vec();
                    self.add_permute(cid, *x, axes, &mut visited, &mut rcs, &mut pending_stores, shapes);
                }
                Node::Reshape { x, shape } => {
                    let shape_vec: Vec<Dim> = shapes[*shape].clone();
                    self.add_reshape(cid, *x, shape_vec, &mut visited, &mut rcs, &mut pending_stores, shapes);
                }
                Node::PadZeros { x, padding } => {
                    let padding: Vec<(i64, i64)> = padding.to_vec();
                    self.add_pad(cid, *x, padding, &mut visited, &mut rcs, &mut pending_stores, shapes);
                }
                Node::ToDevice { x, .. } => {
                    let x = *x;
                    let (child_kid, child_op) = self.child_to_kid(x, &mut visited, shapes);
                    self.add_store(x, child_kid, child_op, &mut visited, &mut pending_stores, &rcs, shapes);
                    let (kid, op_id) = self.child_to_kid(x, &mut visited, shapes);
                    visited.insert(cid, (kid, op_id));
                }
                Node::Kernel { .. } => {}
            }

            // Post-processing: store if final output
            if outputs.contains(&cid) {
                let (kid, op_id) = visited[&cid];
                *rcs.get_mut(&cid).unwrap() -= 1;
                self.add_store(cid, kid, op_id, &mut visited, &mut pending_stores, &rcs, shapes);
            }

            // Highly important invariant checks, DO NOT TOUCH
            for ek in self.ekernels.values() {
                let mut counts: Map<ClassId, u32> = Map::default();
                for &ocid in &ek.outputs {
                    *counts.entry(ocid).or_default() += 1;
                }
                if !counts.is_empty() && counts.iter().all(|(c, &n)| rcs.get(c).copied().unwrap_or(0) != n) {
                    println!("outputs={:?}", ek.outputs);
                    ek.kernel.debug();
                    panic!("output:rcs invariant violated for kernel after {cid:?}");
                }
            }
            debug_assert!(
                order[..=i].iter().all(|&c| if let Some(&rc) = rcs.get(&c) {
                    if rc == 0 {
                        !visited.contains_key(&c)
                    } else {
                        visited.contains_key(&c)
                    }
                } else {
                    true
                }),
                "class with rcs>0 not in visited after {cid:?}"
            );
        }

        // Highly important invariant checks, DO NOT TOUCH
        debug_assert!(rcs.values().all(|&r| r == 0), "all rcs must be zero");
        debug_assert!(visited.is_empty(), "visited must be empty");
        for kernel in self.ekernels.values() {
            debug_assert!(kernel.outputs.is_empty());
            debug_assert!(kernel.stores.len() > 0);
        }
    }

    fn new_load_kernel(&mut self, cid: ClassId, shapes: &Slab<ShapeId, Vec<Dim>>) -> EKernelId {
        let mut kernel = Kernel::new(DeviceId::NULL);
        let shape: Vec<Dim> = shapes[self.classes[cid].shape].clone();
        let is_const = self.classes[cid].nodes.iter().any(|&nid| matches!(&self.nodes[nid].node, Node::Const(_)));
        if is_const {
            let value = self.classes[cid]
                .nodes
                .iter()
                .copied()
                .find_map(|nid| {
                    if let Node::Const(v) = &self.nodes[nid].node {
                        Some(*v)
                    } else {
                        None
                    }
                })
                .unwrap();
            kernel.push_back(Op::ConstView(Box::new((value, View::contiguous(&[1])))));
        } else {
            kernel.load_contiguous(self.classes[cid].dtype, &shape);
        }
        let kid = self.ekernels.push(EKernelData {
            kernel,
            outputs: Vec::new(),
            loads: if is_const { Vec::new() } else { vec![cid] },
            stores: Vec::new(),
        });
        kid
    }

    fn child_to_kid(
        &mut self,
        child: ClassId,
        visited: &mut Map<ClassId, (EKernelId, OpId)>,
        shapes: &Slab<ShapeId, Vec<Dim>>,
    ) -> (EKernelId, OpId) {
        match visited.get(&child) {
            Some(&v) => v,
            None => {
                let kid = self.new_load_kernel(child, shapes);
                let op = self.ekernels[kid].kernel.head;
                visited.insert(child, (kid, op));
                (kid, op)
            }
        }
    }

    fn add_store(
        &mut self,
        cid: ClassId,
        kid: EKernelId,
        op_id: OpId,
        visited: &mut Map<ClassId, (EKernelId, OpId)>,
        pending_stores: &mut Set<ClassId>,
        rcs: &Map<ClassId, u32>,
        shapes: &Slab<ShapeId, Vec<Dim>>,
    ) {
        debug_assert!(!pending_stores.contains(&cid));
        pending_stores.insert(cid);

        let dtype = self.classes[cid].dtype;
        self.ekernels[kid].kernel.store_contiguous(op_id, dtype);
        self.ekernels[kid].stores.push(cid);
        visited.remove(&cid);

        // Remove all occurences of x
        let outputs = &mut self.ekernels[kid].outputs;
        debug_assert_eq!(rcs[&cid], outputs.iter().filter(|&&x| x == cid).count() as u32 - 1);
        outputs.retain(|&x| x != cid);

        if let Some(rc) = rcs.get(&cid).copied()
            && rc > 0
        {
            let new_kid = self.new_load_kernel(cid, shapes);
            let new_op = self.ekernels[new_kid].kernel.head;
            self.ekernels[new_kid].outputs = vec![cid; rc as usize];
            visited.insert(cid, (new_kid, new_op));
        }
    }

    fn merge_kernels(&mut self, src: EKernelId, dst: EKernelId, visited: &mut Map<ClassId, (EKernelId, OpId)>) {
        let EKernelData { kernel: src_kernel, outputs, loads, stores } = unsafe { self.ekernels.remove_and_return(src) };

        {
            let dst_data = &mut self.ekernels[dst];
            dst_data.outputs.extend(outputs);
            dst_data.loads.extend(loads);
            dst_data.stores.extend(stores);
        }

        let mut op_map: Map<OpId, OpId> = Map::default();
        let mut i = src_kernel.head;
        while !i.is_null() {
            let mut op = src_kernel.ops[i].op.clone();
            for param in op.parameters_mut() {
                if !param.is_null() {
                    if let Some(&new_param) = op_map.get(param) {
                        *param = new_param;
                    }
                }
            }
            let new_id = self.ekernels[dst].kernel.push_back(op);
            op_map.insert(i, new_id);
            i = src_kernel.ops[i].next;
        }

        for (_, (kid, op_id)) in visited.iter_mut() {
            if *kid == src {
                *kid = dst;
                if let Some(&new_op) = op_map.get(op_id) {
                    *op_id = new_op;
                }
            }
        }
    }

    fn duplicate_or_store_class(
        &mut self,
        child: ClassId,
        mut kid: EKernelId,
        mut op_id: OpId,
        visited: &mut Map<ClassId, (EKernelId, OpId)>,
        pending_stores: &mut Set<ClassId>,
        rcs: &mut Map<ClassId, u32>,
        shapes: &Slab<ShapeId, Vec<Dim>>,
    ) -> (EKernelId, OpId) {
        // if kernel has stores, store child and create fresh load kernel
        let has_stores = self.ekernels[kid].kernel.contains_stores();
        if has_stores {
            self.add_store(child, kid, op_id, visited, pending_stores, rcs, shapes);
            if rcs.get(&child).copied().unwrap_or(0) > 0 {
                let (nk, no) = visited[&child];
                kid = nk;
                op_id = no;
            } else {
                let new_kid = self.new_load_kernel(child, shapes);
                let new_op = self.ekernels[new_kid].kernel.head;
                visited.insert(child, (new_kid, new_op));
                kid = new_kid;
                op_id = new_op;
            }
        }

        // if kernel has multiple outputs, duplicate the kernel
        let n_outputs = self.ekernels[kid].outputs.len();
        if n_outputs > 1 {
            let preceded_by_reduce = self.ekernels[kid].kernel.is_preceded_by_reduce(op_id);
            if preceded_by_reduce {
                self.add_store(child, kid, op_id, visited, pending_stores, rcs, shapes);
                if rcs.get(&child).copied().unwrap_or(0) > 0 {
                    let (nk, no) = visited[&child];
                    kid = nk;
                    op_id = no;
                } else {
                    let new_kid = self.new_load_kernel(child, shapes);
                    let new_op = self.ekernels[new_kid].kernel.head;
                    visited.insert(child, (new_kid, new_op));
                    kid = new_kid;
                    op_id = new_op;
                }
            } else {
                remove_first_output(&mut self.ekernels, kid, child);

                let out_op_ids: Vec<OpId> = self.ekernels[kid].outputs.iter().map(|&cid| visited[&cid].1).collect();
                let loads = self.ekernels[kid].loads.clone();
                let (new_kernel, new_op_id, self_loads, new_loads) =
                    self.ekernels[kid].kernel.extract_subkernel(op_id, &out_op_ids, &loads);
                self.ekernels[kid].loads = self_loads;

                let new_kid = self.ekernels.push(EKernelData {
                    kernel: new_kernel,
                    outputs: vec![child],
                    loads: new_loads,
                    stores: Vec::new(),
                });
                op_id = new_op_id;
                kid = new_kid;
            }
        }

        (kid, op_id)
    }

    fn add_unary(
        &mut self,
        cid: ClassId,
        child: ClassId,
        uop: UOp,
        visited: &mut Map<ClassId, (EKernelId, OpId)>,
        rcs: &mut Map<ClassId, u32>,
        shapes: &Slab<ShapeId, Vec<Dim>>,
    ) {
        let (kid, op_id) = self.child_to_kid(child, visited, shapes);
        *rcs.get_mut(&child).unwrap() -= 1;
        remove_first_output(&mut self.ekernels, kid, child);
        if rcs.get(&child).copied().unwrap_or(0) == 0 {
            visited.remove(&child);
        }
        let kernel = &mut self.ekernels[kid].kernel;
        let result_op = kernel.unary(op_id, uop);
        let n_consumers = rcs.get(&cid).copied().unwrap_or(0) as usize;
        for _ in 0..n_consumers {
            self.ekernels[kid].outputs.push(cid);
        }
        visited.insert(cid, (kid, result_op));
    }

    fn add_cast(
        &mut self,
        cid: ClassId,
        child: ClassId,
        dtype: DType,
        visited: &mut Map<ClassId, (EKernelId, OpId)>,
        rcs: &mut Map<ClassId, u32>,
        shapes: &Slab<ShapeId, Vec<Dim>>,
    ) {
        let (kid, op_id) = self.child_to_kid(child, visited, shapes);
        *rcs.get_mut(&child).unwrap() -= 1;
        remove_first_output(&mut self.ekernels, kid, child);
        if rcs.get(&child).copied().unwrap_or(0) == 0 {
            visited.remove(&child);
        }
        let kernel = &mut self.ekernels[kid].kernel;
        let result_op = kernel.cast(op_id, dtype);
        let n_consumers = rcs.get(&cid).copied().unwrap_or(0) as usize;
        for _ in 0..n_consumers {
            self.ekernels[kid].outputs.push(cid);
        }
        visited.insert(cid, (kid, result_op));
    }

    fn add_binary(
        &mut self,
        cid: ClassId,
        lhs: ClassId,
        rhs: ClassId,
        bop: BOp,
        visited: &mut Map<ClassId, (EKernelId, OpId)>,
        rcs: &mut Map<ClassId, u32>,
        pending_stores: &mut Set<ClassId>,
        shapes: &Slab<ShapeId, Vec<Dim>>,
    ) {
        let (mut kid, mut op_id) = self.child_to_kid(lhs, visited, shapes);
        let (mut kidy, op_idy) = self.child_to_kid(rhs, visited, shapes);

        let kid_stores = self.ekernels[kid].kernel.contains_stores();
        let kidy_stores = self.ekernels[kidy].kernel.contains_stores();

        if kid == kidy {
            *rcs.get_mut(&lhs).unwrap() -= 1;
            *rcs.get_mut(&rhs).unwrap() -= 1;
            remove_first_output(&mut self.ekernels, kid, lhs);
            remove_first_output(&mut self.ekernels, kid, rhs);
            if rcs.get(&lhs).copied().unwrap_or(0) == 0 {
                visited.remove(&lhs);
            }
            if rcs.get(&rhs).copied().unwrap_or(0) == 0 {
                visited.remove(&rhs);
            }
            let result_op = self.ekernels[kid].kernel.binary(op_id, op_idy, bop);
            let n_consumers = rcs.get(&cid).copied().unwrap_or(0) as usize;
            for _ in 0..n_consumers {
                self.ekernels[kid].outputs.push(cid);
            }
            visited.insert(cid, (kid, result_op));
        } else {
            *rcs.get_mut(&lhs).unwrap() -= 1;
            *rcs.get_mut(&rhs).unwrap() -= 1;
            match (kid_stores, kidy_stores) {
                (true, true) => {
                    self.add_store(lhs, kid, op_id, visited, pending_stores, rcs, shapes);
                    let (nk, no) = self.child_to_kid(lhs, visited, shapes);
                    (kid, op_id) = (nk, no);

                    self.add_store(rhs, kidy, op_idy, visited, pending_stores, rcs, shapes);
                    let (nk, _) = self.child_to_kid(rhs, visited, shapes);
                    (kidy, _) = (nk, nk);
                }
                (true, false) => {
                    self.add_store(lhs, kid, op_id, visited, pending_stores, rcs, shapes);
                    let (nk, no) = self.child_to_kid(lhs, visited, shapes);
                    (kid, op_id) = (nk, no);
                }
                (false, true) => {
                    self.add_store(rhs, kidy, op_idy, visited, pending_stores, rcs, shapes);
                    let (nk, _) = self.child_to_kid(rhs, visited, shapes);
                    (kidy, _) = (nk, nk);
                }
                (false, false) => {}
            }

            self.merge_kernels(kidy, kid, visited);
            let (_, op_idy) = visited[&rhs];
            remove_first_output(&mut self.ekernels, kid, lhs);
            remove_first_output(&mut self.ekernels, kid, rhs);
            if rcs.get(&lhs).copied().unwrap_or(0) == 0 {
                visited.remove(&lhs);
            }
            if rcs.get(&rhs).copied().unwrap_or(0) == 0 {
                visited.remove(&rhs);
            }
            let result_op = self.ekernels[kid].kernel.binary(op_id, op_idy, bop);
            let n_consumers = rcs.get(&cid).copied().unwrap_or(0) as usize;
            for _ in 0..n_consumers {
                self.ekernels[kid].outputs.push(cid);
            }
            visited.insert(cid, (kid, result_op));
        }
    }

    fn add_reduce(
        &mut self,
        cid: ClassId,
        child: ClassId,
        rop: BOp,
        axes: Vec<UAxis>,
        visited: &mut Map<ClassId, (EKernelId, OpId)>,
        rcs: &mut Map<ClassId, u32>,
        pending_stores: &mut Set<ClassId>,
        shapes: &Slab<ShapeId, Vec<Dim>>,
    ) {
        let n_axes = axes.len() as UAxis;
        let (mut kid, mut op_id) = self.child_to_kid(child, visited, shapes);
        *rcs.get_mut(&child).unwrap() -= 1;
        (kid, op_id) = self.duplicate_or_store_class(child, kid, op_id, visited, pending_stores, rcs, shapes);
        remove_first_output(&mut self.ekernels, kid, child);
        if rcs.get(&child).copied().unwrap_or(0) == 0 {
            visited.remove(&child);
        }

        let in_shape: Vec<Dim> = shapes[self.classes[child].shape].clone();
        let kernel = &mut self.ekernels[kid].kernel;
        let permuted = {
            let n = in_shape.len();
            let max_axis = *axes.last().unwrap() as usize;
            let mut permute_axes = Vec::with_capacity(n);
            let mut ai = 0;
            for i in 0..=max_axis {
                if axes[ai] as usize == i {
                    ai += 1;
                } else {
                    permute_axes.push(i as UAxis);
                }
            }
            permute_axes.extend((max_axis + 1..n).map(|i| i as UAxis));
            permute_axes.extend_from_slice(&axes);
            if !permute_axes.iter().copied().eq(0..permute_axes.len() as UAxis) {
                let shape = crate::shape::permute(&in_shape, &permute_axes);
                kernel.push_back(Op::Move { x: op_id, mop: Box::new(MoveOp::Permute { axes: permute_axes, shape }) })
            } else {
                op_id
            }
        };
        let mut result_op = kernel.push_back(Op::Reduce { x: permuted, rop, n_axes: n_axes as UAxis });
        if in_shape.len() == n_axes as usize {
            result_op = kernel.reshape(result_op, &[1]);
        }

        let n_consumers = rcs.get(&cid).copied().unwrap_or(0) as usize;
        for _ in 0..n_consumers {
            self.ekernels[kid].outputs.push(cid);
        }
        visited.insert(cid, (kid, result_op));
    }

    fn add_expand(
        &mut self,
        cid: ClassId,
        child: ClassId,
        _shape: ShapeId,
        visited: &mut Map<ClassId, (EKernelId, OpId)>,
        rcs: &mut Map<ClassId, u32>,
        pending_stores: &mut Set<ClassId>,
        shapes: &Slab<ShapeId, Vec<Dim>>,
    ) {
        let (mut kid, mut op_id) = self.child_to_kid(child, visited, shapes);
        *rcs.get_mut(&child).unwrap() -= 1;
        (kid, op_id) = self.duplicate_or_store_class(child, kid, op_id, visited, pending_stores, rcs, shapes);
        if self.ekernels[kid].kernel.is_preceded_by_compute(op_id) {
            self.add_store(child, kid, op_id, visited, pending_stores, rcs, shapes);
            if rcs.get(&child).copied().unwrap_or(0) > 0 {
                let (nk, no) = visited[&child];
                kid = nk;
                op_id = no;
            } else {
                let new_kid = self.new_load_kernel(child, shapes);
                let new_op = self.ekernels[new_kid].kernel.head;
                visited.insert(child, (new_kid, new_op));
                kid = new_kid;
                op_id = new_op;
            }
        }
        remove_first_output(&mut self.ekernels, kid, child);
        if rcs.get(&child).copied().unwrap_or(0) == 0 {
            visited.remove(&child);
        }
        let shape: Vec<Dim> = shapes[self.classes[cid].shape].clone();
        let kernel = &mut self.ekernels[kid].kernel;
        let result_op = kernel.push_back(Op::Move { x: op_id, mop: Box::new(MoveOp::Expand { shape }) });
        let n_consumers = rcs.get(&cid).copied().unwrap_or(0) as usize;
        for _ in 0..n_consumers {
            self.ekernels[kid].outputs.push(cid);
        }
        visited.insert(cid, (kid, result_op));
    }

    fn add_permute(
        &mut self,
        cid: ClassId,
        child: ClassId,
        axes: Vec<UAxis>,
        visited: &mut Map<ClassId, (EKernelId, OpId)>,
        rcs: &mut Map<ClassId, u32>,
        pending_stores: &mut Set<ClassId>,
        shapes: &Slab<ShapeId, Vec<Dim>>,
    ) {
        let (mut kid, mut op_id) = self.child_to_kid(child, visited, shapes);
        *rcs.get_mut(&child).unwrap() -= 1;
        (kid, op_id) = self.duplicate_or_store_class(child, kid, op_id, visited, pending_stores, rcs, shapes);
        remove_first_output(&mut self.ekernels, kid, child);
        if rcs.get(&child).copied().unwrap_or(0) == 0 {
            visited.remove(&child);
        }
        let shape: Vec<Dim> = shapes[self.classes[cid].shape].clone();
        let kernel = &mut self.ekernels[kid].kernel;
        let result_op = kernel.push_back(Op::Move { x: op_id, mop: Box::new(MoveOp::Permute { axes, shape }) });
        let n_consumers = rcs.get(&cid).copied().unwrap_or(0) as usize;
        for _ in 0..n_consumers {
            self.ekernels[kid].outputs.push(cid);
        }
        visited.insert(cid, (kid, result_op));
    }

    fn add_reshape(
        &mut self,
        cid: ClassId,
        child: ClassId,
        shape: Vec<Dim>,
        visited: &mut Map<ClassId, (EKernelId, OpId)>,
        rcs: &mut Map<ClassId, u32>,
        pending_stores: &mut Set<ClassId>,
        shapes: &Slab<ShapeId, Vec<Dim>>,
    ) {
        let (mut kid, mut op_id) = self.child_to_kid(child, visited, shapes);
        *rcs.get_mut(&child).unwrap() -= 1;
        (kid, op_id) = self.duplicate_or_store_class(child, kid, op_id, visited, pending_stores, rcs, shapes);
        remove_first_output(&mut self.ekernels, kid, child);
        if rcs.get(&child).copied().unwrap_or(0) == 0 {
            visited.remove(&child);
        }
        let kernel = &mut self.ekernels[kid].kernel;
        let result_op = kernel.push_back(Op::Move { x: op_id, mop: Box::new(MoveOp::Reshape { shape }) });
        let n_consumers = rcs.get(&cid).copied().unwrap_or(0) as usize;
        for _ in 0..n_consumers {
            self.ekernels[kid].outputs.push(cid);
        }
        visited.insert(cid, (kid, result_op));
    }

    fn add_pad(
        &mut self,
        cid: ClassId,
        child: ClassId,
        padding: Vec<(i64, i64)>,
        visited: &mut Map<ClassId, (EKernelId, OpId)>,
        rcs: &mut Map<ClassId, u32>,
        pending_stores: &mut Set<ClassId>,
        shapes: &Slab<ShapeId, Vec<Dim>>,
    ) {
        let child_shape: Vec<Dim> = shapes[self.classes[child].shape].clone();
        let child_n: Dim = child_shape.iter().product();
        let cid_shape: Vec<Dim> = shapes[self.classes[cid].shape].clone();
        let pad_n: Dim = cid_shape.iter().product();
        let expands = pad_n > child_n;

        *rcs.get_mut(&child).unwrap() -= 1;
        let (mut kid, mut op_id);
        if expands {
            if let Some(&(ckid, cop_id)) = visited.get(&child) {
                self.add_store(child, ckid, cop_id, visited, pending_stores, rcs, shapes);
            }
            (kid, op_id) = self.child_to_kid(child, visited, shapes);
        } else {
            (kid, op_id) = self.child_to_kid(child, visited, shapes);
            (kid, op_id) = self.duplicate_or_store_class(child, kid, op_id, visited, pending_stores, rcs, shapes);
        }
        remove_first_output(&mut self.ekernels, kid, child);
        if rcs.get(&child).copied().unwrap_or(0) == 0 {
            visited.remove(&child);
        }
        let shape: Vec<Dim> = cid_shape;
        let kernel = &mut self.ekernels[kid].kernel;
        let result_op = kernel.push_back(Op::Move { x: op_id, mop: Box::new(MoveOp::Pad { padding, shape }) });
        let n_consumers = rcs.get(&cid).copied().unwrap_or(0) as usize;
        for _ in 0..n_consumers {
            self.ekernels[kid].outputs.push(cid);
        }
        visited.insert(cid, (kid, result_op));
    }
}

fn remove_first_output(kernels: &mut Slab<EKernelId, EKernelData>, kid: EKernelId, cid: ClassId) {
    if let Some(pos) = kernels[kid].outputs.iter().position(|&x| x == cid) {
        kernels[kid].outputs.remove(pos);
    }
}
