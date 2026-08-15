use std::collections::BTreeSet;

use crate::{
    Map, Set,
    graph::{ClassId, Graph, JitKernelData, JitKernelId, Node},
    kernel::{DeviceId, IDX_T, Kernel, MemLayout, MemScope, MoveOp, Op, OpId, ParamKind},
    runtime::ShapeId,
    shape::{Dim, UAxis},
    slab::{Slab, SlabId},
};

impl Graph {
    /// Fuses remaining non-Kernel nodes into kernels so every output has an all-Kernel path to leaves.
    ///
    /// The zyx graph is very granular — a single matmul produces dozens of structural nodes (Expand,
    /// Reduce, Permute, Cast, etc.). Materializing each as a separate kernel is impossible (e.g.,
    /// Expand to 2048×2048×2048 would OOM). [`kernelize`] uses eager fusion to batch structural
    /// nodes into larger kernels, ensuring the [`extract`](Graph::extract) invariant holds:
    ///
    /// > A path composed exclusively of [`Node::Kernel`] and [`Node::ToDevice`] nodes must exist
    /// > from leaves (realized classes) to every output.
    ///
    /// Not every class needs a kernel — only those that lie on output computation paths. Dead graph
    /// regions without kernels are harmless. After this function returns, all classes that *are* on
    /// output paths must be covered by a kernel.
    ///
    /// # Inputs
    ///
    /// Classes in `inputs` are treated as realized boundary values — the kernelizer
    /// never fuses *into* them, it only loads them (exactly like [`Node::Leaf`]s).
    /// For the whole graph these are the leaf classes; for a subregion (the gap
    /// between two AOT kernels) they are the region's boundary inputs.
    ///
    /// # Allowed set
    ///
    /// If `Some`, traversal is restricted to classes in `allowed` — classes outside
    /// it are never fused, even if they feed an output. [`fill_gaps`] uses this to
    /// kernelize each connected structural region in isolation; `None` allows the
    /// whole graph (leaf classes are always excluded via `inputs`).
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
    pub fn kernelize(
        &mut self,
        inputs: &Set<ClassId>,
        outputs: &BTreeSet<ClassId>,
        shapes: &Slab<ShapeId, Vec<Dim>>,
        allowed: Option<&Set<ClassId>>,
    ) {
        //self.debug(shapes);
        // A class can't be both a boundary input and a region output — that
        // would make a fused kernel load and store the same class.
        if cfg!(debug_assertions) {
            for cid in inputs {
                debug_assert!(
                    !outputs.contains(cid),
                    "class {cid:?} is both a kernelize input and output: inputs={inputs:?} outputs={outputs:?}"
                );
            }
        }

        let order = self.topo_sort_classes_without_kernels(inputs, outputs, allowed);

        let mut rcs: Map<ClassId, u32> = Map::default();
        for &cid in &order {
            // Boundary inputs are loaded, not fused — their structural nodes
            // (e.g. the matmul form of an AOT kernel output) must not count
            // children that live outside this region.
            if inputs.contains(&cid) {
                continue;
            }
            for nid in &self.classes[cid].nodes {
                // Kernel nodes added by pattern matching (e.g. cblas) are never
                // consumed here — kernelize only processes structural nodes.
                if matches!(&self.nodes[*nid].node, Node::Kernel { .. }) {
                    continue;
                }
                for child in self.nodes[*nid].node.class_params() {
                    *rcs.entry(child).or_default() += 1;
                }
            }
        }
        // User-requested outputs are consumers too — extraction needs a producer path for them.
        for &cid in outputs {
            *rcs.entry(cid).or_default() += 1;
        }

        let mut visited: Map<ClassId, (JitKernelId, OpId)> = Map::default();

        //println!("order={:?}", order);

        for (i, &cid) in order.iter().enumerate() {
            debug_assert!(!visited.contains_key(&cid), "class {cid:?} already visited");

            let nid = self.classes[cid].nodes[0];
            /*println!(
                "cid={} nid={} rc={} shape={:?}, {:?}, n_kernels={:?}",
                cid.0,
                nid.0,
                rcs[&cid],
                shapes[self.classes[cid].shape],
                self.nodes[nid].node,
                self.jit_kernels.len()
            );*/
            if inputs.contains(&cid) {
                // Boundary input: load the class from storage, same as a leaf.
                let (kid, op_id) = self.new_load_kernel(cid, rcs[&cid]);
                visited.insert(cid, (kid, op_id));
            } else {
                match self.nodes[nid].node {
                    Node::Leaf { .. } => {
                        let (kid, op_id) = self.new_load_kernel(cid, rcs[&cid]);
                        visited.insert(cid, (kid, op_id));
                    }
                    Node::Const(value) => {
                        let rc = *rcs.get(&cid).unwrap();
                        let mut kernel = Kernel::new(DeviceId::NULL);
                        kernel.push_back(Op::Const(value));
                        let op_id = kernel.head;
                        let kid = self.jit_kernels.push(JitKernelData {
                            kernel,
                            outputs: vec![cid; rc as usize],
                            loads: Vec::new(),
                            stores: Vec::new(),
                        });
                        visited.insert(cid, (kid, op_id));
                    }
                    Node::Unary { x, uop } => {
                        let (kid, op_id) = visited[&x];
                        self.consume(x, kid, &mut visited, &mut rcs);
                        let result_op = self.jit_kernels[kid].kernel.unary(op_id, uop);
                        self.push_outputs(kid, cid, rcs[&cid]);
                        visited.insert(cid, (kid, result_op));
                    }
                    Node::Cast { x, dtype } => {
                        let (kid, op_id) = visited[&x];
                        self.consume(x, kid, &mut visited, &mut rcs);
                        let result_op = self.jit_kernels[kid].kernel.cast(op_id, dtype);
                        self.push_outputs(kid, cid, rcs[&cid]);
                        visited.insert(cid, (kid, result_op));
                    }
                    Node::Binary { x, y, bop } => {
                        let (mut kid, mut op_id) = visited[&x];
                        let (mut kidy, mut op_idy) = visited[&y];

                        if kid != kidy {
                            let kid_stores = self.jit_kernels[kid].kernel.contains_stores();
                            let kidy_stores = self.jit_kernels[kidy].kernel.contains_stores();
                            match (kid_stores, kidy_stores) {
                                (true, true) => {
                                    (kid, op_id) = self.add_store(x, kid, op_id, &mut visited, &rcs);
                                    (kidy, _) = self.add_store(y, kidy, op_idy, &mut visited, &rcs);
                                }
                                (true, false) => (kid, op_id) = self.add_store(x, kid, op_id, &mut visited, &rcs),
                                (false, true) => (kidy, _) = self.add_store(y, kidy, op_idy, &mut visited, &rcs),
                                (false, false) => {}
                            }

                            self.merge_kernels(kidy, kid, &mut visited);
                            (_, op_idy) = visited[&y];
                        }

                        self.consume(x, kid, &mut visited, &mut rcs);
                        self.consume(y, kid, &mut visited, &mut rcs);
                        let result_op = self.jit_kernels[kid].kernel.binary(op_id, op_idy, bop);
                        self.push_outputs(kid, cid, rcs[&cid]);
                        visited.insert(cid, (kid, result_op));
                    }
                    Node::Reduce { x, rop: bop, ref axes } => {
                        let axes: Vec<UAxis> = axes.to_vec();
                        let n_axes: UAxis = axes.len() as UAxis;
                        let (mut kid, mut op_id) = visited[&x];
                        (kid, op_id) = self.duplicate_or_store_class(x, kid, op_id, &mut visited, &rcs, shapes, false);
                        self.consume(x, kid, &mut visited, &mut rcs);

                        // Permute so that reduce dimensions are last
                        let in_shape: Vec<Dim> = shapes[self.classes[x].shape].clone();
                        let kernel = &mut self.jit_kernels[kid].kernel;
                        let permuted = {
                            let n = in_shape.len();
                            let permute_axes: Vec<UAxis> =
                                (0..n as UAxis).filter(|&i| !axes.contains(&i)).chain(axes.iter().copied()).collect();
                            if permute_axes.iter().copied().ne(0..n as UAxis) {
                                kernel.permute(op_id, &permute_axes)
                            } else {
                                op_id
                            }
                        };

                        let mut result_op = kernel.push_back(Op::Reduce { x: permuted, rop: bop, n_axes });

                        // reshape if only 1 function remains
                        if in_shape.len() == n_axes as usize {
                            result_op = kernel.reshape(result_op, &[1]);
                        }

                        self.push_outputs(kid, cid, rcs[&cid]);
                        visited.insert(cid, (kid, result_op));
                    }
                    Node::After { x, dep } => {
                        // dep (the assign) wrote the new value in-place into x's
                        // base leaf buffer; cid aliases that buffer. Consume dep
                        // and keep its store so extract sees the assign kernel as
                        // cid's producer (this also orders any reader of cid after
                        // the in-place write).
                        let (dep_kid, _) = visited[&dep];
                        self.consume(dep, dep_kid, &mut visited, &mut rcs);

                        // Structurally the After consumes x; the assign kernel now
                        // owns the in-place store, so x's output slot is dropped.
                        let (kid, _) = visited[&x];
                        self.consume(x, kid, &mut visited, &mut rcs);
                        if self.jit_kernels[kid].outputs.is_empty() && self.jit_kernels[kid].stores.is_empty() {
                            self.jit_kernels.remove(kid);
                        }

                        // Re-expose the post-assign buffer via a fresh load kernel
                        // instead of aliasing x's op into dep's kernel (which
                        // breaks across chained Afters).
                        self.jit_kernels[dep_kid].stores.retain(|&z| z != x);
                        self.jit_kernels[dep_kid].stores.push(cid);
                        let (new_kid, new_op) = self.new_load_kernel(cid, rcs[&cid]);
                        visited.insert(cid, (new_kid, new_op));
                    }
                    Node::Assign { dst, src } => {
                        let (kid, src_op) = visited[&src];
                        let (dst_kid, dst_op) = visited[&dst];

                        assert_ne!(kid, dst_kid, "assign: src and dst must not share kernel {kid:?}");
                        let dst_leaf = self.jit_kernels[dst_kid].loads[0];
                        assert!(
                            !self.jit_kernels[kid].loads.contains(&dst_leaf),
                            "assign: src kernel loads dst tensor, not allowed to avoid data races"
                        );
                        // The assign's dst is normally consumed only by the
                        // assign itself. An After node re-exposes the post-assign
                        // version of dst, so each `After { x: dst, .. }` is a
                        // legitimate extra consumer.
                        let n_after_dst: usize = order
                            .iter()
                            .map(|&c| {
                                self.classes[c]
                                    .nodes
                                    .iter()
                                    .filter(|&&nid| matches!(&self.nodes[nid].node, Node::After { x, .. } if *x == dst))
                                    .count()
                            })
                            .sum();
                        let expected_rcs = 1 + n_after_dst + outputs.contains(&dst) as usize;
                        assert_eq!(
                            rcs[&dst], expected_rcs as u32,
                            "assign: dst class {dst:?} must be consumed only by the assign and its After(s) \
                             (rcs={}, expected {expected_rcs})",
                            rcs[&dst]
                        );

                        // Remove dst's movement-only kernel; its base buffer is
                        // dst_leaf's. The assign store reuses that buffer in-place.
                        let JitKernelData { kernel: dst_kernel, loads, stores, outputs } =
                            unsafe { self.jit_kernels.remove_and_return(dst_kid) };
                        debug_assert!(stores.is_empty());
                        debug_assert!(outputs.iter().all(|&x| x == dst));

                        // Backtrace to dst's base define.
                        let mut dst_define = dst_op;
                        for _ in 0..100 {
                            match dst_kernel.ops[dst_define].op {
                                Op::Move { x, .. } => dst_define = x,
                                Op::Storage { .. } => break,
                                _ => {}
                            }
                        }

                        // Replay dst's movement chain into src's kernel. The
                        // replayed base define becomes the mutable (ro=false)
                        // store target; the last replayed move yields dst's final
                        // position.
                        let mut op_map: Map<OpId, OpId> = Map::default();
                        let mut op_id = dst_kernel.head;
                        while !op_id.is_null() {
                            match dst_kernel.ops[op_id].op {
                                Op::Const(value) => {
                                    let id = self.jit_kernels[kid].kernel.push_back(Op::Const(value));
                                    op_map.insert(op_id, id);
                                }
                                Op::Param { dtype, ref mut kind } => {
                                    if op_id == dst_define {
                                        *kind = ParamKind::GlobalMut;
                                    }
                                    let id = self.jit_kernels[kid].kernel.push_back(Op::Param { dtype, kind: *kind });
                                    op_map.insert(op_id, id);
                                }
                                Op::Move { x, ref mop } => {
                                    let x = op_map.get(&x).copied().unwrap_or(op_map[&dst_define]);
                                    let mop = mop.remap(&op_map, dst_define);
                                    let id = self.jit_kernels[kid].kernel.push_back(Op::Move { x, mop });
                                    op_map.insert(op_id, id);
                                }
                                _ => unreachable!("assign: dst kernel must be movement-only"),
                            }
                            op_id = dst_kernel.next_op(op_id);
                        }

                        let dst_op = op_map.get(&dst_op).copied().unwrap_or(op_map[&dst_define]);
                        self.jit_kernels[kid].kernel.store(dst_op, src_op, OpId::NULL, MemLayout::Scalar);
                        self.jit_kernels[kid].stores.push(dst_leaf);
                        // Extra loads (variable defines like a narrow start, all
                        // but the base define) are replayed as new defines in
                        // src's kernel and must be passed at launch too.
                        self.jit_kernels[kid].loads.extend(loads.iter().skip(1).copied());

                        self.consume(src, kid, &mut visited, &mut rcs);
                        *rcs.get_mut(&dst).unwrap() -= 1;
                        if rcs[&dst] > 0 {
                            // The After(s) still consume dst: after the in-place
                            // store, dst's value lives in the (replayed) base
                            // define inside src's kernel. Point the remaining
                            // consumers at it — the assign already removed dst's
                            // old kernel, so the After arm cannot resolve it
                            // otherwise.
                            visited.insert(dst, (kid, dst_op));
                        } else {
                            visited.remove(&dst);
                        }

                        self.push_outputs(kid, src, rcs[&src]);
                        if rcs[&src] > 0 {
                            visited.insert(src, (kid, op_id));
                        }

                        self.push_outputs(kid, cid, rcs[&cid]);
                        if rcs[&cid] > 0 {
                            visited.insert(cid, (kid, op_id));
                        }

                        /*println!("\ncid={cid:?} src={src:?} dst={dst:?}, n_kernels={:?}", self.jit_kernels.len());
                        println!("outputs={:?}", self.jit_kernels[kid].outputs);
                        println!("loads={:?}", self.jit_kernels[kid].loads);
                        println!("stores={:?}", self.jit_kernels[kid].stores);
                        self.jit_kernels[kid].kernel.debug();*/
                    }
                    Node::Expand { x, .. } => {
                        let (kid, op_id) = visited[&x];
                        let force_store = self.jit_kernels[kid].kernel.is_preceded_by_compute(op_id);
                        let shape = shapes[self.classes[cid].shape].clone();
                        self.add_move(cid, x, MoveOp::Expand { shape }, force_store, &mut visited, &mut rcs, shapes);
                    }
                    Node::Permute { x, ref axes } => {
                        let shape = shapes[self.classes[cid].shape].clone();
                        self.add_move(
                            cid,
                            x,
                            MoveOp::Permute { axes: axes.to_vec(), shape },
                            false,
                            &mut visited,
                            &mut rcs,
                            shapes,
                        );
                    }
                    Node::Reshape { x, ref shape } => {
                        let shape = shape.clone();
                        self.add_move(cid, x, MoveOp::Reshape { shape }, false, &mut visited, &mut rcs, shapes);
                    }
                    Node::PadZeros { x, ref padding } => {
                        let (kid, op_id) = visited[&x];
                        let child_n: Dim = shapes[self.classes[x].shape].iter().product();
                        let shape = shapes[self.classes[cid].shape].clone();
                        // if shape after expand is larger than original and is compute kernel
                        let force_store =
                            shape.iter().product::<Dim>() > child_n && self.jit_kernels[kid].kernel.is_preceded_by_compute(op_id);
                        self.add_move(
                            cid,
                            x,
                            MoveOp::Pad { padding: padding.to_vec(), shape },
                            force_store,
                            &mut visited,
                            &mut rcs,
                            shapes,
                        );
                    }
                    Node::Narrow { x, axis, start, len } => {
                        // The start class is a leaf holding the crop offset (see
                        // runtime::narrow). Mirror the eager path: load it as a
                        // read-only variable define in this kernel, backed by the
                        // leaf's host buffer.
                        let start_cid = start;
                        let (mut kid, mut op_id) = visited[&x];
                        (kid, op_id) = self.duplicate_or_store_class(x, kid, op_id, &mut visited, &mut rcs, shapes, false);
                        self.consume(x, kid, &mut visited, &mut rcs);
                        let start_op = self.jit_kernels[kid].kernel.param(IDX_T, MemScope::Variable, true, &[1]);
                        self.jit_kernels[kid].loads.push(start_cid);
                        let result_op = self.jit_kernels[kid]
                            .kernel
                            .push_back(Op::Move { x: op_id, mop: Box::new(MoveOp::Narrow { axis, start: start_op, len }) });
                        self.push_outputs(kid, cid, *rcs.get(&cid).unwrap());
                        visited.insert(cid, (kid, result_op));
                    }
                    Node::Flip { x, ref axes } => {
                        self.add_move(cid, x, MoveOp::Flip { axes: axes.to_vec() }, false, &mut visited, &mut rcs, shapes);
                    }
                    Node::ToDevice { x, .. } => {
                        let (kid, op_id) = visited[&x];
                        self.consume(x, kid, &mut visited, &mut rcs);
                        let (kid, op_id) = self.add_store(x, kid, op_id, &mut visited, &rcs, shapes);
                        visited.insert(cid, (kid, op_id));
                    }
                    Node::Contiguous { x } => {
                        let (kid, op_id) = visited[&x];
                        self.consume(x, kid, &mut visited, &mut rcs);
                        let (kid, op_id) = self.add_store(x, kid, op_id, &mut visited, &rcs, shapes);
                        visited.insert(cid, (kid, op_id));
                    }
                    Node::Kernel { .. } => {}
                }
            }

            // AOT kernel classes (e.g. a cblas matmul output) are computed by a
            // backend kernel, not by this fused kernel. Materialize the class into
            // storage and hand off to a fresh load kernel, so downstream ops (e.g.
            // relu) start from the stored class instead of fusing into this kernel.
            if !inputs.contains(&cid)
                && self.classes[cid].nodes.iter().any(|&nid| matches!(&self.nodes[nid].node, Node::Kernel { .. }))
            {
                let (kid, op_id) = visited[&cid];
                let _ = self.add_store(cid, kid, op_id, &mut visited, &rcs, shapes);
            }

            // Post-processing: store if final output
            if outputs.contains(&cid) {
                let (mut kid, op_id) = visited[&cid];
                // Assign classes are in-place aliases of dst's (leaf) buffer; the
                // kernelizer already recorded the in-place store, so do not add a
                // fresh-buffer store. AOT kernel classes are already materialized
                // into storage by the backend kernel — storing the load kernel
                // again would produce a self-copying kernel.
                if !self.classes[cid]
                    .nodes
                    .iter()
                    .any(|&nid| matches!(&self.nodes[nid].node, Node::After { .. } | Node::Kernel { .. }))
                {
                    (kid, _) = self.add_store(cid, kid, op_id, &mut visited, &rcs, shapes);
                }
                *rcs.get_mut(&cid).unwrap() -= 1;
                remove_first_output(&mut self.jit_kernels, kid, cid);
                if rcs[&cid] == 0 {
                    visited.remove(&cid);
                }
                if self.jit_kernels[kid].outputs.is_empty() && self.jit_kernels[kid].stores.is_empty() {
                    self.jit_kernels.remove(kid);
                }
            }

            if cfg!(debug_assertions) {
                for kid in self.jit_kernels.ids() {
                    let kernel = &self.jit_kernels[kid];
                    // A kernel must never load a class it also stores — that would
                    // create a self-referential producer path and break extract.
                    for load in &kernel.loads {
                        debug_assert!(
                            !kernel.stores.contains(load),
                            "kernel {kid:?} loads and stores class {load:?}: loads={:?} stores={:?}",
                            kernel.loads,
                            kernel.stores,
                        );
                    }
                }
            }

            if cfg!(debug_assertions) {
                for ek in self.jit_kernels.values() {
                    let mut counts: Map<ClassId, u32> = Map::default();
                    for &ocid in &ek.outputs {
                        *counts.entry(ocid).or_default() += 1;
                    }
                    if !counts.is_empty() && counts.iter().any(|(c, &n)| *rcs.get(c).unwrap() != n) {
                        println!("outputs={:?}, counts={counts:?}", ek.outputs);
                        for (c, n) in counts.iter() {
                            println!("class={c:?}, rcs={}, n={n}", rcs[c]);
                        }
                        ek.kernel.debug();
                        panic!("output != rcs");
                    }
                }
                for c in &order[..=i] {
                    if let Some(&rc) = rcs.get(c) {
                        if rc == 0 {
                            if visited.contains_key(c) {
                                panic!("class={c:?} with rcs=0 in visited");
                            }
                        } else {
                            if !visited.contains_key(c) {
                                panic!("class={c:?} with rcs>0 not in visited");
                            }
                        }
                    }
                }
            }
        }

        if cfg!(debug_assertions) {
            for (c, &r) in rcs.iter() {
                if r != 0 {
                    eprintln!("leaked rc: class={c:?} rc={r} inputs={inputs:?}");
                }
            }
            if rcs.values().any(|&r| r != 0) {
                self.debug();
            }
            debug_assert!(rcs.values().all(|&r| r == 0), "all rcs must be zero");
            debug_assert!(visited.is_empty(), "visited must be empty");
            for kid in self.jit_kernels.ids() {
                let kernel = &self.jit_kernels[kid];
                debug_assert!(kernel.outputs.is_empty());
                if kernel.stores.is_empty() {
                    kernel.kernel.debug();
                    panic!("encountered kernel without stores");
                }
                // A kernel must never load a class it also stores — that would
                // create a self-referential producer path and break extract.
                for load in &kernel.loads {
                    debug_assert!(
                        !kernel.stores.contains(load),
                        "kernel {kid:?} loads and stores class {load:?}: loads={:?} stores={:?}",
                        kernel.loads,
                        kernel.stores,
                    );
                }
            }
        }

        /*for kernel in self.jit_kernels.values() {
            println!("loads={:?}", kernel.loads);
            println!("stores={:?}", kernel.stores);
            kernel.kernel.debug();
        }
        panic!();*/
    }

    fn new_load_kernel(&mut self, cid: ClassId, rc: u32) -> (JitKernelId, OpId) {
        let mut kernel = Kernel::new(DeviceId::NULL);
        let op_id = kernel.param(self.classes[cid].dtype, ParamKind::Global);
        let kid = self.jit_kernels.push(JitKernelData {
            kernel,
            outputs: vec![cid; rc as usize],
            loads: vec![cid],
            stores: Vec::new(),
        });
        (kid, op_id)
    }

    #[must_use]
    fn add_store(
        &mut self,
        cid: ClassId,
        kid: JitKernelId,
        op_id: OpId,
        visited: &mut Map<ClassId, (JitKernelId, OpId)>,
        rcs: &Map<ClassId, u32>,
    ) -> (JitKernelId, OpId) {
        //println!("add store cid={cid:?} kid={kid:?} op_id={op_id:?} rc={}", rcs.get(&cid).unwrap());
        //println!("outputs={:?}", self.ekernels[kid].outputs);

        if !self.jit_kernels[kid].loads.contains(&cid) {
            let dst = self.jit_kernels[kid].kernel.param(dtype, ParamKind::Global);
            self.jit_kernels[kid].kernel.store(dst, op_id, OpId::NULL, MemLayout::Scalar);
            self.jit_kernels[kid].stores.push(cid);
            visited.remove(&cid);
        }

        // Remove all occurences of x
        let outputs = &mut self.jit_kernels[kid].outputs;
        debug_assert_eq!(rcs[&cid], outputs.iter().filter(|&&x| x == cid).count() as u32);
        outputs.retain(|&x| x != cid);

        if let Some(rc) = rcs.get(&cid).copied()
            && rc > 0
        {
            let (new_kid, new_op) = self.new_load_kernel(cid, rc);
            visited.insert(cid, (new_kid, new_op));
            (new_kid, new_op)
        } else {
            (kid, op_id)
        }
    }

    fn merge_kernels(&mut self, src: JitKernelId, dst: JitKernelId, visited: &mut Map<ClassId, (JitKernelId, OpId)>) {
        let JitKernelData { kernel: src_kernel, outputs, loads, stores } = unsafe { self.jit_kernels.remove_and_return(src) };

        {
            let dst_data = &mut self.jit_kernels[dst];
            dst_data.outputs.extend(outputs);
            dst_data.loads.extend(loads);
            dst_data.stores.extend(stores);
        }

        let mut op_map: Map<OpId, OpId> = Map::default();
        let mut i = src_kernel.head;
        while !i.is_null() {
            let mut op = src_kernel.ops[i].op.clone();
            for param in op.parameters_mut() {
                if !param.is_null()
                    && let Some(&new_param) = op_map.get(param)
                {
                    *param = new_param;
                }
            }
            let new_id = self.jit_kernels[dst].kernel.push_back(op);
            op_map.insert(i, new_id);
            i = src_kernel.ops[i].next;
        }

        for (kid, op_id) in visited.values_mut() {
            if *kid == src {
                *kid = dst;
                if let Some(&new_op) = op_map.get(op_id) {
                    *op_id = new_op;
                }
            }
        }
    }

    /// Duplicate or store is heuristics that checks if it's better to store and create new load kernel
    /// or duplicate the original kernel. If force_store is set to true, it always stores.
    /// The original kernel is left with one fewer child class in it's outputs.
    /// The new kernel contains this one child class and the new kernel is guaranteed to have only one output.
    /// This is called by functions that HAVE TO have only 1 output, because they are movement or reduce.
    /// Movement or reduce change the view of the load, that's why they require that the load is duplicated.
    #[allow(clippy::too_many_arguments)] // graph kernel API, arguments are structural parameters
    fn duplicate_or_store_class(
        &mut self,
        child: ClassId,
        mut kid: JitKernelId,
        mut op_id: OpId,
        visited: &mut Map<ClassId, (JitKernelId, OpId)>,
        rcs: &Map<ClassId, u32>,
        shapes: &Slab<ShapeId, Vec<Dim>>,
        force_store: bool,
    ) -> (JitKernelId, OpId) {
        // if kernel has stores, store child and create fresh load kernel
        let force_store = force_store || self.jit_kernels[kid].kernel.contains_stores();

        // if kernel has multiple outputs, duplicate the kernel
        //println!("n_outputs={}", self.ekernels[kid].outputs.len());
        if self.jit_kernels[kid].outputs.len() > 1 || force_store {
            if force_store || self.jit_kernels[kid].kernel.is_preceded_by_reduce(op_id) {
                (kid, op_id) = self.add_store(child, kid, op_id, visited, rcs);

                // After storing, the new kernel can have more than one output. If it does, we have to split into another kernel
                debug_assert!(self.jit_kernels[kid].outputs.iter().all(|&x| x == child));
                if self.jit_kernels[kid].outputs.len() > 1 {
                    // Remove from the original kernel
                    remove_first_output(&mut self.jit_kernels, kid, child);
                    // Create another kernel with just one output
                    (kid, op_id) = self.new_load_kernel(child, 1);
                }
            } else {
                remove_first_output(&mut self.jit_kernels, kid, child);
                let out_op_ids: Vec<OpId> = self.jit_kernels[kid].outputs.iter().map(|&cid| visited[&cid].1).collect();
                let loads = self.jit_kernels[kid].loads.clone();
                let (new_kernel, new_op_id, self_loads, new_loads) =
                    self.jit_kernels[kid].kernel.extract_subkernel(op_id, &out_op_ids, &loads);
                self.jit_kernels[kid].loads = self_loads;

                debug_assert_eq!(self.jit_kernels[kid].outputs.iter().filter(|&&x| x == child).count(), rcs[&child] as usize - 1);

                let new_kid = self.jit_kernels.push(JitKernelData {
                    kernel: new_kernel,
                    outputs: vec![child],
                    loads: new_loads,
                    stores: Vec::new(),
                });
                op_id = new_op_id;
                kid = new_kid;
            }
        }

        debug_assert_eq!(self.jit_kernels[kid].outputs.len(), 1);

        (kid, op_id)
    }

    fn consume(
        &mut self,
        cid: ClassId,
        kid: JitKernelId,
        visited: &mut Map<ClassId, (JitKernelId, OpId)>,
        rcs: &mut Map<ClassId, u32>,
    ) {
        *rcs.get_mut(&cid).unwrap() -= 1;
        remove_first_output(&mut self.jit_kernels, kid, cid);
        if *rcs.get(&cid).unwrap() == 0 {
            visited.remove(&cid);
        }
    }

    fn push_outputs(&mut self, kid: JitKernelId, cid: ClassId, n: u32) {
        self.jit_kernels[kid].outputs.extend(std::iter::repeat_n(cid, n as usize));
    }

    #[allow(clippy::too_many_arguments)] // graph kernel API, arguments are structural parameters
    fn add_move(
        &mut self,
        cid: ClassId,
        child: ClassId,
        mop: MoveOp,
        force_store: bool,
        visited: &mut Map<ClassId, (JitKernelId, OpId)>,
        rcs: &mut Map<ClassId, u32>,
        shapes: &Slab<ShapeId, Vec<Dim>>,
    ) {
        let (mut kid, mut op_id) = visited[&child];
        (kid, op_id) = self.duplicate_or_store_class(child, kid, op_id, visited, rcs, shapes, force_store);
        self.consume(child, kid, visited, rcs);
        let kernel = &mut self.jit_kernels[kid].kernel;
        let result_op = kernel.push_back(Op::Move { x: op_id, mop: Box::new(mop) });
        self.push_outputs(kid, cid, *rcs.get(&cid).unwrap());
        visited.insert(cid, (kid, result_op));
    }

    /// Fills the gaps between AOT kernels with fused kernels.
    ///
    /// The classes in `active_outputs` are outputs of AOT kernels that are in
    /// play for this pass. Together with the leaf classes they form producer
    /// boundaries: the kernelizer never fuses into them, it only loads them.
    /// Everything else on output paths decomposes into connected structural
    /// regions, each bounded by producer boundaries on the input side and by
    /// AOT kernel inputs / final outputs on the output side. Each region is
    /// kernelized independently, so the gaps between AOT kernels get filled
    /// while each AOT kernel keeps its own subgraph.
    pub fn fill_gaps(&mut self, active_outputs: &Set<ClassId>, outputs: &BTreeSet<ClassId>) {
        let mut producer_boundaries: Set<ClassId> = self.leaf_classes.iter().copied().collect();
        producer_boundaries.extend(active_outputs.iter().copied());

        // Classes consumed by active AOT kernels — region outputs that must be
        // stored so the backend kernel can read them.
        let mut kernel_inputs: Set<ClassId> = Set::default();
        for &cid in active_outputs {
            for nid in &self.classes[cid].nodes {
                if let Node::Kernel { inputs: kin, .. } = &self.nodes[*nid].node {
                    kernel_inputs.extend(kin.iter().copied());
                }
            }
        }

        let order = self.topo_sort_classes_without_kernels(&producer_boundaries, outputs, None);

        // Union-find the structural classes into connected regions.
        let structural: Vec<ClassId> = order.iter().copied().filter(|&c| !producer_boundaries.contains(&c)).collect();
        let idx: Map<ClassId, usize> = structural.iter().enumerate().map(|(i, &c)| (c, i)).collect();
        let mut parent: Vec<usize> = (0..structural.len()).collect();
        fn find(parent: &mut [usize], mut i: usize) -> usize {
            while parent[i] != i {
                parent[i] = parent[parent[i]];
                i = parent[i];
            }
            i
        }
        for (i, &cid) in structural.iter().enumerate() {
            for nid in &self.classes[cid].nodes {
                for p in self.nodes[*nid].node.class_params() {
                    if let Some(&j) = idx.get(&p) {
                        let (a, b) = (find(&mut parent, i), find(&mut parent, j));
                        parent[a.max(b)] = a.min(b);
                    }
                }
            }
        }
        let mut regions: Map<usize, Vec<ClassId>> = Map::default();
        for (i, &cid) in structural.iter().enumerate() {
            regions.entry(find(&mut parent, i)).or_default().push(cid);
        }

        for region_classes in regions.values() {
            let region: Set<ClassId> = region_classes.iter().copied().collect();

            let mut region_inputs: Set<ClassId> = Set::default();
            for &cid in &region {
                for nid in &self.classes[cid].nodes {
                    for p in self.nodes[*nid].node.class_params() {
                        if producer_boundaries.contains(&p) {
                            region_inputs.insert(p);
                        }
                    }
                }
            }

            let mut region_outputs: BTreeSet<ClassId> = BTreeSet::new();
            for &cid in &region {
                if outputs.contains(&cid) || kernel_inputs.contains(&cid) {
                    region_outputs.insert(cid);
                }
            }

            if region_outputs.is_empty() {
                continue;
            }
            let region_allowed: Set<ClassId> = region.union(&region_inputs).copied().collect();
            self.kernelize(&region_inputs, &region_outputs, shapes, Some(&region_allowed));
        }
    }
}

fn remove_first_output(kernels: &mut Slab<JitKernelId, JitKernelData>, kid: JitKernelId, cid: ClassId) {
    if let Some(pos) = kernels[kid].outputs.iter().position(|&x| x == cid) {
        kernels[kid].outputs.remove(pos);
    }
}
