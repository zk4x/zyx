use std::collections::BTreeSet;

use crate::{
    shape::Dim,
    Map, Set,
    graph::Constant,
    graph::{ClassId, Graph, JitKernelData, JitKernelId, Node},
    kernel::{DeviceId, IDX_T, Kernel, MemLayout, MoveOp, Op, OpId, ParamKind},
    shape::UAxis,
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
    /// 3. **Shape replay**: Shape-descriptor classes — the `shape` of `Reshape`/`Expand`, the
    ///    `lp`/`len` of `Pad`, the `start`/`len` of `Narrow` — are pure symbolic metadata and
    ///    are **never** kernelized. Their transitive subgraph (`Stack` elements, `Binary`/`Unary`/
    ///    `Cast` operands, `Const` leaves, IDX_T scalar `Leaf` dim-variables) is collected
    ///    before the refcount walk; those classes are excluded from `rcs`, skipped by the
    ///    kernelize loop, and replayed into each consumer on demand via
    ///    [`Graph::replay_shape_into_kernel`]. Replay panics on a non-symbolic node — shapes
    ///    are pure metadata and must never be computed by a kernel.
    /// 4. **Eager parity**: The narrow/assign/contiguous arms in this kernelizer mirror
    ///    `Runtime::narrow`/`assign`/`contiguous` exactly. The narrow arm requires the input
    ///    kernel to have empty `outputs` after the input is consumed (mirroring `Runtime::narrow`'s
    ///    "input into narrow must have empty outputs" check); the assign arm replays dst's
    ///    movement chain into src's kernel and uses an in-place store, then re-points dst's
    ///    remaining consumers at a fresh load kernel (the same contract `add_store` uses for
    ///    every other stored class).
    pub fn kernelize(&mut self, inputs: &Set<ClassId>, outputs: &BTreeSet<ClassId>, allowed: Option<&Set<ClassId>>) {
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

        let order = self.topo_sort_classes::<true>(inputs, outputs, allowed);

        let mut rcs: Map<ClassId, u32> = Map::default();
        // Shape descriptors (Reshape/Expand shape, Pad lp/len, Narrow
        // start/len) are purely symbolic metadata: their whole subgraph is
        // replayed into each consuming kernel (see
        // `Graph::replay_shape_into_kernel`) and never kernelized. Collect
        // them before the refcount walk so descriptor params leave no rc
        // entries and their classes never enter the kernelize loop.
        let mut shape_classes: Set<ClassId> = Set::default();
        let mut desc_stack: Vec<ClassId> = Vec::new();
        for &cid in &order {
            if inputs.contains(&cid) {
                continue;
            }
            for nid in &self.classes[cid].nodes {
                match &self.nodes[*nid].node {
                    Node::Reshape { shape, .. } | Node::Expand { shape, .. } => desc_stack.push(*shape),
                    Node::Pad { lp, len, .. } => {
                        desc_stack.push(*lp);
                        desc_stack.push(*len);
                    }
                    Node::Narrow { start, len, .. } => {
                        desc_stack.push(*start);
                        desc_stack.push(*len);
                    }
                    _ => {}
                }
            }
        }
        while let Some(c) = desc_stack.pop() {
            if !shape_classes.insert(c) {
                continue;
            }
            for nid in &self.classes[c].nodes {
                match &self.nodes[*nid].node {
                    Node::Stack { ops } => desc_stack.extend(ops.iter().copied()),
                    Node::Binary { x, y, .. } => {
                        desc_stack.push(*x);
                        desc_stack.push(*y);
                    }
                    Node::Cast { x, .. } | Node::Unary { x, .. } => desc_stack.push(*x),
                    Node::Const { .. } | Node::Leaf { .. } => {}
                    n => panic!("shape descriptor contains non-symbolic node {n:?}; shapes are purely symbolic"),
                }
            }
        }

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
                    if shape_classes.contains(&child) {
                        continue;
                    }
                    *rcs.entry(child).or_default() += 1;
                }
            }
        }
        // User-requested outputs are consumers too — extraction needs a producer path for them.
        for &cid in outputs {
            *rcs.entry(cid).or_default() += 1;
        }

        let mut visited: Map<ClassId, (JitKernelId, OpId)> = Map::default();

        for (i, &cid) in order.iter().enumerate() {
            debug_assert!(!visited.contains_key(&cid), "class {cid:?} already visited");

            // Pure shape-descriptor classes are replayed into consumers and
            // never kernelized.
            if shape_classes.contains(&cid) {
                continue;
            }

            let nid = self.classes[cid].nodes[0];

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
                    Node::Const { value, .. } => {
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
                    Node::Stack { ref ops } => {
                        // Copy the element list out of the node so the shared
                        // borrow of self.nodes ends before we mutate kernels.
                        let ops: Vec<ClassId> = ops.iter().copied().collect();
                        // Merge every element into the first element's kernel,
                        // mirroring `Runtime::stack`: fresh `(kid, op)` read
                        // per element; a source kernel with stores is stored
                        // and re-read, anything still foreign is merged in.
                        let (kid, _) = visited[&ops[0]];
                        // All inputs merge into one kernel with one shared gws
                        // grid derived from the elements' shapes — they must agree.
                        if cfg!(debug_assertions) {
                            let s0 = self.shape(ops[0]);
                            for &e in ops.iter().skip(1) {
                                debug_assert_eq!(
                                    self.shape(e),
                                    s0,
                                    "Stack inputs must have identical shapes: {s0:?} vs {:?}",
                                    self.shape(e)
                                );
                            }
                        }
                        let mut op_ids: Vec<OpId> = Vec::with_capacity(ops.len());
        for &elem in ops.iter() {
            let (mut ekid, mut eop) = visited[&elem];
            if ekid != kid {
                                if self.jit_kernels[ekid].kernel.contains_stores() {
                                    (ekid, eop) = self.add_store(elem, ekid, eop, &mut visited, &rcs);
                                }
                                if ekid != kid {
                                    self.merge_kernels(ekid, kid, &mut visited);
                                    (_, eop) = visited[&elem];
                                }
                            }
                            op_ids.push(eop);
                        }
                        for &elem in ops.iter() {
                            self.consume(elem, kid, &mut visited, &mut rcs);
                        }
                        let result_op = self.jit_kernels[kid].kernel.stack(&op_ids);
                        self.push_outputs(kid, cid, rcs[&cid]);
                        visited.insert(cid, (kid, result_op));
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
                        // NOTE: `Node::Binary` does NOT broadcast. Broadcasting is
                        // performed upstream by `Tensor::broadcast` / `Graph::push_binary_node`,
                        // so by the time a binary node reaches the kernelizer its
                        // two operands already have the same (broadcast-compatible)
                        // shape. The kernelizer must never attempt to broadcast here.
                        let (mut kid, mut op_id) = visited[&x];
                        let (mut kidy, mut op_idy) = visited[&y];

                        if kid != kidy {
                            // Two kernels whose inputs disagree on dynamism can
                            // never merge: one kernel runs on ONE global work
                            // grid, and a static grid length cannot drive a
                            // dynamic computation (or vice versa). Materialize
                            // the STATIC side; `add_store` returns a fresh load
                            // kernel which becomes the merge destination, so the
                            // result (whose broadcast shape is the dynamic one)
                            // stays inside the dynamic kernel.
                            // Kernel-level dynamism, straight from the kernel
                            // IR: a kernel is dynamic if ANY param's shape has
                            // an unresolved (zero) dim.
                            match (
                                self.jit_kernels[kid].kernel.shape(op_id).contains(&0),
                                self.jit_kernels[kidy].kernel.shape(op_idy).contains(&0),
                            ) {
                                (false, true) => {
                                    (kid, op_id) = self.add_store(x, kid, op_id, &mut visited, &rcs);
                                }
                                (true, false) => {
                                    (kidy, op_idy) = self.add_store(y, kidy, op_idy, &mut visited, &rcs);
                                }
                                (_, _) => {
                                    // Both operands share the same dynamism, so they
                                    // must already be broadcast-compatible (broadcasting
                                    // is performed upstream by `Tensor::broadcast` /
                                    // `Graph::push_binary_node`); the kernelizer's
                                    // `Node::Binary` does NOT broadcast — except that
                                    // scalars broadcast implicitly in kernel IR.
                                    let sx = self.jit_kernels[kid].kernel.shape(op_id);
                                    let sy = self.jit_kernels[kidy].kernel.shape(op_idy);
                                    debug_assert!(
                                        sx == sy || sx.is_empty() || sy.is_empty(),
                                        "binary operands {sx:?} vs {sy:?} are not broadcast-compatible"
                                    );
                                }
                            }

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

                            // Restore of the original Binary merge rule
                            // (commit 7786c15): the reduce kernel must be the
                            // merge DESTINATION — a reduce's output grid is
                            // smaller than its input's, so pulling plain
                            // compute ops into it is safe, while merging a
                            // reduce into a plain compute kernel corrupts the
                            // gws. If x's kernel reduces and y's does not,
                            // swap the two slots so `merge_kernels` below pulls
                            // the non-reduce side into the reduce kernel.
                            // Operand order is preserved: both op ids are
                            // re-read from `visited` after the merge.
                            if self.jit_kernels[kid].kernel.is_reduce() && !self.jit_kernels[kidy].kernel.is_reduce() {
                                std::mem::swap(&mut kid, &mut kidy);
                                std::mem::swap(&mut op_id, &mut op_idy);
                            }

                            self.merge_kernels(kidy, kid, &mut visited);
                            (kid, op_idy) = visited[&y];
                            op_id = visited[&x].1;
                        }

                        self.consume(x, kid, &mut visited, &mut rcs);
                        self.consume(y, kid, &mut visited, &mut rcs);
                        let result_op = self.jit_kernels[kid].kernel.binary(op_id, op_idy, bop);
                        self.push_outputs(kid, cid, rcs[&cid]);
                        visited.insert(cid, (kid, result_op));
                    }
                    Node::Reduce { x, rop, ref axes } => {
                        // Assumed unique (backed by debug_assert) and in range
                        // (asserted by push_node).
                        debug_assert!(
                            axes.iter().collect::<BTreeSet<_>>().len() == axes.len(),
                            "Reduce: duplicate axes {axes:?}"
                        );
                        let axes: Vec<UAxis> = axes.to_vec();
                        let rank = self.shape(x).len();
                        let (mut kid, mut op_id) = visited[&x];
                        (kid, op_id) = self.duplicate_or_store_class(x, kid, op_id, &mut visited, &mut rcs, false);
                        // Single permute: non-reduced axes first, reduced axes
                        // trailing (order preserved), so each reduce in the
                        // sequence below sees its axis last.
                        let perm: Vec<UAxis> = (0..rank).filter(|i| !axes.contains(i)).chain(axes.iter().copied()).collect();
                        if !perm.iter().copied().eq(0..rank) {
                            let kernel = &mut self.jit_kernels[kid].kernel;
                            op_id = kernel.push_back(Op::Move { x: op_id, mop: Box::new(MoveOp::Permute { axes: perm.into() }) });
                        }
                        // Sequence of single trailing-axis reduces.
                        for _ in 0..axes.len() {
                            let kernel = &mut self.jit_kernels[kid].kernel;
                            let dims = kernel.shape_ids(op_id);
                            debug_assert!(!dims.is_empty(), "reduce of scalar");
                            let reduce_axis = *dims.last().unwrap();
                            op_id = kernel.push_back(Op::Reduce { x: op_id, rop, reduce_axis });
                        }
                        // All dims reduced: reshape the scalar to [1].
                        if axes.len() == rank {
                            let kernel = &mut self.jit_kernels[kid].kernel;
                            let shape_op = kernel.add_shape(&[1]);
                            op_id = kernel.push_back(Op::Move { x: op_id, mop: Box::new(MoveOp::Reshape { shape: shape_op }) });
                        }
                        self.consume(x, kid, &mut visited, &mut rcs);
                        self.push_outputs(kid, cid, rcs[&cid]);
                        visited.insert(cid, (kid, op_id));
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
                        // The dst kernel's loads mix the owning buffer with
                        // dim-variable classes. Exactly ONE buffer entry may
                        // exist — trace it instead of assuming a position,
                        // fail loud otherwise.
                        let dst_loads = self.jit_kernels[dst_kid].loads.clone();
                        let is_var_class =
                            |g: &Self, c: ClassId| matches!(&g.nodes[g.classes[c].nodes[0]].node, Node::Leaf { dtype, shape, .. } if *dtype == IDX_T && shape.is_null());
                        let mut buffer_classes = dst_loads.iter().copied().filter(|&c| !is_var_class(self, c));
                        let dst_leaf = match (buffer_classes.next(), buffer_classes.next()) {
                            (Some(c), None) => c,
                            found => panic!("assign: dst kernel must contain exactly one buffer load, got {:?}", found.0),
                        };
                        for &c in &dst_loads {
                            assert!(
                                c == dst_leaf || is_var_class(self, c),
                                "assign: dst kernel load class {c:?} is neither the buffer nor a dim variable"
                            );
                        }
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
                        let JitKernelData { kernel: dst_kernel, stores, outputs, .. } =
                            unsafe { self.jit_kernels.remove_and_return(dst_kid) };
                        debug_assert!(stores.is_empty());

                        // Backtrace to dst's base param.
                        let mut dst_param = dst_op;
                        for _ in 0..100 {
                            match dst_kernel.ops[dst_param].op {
                                Op::Move { x, .. } => dst_param = x,
                                Op::Storage { .. } => break,
                                _ => {}
                            }
                        }

                        // Replay dst's movement chain into src's kernel. The
                        // replayed base param becomes the mutable (GlobalMut)
                        // store target; the last replayed move yields dst's final
                        // position. Every replayed define keeps its load class,
                        // aligned in define order (positional args law).
                        let mut op_map: Map<OpId, OpId> = Map::default();
                        let mut new_def_loads: Vec<ClassId> = Vec::new();
                        let mut def_i = 0usize;
                        let mut op_id = dst_kernel.head;
                        while !op_id.is_null() {
                            match dst_kernel.ops[op_id].op {
                                Op::Const(value) => {
                                    let id = self.jit_kernels[kid].kernel.push_back(Op::Const(value));
                                    op_map.insert(op_id, id);
                                }
                                Op::Param { dtype, mut kind, shape } => {
                                    if op_id == dst_param {
                                        kind = ParamKind::GlobalMut;
                                    }
                                    assert!(
                                        matches!(kind, ParamKind::GlobalMut | ParamKind::Variable),
                                        "assign: unexpected param kind {kind:?} in dst movement kernel"
                                    );
                                    let id = self.jit_kernels[kid].kernel.push_back(Op::Param { dtype, kind, shape });
                                    // Assign turns dst's base from a load into a
                                    // PURE STORE: it must NOT register in loads —
                                    // its buffer slot comes via `stores` instead.
                                    if kind == ParamKind::Variable {
                                        new_def_loads.push(dst_loads[def_i]);
                                    }
                                    def_i += 1;
                                    op_map.insert(op_id, id);
                                }
                                Op::Move { x, ref mop } => {
                                    let x = op_map.get(&x).copied().unwrap_or(op_map[&dst_param]);
                                    let mop = mop.remap(&op_map);
                                    let id = self.jit_kernels[kid].kernel.push_back(Op::Move { x, mop });
                                    op_map.insert(op_id, id);
                                }
                                Op::Stack { ref ops } => {
                                    let mapped: Box<[OpId]> = ops.iter().map(|&o| op_map[&o]).collect();
                                    let id = self.jit_kernels[kid].kernel.push_back(Op::Stack { ops: mapped });
                                    op_map.insert(op_id, id);
                                }
                                _ => unreachable!("assign: dst kernel must be movement-only, got {:?}", dst_kernel.ops[op_id].op),
                            }
                            op_id = dst_kernel.next_op(op_id);
                        }

                        // Classes still living in the removed dst kernel (e.g.
                        // shape consts merged into the movement chain) move to
                        // the replayed kernel: remap visited and carry their
                        // outstanding output entries over, keeping rc balanced.
                        for (&vclass, (vkid, vop)) in visited.iter_mut() {
                            if *vkid == dst_kid && vclass != dst {
                                let count = outputs.iter().filter(|&&c| c == vclass).count() as u32;
                                self.jit_kernels[kid].outputs.extend(std::iter::repeat_n(vclass, count as usize));
                                *vkid = kid;
                                if let Some(&new_op) = op_map.get(vop) {
                                    *vop = new_op;
                                }
                            }
                        }

                        let dst_op = op_map.get(&dst_op).copied().unwrap_or(op_map[&dst_param]);
                        self.jit_kernels[kid].kernel.store(dst_op, src_op, OpId::NULL, MemLayout::Scalar);
                        self.jit_kernels[kid].stores.push(dst_leaf);
                        // Register every replayed define's load class in define
                        // order (variables and the GlobalMut base buffer alike)
                        // — positional args law.
                        self.jit_kernels[kid].loads.extend(new_def_loads);

                        self.consume(src, kid, &mut visited, &mut rcs);
                        *rcs.get_mut(&dst).unwrap() -= 1;
                        if rcs[&dst] > 0 {
                            // The After(s) still consume dst: after the in-place
                            // store, dst's value lives in the (replayed) base
                            // param inside src's kernel. Remaining consumers load
                            // it from a fresh loader kernel — the same contract
                            // `add_store` uses for every other stored class
                            // (pointing them at the in-kernel op instead breaks
                            // any later consumer whose force_store flushes).
                            let (new_kid, new_op) = self.new_load_kernel(dst, rcs[&dst]);
                            visited.insert(dst, (new_kid, new_op));
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
                    Node::Expand { x, shape } => {
                        // Dtypes are fully static: every dim of the result
                        // must be integer-typed.
                        if cfg!(debug_assertions) {
                            for dim in self.shape(cid) {
                                let dt = self.dtype(dim);
                                debug_assert!(dt.is_int(), "Expand {cid:?} has non-integer dim dtype {dt:?}");
                            }
                        }
                        let (mut kid, mut op_id) = visited[&x];
                        let force_store = self.jit_kernels[kid].kernel.is_preceded_by_compute(op_id);
                        (kid, op_id) = self.duplicate_or_store_class(x, kid, op_id, &mut visited, &mut rcs, force_store);
                        // The shape descriptor is pure metadata — replay its
                        // symbolic expression directly into this kernel.
                        let sop = self.replay_shape_into_kernel(kid, shape);
                        self.consume(x, kid, &mut visited, &mut rcs);
                        let result_op = self.jit_kernels[kid]
                            .kernel
                            .push_back(Op::Move { x: op_id, mop: Box::new(MoveOp::Expand { shape: sop }) });
                        self.push_outputs(kid, cid, *rcs.get(&cid).unwrap());
                        visited.insert(cid, (kid, result_op));
                    }
                    Node::Permute { x, ref axes } => {
                        self.add_move(cid, x, MoveOp::Permute { axes: axes.clone() }, false, &mut visited, &mut rcs);
                    }
                    Node::Reshape { x, shape } => {
                        // Dtypes are fully static: every dim of the result
                        // must be integer-typed.
                        if cfg!(debug_assertions) {
                            for dim in self.shape(cid) {
                                let dt = self.dtype(dim);
                                debug_assert!(dt.is_int(), "Reshape {cid:?} has non-integer dim dtype {dt:?}");
                            }
                        }
                        let (mut kid, mut op_id) = visited[&x];
                        let force_store = false;
                        (kid, op_id) = self.duplicate_or_store_class(x, kid, op_id, &mut visited, &mut rcs, force_store);
                        // The shape descriptor is pure metadata — replay its
                        // symbolic expression directly into this kernel.
                        let sop = self.replay_shape_into_kernel(kid, shape);
                        self.consume(x, kid, &mut visited, &mut rcs);
                        let result_op = self.jit_kernels[kid]
                            .kernel
                            .push_back(Op::Move { x: op_id, mop: Box::new(MoveOp::Reshape { shape: sop }) });
                        self.push_outputs(kid, cid, *rcs.get(&cid).unwrap());
                        visited.insert(cid, (kid, result_op));
                    }
                    Node::Pad { x, axis, lp, len } => {
                        let (mut kid, mut op_id) = visited[&x];
                        let force_store = false;
                        (kid, op_id) = self.duplicate_or_store_class(x, kid, op_id, &mut visited, &mut rcs, force_store);
                        // Bounds are pure metadata — replay their symbolic
                        // expressions directly into this kernel.
                        let lp_op = self.replay_shape_into_kernel(kid, lp);
                        let len_op = self.replay_shape_into_kernel(kid, len);
                        self.consume(x, kid, &mut visited, &mut rcs);
                        let result_op = self.jit_kernels[kid].kernel.push_back(Op::Move {
                            x: op_id,
                            mop: Box::new(MoveOp::Pad { axis, lp: lp_op, len: len_op }),
                        });
                        self.push_outputs(kid, cid, *rcs.get(&cid).unwrap());
                        visited.insert(cid, (kid, result_op));
                    }
                    Node::Narrow { x, axis, start, len } => {
                        let (mut kid, mut op_id) = visited[&x];
                        let force_store = false;
                        (kid, op_id) = self.duplicate_or_store_class(x, kid, op_id, &mut visited, &mut rcs, force_store);
                        // Bounds are pure metadata — replay their symbolic
                        // expressions directly into this kernel.
                        let start_op = self.replay_shape_into_kernel(kid, start);
                        let len_op = self.replay_shape_into_kernel(kid, len);
                        self.consume(x, kid, &mut visited, &mut rcs);
                        // Eager parity: `runtime::narrow` requires the input to
                        // sit alone in a store-free kernel before the bound
                        // kernels merge ("input into narrow must have empty
                        // outputs"). `duplicate_or_store_class` + `consume`
                        // guarantee exactly that here.
                        debug_assert!(
                            self.jit_kernels[kid].outputs.is_empty(),
                            "narrow: input kernel must have empty outputs before the narrow merges (eager parity)"
                        );
                        let result_op = self.jit_kernels[kid].kernel.push_back(Op::Move {
                            x: op_id,
                            mop: Box::new(MoveOp::Narrow { axis, start: start_op, len: len_op }),
                        });
                        self.push_outputs(kid, cid, *rcs.get(&cid).unwrap());
                        visited.insert(cid, (kid, result_op));
                    }
                    Node::Flip { x, ref axes } => {
                        self.add_move(cid, x, MoveOp::Flip { axes: axes.clone() }, false, &mut visited, &mut rcs);
                    }
                    Node::ToDevice { x, .. } => {
                        let (kid, op_id) = visited[&x];
                        self.consume(x, kid, &mut visited, &mut rcs);
                        let (kid, op_id) = self.add_store(x, kid, op_id, &mut visited, &rcs);
                        visited.insert(cid, (kid, op_id));
                    }
                    Node::Contiguous { x } => {
                        let (kid, op_id) = visited[&x];
                        self.consume(x, kid, &mut visited, &mut rcs);
                        // Cast-shim semantics (mirrors eager runtime::contiguous):
                        // a same-dtype Cast (value identity) becomes the stored
                        // class's own op, so `cid` gets a distinct op and its own
                        // backing buffer instead of aliasing x's load op.
                        let dtype = self.dtype(cid);
                        let cast_op = self.jit_kernels[kid].kernel.cast(op_id, dtype);
                        let rc = rcs.get(&cid).copied().unwrap_or_else(|| panic!("contiguous: class {cid:?} has no rc entry"));
                        self.jit_kernels[kid].outputs.extend(std::iter::repeat_n(cid, rc as usize));
                        let (kid, op_id) = self.add_store(cid, kid, cast_op, &mut visited, &mut rcs);
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
                let _ = self.add_store(cid, kid, op_id, &mut visited, &rcs);
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
                    (kid, _) = self.add_store(cid, kid, op_id, &mut visited, &rcs);
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
            for kid in self.jit_kernels.ids().collect::<Vec<_>>() {
                let kernel = &self.jit_kernels[kid];
                // Fully-consumed pure value kernels (e.g. private const
                // kernels) are dead: nothing references them.
                if kernel.outputs.is_empty() && kernel.loads.is_empty() && kernel.stores.is_empty() {
                    self.jit_kernels.remove(kid);
                    continue;
                }
                debug_assert!(kernel.outputs.is_empty());
                if kernel.stores.is_empty() {
                    eprintln!("DEBUG kernel {kid:?} without stores: outputs={:?} loads={:?}", kernel.outputs, kernel.loads);
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
                // Invariant: `loads` is parallel to the Global/Variable Param
                // ops in head order (extract_subkernel and launch both rely on
                // this).
                let mut n_params = 0;
                let mut oid = kernel.kernel.head;
                for _ in 0..10_000 {
                    if oid.is_null() {
                        break;
                    }
                    if matches!(kernel.kernel.at(oid), Op::Param { kind: ParamKind::Global | ParamKind::Variable, .. }) {
                        n_params += 1;
                    }
                    oid = kernel.kernel.next_op(oid);
                }
                assert!(!oid.is_null() || true);
                if n_params != kernel.loads.len() {
                    panic!(
                        "DEBUG kernelize invariant broken: kernel {kid:?} has {n_params} Global/Variable params but {} loads entries. stores={:?} outputs={:?}",
                        kernel.loads.len(),
                        kernel.stores,
                        kernel.outputs,
                    );
                }
                // Every load class must be produced (stored) by some kernel, or
                // be a graph input / leaf.
                for load in &kernel.loads {
                    let stored = self.jit_kernels.values().any(|k| k.stores.contains(load));
                    let in_outputs = self.jit_kernels.values().any(|k| k.outputs.contains(load));
                    let is_input =
                        inputs.contains(load) || matches!(self.nodes[self.classes[*load].nodes[0]].node, Node::Leaf { .. });
                    if !stored && !is_input {
                        panic!(
                            "DEBUG kernelize: load class {load:?} (node {:?}) of kernel {kid:?} is not stored anywhere (in_outputs={in_outputs}) and is not an input",
                            self.nodes[self.classes[*load].nodes[0]].node
                        );
                    }
                }
            }
        }

        /*for kernel in self.jit_kernels.values() {
            println!("loads={:?}", kernel.loads);
            println!("stores={:?}", kernel.stores);
            kernel.kernel.debug();
        }
        panic!();*/

        self.verify();
    }

    /// Creates a fresh **load kernel** for class `cid` that re-exposes its stored value to
    /// `rc` remaining consumers.
    ///
    /// The load kernel holds a single `Param(Global)` (its only load) whose shape is replayed
    /// symbolically from the egraph (see [`Graph::replay_symbolic_into_kernel`]). Its `outputs`
    /// list contains exactly `rc` copies of `cid` — one per remaining consumer; each
    /// `consume(cid, kid, ...)` will pop one. This is the canonical "class was stored, point
    /// remaining consumers at a fresh loader" contract used by [`add_store`] and the assign
    /// arm's post-in-place-store handling — the inverse of placement, so consumers never
    /// re-enter a kernel whose `outputs` no longer contains the class.
    fn new_load_kernel(&mut self, cid: ClassId, rc: u32) -> (JitKernelId, OpId) {
        let kid = self.jit_kernels.push(JitKernelData {
            kernel: Kernel::new(DeviceId::NULL),
            outputs: Vec::new(),
            loads: Vec::new(),
            stores: Vec::new(),
        });
        // Shapes are purely symbolic metadata: they are replayed directly from
        // the egraph into this kernel via `replay_symbolic_into_kernel` — the
        // graph-side mirror of eager's `Runtime::replay_symbolic_into_kernel`.
        // Variables register in `loads` at mint time inside the replay, before
        // the buffer param below, so define order == loads order and the
        // positional args law holds. No constant folding, no anonymous
        // variables, no fallbacks.
        let dims = self.shape(cid);
        let shape = self.replay_symbolic_into_kernel(kid, &dims);
        let dtype = self.dtype(cid);
        let op_id = self.jit_kernels[kid].kernel.param(dtype, ParamKind::Global, shape);
        let data = &mut self.jit_kernels[kid];
        data.outputs = vec![cid; rc as usize];
        data.loads.push(cid);
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

        // If the kernel already consumes `cid` as a plain load and stores
        // nothing, there is nothing to materialize. Mirror zyx2: strip this
        // class's output slots and re-point `cid` at a fresh loader so the
        // consumer boundary is preserved (prevents over-fusion) without
        // abandoning the old loader.
        if self.jit_kernels[kid].loads.contains(&cid) && !self.jit_kernels[kid].kernel.contains_stores() {
            self.jit_kernels[kid].outputs.retain(|&x| x != cid);
            if let Some(rc) = rcs.get(&cid).copied()
                && rc > 0
            {
                let (new_kid, new_op) = self.new_load_kernel(cid, rc);
                visited.insert(cid, (new_kid, new_op));
                // The old loader is now an empty husk (no stores, output
                // stripped) — drop it so it doesn't trip the no-stores assert.
                if self.jit_kernels[kid].outputs.is_empty() {
                    self.jit_kernels.remove(kid);
                }
                return (new_kid, new_op);
            } else {
                return (kid, op_id);
            }
        }

        if !self.jit_kernels[kid].loads.contains(&cid) {
            let dtype = self.dtype(cid);
            let kernel = &mut self.jit_kernels[kid].kernel;
            let shape = kernel.stack_shape_dims(op_id);
            let dst = kernel.param(dtype, ParamKind::GlobalMut, shape);
            kernel.store(dst, op_id, OpId::NULL, MemLayout::Scalar);
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
        force_store: bool,
    ) -> (JitKernelId, OpId) {
        // if kernel has stores, store child and create fresh load kernel
        let force_store = force_store || self.jit_kernels[kid].kernel.contains_stores();

        // if kernel has multiple outputs, duplicate the kernel
        //println!("n_outputs={}", self.ekernels[kid].outputs.len());
        let log = std::env::var("ZYX_KERN_TRACE").is_ok();
        if log {
            eprintln!(
                "DOS child={child:?} kid={kid:?} n_out={} force_store={force_store} preced_red={} node={:?}",
                self.jit_kernels[kid].outputs.len(),
                self.jit_kernels[kid].kernel.is_preceded_by_reduce(op_id),
                self.nodes[*self.classes[child].nodes.last().unwrap()].node
            );
        }
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
    ) {
        let (mut kid, mut op_id) = visited[&child];
        (kid, op_id) = self.duplicate_or_store_class(child, kid, op_id, visited, rcs, force_store);
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

        let order = self.topo_sort_classes::<true>(&producer_boundaries, outputs, None);

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

        // Region id per structural class.
        let region_of: Map<ClassId, usize> = structural.iter().map(|&c| (c, find(&mut parent, idx[&c]))).collect();
        // A class consumed by a node in a *different* region must be stored —
        // the consumer region loads it through a global param ("a shape
        // dimension is a result of a kernel now and loaded into a new one").
        let mut cross_region_outputs: Map<usize, BTreeSet<ClassId>> = Map::default();
        for (i, &cid) in structural.iter().enumerate() {
            for nid in &self.classes[cid].nodes {
                for p in self.nodes[*nid].node.class_params() {
                    if producer_boundaries.contains(&p) {
                        continue;
                    }
                    if let Some(&pr) = region_of.get(&p)
                        && pr != find(&mut parent, i)
                    {
                        cross_region_outputs.entry(pr).or_default().insert(p);
                    }
                }
            }
        }

        for (root, region_classes) in regions.iter_mut() {
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
            if let Some(extra) = cross_region_outputs.remove(root) {
                region_outputs.extend(extra);
            }

            if region_outputs.is_empty() {
                continue;
            }
            let region_allowed: Set<ClassId> = region.union(&region_inputs).copied().collect();
            self.kernelize(&region_inputs, &region_outputs, Some(&region_allowed));
        }
    }
}

fn remove_first_output(kernels: &mut Slab<JitKernelId, JitKernelData>, kid: JitKernelId, cid: ClassId) -> bool {
    match kernels[kid].outputs.iter().position(|&x| x == cid) {
        Some(pos) => {
            kernels[kid].outputs.remove(pos);
            true
        }
        None => false,
    }
}
