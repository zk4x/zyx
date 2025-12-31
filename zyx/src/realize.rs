//! Converts graph to kernels and schedules them to devices

use nanoserde::SerBin;

use crate::{
    DType, DebugMask, Map, Set, ZyxError,
    backend::{Device, ProgramId, SearchConfig},
    cache::{Cache, get_perf},
    dtype::Constant,
    error::{BackendError, ErrorStatus},
    graph::{BOp, Graph, Node, ROp, UOp},
    kernel::{Kernel, Op, OpId},
    optimizer::Optimizer,
    prog_bar::ProgressBar,
    runtime::{Pool, Runtime, deallocate_tensors},
    shape::Dim,
    slab::{Slab, SlabId},
    tensor::TensorId,
    view::View,
};
use std::{collections::BTreeSet, hash::BuildHasherDefault, path::PathBuf};

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct KMKernelId(u32);

impl SlabId for KMKernelId {
    const ZERO: Self = Self(0);

    fn inc(&mut self) {
        self.0 += 1;
    }
}

impl From<usize> for KMKernelId {
    fn from(value: usize) -> Self {
        KMKernelId(value as u32)
    }
}

impl From<KMKernelId> for usize {
    fn from(value: KMKernelId) -> Self {
        value.0 as usize
    }
}

#[derive(Debug, Clone)]
struct KMKernel {
    kernel: Kernel,
    outputs: Vec<TensorId>,
    loads: Vec<TensorId>,
    stores: Vec<TensorId>,
}

impl KMKernel {
    fn contains_stores(&self) -> bool {
        self.kernel.contains_stores()
    }

    fn is_reduce(&self) -> bool {
        self.kernel.is_reduce()
    }

    fn total_reduce_dim(&self, op: OpId) -> Dim {
        self.kernel.total_reduce_dim(op)
    }

    fn apply_movement(&mut self, func: impl Fn(&mut View)) {
        self.kernel.apply_movement(func);
    }

    fn push(&mut self, op: Op) -> OpId {
        let op_id = self.kernel.ops.push(op);
        self.kernel.order.push(op_id);
        op_id
    }

    fn debug(&self) {
        println!("\nloads={:?}", self.loads);
        println!("stores={:?}", self.stores);
        print!("outputs={:?}", self.outputs);
        self.kernel.debug();
    }

    fn remove_first_output(&mut self, x: TensorId) {
        //println!("removing tensor {x} from kernel {kid:?}");
        let outputs = &mut self.outputs;
        outputs.iter().position(|elem| *elem == x).map(|i| outputs.remove(i));
    }

    /*fn drop_unused_ops(&mut self, visited: &Map<TensorId, (KernelId, OpId)>) {
        let params = self.outputs.iter().map(|tid| visited[tid].1).collect();
        let required = self.get_required_ops(params);
        let mut loaded_tensors = Vec::new();
        let mut load_index = 0;
        let ops = &mut self.kernel.ops;
        let loads = &self.loads;
        for (op_id, op) in ops.iter_mut().enumerate() {
            let is_required = required.contains(&op_id);
            if let Op::LoadView { .. } = op {
                if is_required {
                    loaded_tensors.push(loads[load_index]);
                }
                load_index += 1;
            }
            if !is_required {
                *op = Op::Null;
            }
        }
        self.loads = loaded_tensors;
        #[cfg(debug_assertions)]
        if self.loads.len() != self.kernel.ops.iter().filter(|op| matches!(op, Op::LoadView { .. })).count() {
            self.debug();
            panic!();
        }
    }

    fn get_required_ops(
        &self,
        mut params: Vec<OpId>,
    ) -> std::collections::HashSet<usize, BuildHasherDefault<crate::chasher::CHasher>> {
        let mut required = Set::default();
        while let Some(param) = params.pop() {
            if !required.insert(param) {
                continue;
            }
            //println!("param={param}");
            match &self.kernel.ops[param] {
                Op::Reduce { x, .. } | Op::Cast { x, .. } | Op::Unary { x, .. } => {
                    params.push(*x);
                }
                Op::Binary { x, y, .. } => {
                    params.push(*x);
                    params.push(*y);
                }
                Op::Const(..) | Op::ConstView { .. } | Op::LoadView { .. } => {}
                Op::Define { .. }
                | Op::StoreView { .. }
                | Op::Load { .. }
                | Op::Store { .. }
                | Op::Null
                | Op::Loop { .. }
                | Op::EndLoop => unreachable!(),
            }
        }
        required
    }*/
}

impl std::ops::Index<OpId> for KMKernel {
    type Output = Op;

    fn index(&self, index: OpId) -> &Self::Output {
        &self.kernel.ops[index]
    }
}

struct Kernelizer<'a> {
    // TODO merge as many of these as possible. Perhaps start by mergins rcs and visited
    // Those nodes that have been store ops in some kernel, but those kernels may have not yet run (must be checked in realized_nodex).
    virt_realized_nodes: Set<TensorId>,
    must_keep_nodes: Set<TensorId>,
    realized_nodes: Set<TensorId>,
    kernels: Slab<KMKernelId, KMKernel>,
    visited: Map<TensorId, (KMKernelId, OpId)>,
    rcs: Map<TensorId, u32>,
    graph: &'a Graph,
    pools: &'a mut [Pool],
    devices: &'a mut [Device],
    cache: &'a mut Cache,
    search_config: &'a SearchConfig,
    config_dir: Option<&'a PathBuf>,
    debug: &'a DebugMask,
}

impl<'a> Kernelizer<'a> {
    fn new(
        realized_nodes: Set<TensorId>,
        to_eval: &'a Set<TensorId>,
        rcs: Map<TensorId, u32>,
        graph: &'a Graph,
        pools: &'a mut [Pool],
        devices: &'a mut [Device],
        cache: &'a mut Cache,
        search_config: &'a SearchConfig,
        config_dir: Option<&'a PathBuf>,
        debug: &'a DebugMask,
    ) -> Self {
        let mut must_keep_nodes = realized_nodes.clone();
        must_keep_nodes.extend(to_eval);
        Self {
            // Those nodes that have been store ops in some kernel, but those kernels may have not yet run (must be checked in realized_nodex).
            must_keep_nodes,
            virt_realized_nodes: realized_nodes.clone(),
            realized_nodes,
            kernels: Slab::with_capacity(30),
            visited: Map::with_capacity_and_hasher(100, BuildHasherDefault::new()),
            rcs,
            graph,
            pools,
            devices,
            cache,
            search_config,
            config_dir,
            debug,
        }
    }

    #[allow(unused)]
    fn debug(&self) {
        for kernel in self.kernels.values() {
            kernel.debug();
        }
        println!();
    }

    fn is_virt_realized(&self, nid: TensorId) -> bool {
        self.virt_realized_nodes.contains(&nid)
    }

    fn duplicate_or_store(&mut self, x: TensorId, reduce_dims: Option<Dim>) -> Result<(KMKernelId, OpId), ZyxError> {
        let (mut kid, mut op_id) = self.visited[&x];

        if self.kernels[kid].contains_stores() {
            self.add_store(x)?;
            (kid, op_id) = self.create_load_kernel(x);
            if self.kernels[kid].outputs.len() > 1 {
                kid = self.duplicate_kernel(x, kid);
            }
        }

        // if it's reduce
        if let Some(reduce_dims) = reduce_dims {
            if self.kernels[kid].is_reduce() {
                if reduce_dims * self.kernels[kid].total_reduce_dim(op_id) > 32000 {
                    self.add_store(x)?;
                    (kid, op_id) = self.create_load_kernel(x);
                    if self.kernels[kid].outputs.len() > 1 {
                        kid = self.duplicate_kernel(x, kid);
                    }
                }
            }
        }

        if self.kernels[kid].outputs.len() > 1 {
            //println!("Duplicating kernel");
            //self.kernels[kid].debug();
            if self.kernels[kid].is_reduce() {
                self.add_store(x)?;
                (kid, op_id) = self.create_load_kernel(x);
                if self.kernels[kid].outputs.len() > 1 {
                    kid = self.duplicate_kernel(x, kid);
                }
            } else {
                kid = self.duplicate_kernel(x, kid);
            }
        }
        Ok((kid, op_id))
    }

    fn duplicate_kernel(&mut self, x: TensorId, kid: KMKernelId) -> KMKernelId {
        //println!("op_id={op_id}");
        //println!("Duplicating");
        // Instead of copy of the whole kernel, copy only relevant ops
        // and remove these ops from the original if not needed.
        let mut kernel = self.kernels[kid].clone();
        kernel.outputs = vec![x];
        //kernel.drop_unused_ops(&self.visited);
        self.kernels[kid].remove_first_output(x);
        //self.kernels[kid].drop_unused_ops(&self.visited);
        //self.debug();
        self.kernels.push(kernel)
    }

    fn create_load_kernel(&mut self, nid: TensorId) -> (KMKernelId, OpId) {
        //println!("ADDING LOAD for {x} x {}", rc);
        let shape = self.graph.shape(nid);
        let dtype = self.graph.dtype(nid);
        let mut ops = Slab::with_capacity(100);
        let op_id = ops.push(Op::LoadView { dtype, view: View::contiguous(shape) });
        let kernel = KMKernel {
            kernel: Kernel { ops, order: vec![op_id] },
            outputs: vec![nid; self.rcs[&nid] as usize],
            loads: vec![nid],
            stores: Vec::new(),
        };
        let kid = self.kernels.push(kernel);
        self.visited.insert(nid, (kid, op_id));
        (kid, op_id)
    }

    fn create_const_kernel(&mut self, nid: TensorId, value: Constant) {
        let mut ops = Slab::with_capacity(100);
        let op_id = ops.push(Op::ConstView { value, view: View::contiguous(&[1]) });
        let kernel = KMKernel {
            kernel: Kernel { ops, order: vec![op_id] },
            outputs: vec![nid; self.rcs[&nid] as usize],
            loads: Vec::new(),
            stores: Vec::new(),
        };
        let kid = self.kernels.push(kernel);
        self.visited.insert(nid, (kid, op_id));
    }

    fn add_expand_op(&mut self, nid: TensorId, x: TensorId) -> Result<(), ZyxError> {
        // TODO instead of store add expand op that inserts loop in IR
        let (kid, op_id) = self.duplicate_or_store(x, Some(1))?;
        let shape = self.graph.shape(nid);
        let kernel = &mut self.kernels[kid];
        kernel.apply_movement(|view| view.expand(shape));
        kernel.remove_first_output(x);
        kernel.outputs.extend(vec![nid; self.rcs[&nid] as usize]);
        *self.rcs.get_mut(&x).unwrap() -= 1;
        debug_assert_eq!(self.graph.shape(nid), kernel.kernel.shape());
        self.visited.insert(nid, (kid, op_id));
        Ok(())
    }

    fn add_permute_op(&mut self, nid: TensorId, x: TensorId) -> Result<(), ZyxError> {
        // TODO instead of store add permute op that swaps indices in IR
        let (kid, op_id) = self.duplicate_or_store(x, None)?;
        let axes = self.graph.axes(nid);
        let kernel = &mut self.kernels[kid];
        kernel.apply_movement(|view| view.permute(axes));
        kernel.remove_first_output(x);
        kernel.outputs.extend(vec![nid; self.rcs[&nid] as usize]);
        *self.rcs.get_mut(&x).unwrap() -= 1;
        debug_assert_eq!(self.graph.shape(nid), kernel.kernel.shape());
        self.visited.insert(nid, (kid, op_id));
        Ok(())
    }

    fn add_reshape_op(&mut self, nid: TensorId, x: TensorId) -> Result<(), ZyxError> {
        debug_assert!(self.visited.contains_key(&x), "Missing tensor {x} in visited.");
        // TODO duplicate or store only if this is not mergeable.
        // Otherwise if it is like unsqueeze or splitting two dims
        // or fusing two dims, it can be represented by a custom
        // op that is unfoldable into indices, since it does not change
        // global work size.
        let (kid, op_id) = self.duplicate_or_store(x, None)?;
        let n = self.graph.shape(x).len();
        let shape = self.graph.shape(nid);
        let kernel = &mut self.kernels[kid];
        kernel.apply_movement(|view| view.reshape(0..n, shape));
        kernel.remove_first_output(x);
        kernel.outputs.extend(vec![nid; self.rcs[&nid] as usize]);
        *self.rcs.get_mut(&x).unwrap() -= 1;
        debug_assert_eq!(self.graph.shape(nid), kernel.kernel.shape());
        self.visited.insert(nid, (kid, op_id));
        Ok(())
    }

    fn add_pad_op(&mut self, nid: TensorId, x: TensorId) -> Result<(), ZyxError> {
        // TODO instead of duplication add pad op that adds if statement into ir (e.g. if idx < padding)
        let (kid, op_id) = self.duplicate_or_store(x, Some(1))?;
        let padding = self.graph.padding(nid);
        let rank = self.graph.shape(nid).len();
        let kernel = &mut self.kernels[kid];
        kernel.apply_movement(|view| view.pad(rank, padding));
        kernel.remove_first_output(x);
        kernel.outputs.extend(vec![nid; self.rcs[&nid] as usize]);
        *self.rcs.get_mut(&x).unwrap() -= 1;
        debug_assert_eq!(self.graph.shape(nid), kernel.kernel.shape());
        self.visited.insert(nid, (kid, op_id));
        Ok(())
    }

    fn add_reduce_op(&mut self, nid: TensorId, x: TensorId, rop: ROp) -> Result<(), ZyxError> {
        // Don't apply reduce if the kernel already contains reduce
        // and the resulting shape's dimension is less than 256

        let axes = self.graph.axes(nid);
        let shape = self.graph.shape(x);
        let dims: Vec<Dim> = axes.iter().map(|&a| shape[a]).collect();

        // If the kernel has more than one output, or rc of x is more than one,
        // we have to either copy it (if it is small), or store x (if kid is big)
        let reduce_dims_product: Dim = dims.iter().product();
        let (kid, op_id) = self.duplicate_or_store(x, Some(reduce_dims_product))?;
        //self.debug();

        /*for kernel in kernels.iter() {
            println!("{:?}, {}", kernel.0, kernel.1.n_outputs);
        }*/

        #[cfg(debug_assertions)]
        {
            use crate::shape::UAxis;
            let mut sorted_axes: Vec<UAxis> = axes.into();
            sorted_axes.sort_unstable();
            debug_assert_eq!(axes, sorted_axes, "Reduce axes must be sorted.");
        }

        //kernels[kid].debug();
        //println!("{axes:?}");

        {
            // Permute before reduce so that reduce axes are last
            let n = shape.len();
            let mut permute_axes = Vec::with_capacity(n);
            let max_axis = *axes.last().unwrap();
            let mut ai = 0;
            for i in 0..=max_axis {
                if axes[ai] == i {
                    ai += 1;
                } else {
                    permute_axes.push(i);
                }
            }
            permute_axes.extend(max_axis + 1..n);
            permute_axes.extend_from_slice(axes);
            self.kernels[kid].apply_movement(|v| v.permute(&permute_axes));
        }

        // If all dims are reduced
        if shape == dims {
            self.kernels[kid].apply_movement(|v| v.reshape(0..1, &[1, shape[0]]));
        }
        let kernel = &mut self.kernels[kid];
        let op_id = kernel.push(Op::Reduce { x: op_id, rop, dims });
        kernel.remove_first_output(x);
        kernel.outputs.extend(vec![nid; self.rcs[&nid] as usize]);
        *self.rcs.get_mut(&x).unwrap() -= 1;
        debug_assert_eq!(self.graph.shape(nid), kernel.kernel.shape());
        self.visited.insert(nid, (kid, op_id));
        Ok(())
    }

    fn add_cast_op(&mut self, nid: TensorId, x: TensorId, dtype: DType) {
        let (kid, op_id) = self.visited[&x];
        let kernel = &mut self.kernels[kid];
        let op_id = kernel.push(Op::Cast { x: op_id, dtype });
        kernel.remove_first_output(x);
        kernel.outputs.extend(vec![nid; self.rcs[&nid] as usize]);
        *self.rcs.get_mut(&x).unwrap() -= 1;
        self.visited.insert(nid, (kid, op_id));
    }

    fn add_unary_op(&mut self, nid: TensorId, x: TensorId, uop: UOp) {
        let (kid, op_id) = self.visited[&x];
        let kernel = &mut self.kernels[kid];
        let op_id = kernel.push(Op::Unary { x: op_id, uop });
        kernel.remove_first_output(x);
        kernel.outputs.extend(vec![nid; self.rcs[&nid] as usize]);
        *self.rcs.get_mut(&x).unwrap() -= 1;
        self.visited.insert(nid, (kid, op_id));
    }

    fn add_binary_op(&mut self, nid: TensorId, x: TensorId, y: TensorId, bop: BOp) -> Result<(), ZyxError> {
        let (mut kid, mut op_id) = self.visited[&x];
        let (mut kidy, mut op_idy) = self.visited[&y];

        //self.kernels[kid].debug();
        //self.kernels[kidy].debug();

        let kid_stores = !self.kernels[kid].stores.is_empty();
        let kidy_stores = !self.kernels[kidy].stores.is_empty();

        let new_op_id = if kid == kidy {
            //println!("Same kernels for binary");
            let kernel = &mut self.kernels[kid];
            kernel.remove_first_output(x);
            kernel.remove_first_output(y);
            kernel.outputs.extend(vec![nid; self.rcs[&nid] as usize]);
            kernel.push(Op::Binary { x: op_id, y: op_idy, bop })
        } else {
            //println!("Different kernels for binary");
            // TODO later use this, but this requires global memory sync inside of the kernel
            // as it loads and stores from the same kernel
            //if kid_stores && kidy_stores {
            match (kid_stores, kidy_stores) {
                (true, true) => {
                    self.add_store(x)?;
                    (kid, op_id) = self.create_load_kernel(x);
                    if self.kernels[kid].outputs.len() > 1 {
                        kid = self.duplicate_kernel(x, kid);
                        self.kernels[kid].outputs.push(x);
                    }
                    self.add_store(y)?;
                    (kidy, op_idy) = self.create_load_kernel(y);
                    //println!("kidy={:?}", kidy);
                    if self.kernels[kidy].outputs.len() > 1 {
                        kidy = self.duplicate_kernel(y, kidy);
                        self.kernels[kidy].outputs.push(y);
                    }
                    //println!("kidy={:?}", kidy);
                }
                (true, false) => {
                    self.add_store(x)?;
                    (kid, op_id) = self.create_load_kernel(x);
                    if self.kernels[kid].outputs.len() > 1 {
                        kid = self.duplicate_kernel(x, kid);
                        self.kernels[kid].outputs.push(x);
                    }
                }
                (false, true) => {
                    self.add_store(y)?;
                    (kidy, op_idy) = self.create_load_kernel(y);
                    //println!("kidy={:?}", kidy);
                    if self.kernels[kidy].outputs.len() > 1 {
                        kidy = self.duplicate_kernel(y, kidy);
                        self.kernels[kidy].outputs.push(y);
                    }
                    //println!("kidy={:?}", kidy);
                }
                (false, false) => {}
            }

            /*if !kid_stores && kidy_stores {
                //println!("Swap x, y");
                (kid, kidy) = (kidy, kid);
                (op_id, op_idy) = (op_idy, op_id);
            }*/

            self.kernels[kidy].remove_first_output(y);
            let KMKernel { kernel, outputs, loads, stores } = unsafe { self.kernels.remove_and_return(kidy) };

            // Extend x kernel with y ops
            let mut y_ops_map = Map::with_capacity_and_hasher(50, BuildHasherDefault::new());
            for (op_id, op) in kernel.ops.iter() {
                let mut op = op.clone();
                for param in op.parameters_mut() {
                    *param = y_ops_map[param];
                }

                let new_op_id = self.kernels[kid].push(op);
                y_ops_map.insert(op_id, new_op_id);
            }
            // Fix visited
            for (kidm, op_id) in self.visited.values_mut() {
                if *kidm == kidy {
                    *kidm = kid;
                    *op_id = y_ops_map[op_id];
                }
            }

            self.kernels[kid].loads.extend(loads);
            self.kernels[kid].stores.extend(stores);

            self.kernels[kid].remove_first_output(x);
            self.kernels[kid].outputs.extend(outputs);
            self.kernels[kid].outputs.extend(vec![nid; self.rcs[&nid] as usize]);

            self.kernels[kid].push(Op::Binary { x: op_id, y: y_ops_map[&op_idy], bop })
        };

        *self.rcs.get_mut(&x).unwrap() -= 1;
        *self.rcs.get_mut(&y).unwrap() -= 1;
        self.visited.insert(nid, (kid, new_op_id));
        //self.kernels[kid].debug();
        Ok(())
    }

    /// Stores x and returns remaining reference count to x
    fn add_store(&mut self, x: TensorId) -> Result<(), ZyxError> {
        let (kid, op_id) = self.visited[&x];
        if self.virt_realized_nodes.contains(&x) {
            self.visited.remove(&x).unwrap();
            self.kernels[kid].outputs.retain(|&elem| elem != x);
        } else {
            self.visited.remove(&x).unwrap();
            self.virt_realized_nodes.insert(x);
            let dtype = self.graph.dtype(x);
            self.kernels[kid].push(Op::StoreView { src: op_id, dtype });
            self.kernels[kid].stores.push(x);

            // remove all references to x
            self.kernels[kid].outputs.retain(|&elem| elem != x);
        }

        //kernels[kid].debug();
        if self.kernels[kid].outputs.is_empty()
            && self.kernels[kid].loads.iter().all(|x| self.realized_nodes.contains(x))
        {
            let kernel = unsafe { self.kernels.remove_and_return(kid) };
            let loads = kernel.loads.clone();
            let stores = kernel.stores.clone();
            self.launch_kernel(kernel)?;
            self.realized_nodes.extend(stores);
            // Delete unneeded intermediate tensors from memory pools
            let mut to_remove = Set::with_capacity_and_hasher(1, BuildHasherDefault::new());
            for tid in loads {
                if !self.kernels.values().any(|kernel| kernel.loads.contains(&tid))
                    && !self.must_keep_nodes.contains(&tid)
                {
                    to_remove.insert(tid);
                }
            }
            deallocate_tensors(&to_remove, self.pools);
        }
        //println!("ADDED STORE for {x} x {xrc_rem}");
        Ok(())
    }

    fn launch_kernel(&mut self, kernel: KMKernel) -> Result<(), ZyxError> {
        // Iterate over all memory pools ordered by device speed.
        // Then select first fastest device that has associated memory pool which fits all tensors used
        // as arguments for the kernel that are not yet allocated on that memory pool.

        if kernel.stores.is_empty() {
            kernel.debug();
        }
        let KMKernel { mut kernel, outputs: _, loads, stores } = kernel;
        //debug_assert!(outputs.is_empty());
        debug_assert!(!stores.is_empty());
        debug_assert!(!kernel.ops.is_empty());

        let required_stores_memory: Dim = stores
            .iter()
            .map(|&tid| self.graph.shape(tid).iter().product::<Dim>() * self.graph.dtype(tid).byte_size() as Dim)
            .sum::<Dim>();
        //println!("Kernel requires {required_kernel_memory} B");
        let mut dev_ids: Vec<usize> = (0..self.devices.len()).collect();
        dev_ids.sort_unstable_by_key(|&dev_id| self.devices[dev_id].free_compute());
        dev_ids.reverse();
        //println!("dev_ids={dev_ids:?}");
        let mut device_id = None;
        for dev_id in dev_ids {
            let mpid = self.devices[dev_id].memory_pool_id() as usize;
            // Check if kernel arguments fit into associated memory pool
            let free_memory = self.pools[mpid].pool.free_bytes();
            // required memory is lowered by the amount of tensors already stored in that memory pool
            let missing_loads_memory = loads
                .iter()
                .map(|tid| {
                    if self.pools[mpid].buffer_map.contains_key(tid) {
                        0
                    } else {
                        self.graph.shape(*tid).iter().product::<Dim>() * self.graph.dtype(*tid).byte_size() as Dim
                    }
                })
                .sum::<Dim>();
            //println!("Free memory {free_memory} B, missing loads memory {missing_loads_memory} B");
            let required_memory = required_stores_memory + missing_loads_memory;
            if free_memory > required_memory {
                device_id = Some(dev_id);
                break;
            }
        }
        // else
        let Some(dev_id) = device_id else {
            return Err(ZyxError::AllocationError(
                format!("no device has enough memory to store {required_stores_memory} B for intermedite tensors.")
                    .into(),
            ));
        };
        let _ = device_id;
        let mpid = self.devices[dev_id].memory_pool_id() as usize;

        let mut event_wait_list = Vec::new();
        // Move all loads to that pool if they are not there already.
        for &tid in &loads {
            if !self.pools[mpid].buffer_map.contains_key(&tid) {
                if !self.pools[mpid].buffer_map.contains_key(&tid) {
                    // Check where the tensor is
                    let mut old_mpid = usize::MAX;
                    for (i, pool) in self.pools.iter().enumerate() {
                        if pool.buffer_map.contains_key(&tid) {
                            old_mpid = i;
                            break;
                        }
                    }
                    debug_assert_ne!(old_mpid, usize::MAX);

                    let bytes =
                        self.graph.shape(tid).iter().product::<Dim>() * self.graph.dtype(tid).byte_size() as Dim;
                    // No need to initialize here, other than rust is bad.
                    let mut byte_slice = vec![0u8; bytes as usize];
                    let src = self.pools[old_mpid].buffer_map[&tid];

                    // Move the tensor from old pool into temporary in RAM
                    // TODO later we can implement direct GPU to GPU movement, it's easy here,
                    // a bit harder for the backends.
                    // Pool to host blocks on event, so we can remove that event.
                    let mut event_wait_list = Vec::new();
                    for buffers in self.pools[old_mpid].events.keys() {
                        if buffers.contains(&src) {
                            let buffers = buffers.clone();
                            // Pool to host blocks on event, so we can remove that event.
                            let event = self.pools[old_mpid].events.remove(&buffers).unwrap();
                            event_wait_list.push(event);
                            break;
                        }
                    }
                    self.pools[old_mpid].pool.pool_to_host(src, &mut byte_slice, event_wait_list)?;

                    // Delete the tensor from the old pool
                    self.pools[old_mpid].pool.deallocate(src, vec![]);
                    self.pools[old_mpid].buffer_map.remove(&tid);
                    //println!("{byte_slice:?}");

                    let (dst, event) = self.pools[mpid].pool.allocate(bytes)?;
                    let event = self.pools[mpid].pool.host_to_pool(&byte_slice, dst, vec![event])?;
                    // We have to sync here, because byte_slice does not exist any more.
                    // The other solution would be to put this into temp_data.
                    // But perhaps we should figure some better async.
                    self.pools[mpid].pool.sync_events(vec![event])?;
                    self.pools[mpid].buffer_map.insert(tid, dst);
                    //memory_pools[mpid].events.insert(BTreeSet::from([dst]), event);
                }
            }
        }

        // Allocate space for all stores (outputs)
        let mut output_buffers = BTreeSet::new();
        for &tid in &stores {
            let bytes = self.graph.shape(tid).iter().product::<Dim>() * self.graph.dtype(tid).byte_size() as Dim;
            let (buffer_id, event) = self.pools[mpid].pool.allocate(bytes)?;
            self.pools[mpid].buffer_map.insert(tid, buffer_id);
            event_wait_list.push(event);
            output_buffers.insert(buffer_id);
        }

        // Get a list of all arg buffers. These must be specifically in order as they are mentioned in kernel ops
        let mut args = Vec::new();
        for tid in &loads {
            args.push(self.pools[mpid].buffer_map[tid]);
        }
        for tid in &stores {
            args.push(self.pools[mpid].buffer_map[tid]);
        }

        /***** CACHE and OPTIMIZATION SEARCH *****/

        let device = &mut self.devices[dev_id];
        let dev_id = crate::cache::DeviceId(dev_id as u32);
        let pool = &mut self.pools[mpid];

        // Send the kernel to kernel cache.
        let dev_info_id = self.cache.get_or_add_dev_info(device.info());

        // Launch if it is in cache
        let kernel_id;
        let mut optimizer;
        if let Some(&kid) = self.cache.kernels.get(&kernel) {
            // If it has been compiled for the device
            if let Some(&program_id) = self.cache.programs.get(&(kid, dev_id)) {
                if self.debug.kmd() {
                    println!("Kernel launch from memory pool {mpid} with args: {args:?}");
                }
                let event = device.launch(program_id, &mut pool.pool, &args, event_wait_list)?;
                self.pools[mpid].events.insert(output_buffers, event);
                return Ok(());
            } else if let Some(opt) = self.cache.optimizations.get(&(kid, dev_info_id)) {
                // Continue optimizing using optimizations cached to disk
                optimizer = opt.clone();
            } else {
                // It was optimized for different device
                optimizer = Optimizer::new(&kernel, device.info());
            }
            kernel_id = kid;
        } else {
            // If it is not in cache, we just get new empty kernel id where we insert the kernel
            optimizer = Optimizer::new(&kernel, device.info());
            // a bit unnecessay kernel clone here, but it does not really matter
            kernel_id = self.cache.insert_kernel(kernel.clone());
        }

        // Check if best optimization already found
        if optimizer.fully_optimized() || (self.search_config.iterations == 0 && !optimizer.is_new()) {
            // done optimizing, loaded best from disk
            let opt_res = optimizer.apply_optimization(&mut kernel, optimizer.best_optimization(), self.debug.ir());
            debug_assert!(opt_res);
            let program_id = device.compile(&kernel, self.debug.asm())?;
            if self.debug.kmd() {
                println!("Kernel launch from memory pool {mpid} with args: {args:?}");
            }
            self.cache.programs.insert((kernel_id, dev_id), program_id);
            let event = device.launch(program_id, &mut pool.pool, &args, event_wait_list)?;
            self.pools[mpid].events.insert(output_buffers, event);
            return Ok(());
        }

        if self.debug.sched() {
            println!(
                "Optimizing kernel stores {stores:?}, loads {loads:?}, max iterations: {}",
                optimizer.max_iters()
            );
            kernel.debug();
        }

        // If search_iters == 0, we use default optimizations
        if self.search_config.iterations == 0 {
            let mut okernel;
            let program_id;
            let nanos;

            loop {
                okernel = kernel.clone();

                let Some(optimization) = optimizer.next_optimization(u64::MAX) else {
                    return Err(ZyxError::KernelLaunchFailure);
                };

                if !optimizer.apply_optimization(&mut okernel, optimization, self.debug.ir()) {
                    continue;
                }

                if let Ok(pid) = device.compile(&okernel, self.debug.asm()) {
                    program_id = pid;
                } else {
                    continue;
                }

                if self.debug.kmd() {
                    println!("Kernel launch from memory pool {mpid} with args: {args:?}");
                }
                let timer = std::time::Instant::now();
                let event = device.launch(program_id, &mut pool.pool, &args, event_wait_list)?;
                pool.pool.sync_events(vec![event])?;
                nanos = timer.elapsed().as_nanos() as u64;

                break;
            }

            if nanos < optimizer.best_time_nanos {
                self.cache.programs.insert((kernel_id, dev_id), program_id);
            }
            if self.debug.perf() {
                let (flop, mem_read, mem_write) = kernel.flop_mem_rw();
                println!("{}", get_perf(flop, mem_read, mem_write, nanos));
            }
            optimizer.best_time_nanos = nanos;
        } else {
            let mut last_time_nanos = u64::MAX;

            pool.pool.sync_events(event_wait_list)?;

            let mut progress_bar = if self.debug.perf() {
                let (flop, mem_read, mem_write) = kernel.flop_mem_rw();
                Some((
                    ProgressBar::new(self.search_config.iterations.min(optimizer.max_iters() as usize) as u64),
                    flop,
                    mem_read,
                    mem_write,
                ))
            } else {
                None
            };

            /*for &arg in &args {
                let mut data = [0f32; 10];
                Runtime::load_buffer(&mut data, pool, arg)?;
                println!("{data:?}");
            }*/

            let mut i = 0;
            while let Some(optimization) = optimizer.next_optimization(last_time_nanos)
                && i < self.search_config.iterations
            {
                i += 1;
                let mut kernel = kernel.clone();
                if !optimizer.apply_optimization(&mut kernel, optimization, self.debug.ir()) {
                    continue;
                }

                let res = (|| -> Result<(ProgramId, u64), BackendError> {
                    let program_id = device.compile(&kernel, self.debug.asm())?;
                    let begin = std::time::Instant::now();
                    let event = device.launch(program_id, &mut pool.pool, &args, Vec::new())?;
                    pool.pool.sync_events(vec![event])?;
                    Ok((program_id, begin.elapsed().as_nanos() as u64))
                })();

                last_time_nanos = if let Ok((program_id, last_time_nanos)) = res {
                    if last_time_nanos < optimizer.best_time_nanos {
                        self.cache.programs.insert((kernel_id, dev_id), program_id);
                    }
                    last_time_nanos
                } else {
                    if let Err(err) = res {
                        match err.status {
                            ErrorStatus::KernelCompilation
                            | ErrorStatus::IncorrectKernelArg
                            | ErrorStatus::KernelLaunch
                            | ErrorStatus::KernelSync => {}
                            _ => {
                                println!();
                                return Err(ZyxError::BackendError(err));
                            }
                        }
                    }
                    u64::MAX
                };

                if let Some((prog_bar, flop, mem_read, mem_write)) = &mut progress_bar {
                    prog_bar.inc(
                        1,
                        &format!(
                            "{}, best={}μs",
                            get_perf(*flop, *mem_read, *mem_write, last_time_nanos),
                            if optimizer.best_time_nanos == u64::MAX {
                                ("inf").into()
                            } else {
                                (optimizer.best_time_nanos / 1000).to_string()
                            }
                        ),
                    );
                }
            }
            if let Some((_, flop, mem_read, mem_write)) = &progress_bar {
                println!();
                println!(
                    "Best: {}",
                    get_perf(*flop, *mem_read, *mem_write, optimizer.best_time_nanos)
                );
            }
        }

        self.cache.optimizations.insert((kernel_id, dev_info_id), optimizer);
        if self.search_config.save_to_disk {
            if let Some(mut path) = self.config_dir.cloned() {
                path.push("cached_kernels");
                let ser_cache: Vec<u8> = self.cache.serialize_bin();
                std::fs::write(path, ser_cache)?;
            }
        }

        Ok(())
    }
}

impl Runtime {
    fn realize_with_order(
        &mut self,
        rcs: Map<TensorId, u32>,
        realized_nodes: Set<TensorId>,
        order: &[TensorId],
        to_eval: &Set<TensorId>,
    ) -> Result<(), ZyxError> {
        /*let rcs = if rcs.is_empty() {
            let mut rcs = Map::with_capacity_and_hasher(100, BuildHasherDefault::default());
            for &nid in order {
                if !realized_nodes.contains(&nid) {
                    for nid in self.graph[nid].parameters() {
                        rcs.entry(nid).and_modify(|rc| *rc += 1).or_insert(1);
                    }
                }
            }
            for &nid in to_eval {
                rcs.entry(nid).and_modify(|rc| *rc += 1).or_insert(1);
            }
            rcs
        } else {
            rcs
        };*/

        #[cfg(debug_assertions)]
        {
            let mut rcs2 = Map::with_hasher(BuildHasherDefault::default());
            for &nid in order {
                if !realized_nodes.contains(&nid) {
                    for nid in self.graph[nid].parameters() {
                        rcs2.entry(nid).and_modify(|rc| *rc += 1).or_insert(1);
                    }
                }
            }
            for &nid in to_eval {
                rcs2.entry(nid).and_modify(|rc| *rc += 1).or_insert(1);
            }
            if rcs2 != rcs {
                println!("Realized nodes: {realized_nodes:?}");
                for &nid in order {
                    println!(
                        "ID({nid:?}): {:?}, sh: {:?}, rcs: {}, rcs actual: {}",
                        self.graph[nid],
                        self.graph.shape(nid),
                        rcs.get(&nid).copied().unwrap_or(0),
                        rcs2.get(&nid).copied().unwrap_or(0),
                    );
                }
                panic!("rcs are incorrect, rcs: {rcs:?}\nrcs2: {rcs2:?}");
            }
        }

        //println!("{rcs:?}");
        //println!("realized_nodes: {realized_nodes:?}");
        //println!("to_eval: {to_eval:?}");

        let begin = std::time::Instant::now();

        let mut kernelizer = Kernelizer::new(
            realized_nodes,
            to_eval,
            rcs,
            &self.graph,
            &mut self.pools,
            &mut self.devices,
            &mut self.cache,
            &self.search_config,
            self.config_dir.as_ref(),
            &self.debug,
        );

        for &nid in order {
            /*use crate::{RED, RESET};
            println!(
                "{RED}{}{nid} x {} -> {:?}  {}  {:?}{RESET}",
                if kernelizer.is_virt_realized(nid) { "LOAD " } else { "" },
                kernelizer.rcs[&nid],
                self.graph[nid],
                self.graph.dtype(nid),
                self.graph.shape(nid)
            );*/
            if kernelizer.is_virt_realized(nid) {
                kernelizer.create_load_kernel(nid);
            } else {
                match self.graph[nid] {
                    Node::Leaf { .. } => unreachable!(),
                    Node::Const { value } => kernelizer.create_const_kernel(nid, value),
                    Node::Cast { x, dtype } => kernelizer.add_cast_op(nid, x, dtype),
                    Node::Unary { x, uop } => kernelizer.add_unary_op(nid, x, uop),
                    Node::Expand { x } => kernelizer.add_expand_op(nid, x)?,
                    Node::Permute { x } => kernelizer.add_permute_op(nid, x)?,
                    Node::Reshape { x } => kernelizer.add_reshape_op(nid, x)?,
                    Node::Pad { x } => kernelizer.add_pad_op(nid, x)?,
                    Node::Reduce { x, rop } => kernelizer.add_reduce_op(nid, x, rop)?,
                    Node::Binary { x, y, bop } => kernelizer.add_binary_op(nid, x, y, bop)?,
                }
            }

            if to_eval.contains(&nid) {
                kernelizer.add_store(nid)?;
                *kernelizer.rcs.get_mut(&nid).unwrap() -= 1;
                if kernelizer.rcs[&nid] > 0 {
                    kernelizer.create_load_kernel(nid);
                }
            }

            //kernelizer.debug();
        }

        if kernelizer.kernels.len() > KMKernelId(0) {
            let mut kids: Vec<KMKernelId> = kernelizer.kernels.ids().collect();
            while let Some(kid) = kids
                .iter()
                .find(|&&kid| kernelizer.kernels[kid].loads.iter().all(|x| kernelizer.realized_nodes.contains(x)))
                .copied()
            {
                kids.retain(|x| *x != kid);
                let kernel = unsafe { kernelizer.kernels.remove_and_return(kid) };
                let loads = kernel.loads.clone();
                let stores = kernel.stores.clone();
                kernelizer.launch_kernel(kernel)?;
                kernelizer.realized_nodes.extend(stores);

                // Delete unneeded intermediate tensors in memory pools
                let mut to_remove = Set::with_capacity_and_hasher(1, BuildHasherDefault::new());
                for tid in loads {
                    if !kernelizer.kernels.values().any(|kernel| kernel.loads.contains(&tid))
                        && !kernelizer.must_keep_nodes.contains(&tid)
                    {
                        to_remove.insert(tid);
                    }
                }
                deallocate_tensors(&to_remove, kernelizer.pools);
            }
        }

        #[cfg(debug_assertions)]
        {
            if kernelizer.kernels.len() > KMKernelId(0) {
                println!("realized_nodes={:?}", kernelizer.realized_nodes);
                println!("Unrealized kernels:");
                for (kid, kernel) in kernelizer.kernels.iter() {
                    println!("loads={:?}", kernel.loads);
                    println!("stores={:?}", kernel.stores);
                    println!("{kid:?}, outputs={:?}", kernel.outputs);
                    kernel.debug();
                    println!();
                }
                panic!();
            }
            debug_assert!(to_eval.is_subset(&kernelizer.realized_nodes));
            //println!("Realized nodes in kernelizer: {:?}", kernelizer.realized_nodes);
        }

        let elapsed = begin.elapsed();
        if self.debug.perf() {
            println!("Kernelizer took {} μs", elapsed.as_micros());
        }
        Ok(())
    }

    pub fn realize_selected(&mut self, to_eval: &Set<TensorId>) -> Result<(), ZyxError> {
        let realized_nodes: Set<TensorId> =
            self.pools.iter().flat_map(|pool| pool.buffer_map.keys()).copied().collect();

        let to_eval: Set<TensorId> = to_eval.difference(&realized_nodes).copied().collect();

        if to_eval.is_empty() {
            return Ok(());
        }

        if self.devices.is_empty() {
            self.initialize_devices()?;
        }

        let mut params: Vec<TensorId> = to_eval.iter().copied().collect();
        let mut rcs: Map<TensorId, u32> = Map::with_capacity_and_hasher(100, BuildHasherDefault::default());
        while let Some(nid) = params.pop() {
            rcs.entry(nid).and_modify(|rc| *rc += 1).or_insert_with(|| {
                if !realized_nodes.contains(&nid) {
                    params.extend(self.graph.nodes[nid].1.parameters());
                }
                1
            });
        }
        // Order them using rcs reference counts
        let mut order = Vec::new();
        let mut internal_rcs: Map<TensorId, u32> = Map::with_capacity_and_hasher(100, BuildHasherDefault::default());
        let mut params: Vec<TensorId> = to_eval.iter().copied().collect();
        while let Some(nid) = params.pop() {
            if let Some(&rc) = rcs.get(&nid) {
                if rc == *internal_rcs.entry(nid).and_modify(|rc| *rc += 1).or_insert(1) {
                    order.push(nid);
                    if !realized_nodes.contains(&nid) {
                        params.extend(self.graph.nodes[nid].1.parameters());
                    }
                }
            }
        }
        order.reverse();
        //println!("Order {order:?}");
        //println!("To eval {to_eval:?}");

        debug_assert!(!order.is_empty());
        debug_assert!(!to_eval.is_empty());

        if self.debug.perf() {
            println!(
                "Runtime realize graph order for {}/{} tensors with gradient_tape={}",
                order.len(),
                usize::from(self.graph.nodes.len()),
                self.graph.gradient_tape.is_some(),
            );
        }

        self.realize_with_order(rcs, realized_nodes, &order, &to_eval)?;

        // Delete all unnecessary nodes no longer needed after realization
        let mut to_release = Vec::new();
        if let Some(tape) = self.graph.gradient_tape.as_ref() {
            for &nid in &to_eval {
                if !tape.contains(&nid) {
                    let dtype = self.dtype(nid);
                    let shape = self.shape(nid).into();
                    self.graph.shapes.insert(nid, shape);
                    to_release.extend(self.graph[nid].parameters());
                    self.graph.nodes[nid].1 = Node::Leaf { dtype };
                }
            }
            let to_remove = self.graph.release(&to_release);
            deallocate_tensors(&to_remove, &mut self.pools);
        } else {
            for &nid in &to_eval {
                self.graph.add_shape(nid);
                let dtype = self.dtype(nid);
                to_release.extend(self.graph[nid].parameters());
                self.graph[nid] = Node::Leaf { dtype };
            }
            let to_remove = self.graph.release(&to_release);
            deallocate_tensors(&to_remove, &mut self.pools);
        }

        #[cfg(debug_assertions)]
        {
            let realized_nodes: Set<TensorId> =
                self.pools.iter().flat_map(|pool| pool.buffer_map.keys()).copied().collect();
            debug_assert!(realized_nodes.is_superset(&to_eval));
        }

        Ok(())
    }

    pub fn realize_all(&mut self) -> Result<(), ZyxError> {
        let realized_nodes: Set<TensorId> =
            self.pools.iter().flat_map(|pool| pool.buffer_map.keys()).copied().collect();

        if self.devices.is_empty() {
            self.initialize_devices()?;
        }

        let mut rcs: Map<TensorId, u32> = Map::with_capacity_and_hasher(100, BuildHasherDefault::default());
        for (_, node) in self.graph.nodes.values() {
            for nid in node.parameters() {
                rcs.entry(nid).and_modify(|rc| *rc += 1).or_insert(1);
            }
        }

        let mut to_eval = Set::with_hasher(BuildHasherDefault::new()); // TODO
        for (id, (rc, _)) in self.graph.nodes.iter() {
            if let Some(graph_rc) = rcs.get(&id) {
                if rc > graph_rc {
                    to_eval.insert(id);
                }
            } else {
                to_eval.insert(id);
            }
        }
        for id in &to_eval {
            rcs.entry(*id).and_modify(|rc| *rc += 1).or_insert(1);
        }

        // Order them using rcs reference counts
        let mut order = Vec::new();
        let mut internal_rcs: Map<TensorId, u32> = Map::with_capacity_and_hasher(100, BuildHasherDefault::default());
        let mut params: Vec<TensorId> = to_eval.iter().copied().collect();
        while let Some(nid) = params.pop() {
            if let Some(&rc) = rcs.get(&nid) {
                if rc == *internal_rcs.entry(nid).and_modify(|rc| *rc += 1).or_insert(1) {
                    order.push(nid);
                    if !realized_nodes.contains(&nid) {
                        params.extend(self.graph.nodes[nid].1.parameters());
                    }
                }
            }
        }
        order.reverse();
        //println!("Order {order:?}");
        //println!("To eval {to_eval:?}");

        debug_assert!(!order.is_empty());
        debug_assert!(!to_eval.is_empty());

        if self.debug.perf() {
            println!(
                "Runtime realize graph order for {}/{} tensors with gradient_tape={}",
                order.len(),
                usize::from(self.graph.nodes.len()),
                self.graph.gradient_tape.is_some(),
            );
        }

        self.realize_with_order(rcs, realized_nodes, &order, &to_eval)?;

        // Delete all unnecessary nodes no longer needed after realization
        let mut to_release = Vec::new();
        if let Some(tape) = self.graph.gradient_tape.as_ref() {
            for &nid in &to_eval {
                if !tape.contains(&nid) {
                    let dtype = self.dtype(nid);
                    let shape = self.shape(nid).into();
                    self.graph.shapes.insert(nid, shape);
                    to_release.extend(self.graph[nid].parameters());
                    self.graph.nodes[nid].1 = Node::Leaf { dtype };
                }
            }
            let to_remove = self.graph.release(&to_release);
            deallocate_tensors(&to_remove, &mut self.pools);
        } else {
            for &nid in &to_eval {
                self.graph.add_shape(nid);
                let dtype = self.dtype(nid);
                to_release.extend(self.graph[nid].parameters());
                self.graph[nid] = Node::Leaf { dtype };
            }
            let to_remove = self.graph.release(&to_release);
            deallocate_tensors(&to_remove, &mut self.pools);
        }

        #[cfg(debug_assertions)]
        {
            let realized_nodes: Set<TensorId> =
                self.pools.iter().flat_map(|pool| pool.buffer_map.keys()).copied().collect();
            debug_assert!(realized_nodes.is_superset(&to_eval));
        }

        Ok(())
    }
}
