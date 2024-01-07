extern crate alloc;

use cl3;
use cl3::memory::CL_MEM_READ_ONLY;
use alloc::boxed::Box;
use rand;
use alloc::collections::{BTreeMap, BTreeSet};
use alloc::vec::Vec;
use core::ffi::c_void;
use cl3::command_queue::CL_NON_BLOCKING;
use cl3::error_codes::ClError;
use zyx_core::backend::BufferView;
use zyx_core::dtype::DType;
use zyx_core::node::Node;
use zyx_core::scalar::Scalar;
use zyx_core::shape::Shape;
use zyx_core::tensor::{Id, id};

// We need to remember repeating parts of graph, then find the best way to evaluate them.
// Repeating means single training/inference loop.

struct Buffer {
    mem: *mut c_void,
    event: *mut c_void,
}

impl Drop for Buffer {
    fn drop(&mut self) {
        unsafe { cl3::memory::release_mem_object(self.mem) }.unwrap();
        unsafe { cl3::event::release_event(self.event) }.unwrap();
    }
}

pub(super) struct Inner {
    rng: rand::rngs::SmallRng,
    rcs: Vec<u8>,
    order: Vec<Id>,
    nodes: Vec<Node>,
    leafs: BTreeSet<Id>, // these do not need backward graph
    context: *mut c_void,
    devices: BTreeSet<*mut c_void>,
    queues: Box<[*mut c_void]>,
    queue_id: usize,
    buffers: BTreeMap<Id, Buffer>,
}

impl Inner {
    pub(super) fn new() -> Result<Self, ClError> {
        use cl3::ext::CL_DEVICE_TYPE_ALL;
        let platform_ids = cl3::platform::get_platform_ids()?;
        let Some(platform) = platform_ids.get(0) else {
            panic!("There are no available OpenCL platforms.");
        };
        let platform = *platform;
        #[cfg(feature = "debug1")]
        std::println!(
            "Using OpenCL platform: {}",
            String::from_utf8(cl3::platform::get_platform_data(
                platform,
                cl3::ext::CL_PLATFORM_NAME
            )?)
                .unwrap()
        );
        let device_ids = cl3::device::get_device_ids(platform, CL_DEVICE_TYPE_ALL)?;
        #[cfg(feature = "debug1")]
        std::println!("Using devices:");
        #[cfg(feature = "debug1")]
        for dev in &device_ids {
            std::println!(
                "{}",
                String::from_utf8(cl3::device::get_device_data(
                    *dev,
                    cl3::ext::CL_DEVICE_NAME
                )?)
                    .unwrap()
            );
        }
        let context = cl3::context::create_context(
            &device_ids,
            core::ptr::null(),
            None,
            core::ptr::null_mut(),
        )?;
        // This makes our code asynchronous. Creating graph would actually make us 2 times slower (can be optimized),
        // if we couldn't execute kernels asynchronously. We don't need this to be huge. 2 seems to
        // be plenty. And lower values also lower memory usage.
        let queues_per_device: u32 = 8; //device_ids.iter().map(|dev| cl3::device::get_device_info(*dev, CL_DEVICE_MAX_ON_DEVICE_QUEUES).unwrap().into()).min().unwrap();
        #[cfg(feature = "debug1")]
        std::println!("Working with {queues_per_device} queues on each device.");
        let queues = (0..queues_per_device)
            .flat_map(|_| {
                device_ids.iter().map(|dev| {
                    unsafe { cl3::command_queue::create_command_queue(context, *dev, 0) }.unwrap()
                })
            })
            .collect();
        let mut devices = BTreeSet::new();
        for dev in device_ids {
            devices.insert(dev);
        }
        use rand::SeedableRng;
        Ok(Self {
            rng: rand::rngs::SmallRng::seed_from_u64(
                420_694_206_942_069),
            rcs: Vec::new(),
            order: Vec::new(),
            nodes: Vec::new(),
            leafs: BTreeSet::new(),
            context,
            devices,
            queues,
            queue_id: 0,
            buffers: BTreeMap::new(),
        })
    }

    pub(super) fn randn(&mut self, shape: Shape, dtype: DType) -> Id {
        let shape: Shape = shape.into();
        use rand::Rng;
        let n = shape.numel();
        let mut rng = self.rng.clone();
        use rand::distributions::Standard;
        let data = match dtype {
            DType::F32 => self.push(Node::IterF32(Box::new((0..n).map(move |_| rng.sample(Standard))), shape)),
            DType::I32 => self.push(Node::IterI32(Box::new((0..n).map(move |_| rng.sample(Standard))), shape)),
        };
        // change the state of the random seed in rng
        for _ in 0..n {
            self.rng.sample::<f32, _>(rand::distributions::Standard);
        }
        data
    }

    pub(super) fn uniform<T: Scalar>(&mut self, shape: Shape, low: T, high: T) -> Id {
        match T::dtype() {
            DType::F32 => self.push(Node::UniformF32(shape)),
            DType::I32 => self.push(Node::UniformI32(shape))
        }
    }

    pub(super) fn full<T: Scalar>(&mut self, shape: Shape, value: T) -> Id {
        match T::dtype() {
            DType::F32 => self.push(Node::IterF32(Box::new(core::iter::repeat(value.into_f32()).take(shape.numel())), shape)),
            DType::I32 => self.push(Node::IterI32(Box::new(core::iter::repeat(value.into_i32()).take(shape.numel())), shape)),
        }
    }

    pub(super) fn eye(&mut self, n: usize, dtype: DType) -> Id {
        match dtype {
            DType::F32 => self.push(Node::IterF32(Box::new((0..n).flat_map(move |i| (0..n).map(move |j| if j == i { 1. } else { 0. }))), [n, n].into())),
            DType::I32 => self.push(Node::IterI32(Box::new((0..n).flat_map(move |i| (0..n).map(move |j| if j == i { 1 } else { 0 }))), [n, n].into())),
        }
    }

    pub(super) fn shape(&self, mut x: Id) -> &Shape {
        loop {
            let node = self.nodes.get(x.i()).unwrap();
            match node {
                Node::LeafF32(shape)
                | Node::IterF32(_, shape)
                | Node::UniformF32(shape)
                | Node::LeafI32(shape)
                | Node::IterI32(_, shape)
                | Node::UniformI32(shape)
                | Node::Reshape(_, shape)
                | Node::Expand(_, shape)
                | Node::Permute(.., shape)
                | Node::Sum(.., shape)
                | Node::Max(.., shape) => return shape,
                _ => x = node.parameters().next().unwrap(),
            }
        }
    }

    pub(super) fn dtype(&self, mut x: Id) -> DType {
        loop {
            let node = self.nodes.get(x.i()).unwrap();
            match node {
                Node::LeafF32(..)
                | Node::IterF32(..)
                | Node::UniformF32(..)
                | Node::CastF32(..) => return DType::F32,
                Node::LeafI32(..)
                | Node::IterI32(..)
                | Node::UniformI32(..)
                | Node::CastI32(..) => return DType::I32,
                _ => x = node.parameters().next().unwrap(),
            }
        }
    }

    pub(super) fn release(&mut self, x: Id) {
        let mut params = Vec::with_capacity(10);
        params.push(x);
        while let Some(p) = params.pop() {
            self.rcs[p.i()] -= 1;
            if self.rcs[p.i()] == 0 {
                params.extend(self.nodes[p.i()].parameters());
                self.leafs.remove(&p);
                self.buffers.remove(&p);
            }
        }
    }

    pub(super) fn retain(&mut self, x: Id) { self.rcs[x.i()] += 1; }

    pub(super) fn load(&mut self, x: Id) -> BufferView {
        // This may need to evaluate, therefore we need to take mutable reference to self
        if !self.buffers.contains_key(&x) {
            // TODO also check if these are only movements ops,
            // in which case we can directly return iterator with view
            self.evaluate(BTreeSet::from([x]));
        }
        todo!()
    }

    pub(super) fn push(&mut self, node: Node) -> Id {
        for nid in node.parameters() {
            self.rcs[nid.i()] += 1;
        }
        let (i, new_node) = if let Some(i) = self.rcs.iter().position(|rc| *rc == 0) {
            (id(i), false)
        } else {
            (id(self.rcs.len()), true)
        };
        if new_node {
            self.rcs.push(1);
            self.nodes.push(node);
            self.order.push(i);
        } else {
            self.rcs[i.i()] = 1;
            self.nodes[i.i()] = node;
            // Keep the ordering, this is probably as fast as it gets
            let prev = self.order[i.i()];
            for x in self.order.iter_mut() {
                if *x > prev {
                    *x -= 1;
                }
            }
            self.order[i.i()] = id(self.order.len() - 1);
        }
        i
    }

    pub(super) fn set_leaf(&mut self, x: Id) { self.leafs.insert(x); }

    pub(super) fn backward(&mut self, x: Id, sources: &BTreeSet<Id>) -> BTreeMap<Id, Id> {
        fn build_topo(x: Id, sources: &BTreeSet<Id>, nodes: &[Node], order: &[Id]) -> Vec<Id> {
            // First we need to know which nodes require gradient
            let mut req_grad = BTreeSet::new();
            for i in order {
                for p in nodes[i.i()].parameters() {
                    if sources.contains(&p) || req_grad.contains(&p) {
                        req_grad.insert(i);
                    }
                }
            }
            // Here we build topo
            let mut topo = Vec::new();
            if !req_grad.contains(&x) {
                return topo
            }
            let mut visited = BTreeSet::new();
            let mut params = alloc::collections::VecDeque::with_capacity(32);
            params.push_back(x);
            while let Some(p) = params.pop_front() {
                if req_grad.contains(&p) {
                    topo.push(p);
                    if visited.insert(p) {
                        params.extend(nodes[p.i()].parameters());
                    }
                }
            }
            topo
        }

        let topo = build_topo(x, sources, &self.nodes, &self.order);
        let req_grad: BTreeSet<Id> = topo.iter().copied().chain(sources.iter().copied()).collect();
        //extern crate std;
        //std::println!("These nodes require gradient: {:?}", req_grad);
        // Node -> Grad
        let mut grads: BTreeMap<Id, Id> = BTreeMap::new();
        // Initial gradient of ones
        let grad1 = match self.dtype(x) {
            DType::F32 => self.push(Node::IterF32(Box::new([1.].into_iter()), self.shape(x).clone())),
            DType::I32 => self.push(Node::IterF32(Box::new([1.].into_iter()), self.shape(x).clone())),
        };
        grads.insert(x, self.push(Node::Expand(grad1, self.shape(x).clone())));
        self.release(grad1);
        // backpropagate
        // TODO this is not very clean code. Can we make it cleaner?
        for nid in topo {
            let grad = grads[&nid];
            match self.nodes[nid.i()] {
                Node::LeafF32(..)
                | Node::LeafI32(..)
                | Node::UniformF32(..)
                | Node::UniformI32(..)
                | Node::IterF32(..)
                | Node::IterI32(..) => {}
                Node::Add(x, y) => {
                    if req_grad.contains(&x) && grads.insert(x, grad).is_none() {
                        self.retain(grad);
                    }
                    if req_grad.contains(&y) && grads.insert(y, grad).is_none() {
                        self.retain(grad);
                    }
                }
                Node::Sub(x, y) => {
                    if req_grad.contains(&x) && grads.insert(x, grad).is_none() {
                        self.retain(grad);
                    }
                    if req_grad.contains(&y) && !grads.contains_key(&y) {
                        grads.insert(y, self.push(Node::Neg(grad)));
                    }
                }
                Node::Mul(x, y) => {
                    if req_grad.contains(&x) && !grads.contains_key(&x) {
                        grads.insert(x, self.push(Node::Mul(y, grad)));
                    }
                    if req_grad.contains(&y) && !grads.contains_key(&y) {
                        grads.insert(y, self.push(Node::Mul(x, grad)));
                    }
                }
                Node::Div(x, y) => {
                    if req_grad.contains(&x) && !grads.contains_key(&x) {
                        grads.insert(x, self.push(Node::Div(grad, y)));
                    }
                    if req_grad.contains(&y) && !grads.contains_key(&y) {
                        // -grad*x/(y^2)
                        let two = match self.dtype(y) {
                            DType::F32 => self.push(Node::IterF32(Box::new([2.].into_iter()), 1.into())),
                            DType::I32 => self.push(Node::IterI32(Box::new([2].into_iter()), 1.into())),
                        };
                        let two_e = self.push(Node::Expand(two, self.shape(y).clone()));
                        self.release(two);
                        let two_2 = self.push(Node::Pow(y, two_e));
                        self.release(two_e);
                        let temp = self.push(Node::Mul(x, grad));
                        let temp_neg = self.push(Node::Neg(temp));
                        self.release(temp);
                        let y_grad = self.push(Node::Div(temp_neg, two_2));
                        self.release(temp_neg);
                        self.release(two_2);
                        grads.insert(y, y_grad);
                    }
                }
                Node::Pow(x, y) => {
                    if req_grad.contains(&x) && !grads.contains_key(&x) {
                        // grad * y * x.pow(y-1)
                        let one = match self.dtype(y) {
                            DType::F32 => self.push(Node::IterF32(Box::new([1.].into_iter()), 1.into())),
                            DType::I32 => self.push(Node::IterI32(Box::new([1].into_iter()), 1.into())),
                        };
                        let one1 = self.push(Node::Expand(one, self.shape(y).clone()));
                        self.release(one);
                        let y_1 = self.push(Node::Sub(y, one1));
                        self.release(one1);
                        let pow_y_1 = self.push(Node::Pow(x, y_1));
                        self.release(y_1);
                        let y_mul = self.push(Node::Mul(y, pow_y_1));
                        self.release(pow_y_1);
                        let x_grad = self.push(Node::Mul(grad, y_mul));
                        self.release(y_mul);
                        grads.insert(x, x_grad);
                    }
                    if req_grad.contains(&y) && !grads.contains_key(&y) {
                        // grad * x.pow(y) * ln(x)
                        let temp1 = self.push(Node::Ln(x));
                        let temp2 = self.push(Node::Mul(nid, temp1));
                        self.release(temp1);
                        let y_grad = self.push(Node::Mul(grad, temp2));
                        self.release(temp2);
                        grads.insert(y, y_grad);
                    }
                }
                Node::Cmplt(_x, _y) => {
                    todo!()
                }
                Node::ReLU(x) => {
                    // TODO is grads.contains_key useless for unary ops?
                    grads.entry(x).or_insert_with(|| {
                        let zero = match self.dtype(x) {
                            DType::F32 => self.push(Node::IterF32(Box::new([0.].into_iter()), 1.into())),
                            DType::I32 => self.push(Node::IterI32(Box::new([0].into_iter()), 1.into())),
                        };
                        let zeros = self.push(Node::Expand(zero, self.shape(x).clone()));
                        self.release(zero);
                        let zl = self.push(Node::Cmplt(zeros, x));
                        self.release(zeros);
                        let x_grad = self.push(Node::Mul(zl, grad));
                        self.release(zl);
                        x_grad
                    });
                }
                Node::Exp(x) => {
                    grads
                        .entry(x)
                        .or_insert_with(|| self.push(Node::Mul(nid, grad)));
                }
                Node::Ln(x) => {
                    grads
                        .entry(x)
                        .or_insert_with(|| self.push(Node::Div(grad, x)));
                }
                Node::Sin(x) => {
                    grads.entry(x).or_insert_with(|| {
                        let x_temp = self.push(Node::Cos(x));
                        let x_grad = self.push(Node::Mul(x_temp, grad));
                        self.release(x_temp);
                        x_grad
                    });
                }
                Node::Cos(x) => {
                    grads.entry(x).or_insert_with(|| {
                        let x_temp1 = self.push(Node::Sin(x));
                        let x_temp = self.push(Node::Neg(x_temp1));
                        self.release(x_temp1);
                        let x_grad = self.push(Node::Mul(x_temp, grad));
                        self.release(x_temp);
                        x_grad
                    });
                }
                Node::Sqrt(x) => {
                    grads.entry(x).or_insert_with(|| {
                        // x_grad = grad/(2*sqrt(x))
                        let x_shape = self.shape(x).clone();
                        let two1 = self.push(Node::IterF32(Box::new([2.].into_iter()), 1.into()));
                        let two2 = self.push(Node::Expand(two1, x_shape));
                        self.release(two1);
                        let x_temp = self.push(Node::Mul(two2, nid));
                        self.release(two2);
                        let x_grad = self.push(Node::Div(grad, x_temp));
                        self.release(x_temp);
                        x_grad
                    });
                }
                Node::CastF32(x) => {
                    grads
                        .entry(x)
                        .or_insert_with(|| match self.dtype(x) {
                            DType::F32 => self.push(Node::CastF32(grad)),
                            DType::I32 => self.push(Node::CastI32(grad)),
                        });
                }
                Node::CastI32(x) => {
                    grads
                        .entry(x)
                        .or_insert_with(|| match self.dtype(x) {
                            DType::F32 => self.push(Node::CastF32(grad)),
                            DType::I32 => self.push(Node::CastI32(grad)),
                        });
                }
                Node::Neg(x) => {
                    grads.entry(x).or_insert_with(|| self.push(Node::Neg(grad)));
                }
                Node::Tanh(x) => {
                    grads.entry(x).or_insert_with(|| {
                        // 1 - tanh^2(x)
                        let shape = self.shape(x).clone();
                        let (two1, one1) = match self.dtype(x) {
                            DType::F32 => {
                                (self.push(Node::IterF32(Box::new([2.].into_iter()), 1.into())),
                                 self.push(Node::IterF32(Box::new([1.].into_iter()), 1.into())))
                            }
                            DType::I32 => {
                                (self.push(Node::IterI32(Box::new([2].into_iter()), 1.into())),
                                 self.push(Node::IterI32(Box::new([1].into_iter()), 1.into())))
                            }
                        };
                        let two2 = self.push(Node::Expand(two1, shape.clone()));
                        self.release(two1);
                        let two = self.push(Node::Pow(nid, two2));
                        self.release(two2);
                        let one2 = self.push(Node::Expand(one1, shape));
                        self.release(one1);
                        let one = self.push(Node::Sub(one2, two));
                        self.release(one2);
                        self.release(two);
                        let x_grad = self.push(Node::Mul(one, grad));
                        self.release(one);
                        x_grad
                    });
                }
                Node::Reshape(x, ..) => {
                    grads
                        .entry(x)
                        .or_insert_with(|| self.push(Node::Reshape(grad, self.shape(x).clone())));
                }
                Node::Expand(x, ref shape) => {
                    if !grads.contains_key(&x) {
                        let org_shape = self.shape(x).clone();
                        let axes = org_shape.expand_axes(shape);
                        let temp = self.push(Node::Sum(grad, axes, org_shape.clone()));
                        let x_grad = self.push(Node::Reshape(temp, org_shape));
                        self.release(temp);
                        grads.insert(x, x_grad);
                    }
                }
                Node::Permute(x, ref axes, _) => {
                    if !grads.contains_key(&x) {
                        let shape = self.shape(x);
                        grads.insert(x, self.push(Node::Permute(grads[&nid], axes.argsort(), shape.clone())));
                    }
                }
                Node::Sum(x, ..) => {
                    grads
                        .entry(x)
                        .or_insert_with(|| self.push(Node::Expand(grad, self.shape(x).clone())));
                }
                Node::Max(x, ..) => {
                    grads.entry(x).or_insert_with(|| {
                        // x_grad = (1 - (x < z.expand(x.shape()))) * grad
                        let x_shape = self.shape(x).clone();
                        let z_temp = self.push(Node::Expand(nid, x_shape.clone()));
                        let cmp_t = self.push(Node::Cmplt(x, z_temp));
                        self.release(z_temp);
                        let one1 = self.push(Node::IterF32(Box::new([1.].into_iter()), 1.into()));
                        let one2 = self.push(Node::Expand(one1, x_shape));
                        self.release(one1);
                        let max_1s = self.push(Node::Sub(one2, cmp_t));
                        self.release(one2);
                        self.release(cmp_t);
                        let x_grad = self.push(Node::Mul(max_1s, grad));
                        self.release(max_1s);
                        x_grad
                    });
                }
            }
        }
        grads.into_iter().flat_map(|x| if sources.contains(&x.0) { Some(x) } else { self.release(x.1); None }).collect()
    }

    fn queue(&mut self) -> *mut c_void {
        let res = self.queues[self.queue_id];
        self.queue_id = (self.queue_id + 1) % self.queues.len();
        res
    }

    fn store<T>(&mut self, iter: Box<dyn Iterator<Item = T>>) -> Buffer {
        let data: Vec<T> = iter.collect();
        let size = data.len() * core::mem::size_of::<T>();
        let mem = unsafe {
            cl3::memory::create_buffer(
                self.context,
                CL_MEM_READ_ONLY,
                size,
                core::ptr::null_mut(),
            )
        }.unwrap();
        let queue = self.queue();
        let event = unsafe {
            cl3::command_queue::enqueue_write_buffer(
                queue,
                mem,
                CL_NON_BLOCKING,
                0,
                size,
                data.as_ptr().cast(),
                0,
                core::ptr::null(),
            )
        }
            .unwrap();
        cl3::event::wait_for_events(&[event]).unwrap();
        Buffer { mem, event }
    }

    fn evaluate(&mut self, nodes: BTreeSet<Id>) {
        // TODO we are probably going too many times back and forth in the graph.
        // First we go back to create graph of all nodes that need to be evaluated.
        // Then we go forward to find which nodes are kernel subgraphs.
        // Then we go back again to create subgraphs for individual kernels.
        // Then we go forward again to create the kernel itself.

        // Find all needed parameters for calculation of nodes
        let mut params: Vec<Id> = nodes.iter().copied().collect();
        let mut rcs: BTreeMap<Id, u8> = BTreeMap::new();
        while let Some(nid) = params.pop() {
            rcs.entry(nid)
                .and_modify(|rc| *rc += 1)
                .or_insert_with(|| {
                    params.extend(self.nodes[nid.i()].parameters());
                    1
                });
        }
        let mut order: Vec<Id> = rcs.keys().copied().collect();
        order.sort_by_cached_key(|nid| self.order[nid.i()]);

        for nid in order {
            match &mut self.nodes[nid.i()] {
                Node::LeafF32(..)
                | Node::LeafI32(..)
                | Node::UniformF32(..)
                | Node::UniformI32(..)
                | Node::CastF32(..)
                | Node::CastI32(..)
                | Node::Neg(..)
                | Node::ReLU(..)
                | Node::Sin(..)
                | Node::Cos(..)
                | Node::Ln(..)
                | Node::Exp(..)
                | Node::Tanh(..)
                | Node::Sqrt(..)
                | Node::Add(..)
                | Node::Sub(..)
                | Node::Mul(..)
                | Node::Div(..)
                | Node::Pow(..)
                | Node::Cmplt(..)
                | Node::Reshape(..)
                | Node::Permute(..)
                | Node::Sum(..)
                | Node::Max(..) => {}
                Node::IterF32(_, shape) => {
                    let mut new_node = Node::LeafF32(shape.clone());
                    core::mem::swap(&mut self.nodes[nid.i()], &mut new_node);
                    if let Node::IterF32(iter, _) = new_node {
                        self.store(iter);
                    }
                }
                Node::IterI32(_, shape) => {
                    let mut new_node = Node::LeafI32(shape.clone());
                    core::mem::swap(&mut self.nodes[nid.i()], &mut new_node);
                    if let Node::IterI32(iter, _) = new_node {
                        self.store(iter);
                    }
                }
                Node::Expand(x, _) => {
                    // if reduce operation preceded expand, we call evaluate_buffer
                    let mut params = alloc::vec![*x];
                    while let Some(p) = params.pop() {
                        if matches!(self.nodes[p.i()], Node::Sum(..) | Node::Max(..)) {
                            evaluate_buffer(&self.nodes, p);
                            break;
                        }
                        params.extend(self.nodes[p.i()].parameters());
                    }
                }
            }
            // TODO release nodes that are no longer needed.
            // And release intermediate buffers.
        }

        // Release parts of graph that are not needed for backpropagation
        //while let Some(leaf) = self.leafs.pop_last() {
            //std::println!("Releasing leaf {leaf}");
            //let mut node = Node::Leaf(self.dtype(leaf));
            //let shape = self.shape(leaf);
            //self.shapes.insert(leaf, shape);
            //core::mem::swap(self.nodes.get_mut(leaf.i()).unwrap(), &mut node);
            //for nid in &*node.parameters() {
                //self.release(*nid);
            //}
        //}

        /// This function evaluates concrete buffer that we know can be directly evaluated,
        /// that is it all of it's leafs are already evaluated.
        fn evaluate_buffer(nodes: &[Node], x: Id) {
            // create list of nodes that need to be evaluated
        }
    }
}
