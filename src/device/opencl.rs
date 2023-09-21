extern crate alloc;
use super::Storage;
use crate::{
    node_id::NodeId,
    dtype::DType,
    graph::Node,
    shape::Shape,
    OutOfMemoryError,
};
use alloc::{
    boxed::Box,
    collections::{BTreeMap, BTreeSet},
    ffi::CString,
    format,
    rc::Rc,
    string::String,
    vec::Vec,
};
use cl3::{memory::CL_MEM_READ_ONLY, program::CL_PROGRAM_BUILD_LOG, types::CL_NON_BLOCKING, ext::{CL_DEVICE_GLOBAL_MEM_SIZE, CL_DEVICE_MAX_WORK_GROUP_SIZE, CL_DEVICE_LOCAL_MEM_SIZE}, error_codes::ClError};
use core::{ffi::c_void, mem::size_of};

// TODO deduplicate everything

#[derive(Debug)]
pub(crate) struct OpenCLDev {
    context: *mut c_void,
    devices: BTreeMap<*mut c_void, DeviceInfo>,
    queues: Box<[*mut c_void]>,
    queue_id: usize,
    programs: BTreeMap<String, *mut c_void>,
    #[allow(dead_code)]
    max_mem: usize,
}

#[derive(Debug)]
struct DeviceInfo {
    #[allow(dead_code)]
    max_lws: usize,
    #[allow(dead_code)]
    local_mem_size: usize,
}

impl DeviceInfo {
    fn new(dev: *mut c_void) -> Result<Self, ClError> {
        let max_lws: usize = cl3::device::get_device_info(dev, CL_DEVICE_MAX_WORK_GROUP_SIZE)?.into();
        let local_mem_size: u64 = cl3::device::get_device_info(dev, CL_DEVICE_LOCAL_MEM_SIZE)?.into();
        Ok(Self {
            max_lws,
            local_mem_size: usize::try_from(local_mem_size).unwrap(),
        })
    }
}

#[cfg(feature = "opencl")]
impl Drop for OpenCLDev {
    fn drop(&mut self) {
        unsafe { cl3::context::release_context(self.context) }.unwrap();
        for device in self.devices.keys() {
            unsafe { cl3::device::release_device(*device) }.unwrap();
        }
        for queue in &*self.queues {
            unsafe { cl3::command_queue::release_command_queue(*queue) }.unwrap();
        }
        for program in self.programs.values() {
            unsafe { cl3::program::release_program(*program) }.unwrap();
        }
    }
}

enum Op {
    Load {
        dtype: DType,
        shape: Shape,
        parameters: BTreeSet<NodeId>,
        idx: String,
        storage: ClStorage,
    },
    Basic {
        dtype: DType,
        shape: Shape,
        parameters: BTreeSet<NodeId>,
        kernel: String,
    },
    Custom {
        dtype: DType,
        shape: Shape,
        parameters: BTreeSet<NodeId>,
        kernel: String,
        prefix: String,
        suffix: String,
        gws: Box<[usize]>,
        lws: Option<Box<[usize]>>,
        skip_params: Box<[NodeId]>,
        idx: String,
    },
}

#[derive(Debug)]
pub(crate) struct ClStorage(Rc<ClBuf>);

impl ClStorage {
    pub(super) fn new(buffer: *mut c_void, event: *mut c_void) -> Self {
        Self(Rc::new(ClBuf { buffer, event }))
    }

    pub(super) fn buffer(&self) -> *mut c_void {
        self.0.buffer
    }

    pub(super) fn event(&self) -> *mut c_void {
        self.0.event
    }
}

// This is nice way to make sure deallocation is ok, perhaps could be without Rc,
// especially since opencl implements it's own reference counting, but its hard.
#[derive(Debug)]
struct ClBuf {
    buffer: *mut c_void,
    event: *mut c_void,
}

impl Drop for ClBuf {
    fn drop(&mut self) {
        cl3::event::wait_for_events(&[self.event]).unwrap();
        unsafe { cl3::event::release_event(self.event) }.unwrap();
        unsafe { cl3::memory::release_mem_object(self.buffer) }.unwrap();
    }
}

impl Clone for ClStorage {
    fn clone(&self) -> Self {
        Self(Rc::clone(&self.0))
    }
}

impl Op {
    fn is_custom(&self) -> bool {
        matches!(self, Op::Custom { .. })
    }

    fn dtype(&self) -> DType {
        match self {
            Op::Load { dtype, .. } | Op::Basic { dtype, .. } | Op::Custom { dtype, .. } => *dtype,
        }
    }

    fn shape(&self) -> &Shape {
        match self {
            Op::Load { shape, .. } | Op::Basic { shape, .. } | Op::Custom { shape, .. } => shape,
        }
    }

    fn parameters(&self) -> &BTreeSet<NodeId> {
        match self {
            Op::Load { parameters, .. }
            | Op::Basic { parameters, .. }
            | Op::Custom { parameters, .. } => parameters,
        }
    }

    fn from_const(id: NodeId, dtype: DType, shape: Shape, storage: ClStorage) -> Self {
        Op::Load {
            dtype,
            shape,
            idx: "IDX".into(),
            storage,
            parameters: BTreeSet::from([id]),
        }
    }
}

impl OpenCLDev {
    pub(super) fn new() -> Result<Self, cl3::error_codes::ClError> {
        use cl3::ext::CL_DEVICE_TYPE_ALL;
        let platform_ids = cl3::platform::get_platform_ids()?;
        let Some(platform) = platform_ids.get(0) else {
            panic!("There are no available OpenCL platforms.");
        };
        let platform = *platform;
        #[cfg(feature = "debug1")]
        std::println!(
            "Using OpenCL platform: {}",
            alloc::string::String::from_utf8(cl3::platform::get_platform_data(
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
                alloc::string::String::from_utf8(cl3::device::get_device_data(
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
        let max_mem: u64 = cl3::device::get_device_info(device_ids[0], CL_DEVICE_GLOBAL_MEM_SIZE)?.into();
        let mut devices = BTreeMap::new();
        for dev in device_ids {
            devices.insert(dev, DeviceInfo::new(dev)?);
        }
        Ok(Self {
            queues,
            queue_id: 0,
            context,
            devices,
            programs: BTreeMap::new(),
            max_mem: (max_mem as f64 * 0.8) as usize, // uses 80% of available memory by default
        })
    }

    // Get access to next empty/least pressured queue
    pub(super) fn queue(&mut self) -> *mut c_void {
        let res = self.queues[self.queue_id];
        self.queue_id = (self.queue_id + 1) % self.queues.len();
        res
    }
    
    // Get maximum local work size (work group size) for device
    // that will run next queue
    fn max_lws(&self) -> usize {
        //self.devices[self.devices.keys().nth((self.queue_id + 1) * self.devices.len() / self.queues.len()).unwrap()].max_lws
        256
    }

    // TODO this can be cleaned up
    #[allow(clippy::too_many_lines)]
    pub(super) fn realize(
        &mut self,
        graph: &mut BTreeMap<NodeId, (usize, Node)>, // id, refcount and Node
        order: &[NodeId],                            // recommended realization order
        nodes: &BTreeSet<NodeId>,                    // which nodes need to be realized
    ) -> Result<(), OutOfMemoryError> {
        let mut buffers: BTreeMap<NodeId, Op> = BTreeMap::new();
        for node_id in order {
            //std::println!("Node id {}", node_id);
            match graph[node_id].1 {
                Node::Expand(x, _) => {
                    if buffers[&x]
                        .parameters()
                        .iter()
                        .any(|x| buffers[x].is_custom())
                    {
                        graph.get_mut(&x).unwrap().1 =
                            Node::Const(self.eval(x, &mut buffers, order)?);
                    }
                }
                Node::Add(x, y)
                | Node::Sub(x, y)
                | Node::Mul(x, y)
                | Node::Div(x, y)
                | Node::Pow(x, y) => {
                    if buffers[&x]
                        .parameters()
                        .iter()
                        .any(|x| buffers[x].is_custom())
                        && buffers[&y]
                            .parameters()
                            .iter()
                            .any(|x| buffers[x].is_custom())
                    {
                        graph.get_mut(&y).unwrap().1 =
                            Node::Const(self.eval(y, &mut buffers, order)?);
                    }
                }
                Node::TDot(..) | Node::Permute(..) | Node::Sum(..) | Node::Max(..) => {
                    for param in &*graph[node_id].1.parameters() {
                        graph.get_mut(param).unwrap().1 =
                            Node::Const(self.eval(*param, &mut buffers, order)?);
                    }
                }
                _ => {}
            }
            match &graph.get(node_id).unwrap().1 {
                Node::None | Node::Leaf => panic!(),
                Node::Const(storage) => match storage {
                    Storage::OpenCLF32(shape, storage) => {
                        buffers.insert(
                            *node_id,
                            Op::from_const(*node_id, DType::F32, shape.clone(), storage.clone()),
                        );
                    }
                    Storage::OpenCLI32(shape, storage) => {
                        buffers.insert(
                            *node_id,
                            Op::from_const(*node_id, DType::I32, shape.clone(), storage.clone()),
                        );
                    }
                    _ => panic!("Opencl device can not access memory stored in other device."),
                },
                Node::StoreF32(data, shape) => {
                    let n = data.len() * size_of::<f32>();
                    let buffer = unsafe {
                        cl3::memory::create_buffer(
                            self.context,
                            CL_MEM_READ_ONLY,
                            n,
                            core::ptr::null_mut(),
                        )
                    }
                    .map_err(|_| OutOfMemoryError)?;
                    let queue = self.queue();
                    let event = unsafe {
                        cl3::command_queue::enqueue_write_buffer(
                            queue,
                            buffer,
                            CL_NON_BLOCKING,
                            0,
                            n,
                            data.as_ptr().cast(),
                            0,
                            core::ptr::null(),
                        )
                    }
                    .unwrap();
                    //cl3::command_queue::finish(queue).unwrap();
                    cl3::event::wait_for_events(&[event]).unwrap();
                    let dtype = DType::F32;
                    buffers.insert(
                        *node_id,
                        Op::from_const(
                            *node_id,
                            dtype,
                            shape.clone(),
                            ClStorage::new(buffer, event),
                        ),
                    );
                }
                Node::StoreI32(data, shape) => {
                    let n = data.len() * size_of::<i32>();
                    let buffer = unsafe {
                        cl3::memory::create_buffer(
                            self.context,
                            CL_MEM_READ_ONLY,
                            n,
                            core::ptr::null_mut(),
                        )
                    }
                    .map_err(|_| OutOfMemoryError)?;
                    let queue = self.queue();
                    let event = unsafe {
                        cl3::command_queue::enqueue_write_buffer(
                            queue,
                            buffer,
                            CL_NON_BLOCKING,
                            0,
                            n,
                            data.as_ptr().cast(),
                            0,
                            core::ptr::null(),
                        )
                    }
                    .unwrap();
                    //cl3::command_queue::finish(queue).unwrap();
                    cl3::event::wait_for_events(&[event]).unwrap();
                    let dtype = DType::I32;
                    buffers.insert(
                        *node_id,
                        Op::from_const(
                            *node_id,
                            dtype,
                            shape.clone(),
                            ClStorage::new(buffer, event),
                        ),
                    );
                }
                Node::Cast(x, dtype) => {
                    let dtype = *dtype;
                    let mut parameters = buffers[x].parameters().clone();
                    parameters.insert(*node_id);
                    buffers.insert(
                        *node_id,
                        Op::Basic {
                            dtype,
                            shape: buffers[x].shape().clone(),
                            parameters,
                            kernel: format!(
                                "  {} res{} = ({})res{};\n",
                                dtype.cl_type(),
                                node_id.i(),
                                dtype.cl_type(),
                                x.i()
                            ),
                        },
                    );
                }
                Node::Reshape(x, shape) => {
                    let dtype = buffers[x].dtype();
                    let mut parameters = buffers[x].parameters().clone();
                    parameters.insert(*node_id);
                    buffers.insert(
                        *node_id,
                        Op::Basic {
                            dtype,
                            shape: shape.clone(),
                            parameters,
                            kernel: format!(
                                "  {} res{} = ({})res{};\n",
                                dtype.cl_type(),
                                node_id.i(),
                                dtype.cl_type(),
                                x.i()
                            ),
                        },
                    );
                }
                Node::Neg(x) => unary_op(*node_id, *x, &mut buffers, "-"),
                Node::Exp(x) => unary_op(*node_id, *x, &mut buffers, "exp"),
                Node::Ln(x) => unary_op(*node_id, *x, &mut buffers, "log"),
                Node::Tanh(x) => unary_op(*node_id, *x, &mut buffers, "tanh"),
                Node::Add(x, y) => binary_op(*node_id, *x, *y, &mut buffers, "+"),
                Node::Sub(x, y) => binary_op(*node_id, *x, *y, &mut buffers, "-"),
                Node::Mul(x, y) => binary_op(*node_id, *x, *y, &mut buffers, "*"),
                Node::Div(x, y) => binary_op(*node_id, *x, *y, &mut buffers, "/"),
                Node::Pow(x, y) => {
                    let dtype = buffers[x].dtype();
                    let mut parameters = buffers[x].parameters().clone();
                    parameters.extend(buffers[y].parameters());
                    parameters.insert(*node_id);
                    let kernel = match dtype {
                        DType::F32 => format!(
                            "  {} res{} = pow(res{}, res{});\n",
                            dtype.cl_type(),
                            node_id.i(),
                            x.i(),
                            y.i()
                        ),
                        DType::I32 => format!(
                            "  {} res{} = pown(res{}, res{});\n",
                            dtype.cl_type(),
                            node_id.i(),
                            x.i(),
                            y.i()
                        ),
                    };
                    buffers.insert(
                        *node_id,
                        Op::Basic {
                            dtype,
                            shape: buffers[x].shape().clone(),
                            parameters,
                            kernel,
                        },
                    );
                }
                Node::Expand(x, shape) => {
                    let dtype = buffers[x].dtype();
                    let mut parameters = buffers[x].parameters().clone();
                    let estrides = &buffers[x].shape().expand_strides(shape);
                    for param in &parameters {
                        if let Op::Load { idx, .. } = buffers.get_mut(param).unwrap() {
                            *idx = shape
                                .strides()
                                .into_iter()
                                .zip(estrides)
                                .zip(shape.into_iter())
                                .map(|((nst, st), d)| format!("({idx})/{nst}%{d}*{st}"))
                                .collect::<Box<[String]>>()
                                .join("+");
                        }
                    }
                    parameters.insert(*node_id);
                    buffers.insert(
                        *node_id,
                        Op::Basic {
                            dtype,
                            shape: shape.clone(),
                            parameters,
                            kernel: format!(
                                "  {} res{} = res{};\n",
                                dtype.cl_type(),
                                node_id.i(),
                                x.i()
                            ),
                        },
                    );
                }
                Node::TDot(x, y, shape) => {
                    let dtype = buffers[x].dtype();
                    let mut parameters = buffers[x].parameters().clone();
                    parameters.extend(buffers[y].parameters());
                    parameters.insert(*node_id);

                    // k, m @ k, n -> m, n
                    let x_shape = buffers[x].shape();
                    let m: usize = shape[-2];
                    let k = if x_shape.rank() > 1 { x_shape[-2] } else { 1 };
                    let n: usize = shape[-1];
                    let skip_params = [*x, *y].into();
                    let x = x.i();
                    let y = y.i();
                    let z = node_id.i();

                    // TODO runtime optimization
                    //let ts = (self.max_lws() as f32).sqrt() as usize;
                    let tsm = 128; // 128
                    let wptm = 8; // 16
                    let tsn = 128; // 256
                    let wptn = 8; // 8
                    let tsk = 16; // 16
                    let rtsm = tsm/wptm;
                    let rtsn = tsn/wptn;
                    let rts = rtsm*rtsn;
                    let lpta = (tsk*tsm)/rts;
                    let om = m;
                    let ok = k;
                    let on = n;
                    let m = m + if m%tsm == 0 { 0 } else { tsm - m%tsm };
                    let k = k + if k%tsk == 0 { 0 } else { tsk - k%tsk };
                    let n = n + if n%tsn == 0 { 0 } else { tsn - n%tsn };
                    //std::println!("{m}, {k}, {n}");
                    let load_x = if om%tsm == 0 && ok%tsk == 0 {
                        format!("tile_x[ki][row] = data{x}[tiki*{m} + grid0 + row];")
                    } else {
                        format!("if ((grid0 + row < {om}) && (tiki < {ok})) {{
            tile_x[ki][row] = data{x}[tiki*{om} + grid0 + row];
          }} else tile_x[ki][row] = 0;")
                    };
                    let load_y = if on%tsm == 0 && ok%tsk == 0 {
                        format!("tile_y[ki][row] = data{y}[tiki*{n} + grid1 + row];")
                    } else {
                        format!("if ((grid1 + row < {on}) && (tiki < {ok})) {{
            tile_y[ki][row] = data{y}[tiki*{on} + grid1 + row];
          }} else tile_y[ki][row] = 0;")
                    };
                    let load_z = if om%tsm == 0 && on%tsn == 0 { String::new() } else {
                        format!("if((z_row < {om}) && (z_col < {on})) ")
                    };
                    let kernel = format!("
  const int lid0 = get_local_id(0);
  const int lid1 = get_local_id(1);
  const int grid0 = get_group_id(0)*{tsm};
  const int grid1 = get_group_id(1)*{tsn};
  __local {dtype} tile_x[{tsk}][{tsm}];
  __local {dtype} tile_y[{tsk}][{tsn}];
  {dtype} acc[{wptm}][{wptn}];
  for (int jm = 0; jm < {wptm}; ++jm)
    for (int jn = 0; jn < {wptn}; ++jn)
      acc[jm][jn] = 0.0f;
  {dtype} x_reg;
  {dtype} y_reg[{wptn}];
  for (int ti = 0; ti < {k}; ti += {tsk}) {{
    for (int la=0; la<{lpta}; la++) {{
        int id = la*{rts} + lid1*{rtsm} + lid0;
        int row = id % {tsm};
        int ki = id / {tsm};
        int tiki = ti + ki;
        {load_x}
        {load_y}
    }}
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int ki = 0; ki < {tsk}; ++ki) {{
      for (int jn = 0; jn < {wptn}; ++jn)
        y_reg[jn] = tile_y[ki][jn*{rtsn} + lid1];
      for (int jm = 0; jm < {wptm}; ++jm) {{
        x_reg = tile_x[ki][jm*{rtsm} + lid0];
        for (int jn = 0; jn < {wptn}; ++jn)
          acc[jm][jn] = x_reg * y_reg[jn] + acc[jm][jn]; }} }}
    barrier(CLK_LOCAL_MEM_FENCE); }}
  for (int jn = 0; jn < {wptn}; ++jn) {{
    int z_col = grid1 + jn*{rtsn} + lid1;
    for (int jm = 0; jm < {wptm}; ++jm) {{
      int z_row = grid0 + jm*{rtsm} + lid0;
        {load_z}{{
        {dtype} res{z} = acc[jm][jn];
", dtype=dtype.cl_type());
                    let gws = [m/wptm, n/wptn].into();
                    let lws = Some([rtsm, rtsn].into());
                    let prefix = format!("        datar[z_row*{on} + z_col] = ");
                    let suffix = ";  } } }\n".into();
                    let idx = format!("z_row*{on} + z_col");

                    buffers.insert(
                        *node_id,
                        Op::Custom {
                            dtype,
                            shape: shape.clone(),
                            parameters,
                            kernel,
                            prefix,
                            suffix,
                            gws,
                            lws,
                            skip_params,
                            idx,
                        }
                    );
                }
                Node::Permute(x, axes, shape) => {
                    let dtype = buffers[x].dtype();
                    let mut parameters = buffers[x].parameters().clone();
                    parameters.insert(*node_id);
                    let skip_params = [*x].into();
                    let z = node_id.i();
                    // TODO runtime optimization of this parameter
                    let ts = 64;
                    let (kernel, prefix, suffix, idx, gws, lws) = if shape.rank() == 2 && shape[-1]%ts == 0usize && shape[-2]%ts == 0usize && shape[-1] == shape[2] {
                        let dtype = dtype.cl_type();
                        let z = node_id.i();
                        let [m, n]: [usize; 2] = shape.clone().try_into().unwrap();
                        let x = x.i();
                        let wpt = ts*ts/self.max_lws();
  //for (size_t w = 0; w < {wpt}; ++w)
      //tile[lid0][lid1+w] = data{x}[x + w];
                        let kernel = format!("
  const int lid0 = get_local_id(0);
  const int lid1 = get_local_id(1)*{wpt};
  const int grid0 = get_group_id(0);
  const int grid1 = get_group_id(1);
  __local {dtype} tile[{ts}][{ts}+1];
  int x = (grid0*{ts} + lid0)*{n} + grid1*{ts} + lid1;
  float16 vec = (float16)(*((__global float16*)(data{x}+x)));
  tile[lid0][lid1] = vec.s0;
  tile[lid0][lid1 + 1] = vec.s1;
  tile[lid0][lid1 + 2] = vec.s2;
  tile[lid0][lid1 + 3] = vec.s3;
  tile[lid0][lid1 + 4] = vec.s4;
  tile[lid0][lid1 + 5] = vec.s5;
  tile[lid0][lid1 + 6] = vec.s6;
  tile[lid0][lid1 + 7] = vec.s7;
  tile[lid0][lid1 + 8] = vec.s8;
  tile[lid0][lid1 + 9] = vec.s9;
  tile[lid0][lid1 + 10] = vec.sA;
  tile[lid0][lid1 + 11] = vec.sB;
  tile[lid0][lid1 + 12] = vec.sC;
  tile[lid0][lid1 + 13] = vec.sD;
  tile[lid0][lid1 + 14] = vec.sE;
  tile[lid0][lid1 + 15] = vec.sF;
  barrier(CLK_LOCAL_MEM_FENCE);
  x = (grid1*{ts} + lid0)*{n} + grid0*{ts} + lid1;
  for (size_t w = 0; w < {wpt}; ++w) {{
    {dtype} res{z} = tile[lid1 + w][lid0];\n");
                        let prefix = "    datar[x + w] = ".into();
                        let suffix = ";\n  }\n".into();
                        let m = m + if m%ts == 0 { 0 } else { ts - m % ts };
                        let n = n + if n%ts == 0 { 0 } else { ts - n % ts };
                        let idx = "(x + w)".into();
                        let gws = [m, n/wpt].into();
                        let lws = Some([ts, ts/wpt].into());
                        (kernel, prefix, suffix, idx, gws, lws)
                    } else {
                        // Non coalesced load, so that operation fused with permute
                        // can have coalesced load. This may not be the fastest solution.
                        let data_idx = buffers[x]
                            .shape()
                            .strides()
                            .permute(axes)
                            .into_iter()
                            .enumerate()
                            .map(|(i, st)| format!("gid{i}*{st}"))
                            .collect::<Box<[String]>>()
                            .join("+");
                        let idx = shape
                            .strides()
                            .into_iter()
                            .enumerate()
                            .map(|(i, st)| format!("gid{i}*{st}"))
                            .collect::<Box<[String]>>()
                            .join("+");
                        let gws = shape.into_iter().copied().collect();
                        let x = x.i();
                        let kernel = format!("  {0} res{z} = data{x}[{data_idx}];\n", dtype.cl_type());
                        let prefix = format!("  datar[{idx}] = ");
                        let suffix = ";\n".into();
                        let d: usize = shape[-1];
                        let lws = Some(core::iter::repeat(1).take(shape.rank() - 1).chain([d.min(256)]).collect());
                        (kernel, prefix, suffix, idx, gws, lws)
                    };
                    buffers.insert(
                        *node_id,
                        Op::Custom {
                            dtype,
                            shape: shape.clone(),
                            parameters,
                            kernel,
                            prefix,
                            suffix,
                            gws,
                            lws,
                            skip_params,
                            idx,
                        },
                    );
                }
                Node::Sum(x, axes, shape) => {
                    // TODO vectorized reduce
                    use core::fmt::Write;
                    let dtype = buffers[x].dtype();
                    let mut parameters = buffers[x].parameters().clone();
                    parameters.insert(*node_id);
                    let skip_params = [*x].into();
                    let z = node_id.i();
                    let mut kernel = format!("  {0} res{z} = 0;\n", dtype.cl_type());
                    let idx = buffers[x]
                        .shape()
                        .into_iter()
                        .zip(buffers[x].shape().strides().into_iter())
                        .zip(shape.strides().into_iter())
                        .enumerate()
                        // TODO reorder this loop to get more performance
                        .map(|(i, ((d, rst), st))| {
                            if axes.contains(i) {
                                writeln!(
                                    kernel,
                                    "  for (size_t a{i} = 0; a{i} < {}; a{i} += {st}) {{",
                                    d*st,
                                )
                                .unwrap();
                                format!("a{i}")
                            } else {
                                format!("gid{i}*{rst}")
                            }
                        })
                        .collect::<Box<[String]>>()
                        .join("+");
                    let x = x.i();
                    writeln!(kernel, "    res{z} = data{x}[{idx}] + res{z};").unwrap();
                    for _ in 0..axes.len() {
                        writeln!(kernel, "  }}").unwrap();
                    }
                    let res_idx = shape
                        .strides()
                        .into_iter()
                        .enumerate()
                        .map(|(i, st)| format!("gid{i}*{st}"))
                        .collect::<Box<[String]>>()
                        .join("+");
                    buffers.insert(
                        *node_id,
                        Op::Custom {
                            dtype,
                            shape: shape.clone(),
                            parameters,
                            kernel,
                            prefix: format!("  datar[{res_idx}] = "),
                            suffix: ";\n".into(),
                            gws: shape.into_iter().copied().collect(),
                            lws: None,
                            skip_params,
                            idx: res_idx,
                        },
                    );
                }
                Node::Max(x, axes, shape) => {
                    use core::fmt::Write;
                    let dtype = buffers[x].dtype();
                    let mut parameters = buffers[x].parameters().clone();
                    parameters.insert(*node_id);
                    let skip_params = [*x].into();
                    let z = node_id.i();
                    let mut kernel = format!("  {0} res{z} = 0;\n", dtype.cl_type());
                    let idx = buffers[x]
                        .shape()
                        .into_iter()
                        .zip(buffers[x].shape().strides().into_iter())
                        .zip(shape.strides().into_iter())
                        .enumerate()
                        .map(|(i, ((d, rst), st))| {
                            if axes.contains(i) {
                                writeln!(
                                    kernel,
                                    "  for (size_t a{i} = 0; a{i} < {}; a{i} += {st}) {{",
                                    d*st,
                                )
                                .unwrap();
                                format!("a{i}")
                            } else {
                                format!("gid{i}*{rst}")
                            }
                        })
                        .collect::<Box<[String]>>()
                        .join("+");
                    let x = x.i();
                    writeln!(kernel, "    res{z} = max(data{x}[{idx}], res{z});").unwrap();
                    for _ in 0..axes.len() {
                        writeln!(kernel, "  }}").unwrap();
                    }
                    let res_idx = shape
                        .strides()
                        .into_iter()
                        .enumerate()
                        .map(|(i, st)| format!("gid{i}*{st}"))
                        .collect::<Box<[String]>>()
                        .join("+");
                    buffers.insert(
                        *node_id,
                        Op::Custom {
                            dtype,
                            shape: shape.clone(),
                            parameters,
                            kernel,
                            prefix: format!("  datar[{res_idx}] = "),
                            suffix: ";\n".into(),
                            gws: shape.into_iter().copied().collect(),
                            lws: None,
                            skip_params,
                            idx: res_idx,
                        },
                    );
                }
            }
            if nodes.contains(node_id) {
                graph.get_mut(node_id).unwrap().1 =
                    Node::Const(self.eval(*node_id, &mut buffers, order)?);
            }
            let parameters = graph[node_id].1.parameters();
            //std::println!("Relesing parameters {:?}.", parameters);
            for parameter in &*parameters {
                //std::println!("Releasing param {}", parameter);
                let val = graph.get_mut(parameter).unwrap();
                val.0 -= 1;
                if val.0 == 0 {
                    val.1 = Node::None;
                }
            }
        }
        Ok(())
    }

    // launches this kernel (asynchronously)
    #[allow(clippy::too_many_lines)]
    fn eval(
        &mut self,
        id: NodeId,
        buffers: &mut BTreeMap<NodeId, Op>,
        order: &[NodeId],
    ) -> Result<Storage, OutOfMemoryError> {
        //std::println!("Realizing {}", id);
        use core::fmt::Write;
        let (
            global_work_size,
            local_work_size,
            dtype,
            shape,
            kernel,
            prefix,
            suffix,
            parameters,
            skip_params,
            data_idx,
        ) = match &buffers[&id] {
            Op::Custom {
                dtype,
                shape,
                kernel,
                prefix,
                suffix,
                gws,
                lws,
                parameters,
                skip_params,
                idx,
            } => (
                gws.clone(),
                lws.clone(),
                *dtype,
                shape.clone(),
                kernel.as_str(),
                prefix.as_str(),
                suffix.as_str(),
                parameters,
                skip_params.clone(),
                idx.as_str(),
            ),
            Op::Basic {
                dtype,
                shape,
                kernel: _,
                parameters,
            } => {
                if let Some(custom_op) = buffers[&id]
                    .parameters()
                    .iter()
                    .find(|x| buffers[x].is_custom())
                {
                    let Op::Custom {
                        gws,
                        lws,
                        kernel,
                        prefix,
                        suffix,
                        skip_params,
                        idx,
                        ..
                    } = &buffers[custom_op]
                    else {
                        panic!()
                    };
                    (
                        gws.clone(),
                        lws.clone(),
                        *dtype,
                        shape.clone(),
                        kernel.as_str(),
                        prefix.as_str(),
                        suffix.as_str(),
                        parameters,
                        skip_params.clone(),
                        idx.as_str(),
                    )
                } else {
                    (
                        [shape.numel()].into(),
                        None,
                        *dtype,
                        shape.clone(),
                        "",
                        "  datar[gid0] = ",
                        ";\n",
                        parameters,
                        [].into(),
                        "gid0",
                    )
                }
            }
            Op::Load {
                dtype,
                shape,
                idx: _,
                storage,
                parameters: _,
            } => {
                // Carefull here, this is clone, so we increase refcount
                unsafe { cl3::event::retain_event(storage.event()) }.unwrap();
                unsafe { cl3::memory::retain_mem_object(storage.buffer()) }.unwrap();
                return Ok(match dtype {
                    DType::F32 => Storage::OpenCLF32(shape.clone(), storage.clone()),
                    DType::I32 => Storage::OpenCLI32(shape.clone(), storage.clone()),
                });
            }
        };
        let ids = (0..global_work_size.len()).fold(String::new(), |mut out, i| {
            writeln!(out, "  size_t gid{i} = get_global_id({i});").unwrap();
            //writeln!(out, "  size_t lid{i} = get_local_id({i});").unwrap();
            out
        });
        let (params, buffers_params): (Vec<(DType, NodeId)>, Vec<*mut c_void>) = parameters
            .iter()
            .filter_map(|x| {
                if let Op::Load {
                    dtype,
                    shape: _,
                    idx: _,
                    storage,
                    parameters,
                } = &buffers[x]
                {
                    Some(((*dtype, *parameters.first().unwrap()), storage.buffer()))
                } else {
                    None
                }
            })
            .unzip();
        let params = params.iter().fold(String::new(), |mut out, (dtype, id)| {
            write!(out, ",\n  __global const {}* data{id}", dtype.cl_type()).unwrap();
            out
        });
        let load_kernels = parameters
            .iter()
            .filter_map(|x| {
                if skip_params.contains(x) {
                    None
                } else if let Op::Load {
                    dtype,
                    shape: _,
                    idx,
                    storage: _,
                    parameters,
                } = &buffers[x]
                {
                    Some((
                        dtype,
                        *parameters.first().unwrap(),
                        idx.replace("IDX", data_idx),
                    ))
                } else {
                    None
                }
            })
            .fold(String::new(), |mut out, (dtype, id, idx)| {
                writeln!(
                    out,
                    "  {} res{1} = data{1}[{idx}];",
                    dtype.cl_type(),
                    id.i()
                )
                .unwrap();
                out
            });
        let mut sec_kernels: Vec<(NodeId, String)> = parameters
            .iter()
            .filter_map(|x| {
                if let Op::Basic {
                    dtype: _,
                    shape: _,
                    kernel,
                    parameters: _,
                } = &buffers[x]
                {
                    Some((*x, kernel.clone()))
                } else {
                    None
                }
            })
            .collect();
        // TODO can this be sorted in faster way?
        sec_kernels.sort_by_key(|(k, _)| order.iter().position(|x| x == k));
        let sec_kernels = sec_kernels.iter().fold(String::new(), |mut out, (_, v)| {
            write!(out, "{v}",).unwrap();
            out
        });
        let mut kernel_name = global_work_size
            .iter()
            .map(alloc::string::ToString::to_string)
            .collect::<Box<[String]>>()
            .join("_");
        if let Some(local_work_size) = &local_work_size {
            kernel_name.push_str("__");
            kernel_name.push_str(local_work_size
                .iter()
                .map(alloc::string::ToString::to_string)
                .collect::<Box<[String]>>()
                .join("_").as_str());
        }
        kernel_name.insert_str(0, "kernel_");
        let source = format!(
            "__kernel void {kernel_name} (\n  __global {}* \
            datar{params}\n) {{\n{ids}{kernel}{load_kernels}{sec_kernels}{prefix}res{id}{suffix}}}",
            dtype.cl_type()
        );
        let program = if let Some(program) = self.programs.get(&source) {
            *program
        } else {
            #[cfg(feature = "debug1")]
            std::println!("Compiling OpenCL kernel:\n{source}");
            let program =
                cl3::program::create_program_with_source(self.context, &[&source]).unwrap();
            let devices = self.devices.keys().copied().collect::<Box<[*mut c_void]>>();
            if let Err(er) = cl3::program::build_program(
                program,
                devices.as_ref(),
                core::ffi::CStr::from_bytes_with_nul(b"-cl-fast-relaxed-math\0").unwrap(),
                None,
                core::ptr::null_mut(),
            ) {
                let log = cl3::program::get_program_build_info(
                    program,
                    devices[0],
                    CL_PROGRAM_BUILD_LOG,
                )
                .unwrap();
                panic!("Compilation failed with error {er}:\n{log}");
            };
            self.programs.insert(source, program);
            program
        };
        //std::println!("Creating kernel {}", id);
        let kernel =
            cl3::kernel::create_kernel(program, &CString::new(kernel_name).unwrap()).unwrap();
        let buffer = unsafe {
            cl3::memory::create_buffer(
                self.context,
                CL_MEM_READ_ONLY,
                shape.numel() * dtype.byte_size(),
                core::ptr::null_mut(),
            )
        }
        .map_err(|_| OutOfMemoryError)?;
        let ptr: *const _ = &buffer;
        //std::println!("Setting kernel args.");
        unsafe {
            cl3::kernel::set_kernel_arg(kernel, 0, core::mem::size_of::<*mut c_void>(), ptr.cast())
        }
        .unwrap();
        let mut i = 1;
        for buffer in buffers_params {
            // This is POINTER MAGIC. Careful.
            let ptr: *const _ = &buffer;
            //std::println!("Setting args: {ptr:?}");
            unsafe {
                cl3::kernel::set_kernel_arg(
                    kernel,
                    i,
                    core::mem::size_of::<*mut c_void>(),
                    ptr.cast(),
                )
            }
            .unwrap();
            i += 1;
        }
        //let events: Box<[*mut c_void]> = events.iter().map(|(_, event)| *event).collect();
        let events = parameters
            .iter()
            .filter_map(|x| match &buffers[x] {
                Op::Load { storage, .. } => Some(storage.event()),
                _ => None,
            })
            .collect::<Box<[*mut c_void]>>();
        //std::println!("Launching kernel.");
        let event = unsafe {
            cl3::command_queue::enqueue_nd_range_kernel(
                self.queue(),
                kernel,
                u32::try_from(global_work_size.len()).unwrap(),
                core::ptr::null(),
                global_work_size.as_ptr(),
                // Following line is funny. Removing as_ref causes local_work_size to be moved, thus as_ptr would be use after free
                local_work_size.as_ref().map_or(core::ptr::null(), |x| x.as_ptr()),
                u32::try_from(events.len()).unwrap(),
                if events.is_empty() {
                    core::ptr::null()
                } else {
                    events.as_ptr()
                },
            )
        }
        .expect("could not execute opencl kernel.");
        //std::println!("Syncing.");
        // So that we can measure performance
        #[cfg(feature = "debug1")]
        cl3::event::wait_for_events(&[event]).unwrap();
        // mutate self so that we can reuse calculated buffer
        let storage = ClStorage::new(buffer, event);
        *buffers.get_mut(&id).unwrap() =
            Op::from_const(id, dtype, shape.clone(), storage.clone());
        //std::println!("Return.");
        Ok(match dtype {
            DType::F32 => Storage::OpenCLF32(shape, storage),
            DType::I32 => Storage::OpenCLI32(shape, storage),
        })
    }
}

fn unary_op(id: NodeId, x: NodeId, buffers: &mut BTreeMap<NodeId, Op>, op: &str) {
    let dtype = buffers[&x].dtype();
    let mut parameters = buffers[&x].parameters().clone();
    parameters.insert(id);
    buffers.insert(
        id,
        Op::Basic {
            dtype,
            shape: buffers[&x].shape().clone(),
            parameters,
            kernel: format!(
                "  {} res{} = {}(res{});\n",
                dtype.cl_type(),
                id.i(),
                op,
                x.i()
            ),
        },
    );
}

fn binary_op(id: NodeId, x: NodeId, y: NodeId, buffers: &mut BTreeMap<NodeId, Op>, op: &str) {
    let dtype = buffers[&x].dtype();
    let mut parameters = buffers[&x].parameters().clone();
    parameters.extend(buffers[&y].parameters());
    parameters.insert(id);
    buffers.insert(
        id,
        Op::Basic {
            dtype,
            shape: buffers[&x].shape().clone(),
            parameters,
            kernel: format!(
                "  {} res{} = res{} {op} res{};\n",
                dtype.cl_type(),
                id.i(),
                x.i(),
                y.i()
            ),
        },
    );
}

#[cfg(all(feature = "debug1", feature = "rand"))]
#[test]
fn test1() {
    use crate::context::Context;
    let ctx = Context::opencl().unwrap();
    let x = ctx.randn((2, 3, 1)).permute((1, 0, 2)).expand((3, 2, 4));
    let mut z = x.exp();
    //let y = ctx.randn((2, 3, 1)).expand((2, 3, 4));
    //let mut z = &y + x + &y.exp();
    z.realize().unwrap();
    std::println!("{}", z);
    //panic!();
}

#[cfg(feature = "debug1")]
#[test]
fn test2() {
    use crate::context::Context;
    let ctx = Context::opencl().unwrap();
    let x = ctx.tensor([[1., 2., 3.], [4., 5., 6.]]);
    let y = ctx.tensor([[1., 2., 3.], [4., 5., 6.]]);
    let mut z = x.transpose().dot(y);
    z.realize().unwrap();
    std::println!("{}", z);
    //panic!();
}

#[cfg(feature = "debug1")]
#[test]
fn test3() -> Result<(), OutOfMemoryError> {
    use crate::context::Context;

    let ctx = Context::new();
    let x = ctx.tensor([[1., 2., 3.], [4., 5., 6.]]); //.cast(DType::I32);
    let y = ctx.tensor([[1., 2., 3.], [4., 5., 6.], [7., 8., 9.]]);
    let mut z = x.dot(y);
    z.realize().unwrap();
    std::println!("{z}");

    let ctx = Context::opencl().unwrap();
    let x = ctx.tensor([[1., 2., 3.], [4., 5., 6.]]); //.cast(DType::I32);
    let y = ctx.tensor([[1., 2., 3.], [4., 5., 6.], [7., 8., 9.]]);
    //.cast(DType::I32);
    let mut z = x.dot(y); // * 0.01; //.tanh();

    z.realize().unwrap();
    std::println!("{z}");
    //panic!();

    Ok(())
}

#[cfg(all(feature = "debug1", feature = "rand"))]
#[test]
fn test4() -> Result<(), OutOfMemoryError> {
    use crate::context::Context;
    let dim = 256;
    let dim2 = 256;

    let ctx = Context::new();
    let x = ctx.uniform_i32((dim, dim2), 0..10);
    let y = ctx.uniform_i32((dim, dim2), 0..10);
    let z = x.dot(&y);

    let mut x = x.transpose();
    let mut y = y;
    let mut z = z;

    x.realize()?;
    y.realize()?;
    z.realize()?;
    
    std::println!("{x}\n{y}\n{z}");

    let vecx = x.to_vec_i32().unwrap();
    let vecy = y.to_vec_i32().unwrap();
    let vecz = z.to_vec_i32().unwrap();

    // OpenCL
    let ctx = Context::opencl().unwrap();

    let x = ctx.tensor_from_iter_i32((dim, dim2), vecx);
    let y = ctx.tensor_from_iter_i32((dim, dim2), vecy);

    let mut z = x.t_dot(&y);
    z.realize()?;
    
    std::println!("{z}");

    let cl_vecz = z.to_vec_i32().unwrap();

    for (x, y) in vecz.into_iter().zip(cl_vecz) {
        assert!(x - y == 0, "{x} != {y}");
    }

    let dim = 2048;
    let dim2 = 2048;
    let ctx = Context::opencl().unwrap();
    let x = ctx.uniform((dim, dim2), 0f32..10.);
    let y = ctx.uniform((dim, dim2), 0f32..10.);
    //let k = ctx.randn((dim, dim));

    let begin = std::time::Instant::now();
    let iters = 20;
    for _ in 0..iters {
        let mut z = x.t_dot(&y); //.tanh() + &k;
        z.realize()?;
    }
    let elapsed = begin.elapsed();

    std::println!("Elapsed {} GFLOPS", iters*2048*2048*2*2047/elapsed.as_nanos());
    //panic!()
    Ok(())
}

#[cfg(feature = "rand")]
#[test]
fn test5() {
    use crate::context::Context;
    use crate::parameters::IntoParameters;

    let ctx = Context::new();
    let mut x = ctx.randn((2048, 2048));
    let mut z = x.transpose();
    (&mut x, &mut z).realize().unwrap();
    let xvec = x.to_vec().unwrap();
    let zvec = z.to_vec().unwrap();
    
    let ctx = Context::opencl().unwrap();
    let x = ctx.tensor_from_iter_f32((2048, 2048), xvec.into_iter());
    let mut z = x.transpose();
    z.realize().unwrap();
    let cl_zvec = z.to_vec().unwrap();

    for (x, y) in zvec.into_iter().zip(cl_zvec) {
        assert!((x - y).abs() < 0.00001, "{x} != {y}");
    }

    for _ in 0..5 {
        let mut z = x.transpose();
        z.realize().unwrap();
    }
}

#[cfg(feature = "rand")]
#[test]
fn test6() {
    use crate::context::Context;

    let ctx = Context::opencl().unwrap();
    let x = ctx.randn((2048, 2048));

    for _ in 0..10 {
        let mut z = x.sum(0);
        z.realize().unwrap();
    }
}

#[cfg(all(feature = "debug1", feature = "rand"))]
#[test]
fn test7() {
    use crate::context::Context;
    let ctx = Context::opencl().unwrap();
    let mut x = ctx.tensor([[2, 4, 3], [5, 2, 4]]);
    let y = ctx.tensor([[2, 2, 4], [1, 2, 1], [3, 4, 2]]);
    let mut z = x.dot(&y);
    z.realize().unwrap();
    std::println!("{}", z);
    //assert_eq!(z, [[17, 24, 18], [24, 30, 30]]);
    let mut y = y.transpose();
    x.realize().unwrap();
    y.realize().unwrap();
    std::println!("{}\n{}", x, x.shape());
    std::println!("{}\n{}", y, y.shape());
    let mut z = x.dot(y);
    z.realize().unwrap();
    std::println!("{}\n{}", z, z.shape());
}
