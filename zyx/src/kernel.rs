use crate::{
    DType, Map, Set,
    backend::{Device, DeviceInfo, ProgramId},
    dtype::Constant,
    graph::{BOp, ROp, UOp},
    shape::Dim,
    view::View,
};
use std::{hash::BuildHasherDefault, ops::Range};

pub type OpId = usize;

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Kernel {
    pub ops: Vec<Op>,
    //pub shape: Vec<Dim>,
    pub n_outputs: u32,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Op {
    // ops that exist only in kernelizer
    ConstView { value: Constant, view: View },
    LoadView { dtype: DType, view: View },
    Reduce { x: OpId, rop: ROp, dims: Vec<Dim> },

    // ops that only exist after unfolding views and reduces
    Const(Constant),
    Load { dtype: DType, index: OpId },
    DeclareAcc { dtype: DType, rop: ROp },
    Loop { dim: Dim, vectorize: bool }, // vectorize means both vectorization and tensor cores
    Accumulate { x: OpId, rop: ROp },
    EndLoop,

    // ops that exist in both
    Store { x: OpId, index: OpId },
    Cast { x: OpId, dtype: DType },
    Unary { x: OpId, uop: UOp },
    Binary { x: OpId, y: OpId, bop: BOp },
}

#[derive(Debug)]
pub struct Cache {
    pub device_infos: Map<DeviceInfo, u32>,
    pub kernels: Map<Kernel, u32>,
    // Finished optimizations of kernels for given devices
    // kernel id, device info id => optimization, time to run in nanos
    pub optimizations: Map<(u32, u32), (Optimizer, u64)>,
    // This last one is not stored to disk
    // kernel id, device id => program id
    pub programs: Map<(u32, u32), ProgramId>,
}

#[derive(Debug, Clone)]
pub enum Optimization {
    Basic { shape: Vec<Dim>, loop_unroll_size: Dim },
}

#[derive(Debug)]
pub enum Optimizer {
    Done(Optimization),
    Default(Optimization),
    Ongoing { current: Optimization, best: Optimization },
}

impl Optimizer {
    fn next_optimization(&self) -> Option<&Optimization> {
        match self {
            Optimizer::Done(optimization) => todo!(),
            Optimizer::Default(optimization) => Some(&optimization),
            Optimizer::Ongoing { current, best } => todo!(),
        }
    }
}

impl Cache {
    pub const fn new() -> Cache {
        Cache {
            device_infos: Map::with_hasher(BuildHasherDefault::new()),
            kernels: Map::with_hasher(BuildHasherDefault::new()),
            optimizations: Map::with_hasher(BuildHasherDefault::new()),
            programs: Map::with_hasher(BuildHasherDefault::new()),
        }
    }

    pub(super) fn deinitialize(&mut self, devices: &mut [Device]) {
        for (&(_, device_id), program_id) in &mut self.programs {
            devices[device_id as usize].release(*program_id);
        }
        self.device_infos = Map::with_hasher(BuildHasherDefault::new());
        self.kernels = Map::with_hasher(BuildHasherDefault::new());
        self.optimizations = Map::with_hasher(BuildHasherDefault::new());
    }
}

#[allow(clippy::similar_names)]
pub fn get_perf(flop: u128, bytes_read: u128, bytes_written: u128, nanos: u128) -> String {
    const fn value_unit(x: u128) -> (u128, &'static str) {
        match x {
            0..1000 => (x * 100, ""),
            1_000..1_000_000 => (x / 10, "k"),
            1_000_000..1_000_000_000 => (x / 10_000, "M"),
            1_000_000_000..1_000_000_000_000 => (x / 10_000_000, "G"),
            1_000_000_000_000..1_000_000_000_000_000 => (x / 10_000_000_000, "T"),
            1_000_000_000_000_000..1_000_000_000_000_000_000 => (x / 10_000_000_000_000, "P"),
            1_000_000_000_000_000_000.. => (x / 10_000_000_000_000_000, "E"),
        }
    }

    let (f, f_u) = value_unit(flop);
    let (br, br_u) = value_unit(bytes_read);
    let (bw, bw_u) = value_unit(bytes_written);
    let (t, t_u) = match nanos {
        0..1_000 => (nanos * 10, "ns"),
        1_000..1_000_000 => (nanos / 100, "μs"),
        1_000_000..1_000_000_000 => (nanos / 100_000, "ms"),
        1_000_000_000..1_000_000_000_000 => (nanos / 100_000_000, "s"),
        1_000_000_000_000.. => (nanos / 6_000_000_000, "min"),
    };

    let (fs, f_us) = value_unit(flop * 1_000_000_000 / nanos);
    let (brs, br_us) = value_unit(bytes_read * 1_000_000_000 / nanos);
    let (bws, bw_us) = value_unit(bytes_written * 1_000_000_000 / nanos);

    format!(
        "{}.{} {t_u} ~ {}.{:02} {f_us}FLOP/s, {}.{:02} {br_us}B/s r, {}.{:02} {bw_us}B/s w, {}.{:02} {f_u}FLOP, {}.{:02} {br_u}B r, {}.{:02} {bw_u}B w",
        t / 10,
        t % 10,
        fs / 100,
        fs % 100,
        brs / 100,
        brs % 100,
        bws / 100,
        bws % 100,
        f / 100,
        f % 100,
        br / 100,
        br % 100,
        bw / 100,
        bw % 100,
    )
}

impl Kernel {
    pub fn apply_movement(&mut self, func: impl Fn(&mut View)) {
        for op in &mut self.ops {
            match op {
                Op::ConstView { view, .. } | Op::LoadView { view, .. } => {
                    func(view);
                }
                _ => {}
            }
        }
    }

    pub fn debug(&self) {
        //println!("Kernel shape {:?}", self.shape);
        for (i, op) in self.ops.iter().enumerate() {
            match op {
                Op::Const(x) => println!("{i:>3} CONST {x}"),
                Op::ConstView { value, view } => println!("{i:>3} CONST VIEW {value} {view}"),
                Op::LoadView { dtype, view } => println!("{i:>3} LOAD VIEW {dtype} {view}"),
                Op::Load { dtype, index } => println!("{i:>3} LOAD {dtype} at {index}"),
                Op::Store { x, index } => println!("{i:>3} STORE {x} at {index}"),
                Op::Cast { x, dtype } => println!("{i:>3} CAST {x} {dtype:?}"),
                Op::Unary { x, uop } => println!("{i:>3} UNARY {uop:?} {x}"),
                Op::Binary { x, y, bop } => println!("{i:>3} BINARY {bop:?} {x} {y}"),
                Op::Loop { dim, vectorize: tiled } => println!("{i:>3} LOOP dim={dim} tiled={tiled}"),
                Op::EndLoop => println!("{i:>3} ENDLOOP"),
                Op::Reduce { x, rop, dims } => println!("{i:>3} REDUCE {} {x}, dims={dims:?}", match rop {
                    ROp::Sum => "SUM",
                    ROp::Max => "MAX",
                }),
                Op::DeclareAcc { dtype, rop } => println!("{i:>3} DECLARE ACC {dtype} {rop:?}"),
                Op::Accumulate { x, rop } => println!("{i:>3} ACCUMULATE {x} {rop:?}"),
            }
        }
    }

    pub fn flop_mem_rw(&self) -> (u128, u128, u128) { (0, 0, 0) }

    pub fn is_reduce(&self) -> bool { self.ops.iter().any(|x| matches!(x, Op::Reduce { .. })) }

    pub(super) fn default_optimization(&self, dev_info: &DeviceInfo) -> Optimizer {
        fn get_equal_factors(x: Dim) -> [Dim; 3] {
            fn get_factors(n: Dim) -> Vec<Dim> {
                let mut factors = Vec::new();
                for i in 1..=n.isqrt() {
                    if n % i == 0 {
                        factors.insert(0, i);
                        factors.push(n / i);
                    }
                }
                factors
            }
            let f = get_factors(x);
            let mut res = [1, 1, 1];
            let mut min_dist = x as isize;
            for i in 0..f.len() {
                for j in i..f.len() {
                    for k in j..f.len() {
                        if f[i] * f[j] * f[k] == x {
                            let r = (f[i] as isize - f[j] as isize).abs() + (f[i] as isize - f[k] as isize).abs();
                            if r < min_dist {
                                min_dist = r;
                                res = [f[i], f[j], f[k]];
                            }
                        }
                    }
                }
            }
            res
        }

        let n = self.shape().iter().product();
        let mut global_work_size = get_equal_factors(n);

        let mut d = dev_info.max_local_threads;
        while n % d != 0 {
            d -= 1;
        }
        let local_work_size = get_equal_factors(d);
        global_work_size[0] /= local_work_size[0];
        global_work_size[1] /= local_work_size[1];
        global_work_size[2] /= local_work_size[2];

        // Concatenate global and local work sizes to get the final 6D shape
        let mut shape = vec![];
        shape.extend(global_work_size);
        shape.extend(local_work_size);

        Optimizer::Default(Optimization::Basic { shape, loop_unroll_size: 16 })
    }

    pub fn apply_optimization(&mut self, optimizer: &Optimizer) {
        let Some(optimization) = optimizer.next_optimization() else { return };
        let loop_unroll_size = match optimization {
            Optimization::Basic { shape, loop_unroll_size } => {
                let n = self.shape().len();
                self.apply_movement(|view| view.reshape(0..n, &shape));
                *loop_unroll_size
            }
        };
        /*let n = self.shape.len();
        let shape = vec![1, 1, 1, 1, 4, 2];
        self.apply_movement(|view| view.reshape(0..n, &shape));
        self.shape = shape.clone();*/

        //let loop_unroll_size = 8;

        self.unfold_shape();
        self.unfold_reduces();
        self.unfold_views();

        let mut kernel = self.clone();
        loop {
            self.move_constants_to_beginning();
            self.constant_folding();
            self.common_subexpression_elimination();

            //self.dead_code_elimination();
            self.loop_invariant_code_motion();
            self.loop_unrolling(loop_unroll_size);

            if *self == kernel {
                break;
            }
            kernel = self.clone();
        }
    }

    fn shape(&self) -> Vec<Dim> {
        if self.ops.iter().any(|op| matches!(op, Op::Loop { .. })) {
            return self
                .ops
                .iter()
                .map_while(|op| {
                    if let Op::Loop { dim, .. } = op {
                        Some(*dim)
                    } else {
                        None
                    }
                })
                .collect();
        }
        let mut reduce_dims = 0;
        for op in self.ops.iter().rev() {
            match op {
                Op::ConstView { view, .. } => {
                    let mut shape = view.shape();
                    for _ in 0..reduce_dims {
                        shape.pop();
                    }
                    return shape;
                }
                Op::LoadView { view, .. } => {
                    let mut shape = view.shape();
                    for _ in 0..reduce_dims {
                        shape.pop();
                    }
                    return shape;
                }
                Op::Reduce { dims, .. } => {
                    reduce_dims += dims.len();
                }
                _ => {}
            }
        }
        unreachable!()
    }

    fn unfold_shape(&mut self) {
        let shape = self.shape();
        let n = self.ops.len();
        increment(&mut self.ops, shape.len(), 0..n);
        for dim in shape.into_iter().rev() {
            self.ops.insert(0, Op::Loop { dim, vectorize: false });
        }
    }

    fn unfold_reduces(&mut self) {
        // Check the reduce op, trace all of it's dependencies,
        // put Loop op before dependency with lowest ID
        // increase all ids higher than that by one

        let reduce_ops: Vec<OpId> =
            self.ops.iter().enumerate().filter(|(_, op)| matches!(op, Op::Reduce { .. })).map(|(i, _)| i).collect();
        for op_id in reduce_ops.into_iter().rev() {
            let Op::Reduce { x, rop, dims } = self.ops[op_id].clone() else { unreachable!() };
            let mut min_param = x;
            let mut params = vec![x];
            let mut acc_dtype = None;
            while let Some(param) = params.pop() {
                match self.ops[param] {
                    Op::ConstView { value, .. } => {
                        if acc_dtype.is_none() {
                            acc_dtype = Some(value.dtype());
                        }
                    }
                    Op::Const { .. } => unreachable!(),
                    Op::LoadView { dtype, .. } => {
                        if acc_dtype.is_none() {
                            acc_dtype = Some(dtype);
                        }
                    }
                    Op::Load { .. } => unreachable!(),
                    Op::Store { x, index } => {
                        params.push(index);
                        if index < min_param {
                            min_param = index;
                        }
                        params.push(x);
                        if x < min_param {
                            min_param = x;
                        }
                    }
                    Op::Cast { x, dtype } => {
                        params.push(x);
                        if x < min_param {
                            min_param = x;
                        }
                        if acc_dtype.is_none() {
                            acc_dtype = Some(dtype);
                        }
                    }
                    Op::Unary { x, .. } | Op::Reduce { x, .. } => {
                        params.push(x);
                        if x < min_param {
                            min_param = x;
                        }
                    }
                    Op::Binary { x, y, .. } => {
                        params.push(x);
                        if x < min_param {
                            min_param = x;
                        }
                        params.push(y);
                        if y < min_param {
                            min_param = y;
                        }
                    }
                    Op::DeclareAcc { .. } => unreachable!(),
                    Op::Accumulate { .. } => unreachable!(),
                    Op::Loop { .. } => unreachable!(),
                    Op::EndLoop { .. } => unreachable!(),
                }
            }
            let n = dims.len();
            self.ops[op_id] = Op::EndLoop;
            for _ in 0..(n - 1) {
                self.ops.insert(op_id, Op::EndLoop);
            }
            self.ops.insert(op_id, Op::Accumulate { x, rop });
            for dim in dims {
                self.ops.insert(min_param, Op::Loop { dim, vectorize: false });
            }
            self.ops.insert(min_param, Op::DeclareAcc { dtype: acc_dtype.unwrap(), rop });
            let ops_len = self.ops.len();
            increment(&mut self.ops[min_param + 1..], 1 + n, min_param..ops_len);
            /*for (i, op) in self.ops.iter().enumerate() {
                println!("{i} -> {op:?}");
            }
            println!("n={n}\n");*/
        }
    }

    fn unfold_views(&mut self) {
        // First we generate the whole view into a new vec,
        // then we insert the vec into existing ops
        // Convert view
        fn new_op(ops: &mut Vec<Op>, op: Op) -> OpId {
            let op_id = ops.len();
            ops.push(op);
            op_id
        }

        let mut op_id = 0;
        while op_id < self.ops.len() {
            match self.ops[op_id] {
                Op::ConstView { value, view: _ } => {
                    // TODO process view
                    self.ops[op_id] = Op::Const(value);
                }
                Op::LoadView { dtype, ref view } => {
                    // With padding, right padding does not affect offset
                    // offset = (a0-lp0)*st0 + a1*st1
                    // Padding condition, negative right padding does not affect it
                    // pc = a0 > lp0-1 && a0 < d0-rp0
                    // pc = pc.cast(dtype)
                    // x = pc * value[offset]

                    let view = view.clone();
                    //println!("Unfolding view: {view}");
                    let temp_ops: Vec<Op> = self.ops.split_off(op_id + 1);
                    self.ops.pop();
                    let ops = &mut self.ops;
                    let axes = get_axes(&ops);
                    let mut pc = new_op(ops, Op::Const(Constant::Bool(true)));
                    let mut offset = new_op(ops, Op::Const(Constant::U32(0)));
                    let mut old_offset: Option<OpId> = None;
                    //println!("View");
                    //for inner in self.0.iter() { println!("{inner:?}") }
                    //println!();
                    for inner in view.0.iter().rev() {
                        //println!("\n{inner:?}");
                        // a = offset / ost % dim
                        let mut ost = 1;
                        offset = new_op(ops, Op::Const(Constant::U32(0)));
                        for (a, dim) in inner.iter().enumerate().rev() {
                            let a = if let Some(old_offset) = old_offset {
                                let ost_c = new_op(ops, Op::Const(Constant::U32(ost)));
                                let a = new_op(ops, Op::Binary { x: old_offset, y: ost_c, bop: BOp::Div });
                                ost *= dim.d as u32;
                                let dimd_c = new_op(ops, Op::Const(Constant::U32(dim.d as u32)));
                                new_op(ops, Op::Binary { x: a, y: dimd_c, bop: BOp::Mod })
                            } else {
                                if dim.d == 1 {
                                    new_op(ops, Op::Const(Constant::U32(0)))
                                } else {
                                    axes[a]
                                }
                            };
                            //println!("ost: {ost}, a: {a:?}, {dim:?}");
                            // Offset
                            let t = if dim.lp != 0 {
                                let lp = new_op(ops, Op::Const(Constant::U32(dim.lp.abs() as u32)));
                                if dim.lp > 0 {
                                    new_op(ops, Op::Binary { x: a, y: lp, bop: BOp::Sub })
                                } else {
                                    new_op(ops, Op::Binary { x: a, y: lp, bop: BOp::Add })
                                }
                            } else {
                                a
                            };
                            let stride = new_op(ops, Op::Const(Constant::U32(dim.st as u32)));
                            let x = new_op(ops, Op::Binary { x: t, y: stride, bop: BOp::Mul });
                            offset = new_op(ops, Op::Binary { x, y: offset, bop: BOp::Add });

                            // Padding condition
                            if dim.lp > 0 {
                                let lp = new_op(ops, Op::Const(Constant::U32((dim.lp - 1) as u32)));
                                let t = new_op(ops, Op::Binary { x: a, y: lp, bop: BOp::Cmpgt });
                                pc = new_op(ops, Op::Binary { x: t, y: pc, bop: BOp::And });
                            }
                            if dim.rp > 0 {
                                let rp = new_op(ops, Op::Const(Constant::U32((dim.d as isize - dim.rp) as u32)));
                                let t = new_op(ops, Op::Binary { x: a, y: rp, bop: BOp::Cmplt });
                                pc = new_op(ops, Op::Binary { x: t, y: pc, bop: BOp::And });
                            }
                        }
                        old_offset = Some(offset);
                    }
                    let pcu32 = new_op(ops, Op::Cast { x: pc, dtype: DType::U32 });
                    let offset = new_op(ops, Op::Binary { x: pcu32, y: offset, bop: BOp::Mul });
                    let z = new_op(ops, Op::Load { dtype, index: offset });
                    let pcd = new_op(ops, Op::Cast { x: pc, dtype });
                    // Nullify z if padding condition is false (if there is padding at that index)
                    _ = new_op(ops, Op::Binary { x: pcd, y: z, bop: BOp::Mul });

                    let n = self.ops.len();
                    self.ops.extend(temp_ops);
                    let ops_len = self.ops.len();
                    increment(&mut self.ops[n..], n - op_id - 1, op_id..ops_len);
                    op_id = n;
                    continue;
                }
                Op::Store { x, .. } => {
                    let temp_ops: Vec<Op> = self.ops.split_off(op_id + 1);
                    self.ops.pop();
                    let axes = get_axes(&self.ops);
                    let mut index = new_op(&mut self.ops, Op::Const(Constant::U32(0)));
                    let mut st = 1;
                    let shape = self.shape().clone();
                    for (id, d) in shape.iter().enumerate().rev() {
                        let stride = Constant::U32(st as u32);
                        let x = if *d > 1 {
                            axes[id]
                        } else {
                            new_op(&mut self.ops, Op::Const(Constant::U32(0)))
                        };
                        let y = new_op(&mut self.ops, Op::Const(stride));
                        let x = new_op(&mut self.ops, Op::Binary { x, y, bop: BOp::Mul });
                        index = new_op(&mut self.ops, Op::Binary { x, y: index, bop: BOp::Add });
                        st *= d;
                    }
                    _ = new_op(&mut self.ops, Op::Store { x, index });

                    let n = self.ops.len();
                    /*for (i, op) in self.ops.iter().enumerate() {
                        println!("{i} -> {op:?}");
                    }
                    println!("n={n}");*/
                    self.ops.extend(temp_ops);
                    let ops_len = self.ops.len();
                    increment(&mut self.ops[n..], n - op_id - 1, op_id..ops_len);
                    op_id = n;
                    continue;
                }
                _ => {}
            }
            op_id += 1;
        }
    }

    fn decrement_range(&mut self, range: Range<usize>, n: usize) {
        for op in &mut self.ops[range.clone()] {
            match op {
                Op::ConstView { .. } | Op::Const { .. } | Op::LoadView { .. } | Op::Loop { .. } => {}
                Op::Load { index, .. } => {
                    if *index >= range.start {
                        *index -= n;
                    }
                }
                Op::Store { x, index } => {
                    if *index >= range.start {
                        *index -= n;
                    }
                    if *x >= range.start {
                        *x -= n;
                    }
                }
                Op::Cast { x, .. } => {
                    if *x >= range.start {
                        *x -= n;
                    }
                }
                Op::Reduce { x, .. } => {
                    if *x >= range.start {
                        *x -= n;
                    }
                }
                Op::Unary { x, .. } => {
                    if *x >= range.start {
                        *x -= n;
                    }
                }
                Op::Binary { x, y, .. } => {
                    if *x >= range.start {
                        *x -= n;
                    }
                    if *y >= range.start {
                        *y -= n;
                    }
                }
                Op::Accumulate { x, .. } => {
                    if *x >= range.start {
                        *x -= n;
                    }
                }
                Op::DeclareAcc { .. } => {}
                Op::EndLoop { .. } => {}
            }
        }
    }

    fn dead_code_elimination(&mut self) {
        let mut params = Vec::new();
        for op_id in 0..self.ops.len() {
            if matches!(
                self.ops[op_id],
                Op::Store { .. } | Op::Loop { .. } | Op::EndLoop { .. } | Op::DeclareAcc { .. }
            ) {
                params.push(op_id);
            }
        }
        let mut needed = Set::with_capacity_and_hasher(self.ops.len(), BuildHasherDefault::new());
        while let Some(param) = params.pop() {
            needed.insert(param);
            match self.ops[param] {
                Op::ConstView { .. } => unreachable!(),
                Op::Const(..) => {}
                Op::LoadView { .. } => unreachable!(),
                Op::Load { index, .. } => {
                    params.push(index);
                }
                Op::Store { x, index } => {
                    params.push(x);
                    params.push(index);
                }
                Op::Cast { x, .. } => {
                    params.push(x);
                }
                Op::Unary { x, .. } => {
                    params.push(x);
                }
                Op::Binary { x, y, .. } => {
                    params.push(x);
                    params.push(y);
                }
                Op::Loop { .. } => {}
                Op::Reduce { x, .. } => {
                    params.push(x);
                }
                Op::Accumulate { x, .. } => {
                    params.push(x);
                }
                Op::DeclareAcc { .. } => {}
                Op::EndLoop { .. } => {}
            }
        }
        for op_id in (0..self.ops.len()).rev() {
            if needed.contains(&op_id) {
                continue;
            }
            // Remove this op from kernel
            self.ops.remove(op_id);
            self.decrement_range(op_id..self.ops.len(), 1);
        }
    }

    fn loop_invariant_code_motion(&mut self) {}

    fn common_subexpression_elimination(&mut self) {
        // TODO deduplication should preserve loop boundaries
        let mut unique_stack: Vec<Map<Op, OpId>> = Vec::new();
        let mut remaps = Map::with_hasher(BuildHasherDefault::new());
        for (op_id, op) in self.ops.iter().enumerate() {
            match op {
                Op::Loop { .. } => {
                    unique_stack.push(Map::with_capacity_and_hasher(10, BuildHasherDefault::new()));
                }
                Op::EndLoop => {
                    unique_stack.pop();
                }
                _ => {
                    for unique in &unique_stack {
                        if let Some(&id) = unique.get(op) {
                            remaps.insert(op_id, id);
                            break;
                        }
                    }

                    if !remaps.contains_key(&op_id)
                        && !matches!(
                            op,
                            Op::Load { .. } | Op::Loop { .. } | Op::DeclareAcc { .. } | Op::Store { .. }
                        )
                    {
                        unique_stack.last_mut().unwrap().insert(op.clone(), op_id);
                    }
                }
            }
        }
        self.remap(&remaps);
    }

    fn move_constants_to_beginning(&mut self) {
        let tail = self.ops.split_off(6);
        let mut remaps = Map::with_hasher(BuildHasherDefault::new());
        let n_constants = tail.iter().filter(|op| matches!(op, Op::Const(_))).count();

        for (i, op) in tail.iter().enumerate() {
            if matches!(op, Op::Const(_)) {
                let new_index = self.ops.len();
                self.ops.push(op.clone());
                remaps.insert(i + 6 + n_constants, new_index);
            }
        }
        self.ops.extend(tail);
        let ops_len = self.ops.len();
        increment(&mut self.ops[6 + remaps.len()..], remaps.len(), 6..ops_len);
        self.remap(&remaps);
    }

    /// Constant folding
    fn constant_folding(&mut self) {
        let mut remaps = Map::with_hasher(BuildHasherDefault::new());
        for op_id in 0..self.ops.len() {
            match self.ops[op_id] {
                Op::ConstView { .. } | Op::LoadView { .. } => unreachable!(),
                Op::Const { .. }
                | Op::Load { .. }
                | Op::Store { .. }
                | Op::Loop { .. }
                | Op::EndLoop { .. }
                | Op::DeclareAcc { .. }
                | Op::Accumulate { .. }
                | Op::Reduce { .. } => {}
                Op::Cast { x, dtype } => {
                    if let Op::Const(x) = self.ops[x] {
                        self.ops[op_id] = Op::Const(x.cast(dtype));
                    }
                }
                Op::Unary { x, uop } => {
                    if let Op::Const(x) = self.ops[x] {
                        self.ops[op_id] = Op::Const(x.unary(uop));
                    }
                }
                Op::Binary { x, y, bop } => match (&self.ops[x], &self.ops[y]) {
                    (&Op::Const(cx), &Op::Const(cy)) => {
                        self.ops[op_id] = Op::Const(Constant::binary(cx, cy, bop));
                    }
                    (&Op::Const(cx), _) => match bop {
                        BOp::Add => {
                            if cx.is_zero() {
                                remaps.insert(op_id, y);
                            }
                        }
                        BOp::Sub => todo!(),
                        BOp::Mul => {
                            if cx.is_zero() {
                                remaps.insert(op_id, x);
                            } else if cx.is_one() {
                                remaps.insert(op_id, y);
                            }
                        }
                        BOp::Div => todo!(),
                        BOp::Pow => todo!(),
                        BOp::Mod => todo!(),
                        BOp::Cmplt => todo!(),
                        BOp::Cmpgt => todo!(),
                        BOp::Max => todo!(),
                        BOp::Or => todo!(),
                        BOp::And => todo!(),
                        BOp::BitXor => todo!(),
                        BOp::BitOr => todo!(),
                        BOp::BitAnd => todo!(),
                        BOp::BitShiftLeft => todo!(),
                        BOp::BitShiftRight => todo!(),
                        BOp::NotEq => todo!(),
                    },
                    (_, &Op::Const(cy)) => match bop {
                        BOp::Add => {
                            if cy.is_zero() {
                                remaps.insert(op_id, x);
                            }
                        }
                        BOp::Sub => {
                            if cy.is_zero() {
                                remaps.insert(op_id, x);
                            }
                        }
                        BOp::Mul => {
                            if cy.is_zero() {
                                remaps.insert(op_id, y);
                            } else if cy.is_one() {
                                remaps.insert(op_id, x);
                            }
                        }
                        BOp::Div => {
                            if cy.is_zero() {
                                panic!("Division by zero constant.");
                            } else if cy.is_one() {
                                remaps.insert(op_id, x);
                            }
                        }
                        BOp::Pow => todo!(),
                        BOp::Mod => {
                            if cy.is_zero() {
                                panic!("Modulo by zero constant.");
                            } else if cy.is_one() {
                                self.ops[op_id] = Op::Const(cy.dtype().zero_constant());
                            }
                        }
                        BOp::Cmplt => {}
                        BOp::Cmpgt => {}
                        BOp::Max => todo!(),
                        BOp::Or => todo!(),
                        BOp::And => {}
                        BOp::BitXor => todo!(),
                        BOp::BitOr => todo!(),
                        BOp::BitAnd => todo!(),
                        BOp::BitShiftLeft => todo!(),
                        BOp::BitShiftRight => todo!(),
                        BOp::NotEq => {}
                    },
                    _ => {}
                },
            }
        }
        self.remap(&remaps);
    }

    fn remap(&mut self, remap: &Map<OpId, OpId>) {
        let h = |x: &mut usize| {
            if let Some(v) = remap.get(x) {
                *x = *v;
            }
        };
        for op in &mut self.ops {
            match op {
                Op::ConstView { .. } => {}
                Op::LoadView { .. } => {}
                Op::Reduce { x, .. } => h(x),
                Op::Const(_) => {}
                Op::Load { index, .. } => h(index),
                Op::Store { x, index } => {
                    h(x);
                    h(index);
                }
                Op::Cast { x, .. } => h(x),
                Op::Unary { x, .. } => h(x),
                Op::Binary { x, y, .. } => {
                    h(x);
                    h(y);
                }
                Op::DeclareAcc { .. } => {}
                Op::Loop { .. } => {}
                Op::Accumulate { x, .. } => h(x),
                Op::EndLoop { .. } => {}
            }
        }
    }

    /// Unroll all loops with dimension <= loop_unroll_size
    fn loop_unrolling(&mut self, loop_unroll_size: usize) {
        fn unroll_innermost_loop(ir: &mut Vec<Op>, loop_unroll_size: usize) -> bool {
            let mut stack = Vec::new();
            let mut innermost_range: Option<std::ops::Range<usize>> = None;

            // Reverse scan to find the innermost matched Loop..EndLoop
            for (i, op) in ir.iter().enumerate().rev() {
                match op {
                    Op::EndLoop => {
                        stack.push(i);
                    }
                    &Op::Loop { dim, .. } => {
                        if dim > loop_unroll_size {
                            stack.pop();
                            continue;
                        }
                        if let Some(end) = stack.pop() {
                            innermost_range = Some(i..end + 1);
                            break; // Unroll one innermost loop per call
                        }
                    }
                    _ => {}
                }
            }

            let Some(range) = innermost_range else {
                return false; // no matched loop found
            };

            let Op::Loop { dim, .. } = ir[range.start] else {
                unreachable!("Expected Op::Loop at start of matched loop range");
            };

            /*for (i, op) in ir.iter().enumerate() {
                println!("{i} -> {op:?}");
            }
            println!();*/

            let mut body = ir.split_off(range.start);
            let mut tail = body.split_off(range.end - range.start);
            body.pop();

            // Append body dim times
            for i in 0..dim {
                let mut body = body.clone();
                let n = body.len();
                increment(&mut body, i * n, range.clone());
                body[0] = Op::Const(Constant::U32(i as u32));

                ir.extend(body);
            }

            increment(&mut tail, (dim - 1) * body.len() - 1, range.end..usize::MAX);
            increment(&mut tail, (dim - 1) * body.len(), range);
            ir.extend(tail);

            /*for (i, op) in ir.iter().enumerate() {
                println!("{i} -> {op:?}");
            }*/
            //panic!();

            true
        }
        while unroll_innermost_loop(&mut self.ops, loop_unroll_size) {}
    }
}

fn get_axes(ops: &[Op]) -> Vec<OpId> {
    let mut axes = Vec::new();
    for (i, op) in ops.iter().enumerate() {
        if matches!(op, Op::Loop { .. }) {
            axes.push(i);
        }
    }
    axes
}

fn increment(ops: &mut [Op], d: usize, range: Range<usize>) {
    let h = |x: &mut usize| {
        if range.contains(x) {
            *x += d
        }
    };
    for op in ops {
        match op {
            Op::ConstView { .. }
            | Op::Const { .. }
            | Op::LoadView { .. }
            | Op::Loop { .. }
            | Op::DeclareAcc { .. }
            | Op::EndLoop { .. } => {}
            Op::Load { index, .. } => h(index),
            Op::Store { x, index } => {
                h(index);
                h(x);
            }
            Op::Cast { x, .. } | Op::Reduce { x, .. } | Op::Unary { x, .. } | Op::Accumulate { x, .. } => h(x),
            Op::Binary { x, y, .. } => {
                h(x);
                h(y);
            }
        }
    }
}
