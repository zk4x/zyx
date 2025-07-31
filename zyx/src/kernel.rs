use crate::{
    DType, Map,
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
    pub shape: Vec<Dim>,
    pub n_outputs: u32,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Op {
    //Sink { stores: Vec<Op> }, // A way to put multiple stores in one kernel
    ConstView { value: Constant, view: View },
    Const { value: Constant },
    Index { id: u8 },
    LoadView { dtype: DType, view: View },
    Load { dtype: DType, index: OpId },
    Store { x: OpId, index: OpId },
    Cast { x: OpId, dtype: DType },
    Unary { x: OpId, uop: UOp },
    Binary { x: OpId, y: OpId, bop: BOp },
    Loop { dtype: DType, rop: ROp, dims: Vec<Dim> },
    Reduce { x: OpId, rop: ROp, dims: Vec<Dim> }, // Loops are always reduce loops
}

#[derive(Debug)]
pub struct Cache {
    pub device_infos: Map<DeviceInfo, u32>,
    pub kernels: Map<Kernel, u32>,
    // Finished optimizations of kernels for given devices
    // kernel id, device info id => optimization, time to run in nanos
    pub optimizations: Map<(u32, u32), (Optimization, u64)>,
    // This last one is not stored to disk
    // kernel id, device id => program id
    pub programs: Map<(u32, u32), ProgramId>,
}

#[derive(Debug)]
pub enum Optimization {
    Basic { shape: Vec<Dim> },
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
        println!("Kernel shape {:?}", self.shape);
        for (i, op) in self.ops.iter().enumerate() {
            match op {
                Op::Const { value } => println!("{i:>3} CONST {value}"),
                Op::Index { id } => println!("{i:>3} INDEX {id}"),
                Op::ConstView { value, view } => println!("{i:>3} CONST VIEW {value} {view}"),
                Op::LoadView { dtype, view } => println!("{i:>3} LOAD VIEW {dtype} {view}"),
                Op::Load { dtype, index } => println!("{i:>3} LOAD {dtype} at {index}"),
                Op::Store { x, index } => println!("{i:>3} STORE {x} at {index}"),
                Op::Cast { x, dtype } => println!("{i:>3} CAST {x} {dtype:?}"),
                Op::Unary { x, uop } => println!("{i:>3} UNARY {uop:?} {x}"),
                Op::Binary { x, y, bop } => println!("{i:>3} BINARY {bop:?} {x} {y}"),
                Op::Loop { dtype, rop, dims } => println!(
                    "{i:>3} ACC {dtype} LOOP {} dims={dims:?}",
                    match rop {
                        ROp::Sum => "SUM",
                        ROp::Max => "MAX",
                    }
                ),
                Op::Reduce { x, rop, dims } => println!(
                    "{i:>3} ENDLOOP {} {x}, dims={dims:?}",
                    match rop {
                        ROp::Sum => "SUM",
                        ROp::Max => "MAX",
                    }
                ),
            }
        }
    }

    pub fn flop_mem_rw(&self) -> (u128, u128, u128) {
        (0, 0, 0)
    }

    pub(super) fn default_optimization(&self, dev_info: &DeviceInfo) -> Optimization {
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
                            let r = (f[i] as isize - f[j] as isize).abs()
                                + (f[i] as isize - f[k] as isize).abs();
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

        let n = self.shape.iter().product();
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

        Optimization::Basic { shape }
    }

    pub fn apply_optimization(&mut self, optimization: &Optimization) {
        match optimization {
            Optimization::Basic { shape } => {
                let n = self.shape.len();
                self.apply_movement(|view| view.reshape(0..n, &shape));
                self.shape = shape.clone();
            }
        }

        //let loop_unroll_size = 8;

        self.insert_loops_for_reduces();
        self.unfold_views();

        let mut kernel = self.clone();
        loop {
            //self.constant_folding();
            //self.loop_unrolling(loop_unroll_size);
            //self.deduplicate();
            //self.dead_code_elimination();
            //self.loop_invariant_code_motion();
            if *self == kernel {
                break;
            }
            kernel = self.clone();
        }
    }

    fn insert_loops_for_reduces(&mut self) {
        // Check the reduce op, trace all of it's dependencies,
        // put Loop op before dependency with lowest ID
        // increase all ids higher than that by one

        let reduce_ops: Vec<OpId> = self
            .ops
            .iter()
            .enumerate()
            .filter(|(_, op)| matches!(op, Op::Reduce { .. }))
            .map(|(i, _)| i)
            .collect();
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
                    Op::Index { .. } => {}
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
                    Op::Loop { .. } => unreachable!(),
                }
            }
            self.ops.insert(min_param, Op::Loop { dtype: acc_dtype.unwrap(), rop, dims });
            self.increment_range(min_param + 1..self.ops.len(), 1);
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
                Op::ConstView { value, ref view } => {
                    self.ops[op_id] = Op::Const { value };
                }
                Op::LoadView { dtype, ref view } => {
                    let mut ops = Vec::new();
                    let view = view.clone();
                    let mut index = new_op(&mut ops, Op::Const { value: Constant::U32(0) });
                    for (id, dim) in view.0.last().unwrap().iter().enumerate() {
                        let stride = Constant::U32(dim.st as u32);
                        let x = new_op(&mut ops, Op::Index { id: id as u8 });
                        let y = new_op(&mut ops, Op::Const { value: stride });
                        let x = new_op(&mut ops, Op::Binary { x, y, bop: BOp::Mul });
                        index = new_op(&mut ops, Op::Binary { x, y: index, bop: BOp::Add });
                    }
                    let n = ops.len() - 1;
                    self.ops[op_id] = Op::Load { dtype, index: op_id + n };
                    for op in ops.into_iter().rev() {
                        self.ops.insert(op_id, op);
                    }
                    self.increment_range(op_id..op_id + n + 1, op_id);
                    self.increment_range(op_id + n + 2..self.ops.len(), n + 1);
                    op_id += n + 1;
                }
                Op::Store { x, .. } => {
                    let mut ops = Vec::new();
                    let mut index = new_op(&mut ops, Op::Const { value: Constant::U32(0) });
                    let mut st = 1;
                    let shape = self.shape.clone();
                    for (id, d) in shape.iter().enumerate().rev() {
                        let stride = Constant::U32(st as u32);
                        let x = new_op(&mut ops, Op::Index { id: id as u8 });
                        let y = new_op(&mut ops, Op::Const { value: stride });
                        let x = new_op(&mut ops, Op::Binary { x, y, bop: BOp::Mul });
                        index = new_op(&mut ops, Op::Binary { x, y: index, bop: BOp::Add });
                        st *= d;
                    }
                    let n = ops.len() - 1;
                    self.ops[op_id] = Op::Store { x, index: op_id + n };
                    for op in ops.into_iter().rev() {
                        self.ops.insert(op_id, op);
                    }
                    self.increment_range(op_id..op_id + n + 1, op_id);
                    self.increment_range(op_id + n + 2..self.ops.len(), n + 1);
                    op_id += n + 1;
                }
                _ => {}
            }
            op_id += 1;
        }
    }

    fn increment_range(&mut self, range: Range<usize>, n: usize) {
        for op in &mut self.ops[range] {
            match op {
                Op::ConstView { .. }
                | Op::Const { .. }
                | Op::Index { .. }
                | Op::LoadView { .. }
                | Op::Loop { .. } => {}
                Op::Load { index, .. } => {
                    *index += n;
                }
                Op::Store { x, index } => {
                    *index += n;
                    *x += n;
                }
                Op::Cast { x, .. } => {
                    *x += n;
                }
                Op::Reduce { x, .. } => {
                    *x += n;
                }
                Op::Unary { x, .. } => {
                    *x += n;
                }
                Op::Binary { x, y, .. } => {
                    *x += n;
                    *y += n;
                }
            }
        }
    }

    fn dead_code_elimination(&mut self) {}

    fn loop_invariant_code_motion(&mut self) {}

    fn deduplicate(&mut self) {}

    /// Constant folding
    fn constant_folding(&mut self) {
        let mut change = true;
        while change {
            change = false;
            for op_id in 0..self.ops.len() {
                match self.ops[op_id] {
                    Op::ConstView { .. } | Op::LoadView { .. } => unreachable!(),
                    Op::Const { .. }
                    | Op::Index { .. }
                    | Op::Load { .. }
                    | Op::Store { .. }
                    | Op::Loop { .. }
                    | Op::Reduce { .. } => {}
                    Op::Cast { x, dtype } => {
                        if let Op::Const { value } = self.ops[x] {
                            self.ops[op_id] = Op::Const { value: value.cast(dtype) };
                            change = true;
                        }
                    }
                    Op::Unary { x, uop } => {
                        if let Op::Const { value } = self.ops[x] {
                            self.ops[op_id] = Op::Const { value: value.unary(uop) };
                            change = true;
                        }
                    }
                    Op::Binary { x, y, bop } => match (&self.ops[x], &self.ops[y]) {
                        (&Op::Const { value: cx }, &Op::Const { value: cy }) => {
                            self.ops[op_id] = Op::Const { value: Constant::binary(cx, cy, bop) };
                            change = true;
                        }
                        (&Op::Const { value: cx }, _) => {
                            todo!()
                        }
                        (x, &Op::Const { value: cy }) => match bop {
                            BOp::Add => {
                                if cy.is_zero() {
                                    self.ops[op_id] = x.clone();
                                    change = true;
                                }
                            }
                            BOp::Sub => todo!(),
                            BOp::Mul => {
                                if cy.is_zero() {
                                    self.ops[op_id] = Op::Const { value: cy };
                                    change = true;
                                } else if cy.is_one() {
                                    self.ops[op_id] = x.clone();
                                    change = true;
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
                        _ => {}
                    },
                }
            }
        }
    }

    // Unroll all loops with dimension <= loop_unroll_size
    fn loop_unrolling(&mut self, loop_unroll_size: usize) {}
}
