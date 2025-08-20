use crate::{
    backend::{Device, DeviceInfo, ProgramId}, dtype::Constant, graph::{BOp, ROp, UOp}, optimizer::Optimizer, shape::Dim, view::View, DType, Map, Set
};
use std::{
    fmt::Display,
    hash::BuildHasherDefault,
    ops::{Range, RangeBounds},
};

pub type OpId = usize;

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Kernel {
    pub ops: Vec<Op>,
    //pub shape: Vec<Dim>,
    pub n_outputs: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Scope {
    Global,
    Local,
    Register,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Op {
    // ops that exist only in kernelizer
    ConstView { value: Constant, view: View },
    LoadView { dtype: DType, view: View },
    StoreView { src: OpId, dtype: DType },
    Reduce { x: OpId, rop: ROp, dims: Vec<Dim> },

    // ops that only exist after unfolding views and reduces
    Const(Constant),
    Define { dtype: DType, scope: Scope, ro: bool, len: Dim }, // len is 0 for globals
    Load { src: OpId, index: OpId },
    Loop { dim: Dim, scope: Scope },
    EndLoop,

    //DeclareAcc { dtype: DType, rop: ROp },
    //Accumulate { x: OpId, rop: ROp },

    // ops that exist in both
    Store { dst: OpId, src: OpId, index: OpId },
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
    pub optimizations: Map<(u32, u32), Optimizer>,
    // This last one is not stored to disk
    // kernel id, device id => program id
    pub programs: Map<(u32, u32), ProgramId>,
}

impl Display for Scope {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Scope::Global => "GLOBAL",
            Scope::Local => "LOCAL",
            Scope::Register => "REG",
        })
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
            const RED: &str = "\x1b[31m";
            const GREEN: &str = "\x1b[32m";
            const YELLOW: &str = "\x1b[33m";
            const BLUE: &str = "\x1b[34m";
            const MAGENTA: &str = "\x1b[35m";
            const CYAN: &str = "\x1b[36m";
            const RESET: &str = "\x1b[0m";
            match op {
                Op::ConstView { value, view } => println!("{i:>3} {CYAN}CONST VIEW{RESET} {value} {view}"),
                Op::LoadView { dtype, view } => println!("{i:>3} {CYAN}LOAD VIEW{RESET} {dtype} {view}"),
                Op::StoreView { src, dtype } => println!("{i:>3} {CYAN}STORE VIEW{RESET} {src} {dtype}"),
                Op::Reduce { x, rop, dims } => {
                    println!(
                        "{i:>3} {CYAN}REDUCE{RESET} {} {x}, dims={dims:?}",
                        match rop {
                            ROp::Sum => "SUM",
                            ROp::Max => "MAX",
                        }
                    );
                }
                Op::Define { dtype, scope, ro, len } => {
                    println!("{i:>3} {YELLOW}DEFINE{RESET} {scope} {dtype}, len={len}, ro={ro}");
                }
                Op::Const(x) => println!("{i:>3} {MAGENTA}CONST{RESET} {x}"),
                Op::Load { src, index } => println!("{i:>3} {GREEN}LOAD{RESET} p{src}[{index}]"),
                Op::Store { dst, src, index } => println!("{i:>3} {RED}STORE{RESET} p{dst}[{index}] <- {src}"),
                Op::Cast { x, dtype } => println!("{i:>3} CAST {x} {dtype:?}"),
                Op::Unary { x, uop } => println!("{i:>3} UNARY {uop:?} {x}"),
                Op::Binary { x, y, bop } => println!("{i:>3} BINARY {bop:?} {x} {y}"),
                Op::Loop { dim, scope } => println!("{i:>3} {BLUE}LOOP{RESET} {scope} dim={dim}"),
                Op::EndLoop => println!("{i:>3} {BLUE}ENDLOOP{RESET}"),
            }
        }
    }

    pub const fn flop_mem_rw(&self) -> (u128, u128, u128) {
        (0, 0, 0)
    }

    pub fn is_reduce(&self) -> bool {
        self.ops.iter().any(|x| matches!(x, Op::Reduce { .. }))
    }

    /*pub(super) fn new_optimizer(&self, dev_info: &DeviceInfo) -> Optimizer {
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

        let n: Dim = self.shape().iter().product();

        let mut d = dev_info.max_local_threads;
        while n % d != 0 {
            d -= 1;
        }
        debug_assert_eq!(n % d, 0);

        let local_work_size: Vec<usize> = get_equal_factors(d).into();
        let global_work_size: Vec<usize> = get_equal_factors(n / d).into();

        debug_assert_eq!(global_work_size.iter().product::<Dim>(), n / d);
        debug_assert_eq!(local_work_size.iter().product::<Dim>(), d);

        //Optimizer::Default(Optimization::Basic { global_work_size, local_work_size, loop_unroll_size: 0 })
        todo!()
    }*/

    pub fn shape(&self) -> Vec<Dim> {
        if self.ops.iter().any(|op| matches!(op, Op::Loop { .. })) {
            return self
                .ops
                .iter()
                .filter_map(|op| {
                    if let Op::Loop { dim, scope } = op {
                        if matches!(scope, Scope::Global | Scope::Local) {
                            Some(*dim)
                        } else {
                            None
                        }
                    } else {
                        None
                    }
                })
                .collect();
        }
        let mut reduce_dims = 0;
        for op in self.ops.iter().rev() {
            match op {
                Op::ConstView { view, .. } | Op::LoadView { view, .. } => {
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

    pub fn unfold_shape(&mut self, global_work_size: &[Dim], local_work_size: &[Dim]) {
        let k = global_work_size.len() + local_work_size.len();
        let n = self.ops.len();
        increment(&mut self.ops, k, 0..n);
        for &dim in local_work_size.iter().rev() {
            self.ops.insert(0, Op::Loop { dim, scope: Scope::Local });
        }
        for &dim in global_work_size.iter().rev() {
            self.ops.insert(0, Op::Loop { dim, scope: Scope::Global });
        }
    }

    pub fn unfold_reduces(&mut self) {
        // Check the reduce op, trace all of it's dependencies,
        // put Loop op before dependency with lowest ID
        // increase all ids higher than that by one

        #[allow(clippy::needless_collect)] // false positive
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
                    Op::Const { .. } | Op::Load { .. } | Op::Loop { .. } | Op::EndLoop => unreachable!(),
                    Op::Define { .. } => {}
                    Op::LoadView { dtype, .. } => {
                        if acc_dtype.is_none() {
                            acc_dtype = Some(dtype);
                        }
                    }
                    Op::StoreView { src, .. } => {
                        params.push(src);
                        if src < min_param {
                            min_param = src;
                        }
                    }
                    Op::Store { src, index, .. } => {
                        params.push(index);
                        if index < min_param {
                            min_param = index;
                        }
                        params.push(src);
                        if src < min_param {
                            min_param = src;
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
                }
            }
            let dtype = acc_dtype.unwrap();

            let n_dims = dims.len();
            self.ops[op_id] = Op::EndLoop;

            let mut body = self.ops.split_off(min_param);
            let mut tail = body.split_off(op_id - self.ops.len());

            // Declare accumulator
            let c_0 = self.ops.len();
            self.ops.push(Op::Const(dtype.zero_constant()));
            let acc_init = self.ops.len();
            self.ops.push(Op::Const(match rop {
                ROp::Sum => dtype.zero_constant(),
                ROp::Max => dtype.min_constant(),
            }));
            let acc = self.ops.len();
            self.ops.push(Op::Define { dtype, scope: Scope::Register, ro: false, len: 1 });
            self.ops.push(Op::Store { dst: min_param + 2, src: acc_init, index: c_0 });

            // Insert Loops
            for dim in dims {
                self.ops.push(Op::Loop { dim, scope: Scope::Register });
            }

            increment(&mut body, 4 + n_dims, min_param..);
            self.ops.extend(body);

            // Insert reduce op (load + binary + store)
            let y = self.ops.len();
            self.ops.push(Op::Load { src: acc, index: c_0 });
            self.ops.push(Op::Binary {
                x: x + n_dims + 4,
                y,
                bop: match rop {
                    ROp::Sum => BOp::Add,
                    ROp::Max => BOp::Max,
                },
            });
            self.ops.push(Op::Store { dst: acc, src: y + 1, index: c_0 });

            // Insert endloops
            for _ in 0..(n_dims - 1) {
                self.ops.push(Op::EndLoop);
            }

            // Update tail by adding load acc before all ops referencing op_id
            // op_id -> acc
            let mut i = 0;
            while i < tail.len() {
                match &mut tail[i] {
                    Op::ConstView { .. } | Op::LoadView { .. } | Op::Define { .. } | Op::Loop { .. } | Op::EndLoop => {}
                    Op::StoreView { src, .. } => {
                        if *src == op_id {
                            *src = self.ops.len() + i;
                            tail.insert(i, Op::Load { src: acc, index: c_0 });
                        }
                    }
                    Op::Reduce { x, .. } | Op::Cast { x, .. } | Op::Unary { x, .. } => {
                        if *x == op_id {
                            *x = self.ops.len() + i;
                            tail.insert(i, Op::Load { src: acc, index: c_0 });
                        }
                    }
                    Op::Const(_) => todo!(),
                    Op::Load { src, index } | Op::Store { src, index, .. } => {
                        assert_ne!(*index, op_id);
                        if *src == op_id {
                            *src = self.ops.len() + i;
                            tail.insert(i, Op::Load { src: acc, index: c_0 });
                        }
                    }
                    Op::Binary { x, y, .. } => {
                        let tail1 = if *x == op_id {
                            *x = self.ops.len() + i;
                            true
                        } else {
                            false
                        };
                        let tail2 = if *y == op_id {
                            *y = self.ops.len() + i;
                            true
                        } else {
                            false
                        };
                        if tail1 {
                            tail.insert(i, Op::Load { src: acc, index: c_0 });
                        }
                        if tail2 {
                            tail.insert(i, Op::Load { src: acc, index: c_0 });
                        }
                    }
                }
                i += 1;
            }

            self.ops.extend(tail);

            //self.debug();
            //panic!();
        }
    }

    pub fn define_globals(&mut self) {
        let mut loads = Vec::new();
        let mut stores = Vec::new();
        for op in &self.ops {
            match *op {
                Op::LoadView { dtype, .. } => {
                    loads.push(dtype);
                }
                Op::StoreView { dtype, .. } => {
                    stores.push(dtype);
                }
                _ => {}
            }
        }
        let k = loads.len() + stores.len();
        let temp_ops = self.ops.split_off(0);
        for dtype in loads {
            self.ops.push(Op::Define { dtype, scope: Scope::Global, ro: true, len: 0 });
        }
        for dtype in stores {
            self.ops.push(Op::Define { dtype, scope: Scope::Global, ro: false, len: 0 });
        }
        self.ops.extend(temp_ops);
        let n = self.ops.len();
        increment(&mut self.ops, k, 0..n);
    }

    pub fn unfold_views(&mut self) {
        // First we generate the whole view into a new vec,
        // then we insert the vec into existing ops
        // Convert view
        fn new_op(ops: &mut Vec<Op>, op: Op) -> OpId {
            let op_id = ops.len();
            ops.push(op);
            op_id
        }

        let n_loads = self.ops.iter().filter(|op| matches!(op, Op::LoadView { .. })).count();
        let mut load_id = 0;
        let mut store_id = n_loads;
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
                    let axes = get_axes(&ops[0..op_id]);
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
                            } else if dim.d == 1 {
                                new_op(ops, Op::Const(Constant::U32(0)))
                            } else {
                                axes[a]
                            };
                            //println!("ost: {ost}, a: {a:?}, {dim:?}");
                            // Offset
                            let t = if dim.lp != 0 {
                                let lp = new_op(ops, Op::Const(Constant::U32(dim.lp.unsigned_abs() as u32)));
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

                    let z = new_op(ops, Op::Load { src: load_id, index: offset });

                    let pcd = new_op(ops, Op::Cast { x: pc, dtype });
                    // Nullify z if padding condition is false (if there is padding at that index)
                    _ = new_op(ops, Op::Binary { x: pcd, y: z, bop: BOp::Mul });

                    let n = self.ops.len();
                    self.ops.extend(temp_ops);
                    increment(&mut self.ops[n..], n - op_id - 1, op_id..);
                    op_id = n;
                    load_id += 1;
                    continue;
                }
                Op::StoreView { src, .. } => {
                    let temp_ops: Vec<Op> = self.ops.split_off(op_id + 1);
                    self.ops.pop();
                    let axes = get_axes(&self.ops);
                    let mut index = new_op(&mut self.ops, Op::Const(Constant::U32(0)));
                    let mut st = 1;
                    let shape = self.shape();
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

                    _ = new_op(&mut self.ops, Op::Store { dst: store_id, src, index });

                    let n = self.ops.len();
                    /*for (i, op) in self.ops.iter().enumerate() {
                        println!("{i} -> {op:?}");
                    }
                    println!("n={n}");*/
                    self.ops.extend(temp_ops);
                    increment(&mut self.ops[n..], n - op_id - 1, op_id..);
                    op_id = n;
                    store_id += 1;
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
                Op::ConstView { .. }
                | Op::Const { .. }
                | Op::LoadView { .. }
                | Op::StoreView { .. }
                | Op::Loop { .. }
                | Op::Define { .. }
                | Op::EndLoop => {}
                Op::Load { src, index } => {
                    if *src >= range.start {
                        *src -= n;
                    }
                    if *index >= range.start {
                        *index -= n;
                    }
                }
                Op::Store { dst, src, index } => {
                    if *index >= range.start {
                        *index -= n;
                    }
                    if *dst >= range.start {
                        *dst -= n;
                    }
                    if *src >= range.start {
                        *src -= n;
                    }
                }
                Op::Cast { x, .. } | Op::Reduce { x, .. } | Op::Unary { x, .. } => {
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
            }
        }
    }

    pub fn dead_code_elimination(&mut self) {
        let mut params = Vec::new();
        for op_id in 0..self.ops.len() {
            if matches!(self.ops[op_id], Op::Store { .. } | Op::Loop { .. } | Op::EndLoop) {
                params.push(op_id);
            }
        }
        let mut needed = Set::with_capacity_and_hasher(self.ops.len(), BuildHasherDefault::new());
        while let Some(param) = params.pop() {
            if needed.insert(param) {
                match self.ops[param] {
                    Op::Const(..) | Op::Define { .. } | Op::Loop { .. } | Op::EndLoop => {}
                    Op::ConstView { .. } | Op::LoadView { .. } | Op::StoreView { .. } => unreachable!(),
                    Op::Load { src, index } => {
                        params.push(src);
                        params.push(index);
                    }
                    Op::Store { dst, src, index } => {
                        params.push(dst);
                        params.push(src);
                        params.push(index);
                    }
                    Op::Binary { x, y, .. } => {
                        params.push(x);
                        params.push(y);
                    }
                    Op::Cast { x, .. } | Op::Unary { x, .. } | Op::Reduce { x, .. } => {
                        params.push(x);
                    }
                }
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

    pub fn common_subexpression_elimination(&mut self) {
        // TODO deduplication should preserve loop boundaries
        let mut unique_stack: Vec<Map<Op, OpId>> = Vec::new();
        unique_stack.push(Map::with_capacity_and_hasher(10, BuildHasherDefault::new()));
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

                    if !remaps.contains_key(&op_id) && !matches!(op, Op::Define { .. } | Op::Loop { .. } | Op::EndLoop)
                    {
                        unique_stack.last_mut().unwrap().insert(op.clone(), op_id);
                    }
                }
            }
        }
        remap(&mut self.ops, &remaps);
    }

    pub fn move_constants_to_beginning(&mut self) {
        let n_defines = self.ops.iter().position(|op| !matches!(op, Op::Define { .. })).unwrap();
        let tail = self.ops.split_off(n_defines);
        let mut remaps = Map::with_hasher(BuildHasherDefault::new());
        let n_constants = tail.iter().filter(|op| matches!(op, Op::Const(_))).count();

        for (i, op) in tail.iter().enumerate() {
            if matches!(op, Op::Const(_)) {
                let new_index = self.ops.len();
                self.ops.push(op.clone());
                remaps.insert(i + n_defines + n_constants, new_index);
            }
        }
        self.ops.extend(tail);
        increment(&mut self.ops[n_defines + remaps.len()..], remaps.len(), n_defines..);
        remap(&mut self.ops, &remaps);
    }

    /// Constant folding
    pub fn constant_folding(&mut self) {
        let mut remaps = Map::with_hasher(BuildHasherDefault::new());
        for op_id in 0..self.ops.len() {
            match self.ops[op_id] {
                Op::ConstView { .. } | Op::LoadView { .. } | Op::StoreView { .. } | Op::Reduce { .. } => unreachable!(),
                Op::Const { .. }
                | Op::Load { .. }
                | Op::Store { .. }
                | Op::Loop { .. }
                | Op::EndLoop
                | Op::Define { .. } => {}
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
                        BOp::Add | BOp::Sub => {
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
                        BOp::Cmplt | BOp::Cmpgt | BOp::NotEq | BOp::And => {}
                        BOp::Max => todo!(),
                        BOp::Or => todo!(),
                        BOp::BitXor => todo!(),
                        BOp::BitOr => todo!(),
                        BOp::BitAnd => todo!(),
                        BOp::BitShiftLeft => todo!(),
                        BOp::BitShiftRight => todo!(),
                    },
                    _ => {}
                },
            }
        }
        remap(&mut self.ops, &remaps);
    }

    /// Unroll all loops with dimension <= `loop_unroll_size`
    pub fn loop_optimization(&mut self, loop_unroll_size: usize) {
        fn unroll_loop(ir: &mut Vec<Op>, range: Range<usize>) {
            let Op::Loop { dim, .. } = ir[range.start] else {
                unreachable!("Expected Op::Loop at start of matched loop range");
            };

            let mut body = ir.split_off(range.start);
            let mut tail = body.split_off(range.end - range.start);
            body.pop();

            // If body contains accumulator, we replace it with binary ops and DeclareAcc with constant
            /*let mut replace_acc = if body.iter().any(|op| matches!(op, Op::Accumulate { .. })) {
                ir.iter().rposition(|op| matches!(op, Op::DeclareAcc { .. }))
            } else {
                None
            };
            if let Some(decl_acc_id) = replace_acc {
                if let Op::DeclareAcc { dtype, rop } = ir[decl_acc_id] {
                    ir[decl_acc_id] = Op::Const(match rop {
                        ROp::Sum => dtype.zero_constant(),
                        ROp::Max => dtype.min_constant(),
                    });
                }
            }*/

            // Append body dim times
            for i in 0..dim {
                let mut body = body.clone();
                let n = body.len();
                increment(&mut body, i * n, range.clone());
                body[0] = Op::Const(Constant::U32(i as u32));

                /*if let Some(decl_acc_id) = replace_acc {
                    for (op_id, op) in body.iter_mut().enumerate() {
                        if let &mut Op::Accumulate { x, rop } = op {
                            *op = Op::Binary {
                                x,
                                y: decl_acc_id,
                                bop: match rop {
                                    ROp::Sum => BOp::Add,
                                    ROp::Max => BOp::Max,
                                },
                            };
                            replace_acc = Some(op_id + ir.len());
                            break;
                        }
                    }
                }*/

                ir.extend(body);
            }

            increment(&mut tail, (dim - 1) * body.len() - 1, range.end..usize::MAX);
            increment(&mut tail, (dim - 1) * body.len(), range);
            ir.extend(tail);

            /*for (i, op) in ir.iter().enumerate() {
                println!("{i} -> {op:?}");
            }*/
        }

        /*fn loop_invariant_code_motion(ir: &mut Vec<Op>, range: Range<usize>) {
            for op_id in range {
                match &ir[op_id] {
                    Op::ConstView { value, view } => todo!(),
                    Op::LoadView { dtype, view } => todo!(),
                    Op::Reduce { x, rop, dims } => todo!(),
                    Op::Const(constant) => todo!(),
                    Op::Load { dtype, index, arg_id } => todo!(),
                    Op::DeclareAcc { dtype, rop } => todo!(),
                    Op::Loop { dim, vectorize } => todo!(),
                    Op::Accumulate { x, rop } => todo!(),
                    Op::EndLoop => todo!(),
                    Op::Store { x, index } => todo!(),
                    Op::Cast { x, dtype } => todo!(),
                    Op::Unary { x, uop } => todo!(),
                    Op::Binary { x, y, bop } => todo!(),
                }
            }
        }*/

        let mut ranges = Vec::new();
        let mut stack = Vec::new();

        for (i, op) in self.ops.iter().enumerate() {
            match op {
                Op::Loop { dim, .. } => {
                    stack.push((i, dim));
                }
                &Op::EndLoop => {
                    if let Some((start, dim)) = stack.pop()
                        && *dim <= loop_unroll_size
                    {
                        ranges.push(start..i + 1);
                    }
                }
                _ => {}
            }
        }
        //println!("{ranges:?}");

        for range in ranges {
            unroll_loop(&mut self.ops, range);
        }
    }
}

fn get_axes(ops: &[Op]) -> Vec<OpId> {
    let mut axes = Vec::new();
    for (i, op) in ops.iter().enumerate() {
        match op {
            Op::Loop { .. } => {
                axes.push(i);
            }
            Op::EndLoop => {
                axes.pop();
            }
            _ => {}
        }
    }
    axes
}

fn increment(ops: &mut [Op], d: usize, range: impl RangeBounds<usize>) {
    let start = match range.start_bound() {
        std::ops::Bound::Included(x) => *x,
        std::ops::Bound::Excluded(x) => *x + 1,
        std::ops::Bound::Unbounded => 0,
    };
    let end = match range.end_bound() {
        std::ops::Bound::Included(x) => *x + 1,
        std::ops::Bound::Excluded(x) => *x,
        std::ops::Bound::Unbounded => usize::MAX,
    };
    debug_assert!(start < end);
    let range = start..end;

    let h = |x: &mut usize| {
        //println!("{x}, {range:?}, contains={}", range.contains(x));
        if range.contains(x) {
            *x += d;
        }
    };
    for op in ops {
        match op {
            Op::ConstView { .. }
            | Op::Const { .. }
            | Op::LoadView { .. }
            | Op::Loop { .. }
            | Op::Define { .. }
            | Op::EndLoop => {}
            Op::StoreView { src, .. } => {
                h(src);
            }
            Op::Load { src, index } => {
                h(src);
                h(index);
            }
            Op::Store { dst, src, index } => {
                h(dst);
                h(src);
                h(index);
            }
            Op::Cast { x, .. } | Op::Reduce { x, .. } | Op::Unary { x, .. } => h(x),
            Op::Binary { x, y, .. } => {
                h(x);
                h(y);
            }
        }
    }
}

fn remap(ops: &mut [Op], remap: &Map<OpId, OpId>) {
    let h = |x: &mut usize| {
        if let Some(v) = remap.get(x) {
            *x = *v;
        }
    };
    for op in ops {
        match op {
            Op::ConstView { .. }
            | Op::LoadView { .. }
            | Op::Const(_)
            | Op::Loop { .. }
            | Op::EndLoop
            | Op::Define { .. } => {}
            Op::StoreView { src, .. } => {
                h(src);
            }
            Op::Load { src, index, .. } => {
                h(src);
                h(index);
            }
            Op::Store { dst, src, index } => {
                h(dst);
                h(src);
                h(index);
            }
            Op::Cast { x, .. } | Op::Unary { x, .. } | Op::Reduce { x, .. } => h(x),
            Op::Binary { x, y, .. } => {
                h(x);
                h(y);
            }
        }
    }
}
