use nanoserde::{DeBin, SerBin};

use crate::{
    BLUE, CYAN, DType, GREEN, MAGENTA, Map, RED, RESET, Set, YELLOW,
    backend::{Device, DeviceInfo, ProgramId},
    dtype::Constant,
    graph::{BOp, ROp, UOp},
    optimizer::Optimizer,
    shape::{Dim, UAxis},
    view::View,
};
use std::{
    fmt::Display,
    hash::BuildHasherDefault,
    ops::{Range, RangeBounds},
};

pub type OpId = usize;
pub const IDX_T: DType = DType::U32;

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, SerBin, DeBin)]
pub struct Kernel {
    pub ops: Vec<Op>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, SerBin, DeBin)]
pub enum Scope {
    Global,
    Local,
    Register,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, SerBin, DeBin)]
pub enum Op {
    // ops that exist only in kernelizer
    ConstView { value: Constant, view: View },
    LoadView { dtype: DType, view: View },
    StoreView { src: OpId, dtype: DType },
    Reduce { x: OpId, rop: ROp, dims: Vec<Dim> },
    //MergeIndices { x: OpId, y: OpId }, // creates index for merge of loops x and y (i.e. x * y_len + y)
    //PermuteIndices(Vec<OpId>), // Permute for indices, just swapping indices around
    //PadIndex(OpId, isize, isize), // Pad index with padding
    //Unsqueeze { axis: Axis, dim: Dim } // Inserts a new loop at given axis

    // ops that exist in both
    Store { dst: OpId, x: OpId, index: OpId },
    Cast { x: OpId, dtype: DType },
    Unary { x: OpId, uop: UOp },
    Binary { x: OpId, y: OpId, bop: BOp },
    Null,

    // ops that only exist after unfolding views and reduces
    Const(Constant),
    Define { dtype: DType, scope: Scope, ro: bool, len: Dim }, // len is 0 for globals
    Load { src: OpId, index: OpId },
    Loop { dim: Dim, scope: Scope },
    EndLoop,
}

impl Op {
    fn parameters(&self) -> impl Iterator<Item = OpId> {
        match self {
            Op::ConstView { .. } => vec![],
            Op::LoadView { .. } => vec![],
            Op::StoreView { src, .. } => vec![*src],
            Op::Reduce { x, .. } => vec![*x],
            Op::Store { dst, x, index } => vec![*dst, *x, *index],
            Op::Cast { x, .. } => vec![*x],
            Op::Unary { x, .. } => vec![*x],
            Op::Binary { x, y, .. } => vec![*x, *y],
            Op::Null => vec![],
            Op::Const(..) => vec![],
            Op::Define { .. } => vec![],
            Op::Load { src, index } => vec![*src, *index],
            Op::Loop { .. } => vec![],
            Op::EndLoop => vec![],
        }
        .into_iter()
    }
}

// This is SSA representation. All ops return immutable variables.
// The Define op can define mutable variables.
// Variables defined by define op can only be accessed with Load on Store ops,
// using their src and dst fields.
/*pub enum Op {
    Store { dst: OpId, x: OpId, index: OpId },
    Cast { x: OpId, dtype: DType },
    Unary { x: OpId, uop: UOp },
    Binary { x: OpId, y: OpId, bop: BOp },
    Null,
    Const(Constant),
    Define { dtype: DType, scope: Scope, ro: bool, len: Dim }, // len is 0 for globals
    Load { src: OpId, index: OpId },
    Loop { dim: Dim, scope: Scope },
    EndLoop,
}*/

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

impl SerBin for Cache {
    fn ser_bin(&self, output: &mut Vec<u8>) {
        self.device_infos.len().ser_bin(output);
        for (key, value) in &self.device_infos {
            key.ser_bin(output);
            value.ser_bin(output);
        }
        self.kernels.len().ser_bin(output);
        for (key, value) in &self.kernels {
            key.ser_bin(output);
            value.ser_bin(output);
        }
        self.optimizations.len().ser_bin(output);
        for (key, value) in &self.optimizations {
            key.ser_bin(output);
            value.ser_bin(output);
        }
    }
}

impl DeBin for Cache {
    fn de_bin(offset: &mut usize, bytes: &[u8]) -> Result<Self, nanoserde::DeBinErr> {
        let len = usize::de_bin(offset, bytes)?;
        if len > bytes.len() - *offset {
            return Err(nanoserde::DeBinErr::new(*offset, len, bytes.len() - *offset));
        }
        let mut device_infos = Map::with_capacity_and_hasher(len, BuildHasherDefault::new());
        for _ in 0..len {
            let key = DeviceInfo::de_bin(offset, bytes)?;
            let value = u32::de_bin(offset, bytes)?;
            device_infos.insert(key, value);
        }

        let len = usize::de_bin(offset, bytes)?;
        if len > bytes.len() - *offset {
            return Err(nanoserde::DeBinErr::new(*offset, len, bytes.len() - *offset));
        }
        let mut kernels = Map::with_capacity_and_hasher(len, BuildHasherDefault::new());
        for _ in 0..len {
            let key = Kernel::de_bin(offset, bytes)?;
            let value = u32::de_bin(offset, bytes)?;
            kernels.insert(key, value);
        }

        let len = usize::de_bin(offset, bytes)?;
        if len > bytes.len() - *offset {
            return Err(nanoserde::DeBinErr::new(*offset, len, bytes.len() - *offset));
        }
        let mut optimizations = Map::with_capacity_and_hasher(len, BuildHasherDefault::new());
        for _ in 0..len {
            let k1 = u32::de_bin(offset, bytes)?;
            let k2 = u32::de_bin(offset, bytes)?;
            let key = (k1, k2);
            let value = Optimizer::de_bin(offset, bytes)?;
            optimizations.insert(key, value);
        }

        let programs = Map::with_hasher(BuildHasherDefault::new());
        Ok(Cache { device_infos, kernels, optimizations, programs })
    }
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
pub fn get_perf(flop: u64, bytes_read: u64, bytes_written: u64, nanos: u64) -> String {
    if nanos == u64::MAX {
        return format!("INF time taken");
    }
    const fn value_unit(x: u64) -> (u64, &'static str) {
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

    //let (f, f_u) = value_unit(flop);
    //let (br, br_u) = value_unit(bytes_read);
    //let (bw, bw_u) = value_unit(bytes_written);
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

    /*format!(
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
    )*/

    format!(
        "{}.{} {t_u} ~ {}.{:02} {f_us}FLOP/s, {}.{:02} {br_us}B/s r, {}.{:02} {bw_us}B/s w",
        t / 10,
        t % 10,
        fs / 100,
        fs % 100,
        brs / 100,
        brs % 100,
        bws / 100,
        bws % 100,
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
        let mut indent = String::from(" ");
        for (i, op) in self.ops.iter().enumerate() {
            match op {
                Op::ConstView { value, view } => println!("{i:>3}{indent}{CYAN}CONST VIEW{RESET} {value} {view}"),
                Op::LoadView { dtype, view } => println!("{i:>3}{indent}{CYAN}LOAD VIEW{RESET} {dtype} {view}"),
                Op::StoreView { src, dtype } => println!("{i:>3}{indent}{CYAN}STORE VIEW{RESET} {src} {dtype}"),
                Op::Reduce { x, rop, dims } => {
                    println!(
                        "{i:>3}{indent}{RED}REDUCE{RESET} {} {x}, dims={dims:?}",
                        match rop {
                            ROp::Sum => "SUM",
                            ROp::Max => "MAX",
                        }
                    );
                }
                Op::Define { dtype, scope, ro, len } => {
                    println!("{i:>3}{indent}{YELLOW}DEFINE{RESET} {scope} {dtype}, len={len}, ro={ro}");
                }
                Op::Const(x) => println!("{i:>3}{indent}{MAGENTA}CONST{RESET} {} {x}", x.dtype()),
                Op::Load { src, index } => println!("{i:>3}{indent}{GREEN}LOAD{RESET} p{src}[{index}]"),
                Op::Store { dst, x: src, index } => {
                    println!("{i:>3}{indent}{RED}STORE{RESET} p{dst}[{index}] <- {src}")
                }
                Op::Cast { x, dtype } => println!("{i:>3}{indent}CAST {x} {dtype:?}"),
                Op::Unary { x, uop } => println!("{i:>3}{indent}UNARY {uop:?} {x}"),
                Op::Binary { x, y, bop } => println!("{i:>3}{indent}BINARY {bop:?} {x} {y}"),
                Op::Loop { dim, scope } => {
                    println!("{i:>3}{indent}{BLUE}LOOP{RESET} {scope} dim={dim}");
                    indent += " ";
                }
                Op::EndLoop => {
                    indent.pop();
                    println!("{i:>3}{indent}{BLUE}ENDLOOP{RESET}");
                }
                Op::Null => {}
            }
        }
    }

    pub fn flop_mem_rw(&self) -> (u64, u64, u64) {
        let stores: Vec<OpId> =
            self.ops.iter().enumerate().filter(|(_, op)| matches!(op, Op::StoreView { .. })).map(|(i, _)| i).collect();

        let mut flop = 0;
        let mut mr = 0;
        let mut mw = 0;
        let mut visited = Map::with_hasher(BuildHasherDefault::new());

        // flop, memory read, memory write, number of elements being processed
        fn recursive(x: OpId, ops: &[Op], visited: &mut Map<OpId, u64>) -> (u64, u64, u64) {
            if visited.contains_key(&x) {
                return (0, 0, 0);
            }
            let (f, r, w, n) = match &ops[x] {
                Op::ConstView { view, .. } => (0, 0, 0, view.numel() as u64),
                Op::LoadView { view, .. } => (0, view.original_numel() as u64, 0, view.numel() as u64),
                Op::StoreView { src, .. } => {
                    let (f, r, w) = recursive(*src, ops, visited);
                    let n = visited[src];
                    (f, r, w + n, 0)
                }
                Op::Cast { x, .. } | Op::Unary { x, .. } => {
                    let (f, r, w) = recursive(*x, ops, visited);
                    let n = visited[x];
                    (f + n, r, w, n)
                }
                Op::Binary { x, y, .. } => {
                    let (fx, rx, wx) = recursive(*x, ops, visited);
                    let (fy, ry, wy) = recursive(*y, ops, visited);
                    let n = visited[x];
                    debug_assert_eq!(n, visited[y]);
                    (fx + fy + n, rx + ry, wx + wy, n)
                }
                Op::Reduce { x, dims, .. } => {
                    let (mut f, r, w) = recursive(*x, ops, visited);
                    let mut n = visited[x];
                    let rd = dims.iter().product::<usize>() as u64;
                    n /= rd;
                    f += n * (rd - 1);
                    (f, r, w, n)
                }
                Op::Const(..) => unreachable!(),
                Op::Define { .. } => unreachable!(),
                Op::Load { .. } => unreachable!(),
                Op::Loop { .. } => unreachable!(),
                Op::EndLoop => unreachable!(),
                Op::Store { .. } => unreachable!(),
                Op::Null => unreachable!(),
            };
            visited.insert(x, n);
            (f, r, w)
        }

        for store in stores {
            let (f, r, w) = recursive(store, &self.ops, &mut visited);
            flop += f;
            mr += r;
            mw += w;
        }

        //panic!("{}, {}, {}", flop, mr, mw);

        (flop, mr, mw)
    }

    pub fn is_reduce(&self) -> bool {
        self.ops.iter().any(|x| matches!(x, Op::Reduce { .. }))
    }

    pub fn contains_stores(&self) -> bool {
        self.ops.iter().any(|x| matches!(x, Op::StoreView { .. }))
    }

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

    pub fn unfold_reduces(&mut self) {
        // Check the reduce op, trace all of it's dependencies,
        // put Loop op before dependency with lowest ID
        // increase all ids higher than that by one

        while let Some(op_id) = self.ops.iter().rev().position(|op| matches!(op, Op::Reduce { .. })) {
            //for op_id in reduce_ops.into_iter().rev() {
            //println!();
            //self.debug();
            let op_id = self.ops.len() - op_id - 1;
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
                    Op::Const(c) => {
                        if acc_dtype.is_none() {
                            acc_dtype = Some(c.dtype());
                        }
                    }
                    Op::Load { src, .. } => {
                        params.push(src);
                        if src < min_param {
                            min_param = src;
                        }
                    }
                    Op::Loop { .. } | Op::EndLoop => {}
                    Op::Define { dtype, .. } => {
                        if acc_dtype.is_none() {
                            acc_dtype = Some(dtype);
                        }
                    }
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
                    Op::Store { x: src, index, .. } => {
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
                    Op::Null => unreachable!(),
                }
            }
            //println!("op_id={op_id}, min_param={min_param}");

            let dtype = acc_dtype.unwrap();

            let n_dims = dims.len();
            self.ops[op_id] = Op::EndLoop;

            let mut body = self.ops.split_off(min_param);
            let mut tail = body.split_off(op_id - self.ops.len());

            // Declare accumulator
            let c_0 = self.ops.len();
            self.ops.push(Op::Const(Constant::U32(0)));
            let acc_init = self.ops.len();
            self.ops.push(Op::Const(match rop {
                ROp::Sum => dtype.zero_constant(),
                ROp::Max => dtype.min_constant(),
            }));
            let acc = self.ops.len();
            self.ops.push(Op::Define { dtype, scope: Scope::Register, ro: false, len: 1 });
            self.ops.push(Op::Store { dst: min_param + 2, x: acc_init, index: c_0 });

            // Insert Loops
            for dim in dims {
                self.ops.push(Op::Loop { dim, scope: Scope::Register });
            }

            increment(&mut body, 4 + n_dims as isize, min_param..);
            self.ops.extend(body);

            // Insert reduce op (load + binary + store)
            let y = self.ops.len();
            self.ops.push(Op::Load { src: acc, index: c_0 });
            self.ops.push(Op::Binary {
                x: x + n_dims + 4,
                y,
                bop: match rop {
                    ROp::Sum => BOp::Add,
                    ROp::Max => BOp::Maximum,
                },
            });
            self.ops.push(Op::Store { dst: acc, x: y + 1, index: c_0 });

            // Insert endloops
            for _ in 0..n_dims {
                self.ops.push(Op::EndLoop);
            }

            // Load the accumulator for access in the tail
            self.ops.push(Op::Load { src: acc, index: c_0 });

            tail.remove(0);
            increment(&mut tail, (7 + n_dims * 2) as isize, op_id..);

            //println!("{tail:?}");

            self.ops.extend(tail);
        }
        //println!();
        //self.debug();
    }

    pub fn define_globals(&mut self) {
        let mut loads = Vec::new();
        let mut stores = Vec::new();
        for op in &self.ops {
            match *op {
                Op::LoadView { dtype, ref view } => {
                    loads.push((dtype, view.original_numel()));
                }
                Op::StoreView { dtype, .. } => {
                    stores.push((dtype, 0));
                }
                _ => {}
            }
        }
        let k = loads.len() + stores.len();
        let temp_ops = self.ops.split_off(0);
        for (dtype, len) in loads {
            self.ops.push(Op::Define { dtype, scope: Scope::Global, ro: true, len });
        }
        for (dtype, len) in stores {
            self.ops.push(Op::Define { dtype, scope: Scope::Global, ro: false, len });
        }
        self.ops.extend(temp_ops);
        let n = self.ops.len();
        increment(&mut self.ops, k as isize, 0..n);
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
                Op::ConstView { value, ref view } => {
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
                    let constant_zero = new_op(ops, Op::Const(Constant::idx(0)));
                    #[allow(unused)] // false positive
                    let mut offset = constant_zero;
                    let mut old_offset: Option<OpId> = None;
                    //println!("View");
                    //for inner in self.0.iter() { println!("{inner:?}") }
                    //println!();
                    for inner in view.0.iter().rev() {
                        //println!("\n{inner:?}");
                        // a = offset / ost % dim
                        let mut ost = 1;
                        offset = constant_zero;
                        for (a, dim) in inner.iter().enumerate().rev() {
                            let a = if let Some(old_offset) = old_offset {
                                let t_ost = ost;
                                ost *= dim.d as u64;
                                let x = if t_ost == 1 {
                                    old_offset
                                } else {
                                    let ost_c = new_op(ops, Op::Const(Constant::idx(t_ost)));
                                    new_op(ops, Op::Binary { x: old_offset, y: ost_c, bop: BOp::Div })
                                };
                                if dim.d == 1 {
                                    constant_zero
                                } else {
                                    let dimd_c = new_op(ops, Op::Const(Constant::idx(dim.d as u64)));
                                    new_op(ops, Op::Binary { x, y: dimd_c, bop: BOp::Mod })
                                }
                            } else if dim.d == 1 {
                                new_op(ops, Op::Const(Constant::idx(0u64)))
                            } else {
                                axes[a]
                            };
                            //println!("ost: {ost}, a: {a:?}, {dim:?}");
                            // Offset
                            let t = if dim.lp != 0 {
                                let lp = new_op(ops, Op::Const(Constant::idx(dim.lp.unsigned_abs() as u64)));
                                if dim.lp > 0 {
                                    new_op(ops, Op::Binary { x: a, y: lp, bop: BOp::Sub })
                                } else {
                                    new_op(ops, Op::Binary { x: a, y: lp, bop: BOp::Add })
                                }
                            } else {
                                a
                            };

                            if dim.st != 0 {
                                let stride = new_op(ops, Op::Const(Constant::idx(dim.st as u64)));
                                let x = new_op(ops, Op::Binary { x: t, y: stride, bop: BOp::Mul });
                                offset = new_op(ops, Op::Binary { x, y: offset, bop: BOp::Add });
                            }

                            // Padding condition
                            if dim.lp > 0 {
                                let lp = new_op(ops, Op::Const(Constant::idx((dim.lp - 1) as u64)));
                                let t = new_op(ops, Op::Binary { x: a, y: lp, bop: BOp::Cmpgt });
                                pc = new_op(ops, Op::Binary { x: t, y: pc, bop: BOp::And });
                            }
                            if dim.rp > 0 {
                                let rp = new_op(ops, Op::Const(Constant::idx((dim.d as isize - dim.rp) as u64)));
                                let t = new_op(ops, Op::Binary { x: a, y: rp, bop: BOp::Cmplt });
                                pc = new_op(ops, Op::Binary { x: t, y: pc, bop: BOp::And });
                            }
                        }
                        old_offset = Some(offset);
                    }

                    //let pcu32 = new_op(ops, Op::Cast { x: pc, dtype: DType::U32 });
                    //let offset = new_op(ops, Op::Binary { x: pcu32, y: offset, bop: BOp::Mul });

                    let z = new_op(ops, Op::Const(value));

                    // TODO process view
                    //self.ops[op_id] = Op::Const(value);
                    let dtype = value.dtype();
                    let pcd = new_op(ops, Op::Cast { x: pc, dtype });
                    // Nullify z if padding condition is false (if there is padding at that index)
                    _ = new_op(ops, Op::Binary { x: pcd, y: z, bop: BOp::Mul });

                    let n = self.ops.len();
                    self.ops.extend(temp_ops);
                    increment(&mut self.ops[n..], (n - op_id - 1) as isize, op_id..);
                    op_id = n;
                    continue;
                }
                Op::LoadView { dtype, ref view } => {
                    // With padding, right padding does not affect offset
                    // offset = (a0-lp0)*st0 + a1*st1 + a2*st2 + (a3-lp3)*st3 + ...
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
                    let constant_zero = new_op(ops, Op::Const(Constant::idx(0)));
                    let mut offset = constant_zero;
                    let mut old_offset: Option<OpId> = None;
                    //println!("View");
                    //for inner in self.0.iter() { println!("{inner:?}") }
                    //println!();
                    for inner in view.0.iter().rev() {
                        //println!("\n{inner:?}");
                        // a = offset / ost % dim
                        let mut ost = 1;
                        offset = constant_zero;
                        for (a, dim) in inner.iter().enumerate().rev() {
                            let a = if let Some(old_offset) = old_offset {
                                /*let ost_c = new_op(ops, Op::Const(Constant::U32(ost)));
                                ost *= dim.d as u32;
                                let x = new_op(ops, Op::Binary { x: old_offset, y: ost_c, bop: BOp::Div });
                                let dimd_c = new_op(ops, Op::Const(Constant::U32(dim.d as u32)));
                                new_op(ops, Op::Binary { x, y: dimd_c, bop: BOp::Mod })*/
                                let t_ost = ost;
                                ost *= dim.d as u64;
                                let x = if t_ost == 1 {
                                    old_offset
                                } else {
                                    let ost_c = new_op(ops, Op::Const(Constant::idx(t_ost)));
                                    new_op(ops, Op::Binary { x: old_offset, y: ost_c, bop: BOp::Div })
                                };
                                if dim.d == 1 {
                                    constant_zero
                                } else {
                                    let dimd_c = new_op(ops, Op::Const(Constant::idx(dim.d as u64)));
                                    new_op(ops, Op::Binary { x, y: dimd_c, bop: BOp::Mod })
                                }
                            } else if dim.d == 1 {
                                constant_zero
                            } else {
                                axes[a]
                            };
                            //println!("ost: {ost}, a: {a:?}, {dim:?}");
                            // Offset
                            let t = if dim.lp != 0 {
                                let lp = new_op(ops, Op::Const(Constant::idx(dim.lp.unsigned_abs() as u64)));
                                if dim.lp > 0 {
                                    new_op(ops, Op::Binary { x: a, y: lp, bop: BOp::Sub })
                                } else {
                                    new_op(ops, Op::Binary { x: a, y: lp, bop: BOp::Add })
                                }
                            } else {
                                a
                            };

                            if dim.st != 0 {
                                let stride = new_op(ops, Op::Const(Constant::idx(dim.st as u64)));
                                let x = new_op(ops, Op::Binary { x: t, y: stride, bop: BOp::Mul });
                                offset = new_op(ops, Op::Binary { x, y: offset, bop: BOp::Add });
                            }

                            // Padding condition
                            if dim.lp > 0 {
                                let lp = new_op(ops, Op::Const(Constant::idx((dim.lp - 1) as u64)));
                                let t = new_op(ops, Op::Binary { x: a, y: lp, bop: BOp::Cmpgt });
                                pc = new_op(ops, Op::Binary { x: t, y: pc, bop: BOp::And });
                            }
                            if dim.rp > 0 {
                                let rp = new_op(ops, Op::Const(Constant::idx((dim.d as isize - dim.rp) as u64)));
                                let t = new_op(ops, Op::Binary { x: a, y: rp, bop: BOp::Cmplt });
                                pc = new_op(ops, Op::Binary { x: t, y: pc, bop: BOp::And });
                            }
                        }
                        old_offset = Some(offset);
                    }

                    let pcu = new_op(ops, Op::Cast { x: pc, dtype: IDX_T });
                    let offset = new_op(ops, Op::Binary { x: pcu, y: offset, bop: BOp::Mul });

                    let z = new_op(ops, Op::Load { src: load_id, index: offset });

                    let pcd = new_op(ops, Op::Cast { x: pc, dtype });
                    // Nullify z if padding condition is false (if there is padding at that index)
                    _ = new_op(ops, Op::Binary { x: pcd, y: z, bop: BOp::Mul });

                    let n = self.ops.len();
                    self.ops.extend(temp_ops);
                    increment(&mut self.ops[n..], (n - op_id - 1) as isize, op_id..);
                    op_id = n;
                    load_id += 1;
                    continue;
                }
                Op::StoreView { src, .. } => {
                    let temp_ops: Vec<Op> = self.ops.split_off(op_id + 1);
                    self.ops.pop();
                    let axes = get_axes(&self.ops);
                    let mut index = new_op(&mut self.ops, Op::Const(Constant::idx(0u64)));
                    let mut st = 1;

                    let shape = {
                        let mut shape = Vec::new();
                        for op in &self.ops {
                            match op {
                                Op::Loop { dim, .. } => {
                                    shape.push(*dim);
                                }
                                Op::EndLoop => {
                                    shape.pop();
                                }
                                _ => {}
                            }
                        }
                        shape
                    };

                    for (id, d) in shape.iter().enumerate().rev() {
                        let stride = Constant::idx(st as u64);
                        let x = if *d > 1 {
                            axes[id]
                        } else {
                            new_op(&mut self.ops, Op::Const(Constant::idx(0)))
                        };
                        let y = new_op(&mut self.ops, Op::Const(stride));
                        let x = new_op(&mut self.ops, Op::Binary { x, y, bop: BOp::Mul });
                        index = new_op(&mut self.ops, Op::Binary { x, y: index, bop: BOp::Add });
                        st *= d;
                    }

                    _ = new_op(&mut self.ops, Op::Store { dst: store_id, x: src, index });

                    let n = self.ops.len();
                    /*for (i, op) in self.ops.iter().enumerate() {
                        println!("{i} -> {op:?}");
                    }
                    println!("n={n}");*/
                    self.ops.extend(temp_ops);
                    increment(&mut self.ops[n..], (n - op_id - 1) as isize, op_id..);
                    op_id = n;
                    store_id += 1;
                    continue;
                }
                _ => {}
            }
            op_id += 1;
        }
    }

    pub fn unfold_pows(&mut self) {
        let mut op_id = 0;
        while op_id < self.ops.len() {
            if let Op::Binary { x, y, bop } = self.ops[op_id] {
                if bop == BOp::Pow {
                    let mut tail: Vec<Op> = self.ops.split_off(op_id + 1);
                    self.ops.pop();
                    self.ops.push(Op::Unary { x, uop: UOp::Log2 });
                    self.ops.push(Op::Binary { x: op_id, y, bop: BOp::Mul });
                    self.ops.push(Op::Unary { x: op_id + 1, uop: UOp::Exp2 });
                    increment(&mut tail, 2, op_id..);
                    self.ops.extend(tail);
                }
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
                | Op::EndLoop
                | Op::Null => {}
                Op::Load { src, index } => {
                    if *src >= range.start {
                        *src -= n;
                    }
                    if *index >= range.start {
                        *index -= n;
                    }
                }
                Op::Store { dst, x: src, index } => {
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

    pub fn close_loops(&mut self) {
        let mut loop_id = 0;
        for op in &self.ops {
            match op {
                Op::Loop { .. } => loop_id += 1,
                Op::EndLoop => loop_id -= 1,
                _ => {}
            }
        }
        while loop_id > 0 {
            self.ops.push(Op::EndLoop);
            loop_id -= 1;
        }
    }

    pub fn loop_invariant_code_motion(&mut self, loop_id: OpId) {
        //self.debug();
        // TODO Reorder commutative
        // Iterate:
        //   find a chain of commutative ops like add/sub
        //   reoder by moving loop index last

        // LICM
        // Extract loop body and tail
        let end_loop_id = self.get_end_loop_id(loop_id);

        //println!("end_loop_id={end_loop_id}");
        let tail = self.ops.split_off(end_loop_id);
        let mut body = self.ops.split_off(loop_id);
        // for each op in loop body - if all parameters are invariant, mark as invariant, otherwise do nothing
        let mut i = 0;
        while i < body.len() {
            // If op is invariant
            if !matches!(
                body[i],
                Op::Loop { .. } | Op::Store { .. } | Op::Load { .. } | Op::EndLoop | Op::Define { .. }
            ) && body[i].parameters().all(|x| x < loop_id)
            {
                //println!("invariant op: {:?}", body[i]);
                // Move op out of the loop
                let op = body.remove(i);
                self.ops.push(op);

                // Increment and remap all ops in the body accordingly
                remap_or_increment(
                    &mut body,
                    self.ops.len() + i - 1,
                    self.ops.len() - 1,
                    1,
                    self.ops.len() - 1..self.ops.len() + i - 1,
                );
            } else {
                i += 1;
            }
        }

        // Add body back to ops
        self.ops.extend(body);
        self.ops.extend(tail);
        //println!();
        //self.debug();
        //panic!();
    }

    pub fn loop_unroll(&mut self, loop_id: OpId) {
        let Op::Loop { dim, .. } = self.ops[loop_id] else { unreachable!() };

        // Get tail and body
        let end_loop_id = self.get_end_loop_id(loop_id);

        // Don't unroll super huge loops with many iterations
        if (end_loop_id - loop_id) * dim > 1024 {
            return;
        }

        let mut tail = self.ops.split_off(end_loop_id + 1);
        self.ops.pop();
        let loop_body = self.ops.split_off(loop_id + 1);
        self.ops.pop();

        // Repeat loop body
        let mut n = 1;
        for idx in 0..dim {
            let mut body = loop_body.clone();
            // First index as constant
            let idx_const = self.ops.len();
            self.ops.push(Op::Const(Constant::idx(idx as u64)));

            // Increment body
            increment(&mut body, n as isize - 1, loop_id + 1..end_loop_id);
            n += body.len() + 1;

            // Remap body to use constant
            let mut map = Map::default();
            map.insert(loop_id, idx_const);
            remap(&mut body, &map);
            self.ops.extend(body);
        }

        // Add tail, increment ops
        let d = ((end_loop_id - loop_id) * (dim - 1)) as isize - 1;
        increment(&mut tail, d, end_loop_id..);
        self.ops.extend(tail);
    }

    pub fn loop_unroll_and_jam(&mut self, loop_id: OpId) {
        // Assumes there is outer loop at loop_id and at least one inner loop in this outer loop
        let Op::Loop { dim: loop_dim, scope } = self.ops[loop_id] else { unreachable!() };
        debug_assert_eq!(scope, Scope::Register);

        // Find first inner loop and jam it in there
        let inner_loop_id = self.ops[loop_id..].iter().position(|op| matches!(op, Op::Loop { .. })).unwrap();
        let end_inner_loop_id = self.get_end_loop_id(inner_loop_id);
        // Get the body of the inner loop
        let mut tail = self.ops.split_off(end_inner_loop_id);
        let mut body = self.ops.split_off(inner_loop_id + 1);
        // Jam the outer loop in the inner loop
        self.ops.push(Op::Loop { dim: loop_dim, scope: Scope::Register });
        increment(&mut body, 1, inner_loop_id + 1..);
        self.ops.extend(body);
        self.ops.push(Op::EndLoop);
        increment(&mut tail, 2, end_inner_loop_id..);
        self.ops.extend(tail);

        // Unroll everything else in the outer loop, except for the tail (inner loop + ops after it, till the end of the kernel)

        // What we need to do is to repeat all ops loop_dim times, but define needs to be
        // instead multiplied (it's len) by loop_dim.
        // Existing index that is used to index define both for loads and stores needs to be incremented:
        //  - outside of the inner loop it needs to be incremented by constant
        //  - in the inner loop it needs to be incremented by the jammed loop index

        todo!()
    }

    // Loop tiling/vectorization. Tiles all loads.
    /*pub fn loop_tile(&mut self, loop_id: OpId) {
        todo!()
    }

    pub fn loop_tile_and_jam(&mut self, loop_id: OpId) {
        todo!()
    }*/

    pub fn reshape_reduce(&mut self, reduce_id: OpId, new_dims: &[Dim]) {
        let Op::Reduce { x, ref mut dims, .. } = self.ops[reduce_id] else { return };
        let n_old_dims = dims.len();
        *dims = new_dims.into();

        let mut visited = Set::default();
        self.recursively_apply_reshape(x, n_old_dims, new_dims, &mut visited, 0);
    }

    fn recursively_apply_reshape(
        &mut self,
        op_id: OpId,
        n_old_dims: usize,
        new_dims: &[Dim],
        visited: &mut Set<OpId>,
        skip_last: usize,
    ) {
        if !visited.insert(op_id) {
            return;
        }
        match self.ops[op_id] {
            Op::LoadView { ref mut view, .. } | Op::ConstView { ref mut view, .. } => {
                let rank = view.rank();
                view.reshape(rank - skip_last - n_old_dims..rank - skip_last, new_dims);
            }
            Op::Reduce { x, ref dims, .. } => {
                let skip_last = skip_last + dims.len();
                self.recursively_apply_reshape(x, n_old_dims, new_dims, visited, skip_last);
            }
            Op::Cast { x, .. } | Op::Unary { x, .. } => {
                self.recursively_apply_reshape(x, n_old_dims, new_dims, visited, skip_last);
            }
            Op::Binary { x, y, .. } => {
                self.recursively_apply_reshape(x, n_old_dims, new_dims, visited, skip_last);
                self.recursively_apply_reshape(y, n_old_dims, new_dims, visited, skip_last);
            }
            _ => {}
        }
    }

    fn get_end_loop_id(&self, loop_id: OpId) -> OpId {
        let mut end_loop_id = loop_id;
        let mut n_loops = 1;
        while end_loop_id < self.ops.len() {
            end_loop_id += 1;
            match self.ops[end_loop_id] {
                Op::Loop { .. } => n_loops += 1,
                Op::EndLoop => n_loops -= 1,
                _ => {}
            }
            if n_loops == 0 {
                break;
            }
        }
        end_loop_id
    }

    pub fn dead_code_elimination(&mut self) {
        let mut params = Vec::new();
        for op_id in 0..self.ops.len() {
            // TODO remove Op::Load from here, it has no reason to be here other than compatibility with predefined loads
            if matches!(
                self.ops[op_id],
                Op::Store { .. } | Op::Loop { .. } | Op::EndLoop | Op::Load { .. }
            ) {
                params.push(op_id);
            }
        }
        let mut needed = Set::with_capacity_and_hasher(self.ops.len(), BuildHasherDefault::new());
        while let Some(param) = params.pop() {
            if needed.insert(param) {
                match self.ops[param] {
                    Op::Const(..) | Op::Define { .. } | Op::Loop { .. } | Op::EndLoop | Op::Null => {}
                    Op::ConstView { .. } | Op::LoadView { .. } | Op::StoreView { .. } => unreachable!(),
                    Op::Load { src, index } => {
                        params.push(src);
                        params.push(index);
                    }
                    Op::Store { dst, x: src, index } => {
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

                    if !remaps.contains_key(&op_id)
                        && !matches!(
                            op,
                            Op::Define { .. } | Op::Loop { .. } | Op::EndLoop | Op::Load { .. } | Op::Store { .. }
                        )
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
        increment(
            &mut self.ops[n_defines + remaps.len()..],
            remaps.len() as isize,
            n_defines..,
        );
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
                | Op::Define { .. }
                | Op::Null => {}
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
                        BOp::Sub => {
                            if cx.is_zero() {
                                self.ops[op_id] = Op::Unary { x: y, uop: UOp::Neg };
                            }
                        }
                        BOp::Mul => {
                            if cx.is_zero() {
                                remaps.insert(op_id, x);
                            } else if cx.is_one() {
                                remaps.insert(op_id, y);
                            }
                        }
                        BOp::Div => {
                            if cx.is_zero() {
                                remaps.insert(op_id, x);
                            } else if cx.is_one() {
                                self.ops[op_id] = Op::Unary { x: y, uop: UOp::Reciprocal };
                            }
                        }
                        BOp::Pow => {
                            if cx.is_zero() {
                                remaps.insert(op_id, x);
                            } else if cx.is_one() {
                                remaps.insert(op_id, x);
                            } //else if cx.is_two() && cx.dtype().is_shiftable() {
                            //self.ops.insert(op_id, Op::Constant(cx.dtype().one())); // but we can't insert with remaps
                            //self.ops[op_id] = Op::Binary { x, y, bop: BOp::BitShiftLeft };
                            //}
                        }
                        BOp::Mod => todo!(),
                        BOp::Cmplt => todo!(),
                        BOp::Cmpgt => {}
                        BOp::Maximum => todo!(),
                        BOp::Or => todo!(),
                        BOp::And => todo!(),
                        BOp::BitXor => todo!(),
                        BOp::BitOr => todo!(),
                        BOp::BitAnd => todo!(),
                        BOp::BitShiftLeft => todo!(),
                        BOp::BitShiftRight => todo!(),
                        BOp::NotEq => todo!(),
                        BOp::Eq => todo!(),
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
                        BOp::Pow => {
                            if cy.is_zero() {
                                self.ops[op_id] = Op::Const(cy.dtype().one_constant());
                            } else if cy.is_one() {
                                remaps.insert(op_id, x);
                            } else if cy.is_two() {
                                self.ops[op_id] = Op::Binary { x, y: x, bop: BOp::Mul };
                            }
                        }
                        BOp::Mod => {
                            if cy.is_zero() {
                                panic!("Modulo by zero constant.");
                            } else if cy.is_one() {
                                self.ops[op_id] = Op::Const(cy.dtype().zero_constant());
                            }
                        }
                        BOp::Cmplt | BOp::Cmpgt | BOp::NotEq | BOp::And | BOp::Eq => {}
                        BOp::Maximum => todo!(),
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
}

fn get_axes(ops: &[Op]) -> Vec<UAxis> {
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

pub fn increment(ops: &mut [Op], d: isize, range: impl RangeBounds<usize>) {
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
            *x = (*x as isize + d) as usize;
        }
    };
    for op in ops {
        match op {
            Op::ConstView { .. }
            | Op::Const { .. }
            | Op::LoadView { .. }
            | Op::Loop { .. }
            | Op::Define { .. }
            | Op::EndLoop
            | Op::Null => {}
            Op::StoreView { src, .. } => {
                h(src);
            }
            Op::Load { src, index } => {
                h(src);
                h(index);
            }
            Op::Store { dst, x: src, index } => {
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
            | Op::Define { .. }
            | Op::Null => {}
            Op::StoreView { src, .. } => {
                h(src);
            }
            Op::Load { src, index, .. } => {
                h(src);
                h(index);
            }
            Op::Store { dst, x: src, index } => {
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

fn remap_or_increment(ops: &mut [Op], from: OpId, to: OpId, d: usize, range: impl RangeBounds<usize>) {
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

    let hi = |x: &mut usize| {
        //println!("{x}, {range:?}, contains={}", range.contains(x));
        if range.contains(x) {
            *x += d;
        }
    };

    let hr = |x: &mut usize| -> bool {
        if *x == from {
            *x = to;
            true
        } else {
            false
        }
    };

    for op in ops {
        match op {
            Op::ConstView { .. }
            | Op::LoadView { .. }
            | Op::Const(_)
            | Op::Loop { .. }
            | Op::EndLoop
            | Op::Define { .. }
            | Op::Null => {}
            Op::StoreView { src, .. } => {
                if !hr(src) {
                    hi(src);
                }
            }
            Op::Load { src, index, .. } => {
                if !hr(src) {
                    hi(src);
                }
                if !hr(index) {
                    hi(index);
                }
            }
            Op::Store { dst, x: src, index } => {
                if !hr(dst) {
                    hi(dst);
                }
                if !hr(src) {
                    hi(src);
                }
                if !hr(index) {
                    hi(index);
                }
            }
            Op::Cast { x, .. } | Op::Unary { x, .. } | Op::Reduce { x, .. } => {
                if !hr(x) {
                    hi(x)
                }
            }
            Op::Binary { x, y, .. } => {
                if !hr(x) {
                    hi(x);
                }
                if !hr(y) {
                    hi(y);
                }
            }
        }
    }
}
