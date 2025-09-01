use nanoserde::{DeBin, SerBin};

use crate::{
    DType, Map, Set,
    backend::{Device, DeviceInfo, ProgramId},
    dtype::Constant,
    graph::{BOp, ROp, UOp},
    optimizer::Optimizer,
    shape::Dim,
    view::View,
};
use std::{
    fmt::Display,
    hash::BuildHasherDefault,
    ops::{Range, RangeBounds},
};

pub type OpId = usize;

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, SerBin, DeBin)]
pub struct Kernel {
    pub ops: Vec<Op>,
    //pub n_outputs: u32, // TODO remove this from here
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

    // ops that only exist after unfolding views and reduces
    Const(Constant),
    Define { dtype: DType, scope: Scope, ro: bool, len: Dim }, // len is 0 for globals
    Load { src: OpId, index: OpId },
    Loop { dim: Dim, scope: Scope },
    EndLoop,

    // ops that exist in both
    Store { dst: OpId, x: OpId, index: OpId },
    Cast { x: OpId, dtype: DType },
    Unary { x: OpId, uop: UOp },
    Binary { x: OpId, y: OpId, bop: BOp },
}

// This is SSA representation. All ops return immutable variables.
// The Define op can define mutable variables.
// Variables defined by define op can only be accessed with Load on Store ops,
// using their src and dst fields.
/*pub enum Op {
    Const(Constant),
    Define { dtype: DType, scope: Scope, ro: bool, len: Dim }, // len is 0 for globals
    Load { src: OpId, index: OpId },
    Store { dst: OpId, x: OpId, index: OpId },

    Loop { dim: Dim, scope: Scope },
    EndLoop,

    Cast { x: OpId, dtype: DType },
    Unary { x: OpId, uop: UOp },
    Binary { x: OpId, y: OpId, bop: BOp },
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
        let mut device_infos = Map::with_capacity_and_hasher(len, BuildHasherDefault::new());
        for _ in 0..len {
            let key = DeviceInfo::de_bin(offset, bytes)?;
            let value = u32::de_bin(offset, bytes)?;
            device_infos.insert(key, value);
        }

        let len = usize::de_bin(offset, bytes)?;
        let mut kernels = Map::with_capacity_and_hasher(len, BuildHasherDefault::new());
        for _ in 0..len {
            let key = Kernel::de_bin(offset, bytes)?;
            let value = u32::de_bin(offset, bytes)?;
            kernels.insert(key, value);
        }

        let len = usize::de_bin(offset, bytes)?;
        let mut optimizations = Map::with_capacity_and_hasher(len, BuildHasherDefault::new());
        for _ in 0..len {
            let k1 = u32::de_bin(offset, bytes)?;
            let k2 = u32::de_bin(offset, bytes)?;
            let key = (k1, k2);
            let value = Optimizer::de_bin(offset, bytes).unwrap();
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
pub fn get_perf(flop: u128, bytes_read: u128, bytes_written: u128, nanos: u128) -> String {
    if nanos == u128::MAX {
        return format!("INF time taken");
    }
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
        const RED: &str = "\x1b[31m";
        const GREEN: &str = "\x1b[32m";
        const YELLOW: &str = "\x1b[33m";
        const BLUE: &str = "\x1b[34m";
        const MAGENTA: &str = "\x1b[35m";
        const CYAN: &str = "\x1b[36m";
        const RESET: &str = "\x1b[0m";
        let mut indent = String::from(" ");
        for (i, op) in self.ops.iter().enumerate() {
            match op {
                Op::ConstView { value, view } => println!("{i:>3}{indent}{CYAN}CONST VIEW{RESET} {value} {view}"),
                Op::LoadView { dtype, view } => println!("{i:>3}{indent}{CYAN}LOAD VIEW{RESET} {dtype} {view}"),
                Op::StoreView { src, dtype } => println!("{i:>3}{indent}{CYAN}STORE VIEW{RESET} {src} {dtype}"),
                Op::Reduce { x, rop, dims } => {
                    println!("{i:>3}{indent}{CYAN}REDUCE{RESET} {} {x}, dims={dims:?}", match rop {
                        ROp::Sum => "SUM",
                        ROp::Max => "MAX",
                    });
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
            }
        }
    }

    pub fn flop_mem_rw(&self) -> (u128, u128, u128) {
        let stores: Vec<OpId> =
            self.ops.iter().enumerate().filter(|(_, op)| matches!(op, Op::StoreView { .. })).map(|(i, _)| i).collect();

        let mut flop = 0;
        let mut mr = 0;
        let mut mw = 0;
        let mut visited = Map::with_hasher(BuildHasherDefault::new());

        // flop, memory read, memory write, number of elements being processed
        fn recursive(x: OpId, ops: &[Op], visited: &mut Map<OpId, u128>) -> (u128, u128, u128) {
            if visited.contains_key(&x) {
                return (0, 0, 0);
            }
            let (f, r, w, n) = match &ops[x] {
                Op::ConstView { view, .. } => (0, 0, 0, view.numel() as u128),
                Op::LoadView { view, .. } => (0, view.original_numel() as u128, 0, view.numel() as u128),
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
                    let rd = dims.iter().product::<usize>() as u128;
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

    pub fn is_reduce(&self) -> bool { self.ops.iter().any(|x| matches!(x, Op::Reduce { .. })) }

    pub fn contains_stores(&self) -> bool { self.ops.iter().any(|x| matches!(x, Op::StoreView { .. })) }

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
                }
            }

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
                    ROp::Max => BOp::Maximum,
                },
            });
            self.ops.push(Op::Store { dst: acc, x: y + 1, index: c_0 });

            // Insert endloops
            for _ in 0..(n_dims - 1) {
                self.ops.push(Op::EndLoop);
            }

            // Update tail by adding load acc before all ops referencing op_id
            // op_id -> acc
            //let mut n_inserted_loads = 0;

            // number of new inserted ops before the tail section
            let n = 7 + n_dims * 2 - 1;

            let mut inserted_loads = Vec::new();
            let mut i = 0;
            while i < tail.len() {
                match &mut tail[i] {
                    Op::ConstView { .. } | Op::LoadView { .. } | Op::Define { .. } | Op::Loop { .. } | Op::EndLoop => {}
                    Op::StoreView { src, .. } => {
                        if *src == op_id {
                            *src = self.ops.len() + i;
                            tail.insert(i, Op::Load { src: acc, index: c_0 });
                            i += 1;
                            //n_inserted_loads += 1;
                            inserted_loads.push(i);
                        } else if *src > op_id {
                            //*src += n_inserted_loads + 8;
                            //println!("src={src}, {inserted_loads:?}");
                            *src += inserted_loads.iter().filter(|&&v| v + self.ops.len() - 1 < *src + n).count() + n;
                        }
                    }
                    Op::Reduce { x, .. } | Op::Cast { x, .. } | Op::Unary { x, .. } => {
                        if *x == op_id {
                            *x = self.ops.len() + i;
                            tail.insert(i, Op::Load { src: acc, index: c_0 });
                            i += 1;
                            inserted_loads.push(i);
                        } else if *x > op_id {
                            *x += inserted_loads.iter().filter(|&&v| v + self.ops.len() - 1 < *x + n).count() + n;
                        }
                    }
                    Op::Const(_) => {}
                    Op::Load { src, index } => {
                        debug_assert_ne!(*index, op_id);
                        if *index > op_id {
                            *index += inserted_loads.iter().filter(|&&v| v + self.ops.len() - 1 < *index + n).count() + n;
                        }
                        if *src == op_id {
                            *src = self.ops.len() + i;
                            tail.insert(i, Op::Load { src: acc, index: c_0 });
                            i += 1;
                            inserted_loads.push(i);
                        } else if *src > op_id {
                            *src += inserted_loads.iter().filter(|&&v| v + self.ops.len() - 1 < *src + n).count() + n;
                        }
                    }
                    Op::Store { dst, x, index } => {
                        debug_assert_ne!(*index, op_id);
                        if *index > op_id {
                            *index += inserted_loads.iter().filter(|&&v| v + self.ops.len() - 1 < *index + n).count() + n;
                        }
                        if *dst > op_id {
                            *dst += inserted_loads.iter().filter(|&&v| v + self.ops.len() - 1 < *dst + n).count() + n;
                        }
                        if *x == op_id {
                            *x = self.ops.len() + i;
                            tail.insert(i, Op::Load { src: acc, index: c_0 });
                            i += 1;
                            inserted_loads.push(i);
                        } else if *x > op_id {
                            *x += inserted_loads.iter().filter(|&&v| v + self.ops.len() - 1 < *x + n).count() + n;
                        }
                    }
                    Op::Binary { x, y, .. } => {
                        let tailx = if *x == op_id {
                            *x = self.ops.len() + i;
                            true
                        } else {
                            false
                        };
                        let taily = if *y == op_id {
                            *y = self.ops.len() + i;
                            true
                        } else {
                            false
                        };
                        if tailx || taily {
                            inserted_loads.push(i);
                        }
                        if *x > op_id && !tailx {
                            *x += inserted_loads.iter().filter(|&&v| v + self.ops.len() - 1 < *x + n).count() + n;
                        }
                        if *y > op_id && !taily {
                            *y += inserted_loads.iter().filter(|&&v| v + self.ops.len() - 1 < *y + n).count() + n;
                        }
                        if tailx || taily {
                            tail.insert(i, Op::Load { src: acc, index: c_0 });
                            i += 1;
                        }
                    }
                }
                i += 1;
            }

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
                    let constant_zero = new_op(ops, Op::Const(Constant::U32(0)));
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
                                ost *= dim.d as u32;
                                let x = if t_ost == 1 {
                                    old_offset
                                } else {
                                    let ost_c = new_op(ops, Op::Const(Constant::U32(t_ost)));
                                    new_op(ops, Op::Binary { x: old_offset, y: ost_c, bop: BOp::Div })
                                };
                                if dim.d == 1 {
                                    constant_zero
                                } else {
                                    let dimd_c = new_op(ops, Op::Const(Constant::U32(dim.d as u32)));
                                    new_op(ops, Op::Binary { x, y: dimd_c, bop: BOp::Mod })
                                }
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

                            if dim.st != 0 {
                                let stride = new_op(ops, Op::Const(Constant::U32(dim.st as u32)));
                                let x = new_op(ops, Op::Binary { x: t, y: stride, bop: BOp::Mul });
                                offset = new_op(ops, Op::Binary { x, y: offset, bop: BOp::Add });
                            }

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
                    increment(&mut self.ops[n..], n - op_id - 1, op_id..);
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
                    let constant_zero = new_op(ops, Op::Const(Constant::U32(0)));
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
                                ost *= dim.d as u32;
                                let x = if t_ost == 1 {
                                    old_offset
                                } else {
                                    let ost_c = new_op(ops, Op::Const(Constant::U32(t_ost)));
                                    new_op(ops, Op::Binary { x: old_offset, y: ost_c, bop: BOp::Div })
                                };
                                if dim.d == 1 {
                                    constant_zero
                                } else {
                                    let dimd_c = new_op(ops, Op::Const(Constant::U32(dim.d as u32)));
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
                                let lp = new_op(ops, Op::Const(Constant::U32(dim.lp.unsigned_abs() as u32)));
                                if dim.lp > 0 {
                                    new_op(ops, Op::Binary { x: a, y: lp, bop: BOp::Sub })
                                } else {
                                    new_op(ops, Op::Binary { x: a, y: lp, bop: BOp::Add })
                                }
                            } else {
                                a
                            };

                            if dim.st != 0 {
                                let stride = new_op(ops, Op::Const(Constant::U32(dim.st as u32)));
                                let x = new_op(ops, Op::Binary { x: t, y: stride, bop: BOp::Mul });
                                offset = new_op(ops, Op::Binary { x, y: offset, bop: BOp::Add });
                            }

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

                    _ = new_op(&mut self.ops, Op::Store { dst: store_id, x: src, index });

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
                        BOp::Pow => {}
                        BOp::Mod => todo!(),
                        BOp::Cmplt => todo!(),
                        BOp::Cmpgt => todo!(),
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
                        BOp::Pow => todo!(),
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

pub fn increment(ops: &mut [Op], d: usize, range: impl RangeBounds<usize>) {
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
            | Op::Define { .. } => {}
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
