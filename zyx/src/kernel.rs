use crate::{
    DType, Map,
    backend::{Device, DeviceInfo, ProgramId},
    dtype::Constant,
    graph::{BOp, ROp, UOp},
    shape::{Axis, Dim},
    view::View,
};
use std::hash::BuildHasherDefault;

pub type OpId = usize;

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Kernel {
    pub ops: Vec<Op>,
    pub n_outputs: u32,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Op {
    //Sink { stores: Vec<Op> }, // A way to put multiple stores in one kernel
    ConstView { value: Constant, view: View },
    LoadView { dtype: DType, view: View },
    Store { x: OpId },
    Cast { x: OpId, dtype: DType },
    Unary { x: OpId, uop: UOp },
    Binary { x: OpId, y: OpId, bop: BOp },
    Reduce { x: OpId, rop: ROp, num_axes: usize },
}

#[derive(Debug)]
pub struct Cache {
    pub device_infos: Map<DeviceInfo, u32>,
    pub kernels: Map<Kernel, u32>,
    // Finished optimizations of kernels for given devices
    // kernel id, device info id => optimization
    pub optimizations: Map<(u32, u32), Optimization>,
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

/*impl Optimization {
    fn default(kernel: &Op, dev_info: &DeviceInfo) -> Self {
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

        // TODO
        /*let shape = kernel.shape();
        let n = shape.iter().product();*/
        let n = 3;
        let mut global_work_size = get_equal_factors(n);

        let mut d = dev_info.max_local_threads;
        while n % d != 0 {
            d -= 1;
        }
        let local_work_size = get_equal_factors(n / d);
        global_work_size[0] /= local_work_size[0];
        global_work_size[1] /= local_work_size[1];
        global_work_size[2] /= local_work_size[2];

        // Concatenate global and local work sizes to get the final 6D shape
        let mut shape = vec![];
        shape.extend(global_work_size);
        shape.extend(local_work_size);

        Self::Basic { shape }
    }
}*/

#[allow(clippy::similar_names)]
fn get_perf(flop: u128, bytes_read: u128, bytes_written: u128, nanos: u128) -> String {
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
        for (id, op) in self.ops.iter().enumerate() {
            match op {
                Op::ConstView { value, view } => println!("{id:>3} CONST {value} {view}"),
                Op::LoadView { dtype, view } => println!("{id:>3} LOAD {dtype} {view}"),
                Op::Store { x } => println!("{id:>3} STORE {x}"),
                Op::Cast { x, dtype } => println!("{id:>3} CAST {x} {dtype:?}"),
                Op::Unary { x, uop } => println!("{id:>3} UNARY {uop:?} {x}"),
                Op::Binary { x, y, bop } => println!("{id:>3} BINARY {bop:?} {x} {y}"),
                Op::Reduce { x, rop, num_axes } => println!(
                    "{id:>3} {} {x} num_axes={num_axes:?}",
                    match rop {
                        ROp::Sum => "SUM",
                        ROp::Max => "MAX",
                    }
                ),
            }
        }
    }
}
