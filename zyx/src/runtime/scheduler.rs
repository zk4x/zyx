use crate::runtime::v::{self, VOp};
use crate::runtime::{ir, TensorId};
use crate::scalar::Scalar;
use std::collections::{BTreeMap, BTreeSet};
use super::graph::Graph;
use super::executor::Executor;

pub(super) struct Backend<P: Executor> {
    compiler: P,
    buffers: BTreeMap<TensorId, P::Buffer>,
    compiled_graphs: BTreeMap<Graph, CompiledGraph<P::Program>>,
}

impl<C: Executor> Drop for Backend<C> {
    fn drop(&mut self) {
        while let Some((_, buffer)) = self.buffers.pop_last() {
            self.compiler.deallocate_memory(buffer).unwrap();
        }
        while let Some((_, graph)) = self.compiled_graphs.pop_last() {
            for (_, program) in graph.programs {
                self.compiler.release_program(program).unwrap();
            }
        }
    }
}

/// Compiled graph
pub(super) struct CompiledGraph<Program> {
    // Ordered programs and arguments to them
    programs: Vec<(Vec<TensorId>, Program)>,
    flop: u128,
    bytes_read: u128,
    bytes_written: u128,
}

impl<P: Executor> Backend<P> {
    pub(super) fn initialize() -> Result<Self, P::Error> {
        let mut compiler = P::initialize()?;
        Ok(Self {
            hwinfo: compiler.hardware_information()?,
            compiler,
            buffers: BTreeMap::new(),
            compiled_graphs: BTreeMap::new(),
        })
    }

    pub(super) fn is_realized(&self, x: TensorId) -> bool {
        self.buffers.contains_key(&x)
    }

    // TODO no point in this. Just create arena allocator and use offset to access different buffers
    // But we have to be able to work with multiple memory pools, because max buffer size is usually
    // about quarter of gpu memory.
    // With multiple memory pools we can also make it work for multiple devices.
    // But if we can't make sure that the next allocation is next to previous one,
    // then we need to allocate in large chunks, which may not be optimal.
    // More research needed before making changes.
    pub(super) fn store<T: Scalar>(
        &mut self,
        x: TensorId,
        data: Vec<T>,
    ) -> Result<(), P::Error> {
        //std::println!("Memory alignment: {}", self.hwinfo.page_size);
        let mut buffer = self.compiler.allocate_memory(data.len() * T::byte_size())?;
        self.compiler.store_memory(&mut buffer, data)?;
        self.buffers.insert(x, buffer);
        return Ok(());
    }

    // Load values at x, if x is not evaluated, it will return error
    pub(super) fn load<T: Scalar>(
        &mut self,
        x: TensorId,
        length: usize,
    ) -> Result<Vec<T>, P::Error> {
        //println!("Attempting to load buffer with id {x}");
        if let Some(buffer) = self.buffers.get(&x) {
            return self.compiler.load_memory(buffer, length);
        } else {
            panic!("Buffer with given id does not exist. Internal bug.");
        }
    }

    pub(super) fn remove(&mut self, x: TensorId) -> Result<(), P::Error> {
        if let Some(buffer) = self.buffers.remove(&x) {
            return self.compiler.deallocate_memory(buffer);
        }
        return Ok(());
    }

    /// Compiles and caches graph
    pub(super) fn compile_graph(
        &mut self,
        org_graph: &Graph,
        to_eval: BTreeSet<TensorId>,
    ) -> Result<(), P::Error> {
        //println!("{:#?}", self.hwinfo);
        //#[cfg(feature = "debug1")]
        //println!("Evaluating {to_eval:?}");
        if self.compiled_graphs.contains_key(&org_graph) {
            return Ok(());
        }
        let mut graph = org_graph.clone();
        let (order, flop, bytes_read, bytes_written) = graph.execution_order(&to_eval);
        let mut kernels = v::generate_kernels(&graph, &order, &to_eval);

        let mut programs = Vec::new();
        #[cfg(feature = "debug1")]
        println!("Compiling kernels");
        // Rewrite tiled representation to IR representation and compile it for HW device
        // TODO kernels can be compiled in parallel
        for kernel in &mut kernels {
            // Get the number of loops before any other operation
            let num_loops = kernel
                .ops
                .iter()
                .position(|kernel| !matches!(kernel, VOp::Loop { .. }))
                .unwrap();

            // If this is full reduce kernel
            if num_loops == 0 {
                // this should never happen, because we should use local and register memory
                // and always spread work across multiple threads
                todo!("Full reduce")
            }

            // If there is more loops than 3, pick first three loops as global loops,
            // rest is register loops.
            // So nothing needs to be done.
            // If there is less than three loops, add loops with dimension 1
            if num_loops < 3 {
                let dims: Vec<usize> = core::iter::repeat(1)
                    .take(3 - num_loops)
                    .chain([kernel.shape[0]])
                    .collect();
                kernel.split_axis(0, &dims);
            }

            // Split first three loops into global and local loops.
            let mut gws = [1; 3];
            for op in &kernel.ops {
                if let VOp::Loop { axis, dimension } = op {
                    if *axis > 2 {
                        break;
                    }
                    gws[*axis] = *dimension;
                }
            }

            // Reorder global loops from smallest to largest
            //gws.sort();
            // Get sort indices and permute both kernel and gws
            // by those indices

            // Determine the best possible work size
            let lws = best_local_work_size(gws, self.hwinfo.max_work_group_size);
            gws[0] /= lws[0];
            gws[1] /= lws[1];
            gws[2] /= lws[2];

            kernel.split_axis(0, &[gws[0], lws[0]]);
            kernel.split_axis(2, &[gws[1], lws[1]]);
            kernel.split_axis(4, &[gws[2], lws[2]]);

            //println!("Kernel in virtual ops");
            #[cfg(feature = "debug1")]
            {
                for op in &kernel.ops {
                    println!("{op:?}");
                }
            }

            let ir_kernel = ir::vops_to_ir(&kernel.ops, &graph, &self.hwinfo);
            let str_kernel = ir::to_str_kernel(&ir_kernel);
            //println!("\n\n{str_kernel}\n\n");

            /*let args: Vec<TensorId> = ir_kernel.args.keys().copied().collect();
            programs.push((args.clone(), self.compiler.compile_program(&ir_kernel)?));

            // Allocate memory for intermediate args and results
            for arg in args.iter().copied() {
                if !self.buffers.contains_key(&arg) {
                    self.buffers.insert(
                        arg,
                        self.compiler.allocate_memory(
                            graph.shape(arg).iter().product::<usize>()
                                * graph.dtype(arg).byte_size(),
                        )?,
                    );
                }
            }*/
        }

        self.compiled_graphs.insert(
            org_graph.clone(),
            CompiledGraph {
                programs,
                flop: flop as u128,
                bytes_read: bytes_read as u128,
                bytes_written: bytes_written as u128,
            },
        );

        return Ok(());
    }

    pub(super) fn launch_graph(&mut self, graph: &Graph) -> Result<(), P::Error> {
        let graph = self.compiled_graphs.get(graph).unwrap();

        #[cfg(feature = "debug1")]
        let begin = std::time::Instant::now();

        for (args, program) in &graph.programs {
            let mut buffers = Vec::with_capacity(args.len());
            for arg in args {
                let buffer = self.buffers.remove(arg).unwrap();
                buffers.push(buffer);
            }
            self.compiler.launch_program(&program, &mut buffers)?;
            for arg in args.iter().copied().rev() {
                self.buffers.insert(arg, buffers.pop().unwrap());
            }
        }

        #[cfg(feature = "debug1")]
        {
            let duration = begin.elapsed();
            let nanos = duration.as_nanos();

            fn value_unit(x: u128) -> (u128, &'static str) {
                match x {
                    0..1000 => (x, ""),
                    1000..1000000 => (x / 1000, "k"),
                    1000_000..1000000000 => (x / 1000_000, "M"),
                    1000_000_000..1000_000_000_000 => (x / 1000_000_000, "G"),
                    1000_000_000_000..1000_000_000_000_000 => (x / 1000_000_000_000, "T"),
                    1000_000_000_000_000..1000_000_000_000_000_000 => {
                        (x / 1000_000_000_000_000, "P")
                    }
                    1000_000_000_000_000_000.. => (x / 1000_000_000_000_000_000, "E"),
                }
            }

            let (f, f_u) = value_unit(graph.flop);
            let (br, br_u) = value_unit(graph.bytes_read);
            let (bw, bw_u) = value_unit(graph.bytes_written);
            let (t_d, t_u) = match nanos {
                0..1000 => (1, "ns"),
                1000..1000_000 => (1000, "μs"),
                1000_000..1000_000_000 => (1000_000, "ms"),
                1000_000_000..1000_000_000_000 => (1000_000_000, "s"),
                1000_000_000_000.. => (60_000_000_000, "min"),
            };

            let (fs, f_us) = value_unit(graph.flop * 1000_000_000 / nanos);
            let (brs, br_us) = value_unit(graph.bytes_read * 1000_000_000 / nanos);
            let (bws, bw_us) = value_unit(graph.bytes_written * 1000_000_000 / nanos);

            println!(
            "Graph {f} {f_u}FLOP, {br} {br_u}B read, {bw} {bw_u}B write, took {} {t_u} ~ {fs} {f_us}FLOP/s, {brs} {br_us}B/s read, {bws} {bw_us}B/s write.",
            nanos/t_d,
            );
        }

        return Ok(());
    }
}

// Takes global work size (gws) and maximum work group size (mwgs)
fn best_local_work_size(mut gws: [usize; 3], mwgs: usize) -> [usize; 3] {
    let mut lws = [1; 3];
    //println!("Max {mwgs:?}");
    let rwgs = (mwgs as i64).sqrt() as usize;
    //println!("Root {rwgs:?}");

    let mut total = 1;
    let mut n = 1;
    while gws[1] % (n * 2) == 0 && n * 2 <= rwgs {
        n *= 2;
    }
    gws[1] /= n;
    lws[1] *= n;
    total *= n;
    // put the rest into third dimension
    let mut n = 1;
    while gws[2] % (n * 2) == 0 && n * 2 * total <= mwgs {
        n *= 2;
    }
    gws[2] /= n;
    lws[2] *= n;
    total *= n;
    // if third dimension was too small, put the rest into second dimension
    let mut n = 1;
    while gws[1] % (n * 2) == 0 && n * 2 * total <= mwgs {
        n *= 2;
    }
    gws[1] /= n;
    lws[1] *= n;

    return lws;
}

//fn local_memory_tiles(ops: &mut Vec<VOp>) {}

//fn more_work_per_thread(ops: &mut Vec<VOp>, gws: &mut [usize; 3]) {}
