use crate::runtime::TensorId;
use crate::scalar::Scalar;
use alloc::collections::{BTreeMap, BTreeSet};
use alloc::vec::Vec;
use core::fmt::{Display, Formatter};
use v::VOp;

#[cfg(feature = "debug1")]
use std::println;

use super::graph::Graph;
use super::node::{BOp, UOp};

mod ir;
use ir::{IRArg, IRKernel, IROp};
mod v;

#[cfg(feature = "cuda")]
pub(super) mod cuda;

#[cfg(feature = "hsa")]
pub(super) mod hsa;

#[cfg(feature = "opencl")]
pub(super) mod opencl;

#[cfg(feature = "wgsl")]
pub(super) mod wgsl;

#[allow(dead_code)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) enum Scope {
    Global,
    Local,
    Register,
}

impl Display for Scope {
    fn fmt(&self, f: &mut Formatter<'_>) -> core::fmt::Result {
        f.write_str(match self {
            Self::Global => "g",
            Self::Local => "l",
            Self::Register => "r",
        })
    }
}

pub(super) trait Compiler: Sized {
    type Buffer;
    type Program;
    type Error: core::fmt::Debug;
    fn initialize() -> Result<Self, Self::Error>;
    fn hardware_information(&mut self) -> Result<HWInfo, Self::Error>;
    fn allocate_memory(&mut self, byte_size: usize) -> Result<Self::Buffer, Self::Error>;
    fn store_memory<T: Scalar>(
        &mut self,
        buffer: &mut Self::Buffer,
        data: Vec<T>,
    ) -> Result<(), Self::Error>;
    fn load_memory<T: Scalar>(
        &mut self,
        buffer: &Self::Buffer,
        length: usize,
    ) -> Result<Vec<T>, Self::Error>;
    fn deallocate_memory(&mut self, buffer: Self::Buffer) -> Result<(), Self::Error>;
    fn compile_program(&mut self, kernel: &IRKernel) -> Result<Self::Program, Self::Error>;
    fn launch_program(
        &mut self,
        program: &Self::Program,
        args: &mut [Self::Buffer],
    ) -> Result<(), Self::Error>;
    fn release_program(&mut self, program: Self::Program) -> Result<(), Self::Error>;
}

pub(super) struct CompiledBackend<C: Compiler> {
    compiler: C,
    buffers: BTreeMap<TensorId, C::Buffer>,
    compiled_graphs: BTreeMap<Graph, CompiledGraph<C::Program>>,
    hwinfo: HWInfo,
}

impl<C: Compiler> Drop for CompiledBackend<C> {
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

/*#[derive(Debug)]
pub enum CompilerError {
    InitializationFailure(&'static str),
    OutOfDeviceMemory(&'static str),
    OutOfHostMemory(&'static str),
    BufferDoesNotExist(&'static str),
    // For all unknown errors
    GeneralExecutionError(&'static str),
}*/

/// Hardware information needed for applying optimizations
#[allow(unused)]
#[derive(Debug)]
pub struct HWInfo {
    /// Biggest kernel dimensions
    pub max_work_item_sizes: Vec<usize>,
    /// Maximum local work size threads
    pub max_work_group_size: usize,
    /// Preferred vector size in bytes
    pub preferred_vector_size: usize,
    /// Is half supported?
    pub f16_support: bool,
    /// Is double supported?
    pub f64_support: bool,
    /// Is fused multiply add supported?
    pub fmadd: bool,
    /// Global (VRAM, RAM) memory size in bytes
    pub global_mem_size: usize,
    /// Maximum memory allocation for single buffer in bytes
    pub max_mem_alloc: usize,
    /// Alignment for data types in bytes
    pub mem_align: usize,
    /// Page size (base address alignment) in bytes
    pub page_size: usize,
    /// Local memory size in bytes
    pub local_mem_size: usize,
    /// Number of registers per thread
    pub num_registers: usize,
    /// Does this hardware support native matmul of 16x16 local tiles?
    pub native_mm16x16_support: bool,
}

impl<C: Compiler> CompiledBackend<C> {
    pub(super) fn initialize() -> Result<Self, C::Error> {
        let mut compiler = C::initialize()?;
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

    pub(super) fn store<T: Scalar>(
        &mut self,
        x: TensorId,
        data: Vec<T>,
    ) -> Result<(), C::Error> {
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
    ) -> Result<Vec<T>, C::Error> {
        //println!("Attempting to load buffer with id {x}");
        if let Some(buffer) = self.buffers.get(&x) {
            return self.compiler.load_memory(buffer, length);
        } else {
            panic!("Buffer with given id does not exist. Internal bug.");
        }
    }

    pub(super) fn remove(&mut self, x: TensorId) -> Result<(), C::Error> {
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
    ) -> Result<(), C::Error> {
        //#[cfg(feature = "debug1")]
        //libc_print::libc_println!("Evaluating {to_eval:?}");
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

            // Then split those first three loops into global and local loops.
            // Reshape kernels to 6d (global, local) with some register loops
            // should be shape: [gws[0], lws[0], gws[1], lws[1], gws[2], lws[2]]
            let mut gws = [1; 3];
            let lws = [1; 3]; // TODO
            for op in &kernel.ops {
                if let VOp::Loop { axis, dimension } = op {
                    if *axis > 2 {
                        break;
                    }
                    gws[*axis] = *dimension;
                }
            }

            kernel.split_axis(0, &[gws[0], lws[0]]);
            kernel.split_axis(2, &[gws[1], lws[1]]);
            kernel.split_axis(4, &[gws[2], lws[2]]);

            /*#[cfg(feature = "debug1")]
            {
                println!();
                for op in &kernel.ops {
                    println!("{op:?}");
                }
                println!();
            }*/

            let ir_kernel = ir::compile_ir(
                &graph,
                gws,
                lws,
                &kernel.inputs,
                &kernel.outputs,
                &kernel.ops,
                &self.hwinfo,
            );

            let args: Vec<TensorId> = ir_kernel.args.keys().copied().collect();
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
            }
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

    pub(super) fn launch_graph(&mut self, graph: &Graph) -> Result<(), C::Error> {
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
