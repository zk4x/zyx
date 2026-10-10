// Copyright (C) 2025 zk4x
// SPDX-License-Identifier: LGPL-3.0-only WITH Classpath-exception-2.0

//! C/Clang CPU backend — compiles zyx kernel IR to C, compiles with clang, loads via dlopen

#![allow(non_camel_case_types)]
#![allow(non_snake_case)]
#![allow(clippy::question_mark)]
#![allow(clippy::needless_pass_by_ref_mut)]
#![allow(clippy::unused_self)]

use super::{ChunkId, Cmd, DTypeCapability, Dev, DeviceInfo, DeviceProgramId, GwsDim, LaunchArg, Placement, Pool, Shard};
use crate::DType;
use crate::dtype::Constant;
use crate::error::{BackendError, ErrorStatus};
use crate::kernel::{GPUOp, Kernel, Op, OpId};
use crate::shape::Dim;
use crate::slab::Slab;
use crate::{Map, Set};
use libloading::{Library, Symbol};
use nanoserde::DeJson;
use std::{
    ffi::CString,
    path::PathBuf,
    process::Command,
    sync::{Arc, Mutex, OnceLock},
    time::Instant,
};

// ── Global state ──────────────────────────────────────────────────────────────

static C_DEVICE: OnceLock<Mutex<CDevice>> = OnceLock::new();

#[derive(Debug, DeJson)]
#[nserde(default)]
pub struct CConfig {
    /// Enable this backend
    pub enabled: bool,
}

impl Default for CConfig {
    fn default() -> Self {
        Self { enabled: true }
    }
}

#[derive(Debug)]
pub struct CProgram {
    lib: Library,
    name: String,
}

#[derive(Debug)]
pub struct CDevice {
    device_info: Arc<DeviceInfo>,
    programs: Slab<DeviceProgramId, CProgram>,
    pub has_openmp: bool,
}

/// Preplanned CPU partition: the ordered commands plus per-command death
/// lists (slots whose final read is that command and which nothing later
/// needs). Replay allocates unbound defs on the fly from their specs,
/// launches programs back-to-back, and drops dead slots per the lists —
/// no address resolution beyond direct map indexing.
#[derive(Debug)]
pub(crate) struct CPartition {
    pub(crate) cmds: Vec<Cmd>,
    pub(crate) deaths: Vec<Vec<OpId>>,
}

impl CPartition {
    /// Replay allocates unbound defs on the fly and drops dead slots per
    /// the lists. Slots dying mid-replay are intermediaries: their chunks
    /// stay bare in a replay-local map and are released explicitly. Only
    /// final outputs (live past replay) become Arc<Placement>.
    ///
    /// The Host pool guard is never held across arg use, launches, or
    /// death drops — it is acquired per use below. Any Arc drop (a
    /// cross-partition sole-owner Arc dying here, an overwritten alias)
    /// therefore re-locks the pool mutex freely instead of self-deadlocking
    /// on the non-reentrant mutex (Placement::drop releases each shard
    /// through Pool::release, which locks).
    pub(crate) fn replay(
        &self,
        dev: &mut CDevice,
        resolved: &mut Map<OpId, Arc<Placement>>,
        vars: &Map<OpId, Constant>,
    ) -> Result<(), BackendError> {
        let host = super::host::pool();
        let dying: Set<OpId> = self.deaths.iter().flatten().copied().collect();
        let mut local: Map<OpId, ChunkId> = Map::default();
        for (idx, cmd) in self.cmds.iter().enumerate() {
            //println!("{idx} -> {cmd:?}");
            match cmd {
                Cmd::Launch { program, args, outputs } => {
                    debug_assert_eq!(program.dev, Dev::C, "C partition holds a non-C program");
                    {
                        let mut pool = super::lock(Pool::Host, host);
                        for (slot, dtype, dims) in outputs {
                            if resolved.contains_key(slot) || local.contains_key(slot) {
                                continue;
                            }
                            let bytes = dims.iter().map(|d| d.eval(vars)).fold(dtype.bit_size() as i64 / 8, |a, b| a * b);
                            debug_assert!(bytes >= 0, "C replay allocated negative bytes");
                            let chunk = pool.allocate(bytes)?;
                            if dying.contains(slot) {
                                local.insert(*slot, chunk);
                            } else {
                                resolved.insert(
                                    *slot,
                                    Arc::new(Placement {
                                        shards: vec![Shard { pool: Pool::Host, chunk, offset: 0, len: bytes as usize }],
                                    }),
                                );
                            }
                        }
                    }
                    // Resolve args to pointers. Variables are not stored
                    // anywhere — the value is copied into a local byte box
                    // here, so the kernel reads it by pointer. The raw
                    // pointers outlive the scoped guard: chunks stay owned
                    // until their death below, and chunk addresses are stable.
                    let mut var_boxes: Vec<Box<[u8]>> = Vec::new();
                    let mut ptrs: Vec<*mut u8> = Vec::with_capacity(args.len());
                    {
                        let mut pool = super::lock(Pool::Host, host);
                        for arg in args {
                            if let Some(placement) = resolved.get(arg) {
                                let [shard] = &placement.shards[..] else {
                                    todo!("multi-shard slot in C launch")
                                };
                                ptrs.push(host_ptr(&mut pool, shard.chunk, shard.offset));
                            } else if let Some(chunk) = local.get(arg) {
                                ptrs.push(host_ptr(&mut pool, *chunk, 0));
                            } else if let Some(constant) = vars.get(arg) {
                                var_boxes.push(constant.to_le_bytes().into_boxed_slice());
                                ptrs.push(var_boxes.last_mut().unwrap().as_mut_ptr());
                            } else {
                                panic!("C replay: launch arg {arg:?} is neither placed nor bound");
                            }
                        }
                    }
                    let program_ref = &dev.programs[program.program_id];
                    let func_name = CString::new(program_ref.name.as_str()).unwrap();
                    unsafe {
                        let func: Symbol<unsafe extern "C" fn(*const *mut std::ffi::c_void, usize)> =
                            program_ref.lib.get(func_name.as_bytes()).map_err(|e| BackendError {
                                status: ErrorStatus::KernelCompilation,
                                context: format!("Failed to find kernel symbol: {e}").into(),
                            })?;
                        let ptrs_raw: Vec<*mut std::ffi::c_void> = ptrs.iter().map(|p| (*p).cast::<std::ffi::c_void>()).collect();
                        // Dry run: skip device execution, keep arg binding validation.
                        // Output buffers hold uninitialized contents; callers must not read them.
                        if std::env::var("ZYX_DRY_RUN").is_err() {
                            func(ptrs_raw.as_ptr(), ptrs_raw.len());
                        }
                    }
                }
                Cmd::Alias { class, to } => {
                    let placed = resolved.get(to).unwrap_or_else(|| panic!("C replay: alias target {to:?} is unplaced")).clone();
                    resolved.insert(*class, placed);
                }
                Cmd::Copy { .. } => unreachable!("copies are Copy partitions, never device runs"),
            }
            for dead in &self.deaths[idx] {
                // No pool guard is held here, so every drop re-locks freely:
                // intermediaries release explicitly, Arc drops (including a
                // cross-partition sole-owner Arc dying in this partition)
                // run Placement::drop through Pool::release.
                if let Some(chunk) = local.remove(dead) {
                    super::lock(Pool::Host, host).release(chunk);
                } else {
                    resolved.remove(dead);
                }
            }
        }
        Ok(())
    }
}

fn device_with(config: &CConfig, debug_dev: bool) -> Result<&'static Mutex<CDevice>, BackendError> {
    if let Some(dev) = C_DEVICE.get() {
        return Ok(dev);
    }
    if !config.enabled {
        if debug_dev {
            println!("[c] configured out");
        }
        return Err(configured_out());
    }
    if debug_dev {
        println!("[c] initialized");
    }
    // C backend reuses the global host pool — no init ordering needed.
    let compilers = ["clang-11", "clang", "gcc", "cc"];
    let compiler = compilers.iter().find(|c| Command::new(c).arg("--version").output().is_ok()).copied().unwrap_or("cc");
    let has_vector_exts = Command::new(compiler)
        .args(["-O2", "-x", "c", "-", "-o", "/dev/null"])
        .arg("-Werror")
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .spawn()
        .and_then(|mut child| {
            use std::io::Write;
            child
                .stdin
                .take()
                .unwrap()
                .write_all(
                    b"typedef float float4 __attribute__((ext_vector_type(4)));\n\
                      int main() {\n\
                        float data[4] = {1,2,3,4};\n\
                        float4* p = (float4*)data;\n\
                        float4 v = *p;\n\
                        p[0] = v;\n\
                        return 0;\n\
                      }",
                )
                .ok();
            child.wait()
        })
        .map(|s| s.success())
        .unwrap_or(false);
    let has_openmp = Command::new(compiler)
        .args(["-shared", "-O3", "-fopenmp", "-x", "c", "-", "-o", "/dev/null"])
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .spawn()
        .and_then(|mut child| {
            use std::io::Write;
            child.stdin.take().unwrap().write_all(b"int main(){return 0;}").ok();
            child.wait()
        })
        .map(|s| s.success())
        .unwrap_or(false);
    let dev = Mutex::new(CDevice {
        device_info: Arc::new(DeviceInfo {
            compute: 10 * 1024 * 1024 * 1024 * 1024,
            max_global_work_dims: vec![Dim::from(1_000_000_000); 3],
            max_local_threads: 1,
            max_local_work_dims: vec![1, 1, 1],
            preferred_vector_size: 8,
            local_mem_size: 0,
            max_register_bytes: 1000,
            tensor_cores: false,
            warp_size: 1,
            cc: [0, 0],
            dtype_capability: [DTypeCapability::all(); DType::N_DTYPES],
            has_native_exp2: false,
            supported_vec_lens: vec![2, 4, 8, 16],
            tenstorrent: false,
            tile: [1, 1],
            tile_sizes: vec![],
            wmma_layouts: vec![],
            num_circular_buffers: 0,
            has_openmp,
        }),
        programs: Slab::new(),
        has_openmp,
    });
    if debug_dev {
        println!("[c] vector extensions: {has_vector_exts}");
        println!("[c] OpenMP: {has_openmp}");
    }
    let _ = C_DEVICE.set(dev);
    Ok(C_DEVICE.get().unwrap())
}

fn configured_out() -> BackendError {
    BackendError { status: ErrorStatus::Initialization, context: "C backend configured out".into() }
}

pub(super) fn device() -> Result<&'static Mutex<CDevice>, BackendError> {
    device_with(&super::config().c, super::debug_backends())
}

/// Resolves a launch arg region to a host pointer: chunk base plus the
/// region offset. Host memory is directly addressable, so any offset
/// applies here.
fn host_ptr(memory_pool: &mut super::host::HostMemoryPool, chunk: ChunkId, offset: usize) -> *mut u8 {
    unsafe { memory_pool.buffer_ptr_mut(chunk).add(offset) }
}

impl CDevice {
    /// Schedule a command run for the C device: orders nothing (queue order
    /// is program order), precomputes per-command death lists from the final
    /// read of every slot. A slot dies at its last use unless pinned
    /// (a plan output or read after this partition).
    pub(crate) fn schedule(cmds: Vec<Cmd>, outputs: &Set<OpId>, live_out: Set<OpId>) -> CPartition {
        let mut last_use: Map<OpId, usize> = Map::default();
        for (idx, cmd) in cmds.iter().enumerate() {
            for r in cmd.reads() {
                last_use.insert(r, idx);
            }
        }
        let mut pinned = outputs.clone();
        pinned.extend(live_out);
        let mut deaths: Vec<Vec<OpId>> = vec![Vec::new(); cmds.len()];
        for (slot, idx) in last_use {
            if !pinned.contains(&slot) {
                deaths[idx].push(slot);
            }
        }
        CPartition { cmds, deaths }
    }

    /// Copy executing a transfer into the host pool. Delegates to
    /// [`super::host::copy`]; the C device itself needs no init for this.
    /// The destination is always host-resident. Single-shard placements only.
    pub fn copy(&self, src: &Placement, dst: &Placement, bytes: Dim) -> Result<(), BackendError> {
        super::host::copy(src, dst, bytes)
    }

    pub fn info(&self) -> Arc<DeviceInfo> {
        self.device_info.clone()
    }

    pub fn free_compute(&self) -> u128 {
        self.device_info.compute
    }

    pub fn release(&mut self, program_id: DeviceProgramId) {
        self.programs.remove(program_id);
    }

    pub fn compile(&mut self, kernel: &Kernel, debug_asm: bool) -> Result<DeviceProgramId, BackendError> {
        // --- Phase 0: Compute kernel hash and check disk cache ---
        // Keyed on the ORIGINAL kernel hash: stable across the render
        // port. render_c computes the same hash pre-render for the
        // compiled symbol name, so cache hits still resolve.
        let hash = kernel.get_hash();
        let name = format!("k_{hash:016x}");

        let cache_dir = std::env::var_os("XDG_CONFIG_HOME")
            .and_then(|p| {
                let p = PathBuf::from(p);
                if p.is_absolute() { Some(p) } else { None }
            })
            .or_else(|| std::env::home_dir().map(|h| h.join(".config")))
            .map(|p| p.join("zyx/cache/c"));

        // Skip the disk cache when debug_asm is set so the generated source is
        // always printed below.
        if !debug_asm && let Some(ref cache_dir) = cache_dir {
            let cached_so = cache_dir.join(format!("{hash:016x}.so"));
            if cached_so.is_file()
                && let Ok(lib) = unsafe { Library::new(&cached_so) }
            {
                let program_id = self.programs.push(CProgram { lib, name });
                return Ok(program_id);
            }
        }

        // --- Render to descriptor; decode head positions ---
        let rendered = kernel.render()?;
        let mut order = rendered.ops_in_order();
        let Some(GPUOp::Params(_params)) = order.next().and_then(Op::as_gpu) else {
            return Err(BackendError { status: ErrorStatus::KernelCompilation, context: "head op is not Params".into() });
        };
        let Some(GPUOp::Grid(gws)) = order.next().and_then(Op::as_gpu) else {
            return Err(BackendError { status: ErrorStatus::KernelCompilation, context: "second op is not Grid".into() });
        };
        let Some(Op::Source(source)) = order.next() else {
            return Err(BackendError { status: ErrorStatus::KernelCompilation, context: "third op is not Source".into() });
        };
        // The -fopenmp link flag only pays off above one thread. A dynamic
        // first axis falls back to serial — the same condition the old
        // resolve-const unwrap_or(1) produced.
        let gws0 = match &gws[0] {
            GwsDim::Const(d) => *d,
            _ => 1,
        }
        .max(1);

        // --- Codegen: the source rides in the descriptor; never re-run ---
        let tmp_dir = std::env::temp_dir().join(format!("zyx_c_{}", std::process::id()));
        let _ = std::fs::create_dir_all(&tmp_dir);
        let c_path = tmp_dir.join(format!("{name}.c"));
        let so_path = tmp_dir.join(format!("{name}.so"));

        let full_source = source.as_str();
        std::fs::write(&c_path, &full_source).map_err(|e| BackendError {
            status: ErrorStatus::KernelCompilation,
            context: format!("Failed to write C source: {e}").into(),
        })?;

        // Try clang-11, clang, gcc, cc in order
        let compilers = ["clang-11", "clang", "gcc", "cc"];
        let compiler = compilers.iter().find(|c| Command::new(c).arg("--version").output().is_ok()).copied().unwrap_or("cc");
        let is_clang = compiler.contains("clang");
        let mut cmd = Command::new(compiler);
        // -fno-associative-math is necessary for numerical stability under
        // LLVM's -ffast-math reassociation: clang rewrites x*a + (1-x)*b into
        // x*(a-b) + b, which catastrophically cancels when b is a huge value
        // like the -1e30 log-prob used by ctc_loss.
        cmd.args(["-shared", "-O3", "-ffast-math", "-fno-associative-math", "-fPIC", "-o"])
            .arg(&so_path)
            .arg(&c_path)
            .arg("-lm");
        if self.has_openmp && gws0 > 1 {
            cmd.arg(if is_clang { "-fopenmp=libgomp" } else { "-fopenmp" });
        }
        let output = cmd.output().map_err(|e| BackendError {
            status: ErrorStatus::KernelCompilation,
            context: format!("Failed to run compiler '{compiler}': {e}. Is a C compiler installed?").into(),
        })?;
        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            if debug_asm {
                println!("[C] compiler stderr:\n{stderr}");
            }
            return Err(BackendError {
                status: ErrorStatus::KernelCompilation,
                context: format!("Compiler '{compiler}' compilation failed:\n{stderr}").into(),
            });
        }

        if debug_asm {
            println!();
            println!("{full_source}");
        }

        // Cache the compiled .so for future runs
        if let Some(ref cache_dir) = cache_dir {
            let _ = std::fs::create_dir_all(cache_dir);
            let cached_so = cache_dir.join(format!("{hash:016x}.so"));
            let _ = std::fs::copy(&so_path, &cached_so);
        }

        // Load the shared library
        let lib = unsafe { Library::new(&so_path) }.map_err(|e| BackendError {
            status: ErrorStatus::KernelCompilation,
            context: format!("Failed to dlopen compiled kernel: {e}").into(),
        })?;

        let program_id = self.programs.push(CProgram { lib, name });
        Ok(program_id)
    }

    #[allow(clippy::needless_pass_by_value)]
    pub fn launch_timed(
        &mut self,
        program_id: DeviceProgramId,
        pool_handle: Pool,
        args: &[LaunchArg],
    ) -> Result<u64, BackendError> {
        // Sequential CPU: no queues, no pending window — the kernel runs to
        // completion before returning, so a wall-clock bracket is already an
        // uncontended measurement.
        let start = Instant::now();
        // Sequential CPU: the kernel runs to completion before returning.
        debug_assert_eq!(pool_handle, Pool::Host);
        let host = super::host::pool();
        let mut memory_pool = super::lock(pool_handle, host);

        let program = &self.programs[program_id];

        // Get buffer pointers. Variables are not stored anywhere — the value is
        // copied into a local byte box here, so the kernel reads it by pointer.
        let mut vars: Vec<Box<[u8]>> = Vec::new();
        let mut ptrs: Vec<*mut u8> = Vec::with_capacity(args.len());
        for arg in args {
            match *arg {
                LaunchArg::Buffer { chunk, offset, .. } => {
                    ptrs.push(host_ptr(&mut memory_pool, chunk, offset));
                }
                LaunchArg::Variable(constant) => {
                    vars.push(constant.to_le_bytes().into_boxed_slice());
                    let var = vars.last_mut().unwrap();
                    ptrs.push(var.as_mut_ptr());
                }
            }
        }

        let func_name = CString::new(program.name.as_str()).unwrap();
        unsafe {
            let func: Symbol<unsafe extern "C" fn(*const *mut std::ffi::c_void, usize)> =
                program.lib.get(func_name.as_bytes()).map_err(|e| BackendError {
                    status: ErrorStatus::KernelCompilation,
                    context: format!("Failed to find kernel symbol: {e}").into(),
                })?;
            let ptrs_raw: Vec<*mut std::ffi::c_void> = ptrs.iter().map(|p| (*p).cast::<std::ffi::c_void>()).collect();
            func(ptrs_raw.as_ptr(), ptrs_raw.len());
        }

        Ok(start.elapsed().as_nanos() as u64)
    }
}
