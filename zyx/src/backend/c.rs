// Copyright (C) 2025 zk4x
// SPDX-License-Identifier: LGPL-3.0-only WITH Classpath-exception-2.0

//! C/Clang CPU backend — compiles zyx kernel IR to C, compiles with clang, loads via dlopen

#![allow(non_camel_case_types)]
#![allow(non_snake_case)]
#![allow(clippy::question_mark)]
#![allow(clippy::needless_pass_by_ref_mut)]
#![allow(clippy::unused_self)]

use super::{DTypeCapability, DeviceInfo, DeviceProgramId, LaunchArg, Pool};
use crate::DType;
use crate::error::{BackendError, ErrorStatus};
use crate::kernel::{Kernel, Op, RangeKind};
use crate::shape::Dim;
use crate::slab::Slab;
use libloading::{Library, Symbol};
use nanoserde::DeJson;
use std::{
    ffi::CString,
    path::PathBuf,
    process::Command,
    sync::{Arc, Mutex, OnceLock},
};

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

/// Process-wide C device. Owned here — `mod.rs` only holds the `Dev::C`
/// handle. `C_INIT` serializes first construction only; compile/launch
/// take the device lock, never the init lock.
static C_DEVICE: OnceLock<Arc<Mutex<CDevice>>> = OnceLock::new();
static C_INIT: Mutex<()> = Mutex::new(());

fn device_with(config: &CConfig, debug_dev: bool) -> Result<Arc<Mutex<CDevice>>, BackendError> {
    if let Some(dev) = C_DEVICE.get() {
        return Ok(dev.clone());
    }
    let _init = C_INIT.lock().unwrap_or_else(|_| panic!("c device init lock poisoned"));
    if let Some(dev) = C_DEVICE.get() {
        return Ok(dev.clone());
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
    let dev = Arc::new(Mutex::new(CDevice {
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
    }));
    if debug_dev {
        println!("[c] vector extensions: {has_vector_exts}");
        println!("[c] OpenMP: {has_openmp}");
    }
    let _ = C_DEVICE.set(dev.clone());
    Ok(dev)
}

fn configured_out() -> BackendError {
    BackendError { status: ErrorStatus::Initialization, context: "C backend configured out".into() }
}

pub(super) fn device() -> Result<Arc<Mutex<CDevice>>, BackendError> {
    device_with(&super::config().c, super::debug_backends())
}

impl CDevice {
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

        // --- Compute global work size ---
        let mut gws0 = 1i64;
        let mut op_id = kernel.head;
        let mut steps_op_id = 0usize;
        while !op_id.is_null() {
            steps_op_id += 1;
            if steps_op_id > 10_000 {
                panic!("compile did not finish in 10000 steps");
            }
            if let Op::Range { axis, kind: RangeKind::Group(len) } = kernel.ops[op_id].op
                && axis == 0
            {
                gws0 = kernel.resolve_const(len).and_then(crate::dtype::Constant::as_dim).unwrap_or(1).max(1);
            }
            op_id = kernel.next_op(op_id);
        }

        // --- Codegen ---
        let tmp_dir = std::env::temp_dir().join(format!("zyx_c_{}", std::process::id()));
        let _ = std::fs::create_dir_all(&tmp_dir);
        let c_path = tmp_dir.join(format!("{name}.c"));
        let so_path = tmp_dir.join(format!("{name}.so"));

        let full_source = kernel.generate_c(self.has_openmp, &name)?;
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
    pub fn launch(
        &mut self,
        program_id: DeviceProgramId,
        pool_handle: Pool,
        args: &[LaunchArg],
    ) -> Result<(), BackendError> {
        // Sequential CPU: the kernel runs to completion before returning.
        debug_assert_eq!(pool_handle, Pool::Host);
        let host = super::host::pool();
        let mut memory_pool = super::lock(pool_handle, &host);

        let program = &self.programs[program_id];

        // Get buffer pointers. Variables are not stored anywhere — the value is
        // copied into a local byte box here, so the kernel reads it by pointer.
        let mut vars: Vec<Box<[u8]>> = Vec::new();
        let mut ptrs: Vec<*mut u8> = Vec::with_capacity(args.len());
        for arg in args {
            match *arg {
                LaunchArg::Buffer(buffer_id) => {
                    let ptr = memory_pool.buffer_ptr_mut(buffer_id);
                    ptrs.push(ptr);
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

        Ok(())
    }
}
