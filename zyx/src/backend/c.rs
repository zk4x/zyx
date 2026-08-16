// Copyright (C) 2025 zk4x
// SPDX-License-Identifier: LGPL-3.0-only

//! C/Clang CPU backend — compiles zyx kernel IR to C, compiles with clang, loads via dlopen

#![allow(non_camel_case_types)]
#![allow(non_snake_case)]
#![allow(clippy::question_mark)]
#![allow(clippy::needless_pass_by_ref_mut)]
#![allow(clippy::unused_self)]

use super::{
    DTypeCapability, Device, DeviceId, DeviceInfo, DeviceProgramId, Event, MemoryPool, PoolBufferId, PoolId, host::HostMemoryPool,
};
use crate::DType;
use crate::error::{BackendError, ErrorStatus};
use crate::kernel::{IdxKind, Kernel, Op};
use crate::shape::Dim;
use crate::slab::Slab;
use libloading::{Library, Symbol};
use nanoserde::DeJson;
use std::{ffi::CString, path::PathBuf, process::Command};

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
    device_info: DeviceInfo,
    memory_pool_id: PoolId,
    programs: Slab<DeviceProgramId, CProgram>,
    has_openmp: bool,
}

pub(super) fn initialize_device(
    config: &CConfig,
    memory_pools: &mut Slab<PoolId, MemoryPool>,
    devices: &mut Slab<DeviceId, Device>,
    debug_dev: bool,
) -> Result<(), BackendError> {
    if !config.enabled {
        if debug_dev {
            println!("[c] configured out");
        }
        return Ok(());
    }
    if debug_dev {
        println!("[c] initialized");
    }
    // C backend reuses HostMemoryPool — doesn't create its own pool
    // Just register the device with the host pool
    if memory_pools.is_empty() {
        return Err(BackendError {
            status: ErrorStatus::Initialization,
            context: "C backend requires HostMemoryPool to be initialized first.".into(),
        });
    }
    let pool_id = PoolId::from(0); // use the first (host) pool
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
    devices.push(Device::C(CDevice {
        device_info: DeviceInfo {
            compute: 10 * 1024 * 1024 * 1024 * 1024,
            max_global_work_dims: vec![Dim::from(1_000_000_000u64); 3],
            max_local_threads: 1,
            max_local_work_dims: vec![1, 1, 1],
            preferred_vector_size: 8,
            local_mem_size: 0,
            max_register_bytes: 1000,
            tensor_cores: false,
            warp_size: 1,
            dtype_capability: [DTypeCapability::all(); DType::N_DTYPES],
            has_native_exp2: false,
            supported_vec_lens: vec![2, 4, 8, 16],
        },
        memory_pool_id: pool_id,
        programs: Slab::new(),
        has_openmp,
    }));
    if debug_dev {
        println!("[c] vector extensions: {has_vector_exts}");
        println!("[c] OpenMP: {has_openmp}");
    }
    Ok(())
}

impl CDevice {
    pub const fn deinitialize(&mut self) {}

    pub const fn info(&self) -> &DeviceInfo {
        &self.device_info
    }

    pub const fn memory_pool_id(&self) -> PoolId {
        self.memory_pool_id
    }

    pub const fn free_compute(&self) -> u128 {
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
        let mut gws0 = 1u64;
        let mut op_id = kernel.head;
        while !op_id.is_null() {
            if let Op::Index { len, axis, kind: IdxKind::Group } = kernel.ops[op_id].op
                && axis == 0
            {
                gws0 = kernel.index_len(len).max(1);
            }
            op_id = kernel.next_op(op_id);
        }

        // --- Codegen ---
        let tmp_dir = std::env::temp_dir().join(format!("zyx_c_{}", std::process::id()));
        let _ = std::fs::create_dir_all(&tmp_dir);
        let c_path = tmp_dir.join(format!("{name}.c"));
        let so_path = tmp_dir.join(format!("{name}.so"));

        let full_source = kernel.generate_c(&self.device_info, self.has_openmp, &name)?;
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
        memory_pool: &mut HostMemoryPool,
        args: &[PoolBufferId],
        event_wait_list: Vec<Event>,
    ) -> Result<Event, BackendError> {
        let _ = event_wait_list; // sync not needed for sequential CPU

        let program = &self.programs[program_id];

        // Get buffer pointers
        let mut ptrs: Vec<*mut u8> = Vec::with_capacity(args.len());
        for &arg in args {
            let ptr = memory_pool.buffer_ptr_mut(arg);
            ptrs.push(ptr);
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

        Ok(Event::Host(super::host::HostEvent))
    }
}
