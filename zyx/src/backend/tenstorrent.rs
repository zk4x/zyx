// Copyright (C) 2025 zk4x
// SPDX-License-Identifier: LGPL-3.0-only
//
// Tenstorrent backend for zyx.
//
// # Grid indexing (gidx)
//
// The tensix cores on a device form a 2D grid. Two kernel-index
// dimensions are available, mapped to the core's logical coordinate:
//
//   - gidx0 → core row (y)
//   - gidx1 → core column (x)
//
// For Blackhole P100a the worker grid is 10×12 (rows × columns),
// giving 120 cores total. A single-core launch uses `gidx0 = 0,
// gidx1 = 0` (also written `{0, 0}` in CoreCoord notation).

use super::{Device, DeviceId, DeviceInfo, DeviceProgramId, Event, Kernel, MemoryPool, PoolBufferId, PoolId};
use crate::{
    DType, Map,
    backend::DTypeCapability,
    error::{BackendError, ErrorStatus},
    kernel::{BOp, MemLayout, Op, OpId, Scope, UOp},
    shape::Dim,
    slab::Slab,
};
use nanoserde::DeJson;
use std::{
    ffi::CString,
    fmt::Write,
    io::{BufRead, BufReader, BufWriter, Write as IoWrite},
    path::PathBuf,
    process::{Child, ChildStdin, ChildStdout, Command},
    sync::{Arc, Mutex},
};

// ---------------------------------------------------------------------------
// DRAM size lookup
// ---------------------------------------------------------------------------

/// GDDR6 sizes for known Blackhole PCI subsystem IDs.
///
/// Blackhole has 8 DRAM channels, each connected to a 4 GB GDDR6 chip.
/// Some boards have channels harvested (fused off) for binning.
/// P100/P100A: 7 usable channels → 28 GB.
/// P150:        8 usable channels → 32 GB.
/// P300:        2 chips × 8 channels → 64 GB.
///
/// These are the total per-board values. The kernel driver does not expose
/// GDDR6 capacity — `dma_alloc_coherent` without IOMMU draws from system
/// memory, not device GDDR6 — so we use this table as a fallback.
///
/// Sources:
/// - https://docs.tenstorrent.com/aibs/blackhole/specifications.html
/// - tt-umd `board_upi_map` and `expected_dram_harvested_units_map`
/// - tt-metal `blackhole_140_arch.yaml` (dram_bank_size: 4278190080 ≈ 4 GB)
const DRAM_SIZE_TABLE: &[(u16, &str, u64)] = &[
    (0x0036, "p100", 28u64 * 1024 * 1024 * 1024),
    (0x0040, "p150a", 32u64 * 1024 * 1024 * 1024),
    (0x0041, "p150b", 32u64 * 1024 * 1024 * 1024),
    (0x0042, "p150c", 32u64 * 1024 * 1024 * 1024),
    (0x0043, "p100a", 28u64 * 1024 * 1024 * 1024),
    (0x0044, "p300b", 64u64 * 1024 * 1024 * 1024),
    (0x0045, "p300a", 64u64 * 1024 * 1024 * 1024),
    (0x0046, "p300c", 64u64 * 1024 * 1024 * 1024),
];

fn detect_dram_bytes() -> u64 {
    let pci_devices = std::path::Path::new("/sys/bus/pci/devices");
    if let Ok(entries) = std::fs::read_dir(pci_devices) {
        for entry in entries.flatten() {
            let vendor_path = entry.path().join("vendor");
            let vendor = std::fs::read_to_string(&vendor_path).unwrap_or_default();
            if vendor.trim() == "0x1e52" {
                let subsys = std::fs::read_to_string(entry.path().join("subsystem_device")).unwrap_or_default();
                if let Ok(id) = u16::from_str_radix(subsys.trim().trim_start_matches("0x"), 16) {
                    for &(sid, _name, size) in DRAM_SIZE_TABLE {
                        if sid == id {
                            return size;
                        }
                    }
                }
            }
        }
    }
    64u64 * 1024 * 1024 * 1024
}
// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

#[derive(Default, Debug, DeJson)]
#[nserde(default)]
pub struct TTConfig {
    /// If set to None, then it will automatically use all Tenstorrent devices,
    /// otherwise it uses only selected devices
    device_ids: Option<Vec<i32>>,
}

// ---------------------------------------------------------------------------
// Per-buffer tracking: index into C++ runtime's vector<MeshBuffer>
// ---------------------------------------------------------------------------

#[derive(Debug)]
struct TTBuffer {
    dev_index: u32,
    size: u64,
}

// ---------------------------------------------------------------------------
// Memory pool — device DRAM buffers managed by C++ runtime.
// TTBuffer is a handle (u32 dev_index) into the runtime's buffer list.
// The pool shares the runtime IPC channel with TTDevice via Arc<Mutex>.
// ---------------------------------------------------------------------------

#[derive(Debug)]
pub struct TTMemoryPool {
    buffers: Slab<PoolBufferId, TTBuffer>,
    runtime: Arc<Mutex<RuntimeProcess>>,
    free_bytes: Dim,
}

#[derive(Debug, Clone)]
pub struct TTEvent;

pub(super) fn initialize_device(
    config: &TTConfig,
    memory_pools: &mut Slab<PoolId, MemoryPool>,
    devices: &mut Slab<DeviceId, Device>,
    debug_dev: bool,
) -> Result<(), BackendError> {
    if let Some(device_ids) = &config.device_ids
        && device_ids.is_empty()
    {
        if debug_dev {
            println!("[tenstorrent] configured out");
        }
        return Ok(());
    }

    let dram_bytes = detect_dram_bytes();
    if debug_dev {
        println!("[tenstorrent] device initialized");
        println!("[tenstorrent] device total memory: {} MB", dram_bytes / (1024 * 1024));
    }

    // Compute config dir from XDG convention
    let config_base = std::env::var_os("XDG_CONFIG_HOME")
        .and_then(|p| {
            let p = PathBuf::from(p);
            if p.is_absolute() { Some(p) } else { None }
        })
        .or_else(|| std::env::home_dir().map(|h| h.join(".config")))
        .unwrap_or_else(|| PathBuf::from("/tmp"));

    let cache_dir = config_base.join("zyx/cache/tt");

    // The runtime binary must be installed at the config dir by build.rs
    let runtime_path = config_base.join("zyx/zyx-tt-runtime");
    if !runtime_path.exists() {
        return Err(BackendError {
            status: ErrorStatus::Initialization,
            context: format!("runtime not found at {}. Rebuild with TT_METAL_ROOT set.", runtime_path.display()).into(),
        });
    }

    // Spawn the runtime eagerly — both pool and device need it
    let runtime = Arc::new(Mutex::new(RuntimeProcess::new(&runtime_path.to_string_lossy(), &cache_dir.to_string_lossy())?));

    let pool_id = memory_pools.len();
    let pool = MemoryPool::TT(TTMemoryPool { buffers: Slab::new(), runtime: runtime.clone(), free_bytes: Dim::from(dram_bytes) });
    memory_pools.push(pool);

    let _device_id = devices.len();
    devices.push(Device::TT(TTDevice {
        device_info: DeviceInfo {
            compute: 200_000_000_000_000, // ~200 TFLOPS BF16
            max_global_work_dims: vec![Dim::from(u32::MAX); 3],
            max_local_threads: 1024,
            max_local_work_dims: vec![1, 1024, 1],
            preferred_vector_size: 32,
            local_mem_size: 1_500_000, // 1.5 MB L1 per Tensix core
            max_register_bytes: 128,
            tensor_cores: true,
            warp_size: 1, // Tensix has no SIMT warps
            dtype_capability: [DTypeCapability::all(); DType::N_DTYPES],
            has_native_exp2: false,
            supported_vec_lens: vec![32],
        },
        memory_pool_id: pool_id,
        runtime,
        programs: Slab::new(),
    }));
    Ok(())
}

fn create_temp_shm(size: u64) -> Result<(CString, *mut u8, u64), BackendError> {
    let pid = std::process::id();
    let ns = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap_or_default().as_nanos();
    let name = format!("/zyx-tt-{pid:x}-{ns:x}");
    let cname = CString::new(name.clone())
        .map_err(|_| BackendError { status: ErrorStatus::MemoryAllocation, context: "invalid shm path".into() })?;

    let fd = unsafe { libc::shm_open(cname.as_ptr(), libc::O_CREAT | libc::O_RDWR | libc::O_EXCL, 0o600) };
    if fd < 0 {
        return Err(BackendError {
            status: ErrorStatus::MemoryAllocation,
            context: format!("shm_open errno={}", std::io::Error::last_os_error()).into(),
        });
    }

    if unsafe { libc::ftruncate(fd, size as i64) } < 0 {
        unsafe { libc::close(fd) };
        let _ = unsafe { libc::shm_unlink(cname.as_ptr()) };
        return Err(BackendError { status: ErrorStatus::MemoryAllocation, context: "ftruncate shm".into() });
    }

    let ptr =
        unsafe { libc::mmap(std::ptr::null_mut(), size as usize, libc::PROT_READ | libc::PROT_WRITE, libc::MAP_SHARED, fd, 0) };
    if ptr == libc::MAP_FAILED {
        unsafe { libc::close(fd) };
        let _ = unsafe { libc::shm_unlink(cname.as_ptr()) };
        return Err(BackendError { status: ErrorStatus::MemoryAllocation, context: "mmap shm".into() });
    }

    unsafe { libc::close(fd) };
    Ok((cname, ptr as *mut u8, size))
}

impl TTMemoryPool {
    pub fn deinitialize(&mut self) {
        let _ = self.runtime.lock().unwrap().exit();
    }

    pub fn free_bytes(&self) -> Dim {
        self.free_bytes
    }

    pub fn allocate(&mut self, bytes: Dim) -> Result<(PoolBufferId, Event), BackendError> {
        let bytes_u64: u64 = u64::try_from(bytes).map_err(|_| BackendError {
            status: ErrorStatus::MemoryAllocation,
            context: "allocation size exceeds 64-bit".into(),
        })?;
        if bytes > self.free_bytes {
            return Err(BackendError { status: ErrorStatus::MemoryAllocation, context: "out of device memory".into() });
        }
        let rt = &self.runtime;
        let tile_bytes: u64 = 2048;
        let dev_index = rt.lock().unwrap().alloc_buf(bytes_u64, tile_bytes)?;
        let buf = TTBuffer { dev_index, size: bytes_u64 };
        let id = self.buffers.push(buf);
        Ok((id, Event::TT(TTEvent)))
    }

    pub fn deallocate(&mut self, buffer_id: PoolBufferId, event_wait_list: Vec<Event>) {
        let _ = event_wait_list;
        if self.buffers.contains_key(buffer_id) {
            let buf = unsafe { self.buffers.remove_and_return(buffer_id) };
            let _ = self.runtime.lock().unwrap().free_buf(buf.dev_index);
        }
    }

    pub fn host_to_pool(&mut self, src: &[u8], dst: PoolBufferId, event_wait_list: Vec<Event>) -> Result<Event, BackendError> {
        let _ = event_wait_list;
        let rt = &self.runtime;
        let buf = self
            .buffers
            .get_mut(dst)
            .ok_or_else(|| BackendError { status: ErrorStatus::MemoryCopyH2P, context: "invalid buffer id".into() })?;
        let len = src.len().min(buf.size as usize);
        let (cname, shm_ptr, _) = create_temp_shm(len as u64)?;
        let shm_path = cname.to_str().unwrap_or("/none");
        unsafe { std::ptr::copy_nonoverlapping(src.as_ptr(), shm_ptr, len) };
        rt.lock().unwrap().write_buf(buf.dev_index, shm_path, len as u64)?;
        unsafe {
            libc::munmap(shm_ptr as *mut libc::c_void, len as usize);
            libc::shm_unlink(cname.as_ptr());
        }
        Ok(Event::TT(TTEvent))
    }

    pub fn pool_to_host(&mut self, src: PoolBufferId, dst: &mut [u8], event_wait_list: Vec<Event>) -> Result<(), BackendError> {
        let _ = event_wait_list;
        let rt = &self.runtime;
        let buf = self
            .buffers
            .get_mut(src)
            .ok_or_else(|| BackendError { status: ErrorStatus::MemoryCopyP2H, context: "invalid buffer id".into() })?;
        let len = dst.len().min(buf.size as usize);
        let (cname, shm_ptr, _) = create_temp_shm(len as u64)?;
        let shm_path = cname.to_str().unwrap_or("/none");
        rt.lock().unwrap().read_buf(buf.dev_index, shm_path, len as u64)?;
        unsafe {
            std::ptr::copy_nonoverlapping(shm_ptr, dst.as_mut_ptr(), len);
            libc::munmap(shm_ptr as *mut libc::c_void, len as usize);
            libc::shm_unlink(cname.as_ptr());
        }
        Ok(())
    }

    pub fn pool_to_pool(
        &mut self,
        src_pool: &mut MemoryPool,
        src: PoolBufferId,
        dst: PoolBufferId,
        event_wait_list: Vec<Event>,
    ) -> Result<Event, BackendError> {
        match src_pool {
            MemoryPool::Host(host_pool) => {
                let data = host_pool.get_buffer(src);
                self.host_to_pool(data, dst, event_wait_list)
            }
            _ => todo!(),
        }
    }

    pub fn sync_events(&mut self, events: Vec<Event>) -> Result<(), BackendError> {
        let _ = self;
        let _ = events;
        Ok(())
    }

    pub fn release_events(&mut self, events: Vec<Event>) {
        let _ = self;
        let _ = events;
    }

    pub fn dev_index(&self, buffer_id: PoolBufferId) -> Result<u32, BackendError> {
        if self.buffers.contains_key(buffer_id) {
            Ok(self.buffers[buffer_id].dev_index)
        } else {
            Err(BackendError { status: ErrorStatus::MemoryAllocation, context: "invalid buffer id".into() })
        }
    }
}

// ---------------------------------------------------------------------------
// Runtime process management (JSON IPC over stdin/stdout)
// ---------------------------------------------------------------------------

#[derive(Debug)]
struct RuntimeProcess {
    stdin: BufWriter<ChildStdin>,
    stdout: BufReader<ChildStdout>,
    child: Child,
    timeout_ms: u64,
}

impl RuntimeProcess {
    fn new(runtime_path: &str, cache_dir: &str) -> Result<Self, BackendError> {
        eprintln!("[TT_DEBUG] spawning tt-runtime from {runtime_path}");

        // Kill any previous zyx-tt-runtime that might still hold the device
        let _ = std::process::Command::new("pkill").arg("-9").arg("zyx-tt-runtime").output();

        let mut child = Command::new(runtime_path)
            .stdin(std::process::Stdio::piped())
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::inherit())
            .spawn()
            .map_err(|e| BackendError {
                status: ErrorStatus::Initialization,
                context: format!("spawn tt-runtime {runtime_path}: {e}").into(),
            })?;

        eprintln!("[TT_DEBUG] child spawned, taking stdin/stdout");
        let stdin = child
            .stdin
            .take()
            .ok_or_else(|| BackendError { status: ErrorStatus::Initialization, context: "tt-runtime: no stdin".into() })?;
        let stdout = child
            .stdout
            .take()
            .ok_or_else(|| BackendError { status: ErrorStatus::Initialization, context: "tt-runtime: no stdout".into() })?;

        let mut rt = RuntimeProcess { stdin: BufWriter::new(stdin), stdout: BufReader::new(stdout), child, timeout_ms: 30000 };

        eprintln!("[TT_DEBUG] sending init");
        let init_json = format!(r#"{{"cmd":"init","cache_dir":"{cache_dir}"}}"#);
        rt.send(&init_json)?;
        eprintln!("[TT_DEBUG] init sent, waiting for response");
        let resp = rt.recv_with_timeout(rt.timeout_ms)?;
        eprintln!("[TT_DEBUG] init response: {resp}");
        if resp.contains("\"error\"") {
            let msg = extract_json_str(&resp, "msg").unwrap_or_else(|| "unknown".into());
            return Err(BackendError {
                status: ErrorStatus::Initialization,
                context: format!("tt-runtime init error: {msg}").into(),
            });
        }
        Ok(rt)
    }

    fn send(&mut self, json: &str) -> Result<(), BackendError> {
        eprintln!("[RUST_SEND] {}", &json[..json.len().min(200)]);
        self.stdin
            .write_all(json.as_bytes())
            .map_err(|e| BackendError { status: ErrorStatus::KernelLaunch, context: format!("tt-runtime write: {e}").into() })?;
        self.stdin.write_all(b"\n").map_err(|e| BackendError {
            status: ErrorStatus::KernelLaunch,
            context: format!("tt-runtime write nl: {e}").into(),
        })?;
        self.stdin
            .flush()
            .map_err(|e| BackendError { status: ErrorStatus::KernelLaunch, context: format!("tt-runtime flush: {e}").into() })?;
        Ok(())
    }

    fn poll_read(&mut self, timeout_ms: u64) -> Result<bool, BackendError> {
        match self.child.try_wait() {
            Ok(Some(status)) => {
                return Err(BackendError {
                    status: ErrorStatus::KernelLaunch,
                    context: format!("tt-runtime exited unexpectedly (status {status})").into(),
                });
            }
            Err(e) => {
                return Err(BackendError {
                    status: ErrorStatus::KernelLaunch,
                    context: format!("tt-runtime wait error: {e}").into(),
                });
            }
            Ok(None) => {}
        }

        let fd = std::os::unix::io::AsRawFd::as_raw_fd(self.stdout.get_mut());
        let mut pollfd = libc::pollfd { fd, events: libc::POLLIN, revents: 0 };

        let timeout_ms = i32::try_from(timeout_ms).unwrap_or(i32::MAX);
        let ret = unsafe { libc::poll(&mut pollfd, 1, timeout_ms) };

        match ret {
            -1 => {
                let err = std::io::Error::last_os_error();
                Err(BackendError { status: ErrorStatus::KernelLaunch, context: format!("poll error: {err}").into() })
            }
            0 => Ok(false),
            _ => Ok(pollfd.revents & libc::POLLIN != 0),
        }
    }

    fn recv_with_timeout(&mut self, timeout_ms: u64) -> Result<String, BackendError> {
        let mut attempts = 0;
        let max_attempts = 3;
        let poll_timeout = timeout_ms / max_attempts;

        while attempts < max_attempts {
            if self.poll_read(poll_timeout)? {
                let mut line = String::new();
                match self.stdout.read_line(&mut line) {
                    Ok(0) => {
                        return Err(BackendError {
                            status: ErrorStatus::KernelLaunch,
                            context: "tt-runtime closed stdout".into(),
                        });
                    }
                    Ok(_) => {
                        let trimmed = line.trim().to_string();
                        // Skip non-JSON lines (UMD log messages leaking to stdout)
                        if trimmed.starts_with('{') {
                            eprintln!("[RUST_RECV] {trimmed}");
                            return Ok(trimmed);
                        }
                        // Log line — keep reading
                        continue;
                    }
                    Err(_) => {
                        attempts += 1;
                        continue;
                    }
                }
            }
            match self.child.try_wait() {
                Ok(Some(status)) => {
                    return Err(BackendError {
                        status: ErrorStatus::KernelLaunch,
                        context: format!("tt-runtime exited unexpectedly during read (status {status})").into(),
                    });
                }
                Err(e) => {
                    return Err(BackendError {
                        status: ErrorStatus::KernelLaunch,
                        context: format!("tt-runtime wait error during read: {e}").into(),
                    });
                }
                Ok(None) => {
                    attempts += 1;
                }
            }
        }
        Err(BackendError {
            status: ErrorStatus::KernelLaunch,
            context: format!("tt-runtime read timeout after {}ms", timeout_ms).into(),
        })
    }

    fn alloc_buf(&mut self, size: u64, tile_bytes: u64) -> Result<u32, BackendError> {
        let cmd = format!(r#"{{"cmd":"alloc_buf","size":{size},"tile_bytes":{tile_bytes}}}"#);
        self.send(&cmd)?;
        let resp = self.recv_with_timeout(self.timeout_ms)?;
        if resp.contains("\"error\"") {
            let msg = extract_json_str(&resp, "msg").unwrap_or_else(|| "unknown".into());
            return Err(BackendError {
                status: ErrorStatus::MemoryAllocation,
                context: format!("alloc_buf error: {msg}").into(),
            });
        }
        let idx_str = extract_json_str(&resp, "index").ok_or_else(|| BackendError {
            status: ErrorStatus::MemoryAllocation,
            context: "alloc_buf: no index in response".into(),
        })?;
        let idx: u32 = idx_str.parse().map_err(|_| BackendError {
            status: ErrorStatus::MemoryAllocation,
            context: format!("alloc_buf: invalid index '{idx_str}'").into(),
        })?;
        Ok(idx)
    }

    fn free_buf(&mut self, dev_index: u32) -> Result<(), BackendError> {
        let cmd = format!(r#"{{"cmd":"free_buf","index":{dev_index}}}"#);
        self.send(&cmd)?;
        let resp = self.recv_with_timeout(self.timeout_ms)?;
        if resp.contains("\"error\"") {
            let msg = extract_json_str(&resp, "msg").unwrap_or_else(|| "unknown".into());
            return Err(BackendError { status: ErrorStatus::MemoryAllocation, context: format!("free_buf error: {msg}").into() });
        }
        Ok(())
    }

    fn write_buf(&mut self, dev_index: u32, shm_path: &str, size: u64) -> Result<(), BackendError> {
        let cmd = format!(r#"{{"cmd":"write_buf","index":{dev_index},"shm_path":"{shm_path}","size":{size}}}"#);
        self.send(&cmd)?;
        let resp = self.recv_with_timeout(self.timeout_ms)?;
        if resp.contains("\"error\"") {
            let msg = extract_json_str(&resp, "msg").unwrap_or_else(|| "unknown".into());
            return Err(BackendError { status: ErrorStatus::MemoryCopyH2P, context: format!("write_buf error: {msg}").into() });
        }
        Ok(())
    }

    fn read_buf(&mut self, dev_index: u32, shm_path: &str, size: u64) -> Result<(), BackendError> {
        let cmd = format!(r#"{{"cmd":"read_buf","index":{dev_index},"shm_path":"{shm_path}","size":{size}}}"#);
        self.send(&cmd)?;
        let resp = self.recv_with_timeout(self.timeout_ms)?;
        if resp.contains("\"error\"") {
            let msg = extract_json_str(&resp, "msg").unwrap_or_else(|| "unknown".into());
            return Err(BackendError { status: ErrorStatus::MemoryCopyP2H, context: format!("read_buf error: {msg}").into() });
        }
        Ok(())
    }

    fn compile_program(
        &mut self,
        id: u32,
        reader_source: &str,
        compute_source: &str,
        writer_source: &str,
        cb_config: &[(u32, u32, u32)],
    ) -> Result<(), BackendError> {
        let reader_source_len = reader_source.len();
        let compute_source_len = compute_source.len();
        let writer_source_len = writer_source.len();
        let n_cbs = cb_config.len();
        let mut cmd = format!(
            r#"{{"cmd":"compile_program","id":{id},"reader_source_len":{reader_source_len},"compute_source_len":{compute_source_len},"writer_source_len":{writer_source_len},"n_cbs":{n_cbs}"#
        );
        for (i, (idx, fmt, tb)) in cb_config.iter().enumerate() {
            cmd.push_str(&format!(r#","cb_idx{i}":{idx},"cb_fmt{i}":{fmt},"cb_tb{i}":{tb}"#));
        }
        cmd.push('}');
        self.send(&cmd)?;
        self.stdin.write_all(reader_source.as_bytes()).map_err(|e| BackendError {
            status: ErrorStatus::KernelCompilation,
            context: format!("tt-runtime write reader: {e}").into(),
        })?;
        self.stdin.write_all(compute_source.as_bytes()).map_err(|e| BackendError {
            status: ErrorStatus::KernelCompilation,
            context: format!("tt-runtime write compute: {e}").into(),
        })?;
        self.stdin.write_all(writer_source.as_bytes()).map_err(|e| BackendError {
            status: ErrorStatus::KernelCompilation,
            context: format!("tt-runtime write writer: {e}").into(),
        })?;
        self.stdin.flush().map_err(|e| BackendError {
            status: ErrorStatus::KernelCompilation,
            context: format!("tt-runtime flush: {e}").into(),
        })?;
        let resp = self.recv_with_timeout(self.timeout_ms)?;
        if resp.contains("\"error\"") {
            let msg = extract_json_str(&resp, "msg").unwrap_or_else(|| "unknown".into());
            return Err(BackendError {
                status: ErrorStatus::KernelCompilation,
                context: format!("tt-runtime compile error: {msg}").into(),
            });
        }
        Ok(())
    }

    fn run(&mut self, id: u32, src_indices: &[u32], dst_indices: &[u32], grid_dims: [u32; 2]) -> Result<(), BackendError> {
        let mut cmd = format!(r#"{{"cmd":"run","id":{id},"gd0":{gd0},"gd1":{gd1}"#, gd0 = grid_dims[0], gd1 = grid_dims[1]);
        for (i, idx) in src_indices.iter().enumerate() {
            cmd.push_str(&format!(r#","src{i}":{idx}"#));
        }
        for (i, idx) in dst_indices.iter().enumerate() {
            cmd.push_str(&format!(r#","dst{i}":{idx}"#));
        }
        cmd.push('}');
        self.send(&cmd)?;
        let resp = self.recv_with_timeout(self.timeout_ms)?;
        if resp.contains("\"error\"") {
            let msg = extract_json_str(&resp, "msg").unwrap_or_else(|| "unknown".into());
            return Err(BackendError {
                status: ErrorStatus::KernelLaunch,
                context: format!("tt-runtime run error: {msg}").into(),
            });
        }
        Ok(())
    }

    fn exit(&mut self) -> Result<(), BackendError> {
        self.send(r#"{"cmd":"exit"}"#)?;
        let resp = self.recv_with_timeout(self.timeout_ms)?;
        if resp.contains("\"error\"") {
            let msg = extract_json_str(&resp, "msg").unwrap_or_else(|| "unknown".into());
            return Err(BackendError {
                status: ErrorStatus::KernelLaunch,
                context: format!("tt-runtime exit error: {msg}").into(),
            });
        }
        self.child.wait().ok();
        Ok(())
    }
}

fn extract_json_str(json: &str, key: &str) -> Option<String> {
    let k = json.find(&format!("\"{key}\""))?;
    let after_colon = &json[k + key.len() + 3..]; // skip past "key":
    let start = after_colon.find('"')? + 1;
    let end = after_colon[start..].find('"')?;
    Some(after_colon[start..start + end].to_string())
}

// ---------------------------------------------------------------------------
// Compiled program tracking
// ---------------------------------------------------------------------------

#[derive(Debug)]
struct TTProgram {
    input_dtypes: Vec<DType>,
    output_dtypes: Vec<DType>,
    /// Grid dimensions for gidx0 (rows) and gidx1 (cols).
    /// Each dimension defaults to 1 if no corresponding gidx is used.
    grid_dims: [u32; 2],
}

// ---------------------------------------------------------------------------
// Device
// ---------------------------------------------------------------------------

#[derive(Debug)]
pub struct TTDevice {
    device_info: DeviceInfo,
    memory_pool_id: PoolId,
    runtime: Arc<Mutex<RuntimeProcess>>,
    programs: Slab<DeviceProgramId, TTProgram>,
}

impl TTDevice {
    pub fn deinitialize(&mut self) {}

    pub const fn info(&self) -> &DeviceInfo {
        &self.device_info
    }

    pub const fn memory_pool_id(&self) -> PoolId {
        self.memory_pool_id
    }

    pub const fn free_compute(&self) -> u128 {
        self.device_info.compute
    }

    #[allow(unused_must_use)]
    pub fn compile(&mut self, kernel: &Kernel, debug_asm: bool) -> Result<DeviceProgramId, BackendError> {
        let mut input_dtypes: Vec<DType> = Vec::new();
        let mut output_dtypes: Vec<DType> = Vec::new();

        let mut indent = String::new();

        // Generate reader kernel source
        let mut reader = String::new();
        writeln!(reader, "#include <cstdint>");
        writeln!(reader, "#include \"api/dataflow/dataflow_api.h\"");
        writeln!(reader, "#include \"api/dataflow/noc.h\"");
        writeln!(reader, "#include \"api/dataflow/circular_buffer.h\"");
        writeln!(reader, "#include \"api/tensor/noc_traits.h\"");
        writeln!(reader, "void kernel_main() {{");
        indent += "  ";
        writeln!(reader, "{indent}Noc noc;");
        let mut input_cb_map = Map::default();
        let mut output_cb_map = Map::default();

        {
            let mut max_cb = 0;
            let mut op_id = kernel.head;
            while !op_id.is_null() {
                if let Op::Store { dst, x, .. } = kernel.ops[op_id].op {
                    if let Op::Define { scope: Scope::Local, .. } = kernel.ops[dst].op {
                        if let Op::Load { src, .. } = kernel.ops[x].op {
                            if let Op::Define { ro: true, .. } = kernel.ops[src].op {
                                input_cb_map.insert(dst, max_cb);
                                max_cb += 1;
                            } else {
                                unreachable!()
                            }
                        } else {
                            output_cb_map.insert(dst, max_cb);
                            max_cb += 1;
                        }
                    }
                }
                op_id = kernel.next_op(op_id);
            }
        }

        let mut op_id = kernel.head;
        {
            const PAGE_SIZE: u32 = 4096;
            let mut input_arg_idx = 0u32;
            let mut loop_depth = 0u32;
            while !op_id.is_null() {
                match kernel.ops[op_id].op {
                    Op::Define { dtype, scope, ro, .. } => match scope {
                        Scope::Global => {
                            if ro {
                                input_dtypes.push(dtype);
                                writeln!(reader, "{indent}uint32_t src{op_id} = get_arg_val<uint32_t>({input_arg_idx});");
                                writeln!(
                                    reader,
                                    "{indent}auto args{op_id} = TensorAccessorArgs<{}>({input_arg_idx});",
                                    input_arg_idx * 2
                                );
                                writeln!(reader, "{indent}auto p{op_id} = TensorAccessor(args{op_id}, src{op_id}, {PAGE_SIZE});");
                                input_arg_idx += 1;
                            } else {
                                output_dtypes.push(dtype);
                            }
                        }
                        Scope::Local => {
                            if let Some(cb_id) = input_cb_map.get(&op_id) {
                                writeln!(reader, "{indent}CircularBuffer cb{cb_id}(tt::CBIndex::c_{cb_id});");
                            }
                        }
                        Scope::Register => todo!(),
                    },
                    Op::Load { .. } => {}
                    Op::Store { dst, x, index: st_idx, layout: st_layout } => {
                        let Op::Load { src, index: ld_idx, layout: ld_layout } = kernel.ops[x].op else {
                            panic!("tenstorrent supports only global to local loads in reader kernels with no ops inbetween")
                        };
                        let Op::Define { scope: Scope::Global, ro, .. } = kernel.ops[src].op else {
                            unreachable!()
                        };
                        if !ro {
                            continue;
                        }
                        let Op::Define { dtype, scope: Scope::Local, .. } = kernel.ops[dst].op else {
                            unreachable!()
                        };

                        let elem_size = dtype.bit_size() as u32 / 8;
                        if let Some(cb_id) = input_cb_map.get(&dst) {
                            match (ld_layout, st_layout) {
                                (MemLayout::Scalar, MemLayout::Scalar) => {
                                    if loop_depth == 0 {
                                        writeln!(reader, "{indent}cb{cb_id}.reserve_back(1);");
                                    }
                                    writeln!(
                                        reader,
                                        "{indent}noc.async_read(p{src}, cb{cb_id}, {elem_size},\n{indent}  {{ .page_id = (r{ld_idx}*{elem_size})/{PAGE_SIZE}, .offset_bytes = (r{ld_idx}*{elem_size})%{PAGE_SIZE} }},\n{indent}  {{ .offset_bytes = r{st_idx}*{elem_size} }});"
                                    );
                                }
                                _ => todo!(),
                            }
                        }
                    }
                    Op::Binary { x, y, bop } => {
                        let dt = kernel.dtype(op_id);
                        let _ = match bop {
                            BOp::Add => writeln!(reader, "{indent}{} r{op_id} = r{x} + r{y};", dt.c_type()),
                            BOp::Sub => writeln!(reader, "{indent}{} r{op_id} = r{x} - r{y};", dt.c_type()),
                            BOp::Mul => writeln!(reader, "{indent}{} r{op_id} = r{x} * r{y};", dt.c_type()),
                            BOp::Max => writeln!(reader, "{indent}{} r{op_id} = r{x} > r{y} ? r{x} : r{y};", dt.c_type()),
                            BOp::BitShiftLeft => writeln!(reader, "{indent}{} r{op_id} = r{x} << r{y};", dt.c_type()),
                            _ => unreachable!("{bop:?}"),
                        };
                    }
                    Op::Loop { len } => {
                        if loop_depth == 0 {
                            // Reserve CB space before the loop — enough for one tile per CB
                            for cb_id in input_cb_map.values() {
                                writeln!(reader, "{indent}cb{cb_id}.reserve_back(1);");
                            }
                        }
                        writeln!(reader, "{indent}for (uint32_t r{op_id} = 0; r{op_id} < r{len}; r{op_id}++) {{");
                        indent += "  ";
                        loop_depth += 1;
                    }
                    Op::EndLoop => {
                        indent.pop();
                        indent.pop();
                        writeln!(reader, "{indent}}}");
                        loop_depth -= 1;
                    }
                    Op::Const(val) => {
                        writeln!(reader, "{indent}{} r{op_id} = {};", val.dtype().c_type(), val.c_code());
                    }
                    Op::GroupIndex { axis, .. } => {
                        writeln!(
                            reader,
                            "{indent}uint32_t r{op_id} = get_arg_val<uint32_t>({});",
                            input_dtypes.len() + axis as usize
                        );
                    }
                    // Barrier means reader kernel is over
                    Op::Barrier => break,
                    ref op => unreachable!("{op:?}"),
                }
                op_id = kernel.next_op(op_id);
            }
            writeln!(reader, "{indent}noc.async_read_barrier();");
            for cb_id in input_cb_map.values() {
                writeln!(reader, "{indent}cb{cb_id}.push_back(1);");
            }
            writeln!(reader, "}}");
        }
        // Advance past the barrier into the compute section
        op_id = kernel.next_op(op_id);

        if debug_asm {
            println!("[tenstorrent] reader:\n{reader}");
        }

        // Generate compute kernel source
        let mut compute = String::new();
        writeln!(compute, "#include <cstdint>");
        writeln!(compute, "#include \"api/compute/common.h\"");
        writeln!(compute, "#include \"api/compute/compute_kernel_api.h\"");
        writeln!(compute, "#include \"api/compute/eltwise_binary_sfpu.h\"");
        writeln!(compute, "#include \"api/compute/tile_move_copy.h\"");
        writeln!(compute, "#include \"api/compute/eltwise_unary/eltwise_unary.h\"");
        writeln!(compute, "#include \"api/compute/eltwise_unary/trigonometry.h\"");
        writeln!(compute, "#include \"api/dataflow/circular_buffer.h\"");
        writeln!(compute, "void kernel_main() {{");
        let mut indent = String::from("  ");
        {
            let mut cb_ids: Vec<u32> = input_cb_map.values().copied().collect();
            for cb_id in output_cb_map.values() {
                if !cb_ids.contains(cb_id) {
                    cb_ids.push(*cb_id);
                }
            }
            cb_ids.sort();
            for cb_id in &cb_ids {
                writeln!(compute, "{indent}CircularBuffer cb{cb_id}(tt::CBIndex::c_{cb_id});");
            }

            let input_ids: Vec<u32> = input_cb_map.values().copied().collect();
            let output_ids: Vec<u32> = output_cb_map.values().copied().collect();
            if !input_ids.is_empty() && !output_ids.is_empty() {
                let in0 = input_ids[0];
                let _in1 = input_ids.get(1).copied().unwrap_or(in0);
                let out0 = output_ids[0];
                writeln!(compute, "{indent}init_sfpu({in0}, {out0});");
            }

            // Emit init headers for ops we might encounter
            let mut has_sin = false;
            let mut has_binary = false;
            let (_dtypes, rcs) = kernel.compute_dtypes_and_rcs();
            let mut dst_slots: Map<OpId, Vec<u32>> = Map::default();
            let mut consumer_count: Map<OpId, u32> = Map::default();
            let mut next_slot = 0u32;
            let mut output_stores: Vec<(u32, u32)> = Vec::new();

            // First pass: collect init headers from ops
            let mut scan = op_id;
            while !scan.is_null() {
                match kernel.ops[scan].op {
                    Op::Cast { .. } => {}
                    Op::Unary { uop: UOp::Sin, .. } => has_sin = true,
                    Op::Binary { bop: BOp::Add, .. } => has_binary = true,
                    Op::Barrier => break,
                    _ => {}
                }
                scan = kernel.next_op(scan);
            }

            // Emit init calls based on scanned ops
            if has_sin {
                writeln!(compute, "{indent}sin_tile_init();");
            }
            if has_binary {
                writeln!(compute, "{indent}add_binary_tile_init();");
            }

            // Collect all unique input CBs from Load ops
            let mut load_input_cbs: Vec<u32> = Vec::new();
            let mut pre_scan = op_id;
            while !pre_scan.is_null() {
                match kernel.ops[pre_scan].op {
                    Op::Load { src, layout: MemLayout::Tile { .. }, .. } => {
                        if let Some(&cb_id) = input_cb_map.get(&src) {
                            if !load_input_cbs.contains(&cb_id) {
                                load_input_cbs.push(cb_id);
                            }
                        }
                    }
                    Op::Barrier => break,
                    _ => {}
                }
                pre_scan = kernel.next_op(pre_scan);
            }
            for &cb_id in &load_input_cbs {
                writeln!(compute, "{indent}cb{cb_id}.wait_front(1);");
            }
            writeln!(compute, "{indent}tile_regs_acquire();");

            while !op_id.is_null() {
                match kernel.ops[op_id].op {
                    Op::Load { src, index: _, layout: MemLayout::Tile { .. } } => {
                        if let Some(&cb_id) = input_cb_map.get(&src) {
                            let n = rcs.get(&op_id).copied().unwrap_or(1).max(1) as usize;
                            let mut slots = Vec::with_capacity(n);
                            for _ in 0..n {
                                let slot = next_slot;
                                next_slot += 1;
                                slots.push(slot);
                                writeln!(compute, "{indent}copy_tile({cb_id}, 0, {slot});");
                            }
                            dst_slots.insert(op_id, slots);
                        }
                    }
                    Op::Cast { x, dtype: DType::BF16 | DType::F16 | DType::F32 } => {
                        let idx = consumer_count.entry(x).or_insert(0);
                        let slot = dst_slots[&x][*idx as usize];
                        *idx += 1;
                        let n = rcs.get(&op_id).copied().unwrap_or(1).max(1) as usize;
                        dst_slots.insert(op_id, vec![slot; n]);
                    }
                    Op::Unary { x, uop: UOp::Sin } => {
                        let idx = consumer_count.entry(x).or_insert(0);
                        let slot = dst_slots[&x][*idx as usize];
                        *idx += 1;
                        let n = rcs.get(&op_id).copied().unwrap_or(1).max(1) as usize;
                        dst_slots.insert(op_id, vec![slot; n]);
                        writeln!(compute, "{indent}sin_tile({slot});");
                    }
                    Op::Binary { x, y, bop: BOp::Add } => {
                        let x_idx = consumer_count.entry(x).or_insert(0);
                        let slot_x = dst_slots[&x][*x_idx as usize];
                        *x_idx += 1;
                        let y_idx = consumer_count.entry(y).or_insert(0);
                        let slot_y = dst_slots[&y][*y_idx as usize];
                        *y_idx += 1;
                        let n = rcs.get(&op_id).copied().unwrap_or(1).max(1) as usize;
                        dst_slots.insert(op_id, vec![slot_x; n]);
                        writeln!(compute, "{indent}add_binary_tile({slot_x}, {slot_y}, {slot_x});");
                    }
                    Op::Store { dst, x, index: _, layout: MemLayout::Tile { .. } } => {
                        if let Some(&cb_id) = output_cb_map.get(&dst) {
                            let idx = consumer_count.entry(x).or_insert(0);
                            let slot = dst_slots[&x][*idx as usize];
                            *idx += 1;
                            output_stores.push((slot, cb_id));
                        }
                    }
                    Op::Barrier => break,
                    _ => {}
                }
                op_id = kernel.next_op(op_id);
            }

            writeln!(compute, "{indent}tile_regs_commit();");
            writeln!(compute, "{indent}tile_regs_wait();");
            for &(slot, cb_id) in &output_stores {
                writeln!(compute, "{indent}cb{cb_id}.reserve_back(1);");
                writeln!(compute, "{indent}pack_tile({slot}, {cb_id});");
            }
            for &loaded_cb in &load_input_cbs {
                writeln!(compute, "{indent}cb{loaded_cb}.pop_front(1);");
            }
            writeln!(compute, "{indent}tile_regs_release();");
            for &(_, cb_id) in &output_stores {
                writeln!(compute, "{indent}cb{cb_id}.push_back(1);");
            }
            writeln!(compute, "}}");
        }

        if debug_asm {
            println!("[tenstorrent] compute:\n{compute}");
        }

        // Generate writer kernel source
        let mut writer = String::new();
        // Advance past the second barrier into the writer section
        op_id = kernel.next_op(op_id);

        const PAGE_SIZE: u32 = 4096;
        writeln!(writer, "#include <cstdint>");
        writeln!(writer, "#include \"api/dataflow/dataflow_api.h\"");
        writeln!(writer, "#include \"api/dataflow/noc.h\"");
        writeln!(writer, "#include \"api/dataflow/circular_buffer.h\"");
        writeln!(writer, "#include \"api/tensor/noc_traits.h\"");
        writeln!(writer, "void kernel_main() {{");
        writeln!(writer, "{indent}Noc noc(1);");

        for cb_id in output_cb_map.values() {
            writeln!(writer, "{indent}CircularBuffer cb{cb_id}(tt::CBIndex::c_{cb_id});");
        }

        // Emit TensorAccessor for each output global (ro=false) in IR order
        let mut out_global_count = 0u32;
        {
            let mut scan = kernel.head;
            while !scan.is_null() {
                if let Op::Define { scope: Scope::Global, ro: false, .. } = kernel.ops[scan].op {
                    writeln!(writer, "{indent}uint32_t out{scan} = get_arg_val<uint32_t>({out_global_count});");
                    writeln!(
                        writer,
                        "{indent}auto args_out{scan} = TensorAccessorArgs<{}>({out_global_count});",
                        out_global_count * 2
                    );
                    writeln!(writer, "{indent}auto p_out{scan} = TensorAccessor(args_out{scan}, out{scan}, {PAGE_SIZE});");
                    out_global_count += 1;
                }
                scan = kernel.next_op(scan);
            }
        }

        let mut writer_loop_cbs: Vec<u32> = output_cb_map.values().copied().collect();
        writer_loop_cbs.sort();
        // scan the writer section to collect output CBs used inside the loop
        {
            let mut scan = op_id;
            let mut depth = 0u32;
            let mut in_loop_cbs: Vec<u32> = Vec::new();
            while !scan.is_null() {
                match kernel.ops[scan].op {
                    Op::Loop { .. } => depth += 1,
                    Op::EndLoop => depth -= 1,
                    Op::Store { x, .. } if depth > 0 => {
                        if let Op::Load { src, .. } = kernel.ops[x].op {
                            if let Some(&cb_id) = output_cb_map.get(&src) {
                                if !in_loop_cbs.contains(&cb_id) {
                                    in_loop_cbs.push(cb_id);
                                }
                            }
                        }
                    }
                    Op::Barrier if depth == 0 => break,
                    _ => {}
                }
                scan = kernel.next_op(scan);
            }
            if !in_loop_cbs.is_empty() {
                writer_loop_cbs = in_loop_cbs;
                writer_loop_cbs.sort();
            }
        }

        // Pre-scan: emit any Index/Const ops referenced by writer-section ops
        // but located before the barriers (moved there by move_constants_to_beginning)
        {
            let mut emitted: Vec<OpId> = Vec::new();
            let mut scan = op_id;
            while !scan.is_null() {
                if let Op::Barrier = kernel.ops[scan].op {
                    break;
                }
                let mut work: Vec<OpId> = kernel.ops[scan].op.parameters().collect();
                while let Some(param) = work.pop() {
                    if emitted.contains(&param) {
                        continue;
                    }
                    emitted.push(param);
                    match &kernel.ops[param].op {
                        Op::GroupIndex { axis, .. } => {
                            writeln!(
                                writer,
                                "{indent}uint32_t r{param} = get_arg_val<uint32_t>({});",
                                output_dtypes.len() + *axis as usize
                            );
                        }
                        Op::Const(val) => {
                            writeln!(writer, "{indent}{} r{param} = {};", val.dtype().c_type(), val.c_code());
                        }
                        _ => {}
                    }
                    // Walk parameters of this dependency too (e.g. Binary referencing Const)
                    work.extend(kernel.ops[param].op.parameters());
                }
                scan = kernel.next_op(scan);
            }
        }

        let mut loop_depth = 0u32;
        while !op_id.is_null() {
            match kernel.ops[op_id].op {
                Op::Store { dst, x, index: st_idx, layout } => {
                    if layout != MemLayout::Scalar {
                        todo!("add support for non-scalar stores back to DRAM")
                    }
                    // If storing a Load-from-local value to global → writer CB→DRAM
                    if let Op::Load { src, .. } = kernel.ops[x].op {
                        if let Some(&cb_id) = output_cb_map.get(&src) {
                            let Op::Define { dtype, .. } = kernel.ops[dst].op else {
                                unreachable!()
                            };
                            let elem_size = dtype.bit_size() as u32 / 8;
                            if loop_depth == 0 {
                                writeln!(writer, "{indent}cb{cb_id}.wait_front(1);");
                            }
                            writeln!(
                                writer,
                                "{indent}noc.async_write(use<CircularBuffer::AddrSelector::READ_PTR>(cb{cb_id}),\n{indent}  p_out{dst}, {elem_size}, {{ .offset_bytes = r{st_idx}*{elem_size} }},\n{indent}  {{ .page_id = (r{st_idx}*{elem_size})/{PAGE_SIZE}, .offset_bytes = (r{st_idx}*{elem_size})%{PAGE_SIZE} }});"
                            );
                            if loop_depth == 0 {
                                writeln!(writer, "{indent}cb{cb_id}.pop_front(1);");
                            }
                        }
                    }
                    // If storing a compute result to local → compute writing to output CB,
                    // handled by compute kernel, skip in writer.
                }
                Op::Load { .. } => {
                    // Load from CB in writer section — handled implicitly by the Store that consumes it
                }
                Op::Const(val) => {
                    writeln!(writer, "{indent}{} r{op_id} = {};", val.dtype().c_type(), val.c_code());
                }
                Op::GroupIndex { axis, .. } => {
                    writeln!(
                        writer,
                        "{indent}uint32_t r{op_id} = get_arg_val<uint32_t>({});",
                        output_dtypes.len() + axis as usize
                    );
                }
                Op::Binary { x, y, bop } => {
                    let dt = kernel.dtype(op_id);
                    let _ = match bop {
                        BOp::Add => writeln!(writer, "{indent}{} r{op_id} = r{x} + r{y};", dt.c_type()),
                        BOp::Sub => writeln!(writer, "{indent}{} r{op_id} = r{x} - r{y};", dt.c_type()),
                        BOp::Mul => writeln!(writer, "{indent}{} r{op_id} = r{x} * r{y};", dt.c_type()),
                        BOp::Max => writeln!(writer, "{indent}{} r{op_id} = r{x} > r{y} ? r{x} : r{y};", dt.c_type()),
                        BOp::BitShiftLeft => writeln!(writer, "{indent}{} r{op_id} = r{x} << r{y};", dt.c_type()),
                        _ => unreachable!("{bop:?}"),
                    };
                }
                Op::Loop { len } => {
                    if loop_depth == 0 {
                        for cb_id in &writer_loop_cbs {
                            writeln!(writer, "{indent}cb{cb_id}.wait_front(1);");
                        }
                    }
                    writeln!(writer, "{indent}for (uint32_t r{op_id} = 0; r{op_id} < r{len}; r{op_id}++) {{");
                    indent += "  ";
                    loop_depth += 1;
                }
                Op::EndLoop => {
                    indent.pop();
                    indent.pop();
                    writeln!(writer, "{indent}}}");
                    if loop_depth == 1 {
                        writeln!(writer, "{indent}noc.async_write_barrier();");
                        for cb_id in &writer_loop_cbs {
                            writeln!(writer, "{indent}cb{cb_id}.pop_front(1);");
                        }
                    }
                    loop_depth -= 1;
                }
                Op::Barrier => break,
                _ => {}
            }
            op_id = kernel.next_op(op_id);
        }
        writeln!(writer, "}}");

        if debug_asm {
            println!("[tenstorrent] writer:\n{writer}");
        }

        // Scan kernel for global index dimensions
        let mut grid_dims = [1u32, 1u32];
        {
            let mut scan = kernel.head;
            while !scan.is_null() {
                if let Op::GroupIndex { len, axis } = &kernel.ops[scan].op {
                    if (*axis as usize) < 2 {
                        grid_dims[*axis as usize] = *len as u32;
                    }
                }
                scan = kernel.next_op(scan);
            }
        }

        let prog_id =
            self.programs.push(TTProgram { input_dtypes: input_dtypes.clone(), output_dtypes: output_dtypes.clone(), grid_dims });

        {
            let mut cb_config = Vec::with_capacity(input_cb_map.len() + output_cb_map.len());
            let dtype_to_tt_fmt = |dt: DType| -> u32 {
                match dt {
                    DType::F32 => 0,
                    DType::F16 => 1,
                    DType::BF16 => 2,
                    _ => 0,
                }
            };
            let tile_bytes_of = |dt: DType| -> u32 {
                let te = 1024u64;
                (match dt {
                    DType::F32 => 4 * te,
                    DType::F16 | DType::BF16 => 2 * te,
                    _ => 4 * te,
                }) as u32
            };

            let mut cb_ids: Vec<u32> = input_cb_map.values().copied().collect();
            for cb_id in output_cb_map.values() {
                if !cb_ids.contains(cb_id) {
                    cb_ids.push(*cb_id);
                }
            }
            cb_ids.sort();
            for cb_id in &cb_ids {
                // Find the local define for this CB to get its dtype
                let local_op = input_cb_map
                    .iter()
                    .find(|(_, v)| *v == cb_id)
                    .or_else(|| output_cb_map.iter().find(|(_, v)| *v == cb_id))
                    .map(|(op, _)| *op);
                let dt = local_op
                    .and_then(|op| {
                        if let Op::Define { dtype, .. } = &kernel.ops[op].op {
                            Some(*dtype)
                        } else {
                            None
                        }
                    })
                    .unwrap_or(DType::BF16);
                let fmt = dtype_to_tt_fmt(dt);
                let tb = tile_bytes_of(dt);
                cb_config.push((*cb_id, fmt, tb));
            }

            let mut rt_guard = self.runtime.lock().unwrap();
            rt_guard.compile_program(prog_id.0, &reader, &compute, &writer, &cb_config)?;
        }

        Ok(prog_id)
    }

    pub fn release(&mut self, program_id: DeviceProgramId) {
        if self.programs.contains_key(program_id) {
            unsafe { self.programs.remove_and_return(program_id) };
        }
    }

    pub fn launch(
        &mut self,
        program_id: DeviceProgramId,
        memory_pool: &mut TTMemoryPool,
        args: &[PoolBufferId],
        event_wait_list: Vec<Event>,
    ) -> Result<Event, BackendError> {
        let _ = event_wait_list;
        let prog = if self.programs.contains_key(program_id) {
            &self.programs[program_id]
        } else {
            return Err(BackendError { status: ErrorStatus::KernelLaunch, context: "invalid program id".into() });
        };

        let rt = &self.runtime;

        let n_inputs = prog.input_dtypes.len();
        let n_outputs = prog.output_dtypes.len();

        if args.len() < n_inputs + n_outputs {
            return Err(BackendError {
                status: ErrorStatus::KernelLaunch,
                context: format!("expected {} buffers, got {}", n_inputs + n_outputs, args.len()).into(),
            });
        }

        let mut src_indices: Vec<u32> = Vec::with_capacity(n_inputs);
        for i in 0..n_inputs {
            let idx = memory_pool.dev_index(args[i]).map_err(|e| BackendError {
                status: ErrorStatus::KernelLaunch,
                context: format!("src{i} dev_index: {e}").into(),
            })?;
            src_indices.push(idx);
        }
        let mut dst_indices: Vec<u32> = Vec::with_capacity(n_outputs);
        for i in 0..n_outputs {
            let idx = memory_pool.dev_index(args[n_inputs + i]).map_err(|e| BackendError {
                status: ErrorStatus::KernelLaunch,
                context: format!("dst{i} dev_index: {e}").into(),
            })?;
            dst_indices.push(idx);
        }

        let mut rt_guard = rt.lock().unwrap();
        rt_guard.run(program_id.0, &src_indices, &dst_indices, prog.grid_dims)?;

        Ok(Event::TT(TTEvent))
    }
}
