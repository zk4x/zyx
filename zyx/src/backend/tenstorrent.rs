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

use super::{Device, DeviceId, DeviceInfo, DeviceProgramId, Event, GwsDim, Kernel, LaunchArg, MemoryPool, PoolBufferId, PoolId};
use crate::{
    DType, Map, Set,
    backend::DTypeCapability,
    error::{BackendError, ErrorStatus},
    kernel::{MemScope, Op, OpId, ParamKind, RangeKind},
    shape::Dim,
    slab::Slab,
};
use nanoserde::DeJson;
use std::{
    ffi::CString,
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
pub(crate) struct TTBuffer {
    dev_index: u32,
    pub(crate) size: u64,
}

// ---------------------------------------------------------------------------
// Memory pool — device DRAM buffers managed by C++ runtime.
// TTBuffer is a handle (u32 dev_index) into the runtime's buffer list.
// The pool shares the runtime IPC channel with TTDevice via Arc<Mutex>.
// ---------------------------------------------------------------------------

#[derive(Debug)]
pub struct TTMemoryPool {
    pub(crate) buffers: Slab<PoolBufferId, TTBuffer>,
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
    let pool =
        MemoryPool::TT(TTMemoryPool { buffers: Slab::new(), runtime: runtime.clone(), free_bytes: Dim::from(dram_bytes as i64) });
    memory_pools.push(pool);

    let _device_id = devices.len();
    let dev_id = config.device_ids.as_ref().and_then(|ids| ids.first().copied()).unwrap_or(0);
    devices.push(Device::TT(TTDevice {
        dev_id: u32::try_from(dev_id).unwrap(),
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
            tenstorrent: true,
            tile: [32, 32],
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
        if self.buffers.contains_id(buffer_id) {
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
        eprintln!("[TT-MARK] pool_to_pool start");
        match src_pool {
            MemoryPool::Host(host_pool) => {
                eprintln!("[TT-MARK] pool_to_pool src=host");
                let data = host_pool.get_buffer(src);
                self.host_to_pool(data, dst, event_wait_list)
            }
            // No P2P path in the tt-runtime shim yet — stage through host.
            _ => {
                eprintln!("[TT-MARK] pool_to_pool src=other-pool, staging via host");
                let len = {
                    let dst_buf = self.buffers.get(dst).ok_or_else(|| BackendError {
                        status: ErrorStatus::MemoryCopyP2H,
                        context: "invalid dst buffer id".into(),
                    })?;
                    dst_buf.size as usize
                };
                let mut staging = vec![0u8; len];
                src_pool.pool_to_host(src, &mut staging, event_wait_list)?;
                self.host_to_pool(&staging, dst, Vec::new())
            }
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
        if self.buffers.contains_id(buffer_id) {
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
        n_params: u32,
        reader_params: &[u32],
        compute_params: &[u32],
        writer_params: &[u32],
    ) -> Result<(), BackendError> {
        let reader_source_len = reader_source.len();
        let compute_source_len = compute_source.len();
        let writer_source_len = writer_source.len();
        let n_cbs = cb_config.len();
        let mut cmd = format!(
            r#"{{"cmd":"compile_program","id":{id},"reader_source_len":{reader_source_len},"compute_source_len":{compute_source_len},"writer_source_len":{writer_source_len},"n_cbs":{n_cbs},"n_params":{n_params},"n_reader_params":{},"n_compute_params":{},"n_writer_params":{}"#,
            reader_params.len(),
            compute_params.len(),
            writer_params.len()
        );
        for (i, p) in reader_params.iter().enumerate() {
            cmd.push_str(&format!(r#","rp{i}":{p}"#));
        }
        for (i, p) in compute_params.iter().enumerate() {
            cmd.push_str(&format!(r#","cp{i}":{p}"#));
        }
        for (i, p) in writer_params.iter().enumerate() {
            cmd.push_str(&format!(r#","wp{i}":{p}"#));
        }
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

    fn run(
        &mut self,
        id: u32,
        src_indices: &[u32],
        dst_indices: &[u32],
        grid_dims: [u32; 2],
        vars: &[(u32, u32)],
    ) -> Result<(), BackendError> {
        let mut cmd = format!(
            r#"{{"cmd":"run","id":{id},"gd0":{gd0},"gd1":{gd1},"n_vars":{}"#,
            vars.len(),
            gd0 = grid_dims[0],
            gd1 = grid_dims[1]
        );
        for (i, idx) in src_indices.iter().enumerate() {
            cmd.push_str(&format!(r#","src{i}":{idx}"#));
        }
        for (i, idx) in dst_indices.iter().enumerate() {
            cmd.push_str(&format!(r#","dst{i}":{idx}"#));
        }
        for (i, (ordinal, value)) in vars.iter().enumerate() {
            cmd.push_str(&format!(r#","vord{i}":{ordinal},"vval{i}":{value}"#));
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
    /// Group-range lengths in axis order (gws): Const resolved at compile,
    /// Param(ordinal) resolved from the launch args.
    gws: Vec<GwsDim>,
}

// ---------------------------------------------------------------------------
// Device
// ---------------------------------------------------------------------------

#[derive(Debug)]
pub struct TTDevice {
    device_info: DeviceInfo,
    /// Real Tenstorrent chip id (from device_ids config), set at init. Not the slab index.
    pub(crate) dev_id: u32,
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
        // Build CB maps and dtypes from the kernel.
        //
        // Sections are delimited by Barriers: reader (head -> 1st barrier),
        // compute (1st -> 2nd), writer (2nd -> end). Circular storages are
        // plain rw L1 SRAM with ONE id each: a CB touched by any section is
        // registered once, in head order, and every section uses that id.
        let mut cb_map: Map<OpId, u32> = Map::default();
        let mut input_dtypes: Vec<DType> = Vec::new();
        let mut output_dtypes: Vec<DType> = Vec::new();
        {
            let mut max_cb = 0u32;
            let mut section = 0u32;
            let mut scan = kernel.head;
            let mut steps_scan = 0usize;
            while !scan.is_null() {
                steps_scan += 1;
                if steps_scan > 10_000 {
                    panic!("compile did not finish in 10000 steps");
                }
                match &kernel.ops[scan].op {
                    Op::Barrier => section += 1,
                    Op::Param { dtype, kind: ParamKind::Global, .. } => input_dtypes.push(*dtype),
                    Op::Param { dtype, kind: ParamKind::GlobalMut, .. } => output_dtypes.push(*dtype),
                    Op::Load { src, .. } => {
                        if let Op::Storage { scope: MemScope::Circular, .. } = kernel.ops[*src].op {
                            // Reader loads and writer loads register; compute
                            // loads use CBs already registered by whoever
                            // fills/drains them.
                            if section != 1 && !cb_map.contains_key(src) {
                                cb_map.insert(*src, max_cb);
                                max_cb += 1;
                            }
                        }
                    }
                    Op::Store { dst, .. } => {
                        match &kernel.ops[*dst].op {
                            Op::Storage { scope: MemScope::Circular, .. } => {
                                if section == 0 && !cb_map.contains_key(dst) {
                                    cb_map.insert(*dst, max_cb);
                                    max_cb += 1;
                                }
                                // compute/writer CB stores need no registration:
                                // the loads register CBs.
                            }
                            Op::Storage { scope, .. } => {
                                todo!("store into non-CB storage scope {scope:?}")
                            }
                            _ => {}
                        }
                    }
                    _ => {}
                }
                scan = kernel.next_op(scan);
            }
        }

        // ---- Per-section param requirements ----
        // Sections are delimited by Barriers: reader (head -> 1st barrier),
        // compute (1st -> 2nd), writer (2nd -> end). Each section needs the
        // params in the transitive closure of its stores' scalar deps.
        let n_params = {
            let mut n = 0u32;
            let mut scan = kernel.head;
            let mut steps = 0usize;
            while !scan.is_null() {
                steps += 1;
                if steps > 10_000 {
                    panic!("tt param scan did not finish in 10000 steps");
                }
                if matches!(kernel.ops[scan].op, Op::Param { .. }) {
                    n += 1;
                }
                scan = kernel.next_op(scan);
            }
            n
        };
        let param_ordinal_of: Map<OpId, u32> = {
            let mut map = Map::default();
            let mut idx = 0u32;
            let mut scan = kernel.head;
            let mut steps = 0usize;
            while !scan.is_null() {
                steps += 1;
                if steps > 10_000 {
                    panic!("tt param ordinal scan did not finish in 10000 steps");
                }
                if matches!(kernel.ops[scan].op, Op::Param { .. }) {
                    map.insert(scan, idx);
                    idx += 1;
                }
                scan = kernel.next_op(scan);
            }
            map
        };
        // Stores per section (0 = reader, 1 = compute, 2 = writer).
        let mut section_stores: [Vec<OpId>; 3] = [Vec::new(), Vec::new(), Vec::new()];
        let mut gws_lens: Vec<OpId> = Vec::new();
        {
            let mut section = 0usize;
            let mut scan = kernel.head;
            let mut steps = 0usize;
            while !scan.is_null() {
                steps += 1;
                if steps > 10_000 {
                    panic!("tt section scan did not finish in 10000 steps");
                }
                match kernel.ops[scan].op {
                    Op::Barrier => section += 1,
                    Op::Store { .. } => section_stores[section].push(scan),
                    Op::Range { kind: RangeKind::Group(len), .. } => gws_lens.push(len),
                    _ => {}
                }
                scan = kernel.next_op(scan);
            }
        }
        // Per-section params (0 = reader, 1 = compute, 2 = writer): the
        // ordinals of the params each section's stores depend on, in
        // ascending head order. These lists define the sections' runtime
        // args: each section gets exactly its own params — its Global +
        // Variable params interleaved in head order first, then its
        // GlobalMut params — followed by the core's tensix-grid coordinates
        // gidx0 (row) / gidx1 (col). GlobalMut occupies the tail of the
        // head-order param list, so the ascending sort already yields the
        // Global|Variable-then-GlobalMut layout; see
        // `Kernel::generate_tenstorrent` and `tt_runtime.cpp`
        // `section_rt_args` for the consumption side.
        let per_section_params: Vec<Vec<u32>> = section_stores
            .iter()
            .map(|stores| {
                let mut deps: Set<OpId> = Set::default();
                let mut stack: Vec<OpId> = stores.iter().copied().collect();
                while let Some(id) = stack.pop() {
                    if !deps.insert(id) {
                        continue;
                    }
                    stack.extend(kernel.ops[id].op.parameters());
                }
                let mut params: Vec<u32> = deps
                    .iter()
                    .filter(|id| matches!(kernel.ops[**id].op, Op::Param { .. }))
                    .filter_map(|id| param_ordinal_of.get(id).copied())
                    .collect();
                params.sort_unstable();
                params
            })
            .collect();
        let [reader_params, compute_params, writer_params] = per_section_params.try_into().expect("3 sections");

        // Group-range lengths in axis order -> GwsDim (Const resolved now,
        // Param resolved at launch from the Variable arg).
        let gws: Vec<GwsDim> = gws_lens
            .iter()
            .map(|&len| match &kernel.ops[len].op {
                Op::Const(c) => GwsDim::Const(c.as_dim().expect("gws const length has a concrete dim")),
                Op::Param { kind: ParamKind::Variable, .. } => GwsDim::Param(param_ordinal_of[&len] as usize),
                op => todo!("tenstorrent gws: group length must be Const or Param Variable, got {op:?}"),
            })
            .collect();

        let (reader, compute, writer) = kernel.generate_tenstorrent(
            debug_asm,
            &cb_map,
            &reader_params,
            &compute_params,
            &writer_params,
        )?;

        let prog_id = self.programs.push(TTProgram { input_dtypes, output_dtypes, gws });

        {
            let mut cb_config = Vec::with_capacity(cb_map.len());
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

            let mut cb_ids: Vec<u32> = cb_map.values().copied().collect();
            cb_ids.sort();
            for cb_id in &cb_ids {
                // Find the local define for this CB to get its dtype
                let local_op = cb_map.iter().find(|(_, v)| *v == cb_id).map(|(op, _)| *op);
                let dt = local_op
                    .and_then(|op| {
                        if let Op::Storage { dtype, .. } = &kernel.ops[op].op {
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
            rt_guard.compile_program(
                prog_id.0,
                &reader,
                &compute,
                &writer,
                &cb_config,
                n_params,
                &reader_params,
                &compute_params,
                &writer_params,
            )?;
        }

        Ok(prog_id)
    }

    pub fn release(&mut self, program_id: DeviceProgramId) {
        if self.programs.contains_id(program_id) {
            unsafe { self.programs.remove_and_return(program_id) };
        }
    }

    pub fn launch(
        &mut self,
        program_id: DeviceProgramId,
        memory_pool: &mut TTMemoryPool,
        args: &[LaunchArg],
        event_wait_list: Vec<Event>,
    ) -> Result<Event, BackendError> {
        let _ = event_wait_list;
        let prog = if self.programs.contains_id(program_id) {
            &self.programs[program_id]
        } else {
            return Err(BackendError { status: ErrorStatus::KernelLaunch, context: "invalid program id".into() });
        };

        let rt = &self.runtime;

        let n_inputs = prog.input_dtypes.len();
        let n_outputs = prog.output_dtypes.len();

        // One arg per param, head order: Global + Variable interleaved,
        // GlobalMut at the tail. Kinds are derivable from the args themselves:
        // Variable -> Variable; Buffer above the GlobalMut tail -> Global.
        if args.len() < n_inputs + n_outputs {
            return Err(BackendError {
                status: ErrorStatus::KernelLaunch,
                context: format!(
                    "expected at least {} args ({} inputs + {} outputs), got {}",
                    n_inputs + n_outputs,
                    n_inputs,
                    n_outputs,
                    args.len()
                )
                .into(),
            });
        }
        let n_params = args.len();
        debug_assert!(n_params >= n_inputs + n_outputs, "tt launch: {n_params} args for {n_inputs} inputs + {n_outputs} outputs");

        let globalmut_start = n_params - n_outputs;
        let mut src_indices: Vec<u32> = Vec::with_capacity(n_inputs);
        let mut dst_indices: Vec<u32> = Vec::with_capacity(n_outputs);
        // Variable params: (ordinal, value) pairs.
        let mut vars: Vec<(u32, u32)> = Vec::new();
        for (ordinal, arg) in args.iter().enumerate() {
            let ordinal = ordinal as u32;
            match arg {
                LaunchArg::Buffer(buffer_id) => {
                    let idx = memory_pool.dev_index(*buffer_id).map_err(|e| BackendError {
                        status: ErrorStatus::KernelLaunch,
                        context: format!("param {ordinal} dev_index: {e}").into(),
                    })?;
                    if ordinal as usize >= globalmut_start {
                        dst_indices.push(idx);
                    } else {
                        src_indices.push(idx);
                    }
                }
                LaunchArg::Variable(value) => {
                    debug_assert!(
                        (ordinal as usize) < globalmut_start,
                        "tt launch: Variable arg at ordinal {ordinal} inside GlobalMut tail"
                    );
                    let dim = value.as_dim().expect("variable launch arg has a concrete dim");
                    let v = u32::try_from(dim).map_err(|_| BackendError {
                        status: ErrorStatus::KernelLaunch,
                        context: format!("param {ordinal} variable value {dim} does not fit u32").into(),
                    })?;
                    vars.push((ordinal, v));
                }
            }
        }
        debug_assert_eq!(src_indices.len(), n_inputs, "tt launch: {} src args for {} Global params", src_indices.len(), n_inputs);
        debug_assert_eq!(dst_indices.len(), n_outputs, "tt launch: {} dst args for {} outputs", dst_indices.len(), n_outputs);

        // Grid dims from the group-range lengths (gws), in axis order.
        let mut grid_dims = [1u32, 1u32];
        if prog.gws.len() > 2 {
            return Err(BackendError {
                status: ErrorStatus::KernelLaunch,
                context: format!("tenstorrent supports at most 2 group axes, got {}", prog.gws.len()).into(),
            });
        }
        for (axis, g) in prog.gws.iter().enumerate() {
            grid_dims[axis] = match g {
                GwsDim::Const(dim) => u32::try_from(*dim).map_err(|_| BackendError {
                    status: ErrorStatus::KernelLaunch,
                    context: format!("gws axis {axis} const dim {dim} does not fit u32").into(),
                })?,
                GwsDim::Param(ordinal) => {
                    vars.iter().find(|(o, _)| *o == *ordinal as u32).map(|(_, v)| *v).ok_or_else(|| BackendError {
                        status: ErrorStatus::KernelLaunch,
                        context: format!("gws axis {axis} param {ordinal} has no Variable launch arg").into(),
                    })?
                }
                g => todo!("tenstorrent gws: unhandled GwsDim variant {g:?}"),
            };
        }
        let mut rt_guard = rt.lock().unwrap();
        rt_guard.run(program_id.0, &src_indices, &dst_indices, grid_dims, &vars)?;

        Ok(Event::TT(TTEvent))
    }
}
