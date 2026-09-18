// Copyright (C) 2025 zk4x
// SPDX-License-Identifier: LGPL-3.0-only WITH Classpath-exception-2.0
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
// For Blackhole P100a the logical worker grid is 10 rows × 13 columns
// (x ∈ 0..13, y ∈ 0..10), giving 130 cores total (140 physical workers
// minus the 10-core dispatch column; harvested boards shrink x).
// Source: tt-metal `core_descriptors/blackhole_140_arch.yaml` (range
// end [12, 9], sized (end-start)+1 in `core_descriptor.cpp`). A
// single-core launch uses `gidx0 = 0, gidx1 = 0` (also written `{0, 0}`
// in CoreCoord notation).

use super::{DeviceInfo, DeviceProgramId, GwsDim, Kernel, LaunchArg, Pool, PoolBufferId, gws_from_kernel};
use crate::{
    DType,
    backend::DTypeCapability,
    codegen::tenstorrent::{CBId, TTCompiler, TTKernel},
    error::{BackendError, ErrorStatus},
    shape::Dim,
    slab::Slab,
};
use nanoserde::DeJson;
use std::{
    ffi::CString,
    io::{BufRead, BufReader, BufWriter, Write as IoWrite},
    path::PathBuf,
    process::{Child, ChildStdin, ChildStdout, Command},
    sync::{Arc, Mutex, OnceLock},
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
            let vendor = std::fs::read_to_string(&vendor_path).unwrap();
            if vendor.trim() == "0x1e52" {
                let subsys = std::fs::read_to_string(entry.path().join("subsystem_device")).unwrap();
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
    rc: u16,
}

// ---------------------------------------------------------------------------
// Memory pool — device DRAM buffers managed by C++ runtime.
// TTBuffer is a handle (u32 dev_index) into the runtime's buffer list.
// The pool shares the runtime IPC channel with TTDevice via Arc<Mutex>.
// ---------------------------------------------------------------------------

/// Process-wide per-device pools. Owned here — `mod.rs` only holds
/// `Pool::TT(i)` handles. `TT_INIT` serializes first construction only;
/// the alloc/free path never takes it. The pool carries its `DeviceInfo`
/// for later device registration.
static TT_POOLS: OnceLock<Vec<Arc<Mutex<TTMemoryPool>>>> = OnceLock::new();
static TT_INIT: Mutex<()> = Mutex::new(());

fn pools_with(config: &TTConfig, debug_dev: bool) -> Result<&'static Vec<Arc<Mutex<TTMemoryPool>>>, BackendError> {
    if let Some(pools) = TT_POOLS.get() {
        return Ok(pools);
    }
    let _init = TT_INIT.lock().unwrap_or_else(|_| panic!("tt pool init lock poisoned"));
    if let Some(pools) = TT_POOLS.get() {
        return Ok(pools);
    }
    let pools = ensure_pool_table(config, debug_dev)?;
    let _ = TT_POOLS.set(pools);
    TT_POOLS
        .get()
        .ok_or_else(|| BackendError { status: ErrorStatus::Initialization, context: "TT pool init failed".into() })
}

fn pools() -> Result<&'static Vec<Arc<Mutex<TTMemoryPool>>>, BackendError> {
    pools_with(&TTConfig::default(), false)
}

pub(super) fn pool(id: u16) -> Result<Arc<Mutex<TTMemoryPool>>, BackendError> {
    pools()?.get(id as usize).cloned().ok_or_else(|| no_pool(id))
}

pub(super) fn pool_count() -> u16 {
    pools().map(|pools| pools.len() as u16).unwrap_or(0)
}

fn no_pool(id: u16) -> BackendError {
    BackendError { status: ErrorStatus::Initialization, context: format!("Pool::TT({id}) is not available").into() }
}

#[derive(Debug)]
pub struct TTMemoryPool {
    pub(crate) buffers: Slab<PoolBufferId, TTBuffer>,
    runtime: Arc<Mutex<RuntimeProcess>>,
    free_bytes: Dim,
    dev_info: DeviceInfo,
    /// Real Tenstorrent chip id (from device_ids config). Not the pool ordinal.
    dev_id: u32,
}

pub(super) fn ensure_pool_table(config: &TTConfig, debug_dev: bool) -> Result<Vec<Arc<Mutex<TTMemoryPool>>>, BackendError> {
    let mut pools: Vec<Arc<Mutex<TTMemoryPool>>> = Vec::new();
    if let Some(device_ids) = &config.device_ids
        && device_ids.is_empty()
    {
        if debug_dev {
            println!("[tenstorrent] configured out");
        }
        return Ok(pools);
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
        .unwrap();

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

    // Real tensix grid (harvest-aware) bounds every group axis: const
    // grid sizes fail at compile (via gws_from_kernel), dynamic ones at
    // launch. max_global_work_dims IS the grid for TT.
    let (grid_rows, grid_cols) = runtime.lock().unwrap().grid()?;
    if debug_dev {
        println!("[tenstorrent] tensix grid {grid_rows} rows x {grid_cols} cols");
    }

    let dev_id = config.device_ids.as_ref().and_then(|ids| ids.first().copied()).unwrap();
    // F8E5M2 has no Blackhole DataFormat: not a capable dtype, codegen rejects it.
    let mut dtype_capability = [DTypeCapability::all(); DType::N_DTYPES];
    dtype_capability[DType::F8E5M2 as usize] = DTypeCapability::ZERO;
    pools.push(Arc::new(Mutex::new(TTMemoryPool {
        buffers: Slab::new(),
        runtime: runtime.clone(),
        free_bytes: Dim::from(dram_bytes as i64),
        dev_info: DeviceInfo {
            compute: 200_000_000_000_000, // ~200 TFLOPS BF16
            // Grid axes only (gidx0 row, gidx1 col); TT launches at most
            // 2 group axes, and the launch path rejects more.
            max_global_work_dims: vec![Dim::from(grid_rows), Dim::from(grid_cols)],
            max_local_threads: 1024,
            max_local_work_dims: vec![1, 1024, 1],
            preferred_vector_size: 32,
            local_mem_size: 1_500_000, // 1.5 MB L1 per Tensix core
            max_register_bytes: 128,
            tensor_cores: true,
            warp_size: 1, // Tensix has no SIMT warps
            cc: [0, 0],
            dtype_capability,
            has_native_exp2: false,
            supported_vec_lens: vec![32],
            tenstorrent: true,
            tile: [32, 32],
            tile_sizes: vec![[32, 32]],
            wmma_layouts: vec![],
            num_circular_buffers: 32, // architectural CB0-CB31
            has_openmp: false,
        },
        dev_id: u32::try_from(dev_id).unwrap(),
    })));

    Ok(pools)
}

/// Process-wide per-chip Tenstorrent devices. Owned here — `mod.rs` only holds
/// `Dev::TT(i)` handles. `TT_DEV_INIT` serializes first construction
/// only; compile/launch take the device lock, never the init lock.
static TT_DEVICES: OnceLock<Vec<Arc<Mutex<TTDevice>>>> = OnceLock::new();
static TT_DEV_INIT: Mutex<()> = Mutex::new(());

fn devices_with(config: &TTConfig, debug_dev: bool) -> Result<&'static Vec<Arc<Mutex<TTDevice>>>, BackendError> {
    if let Some(devs) = TT_DEVICES.get() {
        return Ok(devs);
    }
    let _init = TT_DEV_INIT.lock().unwrap_or_else(|_| panic!("tt device init lock poisoned"));
    if let Some(devs) = TT_DEVICES.get() {
        return Ok(devs);
    }
    let devs = ensure_device_table(config, debug_dev)?;
    let _ = TT_DEVICES.set(devs);
    TT_DEVICES.get().ok_or_else(|| BackendError {
        status: ErrorStatus::Initialization,
        context: "TT device init failed".into(),
    })
}

fn devices() -> Result<&'static Vec<Arc<Mutex<TTDevice>>>, BackendError> {
    devices_with(&super::config().tenstorrent, super::debug_backends())
}

pub(super) fn device(id: u16) -> Result<Arc<Mutex<TTDevice>>, BackendError> {
    devices()?.get(id as usize).cloned().ok_or_else(|| BackendError {
        status: ErrorStatus::Initialization,
        context: format!("Dev::TT({id}) is not available").into(),
    })
}

pub(super) fn device_count() -> u16 {
    devices().map(|devs| devs.len() as u16).unwrap_or(0)
}

fn ensure_device_table(config: &TTConfig, debug_dev: bool) -> Result<Vec<Arc<Mutex<TTDevice>>>, BackendError> {
    let pools = pools_with(config, debug_dev)?;
    let mut devs = Vec::with_capacity(pools.len());
    for (idx, pool_arc) in pools.iter().enumerate() {
        let pool_id = Pool::TT(u16::try_from(idx).expect("So many Tenstorrent devices..."));
        let guard = super::lock(pool_id, pool_arc);
        devs.push(Arc::new(Mutex::new(TTDevice {
            device_info: Arc::new(guard.dev_info.clone()),
            dev_id: guard.dev_id,
            memory_pool: pool_id,
            runtime: guard.runtime.clone(),
            programs: Slab::new(),
        })));
    }
    Ok(devs)
}

fn create_temp_shm(size: u64) -> Result<(CString, *mut u8, u64), BackendError> {
    let pid = std::process::id();
    let ns = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos();
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

    pub fn allocate(&mut self, bytes: Dim) -> Result<PoolBufferId, BackendError> {
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
        let buf = TTBuffer { dev_index, size: bytes_u64, rc: 1 };
        Ok(self.buffers.push(buf))
    }

    /// Increment the buffer's reference count. Checked math: overflow panics.
    pub fn retain(&mut self, buffer_id: PoolBufferId) {
        match self.buffers.get_mut(buffer_id) {
            Some(buffer) => buffer.rc = buffer.rc.checked_add(1).expect("TTBuffer rc overflow"),
            None => debug_assert!(false, "retain of unknown TT buffer {buffer_id:?}"),
        }
    }

    /// Decrement the reference count. At zero the buffer is freed immediately:
    /// the TT shim is synchronous and holds no in-flight work — async
    /// consumers elsewhere retain the buffer while they still need it.
    pub fn release(&mut self, buffer_id: PoolBufferId) {
        let Some(buffer) = self.buffers.get_mut(buffer_id) else {
            debug_assert!(false, "release of unknown TT buffer {buffer_id:?}");
            return;
        };
        buffer.rc = buffer.rc.checked_sub(1).expect("TTBuffer rc underflow");
        if buffer.rc == 0 {
            let buf = unsafe { self.buffers.remove_and_return(buffer_id) };
            self.free_bytes += buf.size as Dim;
            let _ = self.runtime.lock().unwrap().free_buf(buf.dev_index);
        }
    }

    /// Blocking shim upload (sync — the shim call runs to completion).
    pub fn host_to_pool(&mut self, src: &[u8], dst: PoolBufferId) -> Result<(), BackendError> {
        let rt = &self.runtime;
        let buf = self
            .buffers
            .get_mut(dst)
            .ok_or_else(|| BackendError { status: ErrorStatus::MemoryCopyH2P, context: "invalid buffer id".into() })?;
        let len = src.len().min(buf.size as usize);
        let (cname, shm_ptr, _) = create_temp_shm(len as u64)?;
        let shm_path = cname.to_str().unwrap();
        unsafe { std::ptr::copy_nonoverlapping(src.as_ptr(), shm_ptr, len) };
        rt.lock().unwrap().write_buf(buf.dev_index, shm_path, len as u64)?;
        unsafe {
            libc::munmap(shm_ptr as *mut libc::c_void, len as usize);
            libc::shm_unlink(cname.as_ptr());
        }
        Ok(())
    }

    /// Blocking shim download (sync — the shim call runs to completion).
    pub fn pool_to_host(&mut self, src: PoolBufferId, dst: &mut [u8]) -> Result<(), BackendError> {
        let rt = &self.runtime;
        let buf = self
            .buffers
            .get_mut(src)
            .ok_or_else(|| BackendError { status: ErrorStatus::MemoryCopyP2H, context: "invalid buffer id".into() })?;
        let len = dst.len().min(buf.size as usize);
        let (cname, shm_ptr, _) = create_temp_shm(len as u64)?;
        let shm_path = cname.to_str().unwrap();
        rt.lock().unwrap().read_buf(buf.dev_index, shm_path, len as u64)?;
        unsafe {
            std::ptr::copy_nonoverlapping(shm_ptr, dst.as_mut_ptr(), len);
            libc::munmap(shm_ptr as *mut libc::c_void, len as usize);
            libc::shm_unlink(cname.as_ptr());
        }
        Ok(())
    }

    /// Synchronous copy into this pool (the TT shim has no async queues): the
    /// source is consumed within this call, so no retain is needed. Host
    /// sources upload directly; every other pool stages through host memory.
    pub fn pool_to_pool(&mut self, src: Pool, src_buf: PoolBufferId, dst_buf: PoolBufferId) -> Result<(), BackendError> {
        match src {
            Pool::Host => {
                let src_pool = super::host::pool();
                let src_pool = super::lock(src, &src_pool);
                let data = src_pool.get_buffer(src_buf).to_vec();
                drop(src_pool);
                self.host_to_pool(&data, dst_buf)
            }
            // No P2P path in the tt-runtime shim yet — stage through host.
            _ => {
                let len = {
                    let dst_ref = self.buffers.get(dst_buf).ok_or_else(|| BackendError {
                        status: ErrorStatus::MemoryCopyP2H,
                        context: "invalid dst buffer id".into(),
                    })?;
                    dst_ref.size as usize
                };
                let mut staging = vec![0u8; len];
                src.pool_to_host(src_buf, &mut staging)?;
                self.host_to_pool(&staging, dst_buf)
            }
        }
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
            let msg = extract_json_str(&resp, "msg").unwrap();
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

        let timeout_ms = i32::try_from(timeout_ms).unwrap();
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

    /// Logical tensix compute grid (rows, cols), harvest-aware, reported
    /// by the driver. Feeds `max_global_work_dims`, so const grid axes
    /// are bounds-checked at compile and dynamic ones at launch.
    fn grid(&mut self) -> Result<(u32, u32), BackendError> {
        self.send(r#"{"cmd":"grid"}"#)?;
        let resp = self.recv_with_timeout(self.timeout_ms)?;
        if resp.contains("\"error\"") {
            let msg = extract_json_str(&resp, "msg").unwrap();
            return Err(BackendError { status: ErrorStatus::Initialization, context: format!("grid error: {msg}").into() });
        }
        let parse = |key: &str| {
            extract_json_str(&resp, key)
                .ok_or_else(|| BackendError {
                    status: ErrorStatus::Initialization,
                    context: format!("grid: no {key} in response").into(),
                })?
                .parse::<u32>()
                .map_err(|_| BackendError { status: ErrorStatus::Initialization, context: format!("grid: invalid {key}").into() })
        };
        Ok((parse("rows")?, parse("cols")?))
    }

    fn alloc_buf(&mut self, size: u64, tile_bytes: u64) -> Result<u32, BackendError> {
        let cmd = format!(r#"{{"cmd":"alloc_buf","size":{size},"tile_bytes":{tile_bytes}}}"#);
        self.send(&cmd)?;
        let resp = self.recv_with_timeout(self.timeout_ms)?;
        if resp.contains("\"error\"") {
            let msg = extract_json_str(&resp, "msg").unwrap();
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
            let msg = extract_json_str(&resp, "msg").unwrap();
            return Err(BackendError { status: ErrorStatus::MemoryAllocation, context: format!("free_buf error: {msg}").into() });
        }
        Ok(())
    }

    fn write_buf(&mut self, dev_index: u32, shm_path: &str, size: u64) -> Result<(), BackendError> {
        let cmd = format!(r#"{{"cmd":"write_buf","index":{dev_index},"shm_path":"{shm_path}","size":{size}}}"#);
        self.send(&cmd)?;
        let resp = self.recv_with_timeout(self.timeout_ms)?;
        if resp.contains("\"error\"") {
            let msg = extract_json_str(&resp, "msg").unwrap();
            return Err(BackendError { status: ErrorStatus::MemoryCopyH2P, context: format!("write_buf error: {msg}").into() });
        }
        Ok(())
    }

    fn read_buf(&mut self, dev_index: u32, shm_path: &str, size: u64) -> Result<(), BackendError> {
        let cmd = format!(r#"{{"cmd":"read_buf","index":{dev_index},"shm_path":"{shm_path}","size":{size}}}"#);
        self.send(&cmd)?;
        let resp = self.recv_with_timeout(self.timeout_ms)?;
        if resp.contains("\"error\"") {
            let msg = extract_json_str(&resp, "msg").unwrap();
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
        cb_config: &Slab<CBId, (u32, u32, u32)>,
        n_params: u32,
        reader_params: &[u32],
        compute_params: &[u32],
        writer_params: &[u32],
        fp32_dest_acc_en: bool,
    ) -> Result<(), BackendError> {
        let reader_source_len = reader_source.len();
        let compute_source_len = compute_source.len();
        let writer_source_len = writer_source.len();
        let n_cbs = usize::from(cb_config.len());
        let dest_acc = fp32_dest_acc_en as u32;
        let mut cmd = format!(
            r#"{{"cmd":"compile_program","id":{id},"reader_source_len":{reader_source_len},"compute_source_len":{compute_source_len},"writer_source_len":{writer_source_len},"n_cbs":{n_cbs},"n_params":{n_params},"n_reader_params":{},"n_compute_params":{},"n_writer_params":{},"fp32_dest_acc":{dest_acc}"#,
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
        for (i, (cb, (fmt, tb, nt))) in cb_config.iter().enumerate() {
            cmd.push_str(&format!(r#","cb_idx{i}":{cb},"cb_fmt{i}":{fmt},"cb_tb{i}":{tb},"cb_nt{i}":{nt}"#));
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
            let msg = extract_json_str(&resp, "msg").unwrap();
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
            let msg = extract_json_str(&resp, "msg").unwrap();
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
            let msg = extract_json_str(&resp, "msg").unwrap();
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
    /// Tensix grid rows/cols (from DeviceInfo at compile): launch-resolved
    /// grid dims (dynamic sizes) are bounds-checked against this.
    max_grid: [u32; 2],
}

// ---------------------------------------------------------------------------
// Device
// ---------------------------------------------------------------------------

#[derive(Debug)]
pub struct TTDevice {
    device_info: Arc<DeviceInfo>,
    /// Real Tenstorrent chip id (from device_ids config), set at init. Not the slab index.
    pub(crate) dev_id: u32,
    memory_pool: Pool,
    runtime: Arc<Mutex<RuntimeProcess>>,
    programs: Slab<DeviceProgramId, TTProgram>,
}

impl TTDevice {
    pub fn deinitialize(&mut self) {}

    pub fn info(&self) -> Arc<DeviceInfo> {
        self.device_info.clone()
    }

    pub const fn memory_pool(&self) -> Pool {
        self.memory_pool
    }

    pub fn free_compute(&self) -> u128 {
        self.device_info.compute
    }

    #[allow(unused_must_use)]
    pub fn compile(&mut self, kernel: &Kernel, debug_asm: bool) -> Result<DeviceProgramId, BackendError> {
        // CB ids, section params, param ordinals, and input/output dtypes
        // are calculated only in `Kernel::generate_tenstorrent` (single
        // point); the backend consumes the returned tables. What stays
        // here is launch-side assembly: the group-grid walk, the runtime
        // CB config, and the program compile call.
        let compiler = kernel.generate_tenstorrent()?;
        // DST mode is resolved by codegen; both variants expose the
        // same tables.
        let (param_len, reader_k, compute_k, writer_k, input_dtypes, output_dtypes, cb_config) = match &compiler {
            TTCompiler::Bf16(c) => (
                c.noc.param_ordinal_of.len(),
                &c.reader,
                &c.compute,
                &c.writer,
                &c.noc.input_dtypes,
                &c.noc.output_dtypes,
                &c.cb.config,
            ),
            TTCompiler::Fp32(c) => (
                c.noc.param_ordinal_of.len(),
                &c.reader,
                &c.compute,
                &c.writer,
                &c.noc.input_dtypes,
                &c.noc.output_dtypes,
                &c.cb.config,
            ),
        };

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
        let n_params = param_len as u32;
        // Group grid via the shared helper (same as CUDA/OpenCL/wgpu/HIP):
        // axis-ordered, full dim expressions, const lengths validated
        // against the device max. Param-backed lengths resolve at launch
        // from the Variable arg.
        let gws = gws_from_kernel(kernel, &self.device_info.max_global_work_dims)?;

        let TTKernel::Reader { src: reader, ordinals: reader_params, .. } = reader_k else {
            return Err(BackendError {
                status: ErrorStatus::KernelCompilation,
                context: "tenstorrent2 reader kernel missing".into(),
            });
        };
        // A missing compute kernel is valid: pure copy kernels move data
        // without computing. Only a wrong variant in its slot is an error.
        let empty_src = String::new();
        let empty_ord: Vec<u32> = Vec::new();
        let (compute, compute_params) = match compute_k {
            TTKernel::Compute { src, ordinals, .. } => (src, ordinals),
            TTKernel::None => (&empty_src, &empty_ord),
            TTKernel::Reader { .. } | TTKernel::Writer { .. } => {
                return Err(BackendError {
                    status: ErrorStatus::KernelCompilation,
                    context: "tenstorrent2 compute slot holds a non-compute kernel".into(),
                });
            }
        };
        let TTKernel::Writer { src: writer, ordinals: writer_params, .. } = writer_k else {
            return Err(BackendError {
                status: ErrorStatus::KernelCompilation,
                context: "tenstorrent2 writer emission not implemented".into(),
            });
        };
        if debug_asm {
            eprintln!("[tenstorrent2] reader:\n{reader}");
            eprintln!("[tenstorrent2] compute:\n{compute}");
            eprintln!("[tenstorrent2] writer:\n{writer}");
        }

        // DST geometry follows the codegen variant: the Fp32 compiler
        // ran iff compute unpacks an F32 tile into DST, which is
        // exactly when 32-bit Dest mode is required (any F32 tile in
        // DST, per the typecast header).
        let fp32_dest_acc_en = matches!(compiler, TTCompiler::Fp32(_));

        // Snapshot the grid for the launch-time bounds check (dynamic
        // sizes only; const sizes already failed at compile above).
        let mg = &self.device_info.max_global_work_dims;
        let max_grid = [
            u32::try_from(mg[0]).map_err(|_| BackendError {
                status: ErrorStatus::KernelCompilation,
                context: "tenstorrent grid rows do not fit u32".into(),
            })?,
            u32::try_from(mg[1]).map_err(|_| BackendError {
                status: ErrorStatus::KernelCompilation,
                context: "tenstorrent grid cols do not fit u32".into(),
            })?,
        ];
        let prog_id = self.programs.push(TTProgram {
            input_dtypes: input_dtypes.clone(),
            output_dtypes: output_dtypes.clone(),
            gws,
            max_grid,
        });

        {
            let mut rt_guard = self.runtime.lock().unwrap();
            rt_guard.compile_program(
                prog_id.0,
                reader,
                compute,
                writer,
                cb_config,
                n_params,
                reader_params,
                compute_params,
                writer_params,
                fp32_dest_acc_en,
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
        pool_handle: Pool,
        args: &[LaunchArg],
    ) -> Result<(), BackendError> {
        debug_assert_eq!(pool_handle, self.memory_pool);
        let Pool::TT(id) = pool_handle else {
            unreachable!("TT launch with non-TT pool")
        };
        let pool_arc = pool(id).expect("launch on unavailable TT pool");
        let memory_pool = super::lock(pool_handle, &pool_arc);
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
            // Variable ordinals index the launch args (head order), which
            // always carry a Variable value at a Variable param's position.
            let dim = g.eval(&mut |ordinal| {
                vars.iter()
                    .find(|(o, _)| *o == ordinal as u32)
                    .map(|(_, v)| Dim::from(*v))
                    .expect("gws param ordinal has no Variable launch arg")
            });
            grid_dims[axis] = u32::try_from(dim).map_err(|_| BackendError {
                status: ErrorStatus::KernelLaunch,
                context: format!("gws axis {axis} dim {dim} does not fit u32").into(),
            })?;
            // Dynamic sizes skip the compile check: bound them here.
            if grid_dims[axis] > prog.max_grid[axis] {
                return Err(BackendError {
                    status: ErrorStatus::KernelLaunch,
                    context: format!(
                        "tenstorrent grid axis {axis} size {} exceeds device grid {}",
                        grid_dims[axis], prog.max_grid[axis]
                    )
                    .into(),
                });
            }
        }
        let mut rt_guard = rt.lock().unwrap();
        rt_guard.run(program_id.0, &src_indices, &dst_indices, grid_dims, &vars)?;

        Ok(())
    }
}
