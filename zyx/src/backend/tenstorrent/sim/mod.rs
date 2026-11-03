//! A safe, singleton-enforcing wrapper over `libttsim.so`.
//!
//! **This crate is a development and test dependency. It must never appear in the
//! dependency graph of a shipped artifact** — CI enforces that with a `cargo tree`
//! check. Binding the simulator does not compromise the native-Rust goal the way
//! an FFI dependency on a production library would, but only as long as it stays
//! out of shipped builds.
//!
//! # Why this wrapper exists
//!
//! `libttsim.so` has two properties that make direct use hazardous:
//!
//! 1. **It is a process-wide singleton** with no context handle. Its state lives in
//!    file-scope globals, it is single-threaded and non-reentrant, and `init` may be
//!    called exactly once per process. `cargo test`'s thread-per-test default would
//!    otherwise corrupt it in ways that look exactly like simulator bugs.
//! 2. **Every contract violation terminates the process** via `_Exit` — no return
//!    code, no unwinding, no destructors, no panic hook. A misaligned access does
//!    not fail, it ends the test run.
//!
//! The answers are [`Simulator`], which can only be obtained once and is neither
//! `Send` nor `Sync`, and [`fork_scope`], which runs a closure in a forked child so
//! that a fatal error becomes a legible test failure rather than a vanished runner.
//!
//! Property (2) is also why [`transport`] validates every access against the
//! simulator's decode map before making the call. By the time a bad address reaches
//! `libttsim`, there is no one left to report it to.

// `fork_scope` isolates libttsim failures in a child process, which needs
// `fork`. Windows has no `fork`, so the Windows build runs the closure inline
// and loses that isolation; see `fork_windows.rs`.
#[cfg(unix)]
pub mod fork;
#[cfg(windows)]
#[path = "fork_windows.rs"]
pub mod fork;
pub mod sys;
pub mod transport;

pub use fork::{ForkError, fork_scope, outside_fork};
pub use transport::LibTtsim;

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};

use self::sys::{Lib, LoadError};
use tt_isa::noc::ChipId;

/// Set once `Simulator::open` has succeeded. Never cleared: `libttsim_init` has no
/// re-init path, so a second open would be a fatal error inside the library.
static OPENED: AtomicBool = AtomicBool::new(false);

/// Holds the loaded library for the life of the process.
///
/// Leaked deliberately rather than stored in a `OnceLock<Lib>`: the DMA callbacks
/// below are plain `extern "C"` functions that may run at any point after `init`,
/// and a `'static` reference is the simplest way to guarantee the library outlives
/// them. There is exactly one per process, so this leaks a fixed, bounded amount.
static mut LIB: Option<&'static Lib> = None;

/// The environment variable that overrides which `libttsim*.so` is loaded.
pub const LIB_PATH_ENV: &str = "TT_TTSIM_LIB";

/// The same, for the dual-chip build that [`x2_lib_path`] finds.
///
/// Not decoration: pointing this at the single-chip build is how the multi-chip
/// gate is checked for vacuity without editing any code.
pub const LIB_PATH_ENV_X2: &str = "TT_TTSIM_LIB_X2";

/// Overrides [`x4_lib_path`], as [`LIB_PATH_ENV_X2`] does for the dual-chip build.
pub const LIB_PATH_ENV_X4: &str = "TT_TTSIM_LIB_X4";

#[derive(Debug)]
pub enum OpenError {
    /// [`Simulator::open`] was already called in this process.
    AlreadyOpen,
    /// The `.so` could not be found on disk.
    NotFound(PathBuf),
    /// The `.so` could not be loaded or was missing an entry point.
    Load(LoadError),
    /// The simulator came up, but did not report a Blackhole.
    NotBlackhole { vendor: u16, device: u16 },
    /// A BAR base in configuration space did not match the address the library
    /// actually decodes. See [`chip_bar_base`].
    BarMismatch {
        chip: ChipId,
        bar: super::dev::Bar,
        expected: u64,
        found: u64,
    },
    /// Not even chip 0 answered configuration space.
    NoChips,
    /// A chip was present above an absent slot. Every build libttsim ships
    /// numbers its chips contiguously from zero, and [`ChipId`] is used both as
    /// the bdf device field and as the stride multiplier, so an ordinal that is
    /// not an index would put every access in the wrong 64 GiB window.
    SparseTopology { gap: ChipId, present_above: ChipId },
}

impl std::fmt::Display for OpenError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            OpenError::AlreadyOpen => write!(
                f,
                "libttsim was already initialized in this process; it is a singleton \
                 with no re-init path. Use fork_scope() to isolate test cases."
            ),
            OpenError::NotFound(p) => write!(
                f,
                "libttsim not found at {}. Set {LIB_PATH_ENV} to the simulator's shared object, \
                 or put it at vendor/libttsim_bh.so.",
                p.display()
            ),
            OpenError::Load(e) => write!(f, "{e}"),
            OpenError::NotBlackhole { vendor, device } => write!(
                f,
                "simulator reported {vendor:#06x}:{device:#06x}, not Blackhole. \
                 Is this libttsim_wh.so rather than libttsim_bh.so?"
            ),
            OpenError::NoChips => write!(
                f,
                "no chip answered configuration space at bdf 0; the library loaded \
                 but presents no PCIe endpoint"
            ),
            OpenError::SparseTopology { gap, present_above } => write!(
                f,
                "chip {} is absent but chip {} is present. This build numbers its \
                 chips contiguously from zero, so a gap means the bdf-to-chip \
                 mapping in this crate no longer matches the library's.",
                gap.0, present_above.0
            ),
            OpenError::BarMismatch { chip, bar, expected, found } => write!(
                f,
                "chip {}'s {bar:?} base is {found:#x} in config space but this build \
                 expects {expected:#x}; libttsim's hardcoded BAR map has changed",
                chip.0
            ),
        }
    }
}

impl std::error::Error for OpenError {}

impl From<LoadError> for OpenError {
    fn from(e: LoadError) -> Self {
        OpenError::Load(e)
    }
}

/// A running simulator.
///
/// There is at most one of these per process, and it is neither `Send` nor `Sync`:
/// the library is single-threaded and non-reentrant, so the token stays on the
/// thread that created it.
pub struct Simulator {
    lib: &'static Lib,
    /// Chips this build presents, counted by `probe_chips` at open.
    chips: u16,
    /// Makes `Simulator` `!Send + !Sync` without an unstable negative impl.
    _not_send: std::marker::PhantomData<*const ()>,
}

impl Simulator {
    /// Load and initialize the simulator.
    ///
    /// Succeeds at most once per process; the second call returns
    /// [`OpenError::AlreadyOpen`] rather than letting `libttsim_init` abort.
    ///
    /// The path comes from `$TT_TTSIM_LIB` if set, otherwise `vendor/libttsim_bh.so`
    /// beside the workspace root.
    pub fn open() -> Result<Self, OpenError> {
        match std::env::var_os(LIB_PATH_ENV) {
            Some(p) => Self::open_path(PathBuf::from(p)),
            None => Self::open_path(default_lib_path()),
        }
    }

    /// Load and initialize the simulator from a specific `.so`.
    pub fn open_path(path: impl AsRef<Path>) -> Result<Self, OpenError> {
        let path = path.as_ref();

        // Claim the singleton before touching the library, so a losing racer never
        // reaches `libttsim_init`.
        if OPENED.swap(true, Ordering::SeqCst) {
            return Err(OpenError::AlreadyOpen);
        }
        if !path.exists() {
            return Err(OpenError::NotFound(path.to_path_buf()));
        }

        // SAFETY: the path names a libttsim.so, which we are about to verify by
        // resolving all ten entry points and reading a known device ID from it.
        let lib: &'static Lib = Box::leak(Box::new(unsafe { Lib::load(path) }?));

        // SAFETY: single-threaded, and this is the one point in the process where
        // LIB is written — before `init`, and therefore before any callback can run.
        unsafe { LIB = Some(lib) };

        // Callbacks must be installed *before* init; the library rejects them
        // afterwards. Install them even though the baseline performs no DMA:
        // DMA with no callbacks installed is itself a fatal error, and ours at
        // least says what happened.
        //
        // SAFETY: both pointers are `extern "C"` functions with the required
        // signatures, and the simulator is not yet running.
        unsafe { (lib.set_pci_dma_mem_callbacks)(dma_read_unexpected, dma_write_unexpected) };

        // SAFETY: callbacks are installed and this is the first and only init.
        unsafe { (lib.init)() };

        let mut sim = Simulator { lib, chips: 0, _not_send: std::marker::PhantomData };
        sim.chips = sim.probe_chips()?;
        Ok(sim)
    }

    /// Count the chips this build presents, and check each one's BAR map.
    ///
    /// The walk covers every device slot a bdf can name rather than stopping at
    /// the first gap: presence is a prefix in every build libttsim ships, so a
    /// gap would mean this crate's bdf-to-chip mapping no longer matches the
    /// library's — and since [`ChipId`] doubles as the stride multiplier, that
    /// would silently address the wrong 64 GiB window rather than fail. Reading
    /// an absent slot is safe and returns all-ones; only *writing* one is fatal.
    ///
    /// The BAR check is the same guard the single-chip build always had, applied
    /// per chip. The validation layer in [`transport`] decides whether an access
    /// is legal by comparing against libttsim's *hardcoded* decode map, not
    /// against whatever config space reports. Confirm the two still agree: if a
    /// future libttsim moves a BAR, every subsequent access would be validated
    /// against the wrong map and then terminate the process. Better to refuse to
    /// start.
    fn probe_chips(&self) -> Result<u16, OpenError> {
        use super::dev::{Bar, ConfigOffset, DEVICE_ID_BLACKHOLE, VENDOR_ID_TENSTORRENT};

        let mut present = 0u16;
        for index in 0..MAX_MMIO_CHIPS {
            let chip = ChipId(index);
            // SAFETY: offset 0 is always decoded, the bdf is well-formed, and an
            // absent slot reads all-ones rather than faulting.
            let id = unsafe { (self.lib.pci_config_rd32)(chip_bdf(chip), ConfigOffset::VendorDevice as u32) };
            if id == u32::MAX {
                continue;
            }
            if index != present {
                return Err(OpenError::SparseTopology { gap: ChipId(present), present_above: chip });
            }

            let vendor = (id & 0xFFFF) as u16;
            let device = (id >> 16) as u16;
            if vendor != VENDOR_ID_TENSTORRENT || device != DEVICE_ID_BLACKHOLE {
                return Err(OpenError::NotBlackhole { vendor, device });
            }

            for bar in [Bar::Bar0, Bar::Bar2, Bar::Bar4] {
                // SAFETY: BAR offsets are decoded; the simulator is initialized.
                let lo = unsafe { (self.lib.pci_config_rd32)(chip_bdf(chip), bar.config_offset() as u32) };
                let hi = unsafe { (self.lib.pci_config_rd32)(chip_bdf(chip), bar.config_offset_hi() as u32) };
                let found = ((hi as u64) << 32) | ((lo & !0xF) as u64);
                let expected = chip_bar_base(chip, bar);
                if found != expected {
                    return Err(OpenError::BarMismatch { chip, bar, expected, found });
                }
            }
            present += 1;
        }

        if present == 0 {
            return Err(OpenError::NoChips);
        }
        Ok(present)
    }

    /// How many chips this build presents.
    pub fn chip_count(&self) -> u16 {
        self.chips
    }

    /// The chips this build presents, lowest first.
    pub fn chips(&self) -> impl Iterator<Item = ChipId> {
        (0..self.chips).map(ChipId)
    }

    /// Borrow this simulator as a [`Transport`](super::dev::Transport) for chip 0.
    pub fn transport(&mut self) -> LibTtsim<'_> {
        LibTtsim::new(self.lib, ChipId(0))
    }

    /// One transport per chip, lowest chip first.
    ///
    /// Plural because the `&mut self` is reborrowed once, for all of them: two
    /// separate calls could not both be live, and holding two `Device`s at once
    /// is the entire point. They coexist safely because each addresses a
    /// disjoint 64 GiB window of physical address space.
    ///
    /// Note that [`clock`](Self::clock) advances *every* chip — libttsim runs
    /// them on a single global timebase — so ticking one `Device` per chip in a
    /// loop advances time once per chip, not once.
    pub fn transports(&mut self) -> Vec<LibTtsim<'_>> {
        (0..self.chips).map(|i| LibTtsim::new(self.lib, ChipId(i))).collect()
    }

    /// Advance simulated time.
    ///
    /// Nothing else does. The simulator has no free-running clock: outside a call
    /// to this function, the device is frozen.
    pub fn clock(&mut self, n_clocks: u32) {
        // SAFETY: initialized, single-threaded (enforced by `!Send`).
        unsafe { (self.lib.clock)(n_clocks) }
    }
}

/// Bus/device/function of `chip`: bus 0, device `chip`, function 0.
///
/// libttsim's `pci_device_from_bdf` reports a slot as absent unless the function
/// is 0, the bus is 0, and the device index is below the build's chip count, so
/// the chip number *is* the bdf device field. Reading an absent bdf returns
/// all-ones; **writing one is fatal**, which is why nothing in this workspace
/// calls `pci_config_wr32`.
pub const fn chip_bdf(chip: ChipId) -> u32 {
    debug_assert!(chip.0 < MAX_MMIO_CHIPS);
    (chip.0 as u32) << 3
}

/// Device slots a bdf can name: the bdf device field is five bits wide.
pub const MAX_MMIO_CHIPS: u16 = 32;

/// Distance between one chip's BAR windows and the next chip's.
///
/// libttsim places device *i*'s BARs at device 0's bases plus `i` times this
/// (`PER_DEVICE_PADDR_STRIDE`), so a host physical address uniquely identifies
/// both the chip and the offset within it, and the memory-access entry points
/// pick the chip out of the address rather than from any current-chip state.
///
/// The stride is exactly where chip 0's BAR4 ends — see the assertion in
/// [`transport`] — so the bound that keeps an access inside its own BAR is also
/// the bound that keeps it out of the next chip's window.
pub const PER_CHIP_PADDR_STRIDE: u64 = 0x10_0000_0000;

/// The address at which `libttsim` decodes `chip`'s `bar`.
pub const fn chip_bar_base(chip: ChipId, bar: super::dev::Bar) -> u64 {
    expected_bar_base(bar) + (chip.0 as u64) * PER_CHIP_PADDR_STRIDE
}

/// The address at which `libttsim` decodes each BAR, for chip 0 — equivalently,
/// for the whole of the single-chip build, where the stride term is zero.
///
/// These are compile-time constants inside the library, not values derived from
/// configuration space. `libttsim_init` pre-programs config space to match, and
/// config-space BAR writes are accepted but change nothing — so relocating a BAR
/// the way a BIOS would makes config space lie while every memory access continues
/// to decode against the constants below, and the first access outside them
/// terminates the process.
pub const fn expected_bar_base(bar: super::dev::Bar) -> u64 {
    match bar {
        super::dev::Bar::Bar0 => 0x1_0000_0000,
        super::dev::Bar::Bar2 => 0x1_2000_0000,
        super::dev::Bar::Bar4 => 0x8_0000_0000,
    }
}

/// Where `cargo xtask fetch-ttsim` puts the single-chip library.
pub fn default_lib_path() -> PathBuf {
    workspace_root().join("vendor").join("libttsim_bh.so")
}

/// The dual-chip (P300) build, overridable with [`LIB_PATH_ENV_X2`].
///
/// A separate accessor rather than a `Simulator::open` argument because the
/// choice is per test binary — the simulator is a process-wide singleton — and
/// because the override is how the multi-chip gate is run against the
/// single-chip build to confirm it is not vacuous.
pub fn x2_lib_path() -> PathBuf {
    match std::env::var_os(LIB_PATH_ENV_X2) {
        Some(p) => PathBuf::from(p),
        None => workspace_root().join("vendor").join("libttsim_bh_x2.so"),
    }
}

/// The four-chip build (two P300s), overridable with [`LIB_PATH_ENV_X4`].
pub fn x4_lib_path() -> PathBuf {
    match std::env::var_os(LIB_PATH_ENV_X4) {
        Some(p) => PathBuf::from(p),
        None => workspace_root().join("vendor").join("libttsim_bh_x4.so"),
    }
}

fn workspace_root() -> PathBuf {
    // CARGO_MANIFEST_DIR is <root>/crates/tt-ttsim.
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .expect("crate manifest is two levels below the workspace root")
        .to_path_buf()
}

/// Installed so that unexpected DMA produces a diagnosis rather than libttsim's
/// generic fatal error. The baseline performs no DMA; reaching here means the
/// device tried to touch host memory, which is a bug in the code that programmed it.
unsafe extern "C" fn dma_read_unexpected(paddr: u64, dst: *mut core::ffi::c_void, size: u32) {
    let done = host::with_region(paddr, size as u64, |bytes| {
        // SAFETY: libttsim hands a buffer of `size` bytes to fill.
        unsafe { std::ptr::copy_nonoverlapping(bytes.as_ptr(), dst as *mut u8, size as usize) }
    });
    if done.is_none() {
        fatal_unexpected_dma("read", paddr, size)
    }
}

unsafe extern "C" fn dma_write_unexpected(paddr: u64, src: *const core::ffi::c_void, size: u32) {
    let done = host::with_region(paddr, size as u64, |bytes| {
        // SAFETY: libttsim hands `size` bytes to take.
        unsafe { std::ptr::copy_nonoverlapping(src as *const u8, bytes.as_mut_ptr(), size as usize) }
    });
    if done.is_none() {
        fatal_unexpected_dma("write", paddr, size)
    }
}

/// Simulated host memory for the card's DMA ([`super::dev::HostMemory`]):
/// regions the DMA callbacks serve, at made-up physical addresses the card
/// reaches through the PCIe tile's window to the host (`0x0...`). DMA
/// anywhere else is still fatal.
pub mod host {
    use std::sync::{Arc, Mutex};

    type Bytes = Arc<Mutex<Vec<u8>>>;
    static REGIONS: Mutex<Vec<(u64, Bytes)>> = Mutex::new(Vec::new());
    static NEXT: Mutex<u64> = Mutex::new(1 << 36);

    /// Run `f` on the bytes `[paddr, paddr + len)` of the region holding them.
    pub(crate) fn with_region(paddr: u64, len: u64, f: impl FnOnce(&mut [u8])) -> Option<()> {
        let regions = REGIONS.lock().unwrap_or_else(|p| p.into_inner());
        let (base, bytes) = regions.iter().find(|(base, b)| {
            let n = b.lock().unwrap_or_else(|p| p.into_inner()).len() as u64;
            paddr >= *base && paddr + len <= base + n
        })?;
        let mut b = bytes.lock().unwrap_or_else(|p| p.into_inner());
        let at = (paddr - base) as usize;
        f(&mut b[at..at + len as usize]);
        Some(())
    }

    /// One region, registered until dropped.
    pub struct SimHostMemory {
        base: u64,
        bytes: Bytes,
    }

    impl SimHostMemory {
        pub fn new(len: usize) -> Self {
            let len = len.max(1).next_multiple_of(4096);
            let mut next = NEXT.lock().unwrap_or_else(|p| p.into_inner());
            let base = *next;
            *next += (len as u64).next_multiple_of(1 << 30);
            let bytes = Arc::new(Mutex::new(vec![0u8; len]));
            REGIONS.lock().unwrap_or_else(|p| p.into_inner()).push((base, bytes.clone()));
            SimHostMemory { base, bytes }
        }
    }

    impl super::super::dev::HostMemory for SimHostMemory {
        fn noc_address(&self) -> u64 {
            self.base
        }
        fn len(&self) -> usize {
            self.bytes.lock().unwrap_or_else(|p| p.into_inner()).len()
        }
        fn read(&self, offset: usize, dst: &mut [u8]) {
            let b = self.bytes.lock().unwrap_or_else(|p| p.into_inner());
            dst.copy_from_slice(&b[offset..offset + dst.len()]);
        }
        fn write(&mut self, offset: usize, src: &[u8]) {
            let mut b = self.bytes.lock().unwrap_or_else(|p| p.into_inner());
            b[offset..offset + src.len()].copy_from_slice(src);
        }
        fn with_bytes(&mut self, f: &mut dyn FnMut(&mut [u8])) {
            f(&mut self.bytes.lock().unwrap_or_else(|p| p.into_inner()))
        }
    }

    impl Drop for SimHostMemory {
        fn drop(&mut self) {
            REGIONS.lock().unwrap_or_else(|p| p.into_inner()).retain(|(base, _)| *base != self.base);
        }
    }
}

fn fatal_unexpected_dma(kind: &str, paddr: u64, size: u32) -> ! {
    // Cannot unwind out of an `extern "C"` callback into C, and there is no way to
    // tell libttsim "no". Report and stop.
    eprintln!(
        "tt-ttsim: device attempted an unexpected DMA {kind} of {size} bytes at host \
         physical address {paddr:#x}.\n\
         The baseline registers no host memory for DMA, so this means a NoC or PCIe \
         descriptor was programmed with a host address by mistake."
    );
    std::process::abort()
}
