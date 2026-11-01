//! Raw FFI bindings to `libttsim.so`.
//!
//! ttsim ships no header file. The authoritative contract is the linker version
//! script `src/libttsim.map` in the ttsim source tree, which exports exactly the
//! ten symbols declared below; the signatures come from `docs/libttsim_api.md`.
//! Both are pinned in `PINS.toml`.
//!
//! The library is loaded with `dlopen` rather than linked, mirroring how ttsim's
//! own consumers (`rv64_sys_tt_attach`, the ttsim-qemu device model) use it. That
//! keeps the path configurable and keeps a missing simulator out of the link step.
//!
//! # Safety
//!
//! Nothing in this crate is safe to call directly. Every entry point:
//!
//! - is part of a **process-wide singleton** with no context handle,
//! - is **single-threaded and non-reentrant**,
//! - and **terminates the process via `_Exit`** on any contract violation, with
//!   no unwinding, no destructors, and no panic hook.
//!
//! That last property is the important one: a bad address does not return an
//! error, it ends the process. Callers must validate *before* calling. Use
//! [`tt_ttsim`](../tt_ttsim/index.html) instead of this crate.

use std::ffi::OsStr;

use libloading::{Library, Symbol};

/// Callback invoked when the simulated device reads host memory over DMA.
pub type DmaReadFn = unsafe extern "C" fn(paddr: u64, dst: *mut core::ffi::c_void, size: u32);

/// Callback invoked when the simulated device writes host memory over DMA.
pub type DmaWriteFn = unsafe extern "C" fn(paddr: u64, src: *const core::ffi::c_void, size: u32);

type FnInit = unsafe extern "C" fn();
type FnExit = unsafe extern "C" fn();
type FnSetDmaCallbacks = unsafe extern "C" fn(DmaReadFn, DmaWriteFn);
type FnConfigRd32 = unsafe extern "C" fn(bdf: u32, offset: u32) -> u32;
type FnConfigWr32 = unsafe extern "C" fn(bdf: u32, offset: u32, data: u32);
type FnMemRdBytes = unsafe extern "C" fn(paddr: u64, dst: *mut core::ffi::c_void, size: u32);
type FnMemWrBytes = unsafe extern "C" fn(paddr: u64, src: *const core::ffi::c_void, size: u32);
type FnClock = unsafe extern "C" fn(n_clocks: u32);
type FnTileRdBytes = unsafe extern "C" fn(x: u32, y: u32, addr: u64, dst: *mut core::ffi::c_void, size: u32);
type FnTileWrBytes = unsafe extern "C" fn(x: u32, y: u32, addr: u64, src: *const core::ffi::c_void, size: u32);

/// A loaded `libttsim.so` with all ten entry points resolved.
///
/// Holding one of these does not mean the simulator is running; see
/// `tt_ttsim::Simulator` for the lifecycle.
pub struct Lib {
    // Declared last-dropped-last: the function pointers borrow from `library`,
    // so `library` must outlive them. They are raw `fn` pointers rather than
    // `Symbol<'_, _>` to avoid a self-referential struct; `library` is kept
    // alive alongside them and never unloaded early.
    pub init: FnInit,
    pub exit: FnExit,
    pub set_pci_dma_mem_callbacks: FnSetDmaCallbacks,
    pub pci_config_rd32: FnConfigRd32,
    pub pci_config_wr32: FnConfigWr32,
    pub pci_mem_rd_bytes: FnMemRdBytes,
    pub pci_mem_wr_bytes: FnMemWrBytes,
    pub clock: FnClock,
    /// Transitional in the ttsim ABI, and `UnsupportedFunctionality` on both
    /// Wormhole and Blackhole — calling it kills the process. Bound only for
    /// completeness; there is no backdoor into tile memory, everything goes
    /// through PCIe TLB windows.
    pub tile_rd_bytes: FnTileRdBytes,
    /// See [`Lib::tile_rd_bytes`]. Also unsupported on Blackhole.
    pub tile_wr_bytes: FnTileWrBytes,

    _library: Library,
}

/// Why loading `libttsim.so` failed.
#[derive(Debug)]
pub enum LoadError {
    /// `dlopen` failed.
    Open(libloading::Error),
    /// `dlopen` succeeded but a required entry point was missing, which means
    /// the file is not a `libttsim.so` of the expected vintage.
    MissingSymbol { name: &'static str, source: libloading::Error },
}

impl std::fmt::Display for LoadError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            LoadError::Open(e) => write!(f, "could not dlopen libttsim: {e}"),
            LoadError::MissingSymbol { name, source } => {
                write!(f, "libttsim is missing required entry point `{name}`: {source}")
            }
        }
    }
}

impl std::error::Error for LoadError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            LoadError::Open(e) => Some(e),
            LoadError::MissingSymbol { source, .. } => Some(source),
        }
    }
}

impl Lib {
    /// `dlopen` the given `libttsim.so` and resolve every entry point.
    ///
    /// Resolving all ten up front is deliberate: a partially usable simulator
    /// would fail later, at a call site, where the failure mode is a dead
    /// process rather than an error.
    ///
    /// # Safety
    ///
    /// The path must name a genuine `libttsim.so`. Loading an arbitrary shared
    /// object runs its initializers.
    pub unsafe fn load(path: impl AsRef<OsStr>) -> Result<Self, LoadError> {
        let library = unsafe { Library::new(path.as_ref()) }.map_err(LoadError::Open)?;

        /// Resolve one symbol, naming it in the error rather than losing it.
        macro_rules! sym {
            ($name:literal) => {{
                let s: Symbol<'_, _> = unsafe { library.get(concat!($name, "\0").as_bytes()) }
                    .map_err(|source| LoadError::MissingSymbol { name: $name, source })?;
                *s
            }};
        }

        let this = Lib {
            init: sym!("libttsim_init"),
            exit: sym!("libttsim_exit"),
            set_pci_dma_mem_callbacks: sym!("libttsim_set_pci_dma_mem_callbacks"),
            pci_config_rd32: sym!("libttsim_pci_config_rd32"),
            pci_config_wr32: sym!("libttsim_pci_config_wr32"),
            pci_mem_rd_bytes: sym!("libttsim_pci_mem_rd_bytes"),
            pci_mem_wr_bytes: sym!("libttsim_pci_mem_wr_bytes"),
            clock: sym!("libttsim_clock"),
            tile_rd_bytes: sym!("libttsim_tile_rd_bytes"),
            tile_wr_bytes: sym!("libttsim_tile_wr_bytes"),
            _library: library,
        };
        Ok(this)
    }
}
