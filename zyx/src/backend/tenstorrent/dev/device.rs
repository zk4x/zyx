//! A Blackhole device reached through TLB windows.
//!
//! Wraps a [`Transport`] with the two things every caller above it needs: a window
//! allocator, and transfers that are split so no access ever straddles a window.

use std::collections::{BTreeMap, HashMap};
use std::sync::{Arc, Mutex, MutexGuard};
use std::time::Instant;

use tt_isa::noc::{niu, ChipId, NocCoord, NocId, TileType};
use tt_isa::tensix::{self, Core};

use super::tlb::{
    self, TlbConfig, WindowKind, KERNEL_RESERVED_WINDOW, NUM_2MIB_WINDOWS, NUM_WINDOWS,
};
use super::{Bar, Result, Transport, TransportError};

mod fence;
pub use fence::FencedWrite;

/// A TLB window reserved for this `Device`'s use.
///
/// Returned by [`Device::alloc_window`]. Dropping it returns the index to the
/// `Device`'s pool; [`Device::free_window`] does the same, explicitly. Not
/// `Copy`: two callers holding the same index would retarget the window under
/// each other.
///
/// The pool is shared rather than borrowed because `Drop` cannot take
/// `&mut Device`, and a window that borrowed its device would make holding
/// several at once -- what every multi-window transfer does -- a borrow
/// conflict. Before this, a dropped window was lost to the pool for the rest of
/// the process, silently; that cost the Phase 1 silicon gate a run.
pub struct Window {
    index: u16,
    kind: WindowKind,
    pool: Arc<Mutex<Pool>>,
}

impl std::fmt::Debug for Window {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Window")
            .field("index", &self.index)
            .field("kind", &self.kind)
            .finish()
    }
}

impl PartialEq for Window {
    fn eq(&self, other: &Self) -> bool {
        self.index == other.index && Arc::ptr_eq(&self.pool, &other.pool)
    }
}

impl Eq for Window {}

impl Drop for Window {
    fn drop(&mut self) {
        // A poisoned pool means a panic while it was held; the index is lost,
        // which is the old behaviour and no worse than the panic itself.
        if let Ok(mut pool) = self.pool.lock() {
            pool.release(self.index);
        }
    }
}

impl Window {
    pub fn index(&self) -> u16 {
        self.index
    }

    pub fn kind(&self) -> WindowKind {
        self.kind
    }
}

/// What a window is currently pointing at, so that redundant reconfiguration can be
/// skipped.
///
/// Shadowing is not an optimisation, it is a necessity: the configuration registers
/// are write-only. It happens to also be an optimisation, since each configuration
/// write is a slow uncached MMIO round trip — the same reason `ethdump.c:264` caches
/// them.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
struct Shadow {
    words: [u32; 3],
    readable: bool,
}

/// The windows a `Device` has not handed out, and what each configured one
/// points at. Shared with every [`Window`] so that dropping one can return it.
#[derive(Debug)]
struct Pool {
    /// Windows not currently handed out, lowest first.
    free: Vec<u16>,
    shadow: BTreeMap<u16, Shadow>,
}

impl Pool {
    /// Return `index` to the free list, dropping its shadow entry so the next
    /// holder cannot inherit a stale belief about where it points. The hardware
    /// configuration is left as it was: there is no "unconfigured" state to
    /// restore it to, and the next holder reconfigures before use.
    fn release(&mut self, index: u16) {
        self.shadow.remove(&index);
        let insert_at = self.free.partition_point(|&i| i < index);
        debug_assert!(self.free.get(insert_at) != Some(&index), "double release");
        self.free.insert(insert_at, index);
    }
}

/// A tile found by [`Device::discover_tiles`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Tile<N: NocId> {
    pub coord: NocCoord<N>,
    pub kind: TileType,
    /// Index of this tile within its type.
    pub index: u8,
    /// True if the tile is fused off and must not be used.
    pub harvested: bool,
}

/// What a [`Device`] has sent across its transport: PCIe on silicon.
///
/// Counted at the three places a `Device` touches a BAR -- data writes, data
/// reads, and TLB retargets -- so it covers everything above them, including
/// the core-control and ARC paths. `*_calls` counts transport calls, not bus
/// transactions: one call is one contiguous run of dword accesses.
///
/// The Phase 9 metric. Tensors are meant to live on the card, so a steady-state
/// training step should move almost nothing here, and a gate can say so by
/// subtracting two snapshots.
#[derive(Copy, Clone, Debug, Default, Eq, PartialEq)]
pub struct Traffic {
    pub bytes_written: u64,
    pub bytes_read: u64,
    pub write_calls: u64,
    pub read_calls: u64,
    /// How many of the writes were TLB retargets (three config dwords each).
    pub retargets: u64,
}

impl std::ops::Add for Traffic {
    type Output = Traffic;
    fn add(self, other: Traffic) -> Traffic {
        Traffic {
            bytes_written: self.bytes_written + other.bytes_written,
            bytes_read: self.bytes_read + other.bytes_read,
            write_calls: self.write_calls + other.write_calls,
            read_calls: self.read_calls + other.read_calls,
            retargets: self.retargets + other.retargets,
        }
    }
}

impl std::ops::Sub for Traffic {
    type Output = Traffic;
    fn sub(self, earlier: Traffic) -> Traffic {
        Traffic {
            bytes_written: self.bytes_written - earlier.bytes_written,
            bytes_read: self.bytes_read - earlier.bytes_read,
            write_calls: self.write_calls - earlier.write_calls,
            read_calls: self.read_calls - earlier.read_calls,
            retargets: self.retargets - earlier.retargets,
        }
    }
}

pub struct Device<T: Transport> {
    transport: T,
    chip: ChipId,
    pool: Arc<Mutex<Pool>>,
    traffic: Traffic,
    /// When this `Device` last released each core from reset, keyed by
    /// `(NoC index, x, y, core)`. Read by the local-data-RAM accessors, which
    /// must not touch a core's RAM during the zeroing that follows a release.
    pub(crate) released: HashMap<(u8, u8, u8, Core), Instant>,
    /// Did [`Device::open`] raise the chip to its busy operating point? If so,
    /// dropping the `Device` returns it to idle.
    busy: bool,
}

/// Who manages the chip's operating point (ARC `AICLK_GO_BUSY` / `GO_LONG_IDLE`).
#[derive(Copy, Clone, Debug, Default, Eq, PartialEq)]
pub enum PowerPolicy {
    /// Raise the chip to busy on open, and return it to idle on drop. The
    /// default, and what UMD does.
    #[default]
    Busy,
    /// Send nothing: the caller calls [`Device::set_busy`] itself, before any
    /// compute. For explicit power management.
    Manual,
}

/// Return the chip to its idle operating point if [`Device::open`] raised it.
///
/// **One `Device` per chip.** Busy/idle is chip-wide ARC state with no
/// reference count: if two `Device`s -- in this process or another, tt-metal
/// included -- hold the same chip, the first to drop idles it under the other,
/// which then computes at the operating point where the Matrix Unit's `Src`
/// reads are unreliable (divergence row 48). A failure here can only be
/// reported, not returned; the worst case is a chip left busy, which is safe.
impl<T: Transport> Drop for Device<T> {
    fn drop(&mut self) {
        if !self.busy {
            return;
        }
        let Ok(w) = self.alloc_window(super::tlb::WindowKind::TwoMib) else {
            eprintln!(
                "tt-device: no TLB window to return chip {:?} to idle",
                self.chip
            );
            return;
        };
        if let Err(e) = self.set_busy(&w, false) {
            eprintln!(
                "tt-device: could not return chip {:?} to idle: {e}",
                self.chip
            );
        }
        self.free_window(w);
    }
}

/// Refuse a plain access to the local-data-RAM aperture.
///
/// A baby RISC-V's local data RAM does not answer the NoC while its core is held
/// in soft reset: the request is never completed, the NoC hangs, and the ARC
/// recovers with a chip reset that drops the PCIe link. On a card passed through
/// to a VM that takes the host down, which is how this was found
/// (`silicon_local_ram.rs`, 2026-09-30). Whether a core is in reset is state on
/// the chip, not something a plain `read`/`write` knows, so the aperture is only
/// reachable through [`Device::local_ram_read`] and [`Device::local_ram_write`],
/// which check it.
///
/// Ethernet tiles map different registers at these addresses; reaching them
/// will need its own accessor when Phase 8 does.
/// The bulk path is memory-only; see [`Device::l1_write`].
fn refuse_non_l1<N: NocId>(coord: NocCoord<N>, address: u64, len: usize) -> Result<()> {
    let in_l1 = address
        .checked_add(len as u64)
        .is_some_and(|end| end <= tensix::L1_SIZE);
    if !tt_isa::noc::grid::is_tensix_geometry(coord.x(), coord.y()) || !in_l1 {
        return Err(TransportError::Hazard {
            address,
            reason: "the bulk (write-combining) path is for Tensix L1 and GDDR only; \
                     registers must go through Device::write",
        });
    }
    Ok(())
}

fn refuse_local_ram_aperture(address: u64, len: usize) -> Result<()> {
    if touches_local_ram_aperture(address, len) {
        return Err(TransportError::Hazard {
            address,
            reason: "the local-data-RAM aperture hangs the NoC if the owning core is in \
                     reset; use Device::local_ram_read / local_ram_write",
        });
    }
    Ok(())
}

/// Does `[address, address + len)` touch the NoC-visible local-data-RAM
/// aperture (`0xFFB1_4000..0xFFB1_E000`)?
fn touches_local_ram_aperture(address: u64, len: usize) -> bool {
    let end = address.saturating_add(len as u64);
    address < tensix::LOCAL_DATA_RAM_NOC_END && end > tensix::LOCAL_DATA_RAM_NOC_BASE
}

impl<T: Transport> Device<T> {
    /// Take ownership of a transport and confirm it is a Blackhole.
    ///
    /// Every address constant in this crate is Blackhole-specific; against another
    /// part they are merely plausible, which is the worst kind of wrong.
    pub fn open(transport: T) -> Result<Self> {
        Self::open_with_power(transport, PowerPolicy::Busy)
    }

    /// [`Device::open`], choosing who manages the chip's operating point.
    ///
    /// [`PowerPolicy::Busy`] is what `open` does. [`PowerPolicy::Manual`] sends no
    /// power message on open or on drop: the caller owns the operating point and
    /// must call [`Device::set_busy`] before any compute, because at idle the
    /// Matrix Unit's `Src` reads are unreliable (divergence row 48). For power
    /// management that raises the clock only around work.
    pub fn open_with_power(mut transport: T, power: PowerPolicy) -> Result<Self> {
        transport.verify_is_blackhole()?;

        // Window 201 belongs to the kernel driver and must never be handed out --
        // honoured on the simulator too, where there is no driver to collide with,
        // so that allocation behaviour is identical on silicon.
        let free: Vec<u16> = (0..NUM_WINDOWS)
            .filter(|&i| i != KERNEL_RESERVED_WINDOW)
            .collect();

        // Taken from the transport rather than assumed: a transport is one chip,
        // and it is the only thing that knows which. Asserting it here instead
        // would let a `Device` claim to be a chip it does not address.
        let chip = transport.chip();

        let mut dev = Device {
            transport,
            chip,
            pool: Arc::new(Mutex::new(Pool {
                free,
                shadow: BTreeMap::new(),
            })),
            released: HashMap::new(),
            busy: false,
            traffic: Traffic::default(),
        };

        // Compute needs the busy operating point; see `Device::set_busy`. Held
        // for the life of the `Device`, and given back by `Drop`, as UMD's
        // `LocalChip` does. The ARC message touches only the ARC tile, which is
        // safe before the harvesting mask or the translation state is known.
        if power == PowerPolicy::Busy && !dev.transport.is_simulated() {
            let w = dev.alloc_window(super::tlb::WindowKind::TwoMib)?;
            let result = dev.set_busy(&w, true);
            dev.free_window(w);
            result?;
            dev.busy = true;
        }
        Ok(dev)
    }

    /// Everything this `Device` has moved across its transport so far.
    pub fn traffic(&self) -> Traffic {
        self.traffic
    }

    pub fn chip(&self) -> ChipId {
        self.chip
    }

    pub fn transport(&mut self) -> &mut T {
        &mut self.transport
    }

    /// Advance simulated time. A no-op against silicon.
    pub fn tick(&mut self, n: u32) {
        self.transport.tick(n);
    }

    fn pool(&self) -> MutexGuard<'_, Pool> {
        // Poisoning needs a panic inside `Pool::release` or the few lines below
        // that hold the lock, none of which can panic; recover rather than
        // propagate a panic from an unrelated thread.
        self.pool.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// Refuse a window handed out by a different `Device`: its index means a
    /// different window of a different chip's BAR here.
    fn check_owned(&self, window: &Window) -> Result<()> {
        if Arc::ptr_eq(&window.pool, &self.pool) {
            Ok(())
        } else {
            Err(TransportError::Hazard {
                address: 0,
                reason: "this window was allocated by a different Device",
            })
        }
    }

    /// Reserve a window of the requested geometry.
    pub fn alloc_window(&mut self, kind: WindowKind) -> Result<Window> {
        let mut pool = self.pool();
        let position = pool
            .free
            .iter()
            .position(|&i| tlb::window_kind(i) == Some(kind));
        match position {
            Some(p) => {
                let index = pool.free.remove(p);
                drop(pool);
                Ok(Window {
                    index,
                    kind,
                    pool: Arc::clone(&self.pool),
                })
            }
            None => Err(TransportError::OutOfBounds {
                bar: kind.bar(),
                offset: 0,
                len: 0,
            }),
        }
    }

    /// Return a window to the pool now. Dropping it does the same; this is the
    /// spelling for a caller that wants the release to be visible.
    pub fn free_window(&mut self, window: Window) {
        debug_assert!(
            Arc::ptr_eq(&window.pool, &self.pool),
            "freed a window allocated by a different Device"
        );
        drop(window);
    }

    /// Point a window at a device address in a tile, if it is not already there.
    fn retarget<N: NocId>(
        &mut self,
        window: &Window,
        coord: NocCoord<N>,
        base_address: u64,
    ) -> Result<()> {
        self.check_owned(window)?;
        let config = TlbConfig::unicast(base_address, coord);
        let words = config
            .encode(window.kind)
            .map_err(|_| TransportError::Misaligned {
                bar: window.kind.bar(),
                offset: base_address,
                len: window.kind.size(),
                reason: "device address is not window-aligned",
            })?;
        let shadow = Shadow {
            words,
            readable: config.is_readable(),
        };

        if self.pool().shadow.get(&window.index) == Some(&shadow) {
            return Ok(());
        }
        tlb::write_config(&mut self.transport, window.index, &config)?;
        self.traffic.retargets += 1;
        self.traffic.write_calls += 3;
        self.traffic.bytes_written += 12;
        self.pool().shadow.insert(window.index, shadow);
        Ok(())
    }

    /// Write `data` to `address` in the tile at `coord`, through `window`.
    ///
    /// Splits the transfer so that no single access straddles a window boundary,
    /// retargeting the window as it goes.
    ///
    /// Refuses the local-data-RAM aperture; see [`Device::local_ram_write`].
    pub fn write<N: NocId>(
        &mut self,
        window: &Window,
        coord: NocCoord<N>,
        address: u64,
        data: &[u8],
    ) -> Result<()> {
        refuse_local_ram_aperture(address, data.len())?;
        self.write_unchecked(window, coord, address, data)
    }

    /// [`Device::write`] without the aperture refusal, for the accessors that
    /// have established the access is safe.
    pub(crate) fn write_unchecked<N: NocId>(
        &mut self,
        window: &Window,
        coord: NocCoord<N>,
        address: u64,
        data: &[u8],
    ) -> Result<()> {
        self.transfer(
            window,
            coord,
            address,
            data.len(),
            |dev, bar, off, range| {
                dev.traffic.write_calls += 1;
                dev.traffic.bytes_written += range.len() as u64;
                dev.transport.bar_write(bar, off, &data[range])
            },
        )
    }

    /// Write `data` into a Tensix tile's L1, by the fast path.
    ///
    /// The bulk path ([`Transport::bar_write_bulk`]: write-combining on silicon)
    /// is only for memory, and this is the one way to reach it at a Tensix tile:
    /// the coordinate must have Tensix geometry and the range must lie inside
    /// L1, so no register -- all of which sit at `0xFF..` addresses, or on the
    /// ARC and PCIe tiles -- can be written this way. Use [`Device::write`] for
    /// anything else.
    pub fn l1_write<N: NocId>(
        &mut self,
        window: &Window,
        coord: NocCoord<N>,
        address: u64,
        data: &[u8],
    ) -> Result<()> {
        refuse_non_l1(coord, address, data.len())?;
        self.write_memory(window, coord, address, data)
    }

    /// Read a Tensix tile's L1 by the fast path. See [`Device::l1_write`].
    pub fn l1_read<N: NocId>(
        &mut self,
        window: &Window,
        coord: NocCoord<N>,
        address: u64,
        out: &mut [u8],
    ) -> Result<()> {
        refuse_non_l1(coord, address, out.len())?;
        self.read_memory(window, coord, address, out)
    }

    /// The bulk write, for callers that have established `address` is memory.
    pub(crate) fn write_memory<N: NocId>(
        &mut self,
        window: &Window,
        coord: NocCoord<N>,
        address: u64,
        data: &[u8],
    ) -> Result<()> {
        self.transfer(
            window,
            coord,
            address,
            data.len(),
            |dev, bar, off, range| {
                dev.traffic.write_calls += 1;
                dev.traffic.bytes_written += range.len() as u64;
                dev.transport.bar_write_bulk(bar, off, &data[range])
            },
        )
    }

    /// The bulk read. See [`Device::write_memory`].
    pub(crate) fn read_memory<N: NocId>(
        &mut self,
        window: &Window,
        coord: NocCoord<N>,
        address: u64,
        out: &mut [u8],
    ) -> Result<()> {
        let len = out.len();
        let mut done = 0usize;
        for chunk in self.plan(window, address, len)? {
            self.retarget(window, coord, chunk.window_base)?;
            let bar_offset = tlb::window_bar_offset(window.index).expect("allocated window")
                + chunk.offset_in_window;
            self.traffic.read_calls += 1;
            self.traffic.bytes_read += chunk.len as u64;
            self.transport.bar_read_bulk(
                window.kind.bar(),
                bar_offset,
                &mut out[done..done + chunk.len],
            )?;
            done += chunk.len;
        }
        Ok(())
    }

    /// Read from `address` in the tile at `coord` into `out`, through `window`.
    ///
    /// Refuses the local-data-RAM aperture; see [`Device::local_ram_read`].
    pub fn read<N: NocId>(
        &mut self,
        window: &Window,
        coord: NocCoord<N>,
        address: u64,
        out: &mut [u8],
    ) -> Result<()> {
        refuse_local_ram_aperture(address, out.len())?;
        self.read_unchecked(window, coord, address, out)
    }

    /// [`Device::read`] without the aperture refusal.
    pub(crate) fn read_unchecked<N: NocId>(
        &mut self,
        window: &Window,
        coord: NocCoord<N>,
        address: u64,
        out: &mut [u8],
    ) -> Result<()> {
        // Split borrows: the closure needs `out` mutably while `transfer` needs
        // `self`, so the chunking is done against a raw range and applied here.
        let len = out.len();
        let mut done = 0usize;
        let chunks = self.plan(window, address, len)?;
        for chunk in chunks {
            self.retarget(window, coord, chunk.window_base)?;
            let bar_offset = tlb::window_bar_offset(window.index).expect("allocated window")
                + chunk.offset_in_window;
            self.traffic.read_calls += 1;
            self.traffic.bytes_read += chunk.len as u64;
            self.transport.bar_read(
                window.kind.bar(),
                bar_offset,
                &mut out[done..done + chunk.len],
            )?;
            done += chunk.len;
        }
        debug_assert_eq!(done, len);
        Ok(())
    }

    fn transfer<N: NocId, F>(
        &mut self,
        window: &Window,
        coord: NocCoord<N>,
        address: u64,
        len: usize,
        mut apply: F,
    ) -> Result<()>
    where
        F: FnMut(&mut Self, Bar, u64, std::ops::Range<usize>) -> Result<()>,
    {
        let mut done = 0usize;
        for chunk in self.plan(window, address, len)? {
            self.retarget(window, coord, chunk.window_base)?;
            let bar_offset = tlb::window_bar_offset(window.index).expect("allocated window")
                + chunk.offset_in_window;
            apply(self, window.kind.bar(), bar_offset, done..done + chunk.len)?;
            done += chunk.len;
        }
        debug_assert_eq!(done, len);
        Ok(())
    }

    /// Break a transfer into per-window pieces.
    fn plan(&self, window: &Window, address: u64, len: usize) -> Result<Vec<Chunk>> {
        let size = window.kind.size();
        let mut chunks = Vec::new();
        let mut address = address;
        let mut remaining = len as u64;

        while remaining > 0 {
            let window_base = address & !(size - 1);
            let offset_in_window = address - window_base;
            let here = remaining.min(size - offset_in_window);
            chunks.push(Chunk {
                window_base,
                offset_in_window,
                len: here as usize,
            });
            address = address
                .checked_add(here)
                .ok_or(TransportError::OutOfBounds {
                    bar: window.kind.bar(),
                    offset: address,
                    len: remaining,
                })?;
            remaining -= here;
        }
        Ok(chunks)
    }

    /// Read one dword from a tile.
    pub fn read32<N: NocId>(
        &mut self,
        window: &Window,
        coord: NocCoord<N>,
        address: u64,
    ) -> Result<u32> {
        let mut buf = [0u8; 4];
        self.read(window, coord, address, &mut buf)?;
        Ok(u32::from_le_bytes(buf))
    }

    /// Write one dword to a tile.
    pub fn write32<N: NocId>(
        &mut self,
        window: &Window,
        coord: NocCoord<N>,
        address: u64,
        value: u32,
    ) -> Result<()> {
        self.write(window, coord, address, &value.to_le_bytes())
    }

    /// Identify the tile at `coord`, or `None` if nothing answers there.
    ///
    /// Reads `NOC_ENDPOINT_ID` and `NIU_CFG_0` through NoC #0's NIU, the same probe
    /// `ethdump.c:462-474` uses to walk the grid.
    pub fn probe_tile<N: NocId>(
        &mut self,
        window: &Window,
        coord: NocCoord<N>,
    ) -> Result<Option<Tile<N>>> {
        let endpoint = self.read32(window, coord, niu::NOC0_BASE + niu::NOC_ENDPOINT_ID)?;
        if endpoint == 0 || endpoint == u32::MAX {
            return Ok(None);
        }
        let cfg = self.read32(window, coord, niu::NOC0_BASE + niu::NIU_CFG_0)?;
        Ok(Some(Tile {
            coord,
            kind: TileType::from_endpoint_id(endpoint),
            index: tt_isa::noc::endpoint_tile_index(endpoint),
            harvested: cfg & niu::NIU_CFG_0_HARVESTED != 0,
        }))
    }

    /// Walk a rectangle of the NoC grid, reporting every tile that answers.
    ///
    /// Discovery rather than hardcoding: which tiles are present depends on
    /// harvesting, and whether coordinates are translated depends on `NIU_CFG_0`
    /// bit 14. Probing survives both.
    pub fn discover_tiles<N: NocId>(
        &mut self,
        window: &Window,
        xs: std::ops::Range<u8>,
        ys: std::ops::Range<u8>,
    ) -> Result<Vec<Tile<N>>> {
        let mut found = Vec::new();
        for y in ys.clone() {
            for x in xs.clone() {
                let Some(coord) = NocCoord::<N>::new(x, y) else {
                    continue;
                };
                if let Some(tile) = self.probe_tile(window, coord)? {
                    found.push(tile);
                }
            }
        }
        Ok(found)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Chunk {
    /// Window-aligned device address this piece needs the window pointed at.
    window_base: u64,
    offset_in_window: u64,
    len: usize,
}

/// Number of 2 MiB windows available to callers, after the kernel reservation.
pub const USABLE_2MIB_WINDOWS: u16 = NUM_2MIB_WINDOWS - 1;

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::{Bar, ConfigOffset};
    use tt_isa::noc::Noc0;

    /// A transport that models just enough of the chip to test windowing: it
    /// decodes TLB configuration writes and routes data accesses to a per-tile
    /// sparse memory, exactly as the hardware's address translation would.
    ///
    /// Without the decode step two windows onto the same tile address would appear
    /// to be different memory, and a round-trip through two windows -- the whole
    /// point of the step 2 gate -- could not be expressed here at all. The
    /// authoritative version of that gate runs against ttsim; this one keeps the
    /// chunking and shadowing logic testable without a simulator.
    #[derive(Default)]
    pub(super) struct FakeTransport {
        writes: Vec<(Bar, u64, usize)>,
        reads: Vec<(Bar, u64, usize)>,
        /// Every data (non-config) access in order: `(is_write, tile address,
        /// len)`. The order is what a fence is about.
        pub(super) log: Vec<(bool, u64, usize)>,
        /// Fail every data (non-config) write, for "no read-back after a failure".
        pub(super) fail_data_writes: bool,
        /// Raw configuration words, indexed by window.
        tlb_cfg: BTreeMap<u16, [u32; 3]>,
        /// Tile memory, keyed by (packed coordinate, device address).
        pub(super) mem: BTreeMap<(u16, u64), u8>,
    }

    impl FakeTransport {
        /// Reverse of `TlbConfig::encode` for a 2 MiB window: recover the tile and
        /// device address a BAR0 offset resolves to.
        fn translate(&self, offset: u64) -> Option<(u16, u64)> {
            let index = (offset / WindowKind::TwoMib.size()) as u16;
            let offset_in_window = offset % WindowKind::TwoMib.size();
            let w = self.tlb_cfg.get(&index)?;
            let local_offset = (w[0] as u64) | (((w[1] & 0x7FF) as u64) << 32);
            let x = ((w[1] >> 11) & 0x3F) as u16;
            let y = ((w[1] >> 17) & 0x3F) as u16;
            Some((x | (y << 6), (local_offset << 21) | offset_in_window))
        }

        fn is_config(bar: Bar, offset: u64) -> bool {
            bar == Bar::Bar0 && offset >= tlb::CONFIG_BASE
        }
    }

    impl Transport for FakeTransport {
        fn bar_read(&mut self, bar: Bar, offset: u64, dst: &mut [u8]) -> Result<()> {
            self.reads.push((bar, offset, dst.len()));
            assert!(
                !Self::is_config(bar, offset),
                "TLB config registers are write-only"
            );
            let (tile, addr) = self.translate(offset).expect("window is configured");
            self.log.push((false, addr, dst.len()));
            for (i, b) in dst.iter_mut().enumerate() {
                *b = self.mem.get(&(tile, addr + i as u64)).copied().unwrap_or(0);
            }
            Ok(())
        }

        fn bar_write(&mut self, bar: Bar, offset: u64, src: &[u8]) -> Result<()> {
            self.writes.push((bar, offset, src.len()));
            if Self::is_config(bar, offset) {
                let reg = (offset - tlb::CONFIG_BASE) / 4;
                let (index, word) = ((reg / 3) as u16, (reg % 3) as usize);
                assert_eq!(src.len(), 4, "config registers take dwords");
                self.tlb_cfg.entry(index).or_default()[word] =
                    u32::from_le_bytes(src.try_into().unwrap());
                return Ok(());
            }
            if self.fail_data_writes {
                return Err(TransportError::Io(std::io::Error::other("injected")));
            }
            let (tile, addr) = self.translate(offset).expect("window is configured");
            self.log.push((true, addr, src.len()));
            for (i, b) in src.iter().enumerate() {
                self.mem.insert((tile, addr + i as u64), *b);
            }
            Ok(())
        }

        fn config_read32(&mut self, offset: ConfigOffset) -> Result<u32> {
            Ok(match offset {
                ConfigOffset::VendorDevice => 0xB140_1E52,
                _ => 0,
            })
        }

        fn tick(&mut self, _n: u32) {}

        fn is_simulated(&self) -> bool {
            true
        }
    }

    pub(super) fn device() -> Device<FakeTransport> {
        Device::open(FakeTransport::default()).unwrap()
    }

    pub(super) fn c(x: u8, y: u8) -> NocCoord<Noc0> {
        NocCoord::new(x, y).unwrap()
    }

    /// Data writes only; the three dwords of TLB configuration are filtered out.
    fn data_writes(d: &Device<FakeTransport>) -> Vec<(Bar, u64, usize)> {
        d.transport
            .writes
            .iter()
            .copied()
            .filter(|(bar, off, _)| !(*bar == Bar::Bar0 && *off >= tlb::CONFIG_BASE))
            .collect()
    }

    fn config_write_count(d: &Device<FakeTransport>) -> usize {
        d.transport
            .writes
            .iter()
            .filter(|(bar, off, _)| *bar == Bar::Bar0 && *off >= tlb::CONFIG_BASE)
            .count()
    }

    #[test]
    fn the_bulk_path_refuses_anything_but_tensix_l1() {
        let mut d = device();
        let w = d.alloc_window(WindowKind::TwoMib).unwrap();
        let before = d.traffic();
        // The ARC tile, a DRAM column, a Tensix register, and L1's last byte + 1.
        assert!(d.l1_write(&w, c(8, 0), 0x1000, &[0; 4]).is_err());
        assert!(d.l1_write(&w, c(0, 5), 0x1000, &[0; 4]).is_err());
        assert!(d.l1_write(&w, c(3, 4), 0xFFB1_21B0, &[0; 4]).is_err());
        assert!(d
            .l1_read(&w, c(3, 4), tensix::L1_SIZE - 2, &mut [0; 4])
            .is_err());
        assert_eq!(d.traffic(), before, "nothing may be sent");
        d.l1_write(&w, c(3, 4), tensix::L1_SIZE - 4, &[1, 2, 3, 4])
            .unwrap();
        let mut back = [0u8; 4];
        d.l1_read(&w, c(3, 4), tensix::L1_SIZE - 4, &mut back)
            .unwrap();
        assert_eq!(back, [1, 2, 3, 4]);
    }

    #[test]
    fn traffic_counts_every_bar_access() {
        let mut d = device();
        assert_eq!(d.traffic(), Traffic::default());
        let w = d.alloc_window(WindowKind::TwoMib).unwrap();

        // Across a window boundary: two data writes, two retargets.
        let size = WindowKind::TwoMib.size();
        d.write(&w, c(1, 2), size - 8, &[0xAA; 16]).unwrap();
        let t = d.traffic();
        assert_eq!(t.retargets, 2);
        assert_eq!(t.bytes_written, 16 + 2 * 12);
        assert_eq!(t.write_calls as usize, d.transport.writes.len());
        assert_eq!(
            t.bytes_written,
            d.transport.writes.iter().map(|w| w.2 as u64).sum::<u64>()
        );

        // A read in the window it is already pointed at: no retarget.
        let mut out = [0u8; 8];
        d.read(&w, c(1, 2), size, &mut out).unwrap();
        let delta = d.traffic() - t;
        assert_eq!(
            delta,
            Traffic {
                bytes_read: 8,
                read_calls: 1,
                ..Traffic::default()
            }
        );
    }

    #[test]
    fn kernel_window_is_never_allocated() {
        let mut d = device();
        // Drain every 2 MiB window, holding them: a dropped window goes back.
        let held: Vec<Window> = (0..USABLE_2MIB_WINDOWS)
            .map(|_| d.alloc_window(WindowKind::TwoMib).unwrap())
            .collect();
        let seen: Vec<u16> = held.iter().map(Window::index).collect();
        assert!(!seen.contains(&KERNEL_RESERVED_WINDOW));
        assert_eq!(seen.len(), 201);
        // And the pool is now empty for that geometry, but not for the other.
        assert!(d.alloc_window(WindowKind::TwoMib).is_err());
        assert!(d.alloc_window(WindowKind::FourGib).is_ok());
    }

    #[test]
    fn freed_windows_are_reused_and_lose_their_shadow() {
        let mut d = device();
        let w = d.alloc_window(WindowKind::TwoMib).unwrap();
        d.write(&w, c(1, 2), 0, &[0xAA; 4]).unwrap();
        let before = config_write_count(&d);
        assert!(before > 0);

        let index = w.index();
        d.free_window(w);
        let w2 = d.alloc_window(WindowKind::TwoMib).unwrap();
        assert_eq!(
            w2.index(),
            index,
            "lowest free window should come back first"
        );

        // The new holder must not inherit the old shadow, or it would skip the
        // reconfiguration it needs.
        d.write(&w2, c(1, 2), 0, &[0xAA; 4]).unwrap();
        assert!(
            config_write_count(&d) > before,
            "reconfiguration must be repeated"
        );
    }

    #[test]
    fn a_dropped_window_returns_to_the_pool() {
        let mut d = device();
        // Several passes over the whole pool, dropping each window rather than
        // freeing it. Before `Window` had a `Drop`, the second pass failed.
        for _ in 0..3 {
            let held: Vec<Window> = (0..USABLE_2MIB_WINDOWS)
                .map(|_| d.alloc_window(WindowKind::TwoMib).unwrap())
                .collect();
            assert!(d.alloc_window(WindowKind::TwoMib).is_err());
            drop(held);
        }
        // And a dropped window loses its shadow, as a freed one does.
        let w = d.alloc_window(WindowKind::TwoMib).unwrap();
        d.write(&w, c(1, 2), 0, &[1; 4]).unwrap();
        let before = config_write_count(&d);
        drop(w);
        let w = d.alloc_window(WindowKind::TwoMib).unwrap();
        d.write(&w, c(1, 2), 0, &[1; 4]).unwrap();
        assert!(
            config_write_count(&d) > before,
            "reconfiguration must repeat"
        );
    }

    #[test]
    fn a_window_from_another_device_is_refused() {
        let mut a = device();
        let mut b = device();
        let wa = a.alloc_window(WindowKind::TwoMib).unwrap();
        let wb = b.alloc_window(WindowKind::TwoMib).unwrap();
        // Same index, different chips' pools.
        assert_eq!(wa.index(), wb.index());
        assert_ne!(wa, wb);
        let before = b.transport.writes.len();
        let e = b.write32(&wa, c(1, 2), 0x100, 1).unwrap_err();
        assert!(matches!(e, TransportError::Hazard { .. }), "{e}");
        let e = b.read32(&wa, c(1, 2), 0x100).unwrap_err();
        assert!(matches!(e, TransportError::Hazard { .. }), "{e}");
        assert_eq!(b.transport.writes.len(), before, "nothing may be sent");
        b.write32(&wb, c(1, 2), 0x100, 1).unwrap();
    }

    #[test]
    fn transfers_split_at_window_boundaries() {
        let mut d = device();
        let w = d.alloc_window(WindowKind::TwoMib).unwrap();
        let size = WindowKind::TwoMib.size();

        // Start four bytes before a window boundary, run eight bytes past it.
        d.write(&w, c(1, 2), size - 4, &[0x5A; 8]).unwrap();
        let writes = data_writes(&d);
        assert_eq!(
            writes.len(),
            2,
            "should have split into two accesses: {writes:?}"
        );
        assert_eq!(writes[0].2, 4, "first piece runs to the window boundary");
        assert_eq!(writes[1].2, 4, "second piece starts the next window");
        // The second piece lands at the start of the window aperture, because the
        // window was retargeted rather than the offset being advanced.
        assert_eq!(writes[1].1, tlb::window_bar_offset(w.index()).unwrap());
    }

    #[test]
    fn a_transfer_spanning_many_windows_is_fully_chunked() {
        let mut d = device();
        let w = d.alloc_window(WindowKind::TwoMib).unwrap();
        let size = WindowKind::TwoMib.size() as usize;
        let data = vec![7u8; size * 2 + 16];
        d.write(&w, c(3, 4), (size - 8) as u64, &data).unwrap();

        let writes = data_writes(&d);
        assert_eq!(writes.iter().map(|w| w.2).sum::<usize>(), data.len());
        assert_eq!(writes.len(), 4, "8 + 2MiB + 2MiB + 8 = four accesses");
        assert!(writes.iter().all(|(_, _, len)| *len <= size));
    }

    #[test]
    fn aligned_transfers_are_not_split() {
        let mut d = device();
        let w = d.alloc_window(WindowKind::TwoMib).unwrap();
        d.write(&w, c(1, 1), 0x1000, &[0u8; 4096]).unwrap();
        assert_eq!(data_writes(&d).len(), 1);
    }

    #[test]
    fn repeated_access_to_one_window_reconfigures_once() {
        let mut d = device();
        let w = d.alloc_window(WindowKind::TwoMib).unwrap();
        d.write32(&w, c(1, 2), 0x100, 1).unwrap();
        let after_first = config_write_count(&d);
        assert_eq!(after_first, 3, "three dwords of configuration");

        // Same tile, same window -- no reconfiguration.
        d.write32(&w, c(1, 2), 0x200, 2).unwrap();
        assert_eq!(config_write_count(&d), after_first);

        // Different tile -- reconfiguration.
        d.write32(&w, c(5, 6), 0x200, 3).unwrap();
        assert_eq!(config_write_count(&d), after_first + 3);
    }

    #[test]
    fn round_trip_through_two_different_windows() {
        let mut d = device();
        let a = d.alloc_window(WindowKind::TwoMib).unwrap();
        let b = d.alloc_window(WindowKind::TwoMib).unwrap();
        assert_ne!(a.index(), b.index());

        let payload: Vec<u8> = (0..255u8).collect();
        d.write(&a, c(2, 3), 0x4000, &payload).unwrap();
        let mut back = vec![0u8; payload.len()];
        d.read(&b, c(2, 3), 0x4000, &mut back).unwrap();
        assert_eq!(
            back, payload,
            "a second window must see the first window's writes"
        );
    }

    #[test]
    fn probe_reports_absent_tiles_as_none() {
        let mut d = device();
        let w = d.alloc_window(WindowKind::TwoMib).unwrap();
        // The fake serves zeroes, which stands in for "nothing there".
        assert_eq!(d.probe_tile(&w, c(9, 9)).unwrap(), None);
    }

    // -- The local-data-RAM aperture ------------------------------------------

    fn gate_tile() -> NocCoord<Noc0> {
        NocCoord::new(3, 4).unwrap()
    }

    #[test]
    fn plain_accesses_to_the_local_ram_aperture_are_refused() {
        let mut dev = device();
        let w = dev.alloc_window(WindowKind::TwoMib).unwrap();
        let t = gate_tile();
        for core in Core::ALL {
            let at = core.local_data_ram_noc_address();
            let e = dev.write32(&w, t, at, 1).unwrap_err();
            assert!(matches!(e, TransportError::Hazard { .. }), "{e}");
            let e = dev.read32(&w, t, at).unwrap_err();
            assert!(matches!(e, TransportError::Hazard { .. }), "{e}");
        }
        // A transfer that merely overlaps the aperture's first byte is caught too.
        let e = dev
            .write(&w, t, tensix::LOCAL_DATA_RAM_NOC_BASE - 2, &[0; 4])
            .unwrap_err();
        assert!(matches!(e, TransportError::Hazard { .. }), "{e}");
        // And its neighbours are not: the debug registers and the NoC 0 NIU.
        dev.write32(&w, t, tensix::LOCAL_DATA_RAM_NOC_BASE - 4, 0)
            .unwrap();
        dev.read32(&w, t, tensix::LOCAL_DATA_RAM_NOC_END).unwrap();
        assert!(dev.transport.writes.iter().all(|&(_, off, _)| {
            // Nothing that reached the transport landed in the aperture.
            dev.transport
                .translate(off)
                .is_none_or(|(_, a)| !touches_local_ram_aperture(a, 1))
        }));
    }

    #[test]
    fn local_ram_is_refused_while_its_core_is_held_in_reset() {
        let mut dev = device();
        let w = dev.alloc_window(WindowKind::TwoMib).unwrap();
        let t = gate_tile();
        dev.set_core_reset(&w, t, Core::T1, true).unwrap();
        let before = dev.transport.writes.len();
        let e = dev
            .local_ram_write(&w, t, Core::T1, 0, &[1, 2, 3, 4])
            .unwrap_err();
        assert!(matches!(e, TransportError::Hazard { .. }), "{e}");
        assert_eq!(dev.transport.writes.len(), before, "nothing may be sent");
        let mut buf = [0u8; 4];
        let e = dev
            .local_ram_read(&w, t, Core::T1, 0, &mut buf)
            .unwrap_err();
        assert!(matches!(e, TransportError::Hazard { .. }), "{e}");
    }

    #[test]
    fn local_ram_round_trips_once_its_core_is_running() {
        let mut dev = device();
        let w = dev.alloc_window(WindowKind::TwoMib).unwrap();
        let t = gate_tile();
        dev.set_core_reset(&w, t, Core::T1, true).unwrap();
        dev.park_core(&w, t, Core::T1, 0x4_0000).unwrap();
        assert!(!dev.is_core_in_reset(&w, t, Core::T1).unwrap());
        dev.local_ram_write(&w, t, Core::T1, 16, &[9, 8, 7, 6])
            .unwrap();
        let mut back = [0u8; 4];
        dev.local_ram_read(&w, t, Core::T1, 16, &mut back).unwrap();
        assert_eq!(back, [9, 8, 7, 6]);
        // Where it landed: T1's slow-path window.
        let at = Core::T1.local_data_ram_noc_address() + 16;
        assert_eq!(dev.transport.mem.get(&(t.packed(), at)), Some(&9));
    }

    #[test]
    fn local_ram_refuses_the_upper_half_of_a_t_core_window() {
        let mut dev = device();
        let w = dev.alloc_window(WindowKind::TwoMib).unwrap();
        let t = gate_tile();
        let size = u64::from(Core::T0.local_data_ram_size());
        let e = dev
            .local_ram_write(&w, t, Core::T0, size - 4, &[0; 8])
            .unwrap_err();
        assert!(matches!(e, TransportError::Hazard { .. }), "{e}");
        // B's RAM is 8 KiB, so the same offset is fine there.
        dev.local_ram_write(&w, t, Core::B, size - 4, &[0; 8])
            .unwrap();
    }

    // -- Power policy ----------------------------------------------------------

    /// A simulated transport never sends power messages, under either policy;
    /// and the policy is what `open` defaults to.
    #[test]
    fn power_policy_is_busy_by_default_and_inert_on_the_simulator() {
        assert_eq!(PowerPolicy::default(), PowerPolicy::Busy);
        let dev = device();
        assert!(!dev.busy, "no ARC to message on a simulated transport");
        let manual =
            Device::open_with_power(FakeTransport::default(), PowerPolicy::Manual).unwrap();
        assert!(!manual.busy);
    }
}
