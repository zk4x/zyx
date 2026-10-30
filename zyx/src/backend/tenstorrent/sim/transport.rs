//! [`Transport`] over `libttsim`, with a validation layer in front of it.
//!
//! # Why the validation layer exists
//!
//! `libttsim` decodes a fixed map of BAR offsets. An access outside it, or one that
//! violates a region's alignment or direction rule, is not an error — it prints a
//! message and calls `_Exit`. There is no way to catch that in-process.
//!
//! So every access is checked here first, against the same map, and rejected as a
//! [`TransportError`] before the call happens. The map below is therefore a mirror
//! of libttsim's decoder, and drift between them shows up as either a spurious
//! rejection (harmless, loud) or a dead process (not). `Simulator::verify` guards
//! the most likely form of drift by confirming the BAR bases still match.

use super::super::dev::{Bar, ConfigOffset, Result, Transport, TransportError};
use super::sys::Lib;

use tt_isa::noc::ChipId;

use super::{chip_bar_base, chip_bdf};

/// Size of a BAR0 TLB window.
pub const TLB_2MIB_SIZE: u64 = 2 * 1024 * 1024;
/// Size of a BAR4 TLB window.
pub const TLB_4GIB_SIZE: u64 = 4 * 1024 * 1024 * 1024;

/// Number of 2 MiB windows mapped into BAR0 (indices 0..=201).
pub const NUM_BAR0_WINDOWS: u64 = 202;
/// Number of 4 GiB windows mapped into BAR4 (indices 202..=209).
pub const NUM_BAR4_WINDOWS: u64 = 8;

/// End of the TLB window region in BAR0: 202 × 2 MiB = 404 MiB.
const BAR0_WINDOWS_END: u64 = NUM_BAR0_WINDOWS * TLB_2MIB_SIZE;

/// Base of the TLB configuration array in BAR0 (`HostToDeviceTLBs.md:14`).
pub const TLB_CONFIG_BASE: u64 = 0x1FC0_0000;
/// Three `u32` per window, 210 windows.
const TLB_CONFIG_REGS: u64 = 210 * 3;
/// One past the last byte of `windows[210]`.
const TLB_CONFIG_END: u64 = TLB_CONFIG_BASE + TLB_CONFIG_REGS * 4;

// The specification places `uint32_t strided[32]` immediately after `windows[210]`,
// running to 0x1FC0_0A58. libttsim does not decode it: its TLB-config region stops
// at the end of `windows[]`. Non-rectangular multicast therefore cannot be
// configured on the simulator at all, and an attempt to do so would terminate the
// process. Rejected here instead.
const STRIDED_BASE: u64 = TLB_CONFIG_END;
const STRIDED_END: u64 = STRIDED_BASE + 32 * 4;

/// PCIe NIU #0 configuration/status, read-only (`PCIExpressTile/README.md:80`).
const NIU0_BASE: u64 = 0x1FD0_4000;
const NIU0_END: u64 = 0x1FD0_6000;
/// PCIe NIU #1 configuration/status, read-only (`README.md:82`).
const NIU1_BASE: u64 = 0x1FD1_4000;
const NIU1_END: u64 = 0x1FD1_6000;

/// BAR2 DMA read-channel-0 registers.
const BAR2_DMA_BASE: u64 = 0x100;
const BAR2_DMA_END: u64 = 0x1AC;
/// BAR2 iATU region.
const BAR2_IATU_BASE: u64 = 0x1000;
const BAR2_IATU_END: u64 = 0x3000;

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
enum Dir {
    Read,
    Write,
}

/// A [`Transport`] backed by the in-process simulator.
///
/// Borrowed from a [`Simulator`](super::Simulator) rather than constructed, so that
/// the singleton guarantee carries through: there is one simulator, so there is at
/// most one live transport.
pub struct LibTtsim<'a> {
    lib: &'a Lib,
    /// Which chip this transport addresses. Constant for its whole life, and the
    /// only thing that distinguishes one transport from another.
    chip: ChipId,
}

impl<'a> LibTtsim<'a> {
    /// Deliberately `pub(crate)`: the only public constructors go through
    /// `Simulator`, which has confirmed the chip is present. A transport naming
    /// a chip that is not there would put every access in an unpopulated window,
    /// which is fatal rather than an error.
    pub(crate) fn new(lib: &'a Lib, chip: ChipId) -> Self {
        LibTtsim { lib, chip }
    }

    /// Absolute simulator-internal physical address for a BAR offset.
    ///
    /// The only chip-aware code in this workspace. libttsim picks the chip out of
    /// the address and subtracts the stride before its intra-chip decode, so
    /// everything below this line — `validate` included — is chip-independent.
    fn paddr(&self, bar: Bar, offset: u64) -> u64 {
        chip_bar_base(self.chip, bar) + offset
    }

    /// Reject anything libttsim would refuse, before it gets the chance to `_Exit`.
    fn validate(bar: Bar, offset: u64, len: u64, dir: Dir) -> Result<()> {
        let oob = || TransportError::OutOfBounds { bar, offset, len };
        let bad = |reason| TransportError::Misaligned {
            bar,
            offset,
            len,
            reason,
        };

        if len == 0 {
            return Err(bad("zero-length accesses are not meaningful"));
        }
        let end = offset.checked_add(len).ok_or_else(oob)?;
        if end > bar.size() {
            return Err(oob());
        }

        match bar {
            Bar::Bar0 => {
                if end <= BAR0_WINDOWS_END {
                    // A window access may not straddle the end of its window. The
                    // caller is expected to have split the transfer already; this
                    // catches the case where it did not.
                    let window = offset / TLB_2MIB_SIZE;
                    if (end - 1) / TLB_2MIB_SIZE != window {
                        return Err(bad("access straddles a 2 MiB TLB window boundary"));
                    }
                    Ok(())
                } else if (STRIDED_BASE..STRIDED_END).contains(&offset) {
                    Err(bad(
                        "libttsim does not implement the TLB `strided[32]` array, so \
                         non-rectangular multicast cannot be configured on the simulator",
                    ))
                } else if offset >= TLB_CONFIG_BASE && end <= TLB_CONFIG_END {
                    if dir == Dir::Read {
                        return Err(bad("TLB configuration registers are write-only"));
                    }
                    require_dword(offset, len).map_err(bad)
                } else if (offset >= NIU0_BASE && end <= NIU0_END)
                    || (offset >= NIU1_BASE && end <= NIU1_END)
                {
                    if dir == Dir::Write {
                        return Err(bad("PCIe NIU registers are read-only"));
                    }
                    require_dword(offset, len).map_err(bad)
                } else {
                    Err(bad("not a region libttsim decodes in BAR0"))
                }
            }
            Bar::Bar2 => {
                let in_dma = offset >= BAR2_DMA_BASE && end <= BAR2_DMA_END;
                let in_iatu = offset >= BAR2_IATU_BASE && end <= BAR2_IATU_END;
                if !in_dma && !in_iatu {
                    return Err(bad("not a region libttsim decodes in BAR2"));
                }
                require_dword(offset, len).map_err(bad)
            }
            Bar::Bar4 => {
                let window = offset / TLB_4GIB_SIZE;
                if window >= NUM_BAR4_WINDOWS {
                    return Err(oob());
                }
                if (end - 1) / TLB_4GIB_SIZE != window {
                    return Err(bad("access straddles a 4 GiB TLB window boundary"));
                }
                Ok(())
            }
        }
    }
}

/// Reject TLB configuration libttsim refuses to decode.
///
/// libttsim implements a subset of the window configuration the specification
/// describes: for the third dword it accepts only `y_start`, `mcast` and `ordering`,
/// requiring every other bit to be zero. In specification terms that means a window
/// on the simulator must target **NoC #0**, and may not use `linked`, `static_vc`,
/// `static_vc_buddy` or `static_vc_class`.
///
/// None of those restrictions bite for the baseline -- `linked` is never safe from
/// the host anyway, and NoC #0 is the natural choice -- but a NoC #1 window would
/// otherwise terminate the process with a message about a config word rather than
/// about the NoC, which is a long way from the cause.
fn validate_tlb_config_word(offset: u64, value: u32) -> Result<()> {
    let reg = (offset - TLB_CONFIG_BASE) / 4;
    if reg % 3 != 2 {
        return Ok(());
    }
    // Bits 0..=2 y_start[5:3], bit 5 mcast, bits 6..=7 ordering.
    const DECODED: u32 = 0xE7;
    if value & !DECODED == 0 {
        return Ok(());
    }
    let reason = if value & (1 << 3) != 0 {
        "libttsim only decodes NoC #0 TLB windows; this configuration selects NoC #1"
    } else if value & (1 << 8) != 0 {
        "the TLB `linked` bit is never safe to set from the host"
    } else {
        "libttsim does not decode static-VC fields in TLB window configuration"
    };
    Err(TransportError::Misaligned {
        bar: Bar::Bar0,
        offset,
        len: 4,
        reason,
    })
}

/// Regions that libttsim decodes a dword at a time reject anything else outright.
fn require_dword(offset: u64, len: u64) -> std::result::Result<(), &'static str> {
    if len != 4 {
        Err("this region accepts 4-byte accesses only")
    } else if offset % 4 != 0 {
        Err("this region requires 4-byte alignment")
    } else {
        Ok(())
    }
}

impl Transport for LibTtsim<'_> {
    fn host_memory(&mut self, len: usize) -> Result<Box<dyn super::super::dev::HostMemory>> {
        Ok(Box::new(super::host::SimHostMemory::new(len)))
    }

    fn bar_read(&mut self, bar: Bar, offset: u64, dst: &mut [u8]) -> Result<()> {
        Self::validate(bar, offset, dst.len() as u64, Dir::Read)?;
        // SAFETY: the access has been validated against libttsim's decode map, and
        // `dst` is a valid writable slice of exactly `len` bytes.
        unsafe {
            (self.lib.pci_mem_rd_bytes)(
                self.paddr(bar, offset),
                dst.as_mut_ptr().cast(),
                dst.len() as u32,
            )
        };
        Ok(())
    }

    fn bar_write(&mut self, bar: Bar, offset: u64, src: &[u8]) -> Result<()> {
        Self::validate(bar, offset, src.len() as u64, Dir::Write)?;
        if (TLB_CONFIG_BASE..TLB_CONFIG_END).contains(&offset) {
            let value = u32::from_le_bytes(src.try_into().expect("validated as a dword"));
            validate_tlb_config_word(offset, value)?;
        }
        // SAFETY: as above; `src` is a valid readable slice of exactly `len` bytes.
        unsafe {
            (self.lib.pci_mem_wr_bytes)(
                self.paddr(bar, offset),
                src.as_ptr().cast(),
                src.len() as u32,
            )
        };
        Ok(())
    }

    fn config_read32(&mut self, offset: ConfigOffset) -> Result<u32> {
        // `ConfigOffset` is a closed enum of exactly the offsets libttsim decodes,
        // so there is nothing further to validate.
        // SAFETY: the simulator is initialized and the offset is decodable.
        Ok(unsafe { (self.lib.pci_config_rd32)(chip_bdf(self.chip), offset as u32) })
    }

    /// Advance the whole simulator.
    ///
    /// On a multi-chip build `libttsim_clock` steps every chip on a single
    /// global timebase, so this is not per-chip: ticking one transport per chip
    /// in a loop advances time once per chip. Time belongs to the
    /// [`Simulator`](super::Simulator), not to a transport.
    fn tick(&mut self, n: u32) {
        // SAFETY: initialized, and `!Send` keeps this on one thread.
        unsafe { (self.lib.clock)(n) }
    }

    fn is_simulated(&self) -> bool {
        true
    }

    fn chip(&self) -> ChipId {
        self.chip
    }
}

/// Chip 0's BAR4 ends exactly where the next chip's windows begin, so
/// `validate`'s `end > bar.size()` rejection is also what keeps one chip's
/// accesses out of the next chip's 64 GiB window. That is why `validate` needs
/// to know nothing about chips.
const _: () =
    assert!(super::expected_bar_base(Bar::Bar4) + Bar::Bar4.size() == super::PER_CHIP_PADDR_STRIDE);

#[cfg(test)]
mod tests {
    use super::*;

    #[track_caller]
    fn rejected(bar: Bar, offset: u64, len: u64, dir: Dir) -> String {
        LibTtsim::validate(bar, offset, len, dir)
            .expect_err("expected this access to be rejected")
            .to_string()
    }

    #[track_caller]
    fn allowed(bar: Bar, offset: u64, len: u64, dir: Dir) {
        LibTtsim::validate(bar, offset, len, dir).expect("expected this access to be allowed");
    }

    #[test]
    fn window_accesses_must_not_straddle() {
        // Entirely inside window 0.
        allowed(Bar::Bar0, 0, 4096, Dir::Write);
        // Ends exactly at the window boundary.
        allowed(Bar::Bar0, TLB_2MIB_SIZE - 4, 4, Dir::Write);
        // Crosses into window 1.
        assert!(rejected(Bar::Bar0, TLB_2MIB_SIZE - 2, 4, Dir::Write).contains("straddles"));
        // Same rule, 4 GiB windows.
        allowed(Bar::Bar4, TLB_4GIB_SIZE - 8, 8, Dir::Read);
        assert!(rejected(Bar::Bar4, TLB_4GIB_SIZE - 2, 4, Dir::Read).contains("straddles"));
    }

    #[test]
    fn tlb_config_is_write_only_and_dword_sized() {
        allowed(Bar::Bar0, TLB_CONFIG_BASE, 4, Dir::Write);
        assert!(rejected(Bar::Bar0, TLB_CONFIG_BASE, 4, Dir::Read).contains("write-only"));
        assert!(rejected(Bar::Bar0, TLB_CONFIG_BASE, 8, Dir::Write).contains("4-byte accesses"));
        assert!(rejected(Bar::Bar0, TLB_CONFIG_BASE + 2, 4, Dir::Write).contains("alignment"));

        // Last register of windows[210] is in range; one past it is not.
        allowed(Bar::Bar0, TLB_CONFIG_END - 4, 4, Dir::Write);
        assert!(rejected(Bar::Bar0, TLB_CONFIG_END, 4, Dir::Write).contains("strided"));
    }

    #[test]
    fn strided_array_is_named_as_unimplemented() {
        // The spec says this array exists; libttsim does not decode it. The error
        // must say so, because "not a region libttsim decodes" would send the
        // reader looking for a bug in their offset arithmetic.
        let msg = rejected(Bar::Bar0, STRIDED_BASE, 4, Dir::Write);
        assert!(msg.contains("strided"), "{msg}");
        assert!(msg.contains("multicast"), "{msg}");
    }

    #[test]
    fn niu_registers_are_read_only() {
        allowed(Bar::Bar0, NIU0_BASE, 4, Dir::Read);
        allowed(Bar::Bar0, NIU1_BASE, 4, Dir::Read);
        assert!(rejected(Bar::Bar0, NIU0_BASE, 4, Dir::Write).contains("read-only"));
    }

    #[test]
    fn undecoded_bar0_holes_are_rejected() {
        // The reserved span between the windows and the TLB config array.
        assert!(rejected(Bar::Bar0, 0x1A00_0000, 4, Dir::Read).contains("decodes"));
        // The ARC aperture, which exists on Wormhole but not on Blackhole.
        assert!(rejected(Bar::Bar0, 0x1FE0_0000, 4, Dir::Read).contains("decodes"));
    }

    #[test]
    fn bar2_is_dword_only_and_sparse() {
        allowed(Bar::Bar2, BAR2_IATU_BASE, 4, Dir::Write);
        allowed(Bar::Bar2, BAR2_DMA_BASE, 4, Dir::Write);
        assert!(rejected(Bar::Bar2, 0, 4, Dir::Read).contains("decodes"));
        assert!(rejected(Bar::Bar2, BAR2_IATU_BASE, 2, Dir::Read).contains("4-byte accesses"));
    }

    #[test]
    fn out_of_bounds_is_distinguished_from_undecoded() {
        let msg = rejected(Bar::Bar0, Bar::Bar0.size(), 4, Dir::Read);
        assert!(msg.contains("outside the BAR"), "{msg}");
        // Overflow in offset + len must not wrap into a valid-looking range.
        let msg = rejected(Bar::Bar0, u64::MAX, 8, Dir::Read);
        assert!(msg.contains("outside the BAR"), "{msg}");
    }

    #[test]
    fn zero_length_is_rejected() {
        assert!(rejected(Bar::Bar0, 0, 0, Dir::Read).contains("zero-length"));
    }
}
