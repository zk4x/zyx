//! How the host reaches the chip.
//!
//! Three implementations are anticipated: the in-process simulator (`tt-ttsim`),
//! the real kernel driver via `/dev/tenstorrent/N`, and that same driver inside a
//! QEMU guest. Everything above this trait is written once.

use std::fmt;

use tt_isa::noc::ChipId;

/// A PCIe base address register.
///
/// Blackhole exposes three (`PCIExpressTile/README.md:36`). BAR2 is the DBI /
/// iATU region and is not used by the baseline, but it is named here so that a
/// caller cannot reach it by passing a bare index.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum Bar {
    /// 512 MiB. TLB windows 0..=201, the TLB config array, and the PCIe NIUs.
    Bar0,
    /// 1 MiB. DBI / iATU.
    Bar2,
    /// 32 GiB. TLB windows 202..=209, each 4 GiB.
    Bar4,
}

impl Bar {
    pub const fn size(self) -> u64 {
        match self {
            Bar::Bar0 => 512 * 1024 * 1024,
            Bar::Bar2 => 1024 * 1024,
            Bar::Bar4 => 32 * 1024 * 1024 * 1024,
        }
    }

    /// Byte offset of this BAR's low dword in PCI configuration space.
    pub const fn config_offset(self) -> ConfigOffset {
        match self {
            Bar::Bar0 => ConfigOffset::Bar0Lo,
            Bar::Bar2 => ConfigOffset::Bar2Lo,
            Bar::Bar4 => ConfigOffset::Bar4Lo,
        }
    }

    /// Byte offset of this BAR's high dword.
    ///
    /// Exists so that reading a 64-bit base never needs `config_offset() + 4`
    /// arithmetic on the discriminant: [`ConfigOffset`] is a closed enum
    /// precisely so an out-of-range config read cannot be written, and adding to
    /// it as an integer steps straight back out of that guarantee.
    pub const fn config_offset_hi(self) -> ConfigOffset {
        match self {
            Bar::Bar0 => ConfigOffset::Bar0Hi,
            Bar::Bar2 => ConfigOffset::Bar2Hi,
            Bar::Bar4 => ConfigOffset::Bar4Hi,
        }
    }
}

/// A readable offset in PCI configuration space.
///
/// Deliberately a closed enum rather than a `u16`. ttsim decodes only the offsets
/// below; reading anything else — including every offset at or above `0x40` — is a
/// fatal error that terminates the process. A generic "dump config space" helper
/// would be a process-killer, so the type makes one unwritable.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
#[repr(u16)]
pub enum ConfigOffset {
    /// Vendor ID in bits 0..=15, device ID in bits 16..=31.
    VendorDevice = 0x00,
    CommandStatus = 0x04,
    /// Class code, subclass, prog-if, revision ID.
    ClassRevision = 0x08,
    CacheLatencyHeader = 0x0C,
    Bar0Lo = 0x10,
    Bar0Hi = 0x14,
    Bar2Lo = 0x18,
    Bar2Hi = 0x1C,
    Bar4Lo = 0x20,
    Bar4Hi = 0x24,
    CardbusCisPointer = 0x28,
    SubsystemId = 0x2C,
    ExpansionRomPointer = 0x30,
    /// Reads 0 on Blackhole — there is no capability list, and therefore no
    /// MSI/MSI-X capability structure to walk.
    CapabilitiesPointer = 0x34,
    Reserved38 = 0x38,
    InterruptLinePin = 0x3C,
}

/// Tenstorrent's PCI vendor ID.
pub const VENDOR_ID_TENSTORRENT: u16 = 0x1E52;
/// Blackhole's PCI device ID (`ethdump.c:143`).
pub const DEVICE_ID_BLACKHOLE: u16 = 0xB140;

#[derive(Debug)]
pub enum TransportError {
    /// The access would fall outside the BAR.
    OutOfBounds { bar: Bar, offset: u64, len: u64 },
    /// The access violates an alignment or size rule for this region.
    Misaligned {
        bar: Bar,
        offset: u64,
        len: u64,
        reason: &'static str,
    },
    /// The device reported something other than a Blackhole.
    NotBlackhole { vendor: u16, device: u16 },
    /// An implementation-specific failure (an ioctl, a missing device node).
    Io(std::io::Error),
    /// The access is addressable but is known to hang or corrupt the chip in the
    /// current state, so it was refused before reaching the transport.
    Hazard { address: u64, reason: &'static str },
}

impl fmt::Display for TransportError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TransportError::OutOfBounds { bar, offset, len } => write!(
                f,
                "access of {len} bytes at {bar:?}+{offset:#x} falls outside the BAR ({:#x} bytes)",
                bar.size()
            ),
            TransportError::Misaligned {
                bar,
                offset,
                len,
                reason,
            } => write!(
                f,
                "access of {len} bytes at {bar:?}+{offset:#x} is not permitted: {reason}"
            ),
            TransportError::NotBlackhole { vendor, device } => write!(
                f,
                "expected Blackhole ({VENDOR_ID_TENSTORRENT:#06x}:{DEVICE_ID_BLACKHOLE:#06x}), \
                 found {vendor:#06x}:{device:#06x}"
            ),
            TransportError::Io(e) => write!(f, "{e}"),
            TransportError::Hazard { address, reason } => {
                write!(f, "refused access at {address:#x}: {reason}")
            }
        }
    }
}

impl std::error::Error for TransportError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            TransportError::Io(e) => Some(e),
            _ => None,
        }
    }
}

impl From<std::io::Error> for TransportError {
    fn from(e: std::io::Error) -> Self {
        TransportError::Io(e)
    }
}

pub type Result<T> = std::result::Result<T, TransportError>;

/// Host memory the card reaches by DMA ([`Transport::host_memory`]). Its
/// bytes belong to the host only while nothing queued on the card reads or
/// writes them: the caller orders the two, as it would any shared buffer.
pub trait HostMemory: Send {
    /// The NoC address of its first byte at the host-connected PCIe tile.
    fn noc_address(&self) -> u64;
    fn len(&self) -> usize;
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
    /// Copy `dst.len()` bytes from `offset`. Panics past the end.
    fn read(&self, offset: usize, dst: &mut [u8]);
    /// Copy `src` in at `offset`. Panics past the end.
    fn write(&mut self, offset: usize, src: &[u8]);
    /// Run `f` on all of its bytes in place: for filling or reading it
    /// without a copy.
    fn with_bytes(&mut self, f: &mut dyn FnMut(&mut [u8]));
}

/// How the host reaches the chip.
///
/// Implementations must validate every access *before* performing it. On the
/// simulator an out-of-range address does not fail, it terminates the process;
/// on silicon it may silently corrupt unrelated state. Either way, by the time an
/// invalid access reaches the hardware it is too late to report it.
pub trait Transport {
    /// Read from a BAR window.
    fn bar_read(&mut self, bar: Bar, offset: u64, dst: &mut [u8]) -> Result<()>;

    /// Write to a BAR window.
    fn bar_write(&mut self, bar: Bar, offset: u64, src: &[u8]) -> Result<()>;

    /// Read a dword of PCI configuration space.
    fn config_read32(&mut self, offset: ConfigOffset) -> Result<u32>;

    /// Advance the device's notion of time by `n` clocks.
    ///
    /// A no-op on silicon, where time advances by itself. On the simulator this is
    /// the *only* thing that advances time: a poll loop that never ticks spins
    /// forever against a frozen device.
    ///
    /// This is on the trait, rather than on the simulator type alone, so that code
    /// above the trait can be written once. The cost is one no-op call per poll on
    /// silicon; the alternative is teaching every caller about the simulator.
    fn tick(&mut self, n: u32);

    /// Does time stand still between calls to [`tick`](Transport::tick)?
    ///
    /// Decides what a poll loop's budget means. On the simulator it is simulated
    /// cycles, which only `tick` spends. On silicon `tick` is a no-op, so a
    /// budget counted in ticks would be a *poll count* with no bound in time --
    /// fast MMIO makes it expire early, a slow one makes it wait arbitrarily --
    /// and the loop needs a wall clock instead.
    ///
    /// Required rather than defaulted: each transport has to say which it is.
    fn is_simulated(&self) -> bool;

    /// Write device *memory* -- L1 or GDDR -- in bulk.
    ///
    /// A transport may use a faster, weaker path here (write-combining on
    /// silicon), so only [`crate::Device`]'s memory accessors call it: never a
    /// register, which may be dword-only and must see its stores in order. The
    /// write is complete from the CPU's side when this returns (nothing buffered
    /// may be overtaken by a later access), but, as with every PCIe write, not
    /// necessarily landed; see `Device::fence`.
    fn bar_write_bulk(&mut self, bar: Bar, offset: u64, src: &[u8]) -> Result<()> {
        self.bar_write(bar, offset, src)
    }

    /// Read device memory in bulk. See [`Transport::bar_write_bulk`].
    fn bar_read_bulk(&mut self, bar: Bar, offset: u64, dst: &mut [u8]) -> Result<()> {
        self.bar_read(bar, offset, dst)
    }

    /// `len` bytes of host memory the card reads and writes itself, by DMA
    /// through the PCIe tile (`tt_isa::dm::op::HOST_READ`): pinned memory on
    /// silicon, a region the simulator's DMA callbacks serve on ttsim. A
    /// transport without it says so rather than pretending.
    fn host_memory(&mut self, len: usize) -> Result<Box<dyn HostMemory>> {
        let _ = len;
        Err(TransportError::Io(std::io::Error::new(
            std::io::ErrorKind::Unsupported,
            "this transport has no host memory for the card to reach",
        )))
    }

    /// Read a single dword from a BAR.
    fn bar_read32(&mut self, bar: Bar, offset: u64) -> Result<u32> {
        let mut buf = [0u8; 4];
        self.bar_read(bar, offset, &mut buf)?;
        Ok(u32::from_le_bytes(buf))
    }

    /// Write a single dword to a BAR.
    fn bar_write32(&mut self, bar: Bar, offset: u64, value: u32) -> Result<()> {
        self.bar_write(bar, offset, &value.to_le_bytes())
    }

    /// Confirm the device is a Blackhole.
    ///
    /// Worth calling before anything else: every address constant in this crate is
    /// Blackhole-specific, and against a Wormhole they are merely plausible.
    fn verify_is_blackhole(&mut self) -> Result<()> {
        let id = self.config_read32(ConfigOffset::VendorDevice)?;
        let vendor = (id & 0xFFFF) as u16;
        let device = (id >> 16) as u16;
        if vendor == VENDOR_ID_TENSTORRENT && device == DEVICE_ID_BLACKHOLE {
            Ok(())
        } else {
            Err(TransportError::NotBlackhole { vendor, device })
        }
    }

    /// Read a 64-bit BAR base from configuration space, masking the low flag bits.
    fn bar_base(&mut self, bar: Bar) -> Result<u64> {
        let lo = self.config_read32(bar.config_offset())?;
        let hi = self.config_read32(bar.config_offset_hi())?;
        Ok(((hi as u64) << 32) | ((lo & !0xF) as u64))
    }

    /// Which chip this transport reaches.
    ///
    /// One transport is one chip. On silicon that is the device behind a single
    /// `/dev/tenstorrent/N` — no file descriptor reaches another chip's BARs — so
    /// a chip parameter on the access methods would be unimplementable there.
    /// Chip selection is therefore base-address selection, private to the
    /// implementation, and this only reports the answer.
    fn chip(&self) -> ChipId {
        ChipId(0)
    }
}
